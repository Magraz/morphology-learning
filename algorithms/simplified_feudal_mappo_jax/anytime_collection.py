"""Anytime fork collector for `intervention_scheduler: anytime | anytime_random`.

Plan: `/home/magraz/.claude/plans/i-want-to-create-breezy-wind.md` (2026-10-25).

**Streams and slots.** As in `adaptive_collection.py`, `batch_envs / 2`
persistent main streams are reset at the start of every collection and again
whenever their episode ends, and an update collects exactly `n_windows *
batch_envs` window slots (the parent's batch). Each stream owns `2 * n_windows`
of them, filled by its main windows and by the fork windows it commits; it runs
main windows until the two add up, so unspent budget becomes main windows.

**Launches.** At every step of a main window the scheduler
(`anytime_scheduler.py`) reads the pre-step state and either skips or launches
a fork: a focal agent, the teammates to teleport next to it, and a length of k
windows. k is at most the stream's remaining budget `B - committed`, its free
slots after this window, and the windows left before the episode's time limit.
A stream launches at most once per main window (one snapshot slot), and its main
window continues unchanged. The pre-step state is the fork's start.

**Forks.** The recruits are teleported (`env.teleport_agents`, zero waypoint
offset). Then every agent decides from the post-teleport state and the fork runs
k ordinary windows (the trainer's `_decide_window` / `_run_window`), one per
round, beside the main streams. A placed fork commits its k slots; a failed
placement commits nothing. Its windows are written in commitment order to fork
rows of its stream's column. At its last window, unless its episode ended,
`gamma_M^c V^M(end state)` is added to that window's manager reward and the row
becomes a recursion cut, so two forks in one column never share an advantage.
A fork whose episode ends early keeps its k slots: its unrun windows stay masked
placeholder rows and run no physics (the frozen-step convention).

**Execution.** A host loop around jitted primitives. Every lane (main streams,
then running forks) lives in one pool of env states. Each lane runs as a batch
of one under `vmap`, with keys folded from stable identities (stream and main
window, or fork and window), so how lanes are grouped cannot change any random
draw. A window dispatch on an accelerator costs about the same at any width
(measured 16-20 ms from 1 to 16 lanes at 2a/4o), so there each round is ONE
dispatch, padded to a power of two with lanes that write nowhere; on the CPU,
where compute dominates, the active lanes run in exact power-of-two chunks
(`PAD_LANES`). A launch's pre-step state goes to a per-stream snapshot buffer,
so placement compiles once per number of launches placed together.

**After the rollout** the host computes every placed fork's paired gain
(`anytime_scheduler.span_return`) and, under `anytime`, the scheduler's GAE over
each stream's decisions (`anytime_scheduler.decision_gae`), for its update.
"""

import dataclasses
from collections import defaultdict
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.mappo_jax.trainer import RunnerState
from algorithms.simplified_feudal_mappo_jax import anytime_scheduler as asch
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.adaptive_collection import (
    _first,
    _lead,
    _take,
    pow2_chunks,
)
from algorithms.simplified_feudal_mappo_jax.types import (
    AdaptiveBatch,
    Intervention,
    LastValues,
    Rollout,
    SchedulerBatch,
)

# `fold_in` namespaces, all from the per-update collection key (none comes from
# the runner's main RNG chain), plus the scheduler's fit in the update.
NS_RESET = 0xA701
NS_MAIN = 0xA702
NS_FORK = 0xA703
NS_PLACE = 0xA704
NS_SCHED = 0xA705
NS_FIT = 0xA706

# Tests set this to `hook(name, fn) -> fn` to wrap the collector's host-dispatched
# jitted calls ("window", "place") and record what actually runs.
DISPATCH_HOOK = None
# One padded dispatch per round (True) or exact power-of-two chunks (False).
# None picks by the env states' device: padded except on the CPU.
PAD_LANES = None

# Logged means over launches or placed forks: NaN when there was none (not 0,
# which would read as a neutral outcome).
NAN_WHEN_EMPTY = (
    "intervention_valid_frac", "intervention_recruit_distance",
    "intervention_fork_length", "intervention_window_return",
    "intervention_delivery_frac", "intervention_n_recruits_mean",
    "scheduler_mean_duration", "scheduler_mean_kmax", "scheduler_launch_step_mean",
    "scheduler_window_start_frac", "scheduler_value_mean",
    "anytime_gain", "anytime_g_main", "anytime_g_fork", "anytime_bootstrap",
    "anytime_span", "anytime_gain_positive_frac",
)


class Primitives(NamedTuple):
    """The trainer's window and placement functions (`trainer.py`), all batched
    over a leading env axis."""

    decide_window: Callable  # (train_state, env_state, obs, rng) -> Decision
    run_window: Callable  # (train_state, env_state, obs, d, rng, reset_rng, tracker)
    place: Callable  # (env_state, focal, n_recruits, recruit, offset, key) -> Placement
    manager_value: Callable  # (train_state, env_state, obs) -> (B,) V^M, zero context
    context: Callable  # (obs, env_state) -> (actor_in (B, N, d), pos (B, N, 2))


class Record(NamedTuple):
    """The scheduler's view and choice at one step of a main window."""

    eligible: jax.Array  # bool: it decided at this step
    choice: jax.Array  # int: 0 skip, i + 1 a fork around agent i
    recruit: jax.Array  # (N,) bool
    duration: jax.Array  # int: windows - 1
    kmax: jax.Array  # int: the longest allowed fork, in windows
    log_prob: jax.Array
    value: jax.Array  # the scheduler's value
    v_m: jax.Array  # the manager's value of the pre-step state (zero context)
    x: jax.Array  # (N, actor_dim)
    pair: jax.Array  # (N, N, 2)
    glob: jax.Array  # (N_GLOBAL,)


class Buffers(NamedTuple):
    """Fixed-shape `(rows, streams, ...)` buffers; uncollected cells stay
    placeholders (zeros, `done` set)."""

    main: tuple  # (worker Transition (.., H, ...), manager Transition)
    fork: tuple  # the same, one row per fork window
    rec: Record  # (.., H, ...) the scheduler's records of main windows
    fork_vm: jax.Array  # (.., H) pre-step manager values in fork windows
    fork_end: jax.Array  # manager value after each fork window, before any reset
    fork_ended: jax.Array  # bool: the fork's episode ended in that window


@dataclasses.dataclass
class _Lane:
    pool_row: int
    stream: int
    row: int  # buffer row: the main window index, or the fork row
    main: bool
    key: tuple  # (namespace, a, b, c) folded into the collection key
    kmax: int = 0
    budget_left: int = 0
    slots_left: int = 0
    is_last: bool = False
    fork: object = None


@dataclasses.dataclass
class _Fork:
    stream: int
    row0: int  # the main window it was launched in
    step: int  # the step of that window
    k: int
    j0: int  # its first fork row in the stream's column
    pool_row: int
    launch: int  # index into the launch list
    m: int = 0  # windows run
    done: bool = False
    delivered: bool = False


def make_collect(config, env, prims: Primitives, m_gamma: float, saturation_tol: float):
    """The anytime `collect_fn(runner_state) -> (runner_state, Rollout,
    LastValues, rollout_stats)`: the inherited train loop's contract."""
    horizon, n_agents = config.goal_horizon, env.n_agents
    n_streams = config.worker.n_envs
    slots = asch.stream_slots(config)  # per stream; also the buffers' row count
    budget = asch.fork_budget(config)
    pool_size = n_streams + n_streams * budget  # in-flight forks <= B per stream
    total_slots = slots * n_streams
    learned = config.intervention_scheduler == "anytime"
    gamma_window = config.manager.gamma  # gamma_M ** goal_horizon
    bound = config.manager_action_bound
    max_steps = int(env.max_steps)
    world_scale = float(getattr(env, "world_width", 1.0))
    v_reset = jax.vmap(env.reset)
    v_pos = jax.vmap(env.goal_state)

    # ---------------------------------------------------------------- lanes

    def lane_window(train_state, env_state, obs, key, kmax_lane, budget_left,
                    slots_left, is_last):
        """One window of one lane (a batch of one), with the scheduler deciding
        at every pre-step state while `kmax >= 1` (main lanes only)."""
        manager_rng, scan_rng, reset_rng = jax.random.split(key, 3)
        sched_key = jax.random.fold_in(key, NS_SCHED)
        sched = train_state.bandit
        es, ob = _lead(env_state), obs[None]
        d = prims.decide_window(train_state, es, ob, manager_rng)
        x0, _ = prims.context(ob, es)
        actor_dim = x0.shape[-1]

        def track(carry, state, o, alive, k):
            fired, snap_state, snap_obs, launch, rec = carry
            x, pos = prims.context(o, state)
            t = state.t[0]
            # The windows that fit before the time limit: ceil((T - t) / c).
            kmax = jnp.minimum(kmax_lane, -((t - max_steps) // horizon))
            eligible = alive[0] & ~fired & (kmax >= 1)
            glob = jnp.stack(
                [budget_left / budget, slots_left / slots, t / max_steps, k / horizon]
            ).astype(jnp.float32)
            pair = pos[0][None, :, :] - pos[0][:, None, :]  # [i, j] = pos_j - pos_i
            h = sched.apply_fn(sched.params, x[0], pair, glob)
            action, log_prob = asch.sample(h, kmax, jax.random.fold_in(sched_key, k))
            go = eligible & (action.choice > 0)

            def keep(new, old):
                return jnp.where(go, new, old)

            snap_state = jax.tree.map(keep, state, snap_state)
            snap_obs = keep(o, snap_obs)
            launch = jax.tree.map(keep, (k, action, kmax), launch)
            step_record = Record(
                eligible=eligible, choice=action.choice, recruit=action.recruit,
                duration=action.duration, kmax=kmax, log_prob=log_prob,
                value=h.value, v_m=prims.manager_value(train_state, state, o)[0],
                x=x[0], pair=pair, glob=glob,
            )
            rec = jax.tree.map(lambda buf, v: buf.at[k].set(v), rec, step_record)
            return fired | go, snap_state, snap_obs, launch, rec

        zero = jnp.zeros((), jnp.int32)
        no_action = asch.Action(choice=zero, recruit=jnp.zeros(n_agents, bool), duration=zero)
        rec0 = Record(
            eligible=jnp.zeros(horizon, bool),
            choice=jnp.zeros(horizon, jnp.int32),
            recruit=jnp.zeros((horizon, n_agents), bool),
            duration=jnp.zeros(horizon, jnp.int32),
            kmax=jnp.zeros(horizon, jnp.int32),
            log_prob=jnp.zeros(horizon),
            value=jnp.zeros(horizon),
            v_m=jnp.zeros(horizon),
            x=jnp.zeros((horizon, n_agents, actor_dim)),
            pair=jnp.zeros((horizon, n_agents, n_agents, 2)),
            glob=jnp.zeros((horizon, asch.N_GLOBAL)),
        )
        carry0 = (jnp.zeros((), bool), es, ob, (zero, no_action, zero), rec0)
        es, ob, _, (w_traj, m_t, diag, _), (fired, snap_state, snap_obs, launch, rec) = (
            prims.run_window(train_state, es, ob, d, scan_rng, reset_rng, (track, carry0))
        )
        ended = m_t.done[0]
        # The state after the window, before any reset: a reset only follows an
        # ended episode, whose value is never read.
        end_value = prims.manager_value(train_state, es, ob)[0]
        raw_return = m_t.reward[0]
        fold = (is_last & ~ended).astype(jnp.float32)
        m_t = m_t._replace(
            reward=m_t.reward + fold * gamma_window * end_value,
            done=m_t.done | is_last,
        )
        w_lane = jax.tree.map(lambda v: v[:, 0], w_traj._replace(action_mask=None))
        m_lane = _first(m_t._replace(action_mask=None))
        summary = {
            "offset_sum": diag["offset"].sum(),
            "final_error_sum": diag["final_error"].sum(),
            "reached_sum": diag["reached"].sum(),
            "intrinsic_sum": diag["intrinsic_sum"],
            "delivered": diag["delivered"][0],
            "active_agent_steps": w_lane.active_mask.sum(),
            "active_env_steps": w_lane.active_mask[:, 0].sum(),
            "team_reward": w_lane.team_reward.sum(),
            "window_return": raw_return,
            "ended": ended,
            "saturated": (
                jnp.abs(wp.bounded_action(m_lane.action, bound)) >= saturation_tol
            ).sum(),
        }
        step, action, kmax = launch
        host = dict(
            summary, fired=fired, step=step, choice=action.choice,
            recruit=action.recruit, duration=action.duration, kmax=kmax,
        )
        return (
            (_first(es), ob[0]), (w_lane, m_lane), rec, end_value, ended, host,
            (_first(snap_state), snap_obs[0]),
        )

    def run_chunk(train_state, pool, snap, bufs, idx, write, keys, kmax_lane,
                  budget_left, slots_left, is_last, row, col, is_main):
        """One window of the lanes `idx` of the pool: their new states go back
        to pool rows `write`, a launch's pre-step state to `snap[stream]`, and
        their records to the main or the fork buffers. Out-of-bounds indices
        (padding lanes, the other buffer's lanes) write nothing."""
        state, obs = _take(pool, idx)
        new, records, rec, end_value, ended, host, snaps = jax.vmap(
            lane_window, in_axes=(None, 0, 0, 0, 0, 0, 0, 0)
        )(train_state, state, obs, keys, kmax_lane, budget_left, slots_left, is_last)
        pool = jax.tree.map(lambda p, v: p.at[write].set(v, mode="drop"), pool, new)
        launched = jnp.where(is_main & host["fired"], col, n_streams)
        snap = jax.tree.map(lambda s, v: s.at[launched].set(v, mode="drop"), snap, snaps)
        main_row = jnp.where(is_main, row, slots)
        fork_row = jnp.where(is_main, slots, row)

        def put(buf, r, part):
            return jax.tree.map(lambda b, p: b.at[r, col].set(p, mode="drop"), buf, part)

        bufs = Buffers(
            main=put(bufs.main, main_row, records),
            fork=put(bufs.fork, fork_row, records),
            rec=put(bufs.rec, main_row, rec),
            fork_vm=put(bufs.fork_vm, fork_row, rec.v_m),
            fork_end=put(bufs.fork_end, fork_row, end_value),
            fork_ended=put(bufs.fork_ended, fork_row, ended),
        )
        return pool, snap, bufs, host

    def place_chunk(pool, snap, streams, dest, focal, recruit, keys):
        """Teleport the chosen recruits next to the focal agent in the launch
        snapshots of `streams`, each a batch of one with its own key. Placed
        states go to pool rows `dest`; a failed placement leaves its row as it
        was."""
        state, obs = _take(snap, streams)

        def lane(s, f, r, key):
            p = prims.place(
                _lead(s), f[None], r.sum()[None].astype(jnp.int32), r[None],
                jnp.zeros((1, 2)), key,
            )
            return _first(p)

        p = jax.vmap(lane)(state, focal, recruit, keys)
        pos = v_pos(p.state)
        gap = jnp.linalg.norm(pos - pos[jnp.arange(focal.shape[0]), focal][:, None], axis=-1)
        distance = world_scale * (gap * recruit).sum(-1) / jnp.maximum(recruit.sum(-1), 1)

        def keep(new, old):
            return jnp.where(p.valid.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)

        placed = jax.tree.map(keep, (p.state, p.obs), _take(pool, dest))
        pool = jax.tree.map(lambda q, v: q.at[dest].set(v), pool, placed)
        return pool, p.valid, distance

    def lane_keys(base, ns, a, b, c):
        def one(ns, a, b, c):
            key = jax.random.fold_in(jax.random.fold_in(base, ns), a)
            return jax.random.fold_in(jax.random.fold_in(key, b), c)

        return jax.vmap(one)(ns, a, b, c)

    def reset(key):
        """The pool: every main stream reset, every fork row a placeholder copy
        of stream 0 (overwritten before it ever runs)."""
        obs, state = v_reset(
            jax.random.split(jax.random.fold_in(key, NS_RESET), n_streams)
        )
        pad = pool_size - n_streams
        return jax.tree.map(
            lambda x: jnp.concatenate([x, jnp.repeat(x[:1], pad, axis=0)]), (state, obs)
        )

    def placeholders(out):
        """`(rows, streams, ...)` buffers shaped like one lane's records."""
        _, records, rec, end_value, ended, _, _ = out

        def grid(x):
            return jnp.zeros((slots, n_streams) + x.shape, x.dtype)

        def transitions():
            w, m = jax.tree.map(grid, records)
            return (w._replace(done=jnp.ones_like(w.done)),
                    m._replace(done=jnp.ones_like(m.done)))

        return Buffers(
            main=transitions(), fork=transitions(), rec=jax.tree.map(grid, rec),
            fork_vm=grid(rec.v_m), fork_end=grid(end_value), fork_ended=grid(ended),
        )

    hook = DISPATCH_HOOK or (lambda name, fn: fn)
    run_jit = hook("window", jax.jit(run_chunk, donate_argnums=(1, 2, 3)))
    place_jit = hook("place", jax.jit(place_chunk, donate_argnums=0))
    keys_jit = jax.jit(lane_keys)
    reset_jit = jax.jit(reset)
    take_jit = jax.jit(_take)
    value_jit = jax.jit(prims.manager_value)
    cache = {}

    def keys_for(base, lanes_keys):
        cols = np.asarray(lanes_keys, np.int64).T
        return keys_jit(base, *(jnp.asarray(c, jnp.int32) for c in cols))

    # ---------------------------------------------------------------- collect

    def collect_fn(runner_state: RunnerState):
        train_state, rng = runner_state
        rng, collect_key = jax.random.split(rng)
        pool = reset_jit(collect_key)
        if "buffers" not in cache:
            one = jax.tree.map(lambda x: x[0], pool)
            z = jnp.int32(0)
            shapes = jax.eval_shape(
                lane_window, train_state, one[0], one[1], jax.random.PRNGKey(0),
                z, z, z, jnp.bool_(False),
            )
            cache["buffers"] = jax.jit(lambda: placeholders(shapes))
            platform = next(iter(jax.tree.leaves(pool)[0].devices())).platform
            cache["pad"] = platform != "cpu" if PAD_LANES is None else PAD_LANES
        bufs, pad = cache["buffers"](), cache["pad"]
        snap = take_jit(pool, jnp.arange(n_streams))  # launch states, by stream

        main_done = np.zeros(n_streams, np.int64)
        committed = np.zeros(n_streams, np.int64)
        main_collected = np.zeros((slots, n_streams), bool)
        fork_executed = np.zeros((slots, n_streams), bool)
        fork_focal = np.zeros((slots, n_streams), np.int32)
        fork_n = np.zeros((slots, n_streams), np.int32)
        free = list(range(pool_size - 1, n_streams - 1, -1))
        running, forks, launches = [], [], []
        log = _Log()
        rounds = 0
        while True:
            lanes = []
            for e in range(n_streams):
                used = main_done[e] + committed[e]
                if used < slots:
                    left, after = budget - committed[e], slots - used - 1
                    lanes.append(_Lane(
                        pool_row=e, stream=e, row=int(main_done[e]), main=True,
                        key=(NS_MAIN, e, int(main_done[e]), 0), kmax=int(min(left, after)),
                        budget_left=int(left), slots_left=int(after),
                    ))
            for f in running:
                lanes.append(_Lane(
                    pool_row=f.pool_row, stream=f.stream, row=f.j0 + f.m, main=False,
                    key=(NS_FORK, f.stream, f.row0, f.m), is_last=f.m == f.k - 1, fork=f,
                ))
            if not lanes:
                break
            everything = np.arange(len(lanes))
            for chunk in [everything] if pad else pow2_chunks(everything):
                ls = [lanes[i] for i in chunk]
                # Padding lanes read lane 0's state, launch nothing and write
                # nowhere (out-of-bounds pool row and buffer row).
                n_pad = (1 << (len(ls) - 1).bit_length()) - len(ls) if pad else 0

                def arr(name, fill, dtype=jnp.int32, ls=ls, n_pad=n_pad):
                    values = [getattr(lane, name) for lane in ls] + [fill] * n_pad
                    return jnp.asarray(np.array(values), dtype)

                pool, snap, bufs, host = run_jit(
                    train_state, pool, snap, bufs, arr("pool_row", ls[0].pool_row),
                    arr("pool_row", pool_size),
                    keys_for(collect_key, [lane.key for lane in ls] + [ls[0].key] * n_pad),
                    arr("kmax", 0), arr("budget_left", 0), arr("slots_left", 0),
                    arr("is_last", False, bool), arr("row", slots), arr("stream", 0),
                    arr("main", False, bool),
                )
                host = jax.device_get(host)
                launched = []
                for j, lane in enumerate(ls):
                    if lane.main:
                        main_collected[lane.row, lane.stream] = True
                        main_done[lane.stream] += 1
                        log.main_window(host, j)
                        if host["fired"][j]:
                            launched.append(j)
                    else:
                        f = lane.fork
                        log.fork_window(host, j)
                        f.delivered |= bool(host["delivered"][j])
                        f.m += 1
                        if host["ended"][j] or f.m == f.k:
                            f.done = True
                            free.append(f.pool_row)

                # Place this chunk's launches into free pool rows.
                for sub in pow2_chunks(np.asarray(launched, np.int64)):
                    dest = [free.pop() for _ in sub]
                    focal = host["choice"][sub] - 1
                    recruit = host["recruit"][sub]
                    keys = keys_for(
                        collect_key,
                        [(NS_PLACE, ls[j].stream, ls[j].row, 0) for j in sub],
                    )
                    pool, valid, distance = place_jit(
                        pool, snap, jnp.asarray([ls[j].stream for j in sub], jnp.int32),
                        jnp.asarray(dest, jnp.int32), jnp.asarray(focal, jnp.int32),
                        jnp.asarray(recruit), keys,
                    )
                    valid, distance = jax.device_get((valid, distance))
                    for i, j in enumerate(sub):
                        e, k = ls[j].stream, int(host["duration"][j]) + 1
                        launch = {
                            "stream": e, "row": ls[j].row, "step": int(host["step"][j]),
                            "k": k, "kmax": int(host["kmax"][j]),
                            "n_recruits": int(recruit[i].sum()), "valid": bool(valid[i]),
                            "distance": float(distance[i]), "fork": None,
                        }
                        if valid[i]:
                            j0 = int(committed[e])
                            committed[e] += k
                            fork_executed[j0 : j0 + k, e] = True
                            fork_focal[j0 : j0 + k, e] = focal[i]
                            fork_n[j0 : j0 + k, e] = launch["n_recruits"]
                            f = _Fork(
                                stream=e, row0=ls[j].row, step=launch["step"], k=k, j0=j0,
                                pool_row=dest[i], launch=len(launches),
                            )
                            launch["fork"] = len(forks)
                            forks.append(f)
                            running.append(f)
                        else:
                            free.append(dest[i])
                        launches.append(launch)
            running = [f for f in running if not f.done]
            rounds += 1

        state, obs = take_jit(pool, jnp.arange(n_streams))
        final_value = value_jit(train_state, state, obs)
        last_row = main_done - 1
        assert main_collected.sum() + fork_executed.sum() == total_slots

        # ------------------------------------------------ the launches' rewards
        main_w, main_m = bufs.main
        fork_w, _ = bufs.fork
        h = jax.device_get({
            "m_reward": main_w.team_reward, "m_active": main_w.active_mask[..., 0],
            "m_ended": main_m.done, "m_value": bufs.rec.v_m,
            "f_reward": fork_w.team_reward, "f_active": fork_w.active_mask[..., 0],
            "f_ended": bufs.fork_ended, "f_value": bufs.fork_vm, "f_end": bufs.fork_end,
            "eligible": bufs.rec.eligible, "s_value": bufs.rec.value,
            "final": final_value,
        })
        gains = paired_gains(h, forks, main_done, horizon, m_gamma)
        reward = np.zeros((slots, n_streams, horizon), np.float32)
        for f, g in zip(forks, gains):
            reward[f.row0, f.stream, f.step] = g["gain"]

        decided = h["eligible"]
        scheduler = None
        if learned:
            def time_major(x):  # (rows, streams, H) -> (rows * H, streams)
                return np.transpose(x, (0, 2, 1)).reshape(slots * horizon, n_streams)

            def grid(x):
                return np.transpose(x.reshape(slots, horizon, n_streams), (0, 2, 1))

            adv, ret = asch.decision_gae(
                time_major(reward), time_major(h["s_value"]), time_major(decided),
                config.intervention_scheduler_gamma, config.intervention_scheduler_lambda,
            )
            rec = bufs.rec
            scheduler = SchedulerBatch(
                x=rec.x, pair=rec.pair, glob=rec.glob, choice=rec.choice,
                recruit=rec.recruit, duration=rec.duration, kmax=rec.kmax,
                log_prob=rec.log_prob, value=rec.value, reward=jnp.asarray(reward),
                advantage=jnp.asarray(grid(adv)), returns=jnp.asarray(grid(ret)),
                weight=jnp.asarray(decided, jnp.float32),
            )

        diagnostics = log.diagnostics(
            launches=launches, forks=forks, gains=gains, decided=decided,
            s_value=h["s_value"], committed=committed, rounds=rounds,
            n_agents=n_agents, horizon=horizon, total_slots=total_slots,
            budget_total=budget * n_streams,
        )
        rollout = Rollout(
            worker=main_w,
            manager=main_m,
            diagnostics=diagnostics,
            intervention=Intervention(
                worker=fork_w,
                manager=bufs.fork[1],
                focal=jnp.asarray(fork_focal),
                n_recruits=jnp.asarray(fork_n),
                valid=jnp.asarray(fork_executed),
            ),
            adaptive=AdaptiveBatch(
                main_collected=jnp.asarray(main_collected),
                fork_executed=jnp.asarray(fork_executed),
                last_row=jnp.asarray(last_row, jnp.int32),
                final_value=final_value,
            ),
            scheduler=scheduler,
        )
        last_values = LastValues(
            worker=jnp.zeros((n_streams, n_agents)), manager=final_value
        )
        rollout_stats = {
            "mean_reward": diagnostics["rollout_team_reward"],
            "episode_count": diagnostics["main_episodes"],
        }
        return RunnerState(train_state=train_state, rng=rng), rollout, last_values, rollout_stats

    return collect_fn


def paired_gains(h, forks, main_done, horizon: int, gamma: float):
    """Each placed fork's paired gain against its stream's main rollout, from the
    collection's host arrays `h` (`(rows, streams, ...)`).

    The span is the fork's k windows, cut at the end of the stream's rollout.
    Each side's discounted raw team reward over the span stops at its own
    episode's end; a side still in its episode at the span's end adds its
    discounted manager value there (`anytime_scheduler.span_return`). Returns
    one dict per fork: `gain`, `g_main`, `g_fork`, `bootstrap` (the fork's value
    term minus the main's) and `span`."""
    main = {}
    out = []
    for f in forks:
        e = f.stream
        if e not in main:
            rows = int(main_done[e])
            main[e] = (
                h["m_reward"][:rows, e].reshape(-1),
                asch.continuation(h["m_active"][:rows, e], h["m_ended"][:rows, e]),
                np.concatenate([h["m_value"][:rows, e].reshape(-1), [h["final"][e]]]),
            )
        m_reward, m_cont, m_value = main[e]
        rows = slice(f.j0, f.j0 + f.k)
        f_cont = asch.continuation(h["f_active"][rows, e], h["f_ended"][rows, e])
        f_value = np.concatenate(
            [h["f_value"][rows, e].reshape(-1), [h["f_end"][f.j0 + f.k - 1, e]]]
        )
        start = f.row0 * horizon + f.step
        span = min(f.k * horizon, len(m_reward) - start)
        g_main, b_main = asch.span_return(m_reward, m_cont, m_value, gamma, start, span)
        g_fork, b_fork = asch.span_return(
            h["f_reward"][rows, e].reshape(-1), f_cont, f_value, gamma, 0, span
        )
        out.append({
            "gain": g_fork + b_fork - g_main - b_main,
            "g_main": g_main, "g_fork": g_fork, "bootstrap": b_fork - b_main,
            "span": span,
        })
    return out


class _Log:
    """Host-side sums of one collection, turned into the scalar diagnostics
    merged into the losses (one fixed key set per run)."""

    _MAIN = (
        "offset_sum", "final_error_sum", "reached_sum", "intrinsic_sum",
        "active_agent_steps", "active_env_steps", "team_reward", "window_return",
        "ended", "saturated",
    )

    def __init__(self):
        self.main = defaultdict(float)
        self.fork = defaultdict(float)

    def main_window(self, host, j):
        self.main["windows"] += 1
        for key in self._MAIN:
            self.main[key] += float(host[key][j])

    def fork_window(self, host, j):
        self.fork["windows"] += 1
        self.fork["live_steps"] += float(host["active_env_steps"][j])
        self.fork["window_return"] += float(host["window_return"][j])

    def diagnostics(self, launches, forks, gains, decided, s_value, committed,
                    rounds, n_agents, horizon, total_slots, budget_total):
        c, fk = self.main, self.fork
        n_main = c["windows"]
        placed = [launch for launch in launches if launch["valid"]]

        def mean(values):
            return float(np.mean(values)) if len(values) else float("nan")

        n_decisions = float(decided.sum())
        d = {
            # The fixed collector's main-window series, over collected windows.
            "intrinsic_reward": c["intrinsic_sum"] / max(c["active_agent_steps"], 1.0),
            "waypoint_reached_frac": c["reached_sum"] / max(n_main * n_agents, 1.0),
            "waypoint_final_error": c["final_error_sum"] / max(n_main * n_agents, 1.0),
            "waypoint_offset": c["offset_sum"] / max(n_main * n_agents, 1.0),
            "manager_window_return": c["window_return"] / max(n_main, 1.0),
            "manager_action_saturation": c["saturated"] / max(n_main * n_agents * 2, 1.0),
            "rollout_team_reward": c["team_reward"] / max(c["active_env_steps"], 1.0),
            "main_windows": n_main,
            "main_episodes": c["ended"],
            "main_sim_steps": n_main * horizon,
            "main_valid_steps": c["active_env_steps"],
            # Fork windows: every committed slot counts, run or not.
            "fork_windows": float(committed.sum()),
            "fork_sim_steps": float(committed.sum()) * horizon,
            "fork_valid_steps": fk["live_steps"],
            "intervention_attempted": float(len(launches)),
            "intervention_valid_frac": mean([launch["valid"] for launch in launches]),
            "intervention_n_recruits_mean": mean([x["n_recruits"] for x in launches]),
            "intervention_recruit_distance": mean([x["distance"] for x in placed]),
            "intervention_fork_length": (
                fk["live_steps"] / len(forks) if forks else float("nan")
            ),
            "intervention_window_return": (
                fk["window_return"] / fk["windows"] if fk["windows"] else float("nan")
            ),
            "intervention_delivery_frac": mean([f.delivered for f in forks]),
            # The scheduler's decisions.
            "scheduler_rounds": float(rounds),
            "scheduler_decisions": n_decisions,
            "scheduler_launched": float(len(launches)),
            "scheduler_launch_rate": len(launches) / max(n_decisions, 1.0),
            "scheduler_executed": float(len(forks)),
            "scheduler_fork_windows": float(committed.sum()),
            "scheduler_fork_slot_frac": float(committed.sum()) / total_slots,
            "scheduler_budget_spent_frac": float(committed.sum()) / budget_total,
            "scheduler_mean_duration": mean([x["k"] for x in launches]),
            "scheduler_mean_kmax": mean([x["kmax"] for x in launches]),
            "scheduler_launch_step_mean": mean([x["step"] for x in launches]),
            "scheduler_window_start_frac": mean([x["step"] == 0 for x in launches]),
            "scheduler_value_mean": (
                float(s_value[decided].mean()) if n_decisions else float("nan")
            ),
            # Each placed fork's paired gain (`paired_gains`).
            "anytime_gain": mean([g["gain"] for g in gains]),
            "anytime_g_main": mean([g["g_main"] for g in gains]),
            "anytime_g_fork": mean([g["g_fork"] for g in gains]),
            "anytime_bootstrap": mean([g["bootstrap"] for g in gains]),
            "anytime_span": mean([g["span"] for g in gains]),
            "anytime_gain_positive_frac": mean([g["gain"] > 0 for g in gains]),
        }
        return {k: float(v) for k, v in d.items()}
