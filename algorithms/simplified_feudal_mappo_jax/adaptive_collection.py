"""Adaptive fork collector for `intervention_scheduler: bandit | random`.

Plan: `plans/intervention_contextual_bandit_2026-10-16.md`. The fixed scheduler
(`interventions.py`) forks every main env at every `intervention_interval`-th
decision. Here a scheduler decides at each main decision whether to fork it, and
a fork that is not made costs nothing: no placement, no fork physics, no
collection slot. The freed capacity is filled with ordinary main windows.

**Budget.** An update collects exactly `K = n_windows * batch_envs` window slots
(`B = K * goal_horizon` simulator steps, the parent's per-update batch). A main
window and an executed fork each take one slot and `goal_horizon` simulator
calls, frozen steps after a termination included. A failed placement takes none.

**Rounds.** `batch_envs / 2` persistent main streams are reset at the start of
every collection (as everywhere in this stack) and reset again only when their
episode ends. Each round:

1. advances every stream by one main window (in the final round, only as many
   as there are slots left, a random subset);
2. draws a focal agent per window and asks the scheduler whether to fork it,
   reading only agent i's manager actor input (`intervention_bandit.py`);
3. admits at most the remaining slots, a random subset if more were proposed
   (the rest are "budget denied"; with no slot left they are "forced skips");
4. places the admitted forks and runs only the placed ones.

So "never fork" collects `2 * n_windows` main windows per stream and "always
fork" `n_windows` main windows plus `n_windows` forks per stream.

**Execution.** A host loop around jitted primitives. Every lane runs the
trainer's own window / fork code (`trainer.Primitives`) as a batch of one under
`vmap`, with keys folded from (stream, round), so compacting or reordering lanes
cannot change any random draw. The lane sets that run are compacted first and
split into exact power-of-two widths (13 = 8 + 4 + 1), one compiled function per
width; zero lanes means no call, so a skipped fork runs no physics at all.

**Buffers.** Main windows and forks are written into fixed-shape buffers,
`(2 * n_windows rounds, streams, ...)`, whose uncollected rows are masked
placeholders; `trainer.update_fn` packs exactly the collected rows into the PPO
batch (`pack`).
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.mappo_jax.mappo import PPOSamples
from algorithms.mappo_jax.trainer import RunnerState
from algorithms.simplified_feudal_mappo_jax import intervention_bandit as ib
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.types import (
    AdaptiveBatch,
    Intervention,
    LastValues,
    Rollout,
)

# `fold_in` namespaces of the collector's draws, all from the per-update
# collection key: none comes from the runner's main RNG chain.
NS_RESET = 0xAD01
NS_MAIN = 0xAD02
NS_FORK = 0xAD03
NS_PARTIAL = 0xAD04
NS_BANDIT = 0xAD05
# Sub-keys of one candidate's fork key (one per stream and round).
K_FOCAL, K_EXPLORE, K_ACTION, K_ADMIT, K_RECRUIT, K_PLACE, K_SCAN, K_DECIDE = range(8)

# Tests set this to `hook(name, fn) -> fn` to wrap the collector's host-dispatched
# jitted calls ("main_round", "place", "fork") and record what actually runs.
DISPATCH_HOOK = None


class Primitives(NamedTuple):
    """The trainer's width-agnostic window and fork functions (`trainer.py`)."""

    decide_window: Callable  # (train_state, env_state, obs, rng) -> Decision
    run_window: Callable  # (train_state, env_state, obs, d, rng, reset_rng) -> ...
    fork_place: Callable  # (env_state, waypoint, pos, focal, k_recruit, k_place)
    fork_rollout: Callable  # (train_state, waypoint, ..., placement, key, k_scan)
    final_value: Callable  # (train_state, env_state, obs) -> (B,) manager value


class Candidate(NamedTuple):
    """One main decision's fork candidate, per lane."""

    waypoint: jax.Array  # (N, 2)
    pos: jax.Array  # (N, 2)
    actor_in: jax.Array  # (N, actor_dim)
    critic_base: jax.Array  # (critic_dim,)
    action: jax.Array  # (N, 2)
    log_prob: jax.Array  # (N,)
    focal: jax.Array  # () int
    fork_key: jax.Array  # the candidate's fork key
    main_rewards: jax.Array  # (H,) raw team reward of the main window, per step


def lane_key(base, namespace, stream, round_idx):
    """The key of one (stream, round): independent of which lanes run with it."""
    key = jax.random.fold_in(jax.random.fold_in(base, namespace), stream)
    return jax.random.fold_in(key, round_idx)


def pow2_chunks(idx):
    """Consecutive pieces of `idx` with exact power-of-two sizes, largest first
    (13 -> 8, 4, 1); empty for an empty `idx`."""
    out, start, n = [], 0, len(idx)
    for bit in reversed(range(n.bit_length())):
        size = 1 << bit
        if n & size:
            out.append(idx[start : start + size])
            start += size
    return out


def _first(tree):
    return jax.tree.map(lambda x: x[0], tree)


def _lead(tree):
    return jax.tree.map(lambda x: x[None], tree)


def _take(tree, idx):
    return jax.tree.map(lambda x: x[idx], tree)


def window_slots(config) -> int:
    """K: window slots per update (main windows plus executed forks)."""
    return (config.worker.n_steps // config.goal_horizon) * config.intervention_batch_envs


def n_rounds(config) -> int:
    """Buffer rows: the most rounds an update can take (every fork skipped)."""
    return 2 * (config.worker.n_steps // config.goal_horizon)


def make_collect(config, env, prims: Primitives, m_gamma: float, saturation_tol: float):
    """The adaptive `collect_fn(runner_state) -> (runner_state, Rollout,
    LastValues, rollout_stats)`: the inherited train loop's contract."""
    horizon, n_agents = config.goal_horizon, env.n_agents
    n_streams = config.worker.n_envs
    slots, rounds = window_slots(config), n_rounds(config)
    scheduler = config.intervention_scheduler
    epsilon, rate = config.intervention_bandit_epsilon, config.intervention_random_rate
    penalty = config.intervention_bandit_zero_reward_penalty
    cost = config.intervention_bandit_cost_coef
    bound = config.manager_action_bound
    v_reset = jax.vmap(env.reset)

    # ---------------------------------------------------------------- lanes

    def lane_window(train_state, env_state, obs, key):
        """One main window of one stream (a batch of one)."""
        manager_rng, scan_rng, reset_rng = jax.random.split(key, 3)
        es, ob = _lead(env_state), obs[None]
        d = prims.decide_window(train_state, es, ob, manager_rng)
        es, ob, _, (w_traj, m_t, diag, _) = prims.run_window(
            train_state, es, ob, d, scan_rng, reset_rng
        )
        w_lane = jax.tree.map(lambda x: x[:, 0], w_traj._replace(action_mask=None))
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
            "window_return": m_lane.reward,
            "done": m_lane.done,
            "saturated": (
                jnp.abs(wp.bounded_action(m_lane.action, bound)) >= saturation_tol
            ).sum(),
        }
        return _first(es), ob[0], w_lane, m_lane, _first(d), summary

    def main_round(train_state, env_state, obs, streams, round_idx, collect_key):
        """Main windows for the given streams plus their scheduler proposals."""
        keys = jax.vmap(lambda s: lane_key(collect_key, NS_MAIN, s, round_idx))(streams)
        next_state, next_obs, w_lane, m_lane, d, summary = jax.vmap(
            lane_window, in_axes=(None, 0, 0, 0)
        )(train_state, env_state, obs, keys)
        fork_key = jax.vmap(lambda s: lane_key(collect_key, NS_FORK, s, round_idx))(
            streams
        )

        def draw(sub, fn):
            return jax.vmap(lambda k: fn(jax.random.fold_in(k, sub)))(fork_key)

        # Agent i is drawn independently of the manager's sampled action.
        focal = draw(K_FOCAL, lambda k: jax.random.randint(k, (), 0, n_agents))
        u_explore = draw(K_EXPLORE, jax.random.uniform)
        u_action = draw(K_ACTION, jax.random.uniform)
        priority = draw(K_ADMIT, jax.random.uniform)
        lanes = jnp.arange(streams.shape[0])
        x = jax.lax.stop_gradient(d.actor_in[lanes, focal])
        if scheduler == "bandit":
            q = train_state.bandit.apply_fn(train_state.bandit.params, x)
            intervene, explored, propensity = ib.epsilon_greedy(
                q, u_explore, u_action, epsilon
            )
        else:
            q = jnp.zeros(lanes.shape)
            intervene, propensity = ib.random_schedule(u_action, rate)
            explored = jnp.zeros(lanes.shape, bool)
        cand = Candidate(
            waypoint=d.waypoint, pos=d.pos, actor_in=d.actor_in,
            critic_base=d.critic_base, action=d.action, log_prob=d.log_prob,
            focal=focal, fork_key=fork_key, main_rewards=w_lane.team_reward,
        )
        summary = dict(
            summary, q=q, intervene=intervene, explored=explored,
            propensity=propensity, priority=priority, focal=focal, x=x,
        )
        return next_state, next_obs, (w_lane, m_lane), cand, summary

    def lane_place(env_state, c):
        return _first(
            prims.fork_place(
                _lead(env_state), c.waypoint[None], c.pos[None], c.focal[None],
                jax.random.fold_in(c.fork_key, K_RECRUIT),
                jax.random.fold_in(c.fork_key, K_PLACE),
            )
        )

    def place(env_state, cand, idx):
        """Teleports for the admitted streams `idx` (start-of-window states)."""
        return jax.vmap(lane_place)(_take(env_state, idx), _take(cand, idx))

    def lane_fork(train_state, c, placement):
        out = prims.fork_rollout(
            train_state, c.waypoint[None], c.actor_in[None], c.critic_base[None],
            c.action[None], c.log_prob[None], c.focal[None], _lead(placement),
            jax.random.fold_in(c.fork_key, K_DECIDE),
            jax.random.fold_in(c.fork_key, K_SCAN),
        )
        f_w, f_m, _, n_recruits, _, distance, delivered = out[:7]
        extras = out[7] if len(out) > 7 else {}
        f_w = jax.tree.map(lambda x: x[:, 0], f_w._replace(action_mask=None))
        f_m = _first(f_m._replace(action_mask=None))
        target, g_main, g_fork, zero = ib.paired_target(
            c.main_rewards, f_w.team_reward, m_gamma, penalty, cost
        )
        info = {
            "recruit_distance": distance[0],
            "delivered": delivered[0],
            "n_recruits": n_recruits[0],
            "fork_length": f_w.active_mask[:, 0].sum(),
            "window_return": f_m.reward,
            "worker_progress": (f_w.reward * f_w.active_mask).sum(),
            "active_agent_steps": f_w.active_mask.sum(),
            "team_reward": f_w.team_reward.sum(),
            "main_team_reward": c.main_rewards.sum(),
            "main_zero": jnp.all(c.main_rewards == 0),
            "target": target,
            "g_main": g_main,
            "g_fork": g_fork,
            "zero": zero,
            **{k: v[0] for k, v in extras.items()},
        }
        return (f_w, f_m), info

    def fork(train_state, cand, placement, local, idx):
        """Fork windows for placed lanes `local` of `placement`, from streams `idx`."""
        return jax.vmap(lane_fork, in_axes=(None, 0, 0))(
            train_state, _take(cand, idx), _take(placement, local)
        )

    def _write(buf, row, idx, part):
        return jax.tree.map(lambda b, p: b.at[row, idx].set(p), buf, part)

    def _scatter(full, part, idx):
        return jax.tree.map(lambda f, p: f.at[idx].set(p), full, part)

    def record_shapes(train_state, env_state, obs):
        """Shapes of one lane's main and fork records (abstract evaluation, no
        FLOPs), computed once: tracing the window scan costs seconds."""
        st, ob = _first(env_state), obs[0]
        streams = jnp.zeros(1, jnp.int32)
        _, _, main, cand, _ = jax.eval_shape(
            main_round, train_state, _lead(st), ob[None], streams, jnp.int32(0),
            jax.random.PRNGKey(0),
        )
        unlead = lambda t: jax.tree.map(  # noqa: E731
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), t
        )
        c = unlead(cand)
        placement = jax.eval_shape(lane_place, st, c)
        forked, _ = jax.eval_shape(lane_fork, train_state, c, placement)
        return unlead(main), forked

    def placeholders(record):
        """`(rounds, streams, ...)` buffers of a record: zeros, `done` True, so
        every mask is 0 and every row is a recursion cut."""
        w, m = jax.tree.map(
            lambda x: jnp.zeros((rounds, n_streams) + x.shape, x.dtype), record
        )
        return (
            w._replace(done=jnp.ones_like(w.done)),
            m._replace(done=jnp.ones_like(m.done)),
        )

    def reset(key):
        return v_reset(jax.random.split(jax.random.fold_in(key, NS_RESET), n_streams))

    hook = DISPATCH_HOOK or (lambda name, fn: fn)
    main_round_jit = hook("main_round", jax.jit(main_round))
    place_jit = hook("place", jax.jit(place))
    fork_jit = hook("fork", jax.jit(fork))
    # The buffers are rewritten in place (donated); the stream state is not,
    # because the round's start state must outlive the round for the forks.
    write_jit = jax.jit(_write, donate_argnums=0)
    scatter_jit = jax.jit(_scatter)
    take_jit = jax.jit(_take)
    reset_jit = jax.jit(reset)
    final_value_jit = jax.jit(prims.final_value)
    cache = {}

    def collect_fn(runner_state: RunnerState):
        train_state, rng = runner_state
        rng, collect_key = jax.random.split(rng)
        obs, state = reset_jit(collect_key)
        if "buffers" not in cache:
            main_shapes, fork_shapes = record_shapes(train_state, state, obs)
            cache["buffers"] = jax.jit(
                lambda: (placeholders(main_shapes), placeholders(fork_shapes))
            )
        main_buf, fork_buf = cache["buffers"]()

        log = _Log(n_agents, redecide_keys(config.intervention_redecide), scheduler)
        main_collected = np.zeros((rounds, n_streams), bool)
        fork_executed = np.zeros((rounds, n_streams), bool)
        fork_focal = np.zeros((rounds, n_streams), np.int32)
        fork_n = np.zeros((rounds, n_streams), np.int32)
        label_x, label_y, n_labels = None, np.zeros(slots, np.float32), 0
        label_meta = np.zeros((4, slots), np.int32)  # round, stream, focal, placed
        label_q = np.zeros(slots, np.float32)  # the bandit's prediction at the decision
        remaining, r = slots, 0
        all_streams = jnp.arange(n_streams, dtype=jnp.int32)
        while remaining > 0:
            n_main = min(n_streams, remaining)
            pre_state, pre_obs = state, obs
            row = jnp.int32(r)
            if n_main == n_streams:
                streams = np.arange(n_streams)
                state, obs, main, cand, summary = main_round_jit(
                    train_state, state, obs, all_streams, row, collect_key
                )
                main_buf = write_jit(main_buf, row, all_streams, main)
                summary = jax.device_get(summary)
            else:
                # A partial final round: a random subset of the streams. It
                # fills the last slots, so its proposals are all forced skips
                # and no fork candidate is kept.
                order = np.asarray(
                    jax.random.permutation(
                        jax.random.fold_in(collect_key, NS_PARTIAL), n_streams
                    )
                )
                streams = np.sort(order[:n_main])
                cand, summary = None, None
                for chunk in pow2_chunks(streams):
                    idx = jnp.asarray(chunk, jnp.int32)
                    sub_state, sub_obs = take_jit((pre_state, pre_obs), idx)
                    sub_state, sub_obs, main, _, sub_summary = main_round_jit(
                        train_state, sub_state, sub_obs, idx, row, collect_key
                    )
                    state, obs = scatter_jit((state, obs), (sub_state, sub_obs), idx)
                    main_buf = write_jit(main_buf, row, idx, main)
                    sub_summary = jax.device_get(sub_summary)
                    if summary is None:
                        summary = {
                            k: np.zeros((n_streams,) + v.shape[1:], v.dtype)
                            for k, v in sub_summary.items()
                        }
                    for k, v in sub_summary.items():
                        summary[k][chunk] = v
            main_collected[r, streams] = True
            remaining -= n_main
            log.main(summary, streams)
            if label_x is None:
                label_x = np.zeros((slots, summary["x"].shape[-1]), np.float32)

            # --- the scheduler's proposals and the slot budget
            proposed = streams[summary["intervene"][streams]]
            log.count("proposed", len(proposed))
            if remaining == 0:
                log.count("forced_skips", len(proposed))
                admitted = proposed[:0]
            elif len(proposed) > remaining:
                ranked = proposed[np.argsort(-summary["priority"][proposed], kind="stable")]
                admitted = np.sort(ranked[:remaining])
                log.count("budget_denied", len(proposed) - remaining)
            else:
                admitted = proposed

            # --- placement of the admitted forks, then only the placed ones run
            for chunk in pow2_chunks(admitted):
                idx = jnp.asarray(chunk, jnp.int32)
                placement = place_jit(pre_state, cand, idx)
                valid, n_rec = jax.device_get((placement.valid, placement.n_recruits))
                log.attempts(n_rec, valid)
                # Every attempt is labelled: 0 unless its fork runs (below).
                rows = np.arange(n_labels, n_labels + len(chunk))
                label_x[rows] = summary["x"][chunk]
                label_meta[0, rows] = r
                label_meta[1, rows] = chunk
                label_meta[2, rows] = summary["focal"][chunk]
                label_meta[3, rows] = valid
                label_q[rows] = summary["q"][chunk]
                n_labels += len(chunk)
                placed = np.nonzero(valid)[0]
                for sub in pow2_chunks(placed):
                    lanes = chunk[sub]
                    forked, info = fork_jit(
                        train_state, cand, placement,
                        jnp.asarray(sub, jnp.int32), jnp.asarray(lanes, jnp.int32),
                    )
                    fork_buf = write_jit(
                        fork_buf, row, jnp.asarray(lanes, jnp.int32), forked
                    )
                    info = jax.device_get(info)
                    label_y[rows[sub]] = info["target"]
                    log.forks(info, summary, lanes)
                    fork_executed[r, lanes] = True
                    fork_focal[r, lanes] = summary["focal"][lanes]
                    fork_n[r, lanes] = info["n_recruits"]
                    remaining -= len(sub)
            r += 1

        final_value = final_value_jit(train_state, state, obs)
        last_row = rounds - 1 - np.argmax(main_collected[::-1], axis=0)
        diagnostics = log.diagnostics(
            rounds_used=r, slots=slots, horizon=horizon, scheduler=scheduler,
            penalty=penalty, cost=cost, n_labels=n_labels,
        )
        diagnostics["bandit_explained_variance"] = label_explained_variance(
            label_y[:n_labels], label_q[:n_labels]
        )
        rollout = Rollout(
            worker=main_buf[0],
            manager=main_buf[1],
            diagnostics=diagnostics,
            intervention=Intervention(
                worker=fork_buf[0],
                manager=fork_buf[1],
                focal=jnp.asarray(fork_focal),
                n_recruits=jnp.asarray(fork_n),
                valid=jnp.asarray(fork_executed),
            ),
            adaptive=AdaptiveBatch(
                main_collected=jnp.asarray(main_collected),
                fork_executed=jnp.asarray(fork_executed),
                last_row=jnp.asarray(last_row, jnp.int32),
                final_value=final_value,
                label_x=jnp.asarray(label_x),
                label_y=jnp.asarray(label_y),
                label_count=jnp.asarray(n_labels, jnp.int32),
                label_round=jnp.asarray(label_meta[0]),
                label_stream=jnp.asarray(label_meta[1]),
                label_focal=jnp.asarray(label_meta[2]),
                label_placed=jnp.asarray(label_meta[3].astype(bool)),
                label_q=jnp.asarray(label_q),
            ),
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


class _Log:
    """Host-side counters and sums of one collection, turned into the scalar
    diagnostics merged into the losses (one fixed key set per run)."""

    def __init__(self, n_agents, redecide, scheduler):
        self.n_agents = n_agents
        self.redecide = redecide
        self.scheduler = scheduler
        self.c = {
            k: 0.0
            for k in (
                "candidates", "proposed", "explored", "budget_denied", "forced_skips",
                "attempted", "executed", "q_sum", "propensity_sum",
                "offset_sum", "final_error_sum", "reached_sum", "intrinsic_sum",
                "main_agent_steps", "main_env_steps", "main_team_reward",
                "main_window_return", "main_episodes", "saturated",
            )
        }
        self.attempted_n = np.zeros(n_agents)
        self.valid_n = np.zeros(n_agents)
        self.fork = {}

    def count(self, key, n):
        self.c[key] += float(n)

    def main(self, s, streams):
        c = self.c
        c["candidates"] += len(streams)
        c["explored"] += float(s["explored"][streams].sum())
        c["q_sum"] += float(s["q"][streams].sum())
        c["propensity_sum"] += float(s["propensity"][streams].sum())
        for key in ("offset_sum", "final_error_sum", "reached_sum", "intrinsic_sum"):
            c[key] += float(s[key][streams].sum())
        c["main_agent_steps"] += float(s["active_agent_steps"][streams].sum())
        c["main_env_steps"] += float(s["active_env_steps"][streams].sum())
        c["main_team_reward"] += float(s["team_reward"][streams].sum())
        c["main_window_return"] += float(s["window_return"][streams].sum())
        c["main_episodes"] += float(s["done"][streams].sum())
        c["saturated"] += float(s["saturated"][streams].sum())

    def attempts(self, n_recruits, valid):
        self.c["attempted"] += len(valid)
        for n, v in zip(n_recruits, valid):
            self.attempted_n[n] += 1
            self.valid_n[n] += bool(v)

    def forks(self, info, summary, lanes):
        """Per executed fork, with its paired main window."""
        self.c["executed"] += len(lanes)
        paired = {
            "main_delivered": summary["delivered"][lanes],
            "q": summary["q"][lanes],
            # The bandit's own pick, as opposed to the epsilon coin's. Every
            # fork of the `random` scheduler is a random pick.
            "greedy": summary["intervene"][lanes]
            & ~summary["explored"][lanes]
            & (self.scheduler == "bandit"),
        }
        for k, v in {**info, **paired}.items():
            self.fork.setdefault(k, []).append(np.asarray(v, np.float64))

    @staticmethod
    def _selection(f, executed):
        """Greedy forks (the bandit's own choice) against explored ones (the
        epsilon coin's, which ignores `q`, i.e. a random sample of decisions).
        The difference of their means is the selection value on the target."""
        if not executed:
            greedy = np.zeros(0, bool)
        else:
            greedy = f["greedy"].astype(bool)
        out = {
            "bandit_greedy_forks": float(greedy.sum()),
            "bandit_explored_forks": float((~greedy).sum()),
            "bandit_greedy_fork_frac": float(greedy.mean()) if executed else float("nan"),
        }
        for name, values in (
            ("gain", (f["g_fork"] - f["g_main"]) if executed else np.zeros(0)),
            ("target", f["target"] if executed else np.zeros(0)),
        ):
            out[f"bandit_{name}_greedy"] = _split_mean(values, greedy)
            out[f"bandit_{name}_explored"] = _split_mean(values, ~greedy)
        return out

    def diagnostics(self, rounds_used, slots, horizon, scheduler, penalty, cost, n_labels):
        c, n = self.c, self.n_agents
        f = {k: np.concatenate(v) for k, v in self.fork.items()}
        executed = c["executed"]
        n_main = c["candidates"]

        def fmean(key):
            return float(f[key].mean()) if executed else 0.0

        def share(mask):
            return float(np.mean(mask)) if executed else 0.0

        d = {
            # The fixed collector's main-window series, over collected windows.
            "intrinsic_reward": c["intrinsic_sum"] / max(c["main_agent_steps"], 1.0),
            "waypoint_reached_frac": c["reached_sum"] / max(n_main * n, 1.0),
            "waypoint_final_error": c["final_error_sum"] / max(n_main * n, 1.0),
            "waypoint_offset": c["offset_sum"] / max(n_main * n, 1.0),
            "manager_window_return": c["main_window_return"] / max(n_main, 1.0),
            "manager_action_saturation": c["saturated"] / max(n_main * n * 2, 1.0),
            "rollout_team_reward": c["main_team_reward"] / max(c["main_env_steps"], 1.0),
            # The fixed collector's fork series, over attempted / executed forks.
            "intervention_attempted": c["attempted"],
            "intervention_valid_frac": executed / max(c["attempted"], 1.0),
            "intervention_recruit_distance": fmean("recruit_distance"),
            "intervention_fork_length": fmean("fork_length"),
            "intervention_window_return": fmean("window_return"),
            "intervention_worker_progress": (
                float(f["worker_progress"].sum() / max(f["active_agent_steps"].sum(), 1.0))
                if executed
                else 0.0
            ),
            "intervention_sim_steps": float(f["fork_length"].sum()) if executed else 0.0,
            "intervention_stepped_lanes": executed * horizon,
        }
        for k in range(1, n):
            d[f"intervention_attempted_n{k}"] = float(self.attempted_n[k])
            d[f"intervention_valid_frac_n{k}"] = float(
                self.valid_n[k] / max(self.attempted_n[k], 1.0)
            )
        # Each fork against the main window it was cut from (`return_shares`).
        for prefix, reward, delivered in (
            ("intervention", "team_reward", "delivered"),
            ("manager_window", "main_team_reward", "main_delivered"),
        ):
            d[f"{prefix}_return_nonzero_frac"] = share(f[reward] != 0) if executed else 0.0
            d[f"{prefix}_return_positive_frac"] = share(f[reward] > 0) if executed else 0.0
            d[f"{prefix}_delivery_frac"] = share(f[delivered] > 0) if executed else 0.0
        for key in self.redecide:
            d[key] = fmean(key)
        # The scheduler's decisions; denominators are stated in the key names'
        # docs (CLAUDE.md): policy rates over candidates, validity over attempts.
        d.update(
            {
                "scheduler_rounds": float(rounds_used),
                "scheduler_candidates": n_main,
                "scheduler_proposed": c["proposed"],
                "scheduler_proposal_rate": c["proposed"] / max(n_main, 1.0),
                "scheduler_explored": c["explored"],
                "scheduler_policy_skips": n_main - c["proposed"],
                "scheduler_budget_denied": c["budget_denied"],
                "scheduler_forced_skips": c["forced_skips"],
                "scheduler_admitted": c["attempted"],
                "scheduler_invalid": c["attempted"] - executed,
                "scheduler_executed": executed,
                "scheduler_fork_slot_frac": executed / slots,
                "scheduler_propensity": c["propensity_sum"] / max(n_main, 1.0),
                "bandit_predicted_utility": c["q_sum"] / max(n_main, 1.0),
                "bandit_predicted_utility_executed": fmean("q"),
                "bandit_target": fmean("target"),
                "bandit_gain": float((f["g_fork"] - f["g_main"]).mean()) if executed else 0.0,
                "bandit_g_main": fmean("g_main"),
                "bandit_g_fork": fmean("g_fork"),
                "bandit_zero_reward_frac": fmean("zero"),
                "bandit_main_zero_reward_frac": fmean("main_zero"),
                "bandit_penalty": penalty * fmean("zero"),
                "bandit_cost": cost if executed else 0.0,
                "bandit_labels": float(n_labels),
                **self._selection(f, executed),
                "main_windows": n_main,
                "main_episodes": c["main_episodes"],
                "main_sim_steps": n_main * horizon,
                "main_valid_steps": c["main_env_steps"],
                "fork_sim_steps": executed * horizon,
                "fork_valid_steps": float(f["fork_length"].sum()) if executed else 0.0,
            }
        )
        return {k: float(v) for k, v in d.items()}


def _split_mean(values, mask):
    """Mean over `mask`, NaN when it is empty (not 0, which would read as a
    neutral outcome)."""
    return float(values[mask].mean()) if mask.any() else float("nan")


def label_explained_variance(y, q):
    """`1 - Var(y - q) / Var(y)` of the bandit's predictions `q` on this update's
    labels `y`. `q` was predicted before the bandit saw them, so this is a
    held-out score against a constant predictor at the labels' own mean. NaN
    with fewer than two labels or constant labels."""
    if len(y) < 2 or np.var(y) == 0:
        return float("nan")
    return float(1.0 - np.var(y - q) / np.var(y))


def redecide_keys(redecide: str):
    """The per-fork redecide diagnostics `trainer._fork_rollout` returns in each
    mode, logged every update (0 when no fork ran) so the key set is fixed."""
    return {
        "none": (),
        "focal": (
            "intervention_redecide_mean_shift",
            "intervention_redecide_recruit_clip_frac",
        ),
        "all": (
            "intervention_redecide_mean_shift",
            "intervention_redecide_recruit_waypoint_gap",
        ),
    }[redecide]


def adaptive_batches(rollout, config, no_mask):
    """The update's batches from the adaptive buffers.

    Returns `(worker, worker_rows, manager, manager_rows, sources)`: the main and
    fork columns appended along the env axis, time-major like the fixed path's
    batch, with the boolean rows that were collected; `sources` maps
    `(level, source)` to that source's own trajectory, for the per-source logs.

    Each stream's main manager trajectory ends at its last collected round. That
    boundary is bootstrapped explicitly: `gamma_M^c V^M(final state)` is added to
    the last collected row's reward unless its episode ended there, and the row
    becomes a recursion cut (`done`), so the placeholder rows after it can
    neither erase the bootstrap nor add a second one.
    """
    ad = rollout.adaptive
    gamma = config.manager.gamma

    def time_major(w):
        # (rounds, streams, H, ...) -> (rounds * H, streams, ...)
        def leaf(x):
            x = jnp.swapaxes(x, 1, 2)
            return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

        return jax.tree.map(leaf, w)._replace(action_mask=no_mask)

    m = rollout.manager
    cols = jnp.arange(m.reward.shape[1])
    open_end = 1.0 - m.done[ad.last_row, cols].astype(jnp.float32)
    main_m = m._replace(
        reward=m.reward.at[ad.last_row, cols].add(gamma * ad.final_value * open_end),
        done=m.done.at[ad.last_row, cols].set(True),
        action_mask=no_mask,
    )
    fork_m = rollout.intervention.manager._replace(action_mask=no_mask)
    main_w = time_major(rollout.worker)
    fork_w = time_major(rollout.intervention.worker)

    def append(a, b):
        return jax.tree.map(
            lambda x, y: x if x.ndim < 2 else jnp.concatenate([x, y], axis=1), a, b
        )

    manager_rows = jnp.concatenate([ad.main_collected, ad.fork_executed], axis=1)
    horizon = config.goal_horizon
    worker_rows = jnp.repeat(manager_rows, horizon, axis=0)
    sources = {
        ("worker", "main"): main_w,
        ("worker", "fork"): fork_w,
        ("manager", "main"): main_m,
        ("manager", "fork"): fork_m,
    }
    return (
        append(main_w, fork_w), worker_rows, append(main_m, fork_m), manager_rows,
        sources,
    )


def pack(samples: PPOSamples, rows, size: int) -> PPOSamples:
    """Keep exactly the `size` collected rows (row-major order) of flattened
    samples: placeholder rows never reach the optimizer."""
    idx = jnp.nonzero(rows.reshape(-1), size=size)[0]
    return jax.tree.map(lambda x: x[idx], samples)

