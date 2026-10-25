"""Heuristic fork rule for the training forks (`intervention_scheduler: heuristic`).

Plan: `/home/magraz/.claude/plans/peaceful-stargazing-meteor.md` (2026-10-24).
The bandit and random schedulers forked at window starts around a uniformly
drawn focal agent with a random number of recruits, and those forks paid less
than the windows they replaced. This rule forks only where help is
structurally missing, sized to what is missing:

* **When:** at any primitive step of the main rollout, mid-window included. The
  candidate is the state BEFORE that step, a state the main rollout visits.
* **Focal agent:** an agent whose closest undelivered box (by surface distance)
  is within touch range plus `intervention_heuristic_trigger_margin`, and whose
  box does not have its coupling met. One such agent per env and step is drawn
  uniformly.
* **Recruits:** N = the box's coupling minus the agents touching it, counting
  the focal agent as one of them. The N teammates nearest the focal agent among
  those not touching the box are teleported next to it
  (`env.teleport_agents`, with a zero waypoint offset).
* **Goals:** every agent re-decides its goal from the post-teleport state
  (`intervention_redecide: all`), so the fork is an on-policy window of
  `goal_horizon` steps from a teleported start, bootstrapped like the main
  rollout.
* **Budget:** each main env keeps a uniform random sample of at most
  `intervention_heuristic_budget` of its rollout's qualifying moments
  (bottom-K priority sampling, `reservoir_update`). The forks run after the
  main rollout, from the stored states.

Fork transitions are packed into extra env columns (`pack`), valid forks first,
so that no fork column holds a single valid fork: `masked_normalize` leaves a
stream with fewer than two rows unnormalized, which would give that row a raw,
unit-free advantage.

The rule reads privileged state (touch counts, couplings). It only chooses
where training forks start; execution and evaluation are unchanged.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

# `fold_in` constants, so the rule's draws consume no split of the main path's
# keys (as `interventions.FORK_KEY_NAMESPACE` does for the fixed forks).
HEURISTIC_KEY_NAMESPACE = 0x4E55
FORK_RUN_NAMESPACE = 0x4E56


class Moment(NamedTuple):
    """A candidate fork at one step, per env (leading dim E), or unbatched."""

    qualifies: jax.Array  # bool
    focal: jax.Array  # int
    n_recruits: jax.Array  # int, N = the missing agents (0 if not qualifying)
    recruit: jax.Array  # (A,) bool
    box: jax.Array  # int, the focal agent's closest undelivered box
    clearance: jax.Array  # the focal agent's clearance to it, world units
    touching: jax.Array  # bool, the focal agent already touches it


class Reservoir(NamedTuple):
    """Each main env's sample of its qualifying moments, `(E, K, ...)`.

    `priority` is +inf for an empty slot. A slot holds the env state and
    observation BEFORE the step the moment was seen at, plus the moment."""

    priority: jax.Array  # (E, K)
    env_state: object  # (E, K, ...) pytree
    obs: jax.Array  # (E, K, A, obs_dim)
    moment: Moment  # (E, K, ...)
    step: jax.Array  # (E, K) int, the main rollout step (window * H + k)
    events: jax.Array  # (E,) int, qualifying moments seen, before the budget


def candidate(contacts, key, margin: float) -> Moment:
    """The heuristic's candidate fork for ONE env state (`contacts`: the env's
    `box_contacts`, unbatched)."""
    clearance = jnp.where(contacts.delivered[None, :], jnp.inf, contacts.clearance)
    n_agents, n_boxes = clearance.shape
    if n_boxes == 0:
        none = jnp.zeros((), bool)
        zero = jnp.zeros((), jnp.int32)
        return Moment(
            qualifies=none, focal=zero, n_recruits=zero,
            recruit=jnp.zeros(n_agents, bool), box=zero,
            clearance=jnp.zeros(()), touching=none,
        )
    agents = jnp.arange(n_agents)
    box = jnp.argmin(clearance, axis=1)  # (A,) each agent's closest undelivered box
    own = clearance[agents, box]  # (A,), inf when every box is delivered
    touching = contacts.touching  # (A, O)
    on_box = touching[:, box].T  # (A, A): [i, j] = agent j touches agent i's box
    self_touch = jnp.diagonal(on_box)
    # The focal agent counts as one of the box's agents, touching or not.
    have = on_box.sum(axis=1) + (~self_touch).astype(jnp.int32)
    deficit = contacts.coupling[box] - have  # (A,)
    eligible = ~on_box & (agents[None, :] != agents[:, None])  # (A, A) recruits of i
    qualifies = (
        jnp.isfinite(own)
        & (own <= margin)
        & (deficit >= 1)
        & (eligible.sum(axis=1) >= deficit)
    )

    any_q = qualifies.any()
    score = jnp.where(qualifies, jax.random.uniform(key, (n_agents,)), -1.0)
    focal = jnp.argmax(score)
    n_recruits = jnp.where(any_q, deficit[focal], 0).astype(jnp.int32)
    # The N nearest teammates to the focal agent among those not on its box.
    dist = jnp.linalg.norm(contacts.agent_pos - contacts.agent_pos[focal], axis=-1)
    dist = jnp.where(eligible[focal], dist, jnp.inf)
    rank = jnp.argsort(jnp.argsort(dist))
    recruit = eligible[focal] & (rank < n_recruits) & any_q
    return Moment(
        qualifies=any_q,
        focal=focal.astype(jnp.int32),
        n_recruits=n_recruits,
        recruit=recruit,
        box=box[focal].astype(jnp.int32),
        clearance=own[focal],
        touching=self_touch[focal] & any_q,
    )


def reservoir_init(env_state, obs, budget: int) -> Reservoir:
    """An empty sample of `budget` slots per env, shaped from a batch of states."""
    n_envs = obs.shape[0]

    def slots(x):
        return jnp.repeat(x[:, None], budget, axis=1)

    template = Moment(
        qualifies=jnp.zeros(n_envs, bool),
        focal=jnp.zeros(n_envs, jnp.int32),
        n_recruits=jnp.zeros(n_envs, jnp.int32),
        recruit=jnp.zeros(obs.shape[:2], bool),
        box=jnp.zeros(n_envs, jnp.int32),
        clearance=jnp.zeros(n_envs),
        touching=jnp.zeros(n_envs, bool),
    )
    return Reservoir(
        priority=jnp.full((n_envs, budget), jnp.inf),
        env_state=jax.tree.map(slots, env_state),
        obs=slots(obs),
        moment=jax.tree.map(slots, template),
        step=jnp.zeros((n_envs, budget), jnp.int32),
        events=jnp.zeros(n_envs, jnp.int32),
    )


def reservoir_update(res: Reservoir, moment: Moment, env_state, obs, alive, step, key):
    """Offer each env's candidate at this step to its sample.

    Bottom-K priority sampling: every qualifying moment draws u ~ U(0, 1) and
    replaces the env's largest stored priority when u is smaller (an empty slot
    holds +inf). After any number of moments each env holds a uniform random
    sample, without replacement, of min(K, seen) of them. Frozen envs (`alive`
    False, holding a finished episode's last state) offer nothing.
    """
    n_envs = obs.shape[0]
    rows = jnp.arange(n_envs)
    offered = moment.qualifies & alive
    u = jax.random.uniform(key, (n_envs,))
    slot = jnp.argmax(res.priority, axis=1)
    replace = offered & (u < res.priority[rows, slot])

    def put(buf, new):
        old = buf[rows, slot]
        mask = replace.reshape((-1,) + (1,) * (new.ndim - 1))
        return buf.at[rows, slot].set(jnp.where(mask, new, old))

    return Reservoir(
        priority=put(res.priority, u),
        env_state=jax.tree.map(put, res.env_state, env_state),
        obs=put(res.obs, obs),
        moment=jax.tree.map(put, res.moment, moment),
        step=put(res.step, jnp.full(n_envs, step, jnp.int32)),
        events=res.events + offered.astype(jnp.int32),
    )


def flatten_slots(tree):
    """`(E, K, ...)` -> `(E * K, ...)`: one fork lane per slot, env-major.

    The leading size is explicit: `mjx.Data` has zero-size leaves, for which a
    `-1` cannot be inferred."""
    return jax.tree.map(lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), tree)


def n_columns(n_lanes: int, n_windows: int) -> int:
    """Fork columns that hold `n_lanes` forks, one per manager row."""
    return math.ceil(n_lanes / n_windows)


def pack(worker, manager, keep, n_windows: int):
    """Pack per-lane fork transitions into env columns, kept lanes first.

    `worker`: Transition `(H, L, ...)`; `manager`: Transition `(L, ...)`;
    `keep`: `(L,)` bool, the lanes that train (filled and placed). Returns
    `(worker (n_windows * H, C, ...), manager (n_windows, C, ...), order (C *
    n_windows,), kept (C * n_windows,), dropped)`: column c, row w holds lane
    `order[c * n_windows + w]`, and its worker steps fill rows `[w * H, (w + 1) *
    H)` of column c.

    Unkept positions are masked (`active_mask` 0, `done` set). So is a column
    left with exactly one kept fork, since a stream with fewer than two active
    rows is not normalized; kept lanes come first, so only the last occupied
    column can be such a singleton (`dropped` is then 1, else 0).
    """
    horizon, n_lanes = worker.reward.shape[:2]
    n_cols = n_columns(n_lanes, n_windows)
    pad = n_cols * n_windows - n_lanes
    # Kept lanes first, in lane order, then the rest; padding points at lane 0.
    order = jnp.argsort(~keep, stable=True)
    kept = jnp.concatenate([keep[order], jnp.zeros(pad, bool)])
    order = jnp.concatenate([order, jnp.zeros(pad, order.dtype)])
    per_col = kept.reshape(n_cols, n_windows).sum(axis=1)
    singleton = jnp.repeat(per_col == 1, n_windows)
    dropped = (per_col == 1).sum()
    kept = kept & ~singleton

    def mask(t, kept_rows):
        def expand(x):
            return kept_rows.reshape(kept_rows.shape + (1,) * (x.ndim - kept_rows.ndim))

        return t._replace(
            active_mask=t.active_mask * expand(t.active_mask),
            done=t.done | ~expand(t.done),
        )

    def m_leaf(x):
        # (L, ...) -> (C * W, ...) -> (W, C, ...)
        x = x[order].reshape((n_cols, n_windows) + x.shape[1:])
        return jnp.swapaxes(x, 0, 1)

    def w_leaf(x):
        # (H, L, ...) -> (H, C, W, ...) -> (W, H, C, ...) -> (W * H, C, ...)
        x = x[:, order].reshape((horizon, n_cols, n_windows) + x.shape[2:])
        x = jnp.moveaxis(x, 2, 0)
        return x.reshape((n_windows * horizon, n_cols) + x.shape[3:])

    m_kept = jnp.swapaxes(kept.reshape(n_cols, n_windows), 0, 1)  # (W, C)
    w_kept = jnp.repeat(m_kept, horizon, axis=0)  # (W * H, C)
    manager = mask(_map_env_leaves(m_leaf, manager), m_kept)
    worker = mask(_map_env_leaves(w_leaf, worker, lead=1), w_kept)
    return worker, manager, order, kept, dropped


def _map_env_leaves(fn, t, lead=0):
    """Apply `fn` to every leaf with a lane axis at position `lead`; the scalar
    `action_mask` placeholder (no lane axis) passes through."""
    return jax.tree.map(lambda x: fn(x) if x.ndim > lead else x, t)


def paired_main_return(team_reward, active, m_done, step, env, horizon, gamma):
    """Each fork's paired main baseline: the main env's discounted team reward
    over the `horizon` steps from the fork's moment, in the same episode.

    `team_reward` / `active`: the main rollout's `(n_steps, E)` per-step team
    reward and live flag; `m_done`: `(n_windows, E)`, an episode ended in that
    window; `step` / `env`: `(L,)` each fork's moment and env. Mid-window
    termination ends the span through `active` (frozen steps are 0). The next
    window belongs to the same episode only if the moment's window did not end
    one, which also covers an episode ending on a window's last step. Returns
    `(g_main (L,), complete (L,))`, `complete` False when the span runs past the
    end of the rollout.
    """
    n_steps = team_reward.shape[0]
    j = jnp.arange(horizon)
    rows = step[:, None] + j[None, :]  # (L, H)
    inside = rows < n_steps
    safe = jnp.minimum(rows, n_steps - 1)
    r = team_reward[safe, env[:, None]]
    live = active[safe, env[:, None]] * inside
    same_window = (step % horizon)[:, None] + j[None, :] < horizon
    continues = ~m_done[step // horizon, env][:, None]
    mask = jnp.cumprod(live, axis=1) * (same_window | continues)
    g_main = (mask * r * gamma ** j[None, :]).sum(axis=1)
    complete = step + horizon <= n_steps
    return g_main, complete


def window_return(team_reward, gamma):
    """`(L,)` discounted raw team reward of `(H, L)` per-step rewards."""
    horizon = team_reward.shape[0]
    return (team_reward * gamma ** jnp.arange(horizon)[:, None]).sum(axis=0)


def diagnostics(res: Reservoir, f_worker, f_manager, valid, recruit_distance, paired,
                live_main_steps, n_agents: int, horizon: int, dropped):
    """Per-update scalars of the heuristic forks, merged into the losses dict.

    `f_worker` / `f_manager`: the per-lane fork Transitions `(H, L, ...)` /
    `(L, ...)` before packing; `valid`: `(L,)` filled and placed;
    `recruit_distance`: `(L,)` world units; `paired`: `(g_fork, g_main,
    complete)` per lane (`window_return`, `paired_main_return`);
    `live_main_steps`: live env-steps of the main rollout; `dropped`: forks
    masked by `pack`'s singleton rule.
    """
    moment = flatten_slots(res.moment)
    filled = jnp.isfinite(flatten_slots(res.priority))
    filled_f = filled.astype(jnp.float32)
    valid_f = valid.astype(jnp.float32)
    n_filled = jnp.maximum(filled_f.sum(), 1.0)
    n_valid = jnp.maximum(valid_f.sum(), 1.0)
    step = flatten_slots(res.step)

    def over(x, w, n):
        return (x.astype(jnp.float32) * w).sum() / n

    live_fork_steps = f_worker.active_mask[..., 0].sum()
    stats = {
        "intervention_events": res.events.sum().astype(jnp.float32),
        "intervention_event_rate": res.events.sum() / jnp.maximum(live_main_steps, 1.0),
        "intervention_attempted": filled_f.sum(),
        "intervention_budget_frac": filled_f.mean(),
        "intervention_valid_frac": valid_f.sum() / n_filled,
        "intervention_dropped_singleton": jnp.asarray(dropped, jnp.float32),
        "intervention_n_recruits_mean": over(moment.n_recruits, filled_f, n_filled),
        "intervention_focal_touching_frac": over(moment.touching, filled_f, n_filled),
        "intervention_focal_clearance": over(moment.clearance, filled_f, n_filled),
        "intervention_window_step_mean": over(step % horizon, filled_f, n_filled),
        "intervention_window_start_frac": over(step % horizon == 0, filled_f, n_filled),
        "intervention_recruit_distance": over(recruit_distance, valid_f, n_valid),
        "intervention_fork_length": live_fork_steps / n_valid,
        "intervention_window_return": over(f_manager.reward, valid_f, n_valid),
        "intervention_worker_progress": (
            (f_worker.reward * f_worker.active_mask).sum()
            / jnp.maximum(f_worker.active_mask.sum(), 1.0)
        ),
        "intervention_sim_steps": live_fork_steps,
        "intervention_stepped_lanes": jnp.asarray(
            f_worker.reward.shape[0] * f_worker.reward.shape[1], jnp.float32
        ),
    }
    for n in range(1, n_agents):
        tried = filled & (moment.n_recruits == n)
        stats[f"intervention_attempted_n{n}"] = tried.sum().astype(jnp.float32)
        stats[f"intervention_valid_frac_n{n}"] = (tried & valid).sum() / jnp.maximum(
            tried.sum(), 1
        )
    g_fork, g_main, complete = paired
    pair_f = (valid & complete).astype(jnp.float32)
    n_pair = jnp.maximum(pair_f.sum(), 1.0)
    stats.update(
        {
            "intervention_g_fork": over(g_fork, pair_f, n_pair),
            "intervention_g_main": over(g_main, pair_f, n_pair),
            "intervention_paired_gain": over(g_fork - g_main, pair_f, n_pair),
            "intervention_paired": pair_f.sum(),
        }
    )
    return stats
