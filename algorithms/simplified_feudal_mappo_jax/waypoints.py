"""Waypoint geometry and the network inputs of both levels. Pure and jittable.

Positions live in the env's `goal_state` space: each agent's world position as
`(pos - centre) / extent`, so the arena spans roughly [-0.5, 0.5] on each axis.
`radius` (R) is measured in the same units.

Every function takes a leading batch shape in front of `(n_agents, ...)`, so the
training scan (`(n_envs,)`), eval and `view()` share one definition of what each
network reads — a drifted copy of an input layout still runs, it just evaluates a
policy that never trained.
"""

import jax.numpy as jnp

# `goal_state` is normalized by the arena extent about its centre.
ARENA_HALF_EXTENT = 0.5

# How the manager's raw Gaussian action is mapped into [-1, 1] per axis.
#   clip — the original. Past +-1 every sample gives the same waypoint, so only
#          the samples inside the range inform that axis. The trained 1a/3o
#          managers sit mostly outside it (14-51% of samples inside).
#   tanh — a squashed Gaussian. Strictly monotonic, and trained with the squashed
#          entropy (`ppo_update(squash=True)`). In a toy check the mean still
#          saturates, but it turns around 1-5x faster than under `clip` once the
#          reward flips. tanh with the plain Gaussian entropy was slower than clip.
ACTION_BOUNDS = ("clip", "tanh")


def validate_action_bound(bound: str) -> None:
    if bound not in ACTION_BOUNDS:
        raise ValueError(
            f"unknown manager_action_bound {bound!r}; expected one of {ACTION_BOUNDS}"
        )


def bounded_action(action, bound):
    """The manager's raw action mapped into [-1, 1] per axis (see ACTION_BOUNDS)."""
    validate_action_bound(bound)
    if bound == "tanh":
        return jnp.tanh(action)
    return jnp.clip(action, -1.0, 1.0)


def waypoint_from_action(pos, action, radius, bound):
    """`(..., N, 2)` waypoints from the manager's raw Gaussian action.

    The action is bounded to [-1, 1] per axis (`bounded_action`) and scaled by R,
    so the manager picks both the direction AND the distance of each agent's next
    target, up to R per axis. The result is clipped to the arena so no waypoint
    is placed outside it.
    """
    offset = radius * bounded_action(action, bound)
    return jnp.clip(pos + offset, -ARENA_HALF_EXTENT, ARENA_HALF_EXTENT)


def goal_error(waypoint, pos, radius):
    """`(..., N, 2)` live error vector to the waypoint, in units of R.

    This is the worker's goal input. It is recomputed every step from the
    current position, so the directive shrinks to 0 as the agent arrives.
    """
    return (waypoint - pos) / radius


def distance_to_waypoint(waypoint, pos, radius):
    """`(..., N)` distance to the waypoint, in units of R."""
    return jnp.linalg.norm(waypoint - pos, axis=-1) / radius


def intrinsic_reward(waypoint, pos, next_pos, radius):
    """`(..., N)` the worker's ONLY reward: the distance to its waypoint closed
    by this step, in units of R.

    `next_pos` must be the successor the action actually produced (read before
    any reset). Summed over a commitment it telescopes to
    `(||w - s_start|| - ||w - s_end||) / R`, which is bounded by the waypoint's
    initial distance, so the worker cannot farm it; it can only collect the
    distance that exists between it and its target.
    """
    return distance_to_waypoint(waypoint, pos, radius) - distance_to_waypoint(
        waypoint, next_pos, radius
    )


def remaining_fraction(k, horizon):
    """Fraction of the commitment left at step `k` of the window (1.0 at k=0)."""
    return (horizon - k) / horizon


# ---------------------------------------------------------------------------
# Network inputs
# ---------------------------------------------------------------------------


def manager_critic_input(global_state, pos):
    """`(B, G + 2N)` — the team view: the env's global state plus every agent's
    position. Positions are appended because they are the frame the manager's
    actions are expressed in, and the default MJX global state (concatenated
    egocentric observations) carries no absolute coordinates."""
    b = pos.shape[0]
    return jnp.concatenate([global_state, pos.reshape(b, -1)], axis=-1)


def manager_actor_input(global_state, pos):
    """`(B, N, G + 2N + N)` — the team view plus a one-hot agent index.

    The manager actor is shared across agents (MAPPO parameter sharing), so the
    one-hot is what lets it assign a DIFFERENT waypoint to each agent from the
    same team view."""
    b, n = pos.shape[:2]
    team = manager_critic_input(global_state, pos)
    team = jnp.broadcast_to(team[:, None, :], (b, n, team.shape[-1]))
    agent_id = jnp.broadcast_to(jnp.eye(n, dtype=team.dtype), (b, n, n))
    return jnp.concatenate([team, agent_id], axis=-1)


# What the manager ACTOR reads (the manager critic always reads the team view).
#   global   — the original: the env's global state + every agent's position,
#              shared by all agents, plus a one-hot agent index.
#   relative — per agent, everything measured FROM that agent, with delivered
#              boxes removed rather than flagged (`manager_actor_input_relative`).
MANAGER_INPUTS = ("global", "relative")


def validate_manager_input(mode: str) -> None:
    if mode not in MANAGER_INPUTS:
        raise ValueError(
            f"unknown manager_input {mode!r}; expected one of {MANAGER_INPUTS}"
        )


def _sorted_by_distance(feat, dist):
    """Reorder `feat` (..., K, F) along K by ascending `dist` (..., K)."""
    order = jnp.argsort(dist, axis=-1)
    return jnp.take_along_axis(feat, order[..., None], axis=-2)


def manager_actor_input_relative(agents, boxes):
    """`(B, N, 4 + 6*O + 4*(N-1) + N)` — each agent's view measured from itself.

    `agents` (B, N, 4) and `boxes` (B, O, 6) are the env's `entity_state` blocks
    (positions in goal_state units, velocities, and per box: goal distance,
    delivered, touch fraction, coupling fraction). Per agent `i`, in order:

      own     (4)          position, velocity
      boxes   (O, 6)       box - own position (2), goal distance, touch
                           fraction, coupling fraction, 1.0 — sorted nearest
                           first; DELIVERED boxes are zeroed and sorted last
      mates   (N-1, 4)     teammate - own position (2), teammate velocity (2),
                           sorted nearest first
      one-hot (N)          agent index, as in the global input

    Why: the `global` input carries absolute box positions and a `delivered`
    flag per box, so the states after a delivery are a separate input region
    that the manager can learn a separate policy for. Measured 2026-09-27 on
    both `_gs` batches: the trained manager still steers toward the nearest box
    when the agent is moved or a box is marked delivered, but not in the states
    it actually visits after a delivery. Here the box to go for next is always
    slot 0, so what was learned on the first box applies to the next one, as it
    does for the flat policy's `nearest_box_vec`. Residual: the zeroed tail
    still reveals HOW MANY boxes are delivered.
    """
    b, n = agents.shape[:2]
    pos, vel = agents[..., :2], agents[..., 2:4]
    box_pos = boxes[..., :2]
    valid = 1.0 - boxes[..., 3]  # (B, O): 1 while undelivered

    rel_box = box_pos[:, None] - pos[:, :, None]  # (B, N, O, 2)
    box_rest = jnp.concatenate(
        [boxes[..., 2:3], boxes[..., 4:6], valid[..., None]], axis=-1
    )  # (B, O, 4): goal distance, touch, coupling, valid
    box_feat = jnp.concatenate(
        [rel_box, jnp.broadcast_to(box_rest[:, None], (b, n) + box_rest.shape[1:])],
        axis=-1,
    ) * valid[:, None, :, None]  # delivered rows -> all zeros
    box_dist = jnp.where(
        valid[:, None] > 0, jnp.linalg.norm(rel_box, axis=-1), jnp.inf
    )
    box_feat = _sorted_by_distance(box_feat, box_dist)  # (B, N, O, 6)

    rel_mate = pos[:, None] - pos[:, :, None]  # (B, N, N, 2): [i, j] = pos_j - pos_i
    mate_feat = jnp.concatenate(
        [rel_mate, jnp.broadcast_to(vel[:, None], (b, n, n, 2))], axis=-1
    )
    mate_dist = jnp.where(
        jnp.eye(n, dtype=bool), jnp.inf, jnp.linalg.norm(rel_mate, axis=-1)
    )
    mate_feat = _sorted_by_distance(mate_feat, mate_dist)[:, :, : n - 1]  # self last

    agent_id = jnp.broadcast_to(jnp.eye(n, dtype=agents.dtype), (b, n, n))
    return jnp.concatenate(
        [agents, box_feat.reshape(b, n, -1), mate_feat.reshape(b, n, -1), agent_id],
        axis=-1,
    )


def worker_actor_input(obs, error, remaining):
    """`(B, N, obs_dim + 2 + 1)` — own observation, live error to own waypoint,
    and the fraction of the commitment left."""
    rem = jnp.full(obs.shape[:-1] + (1,), remaining, dtype=obs.dtype)
    return jnp.concatenate([obs, error, rem], axis=-1)


def worker_critic_input(global_state, error, remaining):
    """`(B, G + 2N + 1)` — centralized: global state, every agent's error, and
    the fraction of the commitment left (the worker's return depends on it,
    since its episode ends when the waypoint expires)."""
    b = error.shape[0]
    rem = jnp.full((b, 1), remaining, dtype=global_state.dtype)
    return jnp.concatenate([global_state, error.reshape(b, -1), rem], axis=-1)


def input_dims(obs_dim, global_state_dim, n_agents, goal_dim, manager_actor_dim=None):
    """Widths of the four network inputs above, for building train states.

    `manager_actor_dim` overrides the manager actor's width; the trainer passes it
    under `manager_input: relative`, whose width it reads off the builder itself.
    """
    if manager_actor_dim is None:
        manager_actor_dim = global_state_dim + goal_dim * n_agents + n_agents
    return {
        "worker_actor": obs_dim + goal_dim + 1,
        "worker_critic": global_state_dim + goal_dim * n_agents + 1,
        "manager_actor": manager_actor_dim,
        "manager_critic": global_state_dim + goal_dim * n_agents,
    }
