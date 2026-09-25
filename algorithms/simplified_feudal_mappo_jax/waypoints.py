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


def waypoint_from_action(pos, action, radius):
    """`(..., N, 2)` waypoints from the manager's raw Gaussian action.

    The action is clipped to [-1, 1] per axis and scaled by R, so the manager
    picks both the direction AND the distance of each agent's next target, up to
    R per axis. The result is clipped to the arena so no waypoint is placed
    outside it.
    """
    offset = radius * jnp.clip(action, -1.0, 1.0)
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


def input_dims(obs_dim, global_state_dim, n_agents, goal_dim):
    """Widths of the four network inputs above, for building train states."""
    return {
        "worker_actor": obs_dim + goal_dim + 1,
        "worker_critic": global_state_dim + goal_dim * n_agents + 1,
        "manager_actor": global_state_dim + goal_dim * n_agents + n_agents,
        "manager_critic": global_state_dim + goal_dim * n_agents,
    }
