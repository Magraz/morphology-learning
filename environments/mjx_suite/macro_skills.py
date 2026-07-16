"""Scripted JAX skills for the async macro layer (`macro_wrapper.py`).

Each skill is a pure function ``fn(env, env_state) -> (A, 2)`` giving the
low-level action **every** agent would take under that skill; the wrapper then
gathers row ``i`` from the skill agent ``i`` is currently committed to. Computing
all skills for all agents each step is a few jnp ops and keeps the gather
vectorized and jit-friendly.

These are deliberately *scripted* rather than the 4 frozen torch actors in
``algorithms/hierarchical/skills.py``: porting those is a large yak-shave that is
irrelevant to the async-difference-reward claim, and being deterministic and
stateless these make the oracle counterfactual (`algorithms/difference_rewards/
oracle.py`) exact — a forked rollout replays identically.

The goal band spans the **top** wall, so "toward the goal" is ``+y`` (see
``MultiBoxPushMJX.target_y``).
"""

import jax.numpy as jnp

from environments.mjx_suite.multi_box_push_mjx import _AGENT_RADIUS

_EPS = 1e-6
_STAGE_GAP = 0.6  # staging point sits this far below the box surface
_STAGE_TOL = 0.7  # within this of the staging point, switch from approach to push
_CONTACT_GAP = 0.5  # contact skill stops this far from the box surface
_WALL_MARGIN = 3.0  # start steering inward this far from a wall
_WALL_WEIGHT = 3.0  # inward steer strength; must dominate the scatter repulsion


def _unit(v: jnp.ndarray) -> jnp.ndarray:
    """Row-wise normalize; a zero row stays zero (no NaN)."""
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + _EPS)


def _nearest_box(env, env_state):
    """(A,2) nearest-box centre and (A,) its half-extent, per agent."""
    agent_pos = env._agent_pos(env_state.data)  # (A, 2)
    box_pos, _ = env._box_pose(env_state.data)  # (O, 2)
    dist = jnp.linalg.norm(agent_pos[:, None, :] - box_pos[None, :, :], axis=-1)
    idx = jnp.argmin(dist, axis=1)  # (A,)
    return agent_pos, box_pos[idx], env._box_half[idx]


def _local_centroid(env, agent_pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Centroid of the agents each agent can actually *sense*, and a has-neighbor mask.

    Locality is load-bearing, not a detail. A global centroid gives every agent —
    including one parked in a far corner, unable to reach anything — a causal
    channel into every other agent's behaviour. That manufactures long-range
    credit out of nothing: the oracle correctly reports |D_i| ~ 0.16 for a
    provably irrelevant agent (5% of max|D|, larger than genuinely engaged
    agents), which would poison the bias study. It also contradicts the
    observation model, where agents only perceive others within
    `sector_sensor_radius` (`observation.py`). Skills must not reach further than
    the agents can see.

    Returns (A,2) centroid (self's own position where there are no neighbours, so
    the resulting steer is zero) and an (A,1) bool mask.
    """
    radius = env.sector_sensor_radius
    dist = jnp.linalg.norm(agent_pos[:, None, :] - agent_pos[None, :, :], axis=-1)
    in_range = (dist <= radius) & ~jnp.eye(agent_pos.shape[0], dtype=bool)
    weight = in_range.astype(agent_pos.dtype)  # (A, A)
    count = weight.sum(axis=1, keepdims=True)  # (A, 1)
    centroid = (weight[:, :, None] * agent_pos[None, :, :]).sum(axis=1) / jnp.maximum(
        count, 1.0
    )
    has_neighbor = count > 0
    return jnp.where(has_neighbor, centroid, agent_pos), has_neighbor


def skill_push(env, env_state) -> jnp.ndarray:
    """Converge on a staging point below the nearest box, then push it toward +y.

    Generalizes ``multi_box_push_mjx.scripted_push_action`` from a fixed box to
    each agent's nearest box.
    """
    agent_pos, box, half = _nearest_box(env, env_state)
    stage = box - jnp.stack([jnp.zeros_like(half), half + _STAGE_GAP], axis=-1)
    to_stage = stage - agent_pos
    close = jnp.linalg.norm(to_stage, axis=-1, keepdims=True) < _STAGE_TOL
    push = jnp.broadcast_to(jnp.array([0.0, 1.0]), agent_pos.shape)
    return jnp.where(close, push, _unit(to_stage))


def skill_contact(env, env_state) -> jnp.ndarray:
    """Approach the nearest box and stop on its surface (no pushing)."""
    agent_pos, box, half = _nearest_box(env, env_state)
    to_box = box - agent_pos
    dist = jnp.linalg.norm(to_box, axis=-1, keepdims=True)
    touching = dist < (half[:, None] + _CONTACT_GAP)
    return jnp.where(touching, jnp.zeros_like(to_box), _unit(to_box))


def _wall_repulsion(env, agent_pos: jnp.ndarray) -> jnp.ndarray:
    """(A,2) inward steer, ramping 0 -> 1 over `_WALL_MARGIN` before each wall.

    Without this, `skill_scatter` walks agents straight into the boundary, which
    `MultiBoxPushMJX.step` treats as termination-with-zero-reward. That would
    make an agent's difference reward dominated by "did it end the episode"
    rather than by its contribution — a confound the bias study cannot tolerate.
    """
    lo = env.boundary_thickness + _AGENT_RADIUS
    hi = env.world_width - lo
    push_in = jnp.clip((lo + _WALL_MARGIN - agent_pos) / _WALL_MARGIN, 0.0, 1.0)
    push_out = jnp.clip((agent_pos - (hi - _WALL_MARGIN)) / _WALL_MARGIN, 0.0, 1.0)
    return push_in - push_out


def skill_scatter(env, env_state) -> jnp.ndarray:
    """Move away from the centroid of *sensed* agents, steering off the walls."""
    agent_pos = env._agent_pos(env_state.data)
    centroid, has_neighbor = _local_centroid(env, agent_pos)
    away = jnp.where(has_neighbor, _unit(agent_pos - centroid), 0.0)
    return _unit(away + _WALL_WEIGHT * _wall_repulsion(env, agent_pos))


def skill_rendezvous(env, env_state) -> jnp.ndarray:
    """Move toward the centroid of *sensed* agents; idle when alone."""
    agent_pos = env._agent_pos(env_state.data)
    centroid, has_neighbor = _local_centroid(env, agent_pos)
    return jnp.where(has_neighbor, _unit(centroid - agent_pos), 0.0)


def null_action(env, env_state) -> jnp.ndarray:
    """The counterfactual default ``c_i``: apply no force.

    Deliberately **not** in ``SKILL_FNS`` — the policy cannot select it; it only
    ever replaces an agent in a counterfactual rollout.
    """
    return jnp.zeros((env.n_agents, 2), dtype=jnp.float32)


# Index == the high-level discrete action, mirroring the box2d
# `algorithms/hierarchical/skills.py::SKILL_ORDER` convention.
SKILL_ORDER = ["contact", "push", "scatter", "rendezvous"]
SKILL_FNS = (skill_contact, skill_push, skill_scatter, skill_rendezvous)
N_SKILLS = len(SKILL_ORDER)


def all_skill_actions(env, env_state) -> jnp.ndarray:
    """(K, A, 2) — every skill's action for every agent."""
    return jnp.stack([fn(env, env_state) for fn in SKILL_FNS])
