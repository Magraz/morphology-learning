"""Seam tests for `MultiBoxPushMJX`'s opt-in compact global state.

These pin the joints where a mistake is SILENT rather than loud:

- the hook is OFF by default, so no existing arm's critic input width moves and
  no saved `models_*.msgpack` stops loading (the single most important test here);
- the declared `global_state_dim` is the width actually emitted;
- the block layout is what the docstring says (a swapped block would train fine
  and mean something else);
- `delivered` really reaches the vector — delivery LATCHES, so position alone
  cannot distinguish "banked its +100" from "about to pay it";
- the goal distance is recomputed from `data`, not read off the stale-by-name
  `prev_box_goal_dist`;
- every block lands at a comparable scale, which is the design's justification;
- the macro wrapper refuses the hook instead of silently keeping concat-obs.

Run: `uv run pytest algorithms/tests/test_mjx_global_state.py -q`
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from environments.mjx_suite.multi_box_push_mjx import (
    GLOBAL_STATE_AGENT_FEATURES,
    GLOBAL_STATE_BOX_FEATURES,
    MultiBoxPushMJX,
    compact_global_state_dim,
)

A, O = 4, 2


@pytest.fixture(autouse=True)
def _run_on_cpu():
    """Pin to CPU: tiny tests, and sharing a busy GPU fails like an assertion."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.fixture(scope="module")
def env_on():
    return MultiBoxPushMJX(n_agents=A, n_objects=O, use_global_state=True)


@pytest.fixture(scope="module")
def state_on(env_on):
    _, state = jax.jit(env_on.reset)(jax.random.PRNGKey(0))
    return state


def test_hook_is_off_by_default():
    """THE regression gate: every existing arm must stay hookless."""
    env = MultiBoxPushMJX(n_agents=A, n_objects=O)
    assert not hasattr(env, "global_state")
    assert not hasattr(env, "global_state_dim")
    assert env.global_state_enabled is False
    # And it must be off on the CLASS, or binding it on one instance would have
    # flipped every other arm through the trainers' `hasattr` switch.
    assert not hasattr(MultiBoxPushMJX, "global_state")


def test_declared_width_is_the_emitted_width(env_on, state_on):
    expected = A * GLOBAL_STATE_AGENT_FEATURES + O * GLOBAL_STATE_BOX_FEATURES
    assert env_on.global_state_dim == expected == compact_global_state_dim(A, O)
    assert env_on.global_state(state_on).shape == (expected,)


def test_vmapped_and_jitted(env_on):
    n = 3
    _, states = jax.vmap(env_on.reset)(jax.random.split(jax.random.PRNGKey(1), n))
    gs = jax.jit(jax.vmap(env_on.global_state))(states)
    assert gs.shape == (n, env_on.global_state_dim)
    assert bool(jnp.isfinite(gs).all())


def test_block_layout_round_trips(env_on, state_on):
    """Decode each block back to world units — catches a swapped layout."""
    gs = np.asarray(env_on.global_state(state_on))
    centre = np.array([env_on.world_center_x, env_on.world_center_y], np.float32)
    extent = np.array([env_on.world_width, env_on.world_height], np.float32)

    agents = gs[: A * GLOBAL_STATE_AGENT_FEATURES].reshape(A, GLOBAL_STATE_AGENT_FEATURES)
    np.testing.assert_allclose(
        agents[:, :2] * extent + centre,
        np.asarray(env_on._agent_pos(state_on.data)), rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(
        agents[:, 2:4] * env_on.velocity_norm,
        np.asarray(env_on._agent_vel(state_on.data)), rtol=1e-5, atol=1e-5,
    )

    boxes = gs[A * GLOBAL_STATE_AGENT_FEATURES :].reshape(O, GLOBAL_STATE_BOX_FEATURES)
    box_pos, box_yaw = env_on._box_pose(state_on.data)
    np.testing.assert_allclose(
        boxes[:, :2] * extent + centre, np.asarray(box_pos), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        boxes[:, 2] * env_on.world_height,
        np.asarray(env_on._box_goal_distance(box_pos)), rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(boxes[:, 3], np.asarray(state_on.delivered), atol=0)
    touch = env_on._touch_matrix(env_on._agent_pos(state_on.data), box_pos, box_yaw)
    np.testing.assert_allclose(
        boxes[:, 4] * A, np.asarray(touch.sum(axis=0)), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        boxes[:, 5] * A, np.asarray(env_on._coupling), rtol=1e-5, atol=1e-5
    )


def test_delivered_reaches_the_vector(env_on, state_on):
    """Delivery latches, so position alone cannot encode it. Same `data`, both ways."""
    undelivered = dataclasses.replace(state_on, delivered=jnp.zeros(O, dtype=bool))
    delivered = dataclasses.replace(state_on, delivered=jnp.ones(O, dtype=bool))
    a = env_on.global_state(undelivered)
    b = env_on.global_state(delivered)
    assert not bool(jnp.allclose(a, b))
    # ...and ONLY the delivered column moved.
    off = A * GLOBAL_STATE_AGENT_FEATURES
    da = np.asarray(a[off:]).reshape(O, GLOBAL_STATE_BOX_FEATURES)
    db = np.asarray(b[off:]).reshape(O, GLOBAL_STATE_BOX_FEATURES)
    moved = np.where(np.abs(da - db).max(axis=0) > 0)[0]
    assert moved.tolist() == [3], moved


def test_goal_distance_is_recomputed_not_read_from_prev(env_on, state_on):
    """`prev_box_goal_dist` is named "prev" — reading it invites a one-step skew."""
    garbage = dataclasses.replace(state_on, 
        prev_box_goal_dist=jnp.full((O,), 999.0, dtype=jnp.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(env_on.global_state(state_on)),
        np.asarray(env_on.global_state(garbage)),
    )


def test_every_block_is_on_a_comparable_scale(env_on):
    """The design's actual justification: no block dominates the first Tanh."""
    n = 4
    _, states = jax.vmap(env_on.reset)(jax.random.split(jax.random.PRNGKey(2), n))
    step = jax.jit(jax.vmap(env_on.step))
    v_gs = jax.jit(jax.vmap(env_on.global_state))
    key = jax.random.PRNGKey(3)
    worst = 0.0
    for _ in range(50):
        key, sub = jax.random.split(key)
        actions = jax.random.normal(sub, (n, A, 2))
        _, states, *_ = step(states, actions)
        gs = v_gs(states)
        assert bool(jnp.isfinite(gs).all())
        worst = max(worst, float(jnp.abs(gs).max()))
    assert worst < 3.0, worst


def test_macro_wrapper_refuses_the_hook():
    """It re-declares its metadata and has no __getattr__, so it cannot forward."""
    from environments.mjx_suite.macro_wrapper import SyncMacroMJX

    SyncMacroMJX(MultiBoxPushMJX(n_agents=A, n_objects=1))  # hook off: fine
    with pytest.raises(NotImplementedError, match="global_state"):
        SyncMacroMJX(MultiBoxPushMJX(n_agents=A, n_objects=1, use_global_state=True))


@pytest.mark.parametrize("circular", [False, True])
def test_goal_state_to_world_inverts_goal_state(circular):
    """The feudal goal video draws waypoints/headings via this inverse; a wrong
    centre or extent would draw every mark in the wrong place, silently."""
    if circular:
        from environments.mjx_suite.multi_box_multi_goal_push_mjx import (
            MultiBoxMultiGoalPushMJX as Env,
        )
    else:
        Env = MultiBoxPushMJX
    env = Env(n_agents=A, n_objects=O)
    _, state = jax.jit(env.reset)(jax.random.PRNGKey(3))
    world = env.goal_state_to_world(env.goal_state(state))
    np.testing.assert_allclose(world, np.asarray(env._agent_pos(state.data)), atol=1e-5)
