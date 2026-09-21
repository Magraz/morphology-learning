"""Local sector-count semantics and their observation/wrapper integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from environments.mjx_suite.macro_wrapper import (
    COMMITMENT_FEAT_DIM,
    AsyncMacroMJX,
    MacroState,
    SyncMacroMJX,
)
from environments.mjx_suite.multi_box_push_mjx import MultiBoxPushMJX
from environments.mjx_suite.observation import MJXObservationBuilder, OBS_DIM


@pytest.fixture(autouse=True)
def _run_on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _builder(n_agents, **kwargs):
    return MJXObservationBuilder(
        None,
        n_agents=n_agents,
        n_objects=0,
        world_width=30,
        world_height=30,
        velocity_norm=10,
        neighbor_detection_range=3,
        agent_radius=0.4,
        force_multiplier=100,
        sector_sensor_radius=10,
        **kwargs,
    )


def test_count_distinguishes_equal_centroids_and_ignores_team_size():
    one = jnp.array([[0, 0], [6, 0]], dtype=jnp.float32)
    four = jnp.array([[0, 0], [4, 0], [5, 0], [7, 0], [8, 0]], dtype=jnp.float32)
    p_one, c_one = _builder(len(one))._density_sensors_and_counts(one, None)
    p_four, c_four = _builder(len(four))._density_sensors_and_counts(four, None)
    np.testing.assert_allclose(p_one[0], p_four[0])
    assert float(p_four[0, 7]) == pytest.approx(0.4)
    assert float(c_one[0, 7]) == 0.25
    assert float(c_four[0, 7]) == 1.0

    # Additional distant teammates must not change the count's normalization.
    bigger_team = jnp.concatenate([four, jnp.array([[30, 0], [40, 0]])])
    _, bigger_counts = _builder(len(bigger_team))._density_sensors_and_counts(
        bigger_team, None
    )
    np.testing.assert_array_equal(bigger_counts[0], c_four[0])


def test_counts_use_matching_sectors_strict_radius_and_exclude_self():
    # One teammate at the middle of each sector, plus one just inside the
    # radius, one exactly on it, and one outside. +x belongs to sector 7.
    angles = np.deg2rad(np.arange(8) * 45 + 45)
    neighbors = 4 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    positions = jnp.asarray(
        np.vstack([[0, 0], neighbors, [9.5, 0], [10, 0], [10.5, 0]]),
        dtype=jnp.float32,
    )
    builder = _builder(len(positions))
    proximity, counts = builder._density_sensors_and_counts(positions, None)
    np.testing.assert_array_equal(counts[0], [0.25] * 7 + [0.5])
    np.testing.assert_allclose(proximity[0, :7], 0.6, atol=1e-6)
    np.testing.assert_array_equal(proximity[:, 8:], 0)

    _, alone = _builder(1)._density_sensors_and_counts(jnp.zeros((1, 2)), None)
    np.testing.assert_array_equal(alone, np.zeros((1, 8)))


def test_counts_are_not_clipped_and_survive_jit_vmap():
    positions = jnp.array([[0, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]],
                          dtype=jnp.float32)
    builder = _builder(len(positions))
    features = jax.jit(jax.vmap(lambda p: builder._density_sensors_and_counts(p, None)))
    # Translation and teammate ordering do not change the focal agent's sensors.
    permuted = positions[jnp.array([0, 5, 3, 1, 4, 2])] + 20
    proximity, counts = features(jnp.stack([positions, permuted]))
    assert counts.dtype == jnp.float32
    np.testing.assert_array_equal(counts[:, 0, 7], [1.25, 1.25])
    np.testing.assert_array_equal(counts[0, 0], counts[1, 0])
    np.testing.assert_allclose(proximity[0, 0], proximity[1, 0])


@pytest.fixture(scope="module")
def env_and_state():
    with jax.default_device(jax.devices("cpu")[0]):
        env = MultiBoxPushMJX(n_agents=3, n_objects=1)
        obs, state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        yield env, obs, state


def test_full_observation_appends_counts_preserving_original_fields(env_and_state):
    env, obs, state = env_and_state
    current = env.obs_builder
    legacy = MJXObservationBuilder(
        current.model,
        **{name: getattr(current, name) for name in (
            "n_agents", "n_objects", "world_width", "world_height", "velocity_norm",
            "neighbor_detection_range", "agent_radius", "force_multiplier",
            "sector_sensor_radius", "lidar_range",
        )},
        agent_of_geom=current._agent_of_geom,
        object_of_geom=current._object_of_geom,
    )

    @jax.jit
    def legacy_obs(s):
        boxes, yaw = env._box_pose(s.data)
        return legacy.build(
            s.data, env._agent_pos(s.data), env._agent_vel(s.data),
            box_pos=boxes, box_yaw=yaw, box_half=env._box_half,
            goal_coord=env.target_y, delivered=s.delivered,
        )

    assert legacy.obs_dim == OBS_DIM == 40
    assert env.observation_dim == current.obs_dim == 48
    assert obs.shape == (3, 48)
    assert obs.dtype == jnp.float32
    np.testing.assert_allclose(obs[:, :40], legacy_obs(state), rtol=1e-6, atol=1e-6)
    _, counts = current._density_sensors_and_counts(
        env._agent_pos(state.data), env._box_pose(state.data)[0]
    )
    np.testing.assert_array_equal(obs[:, 40:48], counts)


def test_batched_reset_step_and_sync_wrapper_report_actual_width(env_and_state):
    env, _, _ = env_and_state
    wrapper = SyncMacroMJX(env, macro_len=1)
    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    obs, states = jax.jit(jax.vmap(wrapper.reset))(keys)
    assert obs.shape == (2, 3, wrapper.observation_dim) == (2, 3, 48)
    next_obs, *_ = jax.jit(jax.vmap(wrapper.step))(
        states, jnp.zeros((2, 3), dtype=jnp.int32)
    )
    assert next_obs.shape == obs.shape
    assert bool(jnp.isfinite(next_obs).all())


@pytest.mark.parametrize("augment", [False, True])
def test_async_commitment_follows_the_expanded_base_observation(env_and_state, augment):
    env, obs, state = env_and_state
    wrapper = AsyncMacroMJX(env, augment_obs=augment)
    mstate = MacroState(
        env_state=state,
        skill_idx=jnp.zeros(3, dtype=jnp.int32),
        remaining=jnp.ones(3, dtype=jnp.int32),
        elapsed=jnp.zeros(3, dtype=jnp.int32),
        key=jax.random.PRNGKey(0),
    )
    augmented = jax.jit(wrapper._obs)(obs, mstate)
    expected = 48 + (COMMITMENT_FEAT_DIM if augment else 0)
    assert augmented.shape == (3, wrapper.obs_dim) == (3, expected)
    np.testing.assert_array_equal(augmented[:, :48], obs)
