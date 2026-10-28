"""Tests for the pure-JAX rover domain (environments/rover).

Pinned to CPU: the env is tiny, and sharing a GPU with a training job turns
cuSolver / OOM errors into failures that read like assertion errors.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from environments.rover import (
    BaseRoverEnv,
    ObservationModel,
    POIModel,
    RewardModel,
    RewardOutput,
    RoverEnvState,
    RoverState,
    SectorDensityObservation,
    StaticPOIs,
)
from environments.rover.config import (
    list_variants,
    load_env_config,
    load_rover_env,
    make_rover_env,
)
from environments.rover.policies import make_scripted_coalition_policy, random_policy
from environments.rover.rollout import rollout


@pytest.fixture(autouse=True)
def _run_on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def make_env(n_agents=2, entries=None, observation=None, reward=None, **kwargs):
    entries = entries or [{"position": [5.0, 5.0], "coupling": 2}]
    return BaseRoverEnv(
        n_agents,
        StaticPOIs(entries, radius=1.0, value=1.0, rate=0.05),
        observation,
        reward,
        world_size=(10.0, 10.0),
        **kwargs,
    )


def state_at(env, positions, headings=None, remaining=None):
    _, state = env.reset(jax.random.PRNGKey(0))
    positions = jnp.asarray(positions, dtype=jnp.float32)
    headings = (
        jnp.zeros(positions.shape[0]) if headings is None
        else jnp.asarray(headings, dtype=jnp.float32)
    )
    pois = state.pois
    if remaining is not None:
        pois = dataclasses.replace(pois, remaining=jnp.asarray(remaining, jnp.float32))
    return RoverEnvState(RoverState(positions, headings), pois, state.t)


def stay(n_agents):
    """No turn, zero speed."""
    return jnp.tile(jnp.array([0.0, -1.0]), (n_agents, 1))


# ---------------------------------------------------------------- interface


def test_reset_and_step_jit_and_vmap():
    env = load_rover_env("rover_2poi_c2")
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    obs, state = jax.jit(jax.vmap(env.reset))(keys)
    assert obs.shape == (8, env.n_agents, env.observation_dim) == (8, 4, 16)
    actions = jax.random.uniform(jax.random.PRNGKey(1), (8, 4, 2), minval=-1, maxval=1)
    obs, state, reward, terminated, truncated, info = jax.jit(jax.vmap(env.step))(
        state, actions
    )
    assert obs.shape == (8, 4, 16)
    assert reward.shape == terminated.shape == truncated.shape == (8,)
    assert info["agent_rewards"].shape == (8, 4)
    assert info["task_reward"].shape == (8,)
    assert np.all(np.isfinite(np.asarray(obs)))
    assert np.all(np.asarray(state.t) == 1)
    assert (env.action_dim, env.discrete) == (2, False)


def test_reset_spawns_inside_the_spawn_region():
    env = make_env(n_agents=64, spawn_region=(4.0, 6.0, 1.0, 2.0))
    _, state = env.reset(jax.random.PRNGKey(3))
    pos, heading = np.asarray(state.rovers.position), np.asarray(state.rovers.heading)
    assert np.all((pos[:, 0] >= 4.0) & (pos[:, 0] <= 6.0))
    assert np.all((pos[:, 1] >= 1.0) & (pos[:, 1] <= 2.0))
    assert np.all((heading >= -math.pi) & (heading < math.pi))
    np.testing.assert_array_equal(np.asarray(state.pois.remaining), [1.0])


# -------------------------------------------------------------- observation


def test_sectors_are_heading_relative():
    env = make_env()  # R = arena diagonal
    R = env.world_diagonal
    ahead = env.observe(state_at(env, [[5, 5], [7, 5]]))
    np.testing.assert_allclose(ahead[0, :8], np.eye(8)[0] * (1 - 2 / R), rtol=1e-6)
    # the teammate sees rover 0 directly behind it
    np.testing.assert_allclose(ahead[1, :8], np.eye(8)[4] * (1 - 2 / R), rtol=1e-6)
    # +90 degrees in the world with heading 0 is the rover's left: sector 2
    left = env.observe(state_at(env, [[5, 5], [5, 7]]))
    assert int(np.argmax(left[0, :8])) == 2
    # turning the rover +90 degrees puts the teammate ahead on its right: sector 6
    turned = env.observe(state_at(env, [[5, 5], [7, 5]], headings=[math.pi / 2, 0.0]))
    assert int(np.argmax(turned[0, :8])) == 6


def test_sector_zero_spans_plus_minus_half_a_sector():
    env = make_env()
    for degrees, sector in [(22.0, 0), (23.0, 1), (-22.0, 0), (-23.0, 7)]:
        a = math.radians(degrees)
        obs = env.observe(state_at(env, [[5, 5], [5 + 2 * math.cos(a), 5 + 2 * math.sin(a)]]))
        assert int(np.argmax(obs[0, :8])) == sector, degrees


def test_presence_is_a_distance_weighted_share_within_the_radius():
    env = make_env(n_agents=4, observation=SectorDensityObservation(sensor_radius=3.0))
    # teammates ahead at 1.0 and 2.5; the one at 3.5 is out of range
    obs = env.observe(state_at(env, [[5, 5], [6, 5], [7.5, 5], [8.5, 5]]))
    expected = ((1 - 1 / 3) + (1 - 2.5 / 3)) / 3
    np.testing.assert_allclose(obs[0, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(obs[0, 1:8], 0.0)


def test_co_located_teammates_are_sensed_and_presence_reaches_one():
    env = make_env(n_agents=3)
    obs = env.observe(state_at(env, [[5, 5], [5, 5], [5, 5]]))
    np.testing.assert_allclose(obs[:, 0], 1.0)  # self excluded by index, not distance
    assert float(obs.max()) <= 1.0


def test_used_up_pois_are_not_sensed():
    env = make_env(entries=[{"position": [7.0, 5.0], "coupling": 2}])
    R = env.world_diagonal
    live = env.observe(state_at(env, [[5, 5], [1, 1]], remaining=[0.4]))
    np.testing.assert_allclose(live[0, 8:], np.eye(8)[0] * (1 - 2 / R), rtol=1e-6)
    used = env.observe(state_at(env, [[5, 5], [1, 1]], remaining=[0.0]))
    np.testing.assert_allclose(used[:, 8:], 0.0)


def test_observations_stay_in_unit_interval_over_random_rollouts():
    for variant in list_variants():
        env = load_rover_env(variant)

        def episode(key):
            obs, state = env.reset(key)

            def body(carry, k):
                state, lo, hi = carry
                a = jax.random.uniform(k, (env.n_agents, 2), minval=-1, maxval=1)
                obs, state, *_ = env.step(state, a)
                return (state, jnp.minimum(lo, obs.min()), jnp.maximum(hi, obs.max())), None

            (_, lo, hi), _ = jax.lax.scan(
                body, (state, obs.min(), obs.max()), jax.random.split(key, env.max_steps)
            )
            return lo, hi

        lo, hi = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(0), 16))
        assert float(lo.min()) >= 0.0 and float(hi.max()) <= 1.0, variant


# ----------------------------------------------------------------- dynamics


def test_turn_then_move_at_the_set_speed():
    env = make_env(max_speed=0.5, max_turn=math.pi / 4)
    state = state_at(env, [[5, 5], [2, 2]])
    _, nxt, *_ = env.step(state, jnp.array([[1.0, 1.0], [0.0, -1.0]]))
    np.testing.assert_allclose(nxt.rovers.heading[0], math.pi / 4, atol=1e-6)
    step = 0.5 * math.sqrt(0.5)
    np.testing.assert_allclose(nxt.rovers.position[0], [5 + step, 5 + step], atol=1e-6)
    np.testing.assert_allclose(nxt.rovers.position[1], [2, 2])  # speed action -1 = stop
    # half speed, turning right by half the maximum
    _, half, *_ = env.step(state, jnp.array([[-0.5, 0.0], [0.0, -1.0]]))
    np.testing.assert_allclose(half.rovers.heading[0], -math.pi / 8, atol=1e-6)
    np.testing.assert_allclose(
        half.rovers.position[0],
        [5 + 0.25 * math.cos(math.pi / 8), 5 - 0.25 * math.sin(math.pi / 8)],
        atol=1e-6,
    )
    # out-of-range actions are clipped to [-1, 1]
    _, clipped, *_ = env.step(state, jnp.array([[5.0, 9.0], [0.0, -3.0]]))
    np.testing.assert_allclose(clipped.rovers.position, nxt.rovers.position)


def test_heading_wraps_and_walls_clip():
    env = make_env(max_speed=0.5, max_turn=math.pi / 4)
    state = state_at(env, [[9.9, 5.0], [5.0, 0.1]], headings=[3.0, -math.pi / 2])
    _, nxt, _, terminated, *_ = env.step(state, jnp.array([[1.0, -1.0], [0.0, 1.0]]))
    expected = (3.0 + math.pi / 4 + math.pi) % (2 * math.pi) - math.pi
    np.testing.assert_allclose(nxt.rovers.heading[0], expected, atol=1e-6)
    np.testing.assert_allclose(nxt.rovers.position[1], [5.0, 0.0], atol=1e-6)
    _, wall, _, terminated, *_ = env.step(
        state_at(env, [[9.9, 5.0], [5.0, 5.0]]), jnp.array([[0.0, 1.0], [0.0, -1.0]])
    )
    np.testing.assert_allclose(wall.rovers.position[0], [10.0, 5.0])
    assert not bool(terminated)  # walls never end the episode


# ------------------------------------------------------------------- reward


def test_reward_needs_the_coupling_and_is_split_among_observers():
    env = make_env(n_agents=3)
    _, _, r, _, _, info = env.step(state_at(env, [[5, 5], [1, 1], [9, 9]]), stay(3))
    assert float(r) == 0.0 and float(info["poi_remaining"][0]) == 1.0
    # the radius is inclusive: (6, 5) is exactly 1.0 from the POI
    _, _, r, _, _, info = env.step(state_at(env, [[5, 5], [6, 5], [9, 9]]), stay(3))
    np.testing.assert_allclose(r, 0.05)
    np.testing.assert_allclose(info["agent_rewards"], [0.025, 0.025, 0.0])
    # a third rover adds nothing to the payout, only splits it
    _, _, r, _, _, info = env.step(state_at(env, [[5, 5], [6, 5], [4.5, 5]]), stay(3))
    np.testing.assert_allclose(r, 0.05)
    np.testing.assert_allclose(info["agent_rewards"], [0.05 / 3] * 3, rtol=1e-6)
    np.testing.assert_allclose(info["agent_rewards"].sum(), r, rtol=1e-6)


def test_poi_pays_a_steady_rate_until_used_up():
    env = make_env(max_steps=100)
    state = state_at(env, [[5, 5], [5.5, 5]])
    rewards, terminated = [], []
    for _ in range(25):
        _, state, r, term, trunc, _ = env.step(state, stay(2))
        rewards.append(float(r))
        terminated.append(bool(term))
        assert not (bool(term) and bool(trunc))
    np.testing.assert_allclose(rewards[:20], 0.05, rtol=1e-5)
    assert rewards[20:] == [0.0] * 5
    np.testing.assert_allclose(sum(rewards), 1.0, atol=1e-5)
    assert terminated.index(True) == 19  # exactly value / rate coupled steps
    assert float(state.pois.remaining[0]) == 0.0


def test_termination_needs_every_poi_used_up():
    env = make_env(entries=[{"position": [5.0, 5.0], "coupling": 2},
                            {"position": [1.0, 1.0], "coupling": 2}])
    _, _, _, terminated, *_ = env.step(
        state_at(env, [[5, 5], [5, 5]], remaining=[0.05, 1.0]), stay(2)
    )
    assert not bool(terminated)
    _, _, _, terminated, *_ = env.step(
        state_at(env, [[5, 5], [5, 5]], remaining=[0.05, 0.0]), stay(2)
    )
    assert bool(terminated)


def test_truncation_at_max_steps_is_exclusive_with_termination():
    env = make_env(max_steps=5)
    state = state_at(env, [[1, 1], [9, 9]])
    flags = []
    for _ in range(5):
        _, state, _, term, trunc, _ = env.step(state, stay(2))
        flags.append((bool(term), bool(trunc)))
    assert flags == [(False, False)] * 4 + [(False, True)]
    # the last budgeted step ends both ways at once: report termination only
    env = make_env(max_steps=1)
    _, _, _, term, trunc, _ = env.step(state_at(env, [[5, 5], [5, 5]], remaining=[0.05]), stay(2))
    assert (bool(term), bool(trunc)) == (True, False)


# ------------------------------------------------------------ pluggability


class DriftingPOIs(POIModel):
    """One POI that moves +0.5 in x every step."""

    def init(self, key, env):
        return StaticPOIs([{"position": [2.0, 5.0], "coupling": 1}]).init(key, env)

    def step(self, pois, rovers, t):
        return dataclasses.replace(pois, position=pois.position + jnp.array([0.5, 0.0]))


class OwnPosition(ObservationModel):
    obs_dim = 2

    def __call__(self, env, state):
        return state.rovers.position / jnp.asarray(env.world_size)


class NegativeDistance(RewardModel):
    """Per-agent reward: minus the distance to the first POI; never terminates."""

    def __call__(self, env, rovers, pois):
        d = jnp.linalg.norm(rovers.position - pois.position[0], axis=-1)
        return RewardOutput(-d, pois, jnp.array(False), {"distance": d})


def test_components_are_pluggable():
    env = BaseRoverEnv(2, DriftingPOIs(), OwnPosition(), NegativeDistance())
    assert env.observation_dim == 2 and env.n_pois == 1
    obs, state = env.reset(jax.random.PRNGKey(0))
    np.testing.assert_allclose(obs, state.rovers.position / 10.0)
    _, state2, r, terminated, _, info = env.step(state, stay(2))
    np.testing.assert_allclose(state2.pois.position[0], [2.5, 5.0])
    assert r.shape == (2,)
    np.testing.assert_allclose(r, -info["distance"])
    np.testing.assert_allclose(info["task_reward"], r.sum())
    assert not bool(terminated)


def test_an_observation_model_with_the_wrong_declared_width_is_rejected():
    class Wrong(OwnPosition):
        obs_dim = 3

    with pytest.raises(ValueError, match="obs_dim"):
        BaseRoverEnv(2, DriftingPOIs(), Wrong())


# ------------------------------------------------------------------- config


def test_every_rover_variant_composes_and_builds():
    variants = list_variants()
    assert "rover_2poi_c2" in variants and len(variants) >= 3
    default = load_env_config("rover_2poi_c2")
    for variant in variants:
        cfg = load_env_config(variant)
        env = make_rover_env(cfg)
        assert env.observation_dim == 16
        # children override only the POIs (and the team size); dynamics are inherited
        assert cfg["dynamics"] == default["dynamics"]
        assert cfg["observation"] == default["observation"]


def test_default_variant_is_two_pois_of_coupling_two():
    env = load_rover_env("rover_2poi_c2")
    pois = env.pois.init(jax.random.PRNGKey(0), env)
    assert env.n_pois == 2
    np.testing.assert_array_equal(np.asarray(pois.coupling), [2, 2])


def test_hydra_overrides_reach_the_env():
    env = load_rover_env("rover_2poi_c2", overrides=["env.n_agents=6", "env.max_steps=40"])
    assert (env.n_agents, env.max_steps) == (6, 40)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c.update(bogus=1), "unknown rover env keys"),
        (lambda c: c["dynamics"].update(max_sped=1.0), "unknown dynamics keys"),
        (lambda c: c["observation"].update(type="lidar"), "unknown observation type"),
        (lambda c: c.update(environment="multi_box_push"), "environment must be"),
        (lambda c: c["pois"]["entries"][0].update(cupling=2), "unknown keys"),
        (lambda c: c["pois"]["entries"][0].update(position=[11.0, 5.0]), "outside"),
        (lambda c: c["pois"]["entries"][0].update(coupling=0), "coupling"),
        (lambda c: c.update(spawn_region=[4.0, 12.0, 4.0, 6.0]), "spawn_region"),
    ],
)
def test_bad_config_raises(edit, message):
    cfg = load_env_config("rover_2poi_c2")
    edit(cfg)
    with pytest.raises(ValueError, match=message):
        make_rover_env(cfg)


def test_a_poi_needing_more_rovers_than_exist_warns():
    with pytest.warns(UserWarning, match="can never pay"):
        make_env(n_agents=2, entries=[{"position": [5.0, 5.0], "coupling": 3}])


# ----------------------------------------------------------------- rollouts


def test_scripted_team_uses_up_every_poi_and_rollout_freezes_after_the_end():
    env = load_rover_env("rover_2poi_c2")
    policy = make_scripted_coalition_policy(env)
    traj = jax.jit(jax.vmap(lambda k: rollout(env, policy, k)))(
        jax.random.split(jax.random.PRNGKey(0), 4)
    )
    np.testing.assert_allclose(traj.episode_return, 2.0, atol=1e-5)
    done = np.asarray(traj.done)
    assert done[:, -1].all()
    for e in range(4):
        end = int(np.argmax(done[e]))  # step index at which the episode ended
        assert np.all(done[e, end:])
        assert np.all(np.asarray(traj.rewards[e, end + 1:]) == 0.0)
        frozen = np.asarray(traj.positions[e, end + 1:])
        np.testing.assert_array_equal(frozen, np.broadcast_to(frozen[0], frozen.shape))
        np.testing.assert_array_equal(np.asarray(traj.poi_remaining[e, end + 1]), [0.0, 0.0])


def test_rollout_vmaps_over_a_population_of_policy_parameters():
    """Evolutionary-algorithm usage: one fitness per parameter set."""
    env = load_rover_env("rover_2poi_c2")

    def linear_policy(params, key, obs, state):
        return jnp.tanh(obs @ params)

    population = jax.random.normal(jax.random.PRNGKey(0), (5, env.observation_dim, 2))
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    fitness = jax.jit(jax.vmap(lambda p, k: rollout(env, linear_policy, k, p).episode_return))(
        population, keys
    )
    assert fitness.shape == (5,) and np.all(np.isfinite(np.asarray(fitness)))


def test_rollouts_are_deterministic_given_the_key():
    env = load_rover_env("rover_2poi_c2")
    run = jax.jit(lambda k: rollout(env, random_policy, k))
    a, b = run(jax.random.PRNGKey(7)), run(jax.random.PRNGKey(7))
    np.testing.assert_array_equal(np.asarray(a.positions), np.asarray(b.positions))


# ------------------------------------------------------------ visualization


def test_every_variant_writes_arena_and_trajectory_images(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from environments.rover.visualize import plot_arena, plot_trajectory, save_figure

    for variant in list_variants():
        env = load_rover_env(variant)
        arena = save_figure(plot_arena(env, variant), tmp_path / f"{variant}.png")
        policy = make_scripted_coalition_policy(env)
        traj = jax.jit(lambda k: rollout(env, policy, k))(jax.random.PRNGKey(0))
        trajectory = save_figure(plot_trajectory(env, traj, variant), tmp_path / f"{variant}_t.png")
        for path in (arena, trajectory):
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
