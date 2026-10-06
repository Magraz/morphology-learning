"""Support geometry, persistence, untouched focal goals and censored events."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorithms.simplified_feudal_mappo_jax import counterfactual as cf
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.dpp_simulator_probe import METRICS, make_evaluator
from algorithms.simplified_feudal_mappo_jax.dpp_support_probe import (
    EXTRA_METRICS, _commitments, _event_stats, make_extra_metrics, make_intervention,
    select_box, staging_targets,
)
from algorithms.tests.test_dpp_simulator_probe import (
    Data, State, ToyEnv, toy_config, toy_policy, toy_state, toy_ts,
)


@pytest.fixture(autouse=True)
def _cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class GeometryEnv(ToyEnv):
    n_agents = 3
    _centre = jnp.array([0., 0.])
    _extent = jnp.array([30., 30.])
    _box_half = jnp.array([1.5, 1.5])
    _coupling = jnp.array([2, 2])
    world_height = 30.
    target_y = 12.

    def goal_state(self, state):
        return state.data.pos / self._extent

    def _box_pose(self, data):
        return jnp.array([[0., 0.], [6., 0.]]), jnp.zeros(2)

    def _touch_matrix(self, pos, boxes, yaw):
        delta = jnp.maximum(jnp.abs(pos[:, None] - boxes[None]) - self._box_half[None, :, None], 0.)
        return jnp.linalg.norm(delta, axis=-1) <= .45


def geometry_state(pos=None, delivered=None):
    pos = jnp.array([[0., -1.9], [0., 3.], [3., -3.]]) if pos is None else jnp.asarray(pos)
    delivered = jnp.array([False, False]) if delivered is None else jnp.asarray(delivered)
    return State(Data(pos), jnp.array(0), delivered, jnp.array([12., 12.]))


def context(target=0, commitment=1, enabled=True):
    return {"target": jnp.array(target), "commitment": jnp.array(commitment),
            "enabled": jnp.array(enabled), "box": jnp.array(0), "box_valid": jnp.array(True),
            "recruit_mask": jnp.array([False, True, False])}


def test_one_window_focal_position_is_the_production_counterfactual():
    env, state = GeometryEnv(), geometry_state()
    pos = env.goal_state(state)
    goals = pos + jnp.array([[.02, .03], [-.02, .01], [.01, .01]])
    intervene = make_intervention(env, .15)
    result = intervene(state, goals, 0, 0, context())
    support = cf.support_offsets(pos, 0, .15)
    expected = wp.waypoint_from_action(pos, support, .15, "clip")
    np.testing.assert_allclose(result[1], expected[1], atol=1e-7)
    np.testing.assert_array_equal(result[jnp.array([0, 2])], goals[jnp.array([0, 2])])
    np.testing.assert_array_equal(intervene(state, goals, 0, 1, context()), goals)


@pytest.mark.parametrize("target", [0, 1, 2])
def test_all_targets_keep_focal_and_unselected_goals_and_obey_bounds(target):
    env, state = GeometryEnv(), geometry_state()
    pos = env.goal_state(state)
    goals = pos + jnp.array([[.02, .03], [-.02, .01], [.01, .01]])
    result = make_intervention(env, .15)(state, goals, 0, 0, context(target=target))
    np.testing.assert_array_equal(result[jnp.array([0, 2])], goals[jnp.array([0, 2])])
    assert bool((jnp.abs(result[1] - pos[1]) <= .1500001).all())
    assert bool((jnp.abs(result[1]) <= .5000001).all())


def test_waypoint_target_uses_the_managers_current_focal_waypoint():
    env, state = GeometryEnv(), geometry_state()
    pos = env.goal_state(state)
    goals = pos.at[0].set(jnp.array([.1, .1]))
    result = make_intervention(env, .15)(state, goals, 0, 1, context(target=1, commitment=2))
    expected = wp.waypoint_from_action(pos, (goals[0] - pos) / .15, .15, "clip")
    np.testing.assert_allclose(result[1], expected[1], atol=1e-7)


def test_box_choice_prefers_touched_live_coupled_box_and_handles_none():
    env = GeometryEnv()
    index, valid = select_box(env, geometry_state(), 0)
    assert bool(valid) and int(index) == 0
    index, valid = select_box(env, geometry_state(delivered=[True, False]), 0)
    assert bool(valid) and int(index) == 1
    _, valid = select_box(env, geometry_state(delivered=[True, True]), 0)
    assert not bool(valid)


def test_staging_routes_around_box_and_pushes_when_close():
    env = GeometryEnv()
    mask = jnp.array([False, True, False])
    stage = staging_targets(env, geometry_state(), 0, mask, 0) * env._extent
    assert stage[1, 0] > 1.5  # Side bypass, not a waypoint through the box.
    assert stage[1, 1] == pytest.approx(3.)
    near = geometry_state(pos=[[0., -1.9], [.45, -2.1], [3., -3.]])
    pushing = staging_targets(env, near, 0, mask, 0) * env._extent
    assert pushing[1, 1] > near.data.pos[1, 1]


def test_staging_releases_on_delivery_and_focal_position_replans():
    env, state = GeometryEnv(), geometry_state()
    goals = env.goal_state(state)
    intervene = make_intervention(env, .15)
    delivered = geometry_state(delivered=[True, False])
    np.testing.assert_array_equal(intervene(delivered, goals, 0, 1,
                                            context(target=2, commitment=4)), goals)
    moved = geometry_state(pos=[[3., -1.9], [0., 3.], [3., -3.]])
    a = intervene(state, goals, 0, 0, context(commitment=2))
    b = intervene(moved, goals, 0, 1, context(commitment=2))
    assert not np.array_equal(a[1], b[1])
    np.testing.assert_array_equal(intervene(moved, goals, 0, 2, context(commitment=2)), goals)


class SlowToyEnv(ToyEnv):
    _centre = jnp.zeros(2)
    _extent = jnp.ones(2)
    _box_half = jnp.array([.02])
    world_height = 1.
    target_y = .4

    def step(self, state, action):
        return super().step(state, .4 * action)


def test_persistence_enables_delayed_arrival_and_target_delivery():
    env, cfg, state = SlowToyEnv(), toy_config(), toy_state()
    evaluate, _, _ = make_evaluator(env, cfg, 6, toy_policy(),
        make_intervention(env, cfg.waypoint_radius), make_extra_metrics(env, cfg.waypoint_radius))
    states = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    goals = states.data.pos
    common = {"target": jnp.zeros(2, int), "commitment": jnp.array([1, 2]),
              "enabled": jnp.ones(2, bool), "box": jnp.zeros(2, int), "box_valid": jnp.ones(2, bool),
              "recruit_mask": jnp.array([[False, True], [False, True]])}
    keys = jnp.stack([jax.random.PRNGKey(0)] * 2)
    result = evaluate(toy_ts(), states, goals, goals, jnp.zeros(2, int), keys, common)
    assert result["metrics"][0, -1, METRICS.index("task_return")] == 0
    assert result["metrics"][1, -1, METRICS.index("discounted_return")] == pytest.approx(100 * .9 ** 2)
    delivery = EXTRA_METRICS.index("target_delivery_events")
    arrival = EXTRA_METRICS.index("target_recruit_touch_steps")
    assert result["extra_first_steps"][0, -1, arrival] == -1
    assert result["extra_first_steps"][1, -1, arrival] == 3
    assert result["extra_metrics"][1, -1, delivery] == 1


def test_event_means_exclude_censored_cases_and_commitments_include_anchor():
    probability, delay = _event_stats(np.array([-1, 3, -1, 5]))
    assert probability == .5 and delay == 4
    assert _event_stats(np.array([-1, -1])) == (0., None)
    assert _commitments("4,2,2") == [1, 2, 4]
