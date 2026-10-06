"""Counterfactual geometry, paired RNG, continuation and endpoint semantics."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from algorithms.mappo_jax.mappo import ActorCriticTrainState
from algorithms.mappo_jax.types import MAPPOConfig
from algorithms.simplified_feudal_mappo_jax import counterfactual as cf
from algorithms.simplified_feudal_mappo_jax.dpp_simulator_probe import (
    GROUPS, METRICS, StratifiedReservoir, make_candidates, make_evaluator,
    random_matched_offsets, state_group, state_schema, summarize,
)
from algorithms.simplified_feudal_mappo_jax.trainer import HierTrainState, Policy
from algorithms.simplified_feudal_mappo_jax.types import FeudalConfig


@pytest.fixture(autouse=True)
def _cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class Data(NamedTuple):
    pos: jax.Array


class State(NamedTuple):
    data: Data
    t: jax.Array
    delivered: jax.Array
    prev_box_goal_dist: jax.Array


class ToyEnv:
    n_agents = 2
    max_steps = 6
    _coupling = jnp.array([2])

    def goal_state(self, state):
        return state.data.pos

    def _agent_pos(self, data):
        return data.pos

    def _box_pose(self, data):
        return jnp.zeros((1, 2)), jnp.zeros(1)

    def _touch_matrix(self, pos, boxes, yaw):
        return (jnp.linalg.norm(pos - boxes[0], axis=-1) < .05)[:, None]

    def step(self, state, action):
        pos = state.data.pos + action
        met = (jnp.linalg.norm(pos, axis=-1) < .05).all()
        delivered = state.delivered | met
        reward = 100. * (delivered & ~state.delivered).sum()
        next_state = State(Data(pos), state.t + 1, delivered, jnp.array([0.2]))
        return pos, next_state, reward, delivered.all(), next_state.t >= self.max_steps, {"task_reward": reward}


def toy_policy(assist_later=False, noise=0.):
    def observe(obs, states):
        return obs.reshape(obs.shape[0], -1), states.data.pos

    def decide(ts, obs, gs, pos, state, key):
        return (jnp.zeros_like(pos) if assist_later else pos,)

    def act(ts, obs, pos, waypoint, k, key):
        return (waypoint - pos + noise * jax.random.normal(key, pos.shape),)

    return Policy(observe, decide, act)


def toy_state(t=0):
    pos = jnp.array([[0., 0.], [.2, 0.]])
    return State(Data(pos), jnp.array(t), jnp.array([False]), jnp.array([.2]))


def toy_ts(value=0.):
    critic = TrainState.create(apply_fn=lambda p, x: jnp.full(x.shape[:-1], p),
                              params=jnp.array(value), tx=optax.sgd(0.))
    return HierTrainState(worker=None, manager=ActorCriticTrainState(None, critic))


def toy_config():
    worker = MAPPOConfig(gamma=.9)
    manager = MAPPOConfig(gamma=.9 ** 2, gae_lambda=1.)
    return FeudalConfig(worker, manager, goal_horizon=2, waypoint_radius=.3)


def run_toy(goals, state=None, policy=None, value=0., eval_steps=6, keys=None):
    state = toy_state() if state is None else state
    evaluate, labels, stops = make_evaluator(ToyEnv(), toy_config(), eval_steps,
                                             policy or toy_policy())
    states = jax.tree.map(lambda x: x[None], state)
    keys = jax.random.split(jax.random.PRNGKey(0), 1) if keys is None else keys
    result = evaluate(toy_ts(value), states, state.data.pos[None], goals[None], jnp.array([0]), keys)
    return result, labels, stops


def test_candidates_match_training_recruitment_and_keep_focal_slots():
    pos = jnp.array([[0., 0.], [.1, .02], [-.2, .1]])
    offsets = jnp.array([[.3, -.2], [.1, .4], [-.3, -.1]])
    joints, labels, counts = make_candidates(pos, offsets, 0, .15, 2, jax.random.PRNGKey(2))
    assert labels == ["original", "toward", "away", "random", "toward", "away", "random"]
    np.testing.assert_array_equal(joints[0], offsets)
    for c in range(1, len(labels)):
        mask = np.asarray(cf.recruit_mask(pos, 0, counts[c]))
        np.testing.assert_array_equal(joints[c][~mask], offsets[~mask])
        if labels[c] == "toward":
            np.testing.assert_array_equal(joints[c], cf.dpp_joint(offsets, pos, 0, counts[c], .15))
        if labels[c] == "random":
            np.testing.assert_allclose(np.linalg.norm(joints[c][mask], axis=-1),
                                       np.linalg.norm(joints[c-2][mask], axis=-1), atol=1e-7)


def test_random_control_is_distance_matched_and_legal_near_arena_edges():
    pos = jnp.array([[-.49, -.49], [.49, .49], [0., .4], [0., 0.]])
    toward = cf.support_offsets(pos, 0, .15)
    for seed in range(10):
        random = random_matched_offsets(pos, toward, jax.random.PRNGKey(seed), .15)
        np.testing.assert_allclose(np.linalg.norm(random, axis=-1), np.linalg.norm(toward, axis=-1), atol=1e-7)
        assert bool((jnp.abs(pos + .15 * random) <= .5000001).all())
        assert bool((jnp.abs(random) <= 1.000001).all())


def test_intervention_lasts_one_window_then_original_manager_returns():
    goals = toy_state().data.pos
    result, labels, stops = run_toy(goals, policy=toy_policy(assist_later=True))
    assert stops == [2, 6]
    assert result["metrics"][0, 0, METRICS.index("task_return")] == 0
    assert result["metrics"][0, 1, METRICS.index("discounted_return")] == pytest.approx(100 * .9 ** 2)
    assert result["metrics"][0, 1, METRICS.index("deliveries")] == 1


def test_terminal_return_counted_once_and_never_bootstrapped():
    result, _, _ = run_toy(jnp.zeros((2, 2)), value=4.)
    assert result["metrics"][0, -1, METRICS.index("task_return")] == 100
    assert result["metrics"][0, -1, METRICS.index("live_steps")] == 1
    assert result["metrics"][0, -1, METRICS.index("episode_end_events")] == 1
    assert result["training_gae"][0] == pytest.approx(96.)
    assert bool(result["episode_complete"][0])


def test_original_timeout_and_source_time_are_preserved():
    state = toy_state(t=5)
    result, _, _ = run_toy(state.data.pos, state=state, value=4.)
    assert result["metrics"][0, -1, METRICS.index("live_steps")] == 1
    assert result["metrics"][0, -1, METRICS.index("task_return")] == 0
    # Training bootstraps timeout at its actual primitive step; physical return does not.
    assert result["training_gae"][0] == pytest.approx(.9 * 4. - 4.)
    assert bool(result["episode_complete"][0])


def test_artificial_cutoff_is_not_a_terminal_and_same_keys_pair_exactly():
    state = toy_state()
    goals = state.data.pos
    a, _, _ = run_toy(goals, policy=toy_policy(noise=.001), value=4., eval_steps=2)
    b, _, _ = run_toy(goals, policy=toy_policy(noise=.001), value=4., eval_steps=2)
    np.testing.assert_array_equal(a["metrics"], b["metrics"])
    assert not bool(a["episode_complete"][0])
    assert a["metrics"][0, -1, METRICS.index("episode_end_events")] == 0
    assert a["training_gae"][0] == pytest.approx(.9 ** 2 * 4. - 4.)


def test_per_replicate_rng_is_independent_of_chunk_size():
    state = toy_state()
    evaluate, _, _ = make_evaluator(ToyEnv(), toy_config(), 2, toy_policy(noise=.001))
    states = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    combined = evaluate(toy_ts(4.), states, states.data.pos, states.data.pos, jnp.zeros(2, int), keys)
    for i in range(2):
        single = evaluate(toy_ts(4.), jax.tree.map(lambda x: x[i:i+1], states),
                          states.data.pos[i:i+1], states.data.pos[i:i+1], jnp.zeros(1, int), keys[i:i+1])
        np.testing.assert_array_equal(combined["metrics"][i], single["metrics"][0])


def test_reservoir_fills_missing_strata_without_inventing_states():
    reservoir = StratifiedReservoir(8, 0)
    for i in range(40):
        reservoir.offer("other", lambda i=i: {"id": i, "group": "other"})
    samples = reservoir.selected()
    assert len(samples) == 8 and len({s["id"] for s in samples}) == 8
    assert reservoir.seen["other"] == 40
    assert all(reservoir.seen[g] == 0 for g in GROUPS if g != "other")
    assert int(state_group(ToyEnv(), toy_state())[0]) == GROUPS.index("waiting_alone")


def test_state_schema_is_stable_under_host_transfer_and_batching():
    state = toy_state()
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    assert state_schema(state) == state_schema(jax.device_get(state))
    assert state_schema(state) == state_schema(batched, batched=True)


def test_summary_treats_all_zero_returns_as_ties_not_critic_accuracy():
    rows = [dict(trial="0", source_id=i, source_episode=i, group="other",
                 intervention="toward", recruits=1, horizon="episode", predicted_gain=1e-5,
                 actual_gain=0., training_gae_gain=1e-5, selected_toward=True, dpp_applied=True)
            for i in range(4)]
    summary, rankings = summarize(rows, 1e-4, 0)
    assert all(r["predicted_positive_frac"] == 1 for r in summary)
    assert all(r["actual_tied_frac"] == 1 for r in summary)
    assert all(r["sign_accuracy_decisive"] is None for r in summary)
    assert all(r["spearman"] is None for r in summary)
    assert all(r["positive_predictions_harmful_frac"] == 0 for r in summary)
    assert all(r["positive_predictions_tied_frac"] == 1 for r in summary)
    assert all(r["decisive_count_pairs"] == 0 for r in rankings)
