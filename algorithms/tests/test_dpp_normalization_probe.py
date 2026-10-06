"""Numerical failure cases and parity with the real PPO actor gradient."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from algorithms.mappo_jax.mappo import compute_gae, create_train_state, ppo_update
from algorithms.mappo_jax.network import sample_action
from algorithms.mappo_jax.types import MAPPOConfig, Transition
from algorithms.simplified_feudal_mappo_jax.dpp_normalization_probe import (
    analyze_snapshot,
    load_snapshot,
    make_gradient_functions,
    normalized_advantages,
    save_snapshot,
    write_reports,
)
from flax.serialization import to_bytes


@pytest.fixture(autouse=True)
def _cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_tiny_team_variance_amplifies_bonus_and_floor_bounds_it():
    a = jnp.array([[-1e-10, -1.0], [1e-10, 1.0], [-1e-10, -1.0], [1e-10, 1.0]])
    dpp = jnp.broadcast_to(jnp.array([0., 2e-5, 0., 2e-5])[:, None, None], (4, 2, 2))
    original = normalized_advantages(a, dpp, 1.)
    floored = normalized_advantages(a, dpp, 1., .001)
    assert float(jnp.max(jnp.abs(original["bonus"][:, 0]))) > 900
    assert float(jnp.max(jnp.abs(floored["bonus"][:, 0]))) < .011
    # The large-variance env is unaffected: the floor is applied per env.
    np.testing.assert_array_equal(original["corrected"][:, 1], floored["corrected"][:, 1])
    expected_denom = np.std(np.asarray(a), axis=0, ddof=1) + 1e-8
    np.testing.assert_allclose(original["denominator"], expected_denom, rtol=1e-6)


def test_zero_coefficient_and_constant_bonus_do_not_change_centered_advantage():
    a = jnp.array([[-2., 1.], [0., -1.], [2., 0.], [0., 0.]])
    varying = jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2)
    for floor in (0., 10.):
        zero = normalized_advantages(a, varying, 0., floor)
        constant = normalized_advantages(a, jnp.full((4, 2, 2), 3.), 1., floor)
        np.testing.assert_allclose(zero["corrected"], zero["team"], atol=1e-7)
        np.testing.assert_allclose(constant["corrected"], constant["team"], atol=1e-7)


def _batch():
    cfg = MAPPOConfig(hidden_dim=8, n_epochs=1, n_minibatches=1, ent_coef=.03)
    ts = create_train_state(jax.random.PRNGKey(1), cfg, 3, 4, 2, False)
    obs = jax.random.normal(jax.random.PRNGKey(2), (4, 2, 2, 3))
    action, lp = sample_action(jax.random.PRNGKey(3), ts.actor_ts.apply_fn,
                               ts.actor_ts.params, obs.reshape(-1, 3), False)
    reward = jnp.array([[.1, -.2], [.5, .3], [-.3, .8], [.2, -.1]])
    m = Transition(obs=obs, global_state=jnp.ones((4, 2, 4)),
                   action=action.reshape(4, 2, 2, 2), log_prob=lp.reshape(4, 2, 2),
                   reward=reward, done=jnp.zeros((4, 2), bool), value=jnp.zeros((4, 2)),
                   team_reward=reward, active_mask=jnp.ones((4, 2, 2)),
                   action_mask=jnp.zeros((4,)))
    last = jnp.zeros(2)
    a, _ = compute_gae(m.reward, m.value, m.done, last, cfg.gamma, cfg.gae_lambda)
    dpp = jax.random.normal(jax.random.PRNGKey(4), (4, 2, 2))
    return cfg, ts, m, last, a, dpp


@pytest.mark.parametrize("coef", [0., .1, 1.])
def test_probe_gradient_matches_ppo_with_frozen_parameters(coef):
    cfg, ts, m, last, a, dpp = _batch()
    # Record the sum of real minibatch gradients while holding parameters fixed.
    # Equal-size minibatches make their mean equal to the full-batch gradient.
    def init(params):
        return jax.tree.map(jnp.zeros_like, params)

    def update(grads, state, params=None):
        return jax.tree.map(jnp.zeros_like, grads), jax.tree.map(jnp.add, state, grads)

    recorder = optax.GradientTransformation(init, update)
    actor_ts = ts.actor_ts.replace(tx=recorder, opt_state=recorder.init(ts.actor_ts.params))
    ts = ts._replace(actor_ts=actor_ts)
    gradient_fn, _, entropy_grad = make_gradient_functions(
        actor_ts.apply_fn, actor_ts.params, m.obs, m.action, m.log_prob,
        m.active_mask, cfg, squash=True,
    )
    terms = normalized_advantages(a, dpp, coef)
    _, policy_grad = gradient_fn(actor_ts.params, terms["corrected"])
    expected = jax.tree.map(lambda p, e: p + cfg.ent_coef * e, policy_grad, entropy_grad)
    result, _ = ppo_update(ts, jax.random.PRNGKey(5), m, last, cfg, False,
                          squash=True, advantage_correction=-coef * jnp.maximum(dpp, 0.))
    total_ts = m.obs.shape[0] * m.obs.shape[1]
    mb_size = max(1, (total_ts // cfg.n_minibatches) // m.obs.shape[2])
    n_mb = total_ts // mb_size
    actual = jax.tree.map(lambda x: x / n_mb, result.actor_ts.opt_state)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(x, y, rtol=2e-5, atol=2e-6)


def test_snapshot_replay_reports_streams_and_zero_norms(tmp_path):
    cfg, ts, m, _, a, dpp = _batch()
    meta = {"batch": "test", "model": "dpp", "trial": "0", "rollout": 0, "seed": 10,
            "manager_config": dataclasses.asdict(cfg), "action_bound": "tanh"}
    snapshot = {"metadata": meta, "actor_params": to_bytes(ts.actor_ts.params),
                "obs": np.asarray(m.obs), "action": np.asarray(m.action),
                "log_prob": np.asarray(m.log_prob), "active_mask": np.asarray(m.active_mask),
                "a_team": np.asarray(a), "dpp": np.asarray(dpp)}
    save_snapshot(tmp_path / "snapshot.npz", snapshot)
    loaded = load_snapshot(tmp_path / "snapshot.npz")
    assert loaded["metadata"] == meta
    assert loaded["actor_params"] == snapshot["actor_params"]
    np.testing.assert_array_equal(loaded["dpp"], snapshot["dpp"])
    rows, streams = analyze_snapshot(loaded, [0., 1.], [0., 1.])
    assert len(rows) == 4 and len(streams) == 16
    for row in rows:
        if row["coef"] == 0:
            assert row["bonus_grad_norm"] == 0
            assert row["sign_flip_frac"] == 0
            assert row["policy_grad_cos_team"] == pytest.approx(1.)
    write_reports(tmp_path, rows, streams, [meta])
    assert (tmp_path / "streams.csv").read_text().count("\n") == 17
    assert "NaN" not in (tmp_path / "report.json").read_text()

    loaded["a_team"] = np.zeros_like(a)
    loaded["dpp"] = np.zeros_like(dpp)
    zero_rows, _ = analyze_snapshot(loaded, [0.], [0.])
    assert zero_rows[0]["bonus_to_team_grad_norm"] is None
    assert zero_rows[0]["policy_grad_cos_team"] is None
    assert zero_rows[0]["bonus_to_team_rms"] is None
