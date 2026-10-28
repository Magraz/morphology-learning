"""Seam tests for `rmappo_jax` (on-policy's MAPPO in JAX), on CPU stub envs.

They pin the places where a port of on-policy can be silently wrong:
- the GRU against torch's `nn.GRU` driven by on-policy's `RNNLayer` sequence
  logic (gate order, biases, hidden resets at episode boundaries);
- `ValueNorm` against a transcription of on-policy's class;
- the value and policy objectives;
- recurrent re-evaluation: chunks rebuilt from the stored hidden states and
  masks reproduce the rollout's log-probs and values exactly (PPO ratio 1),
  including an episode that resets mid-chunk;
- batch-wide advantage normalization over active rows;
- persistent rollouts, the time-limit handling, the n_steps rounding, the
  linear learning-rate decay, the rejections and the checkpoint round trip.

Run: `uv run pytest algorithms/tests/test_rmappo_jax.py -q`
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorithms.mappo_jax.mappo import compute_gae
from algorithms.rmappo_jax import value_norm as vn_lib
from algorithms.rmappo_jax.mappo import (
    _optimizer,
    chunk_count,
    gather_minibatch,
    policy_objective,
    prepare_advantages,
    unroll,
    value_objective,
)
from algorithms.rmappo_jax.network import GRULayer, evaluate_action
from algorithms.rmappo_jax.trainer import critic_step, make_train
from algorithms.rmappo_jax.types import (
    Model_Params,
    Params,
    RMAPPOConfig,
    make_config,
)


@pytest.fixture(autouse=True)
def _run_on_cpu():
    # Tiny arrays; sharing the GPU with a training job turns resource errors
    # into failures that read like assertion errors.
    with jax.default_device(jax.devices("cpu")[0]):
        yield


N_AGENTS, OBS_DIM = 3, 5


class ContinuousStub:
    """Continuous actions; random terminations, truncation at `max_steps`."""

    n_agents, observation_dim, action_dim = N_AGENTS, OBS_DIM, 2
    discrete, reward_mode = False, "dense"

    def __init__(self, max_steps=7, termination_rate=0.15):
        self.max_steps = max_steps
        self.termination_rate = termination_rate

    def observe(self, key, t):
        return jax.random.normal(jax.random.fold_in(key, t), (N_AGENTS, OBS_DIM))

    def reset(self, key):
        return self.observe(key, 0), {"t": jnp.int32(0), "key": key}

    def step(self, state, actions):
        t, key = state["t"] + 1, state["key"]
        obs = self.observe(key, t)
        reward = -jnp.sum(actions**2) + obs[0, 0]
        draw = jax.random.uniform(jax.random.fold_in(key, 1000 + t))
        terminated = draw < self.termination_rate
        truncated = (t >= self.max_steps) & ~terminated
        return obs, {"t": t, "key": key}, reward, terminated, truncated, {
            "task_reward": reward
        }


class DiscreteStub:
    """Discrete actions with a legal-action mask, a global-state hook and
    agents that sit out some steps (`info["active"]`)."""

    n_agents, observation_dim, action_dim = N_AGENTS, OBS_DIM, 4
    discrete, max_steps, reward_mode = True, 6, "dense"
    global_state_dim = 7

    def reset(self, key):
        return jax.random.normal(key, (N_AGENTS, OBS_DIM)), {
            "t": jnp.int32(0),
            "key": key,
        }

    def avail_actions(self, state):
        illegal = (state["t"] + jnp.arange(N_AGENTS)) % 4
        return jnp.ones((N_AGENTS, 4)).at[jnp.arange(N_AGENTS), illegal].set(0.0)

    def global_state(self, state):
        return jax.random.normal(
            jax.random.fold_in(state["key"], 50 + state["t"]), (7,)
        )

    def step(self, state, actions):
        t, key = state["t"] + 1, state["key"]
        obs = jax.random.normal(jax.random.fold_in(key, t), (N_AGENTS, OBS_DIM))
        reward = jnp.float32(actions.sum()) * 0.1 + obs[1, 1]
        terminated = jax.random.uniform(jax.random.fold_in(key, 1000 + t)) < 0.2
        truncated = (t >= self.max_steps) & ~terminated
        active = (jnp.arange(N_AGENTS) != (t % N_AGENTS)).astype(jnp.float32)
        return obs, {"t": t, "key": key}, reward, terminated, truncated, {
            "task_reward": reward,
            "active": active,
        }


def _config(**overrides):
    base = dict(
        n_envs=4, n_steps=10, n_total_steps=10 * 4 * 3, n_epochs=2,
        n_minibatches=2, data_chunk_length=5, hidden_size=16, n_eval_episodes=3,
    )
    base.update(overrides)
    return RMAPPOConfig(**base)


def _collect(config, env, seed=0, n_rollouts=1):
    init_fn, collect_fn, update_fn, eval_fn, _ = make_train(config, env)
    runner_state = init_fn(jax.random.PRNGKey(seed))
    for _ in range(n_rollouts):
        runner_state, trajectory, last_value, _ = collect_fn(runner_state)
    return runner_state, trajectory, last_value, (update_fn, eval_fn)


# --------------------------------------------------------------------- GRU


def _torch_rnn_layer(gru, norm, x, hxs, masks):
    """on-policy's `RNNLayer.forward` sequence branch, transcribed.

    x (T, N, in), hxs (N, layers, H), masks (T, N). Splits the sequence where
    any mask is 0 (after t=0) and resets the hidden state by the mask at each
    segment start.
    """
    import torch

    T = x.shape[0]
    has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze(-1).cpu()
    has_zeros = [0] + (has_zeros + 1).numpy().tolist() + [T]
    hxs = hxs.transpose(0, 1)
    outputs = []
    for start, end in zip(has_zeros[:-1], has_zeros[1:]):
        temp = hxs * masks[start].view(1, -1, 1)
        rnn_scores, hxs = gru(x[start:end], temp.contiguous())
        outputs.append(rnn_scores)
    out = torch.cat(outputs, dim=0)
    return norm(out), hxs.transpose(0, 1)


@pytest.mark.parametrize("layers", [1, 2])
def test_gru_matches_torch_rnn_layer_with_episode_resets(layers):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    T, N, IN, H = 9, 4, 6, 5
    gru = torch.nn.GRU(IN, H, num_layers=layers)
    for name, p in gru.named_parameters():  # nonzero biases exercise both
        torch.nn.init.normal_(p, std=0.5)
    norm = torch.nn.LayerNorm(H)

    x = torch.randn(T, N, IN)
    h0 = torch.randn(N, layers, H)
    masks = torch.ones(T, N)
    masks[3, 1] = 0.0  # one env restarts mid-sequence
    masks[6, :] = 0.0  # every env restarts
    masks[0, 2] = 0.0  # a reset at the first step

    with torch.no_grad():
        expected, expected_h = _torch_rnn_layer(gru, norm, x, h0, masks)

    params = {"params": {"rnn_norm": {"scale": jnp.ones(H), "bias": jnp.zeros(H)}}}
    for k in range(layers):
        params["params"][f"gru_{k}"] = {
            "weight_ih": jnp.asarray(getattr(gru, f"weight_ih_l{k}").detach().numpy().T),
            "weight_hh": jnp.asarray(getattr(gru, f"weight_hh_l{k}").detach().numpy().T),
            "bias_ih": jnp.asarray(getattr(gru, f"bias_ih_l{k}").detach().numpy()),
            "bias_hh": jnp.asarray(getattr(gru, f"bias_hh_l{k}").detach().numpy()),
        }
    layer = GRULayer(hidden_size=H, recurrent_N=layers)

    def step(h, xs):
        out, h = layer.apply(params, xs[0], h, xs[1])
        return h, out

    final_h, out = jax.lax.scan(
        step, jnp.asarray(h0.numpy()), (jnp.asarray(x.numpy()), jnp.asarray(masks.numpy()))
    )
    np.testing.assert_allclose(np.asarray(out), expected.numpy(), atol=2e-5)
    np.testing.assert_allclose(np.asarray(final_h), expected_h.numpy(), atol=2e-5)


# --------------------------------------------------------------- ValueNorm


class _ReferenceValueNorm:
    """on-policy's ValueNorm (input_shape=1, defaults), transcribed to numpy."""

    def __init__(self, beta=0.99999, epsilon=1e-5):
        self.beta, self.epsilon = beta, epsilon
        self.mean = self.mean_sq = self.debias = 0.0

    def running_mean_var(self):
        d = max(self.debias, self.epsilon)
        mean, mean_sq = self.mean / d, self.mean_sq / d
        return mean, max(mean_sq - mean**2, 1e-2)

    def update(self, x):
        w = self.beta
        self.mean = self.mean * w + x.mean() * (1 - w)
        self.mean_sq = self.mean_sq * w + (x**2).mean() * (1 - w)
        self.debias = self.debias * w + (1 - w)

    def normalize(self, x):
        m, v = self.running_mean_var()
        return (x - m) / np.sqrt(v)

    def denormalize(self, x):
        m, v = self.running_mean_var()
        return x * np.sqrt(v) + m


def test_value_norm_matches_on_policy():
    rng = np.random.default_rng(0)
    ref, state = _ReferenceValueNorm(), vn_lib.init_value_norm()
    probe = rng.normal(size=16).astype(np.float32)
    # Before any update: mean 0, variance at the 1e-2 floor.
    np.testing.assert_allclose(
        np.asarray(vn_lib.denormalize(state, probe)), 0.1 * probe, rtol=1e-6
    )
    for _ in range(30):
        batch = rng.normal(loc=40.0, scale=15.0, size=64).astype(np.float32)
        ref.update(batch.astype(np.float64))
        state = vn_lib.update(state, jnp.asarray(batch))
        np.testing.assert_allclose(
            np.asarray(vn_lib.normalize(state, probe)), ref.normalize(probe), rtol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(vn_lib.denormalize(state, probe)),
            ref.denormalize(probe),
            rtol=1e-4,
        )


def test_huber_and_mse_match_on_policy_util():
    e = jnp.array([-25.0, -10.0, -3.0, 0.0, 4.0, 10.0, 12.0])
    d = 10.0
    expected = [10 * (25 - 5), 50.0, 4.5, 0.0, 8.0, 50.0, 10 * (12 - 5)]
    np.testing.assert_allclose(np.asarray(vn_lib.huber_loss(e, d)), expected)
    np.testing.assert_allclose(np.asarray(vn_lib.mse_loss(e)), np.asarray(e) ** 2 / 2)


# -------------------------------------------------------------- objectives


def test_value_objective_takes_the_larger_of_clipped_and_unclipped():
    config = _config(use_huber_loss=False)
    old = jnp.array([0.0, 0.0])
    values = jnp.array([1.0, -1.0])  # both moved 1.0 > eps_clip 0.2
    target = jnp.array([1.0, 1.0])
    active = jnp.ones(2)
    # Row 0 moved toward the target: unclipped error 0, clipped 0.8 -> max 0.32.
    # Row 1 moved away: unclipped 2.0 -> 2.0, clipped 1.2 -> 0.72; max 2.0.
    loss = value_objective(values, old, target, active, config)
    np.testing.assert_allclose(float(loss), (0.8**2 / 2 + 2.0**2 / 2) / 2, rtol=1e-6)
    unclipped = value_objective(
        values, old, target, active, _config(use_huber_loss=False, use_clipped_value_loss=False)
    )
    np.testing.assert_allclose(float(unclipped), (0.0 + 2.0) / 2, rtol=1e-6)


def test_inactive_rows_do_not_enter_the_objectives():
    config = _config()
    active = jnp.array([1.0, 0.0, 1.0])
    log_prob = jnp.array([0.1, -0.3, 0.2])
    entropy = jnp.array([1.0, 7.0, 3.0])
    adv = jnp.array([1.0, 5.0, -2.0])
    base = policy_objective(log_prob, jnp.zeros(3), adv, entropy, active, config)
    edited = policy_objective(
        log_prob.at[1].set(4.0), jnp.zeros(3), adv.at[1].set(-9.0),
        entropy.at[1].set(-3.0), active, config,
    )
    np.testing.assert_allclose(np.asarray(base[:2]), np.asarray(edited[:2]))
    np.testing.assert_allclose(float(base[1]), 2.0)  # entropy over active rows

    values, old, target = jnp.array([1.0, 2.0, 3.0]), jnp.zeros(3), jnp.zeros(3)
    a = value_objective(values, old, target, active, config)
    b = value_objective(values.at[1].set(50.0), old, target, active, config)
    np.testing.assert_allclose(float(a), float(b))


def test_advantages_are_normalized_over_the_whole_batch_and_active_rows():
    config = _config(use_valuenorm=False)
    _, trajectory, last_value, _ = _collect(config, DiscreteStub())
    normalized, returns, _ = prepare_advantages(trajectory, last_value, None, config)

    advantages, _ = compute_gae(
        trajectory.reward, trajectory.value, trajectory.done, last_value,
        config.gamma, config.gae_lambda,
    )
    adv = np.asarray(advantages, np.float64)
    masked = np.where(np.asarray(trajectory.active_mask) == 0, np.nan, adv)
    assert np.isnan(masked).any()  # the stub really has inactive rows
    expected = (adv - np.nanmean(masked)) / (np.nanstd(masked) + 1e-5)
    np.testing.assert_allclose(np.asarray(normalized), expected, rtol=1e-4, atol=1e-5)


# ---------------------------------------------- recurrent re-evaluation seam


def _reevaluate(train_state, trajectory, config, discrete, masks_override=None):
    """Every chunk rebuilt by the training sampler and run from its stored
    hidden state: `(log_prob, value)` per chunk row, plus the stored ones."""
    ids = jnp.arange(chunk_count(trajectory, config.chunk_length))
    zeros = jnp.zeros_like(trajectory.reward)
    batch = gather_minibatch(trajectory, zeros, zeros, ids, config.chunk_length)
    masks = batch["mask"] if masks_override is None else masks_override(batch["mask"])
    out = unroll(
        train_state.actor_ts.apply_fn, train_state.actor_ts.params,
        batch["obs"], batch["actor_h0"], masks,
    )
    log_prob, _ = evaluate_action(out, batch["action"], discrete, batch["action_mask"])

    def critic(params, x, h, m):
        return train_state.critic_ts.apply_fn(params, x, h, m)

    values = unroll(
        critic, train_state.critic_ts.params, batch["cent_obs"], batch["critic_h0"], masks
    )
    return log_prob, values, batch


@pytest.mark.parametrize("env", [ContinuousStub(), DiscreteStub()], ids=["cont", "disc"])
@pytest.mark.parametrize("recurrent", [True, False], ids=["gru", "mlp"])
def test_chunked_reevaluation_reproduces_the_rollout(env, recurrent):
    config = _config(use_recurrent_policy=recurrent)
    runner_state, trajectory, _, _ = _collect(config, env, n_rollouts=2)
    assert float((trajectory.mask == 0).sum()) > 0  # episodes restart in-rollout
    log_prob, values, batch = _reevaluate(
        runner_state.train_state, trajectory, config, env.discrete
    )
    np.testing.assert_allclose(
        np.asarray(log_prob), np.asarray(batch["old_log_prob"]), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(values), np.asarray(batch["old_value"]), atol=1e-5
    )


def test_ignoring_the_episode_masks_breaks_the_reevaluation():
    """Negative control for the test above: the stored masks are load-bearing."""
    config = _config()
    runner_state, trajectory, _, _ = _collect(config, ContinuousStub(), n_rollouts=2)
    in_chunk_reset = np.asarray(trajectory.mask).reshape(2, 5, -1)[:, 1:] == 0
    assert in_chunk_reset.any()
    log_prob, _, batch = _reevaluate(
        runner_state.train_state, trajectory, config, False,
        masks_override=jnp.ones_like,
    )
    assert np.max(np.abs(np.asarray(log_prob - batch["old_log_prob"]))) > 1e-4


def test_first_minibatch_ratio_is_one():
    config = _config(n_epochs=1, n_minibatches=1)
    runner_state, trajectory, last_value, (update_fn, _) = _collect(
        config, ContinuousStub(), n_rollouts=2
    )
    _, losses = update_fn(runner_state, trajectory, last_value)
    np.testing.assert_allclose(float(losses["ratio"]), 1.0, atol=1e-6)


# ------------------------------------------------------ rollout semantics


def test_rollouts_persist_across_updates():
    config = _config(n_steps=5)
    env = ContinuousStub(max_steps=100, termination_rate=0.0)
    init_fn, collect_fn, update_fn, _, _ = make_train(config, env)
    state = init_fn(jax.random.PRNGKey(0))
    state, first, last_value, _ = collect_fn(state)
    state, _ = update_fn(state, first, last_value)
    carried_h, carried_obs = state.actor_h, state.obs
    state, second, _, _ = collect_fn(state)
    np.testing.assert_array_equal(np.asarray(state.env_state["t"]), 10)
    np.testing.assert_array_equal(np.asarray(second.actor_h[0]), np.asarray(carried_h))
    np.testing.assert_array_equal(np.asarray(second.obs[0]), np.asarray(carried_obs))
    assert float(jnp.abs(carried_h).max()) > 0
    np.testing.assert_array_equal(np.asarray(second.mask), 1.0)


@pytest.mark.parametrize("proper_time_limits", [False, True])
def test_time_limits(proper_time_limits):
    config = _config(use_proper_time_limits=proper_time_limits, n_steps=10)
    env = ContinuousStub(max_steps=4, termination_rate=0.0)
    init_fn, collect_fn, _, _, _ = make_train(config, env)
    state = init_fn(jax.random.PRNGKey(0))
    _, trajectory, _, _ = collect_fn(state)
    done = np.asarray(trajectory.done)
    np.testing.assert_array_equal(done[:, 0], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    # The step after a time limit starts a new episode either way.
    np.testing.assert_array_equal(np.asarray(trajectory.mask)[4], 0.0)

    team = np.asarray(trajectory.team_reward)[:, :, None]
    gap = np.asarray(trajectory.reward) - team
    if not proper_time_limits:
        np.testing.assert_array_equal(gap, 0.0)
        return
    np.testing.assert_array_equal(gap[~done], 0.0)
    # The bootstrap is gamma * V(real successor), valued from the critic's
    # hidden state advanced through the truncated step.
    ts, t = state.train_state, 3
    _, h_next = critic_step(
        ts.critic_ts.apply_fn, ts.critic_ts.params, trajectory.global_state[t],
        trajectory.critic_h[t], trajectory.mask[t], N_AGENTS,
    )
    keys = jax.random.split(jax.random.split(jax.random.PRNGKey(0), 3)[2], config.n_envs)
    successor = jax.vmap(lambda k: env.observe(k, 4))(keys)
    value, _ = critic_step(
        ts.critic_ts.apply_fn, ts.critic_ts.params,
        successor.reshape(config.n_envs, -1), h_next, jnp.ones(config.n_envs), N_AGENTS,
    )
    expected = config.gamma * vn_lib.denormalize(ts.value_norm, value)
    np.testing.assert_allclose(gap[t], np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_terminations_are_not_bootstrapped():
    config = _config(use_proper_time_limits=True)
    env = ContinuousStub(max_steps=1000, termination_rate=0.3)
    _, trajectory, _, _ = _collect(config, env)
    assert np.asarray(trajectory.done).any()
    np.testing.assert_array_equal(
        np.asarray(trajectory.reward), np.asarray(trajectory.team_reward)[:, :, None]
        * np.ones((1, 1, N_AGENTS)),
    )


def test_discrete_actions_respect_the_mask():
    config = _config()
    _, trajectory, _, _ = _collect(config, DiscreteStub(), n_rollouts=2)
    chosen = np.take_along_axis(
        np.asarray(trajectory.action_mask), np.asarray(trajectory.action)[..., None], -1
    )
    np.testing.assert_array_equal(chosen, 1.0)


@pytest.mark.parametrize("env", [ContinuousStub(), DiscreteStub()], ids=["cont", "disc"])
@pytest.mark.parametrize("recurrent", [True, False], ids=["gru", "mlp"])
def test_end_to_end(env, recurrent):
    config = _config(use_recurrent_policy=recurrent)
    init_fn, collect_fn, update_fn, eval_fn, num_updates = make_train(config, env)
    assert num_updates == 3
    state = init_fn(jax.random.PRNGKey(0))
    before = jax.tree.leaves(state.train_state.actor_ts.params)
    for _ in range(num_updates):
        state, trajectory, last_value, stats = collect_fn(state)
        state, losses = update_fn(state, trajectory, last_value)
    after = jax.tree.leaves(state.train_state.actor_ts.params)
    assert any(float(jnp.abs(a - b).max()) > 0 for a, b in zip(before, after))
    assert all(np.isfinite(float(v)) for v in losses.values())
    assert float(state.train_state.value_norm.debiasing_term) > 0
    assert np.isfinite(float(eval_fn(state.train_state, jax.random.PRNGKey(1))))


# --------------------------------------------------------------- config


def _params(**overrides):
    base = dict(
        n_epochs=15, n_total_steps=1e8, n_minibatches=1, n_steps=1024,
        parameter_sharing=True, random_seeds=[0],
    )
    base.update(overrides)
    return Params(**base)


def test_n_steps_rounds_up_to_whole_chunks_under_the_gru_only():
    gru = make_config(_params(), Model_Params(use_recurrent_policy=True), 32)
    mlp = make_config(_params(), Model_Params(use_recurrent_policy=False), 32)
    assert (gru.n_steps, mlp.n_steps) == (1030, 1024)
    assert make_config(_params(n_steps=200), Model_Params(True), 32).n_steps == 200


def test_defaults_are_on_policys():
    config = make_config(_params(), Model_Params(use_recurrent_policy=True), 32)
    expected = dict(
        lr=5e-4, critic_lr=5e-4, opti_eps=1e-5, gamma=0.99, gae_lambda=0.95,
        eps_clip=0.2, ent_coef=0.01, val_coef=1.0, grad_clip=10.0,
        use_valuenorm=True, use_huber_loss=True, huber_delta=10.0,
        use_clipped_value_loss=True, use_value_active_masks=True,
        use_policy_active_masks=True, use_proper_time_limits=False,
        use_linear_lr_decay=False, data_chunk_length=10, hidden_size=64,
        layer_N=1, use_ReLU=True, use_feature_normalization=True, gain=0.01,
        recurrent_N=1,
    )
    assert {k: getattr(config, k) for k in expected} == expected


def test_mappo_jax_model_group_is_rejected():
    with pytest.raises(TypeError):
        Model_Params(hidden_dim=168)


def test_per_agent_rewards_and_unshared_policies_are_rejected():
    env = ContinuousStub()
    env.reward_mode = "difference_rewards"
    with pytest.raises(NotImplementedError, match="per agent"):
        make_train(_config(), env)
    with pytest.raises(NotImplementedError, match="shared policy"):
        make_train(_config(parameter_sharing=False), ContinuousStub())
    with pytest.raises(ValueError, match="not a multiple"):
        make_train(_config(n_steps=12), ContinuousStub())


def test_linear_lr_decay_steps_once_per_update():
    """Adam with a constant gradient moves each step by ~lr, so the step sizes
    expose the schedule: lr * (1 - update / num_updates)."""
    config = _config(use_linear_lr_decay=True, n_epochs=1, n_minibatches=2)
    tx = _optimizer(1.0, config, num_updates=4)
    params = jnp.zeros(())
    opt_state = tx.init(params)
    steps = []
    for _ in range(8):
        updates, opt_state = tx.update(jnp.ones(()), opt_state, params)
        steps.append(-float(updates))
    np.testing.assert_allclose(
        steps, [1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25], rtol=1e-4
    )


# ------------------------------------------------------------- checkpoint


def test_checkpoint_round_trip_and_settings_record(tmp_path):
    from algorithms.rmappo_jax.run import RMAPPO_JAX_Runner

    config = _config()
    env = ContinuousStub()
    runner = RMAPPO_JAX_Runner.__new__(RMAPPO_JAX_Runner)
    runner.config, runner.env = config, env
    runner_state, trajectory, last_value, (update_fn, _) = _collect(config, env)
    runner_state, _ = update_fn(runner_state, trajectory, last_value)
    path = tmp_path / "train_checkpoint.msgpack"
    eval_rng = jax.random.PRNGKey(7)
    runner._save_train_checkpoint(runner_state, eval_rng, path)

    fresh = make_train(config, env)[0](jax.random.PRNGKey(3))
    loaded, loaded_eval_rng = runner._load_train_checkpoint(fresh, eval_rng, path)
    for a, b in zip(
        jax.tree.leaves(runner_state.train_state), jax.tree.leaves(loaded.train_state)
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(loaded_eval_rng), np.asarray(eval_rng))

    runner.config = _config(use_huber_loss=False)
    with pytest.raises(ValueError, match="use_huber_loss"):
        runner._load_train_checkpoint(fresh, eval_rng, path)
