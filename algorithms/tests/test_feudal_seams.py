"""Seam tests for the FeUdal wiring in ``algorithms/feudal_mappo_jax``.

These pin the four joints where the hierarchy meets the flat MAPPO machinery and
where a mistake would be **silent** rather than loud:

1. the in-scan goal ring reproduces ``manager.pool_goals`` exactly, including its
   episode-boundary semantics (the ring is the training path; ``pool_goals`` is
   the oracle, and they must not drift);
2. the goals stored in the trajectory are reproducible by re-scanning the manager
   over the stored global states — the property the manager's update relies on;
3. the agent-major flatten pairs each agent's obs with *its own* goal;
4. the PPO importance ratio is exactly 1 on the first minibatch of the first
   epoch, i.e. the update evaluates the same conditioned policy that acted.

They run against a tiny stub env (no MJX, no GPU), so they are fast and
deterministic — unlike an MJX rollout, which is not reproducible across
processes.

Run: ``uv run pytest algorithms/tests/test_feudal_seams.py -q``
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorithms.feudal_mappo_jax.manager import pool_goals
from algorithms.feudal_mappo_jax.mappo import build_manager
from algorithms.feudal_mappo_jax.network import evaluate_action
from algorithms.feudal_mappo_jax.trainer import make_train
from algorithms.feudal_mappo_jax.types import MAPPOConfig
from algorithms.feudal_mappo_jax.worker import bind_goal

N_AGENTS = 4
OBS_DIM = 6
ACTION_DIM = 2
N_ENVS = 3
N_STEPS = 12
HORIZON = 3
GOAL_DIM = 5
# Deterministic episode length, chosen to force a done partway through the
# rollout: the boundary cases are where the ring and pool_goals can disagree.
EPISODE_LEN = 5


class StubState(NamedTuple):
    t: jnp.ndarray
    seed: jnp.ndarray


class StubEnv:
    """Minimal functional env with the gymnax-style API the trainer expects.

    Observations are a cheap deterministic function of (t, seed) so the manager
    sees a genuinely varying global state; the episode truncates on a fixed
    period so `done` fires mid-rollout.
    """

    n_agents = N_AGENTS
    observation_dim = OBS_DIM
    action_dim = ACTION_DIM
    discrete = False
    reward_mode = "dense"
    max_steps = EPISODE_LEN

    def _obs(self, state):
        base = jnp.arange(N_AGENTS * OBS_DIM, dtype=jnp.float32).reshape(
            N_AGENTS, OBS_DIM
        )
        return jnp.sin(base + state.t + state.seed)

    def reset(self, key):
        state = StubState(
            t=jnp.zeros((), jnp.int32),
            seed=jax.random.uniform(key, ()) * 10.0,
        )
        return self._obs(state), state

    def step(self, state, actions):
        state = state._replace(t=state.t + 1)
        obs = self._obs(state)
        reward = jnp.sum(actions) * 0.01
        terminated = jnp.array(False)
        truncated = state.t >= EPISODE_LEN
        info = {"task_reward": reward}
        return obs, state, reward, terminated, truncated, info


def _config(**overrides):
    cfg = dict(
        n_steps=N_STEPS,
        n_envs=N_ENVS,
        n_epochs=1,
        n_minibatches=2,
        hidden_dim=16,
        goal_dim=GOAL_DIM,
        goal_horizon=HORIZON,
        manager_hidden_dim=16,
        manager_core="mlp",
        n_total_steps=N_STEPS * N_ENVS,
        n_eval_episodes=2,
    )
    cfg.update(overrides)
    return MAPPOConfig(**cfg)


def _collect(config):
    env = StubEnv()
    init_fn, collect_fn, update_fn, eval_fn, _ = make_train(config, env)
    runner_state = init_fn(jax.random.PRNGKey(0))
    new_runner_state, trajectory, bootstrap, stats = collect_fn(runner_state)
    return env, runner_state, new_runner_state, trajectory, bootstrap, update_fn


@pytest.fixture(scope="module")
def rollout():
    config = _config()
    env, rs, new_rs, traj, boot, update_fn = _collect(config)
    return config, env, rs, new_rs, traj, boot, update_fn


def test_rollout_shapes(rollout):
    config, _, _, _, traj, boot, _ = rollout
    assert traj.goal.shape == (N_STEPS, N_ENVS, N_AGENTS, GOAL_DIM)
    assert traj.pooled_goal.shape == traj.goal.shape
    assert traj.state_latent.shape == traj.goal.shape
    # The worker critic is always per-agent (the intrinsic reward is per-agent).
    assert traj.reward.shape == (N_STEPS, N_ENVS, N_AGENTS)
    assert traj.value.shape == (N_STEPS, N_ENVS, N_AGENTS)
    assert boot.worker.shape == (N_ENVS, N_AGENTS)
    # Dense env => scalar V^M head.
    assert traj.manager_reward.shape == (N_STEPS, N_ENVS)
    assert boot.manager.shape == (N_ENVS,)
    # Unit-norm goals, per agent.
    norms = jnp.linalg.norm(traj.goal, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)


def test_done_fires_midrollout(rollout):
    """Guard the guard: if nothing terminates, the ring/pool test is vacuous."""
    _, _, _, _, traj, _, _ = rollout
    assert bool(traj.done.any()), "stub env never terminated — boundary untested"
    assert not bool(traj.done.all())


def test_goal_ring_matches_pool_goals_oracle(rollout):
    """The in-scan ring == the whole-trajectory oracle, episode masking included.

    ``pool_goals`` cannot be used during the scan (it looks across the whole
    trajectory), so the trainer keeps an incremental ring instead. This is the
    assert that stops the two definitions of `w_t` from drifting apart.
    """
    _, _, _, _, traj, _, _ = rollout
    done_a = jnp.broadcast_to(
        traj.done[..., None].astype(jnp.float32), traj.goal.shape[:-1]
    )
    oracle = pool_goals(traj.goal, HORIZON, done=done_a)
    assert jnp.allclose(traj.pooled_goal, oracle, atol=1e-5), (
        "in-scan goal ring disagrees with pool_goals; max diff "
        f"{float(jnp.max(jnp.abs(traj.pooled_goal - oracle)))}"
    )


@pytest.mark.parametrize("core", ["mlp", "dilated_lstm"])
def test_goals_are_reproducible_from_stored_states(core):
    """Re-running the manager over the stored global states reproduces the goals.

    ``manager_update`` recomputes ``(goal, s)`` differentiably from
    ``trajectory.global_state``; if that recomputation did not match what the
    rollout actually emitted, the transition policy gradient would silently be
    optimizing a policy that never acted. Nothing else would show it — the losses
    stay finite and the diagnostics stay healthy.

    This is THE check for the recurrent core, where the two paths must agree on
    the carry convention: same zero-initialized pools, and the per-env reset
    applied *after* the step's goal is emitted (matching `pool_goals`' episode
    masking). The re-scan below is deliberately an independent restatement of
    that convention rather than a call into the trainer's own helper.
    """
    config = _config(manager_core=core)
    _, rs, _, traj = _collect(config)[:4]
    manager = build_manager(config, N_AGENTS)
    params = rs.train_state.manager_ts.params

    if core == "mlp":
        _, goal, s = manager.apply(params, None, traj.global_state)
    else:
        # NOTE: this assumes initialize_carry is deterministic (zeroed pools,
        # rng unused). If that ever stops being true, the rollout and the update
        # would start from different carries and this test is what catches it.
        init_carry = manager.initialize_carry(jax.random.PRNGKey(0), (N_ENVS,))
        dones = traj.done.astype(jnp.float32)

        def _step(carry, xs):
            gs_t, done_t = xs
            carry, goal_t, s_t = manager.apply(params, carry, gs_t)
            carry = carry._replace(
                cell=tuple(
                    jnp.where(done_t[:, None, None], 0.0, p) for p in carry.cell
                )
            )
            return carry, (goal_t, s_t)

        _, (goal, s) = jax.lax.scan(_step, init_carry, (traj.global_state, dones))

    assert jnp.allclose(goal, traj.goal, atol=1e-5), (
        f"core={core}: recomputed goals differ from the rollout's; max diff "
        f"{float(jnp.max(jnp.abs(goal - traj.goal)))}"
    )
    assert jnp.allclose(s, traj.state_latent, atol=1e-5)


@pytest.mark.parametrize("core", ["mlp", "dilated_lstm"])
def test_full_update_runs_for_both_cores(core):
    """collect -> update end-to-end on each core, with the manager actually moving.

    For the recurrent core this exercises the rematerialized BPTT scan inside
    `manager_update`, which is a different code path from the rollout's scan.
    """
    config = _config(manager_core=core)
    _, _, new_rs, traj, boot, update_fn = _collect(config)
    updated_rs, losses = update_fn(new_rs, traj, boot)
    for key in ("policy_loss", "manager_pg_loss", "state_latent_erank"):
        assert np.isfinite(float(losses[key])), (core, key, losses[key])
    before = jax.tree.leaves(new_rs.train_state.manager_ts.params)
    after = jax.tree.leaves(updated_rs.train_state.manager_ts.params)
    assert any(not jnp.array_equal(a, b) for a, b in zip(before, after)), (
        f"core={core}: manager params did not move"
    )


def test_dilated_lstm_carry_is_deterministic():
    """The rollout and the update build their carries from different rngs.

    They only agree because `dilated_lstm_carry` zeroes the pools and ignores the
    key. That is an implicit contract between `trainer.collect_fn` and
    `mappo._recompute`; assert it directly so a future change to
    `initialize_carry` fails here rather than silently desynchronizing the
    manager's gradient from the acting policy.
    """
    config = _config(manager_core="dilated_lstm")
    manager = build_manager(config, N_AGENTS)
    a = manager.initialize_carry(jax.random.PRNGKey(0), (N_ENVS,))
    b = manager.initialize_carry(jax.random.PRNGKey(12345), (N_ENVS,))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert jnp.array_equal(x, y)
    assert all(jnp.all(p == 0.0) for p in a.cell)


def test_ring_helpers_agree_across_batched_and_unbatched_layouts():
    """The `(c,E,N,D)` training ring and the `(c,N,D)` view() ring are one code path.

    `run.py:view()` drives a single unbatched episode while the trainer scans a
    batch, so before these helpers were shared they were two copies of the same
    convention — and a drifted copy (wrong slot index, stale pool) renders
    perfectly happily, just as a *different* policy. Here the unbatched ring is
    checked to equal env-slice 0 of the batched one over a full wrap-around.
    """
    from algorithms.feudal_mappo_jax.manager import (
        goal_ring_pool,
        goal_ring_reset,
        goal_ring_write,
    )

    key = jax.random.PRNGKey(3)
    goals = jax.random.normal(key, (2 * HORIZON, N_ENVS, N_AGENTS, GOAL_DIM))
    batched = jnp.zeros((HORIZON, N_ENVS, N_AGENTS, GOAL_DIM))
    unbatched = jnp.zeros((HORIZON, N_AGENTS, GOAL_DIM))

    for t in range(2 * HORIZON):  # past one wrap, so slot reuse is exercised
        batched = goal_ring_write(batched, goals[t], t)
        unbatched = goal_ring_write(unbatched, goals[t, 0], t)
        assert jnp.allclose(
            goal_ring_pool(batched)[0], goal_ring_pool(unbatched), atol=1e-6
        ), f"t={t}: batched and unbatched rings disagree"

    # The reset is env-selective in the batched layout: env 0 only.
    done = jnp.array([True] + [False] * (N_ENVS - 1))
    cleared = goal_ring_reset(batched, done)
    assert jnp.all(cleared[:, 0] == 0.0)
    assert jnp.allclose(cleared[:, 1:], batched[:, 1:])


def test_agent_major_flatten_pairs_obs_with_own_goal(rollout):
    """Row k = b*n_agents + i must carry agent i's obs AND agent i's goal."""
    _, _, _, _, traj, _, _ = rollout
    obs = traj.obs[0]  # (E, N, obs_dim)
    goal = traj.pooled_goal[0]  # (E, N, goal_dim)
    flat_obs = obs.reshape(N_ENVS * N_AGENTS, OBS_DIM)
    flat_goal = goal.reshape(N_ENVS * N_AGENTS, GOAL_DIM)
    for e in range(N_ENVS):
        for i in range(N_AGENTS):
            k = e * N_AGENTS + i
            assert jnp.allclose(flat_obs[k], obs[e, i])
            assert jnp.allclose(flat_goal[k], goal[e, i])


def test_ppo_ratio_is_one_before_any_update(rollout):
    """`evaluate_action` under the acting params reproduces the stored log_probs.

    This is the strongest available check that the update conditions the worker
    on exactly what the rollout conditioned it on: any mismatch in the goal (a
    stale pool, a transposed flatten, a recomputed-instead-of-stored goal) shows
    up here as ratio != 1.
    """
    _, _, rs, _, traj, _, _ = rollout
    actor_ts = rs.train_state.actor_ts
    n_flat = N_ENVS * N_AGENTS
    for t in (0, N_STEPS // 2, N_STEPS - 1):
        log_probs, _ = evaluate_action(
            bind_goal(
                actor_ts.apply_fn,
                traj.pooled_goal[t].reshape(n_flat, GOAL_DIM),
            ),
            actor_ts.params,
            traj.obs[t].reshape(n_flat, OBS_DIM),
            traj.action[t].reshape(n_flat, ACTION_DIM),
            False,
        )
        ratio = jnp.exp(log_probs - traj.log_prob[t].reshape(n_flat))
        assert jnp.allclose(ratio, 1.0, atol=1e-4), (
            f"step {t}: PPO ratio deviates from 1 before any update "
            f"(max |ratio-1| = {float(jnp.max(jnp.abs(ratio - 1.0)))})"
        )


def test_update_moves_both_learners(rollout):
    """Worker and manager both train, and every logged metric is finite."""
    _, _, _, new_rs, traj, boot, update_fn = rollout
    updated_rs, losses = update_fn(new_rs, traj, boot)
    for key in (
        "total_loss", "policy_loss", "value_loss", "entropy_loss",
        "manager_pg_loss", "manager_value_loss", "manager_explained_variance",
        "d_cos_mean", "d_cos_var", "valid_fraction",
        "goal_pairwise_cos", "goal_pairwise_cos_abs", "goal_direction_count",
        "state_pairwise_cos", "state_latent_erank",
    ):
        assert key in losses, f"missing metric {key}"
        assert np.isfinite(float(losses[key])), (key, losses[key])

    n_agents = traj.goal.shape[-2]
    count = float(losses["goal_direction_count"])
    assert 1.0 - 1e-4 <= count <= n_agents + 1e-4, count

    for name in ("actor_ts", "critic_ts", "manager_ts", "manager_critic_ts"):
        before = jax.tree.leaves(getattr(new_rs.train_state, name).params)
        after = jax.tree.leaves(getattr(updated_rs.train_state, name).params)
        assert any(
            not jnp.array_equal(a, b) for a, b in zip(before, after)
        ), f"{name} did not move"


def test_goal_direction_count_reads_the_collapse_cases():
    """1 for one shared goal (or one shared LINE), N for orthogonal goals."""
    from algorithms.feudal_mappo_jax.mappo import (
        _agent_direction_count,
        _agent_gram,
        _mean_pairwise_cosine,
    )

    n, d = 4, 8
    key = jax.random.PRNGKey(0)
    shared = jnp.broadcast_to(jax.random.normal(key, (1, d)), (n, d))
    orthogonal = jnp.eye(n, d)
    # Antipodal clusters: signed cosine averages to ~0 and reads as "diverse",
    # but every goal lies on one line — this is the case the count exists for.
    antipodal = shared * jnp.array([1.0, 1.0, -1.0, -1.0])[:, None]

    assert np.isclose(float(_agent_direction_count(_agent_gram(shared))), 1.0, atol=1e-3)
    assert np.isclose(float(_agent_direction_count(_agent_gram(orthogonal))), n, atol=1e-3)
    assert np.isclose(
        float(_agent_direction_count(_agent_gram(antipodal))), 1.0, atol=1e-3
    )
    assert abs(float(_mean_pairwise_cosine(_agent_gram(antipodal)))) < 0.4


def test_manager_metrics_are_scalars(rollout):
    """`run.py` casts every loss with float(); a non-scalar would blow up there."""
    _, _, _, new_rs, traj, boot, update_fn = rollout
    _, losses = update_fn(new_rs, traj, boot)
    for key, value in losses.items():
        assert jnp.asarray(value).shape == (), (key, jnp.asarray(value).shape)


def test_valid_fraction_matches_horizon(rollout):
    """`valid` must drop exactly the steps with no real s_{t+c}, plus boundaries.

    With episodes present it is strictly below (T-c)/T; if it collapsed toward 0
    the done-masking would be eating the rollout, and the manager would be
    training on almost nothing.
    """
    _, _, _, new_rs, traj, boot, update_fn = rollout
    _, losses = update_fn(new_rs, traj, boot)
    upper = (N_STEPS - HORIZON) / N_STEPS
    frac = float(losses["valid_fraction"])
    assert 0.0 < frac <= upper + 1e-6, (frac, upper)


def test_detach_rule_holds_in_the_update():
    """The manager's PG must not backprop through the s_{t+c} target arm.

    This is the paper's explicit anti-collapse rule ("the dependence of s on
    theta is ignored when computing grad d_cos"). It is checked at the *update*
    level, not just in the helper, because the update is where a future edit
    could reintroduce an attached path.
    """
    from algorithms.feudal_mappo_jax.manager import transition_cosine

    config = _config()
    _, rs, _, traj = _collect(config)[:4]
    manager = build_manager(config, N_AGENTS)
    done_a = jnp.broadcast_to(
        traj.done[..., None].astype(jnp.float32), traj.goal.shape[:-1]
    )

    def obj(params, detach):
        _, goal, s = manager.apply(params, None, traj.global_state)
        cos, valid = transition_cosine(
            s, goal, HORIZON, done=done_a, detach_states=detach
        )
        return jnp.sum(cos * valid)

    g_detached = jax.grad(obj)(rs.train_state.manager_ts.params, True)
    g_attached = jax.grad(obj)(rs.train_state.manager_ts.params, False)
    # Detaching genuinely changes the gradient (the rule is not a no-op)...
    assert not all(
        jnp.allclose(a, b)
        for a, b in zip(jax.tree.leaves(g_detached), jax.tree.leaves(g_attached))
    )
    # ...and f_Mspace is STILL trained under the detach, via the goal arm,
    # because the manager's core consumes `s`. Wire the core to `z` instead and
    # this is exactly zero and the latent never learns.
    f_mspace = g_detached["params"]["f_Mspace"]["kernel"]
    assert jnp.any(f_mspace != 0.0), (
        "f_Mspace got no gradient under the detach — the core is not consuming s"
    )


def test_intrinsic_stream_is_separate_and_exact():
    """r^I lands in its OWN field, and never touches the extrinsic reward.

    Budget-independent: this pins the *arithmetic* of the intrinsic path (that it
    is built once, unscaled, with its own truncation bootstrap, into its own
    stream), which a learning curve could never isolate. It also checks alpha=0
    is a true no-op — the guard that keeps the machinery inert by default.

    The separation is the load-bearing part. r^I is ~0.155/step and near-flat
    while the extrinsic reward is ~5e-05/step early, so adding the two together
    hands essentially the whole gradient to r^I. They are only allowed to meet in
    `ppo_update`, after each has been normalized to unit std.
    """
    from algorithms.feudal_mappo_jax.manager import worker_intrinsic_reward

    alpha = 0.5
    traj0 = _collect(_config(intrinsic_coef=0.0))[3]
    traj1 = _collect(_config(intrinsic_coef=alpha))[3]

    # Same seed and env => the rollouts are identical (the intrinsic pass runs
    # strictly post-scan, so it cannot alter behaviour within the rollout).
    assert jnp.allclose(traj0.goal, traj1.goal, atol=1e-6)
    assert jnp.allclose(traj0.state_latent, traj1.state_latent, atol=1e-6)
    assert jnp.allclose(traj0.action, traj1.action, atol=1e-6)
    # The manager's stream never sees the intrinsic term at all...
    assert jnp.allclose(traj0.manager_reward, traj1.manager_reward, atol=1e-6)
    # ...and neither does the worker's EXTRINSIC stream. This is the assertion
    # that inverted when the reward-level fold was removed: it used to differ by
    # exactly alpha * r^I, and must now be bit-identical.
    assert jnp.array_equal(traj0.reward, traj1.reward), (
        "extrinsic reward changed with alpha — the intrinsic term is being "
        "folded into `reward` again instead of kept in its own stream"
    )
    # alpha=0 leaves the intrinsic stream as the zeros the scan built.
    assert jnp.array_equal(traj0.intrinsic_reward, jnp.zeros_like(traj0.reward))

    done_a = jnp.broadcast_to(
        traj0.done[..., None].astype(jnp.float32), traj0.state_latent.shape[:-1]
    )
    r_int = worker_intrinsic_reward(
        traj0.state_latent, traj0.goal, HORIZON, done=done_a
    )
    # The stored stream is r^I plus its own truncation bootstrap (taken against
    # V^I in-scan), and nothing else — in particular it is NOT scaled by alpha
    # here; alpha is applied to the normalized advantage in `ppo_update`.
    expected = r_int + traj1.intrinsic_bootstrap
    assert jnp.allclose(traj1.intrinsic_reward, expected, atol=1e-5), (
        "intrinsic stream != r^I + bootstrap; max err "
        f"{float(jnp.max(jnp.abs(traj1.intrinsic_reward - expected)))}"
    )
    assert jnp.any(jnp.abs(r_int) > 1e-6), "r^I is identically zero — vacuous test"
    # r^I is a mean of cosines, so it is bounded; a violation means the mask
    # denominator or the averaging is wrong.
    assert float(jnp.max(jnp.abs(r_int))) <= 1.0 + 1e-5


def test_alpha_zero_builds_no_intrinsic_critic():
    """alpha=0 must stay a STATIC no-op, not a multiply-by-zero.

    The intrinsic critic is what changes the msgpack checkpoint format, so if it
    were built unconditionally every existing `feudal_a0` checkpoint would stop
    resuming. `None` (an empty JAX pytree) is what keeps the slot free.
    """
    _, rs0, _, _, _, _ = _collect(_config(intrinsic_coef=0.0))
    _, rs1, _, _, _, _ = _collect(_config(intrinsic_coef=0.5))
    assert rs0.train_state.intrinsic_critic_ts is None
    assert rs1.train_state.intrinsic_critic_ts is not None


def test_alpha_is_a_gradient_fraction_not_a_reward_coefficient():
    """The two streams meet as unit-std advantages, so alpha is the mix ratio.

    Pins the property the whole redesign exists to establish: the combination
    must be invariant to the RAW scale of r^I. Scaling the stored intrinsic
    stream by 1000x must leave the combined advantage unchanged, because
    normalization divides that factor straight back out. Under the old
    reward-level fold the same 1000x would have swamped the extrinsic term
    entirely — which is exactly how the measured failure happened.
    """
    from algorithms.feudal_mappo_jax.mappo import compute_gae, _annealed_alpha

    config = _config(intrinsic_coef=0.5, intrinsic_anneal="none")
    _, _, _, traj, boot, _ = _collect(config)

    def combined(scale):
        dones = traj.done.astype(jnp.float32)
        adv_e, _ = compute_gae(
            traj.reward, traj.value, dones, boot.worker,
            config.gamma, config.gae_lambda,
        )
        adv_i, _ = compute_gae(
            traj.intrinsic_reward * scale, traj.value_int * scale, dones,
            boot.worker_int * scale, config.gamma, config.gae_lambda,
        )
        norm = lambda a: (a - a.mean(0)) / (a.std(0, ddof=1) + 1e-8)
        alpha = _annealed_alpha(config, jnp.float32(0.0))
        return norm(adv_e) + alpha * norm(adv_i)

    assert jnp.allclose(combined(1.0), combined(1000.0), atol=1e-4), (
        "combined advantage depends on the raw intrinsic scale — the per-stream "
        "normalization is not being applied"
    )


def test_alpha_anneal_schedule():
    """`linear` reaches exactly 0 at the end of training; `none` holds."""
    from algorithms.feudal_mappo_jax.mappo import _annealed_alpha

    lin = _config(intrinsic_coef=0.5, intrinsic_anneal="linear")
    assert float(_annealed_alpha(lin, jnp.float32(0.0))) == pytest.approx(0.5)
    assert float(_annealed_alpha(lin, jnp.float32(0.5))) == pytest.approx(0.25)
    assert float(_annealed_alpha(lin, jnp.float32(1.0))) == pytest.approx(0.0)
    # Clamped, so an overshooting progress cannot flip alpha negative.
    assert float(_annealed_alpha(lin, jnp.float32(1.5))) == pytest.approx(0.0)

    const = _config(intrinsic_coef=0.5, intrinsic_anneal="none")
    assert float(_annealed_alpha(const, jnp.float32(1.0))) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="unknown intrinsic_anneal"):
        _annealed_alpha(_config(intrinsic_anneal="bogus"), jnp.float32(0.0))


def test_update_runs_and_moves_the_intrinsic_critic():
    """End-to-end: the intrinsic path trains V^I and reports its diagnostics."""
    config = _config(intrinsic_coef=0.5)
    _, _, new_rs, traj, boot, update_fn = _collect(config)
    updated_rs, losses = update_fn(new_rs, traj, boot, jnp.float32(0.0))

    before = new_rs.train_state.intrinsic_critic_ts.params
    after = updated_rs.train_state.intrinsic_critic_ts.params
    moved = jax.tree.reduce(
        lambda acc, x: acc or bool(x),
        jax.tree.map(lambda a, b: bool(jnp.any(a != b)), before, after),
        False,
    )
    assert moved, "V^I params did not move"
    for key in (
        "alpha_current",
        "adv_ext_std_raw",
        "adv_int_std_raw",
        "intrinsic_explained_variance",
        "intrinsic_value_loss",
    ):
        assert key in losses, f"missing diagnostic {key}"
        assert jnp.ndim(losses[key]) == 0, f"{key} is not a scalar"
    assert float(losses["alpha_current"]) == pytest.approx(0.5)

    # ...and none of it appears at alpha=0, where the stats keys must be
    # exactly what they were before the intrinsic stream existed.
    _, _, new_rs0, traj0, boot0, update_fn0 = _collect(_config(intrinsic_coef=0.0))
    _, losses0 = update_fn0(new_rs0, traj0, boot0)
    for key in ("alpha_current", "adv_int_std_raw", "intrinsic_value_loss"):
        assert key not in losses0
