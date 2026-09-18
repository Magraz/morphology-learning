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

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorithms.feudal_mappo_jax.manager import (
    GLOBAL_LATENTS,
    GOAL_VARIANTS,
    LATENTS,
    LOCAL_LATENTS,
    PRIVATE_LATENTS,
    constant_goals,
    mean_goal_direction,
    pool_goals,
)
from algorithms.feudal_mappo_jax.mappo import build_manager
from algorithms.feudal_mappo_jax.network import evaluate_action
from algorithms.feudal_mappo_jax.trainer import make_train
from algorithms.feudal_mappo_jax.types import MAPPOConfig
from algorithms.feudal_mappo_jax.worker import bind_goal

@pytest.fixture(autouse=True)
def _run_on_cpu():
    """Pin these tests to CPU — same reasoning as `test_smax_seams.py`, plus one
    of its own.

    They are tiny, so the GPU buys nothing, and sharing it with a training or
    rendering job makes them fail on cuSolver/OOM errors that look exactly like
    assertion failures.

    The extra reason here is `test_goals_are_reproducible_from_stored_states`,
    which asserts a MATHEMATICAL identity (the manager update's recompute equals
    what the rollout emitted) with an f32 tolerance. On GPU it failed for all
    four local latents at 4e-4 - 7e-4 while passing for `centralized`, which
    looked like a `local`-specific logic bug and is recorded in CLAUDE.md as one.
    It is not: MEASURED 2026-09-14, the gap is **exactly 0.0 in float64** for
    every latent, eager and jitted alike. It is XLA associating the f32
    reductions differently between the rollout's `collect_fn` compilation and the
    test's standalone one — the same class of artifact CLAUDE.md already records
    for `mjx.ray` (~3e-4, "compare both under jit"). Loosening the tolerance
    instead would have hidden it behind a number; pinning the platform keeps the
    assertion exact and keeps the suite deterministic, which is the whole reason
    these seams are run on a CPU stub env rather than on MJX.

    `jax.default_device` is used rather than a module-level `JAX_PLATFORMS=cpu`,
    which only takes effect if this module happens to be imported before JAX is
    initialized (i.e. it depends on pytest collection order).
    """
    with jax.default_device(jax.devices("cpu")[0]):
        yield


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
    per_agent_rewards = False
    max_steps = EPISODE_LEN

    def _obs(self, state):
        base = jnp.arange(self.n_agents * OBS_DIM, dtype=jnp.float32).reshape(
            self.n_agents, OBS_DIM
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
        if self.per_agent_rewards:
            reward = jnp.full((self.n_agents,), reward)
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


@pytest.mark.parametrize("latent", ["centralized", *LOCAL_LATENTS])
@pytest.mark.parametrize("core", ["mlp", "dilated_lstm"])
def test_goals_are_reproducible_from_stored_states(core, latent):
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
    config = _config(manager_core=core, manager_latent=latent)
    _, rs, _, traj = _collect(config)[:4]
    manager = build_manager(config, N_AGENTS)
    params = rs.train_state.manager_ts.params

    if core == "mlp":
        _, goal, s = manager.apply(params, None, traj.global_state, traj.obs)
    else:
        # NOTE: this assumes initialize_carry is deterministic (zeroed pools,
        # rng unused). If that ever stops being true, the rollout and the update
        # would start from different carries and this test is what catches it.
        init_carry = manager.initialize_carry(jax.random.PRNGKey(0), (N_ENVS,))
        dones = traj.done.astype(jnp.float32)

        def _step(carry, xs):
            gs_t, obs_t, done_t = xs
            carry, goal_t, s_t = manager.apply(params, carry, gs_t, obs_t)
            carry = carry._replace(
                cell=tuple(
                    jnp.where(done_t[:, None, None], 0.0, p) for p in carry.cell
                )
            )
            return carry, (goal_t, s_t)

        _, (goal, s) = jax.lax.scan(
            _step, init_carry, (traj.global_state, traj.obs, dones)
        )

    assert jnp.allclose(goal, traj.goal, atol=1e-5), (
        f"core={core}/{latent}: recomputed goals differ from the rollout's; max diff "
        f"{float(jnp.max(jnp.abs(goal - traj.goal)))}"
    )
    assert jnp.allclose(s, traj.state_latent, atol=1e-5)


@pytest.mark.parametrize("latent", ["centralized", *LOCAL_LATENTS])
@pytest.mark.parametrize("core", ["mlp", "dilated_lstm"])
def test_full_update_runs_for_both_cores(core, latent):
    """collect -> update end-to-end on each core, with the manager actually moving.

    For the recurrent core this exercises the rematerialized BPTT scan inside
    `manager_update`, which is a different code path from the rollout's scan.
    """
    config = _config(manager_core=core, manager_latent=latent)
    _, _, new_rs, traj, boot, update_fn = _collect(config)
    updated_rs, losses = update_fn(new_rs, traj, boot)
    for key in ("policy_loss", "manager_pg_loss", "state_latent_erank"):
        assert np.isfinite(float(losses[key])), (core, latent, key, losses[key])
    before = jax.tree.leaves(new_rs.train_state.manager_ts.params)
    after = jax.tree.leaves(updated_rs.train_state.manager_ts.params)
    assert any(not jnp.array_equal(a, b) for a, b in zip(before, after)), (
        f"core={core}/{latent}: manager params did not move"
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


# --------------------------------------------------------------------------
# Permutation nulls (the "are the goals useful?" diagnostics)
# --------------------------------------------------------------------------


def test_agent_permutation_commutes_with_the_goal_ring():
    """`roll(pool(h)) == pool(roll(h))`, which is what licenses permuting `w_t`.

    The eval ablation transforms the POOLED goal at the `goal_ring_pool` read
    site. That is only equivalent to permuting the manager's raw output because
    the pool is a plain sum and the shift is fixed across time. If anyone makes
    the permutation time-varying or random-per-step, this equality breaks and
    agent `i` starts receiving a sum of *different agents'* goals at different
    times — which changes ||w|| from ~c to ~sqrt(c) and stops preserving the
    marginal, i.e. the null quietly stops being a null.
    """
    from algorithms.feudal_mappo_jax.manager import (
        goal_ring_pool,
        goal_ring_write,
        permute_agent_goals,
    )

    key = jax.random.PRNGKey(11)
    goals = jax.random.normal(key, (2 * HORIZON, N_ENVS, N_AGENTS, GOAL_DIM))
    ring = jnp.zeros((HORIZON, N_ENVS, N_AGENTS, GOAL_DIM))
    rolled_ring = jnp.zeros((HORIZON, N_ENVS, N_AGENTS, GOAL_DIM))

    for t in range(2 * HORIZON):  # past one wrap, so slot reuse is exercised
        ring = goal_ring_write(ring, goals[t], t)
        rolled_ring = goal_ring_write(
            rolled_ring, permute_agent_goals(goals[t], 1), t
        )
        assert jnp.array_equal(
            permute_agent_goals(goal_ring_pool(ring), 1),
            goal_ring_pool(rolled_ring),
        ), f"t={t}: rolling the pool differs from pooling the rolls"


def test_agent_permutation_rejects_degenerate_shifts():
    """A roll that is the identity must RAISE, not report a vacuous zero gap.

    At n_agents == 1 (or shift % n == 0) the null equals the real value by
    construction, so every gap is exactly 0.0 — which reads as "the goals make
    no difference", the precise conclusion the diagnostic exists to reach. This
    is the self-sealing failure mode, so it is rejected loudly.
    """
    from algorithms.feudal_mappo_jax.manager import (
        permute_agent_goals,
        permute_env_goals,
    )

    one_agent = jnp.zeros((N_ENVS, 1, GOAL_DIM))
    with pytest.raises(ValueError, match="identity"):
        permute_agent_goals(one_agent, 1)

    goals = jnp.zeros((N_ENVS, N_AGENTS, GOAL_DIM))
    with pytest.raises(ValueError, match="identity"):
        permute_agent_goals(goals, 0)
    with pytest.raises(ValueError, match="identity"):
        permute_agent_goals(goals, N_AGENTS)
    with pytest.raises(ValueError, match="identity"):
        permute_env_goals(goals, 0, N_ENVS)

    # A legitimate shift still works on both axes.
    assert permute_agent_goals(goals, 1).shape == goals.shape
    assert permute_env_goals(goals, 0, 1).shape == goals.shape


@pytest.mark.parametrize("n_agents,n_envs", [(1, N_ENVS), (N_AGENTS, 1), (1, 1)])
def test_singleton_permutation_metrics_are_unavailable(n_agents, n_envs):
    from algorithms.feudal_mappo_jax.mappo import manager_cosine_metrics

    shape = (N_STEPS, n_envs, n_agents, GOAL_DIM)
    k_s, k_g = jax.random.split(jax.random.PRNGKey(9))
    s = jax.random.normal(k_s, shape)
    goal = jax.random.normal(k_g, shape)
    metrics = manager_cosine_metrics(
        s, goal, HORIZON, jnp.zeros(shape[:-1]), jnp.ones(shape[:-1])
    )
    unavailable = set()
    if n_agents == 1:
        unavailable.update(("d_cos_null_agent", "d_cos_gap_agent", "goal_perm_cos"))
    if n_envs == 1:
        unavailable.update(("d_cos_null_env", "d_cos_gap_env"))
    for name, value in metrics.items():
        if name in unavailable:
            assert np.isnan(value), name
        else:
            assert np.isfinite(value), name


@pytest.mark.parametrize(
    "eval_goal_variants,intrinsic_coef,per_agent_rewards",
    [(True, 0.1, False), (False, 0.1, False), (True, 0.0, False), (True, 0.1, True)],
)
def test_single_agent_collect_update_and_eval(
    eval_goal_variants, intrinsic_coef, per_agent_rewards
):
    """Single-agent training works with intrinsic reward and either eval mode."""
    config = _config(
        intrinsic_coef=intrinsic_coef, eval_goal_variants=eval_goal_variants,
        per_agent_rewards=per_agent_rewards,
    )
    env = StubEnv()
    env.n_agents = 1
    env.per_agent_rewards = per_agent_rewards
    init_fn, collect_fn, update_fn, eval_fn, _ = make_train(config, env)
    initial = init_fn(jax.random.PRNGKey(0))
    state, trajectory, last_value, _ = collect_fn(initial)
    assert trajectory.reward.shape == trajectory.value.shape == (N_STEPS, N_ENVS, 1)
    assert trajectory.intrinsic_reward.shape == trajectory.value_int.shape == (
        N_STEPS, N_ENVS, 1
    )
    assert last_value.worker.shape == last_value.worker_int.shape == (N_ENVS, 1)
    state, losses = update_fn(state, trajectory, last_value)
    unavailable = {"d_cos_null_agent", "d_cos_gap_agent", "goal_perm_cos"}
    for name, value in losses.items():
        if name in unavailable:
            assert np.isnan(value), name
        else:
            assert np.isfinite(value), name
    for leaf in jax.tree_util.tree_leaves(state.train_state):
        assert np.isfinite(leaf).all()
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(initial.train_state.actor_ts.params),
            jax.tree_util.tree_leaves(state.train_state.actor_ts.params),
        )
    )

    key = jax.random.PRNGKey(1)
    constant = mean_goal_direction(trajectory.pooled_goal)
    rewards, lengths = eval_fn(
        state.train_state, key, detail=True, constant_goal=constant
    )
    assert rewards.shape == lengths.shape == (
        3 if eval_goal_variants else 1, config.n_eval_episodes
    )
    assert np.isfinite(rewards).all()
    assert np.all(lengths == env.max_steps)
    real = eval_fn(state.train_state, key, variants=("real",), detail=True)[0]
    np.testing.assert_allclose(rewards[0], real[0], atol=1e-6)
    # Explicitly requesting an impossible permutation must still fail.
    with pytest.raises(ValueError, match="identity"):
        eval_fn(state.train_state, key, variants=("real", "permuted"))


def test_transition_cosine_valid_is_independent_of_goals():
    """`valid` is a function of (states, done) only — the null's load-bearing fact.

    `manager_cosine_metrics` forms `mask = valid * active` ONCE and reuses it for
    the real cosine and both nulls. That is only sound if permuting the goals
    cannot change which entries are valid; otherwise real and null would be
    averaged over different denominators and the gap would be an artifact.
    """
    from algorithms.feudal_mappo_jax.manager import (
        permute_agent_goals,
        permute_env_goals,
        transition_cosine,
    )

    key = jax.random.PRNGKey(5)
    k_s, k_g = jax.random.split(key)
    s = jax.random.normal(k_s, (N_STEPS, N_ENVS, N_AGENTS, GOAL_DIM))
    goal = jax.random.normal(k_g, (N_STEPS, N_ENVS, N_AGENTS, GOAL_DIM))
    done = jnp.zeros((N_STEPS, N_ENVS, N_AGENTS))
    done = done.at[EPISODE_LEN].set(1.0)

    _, valid = transition_cosine(s, goal, HORIZON, done=done)
    for permuted in (
        permute_agent_goals(goal, 1),
        permute_env_goals(goal, 1, 1),
        jnp.zeros_like(goal),
    ):
        _, valid_p = transition_cosine(s, permuted, HORIZON, done=done)
        assert jnp.array_equal(valid, valid_p)


def test_d_cos_null_equals_real_when_goals_are_collapsed():
    """Collapse => zero gap, per axis. Intended behaviour, not an accident.

    A gap of ~0 is ambiguous on its own: it means EITHER the goals carry no
    information on that axis OR they were already identical, so permuting them
    did nothing. `goal_perm_cos` is what separates the two, so it is asserted
    here alongside each gap.
    """
    from algorithms.feudal_mappo_jax.mappo import manager_cosine_metrics

    key = jax.random.PRNGKey(7)
    k_s, k_g = jax.random.split(key)
    s = jax.random.normal(k_s, (N_STEPS, N_ENVS, N_AGENTS, GOAL_DIM))
    done = jnp.zeros((N_STEPS, N_ENVS, N_AGENTS))
    active = jnp.ones((N_STEPS, N_ENVS, N_AGENTS))

    # One shared goal per (t, env): the agent axis carries nothing.
    shared = jax.random.normal(k_g, (N_STEPS, N_ENVS, 1, GOAL_DIM))
    shared = jnp.broadcast_to(shared, (N_STEPS, N_ENVS, N_AGENTS, GOAL_DIM))
    m = manager_cosine_metrics(s, shared, HORIZON, done, active)
    assert abs(float(m["d_cos_gap_agent"])) < 1e-5
    assert float(m["goal_perm_cos"]) > 0.999  # the permutation changed nothing

    # One goal shared across envs: the state-conditioning axis carries nothing.
    per_agent = jax.random.normal(k_g, (N_STEPS, 1, N_AGENTS, GOAL_DIM))
    per_agent = jnp.broadcast_to(per_agent, (N_STEPS, N_ENVS, N_AGENTS, GOAL_DIM))
    m = manager_cosine_metrics(s, per_agent, HORIZON, done, active)
    assert abs(float(m["d_cos_gap_env"])) < 1e-5
    # ...and this is exactly the case the agent-axis null alone calls healthy:
    # a fixed per-agent code with no state dependence. Its agent gap is NOT ~0,
    # which is why d_cos_gap_env has to be read first.
    assert float(m["goal_perm_cos"]) < 0.9


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
    from algorithms.feudal_mappo_jax.manager import (
        worker_intrinsic_reward_aligned,
    )

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
    # NOTE this is a WIRING test, not a semantics one: it calls the same helper
    # the collector calls, so it pins that the stream is assembled once, into its
    # own field, unscaled — and nothing about whether the helper is right. The
    # timing semantics are pinned separately below, against hand-computed values.
    r_int = worker_intrinsic_reward_aligned(
        traj1.state_latent, traj1.next_state_latent, traj1.goal, HORIZON,
        done=done_a,
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


# ---------------------------------------------------------------------------
# Intrinsic reward TIMING.
#
# `r^I_t` must score the outcome of `a_t`. The paper's literal form ends its
# displacement at the PRE-action latent `s_t`, so every term is fixed before
# `a_t` is sampled while the extrinsic reward on the same transition IS that
# action's consequence; the two streams scored different actions.
#
# These tests compute their expected values BY HAND from a latent that is
# literally a position, so they constrain the semantics rather than the wiring.
# The existing `test_intrinsic_stream_is_separate_and_exact` calls the same
# helper the collector calls and therefore cannot fail for a wrong definition —
# that is the trap this block exists to avoid, not to repeat.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rollout_alpha():
    """A rollout with the intrinsic path LIVE.

    The shared `rollout` fixture runs at the default alpha=0, where the whole
    intrinsic path is a STATIC no-op: `next_state_latent` is the scalar
    placeholder and V^I returns zeros, so every assertion below would pass
    vacuously on it. That is the no-op working, and it is checked separately.
    """
    config = _config(intrinsic_coef=0.5)
    env, rs, new_rs, traj, boot, update_fn = _collect(config)
    return config, env, rs, new_rs, traj, boot, update_fn


_EAST = jnp.array([1.0, 0.0])
_NORTH = jnp.array([0.0, 1.0])
_WEST = jnp.array([-1.0, 0.0])


def _traj4(x):
    """(T, N, 2) -> (T, 1, N, 2): the (T, n_envs, n_agents, dim) layout."""
    return jnp.asarray(x)[:, None]


def _positions(moves):
    """Per-step displacements -> (pre-action latents, successor latents).

    The "env" here is a point whose latent IS its position, so the displacement
    the reward measures is exactly the action taken. That is what makes the
    expected values below hand-computable.
    """
    pos = jnp.concatenate([jnp.zeros_like(moves[:1]), jnp.cumsum(moves, axis=0)])
    return pos[:-1], pos[1:]


def test_intrinsic_reward_credits_the_action_on_its_own_transition():
    """c=1, eastward goal: east / north / west score +1 / 0 / -1 AT step t.

    The headline property. Under the old indexing all three agents score the
    same thing, because the reward never saw the action.
    """
    from algorithms.feudal_mappo_jax.manager import (
        worker_intrinsic_reward,
        worker_intrinsic_reward_aligned,
    )

    moves = jnp.stack([_EAST, _NORTH, _WEST])          # 3 agents, one step
    s = _traj4(jnp.zeros((1, 3, 2)))
    s_plus = _traj4(moves[None])
    g = _traj4(jnp.broadcast_to(_EAST, (1, 3, 2)))

    r = worker_intrinsic_reward_aligned(s, s_plus, g, 1)
    assert np.allclose(np.asarray(r[0, 0]), [1.0, 0.0, -1.0], atol=1e-5), r[0, 0]

    # And the old form is blind to all of it: same three agents, same answer.
    legacy = worker_intrinsic_reward(s, g, 1)
    assert float(jnp.abs(legacy).max()) == 0.0


def test_the_first_action_of_an_episode_earns_a_reward():
    """`k=0` is always valid, so step 0 is scored; the old form paid it 0."""
    from algorithms.feudal_mappo_jax.manager import (
        worker_intrinsic_reward,
        worker_intrinsic_reward_aligned,
    )

    moves = jnp.broadcast_to(_EAST, (4, 1, 2))
    s, s_plus = _positions(moves)
    g = jnp.broadcast_to(_EAST, (4, 1, 2))
    r = worker_intrinsic_reward_aligned(_traj4(s), _traj4(s_plus), _traj4(g), 3)
    assert float(r[0, 0, 0]) == pytest.approx(1.0, abs=1e-5)
    assert float(worker_intrinsic_reward(_traj4(s), _traj4(g), 3)[0, 0, 0]) == 0.0


def test_longer_horizon_matches_a_hand_computed_trajectory():
    """c=3 over east, east, north — every origin, goal index and the denominator.

    r[2] averages three terms, all ending at the successor of a_2 = north:
      k=0  d_cos((0,1), north)         = 0
      k=1  d_cos((1,1), north)         = 1/sqrt(2)
      k=2  d_cos((2,1), east)          = 2/sqrt(5)
    """
    from algorithms.feudal_mappo_jax.manager import worker_intrinsic_reward_aligned

    moves = jnp.stack([_EAST, _EAST, _NORTH])[:, None]      # (3, 1 agent, 2)
    s, s_plus = _positions(moves)
    g = jnp.stack([_EAST, _NORTH, _EAST])[:, None]
    r = worker_intrinsic_reward_aligned(_traj4(s), _traj4(s_plus), _traj4(g), 3)

    expected = (0.0 + 1.0 / np.sqrt(2.0) + 2.0 / np.sqrt(5.0)) / 3.0
    assert float(r[2, 0, 0]) == pytest.approx(expected, abs=1e-5)


def test_the_episode_ending_action_is_paid_against_its_terminal_successor():
    """The one transition a shift-based implementation CANNOT reach.

    `done[t]` is excluded from the mask, so the action that ends the episode is
    still scored — against the successor it actually produced, captured before
    the reset. Deriving the endpoint by shifting the stored latents pays 0 there
    instead, which is a standing bonus for terminating: the same shape as the
    `boundary_truncates` failure the MJX env removed for exactly that reason.
    """
    from algorithms.feudal_mappo_jax.intrinsic_timing_probe import (
        legacy_shift_approximation,
    )
    from algorithms.feudal_mappo_jax.manager import worker_intrinsic_reward_aligned

    moves = jnp.broadcast_to(_EAST, (4, 1, 2))
    s, s_plus = _positions(moves)
    g = jnp.broadcast_to(_EAST, (4, 1, 2))
    done = jnp.zeros((4, 1, 1)).at[1].set(1.0)              # episode ends after t=1

    r = worker_intrinsic_reward_aligned(
        _traj4(s), _traj4(s_plus), _traj4(g), 3, done=done
    )
    assert float(r[1, 0, 0]) == pytest.approx(1.0, abs=1e-5), (
        "the episode-ending action was not credited for its terminal successor"
    )
    shifted = legacy_shift_approximation(_traj4(s), _traj4(g), 3, done=done)
    assert float(shifted[1, 0, 0]) == 0.0, (
        "the shift approximation is supposed to be the thing that pays 0 here — "
        "if it does not, this test no longer demonstrates the difference"
    )


def test_a_reset_cannot_leak_into_the_next_episodes_reward():
    """A teleporting reset creates no reward: pre-done origins are masked out.

    The distances are chosen so that leakage would be unmissable — an unmasked
    `k=1` term would score ~+0.70 against the true +1.00, dragging the mean to
    ~0.85.
    """
    from algorithms.feudal_mappo_jax.manager import worker_intrinsic_reward_aligned

    s = jnp.array([[[0.0, 0.0]], [[100.0, 100.0]]])         # (T=2, N=1, 2)
    s_plus = jnp.array([[[1.0, 0.0]], [[100.0, 101.0]]])    # east, then north
    g = jnp.stack([_EAST, _NORTH])[:, None]
    done = jnp.zeros((2, 1, 1)).at[0].set(1.0)              # reset between them

    r = worker_intrinsic_reward_aligned(
        _traj4(s), _traj4(s_plus), _traj4(g), 3, done=done
    )
    # Exactly the k=0 term, and nothing else: no origin from the old episode.
    assert float(r[1, 0, 0]) == pytest.approx(1.0, abs=1e-5)


def test_the_reward_inputs_receive_no_gradient():
    """r^I is DATA. A gradient here would let the manager raise the worker's
    reward by moving the yardstick, and would backprop the worker's objective
    into the manager — which FuN rules out explicitly."""
    from algorithms.feudal_mappo_jax.manager import worker_intrinsic_reward_aligned

    moves = jnp.broadcast_to(_EAST, (4, 1, 2))
    s, s_plus = _positions(moves)
    g = jnp.broadcast_to(_NORTH, (4, 1, 2))

    def total(a, b, c):
        return worker_intrinsic_reward_aligned(
            _traj4(a), _traj4(b), _traj4(c), 3
        ).sum()

    for grad in jax.grad(total, argnums=(0, 1, 2))(s, s_plus, g):
        assert float(jnp.abs(grad).max()) == 0.0


@pytest.mark.parametrize("latent", list(LATENTS))
@pytest.mark.parametrize("core", ["mlp", "dilated_lstm"])
def test_latent_only_is_exactly_the_full_forwards_latent(core, latent):
    """`latent_only=True` is the same `s`, not a cheaper approximation.

    It is what the collector uses to encode the action's successor. `s` is
    feedforward and upstream of the core in every latent variant, so skipping
    the core and the goal head changes nothing about `s` — and means the
    successor read touches no carry and emits no directive, which is why the
    "do not perturb the live hierarchy" requirement is structural here rather
    than a discipline someone has to remember.
    """
    manager, params, carry = _local_manager(core, latent=latent)
    obs = jax.random.normal(jax.random.PRNGKey(3), (N_AGENTS, OBS_DIM))
    gs = obs.reshape(-1)

    full_carry, _, s_full = manager.apply(params, carry, gs, obs)
    s_only = manager.apply(params, carry, gs, obs, latent_only=True)
    assert jnp.array_equal(s_only, s_full), f"{core}/{latent}: latent_only drifted"

    # Interleaving latent-only reads cannot move the recurrent carry.
    for _ in range(3):
        manager.apply(params, carry, gs, obs, latent_only=True)
    again_carry, _, _ = manager.apply(params, carry, gs, obs)
    assert jax.tree.all(
        jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), full_carry, again_carry)
    ) if core == "dilated_lstm" else again_carry is None


def test_next_state_latent_is_the_true_successor_not_the_reset(rollout_alpha):
    """The ordering trap, asserted in both directions.

    `_restart_done` REBINDS `next_obs`/`next_env_state` in place, so encoding
    the successor after it would silently store the freshly reset episode's
    latent and manufacture a reward out of the teleport. Every other check in
    this file would still pass.

    Both directions are needed. Where no reset happened the successor is exactly
    what the next transition observed, so `next_state_latent[t]` must EQUAL
    `state_latent[t+1]` — which also proves the collector's `latent_only` read
    agrees with the full manager forward in situ. Where a reset happened they
    must DIFFER, because the reset intervened; if they matched, the encode has
    moved below the cond.
    """
    _, _, _, _, traj, _, _ = rollout_alpha
    nsl = np.asarray(traj.next_state_latent[:-1])
    sl = np.asarray(traj.state_latent[1:])
    done = np.asarray(traj.done[:-1]).astype(bool)
    assert done.any() and (~done).any(), "need both cases in one rollout"

    # f32 reassociation between two call sites inside one jit, not a semantic
    # gap — the same artifact the module docstring records for the local latents.
    assert np.abs(nsl[~done] - sl[~done]).max() < 1e-6, (
        "successor latent != the next transition's own latent on a step with no "
        "reset — the collector is not encoding the state it stepped into"
    )
    assert np.abs(nsl[done] - sl[done]).max() > 1e-3, (
        "successor latent == the POST-RESET latent on a done step: the encode "
        "has moved below `_restart_done`"
    )


def test_the_final_rollout_action_gets_its_outcome(rollout_alpha):
    """No transition is dropped, including the one with no successor stored.

    A shift-derived endpoint has nothing to put at `T-1` and must pay 0 there.
    """
    _, _, _, _, traj, _, _ = rollout_alpha
    assert float(jnp.abs(traj.next_state_latent[-1]).max()) > 0.0
    assert float(jnp.abs(traj.intrinsic_reward[-1]).max()) > 1e-6


def test_the_intrinsic_truncation_bootstrap_is_added_exactly_once(rollout_alpha):
    """`gamma * V^I(s_next)` rides on truncated steps only, and only once."""
    _, _, _, _, traj, _, _ = rollout_alpha
    boot = np.asarray(traj.intrinsic_bootstrap)
    done = np.asarray(traj.done).astype(bool)
    # The stub env truncates (never terminates), so done == truncated here.
    assert np.abs(boot[~done]).max() == 0.0, "bootstrap paid off a non-truncated step"
    assert np.abs(boot[done]).max() > 0.0, "truncated steps got no bootstrap"


def test_alpha_zero_stores_a_placeholder_successor_latent():
    """alpha=0 stays a STATIC no-op: no encode, and no (T,E,N,D) buffer for it.

    Same idiom as `action_mask` — at production width that buffer is ~50 MB of
    unused zeros, on top of the `goal`/`state_latent` pair that already
    dominates the manager BPTT's peak memory.
    """
    traj0 = _collect(_config(intrinsic_coef=0.0))[3]
    traj1 = _collect(_config(intrinsic_coef=0.5))[3]
    assert traj0.next_state_latent.shape == (N_STEPS,)
    assert traj1.next_state_latent.shape == traj1.state_latent.shape
    # ...and the rollout itself is untouched: the encode consumes no RNG.
    assert jnp.array_equal(traj0.action, traj1.action)
    assert jnp.array_equal(traj0.reward, traj1.reward)
    assert jnp.array_equal(traj0.state_latent, traj1.state_latent)


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


# ---------------------------------------------------------------------------
# The zero-goal isolate arm (conf/model/feudal_zerogoal.yaml)
# ---------------------------------------------------------------------------


def test_zero_goal_makes_the_policy_independent_of_the_manager():
    """The ablation's whole claim: the worker's output cannot depend on `w_t`.

    Checked against *arbitrary* goals rather than a zero goal, because the
    failure this guards is a half-wired ablation that still lets some goal
    signal through (e.g. zeroing after an embedding with a bias, or zeroing only
    one of the two call paths).
    """
    from algorithms.feudal_mappo_jax.worker import init_worker

    obs = jax.random.normal(jax.random.PRNGKey(1), (N_ENVS, N_AGENTS, OBS_DIM))
    g1 = jax.random.normal(jax.random.PRNGKey(2), (N_ENVS, N_AGENTS, GOAL_DIM))
    g2 = jax.random.normal(jax.random.PRNGKey(3), (N_ENVS, N_AGENTS, GOAL_DIM)) * 17.0

    for embed in (None, 4):
        worker, params = init_worker(
            jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 8,
            discrete=False, goal_embed_dim=embed, zero_goal=True,
        )
        m1, s1 = worker.apply(params, obs, g1)
        m2, s2 = worker.apply(params, obs, g2)
        assert jnp.array_equal(m1, m2), f"goal leaked into the policy (embed={embed})"
        assert jnp.array_equal(s1, s2)

    # ...and the ablation is OFF by default: the live arm must still respond.
    worker, params = init_worker(
        jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 8, discrete=False,
    )
    m1, _ = worker.apply(params, obs, g1)
    m2, _ = worker.apply(params, obs, g2)
    assert not jnp.allclose(m1, m2), "zero_goal must default to False"


def test_zero_goal_keeps_the_param_tree_shape_identical():
    """Checkpoints stay interchangeable: the goal columns are kept, not dropped."""
    from algorithms.feudal_mappo_jax.worker import init_worker

    trees = [
        init_worker(jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 8,
                    discrete=False, zero_goal=z)[1]
        for z in (False, True)
    ]
    shapes = [jax.tree.map(jnp.shape, t) for t in trees]
    assert shapes[0] == shapes[1]
    # Same init, too — the arms differ only in what reaches the input.
    for a, b in zip(jax.tree.leaves(trees[0]), jax.tree.leaves(trees[1])):
        assert jnp.array_equal(a, b)


def test_zero_goal_still_trains_the_manager_and_freezes_the_goal_columns():
    """Rungs (2) and (3) must stay live; only goal conditioning is removed.

    The manager must keep learning (its PG is scored on the observed state
    transition, which the worker still produces), and the worker's goal columns
    must receive EXACTLY zero gradient — which is what makes
    `worker_goal_column_ratio` a wiring check for this arm rather than a
    diagnostic of it.
    """
    config = _config(zero_goal=True)
    _, _, new_rs, traj, boot, update_fn = _collect(config)
    updated_rs, losses = update_fn(new_rs, traj, boot)

    for name in ("manager_ts", "manager_critic_ts", "critic_ts", "actor_ts"):
        before = jax.tree.leaves(getattr(new_rs.train_state, name).params)
        after = jax.tree.leaves(getattr(updated_rs.train_state, name).params)
        assert any(not jnp.array_equal(a, b) for a, b in zip(before, after)), (
            f"{name} did not move under zero_goal"
        )

    def goal_cols(ts):
        return ts.params["params"]["MAPPOActor_0"]["Dense_0"]["kernel"][OBS_DIM:]

    assert jnp.array_equal(
        goal_cols(new_rs.train_state.actor_ts),
        goal_cols(updated_rs.train_state.actor_ts),
    ), "goal columns moved under zero_goal (ablation not wired at the input)"

    # The manager diagnostics must stay live and finite, or the arm is not
    # comparable to feudal_a0 on the axes it is supposed to hold fixed.
    for key in ("state_latent_erank", "goal_direction_count", "d_cos_var",
                "manager_explained_variance", "worker_goal_column_ratio"):
        assert np.isfinite(float(losses[key])), (key, losses[key])


def test_normalize_pooled_goal_false_reproduces_the_raw_sum():
    """The pre-2026-08-28 fusion stays exactly reproducible."""
    from algorithms.feudal_mappo_jax.network import MAPPOActor
    from algorithms.feudal_mappo_jax.worker import init_worker

    obs = jax.random.normal(jax.random.PRNGKey(1), (N_ENVS, OBS_DIM))
    goal = jax.random.normal(jax.random.PRNGKey(2), (N_ENVS, GOAL_DIM)) * 10.0

    worker, params = init_worker(
        jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 8,
        discrete=False, normalize_pooled_goal=False,
    )
    # Same params on both sides: flax init is scope-dependent, so a freshly
    # initialized standalone actor would differ for reasons unrelated to fusion.
    ref = MAPPOActor(action_dim=ACTION_DIM, hidden_dim=8, discrete=False)
    m1, s1 = worker.apply(params, obs, goal)
    m2, s2 = ref.apply({"params": params["params"]["MAPPOActor_0"]},
                       jnp.concatenate([obs, goal], axis=-1))
    assert jnp.array_equal(m1, m2) and jnp.array_equal(s1, s2)

    # ...and the default really does normalize: scale-only changes are invisible.
    worker, params = init_worker(
        jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 8, discrete=False,
    )
    a, _ = worker.apply(params, obs, goal)
    b, _ = worker.apply(params, obs, goal * 3.0)
    assert jnp.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Goal-ablation eval variants (trainer.eval_fn)
# ---------------------------------------------------------------------------


# One arbitrary unit direction, for tests that only need the `constant` variant
# to be WELL-FORMED rather than on-policy. The real call sites derive it with
# `mean_goal_direction` over a rollout.
_A_DIRECTION = jnp.eye(GOAL_DIM)[0]


def _eval_state(**overrides):
    """A config + initialized train state, for the eval-variant tests."""
    config = _config(**overrides)
    env = StubEnv()
    init_fn, _, _, eval_fn, _ = make_train(config, env)
    runner_state = init_fn(jax.random.PRNGKey(0))
    return config, env, runner_state.train_state, eval_fn


def test_real_eval_variant_is_unchanged():
    """Running the ablation blocks must not perturb the `reward` series.

    `eval_fn`'s scalar return is the contract `run.py:evaluate()` and the
    training loop's `reward` stat both depend on. If widening the scan to V*E
    envs changed the real block's result, every future run's reward curve would
    be silently incomparable with every past one.
    """
    _, _, train_state, eval_fn = _eval_state()
    key = jax.random.PRNGKey(4)

    # The config default is all four blocks, so it needs a `constant` direction.
    batched = float(eval_fn(train_state, key, constant_goal=_A_DIRECTION))
    alone = float(eval_fn(train_state, key, variants=("real",)))
    assert batched == alone, (batched, alone)


def test_eval_variants_share_reset_keys():
    """Every block starts from bit-identical initial states.

    The gap is a paired statistic. With n_eval_episodes deliberately small, an
    unpaired design would drown a real gap in reset variance, so the pairing has
    to be structural (tile the keys) rather than a property anyone remembers to
    preserve.
    """
    config = _config()
    n_eps = config.n_eval_episodes
    variants = GOAL_VARIANTS
    keys = jnp.tile(jax.random.split(jax.random.PRNGKey(4), n_eps), (len(variants), 1))
    obs, _ = jax.vmap(StubEnv().reset)(keys)

    blocks = jnp.split(obs, len(variants), axis=0)
    for i, b in enumerate(blocks[1:], start=1):
        assert jnp.array_equal(blocks[0], b), f"block {i} starts from a different state"


def test_zeroed_eval_variant_equals_a_zero_goal_worker():
    """The `zeroed` block must be exactly `FeudalWorker(zero_goal=True)`.

    This is the positive control that validates the whole harness, and it holds
    because `_unit(0) == 0` and the optional `goal_embed_dim` Dense is
    bias-free, so `Dense(0) == 0`. It is also why the ablation is applied to the
    goal OUTSIDE the module: the parameter tree stays shape-identical, so
    checkpoints remain interchangeable.

    Consequence used as the offline probe's stop-the-line check: on a
    `feudal_zerogoal` arm all three variants coincide inside the module, so the
    measured gaps must be exactly 0.0.
    """
    _, _, train_state, eval_fn = _eval_state()
    # Same init rng => identical params, since zero_goal does not change shapes.
    _, _, zg_train_state, zg_eval_fn = _eval_state(zero_goal=True)
    key = jax.random.PRNGKey(6)

    chex_equal = jax.tree.all(
        jax.tree.map(
            lambda a, b: bool(jnp.array_equal(a, b)),
            train_state.actor_ts.params,
            zg_train_state.actor_ts.params,
        )
    )
    assert chex_equal, "zero_goal must not perturb the param tree"

    ablated = float(eval_fn(train_state, key, variants=("zeroed",)))
    native = float(zg_eval_fn(zg_train_state, key, variants=("real",)))
    assert ablated == native, (ablated, native)


def test_eval_variant_blocks_do_not_cross_contaminate():
    """Block v's return must depend only on block v's transform.

    The batched-scan-specific risk: the blocks share one scan, one manager
    carry and one goal ring, so a mis-sliced `jnp.split`/`concatenate` (or a
    roll applied after the agent-major flatten, which crosses env boundaries)
    would let one variant's goals drive another's envs. That produces plausible
    numbers, not an error.
    """
    _, _, train_state, eval_fn = _eval_state()
    key = jax.random.PRNGKey(8)

    both, _ = eval_fn(train_state, key, variants=("real", "zeroed"), detail=True)
    real_alone, _ = eval_fn(train_state, key, variants=("real",), detail=True)
    zero_alone, _ = eval_fn(train_state, key, variants=("zeroed",), detail=True)

    assert jnp.array_equal(both[0], real_alone[0]), "real block contaminated"
    assert jnp.array_equal(both[1], zero_alone[0]), "zeroed block contaminated"

    # Reordering the tuple must move the blocks, not the results.
    swapped, _ = eval_fn(train_state, key, variants=("zeroed", "real"), detail=True)
    assert jnp.array_equal(swapped[0], zero_alone[0])
    assert jnp.array_equal(swapped[1], real_alone[0])


def test_permutation_stays_within_its_env():
    """The agent roll must not move a goal across an env boundary.

    `_actor_forward` flattens the pooled goal agent-major to
    `(n_envs*n_agents, goal_dim)`. A roll applied AFTER that flatten would give
    env b's agent 0 the goal of env b-1's agent N-1 — the exact class of silent
    bug CLAUDE.md records for goal flattening, and it would still produce a
    well-shaped, plausible-looking gap.
    """
    from algorithms.feudal_mappo_jax.manager import permute_agent_goals

    goals = jax.random.normal(
        jax.random.PRNGKey(9), (N_ENVS, N_AGENTS, GOAL_DIM)
    )
    rolled = permute_agent_goals(goals, 1)

    for e in range(N_ENVS):
        # Each env's goal multiset is preserved exactly...
        assert jnp.array_equal(
            jnp.sort(goals[e], axis=0), jnp.sort(rolled[e], axis=0)
        ), f"env {e}: goal multiset not preserved"
        # ...and every row came from THAT env, shifted by one agent.
        for i in range(N_AGENTS):
            assert jnp.array_equal(rolled[e, i], goals[e, (i - 1) % N_AGENTS])

    # The wrong implementation (roll after the agent-major flatten) is caught:
    wrong = jnp.roll(goals.reshape(-1, GOAL_DIM), 1, axis=0).reshape(goals.shape)
    assert not jnp.array_equal(wrong, rolled)


def test_constant_variant_is_one_direction_at_the_original_magnitudes():
    """`constant` must replace the DIRECTION and nothing else.

    Preserving each row's norm is what keeps the variant a pure direction
    intervention under `normalize_pooled_goal=False`, where the worker does see
    ||w_t||. Under `True` the magnitude is discarded and the two coincide — so
    this property is invisible on the default config and has to be pinned here
    rather than noticed in a run.
    """
    goals = jax.random.normal(jax.random.PRNGKey(11), (N_ENVS, N_AGENTS, GOAL_DIM))
    goals = goals * jnp.linspace(0.1, 10.0, N_ENVS)[:, None, None]  # varied norms
    out = constant_goals(goals, _A_DIRECTION)

    dirs = out / jnp.linalg.norm(out, axis=-1, keepdims=True)
    assert jnp.allclose(dirs, _A_DIRECTION, atol=1e-5), "not one shared direction"
    assert jnp.allclose(
        jnp.linalg.norm(out, axis=-1), jnp.linalg.norm(goals, axis=-1), atol=1e-4
    ), "row norms not preserved"

    # The direction need not arrive normalized.
    scaled = constant_goals(goals, 7.5 * _A_DIRECTION)
    assert jnp.allclose(scaled, out, atol=1e-4)


def test_mean_goal_direction_weighs_directions_not_lengths():
    """Rows are normalized BEFORE averaging.

    ``||w_t||`` ramps 1 -> c over the first `goal_horizon` steps of every
    episode, so a raw mean would systematically under-weight early-episode
    goals — i.e. the constant would be drawn from a biased sample of the very
    distribution it is standing in for.
    """
    a, b = jnp.eye(GOAL_DIM)[0], jnp.eye(GOAL_DIM)[1]
    # `b` is 100x longer, but there are equally many of each direction.
    goals = jnp.stack([a, 100.0 * b])[:, None, :]
    d = mean_goal_direction(goals)

    assert jnp.allclose(jnp.linalg.norm(d), 1.0, atol=1e-4)
    assert jnp.allclose(d @ a, d @ b, atol=1e-3), "length-weighted, not direction-weighted"

    # A genuinely collapsed manager gives concentration 1.0 ...
    collapsed = jnp.broadcast_to(a, (5, N_AGENTS, GOAL_DIM))
    assert jnp.allclose(mean_goal_direction(collapsed), a, atol=1e-4)


def test_constant_variant_needs_a_direction():
    """Requesting `constant` without one must RAISE, not substitute something.

    `constant` is the only variant that is not a rearrangement of the goals it
    is given, so there is no default that is not a silent choice. A zero or
    random fallback would still produce a well-shaped gap — and a random
    direction measures something else entirely (robustness to noise, not whether
    the manager is decorative).
    """
    _, _, train_state, eval_fn = _eval_state()
    with pytest.raises(ValueError, match="needs a direction"):
        eval_fn(train_state, jax.random.PRNGKey(12), variants=("constant",))


def test_constant_variant_coincides_with_real_for_a_zero_goal_worker():
    """The probe's stop-the-line control, extended to the new variant.

    On a `feudal_zerogoal` arm the worker zeroes the goal INSIDE the module, so
    every transform applied outside it is a no-op and every measured gap must be
    exactly 0.0 — `constant` included. Adding a variant that the control does
    not cover would leave a block whose harness bugs nothing detects.
    """
    _, _, zg_train_state, zg_eval_fn = _eval_state(zero_goal=True)
    key = jax.random.PRNGKey(13)

    rewards, _ = zg_eval_fn(
        zg_train_state,
        key,
        variants=GOAL_VARIANTS,
        detail=True,
        constant_goal=_A_DIRECTION,
        )
    for i, v in enumerate(GOAL_VARIANTS[1:], start=1):
        assert jnp.array_equal(rewards[0], rewards[i]), f"{v} gap is not exactly 0.0"


def test_constant_variant_equals_real_when_the_manager_has_collapsed():
    """If the manager already emits one frozen direction, `constant` IS `real`.

    The semantics the whole diagnostic rests on: a small `gap_constant` means
    the manager's output is worth no more than a frozen vector. This pins the
    limiting case end-to-end through `eval_fn` — equality here is what makes a
    NON-zero gap elsewhere attributable to the goals actually varying.
    """
    _, _, train_state, eval_fn = _eval_state()
    key = jax.random.PRNGKey(14)

    # Force the collapse the variant is designed to detect, by replacing the
    # manager's goals with one direction before the variant transform runs.
    collapsed, _ = eval_fn(
        train_state, key, variants=("constant", "constant"), detail=True,
        constant_goal=_A_DIRECTION,
    )
    assert jnp.array_equal(collapsed[0], collapsed[1])

    # ... and `constant` must NOT be a no-op on a manager that is not collapsed:
    real, _ = eval_fn(train_state, key, variants=("real",), detail=True)
    assert not jnp.array_equal(real[0], collapsed[0]), (
        "constant made no difference at all — either the manager is already "
        "frozen on this direction, or the transform is not being applied"
    )


# ---------------------------------------------------------------------------
# FiLM goal-influence metrics (mappo._film_goal_metrics)
# ---------------------------------------------------------------------------

_FILM_KEYS = (
    "worker_film_gain_rms",
    "worker_film_shift_ratio",
    "worker_tanh_saturation",
    "worker_goal_action_delta",
)


def _film_update(**overrides):
    """Run one real `ppo_update` on a FiLM worker and return its metrics."""
    config = _config(worker_fusion="film", **overrides)
    env = StubEnv()
    init_fn, collect_fn, update_fn, _, _ = make_train(config, env)
    rs = init_fn(jax.random.PRNGKey(0))
    rs, traj, last_value, _ = collect_fn(rs)
    _, losses = update_fn(rs, traj, last_value, jnp.float32(0.0))
    return losses


def test_film_metrics_replace_the_concat_ratio_and_are_not_nan():
    """The gap this closes: `worker_goal_column_ratio` was NaN on every FiLM run.

    `_goal_column_ratio` slices `kernel[obs_dim:]` for the goal block, which
    under FiLM is an empty (0, hidden) array — `jnp.mean` of nothing is NaN, and
    it was logged as such at every point of 84 of 84 feudal trials across both
    12a batches. So the default arm family trained for 1e8 steps with no logged
    signal at all about whether the manager->worker channel was connected.
    """
    losses = _film_update()
    assert "worker_goal_column_ratio" not in losses, (
        "the concat-only ratio must not be emitted for a FiLM worker — that is "
        "the NaN series this replaces"
    )
    for k in _FILM_KEYS:
        assert k in losses, f"{k} missing"
        assert jnp.isfinite(losses[k]), f"{k} is not finite: {losses[k]}"


def test_concat_still_gets_the_column_ratio_and_not_the_film_metrics():
    """Routing by fusion, both directions. A concat worker has no FiLM layers at
    all, so emitting the FiLM keys there would require inventing them."""
    config = _config(worker_fusion="concat")
    env = StubEnv()
    init_fn, collect_fn, update_fn, _, _ = make_train(config, env)
    rs = init_fn(jax.random.PRNGKey(0))
    rs, traj, last_value, _ = collect_fn(rs)
    _, losses = update_fn(rs, traj, last_value, jnp.float32(0.0))

    assert "worker_goal_column_ratio" in losses
    assert jnp.isfinite(losses["worker_goal_column_ratio"])
    for k in _FILM_KEYS:
        assert k not in losses, f"{k} emitted for a concat worker"


def test_goal_column_ratio_raises_on_a_film_kernel_instead_of_returning_nan():
    """Reaching the concat metric with FiLM params is a routing bug, and a
    routing bug must be loud. Returning NaN is what let this go unnoticed across
    ~84 runs, so the failure mode is pinned rather than merely fixed."""
    from algorithms.feudal_mappo_jax.mappo import _goal_column_ratio

    film_params = {"params": {"MAPPOActor_0": {"Dense_0": {"kernel": jnp.zeros((OBS_DIM, 8))}}}}
    with pytest.raises(ValueError, match="concat-only"):
        _goal_column_ratio(film_params, OBS_DIM)


def test_film_metrics_are_exactly_zero_for_a_zero_goal_worker():
    """POSITIVE CONTROL, and it is exact rather than approximate.

    FiLM's gamma/beta Dense layers are bias-free, so `gamma(0) = beta(0) = 0`
    for the life of the run, not just at init. On a `zero_goal` arm the worker
    zeroes the goal inside the module, so all three goal-driven metrics must be
    **bitwise 0.0**. Anything else means they are reading something other than
    the live modulation. Verified on the trained arms too:
    `mjx_12a_3o_trunc_1024/feudal_film_zerogoal` reads 0.000 on all three seeds.

    `worker_tanh_saturation` is deliberately NOT in this control — it is a
    property of the trunk, not of the goal, and is correctly nonzero (0.56-0.61
    on those same trained arms).
    """
    losses = _film_update(zero_goal=True)
    for k in ("worker_film_gain_rms", "worker_film_shift_ratio",
              "worker_goal_action_delta"):
        assert losses[k] == 0.0, f"{k} = {losses[k]}, must be exactly 0.0"
    assert 0.0 <= losses["worker_tanh_saturation"] <= 1.0


def test_film_metrics_are_zero_at_init_and_move_once_the_goal_is_connected():
    """Zero-init makes the floor a KNOWN 0.0, unlike `worker_goal_column_ratio`
    which starts near 1.0 at orthogonal init and whose movement is therefore
    ambiguous between the numerator and the denominator (both documented
    misreadings). Here any nonzero gain is influence that was earned."""
    from algorithms.feudal_mappo_jax.mappo import _film_goal_metrics
    from flax.training.train_state import TrainState
    import optax
    from algorithms.feudal_mappo_jax.worker import init_worker

    worker, params = init_worker(
        jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 32,
        discrete=False, worker_fusion="film",
    )
    ts = TrainState.create(apply_fn=worker.apply, params=params, tx=optax.sgd(0.0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (64, OBS_DIM))
    goal = jax.random.normal(jax.random.PRNGKey(2), (64, GOAL_DIM))

    at_init = _film_goal_metrics(ts, obs, goal)
    assert at_init["worker_film_gain_rms"] == 0.0
    assert at_init["worker_film_shift_ratio"] == 0.0
    assert at_init["worker_goal_action_delta"] == 0.0

    # Give film_0 a nonzero gain kernel: the metrics must register it.
    connected = jax.tree.map(lambda x: x, ts.params)
    connected["params"]["film_0"]["Dense_0"]["kernel"] = jnp.ones((GOAL_DIM, 32)) * 0.1
    moved = _film_goal_metrics(ts.replace(params=connected), obs, goal)
    assert moved["worker_film_gain_rms"] > 0.0
    assert moved["worker_goal_action_delta"] > 0.0


def test_film_sow_does_not_change_the_param_tree_or_the_forward():
    """The `is_initializing()` guard on the sow is load-bearing.

    `sow` fires under `init` as well, and `init_worker`'s return IS the train
    state's `params`. Without the guard the extra "diagnostics" collection would
    reach `create_train_state`, the optimizer and every msgpack site, making the
    tree incompatible with all 84 existing FiLM checkpoints.
    """
    from algorithms.feudal_mappo_jax.worker import init_worker

    worker, params = init_worker(
        jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 32,
        discrete=False, worker_fusion="film",
    )
    assert list(params.keys()) == ["params"], (
        f"init returned extra collections {list(params.keys())} — this changes "
        f"the checkpoint tree"
    )

    obs = jax.random.normal(jax.random.PRNGKey(1), (5, OBS_DIM))
    goal = jax.random.normal(jax.random.PRNGKey(2), (5, GOAL_DIM))
    plain = worker.apply(params, obs, goal)
    sown, _ = worker.apply(params, obs, goal, mutable=["diagnostics"])
    assert jnp.array_equal(plain[0], sown[0]), "sow perturbed the forward pass"


# ---------------------------------------------------------------------------
# Stats alignment (the resume hazard for any newly-added metric series)
# ---------------------------------------------------------------------------


def test_to_dict_left_pads_short_series():
    """A series that started late must be padded at the FRONT, to alignment.

    `append_agent_stats` is a bare defaultdict append, so a metric added after a
    run began — or absent from the checkpoint a run resumed from — is created on
    its first append and stays permanently shorter than `total_steps`. Nothing
    detects it, and the notebook then plots it against `range(1, len+1)`, i.e.
    silently shifted left and averaged against other runs' wrong iterations.

    Padding the FRONT is what matters: element 0 of a late series belongs to the
    iteration it first existed for, not to iteration 0.
    """
    import math

    from algorithms.mappo_vanilla.trainer_components import TrainingStatsTracker

    tracker = TrainingStatsTracker()
    tracker.training_stats["total_steps"] = [10, 20, 30, 40]
    tracker.training_stats["policy_loss"] = [1.0, 2.0, 3.0, 4.0]  # aligned
    tracker.training_stats["eval_gap_permuted"] = [7.0, 8.0]      # started late
    tracker.training_stats["action_distribution"] = []            # legitimately empty

    out = tracker.to_dict()

    assert out["policy_loss"] == [1.0, 2.0, 3.0, 4.0], "aligned series must not move"
    padded = out["eval_gap_permuted"]
    assert len(padded) == 4
    assert math.isnan(padded[0]) and math.isnan(padded[1]), "must pad the FRONT"
    assert padded[2:] == [7.0, 8.0], "existing values must keep their iterations"
    # Empty is a real state ("never recorded"), not a short one: padding it would
    # make it ragged and break the notebook cell that reads it.
    assert out["action_distribution"] == []


# ---------------------------------------------------------------------------
# FiLM fusion (conf/model/feudal_film.yaml)
# ---------------------------------------------------------------------------


def _film_worker(**kw):
    from algorithms.feudal_mappo_jax.worker import init_worker

    return init_worker(
        jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 8,
        discrete=False, worker_fusion="film", **kw,
    )


def test_film_is_identity_at_init():
    """At init the FiLM worker must be EXACTLY the flat actor, for any goal.

    This is the property the whole arm rests on: zero-init kernels give
    gamma = beta = 0, so `h * (1 + 0) + 0 == h`. It makes the arm start at the
    mappo_jax baseline and forces goal influence to be *earned*, which is the
    reverse of concat -- where orthogonal init makes the goal columns live from
    step 0 and goal-agnosticism is what has to be learned.

    It is also the free pre-flight check for the goal-dependence probe: before
    training, every eval variant must agree bitwise.
    """
    from algorithms.feudal_mappo_jax.network import MAPPOActor

    worker, params = _film_worker()
    obs = jax.random.normal(jax.random.PRNGKey(1), (N_ENVS, N_AGENTS, OBS_DIM))
    g1 = jax.random.normal(jax.random.PRNGKey(2), (N_ENVS, N_AGENTS, GOAL_DIM))
    g2 = jax.random.normal(jax.random.PRNGKey(3), (N_ENVS, N_AGENTS, GOAL_DIM)) * 31.0

    m1, s1 = worker.apply(params, obs, g1)
    m2, s2 = worker.apply(params, obs, g2)
    assert jnp.array_equal(m1, m2), "goal must not reach the policy at init"
    assert jnp.array_equal(s1, s2)

    # ...and it equals the FLAT actor on the same trunk weights, so the identity
    # is against mappo_jax's policy rather than merely self-consistent.
    flat = MAPPOActor(action_dim=ACTION_DIM, hidden_dim=8, discrete=False)
    flat_mean, flat_std = flat.apply(
        {"params": params["params"]["MAPPOActor_0"]}, obs
    )
    assert jnp.array_equal(m1, flat_mean), "FiLM at init must BE the flat actor"
    assert jnp.array_equal(s1, flat_std)

    # The two structural properties, asserted directly on the param tree.
    assert "film_0" in params["params"] and "film_1" in params["params"]
    for name in ("film_0", "film_1"):
        for dense in params["params"][name].values():
            assert jnp.all(dense["kernel"] == 0.0), f"{name} must be zero-init"
            assert "bias" not in dense, f"{name} must be bias-free"


def test_film_becomes_goal_sensitive_once_gamma_is_nonzero():
    """Zero-init sets the DEFAULT; it must not disconnect the goal permanently."""
    worker, params = _film_worker()
    obs = jax.random.normal(jax.random.PRNGKey(1), (N_ENVS, N_AGENTS, OBS_DIM))
    g1 = jax.random.normal(jax.random.PRNGKey(2), (N_ENVS, N_AGENTS, GOAL_DIM))
    g2 = jax.random.normal(jax.random.PRNGKey(3), (N_ENVS, N_AGENTS, GOAL_DIM))

    p = jax.tree.map(lambda x: x, params)  # copy
    k = list(p["params"]["film_0"].keys())[0]
    p["params"]["film_0"][k]["kernel"] = jnp.ones_like(
        p["params"]["film_0"][k]["kernel"]
    ) * 0.1

    m1, _ = worker.apply(p, obs, g1)
    m2, _ = worker.apply(p, obs, g2)
    assert not jnp.allclose(m1, m2), "goal must reach the policy once gamma != 0"

    # And a gradient exists at the zero-init point, so it can get there.
    def loss(prm):
        mean, _ = worker.apply(prm, obs, g1)
        return jnp.sum(mean**2)

    grads = jax.grad(loss)(params)
    gnorm = sum(
        float(jnp.sum(jnp.abs(v["kernel"])))
        for v in grads["params"]["film_0"].values()
    )
    assert gnorm > 0.0, "zero-init must not be a dead start"


def test_film_zero_goal_is_exactly_the_flat_policy_after_training():
    """Bias-free => gamma(0) = beta(0) = 0 FOREVER, not just at init.

    This is what keeps the probe's `zeroed` variant interpretable: with a bias,
    a trained gamma(0) != 0 would make that arm "flat policy + a learned
    constant modulation" and the control would silently drift.
    """
    worker, params = _film_worker()
    obs = jax.random.normal(jax.random.PRNGKey(1), (N_ENVS, N_AGENTS, OBS_DIM))
    zero_g = jnp.zeros((N_ENVS, N_AGENTS, GOAL_DIM))

    # Simulate "after training": arbitrary nonzero FiLM kernels.
    trained = jax.tree.map(lambda x: x, params)
    for name in ("film_0", "film_1"):
        for k in trained["params"][name]:
            trained["params"][name][k]["kernel"] = jax.random.normal(
                jax.random.PRNGKey(hash(name + k) % 2**31),
                trained["params"][name][k]["kernel"].shape,
            )

    at_zero, _ = worker.apply(trained, obs, zero_g)
    at_init, _ = worker.apply(params, obs, zero_g)
    assert jnp.array_equal(at_zero, at_init), (
        "a zero goal must give the flat policy regardless of trained FiLM weights"
    )


def test_film_and_concat_are_not_checkpoint_compatible():
    """Guard the documented shape difference so it cannot be silently assumed."""
    from algorithms.feudal_mappo_jax.worker import init_worker

    _, film_p = _film_worker()
    _, concat_p = init_worker(
        jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 8, discrete=False,
    )
    d0 = lambda p: p["params"]["MAPPOActor_0"]["Dense_0"]["kernel"].shape
    assert d0(concat_p) == (OBS_DIM + GOAL_DIM, 8)
    assert d0(film_p) == (OBS_DIM, 8)


def test_unknown_worker_fusion_raises():
    """Fail loudly rather than silently falling back to one of the two fusions."""
    from algorithms.feudal_mappo_jax.worker import init_worker

    with pytest.raises(ValueError, match="unknown worker_fusion"):
        init_worker(
            jax.random.PRNGKey(0), OBS_DIM, GOAL_DIM, ACTION_DIM, 8,
            discrete=False, worker_fusion="bilinear",
        )


def test_concat_path_is_unchanged_by_the_modulate_hook():
    """`modulate=None` must leave MAPPOActor byte-identical to the pre-hook code."""
    from algorithms.feudal_mappo_jax.network import MAPPOActor

    actor = MAPPOActor(action_dim=ACTION_DIM, hidden_dim=8, discrete=False)
    x = jax.random.normal(jax.random.PRNGKey(1), (N_ENVS, OBS_DIM + GOAL_DIM))
    p = actor.init(jax.random.PRNGKey(0), x)
    a, _ = actor.apply(p, x)
    b, _ = actor.apply(p, x, None)
    assert jnp.array_equal(a, b)


# ---------------------------------------------------------------------------
# Manager latent locality (`manager_latent="local"`)
#
# The centralized latent computes `s = Dense(N*goal_dim)(z)` and reshapes, so
# `s_i = W_i z + b_i`: the agent axis is a SLICE INDEX. Measured over 12 trained
# arms (2026-09-09, `latent_locality_probe.py`), the diagonal share of the block
# Jacobian `d s[i]/d obs_j` is 0.0631 against a uniform 1/16 of 0.0625 — i.e. no
# localization at all. Since `worker_intrinsic_reward` scores `s_t[i]-s_{t-k}[i]`,
# agent i's "own" intrinsic reward then moves as much when a TEAMMATE moves as
# when it does.
#
# `"local"` fixes that structurally, and these tests pin the structure rather
# than any trained outcome — a property that holds by construction is worth
# nothing if a refactor quietly reintroduces the mixing.
# ---------------------------------------------------------------------------


# Every latent that runs the encoder per agent — the 2x2 over the `private` and
# `global` axes. Derived from the manager's own tuples rather than spelled out,
# so adding a fifth variant cannot leave these lists behind (the failure mode
# would be a new arm that silently runs no structural test at all).
ALL_LOCAL_VARIANTS = list(LOCAL_LATENTS)
# The latents whose projections into goal space are SHARED across agents. The
# two tests that assert permutation-equivariance of `s` and a shared
# `f_goalhead` are properties of exactly these; the `PRIVATE_LATENTS` trade both
# away deliberately, while still having to satisfy the locality and detach-rule
# invariants.
LOCAL_VARIANTS = [l for l in ALL_LOCAL_VARIANTS if l not in PRIVATE_LATENTS]
# The projection params each latent carries, for the detach-rule test: the
# gradient must reach the encoder AND whatever plays the role of `f_Mspace`.
_MSPACE_PARAM = {
    l: "f_Mspace_agent_kernel" if l in PRIVATE_LATENTS else "f_Mspace"
    for l in ALL_LOCAL_VARIANTS
}


def _local_manager(core="mlp", n_agents=N_AGENTS, obs_dim=OBS_DIM, latent="local"):
    config = _config(manager_core=core, manager_latent=latent)
    manager = build_manager(config, n_agents)
    carry = manager.initialize_carry(jax.random.PRNGKey(0), ())
    params = manager.init(
        jax.random.PRNGKey(1),
        carry,
        jnp.zeros(n_agents * obs_dim),
        jnp.zeros((n_agents, obs_dim)),
    )
    return manager, params, carry


@pytest.mark.parametrize("latent", ALL_LOCAL_VARIANTS)
@pytest.mark.parametrize("core", ["mlp", "dilated_lstm"])
def test_local_latent_is_exactly_agent_local(core, latent):
    """`d s[i]/d obs_j` is BITWISE zero for j != i — the whole point of the mode.

    Asserted exactly (`== 0.0`), not within a tolerance: the encoder is applied
    per agent, so the off-diagonal blocks are structurally absent from the graph
    rather than merely small. A tolerance here would let a partial reintroduction
    of joint-state mixing pass.
    """
    manager, params, carry = _local_manager(core, latent=latent)
    obs = jax.random.normal(jax.random.PRNGKey(2), (N_AGENTS, OBS_DIM))

    def s_of(o):
        return manager.apply(params, carry, o.reshape(-1), o)[2]

    jac = jax.jacrev(s_of)(obs)  # (N, goal_dim, N, obs_dim)
    blocks = np.asarray(jnp.sqrt((jac**2).sum(axis=(1, 3))))  # (N, N)
    off = blocks - np.diag(np.diag(blocks))
    assert np.abs(off).max() == 0.0, (
        f"core={core}/{latent}: s[i] depends on another agent's observation; max "
        f"off-diagonal block norm {np.abs(off).max()}"
    )
    share = np.diag(blocks) / blocks.sum(axis=1)
    assert np.allclose(share, 1.0), share


@pytest.mark.parametrize("latent", LOCAL_VARIANTS)
def test_local_latent_shares_one_encoder_across_agents(latent):
    """Permuting the agent axis permutes `s` rows identically.

    This is what a SHARED encoder buys beyond locality: all rows land in one
    basis, so "goal coordinate 3" means the same thing for every agent. Per-agent
    encoders would be equally local and still leave N private coordinate systems
    — the second defect named in
    plans/feudal_goal_reward_diagnosis_2026-09-09.md.
    """
    manager, params, carry = _local_manager(latent=latent)
    obs = jax.random.normal(jax.random.PRNGKey(3), (N_AGENTS, OBS_DIM))
    perm = np.array([2, 0, 3, 1])
    s_ref = manager.apply(params, carry, obs.reshape(-1), obs)[2]
    s_perm = manager.apply(params, carry, obs[perm].reshape(-1), obs[perm])[2]
    assert jnp.array_equal(s_ref[perm], s_perm), (
        "encoder is not shared across agents: permuting agents did not permute "
        "the latent rows identically"
    )


@pytest.mark.parametrize("latent", LOCAL_VARIANTS)
def test_local_goal_head_is_shared_so_g_and_s_share_a_basis(latent):
    """The final projection to goal space is one shared `(goal_dim, goal_dim)`.

    `f_goalhead` must be shared for the same reason `f_Mspace` is: `d_cos` scores
    `g_i` against a displacement of `s_i`, so the two have to be expressed in the
    same coordinates. A per-agent final layer (what the centralized branch has,
    `goal_head: (H, N*goal_dim)`) would give every agent its own goal basis.
    """
    _, params, _ = _local_manager(latent=latent)
    mgr = params["params"]
    assert mgr["f_goalhead"]["kernel"].shape == (GOAL_DIM, GOAL_DIM), mgr[
        "f_goalhead"
    ]["kernel"].shape
    assert mgr["f_Mspace"]["kernel"].shape[1] == GOAL_DIM, mgr["f_Mspace"][
        "kernel"
    ].shape


@pytest.mark.parametrize("latent", ALL_LOCAL_VARIANTS)
def test_local_latent_still_trains_f_Mspace_under_the_detach_rule(latent):
    """The core must consume `s`, or the detach starves the encoder.

    `transition_cosine(detach_states=True)` kills the gradient through the TARGET
    arm of the cosine; `f_enc`/`f_Mspace` survive only because `g_t(theta)` still
    depends on them THROUGH the core. Wire the core to anything else and this
    goes to exactly 0.0 while every loss stays finite and every diagnostic stays
    healthy — the same self-sealing failure `manager.py` self-check [8] guards on
    the centralized branch.
    """
    from algorithms.feudal_mappo_jax.manager import transition_cosine

    manager, params, carry = _local_manager(latent=latent)
    T = 8
    obs = jax.random.normal(jax.random.PRNGKey(4), (T, N_ENVS, N_AGENTS, OBS_DIM))

    def loss(p):
        _, goal, s = manager.apply(p, carry, obs.reshape(T, N_ENVS, -1), obs)
        cos, valid = transition_cosine(s, goal, HORIZON, detach_states=True)
        return -(cos * valid).sum() / jnp.maximum(valid.sum(), 1.0)

    grads = jax.grad(loss)(params)["params"]
    for name in ("f_enc_0", "f_enc_1", _MSPACE_PARAM[latent]):
        leaf = grads[name]
        # The stacked per-agent projection is a bare param, not a Dense.
        g = float(jnp.abs(leaf["kernel"] if isinstance(leaf, dict) else leaf).sum())
        assert g > 0.0, (
            f"{latent}/{name} received no gradient ({g}) — the bottleneck broke"
        )


def test_centralized_latent_ignores_the_obs_argument():
    """The default branch is untouched by the new argument.

    `manager.apply(..., gs)` and `manager.apply(..., gs, obs)` must agree
    bitwise, so every existing arm, checkpoint and probe keeps its meaning.
    """
    config = _config()
    manager = build_manager(config, N_AGENTS)
    carry = manager.initialize_carry(jax.random.PRNGKey(0), ())
    gs = jax.random.normal(jax.random.PRNGKey(5), (N_AGENTS * OBS_DIM,))
    params = manager.init(jax.random.PRNGKey(1), carry, gs)
    obs = jax.random.normal(jax.random.PRNGKey(6), (N_AGENTS, OBS_DIM))
    _, g0, s0 = manager.apply(params, carry, gs)
    _, g1, s1 = manager.apply(params, carry, gs, obs)
    assert jnp.array_equal(g0, g1) and jnp.array_equal(s0, s1)


@pytest.mark.parametrize("latent", LOCAL_VARIANTS)
def test_local_and_centralized_are_not_checkpoint_compatible(latent):
    """Loading one into the other must FAIL rather than silently half-work.

    The manager trees differ in both names and shapes (f_enc_0/f_gpre/f_goalhead
    vs f_percept_0/goal_head; f_Mspace is (H, goal_dim) vs (H, N*goal_dim)).
    `goal_dependence_probe._dims_from_checkpoint` keys off `f_enc_0` to tell them
    apart — without that it would infer `goal_dim // n_agents` for a local run.
    """
    _, local_params, _ = _local_manager(latent=latent)
    config = _config()
    central = build_manager(config, N_AGENTS)
    carry = central.initialize_carry(jax.random.PRNGKey(0), ())
    central_params = central.init(
        jax.random.PRNGKey(1), carry, jnp.zeros(N_AGENTS * OBS_DIM)
    )
    assert set(local_params["params"]) != set(central_params["params"])
    assert "f_enc_0" in local_params["params"]
    assert "f_enc_0" not in central_params["params"]
    assert (
        local_params["params"]["f_Mspace"]["kernel"].shape
        != central_params["params"]["f_Mspace"]["kernel"].shape
    )


# ---------------------------------------------------------------------------
# `manager_latent="local_private"` — shared encoder, PER-AGENT projections.
#
# Measured 2026-09-14 (`latent_diversity_probe.py`, trained 12a arms): given the
# SAME observations (participation ratio ~4 of 12 across the agent axis),
# `"local"` produced `s` rows at PR 1.03-2.89 and goals at 1.46-1.67, while
# `"centralized"` produced 8.92-9.22 and 8.73-9.10. A shared projection can only
# TRANSMIT input row diversity; per-agent blocks MANUFACTURE it. These tests pin
# that mechanism exactly, at init, with no training involved.
# ---------------------------------------------------------------------------


def test_local_private_manufactures_row_diversity_that_local_cannot():
    """Identical observations => `"local"` gives identical `s` rows, `"local_private"` does not.

    The sharpest possible statement of why this latent exists, and it needs no
    training: feed every agent the SAME observation. A shared projection is a
    function, so `s_1 == s_2 == ... == s_N` EXACTLY — the rows carry zero
    information about which agent is which, whatever the encoder learned. The
    per-agent blocks break that tie structurally.

    Real observations are correlated rather than identical, which is why the
    measured gap is 1.0-2.9 vs 8.9-9.2 rather than 1.0 vs N; this test is the
    limiting case that isolates the cause.
    """
    obs_one = jax.random.normal(jax.random.PRNGKey(7), (OBS_DIM,))
    obs = jnp.broadcast_to(obs_one, (N_AGENTS, OBS_DIM))

    def rows(latent):
        manager, params, carry = _local_manager(latent=latent)
        return manager.apply(params, carry, obs.reshape(-1), obs)[2]

    shared = rows("local")
    for i in range(1, N_AGENTS):
        assert jnp.array_equal(shared[0], shared[i]), (
            "latent='local' gave distinct rows for identical observations — the "
            "projection is no longer shared"
        )

    private = rows("local_private")
    gram = np.asarray(
        (private / jnp.linalg.norm(private, axis=-1, keepdims=True))
        @ (private / jnp.linalg.norm(private, axis=-1, keepdims=True)).T
    )
    off = gram[~np.eye(N_AGENTS, dtype=bool)]
    assert np.abs(off).max() < 0.99, (
        "latent='local_private' rows are collinear for identical observations; "
        f"max |cos| {np.abs(off).max():.4f} — the per-agent blocks are not "
        "distinguishing agents"
    )
    # Participation ratio: N^2 / ||G||_F^2, the live `goal_direction_count` stat.
    pr = N_AGENTS**2 / float((gram**2).sum())
    assert pr > 1.5, f"local_private participation ratio {pr:.2f} is near-collapsed"


def test_block_orthogonal_blocks_are_mutually_near_orthogonal():
    """Per-agent blocks must be orthogonal to EACH OTHER, not merely internally.

    Independently drawn orthogonal matrices are orthogonal within themselves and
    share directions with one another, which would weaken the diversity this
    latent exists to restore from step 0. `block_orthogonal` splits ONE
    orthogonal matrix instead, reproducing the centralized branch's measured
    block statistics (mean |cos| 0.013-0.019 after training there).
    """
    from algorithms.feudal_mappo_jax.manager import block_orthogonal

    n, d_in, d_out = N_AGENTS, 64, GOAL_DIM
    w = block_orthogonal(n)(jax.random.PRNGKey(0), (n, d_in, d_out))
    assert w.shape == (n, d_in, d_out)
    flat = np.asarray(w).reshape(n, -1)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    gram = flat @ flat.T
    off = np.abs(gram[~np.eye(n, dtype=bool)])
    assert off.max() < 0.2, f"blocks share directions; max |cos| {off.max():.3f}"
    with pytest.raises(ValueError, match="expected"):
        block_orthogonal(n)(jax.random.PRNGKey(0), (n + 1, d_in, d_out))


def test_local_private_pairs_s_and_g_with_matched_per_agent_projections():
    """Both projections are per-agent and the SAME shape — the basis argument.

    `d_cos(s_t[i] - s_{t-k}[i], g_{t-k}[i])` contracts per agent and never
    compares agent i's axes to agent j's, so what it requires is that `s_i` and
    `g_i` agree FOR EACH i. A per-agent `s` with a SHARED goal head would break
    exactly that, which is why `f_gpre`/`f_goalhead` are gone here rather than
    only `f_Mspace` being replaced.
    """
    _, params, _ = _local_manager(latent="local_private")
    mgr = params["params"]
    assert "f_Mspace" not in mgr and "f_goalhead" not in mgr and "f_gpre" not in mgr
    s_w, g_w = mgr["f_Mspace_agent_kernel"], mgr["goal_head_agent_kernel"]
    assert s_w.shape == g_w.shape, (s_w.shape, g_w.shape)
    assert s_w.shape[0] == N_AGENTS and s_w.shape[2] == GOAL_DIM, s_w.shape


def test_local_private_is_not_checkpoint_compatible_with_the_shared_latents():
    """Loading across must FAIL, and the probe must tell them apart.

    `local_private` shares `f_enc_0` with the other local variants, so the
    `(f_enc_0, f_percept_0)` pair that separates `local` from `local_global` is
    NOT enough. `_dims_from_checkpoint` keys off `f_Mspace_agent_kernel` and
    checks it first; reading `f_Mspace` here would raise rather than mis-infer,
    which is the safe direction but only because the key is genuinely absent.
    """
    private = _local_manager(latent="local_private")[1]["params"]
    shared = _local_manager(latent="local")[1]["params"]
    assert set(private) != set(shared)
    assert "f_enc_0" in private, "locality marker missing"
    assert "f_Mspace" in shared and "f_Mspace" not in private

    def infer(mgr):
        if "f_Mspace_agent_kernel" in mgr:
            return "local_private"
        local, percept = "f_enc_0" in mgr, "f_percept_0" in mgr
        return ("local_global" if percept else "local") if local else "centralized"

    assert infer(private) == "local_private"
    assert infer(shared) == "local"
    assert infer(_local_manager(latent="local_global")[1]["params"]) == "local_global"
    # goal_dim must come off the stacked kernel's LAST axis, not the second.
    assert int(private["f_Mspace_agent_kernel"].shape[2]) == GOAL_DIM


def test_unknown_manager_latent_raises():
    """Fail loudly, not open — the `VARIANTS` StrEnum lesson in CLAUDE.md."""
    config = _config(manager_latent="bogus")
    manager = build_manager(config, N_AGENTS)
    with pytest.raises(ValueError, match="latent must be"):
        manager.init(
            jax.random.PRNGKey(0), None, jnp.zeros(N_AGENTS * OBS_DIM)
        )


@pytest.mark.parametrize("latent", LOCAL_VARIANTS)
def test_local_latent_requires_obs(latent):
    """`latent='local'` with no obs raises instead of silently using the wrong input."""
    manager, params, carry = _local_manager(latent=latent)
    with pytest.raises(ValueError, match="needs `obs`"):
        manager.apply(params, carry, jnp.zeros(N_AGENTS * OBS_DIM))


# ---------------------------------------------------------------------------
# `manager_latent="local_global"` — the SMAX variant.
#
# `s` stays a pure function of `obs` (so `r^I` stays clean) while GOAL GENERATION
# regains the env's own global state. It exists because on SMAX `global_state` is
# NOT recoverable from the observations: `SMAX.get_obs` zeroes any unit beyond
# the viewer's sight range, and measured at t=0 on 3m/5m_vs_6m/2s3z/3s5z, 100% of
# alive enemies are invisible to EVERY ally (a spawn property, not a policy
# artifact). On MJX `global_state` IS `obs.reshape(E, -1)`, so this variant is
# redundant there.
#
# These tests feed `global_state` and `obs` as INDEPENDENT arrays, which the
# trainer never does for an MJX env — that is the point: it is the only way to
# separate the two input paths and show which output depends on which.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent", ALL_LOCAL_VARIANTS)
def test_state_latent_never_depends_on_the_global_state(latent):
    """`s` is blind to `global_state` in ALL FOUR local variants.

    This is what keeps `r^I` clean under the `global` axis: whatever the goal
    path gains, the measuring stick stays a function of the agent's own
    observation. If this ever fails, `local_global`/`local_global_private` has
    silently become a centralized latent with extra steps and the whole locality
    result is void.
    """
    manager, params, carry = _local_manager(latent=latent)
    obs = jax.random.normal(jax.random.PRNGKey(7), (N_AGENTS, OBS_DIM))
    gs_a = jax.random.normal(jax.random.PRNGKey(8), (N_AGENTS * OBS_DIM,))
    gs_b = jax.random.normal(jax.random.PRNGKey(9), (N_AGENTS * OBS_DIM,))
    s_a = manager.apply(params, carry, gs_a, obs)[2]
    s_b = manager.apply(params, carry, gs_b, obs)[2]
    assert jnp.array_equal(s_a, s_b), (
        f"{latent}: the state latent moved when only the global state changed"
    )
    # And the gradient path is absent, not merely small.
    g = jax.grad(lambda x: manager.apply(params, carry, x, obs)[2].sum())(gs_a)
    assert float(jnp.abs(g).max()) == 0.0


@pytest.mark.parametrize("latent", ALL_LOCAL_VARIANTS)
def test_the_goal_reads_the_global_state_for_exactly_the_global_latents(latent):
    """The `global` axis, asserted in BOTH directions over the whole 2x2.

    Without it the manager is a pure function of `obs`, so an env whose global
    state carries information the observations do not (SMAX: 100% of alive
    enemies are invisible to every ally at t=0) is unreachable to it. With it the
    goal path sees it. Both directions matter: a `local_global*` arm that
    silently ignored the extra input would be plain `local*` wearing an extra
    `f_percept`, and a `local*` arm that read it would void the locality claim
    for the goal.

    This is the axis that is INDEPENDENT of `private`, so it is parametrized over
    all four rather than asserted on the two single-axis names — that is what
    makes `local_global_private` covered rather than assumed.
    """
    obs = jax.random.normal(jax.random.PRNGKey(7), (N_AGENTS, OBS_DIM))
    gs_a = jax.random.normal(jax.random.PRNGKey(8), (N_AGENTS * OBS_DIM,))
    gs_b = jax.random.normal(jax.random.PRNGKey(9), (N_AGENTS * OBS_DIM,))

    m, p, c = _local_manager(latent=latent)
    moved = not jnp.allclose(
        m.apply(p, c, gs_a, obs)[1], m.apply(p, c, gs_b, obs)[1]
    )
    expected = latent in GLOBAL_LATENTS
    assert moved == expected, (
        f"latent={latent!r}: goal-vs-global_state dependence is wrong "
        f"(moved={moved}, expected={expected})"
    )
    assert ("f_percept_0" in p["params"]) == expected


def test_every_latent_is_distinguishable_from_the_checkpoint_alone(tmp_path):
    """`_dims_from_checkpoint` must separate all FIVE, by PAIRS of keys.

    No single key does it. `f_enc_0` says "local family" but not which of the
    four; `f_percept_0` is shared by `centralized`, `local_global` and
    `local_global_private`; `f_Mspace_agent_kernel` says "private" but not
    whether it also reads the global state. The two axes are independent, so the
    inference needs both bits.

    This calls the REAL `_dims_from_checkpoint` on serialized trees rather than
    reimplementing its rule, because a reimplementation is exactly what would
    have kept passing when `local_global_private` was added: the probe's own
    early-return for `f_Mspace_agent_kernel` hardcoded `"local_private"`, and a
    copy of the rule living here would have been "fixed" in lockstep with
    nothing. The yaml moves while checkpoints do not, so this inference is what
    makes a probe a measurement of what actually trained.
    """
    from flax.serialization import msgpack_serialize

    from algorithms.feudal_mappo_jax.goal_dependence_probe import (
        _dims_from_checkpoint,
    )

    config = _config()
    central = build_manager(config, N_AGENTS)
    trees = {
        "centralized": central.init(
            jax.random.PRNGKey(1),
            central.initialize_carry(jax.random.PRNGKey(0), ()),
            jnp.zeros(N_AGENTS * OBS_DIM),
        )["params"]
    }
    for latent in ALL_LOCAL_VARIANTS:
        trees[latent] = _local_manager(latent=latent)[1]["params"]
    assert set(trees) == {"centralized", *LOCAL_LATENTS}

    for expected, mgr in trees.items():
        # Minimal checkpoint: the probe reads the manager tree plus the actor's
        # first Dense (for `hidden_dim`).
        path = tmp_path / f"{expected}.msgpack"
        path.write_bytes(
            msgpack_serialize(
                jax.device_get(
                    {
                        "manager": {"params": mgr},
                        "actor": {
                            "params": {
                                "MAPPOActor_0": {
                                    "Dense_0": {
                                        "kernel": jnp.zeros(
                                            (OBS_DIM, config.hidden_dim)
                                        )
                                    }
                                }
                            }
                        },
                    }
                )
            )
        )
        dims = _dims_from_checkpoint(path, N_AGENTS)
        assert dims["manager_latent"] == expected, (
            expected, dims["manager_latent"], sorted(mgr)
        )
        # goal_dim must come back as the TRUE width for every latent — the
        # centralized branch stores `n_agents * goal_dim` in one Dense, so a
        # missed division (or a spurious one) silently rebuilds a different
        # network, and for a width that happens to divide it loads without error.
        assert dims["goal_dim"] == GOAL_DIM, (expected, dims["goal_dim"])
        # The private branch reads its widths off a 3-D stacked param
        # `(n_agents, manager_hidden, goal_dim)` rather than a 2-D Dense kernel,
        # so a transposed axis here would produce a plausible-looking wrong width.
        assert dims["manager_hidden_dim"] == config.manager_hidden_dim, (
            expected, dims["manager_hidden_dim"]
        )


# ---------------------------------------------------------------------------
# The pure-intrinsic worker arm
# (conf/model/feudal_film_intrinsic_only_local_private.yaml)
#
# `worker_objective: intrinsic_only` drops the extrinsic advantage from the
# ACTOR's objective, so the worker's only job is to follow the manager's goals.
# These pin the seams where that is silent: a no-op guarantee for every existing
# arm, the actual independence from the extrinsic stream, the fact that alpha
# cannot anneal the objective away, and the four config guards.
# ---------------------------------------------------------------------------


def _actor_params_after_update(config, progress=0.0):
    """Post-update actor params for a config, from a fixed seed."""
    _, _, new_rs, traj, boot, update_fn = _collect(config)
    updated_rs, _ = update_fn(new_rs, traj, boot, jnp.float32(progress))
    return updated_rs.train_state.actor_ts.params


def test_mixed_objective_is_the_default_and_is_unchanged():
    """The no-op guarantee: every existing arm must be bit-identical.

    `worker_objective` defaults to "mixed", and stating it explicitly must be
    the same computation — otherwise adding this knob silently moved 84 trained
    arms' objective.
    """
    assert MAPPOConfig().worker_objective == "mixed"

    implicit = _actor_params_after_update(_config(intrinsic_coef=0.5))
    explicit = _actor_params_after_update(
        _config(intrinsic_coef=0.5, worker_objective="mixed")
    )
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)),
        implicit,
        explicit,
    )


def test_intrinsic_only_drops_the_extrinsic_advantage():
    """The arm's whole claim, stated as an invariance.

    Scaling the stored EXTRINSIC reward by 1000x must leave the actor exactly
    where it was — the extrinsic stream is not in its objective at all. The same
    scaling on the INTRINSIC stream must move it, or the test would also pass
    for an actor that simply gets no gradient.

    (Only the actor: the extrinsic critic still regresses that return and will
    move, which is deliberate — see `..._still_trains_the_worker_critic`.)
    """
    config = _config(
        intrinsic_coef=1.0,
        intrinsic_anneal="none",
        worker_objective="intrinsic_only",
    )
    _, _, new_rs, traj, boot, update_fn = _collect(config)

    base = update_fn(new_rs, traj, boot, jnp.float32(0.0))[0].train_state.actor_ts.params

    scaled_ext = update_fn(
        new_rs,
        traj._replace(reward=traj.reward * 1000.0, value=traj.value * 1000.0),
        boot._replace(worker=boot.worker * 1000.0),
        jnp.float32(0.0),
    )[0].train_state.actor_ts.params
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)),
        base,
        scaled_ext,
    )

    # Positive control: the intrinsic stream IS in the objective. Shift rather
    # than scale — normalization divides a pure scale straight back out (that is
    # what `test_alpha_is_a_gradient_fraction...` pins), so a scale would be a
    # no-op here too and would not distinguish "used" from "ignored".
    perturbed_int = update_fn(
        new_rs,
        traj._replace(intrinsic_reward=traj.intrinsic_reward[::-1]),
        boot,
        jnp.float32(0.0),
    )[0].train_state.actor_ts.params
    moved = any(
        jax.tree.leaves(
            jax.tree.map(
                lambda a, b: bool(jnp.any(a != b)), base, perturbed_int
            )
        )
    )
    assert moved, (
        "the actor is insensitive to the intrinsic stream too — it is getting no "
        "gradient at all, not an intrinsic-only one"
    )


def test_intrinsic_only_ignores_alpha_so_the_anneal_cannot_delete_the_objective():
    """alpha is not a coefficient here, at any point in training.

    `adv_int` is already unit-std, so an alpha factor would be a uniform rescale
    that the default `intrinsic_anneal: "linear"` would drive to EXACTLY 0 —
    silently deleting the worker's whole objective while every logged loss stayed
    healthy. `run.py` forbids the schedule; this pins that the VALUE is inert
    too, so the guard is belt-and-braces rather than the only thing standing
    between the arm and a dead second half.
    """
    kw = dict(intrinsic_anneal="none", worker_objective="intrinsic_only")
    small = _actor_params_after_update(_config(intrinsic_coef=0.1, **kw))
    large = _actor_params_after_update(_config(intrinsic_coef=1.0, **kw))
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6
        ),
        small,
        large,
    )

    # ...and the same params whether we are at the start or the end of training.
    end = _actor_params_after_update(_config(intrinsic_coef=1.0, **kw), progress=1.0)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6
        ),
        large,
        end,
    )


def test_intrinsic_only_still_trains_the_worker_critic_and_the_manager():
    """Only the ACTOR's advantage changes; nothing else is switched off.

    The extrinsic critic is deliberately kept trained: it keeps the param tree
    shape-identical to the matched `mixed` control (so checkpoints stay
    interchangeable, the same reason `zero_goal` zeroes at the input) and keeps
    `explained_variance` a live cross-arm diagnostic. The manager is what holds
    all the task pressure here, so it must move too.
    """
    config = _config(
        intrinsic_coef=1.0,
        intrinsic_anneal="none",
        worker_objective="intrinsic_only",
    )
    _, _, new_rs, traj, boot, update_fn = _collect(config)
    updated_rs, losses = update_fn(new_rs, traj, boot, jnp.float32(0.0))

    for name in ("critic_ts", "manager_ts", "manager_critic_ts"):
        before = getattr(new_rs.train_state, name).params
        after = getattr(updated_rs.train_state, name).params
        assert any(
            jax.tree.leaves(
                jax.tree.map(lambda a, b: bool(jnp.any(a != b)), before, after)
            )
        ), f"{name} did not move under intrinsic_only"

    assert "explained_variance" in losses

    # Shape-identical to the mixed arm => checkpoints remain interchangeable.
    mixed_rs = _collect(_config(intrinsic_coef=1.0, intrinsic_anneal="none"))[2]
    jax.tree.map(
        lambda a, b: (_ for _ in ()).throw(AssertionError((a.shape, b.shape)))
        if a.shape != b.shape
        else None,
        new_rs.train_state.actor_ts.params,
        mixed_rs.train_state.actor_ts.params,
    )


def test_the_logged_weights_say_which_objective_ran():
    """`alpha_current` alone is misleading under intrinsic_only (it is inert).

    The pair `adv_ext_weight` / `adv_int_weight` are the coefficients that
    actually multiply the two normalized streams, so the stats record what the
    objective WAS without the reader having to know the rule.
    """
    _, _, rs_m, traj_m, boot_m, upd_m = _collect(
        _config(intrinsic_coef=0.5, intrinsic_anneal="none")
    )
    _, mixed = upd_m(rs_m, traj_m, boot_m, jnp.float32(0.0))
    assert float(mixed["adv_ext_weight"]) == pytest.approx(1.0)
    assert float(mixed["adv_int_weight"]) == pytest.approx(0.5)

    _, _, rs_i, traj_i, boot_i, upd_i = _collect(
        _config(
            intrinsic_coef=0.5,
            intrinsic_anneal="none",
            worker_objective="intrinsic_only",
        )
    )
    _, only = upd_i(rs_i, traj_i, boot_i, jnp.float32(0.0))
    assert float(only["adv_ext_weight"]) == pytest.approx(0.0)
    assert float(only["adv_int_weight"]) == pytest.approx(1.0)


def test_unknown_worker_objective_raises():
    with pytest.raises(ValueError, match="worker_objective"):
        _collect(_config(worker_objective="bogus"))


def test_intrinsic_only_requires_a_live_intrinsic_stream():
    """At alpha=0 no r^I exists, so the actor would train on a zero advantage."""
    with pytest.raises(ValueError, match="intrinsic_coef"):
        _collect(
            _config(
                intrinsic_coef=0.0,
                intrinsic_anneal="none",
                worker_objective="intrinsic_only",
            )
        )


def test_intrinsic_only_rejects_an_anneal_schedule():
    """A schedule on a factor that is not in the expression reads as a lie."""
    with pytest.raises(ValueError, match="intrinsic_anneal"):
        _collect(
            _config(
                intrinsic_coef=1.0,
                intrinsic_anneal="linear",
                worker_objective="intrinsic_only",
            )
        )


def test_intrinsic_only_warns_on_a_non_local_latent_but_still_runs():
    """Warn, do not block — and neither extreme.

    r^I is the worker's ENTIRE objective here, and under `centralized` it is not
    agent-local (measured diag share of d s[i]/d obs_j = 0.0631 against a uniform
    1/N of 0.0625), so each worker would optimize a team-aggregate signal it does
    not control. That is worth shouting about, but the combination is also the
    direct contrast that TESTS whether locality is what matters, so it must stay
    runnable on purpose.
    """
    from algorithms.feudal_mappo_jax.mappo import validate_worker_objective

    base = dict(
        intrinsic_coef=1.0,
        intrinsic_anneal="none",
        worker_objective="intrinsic_only",
    )
    with pytest.warns(RuntimeWarning, match="not agent-local"):
        validate_worker_objective(_config(manager_latent="centralized", **base))

    # A local latent must be silent, or the warning is noise that gets filtered.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for latent in LOCAL_LATENTS:
            validate_worker_objective(_config(manager_latent=latent, **base))

    # And it really does run: the warning is not a disguised abort.
    with pytest.warns(RuntimeWarning):
        _, _, new_rs, traj, boot, update_fn = _collect(
            _config(manager_latent="centralized", **base)
        )
    update_fn(new_rs, traj, boot, jnp.float32(0.0))


def test_zero_goal_is_still_rejected_under_intrinsic_only():
    """No new guard — the existing zero_goal check must already cover it.

    A zero-goal worker cannot see the goals, so making them its ONLY objective is
    the most uninterpretable arm in the stack. It is caught by run.py's
    pre-existing `zero_goal && intrinsic_coef != 0` rule, which intrinsic_only
    forces the second half of; this pins that so a refactor of that guard cannot
    quietly open the hole.
    """
    config = _config(
        intrinsic_coef=1.0,
        intrinsic_anneal="none",
        worker_objective="intrinsic_only",
        zero_goal=True,
    )
    assert config.zero_goal and config.intrinsic_coef != 0.0, (
        "intrinsic_only no longer implies the condition run.py's zero_goal guard "
        "keys on — that arm needs its own guard now"
    )
