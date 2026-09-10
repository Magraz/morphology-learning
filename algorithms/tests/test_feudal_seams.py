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


@pytest.mark.parametrize("latent", ["centralized", "local"])
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


@pytest.mark.parametrize("latent", ["centralized", "local"])
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

    batched = float(eval_fn(train_state, key))          # 3 variants, one scan
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
    variants = ("real", "permuted", "zeroed")
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


LOCAL_VARIANTS = ["local", "local_global"]


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


@pytest.mark.parametrize("latent", LOCAL_VARIANTS)
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


@pytest.mark.parametrize("latent", LOCAL_VARIANTS)
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
    for name in ("f_enc_0", "f_enc_1", "f_Mspace"):
        g = float(jnp.abs(grads[name]["kernel"]).sum())
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


@pytest.mark.parametrize("latent", LOCAL_VARIANTS)
def test_state_latent_never_depends_on_the_global_state(latent):
    """`s` is blind to `global_state` in BOTH local variants.

    This is what keeps `r^I` clean under `local_global`: whatever the goal path
    gains, the measuring stick stays a function of the agent's own observation.
    If this ever fails, `local_global` has silently become a centralized latent
    with extra steps and the whole locality result is void.
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


def test_local_global_goal_reads_the_global_state_but_plain_local_does_not():
    """The one behavioural difference between the two local variants.

    Under `local` the manager is a pure function of `obs`, so an env whose global
    state carries information the observations do not (SMAX) is unreachable to
    it. Under `local_global` the goal path sees it. Asserted in both directions
    so neither variant can silently drift into the other.
    """
    obs = jax.random.normal(jax.random.PRNGKey(7), (N_AGENTS, OBS_DIM))
    gs_a = jax.random.normal(jax.random.PRNGKey(8), (N_AGENTS * OBS_DIM,))
    gs_b = jax.random.normal(jax.random.PRNGKey(9), (N_AGENTS * OBS_DIM,))

    m_l, p_l, c_l = _local_manager(latent="local")
    assert jnp.array_equal(
        m_l.apply(p_l, c_l, gs_a, obs)[1], m_l.apply(p_l, c_l, gs_b, obs)[1]
    ), "latent='local' goal depends on the global state — f_percept leaked in"
    assert "f_percept_0" not in p_l["params"]

    m_g, p_g, c_g = _local_manager(latent="local_global")
    assert not jnp.allclose(
        m_g.apply(p_g, c_g, gs_a, obs)[1], m_g.apply(p_g, c_g, gs_b, obs)[1]
    ), "latent='local_global' goal ignored the global state"
    assert "f_percept_0" in p_g["params"]


def test_the_three_latents_are_distinguishable_from_the_checkpoint_alone():
    """`_dims_from_checkpoint` must separate all three, and by the PAIR of keys.

    `f_enc_0` alone does not do it (both local variants have it) and
    `f_percept_0` alone does not either (centralized and local_global share it).
    The yaml moves while checkpoints do not, so this inference is what makes a
    probe a measurement of what actually trained.
    """
    trees = {
        "centralized": None,
        "local": _local_manager(latent="local")[1]["params"],
        "local_global": _local_manager(latent="local_global")[1]["params"],
    }
    config = _config()
    central = build_manager(config, N_AGENTS)
    trees["centralized"] = central.init(
        jax.random.PRNGKey(1),
        central.initialize_carry(jax.random.PRNGKey(0), ()),
        jnp.zeros(N_AGENTS * OBS_DIM),
    )["params"]

    def infer(mgr):
        local, percept = "f_enc_0" in mgr, "f_percept_0" in mgr
        return ("local_global" if percept else "local") if local else "centralized"

    for expected, mgr in trees.items():
        assert infer(mgr) == expected, (expected, sorted(mgr))
    # goal_dim inference: divided by n_agents only on the centralized branch.
    assert trees["centralized"]["f_Mspace"]["kernel"].shape[1] == N_AGENTS * GOAL_DIM
    for k in ("local", "local_global"):
        assert trees[k]["f_Mspace"]["kernel"].shape[1] == GOAL_DIM
