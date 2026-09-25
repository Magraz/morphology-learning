"""Seam tests for `algorithms/simplified_feudal_mappo_jax`.

Run on a CPU stub env (deterministic and fast, unlike an MJX rollout). The stub's
observation starts with each agent's own position, so every stored quantity can
be recomputed from the stored data without calling the trainer's code again.

    uv run pytest algorithms/tests/test_simplified_feudal.py -q
"""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorithms.mappo_jax.mappo import create_train_state
from algorithms.mappo_jax.network import MAPPOCritic, evaluate_action
from algorithms.mappo_jax.types import MAPPOConfig
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.trainer import (
    make_train,
    validate_env,
)
from algorithms.simplified_feudal_mappo_jax.types import FeudalConfig, Model_Params


@pytest.fixture(autouse=True)
def _run_on_cpu():
    """Pin to CPU: the tests are tiny, and sharing a GPU with a training job
    produces cuSolver/OOM failures that read like assertion failures (same
    fixture as `test_feudal_seams.py` / `test_smax_seams.py`)."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


OBS_DIM = 4
N_ENVS = 3
HORIZON = 4
N_STEPS = 12  # 3 manager windows
RADIUS = 0.1
SPEED = 0.02  # position change per step at full action


class StubState(NamedTuple):
    pos: jnp.ndarray  # (N, 2)
    t: jnp.ndarray


class StubEnv:
    """Point agents that move `SPEED * clip(action)` per step.

    `goal_state` is the position itself, and the observation begins with it, so a
    test can reconstruct positions (and hence waypoints) from stored inputs.
    `terminate_at` / `max_steps` place a termination / truncation mid-window.
    """

    observation_dim = OBS_DIM
    action_dim = 2
    discrete = False
    goal_state_dim = 2

    def __init__(self, n_agents, max_steps=100, terminate_at=None):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.terminate_at = terminate_at

    def _obs(self, state):
        return jnp.concatenate([state.pos, jnp.sin(3.0 * state.pos)], axis=-1)

    def reset(self, key):
        pos = jax.random.uniform(key, (self.n_agents, 2), minval=-0.3, maxval=0.3)
        state = StubState(pos=pos, t=jnp.zeros((), jnp.int32))
        return self._obs(state), state

    def step(self, state, actions):
        pos = state.pos + SPEED * jnp.clip(actions, -1.0, 1.0)
        state = StubState(pos=pos, t=state.t + 1)
        reward = -jnp.linalg.norm(pos, axis=-1).mean()
        terminated = (
            state.t >= self.terminate_at
            if self.terminate_at is not None
            else jnp.array(False)
        )
        truncated = state.t >= self.max_steps
        info = {"task_reward": reward}
        return self._obs(state), state, reward, terminated, truncated, info

    def goal_state(self, state):
        return state.pos


def _config(n_steps=N_STEPS):
    worker = MAPPOConfig(
        n_steps=n_steps,
        n_envs=N_ENVS,
        n_epochs=2,
        n_minibatches=2,
        hidden_dim=16,
        n_total_steps=n_steps * N_ENVS * 4,
        n_eval_episodes=2,
    )
    manager = dataclasses.replace(
        worker, gamma=worker.gamma**HORIZON, n_minibatches=1
    )
    return FeudalConfig(
        worker=worker, manager=manager, goal_horizon=HORIZON, waypoint_radius=RADIUS
    )


def _collect(env, seed=0):
    config = _config()
    init_fn, collect_fn, update_fn, eval_fn, _ = make_train(config, env)
    runner_state = init_fn(jax.random.PRNGKey(seed))
    out = collect_fn(runner_state)
    return config, runner_state, out, (update_fn, eval_fn)


def _stored_pos_and_waypoint(worker_traj):
    """(T, E, N, 2) positions and waypoints, rebuilt from the worker's input."""
    obs = np.asarray(worker_traj.obs)
    pos = obs[..., :2]
    error = obs[..., OBS_DIM : OBS_DIM + 2]
    return pos, pos + RADIUS * error


def _windows(x):
    """(T, ...) -> (n_windows, HORIZON, ...)."""
    return x.reshape((N_STEPS // HORIZON, HORIZON) + x.shape[1:])


@pytest.mark.parametrize("n_agents", [1, 3])
def test_waypoints_are_fixed_within_a_window_and_redrawn_between(n_agents):
    _, _, (_, rollout, _, _), _ = _collect(StubEnv(n_agents))
    _, waypoint = _stored_pos_and_waypoint(rollout.worker)
    w = _windows(waypoint)
    np.testing.assert_allclose(w, np.broadcast_to(w[:, :1], w.shape), atol=1e-5)
    # A fresh stochastic decision per window.
    assert np.abs(w[1, 0] - w[0, 0]).max() > 1e-4
    # The time-left input counts down 1, 3/4, 1/2, 1/4 in every window.
    rem = _windows(np.asarray(rollout.worker.obs)[..., -1])
    expected = (HORIZON - np.arange(HORIZON)) / HORIZON
    np.testing.assert_allclose(rem, np.broadcast_to(
        expected[None, :, None, None], rem.shape), atol=1e-6)


@pytest.mark.parametrize("n_agents", [1, 3])
def test_worker_reward_telescopes_over_a_commitment(n_agents):
    """Summed over a window, the worker's reward is exactly the distance to the
    waypoint it closed, in units of R."""
    _, _, (_, rollout, _, _), _ = _collect(StubEnv(n_agents))
    pos, waypoint = _stored_pos_and_waypoint(rollout.worker)
    reward = _windows(np.asarray(rollout.worker.reward)).sum(axis=1)
    pos_w, w_w = _windows(pos), _windows(waypoint)
    for j in range(N_STEPS // HORIZON - 1):  # needs the next window's start
        start = np.linalg.norm(w_w[j, 0] - pos_w[j, 0], axis=-1)
        end = np.linalg.norm(w_w[j, 0] - pos_w[j + 1, 0], axis=-1)
        np.testing.assert_allclose(reward[j], (start - end) / RADIUS, atol=1e-4)
    # Only the last step of each window is a worker terminal.
    done = _windows(np.asarray(rollout.worker.done))
    assert done[:, -1].all() and not done[:, :-1].any()


@pytest.mark.parametrize("n_agents", [1, 3])
def test_env_that_ends_mid_window_is_frozen_then_reset(n_agents):
    # Terminates on global step 5 = window 1, step k=1.
    env = StubEnv(n_agents, terminate_at=6)
    config, _, (_, rollout, _, stats), _ = _collect(env)
    w, m = rollout.worker, rollout.manager
    active = np.asarray(w.active_mask)
    assert active[:6].all() and not active[6:8].any() and active[8:].all()
    # Frozen steps hold the terminal state and pay nothing.
    obs = np.asarray(w.obs)[..., :OBS_DIM]
    np.testing.assert_array_equal(obs[6], obs[7])
    assert np.all(np.asarray(w.reward)[6:8] == 0.0)
    assert np.all(np.asarray(w.team_reward)[6:8] == 0.0)
    assert np.asarray(w.done)[5:8].all()
    # The env restarts at the window boundary: positions jump by more than one
    # step could move them.
    assert np.abs(obs[8, ..., :2] - obs[7, ..., :2]).max() > 2 * SPEED
    # Manager: only window 1 ended an episode; its reward is the discounted team
    # reward of the two live steps (no bootstrap: a true termination).
    np.testing.assert_array_equal(
        np.asarray(m.done), np.array([[False] * N_ENVS, [True] * N_ENVS, [False] * N_ENVS])
    )
    team = np.asarray(w.team_reward)
    gamma = config.worker.gamma
    np.testing.assert_allclose(
        np.asarray(m.reward)[1], team[4] + gamma * team[5], rtol=1e-5
    )
    assert int(stats["episode_count"]) == N_ENVS


@pytest.mark.parametrize("n_agents", [1, 3])
def test_time_limit_mid_window_bootstraps_both_levels(n_agents):
    # Truncates on global step 5 = window 1, step k=1.
    env = StubEnv(n_agents, max_steps=6)
    config, _, (runner_state, rollout, _, _), _ = _collect(env)
    ts = runner_state.train_state
    w, m = rollout.worker, rollout.manager
    gamma = config.worker.gamma

    next_obs = np.asarray(w.obs)[6, ..., :OBS_DIM]  # the frozen successor of step 5
    next_pos = next_obs[..., :2]
    next_gs = next_obs.reshape(N_ENVS, -1)  # stub has no global_state hook

    # Manager: r_4 + gamma r_5 + gamma^2 V^M(s_6).
    v_m = ts.manager.critic_ts.apply_fn(
        ts.manager.critic_ts.params, wp.manager_critic_input(next_gs, next_pos)
    )
    team = np.asarray(w.team_reward)
    np.testing.assert_allclose(
        np.asarray(m.reward)[1],
        team[4] + gamma * team[5] + gamma**2 * np.asarray(v_m),
        rtol=1e-5,
    )

    # Worker at step 5: its intrinsic reward + gamma V_w(s_6, time left 1/2).
    pos, waypoint = _stored_pos_and_waypoint(w)
    r_int = (
        np.linalg.norm(waypoint[5] - pos[5], axis=-1)
        - np.linalg.norm(waypoint[5] - next_pos, axis=-1)
    ) / RADIUS
    v_w = ts.worker.critic_ts.apply_fn(
        ts.worker.critic_ts.params,
        wp.worker_critic_input(
            next_gs, wp.goal_error(waypoint[5], next_pos, RADIUS), 0.5
        ),
    )
    np.testing.assert_allclose(
        np.asarray(w.reward)[5], r_int + gamma * np.asarray(v_w), atol=1e-5
    )


@pytest.mark.parametrize("n_agents", [1, 3])
def test_ppo_ratio_is_exactly_one_before_update_at_both_levels(n_agents):
    """The stored inputs must re-evaluate to the stored log-probs, or PPO's
    importance ratio compares two different distributions."""
    _, _, (runner_state, rollout, _, _), _ = _collect(StubEnv(n_agents))
    ts = runner_state.train_state
    for level, traj in (("worker", rollout.worker), ("manager", rollout.manager)):
        actor = getattr(ts, level).actor_ts
        d = traj.obs.shape[-1]
        log_prob, _ = evaluate_action(
            actor.apply_fn,
            actor.params,
            traj.obs.reshape(-1, d),
            traj.action.reshape(-1, traj.action.shape[-1]),
            discrete=False,
        )
        np.testing.assert_allclose(
            log_prob, np.asarray(traj.log_prob).reshape(-1), atol=1e-5, err_msg=level
        )


@pytest.mark.parametrize("n_agents", [1, 3])
def test_collect_update_eval_end_to_end(n_agents):
    config, _, (runner_state, rollout, last_values, _), (update_fn, eval_fn) = (
        _collect(StubEnv(n_agents, max_steps=10))
    )
    n_windows = N_STEPS // HORIZON
    assert rollout.worker.value.shape == (N_STEPS, N_ENVS, n_agents)
    assert rollout.worker.reward.shape == (N_STEPS, N_ENVS, n_agents)
    assert rollout.manager.value.shape == (n_windows, N_ENVS)
    assert rollout.manager.action.shape == (n_windows, N_ENVS, n_agents, 2)
    assert last_values.worker.shape == (N_ENVS, n_agents)

    new_state, losses = update_fn(runner_state, rollout, last_values)
    for key in (
        "worker_policy_loss", "worker_value_loss", "worker_explained_variance",
        "manager_policy_loss", "manager_value_loss", "manager_explained_variance",
        "intrinsic_reward", "waypoint_reached_frac", "waypoint_final_error",
        "waypoint_offset", "manager_window_return", "rollout_team_reward",
    ):
        assert np.isfinite(float(losses[key])), key
    # Both levels actually moved.
    for level in ("worker", "manager"):
        before = getattr(runner_state.train_state, level).actor_ts.params
        after = getattr(new_state.train_state, level).actor_ts.params
        moved = jax.tree.map(lambda a, b: float(jnp.abs(a - b).max()), before, after)
        assert max(jax.tree.leaves(moved)) > 0.0, level

    ret = float(eval_fn(new_state.train_state, jax.random.PRNGKey(1)))
    assert np.isfinite(ret)


def test_waypoints_stay_in_the_arena_and_within_radius():
    pos = jnp.array([[0.45, -0.45], [0.0, 0.0]])
    action = jnp.array([[5.0, -5.0], [0.5, -2.0]])
    w = wp.waypoint_from_action(pos, action, 0.2)
    np.testing.assert_allclose(w, [[0.5, -0.5], [0.1, -0.2]], atol=1e-6)


def test_critic_keep_output_axis_is_inert_by_default():
    """The mappo_jax change must not alter any existing critic: same params,
    and the scalar head still squeezes."""
    x = jnp.ones((5, 7))
    plain, kept = MAPPOCritic(hidden_dim=8), MAPPOCritic(hidden_dim=8, keep_output_axis=True)
    params = plain.init(jax.random.PRNGKey(0), x)
    params_kept = kept.init(jax.random.PRNGKey(0), x)
    assert jax.tree.all(jax.tree.map(lambda a, b: bool((a == b).all()), params, params_kept))
    assert plain.apply(params, x).shape == (5,)
    assert kept.apply(params, x).shape == (5, 1)
    ts = create_train_state(jax.random.PRNGKey(0), MAPPOConfig(hidden_dim=8), 3, 7, 2, False)
    assert ts.critic_ts.apply_fn(ts.critic_ts.params, x).shape == (5,)


def test_env_without_positions_is_rejected():
    class NoPositions(StubEnv):
        goal_state = None

        def __getattribute__(self, name):
            if name == "goal_state":
                raise AttributeError(name)
            return super().__getattribute__(name)

    with pytest.raises(ValueError, match="goal_state"):
        validate_env(NoPositions(2))


def test_model_group_without_hierarchy_keys_is_rejected():
    """`model=mlp` carries only hidden_dim; it must fail rather than write into
    the mappo_jax baseline's results directory."""
    with pytest.raises(TypeError):
        Model_Params(hidden_dim=168)
