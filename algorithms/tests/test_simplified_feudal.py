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
import optax
import pytest
from flax.training.train_state import TrainState

from algorithms.mappo_jax.mappo import create_train_state, ppo_update
from algorithms.mappo_jax.network import (
    MAPPOActor,
    MAPPOCritic,
    _gaussian_entropy,
    _squashed_gaussian_entropy,
    evaluate_action,
    sample_action,
)
from algorithms.mappo_jax.types import MAPPOConfig
from algorithms.simplified_feudal_mappo_jax import counterfactual as cf
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.trainer import (
    create_hier_train_state,
    make_policy,
    make_train,
    validate_env,
)
from algorithms.simplified_feudal_mappo_jax.types import (
    FeudalConfig,
    Model_Params,
    Params,
)


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
    `teleport_agents` is the training-fork hook (`interventions.py`);
    `fail_teleport` makes every placement fail.
    """

    observation_dim = OBS_DIM
    action_dim = 2
    discrete = False
    goal_state_dim = 2

    def __init__(self, n_agents, max_steps=100, terminate_at=None, fail_teleport=False):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.terminate_at = terminate_at
        self.fail_teleport = fail_teleport

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

    def teleport_agents(self, state, focal, recruit_mask, offset, key, radius):
        """Put each recruit `radius` from the focal agent at a random bearing.
        Invalid when `fail_teleport` is set or a recruit's translated waypoint
        `pos + offset` would leave [-0.5, 0.5]; the state is then unchanged."""
        angle = jax.random.uniform(key, (self.n_agents,), maxval=2 * jnp.pi)
        target = state.pos[focal] + radius * jnp.stack(
            [jnp.cos(angle), jnp.sin(angle)], axis=-1
        )
        moved = jnp.where(recruit_mask[:, None], target, state.pos)
        in_bounds = jnp.where(recruit_mask[:, None], jnp.abs(moved + offset) <= 0.5, True)
        valid = jnp.all(in_bounds) & (not self.fail_teleport)
        new = StubState(pos=jnp.where(valid, moved, state.pos), t=state.t)
        return self._obs(new), new, valid

    # Two fixed boxes for `manager_input: relative`; box 0 counts as delivered
    # from step 6 on, so the masking is exercised mid-rollout.
    BOX_POS = jnp.array([[0.2, 0.25], [-0.25, 0.1]])

    def entity_state(self, state):
        agents = jnp.concatenate([state.pos, jnp.zeros_like(state.pos)], axis=-1)
        delivered = jnp.array([1.0, 0.0]) * (state.t >= 6)
        boxes = jnp.concatenate(
            [
                self.BOX_POS,
                (0.5 - self.BOX_POS[:, 1:2]),  # goal distance
                delivered[:, None],
                jnp.zeros((2, 1)),  # touch fraction
                jnp.full((2, 1), 0.5),  # coupling fraction
            ],
            axis=-1,
        )
        return agents, boxes


def _config(
    n_steps=N_STEPS,
    bound="clip",
    manager_input="global",
    manager_step_gamma=None,
    manager_credit="team",
    counterfactual_default="sampled",
    counterfactual_samples=4,
    dpp_coef=1.0,
    dpp_max_recruits=None,
    interventions=False,
    intervention_radius=0.05,
    intervention_interval=1,
):
    worker = MAPPOConfig(
        n_steps=n_steps,
        n_envs=N_ENVS,
        n_epochs=2,
        n_minibatches=2,
        hidden_dim=16,
        n_total_steps=n_steps * N_ENVS * 4,
        n_eval_episodes=2,
    )
    step_gamma = worker.gamma if manager_step_gamma is None else manager_step_gamma
    manager = dataclasses.replace(
        worker, gamma=step_gamma**HORIZON, n_minibatches=1
    )
    return FeudalConfig(
        worker=worker,
        manager=manager,
        goal_horizon=HORIZON,
        waypoint_radius=RADIUS,
        manager_action_bound=bound,
        manager_input=manager_input,
        manager_step_gamma=manager_step_gamma,
        manager_credit=manager_credit,
        counterfactual_default=counterfactual_default,
        counterfactual_samples=counterfactual_samples,
        dpp_coef=dpp_coef,
        dpp_max_recruits=dpp_max_recruits,
        interventions=interventions,
        intervention_radius=intervention_radius,
        intervention_interval=intervention_interval,
    )


def _collect(env, seed=0, bound="clip", **config_kwargs):
    config = _config(bound=bound, **config_kwargs)
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


@pytest.mark.parametrize("manager_input", ["global", "relative", "local"])
@pytest.mark.parametrize("bound", ["clip", "tanh"])
@pytest.mark.parametrize("n_agents", [1, 3])
def test_ppo_ratio_is_exactly_one_before_update_at_both_levels(
    n_agents, bound, manager_input
):
    """The stored inputs must re-evaluate to the stored log-probs, or PPO's
    importance ratio compares two different distributions. Under `tanh` the
    stored manager action is the pre-tanh sample, so this holds unchanged."""
    _, _, (runner_state, rollout, _, _), _ = _collect(
        StubEnv(n_agents), bound=bound, manager_input=manager_input
    )
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


@pytest.mark.parametrize("manager_input", ["global", "relative", "local"])
@pytest.mark.parametrize("bound", ["clip", "tanh"])
@pytest.mark.parametrize("n_agents", [1, 3])
def test_collect_update_eval_end_to_end(n_agents, bound, manager_input):
    config, _, (runner_state, rollout, last_values, _), (update_fn, eval_fn) = (
        _collect(
            StubEnv(n_agents, max_steps=10), bound=bound, manager_input=manager_input
        )
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
        "manager_action_saturation",
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


@pytest.mark.parametrize(
    "bound, expected",
    [
        ("clip", [[0.5, -0.5], [0.1, -0.2]]),
        ("tanh", [[0.5, -0.5], [0.2 * np.tanh(0.5), -0.2 * np.tanh(2.0)]]),
    ],
)
def test_waypoints_stay_in_the_arena_and_within_radius(bound, expected):
    pos = jnp.array([[0.45, -0.45], [0.0, 0.0]])
    action = jnp.array([[5.0, -5.0], [0.5, -2.0]])
    w = wp.waypoint_from_action(pos, action, 0.2, bound)
    np.testing.assert_allclose(w, expected, atol=1e-6)


def test_tanh_moves_the_waypoint_where_clip_is_flat():
    """The defect `tanh` removes: under `clip` two different samples past the
    bound give the SAME waypoint, so the advantage cannot tell them apart."""
    pos = jnp.zeros((1, 2))
    a, b = jnp.array([[2.0, 0.0]]), jnp.array([[5.0, 0.0]])
    clip_a, clip_b = (wp.waypoint_from_action(pos, x, 0.2, "clip") for x in (a, b))
    tanh_a, tanh_b = (wp.waypoint_from_action(pos, x, 0.2, "tanh") for x in (a, b))
    np.testing.assert_array_equal(clip_a, clip_b)
    assert float(tanh_b[0, 0] - tanh_a[0, 0]) > 0.0


def test_unknown_action_bound_is_rejected():
    with pytest.raises(ValueError, match="manager_action_bound"):
        make_train(_config(bound="sigmoid"), StubEnv(2))
    assert Model_Params(
        hidden_dim=8, goal_horizon=4, waypoint_radius=0.1
    ).manager_action_bound == "clip"


# ------------------------------------------------------- relative manager input


def _entity(pos, box_pos, delivered, vel=None):
    """(1, N, 4) agent and (1, O, 6) box blocks, laid out like `entity_state`."""
    pos, box_pos = jnp.asarray(pos, jnp.float32), jnp.asarray(box_pos, jnp.float32)
    vel = jnp.zeros_like(pos) if vel is None else jnp.asarray(vel, jnp.float32)
    o = box_pos.shape[0]
    boxes = jnp.concatenate(
        [
            box_pos,
            0.5 - box_pos[:, 1:2],
            jnp.asarray(delivered, jnp.float32)[:, None],
            jnp.zeros((o, 1)),
            jnp.full((o, 1), 0.5),
        ],
        axis=-1,
    )
    return jnp.concatenate([pos, vel], axis=-1)[None], boxes[None]


def test_relative_input_puts_the_nearest_undelivered_box_first_and_zeroes_delivered():
    agents, boxes = _entity(
        [[0.0, -0.4]], [[0.3, 0.0], [0.0, -0.1], [0.05, -0.35]], [0, 0, 1]
    )
    x = np.asarray(wp.manager_actor_input_relative(agents, boxes))[0, 0]
    box = x[4 : 4 + 18].reshape(3, 6)
    # box 1 (0.3 away) first, then box 0 (0.5 away); box 2 is nearer but delivered.
    np.testing.assert_allclose(box[0], [0.0, 0.3, 0.6, 0.0, 0.5, 1.0], atol=1e-6)
    np.testing.assert_allclose(box[1], [0.3, 0.4, 0.5, 0.0, 0.5, 1.0], atol=1e-6)
    np.testing.assert_array_equal(box[2], np.zeros(6))


def test_relative_input_ignores_box_order_and_where_delivered_boxes_are():
    """What makes the states after a delivery look like the states before one:
    a delivered box is REMOVED, so neither its index nor its position matters."""
    agents = [[0.1, -0.3], [-0.2, -0.3]]
    a = _entity(agents, [[0.3, 0.0], [0.0, -0.1], [0.4, 0.45]], [0, 0, 1])
    b = _entity(agents, [[-0.4, 0.44], [0.0, -0.1], [0.3, 0.0]], [1, 0, 0])
    np.testing.assert_array_equal(
        wp.manager_actor_input_relative(*a), wp.manager_actor_input_relative(*b)
    )


def test_relative_input_lists_teammates_nearest_first_without_self():
    agents, boxes = _entity(
        [[0.0, 0.0], [0.3, 0.0], [0.0, 0.1]],
        [[0.2, 0.2]],
        [0],
        vel=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
    )
    x = np.asarray(wp.manager_actor_input_relative(agents, boxes))[0]
    assert x.shape == (3, 4 + 6 + 4 * 2 + 3)
    # agent 0: agent 2 (0.1 away) before agent 1 (0.3 away), each with its velocity
    np.testing.assert_allclose(
        x[0, 10:18].reshape(2, 4), [[0.0, 0.1, 3.0, 0.0], [0.3, 0.0, 2.0, 0.0]], atol=1e-6
    )
    np.testing.assert_array_equal(x[:, -3:], np.eye(3))
    # One agent: no teammate block at all.
    one = wp.manager_actor_input_relative(*_entity([[0.0, 0.0]], [[0.2, 0.2]], [0]))
    assert one.shape == (1, 1, 4 + 6 + 1)


def test_relative_manager_stores_the_relative_view_of_the_true_state():
    """The stored manager input is the relative view of the state it decided
    in, and delivery (box 0 from step 6) removes that box from it."""
    env = StubEnv(3)
    _, _, (_, rollout, _, _), _ = _collect(env, manager_input="relative")
    m_obs = np.asarray(rollout.manager.obs)  # (windows, E, N, D)
    assert m_obs.shape[-1] == 4 + 6 * 2 + 4 * 2 + 3
    pos, _ = _stored_pos_and_waypoint(rollout.worker)
    for w in range(N_STEPS // HORIZON):
        t = w * HORIZON
        state = StubState(pos=jnp.asarray(pos[t]), t=jnp.full((N_ENVS,), t))
        expected = wp.manager_actor_input_relative(*jax.vmap(env.entity_state)(state))
        np.testing.assert_allclose(m_obs[w], expected, atol=1e-6)
    live_slots = m_obs[..., 4:16].reshape(m_obs.shape[:3] + (2, 6))[..., 5].sum(-1)
    assert np.all(live_slots[:2] == 2.0) and np.all(live_slots[2] == 1.0)


def test_relative_input_needs_entity_state_and_a_known_mode():
    class NoEntities(StubEnv):
        def __getattribute__(self, name):
            if name == "entity_state":
                raise AttributeError(name)
            return super().__getattribute__(name)

    with pytest.raises(ValueError, match="entity_state"):
        make_train(_config(manager_input="relative"), NoEntities(2))
    make_train(_config(), NoEntities(2))  # the global input never asks for it
    make_train(_config(manager_input="local"), NoEntities(2))  # nor does local
    with pytest.raises(ValueError, match="manager_input"):
        make_train(_config(manager_input="egocentric"), StubEnv(2))
    assert Model_Params(
        hidden_dim=8, goal_horizon=4, waypoint_radius=0.1
    ).manager_input == "global"


# ---------------------------------------------------------- local manager input


@pytest.mark.parametrize("n_agents", [1, 3])
def test_local_manager_stores_its_own_observation(n_agents):
    """The stored manager input is the env observation the worker read on the
    first step of the same window: no one-hot, no global state, no position."""
    _, _, (_, rollout, _, _), _ = _collect(StubEnv(n_agents), manager_input="local")
    m_obs = np.asarray(rollout.manager.obs)  # (windows, E, N, OBS_DIM)
    assert m_obs.shape[-1] == OBS_DIM
    w_obs = np.asarray(rollout.worker.obs)[..., :OBS_DIM]
    np.testing.assert_array_equal(m_obs, _windows(w_obs)[:, 0])


def _agent0_action(manager_input, obs, state):
    """Deterministic manager action of agent 0, (E, 2), from a fresh manager."""
    env = StubEnv(3)
    config = _config(manager_input=manager_input)
    policy = make_policy(config, env)
    manager_ts = create_hier_train_state(jax.random.PRNGKey(0), config, env).manager
    gs, pos = policy.observe(obs, state)
    action = policy.decide(
        manager_ts, obs, gs, pos, state, jax.random.PRNGKey(1), True
    )[3]
    return np.asarray(action[:, 0])


def test_local_manager_reads_only_its_own_observation():
    """Under `local` agent 0's decision is a function of agent 0's observation
    alone: moving its teammates (their observations and state), or shifting its
    absolute position while holding its observation fixed, leaves it unchanged,
    and editing its observation changes it. `global` is the positive control:
    it reads the teammates, so the same edit moves its decision.

    In the stub the observation contains the position, so this pins the
    plumbing (what the actor is handed), not partial observability."""
    obs, state = jax.vmap(StubEnv(3).reset)(
        jax.random.split(jax.random.PRNGKey(2), N_ENVS)
    )
    mates_obs = obs.at[:, 1:].add(0.1)
    mates_state = state._replace(pos=state.pos.at[:, 1:].add(0.1))
    shifted_state = state._replace(pos=state.pos + 0.05)
    own_obs = obs.at[:, 0].add(0.1)

    base = _agent0_action("local", obs, state)
    np.testing.assert_array_equal(
        _agent0_action("local", mates_obs, mates_state), base
    )
    np.testing.assert_array_equal(_agent0_action("local", obs, shifted_state), base)
    assert np.any(_agent0_action("local", own_obs, state) != base)

    global_base = _agent0_action("global", obs, state)
    assert np.any(_agent0_action("global", mates_obs, mates_state) != global_base)


# ------------------------------------------------------------ manager discount


@pytest.mark.parametrize("n_agents", [1, 3])
def test_manager_step_gamma_discounts_the_window_and_the_bootstrap(n_agents):
    """A manager discount of its own is used for the window return AND the
    truncation bootstrap; the worker keeps its own gamma."""
    mg = 0.9
    env = StubEnv(n_agents, max_steps=6)  # truncates at window 1, step k=1
    config, _, (runner_state, rollout, _, _), _ = _collect(env, manager_step_gamma=mg)
    assert config.manager.gamma == pytest.approx(mg**HORIZON)
    ts = runner_state.train_state
    w, m = rollout.worker, rollout.manager
    team = np.asarray(w.team_reward)
    np.testing.assert_allclose(
        np.asarray(m.reward)[0],
        sum(mg**k * team[k] for k in range(HORIZON)),
        rtol=1e-5,
    )
    next_obs = np.asarray(w.obs)[6, ..., :OBS_DIM]
    next_pos, next_gs = next_obs[..., :2], next_obs.reshape(N_ENVS, -1)
    v_m = ts.manager.critic_ts.apply_fn(
        ts.manager.critic_ts.params, wp.manager_critic_input(next_gs, next_pos)
    )
    np.testing.assert_allclose(
        np.asarray(m.reward)[1],
        team[4] + mg * team[5] + mg**2 * np.asarray(v_m),
        rtol=1e-5,
    )
    # Worker at step 5: still bootstrapped with the WORKER's gamma.
    pos, waypoint = _stored_pos_and_waypoint(w)
    r_int = (
        np.linalg.norm(waypoint[5] - pos[5], axis=-1)
        - np.linalg.norm(waypoint[5] - next_pos, axis=-1)
    ) / RADIUS
    v_w = ts.worker.critic_ts.apply_fn(
        ts.worker.critic_ts.params,
        wp.worker_critic_input(next_gs, wp.goal_error(waypoint[5], next_pos, RADIUS), 0.5),
    )
    np.testing.assert_allclose(
        np.asarray(w.reward)[5], r_int + config.worker.gamma * np.asarray(v_w), atol=1e-5
    )


def _params(**overrides):
    base = dict(
        n_epochs=6, n_total_steps=1e8, n_minibatches=8, n_steps=1048,
        parameter_sharing=True, random_seeds=[0],
    )
    return Params(**{**base, **overrides})


def test_make_feudal_config_wires_the_manager_knobs():
    """The runner's config resolution: defaults share the worker's gamma and
    entropy coefficient; each manager knob changes only the manager."""
    from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config

    mp = Model_Params(hidden_dim=8, goal_horizon=32, waypoint_radius=0.15)
    default = make_feudal_config(_params(), mp, n_envs=32)
    assert default.worker.n_steps == 1056  # rounded up to whole windows
    assert default.manager_step_gamma is None
    assert default.manager.gamma == default.worker.gamma**32
    assert default.manager.ent_coef == default.worker.ent_coef == 0.01
    make_train(default, StubEnv(2))  # the discount check accepts it

    own = make_feudal_config(
        _params(manager_gamma=0.997, manager_ent_coef=0.001), mp, n_envs=32
    )
    assert own.manager_step_gamma == 0.997
    assert own.manager.gamma == pytest.approx(0.997**32)
    assert own.manager.ent_coef == 0.001
    assert own.worker == default.worker  # the worker is untouched
    make_train(own, StubEnv(2))


def test_manager_ent_coef_changes_only_the_manager_update():
    """Same rollout, two manager entropy coefficients: the worker's update is
    bit-identical, the manager's is not."""
    base = _config(bound="tanh")
    low = dataclasses.replace(
        base, manager=dataclasses.replace(base.manager, ent_coef=0.0)
    )
    env = StubEnv(3)
    init_fn, collect_fn, update_base, _, _ = make_train(base, env)
    update_low = make_train(low, env)[2]
    runner_state, rollout, last, _ = collect_fn(init_fn(jax.random.PRNGKey(0)))
    a, _ = update_base(runner_state, rollout, last)
    b, _ = update_low(runner_state, rollout, last)
    same = jax.tree.map(
        lambda x, y: bool((x == y).all()),
        a.train_state.worker.actor_ts.params, b.train_state.worker.actor_ts.params,
    )
    assert all(jax.tree.leaves(same))
    diff = jax.tree.map(
        lambda x, y: float(jnp.abs(x - y).max()),
        a.train_state.manager.actor_ts.params, b.train_state.manager.actor_ts.params,
    )
    assert max(jax.tree.leaves(diff)) > 0.0


def test_manager_gamma_that_disagrees_with_its_step_discount_is_rejected():
    config = _config(manager_step_gamma=0.9)
    bad = dataclasses.replace(
        config, manager=dataclasses.replace(config.manager, gamma=0.99**HORIZON)
    )
    with pytest.raises(ValueError, match="manager.gamma"):
        make_train(bad, StubEnv(2))
    defaults = {f.name: f.default for f in dataclasses.fields(Params)}
    assert defaults["manager_gamma"] is None
    assert defaults["manager_ent_coef"] is None


@pytest.mark.parametrize("mean, log_std", [(0.0, -0.5), (1.5, -1.0), (-3.0, 0.3)])
def test_squashed_entropy_matches_monte_carlo(mean, log_std):
    """Quadrature vs a Monte Carlo estimate of H[u] + E[log(1 - tanh(u)^2)]."""
    mu = jnp.full((1, 2), mean)
    ls = jnp.full((2,), log_std)
    u = mean + np.exp(log_std) * np.random.default_rng(0).standard_normal(2_000_000)
    jac_mc = np.log1p(-np.tanh(u) ** 2).mean()
    gauss = 0.5 * (1 + np.log(2 * np.pi)) + log_std
    np.testing.assert_allclose(
        float(_squashed_gaussian_entropy(mu, ls)[0]), 2 * (gauss + jac_mc), atol=5e-3
    )


def test_squashed_entropy_pulls_a_saturated_mean_back():
    """Maximizing the squashed entropy moves the mean toward 0; the plain
    Gaussian entropy does not depend on the mean at all."""
    ls = jnp.full((1,), -1.5)

    def grad_at(m):
        return float(jax.grad(lambda x: _squashed_gaussian_entropy(x, ls).sum())(
            jnp.full((1, 1), m)
        )[0, 0])

    assert grad_at(3.0) < -1.0 and grad_at(-3.0) > 1.0
    assert abs(grad_at(0.0)) < 1e-6
    assert float(_squashed_gaussian_entropy(jnp.zeros((1, 1)), ls)[0]) < float(
        _gaussian_entropy(ls, (1,))[0]
    )


def test_squash_changes_the_entropy_and_not_the_log_prob():
    """`squash=False` is the original Gaussian path exactly; `squash=True`
    keeps the log-prob (the tanh Jacobian cancels in the PPO ratio)."""
    actor = MAPPOActor(action_dim=2, hidden_dim=8, discrete=False)
    obs = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    params = actor.init(jax.random.PRNGKey(1), obs)
    action = jax.random.normal(jax.random.PRNGKey(2), (5, 2)) * 3.0
    lp, ent = evaluate_action(actor.apply, params, obs, action, discrete=False)
    lp_s, ent_s = evaluate_action(
        actor.apply, params, obs, action, discrete=False, squash=True
    )
    np.testing.assert_array_equal(lp, lp_s)
    np.testing.assert_array_equal(
        ent, _gaussian_entropy(params["params"]["log_action_std"], (5,))
    )
    assert np.all(np.asarray(ent_s) < np.asarray(ent))
    with pytest.raises(ValueError, match="continuous"):
        evaluate_action(actor.apply, params, obs, action, discrete=True, squash=True)


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


# ---------------------------------------------------------------------------
# Counterfactual-goal credit (`manager_credit: counterfactual`)
# ---------------------------------------------------------------------------


def _tree_max_diff(a, b):
    """Largest absolute difference over the leaves of two pytrees (bool/int as
    floats). Compares leaves, not structure: TrainStates from two `make_train`
    calls hold different optimizer objects as static metadata."""
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    return max(
        (
            float(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)).max())
            for x, y in zip(la, lb)
        ),
        default=0.0,
    )


@pytest.mark.parametrize("n_agents", [1, 3])
def test_counterfactual_credit_is_off_by_default(n_agents):
    config, _, (runner_state, rollout, last, _), (update_fn, _) = _collect(
        StubEnv(n_agents)
    )
    assert config.manager_credit == "team"
    assert runner_state.train_state.manager_adv is None
    assert rollout.manager_goal is None
    new_state, losses = update_fn(runner_state, rollout, last)
    assert new_state.train_state.manager_adv is None
    assert not [key for key in losses if key.startswith("manager_cf_")]
    assert not [key for key in losses if key.startswith("manager_dpp_")]
    defaults = {f.name: f.default for f in dataclasses.fields(Model_Params)}
    assert defaults["manager_credit"] == "team"
    assert defaults["counterfactual_default"] == "sampled"
    assert defaults["dpp_coef"] == 1.0
    assert defaults["dpp_max_recruits"] is None


@pytest.mark.parametrize("n_agents", [1, 3])
def test_zero_advantage_correction_is_team_credit(n_agents):
    """A zero per-agent correction reproduces the team-credit update: the same
    normalization and the same advantage for every agent."""
    config, _, (runner_state, rollout, last, _), _ = _collect(StubEnv(n_agents))
    ts, m = runner_state.train_state.manager, rollout.manager
    key = jax.random.PRNGKey(3)
    team = ppo_update(ts, key, m, last.manager, config.manager, discrete=False)
    zero = ppo_update(
        ts, key, m, last.manager, config.manager, discrete=False,
        advantage_correction=jnp.zeros(m.log_prob.shape),
    )
    assert _tree_max_diff(team[0], zero[0]) < 1e-5
    for key_name in team[1]:
        np.testing.assert_allclose(
            team[1][key_name], zero[1][key_name], atol=1e-5, err_msg=key_name
        )


def test_advantage_correction_needs_a_team_reward():
    config, _, (runner_state, rollout, last, _), _ = _collect(StubEnv(3))
    w = rollout.worker  # per-agent rewards
    with pytest.raises(ValueError, match="per-agent rewards"):
        ppo_update(
            runner_state.train_state.worker, jax.random.PRNGKey(0), w, last.worker,
            config.worker, discrete=False,
            advantage_correction=jnp.zeros(w.log_prob.shape),
        )


@pytest.mark.parametrize("n_agents", [1, 3])
def test_counterfactual_arm_starts_as_exact_team_credit(n_agents):
    """At init the advantage model outputs exactly 0, so beta is 0 and the first
    update equals the team-credit arm's. The rollout is identical too: the model's
    key comes from fold_in and collection never reads it."""
    env = StubEnv(n_agents)
    _, _, (rs_team, ro_team, last_team, _), (update_team, _) = _collect(env)
    _, _, (rs_cf, ro_cf, last_cf, _), (update_cf, _) = _collect(
        env, manager_credit="counterfactual"
    )
    assert _tree_max_diff(
        (ro_team.worker, ro_team.manager), (ro_cf.worker, ro_cf.manager)
    ) == 0.0
    adv = rs_cf.train_state.manager_adv
    in_dim = adv.params["params"]["Dense_0"]["kernel"].shape[0]
    x = jax.random.normal(jax.random.PRNGKey(5), (7, in_dim))
    assert float(jnp.abs(adv.apply_fn(adv.params, x)).max()) == 0.0

    new_team, _ = update_team(rs_team, ro_team, last_team)
    new_cf, losses = update_cf(rs_cf, ro_cf, last_cf)
    assert float(losses["manager_cf_beta"]) == 0.0
    assert float(losses["manager_cf_correction_std"]) == 0.0
    assert _tree_max_diff(new_team.train_state.worker, new_cf.train_state.worker) == 0.0
    assert _tree_max_diff(
        new_team.train_state.manager, new_cf.train_state.manager
    ) < 1e-5
    # ...while the model itself trained on the batch.
    assert _tree_max_diff(adv.params, new_cf.train_state.manager_adv.params) > 0.0


def _nonzero_adv_model(in_dim, seed=0):
    """An advantage model with an ordinary (nonzero) head, for function tests."""
    model = MAPPOCritic(hidden_dim=16)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros(in_dim))
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.sgd(0.0))


def _credit_inputs(n_agents=3, T=2, E=2, C=5, K=4, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    critic_in = jax.random.normal(keys[0], (T, E, C))
    offsets = jax.random.uniform(keys[1], (T, E, n_agents, 2), minval=-1, maxval=1)
    cf_off = jax.random.uniform(keys[2], (T, E, n_agents, K, 2), minval=-1, maxval=1)
    return critic_in, offsets, cf_off


def test_own_slot_joint_replaces_only_the_agents_own_slot():
    _, offsets, cf_off = _credit_inputs()
    n = offsets.shape[-2]
    for i in range(n):
        joint = np.asarray(cf.own_slot_joint(offsets, cf_off[..., i, :, :], i))
        for j in range(n):
            expected = (
                cf_off[..., i, :, :]
                if j == i
                else jnp.broadcast_to(offsets[..., None, j, :], joint[..., j, :].shape)
            )
            np.testing.assert_array_equal(joint[..., j, :], np.asarray(expected))


def test_counterfactual_correction_never_reads_the_agents_own_goal():
    """The unbiasedness precondition: agent i's correction depends only on its
    teammates' goals. Editing agent 1's actual goal leaves c_1 bit-identical and
    moves c_0 and c_2 (the positive control: the model does read goals)."""
    critic_in, offsets, cf_off = _credit_inputs()
    model = _nonzero_adv_model(critic_in.shape[-1] + 2 * offsets.shape[-2])
    c_before, _ = cf.correction(model, critic_in, offsets, cf_off)
    c_after, _ = cf.correction(
        model, critic_in, offsets.at[..., 1, :].add(0.5), cf_off
    )
    np.testing.assert_array_equal(
        np.asarray(c_before[..., 1]), np.asarray(c_after[..., 1])
    )
    assert float(jnp.abs(c_before[..., [0, 2]] - c_after[..., [0, 2]]).max()) > 0.0


def test_counterfactual_offsets_are_draws_of_the_given_policy():
    """`sampled` draws come from the actor passed in, on the stored inputs: with
    the policy's std at its floor every draw lands next to the deterministic
    waypoint. `hold` is the zero offset."""
    _, _, (runner_state, rollout, _, _), _ = _collect(
        StubEnv(3), manager_credit="counterfactual"
    )
    actor = runner_state.train_state.manager.actor_ts
    params = jax.tree.map(lambda p: p, actor.params)
    params["params"]["log_action_std"] = jnp.full_like(
        params["params"]["log_action_std"], -20.0
    )  # clamped to LOG_STD_MIN: std ~ 0.0067
    tight = actor.replace(params=params)
    m, goal = rollout.manager, rollout.manager_goal
    key = jax.random.PRNGKey(0)
    drawn = cf.counterfactual_offsets(
        tight, m.obs, goal.pos, key, 5, RADIUS, "clip", "sampled"
    )
    assert drawn.shape == goal.pos.shape[:-1] + (5, 2)
    assert float(drawn.std(axis=-2).max()) > 0.0  # K distinct draws
    mean_action, _ = sample_action(
        key, tight.apply_fn, tight.params, m.obs.reshape(-1, m.obs.shape[-1]),
        discrete=False, deterministic=True,
    )
    deterministic = wp.waypoint_offset(
        goal.pos, mean_action.reshape(goal.pos.shape), RADIUS, "clip"
    )
    np.testing.assert_allclose(
        drawn, np.broadcast_to(deterministic[..., None, :], drawn.shape), atol=0.05
    )
    hold = cf.counterfactual_offsets(tight, m.obs, goal.pos, key, 5, RADIUS, "clip", "hold")
    assert hold.shape == goal.pos.shape[:-1] + (1, 2)
    assert float(jnp.abs(hold).max()) == 0.0


def test_fit_beta_is_the_clipped_control_variate_coefficient():
    a = jax.random.normal(jax.random.PRNGKey(0), (50, 4))
    ones = jnp.ones((50, 4, 3))
    assert float(cf.fit_beta(a, a[..., None] * ones)) == pytest.approx(1.0, abs=1e-5)
    assert float(cf.fit_beta(a, 2.0 * a[..., None] * ones)) == pytest.approx(0.5, abs=1e-5)
    assert float(cf.fit_beta(a, -a[..., None] * ones)) == 0.0  # never flips sign
    assert float(cf.fit_beta(a, 3.0 * ones)) == 0.0  # constant: no NaN
    assert float(cf.fit_beta(a, jnp.zeros((50, 4, 3)))) == 0.0


def test_counterfactual_credit_isolates_each_agents_contribution_on_a_toy_batch():
    """Synthetic batch: team advantage = f(agent 0's goal) + g(agent 1's goal) +
    noise. After fitting the advantage model, agent 0's corrected advantage
    tracks its own term f much better than the team advantage does, with less
    variance: the teammate's term g has been removed."""
    T, E, N, C = 64, 8, 2, 3
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    critic_in = 0.1 * jax.random.normal(keys[0], (T, E, C))
    offsets = jax.random.uniform(keys[1], (T, E, N, 2), minval=-1, maxval=1)
    f = 2.0 * jnp.sin(2.0 * offsets[..., 0, 0])
    g = 2.0 * jnp.sin(2.0 * offsets[..., 1, 1])
    a_team = f + g + 0.3 * jax.random.normal(keys[2], (T, E))

    model = cf.create_adv_model(
        keys[3], MAPPOConfig(hidden_dim=16, lr=3e-3, grad_clip=10.0), C + 2 * N
    )
    model, _ = cf.fit_adv_model(
        model, cf.adv_model_input(critic_in, offsets), a_team, keys[4],
        n_epochs=300, n_minibatches=4,
    )
    # Counterfactual goals from the synthetic "policy", uniform on [-1, 1]^2.
    cf_off = jax.random.uniform(keys[5], (T, E, N, 32, 2), minval=-1, maxval=1)
    c, _ = cf.correction(model, critic_in, offsets, cf_off)
    beta = float(cf.fit_beta(a_team, c))
    a0 = a_team - beta * c[..., 0]

    def corr(x, y):
        return float(jnp.corrcoef(x.ravel(), y.ravel())[0, 1])

    assert beta > 0.5
    assert corr(a0, f) > corr(a_team, f) + 0.2
    assert float(jnp.var(a0)) < 0.7 * float(jnp.var(a_team))


@pytest.mark.parametrize("default", ["sampled", "hold"])
@pytest.mark.parametrize("n_agents", [1, 3])
def test_counterfactual_collect_update_eval_end_to_end(n_agents, default):
    _, _, (runner_state, rollout, last, _), (update_fn, eval_fn) = _collect(
        StubEnv(n_agents, max_steps=10),
        manager_credit="counterfactual",
        counterfactual_default=default,
    )
    n_windows = N_STEPS // HORIZON
    goal, m = rollout.manager_goal, rollout.manager
    assert goal.pos.shape == goal.offset.shape == (n_windows, N_ENVS, n_agents, 2)
    # The stored offset is the one the stored action produces.
    np.testing.assert_allclose(
        goal.offset, wp.waypoint_offset(goal.pos, m.action, RADIUS, "clip"), atol=1e-6
    )
    # The first update fits the model from zero, so the second one applies a
    # nonzero correction through ppo_update.
    state, _ = update_fn(runner_state, rollout, last)
    state, losses = update_fn(state, rollout, last)
    for key in (
        "manager_cf_beta", "manager_cf_correction_std", "manager_cf_adv_var_ratio",
        "manager_cf_model_ev", "manager_cf_goal_sensitivity", "manager_cf_model_loss",
        "manager_policy_loss", "manager_value_loss",
    ):
        assert np.isfinite(float(losses[key])), key
    assert not [key for key in losses if key.startswith("manager_dpp_")]
    assert 0.0 <= float(losses["manager_cf_beta"]) <= 1.0
    assert float(losses["manager_cf_correction_std"]) > 0.0
    assert _tree_max_diff(
        runner_state.train_state.manager_adv.params, state.train_state.manager_adv.params
    ) > 0.0
    assert np.isfinite(float(eval_fn(state.train_state, jax.random.PRNGKey(1))))


def test_manager_credit_is_validated():
    for field, value in (
        ("manager_credit", "bogus"),
        ("counterfactual_default", "bogus"),
        ("counterfactual_samples", 0),
    ):
        bad = dataclasses.replace(
            _config(manager_credit="counterfactual"), **{field: value}
        )
        with pytest.raises(ValueError, match=field):
            make_train(bad, StubEnv(2))
    with pytest.warns(UserWarning, match="one agent"):
        make_train(_config(manager_credit="counterfactual"), StubEnv(1))


def test_make_feudal_config_wires_the_manager_credit():
    from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config

    base = dict(hidden_dim=8, goal_horizon=32, waypoint_radius=0.15)
    default = make_feudal_config(_params(), Model_Params(**base), n_envs=32)
    assert (
        default.manager_credit,
        default.counterfactual_default,
        default.counterfactual_samples,
    ) == ("team", "sampled", 16)
    own = make_feudal_config(
        _params(),
        Model_Params(
            **base, manager_credit="counterfactual", counterfactual_default="hold",
            counterfactual_samples=4,
        ),
        n_envs=32,
    )
    assert (own.manager_credit, own.counterfactual_default, own.counterfactual_samples) == (
        "counterfactual", "hold", 4,
    )
    assert own.worker == default.worker and own.manager == default.manager


def test_checkpoint_trees_carry_the_advantage_model_only_when_enabled():
    from flax.serialization import from_bytes, to_bytes

    from algorithms.simplified_feudal_mappo_jax.run import (
        Simplified_Feudal_MAPPO_JAX_Runner as Runner,
    )

    eval_rng = jax.random.PRNGKey(7)
    team = make_train(_config(), StubEnv(3))[0](jax.random.PRNGKey(0))
    assert "manager_adv" not in Runner._params_tree(team.train_state)
    assert "manager_adv_ts" not in Runner._checkpoint_tree(team, eval_rng)
    # The default format round-trips unchanged.
    from_bytes(
        Runner._checkpoint_tree(team, eval_rng),
        to_bytes(Runner._checkpoint_tree(team, eval_rng)),
    )

    rs = make_train(_config(manager_credit="counterfactual"), StubEnv(3))[0](
        jax.random.PRNGKey(0)
    )
    assert "manager_adv" in Runner._params_tree(rs.train_state)
    moved_adv = rs.train_state.manager_adv.replace(
        params=jax.tree.map(lambda p: p + 1.0, rs.train_state.manager_adv.params)
    )
    moved = rs._replace(train_state=rs.train_state._replace(manager_adv=moved_adv))
    restored = from_bytes(
        Runner._checkpoint_tree(rs, eval_rng),
        to_bytes(Runner._checkpoint_tree(moved, eval_rng)),
    )
    assert _tree_max_diff(restored["manager_adv_ts"].params, moved_adv.params) == 0.0


# ---------------------------------------------------------------------------
# D++ credit (`manager_credit: dpp`)
# ---------------------------------------------------------------------------


class _IndicatorCritic(NamedTuple):
    """A hand-built joint-goal critic for exact checks of the D++ arithmetic.

    Its value is `sum_j w_j * [agent j's goal differs from ORIGINAL]`, so the gain
    of recruiting a set of teammates is the sum of their weights, known in
    closed form. `params` are the weights `w`; the input layout is
    `adv_model_input`'s (critic input of width `C`, then every agent's offset).
    """

    params: jnp.ndarray
    C: int

    def apply_fn(self, w, x):
        off = x[..., self.C :].reshape(x.shape[:-1] + (-1, 2))
        changed = jnp.linalg.norm(off - ORIGINAL, axis=-1) > 1e-6
        return (changed * w).sum(axis=-1)


ORIGINAL = 0.9  # every realized offset in the hand-built batch is (0.9, 0.9)
# Agent 1 is nearest agent 0 and agent 2 is farther; agent 0 is nearest agent 1.
DPP_POS = jnp.array([[[[0.0, 0.0], [0.05, 0.0], [0.3, 0.0]]]])  # (T=1, E=1, N=3, 2)


def _dpp_batch(weights, max_recruits=None):
    critic = _IndicatorCritic(params=jnp.asarray(weights, jnp.float32), C=2)
    critic_in = jnp.zeros((1, 1, 2))
    offsets = jnp.full((1, 1, 3, 2), ORIGINAL)
    dpp, best_n = cf.dpp_credit(
        critic, critic_in, offsets, DPP_POS, RADIUS, max_recruits
    )
    return np.asarray(dpp[0, 0]), np.asarray(best_n[0, 0])


def test_support_offsets_point_at_the_focal_agent_within_radius_and_arena():
    pos = jax.random.uniform(jax.random.PRNGKey(0), (5, 4, 2), minval=-0.5, maxval=0.5)
    for i in range(4):
        off = np.asarray(cf.support_offsets(pos, i, RADIUS))
        assert np.abs(off).max() <= 1.0 + 1e-6  # within R per axis
        waypoint = np.asarray(pos) + RADIUS * off
        assert np.abs(waypoint).max() <= wp.ARENA_HALF_EXTENT + 1e-6
        to_focal = np.asarray(pos[:, i : i + 1] - pos)
        others = [j for j in range(4) if j != i]
        # Each axis moves toward the focal agent, by the full gap when it is
        # within R and by exactly R otherwise.
        np.testing.assert_allclose(
            off[:, others],
            np.clip(to_focal[:, others] / RADIUS, -1.0, 1.0),
            atol=1e-5,
        )
        np.testing.assert_allclose(off[:, i], 0.0, atol=1e-6)


def test_dpp_joint_keeps_the_focal_goal_and_changes_only_the_nearest_teammates():
    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    pos = jax.random.uniform(keys[0], (6, 4, 2), minval=-0.4, maxval=0.4)
    offsets = jax.random.uniform(keys[1], (6, 4, 2), minval=-1, maxval=1)
    for i in range(4):
        support = np.asarray(cf.support_offsets(pos, i, RADIUS))
        dist = np.linalg.norm(np.asarray(pos - pos[:, i : i + 1]), axis=-1)
        dist[:, i] = np.inf
        order = np.argsort(dist, axis=-1)
        for n in (1, 2, 3):
            joint = np.asarray(cf.dpp_joint(offsets, pos, i, n, RADIUS))
            for b in range(pos.shape[0]):
                recruits = set(order[b, :n].tolist())
                assert i not in recruits
                for j in range(4):
                    expected = support[b, j] if j in recruits else offsets[b, j]
                    np.testing.assert_array_equal(joint[b, j], np.asarray(expected))


def test_dpp_divides_by_the_recruit_count_and_takes_the_best_count():
    # Focal 0: n=1 recruits agent 1 (gain w1), n=2 adds agent 2 ((w1 + w2) / 2).
    dpp, best_n = _dpp_batch([0.0, 2.0, 6.0])
    assert dpp[0] == pytest.approx(4.0) and best_n[0] == 2
    # Focal 1: n=1 recruits agent 0 (w0 = 0), n=2 adds agent 2 ((0 + 6) / 2).
    assert dpp[1] == pytest.approx(3.0) and best_n[1] == 2
    dpp, best_n = _dpp_batch([0.0, 2.0, 1.0])
    assert dpp[0] == pytest.approx(2.0) and best_n[0] == 1  # (2 + 1) / 2 < 2
    dpp, best_n = _dpp_batch([0.0, 2.0, 6.0], max_recruits=1)
    assert dpp[0] == pytest.approx(2.0) and best_n[0] == 1
    # A recruitment that hurts: the raw term is negative, the correction is 0.
    dpp, _ = _dpp_batch([0.0, -2.0, -6.0])
    assert dpp[0] == pytest.approx(-2.0)
    assert float(jnp.abs(cf.dpp_correction(jnp.asarray(dpp), 1.0)).max()) == 0.0


def test_dpp_correction_adds_the_clipped_term_to_the_advantage():
    """`ppo_update` subtracts the correction, so it is minus the clipped term."""
    dpp = jnp.array([-1.0, 0.0, 0.5, 2.0])
    np.testing.assert_array_equal(
        np.asarray(cf.dpp_correction(dpp, 0.5)), [0.0, 0.0, -0.25, -1.0]
    )


def test_dpp_is_zero_without_teammates_and_with_a_zero_model():
    critic_in, offsets, _ = _credit_inputs(n_agents=1)
    model = cf.create_adv_model(
        jax.random.PRNGKey(0), MAPPOConfig(hidden_dim=16), critic_in.shape[-1] + 2
    )
    pos = jax.random.uniform(jax.random.PRNGKey(1), offsets.shape, minval=-0.4, maxval=0.4)
    dpp, best_n = cf.dpp_credit(model, critic_in, offsets, pos, RADIUS)
    assert dpp.shape == offsets.shape[:-1]
    assert float(jnp.abs(dpp).max()) == 0.0 and float(jnp.abs(best_n).max()) == 0.0

    critic_in, offsets, _ = _credit_inputs(n_agents=3)
    model = cf.create_adv_model(
        jax.random.PRNGKey(0), MAPPOConfig(hidden_dim=16), critic_in.shape[-1] + 6
    )
    pos = jax.random.uniform(jax.random.PRNGKey(1), offsets.shape, minval=-0.4, maxval=0.4)
    dpp, _ = cf.dpp_credit(model, critic_in, offsets, pos, RADIUS)
    assert float(jnp.abs(dpp).max()) == 0.0  # the zero-initialized head


def test_dpp_reads_the_agents_own_goal_so_it_is_shaping():
    """The property that separates `dpp` from `counterfactual` (whose correction
    is bit-identical under this edit): agent 1's own goal is kept in o++ and is
    part of o, so editing it moves D++_1."""
    critic_in, offsets, _ = _credit_inputs()
    model = _nonzero_adv_model(critic_in.shape[-1] + 2 * offsets.shape[-2])
    pos = jax.random.uniform(jax.random.PRNGKey(2), offsets.shape, minval=-0.4, maxval=0.4)
    before, _ = cf.dpp_credit(model, critic_in, offsets, pos, RADIUS)
    after, _ = cf.dpp_credit(
        model, critic_in, offsets.at[..., 1, :].add(0.5), pos, RADIUS
    )
    assert float(jnp.abs(before[..., 1] - after[..., 1]).max()) > 0.0


def test_a_positive_dpp_term_raises_the_agents_advantage():
    """Sign check through the real `ppo_update`: crediting agent 0's goals with a
    D++ term on alternate windows makes those actions more likely than
    debiting the same amount does."""
    config, _, (runner_state, rollout, last, _), _ = _collect(StubEnv(3))
    ts, m = runner_state.train_state.manager, rollout.manager
    bonus = jnp.zeros(m.log_prob.shape).at[::2, :, 0].set(5.0)
    key = jax.random.PRNGKey(3)
    credited, _ = ppo_update(
        ts, key, m, last.manager, config.manager, discrete=False,
        advantage_correction=cf.dpp_correction(bonus, 1.0),
    )
    debited, _ = ppo_update(
        ts, key, m, last.manager, config.manager, discrete=False,
        advantage_correction=-cf.dpp_correction(bonus, 1.0),
    )

    def log_prob(state):
        obs = m.obs[::2, :, 0].reshape(-1, m.obs.shape[-1])
        action = m.action[::2, :, 0].reshape(-1, m.action.shape[-1])
        lp, _ = evaluate_action(
            state.actor_ts.apply_fn, state.actor_ts.params, obs, action, False
        )
        return float(lp.mean())

    assert log_prob(credited) > log_prob(debited)


@pytest.mark.parametrize("n_agents", [1, 3])
def test_dpp_arm_starts_as_exact_team_credit(n_agents):
    """At init `Â_φ` outputs exactly 0, so the D++ term is 0 and the first update
    equals the team-credit arm's; the rollout is identical too."""
    env = StubEnv(n_agents)
    _, _, (rs_team, ro_team, last_team, _), (update_team, _) = _collect(env)
    _, _, (rs_dpp, ro_dpp, last_dpp, _), (update_dpp, _) = _collect(
        env, manager_credit="dpp"
    )
    assert _tree_max_diff(
        (ro_team.worker, ro_team.manager), (ro_dpp.worker, ro_dpp.manager)
    ) == 0.0
    new_team, _ = update_team(rs_team, ro_team, last_team)
    new_dpp, losses = update_dpp(rs_dpp, ro_dpp, last_dpp)
    assert float(losses["manager_dpp_positive_frac"]) == 0.0
    assert float(losses["manager_dpp_mean"]) == 0.0
    assert _tree_max_diff(new_team.train_state.worker, new_dpp.train_state.worker) == 0.0
    assert _tree_max_diff(
        new_team.train_state.manager, new_dpp.train_state.manager
    ) < 1e-5
    assert _tree_max_diff(
        rs_dpp.train_state.manager_adv.params, new_dpp.train_state.manager_adv.params
    ) > 0.0


@pytest.mark.parametrize("n_agents", [1, 3])
def test_dpp_collect_update_eval_end_to_end(n_agents):
    _, _, (runner_state, rollout, last, _), (update_fn, eval_fn) = _collect(
        StubEnv(n_agents, max_steps=10), manager_credit="dpp"
    )
    n_windows = N_STEPS // HORIZON
    assert rollout.manager_goal.pos.shape == (n_windows, N_ENVS, n_agents, 2)
    # The first update fits the model from zero, so the second one scores with a
    # nonzero model and applies the term through ppo_update.
    state, _ = update_fn(runner_state, rollout, last)
    state, losses = update_fn(state, rollout, last)
    dpp_keys = (
        "manager_dpp_mean", "manager_dpp_std", "manager_dpp_positive_frac",
        "manager_dpp_mean_best_n", "manager_dpp_adv_shift",
    )
    for key in dpp_keys + (
        "manager_cf_model_ev", "manager_cf_model_loss", "manager_policy_loss",
    ):
        assert np.isfinite(float(losses[key])), key
    assert "manager_cf_beta" not in losses
    assert 0.0 <= float(losses["manager_dpp_positive_frac"]) <= 1.0
    if n_agents == 1:
        assert all(float(losses[key]) == 0.0 for key in dpp_keys)
    else:
        assert float(losses["manager_dpp_positive_frac"]) > 0.0
        assert 1.0 <= float(losses["manager_dpp_mean_best_n"]) <= n_agents - 1
    assert _tree_max_diff(
        runner_state.train_state.manager_adv.params, state.train_state.manager_adv.params
    ) > 0.0
    assert np.isfinite(float(eval_fn(state.train_state, jax.random.PRNGKey(1))))


def test_dpp_credit_is_validated():
    for field, value in (("dpp_coef", -0.1), ("dpp_max_recruits", 0)):
        bad = dataclasses.replace(_config(manager_credit="dpp"), **{field: value})
        with pytest.raises(ValueError, match=field):
            make_train(bad, StubEnv(2))
    with pytest.warns(UserWarning, match="no teammates to recruit"):
        make_train(_config(manager_credit="dpp"), StubEnv(1))


def test_make_feudal_config_wires_the_dpp_knobs():
    from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config

    base = dict(hidden_dim=8, goal_horizon=32, waypoint_radius=0.15)
    default = make_feudal_config(_params(), Model_Params(**base), n_envs=32)
    assert (default.dpp_coef, default.dpp_max_recruits) == (1.0, None)
    own = make_feudal_config(
        _params(),
        Model_Params(**base, manager_credit="dpp", dpp_coef=0.5, dpp_max_recruits=2),
        n_envs=32,
    )
    assert (own.manager_credit, own.dpp_coef, own.dpp_max_recruits) == ("dpp", 0.5, 2)
    assert own.worker == default.worker and own.manager == default.manager


def test_dpp_checkpoint_carries_the_advantage_model():
    from algorithms.simplified_feudal_mappo_jax.run import (
        Simplified_Feudal_MAPPO_JAX_Runner as Runner,
    )

    rs = make_train(_config(manager_credit="dpp"), StubEnv(3))[0](jax.random.PRNGKey(0))
    assert "manager_adv" in Runner._params_tree(rs.train_state)
    assert "manager_adv_ts" in Runner._checkpoint_tree(rs, jax.random.PRNGKey(7))
