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
from algorithms.mappo_jax.network import (
    MAPPOActor,
    MAPPOCritic,
    _gaussian_entropy,
    _squashed_gaussian_entropy,
    evaluate_action,
)
from algorithms.mappo_jax.types import MAPPOConfig
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.trainer import (
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


def _config(n_steps=N_STEPS, bound="clip", manager_input="global", manager_step_gamma=None):
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


@pytest.mark.parametrize("manager_input", ["global", "relative"])
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


@pytest.mark.parametrize("manager_input", ["global", "relative"])
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
    with pytest.raises(ValueError, match="manager_input"):
        make_train(_config(manager_input="egocentric"), StubEnv(2))
    assert Model_Params(
        hidden_dim=8, goal_horizon=4, waypoint_radius=0.1
    ).manager_input == "global"


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
