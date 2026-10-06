"""Seam tests for the simplified feudal stack's training forks
(`interventions: true`, `algorithms/simplified_feudal_mappo_jax/interventions.py`).

Most tests run on the CPU stub env of `test_simplified_feudal.py`, which is
deterministic, so "the main rollout is unchanged by the forks" can be checked
bit for bit. The MJX tests pin the `teleport_agents` geometry and the physics
state it must preserve.

    uv run pytest algorithms/tests/test_simplified_feudal_interventions.py -q
"""

import dataclasses
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# pytest puts this directory on sys.path (no __init__.py), so the stub env and
# config helpers are shared with the main seam tests rather than copied.
from test_simplified_feudal import (
    HORIZON,
    N_ENVS,
    N_STEPS,
    OBS_DIM,
    RADIUS,
    StubEnv,
    _config,
    _params,
    _stored_pos_and_waypoint,
    _tree_max_diff,
)

from algorithms.mappo_jax.mappo import (
    compute_gae,
    masked_normalize,
    ppo_update,
)
from algorithms.mappo_jax.network import evaluate_action
from algorithms.mappo_jax.types import Transition
from algorithms.simplified_feudal_mappo_jax import interventions as iv
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.trainer import make_train
from algorithms.simplified_feudal_mappo_jax.types import Model_Params

N_WINDOWS = N_STEPS // HORIZON


@pytest.fixture(autouse=True)
def _run_on_cpu():
    """Pin to CPU, as in the other seam tests: tiny tests, and a busy GPU fails
    like an assertion."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _forks(env, seed=0, **kwargs):
    """`(config, runner_state, (runner_state, rollout, last_values, stats),
    (update_fn, eval_fn))` with forks on."""
    config = _config(interventions=True, **kwargs)
    init_fn, collect_fn, update_fn, eval_fn, _ = make_train(config, env)
    runner_state = init_fn(jax.random.PRNGKey(seed))
    return config, runner_state, collect_fn(runner_state), (update_fn, eval_fn)


def _parent(env, seed=0, **kwargs):
    config = _config(**kwargs)
    init_fn, collect_fn, _, _, _ = make_train(config, env)
    return collect_fn(init_fn(jax.random.PRNGKey(seed)))


def _lanes(n_agents):
    return N_ENVS * n_agents


# --------------------------------------------------------------------- default


def test_interventions_are_off_by_default():
    mp = Model_Params(hidden_dim=8, goal_horizon=32, waypoint_radius=0.15)
    assert (mp.interventions, mp.intervention_radius, mp.intervention_interval) == (
        False, 1.5, 1,
    )
    _, rollout, _, _ = _parent(StubEnv(3))
    assert rollout.intervention is None


def test_make_feudal_config_wires_the_intervention_keys():
    from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config

    base = dict(hidden_dim=8, goal_horizon=32, waypoint_radius=0.15)
    default = make_feudal_config(_params(), Model_Params(**base), n_envs=32)
    assert (
        default.interventions,
        default.intervention_radius,
        default.intervention_interval,
    ) == (False, 1.5, 1)
    own = make_feudal_config(
        _params(),
        Model_Params(
            **base, interventions=True, intervention_radius=2.0, intervention_interval=4
        ),
        n_envs=32,
    )
    assert (own.interventions, own.intervention_radius, own.intervention_interval) == (
        True, 2.0, 4,
    )
    assert own.worker == default.worker and own.manager == default.manager


@pytest.mark.parametrize(
    "kwargs, env, match",
    [
        (dict(interventions=False, intervention_interval=2), StubEnv(3), "interventions"),
        (dict(interventions=True, manager_credit="dpp"), StubEnv(3), "manager_credit"),
        (dict(interventions=True), StubEnv(1), "two agents"),
        (dict(interventions=True, intervention_radius=0.0), StubEnv(3), "radius"),
        (dict(interventions=True, intervention_interval=0), StubEnv(3), "integer"),
        # 3 windows at interval 3 leave one intervention window.
        (dict(interventions=True, intervention_interval=3), StubEnv(3), "at least 2"),
    ],
)
def test_bad_intervention_configs_are_rejected(kwargs, env, match):
    with pytest.raises(ValueError, match=match):
        make_train(_config(**kwargs), env)


def test_an_env_without_the_teleport_hook_is_rejected():
    class NoTeleport:
        """The stub env with every attribute except `teleport_agents`."""

        def __init__(self, env):
            self._env = env

        def __getattr__(self, name):
            if name == "teleport_agents":
                raise AttributeError(name)
            return getattr(self._env, name)

    env = NoTeleport(StubEnv(3))
    assert not hasattr(env, "teleport_agents")
    with pytest.raises(ValueError, match="teleport_agents"):
        make_train(_config(interventions=True), env)


# ------------------------------------------------------------- main isolation


@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.parametrize("n_agents", [2, 3])
def test_forks_leave_the_main_rollout_unchanged(n_agents, interval):
    """The forks draw their keys from `fold_in`, never from a main split, so the
    main rollout equals the parent's at the same seed (only the manager critic's
    values differ, through its wider input). A failing placement changes nothing
    either."""
    _, parent, _, _ = _parent(StubEnv(n_agents))
    for env in (StubEnv(n_agents), StubEnv(n_agents, fail_teleport=True)):
        _, _, (_, rollout, _, _), _ = _forks(env, intervention_interval=interval)
        assert _tree_max_diff(rollout.worker, parent.worker) == 0.0
        for field in ("obs", "action", "log_prob", "reward", "done", "team_reward"):
            np.testing.assert_array_equal(
                getattr(rollout.manager, field), getattr(parent.manager, field), field
            )


# ------------------------------------------------------------------- interval


@pytest.mark.parametrize("interval", [1, 2])
def test_forks_run_exactly_on_the_interval_windows(interval):
    n_agents = 3
    _, _, (_, rollout, _, stats), _ = _forks(
        StubEnv(n_agents), intervention_interval=interval
    )
    fork = rollout.intervention
    applied = np.arange(N_WINDOWS) % interval == 0
    assert applied.sum() == math.ceil(N_WINDOWS / interval)
    valid, n_rec = np.asarray(fork.valid), np.asarray(fork.n_recruits)
    assert valid[applied].all() and not valid[~applied].any()
    assert (n_rec[applied] >= 1).all() and (n_rec[~applied] == 0).all()
    # Skipped windows hold fully masked placeholder rows.
    w_active = np.asarray(fork.worker.active_mask).reshape(
        N_WINDOWS, HORIZON, _lanes(n_agents), n_agents
    )
    assert w_active[applied].all() and not w_active[~applied].any()
    for traj, shape in (
        (fork.worker, (N_WINDOWS, HORIZON)),
        (fork.manager, (N_WINDOWS,)),
    ):
        r = np.asarray(traj.reward).reshape(shape + (_lanes(n_agents), -1))
        v = np.asarray(traj.value).reshape(shape + (_lanes(n_agents), -1))
        d = np.asarray(traj.done).reshape(shape + (_lanes(n_agents),))
        assert (r[~applied] == 0).all() and (v[~applied] == 0).all()
        assert d[~applied].all()
    assert int(stats["episode_count"]) == 0  # forks are not task episodes


# ------------------------------------------------------- goals and recruitment


def test_recruit_sampling_draws_n_distinct_teammates():
    for n_agents in (2, 3, 6):
        focal = iv.focal_agents(64, n_agents)
        n, mask = iv.sample_recruits(jax.random.PRNGKey(0), focal, n_agents)
        n, mask = np.asarray(n), np.asarray(mask)
        assert n.min() >= 1 and n.max() <= n_agents - 1
        np.testing.assert_array_equal(mask.sum(-1), n)
        assert not mask[np.arange(len(focal)), np.asarray(focal)].any()
        if n_agents == 2:
            assert (n == 1).all()
        else:
            assert len(np.unique(n)) == n_agents - 1  # every count is drawn
        again = iv.sample_recruits(jax.random.PRNGKey(0), focal, n_agents)
        np.testing.assert_array_equal(np.asarray(again[1]), mask)  # reproducible


def test_recruits_get_the_focal_offset_and_everyone_else_keeps_their_goal():
    """At the fork's first step each recruit's error to its waypoint is the focal
    agent's realized offset; the focal agent and every non-recruit keep the main
    decision's waypoint (and position)."""
    n_agents = 3
    _, _, (_, rollout, _, _), _ = _forks(StubEnv(n_agents))
    fork = rollout.intervention
    lanes = _lanes(n_agents)
    f_obs = np.asarray(fork.worker.obs).reshape(N_WINDOWS, HORIZON, lanes, n_agents, -1)
    m_obs = np.asarray(rollout.worker.obs).reshape(N_WINDOWS, HORIZON, N_ENVS, n_agents, -1)
    err_f = f_obs[:, 0, ..., OBS_DIM : OBS_DIM + 2]
    pos_f = f_obs[:, 0, ..., :2]
    err_m = np.repeat(m_obs[:, 0, ..., OBS_DIM : OBS_DIM + 2], n_agents, axis=1)
    pos_m = np.repeat(m_obs[:, 0, ..., :2], n_agents, axis=1)
    focal = np.asarray(fork.focal)
    for w in range(N_WINDOWS):
        moved = np.abs(pos_f[w] - pos_m[w]).max(-1) > 0  # (L, N)
        np.testing.assert_array_equal(moved.sum(-1), np.asarray(fork.n_recruits)[w])
        assert not moved[np.arange(lanes), focal[w]].any()
        focal_err = err_m[w][np.arange(lanes), focal[w]]  # (L, 2)
        for lane in range(lanes):
            for j in range(n_agents):
                expected = focal_err[lane] if moved[lane, j] else err_m[w][lane, j]
                np.testing.assert_allclose(err_f[w][lane, j], expected, atol=1e-5)


def test_the_critic_context_carries_each_forks_own_focal_and_n():
    n_agents = 3
    _, runner_state, (_, rollout, _, _), _ = _forks(StubEnv(n_agents))
    base = wp.input_dims(OBS_DIM, OBS_DIM * n_agents, n_agents, 2)["manager_critic"]
    kernel = runner_state.train_state.manager.critic_ts.params["params"]["Dense_0"]["kernel"]
    assert kernel.shape[0] == base + n_agents + 2
    main_ctx = np.asarray(rollout.manager.global_state)[..., base:]
    assert (main_ctx == 0).all()
    fork = rollout.intervention
    ctx = np.asarray(fork.manager.global_state)[..., base:]
    np.testing.assert_array_equal(ctx[..., 0], 1.0)
    np.testing.assert_array_equal(
        ctx[..., 1 : 1 + n_agents], np.eye(n_agents)[np.asarray(fork.focal)]
    )
    np.testing.assert_allclose(
        ctx[..., -1], np.asarray(fork.n_recruits) / (n_agents - 1), atol=1e-6
    )
    # The rest of the fork input is the main decision's (pre-teleport) state.
    np.testing.assert_array_equal(
        np.asarray(fork.manager.global_state)[..., :base],
        np.repeat(np.asarray(rollout.manager.global_state)[..., :base], n_agents, axis=1),
    )


# ------------------------------------------------------------- PPO provenance


@pytest.mark.parametrize("bound", ["clip", "tanh"])
def test_fork_ratios_are_one_and_only_the_focal_manager_action_is_eligible(bound):
    n_agents = 3
    _, _, (runner_state, rollout, _, _), _ = _forks(StubEnv(n_agents), bound=bound)
    ts, fork = runner_state.train_state, rollout.intervention
    for level, traj in (("worker", fork.worker), ("manager", fork.manager)):
        actor = getattr(ts, level).actor_ts
        d = traj.obs.shape[-1]
        log_prob, _ = evaluate_action(
            actor.apply_fn, actor.params, traj.obs.reshape(-1, d),
            traj.action.reshape(-1, traj.action.shape[-1]), discrete=False,
        )
        np.testing.assert_allclose(
            log_prob, np.asarray(traj.log_prob).reshape(-1), atol=1e-5, err_msg=level
        )
    # The fork's manager record is the main decision itself: same input, action
    # and log-prob, repeated per lane.
    for field in ("obs", "action", "log_prob"):
        np.testing.assert_array_equal(
            getattr(fork.manager, field),
            np.repeat(np.asarray(getattr(rollout.manager, field)), n_agents, axis=1),
        )
    np.testing.assert_array_equal(
        fork.manager.active_mask, np.eye(n_agents)[np.asarray(fork.focal)]
    )
    assert np.asarray(fork.worker.active_mask).all()  # every live agent's action


def test_imposed_manager_actions_carry_no_gradient():
    """Editing the recruits' and other non-focal agents' stored manager actions
    in the fork rows leaves the update bit-identical: only agent i's own sample
    is trained on in its fork."""
    n_agents = 3
    _, _, (runner_state, rollout, last, _), (update_fn, _) = _forks(StubEnv(n_agents))
    fork = rollout.intervention
    focal = np.eye(n_agents)[np.asarray(fork.focal)][..., None]  # (W, L, N, 1)
    edited_action = jnp.where(focal > 0, fork.manager.action, fork.manager.action + 3.0)
    edited = rollout._replace(
        intervention=fork._replace(manager=fork.manager._replace(action=edited_action))
    )
    a, _ = update_fn(runner_state, rollout, last)
    b, _ = update_fn(runner_state, edited, last)
    assert _tree_max_diff(a.train_state, b.train_state) == 0.0


# --------------------------------------------------------- episode separation


def test_fork_columns_are_independent_episodes_under_gae():
    """GAE on the main batch with the fork columns appended equals GAE on each
    source alone, column for column, and a large reward in one fork moves no
    other fork's or main's advantages."""
    n_agents = 3
    config, _, (_, rollout, last, _), _ = _forks(StubEnv(n_agents))
    fork, cfg = rollout.intervention, config.manager
    lanes = _lanes(n_agents)

    def gae(traj, last_value):
        return compute_gae(
            traj.reward, traj.value, traj.done.astype(jnp.float32), last_value,
            cfg.gamma, cfg.gae_lambda,
        )[0]

    joint = gae(
        iv.append_columns(rollout.manager, fork.manager),
        jnp.concatenate([last.manager, jnp.zeros(lanes)]),
    )
    np.testing.assert_array_equal(joint[:, :N_ENVS], gae(rollout.manager, last.manager))
    np.testing.assert_array_equal(joint[:, N_ENVS:], gae(fork.manager, jnp.zeros(lanes)))

    bumped = fork.manager._replace(reward=fork.manager.reward.at[1, 2].add(100.0))
    joint_b = gae(
        iv.append_columns(rollout.manager, bumped),
        jnp.concatenate([last.manager, jnp.zeros(lanes)]),
    )
    changed = np.abs(np.asarray(joint_b - joint)) > 0
    assert changed[1, N_ENVS + 2] and changed.sum() == 1


def test_every_fork_ends_in_a_terminal():
    n_agents = 2
    _, _, (_, rollout, _, _), _ = _forks(StubEnv(n_agents))
    fork = rollout.intervention
    done = np.asarray(fork.worker.done).reshape(N_WINDOWS, HORIZON, -1)
    assert done[:, -1].all() and not done[:, :-1].any()
    assert np.asarray(fork.manager.done).all()


def test_an_early_termination_ends_the_fork_and_pays_only_live_steps():
    """Termination at global step 5 (window 1, k=1): fork steps after it are
    frozen (masked, terminal, unpaid) and the fork's manager return is the
    discounted team reward of its two live steps."""
    n_agents = 2
    config, _, (_, rollout, _, _), _ = _forks(StubEnv(n_agents, terminate_at=6))
    fork = rollout.intervention
    active = np.asarray(fork.worker.active_mask)[..., 0]  # (T, L)
    assert active[4:6].all() and not active[6:8].any()
    assert np.asarray(fork.worker.done)[5:8].all()
    assert (np.asarray(fork.worker.reward)[6:8] == 0).all()
    team = np.asarray(fork.worker.team_reward)
    np.testing.assert_allclose(
        np.asarray(fork.manager.reward)[1],
        team[4] + config.worker.gamma * team[5],
        rtol=1e-5,
    )


@pytest.mark.parametrize("max_steps", [6, 8])
def test_no_time_limit_bootstrap_survives_in_fork_rewards(max_steps):
    """A time limit inside the window (step 5) or at its last step (step 7): the
    main stream bootstraps, the fork does not. Its worker reward is the bare
    intrinsic reward and its manager return the bare discounted team reward.
    (At a window's last step the worker never bootstraps, and the successor
    position is not stored, so only the manager is checked there.)"""
    n_agents = 2
    config, _, (_, rollout, _, _), _ = _forks(StubEnv(n_agents, max_steps=max_steps))
    fork, gamma = rollout.intervention, config.worker.gamma
    t = max_steps - 1
    if t % HORIZON != HORIZON - 1:
        # The frozen row after the time limit holds the true successor.
        pos, waypoint = _stored_pos_and_waypoint(fork.worker)
        r_int = (
            np.linalg.norm(waypoint[t] - pos[t], axis=-1)
            - np.linalg.norm(waypoint[t] - pos[t + 1], axis=-1)
        ) / RADIUS
        np.testing.assert_allclose(np.asarray(fork.worker.reward)[t], r_int, atol=1e-5)
    team = np.asarray(fork.worker.team_reward)
    start = (t // HORIZON) * HORIZON
    expected = sum(gamma**k * team[start + k] for k in range(t - start + 1))
    np.testing.assert_allclose(
        np.asarray(fork.manager.reward)[t // HORIZON], expected, rtol=1e-5
    )
    # The main manager return at that window does carry the bootstrap.
    main_team = np.asarray(rollout.worker.team_reward)
    main_plain = sum(gamma**k * main_team[start + k] for k in range(t - start + 1))
    assert np.abs(np.asarray(rollout.manager.reward)[t // HORIZON] - main_plain).max() > 1e-6


def test_one_step_windows_make_one_step_forks():
    n_agents = 2
    base = _config(interventions=True)
    config = dataclasses.replace(
        base,
        goal_horizon=1,
        manager=dataclasses.replace(base.manager, gamma=base.worker.gamma),
    )
    init_fn, collect_fn, update_fn, _, _ = make_train(config, StubEnv(n_agents))
    rs = init_fn(jax.random.PRNGKey(0))
    rs, rollout, last, _ = collect_fn(rs)
    fork = rollout.intervention
    assert fork.worker.done.shape == (N_STEPS, _lanes(n_agents))
    assert np.asarray(fork.worker.done).all() and np.asarray(fork.manager.done).all()
    _, losses = update_fn(rs, rollout, last)
    assert all(np.isfinite(float(v)) for v in losses.values())


# -------------------------------------------------------------------- masking


@pytest.mark.parametrize("fail", [False, True])
def test_masked_rows_change_nothing_in_the_update(fail):
    """With `masked_statistics`, placeholder rows (interval 2) and failed
    placements enter no normalization, explained variance or loss: overwriting
    their rewards and values with junk leaves the whole update bit-identical."""
    n_agents = 3
    _, _, (runner_state, rollout, last, _), (update_fn, _) = _forks(
        StubEnv(n_agents, fail_teleport=fail), intervention_interval=2
    )
    fork = rollout.intervention

    def junk(traj, active_shape):
        masked = np.asarray(traj.active_mask).reshape(active_shape).max(-1) == 0
        m = jnp.asarray(masked).reshape(traj.reward.shape[:2] + (1,) * (traj.reward.ndim - 2))
        return traj._replace(
            reward=jnp.where(m, 37.0, traj.reward), value=jnp.where(m, -11.0, traj.value)
        )

    edited = rollout._replace(
        intervention=fork._replace(
            worker=junk(fork.worker, fork.worker.active_mask.shape),
            manager=junk(fork.manager, fork.manager.active_mask.shape),
        )
    )
    assert _tree_max_diff(edited.intervention, fork) > 0.0  # the edit landed
    a, la = update_fn(runner_state, rollout, last)
    b, lb = update_fn(runner_state, edited, last)
    assert _tree_max_diff(a.train_state, b.train_state) == 0.0
    for key in ("worker_explained_variance", "manager_explained_variance"):
        assert float(la[key]) == float(lb[key])


def test_masked_normalize_skips_streams_with_fewer_than_two_rows():
    adv = jnp.array([[1.0, 5.0, 2.0], [3.0, 7.0, 4.0], [9.0, 1.0, 6.0]])
    w = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = np.asarray(masked_normalize(adv, w))
    col = np.array([1.0, 3.0, 9.0])
    np.testing.assert_allclose(out[:, 0], (col - col.mean()) / col.std(ddof=1), atol=1e-5)
    np.testing.assert_array_equal(out[:, 1], adv[:, 1])  # one active row
    np.testing.assert_array_equal(out[:, 2], adv[:, 2])  # none
    assert np.isfinite(out).all()


@pytest.mark.parametrize("per_agent", [False, True])
def test_masked_statistics_reduce_to_the_plain_update_when_every_row_is_active(
    per_agent,
):
    from algorithms.mappo_jax.mappo import create_train_state
    from algorithms.mappo_jax.types import MAPPOConfig

    T, E, N, D, G = 8, 3, 2, 5, 7
    cfg = MAPPOConfig(n_steps=T, n_envs=E, n_epochs=2, n_minibatches=2, hidden_dim=8)
    key = jax.random.PRNGKey(0)
    ts = create_train_state(key, cfg, D, G, 2, False, n_critic_outputs=N if per_agent else 1)
    ks = jax.random.split(key, 6)
    shape = (T, E, N) if per_agent else (T, E)
    traj = Transition(
        obs=jax.random.normal(ks[0], (T, E, N, D)),
        global_state=jax.random.normal(ks[1], (T, E, G)),
        action=jax.random.normal(ks[2], (T, E, N, 2)),
        reward=jax.random.normal(ks[3], shape),
        done=jax.random.bernoulli(ks[4], 0.2, (T, E)),
        log_prob=-jnp.ones((T, E, N)),
        value=jax.random.normal(ks[5], shape),
        team_reward=jnp.zeros((T, E)),
        active_mask=jnp.ones((T, E, N)),
        action_mask=jnp.zeros((T,)),
    )
    last = jnp.zeros((E, N) if per_agent else (E,))
    plain, lp = ppo_update(ts, key, traj, last, cfg, False)
    masked, lm = ppo_update(ts, key, traj, last, cfg, False, masked_statistics=True)
    assert _tree_max_diff(plain, masked) < 1e-5
    for k in lp:
        assert float(lp[k]) == pytest.approx(float(lm[k]), rel=1e-4, abs=1e-6), k


def test_masked_statistics_rejects_an_advantage_correction():
    from algorithms.mappo_jax.mappo import create_train_state
    from algorithms.mappo_jax.types import MAPPOConfig

    cfg = MAPPOConfig(n_steps=4, n_envs=2, hidden_dim=8)
    ts = create_train_state(jax.random.PRNGKey(0), cfg, 3, 4, 2, False)
    traj = Transition(
        obs=jnp.zeros((4, 2, 2, 3)), global_state=jnp.zeros((4, 2, 4)),
        action=jnp.zeros((4, 2, 2, 2)), reward=jnp.zeros((4, 2)),
        done=jnp.zeros((4, 2), bool), log_prob=jnp.zeros((4, 2, 2)),
        value=jnp.zeros((4, 2)), team_reward=jnp.zeros((4, 2)),
        active_mask=jnp.ones((4, 2, 2)), action_mask=jnp.zeros((4,)),
    )
    with pytest.raises(ValueError, match="masked_statistics"):
        ppo_update(
            ts, jax.random.PRNGKey(0), traj, jnp.zeros(2), cfg, False,
            advantage_correction=jnp.zeros((4, 2, 2)), masked_statistics=True,
        )


# ------------------------------------------------------------------ end to end


@pytest.mark.parametrize("interval", [1, 2])
def test_n_total_steps_caps_simulator_steps(interval):
    """Under forks an update charges its main rollout plus every stepped fork
    lane to `n_total_steps`, and that charge equals the logged fork lanes. Off,
    the charge is the main rollout, as before."""
    n_agents = 3
    config = _config(interventions=True, intervention_interval=interval)
    main = N_STEPS * N_ENVS
    per_update = iv.simulator_steps_per_update(config, n_agents)
    assert per_update == main + math.ceil(N_WINDOWS / interval) * N_ENVS * n_agents * HORIZON
    assert make_train(config, StubEnv(n_agents))[-1] == (
        config.worker.n_total_steps // per_update
    )
    _, _, (_, rollout, _, _), _ = _forks(StubEnv(n_agents), intervention_interval=interval)
    assert main + float(rollout.diagnostics["intervention_stepped_lanes"]) == per_update

    off = _config()
    assert iv.simulator_steps_per_update(off, n_agents) == main
    assert make_train(off, StubEnv(n_agents))[-1] == off.worker.n_total_steps // main


def test_the_runner_charges_what_the_budget_charges():
    """The runner's per-update step count (logged `total_steps`, resume index)
    is the same function `make_train` divides the budget by."""
    from algorithms.mappo_jax.run import MAPPO_JAX_Runner
    from algorithms.simplified_feudal_mappo_jax.run import (
        Simplified_Feudal_MAPPO_JAX_Runner as Runner,
    )

    runner = Runner.__new__(Runner)
    runner.feudal_config, runner.env = _config(interventions=True), StubEnv(3)
    assert runner._steps_per_update() == iv.simulator_steps_per_update(
        runner.feudal_config, 3
    )
    base = MAPPO_JAX_Runner.__new__(MAPPO_JAX_Runner)
    base.config = runner.feudal_config.worker
    assert base._steps_per_update() == N_STEPS * N_ENVS


@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.parametrize("n_agents", [2, 3])
def test_fork_collect_update_eval_end_to_end(n_agents, interval):
    _, runner_state, (rs, rollout, last, _), (update_fn, eval_fn) = _forks(
        StubEnv(n_agents, max_steps=10), intervention_interval=interval
    )
    lanes = _lanes(n_agents)
    fork = rollout.intervention
    assert fork.worker.reward.shape == (N_STEPS, lanes, n_agents)
    assert fork.manager.reward.shape == (N_WINDOWS, lanes)
    new_state, losses = update_fn(rs, rollout, last)
    for key in (
        "worker_policy_loss", "manager_policy_loss", "manager_value_loss",
        "worker_explained_variance", "manager_explained_variance",
        "intervention_attempted", "intervention_valid_frac",
        "intervention_recruit_distance", "intervention_fork_length",
        "intervention_window_return", "intervention_worker_progress",
        "intervention_sim_steps", "intervention_stepped_lanes",
        "intervention_attempted_n1", "intervention_valid_frac_n1",
        "worker_eligible_main", "worker_eligible_fork",
        "manager_eligible_main", "manager_eligible_fork",
        "worker_adv_std_main", "worker_adv_std_fork",
        "manager_explained_variance_main", "manager_explained_variance_fork",
    ):
        assert np.isfinite(float(losses[key])), key
    assert float(losses["intervention_attempted"]) == math.ceil(N_WINDOWS / interval) * lanes
    assert float(losses["intervention_recruit_distance"]) == pytest.approx(0.05, abs=1e-5)
    assert float(losses["manager_eligible_fork"]) == math.ceil(N_WINDOWS / interval) * lanes
    for level in ("worker", "manager"):
        before = getattr(runner_state.train_state, level).actor_ts.params
        after = getattr(new_state.train_state, level).actor_ts.params
        assert _tree_max_diff(before, after) > 0.0, level
    assert np.isfinite(float(eval_fn(new_state.train_state, jax.random.PRNGKey(1))))


def test_fork_checkpoints_round_trip():
    from flax.serialization import from_bytes, to_bytes

    from algorithms.simplified_feudal_mappo_jax.run import (
        Simplified_Feudal_MAPPO_JAX_Runner as Runner,
    )

    rs = make_train(_config(interventions=True), StubEnv(3))[0](jax.random.PRNGKey(0))
    eval_rng = jax.random.PRNGKey(7)
    moved = rs._replace(
        train_state=rs.train_state._replace(
            manager=rs.train_state.manager._replace(
                critic_ts=rs.train_state.manager.critic_ts.replace(
                    params=jax.tree.map(
                        lambda p: p + 1.0, rs.train_state.manager.critic_ts.params
                    )
                )
            )
        )
    )
    restored = from_bytes(
        Runner._checkpoint_tree(rs, eval_rng), to_bytes(Runner._checkpoint_tree(moved, eval_rng))
    )
    assert _tree_max_diff(
        restored["manager_critic_ts"].params, moved.train_state.manager.critic_ts.params
    ) == 0.0


# ------------------------------------------------------------------ MJX hook


@pytest.fixture(scope="module")
def mjx_env():
    from environments.mjx_suite.multi_box_push_mjx import MultiBoxPushMJX

    return MultiBoxPushMJX(
        n_agents=4, n_objects=2, coupling_def=[2, 2], variant="trunc",
        use_global_state=True,
    )


def _posed(env, agent_pos, box_pos=None, vel=1.0):
    """A reset state with agents (and boxes) placed and nonzero velocities."""
    from environments.mjx_suite.multi_box_push_mjx import _pose

    _, state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = _pose(env, state, agent_pos=agent_pos, box_pos=box_pos)
    qvel = state.data.qvel + vel * jnp.arange(state.data.qvel.shape[0]) / 10.0
    return dataclasses.replace(state, data=state.data.replace(qvel=qvel))


def _teleport(env, state, focal, recruits, offset=(0.0, 0.0), radius=1.5, seed=0):
    mask = jnp.zeros(env.n_agents, bool).at[jnp.asarray(recruits)].set(True)
    fn = jax.jit(partial(env.teleport_agents, radius=radius))
    return fn(state, focal, mask, jnp.asarray(offset), jax.random.PRNGKey(seed)), mask


def test_teleport_places_recruits_next_to_the_focal_agent_and_keeps_the_rest(mjx_env):
    env = mjx_env
    agents = [[10.0, 6.0], [20.0, 6.0], [5.0, 4.0], [25.0, 4.0]]
    state = _posed(env, agents, box_pos=[[10.0, 15.0], [20.0, 15.0]])
    (obs, new, valid), mask = _teleport(env, state, 0, [1, 3])
    assert bool(valid)
    pos0, pos1 = np.asarray(env._agent_pos(state.data)), np.asarray(env._agent_pos(new.data))
    np.testing.assert_array_equal(pos1[[0, 2]], pos0[[0, 2]])
    d = np.linalg.norm(pos1[[1, 3]] - pos1[0], axis=-1)
    assert (d >= 0.85 - 1e-5).all() and (d <= 1.5 + 1e-5).all()
    pair = np.linalg.norm(pos1[:, None] - pos1[None], axis=-1) + 99 * np.eye(4)
    assert pair.min() >= 0.85 - 1e-5
    # Physics state other than the recruits' positions is preserved.
    np.testing.assert_array_equal(new.data.qvel, state.data.qvel)
    np.testing.assert_array_equal(
        np.asarray(new.data.qpos)[np.asarray(env._box_qadr)],
        np.asarray(state.data.qpos)[np.asarray(env._box_qadr)],
    )
    for field in ("t", "prev_box_goal_dist", "delivered"):
        np.testing.assert_array_equal(getattr(new, field), getattr(state, field))
    # The observation is rebuilt from the new state (lidar differs ~1e-4 across
    # compilations, hence the tolerance).
    np.testing.assert_allclose(obs, env._get_obs(new.data, new.delivered), atol=1e-3)


def test_teleport_keeps_clear_of_walls_and_box_faces(mjx_env):
    env = mjx_env
    # Focal agent in the bottom-left corner; a box right next to it.
    agents = [[1.2, 1.2], [20.0, 6.0], [25.0, 6.0], [15.0, 6.0]]
    state = _posed(env, agents, box_pos=[[3.0, 3.6], [20.0, 15.0]])
    placed = 0
    for seed in range(8):
        (_, new, valid), _ = _teleport(env, state, 0, [1], seed=seed)
        if not bool(valid):
            continue
        placed += 1
        pos = np.asarray(env._agent_pos(new.data))
        lo = env.boundary_thickness + 0.4
        assert (pos >= lo).all() and (pos <= env.world_width - lo).all()
        from environments.mjx_suite.observation import box_surface_distance

        box_pos, box_yaw = env._box_pose(new.data)
        gap = np.asarray(box_surface_distance(jnp.asarray(pos), box_pos, box_yaw, env._box_half))
        assert gap.min() >= 0.4
    assert placed > 0  # the corner still leaves room for one recruit


def test_teleport_fails_whole_and_leaves_the_state_unchanged(mjx_env):
    env = mjx_env
    agents = [[10.0, 6.0], [20.0, 6.0], [5.0, 4.0], [25.0, 4.0]]
    state = _posed(env, agents)
    # An offset that puts every translated waypoint past the arena edge.
    (obs, new, valid), _ = _teleport(env, state, 0, [1], offset=(0.0, 0.9))
    assert not bool(valid)
    for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(state)):  # some are empty
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="radius"):
        _teleport(env, state, 0, [1], radius=0.5)


def test_the_first_step_after_a_teleport_pays_no_jump(mjx_env):
    """Teleporting pays nothing: the first step's shaping is exactly the boxes'
    own motion against the unchanged `prev_box_goal_dist`."""
    env = mjx_env
    agents = [[10.0, 6.0], [20.0, 6.0], [5.0, 4.0], [25.0, 4.0]]
    state = _posed(env, agents, box_pos=[[10.0, 12.0], [20.0, 15.0]])
    (_, new, valid), _ = _teleport(env, state, 0, [2])
    assert bool(valid)
    obs, after, reward, *_ = jax.jit(env.step)(new, jnp.zeros((env.n_agents, 2)))
    box_y = np.asarray(env._box_pose(after.data)[0])[:, 1]
    shaping = (np.asarray(new.prev_box_goal_dist) - (env.target_y - box_y)).sum()
    assert np.isfinite(np.asarray(obs)).all()
    assert float(reward) == pytest.approx(shaping, abs=1e-5)
