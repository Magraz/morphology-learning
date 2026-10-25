"""Seam tests for the heuristic fork rule (`intervention_scheduler: heuristic`,
`algorithms/simplified_feudal_mappo_jax/intervention_heuristic.py`).

CPU stub env (deterministic and fast). The rule itself is tested on hand-built
contact geometry; the collector on the stub, whose `box_contacts` treats its two
fixed boxes as discs (coupling 2 and 3).

    uv run pytest algorithms/tests/test_intervention_heuristic.py -q
"""

import dataclasses
import math
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_simplified_feudal import (
    HORIZON,
    N_ENVS,
    N_STEPS,
    OBS_DIM,
    StubContacts,
    StubEnv,
    _config,
    _params,
    _tree_max_diff,
)

from algorithms.mappo_jax.network import evaluate_action
from algorithms.mappo_jax.types import MAPPOConfig, Transition
from algorithms.simplified_feudal_mappo_jax import intervention_heuristic as ih
from algorithms.simplified_feudal_mappo_jax import interventions as iv
from algorithms.simplified_feudal_mappo_jax.trainer import make_train
from algorithms.simplified_feudal_mappo_jax.types import Model_Params

N_WINDOWS = N_STEPS // HORIZON  # 3
MARGIN = 0.1


@pytest.fixture(autouse=True)
def _run_on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _heuristic(budget=2, margin=1.0, **kwargs):
    """A heuristic-scheduler config on the stub. Margin 1.0 makes every agent
    near a box (the stub's world is [-0.3, 0.3]), so every step qualifies."""
    kwargs.setdefault("manager_input", "relative")
    config = _config(interventions=True, intervention_redecide="all", **kwargs)
    return dataclasses.replace(
        config,
        intervention_scheduler="heuristic",
        intervention_heuristic_budget=budget,
        intervention_heuristic_trigger_margin=margin,
    )


def _run(config, env, seed=0):
    init_fn, collect_fn, update_fn, eval_fn, num_updates = make_train(config, env)
    rs = init_fn(jax.random.PRNGKey(seed))
    return rs, collect_fn(rs), (collect_fn, update_fn, eval_fn, num_updates)


def _contacts(pos, clearance, touching=None, coupling=(2, 3, 1), delivered=None):
    """Hand-built contact geometry: `pos` (A, 2), `clearance` (A, O)."""
    clearance = jnp.asarray(clearance, jnp.float32)
    if touching is None:
        touching = clearance <= 0
    n_boxes = clearance.shape[1]
    return StubContacts(
        agent_pos=jnp.asarray(pos, jnp.float32),
        clearance=clearance,
        touching=jnp.asarray(touching),
        coupling=jnp.asarray(coupling, jnp.int32),
        delivered=jnp.zeros(n_boxes, bool) if delivered is None else jnp.asarray(delivered),
        goal_distance=jnp.zeros(n_boxes),
    )


KEY = jax.random.PRNGKey(0)
FAR = 5.0
# Four agents on a line, so "nearest to agent 0" is the index order.
POS = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]


# ------------------------------------------------------------------ the rule


def test_an_agent_near_an_understaffed_box_recruits_exactly_the_missing_agents():
    # Agent 0 is 0.05 from box 1 (coupling 3), nobody touches it: N = 3 - 1 = 2.
    c = _contacts(POS, [[FAR, 0.05, FAR], [FAR] * 3, [FAR] * 3, [FAR] * 3])
    m = ih.candidate(c, KEY, MARGIN)
    assert bool(m.qualifies) and int(m.focal) == 0 and int(m.box) == 1
    assert int(m.n_recruits) == 2 and not bool(m.touching)
    # The two teammates nearest agent 0.
    np.testing.assert_array_equal(np.asarray(m.recruit), [False, True, True, False])
    assert float(m.clearance) == pytest.approx(0.05)


def test_an_agent_already_touching_counts_and_teammates_on_the_box_are_kept():
    # Agents 0 and 3 touch box 1 (coupling 3): agent 0 qualifies with N = 1; the
    # recruit is the nearest teammate NOT on the box (agent 1), never agent 3.
    c = _contacts(POS, [[FAR, -0.1, FAR], [FAR] * 3, [FAR] * 3, [FAR, -0.1, FAR]])
    m = jax.vmap(lambda k: ih.candidate(c, k, MARGIN))(jax.random.split(KEY, 64))
    focal = np.asarray(m.focal)
    assert set(focal) == {0, 3}  # both touching agents qualify
    for f, n, r, t in zip(focal, np.asarray(m.n_recruits), np.asarray(m.recruit),
                          np.asarray(m.touching)):
        assert n == 1 and t
        assert not r[0] and not r[3] and r.sum() == 1
        assert r[1] if f == 0 else r[2]  # nearest non-touching teammate


def test_met_and_single_agent_boxes_never_qualify():
    # Box 0 (coupling 2) touched by agents 0 and 1: met. Box 2 needs one agent.
    met = _contacts(POS, [[-0.1, FAR, FAR], [-0.1, FAR, FAR], [FAR] * 3, [FAR] * 3])
    assert not bool(ih.candidate(met, KEY, MARGIN).qualifies)
    single = _contacts(POS, [[FAR, FAR, 0.0], [FAR] * 3, [FAR] * 3, [FAR] * 3])
    assert not bool(ih.candidate(single, KEY, MARGIN).qualifies)
    # Box 0 with ONE agent touching and agent 2 approaching: once agent 2 is
    # counted the coupling is met, so no help is needed.
    almost = _contacts(POS, [[-0.1, FAR, FAR], [FAR] * 3, [0.05, FAR, FAR], [FAR] * 3])
    m = ih.candidate(almost, KEY, MARGIN)
    assert bool(m.qualifies) and int(m.focal) == 0 and int(m.n_recruits) == 1


def test_the_trigger_margin_and_delivered_boxes():
    far = _contacts(POS, [[FAR, 0.2, FAR], [FAR] * 3, [FAR] * 3, [FAR] * 3])
    assert not bool(ih.candidate(far, KEY, MARGIN).qualifies)
    assert bool(ih.candidate(far, KEY, 0.25).qualifies)
    # The closest box is delivered: the next closest undelivered one counts.
    c = _contacts(
        POS, [[0.0, 0.05, FAR], [FAR] * 3, [FAR] * 3, [FAR] * 3],
        delivered=[True, False, False],
    )
    m = ih.candidate(c, KEY, MARGIN)
    assert bool(m.qualifies) and int(m.box) == 1 and int(m.n_recruits) == 2
    every = _contacts(
        POS, [[0.0, 0.05, 0.0]] * 4, delivered=[True, True, True]
    )
    m = ih.candidate(every, KEY, MARGIN)
    assert not bool(m.qualifies) and int(m.n_recruits) == 0 and not np.asarray(m.recruit).any()


def test_too_few_free_teammates_do_not_qualify():
    # A box needing 5 of a 4-agent team: N = 4 against 3 free teammates.
    c = _contacts(POS, [[FAR, 0.05, FAR], [FAR] * 3, [FAR] * 3, [FAR] * 3],
                  coupling=(2, 5, 1))
    assert not bool(ih.candidate(c, KEY, MARGIN).qualifies)


def test_the_focal_agent_is_uniform_over_qualifiers():
    c = _contacts(POS, [[FAR, 0.05, FAR], [FAR, 0.05, FAR], [FAR] * 3, [FAR] * 3])
    m = jax.vmap(lambda k: ih.candidate(c, k, MARGIN))(jax.random.split(KEY, 4000))
    share = (np.asarray(m.focal) == 0).mean()
    assert abs(share - 0.5) < 0.03


# ------------------------------------------------------------ the reservoir


def _stream(n_envs, budget, qualifies, alive=None, key=KEY):
    """Offer `(T, E)` qualifying flags; the env state at step t is `t`."""
    n_steps = qualifies.shape[0]
    state = jnp.zeros(n_envs, jnp.int32)
    obs = jnp.zeros((n_envs, 2, 3))
    res = ih.reservoir_init(state, obs, budget)
    alive = jnp.ones_like(qualifies) if alive is None else alive
    for t in range(n_steps):
        moment = ih.Moment(
            qualifies=qualifies[t],
            focal=jnp.zeros(n_envs, jnp.int32),
            n_recruits=jnp.full(n_envs, t, jnp.int32),
            recruit=jnp.zeros((n_envs, 2), bool),
            box=jnp.zeros(n_envs, jnp.int32),
            clearance=jnp.zeros(n_envs),
            touching=jnp.zeros(n_envs, bool),
        )
        res = ih.reservoir_update(
            res, moment, jnp.full(n_envs, t, jnp.int32), obs + t, alive[t], t,
            jax.random.fold_in(key, t),
        )
    return res


def test_the_reservoir_keeps_min_budget_and_events_and_their_states():
    q = jnp.array([[True, False], [True, False], [False, True], [True, False],
                   [True, False], [True, False]])
    res = _stream(2, 3, q)
    filled = np.isfinite(np.asarray(res.priority))
    assert filled[0].sum() == 3 and filled[1].sum() == 1
    np.testing.assert_array_equal(np.asarray(res.events), [5, 1])
    steps = np.asarray(res.step)
    assert set(steps[0][filled[0]]) <= {0, 1, 3, 4, 5}
    assert steps[1][filled[1]].tolist() == [2]
    # Each slot holds the state, observation and moment of its own step.
    np.testing.assert_array_equal(np.asarray(res.env_state)[filled], steps[filled])
    np.testing.assert_array_equal(np.asarray(res.obs)[filled][:, 0, 0], steps[filled])
    np.testing.assert_array_equal(np.asarray(res.moment.n_recruits)[filled], steps[filled])


def test_frozen_envs_offer_nothing():
    q = jnp.ones((4, 2), bool)
    alive = jnp.array([[True, False]] * 4)
    res = _stream(2, 3, q, alive=alive)
    np.testing.assert_array_equal(np.asarray(res.events), [4, 0])
    assert not np.isfinite(np.asarray(res.priority))[1].any()


def test_the_reservoir_is_a_uniform_sample():
    n_steps, budget, n_draws = 10, 3, 3000
    q = jnp.ones((n_steps, 1), bool)
    keys = jax.random.split(KEY, n_draws)
    steps = jax.vmap(lambda k: _stream(1, budget, q, key=k).step[0])(keys)
    counts = np.bincount(np.asarray(steps).ravel(), minlength=n_steps) / n_draws
    np.testing.assert_allclose(counts, budget / n_steps, atol=0.035)
    # Without replacement: a step never fills two slots.
    assert all(len(set(row)) == budget for row in np.asarray(steps).tolist())


# ------------------------------------------------------------------- packing


def _lanes(n_lanes, horizon=2, n_agents=2):
    """Per-lane Transitions whose every field carries the lane index."""
    lane = jnp.arange(n_lanes, dtype=jnp.float32)

    def tr(lead):
        shape = lead + (n_lanes,)
        x = jnp.broadcast_to(lane, shape)
        return Transition(
            obs=x[..., None, None] * jnp.ones((n_agents, 3)),
            global_state=x[..., None],
            action=x[..., None, None] * jnp.ones((n_agents, 2)),
            reward=x,
            done=jnp.zeros(shape, bool),
            log_prob=x[..., None] * jnp.ones(n_agents),
            value=x,
            team_reward=x,
            active_mask=jnp.ones(shape + (n_agents,)),
            action_mask=jnp.zeros(()),
        )

    return tr((horizon,)), tr(())


def test_pack_puts_kept_lanes_first_and_maps_worker_rows():
    n_windows, horizon = 3, 2
    worker, manager = _lanes(7, horizon)
    keep = jnp.array([False, True, True, False, True, True, True])  # 5 kept
    w, m, order, kept, dropped = ih.pack(worker, manager, keep, n_windows)
    assert m.reward.shape == (n_windows, 3) and w.reward.shape == (n_windows * horizon, 3)
    assert int(dropped) == 0
    # Column 0 rows 0-2 hold lanes 1, 2, 4; column 1 rows 0-1 hold lanes 5, 6.
    np.testing.assert_array_equal(np.asarray(m.reward)[:, 0], [1, 2, 4])
    np.testing.assert_array_equal(np.asarray(m.reward)[:2, 1], [5, 6])
    m_active = np.asarray(m.active_mask)[..., 0]
    np.testing.assert_array_equal(m_active, [[1, 1, 0], [1, 1, 0], [1, 0, 0]])
    assert np.asarray(m.done)[m_active == 0].all()
    # Worker rows [w*H, (w+1)*H) of column c are the same lane.
    w_reward = np.asarray(w.reward).reshape(n_windows, horizon, 3)
    np.testing.assert_array_equal(w_reward[:, :, 0], [[1, 1], [2, 2], [4, 4]])
    w_active = np.asarray(w.active_mask)[..., 0].reshape(n_windows, horizon, 3)
    np.testing.assert_array_equal(w_active[:, 0], m_active)
    w_done = np.asarray(w.done).reshape(n_windows, horizon, 3).transpose(0, 2, 1)
    assert w_done[m_active == 0].all()
    # The placeholder passes through.
    assert w.action_mask.shape == worker.action_mask.shape


def test_pack_masks_a_singleton_column():
    worker, manager = _lanes(7)
    keep = jnp.array([True, True, True, True, False, False, False])  # 3 + 1
    w, m, _, kept, dropped = ih.pack(worker, manager, keep, 3)
    assert int(dropped) == 1
    m_active = np.asarray(m.active_mask)[..., 0]
    assert m_active[:, 0].sum() == 3 and m_active[:, 1:].sum() == 0
    assert not np.asarray(w.active_mask)[:, 1:].any()
    assert np.asarray(kept).sum() == 3


def test_pack_never_lets_gae_cross_forks():
    worker, manager = _lanes(4)
    # Each lane's worker rows end in a terminal, its manager row is one.
    worker_done = worker._replace(done=worker.done.at[-1].set(True))
    w, m, *_ = ih.pack(worker_done, manager._replace(done=jnp.ones(4, bool)),
                       jnp.ones(4, bool), 2)
    done = np.asarray(w.done).reshape(2, 2, 2)
    assert done[:, -1].all() and np.asarray(m.done).all()


# -------------------------------------------------------- paired baseline


def test_the_paired_main_return_stops_at_the_episode_end():
    horizon, gamma = 4, 0.5
    n_steps, n_envs = 12, 1
    reward = jnp.ones((n_steps, n_envs))
    active = jnp.ones((n_steps, n_envs))
    # Env 0: episode ends at the LAST step of window 0 (step 3): window 1 is a
    # new episode with active == 1 throughout, which only `m_done` reveals.
    m_done = jnp.array([[True], [False], [False]])
    step = jnp.array([2, 5, 10])
    env = jnp.zeros(3, jnp.int32)
    g, complete = ih.paired_main_return(reward, active, m_done, step, env, horizon, gamma)
    np.testing.assert_allclose(np.asarray(g), [1 + 0.5, 1 + 0.5 + 0.25 + 0.125, 1 + 0.5])
    np.testing.assert_array_equal(np.asarray(complete), [True, True, False])
    # A mid-window termination: steps after it are frozen (active 0).
    active = active.at[6:8].set(0.0)
    g, _ = ih.paired_main_return(reward, active, m_done, step, env, horizon, gamma)
    assert float(g[1]) == pytest.approx(1.0)


# --------------------------------------------------------------- collector


def test_heuristic_forks_start_mid_window_from_the_main_rollouts_state():
    """Each fork's start is the main rollout's PRE-step state at its moment, and
    moments fall at every in-window offset, not only at window starts. Non-recruits
    are not moved, so their observation matches the main one exactly."""
    n_agents = 3
    _, (_, rollout, _, stats), _ = _run(_heuristic(budget=3), StubEnv(n_agents))
    fork = rollout.intervention
    valid = np.asarray(fork.valid)
    rows, cols = np.nonzero(valid)
    steps, envs = np.asarray(fork.step)[rows, cols], np.asarray(fork.env)[rows, cols]
    focal, n_rec = np.asarray(fork.focal)[rows, cols], np.asarray(fork.n_recruits)[rows, cols]
    assert valid.sum() == N_ENVS * 3
    assert (steps % HORIZON != 0).any()
    # Every env contributes its own budget of moments.
    np.testing.assert_array_equal(np.bincount(envs, minlength=N_ENVS), [3] * N_ENVS)
    obs = np.asarray(fork.worker.obs)
    w_obs = obs.reshape((N_WINDOWS, HORIZON) + obs.shape[1:])
    main_obs = np.asarray(rollout.worker.obs)
    for r, c, tau, e, f, n in zip(rows, cols, steps, envs, focal, n_rec):
        first = w_obs[r, 0, c]  # (N, actor_dim): the fork's first worker step
        # The focal agent never moves: its observation is the main one at tau.
        np.testing.assert_array_equal(first[f, :OBS_DIM], main_obs[tau, e, f, :OBS_DIM])
        assert 1 <= n <= n_agents - 1


def test_every_agent_re_decides_and_the_ppo_ratio_is_one():
    n_agents = 3
    rs, (_, rollout, _, _), _ = _run(_heuristic(), StubEnv(n_agents))
    m = rollout.intervention.manager
    valid = np.asarray(rollout.intervention.valid)
    active = np.asarray(m.active_mask)
    assert (active[valid] == 1).all() and (active[~valid] == 0).all()
    ts = rs.train_state.manager.actor_ts
    d, a = m.obs.shape[-1], m.action.shape[-1]
    log_prob, _ = evaluate_action(
        ts.apply_fn, ts.params, m.obs.reshape(-1, d), m.action.reshape(-1, a),
        discrete=False,
    )
    log_prob = np.asarray(log_prob).reshape(m.log_prob.shape)
    np.testing.assert_allclose(log_prob[valid], np.asarray(m.log_prob)[valid], atol=1e-5)


def test_the_heuristic_leaves_the_main_rollout_unchanged():
    """Its draws come from `fold_in`, so the main rollout equals the parent's at
    the same env count and seed (only the manager critic's values differ,
    through the fork context block)."""
    n_agents = 3
    _, (_, parent, _, _), _ = _run(_config(manager_input="relative"), StubEnv(n_agents))
    _, (_, rollout, _, _), _ = _run(_heuristic(), StubEnv(n_agents))
    assert _tree_max_diff(rollout.worker, parent.worker) == 0.0
    for field in ("obs", "action", "log_prob", "reward", "done", "team_reward"):
        np.testing.assert_array_equal(
            getattr(rollout.manager, field), getattr(parent.manager, field), field
        )


class FarBoxes(StubEnv):
    """The stub with every box far out of reach: nothing ever qualifies."""

    def box_contacts(self, state):
        c = super().box_contacts(state)
        return c._replace(clearance=c.clearance + 10.0, touching=c.touching & False)


def test_no_qualifying_moment_means_no_fork():
    rs, (_, rollout, last, _), (_, update_fn, _, _) = _run(_heuristic(), FarBoxes(3))
    d = rollout.diagnostics
    assert float(d["intervention_events"]) == 0 and float(d["intervention_attempted"]) == 0
    assert not np.asarray(rollout.intervention.valid).any()
    assert float(d["manager_eligible_fork"]) == 0 and float(d["worker_eligible_fork"]) == 0
    _, losses = update_fn(rs, rollout, last)
    assert np.isfinite(float(losses["manager_total_loss"]))


def test_end_to_end_update_and_logged_series():
    n_agents = 3
    rs, (_, rollout, last, _), (collect_fn, update_fn, eval_fn, num_updates) = _run(
        _heuristic(budget=2), StubEnv(n_agents)
    )
    d = rollout.diagnostics
    for key in (
        "intervention_events", "intervention_event_rate", "intervention_attempted",
        "intervention_budget_frac", "intervention_valid_frac",
        "intervention_dropped_singleton", "intervention_n_recruits_mean",
        "intervention_focal_touching_frac", "intervention_focal_clearance",
        "intervention_window_step_mean", "intervention_window_start_frac",
        "intervention_start_met_frac", "intervention_box_delivered_frac",
        "intervention_box_progress", "intervention_paired_gain",
        "intervention_g_fork", "intervention_g_main", "intervention_paired",
        "intervention_redecide_mean_shift", "intervention_redecide_recruit_waypoint_gap",
        "intervention_delivery_frac", "intervention_attempted_n1",
        "intervention_attempted_n2", "manager_eligible_fork", "worker_eligible_fork",
    ):
        assert np.isfinite(float(d[key])), key
    assert float(d["intervention_attempted"]) == N_ENVS * 2  # every step qualifies
    valid = np.asarray(rollout.intervention.valid)
    assert float(d["manager_eligible_fork"]) == valid.sum() * n_agents
    rs, losses = update_fn(rs, rollout, last)
    assert all(np.isfinite(float(v)) for k, v in losses.items()
               if "explained_variance" not in k)
    assert np.isfinite(float(eval_fn(rs.train_state, jax.random.PRNGKey(1))))


@pytest.mark.parametrize("budget", [1, 4])
def test_failed_placements_and_every_budget_are_masked(budget):
    env = StubEnv(3, fail_teleport=True)
    rs, (_, rollout, last, _), (_, update_fn, _, _) = _run(_heuristic(budget=budget), env)
    assert not np.asarray(rollout.intervention.valid).any()
    assert float(rollout.diagnostics["manager_eligible_fork"]) == 0
    _, losses = update_fn(rs, rollout, last)
    assert np.isfinite(float(losses["manager_total_loss"]))


# ---------------------------------------------------------- config / budget


def test_heuristic_batch_arithmetic():
    assert iv.heuristic_main_envs(32, 33, 5) == 28
    worker = MAPPOConfig(n_steps=1056, n_envs=28, n_total_steps=int(1e8))
    config = types.SimpleNamespace(
        worker=worker, goal_horizon=32, interventions=True,
        intervention_scheduler="heuristic", intervention_heuristic_budget=5,
        intervention_batch_envs=None,
    )
    assert iv.simulator_steps_per_update(config) == 28 * 1056 + 28 * 5 * 32 == 34048
    assert int(1e8) // 34048 == 2937
    # The stub config: main steps plus every fork lane.
    stub = _heuristic(budget=2)
    assert iv.simulator_steps_per_update(stub) == N_ENVS * N_STEPS + N_ENVS * 2 * HORIZON
    _, _, (_, _, _, num_updates) = _run(stub, StubEnv(3))
    assert num_updates == stub.worker.n_total_steps // iv.simulator_steps_per_update(stub)


def test_make_feudal_config_wires_the_heuristic():
    from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config

    mp = Model_Params(
        hidden_dim=8, goal_horizon=32, waypoint_radius=0.15, interventions=True,
        intervention_scheduler="heuristic", intervention_redecide="all",
        intervention_heuristic_budget=5, intervention_heuristic_trigger_margin=0.7,
    )
    config = make_feudal_config(_params(), mp, n_envs=32)
    assert config.worker.n_envs == 28 and config.worker.n_steps == 1056
    assert config.intervention_heuristic_budget == 5
    assert config.intervention_heuristic_trigger_margin == 0.7
    assert config.intervention_batch_envs is None
    assert iv.simulator_steps_per_update(config) == 34048


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"intervention_redecide": "none"}, "must be 'all'"),
        ({"intervention_interval": 2}, "intervention_interval must be 1"),
        ({"intervention_heuristic_budget": 0}, "integer >= 1"),
        ({"intervention_heuristic_trigger_margin": -0.1}, "must be >= 0"),
        ({"intervention_batch_envs": 4}, "must be None"),
        ({"interventions": False}, "has no effect without"),
    ],
)
def test_bad_heuristic_configs_are_rejected(overrides, match):
    config = dataclasses.replace(_heuristic(), **overrides)
    with pytest.raises(ValueError, match=match):
        make_train(config, StubEnv(3))


def test_heuristic_knobs_are_rejected_under_other_schedulers():
    for name, value in (
        ("intervention_heuristic_budget", 3),
        ("intervention_heuristic_trigger_margin", 0.2),
    ):
        config = dataclasses.replace(_config(interventions=True), **{name: value})
        with pytest.raises(ValueError, match="has no effect under"):
            make_train(config, StubEnv(3))


class Hookless(StubEnv):
    """The stub without the `box_contacts` hook."""

    def __getattribute__(self, name):
        if name == "box_contacts":
            raise AttributeError(name)
        return super().__getattribute__(name)


def test_the_heuristic_needs_the_contacts_hook():
    with pytest.raises(ValueError, match="box_contacts"):
        make_train(_heuristic(), Hookless(3))


# -------------------------------------------------------------- checkpoints


def _runner(config):
    from algorithms.simplified_feudal_mappo_jax.run import (
        Simplified_Feudal_MAPPO_JAX_Runner as Runner,
    )

    runner = Runner.__new__(Runner)
    runner.feudal_config = config
    return runner


def test_the_heuristic_checkpoint_record_refuses_other_settings(tmp_path):
    config = _heuristic(budget=2)
    init_fn = make_train(config, StubEnv(3))[0]
    rs, eval_rng = init_fn(jax.random.PRNGKey(0)), jax.random.PRNGKey(7)
    path = tmp_path / "heuristic.msgpack"
    _runner(config)._save_train_checkpoint(rs, eval_rng, path)
    meta = _runner(config)._scheduler_meta()
    np.testing.assert_array_equal(meta, [3, 2, 1.0, 2])  # heuristic, K, margin, all
    restored, _ = _runner(config)._load_train_checkpoint(rs, eval_rng, path)
    assert _tree_max_diff(restored.train_state, rs.train_state) == 0.0
    other = dataclasses.replace(config, intervention_heuristic_budget=3)
    with pytest.raises(ValueError, match="other fork-scheduler settings"):
        _runner(other)._load_train_checkpoint(rs, eval_rng, path)
    fixed = _config(interventions=True, manager_input="relative")
    fixed_rs = make_train(fixed, StubEnv(3))[0](jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="fork-scheduler run"):
        _runner(fixed)._load_train_checkpoint(fixed_rs, eval_rng, path)


def test_the_adaptive_checkpoint_record_is_unchanged():
    """Runs in progress resume against this record, so its fields and codes
    must not move."""
    from algorithms.simplified_feudal_mappo_jax.run import (
        Simplified_Feudal_MAPPO_JAX_Runner as Runner,
    )

    assert Runner._SCHEDULER_META_FIELDS == (
        "intervention_scheduler", "intervention_batch_envs",
        "intervention_bandit_hidden_dim", "intervention_bandit_epsilon",
        "intervention_bandit_lr", "intervention_bandit_cost_coef",
        "intervention_bandit_zero_reward_penalty", "intervention_random_rate",
        "intervention_redecide",
    )
    from algorithms.simplified_feudal_mappo_jax import intervention_bandit as ib

    assert ib.SCHEDULERS[:3] == ("fixed", "bandit", "random")
    config = dataclasses.replace(
        _config(interventions=True), intervention_scheduler="bandit",
        intervention_batch_envs=4,
    )
    meta = _runner(config)._scheduler_meta()
    np.testing.assert_array_equal(meta, [1, 4, 32, 0.1, 3e-4, 0.0, 5.0, 0.5, 0])


def test_the_heuristic_rollout_is_reproducible():
    a = _run(_heuristic(), StubEnv(3), seed=3)[1]
    b = _run(_heuristic(), StubEnv(3), seed=3)[1]
    assert _tree_max_diff(a[1].intervention, b[1].intervention) == 0.0
    assert math.isfinite(float(a[1].diagnostics["intervention_paired_gain"]))
