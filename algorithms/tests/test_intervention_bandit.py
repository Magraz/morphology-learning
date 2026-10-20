"""Seam tests for the adaptive fork scheduler (`intervention_scheduler: bandit |
random`): `intervention_bandit.py`, `adaptive_collection.py` and their update.

Run on the CPU stub env of `test_simplified_feudal.py` (deterministic, so
bit-for-bit claims can be checked). Plan:
`plans/intervention_contextual_bandit_2026-10-16.md`.

    uv run pytest algorithms/tests/test_intervention_bandit.py -q
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.serialization import to_bytes
from test_simplified_feudal import (
    HORIZON,
    N_STEPS,
    StubEnv,
    _config,
    _params,
    _tree_max_diff,
)

from algorithms.mappo_jax.mappo import (
    compute_gae,
    create_train_state,
    flatten_samples,
    optimize_samples,
    ppo_update,
    prepare_advantages,
)
from algorithms.mappo_jax.types import MAPPOConfig, Transition
from algorithms.simplified_feudal_mappo_jax import adaptive_collection as ac
from algorithms.simplified_feudal_mappo_jax import intervention_bandit as ib
from algorithms.simplified_feudal_mappo_jax import interventions as iv
from algorithms.simplified_feudal_mappo_jax.trainer import make_train
from algorithms.simplified_feudal_mappo_jax.types import FeudalConfig, Model_Params

N_WINDOWS = N_STEPS // HORIZON  # 3
NO_MASK = jnp.zeros(())
NAN_OK = {
    "bandit_greedy_fork_frac", "bandit_explained_variance",
    "bandit_gain_greedy", "bandit_gain_explored",
    "bandit_target_greedy", "bandit_target_explored",
}


@pytest.fixture(autouse=True)
def _run_on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class AlwaysPlace(StubEnv):
    """The stub env whose placement never fails (no arena check)."""

    def teleport_agents(self, state, focal, recruit_mask, offset, key, radius):
        angle = jax.random.uniform(key, (self.n_agents,), maxval=2 * jnp.pi)
        target = state.pos[focal] + radius * jnp.stack(
            [jnp.cos(angle), jnp.sin(angle)], axis=-1
        )
        new = state._replace(pos=jnp.where(recruit_mask[:, None], target, state.pos))
        return self._obs(new), new, jnp.array(True)


def _adaptive(scheduler="bandit", batch=4, **kwargs):
    """An adaptive-scheduler config on the stub: `batch` envs, half of them
    main streams (what `make_feudal_config` builds from an env group)."""
    sched = {
        k: kwargs.pop(k)
        for k in list(kwargs)
        if k.startswith("intervention_bandit") or k == "intervention_random_rate"
    }
    config = _config(interventions=True, **kwargs)
    main = batch // 2
    return dataclasses.replace(
        config,
        worker=dataclasses.replace(config.worker, n_envs=main),
        manager=dataclasses.replace(config.manager, n_envs=main),
        intervention_scheduler=scheduler,
        intervention_batch_envs=batch,
        **sched,
    )


def _run(config, env, seed=0):
    """`(runner_state, (runner_state, rollout, last, stats), fns)` after one collect."""
    init_fn, collect_fn, update_fn, eval_fn, _ = make_train(config, env)
    rs = init_fn(jax.random.PRNGKey(seed))
    return rs, collect_fn(rs), (collect_fn, update_fn, eval_fn)


def _labels(ad):
    n = int(ad.label_count)
    return {
        k: np.asarray(getattr(ad, f"label_{k}"))[:n]
        for k in ("x", "y", "round", "stream", "focal", "placed")
    }


# ---------------------------------------------------------------------- config


def test_scheduler_is_fixed_by_default():
    mp = Model_Params(hidden_dim=8, goal_horizon=32, waypoint_radius=0.15)
    assert mp.intervention_scheduler == "fixed"
    assert mp.intervention_bandit_zero_reward_penalty == 5.0
    from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config

    config = make_feudal_config(
        _params(), dataclasses.replace(mp, interventions=True), n_envs=32
    )
    assert config.intervention_scheduler == "fixed"
    assert config.intervention_batch_envs is None
    assert not iv.adaptive(config)


def test_make_feudal_config_wires_the_scheduler_keys():
    """At the production rollout (1056 steps, 33 windows of 32) half of the
    batch's 32 envs are main streams and an update is the parent's 33,792 steps."""
    from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config

    mp = Model_Params(
        hidden_dim=8, goal_horizon=32, waypoint_radius=0.15, interventions=True,
        intervention_scheduler="bandit", intervention_bandit_epsilon=0.2,
        intervention_bandit_zero_reward_penalty=0.0, intervention_bandit_lr=1e-3,
    )
    config = make_feudal_config(_params(), mp, n_envs=32)
    assert (config.intervention_batch_envs, config.worker.n_envs) == (32, 16)
    assert config.intervention_bandit_epsilon == 0.2
    assert config.intervention_bandit_zero_reward_penalty == 0.0
    assert config.intervention_bandit_lr == 1e-3
    assert iv.simulator_steps_per_update(config) == 33_792
    assert ac.window_slots(config) == 33 * 32
    assert ac.n_rounds(config) == 66
    with pytest.raises(ValueError, match="even"):
        make_feudal_config(_params(), mp, n_envs=31)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(intervention_scheduler="greedy"), "unknown intervention_scheduler"),
        (dict(interventions=False), "has no effect without"),
        (dict(intervention_interval=2), "intervention_interval must be 1"),
        (dict(intervention_batch_envs=6), "batch_envs / 2"),
        (dict(intervention_bandit_epsilon=1.5), "epsilon"),
        (dict(intervention_bandit_zero_reward_penalty=-1.0), "zero_reward_penalty"),
        (dict(intervention_bandit_cost_coef=-1.0), "cost_coef"),
        (dict(intervention_bandit_hidden_dim=0), "hidden_dim"),
        (dict(intervention_bandit_lr=0.0), "lr"),
        # A knob of the other scheduler would silently do nothing.
        (dict(intervention_random_rate=0.3), "has no effect under"),
    ],
)
def test_bad_scheduler_configs_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_train(dataclasses.replace(_adaptive(), **overrides), StubEnv(3))


def test_scheduler_knobs_are_rejected_under_the_fixed_scheduler():
    config = dataclasses.replace(
        _config(interventions=True), intervention_bandit_zero_reward_penalty=0.0
    )
    with pytest.raises(ValueError, match="has no effect under"):
        make_train(config, StubEnv(3))
    random = dataclasses.replace(_adaptive("random"), intervention_bandit_epsilon=0.3)
    with pytest.raises(ValueError, match="has no effect under"):
        make_train(random, StubEnv(3))


def test_fixed_scheduler_fields_leave_the_fixed_collector_unchanged():
    """Spelling out every scheduler default changes nothing."""
    base = _config(interventions=True)
    spelled = dataclasses.replace(base, intervention_scheduler="fixed")
    _, (_, a, _, _), _ = _run(base, StubEnv(3))
    _, (_, b, _, _), _ = _run(spelled, StubEnv(3))
    assert _tree_max_diff((a.worker, a.manager), (b.worker, b.manager)) == 0.0


# ---------------------------------------------------------------- bandit units


def test_epsilon_greedy_is_greedy_at_zero_and_a_coin_at_one():
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(k[0], (1000,))
    u_e, u_a = jax.random.uniform(k[1], (1000,)), jax.random.uniform(k[2], (1000,))
    greedy, explored, _ = ib.epsilon_greedy(q, u_e, u_a, 0.0)
    assert np.array_equal(greedy, q > 0) and not np.asarray(explored).any()
    coin, explored, prop = ib.epsilon_greedy(q, u_e, u_a, 1.0)
    assert np.array_equal(coin, u_a < 0.5) and np.asarray(explored).all()
    assert np.allclose(prop, 0.5)
    # Ties skip.
    tie, _, _ = ib.epsilon_greedy(jnp.zeros(5), jnp.ones(5), jnp.zeros(5), 0.0)
    assert not np.asarray(tie).any()


def test_exploration_frequency_and_propensity():
    """A predicted-negative context forks epsilon/2 of the time, a predicted-
    positive one 1 - epsilon/2, and the propensity says so."""
    k = jax.random.split(jax.random.PRNGKey(1), 2)
    n, eps = 200_000, 0.1
    u_e, u_a = jax.random.uniform(k[0], (n,)), jax.random.uniform(k[1], (n,))
    for sign, expected in ((-1.0, eps / 2), (1.0, 1 - eps / 2)):
        act, _, prop = ib.epsilon_greedy(jnp.full(n, sign), u_e, u_a, eps)
        assert abs(float(np.mean(act)) - expected) < 0.005
        assert np.allclose(prop, expected)
    act, prop = ib.random_schedule(u_a, 0.3)
    assert abs(float(np.mean(act)) - 0.3) < 0.005 and np.allclose(prop, 0.3)


def _bandit(seed=0, dim=3):
    return ib.create_bandit_state(jax.random.PRNGKey(seed), dim, 16, 3e-3)


def test_fit_learns_different_actions_for_positive_and_negative_contexts():
    state = _bandit()
    assert float(np.abs(state.apply_fn(state.params, jnp.ones((4, 3)))).max()) == 0.0
    k = jax.random.split(jax.random.PRNGKey(2), 2)
    n = 256
    sign = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
    x = jnp.stack([sign, -sign, jnp.ones(n)], axis=-1) + 0.1 * jax.random.normal(
        k[0], (n, 3)
    )
    y = 2.0 * sign
    for i in range(60):
        state, _, steps = ib.fit_bandit(
            state, x, y, jnp.asarray(n), jax.random.fold_in(k[1], i)
        )
    assert int(steps) == n // ib.BATCH_SIZE
    q = state.apply_fn(state.params, jnp.array([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]]))
    assert float(q[0]) > 0.5 and float(q[1]) < -0.5


def test_padding_rows_are_never_labels():
    """Only the first `count` rows train: whatever the padding holds (a skipped
    decision's outcome is unobserved), the result is bit-identical."""
    k = jax.random.split(jax.random.PRNGKey(3), 2)
    x = jax.random.normal(k[0], (300, 3))
    y = jax.random.normal(k[1], (300,))
    count = jnp.asarray(130)
    garbage = (x.at[130:].set(50.0), y.at[130:].set(-100.0))
    a, loss_a, _ = ib.fit_bandit(_bandit(), x, y, count, jax.random.PRNGKey(4))
    b, loss_b, _ = ib.fit_bandit(_bandit(), *garbage, count, jax.random.PRNGKey(4))
    assert _tree_max_diff(a, b) == 0.0 and float(loss_a) == float(loss_b)


def test_no_labels_leave_the_bandit_exactly_unchanged():
    """Parameters, Adam moments and the step counter: Adam would move them on a
    zero gradient, so an empty update must not step at all."""
    state = _bandit()
    x, y = jnp.ones((300, 3)), jnp.ones(300)
    new, loss, steps = ib.fit_bandit(state, x, y, jnp.asarray(0), jax.random.PRNGKey(5))
    assert _tree_max_diff(state, new) == 0.0
    assert int(new.step) == int(state.step) == 0 and int(steps) == 0
    assert float(loss) == 0.0


def test_paired_target_and_the_zero_reward_penalty():
    gamma = 0.9
    main = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    fork = jnp.array([[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 2.0, 1.0], [3.0, 0.0, 0.0]])
    target, g_main, g_fork, zero = ib.paired_target(main, fork, gamma, 5.0, 0.0)
    assert np.allclose(g_main, [1.0, 0.0, 1.8, 0.0])
    assert np.allclose(g_fork, [0.0, -0.45, 1.8 + 0.81, 3.0])
    # An all-zero fork is penalized; a single nonzero step, even negative, is not;
    # a fork whose window ended after a nonzero step (zeros after) is not.
    assert np.array_equal(zero, [True, False, False, False])
    assert np.allclose(target, g_fork - g_main - 5.0 * np.array([1.0, 0.0, 0.0, 0.0]))
    # Penalty 0 is the unpenalized target, bit for bit.
    plain, *_ = ib.paired_target(main, fork, gamma, 0.0, 0.0)
    assert np.array_equal(np.asarray(plain), np.asarray(g_fork - g_main))
    costed, *_ = ib.paired_target(main, fork, gamma, 0.0, 0.25)
    assert np.allclose(costed, g_fork - g_main - 0.25)


def test_pow2_chunks():
    assert [len(c) for c in ac.pow2_chunks(np.arange(13))] == [8, 4, 1]
    assert [len(c) for c in ac.pow2_chunks(np.arange(16))] == [16]
    assert ac.pow2_chunks(np.arange(0)) == []
    assert np.array_equal(np.concatenate(ac.pow2_chunks(np.arange(7))), np.arange(7))


# ------------------------------------------------------------ inputs and labels


@pytest.mark.parametrize("redecide", ["none", "focal", "all"])
@pytest.mark.parametrize("manager_input", ["global", "relative", "local"])
def test_bandit_input_is_the_focal_agents_pre_teleport_actor_input(
    manager_input, redecide
):
    """Every label's input is exactly agent i's row of the actual main decision
    (the same manager call that acted), also when the fork re-decides goals."""
    config = _adaptive(
        "bandit", batch=8, intervention_bandit_epsilon=1.0, bound="tanh",
        manager_input=manager_input, intervention_redecide=redecide,
    )
    _, (_, rollout, _, _), _ = _run(config, AlwaysPlace(3))
    lab = _labels(rollout.adaptive)
    assert len(lab["y"]) > 4 and len(set(lab["focal"])) >= 2
    main_obs = np.asarray(rollout.manager.obs)
    fork_obs = np.asarray(rollout.intervention.manager.obs)
    expected = main_obs[lab["round"], lab["stream"], lab["focal"]]
    assert np.array_equal(lab["x"], expected)
    assert np.array_equal(
        np.asarray(rollout.intervention.focal)[lab["round"], lab["stream"]], lab["focal"]
    )
    fork_rows = fork_obs[lab["round"], lab["stream"], lab["focal"]]
    if redecide == "none":
        assert np.array_equal(fork_rows, expected)
    elif manager_input != "local":
        # The fork row is the post-teleport decision, which sees the recruits;
        # the bandit never reads it. (Under `local` the stub's observation holds
        # only the agent's own position, and the focal agent is never moved, so
        # its post-teleport input equals the pre-teleport one.)
        assert not np.array_equal(fork_rows, expected)


@pytest.mark.parametrize("redecide", ["none", "all"])
def test_labels_are_raw_paired_returns_of_the_matching_main_window(redecide):
    """The target reads the raw per-step task reward of the fork and of the main
    window it was cut from: no intrinsic reward and no bootstrapped value, even
    when a time limit lands inside the window (both levels bootstrap there)."""
    config = _adaptive(
        "bandit", batch=8, intervention_bandit_epsilon=1.0,
        intervention_redecide=redecide,
    )
    env = AlwaysPlace(3, max_steps=6)  # a time limit inside window 2
    _, (_, rollout, _, _), _ = _run(config, env)
    lab = _labels(rollout.adaptive)
    r, s = lab["round"], lab["stream"]
    main = np.asarray(rollout.worker.team_reward)[r, s]  # (n, H)
    fork = np.asarray(rollout.intervention.worker.team_reward)[r, s]
    m_gamma = config.worker.gamma
    expected, *_ = ib.paired_target(main, fork, m_gamma, 5.0, 0.0)
    assert np.allclose(lab["y"], expected, atol=1e-6)
    # The manager's own window rewards include the bootstraps, so they differ.
    m_main = np.asarray(rollout.manager.reward)[r, s]
    assert not np.allclose(m_main, ib.window_return(jnp.asarray(main), m_gamma))


def test_early_termination_zeroes_the_rest_of_the_window():
    config = _adaptive("bandit", batch=8, intervention_bandit_epsilon=1.0)
    _, (_, rollout, _, _), _ = _run(config, AlwaysPlace(3, terminate_at=6))
    reward = np.asarray(rollout.worker.team_reward)  # (rounds, streams, H)
    alive = np.asarray(rollout.worker.active_mask)[..., 0]
    assert (alive == 0).any()
    assert np.all(reward[alive == 0] == 0.0)
    lab = _labels(rollout.adaptive)
    main = reward[lab["round"], lab["stream"]]
    fork = np.asarray(rollout.intervention.worker.team_reward)[lab["round"], lab["stream"]]
    expected, *_ = ib.paired_target(main, fork, config.worker.gamma, 5.0, 0.0)
    assert np.allclose(lab["y"], expected, atol=1e-6)


def test_failed_placements_run_no_fork_and_are_labelled_zero(monkeypatch):
    calls = _record_dispatch(monkeypatch)
    config = _adaptive("bandit", batch=4, intervention_bandit_epsilon=1.0)
    _, (_, rollout, _, _), _ = _run(config, StubEnv(3, fail_teleport=True))
    ad = rollout.adaptive
    lab = _labels(ad)
    assert len(lab["y"]) > 0
    assert not lab["placed"].any() and np.all(lab["y"] == 0.0)
    assert int(ad.fork_executed.sum()) == 0
    assert int(ad.main_collected.sum()) == ac.window_slots(config)
    assert "place" in {name for name, _ in calls} and "fork" not in {n for n, _ in calls}


# --------------------------------------------------------------- the budget


def _record_dispatch(monkeypatch):
    """Record (name, lanes) of every host-dispatched collector call."""
    calls = []
    lane_arg = {"main_round": 3, "place": 2, "fork": 3}

    def hook(name, fn):
        def wrapped(*args):
            calls.append((name, int(args[lane_arg[name]].shape[0])))
            return fn(*args)

        return wrapped

    monkeypatch.setattr(ac, "DISPATCH_HOOK", hook)
    return calls


@pytest.mark.parametrize(
    "scheduler, kwargs, env",
    [
        ("random", dict(intervention_random_rate=0.0), AlwaysPlace(3)),
        ("random", dict(intervention_random_rate=1.0), AlwaysPlace(3)),
        ("bandit", dict(intervention_bandit_epsilon=1.0), StubEnv(3)),
        ("random", dict(intervention_random_rate=1.0), StubEnv(3, fail_teleport=True)),
    ],
    ids=["all_skip", "all_intervene", "mixed", "all_invalid"],
)
def test_every_update_collects_exactly_the_budget(scheduler, kwargs, env, monkeypatch):
    calls = _record_dispatch(monkeypatch)
    config = _adaptive(scheduler, batch=8, **kwargs)
    _, (_, rollout, _, _), _ = _run(config, env)
    ad, slots = rollout.adaptive, ac.window_slots(config)
    main, forks = int(ad.main_collected.sum()), int(ad.fork_executed.sum())
    assert main + forks == slots
    rate = kwargs.get("intervention_random_rate")
    if rate == 0.0 or isinstance(env, StubEnv) and env.fail_teleport:
        assert (main, forks) == (slots, 0)
        assert rollout.diagnostics["scheduler_rounds"] == ac.n_rounds(config)
    if rate == 1.0 and not getattr(env, "fail_teleport", False):
        assert (main, forks) == (slots // 2, slots // 2)
    # A fork only ever runs from a collected main window.
    assert not np.any(np.asarray(ad.fork_executed) & ~np.asarray(ad.main_collected))
    # Exactly the collected rows reach PPO.
    _, w_rows, _, m_rows, _ = ac.adaptive_batches(rollout, config, NO_MASK)
    assert int(m_rows.sum()) == slots and int(w_rows.sum()) == slots * HORIZON
    # What ran: no zero-width call, power-of-two widths, one fork lane per fork.
    widths = {name: [n for c, n in calls if c == name] for name in ("place", "fork")}
    assert all(n > 0 and n & (n - 1) == 0 for n in widths["place"] + widths["fork"])
    assert sum(widths["fork"]) == forks
    assert sum(widths["place"]) == int(rollout.diagnostics["intervention_attempted"])
    if rate == 0.0:
        assert widths == {"place": [], "fork": []}


def test_a_partial_final_round_finishes_exactly_at_the_budget():
    """With an odd number of forks the last round fits fewer windows than there
    are streams; streams then end with unequal numbers of main windows."""
    config = _adaptive("bandit", batch=4, intervention_bandit_epsilon=1.0)
    for seed in range(40):
        _, (_, rollout, _, _), _ = _run(config, AlwaysPlace(3), seed=seed)
        lengths = np.asarray(rollout.adaptive.main_collected).sum(0)
        if len(set(lengths)) > 1:
            break
    else:
        pytest.fail("no seed produced a partial final round")
    ad = rollout.adaptive
    assert int(ad.main_collected.sum()) + int(ad.fork_executed.sum()) == ac.window_slots(
        config
    )
    assert np.array_equal(np.asarray(ad.last_row), lengths - 1)


# ------------------------------------------------------------------- GAE


def _gae(rewards, values, dones, last, gamma, lam):
    adv, g = np.zeros_like(rewards), 0.0
    for t in reversed(range(len(rewards))):
        nxt = last if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * (1 - dones[t]) - values[t]
        g = delta + gamma * lam * (1 - dones[t]) * g
        adv[t] = g
    return adv


def _partial_round_rollout(redecide="none"):
    config = _adaptive(
        "bandit", batch=4, intervention_bandit_epsilon=1.0,
        intervention_redecide=redecide,
    )
    for seed in range(40):
        _, (_, rollout, _, _), fns = _run(config, AlwaysPlace(3, max_steps=7), seed=seed)
        if len(set(np.asarray(rollout.adaptive.main_collected).sum(0))) > 1:
            return config, rollout
    pytest.fail("no seed produced unequal stream lengths")


@pytest.mark.parametrize("redecide", ["none", "focal", "all"])
def test_gae_matches_independent_ragged_trajectories(redecide):
    """Main streams of unequal length, each bootstrapped from its own final
    state at its own last window; forks are one-window terminals; worker
    windows are terminal at the goal boundary."""
    config, rollout = _partial_round_rollout(redecide)
    ad = rollout.adaptive
    mcfg, wcfg = config.manager, config.worker
    w_batch, _, m_batch, _, _ = ac.adaptive_batches(rollout, config, NO_MASK)
    m_adv, _ = compute_gae(
        m_batch.reward, m_batch.value, m_batch.done.astype(jnp.float32),
        jnp.zeros(m_batch.reward.shape[1]), mcfg.gamma, mcfg.gae_lambda,
    )
    w_adv, _ = compute_gae(
        w_batch.reward, w_batch.value, w_batch.done.astype(jnp.float32),
        jnp.zeros(w_batch.reward.shape[1:]), wcfg.gamma, wcfg.gae_lambda,
    )
    m_adv, w_adv = np.asarray(m_adv), np.asarray(w_adv)
    collected = np.asarray(ad.main_collected)
    n_streams = collected.shape[1]
    main = rollout.manager
    for s in range(n_streams):
        rows = np.nonzero(collected[:, s])[0]
        assert np.array_equal(rows, np.arange(len(rows)))  # a contiguous prefix
        dones = np.asarray(main.done)[rows, s].astype(float)
        expected = _gae(
            np.asarray(main.reward)[rows, s], np.asarray(main.value)[rows, s], dones,
            float(ad.final_value[s]), mcfg.gamma, mcfg.gae_lambda,
        )
        assert np.allclose(m_adv[rows, s], expected, atol=1e-5)
    fork = rollout.intervention.manager
    executed = np.asarray(ad.fork_executed)
    r, s = np.nonzero(executed)
    assert len(r) > 0
    assert np.allclose(
        m_adv[r, n_streams + s],
        np.asarray(fork.reward)[r, s] - np.asarray(fork.value)[r, s],
        atol=1e-6,
    )
    # Worker: every main window is its own terminal episode.
    w = rollout.worker
    for rr, ss in zip(*np.nonzero(collected)):
        for agent in range(3):
            expected = _gae(
                np.asarray(w.reward)[rr, ss, :, agent],
                np.asarray(w.value)[rr, ss, :, agent],
                np.asarray(w.done)[rr, ss].astype(float),
                0.0, wcfg.gamma, wcfg.gae_lambda,
            )
            got = w_adv[rr * HORIZON : (rr + 1) * HORIZON, ss, agent]
            assert np.allclose(got, expected, atol=1e-5)


def test_no_advantage_crosses_between_main_and_fork_columns():
    config, rollout = _partial_round_rollout()
    mcfg = config.manager

    def m_adv(ro):
        _, _, batch, _, _ = ac.adaptive_batches(ro, config, NO_MASK)
        adv, _ = compute_gae(
            batch.reward, batch.value, batch.done.astype(jnp.float32),
            jnp.zeros(batch.reward.shape[1]), mcfg.gamma, mcfg.gae_lambda,
        )
        return np.asarray(adv)

    base = m_adv(rollout)
    fork = rollout.intervention.manager
    bumped = rollout._replace(
        intervention=rollout.intervention._replace(
            manager=fork._replace(reward=fork.reward + 100.0)
        )
    )
    n = rollout.manager.reward.shape[1]
    assert np.array_equal(m_adv(bumped)[:, :n], base[:, :n])
    main = rollout.manager
    bumped = rollout._replace(manager=main._replace(reward=main.reward + 100.0))
    assert np.array_equal(m_adv(bumped)[:, n:], base[:, n:])


# ------------------------------------------------------------- PPO parity


@pytest.mark.parametrize("per_agent", [False, True])
def test_prepared_samples_reproduce_ppo_update(per_agent):
    """Prepare -> flatten -> pack every row -> optimize equals `ppo_update`, bit for
    bit, with masked statistics (masked scalar critic rows / per-agent masks)."""
    T, E, N, D, G = 6, 4, 2, 5, 7
    k = jax.random.split(jax.random.PRNGKey(0), 10)
    cfg = MAPPOConfig(n_steps=T, n_envs=E, n_epochs=2, n_minibatches=2, hidden_dim=8)
    ts = create_train_state(
        k[0], cfg, D, G, 2, False, n_critic_outputs=N if per_agent else 1,
        keep_critic_output_axis=per_agent,
    )
    shape = (T, E, N) if per_agent else (T, E)
    traj = Transition(
        obs=jax.random.normal(k[1], (T, E, N, D)),
        global_state=jax.random.normal(k[2], (T, E, G)),
        action=jax.random.normal(k[3], (T, E, N, 2)),
        reward=jax.random.normal(k[4], shape),
        done=jax.random.uniform(k[5], (T, E)) < 0.3,
        log_prob=-jax.random.uniform(k[6], (T, E, N)),
        value=jax.random.normal(k[7], shape),
        team_reward=jnp.zeros((T, E)),
        active_mask=(jax.random.uniform(k[8], (T, E, N)) > 0.3).astype(jnp.float32),
        action_mask=NO_MASK,
    )
    last = jnp.zeros(shape[1:])
    a_ts, a_losses = ppo_update(ts, k[9], traj, last, cfg, False, masked_statistics=True)
    adv, ret, ev = prepare_advantages(traj, last, cfg, masked_statistics=True)
    samples = ac.pack(flatten_samples(traj, adv, ret), jnp.ones((T, E), bool), T * E)
    b_ts, b_losses = optimize_samples(ts, k[9], samples, cfg, False, masked_statistics=True)
    b_losses["explained_variance"] = ev
    assert _tree_max_diff(a_ts, b_ts) == 0.0
    assert all(float(a_losses[key]) == float(b_losses[key]) for key in a_losses)


# ---------------------------------------------------------------- end to end


@pytest.mark.parametrize("n_agents", [2, 3])
@pytest.mark.parametrize("scheduler", ["bandit", "random"])
def test_collect_update_eval_end_to_end(scheduler, n_agents):
    config = _adaptive(scheduler, batch=4, intervention_bandit_epsilon=0.5)
    if scheduler == "random":
        config = dataclasses.replace(config, intervention_bandit_epsilon=0.1)
    rs, (rs1, rollout, last, stats), (collect_fn, update_fn, eval_fn) = _run(
        config, AlwaysPlace(n_agents)
    )
    rs2, losses = update_fn(rs1, rollout, last)
    # The greedy/explored means and the label explained variance are NaN when
    # undefined (no fork of that kind, fewer than two labels).
    assert all(np.isfinite(float(v)) for k, v in losses.items() if k not in NAN_OK)
    assert np.isfinite(float(eval_fn(rs2.train_state, jax.random.PRNGKey(1))))
    _, rollout2, last2, _ = collect_fn(rs2)
    _, losses2 = update_fn(rs2, rollout2, last2)
    assert set(losses2) == set(losses)  # one key set per run
    assert (rs2.train_state.bandit is None) == (scheduler == "random")


def test_collection_reproduces_from_the_same_runner_state():
    config = _adaptive("bandit", batch=4, intervention_bandit_epsilon=0.5)
    init_fn, collect_fn, _, _, _ = make_train(config, AlwaysPlace(3))
    rs = init_fn(jax.random.PRNGKey(0))
    _, a, _, _ = collect_fn(rs)
    _, b, _, _ = collect_fn(rs)
    assert _tree_max_diff(
        (a.worker, a.manager, a.intervention, a.adaptive),
        (b.worker, b.manager, b.intervention, b.adaptive),
    ) == 0.0


def test_the_bandit_trains_only_on_labels():
    """With labels the bandit moves; with none (epsilon 0: a zero-initialized
    bandit predicts exactly 0 and ties skip) it is bit-identical."""
    for eps, moves in ((1.0, True), (0.0, False)):
        config = _adaptive("bandit", batch=4, intervention_bandit_epsilon=eps)
        rs, (rs1, rollout, last, _), (_, update_fn, _) = _run(config, AlwaysPlace(3))
        assert (int(rollout.adaptive.label_count) > 0) == moves
        rs2, losses = update_fn(rs1, rollout, last)
        moved = _tree_max_diff(rs.train_state.bandit, rs2.train_state.bandit) > 0
        assert moved == moves
        assert (float(losses["bandit_steps"]) > 0) == moves


def _collect_with_bias(config, env, bias=None, seed=0):
    """One collection; with `bias`, the bandit's output bias is set so that it
    predicts exactly `bias` everywhere (its output kernel starts at zero)."""
    init_fn, collect_fn, _, _, _ = make_train(config, env)
    rs = init_fn(jax.random.PRNGKey(seed))
    if bias is not None:
        def set_bias(path, leaf):
            names = [getattr(k, "key", None) for k in path]
            return jnp.full_like(leaf, bias) if "Dense_2" in names and "bias" in names else leaf

        bandit = rs.train_state.bandit
        bandit = bandit.replace(
            params=jax.tree_util.tree_map_with_path(set_bias, bandit.params)
        )
        rs = rs._replace(train_state=rs.train_state._replace(bandit=bandit))
    _, rollout, _, _ = collect_fn(rs)
    return rollout


def test_explored_forks_are_the_coins_and_greedy_forks_the_bandits():
    """Epsilon 1: every fork comes from the coin. Epsilon 0 with q = +1
    everywhere: every fork is the bandit's own choice. The random scheduler's
    forks are all random picks."""
    env = AlwaysPlace(3)
    d = _collect_with_bias(
        _adaptive("bandit", batch=8, intervention_bandit_epsilon=1.0), env
    ).diagnostics
    assert d["scheduler_executed"] > 0 and d["bandit_greedy_forks"] == 0
    assert d["bandit_explored_forks"] == d["scheduler_executed"]
    assert d["bandit_greedy_fork_frac"] == 0.0 and np.isnan(d["bandit_gain_greedy"])
    assert np.isclose(d["bandit_gain_explored"], d["bandit_gain"])
    assert np.isclose(d["bandit_target_explored"], d["bandit_target"])

    d = _collect_with_bias(
        _adaptive("bandit", batch=8, intervention_bandit_epsilon=0.0), env, bias=1.0
    ).diagnostics
    assert d["bandit_predicted_utility"] == 1.0
    assert d["scheduler_executed"] > 0 and d["bandit_explored_forks"] == 0
    assert d["bandit_greedy_fork_frac"] == 1.0 and np.isnan(d["bandit_gain_explored"])
    assert np.isclose(d["bandit_gain_greedy"], d["bandit_gain"])

    d = _collect_with_bias(
        _adaptive("random", batch=8, intervention_random_rate=0.5), env
    ).diagnostics
    assert d["bandit_greedy_forks"] == 0
    assert d["bandit_explored_forks"] == d["scheduler_executed"] > 0


def test_the_greedy_explored_split_recombines_into_the_overall_mean():
    config = _adaptive("bandit", batch=8, intervention_bandit_epsilon=0.6)
    d = _collect_with_bias(config, AlwaysPlace(3), bias=1.0).diagnostics
    frac = d["bandit_greedy_fork_frac"]
    assert 0.0 < frac < 1.0
    for name in ("gain", "target"):
        mixed = frac * d[f"bandit_{name}_greedy"] + (1 - frac) * d[f"bandit_{name}_explored"]
        assert np.isclose(mixed, d[f"bandit_{name}"])


def test_bandit_explained_variance_scores_pre_update_predictions():
    """The predictions stored with the labels are the ones made at the decision,
    before this update's fit, so the score is held out; NaN without labels."""
    config = _adaptive("bandit", batch=8, intervention_bandit_epsilon=0.6)
    rollout = _collect_with_bias(config, AlwaysPlace(3), bias=1.0)
    ad, d = rollout.adaptive, rollout.diagnostics
    n = int(ad.label_count)
    y, q = np.asarray(ad.label_y)[:n], np.asarray(ad.label_q)[:n]
    assert n >= 2 and np.all(q == 1.0)
    assert np.isclose(d["bandit_explained_variance"], 1 - np.var(y - q) / np.var(y))
    # A constant prediction at the labels' mean scores exactly 0.
    assert np.isclose(ac.label_explained_variance(y, np.full(n, y.mean())), 0.0)
    assert np.isnan(ac.label_explained_variance(y[:1], q[:1]))
    # Epsilon 0 and a zero-initialized bandit: no fork, no label, NaN.
    none = _collect_with_bias(
        _adaptive("bandit", batch=8, intervention_bandit_epsilon=0.0), AlwaysPlace(3)
    )
    assert int(none.adaptive.label_count) == 0
    assert np.isnan(none.diagnostics["bandit_explained_variance"])


def test_evaluation_never_reads_the_bandit():
    config = _adaptive("bandit", batch=4)
    rs, _, (_, _, eval_fn) = _run(config, AlwaysPlace(3))
    ts = rs.train_state
    shifted = ts._replace(
        bandit=ts.bandit.replace(params=jax.tree.map(lambda p: p + 1.0, ts.bandit.params))
    )
    key = jax.random.PRNGKey(3)
    assert float(eval_fn(ts, key)) == float(eval_fn(shifted, key))


# ------------------------------------------------------------- checkpoints


def _runner(config):
    from algorithms.simplified_feudal_mappo_jax.run import (
        Simplified_Feudal_MAPPO_JAX_Runner as Runner,
    )

    runner = Runner.__new__(Runner)
    runner.feudal_config = config
    return runner


def test_adaptive_checkpoint_resumes_exactly(tmp_path):
    """Save after one update, load into a fresh init, and the next collect and
    update (decisions, samples, parameters, losses) are bit-identical."""
    config = _adaptive("bandit", batch=4, intervention_bandit_epsilon=0.5)
    init_fn, collect_fn, update_fn, _, _ = make_train(config, AlwaysPlace(3))
    rs = init_fn(jax.random.PRNGKey(0))
    rs, rollout, last, _ = collect_fn(rs)
    rs, _ = update_fn(rs, rollout, last)
    runner, path, eval_rng = _runner(config), tmp_path / "ckpt.msgpack", jax.random.PRNGKey(7)
    runner._save_train_checkpoint(rs, eval_rng, path)
    assert "bandit_ts" in runner._full_checkpoint_tree(rs, eval_rng)
    restored, _ = runner._load_train_checkpoint(init_fn(jax.random.PRNGKey(99)), eval_rng, path)

    def step(state):
        state, ro, lv, _ = collect_fn(state)
        state, losses = update_fn(state, ro, lv)
        return ro, state, losses

    ro_a, state_a, losses_a = step(rs)
    ro_b, state_b, losses_b = step(restored)
    assert _tree_max_diff(
        (ro_a.worker, ro_a.manager, ro_a.adaptive), (ro_b.worker, ro_b.manager, ro_b.adaptive)
    ) == 0.0
    assert _tree_max_diff(state_a, state_b) == 0.0
    assert all(float(losses_a[k]) == float(losses_b[k]) for k in losses_a)


def test_the_scheduler_record_refuses_other_settings(tmp_path):
    config = _adaptive("bandit", batch=4)
    init_fn = make_train(config, AlwaysPlace(3))[0]
    rs, eval_rng = init_fn(jax.random.PRNGKey(0)), jax.random.PRNGKey(7)
    path = tmp_path / "bandit.msgpack"
    _runner(config)._save_train_checkpoint(rs, eval_rng, path)
    other = dataclasses.replace(config, intervention_bandit_epsilon=0.3)
    with pytest.raises(ValueError, match="other fork-scheduler settings"):
        _runner(other)._load_train_checkpoint(rs, eval_rng, path)
    fixed = _config(interventions=True)
    fixed_rs = make_train(fixed, AlwaysPlace(3))[0](jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="adaptive fork-scheduler run"):
        _runner(fixed)._load_train_checkpoint(fixed_rs, eval_rng, path)
    fixed_path = tmp_path / "fixed.msgpack"
    _runner(fixed)._save_train_checkpoint(fixed_rs, eval_rng, fixed_path)
    with pytest.raises(ValueError, match="no fork-scheduler record"):
        _runner(config)._load_train_checkpoint(rs, eval_rng, fixed_path)


def test_fixed_checkpoints_carry_no_scheduler_entries():
    from algorithms.simplified_feudal_mappo_jax.run import (
        Simplified_Feudal_MAPPO_JAX_Runner as Runner,
    )

    fixed = _config(interventions=True)
    rs = make_train(fixed, AlwaysPlace(3))[0](jax.random.PRNGKey(0))
    tree = _runner(fixed)._full_checkpoint_tree(rs, jax.random.PRNGKey(7))
    assert "bandit_ts" not in tree and "scheduler_meta" not in tree
    assert "bandit" not in Runner._params_tree(rs.train_state)
    assert to_bytes(tree) == to_bytes(Runner._checkpoint_tree(rs, jax.random.PRNGKey(7)))


def test_feudal_config_positional_construction_still_works():
    worker = MAPPOConfig(n_steps=4, n_envs=2)
    config = FeudalConfig(worker, worker, goal_horizon=2, waypoint_radius=0.3)
    assert config.intervention_scheduler == "fixed"
