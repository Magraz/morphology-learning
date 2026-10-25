"""Seam tests for the anytime fork scheduler (`intervention_scheduler: anytime |
anytime_random`): `anytime_scheduler.py`, `anytime_collection.py` and their
update.

Run on the CPU stub env of `test_simplified_feudal.py` (deterministic, so
bit-for-bit claims can be checked). Plan:
`/home/magraz/.claude/plans/i-want-to-create-breezy-wind.md`.

    uv run pytest algorithms/tests/test_anytime_scheduler.py -q
"""

import dataclasses
import itertools
import math
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_intervention_bandit import AlwaysPlace, _runner
from test_simplified_feudal import (
    HORIZON,
    N_STEPS,
    StubEnv,
    _config,
    _params,
    _tree_max_diff,
)

from algorithms.mappo_jax.mappo import compute_gae
from algorithms.mappo_jax.network import evaluate_action
from algorithms.simplified_feudal_mappo_jax import adaptive_collection as ac
from algorithms.simplified_feudal_mappo_jax import anytime_collection as anc
from algorithms.simplified_feudal_mappo_jax import anytime_scheduler as asch
from algorithms.simplified_feudal_mappo_jax import intervention_bandit as ib
from algorithms.simplified_feudal_mappo_jax import interventions as iv
from algorithms.simplified_feudal_mappo_jax.trainer import make_train
from algorithms.simplified_feudal_mappo_jax.types import Model_Params

N_WINDOWS = N_STEPS // HORIZON  # 3
SLOTS = 2 * N_WINDOWS  # window slots per stream
NO_MASK = jnp.zeros(())
NEVER, ALWAYS = 50.0, -50.0  # skip-logit biases: never / always launch


@pytest.fixture(autouse=True)
def _run_on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


_SCHEDULER_KEYS = (
    "intervention_fork_budget", "intervention_scheduler_gamma",
    "intervention_scheduler_lambda", "intervention_scheduler_ent_coef",
    "intervention_scheduler_init_forks", "intervention_bandit_hidden_dim",
    "intervention_bandit_lr",
)


def _anytime(scheduler="anytime", batch=4, **kwargs):
    """An anytime-scheduler config on the stub: `batch` envs, half of them main
    streams (what `make_feudal_config` builds from an env group)."""
    sched = {k: kwargs.pop(k) for k in list(kwargs) if k in _SCHEDULER_KEYS}
    kwargs.setdefault("intervention_redecide", "all")
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


def _launch_bias(n_agents, p):
    """The skip bias at which a decision launches with probability `p` (every
    other choice logit is 0 at initialization)."""
    return float(np.log(n_agents * (1 - p) / p))


def _edit(rs, level, fn):
    """Apply `fn(path_string, leaf)` to the params of `bandit` or the manager
    critic."""
    ts = rs.train_state
    if level == "bandit":
        b = ts.bandit
        params = jax.tree_util.tree_map_with_path(
            lambda p, x: fn(jax.tree_util.keystr(p), x), b.params
        )
        return rs._replace(train_state=ts._replace(bandit=b.replace(params=params)))
    critic = ts.manager.critic_ts
    params = jax.tree_util.tree_map_with_path(
        lambda p, x: fn(jax.tree_util.keystr(p), x), critic.params
    )
    manager = ts.manager._replace(critic_ts=critic.replace(params=params))
    return rs._replace(train_state=ts._replace(manager=manager))


def _prepare(rs, bias=None, longest=False, critic=None):
    if bias is not None:
        rs = _edit(rs, "bandit", lambda s, x: (
            jnp.full_like(x, bias) if "'skip'" in s and "bias" in s else x
        ))
    if longest:  # the longest allowed fork, almost surely
        rs = _edit(rs, "bandit", lambda s, x: (
            50.0 * jnp.arange(x.shape[-1], dtype=x.dtype)
            if "'duration'" in s and "bias" in s else x
        ))
    if critic is not None:  # the manager critic outputs `critic` everywhere
        rs = _edit(rs, "critic", lambda s, x: (
            (jnp.full_like(x, critic) if "bias" in s else jnp.zeros_like(x))
            if "Dense_2" in s else x
        ))
    return rs


def _collect(config, env, seed=0, **prepare):
    init_fn, collect_fn, update_fn, eval_fn, _ = make_train(config, env)
    rs = _prepare(init_fn(jax.random.PRNGKey(seed)), **prepare)
    return rs, collect_fn(rs), (collect_fn, update_fn, eval_fn)


def _launches(rollout):
    """Every launch of the rollout, per stream in time order, with its fork's
    first row (forks take their stream's fork rows in launch order; every
    placement succeeds on `AlwaysPlace`)."""
    s = rollout.scheduler
    decided, choice = np.asarray(s.weight) > 0, np.asarray(s.choice)
    out, j0 = [], {}
    for r, e, k in sorted(zip(*np.nonzero(decided & (choice > 0))), key=lambda c: (c[1], c[0], c[2])):
        n = int(np.asarray(s.duration)[r, e, k]) + 1
        out.append({
            "row": int(r), "stream": int(e), "step": int(k), "focal": int(choice[r, e, k] - 1),
            "recruit": np.asarray(s.recruit)[r, e, k], "k": n,
            "kmax": int(np.asarray(s.kmax)[r, e, k]), "glob": np.asarray(s.glob)[r, e, k],
            "j0": j0.get(int(e), 0),
        })
        j0[int(e)] = j0.get(int(e), 0) + n
    return out


def _record_dispatch(monkeypatch):
    """Record (name, lanes) of every host-dispatched collector call."""
    calls = []
    lane_arg = {"window": 4, "place": 2}

    def hook(name, fn):
        def wrapped(*args):
            calls.append((name, int(args[lane_arg[name]].shape[0])))
            return fn(*args)

        return wrapped

    monkeypatch.setattr(anc, "DISPATCH_HOOK", hook)
    return calls


# ---------------------------------------------------------------------- config


def test_make_feudal_config_wires_the_anytime_keys():
    """At the production rollout (33 windows of 32) half of the batch's 32 envs
    are main streams and an update is the parent's 33,792 steps."""
    from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config

    mp = Model_Params(
        hidden_dim=8, goal_horizon=32, waypoint_radius=0.15, interventions=True,
        intervention_redecide="all", intervention_scheduler="anytime",
        intervention_scheduler_gamma=0.9, intervention_scheduler_init_forks=3.0,
    )
    config = make_feudal_config(_params(), mp, n_envs=32)
    assert (config.intervention_batch_envs, config.worker.n_envs) == (32, 16)
    assert iv.anytime(config) and iv.streams(config) and not iv.adaptive(config)
    assert iv.simulator_steps_per_update(config) == 33_792 == ac.window_slots(config) * 32
    assert (asch.fork_budget(config), asch.stream_slots(config)) == (33, 66)
    assert config.intervention_scheduler_gamma == 0.9
    assert config.intervention_scheduler_init_forks == 3.0
    budget = dataclasses.replace(mp, intervention_fork_budget=10)
    assert asch.fork_budget(make_feudal_config(_params(), budget, n_envs=32)) == 10
    # The scheduler codes of existing checkpoints are unchanged.
    assert ib.SCHEDULERS[:4] == ("fixed", "bandit", "random", "heuristic")


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(intervention_redecide="none"), "intervention_redecide must be 'all'"),
        (dict(intervention_fork_budget=0), "intervention_fork_budget"),
        (dict(intervention_fork_budget=N_WINDOWS + 1), "intervention_fork_budget"),
        (dict(intervention_scheduler_gamma=1.5), "intervention_scheduler_gamma"),
        (dict(intervention_scheduler_lambda=-0.1), "intervention_scheduler_lambda"),
        (dict(intervention_scheduler_ent_coef=-1.0), "ent_coef"),
        (dict(intervention_scheduler_init_forks=0.0), "init_forks"),
        (dict(intervention_scheduler_init_forks=2.0 * N_STEPS), "init_forks"),
        # The bandit's knobs would silently do nothing.
        (dict(intervention_bandit_epsilon=0.3), "has no effect under"),
        (dict(intervention_bandit_zero_reward_penalty=0.0), "has no effect under"),
        (dict(intervention_random_rate=0.3), "has no effect under"),
        (dict(interventions=False), "has no effect without"),
        (dict(intervention_batch_envs=6), "batch_envs / 2"),
        (dict(manager_credit="counterfactual"), "manager_credit='team'"),
    ],
)
def test_bad_anytime_configs_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_train(dataclasses.replace(_anytime(), **overrides), AlwaysPlace(3))


def test_learning_knobs_are_rejected_where_they_cannot_act():
    for name, value in (
        ("intervention_scheduler_gamma", 0.5), ("intervention_scheduler_lambda", 0.5),
        ("intervention_scheduler_ent_coef", 0.1), ("intervention_bandit_lr", 1e-3),
    ):
        with pytest.raises(ValueError, match="has no effect under"):
            make_train(
                dataclasses.replace(_anytime("anytime_random"), **{name: value}),
                AlwaysPlace(3),
            )
    for name, value in (
        ("intervention_fork_budget", 2), ("intervention_scheduler_init_forks", 3.0),
    ):
        with pytest.raises(ValueError, match="has no effect under"):
            make_train(
                dataclasses.replace(_config(interventions=True), **{name: value}),
                AlwaysPlace(3),
            )


def test_the_manager_critic_reads_no_fork_context():
    """Every anytime fork row bootstraps to the continuing return, so the
    manager critic is the parent's width; the fixed forks add the context."""

    def width(config):
        rs = make_train(config, AlwaysPlace(3))[0](jax.random.PRNGKey(0))
        return rs.train_state.manager.critic_ts.params["params"]["Dense_0"]["kernel"].shape[0]

    assert iv.context_dim(_anytime(), 3) == 0
    assert width(_anytime()) == width(_config()) == width(
        _config(interventions=True, intervention_redecide="all")
    ) - (3 + 2)


# ------------------------------------------------------------- the policy


def _heads(key, n=4, b=5):
    k = jax.random.split(key, 4)
    return asch.Heads(
        skip=jax.random.normal(k[0], ()), focal=jax.random.normal(k[1], (n,)),
        recruit=jax.random.normal(k[2], (n, n)),
        duration=jax.random.normal(k[3], (n, b)), value=jnp.zeros(()),
    )


def _exact_recruit_logp(logits, focal, subset):
    """log P(subset | at least one recruit), by enumerating every subset."""
    mates = [j for j in range(len(logits)) if j != focal]
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64)))

    def raw(s):
        return np.prod([p[j] if j in s else 1.0 - p[j] for j in mates])

    z = sum(
        raw(set(c)) for r in range(1, len(mates) + 1) for c in itertools.combinations(mates, r)
    )
    return np.log(raw(set(subset)) / z)


def test_recruits_follow_the_conditioned_product_and_log_probs_are_exact():
    n, b = 4, 5
    h = _heads(jax.random.PRNGKey(0), n, b)
    kmax = jnp.int32(b)
    keys = jax.random.split(jax.random.PRNGKey(1), 200_000)
    actions, lps = jax.vmap(lambda k: asch.sample(h, kmax, k))(keys)
    choice, recruit = np.asarray(actions.choice), np.asarray(actions.recruit)
    duration, lps = np.asarray(actions.duration), np.asarray(lps)
    choice_lp = np.asarray(jax.nn.log_softmax(jnp.concatenate([h.skip[None], h.focal])))
    for focal in range(n):
        sel = choice == focal + 1
        assert sel.sum() > 2000
        assert not recruit[sel, focal].any() and recruit[sel].any(axis=1).all()
        row = np.asarray(h.recruit)[focal]
        for subset, count in Counter(map(tuple, recruit[sel])).items():
            members = [j for j in range(n) if subset[j]]
            exact = np.exp(_exact_recruit_logp(row, focal, members))
            se = math.sqrt(exact * (1 - exact) / sel.sum())
            assert abs(count / sel.sum() - exact) < 5 * se + 1e-3
    # The stored log-prob is the decision's whole log probability.
    for j in np.nonzero(choice > 0)[0][:40]:
        focal = choice[j] - 1
        members = [i for i in range(n) if recruit[j, i]]
        dur_lp = np.asarray(jax.nn.log_softmax(np.asarray(h.duration)[focal]))[duration[j]]
        expected = (
            choice_lp[choice[j]]
            + _exact_recruit_logp(np.asarray(h.recruit)[focal], focal, members)
            + dur_lp
        )
        assert np.isclose(lps[j], expected, atol=1e-5)
    skips = np.nonzero(choice == 0)[0][:10]
    assert np.allclose(lps[skips], choice_lp[0], atol=1e-6)
    # The batched evaluation (the update's) agrees with the per-sample one.
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (64,) + x.shape), h)
    first = jax.tree.map(lambda x: x[:64], actions)
    batched = asch.log_prob(batch, first, jnp.full(64, b, jnp.int32))
    assert np.allclose(batched, lps[:64], atol=1e-5)


@pytest.mark.parametrize("kmax", [1, 3, 6])
def test_lengths_respect_kmax_and_the_entropy_is_exact(kmax):
    h = _heads(jax.random.PRNGKey(2), n=3, b=6)
    keys = jax.random.split(jax.random.PRNGKey(kmax), 100_000)
    actions, lps = jax.vmap(lambda k: asch.sample(h, jnp.int32(kmax), k))(keys)
    assert (np.asarray(actions.duration) < kmax).all()
    monte_carlo = -float(np.mean(np.asarray(lps)))
    assert abs(float(asch.entropy(h, jnp.int32(kmax))) - monte_carlo) < 0.02


def test_the_initial_policy_is_uniform_with_the_configured_launch_rate():
    config = _anytime(intervention_scheduler_init_forks=3.0)
    state = make_train(config, AlwaysPlace(3))[0](jax.random.PRNGKey(0)).train_state.bandit
    actor_dim = state.params["params"]["encode_0"]["kernel"].shape[0] - asch.N_GLOBAL
    k = jax.random.split(jax.random.PRNGKey(1), 3)
    h = state.apply_fn(
        state.params, jax.random.normal(k[0], (3, actor_dim)),
        jax.random.normal(k[1], (3, 3, 2)), jax.random.uniform(k[2], (asch.N_GLOBAL,)),
    )
    pi = np.asarray(jax.nn.softmax(jnp.concatenate([h.skip[None], h.focal])))
    assert np.isclose(1 - pi[0], 3.0 / (2 * N_STEPS), rtol=1e-5)
    assert np.allclose(pi[1:], pi[1])
    assert float(np.abs(h.recruit).max()) == 0.0 and float(np.abs(h.duration).max()) == 0.0
    assert float(h.value) == 0.0


# ------------------------------------------------------------- the reward


def _reference_side(reward, active, ended, value_pre, value_after, start, length, gamma):
    """Step by step: discounted reward while the episode lasts, then the value
    of the state `length` steps on if the episode is still running there."""
    horizon = active.shape[1]
    reward, active = reward.reshape(-1), active.reshape(-1)
    live, g = True, 0.0
    for j in range(length + 1):
        q = start + j
        if j > 0:
            if q % horizon == 0:
                live = live and not ended[q // horizon - 1]
            else:
                live = live and bool(active[q])
        if j < length and live:
            g += gamma**j * reward[q]
    q = start + length
    value = value_after if q == reward.size else value_pre.reshape(-1)[q]
    return g, gamma**length * live * value


def test_span_return_matches_a_step_by_step_reference():
    rng = np.random.default_rng(0)
    for _ in range(300):
        windows, horizon = rng.integers(1, 5), rng.integers(1, 5)
        active = np.ones((windows, horizon), bool)
        ended = rng.random(windows) < 0.4
        for w in np.nonzero(ended)[0]:  # an episode ends mid-window or on its last step
            active[w, rng.integers(1, horizon + 1):] = False
        reward = rng.normal(size=(windows, horizon)) * active
        value_pre = rng.normal(size=(windows, horizon))
        value_after = rng.normal()
        start = int(rng.integers(0, windows * horizon))
        if not active.reshape(-1)[start]:
            continue
        length = int(rng.integers(1, windows * horizon - start + 1))
        cont = asch.continuation(active, ended)
        value = np.concatenate([value_pre.reshape(-1), [value_after]])
        got = asch.span_return(reward.reshape(-1), cont, value, 0.9, start, length)
        want = _reference_side(reward, active, ended, value_pre, value_after, start, length, 0.9)
        assert np.allclose(got, want, atol=1e-9)


def test_launch_rewards_are_the_paired_gain_over_the_span():
    """With a constant manager critic the value terms are c on both live sides.
    The stub's episodes end only at its time limit, at the same step on both
    sides, so the gain is the discounted reward difference over the span, cut
    at the time limit (main's post-reset steps are excluded) and at the end of
    the stream's rollout."""
    c, max_steps = 3.0, 7
    config = _anytime(batch=4)
    gamma = config.worker.gamma
    _, (_, rollout, _, _), _ = _collect(
        config, AlwaysPlace(3, max_steps=max_steps), bias=_launch_bias(3, 0.3), critic=c
    )
    launches = _launches(rollout)
    assert len(launches) >= 3 and any(x["step"] > 0 for x in launches)
    main_r = np.asarray(rollout.worker.team_reward)
    fork_r = np.asarray(rollout.intervention.worker.team_reward)
    rows = np.asarray(rollout.adaptive.last_row) + 1
    reward = np.asarray(rollout.scheduler.reward)
    gains = []
    for x in launches:
        e = x["stream"]
        t0 = int(round(x["glob"][2] * max_steps))
        start = x["row"] * HORIZON + x["step"]
        span = min(x["k"] * HORIZON, rows[e] * HORIZON - start)
        m = main_r[: rows[e], e].reshape(-1)[start : start + span]
        f = fork_r[x["j0"] : x["j0"] + x["k"], e].reshape(-1)[:span]
        live = np.array([t0 + j < max_steps for j in range(span + 1)], float)
        disc = gamma ** np.arange(span + 1)
        gain = float((disc[:span] * live[:span] * (f - m)).sum())  # values cancel
        assert np.isclose(reward[x["row"], e, x["step"]], gain, atol=1e-5)
        gains.append(gain)
    # Skips earn nothing.
    s = rollout.scheduler
    skip = (np.asarray(s.weight) > 0) & (np.asarray(s.choice) == 0)
    assert np.all(reward[skip] == 0.0)
    assert np.isclose(rollout.diagnostics["anytime_gain"], np.mean(gains), atol=1e-5)


# -------------------------------------------------------------- collection


@pytest.mark.parametrize(
    "bias, env",
    [
        (NEVER, AlwaysPlace(3)),
        (ALWAYS, AlwaysPlace(3)),
        (_launch_bias(3, 0.3), AlwaysPlace(3)),
        (ALWAYS, StubEnv(3, fail_teleport=True)),
    ],
    ids=["never", "always", "mixed", "all_invalid"],
)
def test_every_update_collects_exactly_the_budget(bias, env, monkeypatch):
    calls = _record_dispatch(monkeypatch)
    config = _anytime(batch=8)
    _, (_, rollout, _, _), _ = _collect(config, env, bias=bias)
    ad, d = rollout.adaptive, rollout.diagnostics
    main, fork = np.asarray(ad.main_collected), np.asarray(ad.fork_executed)
    assert main.sum() + fork.sum() == ac.window_slots(config) == 4 * SLOTS
    for e in range(4):
        assert main[:, e].sum() + fork[:, e].sum() == SLOTS
        assert fork[:, e].sum() <= N_WINDOWS  # the budget, B = n_windows
        for col in (main[:, e], fork[:, e]):  # contiguous prefixes
            assert np.array_equal(np.nonzero(col)[0], np.arange(col.sum()))
    s = rollout.scheduler
    launched = (np.asarray(s.weight) > 0) & (np.asarray(s.choice) > 0)
    assert launched.sum(axis=2).max() <= 1  # one launch per stream and window
    assert np.all((np.asarray(s.duration) + 1)[launched] <= np.asarray(s.kmax)[launched])
    # Exactly the collected rows reach PPO.
    _, w_rows, _, m_rows, _ = ac.adaptive_batches(rollout, config, NO_MASK)
    assert int(m_rows.sum()) == 4 * SLOTS and int(w_rows.sum()) == 4 * SLOTS * HORIZON
    widths = {name: [n for c, n in calls if c == name] for name in ("window", "place")}
    assert all(n > 0 and n & (n - 1) == 0 for n in widths["window"] + widths["place"])
    assert sum(widths["place"]) == launched.sum() == d["scheduler_launched"]
    if bias == NEVER:
        assert fork.sum() == 0 and widths["place"] == [] and launched.sum() == 0
    elif isinstance(env, AlwaysPlace) and bias == ALWAYS:
        assert fork.sum() == 4 * N_WINDOWS  # the whole budget, every stream
    elif not isinstance(env, AlwaysPlace):
        assert fork.sum() == 0 and launched.sum() > 0 and d["intervention_valid_frac"] == 0


def test_padded_rounds_collect_the_same_rollout(monkeypatch):
    """One padded dispatch per round (the accelerator path) against exact
    power-of-two chunks (the CPU path): the same rollout, since every lane is a
    batch of one with its own keys and padding lanes write nowhere."""
    config = _anytime(batch=8)
    bias = _launch_bias(3, 0.3)
    _, (_, a, _, _), _ = _collect(config, AlwaysPlace(3), bias=bias)
    calls = _record_dispatch(monkeypatch)
    monkeypatch.setattr(anc, "PAD_LANES", True)
    _, (_, b, _, _), _ = _collect(config, AlwaysPlace(3), bias=bias)
    windows = [n for name, n in calls if name == "window"]
    assert len(windows) == b.diagnostics["scheduler_rounds"]  # one per round
    assert all(n & (n - 1) == 0 for n in windows) and max(windows) > 4
    assert int(b.adaptive.fork_executed.sum()) > 0
    fields = (a.worker, a.manager, a.intervention, a.adaptive, a.scheduler)
    assert _tree_max_diff(fields, (b.worker, b.manager, b.intervention, b.adaptive, b.scheduler)) < 1e-5
    assert all(
        np.isclose(a.diagnostics[k], b.diagnostics[k], atol=1e-5, equal_nan=True)
        for k in a.diagnostics
    )


def test_main_windows_do_not_depend_on_the_forks():
    config = _anytime(batch=4)
    _, (_, a, _, _), _ = _collect(config, AlwaysPlace(3), bias=NEVER)
    _, (_, b, _, _), _ = _collect(config, AlwaysPlace(3), bias=ALWAYS)
    assert int(b.adaptive.fork_executed.sum()) > 0
    rows = np.minimum(np.asarray(a.adaptive.last_row), np.asarray(b.adaptive.last_row)) + 1
    for e, n in enumerate(rows):
        assert _tree_max_diff(
            jax.tree.map(lambda x: x[:n, e], (a.worker, a.manager)),
            jax.tree.map(lambda x: x[:n, e], (b.worker, b.manager)),
        ) == 0.0


def test_a_fork_starts_from_its_launch_state_with_exactly_the_chosen_recruits():
    """The stub's observation starts with each agent's position: the fork's
    first state is the main window's pre-step state at the launch step, with
    exactly the chosen recruits moved `intervention_radius` from the focal
    agent."""
    config = _anytime(batch=4)
    for seed in range(20):
        _, (_, rollout, _, _), _ = _collect(
            config, AlwaysPlace(3), seed=seed, bias=_launch_bias(3, 0.3)
        )
        launches = _launches(rollout)
        if any(x["step"] > 0 for x in launches):
            break
    else:
        pytest.fail("no mid-window launch")
    main_obs = np.asarray(rollout.worker.obs)  # (rows, streams, H, N, d)
    fork_obs = np.asarray(rollout.intervention.worker.obs)
    for x in launches:
        before = main_obs[x["row"], x["stream"], x["step"], :, :2]
        start = fork_obs[x["j0"], x["stream"], 0, :, :2]
        moved = np.abs(start - before).max(axis=-1) > 1e-7
        assert np.array_equal(moved, x["recruit"]) and not x["recruit"][x["focal"]]
        gap = np.linalg.norm(start[x["recruit"]] - start[x["focal"]], axis=-1)
        assert np.allclose(gap, config.intervention_radius, atol=1e-6)


def test_fork_rows_train_every_agent_with_a_ratio_of_one():
    config = _anytime(batch=4, bound="tanh", manager_input="relative")
    rs, (_, rollout, _, _), _ = _collect(config, AlwaysPlace(3), bias=ALWAYS)
    ts, fork = rs.train_state, rollout.intervention
    executed = np.asarray(rollout.adaptive.fork_executed)
    assert executed.sum() > 0
    assert np.asarray(fork.manager.active_mask)[executed].all()  # every agent's goal
    assert np.asarray(fork.worker.active_mask)[executed].all()
    for level, traj in (("worker", fork.worker), ("manager", fork.manager)):
        actor = getattr(ts, level).actor_ts
        obs = np.asarray(traj.obs)[executed]
        action = np.asarray(traj.action)[executed]
        log_prob, _ = evaluate_action(
            actor.apply_fn, actor.params, obs.reshape(-1, obs.shape[-1]),
            action.reshape(-1, action.shape[-1]), discrete=False,
        )
        np.testing.assert_allclose(
            log_prob, np.asarray(traj.log_prob)[executed].reshape(-1), atol=1e-5,
            err_msg=level,
        )


def test_a_fork_bootstraps_its_continuing_value_at_its_last_window():
    """Constant manager critic c: each fork window's manager reward is its
    discounted raw team reward, plus `gamma_M^c * c` on the fork's last window,
    which is also its only recursion cut."""
    c = 2.0
    config = _anytime(batch=4)
    gamma = config.worker.gamma
    _, (_, rollout, _, _), _ = _collect(
        config, AlwaysPlace(3), bias=ALWAYS, longest=True, critic=c
    )
    fork_m, fork_w = rollout.intervention.manager, rollout.intervention.worker
    reward, done = np.asarray(fork_m.reward), np.asarray(fork_m.done)
    raw = (np.asarray(fork_w.team_reward) * gamma ** np.arange(HORIZON)).sum(-1)
    launches = _launches(rollout)
    assert max(x["k"] for x in launches) > 1
    for x in launches:
        e = x["stream"]
        for w in range(x["k"]):
            row, last = x["j0"] + w, w == x["k"] - 1
            boot = config.manager.gamma * c if last else 0.0
            assert np.isclose(reward[row, e], raw[row, e] + boot, atol=1e-5)
            assert done[row, e] == last


def test_forks_in_one_column_never_share_an_advantage():
    config = _anytime(batch=4)
    mcfg = config.manager
    _, (_, rollout, _, _), _ = _collect(config, AlwaysPlace(3), bias=ALWAYS)
    launches = _launches(rollout)
    assert max(sum(x["stream"] == e for x in launches) for e in range(2)) >= 2

    def m_adv(ro):
        _, _, batch, _, _ = ac.adaptive_batches(ro, config, NO_MASK)
        adv, _ = compute_gae(
            batch.reward, batch.value, batch.done.astype(jnp.float32),
            jnp.zeros(batch.reward.shape[1]), mcfg.gamma, mcfg.gae_lambda,
        )
        return np.asarray(adv)

    base = m_adv(rollout)
    fork = rollout.intervention.manager
    for x in launches:
        rows = np.zeros(fork.reward.shape, bool)
        rows[x["j0"] : x["j0"] + x["k"], x["stream"]] = True
        bumped = rollout._replace(
            intervention=rollout.intervention._replace(
                manager=fork._replace(reward=jnp.where(rows, fork.reward + 100.0, fork.reward))
            )
        )
        changed = np.abs(m_adv(bumped) - base) > 1e-6
        n = fork.reward.shape[1]
        assert np.array_equal(changed[:, n:], rows)  # only this fork's own rows


def test_a_fork_whose_episode_ends_keeps_its_slots_as_masked_rows(monkeypatch):
    calls = _record_dispatch(monkeypatch)
    config = _anytime(batch=4)
    _, (_, rollout, _, _), _ = _collect(
        config, AlwaysPlace(3, terminate_at=6), bias=ALWAYS, longest=True
    )
    executed = np.asarray(rollout.adaptive.fork_executed)
    fork_w = rollout.intervention.worker
    unrun = executed & ~np.asarray(fork_w.active_mask)[..., 0, 0].astype(bool)
    assert unrun.sum() > 0  # counted, never stepped
    assert not np.asarray(fork_w.active_mask)[unrun].any()
    assert np.asarray(rollout.intervention.manager.done)[unrun].all()
    run = sum(n for name, n in calls if name == "window")
    main = int(rollout.adaptive.main_collected.sum())
    assert run - main == executed.sum() - unrun.sum()


def test_no_fork_window_starts_past_the_time_limit():
    max_steps = 6
    config = _anytime(batch=4)
    _, (_, rollout, _, _), _ = _collect(
        config, AlwaysPlace(3, max_steps=max_steps), bias=ALWAYS, longest=True
    )
    launches = _launches(rollout)
    assert launches
    for x in launches:
        t0 = int(round(x["glob"][2] * max_steps))
        assert x["k"] <= x["kmax"] <= math.ceil((max_steps - t0) / HORIZON)


# ------------------------------------------------------------------ update


def _reference_gae(reward, value, decided, gamma, lam):
    out = np.zeros_like(reward)
    for s in range(reward.shape[1]):
        idx = np.nonzero(decided[:, s])[0]
        g = 0.0
        for n, t in reversed(list(enumerate(idx))):
            nxt = value[idx[n + 1], s] if n + 1 < len(idx) else 0.0
            delta = reward[t, s] + gamma * nxt - value[t, s]
            g = delta + gamma * lam * g if n + 1 < len(idx) else delta
            out[t, s] = g
    return out


@pytest.mark.parametrize("gamma, lam", [(1.0, 0.95), (0.97, 0.9), (0.0, 0.95)])
def test_decision_gae_matches_a_ragged_reference(gamma, lam):
    rng = np.random.default_rng(1)
    reward = rng.normal(size=(40, 3)).astype(np.float32)
    value = rng.normal(size=(40, 3)).astype(np.float32)
    decided = rng.random((40, 3)) < 0.4
    decided[:, 2] = False  # a stream that never decided
    adv, ret = asch.decision_gae(reward, value, decided, gamma, lam)
    want = _reference_gae(reward, value, decided, gamma, lam)
    assert np.allclose(adv, want, atol=1e-5)
    assert np.all(adv[~decided] == 0) and np.allclose(ret, (adv + value) * decided)
    if gamma == 0.0:  # a one-step bandit
        assert np.allclose(adv[decided], (reward - value)[decided], atol=1e-6)


def test_stored_log_probs_are_the_pre_update_schedulers():
    config = _anytime(batch=4, manager_input="relative")
    rs, (_, rollout, _, _), _ = _collect(config, AlwaysPlace(3), bias=_launch_bias(3, 0.3))
    s, state = rollout.scheduler, rs.train_state.bandit
    decided = np.asarray(s.weight) > 0
    pick = lambda x: jnp.asarray(np.asarray(x)[decided])  # noqa: E731
    h = state.apply_fn(state.params, pick(s.x), pick(s.pair), pick(s.glob))
    action = asch.Action(pick(s.choice), pick(s.recruit), pick(s.duration))
    lp = asch.log_prob(h, action, pick(s.kmax))
    assert decided.sum() > 10
    np.testing.assert_allclose(lp, pick(s.log_prob), atol=1e-5)


def test_the_scheduler_trains_only_on_its_decisions():
    config = _anytime(batch=4)
    rs, (_, rollout, _, _), _ = _collect(config, AlwaysPlace(3), bias=_launch_bias(3, 0.3))
    s = rollout.scheduler
    off = s.weight == 0
    garbage = s._replace(
        x=jnp.where(off[..., None, None], 99.0, s.x),
        choice=jnp.where(off, 1, s.choice),
        log_prob=jnp.where(off, -7.0, s.log_prob),
        advantage=jnp.where(off, 1e3, s.advantage),
        returns=jnp.where(off, -1e3, s.returns),
    )
    key = jax.random.PRNGKey(5)
    a, stats_a = asch.fit_scheduler(rs.train_state.bandit, s, key, 0.01)
    b, stats_b = asch.fit_scheduler(rs.train_state.bandit, garbage, key, 0.01)
    assert _tree_max_diff(a, b) == 0.0
    assert all(float(stats_a[k]) == float(stats_b[k]) for k in stats_a)
    assert float(stats_a["scheduler_steps"]) > 0
    assert _tree_max_diff(a.params, rs.train_state.bandit.params) > 0


@pytest.mark.parametrize("scheduler", ["anytime", "anytime_random"])
def test_only_the_learned_scheduler_moves(scheduler):
    config = _anytime(scheduler, batch=4)
    rs, (rs1, rollout, last, _), (_, update_fn, _) = _collect(config, AlwaysPlace(3))
    assert (rollout.scheduler is None) == (scheduler == "anytime_random")
    rs2, losses = update_fn(rs1, rollout, last)
    moved = _tree_max_diff(rs.train_state.bandit, rs2.train_state.bandit) > 0
    assert moved == (scheduler == "anytime")
    assert ("scheduler_policy_loss" in losses) == (scheduler == "anytime")


@pytest.mark.parametrize("n_agents", [2, 3])
@pytest.mark.parametrize("scheduler", ["anytime", "anytime_random"])
def test_collect_update_eval_end_to_end(scheduler, n_agents):
    config = _anytime(scheduler, batch=4, intervention_scheduler_init_forks=4.0)
    rs, (rs1, rollout, last, _), (collect_fn, update_fn, eval_fn) = _collect(
        config, AlwaysPlace(n_agents)
    )
    rs2, losses = update_fn(rs1, rollout, last)
    assert all(
        np.isfinite(float(v)) for k, v in losses.items() if k not in anc.NAN_WHEN_EMPTY
    )
    assert np.isfinite(float(eval_fn(rs2.train_state, jax.random.PRNGKey(1))))
    _, rollout2, last2, _ = collect_fn(rs2)
    _, losses2 = update_fn(rs2, rollout2, last2)
    assert set(losses2) == set(losses)  # one key set per run


def test_collection_reproduces_from_the_same_runner_state():
    config = _anytime(batch=4)
    init_fn, collect_fn, _, _, _ = make_train(config, AlwaysPlace(3))
    rs = _prepare(init_fn(jax.random.PRNGKey(0)), bias=_launch_bias(3, 0.3))
    _, a, _, _ = collect_fn(rs)
    _, b, _, _ = collect_fn(rs)
    assert _tree_max_diff(
        (a.worker, a.manager, a.intervention, a.adaptive, a.scheduler),
        (b.worker, b.manager, b.intervention, b.adaptive, b.scheduler),
    ) == 0.0


def test_evaluation_never_reads_the_scheduler():
    config = _anytime(batch=4)
    rs, _, (_, _, eval_fn) = _collect(config, AlwaysPlace(3))
    ts = rs.train_state
    shifted = ts._replace(
        bandit=ts.bandit.replace(params=jax.tree.map(lambda p: p + 1.0, ts.bandit.params))
    )
    key = jax.random.PRNGKey(3)
    assert float(eval_fn(ts, key)) == float(eval_fn(shifted, key))


# ------------------------------------------------------------- checkpoints


def test_anytime_checkpoint_resumes_exactly(tmp_path):
    config = _anytime(batch=4)
    init_fn, collect_fn, update_fn, _, _ = make_train(config, AlwaysPlace(3))
    rs = _prepare(init_fn(jax.random.PRNGKey(0)), bias=_launch_bias(3, 0.3))
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
        (ro_a.worker, ro_a.manager, ro_a.scheduler), (ro_b.worker, ro_b.manager, ro_b.scheduler)
    ) == 0.0
    assert _tree_max_diff(state_a, state_b) == 0.0
    assert all(
        float(losses_a[k]) == float(losses_b[k]) or np.isnan(float(losses_a[k]))
        for k in losses_a
    )


def test_the_anytime_record_refuses_other_settings(tmp_path):
    config = _anytime(batch=4)
    init_fn = make_train(config, AlwaysPlace(3))[0]
    rs, eval_rng = init_fn(jax.random.PRNGKey(0)), jax.random.PRNGKey(7)
    path = tmp_path / "anytime.msgpack"
    _runner(config)._save_train_checkpoint(rs, eval_rng, path)
    meta = _runner(config)._scheduler_meta()
    assert meta[0] == ib.SCHEDULERS.index("anytime") and meta[4] == N_WINDOWS
    for other in (
        dataclasses.replace(config, intervention_scheduler_gamma=0.5),
        dataclasses.replace(config, intervention_fork_budget=2),
        dataclasses.replace(config, intervention_scheduler="anytime_random"),
    ):
        with pytest.raises(ValueError, match="other fork-scheduler settings"):
            _runner(other)._load_train_checkpoint(rs, eval_rng, path)
    # The bandit's record keeps its nine fields, so its runs still resume.
    bandit = dataclasses.replace(config, intervention_scheduler="bandit")
    assert _runner(bandit)._scheduler_meta().shape == (9,)
