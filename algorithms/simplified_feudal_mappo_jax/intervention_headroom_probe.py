"""Offline headroom check for a learned intervention scheduler.

Plan under test: `plans/intervention_contextual_bandit_2026-10-16.md`. Its bandit
reads the focal agent's manager actor input `x` and forks a window when the
predicted in-window gain is positive, with the gain

    y = sum_{k<H} gamma_M^k (fork_task_reward[k] - main_task_reward[k])

of the fork against the main window it was cut from. The rewards are raw task
rewards: no intrinsic reward and no bootstrapped value. A learned bandit can
beat "never fork" only if (1) some contexts have a positive expected gain and
(2) `x` tells them apart. Both are measured here without training: trained
policies are rolled out with the existing fork collector (`collect_fn` from
`trainer.make_train`), and the plan's network is fitted to the `(x, y)` pairs and
scored on held-out collect calls.

Decision rule, fixed before the first run: GO if, on held-out collect calls, the
top 10% of predicted contexts have a realized mean gain above 0 with a 95%
interval excluding 0, on at least 4 of 5 seeds, for the parent's policies on the
dense group under the MSE fit. Otherwise the plan's target cannot drive
selection.

Policy sources (`SOURCES`):

* the two fork arms (`none` and `redecide: all`, interval 1) under their own
  fork semantics;
* the no-fork parent's policies under each fork arm's semantics. Its actors are
  loaded unchanged, and its manager critic's first kernel is zero-padded for
  the fork context block, so values on main rows are exactly the parent's. A
  bandit arm that mostly skips trains on data close to the parent's, so these
  are the policies it would face.

Each collect call gives `n_windows * main_envs` pairs (33 x 16 = 528 at 2a/4o),
one per (window, main env), each with its own uniformly drawn focal agent.

    uv run python -m algorithms.simplified_feudal_mappo_jax.intervention_headroom_probe
    uv run python -m algorithms.simplified_feudal_mappo_jax.intervention_headroom_probe \\
        --analyze-only --out plotting/feudal_goal_analysis/intervention_headroom_<stamp>
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import from_bytes

from algorithms.mappo_jax.run import make_env
from algorithms.mappo_jax.trainer import RunnerState
from algorithms.simplified_feudal_mappo_jax.dpp_probe import REPO_ROOT, compose
# The scheduler's own network, so the probe fits exactly what would train.
from algorithms.simplified_feudal_mappo_jax.intervention_bandit import GainNet
from algorithms.simplified_feudal_mappo_jax.run import (
    Simplified_Feudal_MAPPO_JAX_Runner as Runner,
)
from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config
from algorithms.simplified_feudal_mappo_jax.trainer import (
    HierTrainState,
    create_hier_train_state,
    make_train,
)
from algorithms.simplified_feudal_mappo_jax.types import Model_Params, Params

PARENT = "simplified_feudal_tanh_relative_input"
FORK_NONE = "simplified_feudal_tanh_relative_input_interventions"
FORK_ALL = "simplified_feudal_tanh_relative_input_interventions_redecide_all"
# (label, policy model group, fork-semantics model group)
SOURCES = (
    ("parent_none", PARENT, FORK_NONE),
    ("parent_all", PARENT, FORK_ALL),
    ("fork_none", FORK_NONE, FORK_NONE),
    ("fork_all", FORK_ALL, FORK_ALL),
)
GROUPS = ("mjx_2a_4o_1122_1024_gs", "mjx_2a_4o_1122_1024_gs_sparse")
# A step paying more than this delivered a box: the +100 bonus dwarfs the
# per-step displacement shaping.
DELIVERY_STEP_REWARD = 50.0
LOSSES = ("mse", "huber1")
FRACTIONS = (0.05, 0.10, 0.25)
N_BOOT = 2000


# ------------------------------------------------------------------ loading


def arm_config(batch: str, model: str, trial: str):
    """`(env_config, FeudalConfig)` exactly as the runner resolves them; under
    forks `make_feudal_config` takes the BATCH env count and reduces it."""
    args = compose(batch, model, trial)
    env_config = dict(args["env_config"])
    exp = args["exp_dict"]
    config = make_feudal_config(
        Params(**exp["params"]), Model_Params(**exp["model_params"]), env_config["n_envs"]
    )
    return env_config, config


def _pad_rows(template, params, name):
    """`params` fitted into `template`'s tree. Equal-shape leaves are copied; a
    leaf the template makes taller (the critic's first kernel, which reads the
    fork context block appended to its input) is zero-padded at the end."""

    def pad(t, p):
        if t.shape == p.shape:
            return p
        if t.shape[1:] != p.shape[1:] or t.shape[0] < p.shape[0]:
            raise ValueError(f"{name}: cannot fit {p.shape} into {t.shape}")
        return jnp.concatenate(
            [p, jnp.zeros((t.shape[0] - p.shape[0],) + p.shape[1:], p.dtype)]
        )

    return jax.tree.map(pad, template, params)


def _same_shapes(a, b, name):
    sa = jax.tree.map(lambda x: x.shape, a)
    sb = jax.tree.map(lambda x: x.shape, b)
    if sa != sb:
        raise ValueError(f"{name} shapes differ between the arms: {sa} vs {sb}")


def load_policy(template, env, batch, policy_model, trial):
    """`template` (the fork configuration's train state) carrying
    `policy_model`'s saved weights. Returns `(train_state, checkpoint path)`.

    Only the params are replaced, so every source shares the template's
    `apply_fn` / optimizer objects, which `jit` treats as static: a fresh train
    state per source would recompile the collector each time."""
    ts = template
    models = REPO_ROOT / "experiments" / "results" / batch / policy_model / str(trial) / "models"
    path = models / "models_finished.msgpack"
    if not path.exists():
        path = models / "models_checkpoint.msgpack"
    _, p_config = arm_config(batch, policy_model, trial)
    p_ts = create_hier_train_state(jax.random.PRNGKey(0), p_config, env)
    params = from_bytes(Runner._params_tree(p_ts), path.read_bytes())
    _same_shapes(ts.worker.actor_ts.params, params["worker_actor"], "worker actor")
    _same_shapes(ts.manager.actor_ts.params, params["manager_actor"], "manager actor")
    ts = HierTrainState(
        worker=ts.worker._replace(
            actor_ts=ts.worker.actor_ts.replace(params=params["worker_actor"]),
            critic_ts=ts.worker.critic_ts.replace(
                params=_pad_rows(ts.worker.critic_ts.params, params["worker_critic"], "worker critic")
            ),
        ),
        manager=ts.manager._replace(
            actor_ts=ts.manager.actor_ts.replace(params=params["manager_actor"]),
            critic_ts=ts.manager.critic_ts.replace(
                params=_pad_rows(ts.manager.critic_ts.params, params["manager_critic"], "manager critic")
            ),
        ),
        manager_adv=ts.manager_adv,
    )
    return ts, path


# ------------------------------------------------------------------ collection


def make_extract(config):
    """Jitted `Rollout -> per-pair arrays`, each `(n_windows, main_envs, ...)`."""
    horizon = config.goal_horizon
    n_windows = config.worker.n_steps // horizon
    m_gamma = (
        config.worker.gamma if config.manager_step_gamma is None else config.manager_step_gamma
    )
    discount = m_gamma ** jnp.arange(horizon)

    @jax.jit
    def extract(rollout):
        fork = rollout.intervention
        focal = fork.focal  # (W, E)
        x = jnp.take_along_axis(rollout.manager.obs, focal[:, :, None, None], axis=2)[:, :, 0]
        n_env = focal.shape[1]
        main_r = rollout.worker.team_reward.reshape(n_windows, horizon, n_env)
        fork_r = fork.worker.team_reward.reshape(n_windows, horizon, n_env)
        return {
            "x": x,
            "focal": focal,
            "n_recruits": fork.n_recruits,
            "valid": fork.valid,
            "g_main": jnp.einsum("whe,h->we", main_r, discount),
            "g_fork": jnp.einsum("whe,h->we", fork_r, discount),
            "deliv_main": (main_r > DELIVERY_STEP_REWARD).sum(1),
            "deliv_fork": (fork_r > DELIVERY_STEP_REWARD).sum(1),
        }

    return extract


def collect_source(collect_fn, extract, ts, n_calls, seed):
    """`n_calls` collect calls under fresh keys; each pair is tagged with its
    call (the bootstrap cluster) and window index."""
    chunks, times = [], []
    for c in range(n_calls):
        start = time.perf_counter()
        key = jax.random.fold_in(jax.random.PRNGKey(seed), c)
        _, rollout, _, _ = collect_fn(RunnerState(train_state=ts, rng=key))
        arrays = jax.device_get(extract(rollout))
        times.append(time.perf_counter() - start)
        n_windows, n_env = arrays["focal"].shape
        arrays["call"] = np.full((n_windows, n_env), c)
        arrays["window"] = np.broadcast_to(np.arange(n_windows)[:, None], (n_windows, n_env))
        chunks.append({k: np.asarray(v).reshape((n_windows * n_env,) + np.shape(v)[2:]) for k, v in arrays.items()})
    data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    return data, times


def run_collection(out, groups, n_calls, trials):
    meta = {"groups": list(groups), "trials": list(trials), "n_calls": n_calls, "sources": []}
    for batch in groups:
        for fork_model in sorted({f for _, _, f in SOURCES}):
            env_config, config = arm_config(batch, fork_model, trials[0])
            env = make_env(env_config)
            _, collect_fn, _, _, _ = make_train(config, env)
            extract = make_extract(config)
            template = create_hier_train_state(jax.random.PRNGKey(0), config, env)
            for label, policy_model, semantics in SOURCES:
                if semantics != fork_model:
                    continue
                for trial in trials:
                    _, trial_config = arm_config(batch, fork_model, trial)
                    if trial_config != config:
                        raise ValueError(f"{batch}/{fork_model}: config differs at trial {trial}")
                    ts, path = load_policy(template, env, batch, policy_model, trial)
                    seed = 1_000 * int(trial) + 17
                    data, times = collect_source(collect_fn, extract, ts, n_calls, seed)
                    name = f"{batch}__{label}__{trial}"
                    np.savez_compressed(out / f"{name}.npz", **data)
                    meta["sources"].append(
                        {
                            "name": name,
                            "batch": batch,
                            "label": label,
                            "policy_model": policy_model,
                            "fork_model": fork_model,
                            "trial": trial,
                            "checkpoint": str(path.relative_to(REPO_ROOT)),
                            "redecide": config.intervention_redecide,
                            "main_envs": config.worker.n_envs,
                            "pairs": int(data["focal"].size),
                            "first_call_s": round(times[0], 2),
                            "warm_call_s": round(float(np.median(times[1:])), 3),
                        }
                    )
                    print(
                        f"{name}: {data['focal'].size} pairs, first call {times[0]:.1f}s, "
                        f"warm {np.median(times[1:]):.2f}s, mean y "
                        f"{np.mean(data['g_fork'] - data['g_main']):+.3f}",
                        flush=True,
                    )
    (out / "collection.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ------------------------------------------------------------------ fitting


def fit(x_train, y_train, x_val, y_val, loss, seed, epochs, batch_size=256, lr=1e-3):
    """Fit `GainNet`, keep the epoch with the lowest validation loss, and return
    a predictor in raw `y` units.

    `mse`: squared error on standardized targets (the mean is what an
    intervene-iff-positive rule needs). `huber1`: the plan's Huber loss with
    delta 1 on raw targets.
    """
    if loss == "mse":
        mu, sd = float(y_train.mean()), float(y_train.std() + 1e-8)
    else:
        mu, sd = 0.0, 1.0
    zt, zv = (y_train - mu) / sd, (y_val - mu) / sd
    net = GainNet()
    params = net.init(jax.random.PRNGKey(seed), jnp.zeros((1, x_train.shape[1])))
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
    opt = tx.init(params)

    def objective(p, xb, zb):
        pred = net.apply(p, xb)
        if loss == "mse":
            return jnp.mean((pred - zb) ** 2)
        return jnp.mean(optax.huber_loss(pred, zb, delta=1.0))

    n_batches = max(1, len(zt) // batch_size)
    xt, zt_j = jnp.asarray(x_train), jnp.asarray(zt)
    xv, zv_j = jnp.asarray(x_val), jnp.asarray(zv)

    # The data are arguments, not closed-over constants, so XLA does not embed
    # them in the compiled program.
    @jax.jit
    def epoch(params, opt, key, xt, zt_j, xv, zv_j):
        perm = jax.random.permutation(key, zt_j.shape[0])[: n_batches * batch_size]
        perm = perm.reshape(n_batches, batch_size)

        def step(carry, idx):
            p, o = carry
            g = jax.grad(objective)(p, xt[idx], zt_j[idx])
            updates, o = tx.update(g, o, p)
            return (optax.apply_updates(p, updates), o), None

        (params, opt), _ = jax.lax.scan(step, (params, opt), perm)
        return params, opt, objective(params, xv, zv_j)

    best, best_loss = params, float(objective(params, xv, zv_j))
    key = jax.random.PRNGKey(seed + 1)
    for _ in range(epochs):
        key, sub = jax.random.split(key)
        params, opt, val = epoch(params, opt, sub, xt, zt_j, xv, zv_j)
        if float(val) < best_loss:
            best, best_loss = params, float(val)

    predict = jax.jit(lambda x: mu + sd * net.apply(best, x))
    return lambda x: np.asarray(predict(jnp.asarray(x)))


# ------------------------------------------------------------------ statistics


def cluster_mean_ci(y, clusters, masks, rng, n_boot=N_BOOT):
    """Mean of `y` over each boolean mask, with a 95% bootstrap interval that
    resamples whole collect calls (windows within a call share streams)."""
    ids, inverse = np.unique(clusters, return_inverse=True)
    out = []
    draws = rng.integers(0, len(ids), (n_boot, len(ids)))
    for mask in masks:
        s = np.bincount(inverse, weights=y * mask, minlength=len(ids))
        n = np.bincount(inverse, weights=mask.astype(float), minlength=len(ids))
        if n.sum() == 0:
            out.append({"n": 0, "mean": None, "lo": None, "hi": None})
            continue
        boot = s[draws].sum(1) / np.maximum(n[draws].sum(1), 1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({"n": int(n.sum()), "mean": float(s.sum() / n.sum()), "lo": float(lo), "hi": float(hi)})
    return out


def selection_report(pred, y, clusters, rng):
    """How well the predictor ranks held-out pairs, in raw gain units."""
    order = np.argsort(-pred, kind="stable")
    rank = np.empty(len(pred), int)
    rank[order] = np.arange(len(pred))
    decile = np.minimum(rank * 10 // len(pred), 9)  # 0 = highest predicted
    masks = [decile == d for d in range(10)]
    masks += [rank < int(round(f * len(pred))) for f in FRACTIONS]
    masks += [pred > 0, np.ones(len(pred), bool)]
    stats = cluster_mean_ci(y, clusters, masks, rng)
    sse_model = float(((y - pred) ** 2).sum())
    return {
        "deciles_high_to_low": stats[:10],
        "top": {f"{f:.2f}": s for f, s in zip(FRACTIONS, stats[10 : 10 + len(FRACTIONS)])},
        "predicted_positive": stats[-2],
        "predicted_positive_frac": float((pred > 0).mean()),
        "all": stats[-1],
        "sse_model": sse_model,
    }


def describe(d, rng):
    """Distribution of the gain, and how much of it comes from deliveries."""
    y = d["g_fork"] - d["g_main"]
    valid = d["valid"].astype(bool)
    y, call = y[valid], d["call"][valid]
    deliv = (d["deliv_main"] + d["deliv_fork"])[valid] > 0
    x = d["x"][valid]
    slot0 = x[:, 4:10]  # nearest undelivered box: dx, dy, goal, touch, coupling, valid
    needs_two = (slot0[:, 5] > 0.5) & (slot0[:, 4] > 0.99)
    near = np.linalg.norm(slot0[:, :2], axis=-1) < 0.1
    alone = needs_two & near & (np.abs(slot0[:, 3] - 0.5) < 0.25)
    masks = [np.ones(len(y), bool), deliv, ~deliv, needs_two, alone]
    names = ["all", "delivery_window", "no_delivery_window", "nearest_box_needs_two",
             "alone_at_two_agent_box"]
    stats = dict(zip(names, cluster_mean_ci(y, call, masks, rng)))
    return {
        "pairs": int(valid.size),
        "valid_frac": float(valid.mean()),
        "y_pos_frac": float((y > 0).mean()),
        "y_zero_frac": float((y == 0).mean()),
        "y_neg_frac": float((y < 0).mean()),
        "y_std": float(y.std()),
        "g_main_mean": float(d["g_main"][valid].mean()),
        "g_fork_mean": float(d["g_fork"][valid].mean()),
        "deliveries_per_window_main": float(d["deliv_main"][valid].mean()),
        "deliveries_per_window_fork": float(d["deliv_fork"][valid].mean()),
        "delivery_window_frac": float(deliv.mean()),
        "means": stats,
    }


def _xy(d):
    valid = d["valid"].astype(bool)
    return d["x"][valid], (d["g_fork"] - d["g_main"])[valid], d["call"][valid]


def within_seed(d, loss, seed, epochs):
    """Train on the first half of the collect calls (its last fifth for early
    stopping), test on the second half."""
    x, y, call = _xy(d)
    n_calls = int(call.max()) + 1
    half, val_start = n_calls // 2, int(0.8 * (n_calls // 2))
    train, val, test = call < val_start, (call >= val_start) & (call < half), call >= half
    predict = fit(x[train], y[train], x[val], y[val], loss, seed, epochs)
    pred = predict(x[test])
    report = selection_report(pred, y[test], call[test], np.random.default_rng(seed))
    const = float(y[train].mean())
    report["oos_r2"] = 1.0 - report["sse_model"] / float(((y[test] - const) ** 2).sum())
    return report


def leave_one_seed_out(by_trial, held, loss, seed, epochs):
    """Train on every other seed (10% of their calls for early stopping), test on
    `held`."""
    xs, ys, vs = [], [], []
    for trial, d in by_trial.items():
        if trial == held:
            continue
        x, y, call = _xy(d)
        xs.append(x), ys.append(y), vs.append(call >= int(0.9 * (int(call.max()) + 1)))
    x, y, v = np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)
    predict = fit(x[~v], y[~v], x[v], y[v], loss, seed, epochs)
    xt, yt, ct = _xy(by_trial[held])
    pred = predict(xt)
    report = selection_report(pred, yt, ct, np.random.default_rng(seed))
    const = float(y[~v].mean())
    report["oos_r2"] = 1.0 - report["sse_model"] / float(((yt - const) ** 2).sum())
    return report


def decision(results, batch, label="parent_none"):
    """The pre-registered rule on the within-seed MSE fit."""
    passes = []
    for trial, r in results[batch][label]["within_seed"]["mse"].items():
        top = r["top"]["0.10"]
        passes.append(top["mean"] is not None and top["lo"] > 0)
    return {"batch": batch, "label": label, "seeds_passing": int(sum(passes)),
            "seeds": len(passes), "go": int(sum(passes)) >= 4}


def run_analysis(out, epochs_within, epochs_loso):
    meta = json.loads((out / "collection.json").read_text())
    results = {}
    for src in meta["sources"]:
        results.setdefault(src["batch"], {}).setdefault(src["label"], {"by_trial": {}})
        results[src["batch"]][src["label"]]["by_trial"][src["trial"]] = src["name"]
    report = {"collection": meta, "results": {}}
    for batch, labels in results.items():
        report["results"][batch] = {}
        for label, entry in labels.items():
            data = {t: dict(np.load(out / f"{n}.npz")) for t, n in entry["by_trial"].items()}
            rng = np.random.default_rng(0)
            res = {
                "describe": {t: describe(d, rng) for t, d in data.items()},
                "describe_pooled": describe(
                    {k: np.concatenate([d[k] + (1000 * i if k == "call" else 0)
                                        for i, d in enumerate(data.values())])
                     for k in next(iter(data.values()))},
                    rng,
                ),
                "within_seed": {},
                "leave_one_seed_out": {},
            }
            for loss in LOSSES:
                res["within_seed"][loss] = {
                    t: within_seed(d, loss, 7 + int(t), epochs_within) for t, d in data.items()
                }
                res["leave_one_seed_out"][loss] = {
                    t: leave_one_seed_out(data, t, loss, 11 + int(t), epochs_loso) for t in data
                }
                print(f"fitted {batch} {label} {loss}", flush=True)
            report["results"][batch][label] = res
    report["decision"] = [decision(report["results"], b) for b in report["results"]]
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


# ------------------------------------------------------------------ summary

LABEL_TEXT = {
    "parent_none": "parent policy, `none` fork",
    "parent_all": "parent policy, `all` fork",
    "fork_none": "`none` fork arm",
    "fork_all": "`all` fork arm",
}
# Figure encoding: hue = fork semantics, line style = whose policy.
SEMANTICS_COLOR = {"none": "#2a78d6", "all": "#eb6834"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
)


def _ci(s, digits=2):
    if s["mean"] is None:
        return "—"
    return f"{s['mean']:+.{digits}f} [{s['lo']:+.{digits}f}, {s['hi']:+.{digits}f}]"


def _pass(s):
    return s["mean"] is not None and s["lo"] > 0


def write_summary(out, report):
    """`summary.md` (tables) and `deciles.png` from `report.json`."""
    lines = ["# Intervention headroom probe", "",
             "Gain `y` = discounted raw team return of the fork window minus the main",
             "window it was cut from (per-step discount 0.99, 32 steps). Means with",
             "95% intervals that resample whole collect calls.", ""]
    for d in report["decision"]:
        lines.append(f"**Decision ({d['batch']}, {d['label']}):** {d['seeds_passing']}/"
                     f"{d['seeds']} seeds have a top-10% realized gain above 0 -> "
                     f"{'GO' if d['go'] else 'STOP'}.")
    for batch, labels in report["results"].items():
        lines += ["", f"## {batch}", "", "### Gain distribution (5 seeds pooled)", "",
                  "| source | pairs | G main | G fork | mean gain | >0 / =0 / <0 | "
                  "delivery windows | gain, delivery windows | gain, other windows | "
                  "gain, alone at a two-agent box |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for label, res in labels.items():
            p, m = res["describe_pooled"], res["describe_pooled"]["means"]
            lines.append(
                f"| {LABEL_TEXT[label]} | {p['pairs']} | {p['g_main_mean']:.2f} | "
                f"{p['g_fork_mean']:.2f} | {_ci(m['all'])} | {p['y_pos_frac']:.2f} / "
                f"{p['y_zero_frac']:.2f} / {p['y_neg_frac']:.2f} | "
                f"{p['delivery_window_frac']:.3f} | {_ci(m['delivery_window'], 1)} | "
                f"{_ci(m['no_delivery_window'], 3)} | {_ci(m['alone_at_two_agent_box'])} "
                f"(n={m['alone_at_two_agent_box']['n']}) |")
        for split in ("within_seed", "leave_one_seed_out"):
            title = ("Held-out second half of each seed's calls" if split == "within_seed"
                     else "Held-out seed (trained on the other four)")
            seeds = sorted(next(iter(labels.values()))[split]["mse"])
            lines += ["", f"### {title}: realized gain of the top 10% predicted (MSE fit)", "",
                      "| source | " + " | ".join(f"seed {t}" for t in seeds)
                      + " | seeds > 0 | predicted > 0 | out-of-sample R² |",
                      "|---|" + "---|" * (len(seeds) + 3)]
            for label, res in labels.items():
                fits = res[split]["mse"]
                tops = [fits[t]["top"]["0.10"] for t in sorted(fits)]
                lines.append(
                    f"| {LABEL_TEXT[label]} | " + " | ".join(_ci(s) for s in tops)
                    + f" | {sum(_pass(s) for s in tops)}/{len(tops)} | "
                    f"{np.mean([fits[t]['predicted_positive_frac'] for t in fits]):.3f} | "
                    f"{np.mean([fits[t]['oos_r2'] for t in fits]):+.3f} |")
        lines += ["", "### Loss comparison (held-out calls, mean over seeds)", "",
                  "| source | loss | predicted > 0 | realized gain, predicted > 0 | "
                  "top 5% | top 10% | top 25% | all |", "|---|---|---|---|---|---|---|---|"]
        for label, res in labels.items():
            for loss in LOSSES:
                fits = res["within_seed"][loss]

                def avg(get):
                    vals = [get(f) for f in fits.values()]
                    vals = [v for v in vals if v is not None]
                    return f"{np.mean(vals):+.2f}" if vals else "—"

                lines.append(
                    f"| {LABEL_TEXT[label]} | {loss} | "
                    f"{np.mean([f['predicted_positive_frac'] for f in fits.values()]):.3f} | "
                    f"{avg(lambda f: f['predicted_positive']['mean'])} | "
                    + " | ".join(avg(lambda f, q=q: f["top"][q]["mean"]) for q in ("0.05", "0.10", "0.25"))
                    + f" | {avg(lambda f: f['all']['mean'])} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    _decile_figure(out, report)


def _decile_figure(out, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batches = list(report["results"])
    fig, axes = plt.subplots(1, len(batches), figsize=(6.4 * len(batches), 4.6),
                             facecolor=SURFACE, squeeze=False)
    n_seeds = 0
    x = np.arange(1, 11)
    for ax, batch in zip(axes[0], batches):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ends = []
        for i, (label, res) in enumerate(report["results"][batch].items()):
            fits = res["within_seed"]["mse"]
            curves = np.array([[s["mean"] for s in fits[t]["deciles_high_to_low"]]
                               for t in sorted(fits)], float)
            n_seeds = len(curves)
            mean = np.nanmean(curves, 0)
            sem = np.nanstd(curves, 0, ddof=1) / np.sqrt(len(curves))
            semantics = "none" if label.endswith("none") else "all"
            color = SEMANTICS_COLOR[semantics]
            style = "-" if label.startswith("parent") else "--"
            offset = (i - 1.5) * 0.06
            ax.errorbar(x + offset, mean, yerr=sem, color=color, ls=style, lw=2,
                        marker="o", ms=4.5, mec=SURFACE, mew=1, elinewidth=1,
                        capsize=0, label=LABEL_TEXT[label].replace("`", ""), zorder=3)
            ends.append([mean[-1], LABEL_TEXT[label].replace("`", ""), color])
        # Direct labels at the right end, nudged apart so they do not overlap.
        ends.sort(key=lambda e: e[0])
        span = np.subtract(*ax.get_ylim()[::-1])
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.07 * span)
        for y_end, text, color in ends:
            ax.annotate(text, (10.25, y_end), color=INK_2, fontsize=7.5, va="center",
                        annotation_clip=False)
        ax.set_xlim(0.6, 10.4)
        ax.set_xticks(x)
        ax.set_xlabel("decile of predicted gain (1 = highest)", color=INK_2, fontsize=9)
        ax.set_ylabel("realized gain, held-out calls\n(discounted team return)",
                      color=INK_2, fontsize=9)
        ax.set_title(batch, color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
    # Plain line handles: error-bar handles break up the line and hide the
    # solid / dashed distinction that encodes whose policy it is.
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=SEMANTICS_COLOR["none" if lab.endswith("none") else "all"],
               ls="-" if lab.startswith("parent") else "--", lw=2, marker="o", ms=4.5,
               mec=SURFACE, label=LABEL_TEXT[lab].replace("`", ""))
        for lab in report["results"][batches[0]]
    ]
    axes[0][0].legend(handles=handles, frameon=False, fontsize=7.5, labelcolor=INK_2,
                      loc="lower left", handlelength=3)
    fig.suptitle(f"Fork gain by decile of predicted gain\nwithin-seed fit, mean over "
                 f"{n_seeds} seeds ± standard error", color=INK, fontsize=10.5, x=0.01,
                 ha="left")
    # Room on the right of every panel for its direct labels.
    wide = len(batches) > 1
    fig.subplots_adjust(left=0.08 if wide else 0.14, right=0.87 if wide else 0.76,
                        wspace=0.62, top=0.83, bottom=0.12)
    fig.savefig(out / "deciles.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--trials", default="0,1,2,3,4")
    parser.add_argument("--n-calls", type=int, default=64)
    parser.add_argument("--epochs-within", type=int, default=80)
    parser.add_argument("--epochs-loso", type=int, default=30)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true",
                        help="rewrite summary.md / deciles.png from report.json")
    args = parser.parse_args(argv)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or REPO_ROOT / "plotting" / "feudal_goal_analysis" / f"intervention_headroom_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        write_summary(out, json.loads((out / "report.json").read_text()))
        print(f"wrote {out / 'summary.md'}")
        return
    if not args.analyze_only:
        run_collection(out, args.groups.split(","), args.n_calls, args.trials.split(","))
    report = run_analysis(out, args.epochs_within, args.epochs_loso)
    write_summary(out, report)
    for d in report["decision"]:
        print(f"decision {d['batch']} {d['label']}: {d['seeds_passing']}/{d['seeds']} "
              f"seeds pass -> {'GO' if d['go'] else 'STOP'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
