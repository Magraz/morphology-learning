"""Test 4: paired factorial audit of support targets and commitment lengths.

    uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_support_probe \\
        --source-dir PATH_TO_TEST_3_RESULTS

Reuses complete test-3 source states and frozen weights. Recruited identities
and count stay fixed across target/duration conditions. Support is replanned
only at manager boundaries; all other goals come from the original manager.
"""

import argparse
import datetime
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from algorithms.simplified_feudal_mappo_jax import counterfactual as cf
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.dpp_simulator_probe import (
    METRICS, _batch, _clean, _interval, load_sources, make_evaluator, save_sources,
)
from algorithms.simplified_feudal_mappo_jax.trainer import make_policy

TARGETS = ("focal_position", "focal_waypoint", "box_staging")
EXTRA_METRICS = ("target_focal_touch_steps", "target_recruit_touch_steps",
                 "target_all_recruits_touch_steps", "target_coupled_steps",
                 "target_delivery_events", "recruit_waypoint_reached_steps")


def select_box(env, state, focal):
    """Prefer a touched live coupled box, then the nearest live coupled box."""
    box, yaw = env._box_pose(state.data)
    agent = env._agent_pos(state.data)
    touch = env._touch_matrix(agent, box, yaw)[focal]
    valid = ~state.delivered & (env._coupling >= 2)
    touched = valid & touch
    eligible = jnp.where(touched.any(), touched, valid)
    distance = jnp.linalg.norm(box - agent[focal], axis=-1)
    index = jnp.argmin(jnp.where(eligible, distance, jnp.inf))
    return index, valid.any()


def staging_targets(env, state, focal, recruit_mask, box_id):
    """Routed rear staging with separated slots and an upward pushing phase.

    This is a privileged diagnostic controller adapted from the repo's scripted
    manager, not a learned policy. The focal slot is reserved but its goal is
    never overwritten. Square-box yaw enters the conservative world-axis bounds.
    """
    agents = env._agent_pos(state.data)
    boxes, yaw = env._box_pose(state.data)
    box = boxes[box_id]
    half = env._box_half[box_id] * (jnp.abs(jnp.cos(yaw[box_id])) + jnp.abs(jnp.sin(yaw[box_id])))
    members = recruit_mask | (jnp.arange(env.n_agents) == focal)
    rank = jnp.cumsum(members) - 1
    # Agent diameter 0.8 in MultiBoxPushMJX, plus 0.1 spacing.
    lateral = (rank - (members.sum() - 1) / 2) * .9
    lateral = jnp.clip(lateral, -.8 * half, .8 * half)
    stages = box + jnp.stack([lateral, jnp.full((env.n_agents,), -(half + .6))], axis=-1)
    close = jnp.linalg.norm(stages - agents, axis=-1) < .85
    # One world-width step is later bounded by the legal waypoint radius.
    pushing = agents + jnp.array([0., env.world_height])
    desired = jnp.where(close[:, None], pushing, stages)
    # Route agents that are above the staging line around the box, then below it.
    above = agents[:, 1] > stages[:, 1] + .25
    delta = agents - box
    sign = jnp.where(delta[:, 0] >= 0, 1., -1.)
    bypass_x = box[0] + sign * (half + 1.2)
    beside = jnp.abs(delta[:, 0]) >= half + .8
    around = jnp.stack([bypass_x, jnp.where(beside, stages[:, 1], agents[:, 1])], axis=-1)
    desired = jnp.where((above & ~close)[:, None], around, desired)
    return (desired - env._centre) / env._extent


def make_intervention(env, radius):
    def intervene(state, manager_goals, focal, window, context):
        pos = env.goal_state(state)
        target = jax.lax.switch(context["target"], (
            lambda: jnp.broadcast_to(pos[focal], pos.shape),
            lambda: jnp.broadcast_to(manager_goals[focal], pos.shape),
            lambda: staging_targets(env, state, focal, context["recruit_mask"], context["box"]),
        ))
        # The exact clip mapping used by production support_offsets.
        support = wp.waypoint_from_action(pos, (target - pos) / radius, radius, "clip")
        active = context["enabled"] & (window < context["commitment"])
        # A delivered staging task releases its recruits at the next boundary.
        stage_live = context["box_valid"] & ~state.delivered[context["box"]]
        active &= (context["target"] != 2) | stage_live
        return jnp.where((active & context["recruit_mask"])[:, None], support, manager_goals)
    return intervene


def make_extra_metrics(env, radius):
    def extra(state, next_state, waypoint, focal, live, t, context):
        box, yaw = env._box_pose(next_state.data)
        touch = env._touch_matrix(env._agent_pos(next_state.data), box, yaw)[:, context["box"]]
        valid = live & context["box_valid"]
        undelivered = ~state.delivered[context["box"]]
        recruit_touch = (touch & context["recruit_mask"]).sum()
        count = context["recruit_mask"].sum()
        met = touch.sum() >= env._coupling[context["box"]]
        reached = wp.distance_to_waypoint(waypoint, env.goal_state(next_state), radius) <= .1
        return jnp.array([
            valid & undelivered & touch[focal],
            jnp.where(valid & undelivered, recruit_touch, 0),
            valid & undelivered & (count > 0) & (recruit_touch == count),
            valid & undelivered & met,
            valid & next_state.delivered[context["box"]] & undelivered,
            jnp.where(live, (reached & context["recruit_mask"]).sum(), 0),
        ], dtype=jnp.float32)
    return extra


def source_contexts(env, config, ts, sources, fixed_recruits=None):
    """Choose production DPP's best count once, without changing it by condition."""
    policy = make_policy(config, env)
    contexts, source_predictions = [], []
    n_max = min(config.dpp_max_recruits or env.n_agents - 1, env.n_agents - 1)
    if fixed_recruits is not None and not 1 <= fixed_recruits <= env.n_agents - 1:
        raise ValueError("recruits must be between 1 and n_agents - 1")
    if n_max < 1:
        raise ValueError("Support audit requires at least two agents")
    for source in sources:
        state, focal = source["state"], int(source["focal"])
        pos = env.goal_state(state)
        offsets = wp.goal_error(jnp.asarray(source["waypoint"]), pos, config.waypoint_radius)
        gs, _ = policy.observe(jnp.asarray(source["obs"])[None], _batch(state))
        x = wp.manager_critic_input(gs, pos[None])[0]
        base = ts.manager_adv.apply_fn(ts.manager_adv.params, cf.adv_model_input(x, offsets))
        gains = []
        for n in range(1, n_max + 1):
            joint = cf.dpp_joint(offsets, pos, focal, n, config.waypoint_radius)
            pred = ts.manager_adv.apply_fn(ts.manager_adv.params, cf.adv_model_input(x, joint))
            gains.append(float((pred - base) / n))
        if not np.isfinite(gains).all():
            raise ValueError("Nonfinite DPP predictions at a source state")
        n = fixed_recruits if fixed_recruits is not None else int(np.argmax(gains)) + 1
        box, valid = select_box(env, state, focal)
        contexts.append({"recruit_mask": cf.recruit_mask(pos, focal, n),
                         "box": box, "box_valid": valid})
        source_predictions.append({"recruits": n, "dpp_best_recruits": int(np.argmax(gains)) + 1,
                                   "dpp_raw_gain": max(gains), "box": int(box) if valid else -1,
                                   "box_valid": bool(valid),
                                   "box_coupling": int(env._coupling[box]) if valid else 0})
    return contexts, source_predictions


def _event_stats(times):
    times = np.asarray(times)
    happened = times >= 0
    return float(happened.mean()), float(times[happened].mean()) if happened.any() else None


def evaluate_support(env, config, ts, sources, meta, out, commitments, mc_samples,
                     chunk_size, eval_steps, fixed_recruits=None):
    intervene = make_intervention(env, config.waypoint_radius)
    evaluate, horizon_labels, stops = make_evaluator(env, config, eval_steps,
        waypoint_intervention=intervene, extra_metrics=make_extra_metrics(env, config.waypoint_radius))
    contexts, info = source_contexts(env, config, ts, sources, fixed_recruits)
    policy = make_policy(config, env)
    specs = [("original", 0)] + [(t, k) for t in TARGETS for k in commitments]
    rows, contrasts = [], []
    rng = np.random.default_rng(meta["seed"])
    for start in range(0, len(sources), chunk_size):
        selected = sources[start:start + chunk_size]
        states = jax.tree.map(lambda *xs: jnp.stack(xs), *(s["state"] for s in selected))
        obs = jnp.asarray(np.stack([s["obs"] for s in selected]))
        goals = jnp.asarray(np.stack([s["waypoint"] for s in selected]))
        focal = jnp.asarray([s["focal"] for s in selected])
        base_context = jax.tree.map(lambda *xs: jnp.stack(xs), *contexts[start:start+len(selected)])
        keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(meta["seed"]),
            200000 + int(s["source_id"]) * mc_samples + r)
            for s in selected for r in range(mc_samples)])
        repeated_states = jax.tree.map(lambda x: jnp.repeat(x, mc_samples, axis=0), states)
        repeated_obs, repeated_goals = jnp.repeat(obs, mc_samples, 0), jnp.repeat(goals, mc_samples, 0)
        repeated_focal = jnp.repeat(focal, mc_samples)
        results, first_predictions = [], []
        print(f"  Comparing support for sources {start + 1}–{start + len(selected)}/{len(sources)}...", flush=True)
        for target, commitment in specs:
            target_index = TARGETS.index(target) if target != "original" else 0
            context = {**base_context, "target": jnp.full((len(selected),), target_index, jnp.int32),
                       "commitment": jnp.full((len(selected),), commitment, jnp.int32),
                       "enabled": jnp.full((len(selected),), target != "original")}
            first_goals = jax.vmap(intervene)(states, goals, focal, jnp.zeros(len(selected), jnp.int32), context)
            gs, pos = policy.observe(obs, states)
            x = wp.manager_critic_input(gs, pos)
            predicted = ts.manager_adv.apply_fn(ts.manager_adv.params,
                cf.adv_model_input(x, wp.goal_error(first_goals, pos, config.waypoint_radius)))
            first_predictions.append(np.asarray(predicted))
            repeated_context = jax.tree.map(lambda x: jnp.repeat(x, mc_samples, 0), context)
            result = evaluate(ts, repeated_states, repeated_obs, repeated_goals,
                              repeated_focal, keys, repeated_context)
            results.append(jax.device_get(result))
        shape = (len(selected), mc_samples)
        metrics = np.stack([r["metrics"].reshape(*shape, len(stops), len(METRICS)) for r in results], axis=1)
        extra = np.stack([r["extra_metrics"].reshape(*shape, len(stops), len(EXTRA_METRICS)) for r in results], axis=1)
        first = np.stack([r["extra_first_steps"].reshape(*shape, len(stops), len(EXTRA_METRICS)) for r in results], axis=1)
        gae = np.stack([r["training_gae"].reshape(*shape) for r in results], axis=1)
        predicted = np.stack(first_predictions, axis=1)
        if not all(np.isfinite(a).all() for a in (metrics, extra, gae, predicted)):
            raise ValueError(f"Nonfinite outcomes or predictions in support chunk {start}")
        predicted -= predicted[:, :1]
        np.savez_compressed(out / f"trial_{meta['trial_index']}_chunk_{start}.npz",
            metrics=metrics, extra_metrics=extra, extra_first_steps=first, training_gae=gae,
            initial_predicted_delta=predicted, metric_names=np.asarray(METRICS),
            extra_metric_names=np.asarray(EXTRA_METRICS), targets=np.asarray([s[0] for s in specs]),
            commitments=np.asarray([s[1] for s in specs]), source_ids=np.asarray([s["source_id"] for s in selected]),
            horizon_labels=np.asarray(horizon_labels), horizon_steps=np.asarray(stops),
            continuation_keys=np.asarray(keys), recruit_masks=np.asarray(base_context["recruit_mask"]))
        for i, source in enumerate(selected):
            source_info = info[start + i]
            identity = {"trial": str(meta["trial"]), "source_id": int(source["source_id"]),
                        "source_episode": int(source["source_episode"]), "group": str(source["group"]),
                        "source_window": int(source["source_window"]), "source_time": int(source["state"].t),
                        "focal": int(source["focal"]),
                        "remaining_episode_steps": env.max_steps - int(source["state"].t),
                        **source_info}
            n = source_info["recruits"]
            for c, (target, commitment) in enumerate(specs[1:], 1):
                for h, label in enumerate(horizon_labels):
                    gain = (metrics[i, c, :, h, 0] - metrics[i, 0, :, h, 0]) / n
                    lo, hi = _interval(gain, rng)
                    row = {**identity, "target": target, "commitment_windows": commitment,
                        "horizon": label, "horizon_steps": stops[h], "mc_samples": mc_samples,
                        "initial_goal_predicted_gain": float(predicted[i, c] / n),
                        "prediction_matches_commitment": commitment == 1,
                        "actual_gain": float(gain.mean()), "actual_gain_ci_lo": lo, "actual_gain_ci_hi": hi,
                        "training_gae_gain": float(((gae[i, c] - gae[i, 0]) / n).mean()),
                        "condition_valid": target != "box_staging" or source_info["box_valid"]}
                    for m, name in enumerate(METRICS):
                        row[f"baseline_{name}"] = float(metrics[i, 0, :, h, m].mean())
                        row[f"branch_{name}"] = float(metrics[i, c, :, h, m].mean())
                    for m, name in enumerate(EXTRA_METRICS):
                        row[f"baseline_{name}"] = float(extra[i, 0, :, h, m].mean())
                        row[f"branch_{name}"] = float(extra[i, c, :, h, m].mean())
                        probability, delay = _event_stats(first[i, c, :, h, m])
                        row[f"branch_{name}_event_frac"] = probability
                        row[f"branch_{name}_first_step_if_event"] = delay
                    rows.append(row)
            # Paired factorial contrasts, beyond comparison with the factual policy.
            index = {spec: c for c, spec in enumerate(specs)}
            definitions = []
            for k in commitments:
                for target in TARGETS[1:]:
                    definitions.append(("target", target, k,
                        [(index[(target, k)], 1.), (index[("focal_position", k)], -1.)]))
            for target in TARGETS:
                for k in commitments:
                    if k == 1:
                        continue
                    definitions.append(("duration", target, k,
                        [(index[(target, k)], 1.), (index[(target, 1)], -1.)]))
                    if target != "focal_position":
                        definitions.append(("interaction", target, k,
                            [(index[(target, k)], 1.), (index[("focal_position", k)], -1.),
                             (index[(target, 1)], -1.), (index[("focal_position", 1)], 1.)]))
            for kind, target, k, terms in definitions:
                for h, label in enumerate(horizon_labels):
                    gain = sum(weight * metrics[i, c, :, h, 0] for c, weight in terms) / n
                    lo, hi = _interval(gain, rng)
                    contrasts.append({**identity, "contrast": kind, "target": target,
                        "commitment_windows": k, "horizon": label,
                        "gain": float(gain.mean()), "gain_ci_lo": lo, "gain_ci_hi": hi,
                        "condition_valid": target != "box_staging" or source_info["box_valid"]})
    return rows, contrasts


def _cluster_interval(frame, column, rng):
    blocks = [x[column].to_numpy() for _, x in frame.groupby("source_episode")]
    if len(blocks) < 2:
        return None, None
    sums, sizes = np.array([b.sum() for b in blocks]), np.array([len(b) for b in blocks])
    ids = rng.integers(0, len(blocks), (1000, len(blocks)))
    return tuple(float(x) for x in np.quantile(sums[ids].sum(1) / sizes[ids].sum(1), [.025, .975]))


def write_reports(out, rows, contrasts, metadata, tolerance, seed):
    data, comparison = pd.DataFrame(rows), pd.DataFrame(contrasts)
    summaries, contrast_summaries = [], []
    rng = np.random.default_rng(seed)
    for name, frame in [("all_sampled", data), *list(data.groupby("group"))]:
        for (trial, target, commitment, horizon), g in frame.groupby(
                ["trial", "target", "commitment_windows", "horizon"]):
            g = g[g.condition_valid]
            if g.empty:
                continue
            lo, hi = _cluster_interval(g, "actual_gain", rng)
            summaries.append({"trial": str(trial), "group": name, "target": target,
                "commitment_windows": int(commitment), "horizon": horizon,
                "n_sources": len(g), "n_source_episodes": g.source_episode.nunique(),
                "actual_gain_mean": float(g.actual_gain.mean()), "actual_gain_cluster_ci_lo": lo,
                "actual_gain_cluster_ci_hi": hi,
                "helpful_frac": float((g.actual_gain > tolerance).mean()),
                "harmful_frac": float((g.actual_gain < -tolerance).mean()),
                "tied_frac": float((g.actual_gain.abs() <= tolerance).mean()),
                "initial_goal_predicted_gain_mean": float(g.initial_goal_predicted_gain.mean()),
                "training_gae_gain_mean": float(g.training_gae_gain.mean()),
                "target_delivery_gain_mean": float((g.branch_target_delivery_events - g.baseline_target_delivery_events).mean()),
                "target_coupling_steps_gain_mean": float((g.branch_target_coupled_steps - g.baseline_target_coupled_steps).mean()),
                "recruit_contact_event_frac": float(g.branch_target_recruit_touch_steps_event_frac.mean()),
                "focal_contact_steps_mean": float(g.branch_target_focal_touch_steps.mean()),
                "target_coupling_event_frac": float(g.branch_target_coupled_steps_event_frac.mean())})
    for name, frame in [("all_sampled", comparison), *list(comparison.groupby("group"))]:
        for (trial, kind, target, commitment, horizon), g in frame.groupby(
                ["trial", "contrast", "target", "commitment_windows", "horizon"]):
            g = g[g.condition_valid]
            if g.empty:
                continue
            lo, hi = _cluster_interval(g, "gain", rng)
            contrast_summaries.append({"trial": str(trial), "group": name, "contrast": kind,
                "target": target, "commitment_windows": int(commitment), "horizon": horizon,
                "n_sources": len(g), "gain_mean": float(g.gain.mean()),
                "gain_cluster_ci_lo": lo, "gain_cluster_ci_hi": hi})
    for filename, frame in (("conditions.csv", data), ("contrasts.csv", comparison),
                            ("summary.csv", pd.DataFrame(summaries)),
                            ("contrast_summary.csv", pd.DataFrame(contrast_summaries))):
        frame.to_csv(out / filename, index=False)
    report = {"snapshots": metadata, "gain_tolerance": tolerance,
        "intervention": "Replan support at manager boundaries for 1/2/4 windows; fixed recruit identities/count. Focal/unselected goals remain manager goals. Staging releases recruits after selected-box delivery.",
        "staging": "Privileged routed rear staging with separated slots; switch to upward push when within 0.85 world units. Replanned only at manager boundaries.",
        "prediction": "Only the first joint goal is scored by the learned critic. Persistent-support scores are not predictions of the changed continuation policy.",
        "pairing": "Same full source states and per-replicate random streams for all conditions; source IDs retained after filtering.",
        "recruits": "Production DPP best count chosen once per source, unless a fixed --recruits count is requested; interventions tested regardless of bonus sign.",
        "returns": "Finite discounted environmental returns; GAE with learned bootstraps recorded separately. Episode limit unchanged.",
        "event_times": "First post-step event, relative to intervention start, conditional on event occurrence. No-event first steps are -1 in arrays and null in conditional means.",
        "contrasts": {"target": "target(K) minus focal_position(K)",
                      "duration": "target(K) minus target(1)",
                      "interaction": "[target(K)-focal_position(K)] minus [target(1)-focal_position(1)]"},
        "uncertainty": "Paired MC bootstrap at each source; source-episode clustered bootstrap within each training seed. Filtered stratified samples are not population estimates.",
        "summary": summaries, "contrast_summary": contrast_summaries}
    (out / "report.json").write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n")


def _commitments(value):
    try:
        values = sorted(set(int(x) for x in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("commitments must be positive")
    return sorted(set([1, *values]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-dir", type=Path, help="Directory containing test-3 trial_*_sources.npz files")
    inputs.add_argument("--snapshot", type=Path, help="A single complete source snapshot")
    parser.add_argument("--trials", help="Optional comma-separated saved trial labels")
    parser.add_argument("--groups", default="waiting_alone,understaffed", help="Comma-separated source strata, or all")
    parser.add_argument("--commitments", type=_commitments, default=_commitments("1,2,4"))
    parser.add_argument("--recruits", type=int, help="Fixed count instead of production DPP's best count")
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--gain-tolerance", type=float, default=1e-4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    for name in ("mc_samples", "chunk_size", "eval_steps", "recruits"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.gain_tolerance) or args.gain_tolerance < 0:
        parser.error("gain-tolerance must be finite and nonnegative")
    groups = None if args.groups == "all" else {x.strip() for x in args.groups.split(",") if x.strip()}
    if groups is not None and (not groups or not groups <= {"waiting_alone", "understaffed", "coalition", "other"}):
        parser.error("groups must name source strata or all")
    trials = None if args.trials is None else {x.strip() for x in args.trials.split(",") if x.strip()}
    paths = [args.snapshot] if args.snapshot else sorted(args.source_dir.glob("trial_*_sources.npz"))
    selected_paths = []
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            label = str(json.loads(str(file["metadata"]))["trial"])
        if trials is None or label in trials:
            selected_paths.append(path)
    if not selected_paths:
        parser.error("No matching source snapshots found")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or Path("plotting/feudal_goal_analysis") / f"dpp_support_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    rows, contrasts, metadata = [], [], []
    for trial_index, path in enumerate(selected_paths):
        print(f"Loading frozen sources {path}...", flush=True)
        env, config, ts, sources, meta = load_sources(path)
        original_ids = meta.get("source_ids", list(range(len(sources))))
        if len(original_ids) != len(sources):
            raise ValueError("Saved source-ID mapping has the wrong length")
        for source, sid in zip(sources, original_ids):
            source["source_id"] = int(sid)
        available = {g: sum(str(s["group"]) == g for s in sources) for g in
                     ("waiting_alone", "understaffed", "coalition", "other")}
        sources = [s for s in sources if groups is None or str(s["group"]) in groups]
        if not sources:
            print(f"  No requested strata in trial {meta['trial']}; skipping.", flush=True)
            continue
        required = ("_centre", "_extent", "_box_half", "world_height", "target_y")
        if not all(hasattr(env, name) for name in required):
            parser.error("Box staging requires the single-goal MultiBoxPushMJX world geometry")
        eval_steps = min(args.eval_steps or env.max_steps, env.max_steps)
        if eval_steps < env.max_steps and eval_steps % config.goal_horizon:
            parser.error("Artificial eval-steps cutoffs must contain whole manager windows")
        meta = {**meta, "trial_index": trial_index, "source_snapshot": str(path.resolve()),
                "source_ids": [int(s["source_id"]) for s in sources],
                "available_saved_strata": available, "selected_sources": len(sources),
                "selected_groups": sorted(groups) if groups is not None else "all",
                "commitments": args.commitments, "fixed_recruits": args.recruits,
                "mc_samples": args.mc_samples, "chunk_size": args.chunk_size,
                "eval_steps": eval_steps, "evaluation_backend": jax.default_backend()}
        print(f"  Trial {meta['trial']}: {len(sources)} sources, commitments {args.commitments}.", flush=True)
        save_sources(out / f"trial_{trial_index}_sources.npz", sources, ts, meta)
        result, comparison = evaluate_support(env, config, ts, sources, meta, out,
            args.commitments, args.mc_samples, args.chunk_size, eval_steps, args.recruits)
        rows.extend(result)
        contrasts.extend(comparison)
        metadata.append(meta)
        write_reports(out, rows, contrasts, metadata, args.gain_tolerance, 1234)
    if not rows:
        parser.error("No saved source states belong to the requested strata; try --groups all")
    print(f"Support target/commitment reports and paired outcomes: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
