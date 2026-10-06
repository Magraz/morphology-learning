"""Frozen-rollout audit of DPP advantage normalization and manager gradients.

Collect once with the training collector, then compare coefficients and absolute
floors on the per-env team-advantage std. No parameters or optimizers are updated.
Gradients use the full-batch PPO actor objective at the checkpoint, before Adam
or gradient clipping; they are not the multi-epoch training update. Entropy is
reported separately and included in the total actor gradient.

    uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_normalization_probe \\
        --batch mjx_6a_4o_1024_gs_sparse \\
        --model simplified_feudal_tanh_relative_input_dpp --trials 0,1,2,3,4

The output directory contains summary.csv, streams.csv, report.json, and one
snapshot per trial/rollout. Replay a snapshot without running the simulator:

    uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_normalization_probe \\
        --snapshot PATH.npz --coefs 0,0.05,0.1,1 --std-floors 0,0.001,0.01
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes, to_bytes

from algorithms.mappo_jax.mappo import compute_gae
from algorithms.mappo_jax.network import MAPPOActor, evaluate_action
from algorithms.mappo_jax.trainer import RunnerState
from algorithms.mappo_jax.types import MAPPOConfig
from algorithms.simplified_feudal_mappo_jax import counterfactual as cf


def normalized_advantages(a_team, dpp, coef, std_floor=0.0):
    """Mirror PPO's corrected normalization; floor=0 is the training formula.

    Center over time, independently for each (env, agent), and divide by the
    TEAM std for that env (ddof=1). The floor is in raw advantage/return units.
    Both team and corrected terms use the selected denominator, so their
    difference isolates DPP from the effect of changing the normalization.
    """
    std = a_team.std(axis=0, ddof=1)
    denominator = jnp.maximum(std, std_floor) + 1e-8
    team = (a_team - a_team.mean(axis=0)) / denominator
    raw_bonus = coef * jnp.maximum(dpp, 0.0)
    agent_adv = a_team[..., None] + raw_bonus
    corrected = (agent_adv - agent_adv.mean(axis=0)) / denominator[None, :, None]
    team = jnp.broadcast_to(team[..., None], dpp.shape)
    return {
        "team": team,
        "corrected": corrected,
        "bonus": corrected - team,
        "raw_bonus": raw_bonus,
        "team_std": std,
        "denominator": denominator,
    }


def make_gradient_functions(actor_apply, actor_params, obs, action, old_lp,
                            active_mask, config, squash):
    """The same masked PPO surrogate and entropy as ppo_update, full batch."""
    obs = obs.reshape(-1, obs.shape[-1])
    action = action.reshape(-1, action.shape[-1])
    old_lp, mask = old_lp.reshape(-1), active_mask.reshape(-1)
    count = jnp.maximum(mask.sum(), 1.0)

    def policy_loss(params, advantages):
        lp, _ = evaluate_action(actor_apply, params, obs, action, False, squash=squash)
        ratio = jnp.exp(lp - old_lp)
        adv = jax.lax.stop_gradient(advantages.reshape(-1))
        surrogate = jnp.minimum(
            ratio * adv,
            jnp.clip(ratio, 1 - config.eps_clip, 1 + config.eps_clip) * adv,
        )
        return -(surrogate * mask).sum() / count

    def entropy_loss(params):
        _, entropy = evaluate_action(
            actor_apply, params, obs, action, False, squash=squash
        )
        return -(entropy * mask).sum() / count

    policy_grad = jax.jit(jax.value_and_grad(policy_loss))
    entropy_value, entropy_grad = jax.jit(jax.value_and_grad(entropy_loss))(actor_params)
    return policy_grad, entropy_value, entropy_grad


def _flat_gradient(tree):
    # Host float64 keeps diagnostic norms/dot products stable for large gradients.
    return np.concatenate([np.asarray(x, dtype=np.float64).ravel()
                           for x in jax.tree.leaves(tree)])


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / norm, -1, 1)) if norm > 0 else None


def _distribution(prefix, values):
    x = np.asarray(values, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    quantiles = np.quantile(finite, [0, .05, .5, .95, .99, 1]) if finite.size else [None] * 6
    out = {f"{prefix}_{key}": float(value) if value is not None else None
           for key, value in zip(["min", "p05", "p50", "p95", "p99", "max"], quantiles)}
    out[f"{prefix}_nonfinite_frac"] = float((~np.isfinite(x)).mean()) if x.size else None
    return out


def _signal_metrics(terms, mask):
    team, corrected, bonus, raw = [np.asarray(terms[k], dtype=np.float64)
                                  for k in ("team", "corrected", "bonus", "raw_bonus")]
    team, corrected, bonus, raw = [x[mask] for x in (team, corrected, bonus, raw)]
    rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    return {
        "n_active_decisions": int(team.size),
        "team_normalized_rms": rms(team),
        "corrected_normalized_rms": rms(corrected),
        "bonus_normalized_rms": rms(bonus),
        "bonus_to_team_rms": _ratio(rms(bonus), rms(team)),
        "sign_flip_frac": float((team * corrected < 0).mean()) if team.size else None,
        "bonus_exceeds_team_frac": float((np.abs(bonus) > np.abs(team)).mean()) if team.size else None,
        "raw_bonus_mean": float(raw.mean()) if raw.size else None,
        **_distribution("raw_bonus", raw),
        **_distribution("abs_normalized_bonus", np.abs(bonus)),
    }


def analyze_snapshot(snapshot, coefs, std_floors):
    """Return scenario summaries and one row per (scenario, env, agent)."""
    meta = snapshot["metadata"]
    config = MAPPOConfig(**meta["manager_config"])
    actor = MAPPOActor(action_dim=snapshot["action"].shape[-1],
                       hidden_dim=config.hidden_dim, discrete=False)
    target = actor.init(jax.random.PRNGKey(0), jnp.zeros(snapshot["obs"].shape[-1]))
    params = from_bytes(target, snapshot["actor_params"])
    data = {k: jnp.asarray(snapshot[k]) for k in
            ("obs", "action", "log_prob", "active_mask", "a_team", "dpp")}
    policy_grad, entropy_loss, entropy_grad = make_gradient_functions(
        actor.apply, params, data["obs"], data["action"], data["log_prob"],
        data["active_mask"], config, meta["action_bound"] == "tanh",
    )
    entropy = config.ent_coef * _flat_gradient(entropy_grad)
    mask = np.asarray(data["active_mask"]) > 0
    baseline = normalized_advantages(data["a_team"], data["dpp"], 0.0)
    _, baseline_grad = policy_grad(params, baseline["team"])
    baseline_grad = _flat_gradient(baseline_grad)
    summaries, streams = [], []
    identity = {k: meta[k] for k in ("batch", "model", "trial", "rollout", "seed")}
    for floor in std_floors:
        team_terms = normalized_advantages(data["a_team"], data["dpp"], 0.0, floor)
        _, team_gradient = policy_grad(params, team_terms["team"])
        team_gradient = _flat_gradient(team_gradient)
        for coef in coefs:
            terms = normalized_advantages(data["a_team"], data["dpp"], coef, floor)
            loss, gradient = policy_grad(params, terms["corrected"])
            gradient = _flat_gradient(gradient)
            bonus_gradient, total = gradient - team_gradient, gradient + entropy
            team_total = team_gradient + entropy
            total_norm = np.linalg.norm(total)
            row = {
                **identity, "coef": coef, "std_floor": floor,
                **_distribution("raw_team_advantage", data["a_team"]),
                **_distribution("raw_dpp", data["dpp"]),
                "dpp_positive_frac": float((np.asarray(data["dpp"])[mask] > 0).mean()),
                **_distribution("team_std", terms["team_std"]),
                **_distribution("denominator", terms["denominator"]),
                "floored_env_frac": float((np.asarray(terms["team_std"]) < floor).mean()),
                **_signal_metrics(terms, mask),
                "policy_loss": float(loss), "entropy_loss": float(entropy_loss),
                "team_policy_grad_norm": float(np.linalg.norm(team_gradient)),
                "policy_grad_norm": float(np.linalg.norm(gradient)),
                "bonus_grad_norm": float(np.linalg.norm(bonus_gradient)),
                "weighted_entropy_grad_norm": float(np.linalg.norm(entropy)),
                "total_actor_grad_norm": float(total_norm),
                "bonus_to_team_grad_norm": _ratio(np.linalg.norm(bonus_gradient), np.linalg.norm(team_gradient)),
                "bonus_to_entropy_grad_norm": _ratio(np.linalg.norm(bonus_gradient), np.linalg.norm(entropy)),
                "total_grad_norm_ratio_unfloored_team": _ratio(total_norm, np.linalg.norm(baseline_grad + entropy)),
                "policy_grad_cos_team": _cosine(gradient, team_gradient),
                "bonus_grad_cos_team": _cosine(bonus_gradient, team_gradient),
                "total_grad_cos_team": _cosine(total, team_total),
                "policy_grad_cos_unfloored_team": _cosine(gradient, baseline_grad),
                "total_grad_cos_unfloored_team": _cosine(total, baseline_grad + entropy),
                "full_batch_clip_scale": min(1.0, config.grad_clip / total_norm) if total_norm > 0 else 1.0,
            }
            summaries.append(row)
            host_terms = {k: np.asarray(v) for k, v in terms.items()}
            for env in range(mask.shape[1]):
                for agent in range(mask.shape[2]):
                    local = {k: v[:, env:env+1, agent:agent+1]
                             for k, v in host_terms.items() if v.ndim == 3}
                    streams.append({
                        **identity, "coef": coef, "std_floor": floor,
                        "env": env, "agent": agent,
                        "team_std": float(host_terms["team_std"][env]),
                        "denominator": float(host_terms["denominator"][env]),
                        **_signal_metrics(local, mask[:, env:env+1, agent:agent+1]),
                    })
    return summaries, streams


def collect_snapshot(env, config, train_state, key, metadata, collect_fn=None):
    """Use the real training collector, including timeout bootstraps and resets."""
    from algorithms.simplified_feudal_mappo_jax.trainer import make_train

    if collect_fn is None:
        _, collect_fn, _, _, _ = make_train(config, env)
    _, rollout, last, _ = collect_fn(RunnerState(train_state=train_state, rng=key))
    manager, goal = rollout.manager, rollout.manager_goal
    a_team, _ = compute_gae(manager.reward, manager.value, manager.done.astype(jnp.float32),
                           last.manager, config.manager.gamma, config.manager.gae_lambda)
    dpp, best_n = jax.jit(cf.dpp_credit, static_argnames=("radius", "max_recruits"))(
        train_state.manager_adv, manager.global_state, goal.offset, goal.pos,
        radius=config.waypoint_radius, max_recruits=config.dpp_max_recruits,
    )
    return {
        "metadata": {**metadata, "manager_config": dataclasses.asdict(config.manager),
                     "action_bound": config.manager_action_bound,
                     "goal_horizon": config.goal_horizon,
                     "waypoint_radius": config.waypoint_radius,
                     "dpp_max_recruits": config.dpp_max_recruits,
                     "trained_dpp_coef": config.dpp_coef,
                     "backend": jax.default_backend()},
        "actor_params": to_bytes(train_state.manager.actor_ts.params),
        **{k: np.asarray(getattr(manager, k)) for k in
           ("obs", "action", "log_prob", "active_mask", "reward", "value", "done", "team_reward")},
        "last_value": np.asarray(last.manager),
        "a_team": np.asarray(a_team), "dpp": np.asarray(dpp), "best_n": np.asarray(best_n),
    }


def save_snapshot(path, snapshot):
    np.savez_compressed(path,
        **{k: v for k, v in snapshot.items() if k not in ("metadata", "actor_params")},
        metadata=np.array(json.dumps(snapshot["metadata"])),
        actor_params=np.frombuffer(snapshot["actor_params"], dtype=np.uint8))


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        snapshot = {k: data[k].copy() for k in data.files if k not in ("metadata", "actor_params")}
        snapshot["metadata"] = json.loads(str(data["metadata"]))
        snapshot["actor_params"] = data["actor_params"].tobytes()
    return snapshot


def _numbers(value):
    try:
        result = list(dict.fromkeys(float(x) for x in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result or any(not math.isfinite(x) or x < 0 for x in result):
        raise argparse.ArgumentTypeError("numbers must be finite and nonnegative")
    return result


def _clean_json(value):
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_reports(out, summaries, streams, metadata):
    for filename, rows in (("summary.csv", summaries), ("streams.csv", streams)):
        with (out / filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(_clean_json(row) for row in rows)
    report = {
        "gradient_scope": "Frozen full-batch PPO surrogate at the checkpoint; before clipping/Adam. No updates.",
        "normalization": "Center over time per (env, agent); divide by max(team_std_ddof1, std_floor) + 1e-8.",
        "std_floor_units": "Raw team advantage / discounted return units; absolute, not reward-scale invariant.",
        "comparison": "Each coefficient/floor uses identical rollout, GAE, DPP predictions, and actor parameters.",
        "zero_norm_metrics": "Ratios and cosines with a zero reference norm are null, not zero.",
        "snapshots": metadata, "scenarios": summaries,
    }
    (out / "report.json").write_text(json.dumps(_clean_json(report), indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", default="mjx_6a_4o_1024_gs_sparse")
    parser.add_argument("--model", default="simplified_feudal_tanh_relative_input_dpp")
    parser.add_argument("--trials", default="0,1,2,3,4")
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, help="Override collection length, rounded up to whole goal windows")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--coefs", type=_numbers, default=_numbers("0,0.1,1"))
    parser.add_argument("--std-floors", type=_numbers, default=_numbers("0,0.0001,0.001,0.01"))
    parser.add_argument("--snapshot", type=Path, help="Replay saved data; batch/model/collection arguments are unused")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.n_envs < 1 or args.rollouts < 1 or (args.rollout_steps is not None and args.rollout_steps < 1):
        parser.error("n-envs, rollouts and rollout-steps must be positive")
    # Include both anchors even if the user requests only intermediate settings.
    coefs, floors = list(dict.fromkeys([0.0, *args.coefs])), list(dict.fromkeys([0.0, *args.std_floors]))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or Path("plotting/feudal_goal_analysis") / f"dpp_normalization_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    summaries, streams, metadata = [], [], []

    def analyze(snapshot, name):
        save_snapshot(out / name, snapshot)
        print(f"Analyzing {name} on {jax.default_backend()}...", flush=True)
        rows, local = analyze_snapshot(snapshot, coefs, floors)
        summaries.extend(rows)
        streams.extend(local)
        metadata.append(snapshot["metadata"])
        write_reports(out, summaries, streams, metadata)
        for row in rows:
            ratio = row["bonus_to_team_grad_norm"]
            cosine = row["policy_grad_cos_team"]
            ratio_text = f"{ratio:.3g}" if ratio is not None else "undefined"
            cosine_text = f"{cosine:.3f}" if cosine is not None else "undefined"
            print(f"  coef={row['coef']:g} floor={row['std_floor']:g} "
                  f"bonus_rms={row['bonus_normalized_rms']:.3g} "
                  f"sign_flips={row['sign_flip_frac']:.1%} "
                  f"bonus/team_grad={ratio_text} cos_team={cosine_text}", flush=True)

    if args.snapshot:
        analyze(load_snapshot(args.snapshot), "replayed.npz")
    else:
        from algorithms.simplified_feudal_mappo_jax.dpp_probe import load_arm
        from algorithms.simplified_feudal_mappo_jax.trainer import make_train

        trials = [x.strip() for x in args.trials.split(",") if x.strip()]
        if not trials:
            parser.error("trials must contain at least one trial")
        for trial_index, trial in enumerate(trials):
            print(f"Loading {args.batch}/{args.model}/{trial}...", flush=True)
            env, config, ts, path = load_arm(args.batch, args.model, trial, args.n_envs)
            if args.rollout_steps is not None:
                steps = math.ceil(args.rollout_steps / config.goal_horizon) * config.goal_horizon
                if steps < 2 * config.goal_horizon:
                    parser.error("rollout-steps must allow at least two manager windows")
                config = dataclasses.replace(config,
                    worker=dataclasses.replace(config.worker, n_steps=steps),
                    manager=dataclasses.replace(config.manager, n_steps=steps))
            seed = args.seed + (int(trial) if trial.isdigit() else trial_index)
            checkpoint_meta = {
                "batch": args.batch, "model": args.model, "trial": trial,
                "seed": seed, "checkpoint": str(path),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n_envs": args.n_envs, "rollout_steps": config.worker.n_steps,
            }
            _, collect_fn, _, _, _ = make_train(config, env)
            for rollout in range(args.rollouts):
                print(f"Collecting frozen rollout {rollout} ({config.worker.n_steps} steps/env)...", flush=True)
                key = jax.random.fold_in(jax.random.PRNGKey(seed), rollout)
                snapshot = collect_snapshot(env, config, ts, key,
                    {**checkpoint_meta, "rollout": rollout}, collect_fn=collect_fn)
                analyze(snapshot, f"trial_{trial_index}_rollout_{rollout}.npz")
    print(f"Reports and replayable snapshots: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
