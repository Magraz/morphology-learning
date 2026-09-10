"""Are the feudal manager's goals actually USEFUL? — offline, over many arms.

Every diagnostic this stack logs during training is a **collapse detector**:
``goal_direction_count``, ``goal_pairwise_cos(_abs)``, ``state_latent_erank``,
``d_cos_var``, ``worker_goal_column_ratio``. They answer "are the goals
well-formed?", which is a necessary condition and nothing more. Two of them
mislead outright if read as usefulness — ``d_cos_mean`` is uninterpretable
because the manager owns *both* arguments of the cosine, and
``worker_goal_column_ratio`` already misled once (it fell while the goal block
grew 2.2-4.7x; CLAUDE.md records the correction).

This module measures usefulness directly, on **already-trained checkpoints**, so
the question can be answered without spending another 1e8-step run per arm.

THE METHOD is a permutation that preserves the goal distribution EXACTLY and
destroys exactly one property, so the real-minus-null gap isolates that property:

    variant        agent pairing   state conditioning   what a gap means
    real                 y                y             (the reference)
    permuted             n                y             value of the ASSIGNMENT
    env_permuted         y                n             value of STATE-CONDITIONING
    zeroed               n                n             value of goal conditioning

READ ``env_permuted`` FIRST. A manager that has degenerated into a fixed
per-agent code (an agent-ID label with no dependence on `s_t`) scores a LARGE
agent-permutation gap while doing nothing the hierarchy exists for — and the
existing suite calls that state healthy, since ``goal_direction_count`` reads
~8.25 against a random-direction baseline of 8.26. If the goals do not depend on
the state, the agent-pairing numbers are uninterpretable however large they are.

Two measurements per arm:

* **behavioural** — deterministic eval under each variant, all in ONE scan with
  tiled reset keys, so episode `j` of every block starts from a bit-identical
  state and the gap is a PAIRED statistic (see ``trainer.eval_fn``).
* **latent** — ``d_cos`` against the same two nulls, computed straight off a
  rollout's stored ``goal`` / ``state_latent``. No update, no extra network
  forward: the trajectory already carries everything.

HOW TO READ THE EVAL GAP — the asymmetry is not a caveat, it is the result's
shape. A permuted rollout is off-policy twice (mispaired input, and it then
visits different states), so the gap OVERSTATES the causal value of correct
assignment relative to a manager trained to emit permuted goals.

    gap ~ 0   STRONG negative — the return does not depend on the pairing even
              under a maximally disruptive re-pairing.
    gap > 0   WEAK evidence of dependence. Its magnitude is NOT "the value of
              hierarchy".

    R_real ~ R_perm ~ R_zero   worker ignores the goal; the hierarchy is
                               decorative (the concat degeneracy worker.py warns
                               about).
    R_real ~ R_perm > R_zero   worker uses goal CONTENT but not the ASSIGNMENT.
                               Invisible without the zeroed arm, and the likeliest
                               outcome given goal_pairwise_cos ~ -0.02 alongside
                               0.95-0.99 temporal collinearity.
    R_real > R_perm ~ R_zero   the pairing carries the content — the headline
                               positive.
    R_perm < R_zero            a wrong goal is worse than none: actively
                               misleading directives.

A collapsed-goal manager ALSO gives gap ~ 0, because the permutation is then
nearly the identity. That is correct but ambiguous alone, so ``goal_perm_cos``
(the mean cosine between a goal and the one it was swapped with) is reported
next to every gap: ~1 means the permutation changed nothing and the zero gap
says only that the goals were already identical.

POSITIVE CONTROL, and the stop-the-line check: on a ``feudal_zerogoal`` arm the
worker zeroes the goal inside the module, so all variants coincide and every
measured gap must be **exactly 0.0**. Anything else means the harness is wrong
and no other number here is worth reading.

Run:
    MUJOCO_GL=egl uv run python -m algorithms.feudal_mappo_jax.goal_dependence_probe \\
        --batches mjx_16a_4o_trunc_1024 --models feudal,feudal_zerogoal \\
        --trials 0,1,2 --n-eval-episodes 64 --shifts 1,8
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose(batch: str, model: str, trial: str):
    """Compose one arm's Hydra config exactly as `train.py` would.

    Deliberately NOT the `global_state_probe.py` pattern of rebuilding the env
    from CLI flags (``--hidden-dim 168  help="conf/model/mlp.yaml"``): a
    hand-copied yaml value is precisely how you end up silently measuring a
    different network than the one that trained. `_build_dispatch_args` is
    already factored out of `@hydra.main` for exactly this kind of harness.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from train import _build_dispatch_args

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=feudal_mappo_jax",
                f"env={batch}",
                f"model={model}",
                f"trial_id={trial}",
            ],
        )
    return _build_dispatch_args(cfg, {"env": batch, "model": model})


def _checkpoint_path(batch: str, model: str, trial: str) -> Path | None:
    d = REPO_ROOT / "experiments" / "results" / batch / model / str(trial) / "models"
    for name in ("models_finished.msgpack", "models_checkpoint.msgpack"):
        if (d / name).exists():
            return d / name
    return None


def _dims_from_checkpoint(path: Path, n_agents: int) -> dict:
    """Read the network widths back off the saved params.

    NECESSARY, not defensive. `_load_train_state` rebuilds the networks from the
    CURRENT yaml, but the yaml moves while checkpoints do not: commit 44c3af0
    changed ``goal_dim`` 16 -> 32 *after* every existing feudal arm was trained,
    so composing today's config for those runs fails to load at all (and, for an
    unparameterized setting like ``normalize_pooled_goal``, would silently
    evaluate a different function instead).

    Inferring the widths from the params makes the measurement a property of the
    checkpoint rather than of whatever the config happens to say this week.

    ``goal_dim`` comes from ``f_Mspace`` rather than from the actor's first layer,
    because with ``goal_embed_dim`` set the actor sees the embedding width, not
    the goal width. Its shape depends on ``manager_latent``, which is ALSO read
    back off the tree here:

    * ``"centralized"``: ``f_Mspace`` is ``(manager_hidden, n_agents*goal_dim)``
      and the tree carries ``f_percept_0``/``goal_head``.
    * ``"local"``: ``f_Mspace`` is ``(manager_hidden, goal_dim)`` — one SHARED
      per-agent projection — and the tree carries ``f_enc_0``/``f_gpre``/
      ``f_goalhead`` instead. Dividing by ``n_agents`` here would silently
      produce ``goal_dim // n_agents`` and the reload would fail (or, worse for a
      width that happens to divide, load a different network than trained).
    """
    from flax.serialization import msgpack_restore

    tree = msgpack_restore(path.read_bytes())
    mgr = tree["manager"]["params"]
    actor = tree["actor"]["params"]["MAPPOActor_0"]
    local = "f_enc_0" in mgr
    return {
        "goal_dim": int(mgr["f_Mspace"]["kernel"].shape[1])
        // (1 if local else int(n_agents)),
        "manager_hidden_dim": int(
            mgr["f_enc_0" if local else "f_percept_0"]["kernel"].shape[1]
        ),
        "hidden_dim": int(actor["Dense_0"]["kernel"].shape[1]),
        "manager_latent": "local" if local else "centralized",
    }


def _runner(batch: str, model: str, trial: str, quiet: bool = False):
    """Build the Feudal runner for one arm, mirroring `algorithms._dispatch`.

    Then reconcile the config's network widths with the checkpoint's actual
    ones, so a yaml that has moved since training cannot change what is measured.
    """
    from dataclasses import replace

    from algorithms.feudal_mappo_jax.run import Feudal_MAPPO_JAX_Runner
    from algorithms.feudal_mappo_jax.types import Experiment

    args = _compose(batch, model, trial)
    exp_config = Experiment(**args["exp_dict"])
    runner = Feudal_MAPPO_JAX_Runner(
        exp_config.device,
        args["batch_dir"],
        args["results_dir"],
        args["trial_id"],
        False,  # checkpoint: we load params explicitly, not the train state
        exp_config,
        args["env_config"],
    )

    path = _checkpoint_path(batch, model, trial)
    if path is not None:
        dims = _dims_from_checkpoint(path, runner.env.n_agents)
        drift = {
            k: (getattr(runner.config, k), v)
            for k, v in dims.items()
            if getattr(runner.config, k) != v
        }
        if drift and not quiet:
            print(
                f"    config drift vs checkpoint (using the CHECKPOINT's values): "
                + ", ".join(f"{k}: yaml={a} ckpt={b}" for k, (a, b) in drift.items())
            )
        runner.config = replace(runner.config, **dims)
    return runner


def _has_checkpoint(batch: str, model: str, trial: str) -> bool:
    d = REPO_ROOT / "experiments" / "results" / batch / model / str(trial) / "models"
    return (d / "models_finished.msgpack").exists() or (
        d / "models_checkpoint.msgpack"
    ).exists()


def _fmt_gap(entry, variant):
    lo, hi = entry[f"gap_{variant}_ci"]
    star = "" if (lo <= 0.0 <= hi) else "*"  # * = CI excludes 0
    return f"{entry[f'gap_{variant}']:+8.2f} [{lo:+7.2f},{hi:+7.2f}]{star}"


def _print_arm(batch, model, shift, entries):
    """One table per (batch, model, shift), aggregated over trials."""
    print(f"\n=== {batch} / {model}   shift={shift}   n_trials={len(entries)}")
    if not entries:
        print("    (no trials with a checkpoint)")
        return

    def agg(key):
        vals = np.array([e[key] for e in entries], dtype=np.float64)
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return vals.mean(), sem

    print("    returns   ", end="")
    for v in ("real", "permuted", "env_permuted", "zeroed"):
        m, s = agg(f"return_mean_{v}")
        print(f"{v}={m:7.1f}+-{s:5.1f}  ", end="")
    print()
    print("    ep_len    ", end="")
    for v in ("real", "permuted", "env_permuted", "zeroed"):
        m, _ = agg(f"length_{v}")
        print(f"{v}={m:6.1f}  ", end="")
    print()

    # Per-trial gaps with their paired CIs; * marks a CI excluding 0.
    for i, e in enumerate(entries):
        print(
            f"    trial {e['trial']}: "
            f"perm {_fmt_gap(e, 'permuted')}   "
            f"env {_fmt_gap(e, 'env_permuted')}   "
            f"zero {_fmt_gap(e, 'zeroed')}"
        )

    for key, label in [
        ("d_cos_mean", "d_cos_mean"),
        ("d_cos_gap_env", "d_cos_gap_env  <- READ FIRST"),
        ("d_cos_gap_agent", "d_cos_gap_agent"),
        ("goal_perm_cos", "goal_perm_cos"),
        ("goal_direction_count_raw", "dir_count_raw"),
        ("goal_direction_count_pooled", "dir_count_pooled"),
    ]:
        m, s = agg(key)
        print(f"    {label:32s} {m:+8.4f} +- {s:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--batches", required=True, help="comma-separated env groups")
    p.add_argument("--models", default="feudal,feudal_n01,feudal_n05,feudal_zerogoal")
    p.add_argument("--trials", default="0,1,2")
    p.add_argument(
        "--n-eval-episodes",
        type=int,
        default=64,
        help="episodes per variant block. Width, not depth, so nearly free; "
        "MJX memory scales with it and training only ever uses 32.",
    )
    p.add_argument(
        "--shifts",
        default="1",
        help="permutation shifts to sweep. If results agree across shifts, the "
        "choice of permutation is not load-bearing; if they differ, there is "
        "agent-index structure worth knowing about.",
    )
    p.add_argument("--out", default="algorithms/feudal_mappo_jax/goal_dependence_probe")
    args = p.parse_args()

    batches = [b for b in args.batches.split(",") if b]
    models = [m for m in args.models.split(",") if m]
    trials = [t for t in args.trials.split(",") if t]
    shifts = [int(s) for s in args.shifts.split(",") if s]

    all_results = {}
    t0 = time.time()

    for batch in batches:
        for model in models:
            present = [t for t in trials if _has_checkpoint(batch, model, t)]
            if not present:
                print(f"\n=== {batch} / {model}: no checkpoints, skipping")
                continue

            # Build make_train ONCE per (batch, model): every trial of an arm
            # shares the env and config, so this is 1 compile instead of N.
            shared = None
            per_shift = {s: [] for s in shifts}
            for i, trial in enumerate(present):
                runner = _runner(batch, model, trial, quiet=i > 0)
                if shared is None:
                    from algorithms.feudal_mappo_jax.trainer import make_train
                    from dataclasses import replace as _replace

                    shared = make_train(
                        _replace(
                            runner.config, n_eval_episodes=args.n_eval_episodes
                        ),
                        runner.env,
                    )
                res = runner.goal_dependence(
                    n_eval_episodes=args.n_eval_episodes,
                    shifts=shifts,
                    make_train_out=shared if len(shifts) == 1 else None,
                )
                all_results[(batch, model, trial)] = res
                for s in shifts:
                    e = dict(res["by_shift"][s])
                    e["trial"] = trial
                    for v, r in e.pop("return_mean").items():
                        e[f"return_mean_{v}"] = r
                    for v, l in e.pop("lengths").items():
                        e[f"length_{v}"] = l
                    per_shift[s].append(e)

            for s in shifts:
                _print_arm(batch, model, s, per_shift[s])
                if model == "feudal_zerogoal":
                    bad = [
                        (e["trial"], v, e[f"gap_{v}"])
                        for e in per_shift[s]
                        for v in ("permuted", "env_permuted", "zeroed")
                        if e[f"gap_{v}"] != 0.0
                    ]
                    if bad:
                        print(
                            "    !!! POSITIVE CONTROL FAILED: feudal_zerogoal zeroes "
                            "the goal inside the worker, so every variant must "
                            f"coincide and every gap must be exactly 0.0. Got {bad}. "
                            "The harness is wrong — do not read any other number."
                        )
                    else:
                        print("    positive control OK (all gaps exactly 0.0)")

    out = Path(args.out).with_suffix(".results.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nWrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
