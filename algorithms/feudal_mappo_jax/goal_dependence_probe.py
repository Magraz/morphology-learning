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
    constant             n                n             value of the goal's CONTENT
    zeroed              (absent)                        value of goal conditioning

``constant`` and ``zeroed`` both destroy pairing and conditioning, and the
difference between them is the whole point: ``constant`` still hands the worker
a goal-shaped vector of the usual magnitude, so it separates "the manager's
output carries information" from "the worker has co-adapted to a bias whose
removal is merely off-distribution". ``real ~ constant >> zeroed`` is the
degenerate outcome that every other diagnostic here reports as a healthy,
strongly-used goal channel.

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
    R_real ~ R_const >> R_zero the manager is DECORATIVE: one frozen vector
                               reproduces it. Note this presents as a large
                               `gap_zeroed`, i.e. as the headline SUCCESS
                               condition, unless `constant` is read too.
    R_real ~ R_zero            the channel is WORTH NOTHING: deleting it is
       but R_real > R_perm     free. A positive permutation gap alongside this
       and R_real > R_const    is COHERENCE damage, not value from the
                               assignment -- the worker was co-adapted to a goal
                               consistent with its own observation, and any
                               incoherent goal (wrong agent, wrong state, or
                               frozen) is an off-distribution hit. Measured on
                               mjx_12a_4o_4444_512/feudal_n01_local_private
                               (2026-09-17), which ties its matched
                               feudal_zerogoal control 281.4 vs 281.1.

ACCEPTANCE (conf/model/feudal_film.yaml, CLAUDE.md) therefore needs ALL THREE of
``gap_zeroed > 0``, ``gap_constant > 0`` and ``gap_permuted > 0``, each with a
paired CI excluding 0: the channel must EARN return, from the goal's CONTENT,
and specifically from the per-agent ASSIGNMENT. Necessary, not sufficient --
every variant perturbs an already-trained policy off-distribution, so use this
probe to rule arms OUT cheaply and the between-arm comparison against
``feudal_zerogoal`` to rule one IN.

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

# The variant names are shared with `trainer.eval_fn` and the metric-key
# suffixes; importing rather than re-listing them is what stops the three
# drifting (adding `constant` touched all three at once).
from algorithms.feudal_mappo_jax.manager import GOAL_VARIANTS  # noqa: E402


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
      ``f_goalhead`` instead, with NO ``f_percept_*``. Dividing by ``n_agents``
      here would silently produce ``goal_dim // n_agents`` and the reload would
      fail (or, worse for a width that happens to divide, load a different
      network than trained).
    * ``"local_global"``: as ``"local"``, but ``f_percept_*`` is present TOO (it
      feeds the goal path only). So those two are separated by the PAIR
      ``(f_enc_0, f_percept_0)``, not by either one alone.
    * ``"local_private"``: ``f_enc_0`` as well, but there is NO ``f_Mspace`` at
      all — the projection is the stacked per-agent ``f_Mspace_agent_kernel`` of
      shape ``(n_agents, manager_hidden, goal_dim)``, and the goal path is
      ``goal_head_agent_kernel`` rather than ``f_gpre``/``f_goalhead``. Checked
      FIRST, because it shares ``f_enc_0`` with the other two and reading
      ``f_Mspace`` would raise rather than mis-infer.
    * ``"local_global_private"``: as ``"local_private"``, but ``f_percept_*`` is
      present TOO. So the ``f_Mspace_agent_kernel`` branch must ALSO split on
      ``f_percept_0`` — the ``private`` and ``global`` axes are independent, and
      returning ``"local_private"`` for a tree that carries ``f_percept_*`` would
      build a manager whose target tree is missing those leaves, i.e. a
      ``from_bytes`` failure rather than a silent mis-load. Loud, but still
      wrong, and it fails at the arm you were trying to measure.

    ⚠ **`goal_dim` is read off the GOAL HEAD, never off ``f_Mspace``.** Those
    were the same width until ``manager_latent_dim`` split the manager's
    internal bottleneck from the goal (a grounded arm runs ``goal_dim=2`` with
    ``manager_latent_dim=32``). ``f_Mspace`` / ``f_Mspace_agent_kernel`` emit the
    bottleneck; the goal width lives in ``goal_head`` / ``f_goalhead`` /
    ``goal_head_agent_kernel``. Both are returned, and they are equal **iff** the
    arm ran ``goal_space="latent"`` — which is also the only way to tell a
    grounded checkpoint from a latent one.

    ⚠ **``position_direction`` and ``position_waypoint`` are INDISTINGUISHABLE
    from the tree.** They differ only in the objective and in what the channel
    pools, neither of which is a parameter. So a checkpoint tells you the goal
    space is grounded, not which grounded mode it was — record ``goal_space``
    and ``waypoint_radius`` in the provenance dict (as this module does) and do
    not try to infer them. Pinned by
    ``test_goal_space_is_partially_recoverable_from_the_checkpoint``.
    """
    from flax.serialization import msgpack_restore

    tree = msgpack_restore(path.read_bytes())
    mgr = tree["manager"]["params"]
    actor = tree["actor"]["params"]["MAPPOActor_0"]
    hidden_dim = int(actor["Dense_0"]["kernel"].shape[1])
    worker_in = int(actor["Dense_0"]["kernel"].shape[0])

    def _worker_encoder(manager_hidden_dim: int, goal_dim: int) -> str:
        """Did the worker read `f_enc(obs_i)` instead of the raw observation?

        Recoverable from the tree: the worker's first Dense takes `obs_dim`
        (FiLM) or `obs_dim + goal_width` (concat) normally, and the SAME two
        widths built on `manager_hidden_dim` when the encoder is shared. This
        function exists for the reason the whole `_dims_from_checkpoint` does —
        the yaml moves while checkpoints do not, and `worker_encoder` changes
        what network is evaluated.

        ⚠ Ambiguous if `obs_dim == manager_hidden_dim` (or the concat variants
        collide). Returns None there rather than guessing, so the caller keeps
        the composed config and the arm is not silently mismeasured.
        """
        shared = {manager_hidden_dim, manager_hidden_dim + goal_dim}
        if worker_in not in shared:
            return "none"
        # Only claim "shared" when the raw reading is impossible; `obs_dim` is
        # not in the tree, so a collision cannot be resolved here.
        return "shared"

    percept = "f_percept_0" in mgr

    # ⚠ `goal_dim` MUST come off the GOAL HEAD, not off `f_Mspace`.
    #
    # They were the same number until `manager_latent_dim` split the manager's
    # internal bottleneck from the goal width (grounded arms run goal_dim=2 with
    # latent_dim=32). `f_Mspace` emits the BOTTLENECK; only the goal head emits
    # the goal. Reading the old way would rebuild the target tree with the wrong
    # `goal_dim`, and `from_bytes` would fail at the very arm being measured —
    # the same failure this function's docstring records for
    # `local_global_private`.
    #
    # The two are equal iff `goal_space == "latent"`, which is how a grounded
    # checkpoint is recognized at all. See `_goal_space_from_dims`.
    if "goal_head_agent_kernel" in mgr:          # local_private / local_global_private
        goal_dim = int(mgr["goal_head_agent_kernel"].shape[2])
    elif "f_goalhead" in mgr:                    # local / local_global
        goal_dim = int(mgr["f_goalhead"]["kernel"].shape[1])
    else:                                        # centralized
        goal_dim = int(mgr["goal_head"]["kernel"].shape[1]) // int(n_agents)

    if "f_Mspace_agent_kernel" in mgr:
        # (n_agents, manager_hidden, manager_latent_dim)
        w = mgr["f_Mspace_agent_kernel"]
        manager_hidden_dim = int(w.shape[1])
        latent_dim = int(w.shape[2])
        manager_latent = "local_global_private" if percept else "local_private"
        local = True
    else:
        local = "f_enc_0" in mgr
        manager_latent = (
            ("local_global" if percept else "local") if local else "centralized"
        )
        manager_hidden_dim = int(
            mgr["f_enc_0" if local else "f_percept_0"]["kernel"].shape[1]
        )
        latent_dim = int(mgr["f_Mspace"]["kernel"].shape[1]) // (
            1 if local else int(n_agents)
        )

    return {
        "goal_dim": goal_dim,
        # None when the widths coincide, so a pre-split checkpoint resolves to
        # exactly the config it trained under rather than to an equivalent-but-
        # explicit one.
        "manager_latent_dim": None if latent_dim == goal_dim else latent_dim,
        "manager_hidden_dim": manager_hidden_dim,
        "hidden_dim": hidden_dim,
        "manager_latent": manager_latent,
        # ⚠ The worker's first Dense is `obs_dim + goal_dim` wide under concat,
        # so this needs the GOAL width, not the bottleneck — another reason the
        # two had to be separated above.
        "worker_encoder": _worker_encoder(manager_hidden_dim, goal_dim),
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
    for v in GOAL_VARIANTS:
        m, s = agg(f"return_mean_{v}")
        print(f"{v}={m:7.1f}+-{s:5.1f}  ", end="")
    print()
    print("    ep_len    ", end="")
    for v in GOAL_VARIANTS:
        m, _ = agg(f"length_{v}")
        print(f"{v}={m:6.1f}  ", end="")
    print()

    # Per-trial gaps with their paired CIs; * marks a CI excluding 0.
    for i, e in enumerate(entries):
        print(
            f"    trial {e['trial']}: "
            f"perm {_fmt_gap(e, 'permuted')}   "
            f"env {_fmt_gap(e, 'env_permuted')}"
        )
        print(
            f"    {'':>7}  "
            f"const {_fmt_gap(e, 'constant')}   "
            f"zero {_fmt_gap(e, 'zeroed')}"
        )

    for key, label in [
        ("d_cos_mean", "d_cos_mean"),
        ("d_cos_gap_env", "d_cos_gap_env  <- READ FIRST"),
        ("d_cos_gap_agent", "d_cos_gap_agent"),
        ("goal_perm_cos", "goal_perm_cos"),
        ("goal_direction_count_raw", "dir_count_raw"),
        ("goal_direction_count_pooled", "dir_count_pooled"),
        ("goal_concentration", "goal_concentration (1=frozen)"),
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
                        for v in GOAL_VARIANTS[1:]
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
