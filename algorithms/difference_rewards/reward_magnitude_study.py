"""Magnitude of the reward **stored per transition** in each `macro_mjx_16a_4o`
arm — i.e. the value the critic/actor actually learn from.

Motivation. The `_dr` (single-step) and `_wdr` (windowed) difference-reward arms
train *worse* than the dense team-reward baseline. To see why, compare the
magnitude of the learning signal each arm stores in its transitions:

- **dense**  → the transition reward is the **team scalar `G`** (one value per
  macro decision, shared by every agent through the shared critic + GAE).
- **`_dr`**  → the transition reward is the **per-agent `Dᵢ`** = sum of the base
  env's exact single-step difference rewards over the macro window.
- **`_wdr`** → the transition reward is the **per-agent `Dᵢ`** = the windowed
  counterfactual `G(window) − G₋ᵢ(window)`.

These are exactly `Transition.reward` in each arm (dense: `(n_envs,)`; DR:
`(n_envs, n_agents)`). We roll out the **dense-trained** policies (so the
trajectories are the good ones), read all three rewards off the *same*
trajectory (base physics is independent of `reward_mode`), and plot the signed
mean of each. One plot.

Run:
    MUJOCO_GL=egl uv run python -u -m \
        algorithms.difference_rewards.reward_magnitude_study --n-rollouts 5 --chunk 8
    # re-plot from cached data without recomputing:
    uv run python -m algorithms.difference_rewards.reward_magnitude_study \
        --from-cache algorithms/difference_rewards/reward_magnitude.data.pkl
"""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax.serialization import from_bytes

from algorithms.mappo_jax.network import MAPPOActor, MAPPOCritic
from environments.mjx_suite.macro_skills import N_SKILLS
from environments.mjx_suite.macro_wrapper import (
    WINDOWED_DIFFERENCE_REWARDS,
    SyncMacroMJX,
)
from environments.mjx_suite.multi_box_push_mjx import MultiBoxPushMJX
from environments.mjx_suite.observation import OBS_DIM

RESULTS = Path("experiments/results/macro_mjx_16a_4o/mlp")
N_AGENTS = 16
N_OBJECTS = 4
MACRO_LEN = 20
HIDDEN_DIM = 168  # conf/model/mlp.yaml


# --------------------------------------------------------------------------- io
def load_actor_params(trial_dir: Path):
    """Restore the actor params for one trial from its msgpack checkpoint.

    Mirrors ``MAPPO_JAX_Runner._load_train_state``: prefers ``models_finished``,
    falls back to ``models_checkpoint``. The file stores {"actor", "critic"}, so
    we build both init targets (critic is n_outputs=1 in the dense arm) and keep
    the actor.
    """
    path = trial_dir / "models" / "models_finished.msgpack"
    if not path.exists():
        path = trial_dir / "models" / "models_checkpoint.msgpack"
    actor = MAPPOActor(action_dim=N_SKILLS, hidden_dim=HIDDEN_DIM, discrete=True)
    critic = MAPPOCritic(hidden_dim=2 * HIDDEN_DIM, n_outputs=1)
    target = {
        "actor": actor.init(jax.random.PRNGKey(0), jnp.zeros(OBS_DIM)),
        "critic": critic.init(jax.random.PRNGKey(1), jnp.zeros(OBS_DIM * N_AGENTS)),
    }
    with open(path, "rb") as f:
        loaded = from_bytes(target, f.read())
    return actor, loaded["actor"], path.name


# ------------------------------------------------------------------- rollout fn
def build_rollout(sync_dense, sync_dr, sync_wdr, actor):
    """One deterministic macro-rollout emitting the three stored transition
    rewards per macro decision: team ``G`` (dense), per-agent ``D_ts`` (timestep
    DR), per-agent ``D_w`` (windowed DR). Drives the canonical trajectory with
    ``sync_dense`` and reads the DR rewards off the *same* state / skills; freezes
    after the first done and returns a ``valid`` mask so terminated steps drop
    out.
    """
    horizon = sync_dense.max_steps

    def rollout(actor_params, key):
        obs, state = sync_dense.reset(key)

        def scan_step(carry, _):
            state, obs, done = carry
            logits = actor.apply(actor_params, obs)  # (A, K)
            skills = jnp.argmax(logits, axis=-1).astype(jnp.int32)  # (A,)

            obs_next, next_state, G, term, trunc, _ = sync_dense.step(state, skills)
            _, _, D_ts, _, _, _ = sync_dr.step(state, skills)  # (A,) timestep DR
            _, _, D_w, _, _, _ = sync_wdr.step(state, skills)  # (A,) windowed DR

            valid = ~done
            frozen = jax.tree.map(lambda n, c: jnp.where(done, c, n), next_state, state)
            new_done = done | term | trunc
            return (frozen, obs_next, new_done), (G, D_ts, D_w, valid)

        _, recs = jax.lax.scan(
            scan_step, (state, obs, jnp.zeros((), bool)), None, length=horizon
        )
        return recs  # (G (T,), D_ts (T,A), D_w (T,A), valid (T,))

    return jax.jit(jax.vmap(rollout, in_axes=(None, 0)))


# ----------------------------------------------------------------- aggregation
def collect_trial(recs):
    """Pool valid steps of a (n_rollouts,) batch into flat stored-reward arrays.

    ``G`` per transition (team scalar) and the per-agent ``D_i`` for each DR flavour
    (flattened over transitions × agents — each is one stored per-agent reward).
    """
    G, D_ts, D_w, valid = (np.asarray(x) for x in recs)  # (E,T),(E,T,A),(E,T,A),(E,T)
    m = valid.astype(bool)
    return dict(
        G=G[m],  # (n_transitions,) team reward stored (dense arm)
        Di_ts=D_ts[m].reshape(-1),  # (n_transitions*A,) per-agent reward stored (_dr)
        Di_w=D_w[m].reshape(-1),  # (n_transitions*A,) per-agent reward stored (_wdr)
    )


# ------------------------------------------------------------------- plotting
C_ENV, C_TS, C_WDR = "#1f77b4", "#ff7f0e", "#2ca02c"


def make_figure(per_trial, trials, out):
    """One plot: signed mean magnitude of the reward stored per transition in
    each arm — dense team ``G`` vs per-agent ``D_i`` (timestep) vs per-agent
    ``D_i`` (windowed)."""
    G = np.concatenate([t["G"] for t in per_trial])
    Di_ts = np.concatenate([t["Di_ts"] for t in per_trial])
    Di_w = np.concatenate([t["Di_w"] for t in per_trial])

    means = [G.mean(), Di_ts.mean(), Di_w.mean()]
    labels = [
        "dense arm\nteam $G$",
        "timestep-DR arm\nper-agent $D_i$",
        "windowed-DR arm\nper-agent $D_i$",
    ]
    colors = [C_ENV, C_TS, C_WDR]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    bars = ax.bar(range(3), means, color=colors, alpha=0.9, width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("mean reward stored per transition (signed, log)")
    ax.set_title(
        "macro_mjx_16a_4o — magnitude of the reward stored per transition\n"
        f"(dense-trained policies, {len(trials)} trials; the signal each arm learns from)"
    )
    g = G.mean()
    ax.text(
        0.98,
        0.03,
        f"per-agent $D_i$ vs dense team $G$:\n"
        f"   timestep {Di_ts.mean() / g:.3f}×   ({g / Di_ts.mean():.0f}× smaller)\n"
        f"   windowed {Di_w.mean() / g:.3f}×   ({g / Di_w.mean():.1f}× smaller)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nsaved figure -> {out}")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rollouts", type=int, default=5)
    ap.add_argument(
        "--trials",
        type=int,
        nargs="*",
        default=None,
        help="trial ids (default: all found under the batch)",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=8,
        help="rollouts run concurrently per vmapped batch (caps GPU memory; the "
        "windowed DR forks n_agents counterfactuals per rollout, so peak width is "
        "chunk * n_agents MJX sims). Total = n_rollouts in ceil(n_rollouts/chunk) batches.",
    )
    ap.add_argument(
        "--from-cache",
        type=str,
        default=None,
        help="skip compute and re-plot from a cached .data.pkl (written each run).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=str, default="algorithms/difference_rewards/reward_magnitude.png"
    )
    args = ap.parse_args()

    if args.from_cache:
        with open(args.from_cache, "rb") as f:
            blob = pickle.load(f)
        per_trial, trials = blob["per_trial"], blob["trials"]
        print(f"loaded cached data from {args.from_cache} ({len(trials)} trials)")
    else:
        if args.trials is None:
            trials = sorted((p.name for p in RESULTS.iterdir() if p.is_dir()), key=int)
        else:
            trials = [str(t) for t in args.trials]

        # Three env views sharing macro_len=20 (base physics is identical; only the
        # reward read-out differs): dense drives + gives G, a difference_rewards base
        # gives the timestep D_i, a dense base in windowed mode gives the windowed D_i.
        base_dense = MultiBoxPushMJX(N_AGENTS, N_OBJECTS, reward_mode="dense")
        base_dr = MultiBoxPushMJX(N_AGENTS, N_OBJECTS, reward_mode="difference_rewards")
        sync_dense = SyncMacroMJX(base_dense, macro_len=MACRO_LEN)
        sync_dr = SyncMacroMJX(base_dr, macro_len=MACRO_LEN)
        sync_wdr = SyncMacroMJX(
            base_dense, macro_len=MACRO_LEN, reward_mode=WINDOWED_DIFFERENCE_REWARDS
        )

        actor = MAPPOActor(action_dim=N_SKILLS, hidden_dim=HIDDEN_DIM, discrete=True)
        run_batch = build_rollout(sync_dense, sync_dr, sync_wdr, actor)

        print(
            f"backend={jax.default_backend()} | {N_AGENTS}a/{N_OBJECTS}o | "
            f"macro_len={MACRO_LEN} | horizon={sync_dense.max_steps} decisions | "
            f"n_rollouts={args.n_rollouts} chunk={args.chunk}"
        )
        print(f"trials: {trials}\n")

        def run_all(actor_params, keys):
            chunks = [
                run_batch(actor_params, keys[i : i + args.chunk])
                for i in range(0, len(keys), args.chunk)
            ]
            jax.block_until_ready(chunks)
            return tuple(
                np.concatenate([np.asarray(c[f]) for c in chunks], axis=0)
                for f in range(4)
            )

        per_trial = []
        key = jax.random.PRNGKey(args.seed)
        for trial in trials:
            _, actor_params, ckpt = load_actor_params(RESULTS / trial)
            key, sub = jax.random.split(key)
            keys = jax.random.split(sub, args.n_rollouts)
            stats = collect_trial(run_all(actor_params, keys))
            per_trial.append(stats)
            print(
                f"  trial {trial:>2} [{ckpt}]: "
                f"G={stats['G'].mean():6.3f} | "
                f"D_i timestep={stats['Di_ts'].mean():.3f} | "
                f"D_i windowed={stats['Di_w'].mean():.3f}"
            )

        cache = Path(args.out).with_suffix(".data.pkl")
        with open(cache, "wb") as f:
            pickle.dump({"per_trial": per_trial, "trials": trials}, f)
        print(f"cached raw data -> {cache}")

    make_figure(per_trial, trials, args.out)

    G = np.concatenate([t["G"] for t in per_trial])
    Di_ts = np.concatenate([t["Di_ts"] for t in per_trial])
    Di_w = np.concatenate([t["Di_w"] for t in per_trial])
    print("\n=== stored transition reward (signed mean, pooled) ===")
    print(f"  dense    team    G   = {G.mean():.3f}")
    print(f"  _dr      per-agent D_i = {Di_ts.mean():.3f}  ({G.mean()/Di_ts.mean():.0f}x smaller than G)")
    print(f"  _wdr     per-agent D_i = {Di_w.mean():.3f}  ({G.mean()/Di_w.mean():.1f}x smaller than G)")


if __name__ == "__main__":
    main()
