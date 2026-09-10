"""Summarize the September 2026 goal experiments without running the simulator.

Run from the repository root:
    .venv/bin/python plotting/analyze_feudal_goal_runs.py

Reads trusted local pickle logs. Evaluation summaries use eval_time > 0 to
exclude carried-forward evaluations. Each seed has equal weight; temporal
evaluation points are not treated as independent training replications.
"""

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BATCHES = ("mjx_12a_3o_trunc_1024", "mjx_12a_3o_partition_1024")
MODELS = {
    "mlp": "MAPPO",
    "feudal_film_zerogoal": "Zero goal / MLP",
    "feudal_film_zerogoal_dilated": "Zero goal / dilated",
    "feudal_film_n01": "alpha 0.1 / MLP",
    "feudal_film_n01_dilated": "alpha 0.1 / dilated",
    "feudal_film_n05": "alpha 0.5 / MLP",
    "feudal_film_n05_dilated": "alpha 0.5 / dilated",
}
METRICS = (
    "reward", "eval_reward_zeroed", "eval_gap_zeroed", "eval_gap_permuted",
    "d_cos_mean", "d_cos_gap_agent", "d_cos_gap_env", "d_cos_var",
    "intrinsic_reward", "intrinsic_reward_abs", "alpha_current",
    "adv_ext_std_raw", "adv_int_std_raw", "explained_variance",
    "intrinsic_explained_variance", "manager_explained_variance",
    "state_latent_erank", "goal_direction_count", "valid_fraction",
)


def summarize(results):
    rows = []
    for batch in BATCHES:
        for model in MODELS:
            for trial in sorted((results / batch / model).iterdir()):
                path = trial / "logs/training_stats_finished.pkl"
                finished = path.exists()
                if not finished:
                    path = trial / "logs/training_stats_checkpoint.pkl"
                if not path.exists():
                    continue
                with path.open("rb") as handle:
                    stats = pickle.load(handle)
                steps = np.asarray(stats["total_steps"])
                evaluation = np.asarray(stats["eval_time"]) > 0
                assert len(evaluation) == len(steps)
                for lower, upper in ((0, 2), (8, 10), (28, 30), (70, 80), (90, 100)):
                    window = (steps >= lower * 1e6) & (steps < upper * 1e6)
                    evaluated = window & evaluation
                    row = dict(batch=batch, model=model, seed=trial.name,
                               finished=finished, steps=int(steps[-1]),
                               lower_m=lower, upper_m=upper,
                               window_complete=bool(steps[-1] >= upper * 1e6 - 32768),
                               evaluations=int(evaluated.sum()))
                    for metric in METRICS:
                        mask = evaluated if metric == "reward" or metric.startswith("eval_") else window
                        values = np.asarray(stats.get(metric, []))
                        row[metric] = float(values[mask].mean()) if len(values) and mask.any() else None
                    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--output", type=Path, default=ROOT / "plotting/feudal_goal_analysis")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = summarize(args.results)
    with (args.output / "per_seed_windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # A common, fully covered training window for all 42 local runs.
    for batch in BATCHES:
        print(batch, "— 70–80M steps, mean ± sample standard deviation across seeds")
        for model, label in MODELS.items():
            selected = [r for r in rows if r["batch"] == batch and r["model"] == model and r["lower_m"] == 70]
            assert len(selected) == 3 and all(r["window_complete"] for r in selected)
            values = [r["reward"] for r in selected]
            print(f"  {label:25s} {np.mean(values):6.1f} ± {np.std(values, ddof=1):4.1f}")

    # Contrast task return and goal alignment in the same late-training window.
    # All feudal arms completed this window; MAPPO has incomplete seeds and is
    # deliberately excluded here. Its fair comparison appears above/in the CSV.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = ("#64748b", "#94a3b8", "#2563eb", "#60a5fa", "#dc2626", "#fb7185")
    feudal = list(MODELS)[1:]
    labels = [MODELS[m].replace(" / ", "\n") for m in feudal]
    for col, batch in enumerate(BATCHES):
        selected = [r for r in rows if r["batch"] == batch and r["lower_m"] == 90 and r["model"] != "mlp"]
        assert len(selected) == 18 and all(r["window_complete"] for r in selected)
        for index, model in enumerate(feudal):
            seeds = [r for r in selected if r["model"] == model]
            for row_index, metric in enumerate(("reward", "d_cos_mean")):
                values = np.array([r[metric] for r in seeds])
                ax = axes[row_index, col]
                ax.bar(index, values.mean(), color=colors[index], width=0.72, alpha=0.85)
                ax.errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="none", ecolor="#1f2937", capsize=3)
                ax.scatter(index + np.linspace(-0.13, 0.13, len(values)), values, color="#111827", s=15, zorder=3)
        axes[0, col].set_title("Truncation task" if col == 0 else "Partition task")
        axes[0, col].set_ylim(0, 400)
        axes[1, col].set_ylim(-0.2, 0.35)
        for row_index in range(2):
            ax = axes[row_index, col]
            ax.set_xticks(range(len(feudal)), labels, fontsize=8)
            ax.axhline(0, color="#64748b", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.15)
    axes[0, 0].set_ylabel("Task return (higher is better)")
    axes[1, 0].set_ylabel("Latent goal alignment (cosine)")
    fig.suptitle("Better latent alignment does not imply a better task policy\n90–100M steps; dots: three seeds; error bars: seed standard deviation", fontsize=13)
    fig.savefig(args.output / "task_vs_alignment.png", dpi=180)
    plt.close(fig)
    print(f"Saved summaries and figure to {args.output}")


if __name__ == "__main__":
    main()
