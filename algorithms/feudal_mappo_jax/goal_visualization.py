"""Episode-level PCA views of manager goals and their horizon-later outcomes."""

from pathlib import Path

import numpy as np


def project_goal_outcomes(goals, latents, horizon):
    """Pair g_t with s_(t+c), then fit ONE PCA to all agents and both kinds.

    Goals have shape (T, N, D); latents include the final observation and have
    shape (T+1, N, D). A goal is a direction, so its displayed endpoint is
    s_t + g_t (unit distance for the normal manager), not an absolute state.
    Incomplete horizons at the end of an episode are excluded.
    """
    goals = np.asarray(goals, dtype=np.float64)
    latents = np.asarray(latents, dtype=np.float64)
    if goals.ndim != 3 or latents.shape != (len(goals) + 1, *goals.shape[1:]):
        raise ValueError("Expected goals (T, N, D) and latents (T+1, N, D)")
    if horizon < 1:
        raise ValueError("Goal horizon must be positive")
    if not np.isfinite(goals).all() or not np.isfinite(latents).all():
        raise ValueError("Goals and latents must be finite")

    count = max(0, len(goals) - horizon + 1)
    shape = (count, goals.shape[1], 2)
    if not count:
        return np.empty(shape), np.empty(shape), np.zeros(2)

    targets = latents[:count] + goals[:count]
    reached = latents[horizon : horizon + count]
    points = np.concatenate(
        (targets.reshape(-1, goals.shape[-1]), reached.reshape(-1, goals.shape[-1]))
    )
    centered = points - points.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    # Pad the second coordinate when D=1; zero-variance episodes are valid too.
    dimensions = min(2, len(axes))
    projected = np.zeros((len(points), 2))
    projected[:, :dimensions] = centered @ axes[:dimensions].T
    variance = np.zeros(2)
    total = np.sum(singular_values**2)
    if total > 0:
        variance[:dimensions] = singular_values[:dimensions] ** 2 / total
    return (
        projected[: count * goals.shape[1]].reshape(shape),
        projected[count * goals.shape[1] :].reshape(shape),
        variance,
    )


def save_goal_plot(
    goals,
    latents,
    horizon,
    path: Path,
    episode: int,
    data_pairs: int | None = 10,
):
    """Save one plot per agent, appending ``_agent_<1-based ID>`` to path.

    All plots use the same episode PCA so their coordinates remain comparable.
    ``data_pairs`` limits the displayed pairs per agent; None displays all.
    Sample reached-state times at T / data_pairs, ..., T, rounded to steps.
    If the first time precedes the horizon, spread samples from the earliest
    complete outcome to T instead. Requests exceeding the available complete
    pairs display all pairs. PCA always uses the full episode's complete pairs.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    if data_pairs is not None and (
        isinstance(data_pairs, (bool, np.bool_))
        or not isinstance(data_pairs, (int, np.integer))
        or data_pairs < 1
    ):
        raise ValueError("data_pairs must be a positive integer or None")

    targets, reached, variance = project_goal_outcomes(goals, latents, horizon)
    if data_pairs is not None and 0 < data_pairs < len(targets):
        first_step = max(horizon, len(goals) / data_pairs)
        reached_steps = np.rint(np.linspace(first_step, len(goals), data_pairs)).astype(
            int
        )
        indices = reached_steps - horizon
        targets, reached = targets[indices], reached[indices]
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "brown",
        "olive",
        "pink",
    ]
    colors.extend(plt.get_cmap("tab20b").colors)
    for agent in range(targets.shape[1]):
        fig, ax = plt.subplots(figsize=(10, 7))
        color = (
            colors[agent]
            if agent < len(colors)
            else plt.get_cmap("hsv")(((agent + 1) * 0.61803398875) % 1)
        )
        ax.add_collection(
            LineCollection(
                np.stack((targets[:, agent], reached[:, agent]), axis=1),
                colors=[color],
                linewidths=0.6,
                alpha=0.25,
            )
        )
        ax.scatter(*reached[:, agent].T, color=color, marker="o", s=20, alpha=0.65)
        ax.scatter(
            *targets[:, agent].T,
            color=color,
            marker="*",
            s=95,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.85,
        )
        handles = [
            Line2D(
                [],
                [],
                color=color,
                marker="*",
                markersize=12,
                linestyle="",
                label="Goal endpoint: s_t + g_t",
            ),
            Line2D(
                [],
                [],
                color=color,
                marker="o",
                linestyle="",
                label=f"Reached state: s_(t + {horizon})",
            ),
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.set_xlabel(f"PC 1 ({variance[0]:.1%} variance)")
        ax.set_ylabel(f"PC 2 ({variance[1]:.1%} variance)")
        ax.set_title(
            f"Episode {episode} — Agent {agent + 1}\n"
            "Manager goals and reached latent states"
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        if not len(targets):
            ax.text(
                0.5,
                0.5,
                f"Episode shorter than goal horizon ({horizon} steps):\n"
                "no complete goal/outcome pairs",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        fig.text(
            0.02,
            0.02,
            "Stars mark one goal-vector step from the starting latent; goals specify direction, not distance.\n"
            f"Lines pair goals with outcomes after {horizon} policy steps. Incomplete horizons are omitted.\n"
            "Shared PCA fitted per episode; 2D distances approximate latent-space distances.",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        agent_path = path.with_name(f"{path.stem}_agent_{agent + 1}{path.suffix}")
        try:
            fig.savefig(agent_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        print(f"Goal plot saved to {agent_path}")
