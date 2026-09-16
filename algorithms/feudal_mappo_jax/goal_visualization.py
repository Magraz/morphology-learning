"""Episode-level PCA views of manager goals and their horizon-later outcomes."""

from pathlib import Path

import numpy as np


def project_goal_outcomes(goals, latents, horizon):
    """Pair g_t with s_(t+c), using one PCA for goals and all latent states.

    Goals have shape (T, N, D); latents include the final observation and have
    shape (T+1, N, D). A goal is a direction, so its displayed endpoint is
    s_t + g_t (unit distance for the normal manager), not an absolute state.
    Incomplete horizons at the end of an episode are excluded.
    """
    targets, states, variance = _project_goal_episode(goals, latents, horizon)
    return targets, states[horizon:], variance


def _project_goal_episode(goals, latents, horizon):
    """Project complete goal endpoints and every state in one common basis."""
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
    targets = latents[:count] + goals[:count]
    points = np.concatenate(
        (targets.reshape(-1, goals.shape[-1]), latents.reshape(-1, goals.shape[-1]))
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
        projected[count * goals.shape[1] :].reshape(*latents.shape[:2], 2),
        variance,
    )


def save_goal_plot(
    goals,
    latents,
    horizon,
    path: Path,
    episode: int,
    data_pairs: int | None = 4,
):
    """Save one plot per agent, appending ``_agent_<1-based ID>`` to path.

    All plots use the same episode PCA so their coordinates remain comparable.
    ``data_pairs`` limits the displayed pairs per agent; None displays all.
    Sample reached-state times at T / data_pairs, ..., T, rounded to steps.
    If the first time precedes the horizon, spread samples from the earliest
    complete outcome to T instead. Requests exceeding the available complete
    pairs display all pairs. Every selected pair includes the trajectory from
    s_t through s_(t+c). PCA uses all episode states and complete goal endpoints.
    Trajectory labels report cos(g_t, s_(t+c) - s_t) before projection;
    zero goals or zero displacements have undefined alignment (N/A).
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

    targets, states, variance = _project_goal_episode(goals, latents, horizon)
    indices = np.arange(len(targets))
    if data_pairs is not None and 0 < data_pairs < len(targets):
        first_step = max(horizon, len(goals) / data_pairs)
        reached_steps = np.rint(np.linspace(first_step, len(goals), data_pairs)).astype(
            int
        )
        indices = reached_steps - horizon
    targets = targets[indices]
    trajectories = states[indices[:, None] + np.arange(horizon + 1)]
    reached = trajectories[:, -1]
    # Compute in the original latent space, independently of the PCA display.
    original_states = np.asarray(latents, dtype=np.float64)
    selected_goals = np.asarray(goals, dtype=np.float64)[indices]
    displacement = original_states[indices + horizon] - original_states[indices]
    norm_product = np.linalg.norm(selected_goals, axis=-1) * np.linalg.norm(
        displacement, axis=-1
    )
    alignment = np.divide(
        np.sum(selected_goals * displacement, axis=-1), norm_product,
        out=np.full(norm_product.shape, np.nan), where=norm_product > 0,
    )
    alignment = np.clip(alignment, -1.0, 1.0)
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
                linestyles=":",
                alpha=0.25,
            )
        )
        # Each window is drawn separately: never connect across sampled pairs.
        for pair, (target, trajectory) in enumerate(
            zip(targets[:, agent], trajectories[:, :, agent])
        ):
            ax.plot(
                *trajectory.T,
                color=color,
                linewidth=0.8,
                marker=".",
                markersize=4,
                alpha=0.65,
            )
            ax.annotate(
                "",
                xy=target,
                xytext=trajectory[0],
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linestyle": "--",
                    "linewidth": 0.9,
                    "alpha": 0.7,
                },
            )
            score = alignment[pair, agent]
            score_text = f"{score:+.2f}" if np.isfinite(score) else "N/A"
            start = indices[pair]
            ax.annotate(
                f"t={start}–{start + horizon}\ncos={score_text}",
                xy=trajectory[-1], xytext=(8, 8), textcoords="offset points",
                fontsize=8, zorder=5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": color, "alpha": 0.85},
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.5},
            )
        if len(trajectories):
            origins = trajectories[:, :-1, agent].reshape(-1, 2)
            steps = np.diff(trajectories[:, :, agent], axis=1).reshape(-1, 2)
            # Suppress zero-length arrows for stationary or collapsed latents.
            moving = np.any(steps != 0, axis=1)
            if np.any(moving):
                ax.quiver(
                    *origins[moving].T,
                    *steps[moving].T,
                    color=color,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.003,
                    alpha=0.65,
                    zorder=3,
                )
        ax.scatter(
            *trajectories[:, 0, agent].T,
            color=color,
            marker="s",
            s=35,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
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
                marker="s",
                linestyle="",
                label="Starting state: s_t",
            ),
            Line2D(
                [],
                [],
                color=color,
                marker=".",
                linestyle="-",
                label="Intermediate states (arrows follow time)",
            ),
            Line2D([], [], color=color, linestyle="--", label="Goal direction"),
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
            f"Solid arrows trace {horizon} policy steps; dashed arrows show goal directions. Dotted lines pair goals/outcomes.\n"
            "Shared PCA fitted per episode; 2D distances approximate latent-space distances.\n"
            "cos: full-dimensional goal/displacement alignment (+1 follows, -1 opposes; N/A: zero goal or displacement).",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        agent_path = path.with_name(f"{path.stem}_agent_{agent + 1}{path.suffix}")
        try:
            fig.savefig(agent_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        print(f"Goal plot saved to {agent_path}")
