"""Episode-level goal geometry and alignment through policy timesteps."""

from pathlib import Path

import numpy as np


# Diverging scale for goal-alignment cosines, shared by the raster heatmaps and
# the video overlay so a color means the same thing in both. Blue = follows
# (+1), orange = opposes (-1), neutral gray at 0. Orange rather than a red pole
# because MJX agents are drawn as red discs; the two poles sit at the same OKLCH
# lightness (0.575 / 0.577), so neither arm reads as the stronger one.
ALIGNMENT_POSITIVE = "#2a78d6"
ALIGNMENT_NEUTRAL = "#f0efec"
ALIGNMENT_NEGATIVE = "#c4501f"


def alignment_cmap():
    """The diverging colormap over [-1, 1]; NaN (undefined) is transparent."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "goal_alignment", [ALIGNMENT_NEGATIVE, ALIGNMENT_NEUTRAL, ALIGNMENT_POSITIVE]
    ).with_extremes(bad=(0, 0, 0, 0))


def alignment_rgb(values):
    """Cosines (...,) -> (..., 3) uint8 on `alignment_cmap`; NaN maps to neutral."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    rgba = alignment_cmap()(np.clip((values + 1.0) / 2.0, 0.0, 1.0))
    return np.rint(np.asarray(rgba)[..., :3] * 255).astype(np.uint8)


def project_goal_outcomes(goals, latents, horizon):
    """Pair g_t with s_(t+c), using one PCA for goals and all latent states.

    Goals have shape (T, N, D); latents include the final observation and have
    shape (T+1, N, D). A goal is a direction, so its displayed endpoint is
    s_t + g_t (unit distance for the normal manager), not an absolute state.
    Incomplete horizons at the end of an episode are excluded.
    """
    targets, states, variance = _project_goal_episode(goals, latents, horizon)
    return targets, states[horizon:], variance


def _goal_episode_arrays(goals, latents, horizon):
    """Validate one episode, including its final post-action state."""
    goals = np.asarray(goals, dtype=np.float64)
    latents = np.asarray(latents, dtype=np.float64)
    if goals.ndim != 3 or latents.shape != (len(goals) + 1, *goals.shape[1:]):
        raise ValueError("Expected goals (T, N, D) and latents (T+1, N, D)")
    if not all(goals.shape[1:]):
        raise ValueError("Expected at least one agent and one goal dimension")
    if (isinstance(horizon, (bool, np.bool_))
            or not isinstance(horizon, (int, np.integer)) or horizon < 1):
        raise ValueError("Goal horizon must be a positive integer")
    if not np.isfinite(goals).all() or not np.isfinite(latents).all():
        raise ValueError("Goals and latents must be finite")
    return goals, latents


def _cosine_alignment(directions, displacement):
    """Full-dimensional cosine; zero vectors have undefined alignment."""
    norm_product = np.linalg.norm(directions, axis=-1) * np.linalg.norm(
        displacement, axis=-1
    )
    alignment = np.divide(
        np.sum(directions * displacement, axis=-1), norm_product,
        out=np.full(norm_product.shape, np.nan), where=norm_product > 0,
    )
    return np.clip(alignment, -1.0, 1.0)


def goal_alignment_scores(
    goals, latents, pooled_goals, horizon, *, active=None, goal_space="latent",
):
    """Return horizon and one-step cosines, each shaped (T, N).

    At policy timestep t, the horizon score is cos(g_t, s_(t+c) - s_t);
    the one-step score is cos(w_t, s_(t+1) - s_t), where w_t is the actual
    pooled goal passed to the worker. All coordinates are before projection.
    These are directional diagnostics, not the worker's intrinsic reward.

    NaN marks incomplete horizons, zero vectors, or inactive agents. Activity
    has shape (T, N) and describes the state BEFORE each action, so the transition
    that ends an episode still counts. Every action in a horizon must be active.
    For waypoints, only proposals latched at t=0,c,2c,... count as horizon goals;
    their direction is taken from the worker's live error.
    """
    goals, latents = _goal_episode_arrays(goals, latents, horizon)
    pooled_goals = np.asarray(pooled_goals, dtype=np.float64)
    if pooled_goals.shape != goals.shape or not np.isfinite(pooled_goals).all():
        raise ValueError("Expected finite pooled goals with shape (T, N, D)")
    if goal_space not in ("latent", "position_direction", "position_waypoint"):
        raise ValueError(f"Unknown goal space: {goal_space}")
    if active is None:
        active = np.ones(goals.shape[:2], dtype=bool)
    else:
        active = np.asarray(active)
        if active.shape != goals.shape[:2] or not np.isfinite(active).all():
            raise ValueError("Expected finite activity mask with shape (T, N)")
        active = active > 0

    one_step = _cosine_alignment(pooled_goals, np.diff(latents, axis=0))
    one_step = np.where(active, one_step, np.nan)
    horizon_scores = np.full(goals.shape[:2], np.nan)
    count = max(0, len(goals) - horizon + 1)
    if count:
        directions = pooled_goals if goal_space == "position_waypoint" else goals
        scores = _cosine_alignment(
            directions[:count], latents[horizon:] - latents[:count]
        )
        # Prefix sums avoid building a (T, c, N) array for long episodes.
        inactive = np.concatenate(
            (np.zeros((1, goals.shape[1]), dtype=int), np.cumsum(~active, axis=0))
        )
        valid = inactive[horizon:] == inactive[:count]
        if goal_space == "position_waypoint":
            valid &= (np.arange(count) % horizon == 0)[:, None]
        horizon_scores[:count] = np.where(valid, scores, np.nan)
    return horizon_scores, one_step


def frame_alignment(
    horizon_scores, one_step, frame_decision, horizon, *, goal_space="latent",
):
    """Place `goal_alignment_scores` on rendered frames, as known at each frame.

    `frame_decision` (F,) names the policy step `d` each frame belongs to: the
    identity for flat envs, and constant over a macro window. Both outputs are
    (F, N) and CAUSAL — frame `d` shows the state `s_d` and uses only `s_0..s_d`:

    * ring: ``horizon_scores[d - c]`` = cos(g_(d-c), s_d - s_(d-c)), i.e. was the
      goal issued `c` steps ago followed? For waypoints, which are scored only
      at latches, the most recent COMPLETED latch ``(d // c) * c - c`` is held
      rather than flickering between latches.
    * dot: ``one_step[d - 1]`` = cos(w_(d-1), s_d - s_(d-1)), i.e. did the last
      move follow the goal the worker held?

    NaN before a score exists, and wherever the source score is NaN.
    """
    horizon_scores = np.asarray(horizon_scores, dtype=np.float64)
    one_step = np.asarray(one_step, dtype=np.float64)
    decision = np.asarray(frame_decision)
    if horizon_scores.shape != one_step.shape or one_step.ndim != 2:
        raise ValueError("Expected horizon and one-step scores with shape (T, N)")
    if decision.ndim != 1 or not np.issubdtype(decision.dtype, np.integer):
        raise ValueError("Expected integer frame decisions with shape (F,)")
    if len(decision) and (decision.min() < 0 or decision.max() >= len(one_step)):
        raise ValueError("Frame decisions must index the policy steps")

    if goal_space == "position_waypoint":
        issued = (decision // horizon) * horizon - horizon
    else:
        issued = decision - horizon
    ring = np.full((len(decision), one_step.shape[1]), np.nan)
    ring[issued >= 0] = horizon_scores[issued[issued >= 0]]
    dot = np.full_like(ring, np.nan)
    dot[decision >= 1] = one_step[decision[decision >= 1] - 1]
    return ring, dot


def _mean_valid_scores(scores):
    """Mean over time for each agent, leaving missing scores undefined."""
    valid = np.isfinite(scores)
    count = valid.sum(axis=0)
    return np.divide(
        np.where(valid, scores, 0).sum(axis=0), count,
        out=np.full(scores.shape[1:], np.nan), where=count > 0,
    )


def summarize_task_alignment(
    goals, latents, pooled_goals, horizon, task_rewards, *, episode,
    active=None, goal_space="latent", macro_steps=False, max_windows=8,
):
    """Pair team reward sums with per-agent mean alignment in matching windows.

    Rewards must have one entry per policy decision. Macro callers sum the
    underlying physics-step rewards for each decision, including a short final
    macro step. Windows are non-overlapping multiples of the goal horizon, so
    waypoint latches stay aligned. Only goal horizons wholly contained in a
    window contribute to that window; one-step scores use all its transitions.
    """
    horizon_scores, one_step = goal_alignment_scores(
        goals, latents, pooled_goals, horizon, active=active, goal_space=goal_space,
    )
    task_rewards = np.asarray(task_rewards, dtype=np.float64)
    steps = len(one_step)
    if task_rewards.shape != (steps,) or not np.isfinite(task_rewards).all():
        raise ValueError("Expected finite task rewards, one per policy timestep")
    if (isinstance(max_windows, (bool, np.bool_))
            or not isinstance(max_windows, (int, np.integer)) or max_windows < 1):
        raise ValueError("max_windows must be a positive integer")

    def summarize(start, end):
        return {
            "start": start, "end": end, "steps": end - start,
            "task_return": float(task_rewards[start:end].sum()),
            "horizon_alignment": _mean_valid_scores(
                horizon_scores[start:max(start, end - horizon + 1)]
            ),
            "one_step_alignment": _mean_valid_scores(one_step[start:end]),
        }

    width = max(1, (steps + max_windows * horizon - 1) // (max_windows * horizon)) * horizon
    summary = summarize(0, steps)
    summary.update(
        episode=episode, horizon=horizon, goal_space=goal_space, macro_steps=macro_steps,
        windows=[summarize(start, min(start + width, steps))
                 for start in range(0, steps, width)],
    )
    return summary


def _save_task_alignment_comparison(
    groups, labels, path, *, title, subtitle, return_label, xlabel,
):
    """Stack reward and both alignment metrics over the same categories."""
    import matplotlib.pyplot as plt

    # Repeat category colours down the rows, as in task_vs_alignment.png.
    palette = ("#64748b", "#2563eb", "#dc2626", "#0891b2", "#7c3aed",
               "#16a34a", "#d97706", "#db2777", "#475569", "#4f46e5")
    colors = [palette[i % len(palette)] for i in range(len(groups))]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(groups) * 1.05), 9), sharex=True)
    try:
        returns = [group["task_return"] for group in groups]
        bars = axes[0].bar(x, returns, color=colors, width=0.68, alpha=0.85)
        axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        axes[0].set_ylabel(return_label)
        axes[0].set_title("Task performance", loc="left", fontsize=11)

        for ax, metric, metric_title in zip(
            axes[1:],
            ("horizon_alignment", "one_step_alignment"),
            ("Horizon goal alignment", "One-step worker alignment"),
        ):
            for index, group in enumerate(groups):
                values = np.asarray(group[metric])
                valid = np.isfinite(values)
                if not valid.any():
                    ax.text(index, 0, "N/A", ha="center", va="bottom", fontsize=9,
                            color="#64748b")
                    continue
                # Agents have equal weight even when they have different
                # numbers of valid timesteps (e.g. a unit dies mid-episode).
                mean = values[valid].mean()
                ax.bar(index, mean, color=colors[index], width=0.68, alpha=0.85)
                if valid.sum() > 1:
                    ax.errorbar(index, mean, yerr=values[valid].std(), fmt="none",
                                ecolor="#1f2937", capsize=3, linewidth=1.1)
                offsets = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(index + offsets[valid], values[valid],
                           color="#111827", s=14, alpha=0.8, zorder=3)
            ax.set_ylabel("Mean cosine similarity")
            ax.set_title(metric_title, loc="left", fontsize=11)

        for ax in axes:
            ax.axhline(0, color="#64748b", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.15)
            ax.margins(y=0.25)
            ax.set_xlim(-0.6, max(0.6, len(groups) - 0.4))
            if not groups:
                ax.text(0.5, 0.5, "No episode steps", ha="center", transform=ax.transAxes)
        axes[-1].set_xticks(x, labels, fontsize=9)
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
        fig.text(
            0.02, 0.025,
            "Return: sum of team task rewards. Alignment bars: mean across agents; dots: each agent's mean; whiskers: ±1 SD across agents.\n"
            "Only valid scores contribute; N/A means none are available. Cosine: +1 follows the goal, 0 is perpendicular, -1 opposes it.",
            fontsize=8.5,
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.92))
        fig.savefig(path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Task/alignment plot saved to {path}")


def save_task_alignment_episode(summary, path: Path):
    """Show progress in a few matching reward/alignment windows."""
    windows = summary["windows"]
    labels = [f'{window["start"]}–{window["end"] - 1}\n({window["steps"]} steps)'
              for window in windows]
    unit = "macro decisions" if summary["macro_steps"] else "policy timesteps"
    space = "learned latent" if summary["goal_space"] == "latent" else "position"
    _save_task_alignment_comparison(
        windows, labels, path,
        title=f'Episode {summary["episode"]} — Task return and goal alignment',
        subtitle=(f'Total return: {summary["task_return"]:.2f} · {summary["steps"]} {unit} · '
                  f'{space} goals · horizon {summary["horizon"]}\n'
                  "Each column covers one time window; horizon scores use only complete goals inside that window."),
        return_label="Window task return",
        xlabel=f"Episode {unit} (inclusive)",
    )


def save_task_alignment_overview(summaries, path: Path):
    """Compare whole rendered episodes using the same task/alignment layout."""
    if not summaries:
        return
    labels = [f'Episode {summary["episode"]}\n({summary["steps"]} steps)'
              for summary in summaries]
    _save_task_alignment_comparison(
        summaries, labels, path,
        title="Task return and goal alignment across episodes",
        subtitle="Each column is one rendered episode from the same policy; higher alignment need not mean higher task return.",
        return_label="Episode task return",
        xlabel="Rendered episode (duration in policy steps)",
    )


def save_goal_alignment_plot(
    goals, latents, pooled_goals, horizon, path: Path, episode: int, *,
    active=None, goal_space="latent", macro_steps=False,
):
    """Save both alignment time series in one figure per agent."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    horizon_scores, one_step = goal_alignment_scores(
        goals, latents, pooled_goals, horizon, active=active, goal_space=goal_space,
    )
    steps = np.arange(len(one_step))
    count = max(0, len(steps) - horizon + 1)
    waypoint = goal_space == "position_waypoint"
    for agent in range(one_step.shape[1]):
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        try:
            titles = (
                f"Horizon alignment — {'latched' if waypoint else 'issued'} goal vs. next {horizon} steps",
                "One-step alignment — worker goal vs. next step",
            )
            for ax, scores, title, color in zip(
                axes, (horizon_scores, one_step), titles, ("tab:blue", "tab:orange")
            ):
                ax.plot(steps, scores[:, agent], color=color, linewidth=1,
                        marker=".", markersize=3)
                ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
                ax.set_ylim(-1.05, 1.05)
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_ylabel("Cosine similarity")
                ax.set_title(title, fontsize=11, loc="left")
                ax.grid(alpha=0.2)
                if not np.isfinite(scores[:, agent]).any():
                    message = "No defined alignment (zero vectors or inactive agent)"
                    if not len(steps):
                        message = "No episode steps"
                    elif ax is axes[0] and not count:
                        message = f"No complete {horizon}-step horizon"
                    ax.text(0.5, 0.65, message, ha="center", transform=ax.transAxes)
            if count < len(steps):
                axes[0].axvspan(count - 0.5, len(steps) - 0.5, color="0.5", alpha=0.12)
            axes[-1].set_xlim(-0.5, max(0.5, len(steps) - 0.5))
            axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[-1].set_xlabel(
                "Policy timestep t (macro decision)" if macro_steps else "Policy timestep t"
            )
            space = "learned latent space" if goal_space == "latent" else "position space"
            fig.suptitle(f"Episode {episode} — Agent {agent + 1} — Goal alignment ({space})")
            footer = (
                "+1 follows the goal; 0 is perpendicular; -1 opposes it. Scores use all dimensions, without smoothing.\n"
                "Gaps: zero goal/displacement or inactive agent; shaded tail: incomplete horizon."
            )
            if waypoint:
                footer += "\nHorizon scores appear only at target latches; one-step scores use the live direction to the active target."
            fig.text(0.02, 0.02, footer, fontsize=9)
            fig.tight_layout(rect=(0, 0.13, 1, 0.95))
            agent_path = path.with_name(f"{path.stem}_agent_{agent + 1}{path.suffix}")
            fig.savefig(agent_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        print(f"Goal alignment plot saved to {agent_path}")


def _project_goal_episode(goals, latents, horizon):
    """Project complete goal endpoints and every state in one common basis."""
    goals, latents = _goal_episode_arrays(goals, latents, horizon)

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
    alignment = _cosine_alignment(selected_goals, displacement)
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


# Chart chrome (reference palette ink/surface roles).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_BASELINE = "#c3c2b7"
_SPACE_LABELS = {
    "latent": "learned latent goals",
    "position_direction": "position-direction goals",
    "position_waypoint": "position-waypoint goals",
}


def goal_following_figure(
    ring, dot, rewards, *, horizon, goal_space, episode, macro_steps=False,
    size_px=(1200, 700), hud_px=0,
):
    """Reward trace over two agents x frame heatmaps: `frame_alignment`'s ring
    (horizon) and dot (one-step) values, on one shared diverging scale.

    One definition serves both outputs: saved as the per-episode raster PNG, and
    rasterized as the goal video's side panel (`hud_px` then reserves a blank band
    at the top for the per-frame text). Built on a bare `Figure` with an Agg
    canvas, so it touches no pyplot state. Returns ``(fig, [reward, ring, dot])``.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    ring = np.asarray(ring, dtype=np.float64)
    dot = np.asarray(dot, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if ring.ndim != 2 or dot.shape != ring.shape or rewards.shape != (len(ring),):
        raise ValueError("Expected ring/dot (F, N) and one reward per frame")
    frames, agents = ring.shape
    width, height = size_px
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_SURFACE)
    FigureCanvasAgg(fig)
    top = 1 - hud_px / height
    grid = fig.add_gridspec(
        3, 2, width_ratios=(1, 0.02), height_ratios=(1, 1.3, 1.3),
        left=0.1, right=0.85, top=top - 0.16, bottom=0.24, hspace=0.45, wspace=0.03,
    )
    reward_ax = fig.add_subplot(grid[0, 0])
    ring_ax = fig.add_subplot(grid[1, 0], sharex=reward_ax)
    dot_ax = fig.add_subplot(grid[2, 0], sharex=reward_ax)
    cax = fig.add_subplot(grid[1:, 1])

    unit = "decisions" if macro_steps else "steps"
    reward_ax.plot(np.arange(frames), rewards, color=_INK_SECONDARY, linewidth=1.2)
    reward_ax.set_title("Team task reward", loc="left", fontsize=9, color=_INK)
    reward_ax.set_ylabel("Reward", fontsize=8, color=_INK_SECONDARY)
    reward_ax.grid(axis="y", color=_BASELINE, alpha=0.4, linewidth=0.6)

    cmap = alignment_cmap()
    rows = (
        (ring_ax, ring, f"Horizon: goal issued {horizon} {unit} ago vs. displacement since"),
        (dot_ax, dot, "One-step: goal the worker held vs. its last move"),
    )
    for ax, values, title in rows:
        # Hatching behind a transparent-NaN image marks "undefined" without
        # borrowing a color from the scale.
        ax.add_patch(Rectangle(
            (-0.5, -0.5), max(frames, 1), agents, hatch="////", fill=False,
            edgecolor=_BASELINE, linewidth=0, zorder=0,
        ))
        image = ax.imshow(
            values.T, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
            interpolation="nearest", zorder=1,
            extent=(-0.5, max(frames, 1) - 0.5, agents - 0.5, -0.5),
        )
        ax.set_title(title, loc="left", fontsize=9, color=_INK)
        ax.set_ylabel("Agent", fontsize=8, color=_INK_SECONDARY)
        if agents <= 16:  # label every agent, 0-based like the video's labels
            ax.set_yticks(np.arange(agents))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    colorbar = fig.colorbar(image, cax=cax, ticks=[-1, 0, 1])
    colorbar.ax.set_yticklabels(["−1 opposes", "0", "+1 follows"], fontsize=8)
    colorbar.outline.set_visible(False)

    for ax in (reward_ax, ring_ax, dot_ax):
        ax.set_xlim(-0.5, max(frames, 1) - 0.5)
        ax.tick_params(labelsize=8, colors=_MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(_BASELINE)
    for ax in (reward_ax, ring_ax):
        ax.tick_params(labelbottom=False)
    dot_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    dot_ax.set_xlabel(
        "Physics step (scores update at each macro decision)" if macro_steps
        else "Policy timestep", fontsize=8, color=_INK_SECONDARY,
    )

    fig.text(
        0.1, top - 0.035,
        f"Episode {episode} — goal following", fontsize=12, color=_INK,
        va="top", weight="bold",
    )
    fig.text(
        0.1, top - 0.075,
        f"{_SPACE_LABELS.get(goal_space, goal_space)} · horizon c = {horizon} {unit}"
        f" · return {rewards.sum():.1f}",
        fontsize=9, color=_INK_SECONDARY, va="top",
    )
    caption = [
        "Cosine of each agent's goal with its goal-space displacement, shown when the "
        "outcome is known (horizon: end of the window; one-step: after the move).",
        "Hatched: undefined (no complete horizon yet, inactive agent, or zero vector).",
    ]
    if goal_space == "position_waypoint":
        caption.append("Waypoints are scored only at latches; each score is held "
                       "until the next latch completes.")
    caption.append("Following a goal is not the same as the goal helping: compare "
                   "returns against goal-free controls.")
    fig.text(0.1, 0.02, "\n".join(caption), fontsize=7.5, color=_INK_SECONDARY,
             va="bottom", linespacing=1.4, wrap=True)
    return fig, [reward_ax, ring_ax, dot_ax]


def rasterize_panel(fig, axes, n_frames):
    """Render `fig` to (H, W, 3) uint8, plus where frame f's cursor goes.

    Returns ``(image, cursor_x, (row_top, row_bottom))``: `cursor_x` (F,) is the
    pixel column of each frame's heatmap cell centre, and the rows span every
    axis in `axes` (image coordinates, origin top-left).
    """
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    height = image.shape[0]
    boxes = [ax.get_window_extent() for ax in axes]
    x0, x1 = boxes[0].x0, boxes[0].x1
    cursor_x = x0 + (np.arange(n_frames) + 0.5) / max(n_frames, 1) * (x1 - x0)
    rows = (int(height - max(b.y1 for b in boxes)), int(height - min(b.y0 for b in boxes)))
    return np.rint(cursor_x).astype(int), rows, image


def save_goal_following_raster(ring, dot, rewards, path: Path, **kwargs):
    """Save `goal_following_figure` as one PNG per episode."""
    fig, _ = goal_following_figure(ring, dot, rewards, **kwargs)
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    print(f"Goal-following raster saved to {path}")
