"""Matplotlib figures for the rover domain: arena layouts and example trajectories.

* ``plot_arena(env)`` — the arena and its POIs. A POI is drawn as its
  observation-radius disc with its coupling (rovers needed at once) written
  inside, plus its value and per-step rate. Rover start positions are random,
  so they are not drawn; one rover's sensing radius is overlaid at the
  spawn-region centre (``draw_sensing_radius``).
* ``plot_trajectory(env, traj)`` — one episode: every rover's path (start dot,
  arrow at the final heading) over the arena, when each POI paid, and the
  cumulative team reward. Used-up POIs are drawn hollow with their depletion
  step.

CLI (writes one arena image per ``conf/env/rover_*.yaml`` variant, plus
trajectory figures for the random and scripted demonstration policies)::

    uv run python -m environments.rover.visualize \
        [--variants rover_2poi_c2 ...] [--seeds 0 1] \
        [--policies random scripted] [--out-dir plotting/rover] [--no-trajectories]

Outputs: ``<out-dir>/arenas/<variant>.png`` and
``<out-dir>/trajectories/<variant>_<policy>_seed<k>.png``.

For trained policies, use the runner's view command instead::

    uv run python train.py algorithm=rmappo_jax env=rover_2poi_c2 \
        model=rmappo trial_id=0 view=true

``model=mappo_mlp`` selects the feedforward policy. The runner writes to
``experiments/results/<env>/<model>/<trial>/logs/rover/``.
"""

import argparse
from pathlib import Path

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from environments.rover.config import REPO_ROOT, list_variants, load_rover_env
from environments.rover.policies import make_scripted_coalition_policy, random_policy
from environments.rover.rollout import rollout

# Light-mode chart chrome and the categorical order of the dataviz reference
# palette (fixed order, never cycled; validated for line charts).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

POLICIES = {
    "random": lambda env: random_policy,
    "scripted": make_scripted_coalition_policy,
}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def draw_arena(ax, env, pois=None, depleted_at=None, coupling_inside=True):
    """Draw the arena and POIs on ``ax``.

    ``pois`` defaults to the env's initial POIs (reset key 0). ``depleted_at``
    maps a POI index to the step it was used up; such POIs are drawn hollow.
    ``coupling_inside`` writes the coupling large in the disc centre; set it
    False when paths will be drawn on top (they end on the centre), which moves
    the coupling into the label above the disc.
    """
    if pois is None:
        pois = env.pois.init(jax.random.PRNGKey(0), env)
    depleted_at = depleted_at or {}
    width, height = env.world_size
    _style_axes(ax)
    ax.set_aspect("equal")
    pad = 0.04 * max(width, height)
    ax.set_xlim(-pad, width + pad)
    ax.set_ylim(-pad, height + pad)
    ticks = max(1, round(max(width, height) / 5))
    ax.set_xticks(np.arange(0, width + 1e-9, ticks))
    ax.set_yticks(np.arange(0, height + 1e-9, ticks))
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor=AXIS, linewidth=1.2))

    position = np.asarray(pois.position)
    for k in range(position.shape[0]):
        x, y = position[k]
        radius = float(pois.radius[k])
        used = k in depleted_at
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="none" if used else to_rgba(INK_2, 0.10),
                edgecolor=MUTED if used else INK_2,
                linewidth=1.2,
                zorder=2,
            )
        )
        coupling = int(pois.coupling[k])
        if coupling_inside:
            ax.text(x, y, str(coupling), ha="center", va="center", fontsize=13,
                    fontweight="bold", color=MUTED if used else INK, zorder=4)
            label = f"P{k}"
        else:
            ax.plot(x, y, "+", color=MUTED if used else INK_2, markersize=7, zorder=2)
            label = f"P{k} · c={coupling}"
        ax.text(x, y + radius + 0.12, label, ha="center", va="bottom",
                fontsize=8.5, color=INK_2, zorder=4)
        detail = (
            f"used at t={depleted_at[k]}"
            if used
            else f"v={float(pois.value[k]):g}, {float(pois.rate[k]):g}/step"
        )
        ax.text(x, y - radius - 0.12, detail, ha="center", va="top",
                fontsize=7.5, color=MUTED, zorder=4)


def draw_sensing_radius(ax, env, pois=None):
    """Overlay one rover's sensing radius, centred on the spawn region's centre.

    Rover start positions are random, so the disc shows the sensing range at a
    representative start rather than at any particular rover. Does nothing for
    an observation model without a ``radius(env)``. When the radius covers the
    whole arena from that point (the ``sensor_radius: null`` default, R = the
    arena diagonal) the circle lies outside the plot, so a label says so instead.
    """
    radius_fn = getattr(env.observation, "radius", None)
    if radius_fn is None:
        return
    if pois is None:
        pois = env.pois.init(jax.random.PRNGKey(0), env)
    radius = float(radius_fn(env))
    x_min, x_max, y_min, y_max = env.spawn_region
    centre = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    width, height = env.world_size
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    covers_arena = radius >= np.max(np.linalg.norm(corners - centre, axis=1))
    colour = SERIES[0]

    ax.plot(*centre, "o", color=colour, markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=5)
    if covers_arena:
        ax.text(centre[0], centre[1] - 0.25,
                f"sensing radius {radius:.3g}\n(senses the whole arena)",
                ha="center", va="top", fontsize=7.5, color=colour, zorder=5)
        return

    ax.add_patch(Circle(centre, radius, facecolor=to_rgba(colour, 0.06),
                        edgecolor=colour, linewidth=1.2, linestyle="--", zorder=3))
    # Draw the radius along the direction that stays clearest of the POI discs.
    position = np.asarray(pois.position)
    poi_radius = np.asarray(pois.radius)
    best_angle, best_clearance = 0.0, -np.inf
    for angle in np.arange(8) * np.pi / 4 + np.pi / 4:
        tip = centre + radius * np.array([np.cos(angle), np.sin(angle)])
        samples = centre + np.linspace(0, 1, 9)[:, None] * (tip - centre)
        gaps = np.linalg.norm(samples[:, None] - position[None], axis=-1) - poi_radius
        clearance = gaps.min() if gaps.size else np.inf
        if clearance > best_clearance:
            best_angle, best_clearance = angle, clearance
    direction = np.array([np.cos(best_angle), np.sin(best_angle)])
    tip = centre + radius * direction
    ax.plot(*zip(centre, tip), color=colour, linewidth=1.0, zorder=4)
    middle = centre + 0.5 * radius * direction
    # Anchor the label on the side the line's normal points to, so it never
    # covers the line whatever the angle.
    normal = np.array([-direction[1], direction[0]])
    ha = "left" if normal[0] > 0.3 else "right" if normal[0] < -0.3 else "center"
    va = "bottom" if normal[1] > 0.3 else "top" if normal[1] < -0.3 else "center"
    ax.text(*(middle + 0.12 * normal), f"sensing R = {radius:g}", ha=ha, va=va,
            fontsize=7.5, color=colour, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))


def plot_arena(env, title=""):
    fig, ax = plt.subplots(figsize=(6.2, 6.6), facecolor=SURFACE)
    draw_arena(ax, env)
    draw_sensing_radius(ax, env)
    width, height = env.world_size
    pois = env.pois.init(jax.random.PRNGKey(0), env)
    total = float(np.sum(pois.value))
    ax.set_title(
        f"{title}\n{env.n_agents} rovers · {env.n_pois} POIs · {env.max_steps} steps · "
        f"{width:g} x {height:g} arena · max return {total:g}",
        fontsize=10, color=INK, loc="left",
    )
    fig.text(
        0.02, 0.01,
        "Number = coupling (rovers needed inside the disc at once). "
        "Grey disc = POI observation radius.\n"
        "Blue = one rover's sensing radius, drawn at the spawn-region centre.",
        fontsize=7.5, color=MUTED,
    )
    return fig


def _runs(mask):
    """(start, length) of every run of True in a 1-D bool array."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return [(s, e - s) for s, e in zip(edges[::2], edges[1::2])]


def plot_trajectory(env, traj, title=""):
    """Figure of one ``rollout`` episode (an unbatched ``Trajectory``)."""
    positions = np.asarray(traj.positions)
    headings = np.asarray(traj.headings)
    remaining = np.asarray(traj.poi_remaining)
    rewards = np.asarray(traj.rewards)
    if rewards.ndim == 2:  # per-agent rewards
        rewards = rewards.sum(axis=1)
    done = np.asarray(traj.done)
    n_steps, n_agents = rewards.shape[0], positions.shape[1]
    length = int(np.argmax(done)) + 1 if done.any() else n_steps

    payout = remaining[:-1] - remaining[1:]  # (T, P)
    depleted_at = {
        k: int(np.argmax(remaining[1:, k] <= 0)) + 1
        for k in range(remaining.shape[1])
        if np.any(remaining[1:, k] <= 0)
    }
    pois = env.pois.init(jax.random.PRNGKey(0), env)

    fig = plt.figure(figsize=(13, 6.4), facecolor=SURFACE)
    grid = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.1], hspace=0.45, wspace=0.18)
    ax_arena = fig.add_subplot(grid[:, 0])
    draw_arena(ax_arena, env, pois, depleted_at, coupling_inside=False)

    # More rovers than categorical slots fold into one neutral colour.
    colours = SERIES if n_agents <= len(SERIES) else [INK_2] * n_agents
    arrow = 0.04 * max(env.world_size)
    for i in range(n_agents):
        c = colours[i]
        path = positions[: length + 1, i]
        ax_arena.plot(path[:, 0], path[:, 1], color=c, linewidth=1.5,
                      solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax_arena.plot(*path[0], "o", color=c, markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
        end, h = path[-1], headings[length, i]
        ax_arena.annotate(
            "", xy=(end[0] + arrow * np.cos(h), end[1] + arrow * np.sin(h)), xytext=end,
            arrowprops=dict(arrowstyle="-|>", color=c, linewidth=1.5, mutation_scale=12),
            zorder=5,
        )
    if n_agents <= len(SERIES):
        handles = [Line2D([], [], color=colours[i], linewidth=2, label=f"R{i}")
                   for i in range(n_agents)]
    else:
        handles = [Line2D([], [], color=INK_2, linewidth=2, label=f"{n_agents} rovers")]
    handles.append(Line2D([], [], linestyle="none", marker="o", color=INK_2,
                          markersize=6, label="start"))
    ax_arena.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
                    ncol=min(len(handles), 7), frameon=False, fontsize=8,
                    labelcolor=INK_2, handlelength=1.6)

    # When each POI paid: one row per POI, filled where its coupling was met.
    ax_pay = fig.add_subplot(grid[0, 1])
    _style_axes(ax_pay)
    n_pois = remaining.shape[1]
    for k in range(n_pois):
        row = n_pois - 1 - k
        ax_pay.broken_barh(_runs(payout[:, k] > 0), (row - 0.3, 0.6),
                           facecolors=INK_2, edgecolor=SURFACE, linewidth=0)
        if k in depleted_at:
            ax_pay.text(depleted_at[k] + 0.01 * n_steps, row, "used", va="center",
                        fontsize=7.5, color=MUTED)
        elif not np.any(payout[:, k] > 0):
            ax_pay.text(0.01 * n_steps, row, "never paid", va="center",
                        fontsize=7.5, color=MUTED)
    ax_pay.set_yticks(range(n_pois))
    ax_pay.set_yticklabels([f"P{k} (c={int(pois.coupling[k])})" for k in reversed(range(n_pois))])
    ax_pay.set_xlim(0, n_steps)
    ax_pay.set_ylim(-0.7, n_pois - 0.3)
    ax_pay.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax_pay.set_axisbelow(True)
    ax_pay.set_title("Steps where each POI paid (coupling met)", fontsize=10,
                     color=INK, loc="left")

    # Cumulative team reward, against the most the episode could pay.
    ax_ret = fig.add_subplot(grid[1, 1], sharex=ax_pay)
    _style_axes(ax_ret)
    total = float(np.sum(pois.value))
    cumulative = np.concatenate([[0.0], np.cumsum(rewards)])[: length + 1]
    ax_ret.plot(np.arange(length + 1), cumulative, color=INK, linewidth=1.5)
    ax_ret.axhline(total, color=AXIS, linewidth=1.0)
    ax_ret.text(0.01 * n_steps, total, f"every POI used = {total:g}", ha="left",
                va="bottom", fontsize=7.5, color=MUTED)
    if length < n_steps:
        for ax in (ax_pay, ax_ret):
            ax.axvline(length, color=AXIS, linewidth=1.0)
        # Right of the line when there is room: the curve rises left of it.
        right = length < 0.7 * n_steps
        ax_ret.text(length, 0.02 * total,
                    f" episode ends t={length}" if right else f"episode ends t={length} ",
                    ha="left" if right else "right", va="bottom", fontsize=7.5, color=MUTED)
    ax_ret.set_ylim(0, 1.12 * total)
    ax_ret.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax_ret.set_axisbelow(True)
    ax_ret.set_xlabel("step", fontsize=8, color=MUTED)
    ax_ret.set_title("Cumulative team reward", fontsize=10, color=INK, loc="left")

    fig.suptitle(
        f"{title}   return {float(np.sum(rewards)):.2f} of {total:g} in {length} steps",
        x=0.02, ha="left", fontsize=11, color=INK,
    )
    return fig


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variants", nargs="*", default=None,
                        help="rover env groups (default: every conf/env/rover_*.yaml)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--policies", nargs="+", choices=sorted(POLICIES),
                        default=["random", "scripted"])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "plotting" / "rover")
    parser.add_argument("--no-trajectories", action="store_true",
                        help="only write the arena images")
    args = parser.parse_args(argv)

    for variant in args.variants or list_variants():
        env = load_rover_env(variant)
        print(save_figure(plot_arena(env, variant), args.out_dir / "arenas" / f"{variant}.png"))
        if args.no_trajectories:
            continue
        for name in args.policies:
            policy = POLICIES[name](env)
            run = jax.jit(lambda key: rollout(env, policy, key))
            for seed in args.seeds:
                traj = run(jax.random.PRNGKey(seed))
                fig = plot_trajectory(env, traj, f"{variant} · {name} policy · seed {seed}")
                out = args.out_dir / "trajectories" / f"{variant}_{name}_seed{seed}.png"
                print(save_figure(fig, out))


if __name__ == "__main__":
    main()
