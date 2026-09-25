"""Goal-following video for the feudal `view()`: per-agent overlay + side panel.

Drawn AFTER the episode, onto frames the renderer already produced, because the
horizon score of a goal is only known `c` steps after it is issued and the side
panel needs the whole episode. Every mark therefore reads the same
`frame_alignment` arrays as the raster PNG, so the video and the PNG cannot
disagree. Marks, per agent:

* ring (outer band): horizon alignment, cos(g_(d-c), s_d - s_(d-c)); a thin muted
  outline instead of a band while it is undefined;
* dot (centre): one-step alignment, cos(w_(d-1), s_d - s_(d-1));
* black arrow: the action change the goal causes, clip(mu(obs, w)) -
  clip(mu(obs, 0)), at a FIXED scale (full force = one arrow length), so arrows
  compare across frames and episodes. Continuous-action envs only;
* violet (grounded goal spaces only): the goal in the arena — a heading arrow for
  `position_direction`; the latched waypoint, a line to it, and the path since
  the latch for `position_waypoint`. A LATENT goal gets no arena mark: any
  latent-to-world decode would be a fabricated picture.
"""

from pathlib import Path

import numpy as np
import pygame

from algorithms.feudal_mappo_jax.goal_visualization import (
    alignment_rgb,
    goal_following_figure,
    rasterize_panel,
)
from environments.box2d_suite.renderer import draw_arrow_head

#: Horizon cosine above which the HUD counts an agent as following its goal.
FOLLOW_THRESHOLD = 0.5
#: Height of the blank band atop the side panel that carries per-frame text.
HUD_PX = 64

# Reference-palette ink roles, as RGB for pygame.
_INK = (11, 11, 11)
_INK_SECONDARY = (82, 81, 78)
_MUTED = (137, 135, 129)
_SURFACE = (252, 252, 251)
_GOAL = (74, 58, 167)  # violet: world-space goal marks


def _world_goal_marks(goal_space, goal_states, goals, pooled_goals, radius, to_world):
    """Per-decision arena marks: ``(heading (T,N,2) | None, waypoint (T,N,2) | None)``."""
    if goal_space == "position_direction":
        s = np.asarray(goal_states)[: len(goals)]
        step = to_world(s + np.asarray(goals)) - to_world(s)
        norm = np.linalg.norm(step, axis=-1, keepdims=True)
        heading = np.divide(step, norm, out=np.full_like(step, np.nan), where=norm > 0)
        return heading, None
    if goal_space == "position_waypoint":
        s = np.asarray(goal_states)[: len(pooled_goals)]
        return None, to_world(s + radius * np.asarray(pooled_goals))
    return None, None


def _haloed_line(surface, color, start, end, width):
    """A line over a surface-colored halo, so it reads on boxes and bands too."""
    pygame.draw.line(surface, _SURFACE, start, end, width + 3)
    pygame.draw.line(surface, color, start, end, width)


def _haloed_arrow(surface, color, start, end, width, head=9):
    _haloed_line(surface, color, start, end, width)
    draw_arrow_head(surface, start, end, _SURFACE, size=head + 3)
    draw_arrow_head(surface, start, end, color, size=head)


def draw_goal_overlay(
    surface, to_screen, scale, *, agent_pos, agent_radius, ring, dot,
    influence=None, heading=None, waypoint=None, trail=None,
):
    """Draw one frame's goal marks onto `surface` (world coordinates in).

    agent_pos (N, 2); ring / dot (N,) cosines, NaN = undefined; influence (N, 2)
    in env action units (+-1 = full force) or None; heading (N, 2) unit world
    directions or None; waypoint (N, 2) world points or None; trail (K, N, 2)
    world path since the waypoint latched, or None.
    """
    # float64 throughout: pygame rejects numpy float32 scalars as coordinates,
    # and JAX actions arrive as float32.
    influence, heading, waypoint, trail = (
        None if x is None else np.asarray(x, dtype=np.float64)
        for x in (influence, heading, waypoint, trail)
    )
    r_agent = max(2, int(agent_radius * scale))
    band = 4
    r_ring = r_agent + 2 + band  # 2px surface gap between the agent and its ring
    arrow_px = 3 * r_ring
    ring_rgb, dot_rgb = alignment_rgb(ring), alignment_rgb(dot)
    for i, pos in enumerate(np.asarray(agent_pos, dtype=np.float64)):
        center = to_screen(*pos)
        # Arena goal marks first, so the agent's own marks sit on top.
        if trail is not None and len(trail) > 1:
            points = [to_screen(*p) for p in trail[:, i]]
            pygame.draw.lines(surface, _SURFACE, False, points, 5)
            pygame.draw.lines(surface, _GOAL, False, points, 2)
        if waypoint is not None:
            tip = to_screen(*waypoint[i])
            _haloed_line(surface, _GOAL, center, tip, 1)
            for dx, dy in ((1, 1), (1, -1)):
                _haloed_line(surface, _GOAL, (tip[0] - 6 * dx, tip[1] - 6 * dy),
                             (tip[0] + 6 * dx, tip[1] + 6 * dy), 3)
        if heading is not None and np.isfinite(heading[i]).all():
            # Screen y is flipped relative to world y.
            tip = (center[0] + heading[i, 0] * arrow_px, center[1] - heading[i, 1] * arrow_px)
            _haloed_arrow(surface, _GOAL, center, tip, 3)

        if np.isfinite(ring[i]):
            pygame.draw.circle(surface, tuple(int(c) for c in ring_rgb[i]), center, r_ring, band)
            pygame.draw.circle(surface, _INK_SECONDARY, center, r_ring, 1)
            pygame.draw.circle(surface, _INK_SECONDARY, center, r_ring - band, 1)
        else:
            pygame.draw.circle(surface, _MUTED, center, r_ring, 1)

        if influence is not None:
            tip = (center[0] + influence[i, 0] * arrow_px,
                   center[1] - influence[i, 1] * arrow_px)
            if np.hypot(tip[0] - center[0], tip[1] - center[1]) >= 2:
                _haloed_arrow(surface, _INK, center, tip, 2, head=8)

        if np.isfinite(dot[i]):
            r_dot = max(2, r_agent // 3)  # small, so the red agent disc still shows
            pygame.draw.circle(surface, _SURFACE, center, r_dot + 1)
            pygame.draw.circle(surface, tuple(int(c) for c in dot_rgb[i]), center, r_dot)


def hud_lines(f, decision, ring, dot, *, macro_steps=False):
    """The per-frame numbers shown in the panel's top band."""
    step = f"physics step {f} · decision {decision}" if macro_steps else f"step {f}"
    defined = np.isfinite(ring)
    if defined.any():
        following = int((ring[defined] > FOLLOW_THRESHOLD).sum())
        horizon = (f"horizon cos: mean {np.mean(ring[defined]):+.2f} · following "
                   f"(> +{FOLLOW_THRESHOLD}) {following}/{int(defined.sum())}")
    else:
        horizon = "horizon cos: undefined (no complete horizon yet)"
    one_step = (f"one-step cos: mean {np.nanmean(dot):+.2f}" if np.isfinite(dot).any()
                else "one-step cos: undefined")
    return [step, horizon, one_step]


def _panel_frame(panel, x, rows, lines, font):
    """A copy of `panel` with the cursor at column `x` and `lines` in the HUD band."""
    out = panel.copy()
    out[rows[0]:rows[1], max(0, x - 1):x + 1] = _INK
    surface = pygame.surfarray.make_surface(np.transpose(out, (1, 0, 2)))
    for k, line in enumerate(lines):
        surface.blit(font.render(line, True, _INK if k == 0 else _INK_SECONDARY),
                     (12, 8 + 18 * k))
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def save_goal_video(
    path: Path, renderer, frames, *, agent_pos, frame_decision, ring, dot, rewards,
    horizon, goal_space, episode, agent_radius, macro_steps=False, influence=None,
    goal_states=None, goals=None, pooled_goals=None, waypoint_radius=None,
    to_world=None, fps=30,
):
    """Write `frames` with goal marks beside the synced goal-following panel.

    `renderer` supplies `annotate(frame, draw)`; per-frame inputs are indexed by
    frame (agent_pos (F,N,2), frame_decision (F,), ring/dot (F,N), rewards (F,));
    influence (T,N,2), goal_states (T+1,N,2), goals / pooled_goals (T,N,2) are per
    policy decision. Frames are streamed, never copied as a whole episode.
    """
    import imageio

    if not len(frames):
        return
    agent_pos = np.asarray(agent_pos, dtype=np.float64)
    frame_decision = np.asarray(frame_decision)
    heading = waypoint = None
    if to_world is not None and goal_states is not None:
        heading, waypoint = _world_goal_marks(
            goal_space, goal_states, goals, pooled_goals, waypoint_radius, to_world,
        )

    height = frames[0].shape[0]
    fig, axes = goal_following_figure(
        ring, dot, rewards, horizon=horizon, goal_space=goal_space, episode=episode,
        macro_steps=macro_steps, size_px=(height - height % 2, height), hud_px=HUD_PX,
    )
    # The three rows share one x extent, so the cursor spans all of them.
    cursor_x, rows, panel = rasterize_panel(fig, axes, len(frames))
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 14)

    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for f, frame in enumerate(frames):
            d = int(frame_decision[f])
            trail = None
            if waypoint is not None:
                # The path since this decision's waypoint latched (latch every c).
                first = int(np.searchsorted(frame_decision, (d // horizon) * horizon))
                trail = agent_pos[first:f + 1]

            def draw(surface, to_screen, scale, f=f, d=d, trail=trail):
                draw_goal_overlay(
                    surface, to_screen, scale, agent_pos=agent_pos[f],
                    agent_radius=agent_radius, ring=ring[f], dot=dot[f],
                    influence=None if influence is None else influence[d],
                    heading=None if heading is None else heading[d],
                    waypoint=None if waypoint is None else waypoint[d],
                    trail=trail,
                )

            left = renderer.annotate(frame, draw)
            right = _panel_frame(
                panel, int(cursor_x[f]), rows,
                hud_lines(f, d, ring[f], dot[f], macro_steps=macro_steps), font,
            )
            writer.append_data(np.concatenate([left, right[: left.shape[0]]], axis=1))
    print(f"Goal video saved to {path}")
