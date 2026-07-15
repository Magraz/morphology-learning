"""Renderers for ``MultiBoxPushMJX``: pygame (Box2D-suite look) and native MuJoCo.

The env itself stays pure JAX (no render state, jit/vmap-safe); rendering is a
separate host-side module that consumes an ``EnvState``. Two backends:

- ``MJXRenderer`` (pygame): the Box2D-suite look — black boundary walls,
  translucent green DROP ZONE, colored coupling-numbered boxes, red indexed
  agent discs, and the per-agent sensor overlay driven by the observation.
- ``MuJoCoNativeRenderer`` (OpenGL): the scene as MuJoCo itself sees it,
  rendered with ``mujoco.Renderer`` against a cosmetic *visual twin* of the
  physics model (floor, arena walls, painted target band, lighting/shadows;
  same qpos layout). Cameras: tilted 3D ``"iso"`` or ``"top"`` (matches the
  pygame layout). Headless: set ``MUJOCO_GL=egl`` (or ``osmesa``).

Reuse: ``MJXRenderer`` subclasses the Box2D suite's ``Renderer`` and inherits
everything that doesn't touch Box2D bodies — boundary walls, target areas, the
arrow/HUD primitives, and the full sensor-overlay chain (density sectors,
lidar, nearest-box vector, goal distance). Only the body drawing (rotated
boxes, agent discs, labels) is reimplemented from the numpy snapshot of the
MJX state. The overlay is fed by slicing the actual observation vector, so —
like the Box2D overlay — it cannot drift from what the policy sees.

Extras over the Box2D renderer:
- ``mode="rgb_array"`` (default) draws to an offscreen surface and returns an
  (H, W, 3) uint8 frame — no display needed, works headless.
- a box whose coupling requirement is currently met (light mass) gets a green
  outline; delivered boxes are drawn washed out.
- ``save_video(frames, path)`` writes .mp4 or .gif via imageio.

Typical use::

    env = MultiBoxPushMJX(n_agents=9, n_objects=3)
    renderer = MJXRenderer(env)                    # or mode="human"
    obs, state = jax.jit(env.reset)(key)
    frames = []
    for _ in range(300):
        obs, state, *_ = step(state, actions)
        frames.append(renderer.render(state, obs=obs))
    save_video(frames, "rollout.mp4", fps=30)

For a vmapped batch, render one env with
``jax.tree.map(lambda x: x[i], state)``.

Demo (headless, writes multi_box_push_mjx.mp4 + a sample png; add
``--native iso`` or ``--native top`` for the MuJoCo renderer):
    MUJOCO_GL=egl SDL_VIDEODRIVER=dummy uv run python -m environments.mjx_suite.renderer
"""

from types import SimpleNamespace

import numpy as np
import pygame

from environments.box2d_suite.observation import (
    BASE_OBS_DIM,
    N_LIDAR_RAYS,
    OBS_DIM,
    ObservationManager,
)
from environments.box2d_suite.renderer import Renderer
from environments.box2d_suite.utils import COLORS_LIST
from environments.mjx_suite.multi_box_push_mjx import (
    _AGENT_RADIUS,
    EnvState,
    MultiBoxPushMJX,
)

# Observation layout slices (see ObservationManager.get_observation):
# [vel 0:2 | density 2:18 | touching 18 | neigh 19 | contact 20 |
#  box_vec 21:23 | goal 23 | lidar 24:40]
_DENSITY_SLICE = slice(2, 18)
_BOX_VEC_SLICE = slice(21, 23)
_GOAL_IDX = 23
_LIDAR_SLICE = slice(BASE_OBS_DIM, OBS_DIM)

_AGENT_COLOR = (200, 50, 50)  # Box2D Agent.render_circle "closed" color
_COUPLED_OUTLINE = (0, 170, 0)


class MJXRenderer(Renderer):
    """Draws ``MultiBoxPushMJX`` states with the shared Box2D-suite Renderer."""

    def __init__(
        self,
        env: MultiBoxPushMJX,
        mode: str = "rgb_array",
        screen_size: tuple[int, int] = (700, 700),
        fps: int = 30,
    ):
        super().__init__(env)  # only reads env.world_width (sets scale)
        self.mode = mode
        self.screen_size = screen_size
        self.scale = screen_size[0] / env.world_width
        self.fps = fps

        # The inherited _draw_lidar reaches for
        # env.observation_manager.lidar_directions — a @staticmethod on
        # ObservationManager, so a namespace shim satisfies it without
        # building a Box2D observation manager.
        if not hasattr(env, "observation_manager"):
            env.observation_manager = SimpleNamespace(
                lidar_directions=ObservationManager.lidar_directions
            )

    # ---------------------------------------------------------------- frame

    def render(
        self,
        state: EnvState,
        obs: np.ndarray | None = None,
        focus_agent: int | None = 0,
    ) -> np.ndarray | None:
        """Draw one frame from an ``EnvState``.

        Args:
            state: single-env EnvState (index a vmapped batch first).
            obs: optional (n_agents, OBS_DIM) observation for the sensor
                overlay of ``focus_agent``; overlay is skipped when None.
            focus_agent: whose observation to overlay (None disables it).

        Returns:
            (H, W, 3) uint8 frame in ``rgb_array`` mode, else None.
        """
        self._ensure_screen()
        snap = self._snapshot(state)

        self.screen.fill((255, 255, 255))
        self._draw_boundary_walls()  # inherited
        self._draw_target_areas()  # inherited (env.target_areas)
        self._draw_boxes(snap)
        self._draw_box_coupling(snap)
        self._draw_agents(snap)
        if obs is not None and focus_agent is not None:
            self._draw_obs_overlay(snap, np.asarray(obs), focus_agent)

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2)).copy()

    def close(self):
        if self.screen is not None:
            if self.mode == "human":
                pygame.quit()
            self.screen = None

    # ---------------------------------------------------------------- internals

    def _ensure_screen(self):
        if self.screen is not None:
            return
        if self.mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("MultiBoxPush MJX")
            self.clock = pygame.time.Clock()
        else:
            pygame.font.init()
            self.screen = pygame.Surface(self.screen_size)

    def _snapshot(self, state: EnvState) -> dict:
        """Pull everything drawable out of the jax state as numpy."""
        env = self.env
        agent_pos_j = env._agent_pos(state.data)
        box_pos_j, box_yaw_j = env._box_pose(state.data)
        touch = np.asarray(env._touch_matrix(agent_pos_j, box_pos_j, box_yaw_j))
        return {
            "agent_pos": np.asarray(agent_pos_j),  # (A, 2)
            "box_pos": np.asarray(box_pos_j),  # (O, 2)
            "box_yaw": np.asarray(box_yaw_j),  # (O,)
            "delivered": np.asarray(state.delivered),
            "n_touch": touch.sum(axis=0),  # (O,)
        }

    def _draw_boxes(self, snap):
        env = self.env
        for j in range(env.n_objects):
            color = COLORS_LIST[(env.n_agents + j) % len(COLORS_LIST)]
            if snap["delivered"][j]:  # wash out delivered boxes
                color = tuple(int(c + (255 - c) * 0.65) for c in color)

            h = env.box_half_extents[j]
            cx, cy = snap["box_pos"][j]
            yaw = snap["box_yaw"][j]
            c, s = np.cos(yaw), np.sin(yaw)
            corners = [
                self._to_screen(cx + c * dx - s * dy, cy + s * dx + c * dy)
                for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))
            ]
            pygame.draw.polygon(self.screen, color, corners)
            coupled = snap["n_touch"][j] >= env.objects_push_coupling_list[j]
            pygame.draw.polygon(
                self.screen,
                _COUPLED_OUTLINE if coupled else (0, 0, 0),
                corners,
                4 if coupled else 2,
            )

    def _draw_box_coupling(self, snap):
        """Coupling requirement (and live touch count) on each box."""
        if self._coupling_font is None:
            pygame.font.init()
            self._coupling_font = pygame.font.SysFont("Arial", 20, bold=True)

        for j in range(self.env.n_objects):
            center = self._to_screen(*snap["box_pos"][j])
            text = f"{int(snap['n_touch'][j])}/{self.env.objects_push_coupling_list[j]}"
            surface = self._coupling_font.render(text, True, (255, 255, 255))
            rect = surface.get_rect(center=center)
            for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                outline = self._coupling_font.render(text, True, (0, 0, 0))
                self.screen.blit(outline, rect.move(offset))
            self.screen.blit(surface, rect)

    def _draw_agents(self, snap):
        if self._index_font is None:
            pygame.font.init()
            self._index_font = pygame.font.SysFont("Arial", 12, bold=True)

        radius = int(_AGENT_RADIUS * self.scale)
        for i, pos in enumerate(snap["agent_pos"]):
            center = self._to_screen(*pos)
            pygame.draw.circle(self.screen, _AGENT_COLOR, center, radius)
            pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 2)

            text = str(i)
            surface = self._index_font.render(text, True, (255, 255, 255))
            rect = surface.get_rect(center=(center[0] + 5, center[1] + 5))
            for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                outline = self._index_font.render(text, True, (0, 0, 0))
                self.screen.blit(outline, rect.move(offset))
            self.screen.blit(surface, rect)

    def _draw_obs_overlay(self, snap, obs, focus_agent):
        """Sensor overlay for one agent, fed by its actual observation vector.

        Builds the readout dict the inherited overlay drawers expect
        (Renderer._draw_density_sectors/_draw_lidar/...), sliced straight from
        the policy input.
        """
        if self._sensor_font is None:
            pygame.font.init()
            self._sensor_font = pygame.font.SysFont("Arial", 12)

        idx = focus_agent % self.env.n_agents
        o = obs[idx]
        readout = {
            "density": o[_DENSITY_SLICE],
            "lidar": o[_LIDAR_SLICE],
            "nearest_box_vec": o[_BOX_VEC_SLICE],
            "goal_distance": float(o[_GOAL_IDX]),
            "sector_radius": self.env.sector_sensor_radius,
            "lidar_range": self.env.lidar_range,
            "n_lidar_rays": N_LIDAR_RAYS,
        }
        origin = tuple(snap["agent_pos"][idx])
        center = self._to_screen(*origin)

        self._draw_lidar(origin, center, readout)  # inherited
        self._draw_density_sectors(center, readout)  # inherited
        self._draw_goal_distance(origin, center)  # inherited
        self._draw_nearest_box_vec(origin, center, readout)  # inherited
        pygame.draw.circle(
            self.screen, (0, 0, 0), center, int(_AGENT_RADIUS * self.scale) + 3, 2
        )
        self._draw_sensor_hud(idx, readout)  # inherited


class MuJoCoNativeRenderer:
    """Native MuJoCo (OpenGL) rendering of ``MultiBoxPushMJX`` states.

    Renders through ``mujoco.Renderer`` against the env's *visual twin* model
    (``env._build_xml(..., visual=True)``): identical bodies/joints — so the
    MJX ``qpos`` copies straight across — plus cosmetic-only floor, arena
    walls, painted target band, skybox and lighting. Each ``render(state)``
    copies qpos into a host ``MjData``, runs ``mj_forward`` (kinematics only,
    never stepped), and returns an (H, W, 3) uint8 frame.

    Live state is echoed into geom colors: a box whose coupling requirement is
    currently met is tinted toward green (light mass), delivered boxes fade
    translucent.

    Headless use needs a GL backend: run with ``MUJOCO_GL=egl`` (GPU) or
    ``MUJOCO_GL=osmesa``. Cameras: ``"iso"`` (default, tilted 3D view) or
    ``"top"`` (top-down, matches the pygame layout).
    """

    def __init__(
        self,
        env: MultiBoxPushMJX,
        width: int = 700,
        height: int = 700,
        camera: str = "iso",
    ):
        import mujoco

        self.env = env
        self._model = mujoco.MjModel.from_xml_string(
            env._build_xml(env._heavy_mass_np, visual=True)
        )
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height, width)

        self._box_geom_ids = [
            self._model.geom(f"g_box_{j}").id for j in range(env.n_objects)
        ]
        self._base_box_rgba = self._model.geom_rgba[self._box_geom_ids].copy()

        cx, cy = env.world_width / 2, env.world_height / 2
        self._camera = mujoco.MjvCamera()
        self._camera.lookat[:] = (cx, cy, 0.0)
        if camera == "top":
            self._camera.distance = 1.25 * env.world_width
            self._camera.elevation = -90.0
            self._camera.azimuth = 90.0
        elif camera == "iso":
            self._camera.lookat[1] = cy * 0.9
            self._camera.distance = 1.35 * env.world_width
            self._camera.elevation = -55.0
            self._camera.azimuth = 90.0
        else:
            raise ValueError(f"unknown camera: {camera!r} (use 'iso' or 'top')")

    def render(self, state: EnvState) -> np.ndarray:
        """One (H, W, 3) uint8 frame from a single-env EnvState."""
        import mujoco

        env = self.env
        self._data.qpos[:] = np.asarray(state.data.qpos)
        mujoco.mj_forward(self._model, self._data)

        # coupling / delivered feedback via geom colors
        agent_pos = env._agent_pos(state.data)
        box_pos, box_yaw = env._box_pose(state.data)
        touch = np.asarray(env._touch_matrix(agent_pos, box_pos, box_yaw)).sum(axis=0)
        delivered = np.asarray(state.delivered)
        for j, gid in enumerate(self._box_geom_ids):
            rgba = self._base_box_rgba[j].copy()
            if touch[j] >= env.objects_push_coupling_list[j]:
                rgba[:3] = 0.45 * rgba[:3] + 0.55 * np.array([0.15, 0.85, 0.25])
            if delivered[j]:
                rgba[3] = 0.35
            self._model.geom_rgba[gid] = rgba

        self._renderer.update_scene(self._data, camera=self._camera)
        return self._renderer.render()

    def close(self):
        self._renderer.close()


def save_video(frames, path, fps: int = 30):
    """Write rgb frames to .mp4 or .gif (by extension) via imageio."""
    import imageio.v3 as iio

    frames = np.asarray(frames, dtype=np.uint8)
    if str(path).endswith(".gif"):
        iio.imwrite(path, frames, duration=1000 / fps, loop=0)
    else:
        iio.imwrite(path, frames, fps=fps)
    print(f"wrote {len(frames)} frames to {path}")


if __name__ == "__main__":
    import argparse

    import jax

    from environments.mjx_suite.multi_box_push_mjx import scripted_push_action

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=9)
    parser.add_argument("--n-objects", type=int, default=3)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=2, help="render every Nth step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="multi_box_push_mjx.mp4")
    parser.add_argument("--frame-png", default=None,
                        help="also save one mid-rollout frame to this png path")
    parser.add_argument("--native", choices=["iso", "top"], default=None,
                        help="render with the native MuJoCo (OpenGL) renderer "
                             "instead of pygame; needs MUJOCO_GL=egl headless")
    args = parser.parse_args()

    env = MultiBoxPushMJX(n_agents=args.n_agents, n_objects=args.n_objects)
    if args.native:
        renderer = MuJoCoNativeRenderer(env, camera=args.native)
        draw = lambda state, obs: renderer.render(state)  # noqa: E731
    else:
        renderer = MJXRenderer(env)
        draw = lambda state, obs: renderer.render(state, obs=obs)  # noqa: E731
    step = jax.jit(env.step)

    obs, state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    frames = [draw(state, obs)]
    for i in range(args.steps):
        obs, state, reward, term, trunc, info = step(
            state, scripted_push_action(env, state)
        )
        if (i + 1) % args.every == 0:
            frames.append(draw(state, obs))
        if bool(term) or bool(trunc):
            print(f"episode ended at step {i + 1}")
            break

    save_video(frames, args.out, fps=30)
    if args.frame_png is not None:
        import imageio.v3 as iio

        k = len(frames) * 3 // 4  # late-rollout frame: agents mid-push
        iio.imwrite(args.frame_png, frames[k])
        print(f"wrote frame {k} to {args.frame_png}")
    renderer.close()
