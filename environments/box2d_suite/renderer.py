import math

import numpy as np
import pygame
from Box2D import b2CircleShape, b2PolygonShape

from environments.box2d_suite.utils import COLORS_LIST


class Renderer:
    """Encapsulates all pygame rendering for MultiBoxPushEnv."""

    # Sensor-overlay palette (see _draw_sensor_overlay).
    SECTOR_COLOR = (170, 170, 170)
    AGENT_DENSITY_COLOR = (0, 0, 255)
    OBJECT_DENSITY_COLOR = (230, 120, 0)
    LIDAR_CLEAR_COLOR = (205, 205, 205)
    LIDAR_HIT_COLOR = (220, 50, 50)
    BOX_VEC_COLOR = (190, 0, 190)
    GOAL_COLOR = (0, 150, 0)

    def __init__(self, env):
        self.env = env

        self.screen = None
        self.clock = None
        self.screen_size = (700, 700)
        self.scale = self.screen_size[0] / env.world_width

        # Lazy-initialized fonts
        self._sensor_font = None
        self._target_font = None
        self._index_font = None
        self._coupling_font = None

    def render(self):
        if self.env.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Salp Chain Simulation")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Draw boundary walls correctly positioned
        self._draw_boundary_walls()

        # Draw target areas before or after drawing agents
        self._draw_target_areas()

        # Draw objects
        self._draw_dynamic_objects()
        self._draw_object_coupling()

        # Draw obstacle-course walls / gaps (no-op for envs without walls)
        self._draw_walls()

        # Draw active grab joints (no-op for envs without agent_joints)
        self._draw_grab_joints()

        # Draw agents
        self._render_agents_as_circles()

        # Draw agent indices on top of agents
        self._draw_agent_indices()

        # Observation overlay for one focus agent (env.render_sensor_agent)
        self._draw_sensor_overlay()

        # Draw force vectors
        # self._draw_force_vectors()

        pygame.display.flip()
        self.clock.tick(self.env.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def _render_agents_as_circles(self):
        for agent in self.env.agents:
            agent.render_circle(self.screen, self.screen_size, self.scale, False)

    def _render_agents_as_boxes(self):
        for agent in self.env.agents:
            color = COLORS_LIST[agent.index % len(COLORS_LIST)]
            agent.render_box(self.screen, self.screen_size, self.scale, color)

    def _draw_boundary_walls(self):
        """Draw boundary walls flush with screen edges."""
        sw, sh = self.screen_size
        t = max(1, int(self.env.boundary_thickness * self.scale))

        # Bottom wall
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(0, sh - t, sw, t))
        # Top wall
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(0, 0, sw, t))
        # Left wall
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(0, 0, t, sh))
        # Right wall
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(sw - t, 0, t, sh))

    def _draw_force_vectors(self):
        """Draw force vectors for each agent with enhanced 2D visualization."""
        for agent, force in zip(self.env.agents, self.env.applied_forces):
            color = COLORS_LIST[agent.index % len(COLORS_LIST)]
            agent.render_force(
                self.screen,
                self.screen_size,
                self.scale,
                force,
                self.env.force_scale,
                color,
            )

    def _draw_dynamic_objects(self):
        """Draw the extra objects (Square, Triangle, Circle)"""
        # Not every env has dynamic objects — nothing to draw when absent.
        for body in getattr(self.env, "objects", []):
            color = body.userData.get("color", (128, 128, 128))

            for fixture in body.fixtures:
                shape = fixture.shape

                if isinstance(shape, b2PolygonShape):
                    # Get transformed vertices for polygon/box
                    vertices = [
                        (body.transform * v) * self.scale for v in shape.vertices
                    ]
                    # Convert to screen coordinates (flip Y)
                    vertices = [(v[0], self.screen_size[1] - v[1]) for v in vertices]

                    pygame.draw.polygon(self.screen, color, vertices)
                    pygame.draw.polygon(self.screen, (0, 0, 0), vertices, 2)  # Outline

                elif isinstance(shape, b2CircleShape):
                    # Draw circle
                    center_x = body.position.x * self.scale
                    center_y = self.screen_size[1] - body.position.y * self.scale
                    radius = shape.radius * self.scale

                    pygame.draw.circle(
                        self.screen, color, (int(center_x), int(center_y)), int(radius)
                    )
                    pygame.draw.circle(
                        self.screen,
                        (0, 0, 0),
                        (int(center_x), int(center_y)),
                        int(radius),
                        2,
                    )  # Outline

    def _to_screen(self, x, y):
        """World (x, y) -> screen pixels (y flipped)."""
        return (int(x * self.scale), int(self.screen_size[1] - y * self.scale))

    def _draw_sensor_overlay(self):
        """Draw the observation of a single focus agent on top of the world.

        Shows every spatial component of `ObservationManager.get_observation`:
        the 8+8 density sectors (agents / objects), the lidar scan,
        `nearest_box_vec` and `goal_distance`. The values come straight from
        `ObservationManager.get_sensor_readout`, i.e. the same code paths that
        build the policy's observation, so the overlay cannot drift from it.

        The focus agent is `env.render_sensor_agent` (default 0) — drawing all
        agents at once is unreadable past a handful of them.
        """
        obs_manager = getattr(self.env, "observation_manager", None)
        if obs_manager is None or not self.env.agents:
            return

        if self._sensor_font is None:
            pygame.font.init()
            self._sensor_font = pygame.font.SysFont("Arial", 12)

        idx = getattr(self.env, "render_sensor_agent", 0) % len(self.env.agents)
        agent = self.env.agents[idx]
        readout = obs_manager.get_sensor_readout(idx)

        origin = (agent.position.x, agent.position.y)
        center = self._to_screen(*origin)

        # Lidar goes down first so the density sectors and the goal/box vectors
        # stay legible on top of it.
        self._draw_lidar(origin, center, readout)
        self._draw_density_sectors(center, readout)
        self._draw_goal_distance(origin, center)
        self._draw_nearest_box_vec(origin, center, readout)

        # Ring the focus agent so it is obvious which observation this is.
        pygame.draw.circle(
            self.screen, (0, 0, 0), center, int(agent.radius * self.scale) + 3, 2
        )
        self._draw_sensor_hud(idx, readout)

    def _draw_density_sectors(self, center, readout):
        """8 agent-density + 8 object-density sectors around the focus agent."""
        n_sectors = 8
        sector_step = 360 / n_sectors
        shift_degrees = 22.5  # Sectors are shifted counter-clockwise (see obs manager)

        agent_densities = readout["density"][:n_sectors]
        object_densities = readout["density"][n_sectors : n_sectors * 2]
        sensor_radius = readout["sector_radius"] * self.scale

        center_x, center_y = center
        bounds = pygame.Rect(
            int(center_x - sensor_radius),
            int(center_y - sensor_radius),
            int(sensor_radius * 2),
            int(sensor_radius * 2),
        )

        for sector in range(n_sectors):
            start_rad = math.radians(sector * sector_step + shift_degrees)
            end_rad = math.radians((sector + 1) * sector_step + shift_degrees)

            # Sector boundary spokes + the arc closing them off.
            for ray_rad in (start_rad, end_rad):
                edge = (
                    int(center_x + sensor_radius * math.cos(ray_rad)),
                    int(center_y - sensor_radius * math.sin(ray_rad)),
                )
                pygame.draw.line(self.screen, self.SECTOR_COLOR, center, edge, 1)
            pygame.draw.arc(
                self.screen, self.SECTOR_COLOR, bounds, start_rad, end_rad, 1
            )

            # Values at the middle of the sector: A = agents, O = objects. Both
            # are "closeness" in [0, 1] (1 - centroid_dist / radius), 0 == empty.
            mid_rad = (start_rad + end_rad) / 2
            text_x = center_x + sensor_radius * 0.55 * math.cos(mid_rad)
            text_y = center_y - sensor_radius * 0.55 * math.sin(mid_rad)

            for line, (label, value, color) in enumerate(
                (
                    ("A", agent_densities[sector], self.AGENT_DENSITY_COLOR),
                    ("O", object_densities[sector], self.OBJECT_DENSITY_COLOR),
                )
            ):
                # Grey out empty sectors so the occupied ones stand out.
                shown = color if value > 0 else (150, 150, 150)
                surface = self._sensor_font.render(f"{label}:{value:.2f}", True, shown)
                rect = surface.get_rect(
                    center=(int(text_x), int(text_y) - 6 + line * 12)
                )
                self.screen.blit(surface, rect)

    def _draw_lidar(self, origin, center, readout):
        """Lidar rays out to their hit points, with a dot at each hit."""
        obs_manager = self.env.observation_manager
        lidar = readout["lidar"]
        max_range = readout["lidar_range"]
        dirs = obs_manager.lidar_directions(readout["n_lidar_rays"])

        for r, fraction in enumerate(lidar):
            hit = self._to_screen(
                origin[0] + float(dirs[r, 0]) * max_range * float(fraction),
                origin[1] + float(dirs[r, 1]) * max_range * float(fraction),
            )
            # fraction == 1.0 means the ray reached full range without hitting
            # anything — draw it faint and unterminated.
            clear = fraction >= 1.0
            pygame.draw.line(
                self.screen,
                self.LIDAR_CLEAR_COLOR if clear else self.LIDAR_HIT_COLOR,
                center,
                hit,
                1,
            )
            if not clear:
                pygame.draw.circle(self.screen, self.LIDAR_HIT_COLOR, hit, 3)

    def _draw_nearest_box_vec(self, origin, center, readout):
        """Arrow to the nearest object — the (dx, dy) the policy is fed."""
        # Normalized by world_width in the observation; scale it back to world units.
        vec = readout["nearest_box_vec"] * float(self.env.world_width)
        if not vec.any():  # no objects in this env — the obs is a zero vector
            return

        tip = self._to_screen(origin[0] + float(vec[0]), origin[1] + float(vec[1]))
        pygame.draw.line(self.screen, self.BOX_VEC_COLOR, center, tip, 3)
        self._draw_arrow_head(center, tip, self.BOX_VEC_COLOR)

    def _draw_goal_distance(self, origin, center):
        """Segment from the agent to the target band, along the env's goal axis."""
        target_areas = getattr(self.env, "target_areas", None)
        if not target_areas:  # no goal region — the obs is 0
            return

        target = target_areas[0]
        # Match the observation: the distance is measured along one axis only.
        if getattr(self.env, "goal_axis", "y") == "x":
            tip = self._to_screen(target.x, origin[1])
        else:
            tip = self._to_screen(origin[0], target.y)

        pygame.draw.line(self.screen, self.GOAL_COLOR, center, tip, 3)
        self._draw_arrow_head(center, tip, self.GOAL_COLOR)

    def _draw_arrow_head(self, start, end, color, size=9):
        """Filled triangle at `end`, pointing away from `start`."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        # Two points size/2 either side of the shaft, size back from the tip.
        left = (end[0] - ux * size - uy * size / 2, end[1] - uy * size + ux * size / 2)
        right = (end[0] - ux * size + uy * size / 2, end[1] - uy * size - ux * size / 2)
        pygame.draw.polygon(self.screen, color, [end, left, right])

    def _draw_sensor_hud(self, idx, readout):
        """Legend + the scalar obs values for the focus agent, top-left."""
        box_vec = readout["nearest_box_vec"]
        lines = [
            (f"agent {idx} observation", (0, 0, 0)),
            ("A: agent density (8 sectors)", self.AGENT_DENSITY_COLOR),
            ("O: object density (8 sectors)", self.OBJECT_DENSITY_COLOR),
            (f"lidar: {readout['n_lidar_rays']} rays", self.LIDAR_HIT_COLOR),
            (
                f"nearest_box_vec: ({box_vec[0]:+.3f}, {box_vec[1]:+.3f})",
                self.BOX_VEC_COLOR,
            ),
            (f"goal_distance: {readout['goal_distance']:+.3f}", self.GOAL_COLOR),
        ]
        for i, (text, color) in enumerate(lines):
            surface = self._sensor_font.render(text, True, color)
            self.screen.blit(surface, (10, 10 + i * 14))

    def _draw_target_areas(self):
        """Draw target areas with their coupling requirements"""
        # Not every env defines target areas — nothing to draw when absent.
        target_areas = getattr(self.env, "target_areas", None)
        if not target_areas:
            return

        if self._target_font is None:
            pygame.font.init()
            self._target_font = pygame.font.SysFont("Arial", 14)

        for area in target_areas:

            # Check if it's our new ObjectTargetArea (has width/height)
            if hasattr(area, "width"):
                # Compute pixel bounds from Box2D coords
                left = int((area.x - area.width / 2) * self.scale)
                right = int((area.x + area.width / 2) * self.scale)
                top = int(self.screen_size[1] - (area.y + area.height / 2) * self.scale)
                bottom = int(
                    self.screen_size[1] - (area.y - area.height / 2) * self.scale
                )
                px_w = right - left
                px_h = bottom - top

                # Draw filled transparent rectangle
                rect_surface = pygame.Surface((px_w, px_h), pygame.SRCALPHA)
                rect_surface.fill(area.color)
                self.screen.blit(rect_surface, (left, top))

                # Draw outline
                pygame.draw.rect(
                    self.screen,
                    (0, 100, 0),
                    pygame.Rect(left, top, px_w, px_h),
                    2,
                )

            # Draw label (Optional)
            if hasattr(area, "contains_object"):
                text = "DROP ZONE"
                text_surface = self._target_font.render(text, True, (0, 0, 0))
                text_rect = text_surface.get_rect(
                    center=(
                        area.x * self.scale,
                        self.screen_size[1] - area.y * self.scale,
                    )
                )
                self.screen.blit(text_surface, text_rect)

    def _draw_agent_indices(self):
        """Render the index of each agent on top of them for easy identification."""
        if self._index_font is None:
            pygame.font.init()
            self._index_font = pygame.font.SysFont("Arial", 12, bold=True)

        for agent in self.env.agents:
            agent.render_index(
                self.screen, self.screen_size, self.scale, self._index_font
            )

    def _draw_grab_joints(self):
        """Draw a line between each agent and the object it is grabbing."""
        agent_joints = getattr(self.env, "agent_joints", None)
        if not agent_joints:
            return

        LINE_COLOR = (255, 140, 0)
        ANCHOR_COLOR = (0, 0, 0)
        sh = self.screen_size[1]

        for entry in agent_joints:
            if entry is None:
                continue
            joint, _ = entry
            ax, ay = joint.anchorA
            bx, by = joint.anchorB
            start = (int(ax * self.scale), int(sh - ay * self.scale))
            end = (int(bx * self.scale), int(sh - by * self.scale))
            pygame.draw.line(self.screen, LINE_COLOR, start, end, 2)
            pygame.draw.circle(self.screen, ANCHOR_COLOR, start, 3)
            pygame.draw.circle(self.screen, ANCHOR_COLOR, end, 3)

    def _draw_walls(self):
        """Render obstacle-course walls and gap-coupling labels."""
        walls = getattr(self.env, "walls", None)
        if not walls:
            return

        if self._coupling_font is None:
            pygame.font.init()
            self._coupling_font = pygame.font.SysFont("Arial", 20, bold=True)

        sh = self.screen_size[1]
        gap_pad = getattr(self.env, "gap_region_pad", 1.0)

        for wall in walls:
            wy = wall["y"]
            thickness = wall["thickness"]
            # Wall segments (static bodies).
            for body in wall["segments"]:
                for fixture in body.fixtures:
                    shape = fixture.shape
                    if not isinstance(shape, b2PolygonShape):
                        continue
                    verts = [(body.transform * v) * self.scale for v in shape.vertices]
                    verts = [(v[0], sh - v[1]) for v in verts]
                    pygame.draw.polygon(self.screen, (60, 60, 60), verts)
                    pygame.draw.polygon(self.screen, (0, 0, 0), verts, 2)

            # Gap influence zones + coupling labels.
            half_t = thickness / 2 + gap_pad
            for gap in wall["gaps"]:
                left = int((gap["x_center"] - gap["width"] / 2) * self.scale)
                right = int((gap["x_center"] + gap["width"] / 2) * self.scale)
                top = int(sh - (wy + half_t) * self.scale)
                bottom = int(sh - (wy - half_t) * self.scale)
                w = right - left
                h = bottom - top
                surf = pygame.Surface((w, h), pygame.SRCALPHA)
                surf.fill((255, 165, 0, 70))
                self.screen.blit(surf, (left, top))
                pygame.draw.rect(
                    self.screen, (200, 100, 0), pygame.Rect(left, top, w, h), 1
                )

                text = str(gap["coupling"])
                text_surface = self._coupling_font.render(text, True, (255, 255, 255))
                cx = int(gap["x_center"] * self.scale)
                cy = int(sh - wy * self.scale)
                text_rect = text_surface.get_rect(center=(cx, cy))
                for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    outline = self._coupling_font.render(text, True, (0, 0, 0))
                    self.screen.blit(outline, text_rect.move(offset))
                self.screen.blit(text_surface, text_rect)

    def _draw_object_coupling(self):
        """Render the coupling requirement on top of each object."""
        # Not every env has dynamic objects — nothing to render when absent.
        objects = getattr(self.env, "objects", [])
        if not objects:
            return

        if self._coupling_font is None:
            pygame.font.init()
            self._coupling_font = pygame.font.SysFont("Arial", 20, bold=True)

        for body in objects:
            center_x = body.position.x * self.scale
            center_y = self.screen_size[1] - body.position.y * self.scale

            coupling = body.userData["coupling"]
            text = str(coupling)
            text_surface = self._coupling_font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(int(center_x), int(center_y)))

            for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                outline_rect = text_rect.move(offset)
                outline_surface = self._coupling_font.render(text, True, (0, 0, 0))
                self.screen.blit(outline_surface, outline_rect)

            self.screen.blit(text_surface, text_rect)
