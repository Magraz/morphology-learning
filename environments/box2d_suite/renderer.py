import math

import numpy as np
import pygame
from Box2D import b2CircleShape, b2PolygonShape

from environments.box2d_suite.utils import COLORS_LIST


class Renderer:
    """Encapsulates all pygame rendering for MultiBoxPushEnv."""

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

        # Draw agents
        self._render_agents_as_circles()

        # Draw agent indices on top of agents
        self._draw_agent_indices()

        # self._draw_density_sensors()  # Add this before or after drawing agents

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
            is_open = (
                self.env.attach_values[agent.index] == 1
                and self.env.detach_values[agent.index] == 0
            )
            agent.render_circle(self.screen, self.screen_size, self.scale, is_open)

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
        for body in self.env.objects:
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

    def _draw_density_sensors(self):
        """Draw density sensors for each agent as sector outlines with text values"""
        if self._sensor_font is None:
            pygame.font.init()
            self._sensor_font = pygame.font.SysFont("Arial", 12)

        n_sectors = 8  # Now 8 sectors
        sector_step = 360 / n_sectors
        shift_degrees = 22.5  # Shift sectors counter-clockwise

        for agent in self.env.agents:
            # Get agent position in screen coordinates
            center_x = agent.position.x * self.scale
            center_y = self.screen_size[1] - agent.position.y * self.scale

            # Get sensor values - now returns 16 density values (8 agent + 8 target) + rel coords
            sensors = self.env.observation_manager.calculate_density_sensors(
                agent.index, self.env.sector_sensor_radius
            )
            agent_densities = sensors[:n_sectors]
            target_densities = sensors[n_sectors : n_sectors * 2]

            sensor_radius = (
                self.env.sector_sensor_radius * self.scale
            )  # Radius of detection circle

            # Draw each sector outline
            for sector in range(n_sectors):
                start_angle = sector * sector_step + shift_degrees
                end_angle = (sector + 1) * sector_step + shift_degrees

                # Calculate arc points
                start_rad = math.radians(start_angle)
                end_rad = math.radians(end_angle)

                # Starting point on the arc
                start_x = center_x + sensor_radius * math.cos(start_rad)
                start_y = center_y - sensor_radius * math.sin(start_rad)

                # Ending point on the arc
                end_x = center_x + sensor_radius * math.cos(end_rad)
                end_y = center_y - sensor_radius * math.sin(end_rad)

                # Draw agent density lines (blue)
                pygame.draw.line(
                    self.screen,
                    (0, 0, 255),  # Blue color for agent density
                    (int(center_x), int(center_y)),
                    (int(start_x), int(start_y)),
                    1,
                )

                pygame.draw.line(
                    self.screen,
                    (0, 0, 255),  # Blue color for agent density
                    (int(center_x), int(center_y)),
                    (int(end_x), int(end_y)),
                    1,
                )

                # Draw the arc connecting the two points (for agent density)
                pygame.draw.arc(
                    self.screen,
                    (0, 0, 255),  # Blue color for agent density
                    pygame.Rect(
                        int(center_x - sensor_radius),
                        int(center_y - sensor_radius),
                        int(sensor_radius * 2),
                        int(sensor_radius * 2),
                    ),
                    start_rad,
                    end_rad,
                    1,
                )

                # Calculate text position at the middle of the sector
                mid_angle = math.radians((start_angle + end_angle) / 2)
                text_distance = (
                    sensor_radius * 0.3
                )  # Position text at 70% of the radius
                text_x = center_x + text_distance * math.cos(mid_angle)
                text_y = center_y - text_distance * math.sin(mid_angle)

                # Format the density values
                text_line1 = f"A:{agent_densities[sector]:.3f}"
                text_line2 = f"T:{target_densities[sector]:.3f}"

                # Render the text
                surface1 = self._sensor_font.render(text_line1, True, (0, 0, 0))
                surface2 = self._sensor_font.render(text_line2, True, (0, 0, 0))

                # Position lines centered vertically
                rect1 = surface1.get_rect(center=(int(text_x), int(text_y) - 6))
                rect2 = surface2.get_rect(center=(int(text_x), int(text_y) + 6))

                self.screen.blit(surface1, rect1)
                self.screen.blit(surface2, rect2)

    def _draw_target_areas(self):
        """Draw target areas with their coupling requirements"""
        if self._target_font is None:
            pygame.font.init()
            self._target_font = pygame.font.SysFont("Arial", 14)

        for area in self.env.target_areas:

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

    def _draw_object_coupling(self):
        """Render the coupling requirement on top of each object."""
        if self._coupling_font is None:
            pygame.font.init()
            self._coupling_font = pygame.font.SysFont("Arial", 14, bold=True)

        for body in self.env.objects:
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
