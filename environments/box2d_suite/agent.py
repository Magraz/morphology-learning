import numpy as np
import pygame
from Box2D import b2CircleShape, b2FixtureDef

from environments.box2d_suite.utils import (
    AGENT_CATEGORY,
    BOUNDARY_CATEGORY,
    OBJECT_CATEGORY,
)


class Agent:
    """Reusable Box2D-backed agent for multi-agent environments."""

    def __init__(
        self,
        world,
        position,
        index,
        radius=0.4,
        linear_damping=10.0,
        angular_damping=0.0,
    ):
        fixture_def = b2FixtureDef(
            shape=b2CircleShape(radius=radius),
            isSensor=False,
        )
        fixture_def.filter.categoryBits = AGENT_CATEGORY
        fixture_def.filter.maskBits = (
            AGENT_CATEGORY | BOUNDARY_CATEGORY | OBJECT_CATEGORY
        )

        self.body = world.CreateDynamicBody(
            position=position,
            fixtures=fixture_def,
            linearDamping=linear_damping,
            angularDamping=angular_damping,
        )
        self.body.userData = {"type": "agent", "index": index}
        self.index = index

    @property
    def position(self):
        return self.body.position

    @property
    def linear_velocity(self):
        return self.body.linearVelocity

    @property
    def radius(self):
        return self.body.fixtures[0].shape.radius

    @property
    def fixtures(self):
        return self.body.fixtures

    @property
    def transform(self):
        return self.body.transform

    def apply_force(self, force_x, force_y):
        self.body.ApplyForceToCenter((float(force_x), float(force_y)), True)

    def render_circle(self, screen, screen_size, scale, is_open):
        """Render agent as a colored circle indicating open/closed state."""
        OPEN_COLOR = (50, 200, 50)
        CLOSED_COLOR = (200, 50, 50)
        color = OPEN_COLOR if is_open else CLOSED_COLOR

        cx = int(self.position.x * scale)
        cy = int(screen_size[1] - self.position.y * scale)
        px_radius = int(self.radius * scale)

        pygame.draw.circle(screen, color, (cx, cy), px_radius)
        pygame.draw.circle(screen, (0, 0, 0), (cx, cy), px_radius, 2)

    def render_box(self, screen, screen_size, scale, color):
        """Render agent as a colored polygon from its Box2D shape."""
        for fixture in self.fixtures:
            shape = fixture.shape
            vertices = [(self.transform * v) * scale for v in shape.vertices]
            vertices = [(v[0], screen_size[1] - v[1]) for v in vertices]
            pygame.draw.polygon(screen, color, vertices)

    def render_index(self, screen, screen_size, scale, font):
        """Render the agent index label on top of the agent."""
        cx = int(self.position.x * scale + 5)
        cy = int(screen_size[1] - self.position.y * scale + 5)

        text = str(self.index)
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(cx, cy))

        for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            outline_surface = font.render(text, True, (0, 0, 0))
            screen.blit(outline_surface, text_rect.move(offset))

        screen.blit(text_surface, text_rect)

    def render_force(self, screen, screen_size, scale, force, force_scale, color):
        """Draw force vector arrow on the agent."""
        force_magnitude = np.linalg.norm(force)
        if force_magnitude <= 0.1:
            return

        cx = self.position.x * scale
        cy = screen_size[1] - self.position.y * scale

        scaled_force = force * force_scale
        start_pos = (int(cx), int(cy))
        end_pos = (int(cx + scaled_force[0]), int(cy - scaled_force[1]))
        line_width = max(1, int(force_magnitude * 0.5))

        pygame.draw.line(screen, color, start_pos, end_pos, line_width)
        _draw_arrowhead(screen, start_pos, end_pos, color)


def _draw_arrowhead(screen, start_pos, end_pos, color):
    """Draw an arrowhead at the end of a vector line."""
    if start_pos == end_pos:
        return

    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    length = np.sqrt(dx * dx + dy * dy)

    if length < 5:
        return

    dx /= length
    dy /= length

    arrow_length = min(10, length * 0.3)
    arrow_angle = 0.5

    cos_a = np.cos(arrow_angle)
    sin_a = np.sin(arrow_angle)

    left_x = end_pos[0] - arrow_length * (dx * cos_a - dy * sin_a)
    left_y = end_pos[1] - arrow_length * (dy * cos_a + dx * sin_a)
    right_x = end_pos[0] - arrow_length * (dx * cos_a + dy * sin_a)
    right_y = end_pos[1] - arrow_length * (dy * cos_a - dx * sin_a)

    arrow_points = [
        end_pos,
        (int(left_x), int(left_y)),
        (int(right_x), int(right_y)),
    ]
    pygame.draw.polygon(screen, color, arrow_points)
