import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from Box2D import (
    b2World,
    b2PolygonShape,
    b2FixtureDef,
    b2CircleShape,
)

from environments.box2d_salp.utils import (
    COLORS_LIST,
    AGENT_CATEGORY,
    BOUNDARY_CATEGORY,
    OBJECT_CATEGORY,
    TargetArea,
    BoundaryContactListener,
    get_scatter_positions,
    fixed_position_target_area,
)


class MultiBoxPushEnv(gym.Env):
    metadata = {"render_fps": 30}

    def __init__(self, render_mode=None, n_agents=2, n_target_areas=1, max_steps=512):
        super().__init__()

        self.n_agents = n_agents
        self.render_mode = render_mode

        # Add target areas parameters
        self.n_target_areas = n_target_areas
        self.target_areas = []

        # Update action space to include detach action
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, 4),  # (n_agents, action_dim)
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, 18), dtype=np.float32
        )

        self.world = b2World(gravity=(0, 0))
        self.time_step = 1.0 / 60.0

        self.agents = []

        self.prev_agent_closest_distances = [float("inf") for _ in range(self.n_agents)]
        self.prev_target_closest_distances = [
            float("inf") for _ in range(self.n_agents)
        ]

        # Add contact listener
        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        # Boundary parameters (customize as needed)
        self.world_width = 40
        self.world_height = 40
        self.world_center_x = self.world_width // 2
        self.world_center_y = self.world_height // 2
        self.world_diagonal = np.sqrt(self.world_height**2 + self.world_width**2)
        self.boundary_thickness = 0.5

        # Pygame rendering setup
        self.screen = None
        self.clock = None
        self.screen_size = (800, 800)
        self.scale = 20.0  # Pixels per Box2D meter

        # Create target areas
        self._create_target_areas()

        # Create boundary and agents
        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )

        self._init_agents()

        # Add force tracking
        self.applied_forces = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.force_scale = 2.0  # Scale factor for visualizing forces

        # Add sector sensing threshold
        self.sector_sensor_radius = 10

        # Add parameters for nearest neighbor detection
        self.neighbor_detection_range = 3.0  # Maximum range to detect neighbors

        # Add a field to track link openness for each agent
        self.attach_values = np.zeros(
            self.n_agents, dtype=np.int8
        )  # Default to no attachment (0)

        # Add a field to track detach values for each agent
        self.detach_values = np.zeros(
            self.n_agents, dtype=np.int8
        )  # Default to 0 (no desire to detach)

        # Step tracking for truncation
        self.max_steps = max_steps
        self.current_step = 0

        self.objects = []  # Track dynamic objects

    def _create_target_areas(self):
        """Create target areas at random positions in the environment"""
        self.target_areas = []
        existing_positions = []

        for idx in range(self.n_target_areas):

            # Get position using the new method
            # x, y = dynamic_position_target_area(
            #     self.world_width, self.world_height, existing_positions
            # )
            x, y = fixed_position_target_area(idx, self.world_width, self.world_height)
            existing_positions.append((x, y))

            # Random radius between 2 and 4
            radius = 5

            # Random coupling requirement between 2 and min(5, n_agents)
            # coupling_req = np.random.randint(2, self.n_agents - 1)
            coupling_req = 2

            # Set reward scale based on coupling requirement
            reward_scale = 1.0  # * coupling_req

            # Random color (semi-transparent)
            color = (
                np.random.randint(50, 200),  # R
                np.random.randint(50, 200),  # G
                np.random.randint(50, 200),  # B
                128,  # Alpha (semi-transparent)
            )

            target_area = TargetArea(x, y, radius, coupling_req, reward_scale, color)
            self.target_areas.append(target_area)

    def _init_agents(self):
        self.agents.clear()

        positions = get_scatter_positions(
            self.world_width, self.world_height, self.n_agents
        )
        # positions = get_linear_positions(
        #     self.world_width, self.world_height, self.n_agents
        # )
        self._create_agents(positions)

    def _create_dynamic_objects(self):
        """Create the requested square, triangle, and circle objects"""
        self.objects.clear()

        # Define shapes
        shapes = [
            # Square (Box)
            b2PolygonShape(box=(1.5, 1.5)),
            # Triangle (Polygon vertices must be CCW)
            b2PolygonShape(vertices=[(0, 2.0), (-2.0, -1.5), (2.0, -1.5)]),
            # Circle
            b2CircleShape(radius=1.5),
        ]

        # Colors (R, G, B)
        colors = [
            (100, 100, 255),  # Blue-ish for Square
            (255, 100, 255),  # Magenta-ish for Triangle
            (255, 165, 0),  # Orange for Circle
        ]

        center_x = self.world_width / 2
        center_y = self.world_height / 2

        for i, shape in enumerate(shapes):
            fixture_def = b2FixtureDef(
                shape=shape,
                density=3.0,
                friction=0.3,
                restitution=0.2,
            )

            # Setup collisions
            fixture_def.filter.categoryBits = OBJECT_CATEGORY
            fixture_def.filter.maskBits = (
                AGENT_CATEGORY | BOUNDARY_CATEGORY | OBJECT_CATEGORY
            )

            # Position them slightly offset from center
            offset_x = (i - 1) * 6.0  # Spread out along X
            pos = (center_x + offset_x, center_y)

            body = self.world.CreateDynamicBody(
                position=pos,
                fixtures=fixture_def,
                linearDamping=5.0,
                angularDamping=2.0,
            )

            # Store color in body user data for rendering
            body.userData = {"color": colors[i]}

            self.objects.append(body)

    def _create_agents(self, positions):
        for i in range(self.n_agents):

            fixture_def = b2FixtureDef(
                # shape=b2PolygonShape(box=(0.3, 0.5)),
                shape=b2CircleShape(radius=0.4),
                isSensor=False,
            )

            fixture_def.filter.categoryBits = AGENT_CATEGORY

            fixture_def.filter.maskBits = (
                AGENT_CATEGORY
                | BOUNDARY_CATEGORY  # allow collisions between agents and boundaries
                | OBJECT_CATEGORY
            )

            body = self.world.CreateDynamicBody(
                position=positions[i],
                fixtures=fixture_def,
                linearDamping=10.0,  # High damping prevents drifting
                angularDamping=0.0,  # Prevents excessive spinning
            )

            self.agents.append(body)

    def _render_agents_as_circles(self):
        # Define colors for open and closed agents
        OPEN_COLOR = (50, 200, 50)  # Green for agents open to links
        CLOSED_COLOR = (200, 50, 50)  # Red for agents closed to links

        for idx, body in enumerate(self.agents):
            # Get circle position and radius
            center_x = body.position.x * self.scale
            center_y = self.screen_size[1] - body.position.y * self.scale
            radius = body.fixtures[0].shape.radius * self.scale  # Get radius from shape

            # Choose color based on attach
            if self.attach_values[idx] == 1 and self.detach_values[idx] == 0:
                color = OPEN_COLOR  # Open to links
            else:
                color = CLOSED_COLOR  # Closed to links

            # Draw filled circle
            pygame.draw.circle(
                self.screen,
                color,
                (int(center_x), int(center_y)),
                int(radius),
            )

            # Draw circle outline for better visibility
            pygame.draw.circle(
                self.screen,
                (0, 0, 0),
                (int(center_x), int(center_y)),
                int(radius),
                2,  # Outline thickness
            )

    def _render_agents_as_boxes(self):
        for idx, body in enumerate(self.agents):
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * self.scale for v in shape.vertices]
                vertices = [(v[0], self.screen_size[1] - v[1]) for v in vertices]

                pygame.draw.polygon(self.screen, COLORS_LIST[idx], vertices)

    def _create_boundary(self, width, height, thickness):
        """Create boundary walls that agents can collide with"""

        # Bottom wall
        bottom_wall = self.world.CreateStaticBody(
            position=(width / 2, thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        bottom_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        bottom_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY

        # Top wall
        top_wall = self.world.CreateStaticBody(
            position=(width / 2, height - thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        top_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        top_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY

        # Left wall
        left_wall = self.world.CreateStaticBody(
            position=(thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        left_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        left_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY

        # Right wall
        right_wall = self.world.CreateStaticBody(
            position=(width - thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        right_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        right_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY

    def _draw_boundary_walls(self):
        """Draw the actual boundary walls at their Box2D positions"""
        thickness = self.boundary_thickness

        # Bottom wall
        bottom_rect = pygame.Rect(
            0,  # Left edge
            self.screen_size[1] - thickness * self.scale,  # Bottom of screen
            self.world_width * self.scale,  # Full width
            thickness * self.scale,  # Wall thickness
        )
        pygame.draw.rect(self.screen, (0, 0, 0), bottom_rect)

        # Top wall
        top_rect = pygame.Rect(
            0,  # Left edge
            self.screen_size[1] - self.world_height * self.scale,  # Top position
            self.world_width * self.scale,  # Full width
            thickness * self.scale,  # Wall thickness
        )
        pygame.draw.rect(self.screen, (0, 0, 0), top_rect)

        # Left wall
        left_rect = pygame.Rect(
            0,  # Left edge of screen
            self.screen_size[1] - self.world_height * self.scale,  # Top position
            thickness * self.scale,  # Wall thickness
            self.world_height * self.scale,  # Full height
        )
        pygame.draw.rect(self.screen, (0, 0, 0), left_rect)

        # Right wall
        right_rect = pygame.Rect(
            self.world_width * self.scale - thickness * self.scale,  # Right position
            self.screen_size[1] - self.world_height * self.scale,  # Top position
            thickness * self.scale,  # Wall thickness
            self.world_height * self.scale,  # Full height
        )
        pygame.draw.rect(self.screen, (0, 0, 0), right_rect)

    def _draw_force_vectors(self):
        """Draw force vectors for each agent with enhanced 2D visualization"""
        for idx, (body, force) in enumerate(zip(self.agents, self.applied_forces)):
            # Get agent center position in screen coordinates
            center_x = body.position.x * self.scale
            center_y = self.screen_size[1] - body.position.y * self.scale

            # Calculate force vector magnitude
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 0.1:  # Only draw if force is significant
                # Scale the force vector for visibility
                scaled_force = force * self.force_scale

                end_x = center_x + scaled_force[0]
                end_y = center_y - scaled_force[1]  # Flip Y for screen coordinates

                # Draw force vector as arrow
                start_pos = (int(center_x), int(center_y))
                end_pos = (int(end_x), int(end_y))

                # Use thicker line for stronger forces
                line_width = max(1, int(force_magnitude * 0.5))

                # Draw main force line (thicker, colored by agent)
                pygame.draw.line(
                    self.screen,
                    COLORS_LIST[idx % len(COLORS_LIST)],
                    start_pos,
                    end_pos,
                    line_width,
                )

                # Draw arrowhead
                self._draw_arrowhead(
                    start_pos, end_pos, COLORS_LIST[idx % len(COLORS_LIST)]
                )

    def _draw_arrowhead(self, start_pos, end_pos, color):
        """Draw an arrowhead at the end of a force vector"""
        if start_pos == end_pos:
            return

        # Calculate arrow direction
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = np.sqrt(dx * dx + dy * dy)

        if length < 5:  # Don't draw tiny arrows
            return

        # Normalize direction
        dx /= length
        dy /= length

        # Arrow parameters
        arrow_length = min(10, length * 0.3)
        arrow_angle = 0.5  # radians

        # Calculate arrowhead points
        cos_a = np.cos(arrow_angle)
        sin_a = np.sin(arrow_angle)

        # Left arrowhead point
        left_x = end_pos[0] - arrow_length * (dx * cos_a - dy * sin_a)
        left_y = end_pos[1] - arrow_length * (dy * cos_a + dx * sin_a)

        # Right arrowhead point
        right_x = end_pos[0] - arrow_length * (dx * cos_a + dy * sin_a)
        right_y = end_pos[1] - arrow_length * (dy * cos_a - dx * sin_a)

        # Draw arrowhead
        arrow_points = [
            end_pos,
            (int(left_x), int(left_y)),
            (int(right_x), int(right_y)),
        ]
        pygame.draw.polygon(self.screen, color, arrow_points)

    def _draw_dynamic_objects(self):
        """Draw the extra objects (Square, Triangle, Circle)"""
        for body in self.objects:
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

    def _draw_density_sensors(self, normalization_value=10):
        """Draw density sensors for each agent as sector outlines with text values"""
        # Initialize font if not already done
        if not hasattr(self, "sensor_font"):
            pygame.font.init()
            self.sensor_font = pygame.font.SysFont("Arial", 12)

        for idx, agent in enumerate(self.agents):
            # Get agent position in screen coordinates
            center_x = agent.position.x * self.scale
            center_y = self.screen_size[1] - agent.position.y * self.scale

            # Get sensor values - now returns 8 values (4 agent densities + 4 target densities)
            sensors = self._calculate_density_sensors(idx, self.sector_sensor_radius)
            agent_densities = sensors[:4]
            target_densities = sensors[4:]

            sensor_radius = (
                self.sector_sensor_radius * self.scale
            )  # Radius of detection circle

            # Define sector angles
            sector_angles = [
                (0, 90),  # top-right
                (90, 180),  # top-left
                (180, 270),  # bottom-left
                (270, 360),  # bottom-right
            ]

            # Draw each sector outline
            for sector, (start_angle, end_angle) in enumerate(sector_angles):
                # Agent density (blue lines)
                agent_density = (
                    min(agent_densities[sector], normalization_value)
                    / normalization_value
                )  # Normalize to 0-1
                agent_thickness = max(1, int(1 + 3 * agent_density))  # 1-4 pixels thick

                # Target density (green lines)
                target_density = (
                    min(target_densities[sector], normalization_value)
                    / normalization_value
                )  # Normalize to 0-1
                target_thickness = max(
                    1, int(1 + 3 * target_density)
                )  # 1-4 pixels thick

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
                    agent_thickness,
                )

                pygame.draw.line(
                    self.screen,
                    (0, 0, 255),  # Blue color for agent density
                    (int(center_x), int(center_y)),
                    (int(end_x), int(end_y)),
                    agent_thickness,
                )

                # Draw target density lines (green, slightly offset)
                if target_thickness > 1:  # Only draw if significant target density
                    offset = 3  # Small offset for visibility
                    pygame.draw.line(
                        self.screen,
                        (0, 200, 0),  # Green color for target density
                        (int(center_x), int(center_y)),
                        (int(start_x) + offset, int(start_y) + offset),
                        target_thickness,
                    )

                    pygame.draw.line(
                        self.screen,
                        (0, 200, 0),  # Green color for target density
                        (int(center_x), int(center_y)),
                        (int(end_x) + offset, int(end_y) + offset),
                        target_thickness,
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
                    agent_thickness,
                )

                # Calculate text position at the middle of the sector
                mid_angle = math.radians((start_angle + end_angle) / 2)
                text_distance = (
                    sensor_radius * 0.2
                )  # Position text at 70% of the radius
                text_x = center_x + text_distance * math.cos(mid_angle)
                text_y = center_y - text_distance * math.sin(mid_angle)

                # Format the density values
                text_line1 = f"A:{agent_densities[sector]:.3f}"
                text_line2 = f"T:{target_densities[sector]:.3f}"

                # Render the text
                surface1 = self.sensor_font.render(text_line1, True, (0, 0, 0))
                surface2 = self.sensor_font.render(text_line2, True, (0, 0, 0))

                # Position lines centered vertically
                rect1 = surface1.get_rect(center=(int(text_x), int(text_y) - 6))
                rect2 = surface2.get_rect(center=(int(text_x), int(text_y) + 6))

                self.screen.blit(surface1, rect1)
                self.screen.blit(surface2, rect2)

    def _draw_target_areas(self):
        """Draw target areas with their coupling requirements"""
        if not hasattr(self, "target_font"):
            pygame.font.init()
            self.target_font = pygame.font.SysFont("Arial", 14)

        for area in self.target_areas:
            # Draw the area as a semi-transparent circle
            area_rect = pygame.Rect(
                (area.x - area.radius) * self.scale,
                self.screen_size[1] - (area.y + area.radius) * self.scale,
                area.radius * 2 * self.scale,
                area.radius * 2 * self.scale,
            )

            # Create a surface for the semi-transparent circle
            circle_surface = pygame.Surface(
                (int(area.radius * 2 * self.scale), int(area.radius * 2 * self.scale)),
                pygame.SRCALPHA,
            )
            pygame.draw.circle(
                circle_surface,
                area.color,
                (int(area.radius * self.scale), int(area.radius * self.scale)),
                int(area.radius * self.scale),
            )

            # Draw the circle
            self.screen.blit(circle_surface, area_rect)

            # Draw the coupling requirement text
            req_text = f"Req: {area.coupling_requirement}"
            text_surface = self.target_font.render(req_text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(
                center=(area.x * self.scale, self.screen_size[1] - area.y * self.scale)
            )
            self.screen.blit(text_surface, text_rect)

            # Draw a small indicator of reward scale
            scale_text = f"x{area.reward_scale:.1f}"
            scale_surface = self.target_font.render(scale_text, True, (0, 0, 0))
            scale_rect = scale_surface.get_rect(
                center=(
                    area.x * self.scale,
                    self.screen_size[1] - area.y * self.scale + 20,
                )
            )
            self.screen.blit(scale_surface, scale_rect)

    def _draw_agent_indices(self):
        """Render the index of each agent on top of them for easy identification"""
        # Initialize font if not already done
        if not hasattr(self, "index_font"):
            pygame.font.init()
            self.index_font = pygame.font.SysFont("Arial", 12, bold=True)

        for idx, agent in enumerate(self.agents):
            # Get agent position in screen coordinates
            center_x = agent.position.x * self.scale
            center_y = self.screen_size[1] - agent.position.y * self.scale

            # Render the agent index
            index_text = str(idx)
            text_surface = self.index_font.render(
                index_text, True, (255, 255, 255)
            )  # White text

            # Center the text on the agent
            text_rect = text_surface.get_rect(
                center=(int(center_x + 5), int(center_y + 5))
            )

            # Add a black outline for better visibility
            for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                outline_rect = text_rect.move(offset)
                outline_surface = self.index_font.render(index_text, True, (0, 0, 0))
                self.screen.blit(outline_surface, outline_rect)

            # Draw the actual text
            self.screen.blit(text_surface, text_rect)

    def _get_nearest_non_connected_agent_relative(
        self, agent_idx, all_states, neighbor_detection_range
    ):
        """
        Find the nearest non-connected agent and return relative state information

        Returns:
            numpy array: [relative_x, relative_y, relative_vx, relative_vy, distance]
        """
        agent_position = all_states[agent_idx][:2]
        agent_velocity = all_states[agent_idx][2:4]

        # Find which agents are in the same connected component
        current_component_root = self.union_find.find(agent_idx)

        min_distance = float("inf")
        nearest_relative_state = None

        for other_idx in range(self.n_agents):
            if other_idx == agent_idx:
                continue

            # Check if this agent is in the same connected component
            other_component_root = self.union_find.find(other_idx)
            if current_component_root == other_component_root:
                continue  # Skip agents in the same chain

            # Calculate distance and relative information
            other_position = all_states[other_idx][:2]
            other_velocity = all_states[other_idx][2:4]

            relative_position = other_position - agent_position
            relative_velocity = other_velocity - agent_velocity
            distance = np.linalg.norm(relative_position)

            # Check if within range and closer than previous candidates
            if distance <= neighbor_detection_range and distance < min_distance:
                min_distance = distance
                nearest_relative_state = np.concatenate(
                    [relative_position, relative_velocity, [distance]]
                )

        # Return relative state or zeros if no neighbor found
        if nearest_relative_state is not None:
            return nearest_relative_state
        else:
            return np.zeros(
                5, dtype=np.float32
            )  # [rel_x, rel_y, rel_vx, rel_vy, distance]

    def _get_observation(self):
        # Get all agent states as a matrix
        all_states = np.array(
            [
                [
                    (a.position.x - self.world_center_x),
                    (a.position.y - self.world_center_y),
                    # a.linearVelocity.x,
                    # a.linearVelocity.y,
                ]
                for a in self.agents
            ],
            dtype=np.float32,
        )

        # For each agent, get connected agents' states
        observations = []
        for i in range(self.n_agents):
            # Own state (absolute)
            own_state = all_states[i]

            # Calculate density sensors
            density_sensors = self._calculate_density_sensors(
                i, self.sector_sensor_radius
            )

            # Combine all observations: own absolute state + connected relative states + density sensors
            agent_obs = np.concatenate([own_state, density_sensors])

            observations.append(agent_obs)

        return np.array(observations, dtype=np.float32)

    def _get_chain_size_reward(self):
        """Calculate reward based on the largest connected component of agents"""
        largest_component_size = self._find_largest_connected_component()

        # Normalize by total number of agents to get a value between 0 and 1
        normalized_reward = largest_component_size / self.n_agents

        # Scale the reward (adjust multiplier as needed)
        reward = normalized_reward * 1.0  # Scale to make reward more significant

        terminated = self.n_agents == largest_component_size

        return reward, terminated

    def _calculate_proximity_reward(self):
        agent_proximity_reward = [0.0 for _ in self.agents]

        target_proximity_reward = [0.0 for _ in self.agents]

        # Check all pairs of agents
        for i in range(self.n_agents):
            agent_i_pos = np.array(
                [self.agents[i].position.x, self.agents[i].position.y]
            )

            closest_agent_dist = float("inf")
            closest_target_dist = float("inf")

            # Calculate agent proximity reward
            for j in range(self.n_agents):

                # Skip self
                if i == j:
                    continue
                # Skip if agents are already connected
                if self.union_find.connected(i, j):
                    continue

                agent_j_pos = np.array(
                    [self.agents[j].position.x, self.agents[j].position.y]
                )

                # Calculate distance
                distance = np.linalg.norm(agent_i_pos - agent_j_pos)

                # Update closest agent
                if self.prev_agent_closest_distances[i] == float("inf"):
                    self.prev_agent_closest_distances[i] = distance

                if distance < closest_agent_dist:
                    closest_agent_dist = distance
                    # Calculate reward based on distance
                    agent_proximity_reward[i] = (
                        self.prev_agent_closest_distances[i] - closest_agent_dist
                    )

                    self.prev_agent_closest_distances[i] = closest_agent_dist

            # Calculate target proximity reward
            for t_area in self.target_areas:

                target_pos = np.array([t_area.x, t_area.y])

                # Calculate distance
                distance = np.linalg.norm(agent_i_pos - target_pos)

                # Update closest agent
                if self.prev_target_closest_distances[i] == float("inf"):
                    self.prev_target_closest_distances[i] = distance

                if distance < closest_target_dist:
                    closest_target_dist = distance
                    # Calculate reward based on distance
                    target_proximity_reward[i] = (
                        self.prev_target_closest_distances[i] - closest_target_dist
                    )
                    self.prev_target_closest_distances[i] = closest_target_dist

        return np.array(target_proximity_reward) + np.array(agent_proximity_reward)

    def _get_rewards(self):
        """Calculate combined rewards from chain size and target areas"""
        # Calculate target area rewards
        task_reward = 0.0

        individual_rewards = [0 for _ in range(self.n_agents)]

        # Go over all targets and get the total task reward
        # for target_area in self.target_areas:

        #     # Calculate reward for this target area
        #     reward_map = target_area.calculate_reward(self.agents, self.union_find)

        #     task_reward += np.array(reward_map).mean()

        # Calculate individual rewards
        # individual_rewards = self._calculate_proximity_reward()

        return task_reward, individual_rewards

    def _calculate_density_sensors(self, agent_idx, sensor_radius):
        """
        Calculate density of agents and targets in four sectors around an agent.
        Also returns relative coordinates to closest non-connected agent and target.

        Returns a vector of 12 values:
        - First 4 values: agent density in [top-right, top-left, bottom-left, bottom-right]
        - Next 4 values: target density in [top-right, top-left, bottom-left, bottom-right]
        - Next 2 values: relative [x,y] to closest non-connected agent
        - Last 2 values: relative [x,y] to closest target
        """
        agent_pos = np.array(
            [self.agents[agent_idx].position.x, self.agents[agent_idx].position.y]
        )

        # Initialize densities for the 4 sectors (for both agents and targets)
        agent_densities = np.zeros(4, dtype=np.float32)
        target_densities = np.zeros(4, dtype=np.float32)

        # Variables to track closest agent and target
        closest_agent_dist = float("inf")
        closest_agent_rel = np.zeros(2, dtype=np.float32)
        closest_target_dist = float("inf")
        closest_target_rel = np.zeros(2, dtype=np.float32)

        # Check each other agent
        for other_idx, other_agent in enumerate(self.agents):

            if other_idx == agent_idx:
                continue  # Skip self

            other_pos = np.array([other_agent.position.x, other_agent.position.y])
            relative_pos = other_pos - agent_pos

            # Calculate distance
            distance = np.linalg.norm(relative_pos)

            # Update closest agent tracking
            if distance < closest_agent_dist:
                closest_agent_dist = distance
                closest_agent_rel = relative_pos

            # Skip if outside sensor radius for density calculation
            if distance > sensor_radius:
                continue

            # Determine sector (0: top-right, 1: top-left, 2: bottom-left, 3: bottom-right)
            sector = 0
            if relative_pos[0] < 0:  # Left side
                if relative_pos[1] >= 0:  # Top-left
                    sector = 1
                else:  # Bottom-left
                    sector = 2
            else:  # Right side
                if relative_pos[1] < 0:  # Bottom-right
                    sector = 3
                # else it's already sector 0 (top-right)

            # Calculate density contribution (inverse square of distance)
            density_value = 1.0 / ((distance / self.sector_sensor_radius) + 1.0)

            # Add to appropriate sector
            agent_densities[sector] += density_value

        # Check each target area
        for target in self.target_areas:
            target_pos = np.array([target.x, target.y])
            relative_pos = target_pos - agent_pos

            # Calculate distance
            distance = np.linalg.norm(relative_pos)

            # Update closest target tracking
            if distance < closest_target_dist:
                closest_target_dist = distance
                closest_target_rel = relative_pos

            # Skip if outside sensor radius for density calculation
            if distance > sensor_radius:
                continue

            # Determine sector (0: top-right, 1: top-left, 2: bottom-left, 3: bottom-right)
            sector = 0
            if relative_pos[0] < 0:  # Left side
                if relative_pos[1] >= 0:  # Top-left
                    sector = 1
                else:  # Bottom-left
                    sector = 2
            else:  # Right side
                if relative_pos[1] < 0:  # Bottom-right
                    sector = 3
                # else it's already sector 0 (top-right)

            # Calculate target density contribution
            density_value = target.reward_scale * (
                1.0 / ((distance / self.sector_sensor_radius) + 1.0)
            )

            # Set as sector value if its the highest value
            if target_densities[sector] < density_value:
                target_densities[sector] = density_value

        # Normalize the relative coordinates by world dimensions
        # closest_agent_rel = closest_agent_rel / np.array(
        #     [self.world_width, self.world_height]
        # )
        # closest_target_rel = closest_target_rel / np.array(
        #     [self.world_width, self.world_height]
        # )

        # Combine all values into one array
        return np.concatenate(
            [
                agent_densities,  # 4 values
                target_densities,  # 4 values
                closest_agent_rel,  # 2 values (x,y)
                closest_target_rel,  # 2 values (x,y)
            ]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset previous rewards
        self.prev_rewards = dict.fromkeys(list(range(0, self.n_agents)), 0)

        # Reset step counter
        self.current_step = 0

        for body in self.agents:
            self.world.DestroyBody(body)

        # Clean up existing objects
        for body in self.objects:
            self.world.DestroyBody(body)
        self.objects.clear()

        self._init_agents()

        self._create_dynamic_objects()  # Create new objects

        self._create_target_areas()

        # Reset contact listener
        self.contact_listener.reset()

        obs = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return obs, {}

    def step(self, actions):

        # PREPROCESS ENVIRONMENT ACTION

        # Transform actions into dictionary
        movement_action = actions[:, :2]
        attach_action = actions[:, 2]
        detach_action = actions[:, -1]

        # Update attach and detach states - ensure we get scalar values
        # Convert to flat numpy array if needed
        self.attach_values = np.array(attach_action).flatten()
        self.detach_values = np.array(detach_action).flatten()

        # PROCESS ENVIRONMENT ACTION

        # Apply movement forces
        force_multiplier = 100.0
        for idx, agent in enumerate(self.agents):
            force_x = (
                np.clip(movement_action[idx][0], -1, 1) * force_multiplier
            )  # X component
            force_y = (
                np.clip(movement_action[idx][1], -1, 1) * force_multiplier
            )  # Y component

            # Store the 2D force vector for visualization
            self.applied_forces[idx] = [force_x, force_y]

            # Apply 2D force to agent
            agent.ApplyForceToCenter((float(force_x), float(force_y)), True)

        # Rest of the step method remains the same
        self.world.Step(self.time_step, 6, 2)

        # CALCULATE REWARDS

        # Check for boundary collisions
        task_reward = 0.0
        individual_rewards = np.array([0.0 for _ in range(self.n_agents)])

        terminated = False

        if self.contact_listener.boundary_collision:
            terminated = True
        else:
            # The normal reward calculation
            task_reward, individual_rewards = self._get_rewards()

        # Reset collision flag for next step
        self.contact_listener.reset()

        # Create info dictionary with target positions
        info = {
            "target_positions": [
                {
                    "x": target.x,
                    "y": target.y,
                    "radius": target.radius,
                    "requirement": target.coupling_requirement,
                }
                for target in self.target_areas
            ],
            "agent_positions": [
                {"x": agent.position.x, "y": agent.position.y} for agent in self.agents
            ],
            "local_rewards": individual_rewards,
            "task_reward": task_reward,
        }

        self.current_step += 1

        truncated = self.current_step >= self.max_steps

        # The observation
        return self._get_observation(), task_reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
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

        # Draw agents
        self._render_agents_as_circles()

        # Draw agent indices on top of agents
        self._draw_agent_indices()

        self._draw_density_sensors()  # Add this before or after drawing agents

        # Draw force vectors
        # self._draw_force_vectors()

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    # Create the environment with rendering
    env = MultiBoxPushEnv(
        render_mode="human", n_agents=4, n_target_areas=2, max_steps=10e5
    )
    obs, info = env.reset()

    running = True
    current_agent_idx = 0
    group_control = False  # Toggle for group replication

    print("\n" + "=" * 50)
    print(" SALP CHAIN ENVIRONMENT DEBUGGER")
    print("=" * 50)
    print(f" Controlling Agent: {current_agent_idx}")
    print(" Controls:")
    print("  [ARROWS] : Move Active Agent")
    print("  [SPACE]  : Switch Active Agent")
    print("  [G]      : Toggle Group Control (Radius 5)")
    print("  [ESC]    : Quit")
    print("=" * 50 + "\n")

    while running:
        # 1. Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    current_agent_idx = (current_agent_idx + 1) % env.n_agents
                    print(f">>> Switched control to Agent {current_agent_idx}")
                elif event.key == pygame.K_g:
                    group_control = not group_control
                    status = "ON" if group_control else "OFF"
                    print(f">>> Group Control {status} (Radius 5)")

        # Get continuous key presses
        keys = pygame.key.get_pressed()

        # Initialize actions (n_agents, 4)
        # [force_x, force_y, attach_signal, detach_signal]
        # Default: No movement, Attach=1 (allow), Detach=0 (don't)
        actions = np.zeros((env.n_agents, 4), dtype=np.float32)
        actions[:, 2] = 1.0
        actions[:, 3] = 0.0

        # Set force for controlled agent
        force_x = 0.0
        force_y = 0.0

        if keys[pygame.K_LEFT]:
            force_x = -1.0
        if keys[pygame.K_RIGHT]:
            force_x = 1.0
        if keys[pygame.K_UP]:
            force_y = 1.0
        if keys[pygame.K_DOWN]:
            force_y = -1.0

        # Apply control to active agent
        actions[current_agent_idx, 0] = force_x
        actions[current_agent_idx, 1] = force_y

        # If group control is active, replicate actions to neighbors
        if group_control and (force_x != 0 or force_y != 0):
            current_pos = env.agents[current_agent_idx].position
            radius = 5.0

            for i in range(env.n_agents):
                if i == current_agent_idx:
                    continue

                other_pos = env.agents[i].position
                distance = math.sqrt(
                    (current_pos.x - other_pos.x) ** 2
                    + (current_pos.y - other_pos.y) ** 2
                )

                if distance <= radius:
                    actions[i, 0] = force_x
                    actions[i, 1] = force_y

        # 2. Step environment
        obs, reward, terminated, truncated, info = env.step(actions)

        # 3. Log info nicely
        # Format observation array to be compact
        obs_str = np.array2string(
            obs[current_agent_idx],
            precision=2,
            suppress_small=True,
            separator=",",
            floatmode="fixed",
            max_line_width=np.inf,
        )

        print(
            f"{env.current_step:<5d} | "
            f"{current_agent_idx:<3d} | "
            f"({force_x:>4.1f}, {force_y:>4.1f})  | "
            f"{reward:<8.4f} | "
            f"{obs_str}"
        )

        # 4. Render
        env.render()

        # 5. Handle reset
        if terminated or truncated:
            print(">>> Environment Reset")
            obs, info = env.reset()

    env.close()
