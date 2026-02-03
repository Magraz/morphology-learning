import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from Box2D import (
    b2World,
    b2PolygonShape,
    b2FixtureDef,
    b2RevoluteJointDef,
    b2CircleShape,
)

from learning.environments.box2d_salp.utils import (
    COLORS_LIST,
    AGENT_CATEGORY,
    BOUNDARY_CATEGORY,
    UnionFind,
    BoundaryContactListener,
    get_scatter_positions,
    get_linear_positions,
    dynamic_position_target_area,
    fixed_position_target_area,
    add_dictionary_values,
)


class TargetArea:
    def __init__(
        self,
        x,
        y,
        radius,
        coupling_requirement,
        reward_scale=1.0,
        color=(100, 200, 100, 128),
    ):
        self.x = x  # Center x-coordinate
        self.y = y  # Center y-coordinate
        self.radius = radius  # Area of influence
        self.coupling_requirement = coupling_requirement
        self.reward_scale = reward_scale  # Reward scaling factor
        self.color = color  # Semi-transparent green by default

    def calculate_reward(self, agents, union_find):
        """Calculate reward based on proximity to center if coupling requirement is met"""

        reward_map = dict.fromkeys(list(range(0, len(agents))), 0)

        # Find agents within this target area
        agents_in_area = []
        for i, agent in enumerate(agents):
            dist = np.sqrt(
                (agent.position.x - self.x) ** 2 + (agent.position.y - self.y) ** 2
            )
            if dist <= self.radius:
                agents_in_area.append(i)

        if len(agents_in_area) < self.coupling_requirement:
            return reward_map  # Not enough agents to meet requirement

        # Group agents by their connected component using union_find
        component_map = {}
        for idx in agents_in_area:
            root = union_find.find(idx)
            if root not in component_map:
                component_map[root] = []
            component_map[root].append(idx)

        # Check if any chain meets the coupling requirement
        qualifying_agents = []
        for component in component_map.values():
            if len(component) >= self.coupling_requirement:
                qualifying_agents.extend(component)

        if not qualifying_agents:
            return reward_map  # No component meets requirement

        # Calculate reward based on proximity
        for i in qualifying_agents:
            dist = np.sqrt(
                (agents[i].position.x - self.x) ** 2
                + (agents[i].position.y - self.y) ** 2
            )
            # Reward decreases with distance (1.0 at center, 0.0 at radius)
            reward_map[i] = 1 / (dist**2 + 0.1)

        return reward_map


class SalpChainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, n_agents=6, n_target_areas=1):
        super().__init__()

        self.n_agents = n_agents
        self.render_mode = render_mode

        # Add target areas parameters
        self.n_target_areas = n_target_areas
        self.target_areas = []

        # Add joint limit parameter
        self.max_joints_per_agent = 2

        # Update action space to include detach action
        # self.action_space = spaces.Dict(
        #     {
        #         "movement": spaces.Box(
        #             low=-1, high=1, shape=(self.n_agents, 2), dtype=np.float32
        #         ),
        #         "link_openness": spaces.MultiDiscrete(
        #             [2] * self.n_agents
        #         ),  # Each agent has a 0/1 choice
        #         "detach": spaces.Box(
        #             low=0, high=1, shape=(self.n_agents,), dtype=np.float32
        #         ),  # Detach desire from 0 to 1
        #     }
        # )

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_agents, 2), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, 4), dtype=np.float32
        )

        self.world = b2World(gravity=(0, 0))
        self.time_step = 1.0 / 60.0

        self.agents = []
        self.joints = []

        # Add contact listener
        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        # Boundary parameters (customize as needed)
        self.world_width = 60
        self.world_height = 60
        self.world_center_x = self.world_width // 2
        self.world_center_y = self.world_height // 2
        self.boundary_thickness = 0.5

        # Pygame rendering setup
        self.screen = None
        self.clock = None
        self.screen_size = (1200, 1200)
        self.scale = 20.0  # Pixels per Box2D meter

        # Create target areas
        self._create_target_areas()

        # Create boundary and agents
        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        self._create_chain()

        # Add force tracking
        self.applied_forces = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.force_scale = 2.0  # Scale factor for visualizing forces

        # Add joint limit parameter
        self.max_joints_per_agent = 2

        # Add sector sensing threshold
        self.sector_sensor_radius = 40.0

        # Add parameters for nearest neighbor detection
        self.neighbor_detection_range = 3.0  # Maximum range to detect neighbors

        # Initialize Union-Find for tracking connected agents
        self.union_find = UnionFind(n_agents)

        # Add a field to track link openness for each agent
        self.link_openness = np.ones(
            self.n_agents, dtype=np.int8
        )  # Default to open (1)

        # Add a field to track detach values for each agent
        self.detach_values = np.zeros(
            self.n_agents, dtype=np.float32
        )  # Default to 0 (no desire to detach)

        # Step tracking for truncation
        self.max_steps = 400
        self.current_step = 0

    def _update_union_find(self):
        """Update the Union-Find structure based on current joints"""
        # Reset Union-Find
        self.union_find = UnionFind(self.n_agents)

        # Add all current joints
        for joint in self.joints:
            idx_a = self.agents.index(joint.bodyA)
            idx_b = self.agents.index(joint.bodyB)
            self.union_find.union(idx_a, idx_b)

    def _create_target_areas(self):
        """Create target areas at random positions in the environment"""
        self.target_areas = []
        existing_positions = []

        for idx in range(self.n_target_areas):

            # Get position using the new method
            x, y = dynamic_position_target_area(
                self.world_width, self.world_height, existing_positions
            )
            # x, y = fixed_position_target_area(idx, self.world_width, self.world_height)
            existing_positions.append((x, y))

            # Random radius between 2 and 4
            # radius = np.random.uniform(2.0, 4.0)
            radius = 60

            # Random coupling requirement between 2 and min(5, n_agents)
            # coupling_req = np.random.randint(2, self.n_agents - 1)
            coupling_req = 1

            # Set reward scale based on coupling requirement
            reward_scale = 1.0 * coupling_req

            # Random color (semi-transparent)
            color = (
                np.random.randint(50, 200),  # R
                np.random.randint(50, 200),  # G
                np.random.randint(50, 200),  # B
                128,  # Alpha (semi-transparent)
            )

            target_area = TargetArea(x, y, radius, coupling_req, reward_scale, color)
            self.target_areas.append(target_area)

    def _create_agents(self, positions):
        for i in range(self.n_agents):
            fixture_def = b2FixtureDef(
                # shape=b2PolygonShape(box=(0.3, 0.5)),
                shape=b2CircleShape(radius=0.4),
                density=1.0,
                friction=1.0,
                isSensor=False,
            )

            fixture_def.filter.categoryBits = AGENT_CATEGORY
            fixture_def.filter.maskBits = (
                AGENT_CATEGORY
                | BOUNDARY_CATEGORY  # allow collisions between agents and boundaries
            )

            body = self.world.CreateDynamicBody(
                position=positions[i],
                fixtures=fixture_def,
            )

            self.agents.append(body)

    def _create_sequential_joints(self):
        """
        Creates joints one after the other in self.agents order
        """
        previous_body = None
        for body in self.agents:
            if previous_body:
                self._create_joint(self, bodyA=previous_body, bodyB=body)
            previous_body = body

    def _create_chain(self):
        self.agents.clear()
        self.joints.clear()

        # positions = get_scatter_positions(
        #     self.world_width, self.world_height, self.n_agents
        # )
        positions = get_linear_positions(
            self.world_width, self.world_height, self.n_agents
        )

        self._create_agents(positions)
        # self._create_sequential_joints()

    def _render_agents_as_circles(self):
        # Define colors for open and closed agents
        OPEN_COLOR = (50, 200, 50)  # Green for agents open to links
        CLOSED_COLOR = (200, 50, 50)  # Red for agents closed to links

        for idx, body in enumerate(self.agents):
            # Get circle position and radius
            center_x = body.position.x * self.scale
            center_y = self.screen_size[1] - body.position.y * self.scale
            radius = body.fixtures[0].shape.radius * self.scale  # Get radius from shape

            # Choose color based on link_openness
            if self.link_openness[idx] == 1:
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

            # Visualize detach value as a red ring that grows with detach value
            detach_value = self.detach_values[idx]
            if detach_value > 0.1:  # Only show if significant
                # Calculate ring thickness based on detach value
                ring_thickness = int(3 * detach_value)  # 0 to 3 pixels thick

                # Calculate ring color (get more red as detach increases)
                ring_red = min(255, int(150 + 105 * detach_value))  # 150-255
                ring_color = (ring_red, 50, 50)

                # Draw detach ring
                pygame.draw.circle(
                    self.screen,
                    ring_color,
                    (int(center_x), int(center_y)),
                    int(radius * 0.7),  # Inner ring
                    ring_thickness,  # Thickness varies with detach value
                )

            # Draw plus sign for open agents
            if self.link_openness[idx] == 1:
                # Draw a small plus sign for open agents
                line_length = radius * 0.5
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (int(center_x - line_length), int(center_y)),
                    (int(center_x + line_length), int(center_y)),
                    3,  # Line thickness
                )
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (int(center_x), int(center_y - line_length)),
                    (int(center_x), int(center_y + line_length)),
                    3,  # Line thickness
                )

    def _render_agents_as_boxes(self):
        for idx, body in enumerate(self.agents):
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * self.scale for v in shape.vertices]
                vertices = [(v[0], self.screen_size[1] - v[1]) for v in vertices]

                pygame.draw.polygon(self.screen, COLORS_LIST[idx], vertices)

    def _create_joint(self, bodyA, bodyB):
        anchor = (bodyA.position + bodyB.position) / 2
        joint_def = b2RevoluteJointDef(
            bodyA=bodyA, bodyB=bodyB, anchor=anchor, collideConnected=True
        )
        joint = self.world.CreateJoint(joint_def)
        self.joints.append(joint)
        return joint

    def _break_joint(self, joint):
        """Modified to update Union-Find when joints are broken"""
        self.world.DestroyJoint(joint)
        self.joints.remove(joint)

        # Update Union-Find after breaking a joint
        self._update_union_find()

    def _join_on_proximity(self, min_distance: float = 1.5):
        """Efficient version using Union-Find that respects linking preferences"""
        # Update Union-Find structure
        self._update_union_find()

        for i, bodyA in enumerate(self.agents):
            # Skip if this agent has reached its joint limit
            if self._count_joints_for_agent(bodyA) >= self.max_joints_per_agent:
                continue

            # Skip if this agent is not open to being linked to
            # Use scalar comparison to avoid array comparison issues
            if self.link_openness[i] == 0:
                continue

            for j, bodyB in enumerate(self.agents[i + 1 :], i + 1):
                # Skip if other agent has reached its joint limit
                if self._count_joints_for_agent(bodyB) >= self.max_joints_per_agent:
                    continue

                # Skip if other agent is not open to being linked to
                if self.link_openness[j] == 0:
                    continue

                # Check if already connected using Union-Find
                if self.union_find.connected(i, j):
                    continue

                dist = (bodyA.position - bodyB.position).length
                if dist < min_distance:
                    joint = self._create_joint(bodyA, bodyB)
                    if joint:
                        # Update Union-Find immediately
                        self.union_find.union(i, j)
                        break

    def _count_joints_for_agent(self, agent):
        """Count how many joints an agent is currently part of"""
        count = 0
        for joint in self.joints:
            if joint.bodyA == agent or joint.bodyB == agent:
                count += 1
        return count

    def _create_boundary(self, width, height, thickness):
        """Create boundary walls that agents can collide with"""

        # Bottom wall
        bottom_wall = self.world.CreateStaticBody(
            position=(width / 2, thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        bottom_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        bottom_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY

        # Top wall
        top_wall = self.world.CreateStaticBody(
            position=(width / 2, height - thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        top_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        top_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY

        # Left wall
        left_wall = self.world.CreateStaticBody(
            position=(thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        left_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        left_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY

        # Right wall
        right_wall = self.world.CreateStaticBody(
            position=(width - thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        right_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        right_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY

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

                # Format the density values (2 decimal places)
                density_text = (
                    f"A:{agent_densities[sector]:.3f}|T:{target_densities[sector]:.3f}"
                )

                # Render the text
                text_surface = self.sensor_font.render(density_text, True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(int(text_x), int(text_y)))

                # Draw text with white background for better visibility
                # pygame.draw.rect(
                #     self.screen, (255, 255, 255, 180), text_rect.inflate(4, 4)
                # )

                self.screen.blit(text_surface, text_rect)

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

    def _get_nearest_non_connected_agent_relative(self, agent_idx, all_states):
        """
        Find the nearest non-connected agent and return relative state information

        Returns:
            numpy array: [relative_x, relative_y, relative_vx, relative_vy, distance]
        """
        agent_position = all_states[agent_idx][:2]
        agent_velocity = all_states[agent_idx][2:4]

        # Update Union-Find to get current connected components
        self._update_union_find()

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
            if distance <= self.neighbor_detection_range and distance < min_distance:
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
                    (a.position.x - self.world_center_x) / self.world_center_x,
                    (a.position.y - self.world_center_y) / self.world_center_y,
                    # a.linearVelocity.x,
                    # a.linearVelocity.y,
                ]
                for a in self.agents
            ],
            dtype=np.float32,
        )

        # Build adjacency matrix for connections
        adjacency = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        for joint in self.joints:
            idx_a = self.agents.index(joint.bodyA)
            idx_b = self.agents.index(joint.bodyB)
            adjacency[idx_a, idx_b] = True
            adjacency[idx_b, idx_a] = True

        # For each agent, get connected agents' states
        observations = []
        for i in range(self.n_agents):
            # Own state (absolute)
            own_state = all_states[i]

            # Get indices of connected agents
            connected_indices = np.where(adjacency[i])[0]

            # Get connected states relative to this agent and pad/truncate
            if len(connected_indices) == 0:
                connected_states = np.zeros(
                    self.max_joints_per_agent * 4, dtype=np.float32
                )
            else:
                # Get states of connected agents
                connected_absolute_states = all_states[connected_indices]

                # Calculate relative positions and velocities
                # For each connected agent: [x_rel, y_rel, vx_rel, vy_rel]
                connected_relative_states = np.zeros_like(connected_absolute_states)

                for j, conn_idx in enumerate(connected_indices):
                    # Relative position = connected position - agent position
                    connected_relative_states[j, 0:2] = (
                        connected_absolute_states[j, 0:2] - own_state[0:2]
                    )

                    # Relative velocity = connected velocity - agent velocity
                    connected_relative_states[j, 2:4] = (
                        connected_absolute_states[j, 2:4] - own_state[2:4]
                    )

                # Flatten the relative states
                connected_states = connected_relative_states.flatten()

                # Pad or truncate to fixed size
                target_size = self.max_joints_per_agent * 4
                if len(connected_states) < target_size:
                    connected_states = np.pad(
                        connected_states, (0, target_size - len(connected_states))
                    )
                else:
                    connected_states = connected_states[:target_size]

            # Calculate density sensors
            density_sensors = self._calculate_density_sensors(
                i, self.sector_sensor_radius
            )

            x_diff = self.target_areas[0].x - self.agents[i].position.x
            y_diff = self.target_areas[0].y - self.agents[i].position.y

            diff_pos = np.array([x_diff / self.world_width, y_diff / self.world_height])

            # Combine all observations: own absolute state + connected relative states + density sensors
            # agent_obs = np.concatenate([own_state, connected_states, density_sensors])
            agent_obs = np.concatenate([own_state, diff_pos])

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

    def _get_rewards(self):
        """Calculate combined rewards from chain size and target areas"""
        # Calculate target area rewards
        shared_reward = 0.0

        individual_rewards = {i: 0 for i in range(self.n_agents)}

        # Check if all targets have their coupling requirements met
        all_couplings_met = False

        for target_area in self.target_areas:
            # Calculate reward for this target area
            reward_map = target_area.calculate_reward(self.agents, self.union_find)

            # A target with zero reward means its coupling requirement wasn't met
            if not reward_map:
                all_couplings_met = False
            else:

                for i, _ in enumerate(self.agents):
                    # individual_rewards[i] = reward_map[i] - self.prev_rewards[i]
                    # self.prev_rewards[i] = reward_map[i]

                    individual_rewards[i] = reward_map[i]

                shared_reward += min(reward_map.values())

        # Set terminated based on whether all couplings are met
        terminated = all_couplings_met

        return shared_reward, individual_rewards, terminated

    def _find_largest_connected_component(self):
        """
        Find the size of the largest connected component using graph traversal
        Returns the number of agents in the largest connected group
        """
        if not self.joints:
            return 1  # If no joints, largest component is 1 agent

        # Build adjacency list from joints
        adjacency_list = {i: [] for i in range(self.n_agents)}

        for joint in self.joints:
            idx_a = self.agents.index(joint.bodyA)
            idx_b = self.agents.index(joint.bodyB)
            adjacency_list[idx_a].append(idx_b)
            adjacency_list[idx_b].append(idx_a)

        visited = set()
        largest_component_size = 0

        # Find all connected components using DFS
        for agent_idx in range(self.n_agents):
            if agent_idx not in visited:
                # Start DFS from this unvisited agent
                component_size = self._dfs_component_size(
                    agent_idx, adjacency_list, visited
                )
                largest_component_size = max(largest_component_size, component_size)

        return largest_component_size

    def _dfs_component_size(self, start_idx, adjacency_list, visited):
        """
        Depth-first search to find the size of a connected component
        """
        stack = [start_idx]
        component_size = 0

        while stack:
            current_idx = stack.pop()
            if current_idx not in visited:
                visited.add(current_idx)
                component_size += 1

                # Add all connected agents to the stack
                for neighbor_idx in adjacency_list[current_idx]:
                    if neighbor_idx not in visited:
                        stack.append(neighbor_idx)

        return component_size

    def _calculate_density_sensors(self, agent_idx, sensor_radius):
        """
        Calculate density of agents and targets in four sectors around an agent.
        Returns a vector of 8 values:
        - First 4 values: agent density in [top-right, top-left, bottom-left, bottom-right]
        - Last 4 values: target density in [top-right, top-left, bottom-left, bottom-right]

        Args:
            agent_idx: Index of the agent to calculate sensors for
            sensor_radius: Radius of the detection circle

        Returns:
            numpy array: 8-element vector with density values for each sector
        """
        agent_pos = np.array(
            [self.agents[agent_idx].position.x, self.agents[agent_idx].position.y]
        )

        # Make sure the Union-Find structure is up to date
        self._update_union_find()

        # Find the root component of the current agent
        _ = self.union_find.find(agent_idx)

        # Initialize densities for the 4 sectors (for both agents and targets)
        agent_densities = np.zeros(4, dtype=np.float32)
        target_densities = np.zeros(4, dtype=np.float32)

        # Check each other agent
        for other_idx, other_agent in enumerate(self.agents):
            if other_idx == agent_idx:
                continue  # Skip self

            # Skip connected agents using UnionFind
            if self.union_find.connected(agent_idx, other_idx):
                continue

            other_pos = np.array([other_agent.position.x, other_agent.position.y])
            relative_pos = other_pos - agent_pos

            # Calculate distance
            distance = np.linalg.norm(relative_pos)

            # Skip if outside sensor radius
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
            # Add a small epsilon to avoid division by zero
            density_value = 1.0 / ((distance / self.sector_sensor_radius) + 1.0)

            # Add to appropriate sector
            agent_densities[sector] += density_value

        # Check each target area
        for target in self.target_areas:
            target_pos = np.array([target.x, target.y])
            relative_pos = target_pos - agent_pos

            # Calculate distance
            distance = np.linalg.norm(relative_pos)

            # Skip if outside sensor radius
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

            # Calculate target density contribution, weighted by reward scale and coupling requirement
            # This makes more valuable targets (higher reward scale) provide stronger signals
            # and targets with higher coupling requirements also provide stronger signals
            weight = target.reward_scale
            density_value = weight * (
                1.0 / ((distance / self.sector_sensor_radius) + 1.0)
            )

            # Set as sector value if its the highest value
            if target_densities[sector] < density_value:
                target_densities[sector] = density_value

        # Combine agent and target densities
        return np.concatenate([agent_densities, target_densities])

    def _process_detachments(self, detach_threshold=0.90):
        """Check for agents that want to detach from their connections"""
        # Create a list of joints to remove (we can't modify self.joints while iterating)
        joints_to_remove = []

        for joint in self.joints:
            # Find the indices of connected agents
            idx_a = self.agents.index(joint.bodyA)
            idx_b = self.agents.index(joint.bodyB)

            # Check if both agents want to detach
            if (
                self.detach_values[idx_a] > detach_threshold
                and self.detach_values[idx_b] > detach_threshold
            ):
                joints_to_remove.append(joint)

        # Remove the joints outside the loop
        for joint in joints_to_remove:
            self._break_joint(joint)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset previous rewards
        self.prev_rewards = dict.fromkeys(list(range(0, self.n_agents)), 0)

        # Reset step counter
        self.current_step = 0

        for body in self.agents:
            self.world.DestroyBody(body)

        self._create_chain()

        # Recreate target areas with new random positions
        self._create_target_areas()

        # Reset link_openness to all open
        self.link_openness = np.ones(self.n_agents, dtype=np.int8)

        # Reset detach values to all zero
        self.detach_values = np.zeros(self.n_agents, dtype=np.float32)

        # Reset contact listener
        self.contact_listener.reset()

        obs = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return obs, {}

    def step(self, actions):
        # # Unpack movement, link_openness, and detach actions
        # movement_actions = actions["movement"]
        # link_openness_actions = actions["link_openness"]
        # detach_actions = actions["detach"]

        # # Update link_openness and detach states - ensure we get scalar values
        # # Convert to flat numpy array if needed
        # self.link_openness = np.array(link_openness_actions).flatten()
        # self.detach_values = np.array(detach_actions).flatten()

        # # Check for detachments before applying forces
        # self._process_detachments()

        movement_actions = actions

        # Apply movement forces
        for idx, agent in enumerate(self.agents):
            force_x = float(movement_actions[idx][0])  # X component
            force_y = float(movement_actions[idx][1])  # Y component

            # Store the 2D force vector for visualization
            self.applied_forces[idx] = [force_x, force_y]

            # Apply 2D force to agent
            agent.ApplyForceToCenter((force_x, force_y), True)

        # Rest of the step method remains the same
        self.world.Step(self.time_step, 6, 2)

        # self._join_on_proximity()

        # Check for boundary collisions
        shared_reward = 0.0
        individual_rewards = dict.fromkeys(list(range(0, self.n_agents)), 0)
        if self.contact_listener.boundary_collision:
            terminated = True
            shared_reward = -10.0  # Negative reward for boundary collision
        else:
            # The normal reward calculation
            shared_reward, individual_rewards, terminated = self._get_rewards()

        # The observation
        obs = self._get_observation()

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
            "individual_rewards": individual_rewards,
        }

        self.current_step += 1

        truncated = self.current_step >= self.max_steps

        return obs, shared_reward, terminated, truncated, info

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

        # Draw agents
        self._render_agents_as_circles()

        # Draw agent indices on top of agents
        self._draw_agent_indices()

        self._draw_density_sensors()  # Add this before or after drawing agents

        # Draw joints accurately using anchor points
        for joint in self.joints:
            anchor_a = joint.anchorA * self.scale
            anchor_b = joint.anchorB * self.scale

            # Adjust for pygame's inverted y-axis
            p1 = (anchor_a[0], self.screen_size[1] - anchor_a[1])
            p2 = (anchor_b[0], self.screen_size[1] - anchor_b[1])

            # Draw the joint line (pivot-to-pivot)
            pygame.draw.line(self.screen, (0, 0, 0), p1, p2, width=3)

            # Optionally, draw pivot points explicitly
            pygame.draw.circle(self.screen, (255, 0, 0), p1, radius=5)  # pivot on bodyA
            pygame.draw.circle(self.screen, (0, 0, 255), p2, radius=5)  # pivot on bodyB

        # Draw force vectors
        # self._draw_force_vectors()

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
