import numpy as np
from Box2D import b2ContactListener

AGENT_CATEGORY = 0x0001  # Binary: 0001
BOUNDARY_CATEGORY = 0x0002  # Binary: 0010

COLORS_LIST = [
    # Primary colors
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    # Secondary colors
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (255, 192, 203),  # Pink
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Navy
    (128, 128, 0),  # Olive
    # Tertiary colors
    (255, 99, 71),  # Tomato
    (70, 130, 180),  # Steel Blue
    (255, 20, 147),  # Deep Pink
    (32, 178, 170),  # Light Sea Green
    (255, 215, 0),  # Gold
    (138, 43, 226),  # Blue Violet
    # Earth tones
    (210, 180, 140),  # Tan
    (139, 69, 19),  # Saddle Brown
    (160, 82, 45),  # Sienna
    (205, 92, 92),  # Indian Red
    (222, 184, 135),  # Burlywood
    (188, 143, 143),  # Rosy Brown
    # Cool colors
    (95, 158, 160),  # Cadet Blue
    (72, 61, 139),  # Dark Slate Blue
    (123, 104, 238),  # Medium Slate Blue
    (0, 191, 255),  # Deep Sky Blue
    (30, 144, 255),  # Dodger Blue
    (100, 149, 237),  # Cornflower Blue
    # Warm colors
    (255, 69, 0),  # Red Orange
    (255, 140, 0),  # Dark Orange
]


class UnionFind:
    """Union-Find data structure for efficiently tracking connected components"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already connected

        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)


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

    def calculate_reward(self, agents, union_find: UnionFind):
        """Calculate reward based on proximity to center if coupling requirement is met"""

        reward_map = [0 for _ in range(0, len(agents))]

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

        # # Calculate reward based on proximity
        for i in qualifying_agents:
            dist = np.sqrt(
                (agents[i].position.x - self.x) ** 2
                + (agents[i].position.y - self.y) ** 2
            )
            # Reward decreases with distance (10.0 at center, 0.0 at radius)
            reward_map[i] = 1 / (dist**2 + 1)

        # Give flat positive reward for being at the target
        # for i in qualifying_agents:
        #     reward_map[i] = 1

        return reward_map


class Agent:
    def __init__(self, body, previous_position=(0, 0)):
        self.body = body
        self.prev_position = previous_position


class BoundaryContactListener(b2ContactListener):
    """Contact listener to detect collisions between agents and boundaries"""

    def __init__(self):
        super().__init__()
        self.boundary_collision = False

    def BeginContact(self, contact):
        # Check if this is a collision between an agent and boundary
        fixture_a, fixture_b = contact.fixtureA, contact.fixtureB

        category_a = fixture_a.filterData.categoryBits
        category_b = fixture_b.filterData.categoryBits

        # Check if one fixture is an agent and the other is a boundary
        if (category_a == AGENT_CATEGORY and category_b == BOUNDARY_CATEGORY) or (
            category_b == AGENT_CATEGORY and category_a == BOUNDARY_CATEGORY
        ):
            self.boundary_collision = True

    def reset(self):
        """Reset collision flag"""
        self.boundary_collision = False


def get_linear_positions(world_width, world_height, n_agents, spacing=2):
    """
    Generate positions for agents arranged in a horizontal line centered on the map,
    with half the agents on the left of center and half on the right.

    Args:
        world_width (float): Width of the world
        world_height (float): Height of the world
        n_agents (int): Number of agents to position
        spacing (float): Distance between adjacent agents

    Returns:
        list: List of (x, y) tuples for each agent position
    """
    positions = []

    # Calculate world center
    center_x = world_width / 2
    center_y = world_height / 2

    # Calculate total width of the formation
    total_width = (n_agents - 1) * spacing

    # Calculate starting position (leftmost agent)
    start_x = center_x - (total_width / 2)

    # Position agents in a line
    for i in range(n_agents):
        x_pos = start_x + (i * spacing)
        positions.append((x_pos, center_y))

    return positions


def get_scatter_positions(world_width, world_height, n_agents, min_distance=10):
    """
    Generate random starting positions for all agents

    Args:
        min_distance (float): Minimum distance between agents when maintain_chain=False

    Returns:
        list: List of (x, y) tuples for each agent position
    """
    positions = []

    # Define safe boundaries (away from walls)
    margin = 10.0  # Distance from walls
    safe_x_min = margin
    safe_x_max = world_width - margin
    safe_y_min = margin
    safe_y_max = world_height - margin

    # Completely random positions with minimum distance constraint
    for i in range(n_agents):
        attempts = 0
        max_attempts = 200

        while attempts < max_attempts:
            pos_x = np.random.uniform(safe_x_min, safe_x_max)
            pos_y = np.random.uniform(safe_y_min, safe_y_max)

            # Check minimum distance from other agents
            valid_position = True
            for existing_pos in positions:
                distance = np.sqrt(
                    (pos_x - existing_pos[0]) ** 2 + (pos_y - existing_pos[1]) ** 2
                )
                if distance < min_distance:
                    valid_position = False
                    break

            if valid_position:
                positions.append((pos_x, pos_y))
                break

            attempts += 1

        # Fallback if we can't find a valid position
        if attempts >= max_attempts:
            pos_x = safe_x_min + (i * (safe_x_max - safe_x_min) / n_agents)
            pos_y = safe_y_min + np.random.uniform(0, safe_y_max - safe_y_min)
            positions.append((pos_x, pos_y))

    return positions


def fixed_position_target_area(
    idx,
    width,
    height,
):
    # Calculate world center
    center_x = width / 2
    center_y = height / 2

    x_offset = 15
    y_offset = 8

    # positions = [
    #     (center_x + x_offset, center_y + y_offset),
    #     (center_x + x_offset, center_y - y_offset),
    #     (center_x - x_offset, center_y + y_offset),
    #     (center_x - x_offset, center_y - y_offset),
    # ]

    positions = [
        (center_x + 0, center_y + y_offset),
        (center_x + 0, center_y - y_offset),
        (center_x - x_offset, center_y + y_offset),
        (center_x - x_offset, center_y - y_offset),
    ]
    return positions[idx]


def dynamic_position_target_area(
    width,
    height,
    existing_positions=None,
    min_distance=15.0,
    max_attempts=200,
    min_center_radius=10.0,  # New parameter for minimum distance from center
):
    """
    Position a target area with direction-based offset from the world center,
    ensuring minimum distance from existing targets and placement outside a central circle.

    Args:
        width (float): World width
        height (float): World height
        existing_positions (list): List of (x, y) tuples of existing target positions
        min_distance (float): Minimum distance required between target positions
        max_attempts (int): Maximum number of positioning attempts before falling back
        min_center_radius (float): Minimum radius from world center where targets can be placed

    Returns:
        tuple: (x, y) coordinates for the target area
    """
    # Initialize existing positions if None
    if existing_positions is None:
        existing_positions = []

    # Calculate world center
    center_x = width / 2
    center_y = height / 2

    # Define bounding box dimensions (80% of world size)
    box_width = width * 0.95
    box_height = height * 0.95

    # Calculate bounding box boundaries
    box_left = center_x - box_width / 2
    box_right = center_x + box_width / 2
    box_bottom = center_y - box_height / 2
    box_top = center_y + box_height / 2

    # Margin from edges
    margin = 0
    boundary_margin = 5

    for attempt in range(max_attempts):
        # Step 1: Generate random position within bounding box
        x = np.random.uniform(box_left + margin, box_right - margin)
        y = np.random.uniform(box_bottom + margin, box_top - margin)

        # Step 2: Calculate direction vector from center
        dir_x = x - center_x
        dir_y = y - center_y

        # Step 3: Calculate distance from center
        dist_from_center = np.sqrt(dir_x**2 + dir_y**2)

        # Step 4: Normalize direction vector (if not zero)
        if dist_from_center > 0.001:  # Avoid division by zero
            dir_x /= dist_from_center
            dir_y /= dist_from_center

        # Step 5: Apply offset to ensure position is outside min_center_radius
        # If already outside, add smaller offset; if inside, push outside
        if dist_from_center < min_center_radius:
            # Position is inside forbidden circle, push it outside
            offset_magnitude = (
                min_center_radius - dist_from_center + 5.0
            )  # Extra offset for safety
        else:
            # Position is already outside, add smaller offset
            offset_magnitude = 0.0

        # Apply offset in direction from center
        x = center_x + dir_x * (dist_from_center + offset_magnitude)
        y = center_y + dir_y * (dist_from_center + offset_magnitude)

        # Step 6: Ensure the position stays within world bounds
        x = np.clip(x, boundary_margin, width - boundary_margin)
        y = np.clip(y, boundary_margin, height - boundary_margin)

        # Step 7: Verify the position is still outside the central circle
        # (clipping might have moved it back inside)
        new_dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        if new_dist < min_center_radius:
            continue  # Skip this position as it's inside the forbidden circle

        # Step 8: Check minimum distance from all existing target positions
        valid_position = True
        for pos_x, pos_y in existing_positions:
            distance = np.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2)
            if distance < min_distance:
                valid_position = False
                break

        # If position is valid (meets minimum distance and outside center), return it
        if valid_position:
            return x, y

    # Fallback strategy: if we can't find a valid position, try a systematic approach
    # Use positions at different angles around the min_center_radius circle
    angles = np.linspace(0, 2 * np.pi, max_attempts, endpoint=False)
    np.random.shuffle(angles)  # Randomize the order for better spread

    for angle in angles:
        # Position on circle with min_center_radius + small offset
        x = center_x + (min_center_radius + 2.0) * np.cos(angle)
        y = center_y + (min_center_radius + 2.0) * np.sin(angle)

        # Ensure within bounds
        x = np.clip(x, boundary_margin, width - boundary_margin)
        y = np.clip(y, boundary_margin, height - boundary_margin)

        # Check distance from existing targets
        valid_position = True
        for pos_x, pos_y in existing_positions:
            distance = np.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2)
            if distance < min_distance:
                valid_position = False
                break

        if valid_position:
            return x, y

    # Final fallback: return position on circle at random angle
    angle = np.random.uniform(0, 2 * np.pi)
    x = center_x + (min_center_radius + 2.0) * np.cos(angle)
    y = center_y + (min_center_radius + 2.0) * np.sin(angle)

    # Final bounds check
    x = np.clip(x, boundary_margin, width - boundary_margin)
    y = np.clip(y, boundary_margin, height - boundary_margin)

    return x, y


def add_dictionary_values(dict1, dict2):
    """
    Adds values of two dictionaries, summing values for common keys
    and including all other key-value pairs.
    """
    merged_dict = {}

    # Add values from dict1
    for key, value in dict1.items():
        merged_dict[key] = value

    # Add values from dict2, summing if key exists in merged_dict
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict
