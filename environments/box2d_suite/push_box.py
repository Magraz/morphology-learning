import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from Box2D import (
    b2World,
    b2PolygonShape,
    b2FixtureDef,
)

from environments.box2d_suite.agent import Agent
from environments.box2d_suite.observation import ObservationManager, OBS_DIM
from environments.box2d_suite.renderer import Renderer
from environments.box2d_suite.utils import (
    COLORS_LIST,
    AGENT_CATEGORY,
    BOUNDARY_CATEGORY,
    OBJECT_CATEGORY,
    ObjectTargetArea,
    BoundaryContactListener,
    update_object_mass_from_contacts,
)

# Each episode the target band sits against one of the four walls. A side maps
# to (axis, sign): which world axis the box must travel along, and which
# direction counts as "toward the goal" (+1 toward the high end of that axis).
_GOAL_SIDES = {
    "top": ("y", +1),
    "bottom": ("y", -1),
    "right": ("x", +1),
    "left": ("x", -1),
}

# Agent body radius (matches Agent's default) — used for spawn spacing and to
# keep agents from being placed overlapping the box.
_AGENT_RADIUS = 0.4

# Both the box and every agent must start at least this far (along the goal
# axis) from the target band, expressed as a fraction of the world extent. The
# box is placed exactly at this distance; agents are scattered on the far side
# of it. 0.4 keeps the cluster in roughly the far 60% of the map, leaving a long
# push corridor toward the goal.
_MIN_GOAL_SPAWN_FRACTION = 0.4


class PushBoxEnv(gym.Env):
    """Single-box pushing task.

    A box spawns right next to the cluster of agents, on the side facing the
    goal. Each episode the goal is a target band against one of the four walls
    (top/bottom/left/right, sampled at reset); the agents must cooperatively
    push the box into it. The dense reward is the per-step displacement of the
    box toward the goal wall; a completion bonus is paid once the box reaches
    the target region.

    Shares the box2d_suite observation/renderer/boundary machinery with
    ``multi_box_push`` so the layout stays consistent across the suite.
    """

    metadata = {"render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        n_agents: int = 3,
        max_steps: int = 1024,
        reward_mode: str = "dense",
    ):
        super().__init__()

        self.n_agents = n_agents
        self.n_objects = 1  # single box
        self.render_mode = render_mode
        self.reward_mode = reward_mode

        # Target region (top of the map)
        self.target_areas = []

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, 2),  # (n_agents, action_dim)
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, OBS_DIM), dtype=np.float32
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

        # Auto-scale world size based on entity count
        # Reference: 8 entities (5 agents + 3 objects) -> 30x30
        _total_entities = self.n_agents + self.n_objects
        self.world_width = int(30 * max(1.0, _total_entities / 8) ** 0.5)
        self.world_height = self.world_width
        self.world_center_x = self.world_width // 2
        self.world_center_y = self.world_height // 2
        self.world_diagonal = np.sqrt(self.world_height**2 + self.world_width**2)
        self.boundary_thickness = 0.5

        self.boundary_bodies = []  # Track boundary walls

        # Default goal side so __init__ can build a valid target/observation
        # before the first reset (reset re-samples a random side each episode).
        self.goal_side = "top"
        self.goal_axis, self.goal_sign = _GOAL_SIDES[self.goal_side]

        # Minimum distance the box and agents must start from the goal band,
        # along the goal axis. World is square, so the same value applies to
        # both axes.
        self.min_goal_spawn_distance = _MIN_GOAL_SPAWN_FRACTION * self.world_width

        # Create target area and boundary/agents
        self._create_target_areas()
        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        self._init_agents()

        # Add force tracking
        self.applied_forces = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.force_scale = 2.0  # Scale factor for visualizing forces
        self.force_multiplier = 100.0  # Max force an agent can apply per axis
        # Per-agent normal contact force against the box, averaged over the
        # last physics step. Populated from the contact listener's PostSolve.
        self.agent_contact_forces = np.zeros(self.n_agents, dtype=np.float32)

        # Velocity normalization constant (agents have linear damping=10.0,
        # so terminal velocity is bounded; world_width/10 keeps values ~[-1,1])
        self.velocity_norm = self.world_width / 10.0

        # Scale sector sensor radius proportionally to world size
        self.sector_sensor_radius = self.world_width / 3.0

        # Add parameters for nearest neighbor detection
        self.neighbor_detection_range = 3.0  # Maximum range to detect neighbors

        # Step tracking for truncation
        self.max_steps = max_steps
        self.current_step = 0

        self.objects = []  # Track dynamic objects

        self.observation_manager = ObservationManager(self)
        self.renderer = Renderer(self)

    def _create_target_areas(self):
        """Create a target band spanning the wall on this episode's goal side.

        ``self.goal_side`` (set in reset) selects one of the four walls. The
        band spans the full inner length of that wall and is ``band`` thick.
        """
        self.target_areas = []

        bt = self.boundary_thickness

        if self.goal_axis == "y":
            # Top/bottom band: full inner width, thickness scaled to height.
            band = max(5.0, 5.0 * self.world_height / 30.0)
            x = self.world_width / 2
            width = self.world_width - 2 * bt
            height = band
            if self.goal_sign > 0:  # top
                y = self.world_height - bt - band / 2
            else:  # bottom
                y = bt + band / 2
        else:
            # Left/right band: full inner height, thickness scaled to width.
            band = max(5.0, 5.0 * self.world_width / 30.0)
            y = self.world_height / 2
            height = self.world_height - 2 * bt
            width = band
            if self.goal_sign > 0:  # right
                x = self.world_width - bt - band / 2
            else:  # left
                x = bt + band / 2

        self.target_areas.append(ObjectTargetArea(x, y, width, height))

    def _init_agents(self):
        positions = self._scatter_agent_positions()
        self.agent_spawn_positions = positions
        self._create_agents(positions)

    def _goal_axis_spawn_limit(self):
        """Goal-axis coordinate at exactly ``min_goal_spawn_distance`` from the
        band, on the interior side. Both the box and the agents must stay on the
        far side of this line (i.e. at least that far from the goal)."""
        target = self.target_areas[0]
        d = self.min_goal_spawn_distance
        if self.goal_axis == "y":
            edge = (
                target.y - target.height / 2
                if self.goal_sign > 0
                else target.y + target.height / 2
            )
        else:
            edge = (
                target.x - target.width / 2
                if self.goal_sign > 0
                else target.x + target.width / 2
            )
        return edge - self.goal_sign * d

    def _overlaps_box(self, x, y, clearance=0.3):
        """True if a disc of radius ``_AGENT_RADIUS`` at (x, y) would intersect
        the box (plus a small clearance). No-op when the box isn't placed yet."""
        if not getattr(self, "objects", None):
            return False
        bx, by = self.box_spawn_center
        hx, hy = self.box_half_size
        # Distance from the point to the (axis-aligned) box surface.
        dx = max(abs(x - bx) - hx, 0.0)
        dy = max(abs(y - by) - hy, 0.0)
        return (dx * dx + dy * dy) ** 0.5 < (_AGENT_RADIUS + clearance)

    def _scatter_agent_positions(self):
        """Scatter agents on the far side of the box, away from the goal band.

        Agents are sampled within the interior region that is at least
        ``min_goal_spawn_distance`` from the goal (the same line the box sits
        on), spaced apart, and rejected if they would overlap the box. Uses the
        seeded gym RNG so spawns are reproducible.
        """
        bt = self.boundary_thickness
        r = _AGENT_RADIUS
        margin = 1.0
        min_sep = 2.0

        a_limit = self._goal_axis_spawn_limit()

        # Goal axis (a) range: keep agents on the interior side of a_limit.
        # Perpendicular axis (p) range: the full inner span of that wall.
        if self.goal_axis == "y":
            p_lo, p_hi = bt + r + margin, self.world_width - bt - r - margin
            if self.goal_sign > 0:
                a_lo, a_hi = bt + r + margin, a_limit
            else:
                a_lo, a_hi = a_limit, self.world_height - bt - r - margin
        else:
            p_lo, p_hi = bt + r + margin, self.world_height - bt - r - margin
            if self.goal_sign > 0:
                a_lo, a_hi = bt + r + margin, a_limit
            else:
                a_lo, a_hi = a_limit, self.world_width - bt - r - margin

        # Degenerate guard (shouldn't happen for sane world sizes).
        a_lo, a_hi = min(a_lo, a_hi), max(a_lo, a_hi)

        def to_xy(a, p):
            return (p, a) if self.goal_axis == "y" else (a, p)

        positions = []
        for i in range(self.n_agents):
            placed = False
            for _ in range(200):
                a = float(self.np_random.uniform(a_lo, a_hi))
                p = float(self.np_random.uniform(p_lo, p_hi))
                x, y = to_xy(a, p)
                if self._overlaps_box(x, y):
                    continue
                if any(
                    ((x - qx) ** 2 + (y - qy) ** 2) ** 0.5 < min_sep
                    for qx, qy in positions
                ):
                    continue
                positions.append((x, y))
                placed = True
                break

            if not placed:
                # Fallback: spread evenly along the perpendicular axis at the
                # far edge of the zone, nudged off the box if needed.
                a = a_lo if self.goal_sign > 0 else a_hi
                p = p_lo + (i + 0.5) * (p_hi - p_lo) / self.n_agents
                x, y = to_xy(a, p)
                positions.append((x, y))

        return positions

    def _create_dynamic_objects(self):
        """Create the single box at a fixed minimum distance from the goal.

        The box is placed first (before the agents) so the agent scatter can
        position the cluster behind it without overlapping. Its goal-axis
        coordinate is exactly ``min_goal_spawn_distance`` from the band; its
        perpendicular coordinate is randomized along the wall for variety.
        """
        self.objects.clear()

        # Vary the box size slightly between episodes for robustness. The base
        # 1.5 half-extent is the minimum; each reset samples up to +20% larger
        # (uniformly), keeping the box square. Uses the seeded gym RNG so runs
        # stay reproducible.
        min_half = 1.5
        max_half = min_half * 1.2
        half = float(self.np_random.uniform(min_half, max_half))
        self.box_half_size = (half, half)
        half_size = self.box_half_size

        # Box color, offset by n_agents to avoid clashing with agent colors
        color = COLORS_LIST[self.n_agents % len(COLORS_LIST)]

        # Heavy base density (matches multi_box_push) so the box barely budges
        # until the coupling requirement is met. Once all n_agents push together
        # the density is dropped dramatically by
        # update_object_mass_from_contacts, making it easy to move the length of
        # the map within an episode — so cooperating is what unlocks progress.
        base_density = 20.0
        self.object_base_densities = [base_density]

        # Place the box exactly min_goal_spawn_distance from the band along the
        # goal axis, so it (and the agents behind it) start far from the goal.
        # The perpendicular coordinate is randomized along the wall.
        bt = self.boundary_thickness
        hx, hy = half_size
        a_coord = self._goal_axis_spawn_limit()

        if self.goal_axis == "y":
            spawn_y = float(np.clip(a_coord, bt + hy, self.world_height - bt - hy))
            spawn_x = float(
                self.np_random.uniform(
                    bt + hx + 1.0, self.world_width - bt - hx - 1.0
                )
            )
        else:
            spawn_x = float(np.clip(a_coord, bt + hx, self.world_width - bt - hx))
            spawn_y = float(
                self.np_random.uniform(
                    bt + hy + 1.0, self.world_height - bt - hy - 1.0
                )
            )

        # Remember the center so the agent scatter can avoid overlapping it.
        self.box_spawn_center = (spawn_x, spawn_y)

        shape = b2PolygonShape(box=half_size)
        fixture_def = b2FixtureDef(
            shape=shape,
            density=base_density,
            friction=0.3,
            restitution=0.2,
        )
        fixture_def.filter.categoryBits = OBJECT_CATEGORY
        fixture_def.filter.maskBits = (
            AGENT_CATEGORY | BOUNDARY_CATEGORY | OBJECT_CATEGORY
        )

        body = self.world.CreateDynamicBody(
            position=(spawn_x, spawn_y),
            fixtures=fixture_def,
            linearDamping=5.0,
            angularDamping=8.0,
        )
        body.userData = {
            "type": "object",
            "color": color,
            # Coupling requirement: every agent must push together. Until all
            # n_agents are touching, the box keeps its heavy base density; once
            # the requirement is met it becomes far lighter (see
            # update_object_mass_from_contacts), so cooperating is the only way
            # to move it the length of the map within an episode.
            "index": 0,
            "coupling": self.n_agents,
        }

        self.objects.append(body)

    def _create_agents(self, positions):
        for i in range(self.n_agents):
            agent = Agent(self.world, positions[i], index=i)
            self.agents.append(agent)

    def _create_boundary(self, width, height, thickness):
        """Create boundary walls that agents can collide with"""

        # Bottom wall
        bottom_wall = self.world.CreateStaticBody(
            position=(width / 2, thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        bottom_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        bottom_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
        self.boundary_bodies.append(bottom_wall)

        # Top wall
        top_wall = self.world.CreateStaticBody(
            position=(width / 2, height - thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        top_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        top_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
        self.boundary_bodies.append(top_wall)

        # Left wall
        left_wall = self.world.CreateStaticBody(
            position=(thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        left_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        left_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
        self.boundary_bodies.append(left_wall)

        # Right wall
        right_wall = self.world.CreateStaticBody(
            position=(width - thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        right_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        right_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
        self.boundary_bodies.append(right_wall)

    def get_agents_touching_objects(self):
        """Return result[obj_idx] = list of agent indices touching that object."""
        result = [[] for _ in range(len(self.objects))]
        for obj_idx, obj in enumerate(self.objects):
            obj_pos = np.array([obj.position.x, obj.position.y])
            for agent in self.agents:
                agent_pos = np.array([agent.position.x, agent.position.y])
                dist = self.observation_manager._agent_object_distance(
                    agent_pos, obj, obj_pos
                )
                if dist <= agent.radius + 0.2:
                    result[obj_idx].append(agent.index)
        return result

    def _get_observation(self):
        return self.observation_manager.get_observation()

    def _get_rewards(self):
        reward, done = self._calculate_goal_push_reward()
        individual_rewards = [reward for _ in range(self.n_agents)]
        return reward, individual_rewards, done

    def _calculate_goal_push_reward(self):
        """Reward proportional to how much the box moved toward the goal wall.

        Dense reward = per-step displacement of the box along the goal axis in
        the goal direction (``goal_sign``), so it is positive whenever the box
        moves toward the target band regardless of which wall the band is on. A
        one-time completion bonus is paid when the box enters the band, which
        also terminates the episode.
        """
        task_reward = 0.0
        done = False

        obj = self.objects[0]
        # Position along the goal axis (x for left/right goals, y for top/bottom).
        box_coord = obj.position.x if self.goal_axis == "x" else obj.position.y

        # Initialize tracking on the first step after a reset.
        if not hasattr(self, "prev_box_coord"):
            self.prev_box_coord = box_coord

        # Displacement toward the goal since last step (positive == toward band).
        shaping_reward = (box_coord - self.prev_box_coord) * self.goal_sign
        self.prev_box_coord = box_coord

        completion_reward = 0.0
        target = self.target_areas[0]
        if not getattr(self, "delivered", False) and target.contains_object(obj):
            completion_reward = 100.0
            self.delivered = True
            done = True

        if self.reward_mode == "dense":
            task_reward = shaping_reward + completion_reward
        else:
            task_reward = completion_reward

        return task_reward, done

    def _get_info(self, task_reward=0.0):
        """Build the info dictionary returned by reset() and step()."""
        return {
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
                {"x": ag.position.x, "y": ag.position.y} for ag in self.agents
            ],
            "task_reward": task_reward,
            "agents_2_objects": self.get_agents_touching_objects(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset step counter
        self.current_step = 0

        # Pick this episode's goal wall (top/bottom/left/right) and derive the
        # axis the box must travel along and which direction is "toward goal".
        # Done before objects/targets are built so both can reference it.
        self.goal_side = str(self.np_random.choice(list(_GOAL_SIDES)))
        self.goal_axis, self.goal_sign = _GOAL_SIDES[self.goal_side]

        # Reset reward tracking
        if hasattr(self, "prev_box_coord"):
            del self.prev_box_coord
        self.delivered = False

        # Reset per-step contact force readout
        self.agent_contact_forces.fill(0.0)

        # Create a completely fresh Box2D world to avoid stale references
        self.world = b2World(gravity=(0, 0))

        # Re-attach contact listener to new world
        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        # Clear all body references
        self.agents.clear()
        self.objects.clear()
        self.boundary_bodies.clear()

        # Recreate everything in the fresh world
        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        # Order matters: target band defines the goal, the box is placed at a
        # fixed distance from it, then agents are scattered behind the box
        # (away from the goal and clear of the box).
        self._create_target_areas()
        self._create_dynamic_objects()
        self._init_agents()

        obs = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return obs, self._get_info()

    def step(self, actions):

        # PREPROCESS ENVIRONMENT ACTION
        movement_action = actions[:, :2]

        # PROCESS ENVIRONMENT ACTION
        for agent in self.agents:
            force_x = (
                np.clip(movement_action[agent.index][0], -1, 1) * self.force_multiplier
            )
            force_y = (
                np.clip(movement_action[agent.index][1], -1, 1) * self.force_multiplier
            )

            self.applied_forces[agent.index] = [force_x, force_y]
            agent.apply_force(force_x, force_y)

        # Adjust box mass based on how many agents are pushing: the box only
        # becomes light once the coupling requirement (all n_agents) is met.
        update_object_mass_from_contacts(self)

        # Step physics
        self.world.Step(self.time_step, 6, 2)

        # Convert accumulated agent-object normal impulses into average forces.
        self.agent_contact_forces.fill(0.0)
        for (
            agent_idx,
            impulse,
        ) in self.contact_listener.agent_object_normal_impulse.items():
            self.agent_contact_forces[agent_idx] = impulse / self.time_step

        # CALCULATE REWARDS
        task_reward = 0.0
        individual_rewards = np.array([0.0 for _ in range(self.n_agents)])

        terminated = False

        if self.contact_listener.boundary_collision:
            terminated = True
        else:
            task_reward, individual_rewards, terminated = self._get_rewards()

        # Get observation BEFORE resetting contacts
        obs = self._get_observation()

        # Reset collision flag for next step (AFTER observation and rewards)
        self.contact_listener.reset()

        info = self._get_info(task_reward=task_reward)

        self.current_step += 1

        truncated = self.current_step >= self.max_steps

        return obs, task_reward, terminated, truncated, info

    def render(self):
        self.renderer.render()

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    # Manual keyboard debugger (same controls as the rest of the suite).
    env = PushBoxEnv(render_mode="human", n_agents=6, max_steps=10)
    obs, info = env.reset()

    running = True
    current_agent_idx = 0
    cum_rew = 0
    group_control = False

    print("\n" + "=" * 50)
    print(" PUSH-TO-TOP ENVIRONMENT DEBUGGER")
    print("=" * 50)
    print(f" Controlling Agent: {current_agent_idx}")
    print(" Controls:")
    print("  [ARROWS] : Move Active Agent")
    print("  [SPACE]  : Switch Active Agent")
    print("  [G]      : Toggle Group Control (Radius 5)")
    print("  [ESC]    : Quit")
    print("=" * 50 + "\n")

    while running:
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

        keys = pygame.key.get_pressed()

        actions = np.zeros((env.n_agents, 2), dtype=np.float32)

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

        actions[current_agent_idx, 0] = force_x
        actions[current_agent_idx, 1] = force_y

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

        obs, reward, terminated, truncated, info = env.step(actions)
        cum_rew += reward

        print(f"Cumulative Rew {cum_rew:.3f} Rew {reward:.3f}")

        env.render()

        if terminated or truncated:
            print(">>> Environment Reset")
            cum_rew = 0
            obs, info = env.reset()

    env.close()
