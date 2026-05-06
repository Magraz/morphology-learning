import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from functools import partial
from algorithms.mappo.hypergraph import (
    build_hypergraph,
    compute_hyperedge_structural_entropy_batch,
    distance_based_hyperedges,
    object_contact_hyperedges,
)

from Box2D import (
    b2World,
    b2PolygonShape,
    b2FixtureDef,
    b2DistanceJointDef,
    b2Vec2,
)

from environments.box2d_suite.agent import Agent
from environments.box2d_suite.observation import ObservationManager
from environments.box2d_suite.renderer import Renderer
from environments.box2d_suite.utils import (
    COLORS_LIST,
    AGENT_CATEGORY,
    BOUNDARY_CATEGORY,
    OBJECT_CATEGORY,
    ObjectTargetArea,
    BoundaryContactListener,
    get_scatter_positions,
)


class MultiBoxGrabEnv(gym.Env):
    metadata = {"render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        n_agents=3,
        n_objects=3,
        max_steps=1024,
        reward_mode="dense",
    ):
        super().__init__()

        self.n_agents = n_agents
        self.n_objects = n_objects
        self.render_mode = render_mode
        self.reward_mode = reward_mode

        # Add target areas parameters
        self.target_areas = []

        # Action layout: [force_x, force_y, grab] per agent.
        # grab is binary: 0 = release, 1 = grab the closest object.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, 3),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, 21), dtype=np.float32
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

        # Create target areas
        self._create_target_areas()

        # Create boundary and agents
        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )

        self._init_agents()

        # Create boxes coupling reqs
        self.objects_push_coupling_list = np.random.default_rng(42).integers(
            2, (self.n_agents // 2) + 1, (self.n_objects)
        )

        # Add force tracking
        self.applied_forces = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.force_scale = 2.0  # Scale factor for visualizing forces

        # Velocity normalization constant (agents have linear damping=10.0,
        # so terminal velocity is bounded; world_width/10 keeps values ~[-1,1])
        self.velocity_norm = self.world_width / 10.0

        # Scale sector sensor radius proportionally to world size
        self.sector_sensor_radius = self.world_width / 3.0

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

        # agent_joints[i] = (joint, obj_idx) when agent i is grabbing, else None
        self.agent_joints = [None] * self.n_agents

        self.observation_manager = ObservationManager(self)
        self.renderer = Renderer(self)

    def _create_target_areas(self):
        """Create target areas scaled proportionally to world size."""
        self.target_areas = []

        bt = self.boundary_thickness
        target_h = max(5.0, 5.0 * self.world_height / 30.0)
        inner_w = self.world_width - 2 * bt
        target_area = ObjectTargetArea(
            self.world_width / 2,
            self.world_height - bt - target_h / 2,
            inner_w,
            target_h,
        )
        self.target_areas.append(target_area)

    def _init_agents(self):

        positions = get_scatter_positions(
            self.world_width, self.world_height, self.n_agents
        )

        self._create_agents(positions)

    def _create_dynamic_objects(self):
        """Create n_objects square boxes with dynamically assigned colors."""
        self.objects.clear()

        half_size = (1.5, 1.5)

        # Assign colors from COLORS_LIST, offset by n_agents to avoid collision
        color_offset = self.n_agents
        colors = [
            COLORS_LIST[(color_offset + i) % len(COLORS_LIST)]
            for i in range(self.n_objects)
        ]

        self.object_base_densities = []

        center_x = self.world_width / 2
        center_y = self.world_height / 2

        spawn_width = self.world_width * 0.8
        spawn_height = self.world_height * 0.3

        base_density = 20.0

        placed_positions = []
        min_separation = 4.0

        for i in range(self.n_objects):
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

            min_x = center_x - spawn_width / 2 + half_size[0]
            max_x = center_x + spawn_width / 2 - half_size[0]
            min_y = center_y - spawn_height / 2 + half_size[1]
            max_y = center_y + spawn_height / 2 - half_size[1]

            # Minimum horizontal separation so boxes don't stack vertically
            min_x_separation = half_size[0] * 2 + 1.0

            max_attempts = 100
            for attempt in range(max_attempts):
                pos_x = np.random.uniform(min_x, max_x)
                pos_y = np.random.uniform(min_y, max_y)

                too_close = False
                for prev_pos in placed_positions:
                    dx = abs(pos_x - prev_pos[0])
                    dist = np.sqrt(
                        (pos_x - prev_pos[0]) ** 2 + (pos_y - prev_pos[1]) ** 2
                    )
                    if dist < min_separation or dx < min_x_separation:
                        too_close = True
                        break

                if not too_close:
                    break

            pos = (pos_x, pos_y)
            placed_positions.append(pos)

            body = self.world.CreateDynamicBody(
                position=pos,
                fixtures=fixture_def,
                linearDamping=5.0,
                angularDamping=8.0,
            )

            body.userData = {
                "type": "object",
                "color": colors[i],
                "index": i,
                "coupling": self.objects_push_coupling_list[i],
            }

            self.objects.append(body)
            self.object_base_densities.append(base_density)

    def _update_object_mass_from_grabs(self):
        """
        Object is movable only when enough agents are grabbing it
        (n_grabbing >= coupling). Otherwise density is kept at the heavy
        base value so the joint forces cannot drag it around.
        """
        grabbers_per_object = self._grabbers_per_object()
        for obj_idx, obj in enumerate(self.objects):
            base_density = self.object_base_densities[obj_idx]
            n_grabbing = len(grabbers_per_object[obj_idx])

            if obj.userData["coupling"] <= n_grabbing:
                new_density = 0.05 * obj.userData["coupling"]
            else:
                new_density = base_density

            for fixture in obj.fixtures:
                fixture.density = new_density

            obj.ResetMassData()

    def _grabbers_per_object(self):
        """Return list[obj_idx] -> [agent_idx, ...] currently grabbing it."""
        result = [[] for _ in range(len(self.objects))]
        for agent_idx, entry in enumerate(self.agent_joints):
            if entry is None:
                continue
            _, obj_idx = entry
            result[obj_idx].append(agent_idx)
        return result

    def _process_grabs(self, grab_signals):
        """
        Apply binary grab signals: open joints when an agent grabs and has
        none, release joints when grab=0.
        """
        for agent_idx, signal in enumerate(grab_signals):
            wants_grab = signal >= 0.5
            current = self.agent_joints[agent_idx]

            if wants_grab and current is None:
                self._create_grab_joint(agent_idx)
            elif not wants_grab and current is not None:
                joint, _ = current
                self.world.DestroyJoint(joint)
                self.agent_joints[agent_idx] = None

    def _create_grab_joint(self, agent_idx):
        """Create a distance joint between agent and its closest object."""
        if not self.objects:
            return

        agent = self.agents[agent_idx]
        agent_pos = np.array([agent.position.x, agent.position.y])

        grab_range = agent.radius + 0.2

        closest_idx = None
        closest_dist = float("inf")

        for obj_idx, obj in enumerate(self.objects):
            obj_pos = np.array([obj.position.x, obj.position.y])
            dist = self.observation_manager._agent_object_distance(
                agent_pos, obj, obj_pos
            )
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = obj_idx

        if closest_idx is None or closest_dist > grab_range:
            return

        target_obj = self.objects[closest_idx]

        theta_from_object_to_agent = np.arctan2(
            agent.body.position.y - target_obj.position.y,
            agent.body.position.x - target_obj.position.x,
        )

        norm = np.linalg.norm(
            np.array(agent.body.position.tuple) - np.array(target_obj.position.tuple)
        )

        x = (norm - agent.radius) * np.cos(theta_from_object_to_agent)
        y = (norm - agent.radius) * np.sin(theta_from_object_to_agent)

        joint_def = b2DistanceJointDef(
            bodyA=agent.body,
            bodyB=target_obj,
            anchorA=agent.body.position,
            # anchorB=target_obj.position,
            anchorB=target_obj.position + b2Vec2(x, y),
            collideConnected=True,
        )
        joint = self.world.CreateJoint(joint_def)
        self.agent_joints[agent_idx] = (joint, closest_idx)

    def _release_all_grabs(self):
        for agent_idx, entry in enumerate(self.agent_joints):
            if entry is None:
                continue
            joint, _ = entry
            self.world.DestroyJoint(joint)
            self.agent_joints[agent_idx] = None

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

    def get_agents_grabbing_objects(self):
        """Return a list where result[obj_idx] contains the indices of
        agents currently grabbing that object via an active joint."""
        return self._grabbers_per_object()

    def _get_observation(self):
        return self.observation_manager.get_observation()

    def _get_rewards(self):
        """Calculate combined rewards from chain size and target areas"""
        # Calculate target area rewards
        reward, done = self._calculate_box_push_reward()
        individual_rewards = [reward for _ in range(self.n_agents)]

        return reward, individual_rewards, done

    def _calculate_box_push_reward(self):
        """Calculate reward for getting box closer to goal"""
        # Calculate target area rewards
        task_reward = 0.0
        done = False

        # 1. Check if we need to initialize previous distances (first step or reset)
        if not hasattr(self, "prev_object_distances"):
            self.prev_object_distances = (
                {}
            )  # Map object_id -> min_distance_to_any_target

        if not hasattr(self, "delivered_objects"):
            self.delivered_objects = set()

        current_object_distances = {}
        shaping_reward = 0.0
        completion_reward = 0.0

        # 2. Iterate over all movable objects
        for obj_idx, obj in enumerate(self.objects):

            # Skip objects that have already been delivered
            if obj_idx in self.delivered_objects:
                continue

            target = self.target_areas[0]
            dy = target.y - obj.position.y
            dist = dy

            # Store current distance
            current_object_distances[obj_idx] = dist

            # 3. Calculate reward based on improvement (shaping reward)
            if obj_idx in self.prev_object_distances:
                prev_dist = self.prev_object_distances[obj_idx]
                improvement = prev_dist - dist

                # If improvement is positive (getting closer), give reward
                # If negative (getting farther), give penalty
                # Scaling factor 10.0 makes the signal stronger
                shaping_reward += improvement * 1.0

            # 4. Check for completion (object inside target) — bonus only once
            if target.contains_object(obj):
                completion_reward += 100.0
                self.delivered_objects.add(obj_idx)

        # Update previous distances for next step (only for non-delivered objects)
        self.prev_object_distances = current_object_distances

        # Terminate only when ALL objects have been delivered
        if len(self.delivered_objects) == len(self.objects):
            done = True

        # Dense mode keeps the shaping term; sparse mode only pays on delivery.
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
            "agents_2_objects": self.get_agents_grabbing_objects(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset step counter
        self.current_step = 0

        # Reset distance tracking for rewards
        if hasattr(self, "prev_object_distances"):
            del self.prev_object_distances

        # Reset delivered tracking
        self.delivered_objects = set()

        # Create a completely fresh Box2D world to avoid stale references
        self.world = b2World(gravity=(0, 0))

        # Re-attach contact listener to new world
        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        # Clear all body references. Joints belong to the old world and are
        # discarded with it; just drop the tracking entries.
        self.agents.clear()
        self.objects.clear()
        self.boundary_bodies.clear()
        self.agent_joints = [None] * self.n_agents

        # Recreate everything in the fresh world
        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        self._init_agents()
        self._create_dynamic_objects()
        self._create_target_areas()

        obs = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return obs, self._get_info()

    def step(self, actions):

        # PREPROCESS ENVIRONMENT ACTION

        movement_action = actions[:, :2]
        grab_action = actions[:, 2]

        # PROCESS ENVIRONMENT ACTION

        # Resolve grab/release before stepping so new joints participate in
        # this tick's physics integration.
        self._process_grabs(grab_action)

        # Apply movement forces
        force_multiplier = 100.0
        for agent in self.agents:
            force_x = np.clip(movement_action[agent.index][0], -1, 1) * force_multiplier
            force_y = np.clip(movement_action[agent.index][1], -1, 1) * force_multiplier

            self.applied_forces[agent.index] = [force_x, force_y]
            agent.apply_force(force_x, force_y)

        # Object is only movable when enough agents have it grabbed.
        self._update_object_mass_from_grabs()

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
            task_reward, individual_rewards, terminated = self._get_rewards()

        # Get observation BEFORE resetting contacts
        obs = self._get_observation()

        # Reset collision flag for next step (AFTER observation and rewards)
        self.contact_listener.reset()

        info = self._get_info(task_reward=task_reward)

        self.current_step += 1

        truncated = self.current_step >= self.max_steps

        # The observation
        return obs, task_reward, terminated, truncated, info

    def render(self):
        self.renderer.render()

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    # Create the environment with rendering
    env = MultiBoxGrabEnv(render_mode="human", n_agents=12, n_objects=6, max_steps=10e6)
    obs, info = env.reset()

    running = True
    current_agent_idx = 0
    cum_rew = 0
    group_control = False  # Toggle for group replication
    grab_states = np.zeros(env.n_agents, dtype=np.float32)

    print("\n" + "=" * 50)
    print(" MULTI-BOX GRAB ENVIRONMENT DEBUGGER")
    print("=" * 50)
    print(f" Controlling Agent: {current_agent_idx}")
    print(" Controls:")
    print("  [ARROWS] : Move Active Agent")
    print("  [SPACE]  : Switch Active Agent")
    print("  [F]      : Toggle Grab for Active Agent")
    print("  [G]      : Toggle Group Control (Radius 5)")
    print("  [ESC]    : Quit")
    print("=" * 50 + "\n")

    entropy_log = []  # list of [S_e, S_normalized] per step

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
                elif event.key == pygame.K_f:
                    grab_states[current_agent_idx] = (
                        0.0 if grab_states[current_agent_idx] >= 0.5 else 1.0
                    )
                    state = (
                        "GRAB" if grab_states[current_agent_idx] >= 0.5 else "RELEASE"
                    )
                    print(f">>> Agent {current_agent_idx} {state}")

        # Get continuous key presses
        keys = pygame.key.get_pressed()

        # Action layout: [force_x, force_y, grab]
        actions = np.zeros((env.n_agents, 3), dtype=np.float32)
        actions[:, 2] = grab_states

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

        # Hypergraph testing
        obs = np.expand_dims(obs, axis=0)
        # hypergraphs = build_hypergraph(
        #     1, env.n_agents, obs, partial(distance_based_hyperedges, threshold=1.0)
        # )
        hypergraphs = build_hypergraph(
            1, env.n_agents, [info["agents_2_objects"]], object_contact_hyperedges
        )
        entropies = compute_hyperedge_structural_entropy_batch(hypergraphs)
        entropy_log.append(entropies[0])  # [S_e, S_normalized] for this step

        cum_rew += reward

        # 3. Log info nicely
        # Format observation array to be compact
        obs_str = np.array2string(
            obs[0, current_agent_idx],
            precision=2,
            suppress_small=True,
            separator=",",
            floatmode="fixed",
            max_line_width=np.inf,
        )

        # print(
        #     f"{env.current_step:<5d} | "
        #     f"{current_agent_idx:<3d} | "
        #     f"({force_x:>4.1f}, {force_y:>4.1f})  | "
        #     f"{reward:<8.4f} | "
        #     f"{obs_str}"
        # )

        print(f"Cumulative Rew {cum_rew} Rew {reward}")

        # 4. Render
        env.render()

        # 5. Handle reset
        if terminated or truncated:
            print(">>> Environment Reset")
            cum_rew = 0
            break
            obs, info = env.reset()

    env.close()

    entropy_array = np.array(entropy_log)  # (n_steps, 2)
    steps = np.arange(len(entropy_array))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(steps, entropy_array[:, 0])
    axes[0].set_ylabel("$S_e$ (nats)")
    axes[0].set_title("Hyperedge Structural Entropy")

    axes[1].plot(steps, entropy_array[:, 1])
    axes[1].set_ylabel("$S_{\\mathrm{norm}}$")
    axes[1].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig("entropy.png")
