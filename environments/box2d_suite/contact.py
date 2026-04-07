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
    b2CircleShape,
    b2FixtureDef,
)

from environments.box2d_suite.agent import Agent
from environments.box2d_suite.observation import ObservationManager
from environments.box2d_suite.renderer import Renderer
from environments.box2d_suite.utils import (
    COLORS_LIST,
    AGENT_CATEGORY,
    BOUNDARY_CATEGORY,
    OBJECT_CATEGORY,
    BoundaryContactListener,
)


class ContactEnv(gym.Env):
    """Multi-agent environment where agents must find and touch a random static object.

    Each episode spawns a single immovable object (box, circle, or triangle) at a
    random position. Agents are rewarded for making contact and penalised for
    losing contact.
    """

    metadata = {"render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        n_agents=3,
        max_steps=512,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.render_mode = render_mode

        self.target_areas = []

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, 2),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, 21), dtype=np.float32
        )

        self.world = b2World(gravity=(0, 0))
        self.time_step = 1.0 / 60.0

        self.agents = []

        # Contact listener
        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        # Auto-scale world size based on entity count
        _total_entities = self.n_agents + 1
        self.world_width = int(30 * max(1.0, _total_entities / 8) ** 0.5)
        self.world_height = self.world_width
        self.world_center_x = self.world_width // 2
        self.world_center_y = self.world_height // 2
        self.world_diagonal = np.sqrt(self.world_height**2 + self.world_width**2)
        self.boundary_thickness = 0.5

        self.boundary_bodies = []

        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )

        self._init_agents()

        # Force tracking
        self.applied_forces = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.force_scale = 2.0

        self.velocity_norm = self.world_width / 10.0
        self.sector_sensor_radius = self.world_width / 3.0

        self.max_steps = max_steps
        self.current_step = 0

        self.objects = []

        # Reward state
        self.prev_n_touching = 0

        # Renderer compatibility (no attachment mechanic — show agents as green)
        self.attach_values = np.ones(self.n_agents, dtype=np.int8)
        self.detach_values = np.zeros(self.n_agents, dtype=np.int8)

        self.observation_manager = ObservationManager(self)
        self.renderer = Renderer(self)

    def _init_agents(self):
        rng = self.np_random
        center_x = self.world_width / 2
        center_y = self.world_height / 2
        spawn_radius = max(2.0, min(self.world_width, self.world_height) * 0.15)
        min_distance = 2.0
        positions = []

        for _ in range(self.n_agents):
            for _ in range(100):
                angle = rng.uniform(0.0, 2.0 * np.pi)
                radius = spawn_radius * np.sqrt(rng.uniform(0.0, 8.0))
                candidate = (
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle),
                )

                if all(
                    np.hypot(candidate[0] - px, candidate[1] - py) >= min_distance
                    for px, py in positions
                ):
                    positions.append(candidate)
                    break
            else:
                angle = 2.0 * np.pi * len(positions) / max(self.n_agents, 1)
                radius = min(spawn_radius, min_distance * np.ceil(len(positions) / 6))
                positions.append(
                    (
                        center_x + radius * np.cos(angle),
                        center_y + radius * np.sin(angle),
                    )
                )

        self._create_agents(positions)

    def _create_agents(self, positions):
        for i in range(self.n_agents):
            agent = Agent(self.world, positions[i], index=i)
            self.agents.append(agent)

    def _create_boundary(self, width, height, thickness):
        """Create boundary walls that agents can collide with."""
        walls = [
            ((width / 2, thickness / 2), (width / 2, thickness / 2)),
            ((width / 2, height - thickness / 2), (width / 2, thickness / 2)),
            ((thickness / 2, height / 2), (thickness / 2, height / 2)),
            ((width - thickness / 2, height / 2), (thickness / 2, height / 2)),
        ]
        for pos, half in walls:
            wall = self.world.CreateStaticBody(
                position=pos,
                shapes=b2PolygonShape(box=half),
            )
            wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
            wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
            self.boundary_bodies.append(wall)

    def _create_object(self):
        """Spawn a single random-shaped static (immovable) object at a random position."""
        self.objects.clear()

        margin = 3.0
        pos_x = self.np_random.uniform(margin, self.world_width - margin)
        pos_y = self.np_random.uniform(margin, self.world_height - margin)

        shape_type = self.np_random.choice(["box", "circle", "triangle"])
        # Scale object so more agents can physically surround it
        # Base size 1.5 for 3 agents, grows with sqrt(n_agents)
        size = 1.4 * (self.n_agents / 3) ** 0.5

        if shape_type == "box":
            shape = b2PolygonShape(box=(size, size))
        elif shape_type == "circle":
            shape = b2CircleShape(radius=size)
        else:  # triangle
            shape = b2PolygonShape(vertices=[(-size, -size), (size, -size), (0, size)])

        fixture_def = b2FixtureDef(shape=shape)
        fixture_def.filter.categoryBits = OBJECT_CATEGORY
        fixture_def.filter.maskBits = AGENT_CATEGORY | BOUNDARY_CATEGORY

        body = self.world.CreateStaticBody(
            position=(pos_x, pos_y),
            fixtures=fixture_def,
        )

        body.userData = {
            "type": "object",
            "color": COLORS_LIST[self.n_agents % len(COLORS_LIST)],
            "index": 0,
            "shape_type": shape_type,
            "coupling": self.n_agents,  # displayed on object by renderer
        }

        self.objects.append(body)

    def _count_agents_touching_object(self):
        """Count how many agents are within contact distance of the object."""
        if not self.objects:
            return 0

        obj = self.objects[0]
        obj_pos = np.array([obj.position.x, obj.position.y])
        count = 0

        for agent in self.agents:
            agent_pos = np.array([agent.position.x, agent.position.y])
            dist = self.observation_manager._agent_object_distance(
                agent_pos, obj, obj_pos
            )
            if dist <= agent.radius + 0.2:
                count += 1

        return count

    def _get_observation(self):
        return self.observation_manager.get_observation()

    def _calculate_reward(self):
        n_touching = self._count_agents_touching_object()

        # Delta reward: +1 per new agent making contact, -1 per agent losing contact
        reward = float(n_touching - self.prev_n_touching)
        self.prev_n_touching = n_touching

        # Episode succeeds when every agent is touching the object
        done = n_touching == self.n_agents

        return reward, done

    def _get_info(self):
        return {
            "agent_positions": [
                {"x": ag.position.x, "y": ag.position.y} for ag in self.agents
            ],
            "n_touching": self.prev_n_touching,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_n_touching = 0

        # Fresh Box2D world
        self.world = b2World(gravity=(0, 0))
        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        self.agents.clear()
        self.objects.clear()
        self.boundary_bodies.clear()

        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        self._init_agents()
        self._create_object()

        obs = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return obs, self._get_info()

    def step(self, actions):
        movement_action = actions[:, :2]

        force_multiplier = 100.0
        for agent in self.agents:
            force_x = np.clip(movement_action[agent.index][0], -1, 1) * force_multiplier
            force_y = np.clip(movement_action[agent.index][1], -1, 1) * force_multiplier

            self.applied_forces[agent.index] = [force_x, force_y]
            agent.apply_force(force_x, force_y)

        self.world.Step(self.time_step, 6, 2)

        reward = 0.0
        terminated = False

        if self.contact_listener.boundary_collision:
            terminated = True
        else:
            reward, terminated = self._calculate_reward()

        obs = self._get_observation()

        self.contact_listener.reset()

        info = self._get_info()

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self):
        self.renderer.render()

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    env = ContactEnv(render_mode="human", n_agents=12, max_steps=512)
    obs, info = env.reset()

    running = True
    current_agent_idx = 0
    cum_rew = 0
    group_control = False

    print("\n" + "=" * 50)
    print(" CONTACT ENVIRONMENT DEBUGGER")
    print("=" * 50)
    print(f" Controlling Agent: {current_agent_idx}")
    print(" Controls:")
    print("  [ARROWS] : Move Active Agent")
    print("  [SPACE]  : Switch Active Agent")
    print("  [G]      : Toggle Group Control (Radius 5)")
    print("  [ESC]    : Quit")
    print("=" * 50 + "\n")

    entropy_log = []

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

        # Hypergraph testing
        obs = np.expand_dims(obs, axis=0)
        hypergraphs = build_hypergraph(
            1, env.n_agents, obs, partial(distance_based_hyperedges, threshold=1.0)
        )

        entropies = compute_hyperedge_structural_entropy_batch(hypergraphs)
        entropy_log.append(entropies[0])

        cum_rew += reward

        print(
            f"Cumulative Rew {cum_rew:.2f}  Rew {reward:.2f}  "
            f"Touching {info['n_touching']}/{env.n_agents}"
        )

        env.render()

        if terminated or truncated:
            print(">>> Environment Reset")
            cum_rew = 0
            break

    env.close()

    entropy_array = np.array(entropy_log)
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
