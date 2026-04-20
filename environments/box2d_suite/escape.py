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
)

from Box2D import (
    b2World,
    b2PolygonShape,
    b2RevoluteJointDef,
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
    UnionFind,
)


class EscapeRenderer(Renderer):
    """Renderer extension that draws the annulus push zone and inter-agent joints."""

    def render(self):
        if self.env.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Escape Simulation")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        self._draw_boundary_walls()
        self._draw_annulus()
        self._render_agents_as_circles()
        self._draw_joints()
        self._draw_agent_indices()

        pygame.display.flip()
        self.clock.tick(self.env.metadata["render_fps"])

    def _draw_annulus(self):
        sh = self.screen_size[1]
        cx = int(self.env.world_center_x * self.scale)
        cy = int(sh - self.env.world_center_y * self.scale)
        inner_px = max(1, int(self.env.annulus_inner_radius * self.scale))
        outer_px = max(inner_px + 1, int(self.env.annulus_outer_radius * self.scale))
        spawn_px = max(1, int(self.env.spawn_radius * self.scale))
        ring_thickness = outer_px - inner_px

        size = outer_px * 2 + 4
        ring_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(
            ring_surface,
            (220, 120, 120, 90),
            (size // 2, size // 2),
            outer_px,
            ring_thickness,
        )
        self.screen.blit(ring_surface, (cx - size // 2, cy - size // 2))

        pygame.draw.circle(self.screen, (150, 0, 0), (cx, cy), outer_px, 2)
        pygame.draw.circle(self.screen, (150, 0, 0), (cx, cy), inner_px, 2)
        pygame.draw.circle(self.screen, (0, 150, 0), (cx, cy), spawn_px, 1)

    def _draw_joints(self):
        sh = self.screen_size[1]
        for joint in self.env.joints:
            ax, ay = joint.anchorA
            bx, by = joint.anchorB
            p1 = (int(ax * self.scale), int(sh - ay * self.scale))
            p2 = (int(bx * self.scale), int(sh - by * self.scale))
            pygame.draw.line(self.screen, (0, 0, 0), p1, p2, 3)
            pygame.draw.circle(self.screen, (255, 0, 0), p1, 4)
            pygame.draw.circle(self.screen, (0, 0, 255), p2, 4)


class EscapeEnv(gym.Env):
    metadata = {"render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        n_agents=4,
        max_steps=512,
        min_group_size=2,
        max_joints_per_agent=2,
        join_distance=1.5,
        annulus_push_force=150.0,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.render_mode = render_mode
        self.target_areas = []

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, 4),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, 21), dtype=np.float32
        )

        self.world = b2World(gravity=(0, 0))
        self.time_step = 1.0 / 60.0

        self.agents = []
        self.joints = []

        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        _total_entities = self.n_agents
        self.world_width = int(30 * max(1.0, _total_entities / 8) ** 0.5)
        self.world_height = self.world_width
        self.world_center_x = self.world_width / 2
        self.world_center_y = self.world_height / 2
        self.world_diagonal = np.sqrt(self.world_width**2 + self.world_height**2)
        self.boundary_thickness = 0.5

        self.spawn_radius = self.world_width * 0.10
        self.annulus_inner_radius = self.world_width * 0.18
        self.annulus_outer_radius = self.world_width * 0.32
        self.annulus_push_force = annulus_push_force

        self.boundary_bodies = []

        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        self._init_agents()

        self.applied_forces = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.force_scale = 2.0

        self.velocity_norm = self.world_width / 10.0
        self.sector_sensor_radius = self.world_width / 3.0

        self.max_joints_per_agent = max_joints_per_agent
        self.min_group_size = min_group_size
        self.join_distance = join_distance

        self.union_find = UnionFind(self.n_agents)
        self.attach_values = np.ones(self.n_agents, dtype=np.int8)
        self.detach_values = np.zeros(self.n_agents, dtype=np.int8)

        self._escaped = np.zeros(self.n_agents, dtype=bool)

        self.max_steps = max_steps
        self.current_step = 0

        self.objects = []

        self.observation_manager = ObservationManager(self)
        self.renderer = EscapeRenderer(self)

    def _init_agents(self):
        rng = self.np_random
        cx = self.world_width / 2
        cy = self.world_height / 2
        min_distance = 1.0
        positions = []

        for _ in range(self.n_agents):
            for _ in range(200):
                angle = rng.uniform(0.0, 2.0 * np.pi)
                r = self.spawn_radius * np.sqrt(rng.uniform(0.0, 1.0))
                candidate = (cx + r * np.cos(angle), cy + r * np.sin(angle))
                if all(
                    np.hypot(candidate[0] - px, candidate[1] - py) >= min_distance
                    for px, py in positions
                ):
                    positions.append(candidate)
                    break
            else:
                angle = 2.0 * np.pi * len(positions) / max(self.n_agents, 1)
                positions.append(
                    (
                        cx + 0.5 * np.cos(angle),
                        cy + 0.5 * np.sin(angle),
                    )
                )

        self._create_agents(positions)

    def _create_agents(self, positions):
        for i in range(self.n_agents):
            agent = Agent(self.world, positions[i], index=i)
            self.agents.append(agent)

    def _create_boundary(self, width, height, thickness):
        bottom_wall = self.world.CreateStaticBody(
            position=(width / 2, thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        bottom_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        bottom_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
        self.boundary_bodies.append(bottom_wall)

        top_wall = self.world.CreateStaticBody(
            position=(width / 2, height - thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        top_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        top_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
        self.boundary_bodies.append(top_wall)

        left_wall = self.world.CreateStaticBody(
            position=(thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        left_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        left_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
        self.boundary_bodies.append(left_wall)

        right_wall = self.world.CreateStaticBody(
            position=(width - thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        right_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        right_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
        self.boundary_bodies.append(right_wall)

    def _create_joint(self, bodyA, bodyB):
        anchor = (bodyA.position + bodyB.position) / 2
        joint_def = b2RevoluteJointDef(
            bodyA=bodyA, bodyB=bodyB, anchor=anchor, collideConnected=False
        )
        joint = self.world.CreateJoint(joint_def)
        self.joints.append(joint)
        return joint

    def _break_joint(self, joint):
        self.world.DestroyJoint(joint)
        self.joints.remove(joint)
        self._update_union_find()

    def _update_union_find(self):
        self.union_find = UnionFind(self.n_agents)
        for joint in self.joints:
            idx_a = joint.bodyA.userData["index"]
            idx_b = joint.bodyB.userData["index"]
            self.union_find.union(idx_a, idx_b)

    def _count_joints_for_agent(self, body):
        return sum(
            1 for joint in self.joints if joint.bodyA is body or joint.bodyB is body
        )

    def _join_on_proximity(self):
        self._update_union_find()
        for i, agentA in enumerate(self.agents):
            if self._count_joints_for_agent(agentA.body) >= self.max_joints_per_agent:
                continue
            if self.attach_values[i] == 0:
                continue
            for j in range(i + 1, self.n_agents):
                agentB = self.agents[j]
                if (
                    self._count_joints_for_agent(agentB.body)
                    >= self.max_joints_per_agent
                ):
                    continue
                if self.attach_values[j] == 0:
                    continue
                if self.union_find.connected(i, j):
                    continue
                dist = (agentA.position - agentB.position).length
                if dist < self.join_distance:
                    self._create_joint(agentA.body, agentB.body)
                    self.union_find.union(i, j)
                    break

    def _process_detachments(self):
        joints_to_remove = []
        for joint in self.joints:
            idx_a = joint.bodyA.userData["index"]
            idx_b = joint.bodyB.userData["index"]
            if self.detach_values[idx_a] and self.detach_values[idx_b]:
                joints_to_remove.append(joint)
        for joint in joints_to_remove:
            self._break_joint(joint)

    def _component_sizes(self):
        roots = np.array(
            [self.union_find.find(i) for i in range(self.n_agents)], dtype=np.int32
        )
        _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
        return counts[inverse]

    def _apply_annulus_push(self):
        cx = self.world_center_x
        cy = self.world_center_y
        comp_sizes = self._component_sizes()

        for i, agent in enumerate(self.agents):
            if comp_sizes[i] >= self.min_group_size:
                continue

            pos = agent.position
            dx = cx - pos.x
            dy = cy - pos.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            if self.annulus_inner_radius <= dist <= self.annulus_outer_radius:
                fx = self.annulus_push_force * dx / dist
                fy = self.annulus_push_force * dy / dist
                agent.apply_force(fx, fy)

    def _get_observation(self):
        return self.observation_manager.get_observation()

    def _get_info(self, task_reward=0.0):
        return {
            "agent_positions": [
                {"x": ag.position.x, "y": ag.position.y} for ag in self.agents
            ],
            "n_escaped": int(self._escaped.sum()),
            "task_reward": task_reward,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.world = b2World(gravity=(0, 0))
        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        self.agents.clear()
        self.objects.clear()
        self.boundary_bodies.clear()
        self.joints.clear()

        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        self._init_agents()

        self.attach_values = np.ones(self.n_agents, dtype=np.int8)
        self.detach_values = np.zeros(self.n_agents, dtype=np.int8)
        self.union_find = UnionFind(self.n_agents)
        self._escaped = np.zeros(self.n_agents, dtype=bool)

        obs = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return obs, self._get_info()

    def step(self, actions):
        movement_action = actions[:, :2]
        attach_action = actions[:, 2]
        detach_action = actions[:, 3]

        self.attach_values = (np.asarray(attach_action).flatten() > 0.5).astype(np.int8)
        self.detach_values = (np.asarray(detach_action).flatten() > 0.5).astype(np.int8)

        self._process_detachments()

        force_multiplier = 100.0
        for agent in self.agents:
            fx = np.clip(movement_action[agent.index][0], -1, 1) * force_multiplier
            fy = np.clip(movement_action[agent.index][1], -1, 1) * force_multiplier
            self.applied_forces[agent.index] = [fx, fy]
            agent.apply_force(fx, fy)

        self._apply_annulus_push()

        self.world.Step(self.time_step, 6, 2)

        self._join_on_proximity()

        task_reward = 0.0
        terminated = False

        if self.contact_listener.boundary_collision:
            terminated = True
        else:
            prev_escaped = self._escaped.copy()
            for i, ag in enumerate(self.agents):
                d = math.hypot(
                    ag.position.x - self.world_center_x,
                    ag.position.y - self.world_center_y,
                )
                if d > self.annulus_outer_radius:
                    self._escaped[i] = True
            new_escapees = int(np.sum(self._escaped & ~prev_escaped))
            task_reward = float(new_escapees)
            terminated = bool(self._escaped.all())

        obs = self._get_observation()
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
    env = EscapeEnv(
        render_mode="human",
        n_agents=6,
        max_steps=1024,
        min_group_size=3,
    )
    obs, info = env.reset()

    running = True
    current_agent_idx = 0
    cum_rew = 0
    group_control = False

    print("\n" + "=" * 50)
    print(" ESCAPE ENVIRONMENT DEBUGGER")
    print("=" * 50)
    print(f" Controlling Agent: {current_agent_idx}")
    print(" Controls:")
    print("  [ARROWS] : Move Active Agent")
    print("  [SPACE]  : Switch Active Agent")
    print("  [G]      : Toggle Group Control (Radius 5)")
    print("  [D]      : Toggle Detach Signal for Active Agent")
    print("  [ESC]    : Quit")
    print("=" * 50 + "\n")

    detach_toggle = np.zeros(env.n_agents, dtype=np.float32)

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
                    print(f">>> Group Control {status}")
                elif event.key == pygame.K_d:
                    detach_toggle[current_agent_idx] = (
                        1.0 - detach_toggle[current_agent_idx]
                    )
                    print(
                        f">>> Agent {current_agent_idx} detach = "
                        f"{detach_toggle[current_agent_idx]}"
                    )

        keys = pygame.key.get_pressed()

        actions = np.zeros((env.n_agents, 4), dtype=np.float32)
        actions[:, 2] = 1.0
        actions[:, 3] = detach_toggle

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
                distance = math.hypot(
                    current_pos.x - other_pos.x,
                    current_pos.y - other_pos.y,
                )
                if distance <= radius:
                    actions[i, 0] = force_x
                    actions[i, 1] = force_y

        obs, reward, terminated, truncated, info = env.step(actions)

        cum_rew += reward

        print(
            f"step {env.current_step:<4d} | escaped {info['n_escaped']}/{env.n_agents}"
            f" | rew {reward:.1f} | cum {cum_rew:.1f}"
        )

        env.render()

        if terminated or truncated:
            print(">>> Environment Reset")
            cum_rew = 0
            break

    env.close()
