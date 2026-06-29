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
)

from environments.box2d_suite.agent import Agent
from environments.box2d_suite.observation import ObservationManager, OBS_DIM
from environments.box2d_suite.renderer import Renderer
from environments.box2d_suite.utils import (
    AGENT_CATEGORY,
    BOUNDARY_CATEGORY,
    OBJECT_CATEGORY,
    BoundaryContactListener,
    UnionFind,
)


class MultiPassEnv(gym.Env):
    """Obstacle-course environment.

    Agents spawn at the bottom of the map and must reach the top. The course
    has ``n_walls`` horizontal walls, each pierced by a fixed number of gaps
    (``gaps_per_wall[wall_index]``). Each gap exerts a downward force on
    every agent inside it unless the number of agents currently in that
    gap meets the gap's coupling requirement (``n_agents / n_gaps_in_wall``).
    """

    metadata = {"render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        n_agents: int = 8,
        n_walls: int = 3,
        gaps_per_wall=None,
        max_steps: int = 1024,
        reward_mode: str = "dense",
    ):
        super().__init__()

        if gaps_per_wall is None:
            gaps_per_wall = [2] * n_walls
        assert (
            len(gaps_per_wall) == n_walls
        ), "gaps_per_wall must contain one entry per wall"
        assert all(g >= 1 for g in gaps_per_wall), "every wall needs at least one gap"

        self.n_agents = n_agents
        self.n_walls = n_walls
        self.gaps_per_wall = list(gaps_per_wall)
        # Coupling is proportional to agents vs. gap count on that wall.
        self.wall_couplings = [max(1, n_agents // g) for g in self.gaps_per_wall]

        self.render_mode = render_mode
        self.reward_mode = reward_mode

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, 2),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, OBS_DIM),
            dtype=np.float32,
        )

        self.world = b2World(gravity=(0, 0))
        self.time_step = 1.0 / 60.0

        self.agents = []
        # ObservationManager / Renderer expect these collections to exist.
        self.objects = []
        self.target_areas = []

        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        # Square world; grow with the number of walls so the course has room.
        world_size = max(30, 6 * (self.n_walls + 2))
        self.world_width = world_size
        self.world_height = world_size
        self.world_center_x = self.world_width // 2
        self.world_center_y = self.world_height // 2
        self.world_diagonal = np.sqrt(self.world_height**2 + self.world_width**2)
        self.boundary_thickness = 0.5
        self.wall_thickness = 1.0
        self.gap_width = 3.0
        # Downward push applied per-agent inside an under-coupled gap.
        self.gap_force = 250.0
        # Padding around the wall thickness defining the gap influence zone.
        # Set to 0 so the gap zone height matches the wall thickness exactly.
        self.gap_region_pad = 0.0
        # Agents within this distance are considered part of the same cluster
        # (transitively). Gap coupling is then checked against cluster size,
        # not the literal count of agents inside the gap zone.
        self.cluster_radius = 2.0

        self.boundary_bodies = []
        self.wall_bodies = []
        self.walls = []  # metadata per wall: y, gaps, coupling, etc.

        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        self._create_walls()
        self._init_agents()

        self.applied_forces = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.force_scale = 2.0
        self.force_multiplier = 100.0
        self.agent_contact_forces = np.zeros(self.n_agents, dtype=np.float32)

        self.velocity_norm = self.world_width / 10.0
        self.neighbor_detection_range = 3.0

        self.max_steps = max_steps
        self.current_step = 0

        self.prev_mean_y = None

        self.observation_manager = ObservationManager(self)
        self.renderer = Renderer(self)

    def _spawn_band(self):
        """Return (y_min, y_max) of the bottom spawn band."""
        band_height = self.world_height / (self.n_walls * 5)
        return self.boundary_thickness, self.boundary_thickness + band_height

    def _create_walls(self):
        """Create n_walls horizontal walls with evenly-spaced gaps."""
        self.wall_bodies.clear()
        self.walls.clear()

        _, spawn_y_max = self._spawn_band()
        # Walls span the vertical region between the top of the spawn band and
        # a top margin of similar height.
        top_margin = self.world_height - (self.world_height / (self.n_walls * 5))
        usable = top_margin - spawn_y_max

        inner_left = self.boundary_thickness
        inner_right = self.world_width - self.boundary_thickness
        inner_w = inner_right - inner_left

        # Agent diameter (used to scale gap width with coupling).
        agent_diameter = 0.8

        for w in range(self.n_walls):
            wall_y = spawn_y_max + usable * (w + 1) / (self.n_walls + 1)
            n_gaps = self.gaps_per_wall[w]
            coupling = self.wall_couplings[w]

            gap_centers = [
                inner_left + inner_w * (i + 1) / (n_gaps + 1) for i in range(n_gaps)
            ]
            # Scale gap width with coupling so the required cluster fits
            # through side-by-side, with a minimum of self.gap_width.
            gap_w = max(self.gap_width, coupling * agent_diameter * 1.5)
            # Cap so the wall still has solid segments between gaps.
            max_total_gap = inner_w - 1.0  # leave at least 1 unit of wall total
            gap_w = min(gap_w, max_total_gap / n_gaps)

            gaps = [
                {
                    "x_center": c,
                    "width": gap_w,
                    "coupling": coupling,
                    "crossed": False,
                }
                for c in gap_centers
            ]

            # Compute solid wall segment bounds between gaps.
            edges = [inner_left]
            for c in gap_centers:
                edges.extend([c - gap_w / 2, c + gap_w / 2])
            edges.append(inner_right)

            segments = []
            for k in range(0, len(edges), 2):
                x_left, x_right = edges[k], edges[k + 1]
                if x_right - x_left <= 1e-3:
                    continue
                seg_half_w = (x_right - x_left) / 2
                seg_cx = (x_left + x_right) / 2
                body = self.world.CreateStaticBody(
                    position=(seg_cx, wall_y),
                    shapes=b2PolygonShape(box=(seg_half_w, self.wall_thickness / 2)),
                )
                # OBJECT_CATEGORY so collisions don't trigger boundary handling.
                body.fixtures[0].filterData.categoryBits = OBJECT_CATEGORY
                body.fixtures[0].filterData.maskBits = AGENT_CATEGORY
                body.userData = {"type": "wall", "wall_index": w}
                segments.append(body)
                self.wall_bodies.append(body)

            self.walls.append(
                {
                    "y": wall_y,
                    "gaps": gaps,
                    "coupling": coupling,
                    "n_gaps": n_gaps,
                    "segments": segments,
                    "thickness": self.wall_thickness,
                }
            )

    def _init_agents(self):
        positions = self._get_spawn_positions()
        for i in range(self.n_agents):
            self.agents.append(Agent(self.world, positions[i], index=i))

    def _get_spawn_positions(self):
        """Random non-overlapping positions inside the bottom spawn band."""
        margin = 1.5
        spawn_y_min, spawn_y_max = self._spawn_band()
        x_min = self.boundary_thickness + margin
        x_max = self.world_width - self.boundary_thickness - margin
        y_min = spawn_y_min + margin
        y_max = spawn_y_max - margin
        min_distance = 1.5
        max_attempts = 200

        positions = []
        for i in range(self.n_agents):
            placed = False
            for _ in range(max_attempts):
                pos_x = np.random.uniform(x_min, x_max)
                pos_y = np.random.uniform(y_min, y_max)
                if all(
                    np.hypot(pos_x - p[0], pos_y - p[1]) >= min_distance
                    for p in positions
                ):
                    positions.append((pos_x, pos_y))
                    placed = True
                    break
            if not placed:
                positions.append(
                    (
                        x_min + (i + 0.5) * (x_max - x_min) / self.n_agents,
                        (y_min + y_max) / 2,
                    )
                )
        return positions

    def _create_boundary(self, width, height, thickness):
        """Create boundary walls that agents collide with."""
        specs = [
            ((width / 2, thickness / 2), (width / 2, thickness / 2)),  # bottom
            ((width / 2, height - thickness / 2), (width / 2, thickness / 2)),  # top
            ((thickness / 2, height / 2), (thickness / 2, height / 2)),  # left
            ((width - thickness / 2, height / 2), (thickness / 2, height / 2)),  # right
        ]
        for pos, box in specs:
            body = self.world.CreateStaticBody(
                position=pos, shapes=b2PolygonShape(box=box)
            )
            body.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
            body.fixtures[0].filterData.maskBits = AGENT_CATEGORY | OBJECT_CATEGORY
            self.boundary_bodies.append(body)

    def _compute_clusters(self, agent_pos):
        """Group agents into clusters via pairwise distance <= cluster_radius.

        Returns ``(cluster_id, cluster_size)`` where ``cluster_id[i]`` is the
        root index of agent ``i`` and ``cluster_size[root]`` is the size of
        that cluster.
        """
        n = self.n_agents
        uf = UnionFind(n)
        if n > 1:
            diff = agent_pos[:, np.newaxis, :] - agent_pos[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            i_idx, j_idx = np.where(np.triu(dist <= self.cluster_radius, k=1))
            for i, j in zip(i_idx.tolist(), j_idx.tolist()):
                uf.union(i, j)
        cluster_id = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
        cluster_size = {}
        for cid in cluster_id.tolist():
            cluster_size[cid] = cluster_size.get(cid, 0) + 1
        return cluster_id, cluster_size

    def _apply_gap_forces(self):
        """Push agents whose cluster is too small to meet the gap coupling."""
        if self.n_agents == 0:
            return

        agent_pos = np.array(
            [[a.position.x, a.position.y] for a in self.agents], dtype=np.float32
        )
        half_t = self.wall_thickness / 2 + self.gap_region_pad
        cluster_id, cluster_size = self._compute_clusters(agent_pos)

        for wall in self.walls:
            wy = wall["y"]
            in_y = (agent_pos[:, 1] >= wy - half_t) & (agent_pos[:, 1] <= wy + half_t)
            if not in_y.any():
                continue
            for gap in wall["gaps"]:
                # Once a qualifying cluster has broken through, the gap stays
                # open — otherwise trailing teammates get stranded behind by
                # the very force their cluster just defeated.
                if gap["crossed"]:
                    continue
                half_w = gap["width"] / 2
                in_x = (agent_pos[:, 0] >= gap["x_center"] - half_w) & (
                    agent_pos[:, 0] <= gap["x_center"] + half_w
                )
                inside_idx = np.where(in_x & in_y)[0]
                if inside_idx.size == 0:
                    continue
                for idx in inside_idx.tolist():
                    cid = int(cluster_id[idx])
                    if cluster_size[cid] < gap["coupling"]:
                        self.agents[idx].apply_force(0.0, -self.gap_force)

    def _get_observation(self):
        return self.observation_manager.get_observation()

    def _get_rewards(self):
        reward, done = self._calculate_pass_reward()
        individual_rewards = [reward for _ in range(self.n_agents)]
        return reward, individual_rewards, done

    def _calculate_pass_reward(self):
        agent_pos = np.array(
            [[a.position.x, a.position.y] for a in self.agents], dtype=np.float32
        )
        agent_ys = agent_pos[:, 1]
        mean_y = float(agent_ys.mean())

        shaping_reward = 0.0
        if self.prev_mean_y is not None:
            shaping_reward = mean_y - self.prev_mean_y
        self.prev_mean_y = mean_y

        cluster_id, cluster_size = self._compute_clusters(agent_pos)

        # Crossing bonus: 100 split evenly across the gaps of each wall, paid
        # once per gap when a cluster of size >= coupling has any member above
        # the wall inside the gap's x range.
        crossing_bonus = 0.0
        all_crossed = True
        for wall in self.walls:
            wy = wall["y"]
            per_gap_bonus = 100.0 / wall["n_gaps"]
            for gap in wall["gaps"]:
                if gap["crossed"]:
                    continue
                half_w = gap["width"] / 2
                above_mask = (
                    (agent_pos[:, 0] >= gap["x_center"] - half_w)
                    & (agent_pos[:, 0] <= gap["x_center"] + half_w)
                    & (agent_ys > wy)
                )
                above_idx = np.where(above_mask)[0]
                max_cluster_above = max(
                    (cluster_size[int(cluster_id[i])] for i in above_idx),
                    default=0,
                )
                if max_cluster_above >= gap["coupling"]:
                    gap["crossed"] = True
                    crossing_bonus += per_gap_bonus
                else:
                    all_crossed = False

        done = all_crossed

        if self.reward_mode == "dense":
            return shaping_reward + crossing_bonus, done
        return crossing_bonus, done

    def _get_info(self, task_reward=0.0):
        return {
            "agent_positions": [
                {"x": ag.position.x, "y": ag.position.y} for ag in self.agents
            ],
            "task_reward": task_reward,
            "wall_positions": [
                {
                    "y": w["y"],
                    "coupling": w["coupling"],
                    "n_gaps": w["n_gaps"],
                    "gap_centers": [g["x_center"] for g in w["gaps"]],
                    "gap_crossed": [g["crossed"] for g in w["gaps"]],
                }
                for w in self.walls
            ],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_mean_y = None
        self.agent_contact_forces.fill(0.0)

        self.world = b2World(gravity=(0, 0))
        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        self.agents.clear()
        self.wall_bodies.clear()
        self.boundary_bodies.clear()
        self.walls.clear()

        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        self._create_walls()
        self._init_agents()

        obs = self._get_observation()
        if self.render_mode == "human":
            self.render()
        return obs, self._get_info()

    def step(self, actions):
        movement_action = actions[:, :2]

        for agent in self.agents:
            fx = (
                float(np.clip(movement_action[agent.index][0], -1.0, 1.0))
                * self.force_multiplier
            )
            fy = (
                float(np.clip(movement_action[agent.index][1], -1.0, 1.0))
                * self.force_multiplier
            )
            self.applied_forces[agent.index] = [fx, fy]
            agent.apply_force(fx, fy)

        # Apply gap downward forces before stepping the world.
        self._apply_gap_forces()

        self.world.Step(self.time_step, 6, 2)

        self.agent_contact_forces.fill(0.0)
        for (
            agent_idx,
            impulse,
        ) in self.contact_listener.agent_object_normal_impulse.items():
            self.agent_contact_forces[agent_idx] = impulse / self.time_step

        task_reward, _, terminated = self._get_rewards()
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


class HeuristicController:
    """Splits agents into per-gap teams and steers each team to its gap.

    For every wall, agents that still need to cross it are sorted by x and
    sliced into chunks of size ``wall.coupling``, with each chunk matched
    left-to-right to the wall's gaps. Assignments persist until the agent
    crosses the wall; once crossed, the agent is reassigned to the next
    wall (or steered upward if it's already past every wall). Crossing
    itself is left to the env's gap-force gating — the controller just
    aims each agent at ``(gap_x, wall_y + push_offset)``.
    """

    def __init__(
        self,
        env,
        push_offset: float = 1.5,
        rally_offset: float = 1.5,
        rally_tol: float = 0.5,
        slot_spacing: float = 0.95,
    ):
        self.env = env
        self.push_offset = push_offset
        self.rally_offset = rally_offset
        self.rally_tol = rally_tol
        self.slot_spacing = slot_spacing
        # Per-agent assignment: (wall_idx, gap_idx) or None when past all walls.
        self.assignment = [None] * env.n_agents

    def reset(self):
        self.assignment = [None] * self.env.n_agents

    def _next_wall_indices(self, agent_pos):
        """Lowest-y uncrossed wall for each agent; -1 if past every wall."""
        next_idx = np.full(self.env.n_agents, -1, dtype=int)
        for i, y in enumerate(agent_pos[:, 1]):
            for w_idx, wall in enumerate(self.env.walls):
                if y < wall["y"]:
                    next_idx[i] = w_idx
                    break
        return next_idx

    def _refresh_assignments(self, agent_pos, next_wall_idx):
        # Repartition from scratch every step so stragglers (left behind after
        # most of their wave crossed) regroup on a feasible single team.
        self.assignment = [None] * self.env.n_agents

        for w_idx in range(self.env.n_walls):
            wall = self.env.walls[w_idx]
            coupling = wall["coupling"]
            n_gaps = wall["n_gaps"]

            ag_indices = [
                i for i in range(self.env.n_agents) if next_wall_idx[i] == w_idx
            ]
            if not ag_indices:
                continue

            n = len(ag_indices)
            # Only activate as many gaps as we have full teams for; with too
            # few agents collapse onto one gap so coupling can still be met.
            active_gaps = min(n_gaps, max(1, n // coupling))

            ag_indices.sort(key=lambda i: agent_pos[i, 0])

            # Pick the active_gaps gaps closest to the cluster centroid, then
            # order them left-to-right so chunks line up with neighbouring gaps.
            centroid = float(np.mean([agent_pos[i, 0] for i in ag_indices]))
            gap_centers = [(gi, wall["gaps"][gi]["x_center"]) for gi in range(n_gaps)]
            gap_centers.sort(key=lambda gc: abs(gc[1] - centroid))
            selected = sorted(gap_centers[:active_gaps], key=lambda gc: gc[1])

            chunks = np.array_split(np.array(ag_indices), active_gaps)
            for chunk, (gi, _) in zip(chunks, selected):
                for i in chunk:
                    self.assignment[int(i)] = (w_idx, gi)

    def compute_actions(self):
        env = self.env
        agent_pos = np.array(
            [[a.position.x, a.position.y] for a in env.agents], dtype=np.float32
        )
        next_wall_idx = self._next_wall_indices(agent_pos)
        self._refresh_assignments(agent_pos, next_wall_idx)

        # Group teammates by (wall_idx, gap_idx).
        teams = {}
        for i, a in enumerate(self.assignment):
            if a is not None:
                teams.setdefault(a, []).append(i)

        half_t = env.wall_thickness / 2 + env.gap_region_pad

        actions = np.zeros((env.n_agents, 2), dtype=np.float32)
        for i in range(env.n_agents):
            if self.assignment[i] is None:
                target_x = env.world_width / 2.0
                target_y = float(env.world_height)
            else:
                w_idx, g_idx = self.assignment[i]
                wall = env.walls[w_idx]
                gap = wall["gaps"][g_idx]
                wy = wall["y"]
                rally_y = wy - half_t - self.rally_offset

                # Each teammate gets a unique horizontal rally slot so they
                # line up side-by-side below the gap instead of stacking.
                # Sort by current x so the leftmost agent claims the leftmost
                # slot — otherwise team members end up assigned to slots on
                # the wrong side and try to push through each other.
                team_list = sorted(teams[(w_idx, g_idx)], key=lambda j: agent_pos[j, 0])
                n_team = len(team_list)
                my_slot = team_list.index(i)

                def slot_x(k):
                    return (
                        gap["x_center"] + (k - (n_team - 1) / 2.0) * self.slot_spacing
                    )

                # Ready when every teammate is either at its rally slot or
                # already past the rally band (i.e. mid-push). Without this
                # latch, the moment any teammate leaves rally tol the trigger
                # flips back and the team is yanked down again.
                committed_y = rally_y + self.rally_tol
                ready = True
                for k, j in enumerate(team_list):
                    if agent_pos[j, 1] > committed_y:
                        continue
                    if (
                        abs(agent_pos[j, 0] - slot_x(k)) > self.rally_tol
                        or abs(agent_pos[j, 1] - rally_y) > self.rally_tol
                    ):
                        ready = False
                        break

                if agent_pos[i, 1] > wy - half_t or ready:
                    # Push: converge horizontally to the gap centre while
                    # climbing through the zone.
                    target_x = gap["x_center"]
                    target_y = wy + self.push_offset
                else:
                    target_x = slot_x(my_slot)
                    target_y = rally_y

            dx = target_x - agent_pos[i, 0]
            dy = target_y - agent_pos[i, 1]
            dist = float(np.hypot(dx, dy))
            if dist < 1e-6:
                continue
            # Scale action by distance so agents brake smoothly near targets
            # instead of overshooting on full thrust.
            mag = min(dist, 1.0)
            actions[i, 0] = (dx / dist) * mag
            actions[i, 1] = (dy / dist) * mag
        return actions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MultiPass environment debugger")
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Drive agents with the built-in heuristic controller instead of manual input.",
    )
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--n-walls", type=int, default=3)
    parser.add_argument(
        "--gaps",
        type=int,
        nargs="+",
        default=[5, 2, 1],
        help="Number of gaps per wall (length must equal --n-walls).",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()

    env = MultiPassEnv(
        render_mode="human",
        n_agents=args.n_agents,
        n_walls=args.n_walls,
        gaps_per_wall=args.gaps,
        max_steps=args.max_steps,
    )
    obs, info = env.reset()

    controller = HeuristicController(env)

    running = True
    current_agent_idx = 0
    cum_rew = 0
    group_control = False

    print("\n" + "=" * 50)
    print(" MULTI-PASS ENVIRONMENT DEBUGGER")
    print("=" * 50)
    if args.heuristic:
        print(" Mode: HEURISTIC controller (auto-pilot)")
        print(" Controls:")
        print("  [ESC]    : Quit")
    else:
        print(f" Mode: MANUAL — Controlling Agent: {current_agent_idx}")
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

        if controller is not None:
            actions = controller.compute_actions()
        else:
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

        obs_batched = np.expand_dims(obs, axis=0)
        hypergraphs = build_hypergraph(
            1,
            env.n_agents,
            obs_batched,
            partial(distance_based_hyperedges, threshold=1.0),
        )
        entropies = compute_hyperedge_structural_entropy_batch(hypergraphs)
        entropy_log.append(entropies[0])

        cum_rew += reward

        print(f"Cumulative Rew {cum_rew:.3f} Rew {reward:.3f}")

        env.render()

        # if terminated or truncated:
        #     print(">>> Environment Reset")
        #     cum_rew = 0
        #     break

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
