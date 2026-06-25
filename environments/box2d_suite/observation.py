import numpy as np
from Box2D import b2PolygonShape, b2CircleShape, b2Vec2, b2RayCastCallback

# Number of lidar rays cast per agent (evenly spaced over 360 deg, world frame).
N_LIDAR_RAYS = 16

# Per-agent observation layout (see ObservationManager.get_observation):
#   own_velocity (2) + density_sensors (16) + is_touching_object (1)
#   + neighbor_fraction (1) + contact_force (1) + nearest_box_vec (2)
#   + goal_distance (1) = 24, then + lidar.
BASE_OBS_DIM = 24
OBS_DIM = BASE_OBS_DIM + N_LIDAR_RAYS


class _ClosestHitCallback(b2RayCastCallback):
    """Records the nearest fixture hit by a ray, ignoring the casting agent."""

    def __init__(self, ignore_body):
        super().__init__()
        self.ignore = ignore_body
        self.fraction = 1.0  # 1.0 == nothing hit within range
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body is self.ignore:
            return -1.0  # skip self, let the ray continue
        self.fraction = fraction
        self.hit = True
        return fraction  # clip the ray here so we keep the closest hit


class ObservationManager:
    """Encapsulates all observation computation for Box2D_Suite envs."""

    def __init__(self, env):
        self.env = env

    def _agent_object_distance(self, agent_pos, obj, obj_pos):
        """Compute the surface distance between an agent position and an object."""
        shape = obj.fixtures[0].shape
        if isinstance(shape, b2PolygonShape):
            local_pos = obj.GetLocalPoint((float(agent_pos[0]), float(agent_pos[1])))
            vertices = shape.vertices
            half_w = max(abs(v[0]) for v in vertices)
            half_h = max(abs(v[1]) for v in vertices)
            closest_x = np.clip(local_pos[0], -half_w, half_w)
            closest_y = np.clip(local_pos[1], -half_h, half_h)
            return np.sqrt(
                (local_pos[0] - closest_x) ** 2 + (local_pos[1] - closest_y) ** 2
            )
        elif isinstance(shape, b2CircleShape):
            return np.linalg.norm(agent_pos - obj_pos) - shape.radius
        else:
            return np.linalg.norm(agent_pos - obj_pos)

    def _is_agent_touching_object(self, agent_idx):
        """
        Check if an agent is close enough to any object to be considered 'touching' it.
        Uses distance-based proximity instead of Box2D contacts for stability.
        """
        agent_pos = self._agent_pos_cache[agent_idx]
        agent_radius = self.env.agents[agent_idx].radius

        for obj_idx, obj in enumerate(self._objects):
            obj_pos = self._object_pos_cache[obj_idx]
            dist = self._agent_object_distance(agent_pos, obj, obj_pos)
            if dist <= agent_radius + 0.2:
                return 1.0

        return 0.0

    def get_observation(self):
        # Cache all positions once — eliminates redundant Box2D bridge calls across
        # _calculate_density_sensors_all and _is_agent_touching_object
        self._agent_pos_cache = np.array(
            [[ag.position.x, ag.position.y] for ag in self.env.agents], dtype=np.float32
        )  # (n_agents, 2)
        # Not every env defines `objects` — fall back to an empty list so every
        # downstream consumer (touch check, density sensors, position cache)
        # degrades gracefully instead of raising AttributeError.
        self._objects = getattr(self.env, "objects", [])
        self._object_pos_cache = np.array(
            [[o.position.x, o.position.y] for o in self._objects],
            dtype=np.float32,
        ).reshape(-1, 2)  # (n_objects, 2)

        # Derive all_states from cache (no separate position reads)
        center = np.array(
            [self.env.world_center_x, self.env.world_center_y], dtype=np.float32
        )
        all_states = self._agent_pos_cache - center  # (n_agents, 2)

        # Agent velocities normalized to ~[-1, 1]
        all_velocities = (
            np.array(
                [
                    [ag.linear_velocity.x, ag.linear_velocity.y]
                    for ag in self.env.agents
                ],
                dtype=np.float32,
            )
            / self.env.velocity_norm
        )  # (n_agents, 2)

        # Vectorized density sensors for all agents in one pass (replaces n_agents calls)
        all_density_sensors = self._calculate_density_sensors_all(
            self.env.sector_sensor_radius
        )

        # Fraction of agents within neighbor_detection_range of each agent (incl. self)
        all_neighbor_fractions = self._calculate_neighbor_fractions(
            self.env.neighbor_detection_range
        )

        # Lidar: nearest-obstacle distance along N evenly spaced rays per agent.
        all_lidar = self._calculate_lidar_all(
            getattr(self.env, "n_lidar_rays", N_LIDAR_RAYS),
            getattr(self.env, "lidar_range", self.env.sector_sensor_radius),
        )

        # Goal-relative features (egocentric, so no absolute world anchor):
        #   nearest_box_vec — relative (dx, dy) to the closest object, per axis
        #     normalized by world_width; zero vector when there are no objects.
        #   goal_distance — signed relative y distance from the agent to the
        #     target region center, normalized by world_height; 0 when the env
        #     has no target areas.
        all_nearest_box_vec = self._calculate_nearest_box_vectors()
        all_goal_distance = self._calculate_goal_distances()

        # Normalize per-agent contact force by max applicable force so the
        # observation lives in roughly [0, 1] (can exceed 1 when several
        # bodies pile up against a stalled agent — still well-scaled).
        contact_force_norm = self.env.agent_contact_forces / self.env.force_multiplier

        observations = []
        for i in range(self.env.n_agents):
            own_pos = all_states[i]
            own_velocity = all_velocities[i]
            is_touching_object = np.array([self._is_agent_touching_object(i)])
            neighbor_fraction = np.array([all_neighbor_fractions[i]])
            contact_force = np.array([contact_force_norm[i]], dtype=np.float32)
            nearest_box_vec = all_nearest_box_vec[i]
            goal_distance = np.array([all_goal_distance[i]], dtype=np.float32)
            agent_obs = np.concatenate(
                [
                    # own_pos,
                    own_velocity,
                    all_density_sensors[i],
                    is_touching_object,
                    neighbor_fraction,
                    contact_force,
                    nearest_box_vec,
                    goal_distance,
                    all_lidar[i],
                ]
            )
            observations.append(agent_obs)

        return np.array(observations, dtype=np.float32)

    def _calculate_neighbor_fractions(self, radius):
        """Fraction of all agents within `radius` of each agent (self included)."""
        agent_pos = self._agent_pos_cache  # (A, 2)
        diff = agent_pos[:, np.newaxis, :] - agent_pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)  # (A, A), zero on the diagonal
        within = (dist <= radius).sum(axis=1).astype(np.float32)  # (A,)
        return within / float(self.env.n_agents)

    def _calculate_nearest_box_vectors(self):
        """Relative (dx, dy) from each agent to its nearest object.

        Per-axis normalized by world_width so values land in ~[-1, 1]. Returns a
        zero vector for every agent when the env has no objects.
        """
        n_agents = self.env.n_agents
        if self._object_pos_cache.shape[0] == 0:
            return np.zeros((n_agents, 2), dtype=np.float32)

        # rel[i, o] = obj_pos[o] - agent_pos[i]
        rel = (
            self._object_pos_cache[np.newaxis, :, :]
            - self._agent_pos_cache[:, np.newaxis, :]
        )  # (A, O, 2)
        dist = np.linalg.norm(rel, axis=-1)  # (A, O)
        nearest = np.argmin(dist, axis=1)  # (A,)
        nearest_vec = rel[np.arange(n_agents), nearest]  # (A, 2)
        return (nearest_vec / float(self.env.world_width)).astype(np.float32)

    def _calculate_goal_distances(self):
        """Signed relative distance from each agent to the target region center.

        Measured along the env's goal axis: the y axis by default (normalized by
        world_height), or the x axis (normalized by world_width) when the env
        sets ``goal_axis == "x"`` — e.g. push_box, whose target band can sit
        against the left/right wall. Returns zeros when the env defines no
        target areas (envs without a goal region).
        """
        n_agents = self.env.n_agents
        target_areas = getattr(self.env, "target_areas", None)
        if not target_areas:
            return np.zeros(n_agents, dtype=np.float32)

        target = target_areas[0]
        if getattr(self.env, "goal_axis", "y") == "x":
            goal_d = (target.x - self._agent_pos_cache[:, 0]) / float(
                self.env.world_width
            )
        else:
            goal_d = (target.y - self._agent_pos_cache[:, 1]) / float(
                self.env.world_height
            )
        return goal_d.astype(np.float32)

    def _calculate_lidar_all(self, n_rays, max_range):
        """
        Lidar-style range scan for every agent via Box2D raycasts.

        Casts `n_rays` evenly spaced rays (world frame, matching the density
        sensor convention) out to `max_range` from each agent. Each ray returns
        the normalized distance to the nearest obstacle hit — agents, objects,
        and boundary walls are all picked up by the physics raycast. The casting
        agent is ignored so a ray never hits its own body.

        Returns:
            np.ndarray of shape (n_agents, n_rays) in [0, 1]:
                fraction of `max_range` to the closest hit, or 1.0 if the ray
                reaches its full length without hitting anything.
        """
        angles = np.arange(n_rays) * (2 * np.pi / n_rays)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (R, 2)

        lidar = np.ones((self.env.n_agents, n_rays), dtype=np.float32)
        for i, agent in enumerate(self.env.agents):
            ox, oy = float(self._agent_pos_cache[i][0]), float(
                self._agent_pos_cache[i][1]
            )
            origin = b2Vec2(ox, oy)
            for r in range(n_rays):
                end = b2Vec2(
                    ox + float(dirs[r, 0]) * max_range,
                    oy + float(dirs[r, 1]) * max_range,
                )
                cb = _ClosestHitCallback(agent)
                self.env.world.RayCast(cb, origin, end)
                if cb.hit:
                    lidar[i, r] = cb.fraction  # already normalized to [0, 1]
        return lidar

    def _calculate_density_sensors_all(self, sensor_radius):
        """
        Vectorized density sensors for all agents simultaneously.

        Replaces n_agents calls to _calculate_density_sensors (each with
        O(n_agents + n_objects) Python loops) with a single numpy broadcast pass.

        Returns:
            np.ndarray of shape (n_agents, 16):
                columns 0-7:  normalized centroid distance per sector, agents
                columns 8-15: normalized centroid distance per sector, objects
        """
        n_sectors = 8
        sector_step = (2 * np.pi) / n_sectors
        shift = np.radians(22.5)

        # Use cached positions (zero Box2D reads here)
        agent_pos = self._agent_pos_cache  # (A, 2)

        sensors = np.zeros((self.env.n_agents, 16), dtype=np.float32)

        # === Agent-to-agent ===
        # rel_aa[i, j] = pos[j] - pos[i]: position of j relative to i
        rel_aa = agent_pos[np.newaxis, :, :] - agent_pos[:, np.newaxis, :]  # (A, A, 2)
        dist_aa = np.linalg.norm(rel_aa, axis=-1)  # (A, A)
        in_aa = (dist_aa > 0) & (dist_aa < sensor_radius)  # exclude self & out-of-range

        angle_aa = (np.arctan2(rel_aa[..., 1], rel_aa[..., 0]) - shift) % (2 * np.pi)
        sect_aa = (angle_aa / sector_step).astype(np.int32) % n_sectors  # (A, A)

        for s in range(n_sectors):
            mask = in_aa & (sect_aa == s)  # (A, A)
            count = mask.sum(axis=1)  # (A,)
            has = count > 0
            if not has.any():
                continue
            sum_rel = (rel_aa * mask[:, :, np.newaxis]).sum(axis=1)  # (A, 2)
            denom = np.where(count > 0, count, 1).astype(np.float32)
            centroid_dist = np.linalg.norm(sum_rel / denom[:, np.newaxis], axis=1)
            sensors[:, s] = np.where(has, 1.0 - centroid_dist / sensor_radius, 0.0)

        # === Agent-to-object ===
        if self._object_pos_cache.shape[0] > 0:
            obj_pos = self._object_pos_cache  # (O, 2)

            rel_ao = (
                obj_pos[np.newaxis, :, :] - agent_pos[:, np.newaxis, :]
            )  # (A, O, 2)
            dist_ao = np.linalg.norm(rel_ao, axis=-1)  # (A, O)
            in_ao = dist_ao < sensor_radius

            angle_ao = (np.arctan2(rel_ao[..., 1], rel_ao[..., 0]) - shift) % (
                2 * np.pi
            )
            sect_ao = (angle_ao / sector_step).astype(np.int32) % n_sectors  # (A, O)

            for s in range(n_sectors):
                mask = in_ao & (sect_ao == s)  # (A, O)
                count = mask.sum(axis=1)  # (A,)
                has = count > 0
                if not has.any():
                    continue
                sum_rel = (rel_ao * mask[:, :, np.newaxis]).sum(axis=1)  # (A, 2)
                denom = np.where(count > 0, count, 1).astype(np.float32)
                centroid_dist = np.linalg.norm(sum_rel / denom[:, np.newaxis], axis=1)
                sensors[:, n_sectors + s] = np.where(
                    has, 1.0 - centroid_dist / sensor_radius, 0.0
                )

        return sensors  # (n_agents, 16)

    def calculate_density_sensors(self, agent_idx, sensor_radius):
        """
        Calculate normalized distance to centroid of agents and objects in 8 sectors around an agent.
        Distances are normalized by the sensor_radius so values are in [0, 1].
        0.0 means no entities in that sector, otherwise value is distance/sensor_radius.

        Returns a vector of 16 values:
        - First 8 values: normalized distance to centroid of agents in sectors 0-7
        - Next 8 values: normalized distance to centroid of objects in sectors 0-7
        """
        agent_pos = np.array(
            [
                self.env.agents[agent_idx].position.x,
                self.env.agents[agent_idx].position.y,
            ]
        )

        n_sectors = 8
        sector_radian_step = (2 * np.pi) / n_sectors
        shift_radians = np.radians(22.5)

        # Collect positions per sector for agents and objects
        agent_sector_positions = [[] for _ in range(n_sectors)]
        object_sector_positions = [[] for _ in range(n_sectors)]

        # Check each other agent
        for other_idx, other_agent in enumerate(self.env.agents):
            if other_idx == agent_idx:
                continue

            other_pos = np.array([other_agent.position.x, other_agent.position.y])
            relative_pos = other_pos - agent_pos
            distance = np.linalg.norm(relative_pos)

            # Skip if outside sensor radius
            if distance > sensor_radius:
                continue

            # Calculate angle in range [0, 2pi)
            angle = np.arctan2(relative_pos[1], relative_pos[0])
            angle -= shift_radians
            if angle < 0:
                angle += 2 * np.pi
            elif angle >= 2 * np.pi:
                angle -= 2 * np.pi

            sector = int(angle / sector_radian_step) % n_sectors
            agent_sector_positions[sector].append(other_pos)

        # Check each dynamic object
        for obj in getattr(self.env, "objects", []):
            obj_pos = np.array([obj.position.x, obj.position.y])
            relative_pos = obj_pos - agent_pos
            distance = np.linalg.norm(relative_pos)

            # Skip if outside sensor radius
            if distance > sensor_radius:
                continue

            # Calculate angle in range [0, 2pi)
            angle = np.arctan2(relative_pos[1], relative_pos[0])
            angle -= shift_radians
            if angle < 0:
                angle += 2 * np.pi
            elif angle >= 2 * np.pi:
                angle -= 2 * np.pi

            sector = int(angle / sector_radian_step) % n_sectors
            object_sector_positions[sector].append(obj_pos)

        # Compute normalized distance to centroid per sector
        agent_centroid_distances = np.zeros(n_sectors, dtype=np.float32)
        object_centroid_distances = np.zeros(n_sectors, dtype=np.float32)

        for s in range(n_sectors):
            # Agent centroid distance (inverted and normalized by sensor_radius)
            if len(agent_sector_positions[s]) > 0:
                centroid = np.mean(agent_sector_positions[s], axis=0)
                agent_centroid_distances[s] = 1.0 - (
                    np.linalg.norm(centroid - agent_pos) / sensor_radius
                )
            else:
                agent_centroid_distances[s] = 0.0

            # Object centroid distance (inverted and normalized by sensor_radius)
            if len(object_sector_positions[s]) > 0:
                centroid = np.mean(object_sector_positions[s], axis=0)
                object_centroid_distances[s] = 1.0 - (
                    np.linalg.norm(centroid - agent_pos) / sensor_radius
                )
            else:
                object_centroid_distances[s] = 0.0

        # Combine all values into one array
        return np.concatenate(
            [
                agent_centroid_distances,  # 8 values, normalized [0, 1]
                object_centroid_distances,  # 8 values, normalized [0, 1]
            ]
        )
