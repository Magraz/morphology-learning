import numpy as np
from Box2D import b2PolygonShape, b2CircleShape


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

        for obj_idx, obj in enumerate(self.env.objects):
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
        self._object_pos_cache = (
            np.array(
                [[o.position.x, o.position.y] for o in self.env.objects],
                dtype=np.float32,
            )
            if self.env.objects
            else np.empty((0, 2), dtype=np.float32)
        )  # (n_objects, 2)

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

        observations = []
        for i in range(self.env.n_agents):
            own_state = all_states[i]
            own_velocity = all_velocities[i]
            is_touching_object = np.array([self._is_agent_touching_object(i)])
            agent_obs = np.concatenate(
                [own_state, own_velocity, all_density_sensors[i], is_touching_object]
            )
            observations.append(agent_obs)

        return np.array(observations, dtype=np.float32)

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
        if self.env.objects:
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
        for obj in self.env.objects:
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
