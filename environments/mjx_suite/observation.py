"""Shared JAX observation machinery for the MJX ports of the Box2D suite.

Every ``box2d_suite`` env builds its per-agent observation through the same
``ObservationManager.get_observation`` (see ``box2d_suite/observation.py``), so
every MJX port needs the same 40-dim vector::

    own_velocity(2) + density_sensors(16) + is_touching_object(1)
    + neighbor_fraction(1) + contact_force(1) + nearest_box_vec(2)
    + goal_distance(1) + lidar(N_LIDAR_RAYS)

``MJXObservationBuilder`` is the JAX counterpart of that manager: pure,
``jit``/``vmap``-able, and independent of any particular env. The env owns the
model/qpos layout and hands the builder plain arrays (agent positions and
velocities, box poses); the builder owns the sensor math and the layout.

It covers the whole suite, not just multi_box_push:

- **No objects** (``n_objects=0``, e.g. scatter / rendezvouz): the object half
  of the density sensors, ``is_touching_object``, ``nearest_box_vec`` and the
  contact force all degrade to zeros, exactly as the Box2D manager does when
  ``env.objects`` is empty.
- **No goal region** (``goal_coord=None``, e.g. contact / scatter /
  rendezvouz): ``goal_distance`` is zero, as when ``env.target_areas`` is empty.
- **Per-episode goal wall** (push_box): ``goal_axis`` accepts a *traced* axis
  index (0 = x, 1 = y) as well as the static ``"x"``/``"y"`` strings, so a goal
  side sampled at ``reset`` stays jit-compatible.

Normalizations, sector convention (8 sectors, 22.5 deg shift) and the touch
test are ports of the Box2D code and are verified equal to f32 precision.
"""

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from environments.box2d_suite.observation import BASE_OBS_DIM, N_LIDAR_RAYS, OBS_DIM

__all__ = [
    "BASE_OBS_DIM",
    "N_LIDAR_RAYS",
    "OBS_DIM",
    "MJXObservationBuilder",
    "geom_index_maps",
    "mjx_data_impl",
]

N_SECTORS = 8
TOUCH_EPS = 0.2  # agent counts as touching within radius + eps of a box face
LIDAR_EPS = 1e-3  # ray origin offset past the agent surface (self-hit guard)


def mjx_data_impl(data):
    """mjx 3.10 moved the contact/efc arrays behind ``Data._impl``."""
    return getattr(data, "_impl", data)


def geom_index_maps(mj_model, n_agents, n_objects, agent_geom="g_agent_{}",
                    object_geom="g_box_{}"):
    """``(agent_of_geom, object_of_geom)`` lookups for contact attribution.

    Each is a length-``ngeom`` int array mapping a geom id to its agent/object
    index, or -1 when the geom is neither. Built from the naming convention the
    MJX ports share; pass different format strings if a port deviates.
    """
    agent_of_geom = -np.ones(mj_model.ngeom, dtype=np.int32)
    object_of_geom = -np.ones(mj_model.ngeom, dtype=np.int32)
    for i in range(n_agents):
        agent_of_geom[mj_model.geom(agent_geom.format(i)).id] = i
    for j in range(n_objects):
        object_of_geom[mj_model.geom(object_geom.format(j)).id] = j
    return jnp.asarray(agent_of_geom), jnp.asarray(object_of_geom)


class MJXObservationBuilder:
    """Builds the shared 40-dim per-agent observation from MJX state.

    Args:
        model: the ``mjx.Model`` to raycast the lidar against (the base model —
            not a per-step coupling override, whose masses don't affect rays).
        agent_of_geom / object_of_geom: geom id -> entity index maps for contact
            attribution (see ``geom_index_maps``). Omit when the env has no
            objects; the contact force is then zero everywhere.
        sector_sensor_radius / lidar_range: default to ``world_width / 3`` and
            the sector radius respectively, matching the Box2D manager.
    """

    def __init__(
        self,
        model,
        *,
        n_agents,
        n_objects,
        world_width,
        world_height,
        velocity_norm,
        neighbor_detection_range,
        agent_radius,
        force_multiplier,
        sector_sensor_radius=None,
        lidar_range=None,
        n_lidar_rays=N_LIDAR_RAYS,
        touch_eps=TOUCH_EPS,
        agent_of_geom=None,
        object_of_geom=None,
    ):
        self.model = model
        self.n_agents = int(n_agents)
        self.n_objects = int(n_objects)
        self.world_width = float(world_width)
        self.world_height = float(world_height)
        self.velocity_norm = float(velocity_norm)
        self.neighbor_detection_range = float(neighbor_detection_range)
        self.agent_radius = float(agent_radius)
        self.force_multiplier = float(force_multiplier)
        self.sector_sensor_radius = float(
            self.world_width / 3.0 if sector_sensor_radius is None
            else sector_sensor_radius
        )
        self.lidar_range = float(
            self.sector_sensor_radius if lidar_range is None else lidar_range
        )
        self.n_lidar_rays = int(n_lidar_rays)
        self.touch_eps = float(touch_eps)
        self._agent_of_geom = agent_of_geom
        self._object_of_geom = object_of_geom

        angles = np.arange(self.n_lidar_rays) * (2 * np.pi / self.n_lidar_rays)
        self.lidar_dirs = jnp.asarray(
            np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1),
            dtype=jnp.float32,
        )  # (R, 3) — world frame, same convention as the density sectors

        self.obs_dim = BASE_OBS_DIM + self.n_lidar_rays

    # ------------------------------------------------------------- components

    def touch_matrix(self, agent_pos, box_pos, box_yaw, box_half):
        """(A, O) bool — agent within radius + eps of a (rotated) box surface.

        Port of ``ObservationManager._agent_object_distance`` for the polygon
        case: rotate into the box frame, clamp to the half extents, measure to
        the clamped point. ``box_half`` is the per-box square half-extent (O,).
        """
        if self.n_objects == 0:
            return jnp.zeros((self.n_agents, 0), dtype=bool)
        rel = agent_pos[:, None, :] - box_pos[None, :, :]  # (A, O, 2)
        c, s = jnp.cos(box_yaw), jnp.sin(box_yaw)  # (O,)
        local = jnp.stack(
            [c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]],
            axis=-1,
        )  # (A, O, 2)
        half = jnp.asarray(box_half)[None, :, None]
        clamped = jnp.clip(local, -half, half)
        dist = jnp.linalg.norm(local - clamped, axis=-1)  # (A, O)
        return dist <= self.agent_radius + self.touch_eps

    def density_sensors(self, agent_pos, box_pos):
        """(A, 16) sector centroid-proximity sensors, cols 0-7 agents, 8-15 boxes."""
        R = self.sector_sensor_radius
        shift = jnp.radians(22.5)
        step = 2 * jnp.pi / N_SECTORS

        def sector_block(rel, in_range):
            # rel (A, N, 2), in_range (A, N) -> (A, 8)
            ang = (jnp.arctan2(rel[..., 1], rel[..., 0]) - shift) % (2 * jnp.pi)
            sect = (ang // step).astype(jnp.int32) % N_SECTORS
            onehot = jax.nn.one_hot(sect, N_SECTORS) * in_range[..., None]  # (A,N,8)
            count = onehot.sum(axis=1)  # (A, 8)
            sum_rel = jnp.einsum("ans,anc->asc", onehot, rel)  # (A, 8, 2)
            centroid_dist = jnp.linalg.norm(
                sum_rel / jnp.maximum(count, 1.0)[..., None], axis=-1
            )
            return jnp.where(count > 0, 1.0 - centroid_dist / R, 0.0)

        rel_aa = agent_pos[None, :, :] - agent_pos[:, None, :]  # (A, A, 2)
        dist_aa = jnp.linalg.norm(rel_aa, axis=-1)
        agents_block = sector_block(rel_aa, (dist_aa > 0) & (dist_aa < R))

        if self.n_objects == 0:
            objects_block = jnp.zeros((self.n_agents, N_SECTORS))
        else:
            rel_ao = box_pos[None, :, :] - agent_pos[:, None, :]  # (A, O, 2)
            dist_ao = jnp.linalg.norm(rel_ao, axis=-1)
            objects_block = sector_block(rel_ao, dist_ao < R)

        return jnp.concatenate([agents_block, objects_block], axis=1)

    def neighbor_fractions(self, agent_pos):
        """(A,) fraction of agents within ``neighbor_detection_range`` (incl. self)."""
        pair_dist = self.pairwise_agent_distances(agent_pos)
        return (pair_dist <= self.neighbor_detection_range).mean(axis=1)

    def pairwise_agent_distances(self, agent_pos):
        """(A, A) euclidean distance between every agent pair; zero on the diagonal."""
        return jnp.linalg.norm(agent_pos[:, None, :] - agent_pos[None, :, :], axis=-1)

    def nearest_box_vectors(self, agent_pos, box_pos, delivered=None):
        """(A, 2) relative vector to the nearest *undelivered* box, normalized by
        world_width.

        ``delivered`` is an optional (O,) bool mask; delivered boxes are dropped
        from the nearest-box search (their distance is set to +inf) so an agent
        stops sensing a box that has already been parked in the goal band. The
        vector is zero when the env has no objects, or when every box is
        delivered (no remaining target to point at).
        """
        if self.n_objects == 0:
            return jnp.zeros((self.n_agents, 2))
        rel = box_pos[None, :, :] - agent_pos[:, None, :]  # (A, O, 2)
        dist = jnp.linalg.norm(rel, axis=-1)  # (A, O)
        if delivered is not None:
            dist = jnp.where(delivered[None, :], jnp.inf, dist)
        nearest = jnp.argmin(dist, axis=1)  # (A,)
        vec = rel[jnp.arange(self.n_agents), nearest] / self.world_width
        if delivered is not None:
            # all boxes delivered -> argmin over +inf is meaningless; zero it out
            vec = jnp.where(jnp.all(delivered), 0.0, vec)
        return vec

    def goal_distances(self, agent_pos, goal_coord=None, goal_axis="y"):
        """(A,) signed distance to the goal center along the goal axis.

        ``goal_coord`` is the target center's coordinate *on that axis*; ``None``
        means the env has no target region and the feature is zero. ``goal_axis``
        is ``"y"``/``"x"`` when the axis is fixed, or a traced index (0 = x,
        1 = y) for envs that sample the goal wall per episode (push_box).
        """
        if goal_coord is None:
            return jnp.zeros(self.n_agents)
        if isinstance(goal_axis, str):
            if goal_axis == "x":
                return (goal_coord - agent_pos[:, 0]) / self.world_width
            return (goal_coord - agent_pos[:, 1]) / self.world_height
        is_x = jnp.asarray(goal_axis) == 0
        own = jnp.where(is_x, agent_pos[:, 0], agent_pos[:, 1])
        norm = jnp.where(is_x, self.world_width, self.world_height)
        return (goal_coord - own) / norm

    def lidar(self, data, agent_pos):
        """(A, R) normalized range scan via mjx raycasts.

        ``mjx.ray``'s bodyexclude is static (numpy-side), so instead of excluding
        the caster we start each ray just outside its own sphere surface and add
        the offset back — identical ranges, one fully vmapped call.
        """
        origins3 = jnp.pad(agent_pos, ((0, 0), (0, 1)))  # (A, 3), z = 0
        offset = self.agent_radius + LIDAR_EPS
        start = (
            origins3[:, None, :] + self.lidar_dirs[None, :, :] * offset
        ).reshape(-1, 3)
        vecs = jnp.broadcast_to(
            self.lidar_dirs[None], (self.n_agents, self.n_lidar_rays, 3)
        ).reshape(-1, 3)
        dist, _ = jax.vmap(lambda p, v: mjx.ray(self.model, data, p, v))(start, vecs)
        total = offset + dist
        frac = jnp.where(dist < 0, 1.0, jnp.clip(total / self.lidar_range, 0.0, 1.0))
        return frac.reshape(self.n_agents, self.n_lidar_rays)

    def contact_forces(self, data):
        """(A,) summed contact normal force of each agent against any object.

        With pyramidal friction cones and condim 3, each contact owns 4 efc
        facet rows starting at efc_address, and the normal force is their plain
        sum (mju_decodePyramid) — the counterpart of Box2D's PostSolve
        normal-impulse sum / dt. Zero everywhere for an env with no objects
        (the Box2D listener only ever attributes agent-object contacts).
        """
        if self.n_objects == 0 or self._agent_of_geom is None:
            return jnp.zeros(self.n_agents)
        impl = mjx_data_impl(data)
        contact = impl.contact
        g1, g2 = contact.geom[:, 0], contact.geom[:, 1]
        a_idx = jnp.maximum(self._agent_of_geom[g1], self._agent_of_geom[g2])
        is_pair = (a_idx >= 0) & (
            jnp.maximum(self._object_of_geom[g1], self._object_of_geom[g2]) >= 0
        )
        addr = jnp.maximum(contact.efc_address, 0)[:, None] + jnp.arange(4)[None, :]
        normal = impl.efc_force[addr].sum(axis=1)
        normal = jnp.where(is_pair & (contact.efc_address >= 0), normal, 0.0)
        seg = jnp.where(is_pair, a_idx, self.n_agents)
        return jax.ops.segment_sum(normal, seg, num_segments=self.n_agents + 1)[
            : self.n_agents
        ]

    # ------------------------------------------------------------------ build

    def build(
        self,
        data,
        agent_pos,
        agent_vel,
        box_pos=None,
        box_yaw=None,
        box_half=None,
        goal_coord=None,
        goal_axis="y",
        delivered=None,
    ):
        """(A, obs_dim) float32 observation in the shared Box2D-suite layout.

        ``box_*`` may be omitted when the env has no objects. ``goal_coord`` /
        ``goal_axis`` locate the target band (see ``goal_distances``).
        ``delivered`` is an optional (O,) bool mask of already-delivered boxes,
        excluded from ``nearest_box_vec`` (see ``nearest_box_vectors``).
        """
        touch = self.touch_matrix(agent_pos, box_pos, box_yaw, box_half)
        is_touching = (
            touch.any(axis=1).astype(jnp.float32)[:, None]
            if self.n_objects
            else jnp.zeros((self.n_agents, 1))
        )
        return jnp.concatenate(
            [
                agent_vel / self.velocity_norm,
                self.density_sensors(agent_pos, box_pos),
                is_touching,
                self.neighbor_fractions(agent_pos)[:, None],
                (self.contact_forces(data) / self.force_multiplier)[:, None],
                self.nearest_box_vectors(agent_pos, box_pos, delivered),
                self.goal_distances(agent_pos, goal_coord, goal_axis)[:, None],
                self.lidar(data, agent_pos),
            ],
            axis=1,
        ).astype(jnp.float32)
