"""Default rover observation: heading-relative radial presence sectors."""

import jax.numpy as jnp

from environments.rover.components import ObservationModel
from environments.sectors import sector_onehot


class SectorDensityObservation(ObservationModel):
    """Two blocks of ``n_sectors`` presence values per rover, agents then POIs.

    Sectors rotate with the rover: sector 0 is centred on its heading and spans
    ``[-pi/n, pi/n)``; numbering runs counterclockwise, so with 8 sectors
    sector 2 is to the rover's left and sector 6 to its right. The observation
    has no heading channel, so a world-frame layout would leave a turn-based
    policy unable to tell which way to turn.

    Presence in sector s is a distance-weighted share::

        sum over entities in s with d < R of (1 - d / R),  divided by the total

    where the total is ``n_agents - 1`` for the agent block and ``n_pois`` for
    the POI block. Every term is in (0, 1] and at most ``total`` entities can be
    in one sector, so the value lies in [0, 1]: it grows with how many entities
    are in the sector and with how close they are. The agent block excludes the
    rover itself by index, so a co-located teammate is still sensed. Used-up
    POIs (``remaining == 0``) are not sensed, but still count in the total, so
    the signal of the POIs that remain does not rescale as others deplete.

    Args:
        n_sectors: sectors per block; ``obs_dim = 2 * n_sectors``.
        sensor_radius: R, strict ``d < R``. ``None`` = the arena diagonal, i.e.
            every entity in the arena is sensed.
    """

    def __init__(self, n_sectors=8, sensor_radius=None):
        self.n_sectors = int(n_sectors)
        if self.n_sectors < 1:
            raise ValueError(f"n_sectors must be >= 1, got {n_sectors}")
        if sensor_radius is not None and float(sensor_radius) <= 0:
            raise ValueError(f"sensor_radius must be > 0, got {sensor_radius}")
        self.sensor_radius = None if sensor_radius is None else float(sensor_radius)
        self.obs_dim = 2 * self.n_sectors

    def radius(self, env):
        return env.world_diagonal if self.sensor_radius is None else self.sensor_radius

    def presence(self, rel, valid, heading, radius, total):
        """(A, n_sectors) presence from (A, N, 2) vectors rover -> entity.

        ``valid`` (A, N) masks entities out before the range test; ``total`` is
        the python-int normalizer (0 is treated as 1, giving an all-zero block).
        """
        dist = jnp.linalg.norm(rel, axis=-1)  # (A, N)
        in_range = valid & (dist < radius)
        angle = (
            jnp.arctan2(rel[..., 1], rel[..., 0])
            - heading[:, None]
            + jnp.pi / self.n_sectors
        )
        onehot = sector_onehot(angle, in_range, self.n_sectors)  # (A, N, S)
        weight = 1.0 - dist / radius
        return jnp.einsum("ans,an->as", onehot, weight) / max(total, 1)

    def __call__(self, env, state):
        pos = state.rovers.position
        heading = state.rovers.heading
        n_agents = pos.shape[0]
        R = self.radius(env)

        rel_aa = pos[None, :, :] - pos[:, None, :]  # (A, A, 2): [i, j] = j - i
        not_self = ~jnp.eye(n_agents, dtype=bool)
        agents = self.presence(rel_aa, not_self, heading, R, n_agents - 1)

        pois = state.pois
        n_pois = pois.position.shape[0]
        rel_ap = pois.position[None, :, :] - pos[:, None, :]  # (A, P, 2)
        available = jnp.broadcast_to(pois.remaining > 0, (n_agents, n_pois))
        poi_block = self.presence(rel_ap, available, heading, R, n_pois)

        return jnp.concatenate([agents, poi_block], axis=1)
