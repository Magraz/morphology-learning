"""Pure-jnp radial sector binning shared by the JAX environments.

Kept free of mujoco / Box2D imports so lightweight envs (the rover domain) can
use it without pulling in a physics engine. ``MJXObservationBuilder`` and the
rover ``SectorDensityObservation`` both bin through ``sector_onehot``, so the
two cannot disagree on how an angle maps to a sector.
"""

import jax
import jax.numpy as jnp


def sector_onehot(angle, in_range, n_sectors):
    """Masked one-hot sector assignment.

    Args:
        angle: (..., N) angles in radians, measured from the lower edge of
            sector 0 and increasing counterclockwise. Any real value; it is
            wrapped into [0, 2*pi) here.
        in_range: (..., N) bool / 0-1 mask; masked entries get an all-zero row.
        n_sectors: number of equal sectors covering the full circle.

    Returns:
        (..., N, n_sectors) float one-hot rows, zero where ``in_range`` is False.
    """
    step = 2 * jnp.pi / n_sectors
    ang = angle % (2 * jnp.pi)
    sect = (ang // step).astype(jnp.int32) % n_sectors
    return jax.nn.one_hot(sect, n_sectors) * in_range[..., None]
