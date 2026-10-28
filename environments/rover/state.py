"""Pytree state of the rover domain.

Every leaf is a JAX array (no python scalars, no ``None``), so a caller can
reset finished environments with one ``jax.tree.map(jnp.where(done, reset,
cur))`` over the whole state, as the ``mappo_jax`` collector does for the MJX
envs. Dynamics are deterministic, so no PRNG key is carried: a rollout is a
pure function of its reset key and the actions.
"""

import dataclasses

import jax


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RoverState:
    position: jax.Array  # (A, 2) world coordinates in [0, W] x [0, H]
    heading: jax.Array  # (A,) radians, wrapped to [-pi, pi)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class POIState:
    position: jax.Array  # (P, 2) world coordinates
    coupling: jax.Array  # (P,) int32, rovers needed inside the radius at once
    radius: jax.Array  # (P,) observation radius, inclusive
    value: jax.Array  # (P,) total reward the POI can pay in an episode
    rate: jax.Array  # (P,) reward paid per step while the coupling is met
    remaining: jax.Array  # (P,) budget left this episode; 0 = used up


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RoverEnvState:
    rovers: RoverState
    pois: POIState
    t: jax.Array  # () int32, steps taken this episode
