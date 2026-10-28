"""Demonstration policies for the rover domain (visualization and tests only).

Both follow the ``rollout`` signature ``policy_fn(params, key, obs, state)``.
``scripted_coalition_policy`` reads the privileged state (true POI positions),
which a learned policy must not.
"""

import jax
import jax.numpy as jnp

from environments.rover.base import wrap_angle


def random_policy(params, key, obs, state):
    """Uniform random ``[turn, speed]`` actions."""
    return jax.random.uniform(key, (obs.shape[0], 2), minval=-1.0, maxval=1.0)


def make_scripted_coalition_policy(env):
    """A scripted team that forms coalitions to meet each POI's coupling.

    Undepleted POIs are filled in index order: rovers 0..c0-1 go to the first
    one, the next c1 rovers to the second, and so on; rovers left over join the
    first undepleted POI. When a POI is used up the assignment is recomputed,
    so a team too small to cover every POI at once covers them in sequence.
    Each rover turns toward its target and drives at a speed that stops it on
    the target, slowing while it is still facing away.
    """

    def policy(params, key, obs, state):
        pois = state.pois
        pos, heading = state.rovers.position, state.rovers.heading
        available = pois.remaining > 0
        need = jnp.where(available, pois.coupling, 0)
        cum = jnp.cumsum(need)  # (P,)
        rank = jnp.arange(pos.shape[0])
        first_block = jnp.argmax(rank[:, None] < cum[None, :], axis=1)  # first True
        target = jnp.where(rank < cum[-1], first_block, jnp.argmax(available))

        delta = pois.position[target] - pos
        dist = jnp.linalg.norm(delta, axis=-1)
        error = wrap_angle(jnp.arctan2(delta[:, 1], delta[:, 0]) - heading)
        turn = jnp.clip(error / env.max_turn, -1.0, 1.0)
        turn = jnp.where(dist < 1e-3, 0.0, turn)
        residual = error - turn * env.max_turn  # heading error after this step's turn
        speed = jnp.clip(dist / env.max_speed, 0.0, 1.0) * jnp.maximum(jnp.cos(residual), 0.0)
        speed = jnp.where(available.any(), speed, 0.0)
        return jnp.stack([turn, 2.0 * speed - 1.0], axis=-1)

    return policy
