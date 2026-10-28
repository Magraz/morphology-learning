"""Default rover reward: coupled POIs that pay a steady rate until used up."""

import dataclasses

import jax.numpy as jnp

from environments.rover.components import RewardModel, RewardOutput

# A budget below this is snapped to exactly 0, so float32 accumulation cannot
# leave a ~1e-8 residue that pays one extra step (value 1.0 / rate 0.05 must
# deplete in exactly 20 coupled steps).
DEPLETION_TOL = 1e-6


class CouplingDepletionReward(RewardModel):
    """Team reward from POIs whose coupling is met.

    Per step, on the rovers' post-move positions:

    * a rover observes POI p when ``||rover - p|| <= radius_p`` (inclusive);
    * p is active when at least ``coupling_p`` rovers observe it and it has
      budget left. Rovers beyond the coupling add nothing;
    * an active POI pays ``min(rate_p, remaining_p)`` and its budget drops by
      that much, so it pays a steady rate until used up and then nothing for
      the rest of the episode;
    * the returned reward is the team scalar G = sum of payouts.
      ``info["agent_rewards"]`` splits each payout equally among the rovers
      observing that POI, so it sums to G exactly;
    * the episode terminates once every POI is used up.
    """

    def __call__(self, env, rovers, pois):
        dist = jnp.linalg.norm(
            rovers.position[:, None, :] - pois.position[None, :, :], axis=-1
        )  # (A, P)
        in_radius = dist <= pois.radius[None, :]
        observed = in_radius.sum(axis=0)  # (P,)
        active = (observed >= pois.coupling) & (pois.remaining > 0)
        payout = jnp.where(active, jnp.minimum(pois.rate, pois.remaining), 0.0)
        remaining = pois.remaining - payout
        remaining = jnp.where(remaining < DEPLETION_TOL, 0.0, remaining)

        share = payout / jnp.maximum(observed, 1)  # (P,)
        agent_rewards = in_radius.astype(payout.dtype) @ share  # (A,)
        return RewardOutput(
            reward=payout.sum(),
            pois=dataclasses.replace(pois, remaining=remaining),
            terminated=jnp.all(remaining <= 0.0),
            info={
                "agent_rewards": agent_rewards,
                "poi_observed": observed,
                "poi_active": active,
                "poi_payout": payout,
            },
        )
