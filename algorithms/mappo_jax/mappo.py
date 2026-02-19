from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from algorithms.mappo_jax.types import MAPPOConfig, Transition
from algorithms.mappo_jax.network import (
    MAPPOActor,
    MAPPOCritic,
    evaluate_action,
)


class ActorCriticTrainState(NamedTuple):
    """Immutable container for actor + critic training state."""

    actor_ts: TrainState
    critic_ts: TrainState


def create_train_state(
    rng: jax.random.PRNGKey,
    config: MAPPOConfig,
    obs_dim: int,
    global_state_dim: int,
    action_dim: int,
    discrete: bool,
) -> ActorCriticTrainState:
    """Initialize actor/critic params and optimizers."""
    rng_actor, rng_critic = jax.random.split(rng)

    actor = MAPPOActor(action_dim=action_dim, hidden_dim=128, discrete=discrete)
    critic = MAPPOCritic(hidden_dim=256)

    actor_params = actor.init(rng_actor, jnp.zeros(obs_dim))
    critic_params = critic.init(rng_critic, jnp.zeros(global_state_dim))

    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )

    actor_ts = TrainState.create(
        apply_fn=actor.apply, params=actor_params, tx=actor_tx
    )
    critic_ts = TrainState.create(
        apply_fn=critic.apply, params=critic_params, tx=critic_tx
    )

    return ActorCriticTrainState(actor_ts=actor_ts, critic_ts=critic_ts)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and returns via reverse scan.

    Args:
        rewards:    (n_steps, n_envs)  — summed across agents
        values:     (n_steps, n_envs)
        dones:      (n_steps, n_envs)  — episode-level done
        last_value: (n_envs,)          — bootstrap value
        gamma, gae_lambda: scalars

    Returns:
        advantages: (n_steps, n_envs)
        returns:    (n_steps, n_envs)
    """
    # Append bootstrap value
    values_with_bootstrap = jnp.concatenate(
        [values, last_value[None]], axis=0
    )  # (n_steps+1, n_envs)

    def _scan_fn(gae, t):
        # t counts backward: 0 = last step, 1 = second-to-last, ...
        step = rewards.shape[0] - 1 - t
        delta = (
            rewards[step]
            + gamma * values_with_bootstrap[step + 1] * (1.0 - dones[step])
            - values_with_bootstrap[step]
        )
        gae = delta + gamma * gae_lambda * (1.0 - dones[step]) * gae
        return gae, gae

    _, advantages_reversed = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(last_value),
        jnp.arange(rewards.shape[0]),
    )
    advantages = advantages_reversed[::-1]  # reverse to chronological order
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def ppo_update(
    train_state: ActorCriticTrainState,
    rng: jax.random.PRNGKey,
    trajectory: Transition,
    last_value: jnp.ndarray,
    config: MAPPOConfig,
    discrete: bool,
) -> Tuple[ActorCriticTrainState, dict]:
    """Full PPO update: GAE → multi-epoch minibatch gradient steps.

    Args:
        train_state: current actor/critic TrainState
        rng: PRNG key
        trajectory: Transition with leading dim n_steps
        last_value: (n_envs,) bootstrap values
        config: hyperparameters
        discrete: action space type

    Returns:
        updated train_state, loss metrics dict
    """
    n_steps, n_envs, n_agents = trajectory.obs.shape[:3]
    obs_dim = trajectory.obs.shape[3]

    # Sum per-agent rewards for the shared critic
    rewards_sum = trajectory.reward.sum(axis=-1)  # (n_steps, n_envs)

    # Use episode-level dones
    advantages, returns = compute_gae(
        rewards_sum,
        trajectory.value,
        trajectory.done,
        last_value,
        config.gamma,
        config.gae_lambda,
    )

    # --- Flatten for minibatch training ---
    # Actor data: flatten (steps, envs, agents) → (steps*envs*agents,)
    obs_flat = trajectory.obs.reshape(-1, obs_dim)
    actions_flat = trajectory.action.reshape(-1)
    old_log_probs_flat = trajectory.log_prob.reshape(-1)
    masks_flat = trajectory.action_mask.reshape(-1, trajectory.action_mask.shape[-1])

    # Expand advantages to per-agent: (steps, envs) → (steps, envs, agents) → flat
    adv_expanded = jnp.broadcast_to(
        advantages[:, :, None], (n_steps, n_envs, n_agents)
    ).reshape(-1)
    adv_flat = (adv_expanded - adv_expanded.mean()) / (adv_expanded.std() + 1e-8)

    # Critic data: flatten (steps, envs) → (steps*envs,)
    gs_flat = trajectory.global_state.reshape(-1, trajectory.global_state.shape[-1])
    returns_flat = jnp.broadcast_to(
        returns[:, :, None], (n_steps, n_envs, n_agents)
    ).reshape(-1)

    total_samples = obs_flat.shape[0]
    minibatch_size = total_samples // config.n_minibatches

    def _epoch_step(carry, _epoch_idx):
        train_state, rng = carry
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, total_samples)

        def _minibatch_step(carry, mb_idx):
            actor_ts, critic_ts = carry
            start = mb_idx * minibatch_size
            mb_ids = jax.lax.dynamic_slice(perm, (start,), (minibatch_size,))

            mb_obs = obs_flat[mb_ids]
            mb_actions = actions_flat[mb_ids]
            mb_old_lp = old_log_probs_flat[mb_ids]
            mb_adv = adv_flat[mb_ids]
            mb_masks = masks_flat[mb_ids]
            mb_gs = gs_flat[mb_ids]
            mb_returns = returns_flat[mb_ids]

            # --- Actor loss ---
            def actor_loss_fn(actor_params):
                log_probs, entropy = evaluate_action(
                    actor_ts.apply_fn, actor_params,
                    mb_obs, mb_actions, discrete, mb_masks,
                )
                ratio = jnp.exp(log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = jnp.clip(
                    ratio, 1.0 - config.eps_clip, 1.0 + config.eps_clip
                ) * mb_adv
                policy_loss = -jnp.minimum(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                total = policy_loss + config.ent_coef * entropy_loss
                return total, (policy_loss, entropy_loss)

            (actor_loss, (policy_loss, entropy_loss)), actor_grads = (
                jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_ts.params)
            )
            actor_ts = actor_ts.apply_gradients(grads=actor_grads)

            # --- Critic loss ---
            def critic_loss_fn(critic_params):
                values = critic_ts.apply_fn(critic_params, mb_gs)
                return config.val_coef * jnp.mean((values - mb_returns) ** 2)

            critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
                critic_ts.params
            )
            critic_ts = critic_ts.apply_gradients(grads=critic_grads)

            losses = {
                "total_loss": actor_loss + critic_loss,
                "policy_loss": policy_loss,
                "value_loss": critic_loss,
                "entropy_loss": entropy_loss,
            }
            return (actor_ts, critic_ts), losses

        (actor_ts, critic_ts), mb_losses = jax.lax.scan(
            _minibatch_step,
            (train_state.actor_ts, train_state.critic_ts),
            jnp.arange(config.n_minibatches),
        )
        new_ts = ActorCriticTrainState(actor_ts=actor_ts, critic_ts=critic_ts)
        return (new_ts, rng), mb_losses

    (train_state, rng), epoch_losses = jax.lax.scan(
        _epoch_step,
        (train_state, rng),
        jnp.arange(config.n_epochs),
    )

    # Average losses across epochs and minibatches
    mean_losses = jax.tree.map(lambda x: x.mean(), epoch_losses)

    return train_state, mean_losses
