"""PPO update mirroring ``mappo_vanilla.mappo.MAPPOAgent.update`` in JAX.

Parity notes (vs. the torch implementation):
- GAE runs once per env on the team reward + shared critic value; vanilla tiles
  the reward per agent, but with identical rewards and a shared value the
  per-agent advantages are identical, so env-level GAE broadcast to agents is
  the same computation.
- Advantages are normalized per (env) trajectory over the rollout steps with an
  unbiased std (torch ``.std()``), matching vanilla's per-(env, agent) stream
  normalization.
- Minibatches are timestep-centric like ``update_shared``: one sample is one
  (step, env) element carrying all agents, the critic runs once per element,
  and the timestep minibatch size is ``(batch // n_minibatches) // n_agents``.
  (Deviation: the trailing partial minibatch is dropped — jit needs static
  shapes; torch's DataLoader keeps it.)
- Actor and critic use separate Adam optimizers; since they share no
  parameters this is equivalent to vanilla's single Adam over the combined
  loss ``policy + val_coef * value + ent_coef * entropy``.
- ``explained_variance`` is the same pre-update diagnostic vanilla computes.
"""

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
    rng: jax.Array,
    config: MAPPOConfig,
    obs_dim: int,
    global_state_dim: int,
    action_dim: int,
    discrete: bool,
    n_critic_outputs: int = 1,
) -> ActorCriticTrainState:
    """Initialize actor/critic params and optimizers.

    Matches ``MAPPONetwork``: actor hidden = ``hidden_dim``, centralized critic
    hidden = ``2 * hidden_dim``. ``n_critic_outputs`` > 1 gives the critic a
    per-agent value head (per-agent rewards); 1 keeps the scalar critic.
    """
    rng_actor, rng_critic = jax.random.split(rng)

    actor = MAPPOActor(
        action_dim=action_dim, hidden_dim=config.hidden_dim, discrete=discrete
    )
    critic = MAPPOCritic(hidden_dim=2 * config.hidden_dim, n_outputs=n_critic_outputs)

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

    Shape-agnostic in the trailing axis: with a scalar team reward everything is
    (n_steps, n_envs); with per-agent rewards everything carries a trailing agent
    axis (n_steps, n_envs, n_agents) and the identical recursion runs per agent.

    Args:
        rewards:    (n_steps, n_envs) team | (n_steps, n_envs, n_agents) per-agent
        values:     same shape as rewards
        dones:      (n_steps, n_envs) — episode-level, shared by all agents
        last_value: (n_envs,) | (n_envs, n_agents) — bootstrap value
        gamma, gae_lambda: scalars

    Returns:
        advantages, returns — both shaped like rewards
    """
    # `done` is per-env; add the agent axis so it broadcasts against per-agent
    # values instead of colliding with them.
    if rewards.ndim > dones.ndim:
        dones = dones[..., None]

    values_with_bootstrap = jnp.concatenate(
        [values, last_value[None]], axis=0
    )  # (n_steps+1, ...)

    def _scan_fn(gae, t):
        # t counts backward: 0 = last step, 1 = second-to-last, ...
        step = rewards.shape[0] - 1 - t
        not_done = 1.0 - dones[step]
        delta = (
            rewards[step]
            + gamma * values_with_bootstrap[step + 1] * not_done
            - values_with_bootstrap[step]
        )
        gae = delta + gamma * gae_lambda * not_done * gae
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
    rng: jax.Array,
    trajectory: Transition,
    last_value: jnp.ndarray,
    config: MAPPOConfig,
    discrete: bool,
) -> Tuple[ActorCriticTrainState, dict]:
    """Full PPO update: GAE → multi-epoch timestep-centric minibatch steps.

    Args:
        train_state: current actor/critic TrainState
        rng: PRNG key
        trajectory: Transition with leading dim n_steps
        last_value: (n_envs,) bootstrap values (GAE masks dones internally)
        config: hyperparameters
        discrete: action space type

    Returns:
        updated train_state, loss metrics dict
    """
    n_steps, n_envs, n_agents = trajectory.obs.shape[:3]
    obs_dim = trajectory.obs.shape[3]
    # Static: (n_steps, n_envs, n_agents) rewards => per-agent credit path.
    per_agent = trajectory.reward.ndim == 3

    dones = trajectory.done.astype(jnp.float32)
    advantages, returns = compute_gae(
        trajectory.reward,
        trajectory.value,
        dones,
        last_value,
        config.gamma,
        config.gae_lambda,
    )

    # Pre-update explained variance of the stored critic predictions
    explained_variance = 1.0 - jnp.var(returns - trajectory.value, ddof=1) / (
        jnp.var(returns, ddof=1) + 1e-8
    )

    # Advantage normalization per stream over the rollout steps: per-env for the
    # team reward, per-(env, agent) under per-agent rewards — which is exactly
    # vanilla's per-(env, agent) normalization.
    adv = (advantages - advantages.mean(axis=0)) / (
        advantages.std(axis=0, ddof=1) + 1e-8
    )

    # --- Timestep-centric flattening: one sample per (step, env) ---
    total_ts = n_steps * n_envs
    obs_ts = trajectory.obs.reshape(total_ts, n_agents, obs_dim)
    gs_ts = trajectory.global_state.reshape(total_ts, -1)
    act_ts = trajectory.action.reshape(
        total_ts, n_agents, *trajectory.action.shape[3:]
    )
    lp_ts = trajectory.log_prob.reshape(total_ts, n_agents)
    if per_agent:
        adv_ts = adv.reshape(total_ts, n_agents)
        ret_ts = returns.reshape(total_ts, n_agents)
    else:
        adv_ts = adv.reshape(total_ts)
        ret_ts = returns.reshape(total_ts)

    # minibatch_size agent-samples => minibatch_size // n_agents timesteps
    ts_minibatch_size = max(
        1, (total_ts // config.n_minibatches) // n_agents
    )
    n_minibatches = total_ts // ts_minibatch_size

    def _epoch_step(carry, _epoch_idx):
        train_state, rng = carry
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, total_ts)

        def _minibatch_step(carry, mb_idx):
            actor_ts, critic_ts = carry
            start = mb_idx * ts_minibatch_size
            mb_ids = jax.lax.dynamic_slice(perm, (start,), (ts_minibatch_size,))

            n_flat = ts_minibatch_size * n_agents
            mb_obs = obs_ts[mb_ids].reshape(n_flat, obs_dim)
            mb_actions = act_ts[mb_ids].reshape(n_flat, *act_ts.shape[2:])
            mb_old_lp = lp_ts[mb_ids].reshape(n_flat)
            if per_agent:
                # Each agent carries its own advantage. `.reshape(-1)` is
                # agent-major within a timestep, matching mb_obs' flattening.
                mb_adv = adv_ts[mb_ids].reshape(n_flat)
            else:
                # env-level advantage broadcast to each agent (identical per agent)
                mb_adv = jnp.repeat(adv_ts[mb_ids], n_agents)
            mb_gs = gs_ts[mb_ids]
            mb_returns = ret_ts[mb_ids]

            # --- Actor loss ---
            def actor_loss_fn(actor_params):
                log_probs, entropy = evaluate_action(
                    actor_ts.apply_fn, actor_params, mb_obs, mb_actions, discrete
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

            (_, (policy_loss, entropy_loss)), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(actor_ts.params)
            actor_ts = actor_ts.apply_gradients(grads=actor_grads)

            # --- Critic loss (once per timestep; shared value vs team return) ---
            def critic_loss_fn(critic_params):
                values = critic_ts.apply_fn(critic_params, mb_gs)
                value_loss = jnp.mean((values - mb_returns) ** 2)
                return config.val_coef * value_loss, value_loss

            (_, value_loss), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(critic_ts.params)
            critic_ts = critic_ts.apply_gradients(grads=critic_grads)

            # Stats mirror vanilla: raw component losses + the combined total
            losses = {
                "total_loss": (
                    policy_loss
                    + config.val_coef * value_loss
                    + config.ent_coef * entropy_loss
                ),
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
            }
            return (actor_ts, critic_ts), losses

        (actor_ts, critic_ts), mb_losses = jax.lax.scan(
            _minibatch_step,
            (train_state.actor_ts, train_state.critic_ts),
            jnp.arange(n_minibatches),
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
    mean_losses["explained_variance"] = explained_variance

    return train_state, mean_losses
