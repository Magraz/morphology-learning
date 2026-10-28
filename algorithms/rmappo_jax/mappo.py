"""on-policy's `R_MAPPO.train` / `ppo_update` / `cal_value_loss`, in JAX.

One update, in on-policy's order:
1. GAE on the denormalized values (`SharedReplayBuffer.compute_returns`):
   rewards, values and returns are per (env, agent) row. on-policy's
   `masks[step + 1]` is `1 - done[step]` here, which is what
   `mappo_jax.mappo.compute_gae` already masks with.
2. Advantages normalized over the WHOLE batch (time x env x agent), with the
   mean and the population std (`np.nanstd`) taken over active rows only, and
   `+ 1e-5` on the std (`R_MAPPO.train`).
3. `n_epochs` passes. Each pass draws a fresh permutation of the training
   sequences and splits it into `n_minibatches` (the remainder is dropped, as
   in on-policy's samplers).
   - Recurrent policy (`recurrent_generator`): each (env, agent) trajectory is
     cut into chunks of `data_chunk_length` steps, indexed in on-policy's
     (env, agent, time) order. A chunk starts from the hidden state STORED at
     its first step, and the GRU resets wherever the episode mask is 0.
   - Feedforward policy (`feed_forward_generator`): every row is its own
     sequence of length 1.
4. Per minibatch, the actor step then the critic step, with separate Adams and
   separate gradient-norm clips (`R_MAPPO.ppo_update`):
   - actor: `PPO-clip - ent_coef * entropy`, both averaged over active rows;
   - critic: `ValueNorm` is updated from the minibatch's returns (every row)
     BEFORE its loss, then `max(huber(e), huber(e_clipped))` against the
     normalized returns, averaged over active rows, times `val_coef`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from algorithms.mappo_jax.mappo import compute_gae
from algorithms.rmappo_jax import value_norm as vn_lib
from algorithms.rmappo_jax.network import evaluate_action, make_networks
from algorithms.rmappo_jax.types import RMAPPOConfig, Transition

ADVANTAGE_EPS = 1e-5  # R_MAPPO.train: (adv - mean) / (std + 1e-5)


class RMAPPOTrainState(NamedTuple):
    actor_ts: TrainState
    critic_ts: TrainState
    value_norm: vn_lib.ValueNormState


def zero_hidden(config: RMAPPOConfig, batch_shape) -> jnp.ndarray:
    """Zero GRU hidden states `(*batch_shape, recurrent_N, hidden_size)`, or a
    scalar placeholder the feedforward networks pass through untouched."""
    if not config.use_recurrent_policy:
        return jnp.zeros(())
    return jnp.zeros((*batch_shape, config.recurrent_N, config.hidden_size))


def _optimizer(lr: float, config: RMAPPOConfig, num_updates: int):
    """`clip_grad_norm_(max_grad_norm)` then `Adam(lr, eps=opti_eps)`.

    With `use_linear_lr_decay` the rate is `lr * (1 - update / num_updates)`
    (`update_linear_schedule`), held for every optimizer step of an update.
    """
    if config.use_linear_lr_decay:
        steps_per_update = config.n_epochs * config.n_minibatches

        def learning_rate(count):
            return lr * (1.0 - (count // steps_per_update) / num_updates)

    else:
        learning_rate = lr
    chain = [optax.adam(learning_rate, eps=config.opti_eps)]
    if config.use_max_grad_norm:
        chain.insert(0, optax.clip_by_global_norm(config.grad_clip))
    return optax.chain(*chain)


def create_train_state(
    rng: jax.Array,
    config: RMAPPOConfig,
    obs_dim: int,
    global_state_dim: int,
    action_dim: int,
    discrete: bool,
    num_updates: int,
) -> RMAPPOTrainState:
    actor, critic = make_networks(config, action_dim, discrete)
    rng_actor, rng_critic = jax.random.split(rng)
    h, mask = zero_hidden(config, (1,)), jnp.ones((1,))
    actor_params = actor.init(rng_actor, jnp.zeros((1, obs_dim)), h, mask)
    critic_params = critic.init(rng_critic, jnp.zeros((1, global_state_dim)), h, mask)
    return RMAPPOTrainState(
        actor_ts=TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=_optimizer(config.lr, config, num_updates),
        ),
        critic_ts=TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=_optimizer(config.critic_lr, config, num_updates),
        ),
        value_norm=vn_lib.init_value_norm(),
    )


def _masked_mean(x, weight):
    return (x * weight).sum() / weight.sum()


def prepare_advantages(trajectory: Transition, last_value, value_norm, config):
    """GAE in reward units, then on-policy's batch-wide normalization.

    Returns `(normalized_advantages, returns, explained_variance)`, the first
    two `(n_steps, n_envs, n_agents)`. `trajectory.value` and `last_value` are
    the critic's normalized outputs.
    """
    if config.use_valuenorm:
        values = vn_lib.denormalize(value_norm, trajectory.value)
        last_value = vn_lib.denormalize(value_norm, last_value)
    else:
        values = trajectory.value
    advantages, returns = compute_gae(
        trajectory.reward,
        values,
        trajectory.done,
        last_value,
        config.gamma,
        config.gae_lambda,
    )
    active = trajectory.active_mask
    mean = _masked_mean(advantages, active)
    std = jnp.sqrt(_masked_mean((advantages - mean) ** 2, active))
    normalized = (advantages - mean) / (std + ADVANTAGE_EPS)

    # Pre-update diagnostic (not in on-policy): how much of the returns' variance
    # the stored values explain, over active rows.
    def _var(x):
        return _masked_mean((x - _masked_mean(x, active)) ** 2, active)

    explained_variance = 1.0 - _var(returns - values) / (_var(returns) + 1e-8)
    return normalized, returns, explained_variance


def chunk_count(trajectory: Transition, chunk_length: int) -> int:
    """Training sequences in a rollout: `n_envs * n_agents * n_steps // L`."""
    n_steps, n_envs, n_agents = trajectory.reward.shape
    return n_envs * n_agents * (n_steps // chunk_length)


def gather_minibatch(trajectory: Transition, advantages, returns, chunk_ids, chunk_length):
    """The training sequences `chunk_ids`, each field `(L, B, ...)`.

    Chunk `c` follows on-policy's `_cast` order (env, agent, time): env
    `c // (N * C)`, agent `(c // C) % N`, steps `(c % C) * L` onward, with C
    chunks per (env, agent) trajectory. `actor_h0` / `critic_h0` are the hidden
    states STORED at each chunk's first step (`recurrent_generator`), or the
    feedforward placeholder.
    """
    n_steps, _, n_agents = trajectory.reward.shape
    per_trajectory = n_steps // chunk_length
    env = chunk_ids // (n_agents * per_trajectory)
    agent = (chunk_ids // per_trajectory) % n_agents
    start = (chunk_ids % per_trajectory) * chunk_length
    t = start[None, :] + jnp.arange(chunk_length)[:, None]  # (L, B)
    e, a = env[None, :], agent[None, :]

    def per_agent(x):
        return x[t, e, a]

    def per_env(x):
        return x[t, e]

    def hidden(h):
        return h[start, env, agent] if h.ndim > 1 else jnp.zeros(())

    return {
        "obs": per_agent(trajectory.obs),
        # The centralized state, repeated for every agent row (on-policy's MPE
        # runner repeats share_obs per agent).
        "cent_obs": per_env(trajectory.global_state),
        "mask": per_env(trajectory.mask),
        "action": per_agent(trajectory.action),
        "old_log_prob": per_agent(trajectory.log_prob),
        "old_value": per_agent(trajectory.value),
        "returns": per_agent(returns),
        "advantage": per_agent(advantages),
        "active": per_agent(trajectory.active_mask),
        "action_mask": (
            per_agent(trajectory.action_mask)
            if trajectory.action_mask.ndim == 4
            else None
        ),
        "actor_h0": hidden(trajectory.actor_h),
        "critic_h0": hidden(trajectory.critic_h),
    }


def policy_objective(log_prob, old_log_prob, advantage, entropy, active, config):
    """`ppo_update`'s actor terms: `(policy_loss, dist_entropy, mean ratio)`.

    The clipped surrogate and the entropy are averaged over active rows under
    `use_policy_active_masks`, else over all rows. The actor minimizes
    `policy_loss - ent_coef * dist_entropy`.
    """
    ratio = jnp.exp(log_prob - old_log_prob)
    surr1 = ratio * advantage
    surr2 = jnp.clip(ratio, 1.0 - config.eps_clip, 1.0 + config.eps_clip) * advantage
    surrogate = jnp.minimum(surr1, surr2)
    if config.use_policy_active_masks:
        return (
            -_masked_mean(surrogate, active),
            _masked_mean(entropy, active),
            ratio.mean(),
        )
    return -surrogate.mean(), entropy.mean(), ratio.mean()


def value_objective(values, old_values, target, active, config):
    """`cal_value_loss` after the ValueNorm step: `target` is the (normalized)
    return. The clipped prediction moves at most `eps_clip` from the stored one;
    the loss is the larger of the clipped and unclipped Huber (or e^2/2) errors,
    averaged over active rows under `use_value_active_masks`. Not yet scaled by
    `val_coef`."""
    if config.use_huber_loss:
        def loss_fn(e):
            return vn_lib.huber_loss(e, config.huber_delta)
    else:
        loss_fn = vn_lib.mse_loss
    loss = loss_fn(target - values)
    if config.use_clipped_value_loss:
        clipped = old_values + jnp.clip(
            values - old_values, -config.eps_clip, config.eps_clip
        )
        loss = jnp.maximum(loss, loss_fn(target - clipped))
    if config.use_value_active_masks:
        return _masked_mean(loss, active)
    return loss.mean()


def unroll(apply_fn, params, inputs, h0, masks):
    """Run a one-step network over sequences: `inputs` `(L, B, ...)`, `masks`
    `(L, B)`, `h0` the hidden state at the first step. Returns the per-step
    outputs `(L, B, ...)`."""

    def _step(h, xs):
        x, mask = xs
        out, h = apply_fn(params, x, h, mask)
        return h, out

    _, outputs = jax.lax.scan(_step, h0, (inputs, masks))
    return outputs


def ppo_update(
    train_state: RMAPPOTrainState,
    rng: jax.Array,
    trajectory: Transition,
    last_value: jnp.ndarray,
    config: RMAPPOConfig,
    discrete: bool,
):
    """One `R_MAPPO.train` call. Returns the new train state and the metrics,
    averaged over epochs x minibatches like on-policy's `train_info`."""
    advantages, returns, explained_variance = prepare_advantages(
        trajectory, last_value, train_state.value_norm, config
    )
    length = config.chunk_length
    n_chunks = chunk_count(trajectory, length)
    minibatch_chunks = n_chunks // config.n_minibatches
    if minibatch_chunks < 1:
        raise ValueError(
            f"{n_chunks} training sequences cannot fill {config.n_minibatches} "
            "minibatches"
        )
    actor_apply = train_state.actor_ts.apply_fn
    critic_apply = train_state.critic_ts.apply_fn

    def _actor_loss(params, batch):
        out = unroll(actor_apply, params, batch["obs"], batch["actor_h0"], batch["mask"])
        log_prob, entropy = evaluate_action(
            out, batch["action"], discrete, batch["action_mask"]
        )
        policy_loss, dist_entropy, ratio = policy_objective(
            log_prob,
            batch["old_log_prob"],
            batch["advantage"],
            entropy,
            batch["active"],
            config,
        )
        loss = policy_loss - config.ent_coef * dist_entropy
        return loss, (policy_loss, dist_entropy, ratio)

    def _critic_loss(params, batch, value_norm):
        values = unroll(
            critic_apply, params, batch["cent_obs"], batch["critic_h0"], batch["mask"]
        )
        target = (
            vn_lib.normalize(value_norm, batch["returns"])
            if config.use_valuenorm
            else batch["returns"]
        )
        value_loss = value_objective(
            values, batch["old_value"], target, batch["active"], config
        )
        return value_loss * config.val_coef, value_loss

    def _minibatch(carry, chunk_ids):
        actor_ts, critic_ts, value_norm = carry
        batch = gather_minibatch(trajectory, advantages, returns, chunk_ids, length)

        (_, (policy_loss, dist_entropy, ratio)), actor_grads = jax.value_and_grad(
            _actor_loss, has_aux=True
        )(actor_ts.params, batch)
        actor_grad_norm = optax.tree.norm(actor_grads)  # pre-clip, as torch's
        actor_ts = actor_ts.apply_gradients(grads=actor_grads)

        if config.use_valuenorm:
            value_norm = vn_lib.update(value_norm, batch["returns"])
        (_, value_loss), critic_grads = jax.value_and_grad(
            _critic_loss, has_aux=True
        )(critic_ts.params, batch, value_norm)
        critic_grad_norm = optax.tree.norm(critic_grads)
        critic_ts = critic_ts.apply_gradients(grads=critic_grads)

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "dist_entropy": dist_entropy,
            "ratio": ratio,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
        }
        return (actor_ts, critic_ts, value_norm), metrics

    def _epoch(carry, _):
        networks, rng = carry
        rng, perm_rng = jax.random.split(rng)
        order = jax.random.permutation(perm_rng, n_chunks)
        minibatches = order[: config.n_minibatches * minibatch_chunks].reshape(
            config.n_minibatches, minibatch_chunks
        )
        networks, metrics = jax.lax.scan(_minibatch, networks, minibatches)
        return (networks, rng), metrics

    (networks, _), metrics = jax.lax.scan(
        _epoch,
        (
            (train_state.actor_ts, train_state.critic_ts, train_state.value_norm),
            rng,
        ),
        None,
        length=config.n_epochs,
    )
    actor_ts, critic_ts, value_norm = networks
    metrics = jax.tree.map(jnp.mean, metrics)
    # The repo's usual loss keys, so generic plots of them keep working.
    metrics["entropy_loss"] = -metrics["dist_entropy"]
    metrics["total_loss"] = (
        metrics["policy_loss"]
        + config.val_coef * metrics["value_loss"]
        + config.ent_coef * metrics["entropy_loss"]
    )
    metrics["explained_variance"] = explained_variance
    norm_mean, norm_var = vn_lib.running_mean_var(value_norm)
    metrics["value_norm_mean"] = norm_mean
    metrics["value_norm_std"] = jnp.sqrt(norm_var)
    return (
        RMAPPOTrainState(actor_ts=actor_ts, critic_ts=critic_ts, value_norm=value_norm),
        metrics,
    )
