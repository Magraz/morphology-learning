"""Flax actor/critic mirroring ``mappo_vanilla.networks``.

Same architectures and initialization as the torch versions: 2-layer Tanh MLPs
with orthogonal init (sqrt(2) hidden, 0.01 actor head, 1.0 critic head), a
state-independent learned ``log_action_std`` initialized at -0.5 and clamped to
[-5, 2] for continuous actions. Distributions are implemented inline (diagonal
Gaussian / categorical) instead of pulling in distrax; log-probs and entropies
are summed over action dims exactly like the torch ``Normal`` path.
"""

import math
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
_LOG_2PI = math.log(2.0 * math.pi)


class MAPPOActor(nn.Module):
    """Decentralized actor. 2-layer MLP with Tanh, orthogonal init."""

    action_dim: int
    hidden_dim: int = 128
    discrete: bool = True

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(obs)
        x = nn.tanh(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        action_params = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)

        if self.discrete:
            return action_params  # logits
        else:
            log_std = self.param(
                "log_action_std",
                nn.initializers.constant(-0.5),
                (self.action_dim,),
            )
            return action_params, log_std


class MAPPOCritic(nn.Module):
    """Centralized critic. 2-layer MLP with Tanh.

    `n_outputs` > 1 gives a per-agent value head: one value per agent from the
    same global state, needed when the env emits per-agent rewards (difference
    rewards), since each agent then has its own return to predict. `n_outputs=1`
    (default) keeps the original single-scalar critic exactly.
    """

    hidden_dim: int = 256
    n_outputs: int = 1

    @nn.compact
    def __call__(self, global_state: jnp.ndarray):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(global_state)
        x = nn.tanh(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        value = nn.Dense(
            self.n_outputs,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        if self.n_outputs == 1:
            return jnp.squeeze(value, axis=-1)
        return value  # (..., n_outputs) — one value per agent


# ---------------------------------------------------------------------------
# Distribution math (diagonal Gaussian / masked categorical), JIT-friendly
# ---------------------------------------------------------------------------


def _gaussian_log_prob(action, mean, log_std):
    """Sum of per-dim Normal log-probs (torch's `dist.log_prob(a).sum(-1)`)."""
    var = jnp.exp(2.0 * log_std)
    per_dim = -0.5 * ((action - mean) ** 2 / var) - log_std - 0.5 * _LOG_2PI
    return per_dim.sum(axis=-1)


def _gaussian_entropy(log_std, batch_shape):
    """Sum of per-dim Normal entropies, broadcast to the batch."""
    ent = (0.5 * (1.0 + _LOG_2PI) + log_std).sum()
    return jnp.full(batch_shape, ent)


def _masked_logits(logits, action_mask):
    if action_mask is not None:
        logits = logits + (1.0 - action_mask) * (-1e9)
    return logits


def _categorical_log_prob(logits, action):
    log_p = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_p, action[..., None], axis=-1).squeeze(-1)


def _categorical_entropy(logits):
    log_p = jax.nn.log_softmax(logits, axis=-1)
    return -(jnp.exp(log_p) * log_p).sum(axis=-1)


def sample_action(
    rng: jax.Array,
    actor_apply_fn,
    params,
    obs: jnp.ndarray,
    discrete: bool,
    deterministic: bool = False,
    action_mask: Optional[jnp.ndarray] = None,
):
    """Sample action and log_prob from the actor. Pure function."""
    if discrete:
        logits = _masked_logits(actor_apply_fn(params, obs), action_mask)
        if deterministic:
            action = jnp.argmax(logits, axis=-1)
        else:
            action = jax.random.categorical(rng, logits, axis=-1)
        log_prob = _categorical_log_prob(logits, action)
    else:
        mean, log_std = actor_apply_fn(params, obs)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            action = mean
        else:
            noise = jax.random.normal(rng, mean.shape)
            action = mean + jnp.exp(log_std) * noise
        log_prob = _gaussian_log_prob(action, mean, log_std)

    return action, log_prob


def evaluate_action(
    actor_apply_fn,
    params,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    discrete: bool,
    action_mask: Optional[jnp.ndarray] = None,
):
    """Evaluate log_prob and entropy for given actions. Pure function."""
    if discrete:
        logits = _masked_logits(actor_apply_fn(params, obs), action_mask)
        log_prob = _categorical_log_prob(logits, action)
        entropy = _categorical_entropy(logits)
    else:
        mean, log_std = actor_apply_fn(params, obs)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_prob = _gaussian_log_prob(action, mean, log_std)
        entropy = _gaussian_entropy(log_std, log_prob.shape)

    return log_prob, entropy
