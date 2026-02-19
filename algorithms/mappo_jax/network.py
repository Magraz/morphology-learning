import numpy as np
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


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
    """Centralized critic. 2-layer MLP with Tanh, hidden_dim=256."""

    hidden_dim: int = 256

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
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return jnp.squeeze(value, axis=-1)


# ---------------------------------------------------------------------------
# Pure helper functions for action sampling / evaluation (JIT-friendly)
# ---------------------------------------------------------------------------


def get_pi_discrete(
    actor_apply_fn,
    params,
    obs: jnp.ndarray,
    action_mask: Optional[jnp.ndarray] = None,
) -> distrax.Categorical:
    """Build a masked Categorical distribution from actor output."""
    logits = actor_apply_fn(params, obs)
    if action_mask is not None:
        logits = logits + (1.0 - action_mask) * (-1e9)
    return distrax.Categorical(logits=logits)


def get_pi_continuous(
    actor_apply_fn,
    params,
    obs: jnp.ndarray,
) -> distrax.MultivariateNormalDiag:
    """Build a Normal distribution from actor output."""
    means, log_std = actor_apply_fn(params, obs)
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_std))


def sample_action(
    rng: jax.random.PRNGKey,
    actor_apply_fn,
    params,
    obs: jnp.ndarray,
    discrete: bool,
    deterministic: bool = False,
    action_mask: Optional[jnp.ndarray] = None,
):
    """Sample action and log_prob from the actor. Pure function."""
    if discrete:
        pi = get_pi_discrete(actor_apply_fn, params, obs, action_mask)
        if deterministic:
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
    else:
        pi = get_pi_continuous(actor_apply_fn, params, obs)
        if deterministic:
            action = pi.loc
        else:
            action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)

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
        pi = get_pi_discrete(actor_apply_fn, params, obs, action_mask)
        log_prob = pi.log_prob(action)
        entropy = pi.entropy()
    else:
        pi = get_pi_continuous(actor_apply_fn, params, obs)
        log_prob = pi.log_prob(action)
        entropy = pi.entropy()

    return log_prob, entropy
