"""Actor and critic of on-policy's `R_Actor` / `R_Critic`, in flax.

Ported module by module from `onpolicy/algorithms/utils`:
- `MLPBase` (mlp.py): an optional input LayerNorm, then `1 + layer_N` blocks of
  Dense -> activation -> LayerNorm. Dense weights are orthogonal with the gain
  of the activation (`calculate_gain('relu') = sqrt(2)`, `'tanh' = 5/3`), and
  biases are 0.
- `GRULayer` (rnn.py): torch's `nn.GRU` (gate order r, z, n; separate input and
  hidden biases; `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`), orthogonal
  weights and zero biases, then a LayerNorm. The hidden state is multiplied by
  the episode mask before the step, which is exactly what `RNNLayer.forward`
  does on both its single-step and its segmented sequence branch.
- the action heads (act.py / distributions.py): a Dense with orthogonal init at
  gain `gain` (0.01). Discrete: categorical logits, with unavailable actions set
  to -1e10. Continuous: a diagonal Gaussian whose log-std is a learned,
  state-independent parameter initialized at 0 (std 1) and not clamped.
- the critic head (`R_Critic.v_out`): a Dense with orthogonal init at gain 1.

LayerNorm uses epsilon 1e-5, torch's default (flax's is 1e-6).

Both modules map one step: `(x, h, mask) -> (out, h)`. Sequences are a
`lax.scan` over this step (see `mappo.py`). For the feedforward policy `h` is a
placeholder that passes through untouched.
"""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from algorithms.mappo_jax.network import (
    _LOG_2PI,
    _categorical_entropy,
    _categorical_log_prob,
    _gaussian_log_prob,
)

LAYER_NORM_EPS = 1e-5  # torch.nn.LayerNorm's default
UNAVAILABLE_LOGIT = -1e10  # on-policy's Categorical: x[available_actions == 0]


def _activation(use_ReLU: bool):
    """The activation and the orthogonal gain torch's `calculate_gain` gives it."""
    if use_ReLU:
        return nn.relu, math.sqrt(2.0)
    return nn.tanh, 5.0 / 3.0


def _dense(features: int, gain: float, name: str):
    return nn.Dense(
        features,
        kernel_init=nn.initializers.orthogonal(gain),
        bias_init=nn.initializers.zeros,
        name=name,
    )


class MLPBase(nn.Module):
    hidden_size: int
    layer_N: int
    use_ReLU: bool
    use_feature_normalization: bool

    @nn.compact
    def __call__(self, x):
        if self.use_feature_normalization:
            x = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="feature_norm")(x)
        activation, gain = _activation(self.use_ReLU)
        # fc1, then layer_N hidden blocks; each initialized independently, as in
        # MLPLayer's ModuleList (the get_clones copy there is commented out).
        for i in range(1 + self.layer_N):
            x = _dense(self.hidden_size, gain, f"fc_{i}")(x)
            x = activation(x)
            x = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name=f"fc_norm_{i}")(x)
        return x


class GRUCell(nn.Module):
    """One layer of torch's `nn.GRU`, with torch's parameter layout.

    `weight_ih` is `(in, 3H)` and `weight_hh` is `(H, 3H)`, the transposes of
    torch's `weight_ih_l{k}` / `weight_hh_l{k}`, with the gates stacked in
    torch's order (r, z, n). Orthogonal init over the stacked matrix, as
    `nn.init.orthogonal_` gives the torch parameter.
    """

    hidden_size: int

    @nn.compact
    def __call__(self, h, x):
        size = self.hidden_size
        orthogonal = nn.initializers.orthogonal(1.0)
        weight_ih = self.param("weight_ih", orthogonal, (x.shape[-1], 3 * size))
        weight_hh = self.param("weight_hh", orthogonal, (size, 3 * size))
        bias_ih = self.param("bias_ih", nn.initializers.zeros, (3 * size,))
        bias_hh = self.param("bias_hh", nn.initializers.zeros, (3 * size,))
        i_r, i_z, i_n = jnp.split(x @ weight_ih + bias_ih, 3, axis=-1)
        h_r, h_z, h_n = jnp.split(h @ weight_hh + bias_hh, 3, axis=-1)
        r = jax.nn.sigmoid(i_r + h_r)
        z = jax.nn.sigmoid(i_z + h_z)
        n = jnp.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h


class GRULayer(nn.Module):
    """on-policy's `RNNLayer`: a `recurrent_N`-layer GRU plus a LayerNorm.

    One step. `h` is `(batch, recurrent_N, hidden_size)` and `mask` `(batch,)`;
    the hidden state is zeroed where `mask` is 0 before the step.
    """

    hidden_size: int
    recurrent_N: int

    @nn.compact
    def __call__(self, x, h, mask):
        h = h * mask[:, None, None]
        layers = []
        for layer in range(self.recurrent_N):
            x = GRUCell(self.hidden_size, name=f"gru_{layer}")(h[:, layer], x)
            layers.append(x)
        out = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="rnn_norm")(x)
        return out, jnp.stack(layers, axis=1)


class _Trunk(nn.Module):
    """`MLPBase`, then the GRU when the policy is recurrent."""

    hidden_size: int
    layer_N: int
    use_ReLU: bool
    use_feature_normalization: bool
    use_recurrent_policy: bool
    recurrent_N: int

    @nn.compact
    def __call__(self, x, h, mask):
        x = MLPBase(
            self.hidden_size,
            self.layer_N,
            self.use_ReLU,
            self.use_feature_normalization,
            name="base",
        )(x)
        if self.use_recurrent_policy:
            x, h = GRULayer(self.hidden_size, self.recurrent_N, name="rnn")(
                x, h, mask
            )
        return x, h


class Actor(nn.Module):
    """`R_Actor`. Returns `(logits, h)` for a discrete policy and
    `((mean, log_std), h)` for a continuous one, `log_std` broadcast to `mean`."""

    action_dim: int
    discrete: bool
    hidden_size: int = 64
    layer_N: int = 1
    use_ReLU: bool = True
    use_feature_normalization: bool = True
    use_recurrent_policy: bool = True
    recurrent_N: int = 1
    gain: float = 0.01

    @nn.compact
    def __call__(self, obs, h, mask):
        x, h = _Trunk(
            self.hidden_size,
            self.layer_N,
            self.use_ReLU,
            self.use_feature_normalization,
            self.use_recurrent_policy,
            self.recurrent_N,
            name="trunk",
        )(obs, h, mask)
        out = _dense(self.action_dim, self.gain, "action_out")(x)
        if self.discrete:
            return out, h
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        return (out, jnp.broadcast_to(log_std, out.shape)), h


class Critic(nn.Module):
    """`R_Critic` (without PopArt). Returns `(value, h)`, `value` one per row."""

    hidden_size: int = 64
    layer_N: int = 1
    use_ReLU: bool = True
    use_feature_normalization: bool = True
    use_recurrent_policy: bool = True
    recurrent_N: int = 1

    @nn.compact
    def __call__(self, cent_obs, h, mask):
        x, h = _Trunk(
            self.hidden_size,
            self.layer_N,
            self.use_ReLU,
            self.use_feature_normalization,
            self.use_recurrent_policy,
            self.recurrent_N,
            name="trunk",
        )(cent_obs, h, mask)
        value = _dense(1, 1.0, "v_out")(x)
        return value[..., 0], h


def make_networks(config, action_dim: int, discrete: bool):
    """The actor and critic modules a config describes."""
    trunk = dict(
        hidden_size=config.hidden_size,
        layer_N=config.layer_N,
        use_ReLU=config.use_ReLU,
        use_feature_normalization=config.use_feature_normalization,
        use_recurrent_policy=config.use_recurrent_policy,
        recurrent_N=config.recurrent_N,
    )
    actor = Actor(action_dim=action_dim, discrete=discrete, gain=config.gain, **trunk)
    return actor, Critic(**trunk)


# ---------------------------------------------------------------------------
# Distributions (on-policy's FixedCategorical / FixedNormal)
# ---------------------------------------------------------------------------


def _masked_logits(logits, action_mask):
    if action_mask is None:
        return logits
    return jnp.where(action_mask == 0, UNAVAILABLE_LOGIT, logits)


def sample_action(rng, out, discrete: bool, deterministic: bool, action_mask=None):
    """Sample (or take the mode of) the policy at `out`, the actor's output.
    Returns `(action, log_prob)`; the log-prob of a continuous action is summed
    over action dims (`FixedNormal.log_probs`)."""
    if discrete:
        logits = _masked_logits(out, action_mask)
        if deterministic:
            action = jnp.argmax(logits, axis=-1)
        else:
            action = jax.random.categorical(rng, logits, axis=-1)
        return action, _categorical_log_prob(logits, action)
    mean, log_std = out
    if deterministic:
        action = mean
    else:
        action = mean + jnp.exp(log_std) * jax.random.normal(rng, mean.shape)
    return action, _gaussian_log_prob(action, mean, log_std)


def evaluate_action(out, action, discrete: bool, action_mask=None):
    """`(log_prob, entropy)` of stored actions under the actor's output `out`."""
    if discrete:
        logits = _masked_logits(out, action_mask)
        return _categorical_log_prob(logits, action), _categorical_entropy(logits)
    mean, log_std = out
    log_prob = _gaussian_log_prob(action, mean, log_std)
    # Normal entropy summed over action dims (`FixedNormal.entropy`), per row.
    entropy = (0.5 * (1.0 + _LOG_2PI) + log_std).sum(axis=-1)
    return log_prob, entropy
