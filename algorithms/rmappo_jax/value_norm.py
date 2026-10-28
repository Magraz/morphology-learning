"""on-policy's `ValueNorm` and value-loss helpers, as pure functions.

`ValueNorm` (onpolicy/utils/valuenorm.py) keeps a debiased exponential moving
average of the returns' mean and mean square. The critic regresses
`normalize(returns)`, and its outputs are `denormalize`d wherever a value in
reward units is needed (GAE and the advantages). With the defaults used by
`R_MAPPO`: `beta` 0.99999, `epsilon` 1e-5, `per_element_update` False, and the
variance clamped to at least 1e-2.

Note the start: before the first update the debiasing term is 0, so the mean
is 0 and the variance is the 1e-2 floor, i.e. `denormalize(v) = 0.1 * v`.
on-policy denormalizes its first rollout's values with exactly that, and so
does this port.
"""

from typing import NamedTuple

import jax.numpy as jnp

BETA = 0.99999
EPSILON = 1e-5
VAR_MIN = 1e-2


class ValueNormState(NamedTuple):
    running_mean: jnp.ndarray  # ()
    running_mean_sq: jnp.ndarray  # ()
    debiasing_term: jnp.ndarray  # ()


def init_value_norm() -> ValueNormState:
    zero = jnp.zeros((), jnp.float32)
    return ValueNormState(zero, zero, zero)


def running_mean_var(state: ValueNormState):
    debias = jnp.maximum(state.debiasing_term, EPSILON)
    mean = state.running_mean / debias
    mean_sq = state.running_mean_sq / debias
    return mean, jnp.maximum(mean_sq - mean**2, VAR_MIN)


def update(state: ValueNormState, x) -> ValueNormState:
    """One EMA step from a batch of returns (`ValueNorm.update`; every element
    of `x` counts, inactive rows included)."""
    return ValueNormState(
        running_mean=state.running_mean * BETA + jnp.mean(x) * (1.0 - BETA),
        running_mean_sq=state.running_mean_sq * BETA
        + jnp.mean(x**2) * (1.0 - BETA),
        debiasing_term=state.debiasing_term * BETA + (1.0 - BETA),
    )


def normalize(state: ValueNormState, x):
    mean, var = running_mean_var(state)
    return (x - mean) / jnp.sqrt(var)


def denormalize(state: ValueNormState, x):
    mean, var = running_mean_var(state)
    return x * jnp.sqrt(var) + mean


def huber_loss(e, delta):
    """onpolicy/utils/util.py `huber_loss`: `e^2/2` inside `delta`, linear outside."""
    inside = jnp.abs(e) <= delta
    return jnp.where(inside, e**2 / 2, delta * (jnp.abs(e) - delta / 2))


def mse_loss(e):
    """onpolicy/utils/util.py `mse_loss`: note the 1/2."""
    return e**2 / 2
