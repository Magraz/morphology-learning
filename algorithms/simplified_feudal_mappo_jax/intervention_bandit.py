"""Learned scheduler for the training forks (`intervention_scheduler: bandit`).

Plan: `plans/intervention_contextual_bandit_2026-10-16.md`. At every candidate
main decision of the adaptive fork collector (`adaptive_collection.py`) a focal
agent i is drawn uniformly, and a one-step contextual bandit decides whether to
fork the window around it:

* input `x`: exactly agent i's manager actor input from the actual manager call,
  with no other state (the same `m_actor_in` row PPO trains on);
* output `q(x)`: the predicted utility of intervening. Skipping is defined to be
  worth 0, so the greedy action is "intervene iff q(x) > 0" (ties skip);
* exploration: with probability epsilon the greedy action is replaced by a
  uniformly random skip/intervene;
* target, for every attempted fork (`paired_target`): the discounted raw team
  return of the fork window minus that of the main window it was cut from,
  minus `zero_reward_penalty` when every raw fork step earned exactly 0, minus
  `cost_coef` per fork. A failed placement runs no fork and is labelled 0.

The bandit never reads the manager's sampled action, so the forks it selects
are on-policy for the manager's PPO (selection depends on the state only).
Nothing here enters either PPO policy's gradient, and the bandit is never
consulted at evaluation. `intervention_scheduler: random` is the matched
control: it proposes a fork with a fixed probability, and rate 0 never forks.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

# `intervention_scheduler`: "fixed" is the original fork at every
# `intervention_interval`-th decision; the other two run the adaptive collector.
SCHEDULERS = ("fixed", "bandit", "random")
ADAPTIVE_SCHEDULERS = ("bandit", "random")

# The bandit's optimizer and loss, in one place (the plan's defaults). One pass
# over each update's labelled attempts, in shuffled minibatches.
BATCH_SIZE = 128
N_EPOCHS = 1
HUBER_DELTA = 1.0
GRAD_CLIP = 0.5


class GainNet(nn.Module):
    """Two tanh layers and an unconstrained scalar output, zero-initialized so
    every prediction starts at exactly 0 (and the greedy action at skip)."""

    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(nn.Dense(self.hidden)(x))
        x = jnp.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(1, kernel_init=nn.initializers.zeros)(x)[..., 0]


def create_bandit_state(rng, input_dim: int, hidden_dim: int, lr: float) -> TrainState:
    net = GainNet(hidden=hidden_dim)
    params = net.init(rng, jnp.zeros((input_dim,)))
    tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(lr))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


def epsilon_greedy(q, u_explore, u_action, epsilon: float):
    """`(intervene, explored, propensity)` per candidate.

    Greedy: intervene iff `q > 0`. With probability epsilon (`u_explore <
    epsilon`) the action is replaced by a fair coin (`u_action < 0.5`).
    `propensity` is the probability that this candidate intervenes:
    `(1 - epsilon) * [q > 0] + epsilon / 2`.
    """
    greedy = q > 0
    explored = u_explore < epsilon
    intervene = jnp.where(explored, u_action < 0.5, greedy)
    propensity = (1.0 - epsilon) * greedy.astype(jnp.float32) + 0.5 * epsilon
    return intervene, explored, propensity


def random_schedule(u, rate: float):
    """`(intervene, propensity)`: intervene with probability `rate`."""
    return u < rate, jnp.full(u.shape, rate, jnp.float32)


def window_return(rewards, gamma: float):
    """Discounted sum over the trailing step axis: `sum_k gamma^k r[..., k]`."""
    discount = gamma ** jnp.arange(rewards.shape[-1], dtype=jnp.float32)
    return (rewards * discount).sum(-1)


def paired_target(main_rewards, fork_rewards, gamma, zero_reward_penalty, cost_coef):
    """The bandit's label for executed forks, from the raw per-step team rewards
    `(..., H)` of the fork and of the main window it was cut from.

    Returns `(target, g_main, g_fork, zero_fork)`. `zero_fork` is True when every
    raw fork step is exactly 0 (a negative shaping step counts as a reward; steps
    after a termination are 0). The cost is per executed fork: every launched
    fork window counts all its H simulator calls, so `cost * steps / H` is
    `cost`.
    """
    g_main = window_return(main_rewards, gamma)
    g_fork = window_return(fork_rewards, gamma)
    zero_fork = jnp.all(fork_rewards == 0, axis=-1)
    target = (
        g_fork - g_main - zero_reward_penalty * zero_fork.astype(jnp.float32) - cost_coef
    )
    return target, g_main, g_fork, zero_fork


def fit_bandit(state: TrainState, x, y, count, rng):
    """`N_EPOCHS` passes over the first `count` rows of `(x, y)` (the labelled
    attempts; the rest is padding), in shuffled minibatches of `BATCH_SIZE`,
    with a Huber loss.

    A minibatch with no labelled row is skipped entirely: Adam moves parameters
    on a zero gradient (its first moment and step counter), so with no labels
    the state is returned unchanged. Returns `(state, mean loss over labelled
    rows, number of optimizer steps)`; the loss is measured before each step.
    """
    capacity = x.shape[0]
    n_mb = -(-capacity // BATCH_SIZE)
    pad = n_mb * BATCH_SIZE - capacity
    row = jnp.arange(capacity)
    labelled = row < count

    def loss_fn(params, xb, yb, wb):
        pred = state.apply_fn(params, xb)
        loss = optax.huber_loss(pred, yb, delta=HUBER_DELTA)
        return (loss * wb).sum() / jnp.maximum(wb.sum(), 1.0)

    def epoch(carry, key):
        st, loss_sum, steps = carry
        # A random order of the labelled rows first, padding last.
        order = jnp.argsort(jnp.where(labelled, jax.random.uniform(key, (capacity,)), 2.0))
        order = jnp.concatenate([order, jnp.zeros((pad,), order.dtype)])
        valid = jnp.concatenate([labelled[order[:capacity]], jnp.zeros((pad,), bool)])
        valid = valid & (jnp.arange(n_mb * BATCH_SIZE) < count)

        def minibatch(carry, i):
            st, loss_sum, steps = carry
            idx = jax.lax.dynamic_slice(order, (i * BATCH_SIZE,), (BATCH_SIZE,))
            w = jax.lax.dynamic_slice(valid, (i * BATCH_SIZE,), (BATCH_SIZE,)).astype(
                jnp.float32
            )
            xb, yb = x[idx], y[idx]

            def step(st):
                loss, grads = jax.value_and_grad(loss_fn)(st.params, xb, yb, w)
                return st.apply_gradients(grads=grads), loss

            st, loss = jax.lax.cond(
                w.sum() > 0, step, lambda st: (st, jnp.zeros(())), st
            )
            return (st, loss_sum + loss * w.sum(), steps + (w.sum() > 0)), None

        carry, _ = jax.lax.scan(minibatch, (st, loss_sum, steps), jnp.arange(n_mb))
        return carry, None

    (state, loss_sum, steps), _ = jax.lax.scan(
        epoch,
        (state, jnp.zeros(()), jnp.zeros((), jnp.int32)),
        jax.random.split(rng, N_EPOCHS),
    )
    mean_loss = loss_sum / jnp.maximum(count * N_EPOCHS, 1)
    return state, mean_loss, steps
