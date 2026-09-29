"""Counterfactual-goal credit for the manager (`manager_credit: counterfactual`).

By default the manager's PPO gives every agent's waypoint the same TEAM advantage
(`ppo_update` repeats the env-level GAE advantage over agents). That has two
failure modes: a free rider's useless waypoint is reinforced whenever a teammate
delivers, and under tight coupling an agent waiting at the right box is punished
when its partner never comes. This module gives each agent its own advantage:

    A_i = A_team - beta * c_i
    c_i = mean over k of  Â_φ(x, o with slot i replaced by o'_{i,k})

* `A_team` is the manager's GAE advantage, unchanged.
* `Â_φ(x, o)` is a learned estimate of that advantage from the manager critic's
  input `x` (the team view) and every agent's realized waypoint offset `o`.
* `o'_{i,k}` are counterfactual goals for agent i ALONE: K draws from agent i's
  own rollout policy (`sampled`, Wolpert & Tumer's aristocrat utility), or the
  zero offset (`hold`, their wonderful life utility with "stay where you are"
  as the default action). Teammates keep their actual goals.
* `c_i` is therefore what the team would have scored on average whatever agent i
  had picked; subtracting it leaves agent i's own contribution.

Why this is safe:
  * `c_i` never reads agent i's own sampled goal, and the manager policy factors
    over agents (one Gaussian per agent from the shared actor, independent given
    the state). So `c_i` is a baseline: it changes the variance of agent i's
    policy gradient, never its expectation, however inaccurate `Â_φ` is (the
    action-dependent factorized baseline of Wu et al., ICLR 2018). COMA uses its
    learned critic for the whole advantage, so there a wrong critic biases the
    gradient; here the real returns still come from GAE.
  * `Â_φ` is only queried on joints the policy could have produced: agents sample
    independently, so (o'_i, actual o_-i) is a sample of the joint policy.
  * The model's head is zero-initialized (c = 0, exact team credit until it learns
    goal dependence), and `beta` (a per-batch control-variate coefficient, clipped
    to [0, 1]) shrinks the correction whenever it stops reducing variance.
  * Both critics keep regressing the team return; only the actor's advantage
    changes, so the reward scale is untouched.

What it cannot do: create a signal where no team reward was observed. If no
coalition ever paid off, `Â_φ` stays flat, `beta` -> 0, and this is team credit.

Everything here is pure and jittable; the trainer calls it once per update.
"""

import warnings

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from algorithms.mappo_jax.network import MAPPOCritic, sample_action
from algorithms.simplified_feudal_mappo_jax import waypoints as wp

# How each agent's waypoint is credited.
#   team           — the original: every agent gets the team advantage.
#   counterfactual — the team advantage minus the agent's counterfactual baseline.
MANAGER_CREDITS = ("team", "counterfactual")
# Which counterfactual goals the baseline averages over.
#   sampled — K draws from the agent's own rollout policy (aristocrat utility).
#   hold    — the zero offset: the agent stays where it is (wonderful life utility).
COUNTERFACTUAL_DEFAULTS = ("sampled", "hold")
# Var(c) below this counts as "no correction": beta is then 0, not 0/0.
MIN_CORRECTION_VAR = 1e-8


def validate_manager_credit(config, n_agents: int) -> None:
    """Reject unknown modes; warn when there are no teammates to credit against."""
    if config.manager_credit not in MANAGER_CREDITS:
        raise ValueError(
            f"unknown manager_credit {config.manager_credit!r}; "
            f"expected one of {MANAGER_CREDITS}"
        )
    if config.counterfactual_default not in COUNTERFACTUAL_DEFAULTS:
        raise ValueError(
            f"unknown counterfactual_default {config.counterfactual_default!r}; "
            f"expected one of {COUNTERFACTUAL_DEFAULTS}"
        )
    if config.counterfactual_samples < 1:
        raise ValueError(
            f"counterfactual_samples must be >= 1, got {config.counterfactual_samples}"
        )
    if config.manager_credit == "counterfactual" and n_agents == 1:
        warnings.warn(
            "manager_credit='counterfactual' with one agent: there are no teammates "
            "to remove, so the correction is only a learned state baseline and the "
            "arm should match team credit (useful as the negative control).",
            stacklevel=2,
        )


def adv_model_input(critic_in, offsets):
    """`(..., C + goal_dim*N)` — the manager critic's team view plus every agent's
    goal offset, agent-major. The ONE definition of the advantage model's input
    layout, used by its training and by the correction alike."""
    flat = offsets.reshape(offsets.shape[:-2] + (-1,))
    return jnp.concatenate([critic_in, flat], axis=-1)


def create_adv_model(rng, config, in_dim: int) -> TrainState:
    """The advantage model `Â_φ`: a `MAPPOCritic` body (the manager critic's
    width) with an exactly-zero head, so it outputs 0 until it trains. Same
    optimizer recipe as `mappo_jax.mappo.create_train_state`'s critic, at the
    manager's learning rate (`config` is the manager's `MAPPOConfig`)."""
    model = MAPPOCritic(hidden_dim=2 * config.hidden_dim, head_init_scale=0.0)
    params = model.init(rng, jnp.zeros(in_dim))
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip), optax.adam(config.lr)
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def counterfactual_offsets(
    actor_ts, actor_obs, pos, rng, n_samples, radius, bound, default
):
    """`(T, E, N, K, 2)` counterfactual goal offsets, one set per agent.

    `sampled`: K draws from each agent's OWN policy on its stored manager actor
    input (`actor_obs`, `(T, E, N, d)`). Pass the PRE-update actor, so these are
    draws from the policy that acted. Mapped through `waypoints.waypoint_offset`,
    the same function that measures the actual goal.
    `hold`: the zero offset, i.e. the agent stays where it is (K = 1).
    """
    if default == "hold":
        return jnp.zeros(pos.shape[:-1] + (1,) + pos.shape[-1:], pos.dtype)
    lead = actor_obs.shape[:-1]
    flat = actor_obs.reshape(-1, actor_obs.shape[-1])

    def draw(key):
        action, _ = sample_action(
            key, actor_ts.apply_fn, actor_ts.params, flat, discrete=False
        )
        return action.reshape(lead + action.shape[-1:])  # (T, E, N, 2)

    actions = jax.vmap(draw)(jax.random.split(rng, n_samples))  # (K, T, E, N, 2)
    actions = jnp.moveaxis(actions, 0, -2)  # (T, E, N, K, 2)
    return wp.waypoint_offset(pos[..., None, :], actions, radius, bound)


def own_slot_joint(offsets, cf_i, i):
    """`(..., K, N, 2)` — agent i's counterfactual joints: the actual offsets
    `(..., N, 2)` with ONLY slot i replaced by each of agent i's K counterfactual
    offsets `cf_i` `(..., K, 2)`. A one-hot select (`arange(N) == i`), so slot i
    cannot carry agent i's actual offset and no other slot can change."""
    own = (jnp.arange(offsets.shape[-2]) == i)[:, None]  # (N, 1)
    return jnp.where(own, cf_i[..., None, :], offsets[..., None, :, :])


def correction(adv_ts, critic_in, offsets, cf):
    """`(c, sensitivity)`, each `(T, E, N)`.

    `c[..., i]` = mean over agent i's K counterfactual goals of `Â_φ` on the joint
    where only agent i's goal is replaced: the team advantage the teammates'
    actual goals would have produced whatever agent i had picked.
    `sensitivity[..., i]` = the std of those K values: how much `Â_φ` thinks agent
    i's own goal matters in that state (0 under `hold`, where K = 1).

    `jax.lax.map` over agents keeps memory at O(T*E*K*N) rather than
    O(T*E*K*N^2) for the stacked joints.
    """
    n = offsets.shape[-2]

    def one_agent(i):
        joint = own_slot_joint(offsets, cf[..., i, :, :], i)  # (T, E, K, N, 2)
        x = jnp.broadcast_to(
            critic_in[..., None, :], joint.shape[:-2] + critic_in.shape[-1:]
        )
        return adv_ts.apply_fn(adv_ts.params, adv_model_input(x, joint))  # (T, E, K)

    values = jnp.moveaxis(jax.lax.map(one_agent, jnp.arange(n)), 0, -2)  # (T,E,N,K)
    return values.mean(axis=-1), values.std(axis=-1)


def fit_beta(a_team, c):
    """Scalar in [0, 1]: the control-variate coefficient `Cov(A_team, c) / Var(c)`,
    pooled over (time, env, agent). The variance-minimizing scale of the
    correction; clipped so it never over-corrects or flips sign. 0 when `c` is
    (numerically) constant, e.g. the zero-initialized model, so the update is then
    exactly team credit."""
    a = jnp.broadcast_to(a_team[..., None], c.shape)
    a_c, c_c = a - a.mean(), c - c.mean()
    var = (c_c**2).mean()
    beta = (a_c * c_c).mean() / jnp.maximum(var, MIN_CORRECTION_VAR)
    return jnp.where(var > MIN_CORRECTION_VAR, jnp.clip(beta, 0.0, 1.0), 0.0)


def fit_adv_model(adv_ts, inputs, targets, rng, n_epochs, n_minibatches):
    """Regress `Â_φ` on the unnormalized team advantage. `inputs` `(T, E, D)`,
    `targets` `(T, E)`. Shuffled minibatches, the same cadence as `ppo_update`'s
    critic step. Returns `(adv_ts, mean loss)`."""
    x = inputs.reshape(-1, inputs.shape[-1])
    y = targets.reshape(-1)
    n = x.shape[0]
    mb = max(1, n // n_minibatches)
    n_mb = n // mb

    def epoch(carry, _):
        ts, rng = carry
        rng, key = jax.random.split(rng)
        perm = jax.random.permutation(key, n)

        def step(ts, j):
            ids = jax.lax.dynamic_slice(perm, (j * mb,), (mb,))

            def loss_fn(params):
                return jnp.mean((ts.apply_fn(params, x[ids]) - y[ids]) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(ts.params)
            return ts.apply_gradients(grads=grads), loss

        ts, losses = jax.lax.scan(step, ts, jnp.arange(n_mb))
        return (ts, rng), losses

    (adv_ts, _), losses = jax.lax.scan(epoch, (adv_ts, rng), None, length=n_epochs)
    return adv_ts, losses.mean()


def credit_diagnostics(a_team, c, beta, sensitivity, fitted):
    """Scalars logged per update (prefixed `manager_cf_` by the trainer).

    * `beta` — the applied correction scale. 0 means the model has not learned any
      teammate dependence yet (the update is team credit); read it first.
    * `correction_std` — spread of `c`, the raw correction.
    * `adv_var_ratio` — `Var(A_team - beta*c) / Var(A_team)`: below 1 means the
      correction removed variance from the agents' advantages.
    * `model_ev` — explained variance of the PRE-update `Â_φ(x, o)` on this
      batch's `A_team`: how much of the team advantage the goals predict at all.
    * `goal_sensitivity` — mean std of `Â_φ` over each agent's counterfactual goals.
    """
    a = jnp.broadcast_to(a_team[..., None], c.shape)
    var_a = jnp.var(a) + 1e-8
    return {
        "beta": beta,
        "correction_std": jnp.std(c),
        "adv_var_ratio": jnp.var(a - beta * c) / var_a,
        "model_ev": 1.0 - jnp.var(a_team - fitted) / (jnp.var(a_team) + 1e-8),
        "goal_sensitivity": sensitivity.mean(),
    }
