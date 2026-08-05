"""Feudal worker network: a goal-conditioned actor.

The worker is the low-level policy of the feudal hierarchy. It sees its own
local observation *plus* a latent goal vector `g` produced by the manager
(``manager.py``) and emits a primitive action — the same action space the flat
MAPPO actor drives, so everything downstream of the policy (env stepping,
PPO update, logging) is unchanged.

Fusion is plain concatenation ``[obs, goal] -> MLP`` for now. The network body
is the flat :class:`MAPPOActor` reused verbatim (same 2-layer Tanh MLP, same
orthogonal init, same continuous/discrete head contract), so the worker returns
exactly what ``sample_action`` / ``evaluate_action`` in ``network.py`` expect —
logits when ``discrete``, ``(mean, log_std)`` otherwise.

Known property of concat fusion, relevant once training is wired up: the worker
*can* learn to ignore the goal by zeroing the goal columns of the first layer,
which is precisely the degenerate solution FeUdal Networks (Vezhnevets et al.,
2017) avoids by making the goal enter through a bias-free bilinear projection
(``logits = U(obs) @ phi(g)``, ``phi`` linear without bias) so a zero goal
yields no preference and the goal direction cannot be dropped. If the worker
turns out to be goal-blind, that is the first thing to change here — the rest
of this module's interface stays the same.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from algorithms.feudal_mappo_jax.network import MAPPOActor


class FeudalWorker(nn.Module):
    """Goal-conditioned low-level policy.

    Args (module attributes):
        action_dim: primitive action dimension (or number of discrete actions).
        goal_dim: dimension of the manager's latent goal vector.
        hidden_dim: hidden width of the shared MLP body.
        discrete: action-space type, matching ``MAPPOActor``.
        goal_embed_dim: if set, the goal is first mapped through a **bias-free**
            linear layer of this width before concatenation (FuN's ``phi``);
            ``None`` (default) concatenates the raw goal.

    Call:
        ``__call__(obs, goal) ->`` logits ``(..., action_dim)`` if ``discrete``,
        else ``(mean, log_std)``.

        ``obs`` is ``(..., obs_dim)`` and ``goal`` is ``(..., goal_dim)``. The
        goal broadcasts over ``obs``'s leading axes, so a single team goal
        ``(n_envs, goal_dim)`` and a per-agent goal ``(n_envs, n_agents,
        goal_dim)`` both work against ``obs`` of ``(n_envs, n_agents, obs_dim)``.
    """

    action_dim: int
    goal_dim: int
    hidden_dim: int = 128
    discrete: bool = True
    goal_embed_dim: Optional[int] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, goal: jnp.ndarray):
        if self.goal_embed_dim is not None:
            # Bias-free, as in FuN: a zero goal must contribute nothing.
            goal = nn.Dense(
                self.goal_embed_dim,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(1.0),
            )(goal)

        goal = _broadcast_goal(goal, obs)
        x = jnp.concatenate([obs, goal], axis=-1)

        return MAPPOActor(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            discrete=self.discrete,
        )(x)


def _broadcast_goal(goal: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
    """Line the goal's leading axes up with the observation's.

    A manager that emits one goal per env gives ``(n_envs, goal_dim)`` while the
    worker runs per agent on ``(n_envs, n_agents, obs_dim)``; insert the missing
    axes and expand so the concatenation is well-defined.

    My Comment: This is needed to concatenate the goal with the observation.
    """
    while goal.ndim < obs.ndim:
        goal = jnp.expand_dims(goal, axis=-2)
    return jnp.broadcast_to(goal, obs.shape[:-1] + goal.shape[-1:])


def bind_goal(worker_apply_fn, goal: jnp.ndarray):
    """Freeze `goal` into a worker's apply fn, yielding the flat actor signature.

    ``sample_action`` / ``evaluate_action`` in ``network.py`` call
    ``actor_apply_fn(params, obs)``; the worker needs a second argument. Wrapping
    it here means the sampling/eval path is reused unmodified rather than forked
    for the feudal stack::

        actions, log_probs = sample_action(
            rng, bind_goal(worker_ts.apply_fn, goals), worker_ts.params,
            obs, discrete,
        )
    """

    def apply(params, obs):
        return worker_apply_fn(params, obs, goal)

    return apply


def init_worker(
    rng: jax.Array,
    obs_dim: int,
    goal_dim: int,
    action_dim: int,
    hidden_dim: int,
    discrete: bool,
    goal_embed_dim: Optional[int] = None,
):
    """Build a `FeudalWorker` and its initial params. Returns ``(module, params)``."""
    worker = FeudalWorker(
        action_dim=action_dim,
        goal_dim=goal_dim,
        hidden_dim=hidden_dim,
        discrete=discrete,
        goal_embed_dim=goal_embed_dim,
    )
    params = worker.init(rng, jnp.zeros(obs_dim), jnp.zeros(goal_dim))
    return worker, params
