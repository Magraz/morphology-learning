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

Known property of concat fusion: the worker *can* learn to ignore the goal by
zeroing the goal columns of the first layer, which is precisely the degenerate
solution FeUdal Networks (Vezhnevets et al., 2017) avoids by making the goal
enter through a bias-free bilinear projection (``logits = U(obs) @ phi(g)``,
``phi`` linear without bias) so a zero goal yields no preference and the goal
direction cannot be dropped. If the worker turns out to be goal-blind, that is
the next thing to change here — the rest of this module's interface stays the
same.

``normalize_pooled_goal`` (default True) is a first, cheaper guard on the same
seam, and it was added in response to a MEASURED defect rather than on
principle. What the worker eats is FuN's ``w_t = sum_{i=t-c+1}^{t} g_i`` — a sum
of ``c`` unit vectors. Measured on trained ``mjx_16a_4o`` / ``_trunc``
checkpoints, consecutive goals are 0.95-0.99 collinear, so that sum is not
pooling anything: it is a near-exact ``c``x multiplier on one slowly-varying
direction (``||w_t|| = 9.95`` of a maximum 10 at ``c=10``). Concatenated raw, it
enters the first Dense at ~5x the observation's per-dimension RMS (2.49 vs
0.48), so 16 goal dims own **91.6%** of the layer's preactivation variance
against 40 observation dims — into a Tanh. The worker then spends training
clawing that back (``worker_goal_column_ratio`` decayed 0.94 -> 0.42 over 1e8
steps) and still ends at 66%. On top of the scale, ``||w_t||`` ramps 1 -> c over
the first ``c`` steps of EVERY episode, so the goal block's scale is also
non-stationary in episode phase.

FuN does not hit this because its bilinear fusion makes ``||w_t||`` a pure logit
rescale that never competes with the observation inside a saturating
nonlinearity. Under concatenation the magnitude is not free, so normalize it.
This discards only the magnitude of ``w_t``; the direction — the whole content
of a FuN goal, and the only thing the scale-invariant ``d_cos`` objective
scores — is untouched, which is why it costs the mechanism nothing.

Set ``normalize_pooled_goal=False`` to reproduce the raw-sum behaviour of runs
before this change.

``zero_goal`` is the ABLATION SWITCH, not a fusion knob: it replaces `w_t` with
zeros so the worker's policy is provably independent of the manager, while every
other part of the stack — the manager network, its transition policy gradient,
its critic, its diagnostics, and the worker's per-agent critic head — runs
untouched. It exists because `feudal` at ``intrinsic_coef=0`` is NOT an isolate
of "hierarchy vs flat": it differs from ``algorithm=mappo_jax`` in three ways at
once (goal conditioning, an always-per-agent worker critic head, and manager
training), so a gap against the flat baseline is unattributable. With
``zero_goal=True`` the goal-conditioning term is removed and the other two
remain, which is the rung that says whether a measured feudal deficit comes from
the goals or from everything else. See ``conf/model/feudal_zerogoal.yaml``.

The zeroing is applied to the INPUT, keeping the goal columns of layer 1 in
place, so the parameter tree is shape-identical to a normal feudal run and
checkpoints stay interchangeable. Those columns receive gradient 0 and simply
stay at their init.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from algorithms.feudal_mappo_jax.network import MAPPOActor

# Same epsilon convention as manager.py's `_unit` / `cosine_similarity`:
# a zero goal must stay finite rather than produce NaN. Imported rather
# than redefined so the two cannot drift.
from algorithms.feudal_mappo_jax.manager import _unit


class FeudalWorker(nn.Module):
    """Goal-conditioned low-level policy.

    Args (module attributes):
        action_dim: primitive action dimension (or number of discrete actions).
        goal_dim: dimension of the manager's latent goal vector.
        hidden_dim: hidden width of the shared MLP body.
        discrete: action-space type, matching ``MAPPOActor``.
        goal_embed_dim: if set, the goal is mapped through a **bias-free** linear
            layer of this width before concatenation (FuN's ``phi``); ``None``
            (default) concatenates the goal directly.
        normalize_pooled_goal: L2-normalize the incoming pooled goal ``w_t`` to
            unit length before it meets the observation. Default True; see the
            module docstring for the measured defect this exists to fix. Applied
            BEFORE ``goal_embed_dim`` so ``phi`` sees a unit-scale input.
        zero_goal: ablation — feed a zero goal, so the policy is independent of
            the manager while the rest of the hierarchy still trains. Default
            False. Overrides the two knobs above (they act on a zero vector).

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
    normalize_pooled_goal: bool = True
    zero_goal: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, goal: jnp.ndarray):
        if self.zero_goal:
            # Ablation rung: cut the goal's influence on the POLICY only. Done
            # here (not by skipping the concat) so the kernel keeps its goal
            # columns and the param tree stays shape-identical to a live feudal
            # run — those columns just receive zero gradient. `jnp.zeros_like`
            # rather than dropping the term, so the shape/broadcast path below
            # is exercised identically in both settings.
            goal = jnp.zeros_like(goal)

        if self.normalize_pooled_goal:
            # Direction only. `w_t` is a sum of c unit goals whose magnitude is
            # an artifact of how collinear they happen to be (and of how many
            # ring slots are written yet), not a directive the manager chose --
            # its own d_cos objective cannot even see it. Left raw it dominates
            # the first layer; see the module docstring for the measurement.
            goal = _unit(goal)

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
    normalize_pooled_goal: bool = True,
    zero_goal: bool = False,
):
    """Build a `FeudalWorker` and its initial params. Returns ``(module, params)``."""
    worker = FeudalWorker(
        action_dim=action_dim,
        goal_dim=goal_dim,
        hidden_dim=hidden_dim,
        discrete=discrete,
        goal_embed_dim=goal_embed_dim,
        normalize_pooled_goal=normalize_pooled_goal,
        zero_goal=zero_goal,
    )
    params = worker.init(rng, jnp.zeros(obs_dim), jnp.zeros(goal_dim))
    return worker, params
