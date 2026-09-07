"""Feudal worker network: a goal-conditioned actor.

The worker is the low-level policy of the feudal hierarchy. It sees its own
local observation *plus* a latent goal vector `g` produced by the manager
(``manager.py``) and emits a primitive action — the same action space the flat
MAPPO actor drives, so everything downstream of the policy (env stepping,
PPO update, logging) is unchanged.

``worker_fusion`` selects how the goal reaches the policy: ``"concat"``
(``[obs, goal] -> MLP``, the original and still the default) or ``"film"``
(zero-initialized Feature-wise Linear Modulation, :class:`FiLM`). Either way the
network body is the flat :class:`MAPPOActor` reused verbatim (same 2-layer Tanh
MLP, same orthogonal init, same continuous/discrete head contract), so the worker
returns exactly what ``sample_action`` / ``evaluate_action`` in ``network.py``
expect — logits when ``discrete``, ``(mean, log_std)`` otherwise. FiLM rides an
optional ``modulate`` hook on ``MAPPOActor``; passing ``None`` leaves that module
byte-identical to the pre-hook version.

**Why FiLM exists — MEASURED, not anticipated.** The goal-dependence probe
(``goal_dependence_probe.py``, all 20 trained concat arms, 2026-09-06) found
that permuting the manager's goals across agents changes the return by *nothing*
while **zeroing** them *improves* it on 15 of 16 non-control arms. So under
concat the goal is a net-harmful perturbation whose direction the worker never
learned to use — and the manager is not at fault (its goals are agent-specific
and state-conditioned; see CLAUDE.md).

Two STRUCTURAL properties of concatenation cause that, and neither is about
scale (scale is spent — ``normalize_pooled_goal`` already took the goal block
from 59-78% of layer-1 variance down to a measured 14-19%):

1. **The default is influence, not no-influence.** Orthogonal init makes the
   goal columns live from step 0 — measured at init on real observations,
   swapping the goal moves concat's action mean by 1.40e-03 against an action
   scale of 1.65e-03, i.e. ~85% of the untrained policy's output is goal-driven
   before any learning. Goal-*agnosticism* is therefore something the worker
   must actively learn, and the probe says it never finishes. This is exactly
   the degeneracy FeUdal Networks (Vezhnevets et al., 2017) avoids with a
   bias-free bilinear projection (``logits = U(obs) @ phi(g)``), under which an
   uninformative goal costs nothing by construction. FiLM's zero-init recovers
   that property: measured 0.000e+00 for both a swapped and a zeroed goal.
2. **Concat can only TRANSLATE the policy.** ``d z1/d obs = W_obs`` contains no
   goal term whatsoever, so a concatenated goal cannot change *which*
   observation features matter — only where the operating point sits. A
   directive like "work the box on your left, ignore the one behind you" needs
   the goal to modulate the obs->action map, which is what FiLM's multiplicative
   gain provides.

⚠ FuN's bilinear was deliberately NOT used here: its "zero goal yields no
preference" property is stated for *discrete logits*, but the MJX envs are
continuous force control, where a zero goal would produce a zero mean action —
"stand still", a specific and consequential action rather than neutrality. For
the discrete arms (``macro_mjx``, SMAX) it is the right shape; for continuous it
would have to be residual on the mean.

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


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (Perez et al. 2018), zero-initialized.

    ``h <- (1 + gamma(g)) * h + beta(g)``, feature-wise, with ``gamma``/``beta``
    linear in the goal. Used by ``FeudalWorker(worker_fusion="film")`` to let the
    manager's directive act as a set of **gains on how the observation is read**
    rather than as one more observation channel.

    Two properties do the work, and both are structural rather than incidental:

    **Zero-init kernels => identity at step 0.** ``gamma = beta = 0``, so the
    worker begins bit-identical to the flat ``mappo_jax`` actor and any goal
    influence has to be *earned*. Under concatenation the opposite holds: the
    goal columns are live at orthogonal init (measured: ~14-19% of the worker's
    layer-1 variance on trained checkpoints), so goal-*agnosticism* is what the
    worker would have to learn — and measurement says it never finishes, which
    is why zeroing the goal at eval IMPROVES return on 15 of 16 trained arms.
    The ``1 +`` is load-bearing: a bare ``gamma * h`` with a zero-init kernel
    would output zeros and kill the forward pass. Same trick as adaLN-Zero /
    ReZero / LoRA's zero-init B.
    It does not stall learning: ``dL/dW_gamma = (dL/dz * z) g^T`` is nonzero from
    the first update, so zero-init sets the default, it does not disconnect.

    **Bias-free => ``gamma(0) = beta(0) = 0`` FOREVER**, not just at init, so
    "zero goal ⇒ exactly the flat policy" survives training. That is what keeps
    the ``zeroed`` variant of the goal-dependence probe interpretable: with a
    bias, a trained ``gamma(0) != 0`` would make that arm "flat policy plus a
    learned constant modulation" and the control would silently drift. FuN makes
    its ``phi`` bias-free for the same reason.
    """

    hidden_dim: int

    @nn.compact
    def __call__(self, h: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        def _coef():
            return nn.Dense(
                self.hidden_dim,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
            )

        gamma = _coef()(goal)
        beta = _coef()(goal)
        return h * (1.0 + gamma) + beta


class FeudalWorker(nn.Module):
    """Goal-conditioned low-level policy.

    Args (module attributes):
        action_dim: primitive action dimension (or number of discrete actions).
        goal_dim: dimension of the manager's latent goal vector.
        hidden_dim: hidden width of the shared MLP body.
        discrete: action-space type, matching ``MAPPOActor``.
        worker_fusion: how the goal reaches the policy. ``"concat"`` (default,
            the original) appends it to the observation; ``"film"`` feeds it
            through zero-initialized :class:`FiLM` layers on both hidden
            preactivations. See the module docstring for the measurement that
            motivates ``"film"``. ⚠ The two are NOT checkpoint-compatible:
            concat's first Dense has input width ``obs_dim + goal_dim`` against
            FiLM's ``obs_dim``.
        goal_embed_dim: if set, the goal is mapped through a **bias-free** linear
            layer of this width before concatenation (FuN's ``phi``); ``None``
            (default) concatenates the goal directly. Applies to both fusions.
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
    worker_fusion: str = "concat"

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

        actor = MAPPOActor(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            discrete=self.discrete,
        )

        if self.worker_fusion == "concat":
            return actor(jnp.concatenate([obs, goal], axis=-1))

        if self.worker_fusion == "film":
            # One FiLM per hidden layer, applied to the PREACTIVATION (see
            # MAPPOActor.__call__). Named explicitly so the param tree does not
            # depend on flax's creation-order autonaming.
            films = [FiLM(self.hidden_dim, name=f"film_{i}") for i in range(2)]
            return actor(obs, modulate=lambda h, i: films[i](h, goal))

        raise ValueError(
            f"unknown worker_fusion: {self.worker_fusion!r} (expected "
            f"'concat' or 'film')"
        )


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
    worker_fusion: str = "concat",
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
        worker_fusion=worker_fusion,
    )
    params = worker.init(rng, jnp.zeros(obs_dim), jnp.zeros(goal_dim))
    return worker, params
