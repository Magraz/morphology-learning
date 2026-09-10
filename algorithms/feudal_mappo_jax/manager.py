"""Feudal manager network: a centralized, per-agent latent-goal generator.

The manager is the high-level policy of the feudal hierarchy (Vezhnevets et al.,
2017, "FeUdal Networks for Hierarchical Reinforcement Learning"). It reads the
**global state** and emits, for every agent, a unit-norm goal vector `g` — a
*direction* in a learned latent state space `s` that the goal-conditioned worker
(``worker.py``) is then rewarded for moving along.

Nothing here is wired into training yet: no `types.py`, `trainer.py`, `run.py`,
`algorithms/types.py` or `conf/` change, so no existing run is affected.

Layout
------
`s` and `g` live in the *same* space — in FuN the goal is a direction in the
state embedding, not a separate code — so `goal_dim` is the only width knob::

    global_state (E, N*obs_dim)      # trainer.py: obs.reshape(n_envs, -1)
      -> f_percept 2-layer Tanh MLP   -> z    (E, hidden_dim)   team embedding
      -> f_Mspace  Dense(N*goal_dim)  -> s    (E, N, goal_dim)
      -> f_Mrnn (dilated LSTM | MLP)  -> y    (E, hidden_dim)   consumes s
      -> goal head Dense(N*goal_dim)  -> ghat (E, N, goal_dim)
                     per-agent L2     -> g    (E, N, goal_dim)

`s` is a **bottleneck**, not a side head: the core consumes `s`, matching the
paper's ``h^M_t, ghat_t = f^Mrnn(s_t, h^M_{t-1})``. This is what keeps `f_Mspace`
trainable once the transition PG detaches the target arm of the cosine — see
"Latent collapse" below. It also means `goal_dim` is the manager's whole
information channel, not just the goal space, so shrinking it throttles the goal
RNN too.

Goals are **per agent from a centralized manager**: one shared read of the joint
state, one distinct directive per agent, which is what lets the manager assign a
division of labour. `g` drops straight into :class:`FeudalWorker`, whose
``_broadcast_goal`` is a no-op on an already-per-agent goal.

Recurrence is **team level** — a single dilated LSTM over `z`, expanded to agents
only at the goal head — so the carry carries no agent axis.

Deliberate deviations from the paper
------------------------------------
* FuN's ``f_Mspace`` is ``Dense + ReLU``; this uses the repo's 2-layer Tanh body
  (``network.MAPPOActor``/``MAPPOCritic`` convention). Bounded latents are better
  conditioned for a cosine objective, and it keeps the stack visually uniform.
* The goal head uses ``orthogonal(1.0)``, not the actor head's ``0.01``. A
  near-zero ``ghat`` normalizes to a direction set entirely by init noise.
* FuN shares one ``f_percept`` between manager and worker; here the worker
  consumes the raw local obs, so ``f_Mspace`` is manager-only.

Wiring notes for whoever hooks this up
--------------------------------------
* **Manager value ``V^M``**: reuse :class:`network.MAPPOCritic` on the global
  state with ``n_outputs=n_agents``. Do not add another critic.
* **Train state**: ``mappo.create_train_state`` returns a 2-field
  ``ActorCriticTrainState``; a third manager ``TrainState`` must also be added at
  the checkpoint-restore sites in ``run.py``.
* **Goal layout at the actor**: ``trainer._actor_forward`` flattens obs to
  ``(b*n_agents, obs_dim)`` for the fused shared-actor pass, so the goal has to be
  flattened the same way there — ``worker._broadcast_goal`` handles the
  *un*-flattened layout only.
* **Latent collapse** (handled here, but easy to reintroduce). `d_cos` is
  scale-invariant, so the risk is *directional*, not about the magnitude of `s`.
  If `f_Mspace` collapses to rank 1 — `s_t = phi(x_t) * u` for a fixed `u` — then
  every `s_t - s_{t-i}` is parallel to `u`, the goal head emits `g = u`, and the
  cosine is pinned at +-1 for **every** state and every action the worker could
  take. The intrinsic reward becomes a constant, and a constant is annihilated by
  advantage centering: the whole mechanism goes inert while the loss curves look
  healthy. The manager is pushed toward this because it owns *both* arguments of
  the cosine — it picks the measuring stick (`s`) and the target (`g`), and
  rotating the yardstick is far cheaper than learning what the worker can
  actually achieve. It is self-sealing, since the metric that would expose the
  failure is the one that collapsed (structurally the same trap as the
  ``boundary_truncates`` bug documented in CLAUDE.md).
  The guard is FuN's own, and it is explicit in the paper: *"the dependence of
  `s` on theta is ignored when computing grad d_cos -- this avoids trivial
  solutions."* Implemented as ``transition_cosine(..., detach_states=True)``
  (the default) and an unconditional detach inside
  ``worker_intrinsic_reward``. `f_Mspace` keeps a learning signal anyway, through
  the **goal** arm, because the core consumes `s` — which is exactly why the
  bottleneck topology above is load-bearing rather than cosmetic. Wire the core
  to `z` instead and the detach starves `f_Mspace` to its random init.
  Per-agent-specific residual risk, **not** guarded: `s` and `g` are each one
  ``Dense`` reshaped to ``(N, goal_dim)``, so nothing structurally forces the `N`
  rows to differ. If gradient pressure favours uniformity, per-agent goals
  silently degrade to a single team goal — every shape and assertion still
  passes. Log the mean pairwise cosine between agents' goals, the effective rank
  of `s`, and ``Var_t[d_cos]``; a high *flat* intrinsic reward is the pathology,
  so its mean alone reads as success.
* **The dilated LSTM needs a carry threaded through the rollout scan**, whose
  carry is currently ``(train_state, env_state, obs, rng)`` and fully feedforward.
  ``core="mlp"`` is the stateless variant that wires in without touching it.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Dilated LSTM (FuN section 3)
# ---------------------------------------------------------------------------


class DilatedLSTMState(NamedTuple):
    """Carry of :class:`DilatedLSTM`.

    `cell` is the ``(c_pool, h_pool)`` pair of sub-state pools, each
    ``(..., radius, features)``; `t` is a traced scalar step counter that selects
    which group is live.
    """

    cell: Tuple[jnp.ndarray, jnp.ndarray]
    t: jnp.ndarray


def dilated_lstm_carry(features: int, radius: int, batch_shape=()) -> DilatedLSTMState:
    """Zeroed sub-state pools plus a step counter at 0.

    A module-level function, not a method: flax wraps every public module method
    in its scope machinery, so an *unbound* module cannot build submodules inside
    one. Both ``initialize_carry`` methods below delegate here.
    """
    zeros = jnp.zeros(tuple(batch_shape) + (radius, features))
    return DilatedLSTMState(cell=(zeros, zeros), t=jnp.zeros((), dtype=jnp.int32))


class DilatedLSTM(nn.Module):
    """LSTM whose state is a pool of `radius` groups, one updated per step.

    FuN's temporal dilation: the state is ``{h^i}_{i=1..r}`` and at time `t` only
    group ``t % r`` is read and written, so any single group sees the input every
    `r` steps and its gradient path spans ``r`` times more real time than a plain
    LSTM's. The gate parameters are **shared across groups** (one
    ``nn.LSTMCell`` instantiated once), exactly as in the paper.

    Group selection is a one-hot gather + masked write-back rather than a dynamic
    index, so `t` may be a tracer and the module is safe under ``jit``/``scan``.
    """

    features: int
    radius: int

    @nn.compact
    def __call__(self, state: DilatedLSTMState, x: jnp.ndarray):
        cell = nn.LSTMCell(features=self.features)

        c_pool, h_pool = state.cell
        # (radius,) selector over the sub-state axis, which is axis -2.
        onehot = jax.nn.one_hot(state.t % self.radius, self.radius)
        gather = onehot[..., None]  # (radius, 1), broadcasts over `features`

        c_sel = jnp.sum(c_pool * gather, axis=-2)
        h_sel = jnp.sum(h_pool * gather, axis=-2)

        (c_new, h_new), y = cell((c_sel, h_sel), x)

        # Write back into the live group only; the others carry over untouched.
        c_pool = c_pool * (1.0 - gather) + c_new[..., None, :] * gather
        h_pool = h_pool * (1.0 - gather) + h_new[..., None, :] * gather

        return DilatedLSTMState(cell=(c_pool, h_pool), t=state.t + 1), y

    def initialize_carry(self, rng: jax.Array, batch_shape=()) -> DilatedLSTMState:
        """Initial carry. `batch_shape` is the leading shape of the inputs this
        carry will meet, e.g. ``(n_envs,)``. `rng` is unused (the pools start at
        zero) and kept only for parity with ``nn.RNNCellBase.initialize_carry``.
        """
        return dilated_lstm_carry(self.features, self.radius, batch_shape)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class FeudalManager(nn.Module):
    """Centralized manager emitting one unit-norm latent goal per agent.

    Args (module attributes):
        n_agents: number of agents to emit goals for.
        goal_dim: width of the latent state space `s`; goals are directions in it.
        hidden_dim: width of ``f_Mspace`` and of the recurrent core.
        core: ``"dilated_lstm"`` (FuN, default) or ``"mlp"`` (stateless).
        horizon: goal horizon `c`; doubles as the LSTM dilation radius `r`.
        latent: where `s` comes from — ``"centralized"`` (default, the original),
            ``"local"`` (a shared per-agent encoder), or ``"local_global"`` (the
            same encoder, plus a direct read of the env's global state on the
            GOAL path only). See "Latent locality".

    Call:
        ``__call__(carry, global_state, obs=None) -> (carry, goal, state_latent)``

        ``global_state`` is ``(..., n_agents * obs_dim)``; `goal` and
        `state_latent` are both ``(..., n_agents, goal_dim)`` and `goal` is
        L2-normalized per agent. With ``core="mlp"`` the carry is ``None`` and
        passes through, so the signature is the same for both cores.
        ``obs`` is ``(..., n_agents, obs_dim)`` and is REQUIRED by
        ``latent="local"``; the centralized branch ignores it entirely.

    Latent locality (``latent``)
    ----------------------------
    ``"centralized"`` computes ``s = Dense(N*goal_dim)(z)`` and reshapes, i.e.
    ``s_i = W_i z + b_i``: the agent axis is a SLICE INDEX, not a factorization.
    Two consequences, both measured:

    * **`s_i` is not about agent i.** Block-Jacobian probe over 12 trained arms
      (4 env groups x {feudal, feudal_n01, feudal_n05}, 256 on-policy states,
      2026-09-09): with ``B[i,j] = ||d s[i,:] / d obs_block_j||_F``, the diagonal
      share ``B[i,i]/sum_j B[i,j]`` is **0.0631** against a uniform 1/N of
      **0.0625** and the same net at init of 0.0623. Paired trained-minus-init is
      +0.00086. The metric is not blind: a surgically block-diagonal manager
      reads 1.0000 and a half-local one 0.128, so these arms moved ~1.3% of the
      way to *half* localized. Hungarian assignment over `B` (catching
      localization stored under a relabeling) reads 0.0655 vs 0.0650 at init.
      Since ``worker_intrinsic_reward`` scores ``s_t[i] - s_{t-k}[i]``, agent i's
      "own" intrinsic reward therefore moves as much when a TEAMMATE moves as
      when it does: `r^I` is per-agent in indexing, not in causation, and carries
      exactly the credit-assignment confound a worker-level reward exists to
      remove.
    * **Each row has its own basis.** Nothing ties ``W_i`` and ``W_j`` together,
      so "goal coordinate 3" means something different for every agent, and the
      shared worker must learn N goal-to-action mappings.

    ``"local"`` removes both by construction:

        obs_i --f_enc (SHARED, per agent)--> f_Mspace (SHARED) --> s_i
        [s_1..s_N] --core--> y --f_gpre--> y_i --f_goalhead (SHARED)--> g_i

    * ``d s[i]/d obs_j`` is structurally **zero** for ``j != i`` — exactly, not
      approximately.
    * ``f_enc``/``f_Mspace`` are shared, so all rows land in ONE basis; and the
      final projection producing `g` is shared too, so `g_i` and `s_i` live in
      the same space. Without that second sharing `g` would keep per-agent bases
      and the cosine would still compare incommensurate coordinates.
    * The core still consumes `s`, so the bottleneck property below (what keeps
      ``f_Mspace`` trainable under the detach rule) is preserved.
    * Centralization is preserved through the core, which mixes all N local
      latents. For the MJX envs ``global_state`` IS ``obs.reshape(E, -1)``
      (``trainer._global_state`` falls back to exactly that when the env has no
      ``global_state`` hook), so **nothing is lost** and ``"local"`` is a pure
      refactoring of the same input.

    ``"local_global"`` is ``"local"`` plus ``core_in = concat(s_flat, z)``, where
    ``z = f_percept(global_state)`` as in the centralized branch. `s` remains a
    pure function of `obs` — so ``d s[i]/d obs_j`` is still exactly zero for
    ``j != i`` and `r^I` is still clean — while **goal generation** regains the
    env's own global state.

    It exists for envs where ``global_state`` is NOT recoverable from the
    observations. SMAX is the live case, and the gap is structural rather than a
    matter of formatting:

    * ``SMAX.get_obs`` zeroes unit `j` out of unit `i`'s observation entirely
      unless ``||pos_j - pos_i|| < sight_range_i``; ``SMAX.get_world_state``
      applies no such gate, so a unit invisible to *every* ally is present in the
      world state and in no observation.
    * observation positions are *relative* and sight-normalized, world-state
      positions are *absolute*. (Allies are recoverable either way —
      ``get_self_features`` puts each unit's own absolute position in its own
      observation — so the loss is specifically about ENEMIES.)

    Measured (64 envs x 100 steps, random legal actions): the fraction of alive
    enemies invisible to every ally is **100% at t=0** on all of 3m / 5m_vs_6m /
    2s3z / 3s5z, falling to 28-42% over the first 10 steps. Read the t=0 number
    and not the rest: under random actions the teams never engage, so the
    mid-episode figures are pessimistic and a trained policy would push them far
    lower. But t=0 is a SPAWN property, not a policy artifact — SMAX starts the
    teams beyond sight range — and with ``goal_horizon=10`` that is exactly the
    window in which the first pooled goal ``w_t`` is assembled.

    Note what is NOT lost under plain ``"local"``: all three critics keep reading
    ``global_state`` directly (``trainer._values`` / ``_manager_values`` /
    ``_values_int``), so CTDE is intact either way. Only the manager's directives
    change.

    On an MJX env ``"local_global"`` is redundant (it feeds `z` computed from the
    same numbers the encoder already saw) and costs an extra ``f_percept``, so
    prefer ``"local"`` there and keep arms differing in one thing at a time.

    Residual `"local"` does NOT fix: ``obs_i`` is egocentric but not
    proprioceptive — density sensors, ``neighbor_fraction`` and lidar all respond
    to teammates inside ``sector_sensor_radius``. So `s_i` is agent i's *view*,
    which teammates still perturb through channels agent i can perceive. That is
    arguably the right notion of own-progress under partial observability, and it
    is a large reduction from uniform 1/N mixing, but it is not literally
    own-physical-state. ``latent_locality_probe.py`` measures the physical
    residual separately (Jacobian w.r.t. agent world positions).
    """

    n_agents: int
    goal_dim: int
    hidden_dim: int = 256
    core: str = "dilated_lstm"
    horizon: int = 10
    latent: str = "centralized"

    def _check_core(self):
        if self.core not in ("dilated_lstm", "mlp"):
            raise ValueError(f"core must be 'dilated_lstm' or 'mlp', got {self.core!r}")

    def _check_latent(self):
        # Reject an unknown value rather than failing open. CLAUDE.md records the
        # `VARIANTS` regression where a silently-false guard turned a whole arm
        # into its own baseline for two commits.
        if self.latent not in ("centralized", "local", "local_global"):
            raise ValueError(
                "latent must be 'centralized', 'local' or 'local_global', got "
                f"{self.latent!r}"
            )

    def _dense(self, features: int, scale: float, name: str):
        return nn.Dense(
            features,
            kernel_init=nn.initializers.orthogonal(scale),
            bias_init=nn.initializers.constant(0.0),
            name=name,
        )

    @nn.compact
    def __call__(self, carry, global_state: jnp.ndarray, obs: jnp.ndarray = None):
        self._check_core()
        self._check_latent()

        is_local = self.latent in ("local", "local_global")
        if is_local:
            if obs is None:
                raise ValueError(
                    f"latent={self.latent!r} needs `obs` (..., n_agents, obs_dim): "
                    "`s` is built per agent from its OWN observation, which the "
                    "flattened global state cannot be un-mixed back into for every "
                    "env. Pass obs explicitly — the trainer, the eval scan, view() "
                    "and manager_update all have it in scope."
                )
            # f_enc: ONE encoder, applied to each agent's own observation. Sharing
            # it is what puts every row of `s` in a single basis; applying it per
            # agent is what makes d s[i]/d obs_j structurally zero for j != i.
            h = nn.tanh(self._dense(self.hidden_dim, np.sqrt(2), "f_enc_0")(obs))
            h = nn.tanh(self._dense(self.hidden_dim, np.sqrt(2), "f_enc_1")(h))
            # Shared, so the goal-space coordinates mean the same thing for all
            # agents. Same `orthogonal(1.0)` as the centralized f_Mspace.
            s = self._dense(self.goal_dim, 1.0, "f_Mspace")(h)
            lead = s.shape[:-2]
        else:
            # f_percept: joint observation -> team embedding.
            z = nn.tanh(
                self._dense(self.hidden_dim, np.sqrt(2), "f_percept_0")(global_state)
            )
            z = nn.tanh(self._dense(self.hidden_dim, np.sqrt(2), "f_percept_1")(z))

            # f_Mspace: the latent space the goals live in. Feedforward from `z` and
            # *upstream* of the core, so `s` is a function of the current state alone
            # (no history), exactly as in FuN.
            s = self._dense(self.n_agents * self.goal_dim, 1.0, "f_Mspace")(z)
            s = s.reshape(z.shape[:-1] + (self.n_agents, self.goal_dim))
            lead = z.shape[:-1]

        # f_Mrnn consumes `s`, not `z` — this is load-bearing, see the module
        # docstring: it is what keeps `f_Mspace` trainable once the transition
        # PG detaches the target arm of the cosine. It is ALSO what makes the
        # local branches centralized: the core is where the N per-agent latents
        # are mixed, so goal assignment still reads the whole team.
        core_in = s.reshape(lead + (self.n_agents * self.goal_dim,))
        if self.latent == "local_global":
            # ...plus a direct read of the env's own global state, for envs where
            # that is NOT recoverable from the observations. `s` stays a pure
            # function of `obs` (so `r^I` stays clean); only GOAL GENERATION gains
            # the extra input. See "Latent locality" for the SMAX measurement that
            # motivates it.
            zg = nn.tanh(
                self._dense(self.hidden_dim, np.sqrt(2), "f_percept_0")(global_state)
            )
            zg = nn.tanh(self._dense(self.hidden_dim, np.sqrt(2), "f_percept_1")(zg))
            core_in = jnp.concatenate([core_in, zg], axis=-1)
        if self.core == "dilated_lstm":
            carry, y = DilatedLSTM(
                features=self.hidden_dim, radius=self.horizon, name="core"
            )(carry, core_in)
        else:
            y = nn.tanh(self._dense(self.hidden_dim, np.sqrt(2), "core")(core_in))

        if is_local:
            # Per-agent context out of the centralized mix, THEN a shared
            # projection into goal space. The second step is what puts `g_i` in
            # the same basis as `s_i`; with a per-agent final layer (as the
            # centralized branch has) `g` would keep N private coordinate systems
            # and the cosine would compare incommensurate axes.
            y_i = self._dense(self.n_agents * self.goal_dim, 1.0, "f_gpre")(y)
            y_i = y_i.reshape(y.shape[:-1] + (self.n_agents, self.goal_dim))
            goal = self._dense(self.goal_dim, 1.0, "f_goalhead")(nn.tanh(y_i))
        else:
            goal = self._dense(self.n_agents * self.goal_dim, 1.0, "goal_head")(y)
            goal = goal.reshape(y.shape[:-1] + (self.n_agents, self.goal_dim))
        goal = _unit(goal)

        return carry, goal, s

    def initialize_carry(self, rng: jax.Array, batch_shape=()):
        """Initial carry for this manager: pools for the LSTM core, else ``None``."""
        self._check_core()
        self._check_latent()
        if self.core != "dilated_lstm":
            return None
        return dilated_lstm_carry(self.hidden_dim, self.horizon, batch_shape)


def init_manager(
    rng: jax.Array,
    global_state_dim: int,
    n_agents: int,
    goal_dim: int,
    hidden_dim: int = 256,
    core: str = "dilated_lstm",
    horizon: int = 10,
    batch_shape=(),
    latent: str = "centralized",
    obs_dim: int = None,
):
    """Build a `FeudalManager`, its params and its initial carry.

    Returns ``(module, params, carry)``. Mirrors ``worker.init_worker``.

    ``latent="local"``/``"local_global"`` need `obs_dim` so the init pass can
    supply a correctly shaped ``(*batch, n_agents, obs_dim)`` dummy observation.
    """
    manager = FeudalManager(
        n_agents=n_agents,
        goal_dim=goal_dim,
        hidden_dim=hidden_dim,
        core=core,
        horizon=horizon,
        latent=latent,
    )
    carry_rng, init_rng = jax.random.split(rng)
    carry = manager.initialize_carry(carry_rng, batch_shape)
    dummy = jnp.zeros(tuple(batch_shape) + (global_state_dim,))
    dummy_obs = None
    if latent in ("local", "local_global"):
        if obs_dim is None:
            raise ValueError(f"init_manager(latent={latent!r}) requires obs_dim")
        dummy_obs = jnp.zeros(tuple(batch_shape) + (n_agents, obs_dim))
    params = manager.init(init_rng, carry, dummy, dummy_obs)
    return manager, params, carry


# ---------------------------------------------------------------------------
# Goal semantics: what a goal *means*, as pure jittable functions
# ---------------------------------------------------------------------------


def _unit(v: jnp.ndarray, eps: float = _EPS) -> jnp.ndarray:
    """Row-wise normalize along the last axis; a zero row stays finite (no NaN)."""
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)


def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray, eps: float = _EPS):
    """``d_cos`` over the last axis. Zero vectors score 0 rather than NaN."""
    return jnp.sum(_unit(a, eps) * _unit(b, eps), axis=-1)


def _same_episode(done: Optional[jnp.ndarray], t_from, t_to, length: int):
    """1.0 where steps `t_from` and `t_to` lie in the same episode.

    `done[t]` marks step `t` as terminal (the episode ends *after* it), matching
    ``Transition.done``. Two steps share an episode iff no done falls in
    ``[min, max)``; a cumulative sum turns that into an O(1) comparison per pair.
    """
    if done is None:
        return 1.0
    lo = jnp.minimum(t_from, t_to)
    hi = jnp.maximum(t_from, t_to)
    # cs[t] = number of dones strictly before t, so cs[hi] - cs[lo] counts the
    # dones in [lo, hi).
    cs = jnp.concatenate(
        [jnp.zeros((1,) + done.shape[1:], done.dtype), jnp.cumsum(done, axis=0)],
        axis=0,
    )
    crossed = jnp.take(cs, hi, axis=0) - jnp.take(cs, lo, axis=0)
    return (crossed == 0).astype(jnp.float32)


def _align(mask, ref: jnp.ndarray):
    """Broadcast an episode mask over `ref`'s trailing (agent/feature) axes."""
    if isinstance(mask, float):
        return mask
    while mask.ndim < ref.ndim:
        mask = mask[..., None]
    return mask


def _check_done(done: Optional[jnp.ndarray], expected, fn_name: str):
    """Reject a `done` mask whose shape is not the leading shape it must align to.

    Load-bearing, because the failure it prevents is **silent**. These masks are
    multiplied against tensors whose leading axes are ``(T, n_envs, n_agents)``.
    The trainer's own ``Transition.done`` is ``(T, n_envs)`` — one axis short —
    and numpy right-alignment turns ``(T,1,1) * (T,E)`` into ``(T,T,E)`` rather
    than raising, so the manager loss would end up masked by a tensor that
    measures nothing. Callers must pre-broadcast `done` to the full leading
    shape (e.g. ``jnp.broadcast_to(done[..., None], states.shape[:-1])``).

    Note ``(T, E, 1)`` is rejected too: it broadcasts *correctly* but makes a
    masked-mean denominator ``mask.sum()`` a factor of ``n_agents`` too small.

    Python-level (shapes are static under jit), so this costs nothing.
    """
    if done is None:
        return
    if tuple(done.shape) != tuple(expected):
        raise ValueError(
            f"{fn_name}: `done` must have shape {tuple(expected)} (the leading "
            f"shape of the cosine), got {tuple(done.shape)}. A partially-shaped "
            f"mask broadcasts silently into a wrong result instead of raising — "
            f"pre-broadcast it, e.g. "
            f"jnp.broadcast_to(done[..., None], {tuple(expected)})."
        )


def pool_goals(
    goals: jnp.ndarray, horizon: int, done: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """FuN's goal pooling: ``w_t = sum_{i=t-c+1}^{t} g_i``.

    The worker is conditioned on the *sum of the last `c` goals*, not on the
    single current one, which is what makes the manager's directives persist over
    the horizon instead of flickering step to step.

    Args:
        goals: ``(T, ...)`` with the goal vector on the last axis.
        horizon: `c`.
        done: optional terminal mask, shaped exactly ``goals.shape[:-1]``; goals
            from before an episode boundary are dropped from the sum.

    Returns:
        ``(T, ...)``, same shape as `goals`.
    """
    _check_done(done, goals.shape[:-1], "pool_goals")

    T = goals.shape[0]
    steps = jnp.arange(T)

    def one_offset(i):
        src = jnp.maximum(steps - i, 0)
        g = jnp.take(goals, src, axis=0)
        # Both masks must be aligned to `g` SEPARATELY before being combined:
        # `in_range` is (T,) while `_same_episode` is g.shape[:-1], so
        # multiplying them first right-aligns (T,) against the agent axis and
        # either raises or silently mis-broadcasts.
        in_range = _align((steps >= i).astype(jnp.float32), g)
        same = _align(_same_episode(done, src, steps, T), g)
        return g * in_range * same

    return jnp.sum(jax.vmap(one_offset)(jnp.arange(horizon)), axis=0)


def goal_ring_write(goal_hist: jnp.ndarray, goal: jnp.ndarray, t) -> jnp.ndarray:
    """Write `goal` into ring slot ``t % horizon``; returns the updated ring.

    The incremental, online form of :func:`pool_goals`. The pooled conditioning
    `w_t` has to exist *at act time* — the worker's stored ``log_prob`` and the
    update's ``evaluate_action`` must come from the same conditioned policy or
    the PPO ratio is meaningless — but ``pool_goals`` reads the whole trajectory
    and cannot be called inside a scan. So the rollout carries this ring instead,
    and ``pool_goals`` serves as the offline oracle the ring is tested against.

    Shared by **every** consumer of that convention — the training scan, the eval
    scan, and ``run.py:view()`` — so the rendered policy provably conditions on
    the same thing the trained one did. Rank-agnostic: the ring is
    ``(horizon, *leading, goal_dim)`` and `goal` is ``(*leading, goal_dim)``, so
    it serves the batched ``(c, E, N, D)`` and unbatched ``(c, N, D)`` layouts
    alike.

    Slot selection is a one-hot gather/write rather than a dynamic index, so `t`
    may be a tracer. Slots not yet written hold zeros and contribute nothing to
    the sum — exactly what ``pool_goals`` masks out for ``t < i``.
    """
    horizon = goal_hist.shape[0]
    slot = jax.nn.one_hot(t % horizon, horizon).reshape(
        (horizon,) + (1,) * (goal_hist.ndim - 1)
    )
    return goal_hist * (1.0 - slot) + goal[None] * slot


def goal_ring_pool(goal_hist: jnp.ndarray) -> jnp.ndarray:
    """FuN's `w_t`: the sum over the ring (zeros for never-written slots)."""
    return goal_hist.sum(axis=0)


def goal_ring_reset(goal_hist: jnp.ndarray, done) -> jnp.ndarray:
    """Clear finished envs' rings, preserving ``pool_goals``' episode semantics.

    Must be applied AFTER the step's `w_t` has been consumed: ``_same_episode``
    counts dones in ``[src, t)``, so `g_src` still contributes to `w_t` on the
    step where ``done[t]`` fires, and not afterwards.
    """
    mask = done.reshape((1,) + done.shape + (1,) * (goal_hist.ndim - done.ndim - 1))
    return jnp.where(mask, 0.0, goal_hist)


# ---------------------------------------------------------------------------
# Permutation nulls: the reference level for "are the goals useful?"
#
# Every other manager diagnostic in this stack is a COLLAPSE detector — a
# necessary condition, not a usefulness test. Two of them mislead if read as
# one: `d_cos_mean` is uninterpretable because the manager owns *both*
# arguments of the cosine, and `worker_goal_column_ratio` already misled once
# (it fell while the goal block grew).
#
# The method is a permutation that preserves the goal marginal distribution
# EXACTLY and destroys exactly one property, so the real-minus-null gap
# isolates that property:
#
#     axis      preserves            destroys
#     agents    state-conditioning   the agent<->goal PAIRING
#     envs      the pairing          STATE-CONDITIONING
#
# Both are needed, and the env one has to be read FIRST. A manager that emits a
# fixed per-agent constant (an agent-ID code with no dependence on `s_t`)
# produces a large agent-permutation gap while doing nothing the hierarchy is
# for — and the existing suite endorses that state, since `goal_direction_count`
# reads ~8.25 against a random-direction baseline of 8.26, i.e. "maximally
# diverse". If `d_cos_null_env ~ d_cos_mean` the goals carry no state
# information and the agent-pairing gap is meaningless whatever its value.
# ---------------------------------------------------------------------------


def _check_permutable(goal: jnp.ndarray, shift: int, axis: int, fn_name: str):
    """Reject a permutation that is secretly the identity.

    Load-bearing for the same reason as :func:`_check_done`: the failure is
    silent and self-sealing. At ``n == 1`` (or ``shift % n == 0``) a roll is the
    identity, every real-minus-null gap is exactly 0.0, and the diagnostic
    reports "the goals make no difference" — the precise answer it exists to
    detect — while having measured nothing at all.

    Python-level; shapes and `shift` are static under jit, so this is free.
    """
    n = goal.shape[axis]
    if n < 2:
        raise ValueError(
            f"{fn_name}: axis {axis} has length {n}, so a roll is the identity "
            f"and the null would equal the real value by construction (a "
            f"vacuous 'no dependence' result). This diagnostic needs at least 2."
        )
    if shift % n == 0:
        raise ValueError(
            f"{fn_name}: shift={shift} is a multiple of axis {axis}'s length "
            f"{n}, so the roll is the identity and the null is vacuous. Use a "
            f"shift that is not a multiple of {n}."
        )


def permute_agent_goals(
    goal: jnp.ndarray, shift: int = 1, axis: int = -2
) -> jnp.ndarray:
    """Re-pair goals across the AGENT axis: agent `i` receives agent `i-shift`'s.

    Destroys the agent<->goal pairing while preserving the goal multiset at each
    ``(t, env)`` exactly — so a real-minus-null gap measures the *assignment*,
    not a distribution shift.

    A cyclic shift, deliberately, rather than a random permutation:

    * it is a guaranteed **derangement**. A uniform random permutation has
      ``E[fixed points] = 1``, i.e. on average one of 16 agents silently keeps
      its own goal, systematically diluting the intervention.
    * :func:`~algorithms.feudal_mappo_jax.mappo.manager_update` has no rng in
      scope, and adding one would change its signature and make the metric
      non-reproducible epoch to epoch.
    * determinism is what lets the offline probe and the live training series
      produce the same number on the same data.

    Rank-agnostic like the ring helpers, so ``(N, D)``, ``(E, N, D)`` and
    ``(T, E, N, D)`` all work off the default ``axis=-2``.

    NOTE ON PLACEMENT: apply this to the **pooled** goal ``w_t``, not to the raw
    per-step goals. ``goal_ring_pool`` is a plain sum over the ring, so for a
    shift fixed across time ``roll(pool(h)) == pool(roll(h))`` bit-identically
    (pinned by ``test_agent_permutation_commutes_with_the_goal_ring``) — but
    permuting raw goals per step and *then* pooling would hand agent `i` a sum
    of goals from different agents at different times. Measured consecutive-goal
    collinearity is 0.95-0.99, so ``||w_t|| ~ c`` for a true pool against
    ``~sqrt(c)`` for a cross-agent one: a silent ~3.2x scale intervention under
    ``normalize_pooled_goal=False``, and a partial goal *collapse* under
    ``True``. Either way the marginal is no longer preserved and the null stops
    being a null.
    """
    _check_permutable(goal, shift, axis, "permute_agent_goals")
    return jnp.roll(goal, shift, axis=axis)


def permute_env_goals(goal: jnp.ndarray, axis: int, shift: int = 1) -> jnp.ndarray:
    """Re-pair goals across the ENV axis: env `e` receives env `e-shift`'s goals.

    The companion null to :func:`permute_agent_goals`. It keeps every agent
    paired with an agent-slot goal but takes that goal from an unrelated state,
    so it destroys **state-conditioning** while preserving the pairing. This is
    the null that catches a manager which has degenerated into emitting a fixed
    per-agent code, which the agent-axis null alone reports as healthy.

    `axis` is REQUIRED and never inferred: the eval path's pooled goal is
    ``(E, N, D)`` (env axis 0) while the manager path's is ``(T, E, N, D)``
    (env axis 1), and guessing wrong silently rolls time or agents instead.
    """
    _check_permutable(goal, shift, axis, "permute_env_goals")
    return jnp.roll(goal, shift, axis=axis)


def zero_goals(goal: jnp.ndarray) -> jnp.ndarray:
    """Replace the goal with zeros — destroys pairing AND state-conditioning.

    Equivalent to ``FeudalWorker(zero_goal=True)`` when applied to the pooled
    goal at the worker's input: ``_unit(0) == 0`` and the optional
    ``goal_embed_dim`` Dense is bias-free, so ``Dense(0) == 0``. Doing it here
    rather than by rebuilding the module keeps the parameter tree untouched, so
    checkpoints stay interchangeable and the ablation costs no re-init.
    """
    return jnp.zeros_like(goal)


def _goal_transform_variants(shift: int, env_axis: int):
    """Build the variant name -> transform map for one call site.

    Single source of truth for the variant NAMES, which are simultaneously the
    eval-block labels, the metric-key suffixes and the offline probe's CLI
    strings. Defining them in one place is what stops those three drifting.
    """
    return {
        "real": lambda g: g,
        "permuted": lambda g: permute_agent_goals(g, shift),
        "env_permuted": lambda g: permute_env_goals(g, env_axis, shift),
        "zeroed": zero_goals,
    }


GOAL_VARIANTS = ("real", "permuted", "env_permuted", "zeroed")
"""Every defined goal-ablation variant, in reading order.

The live eval path uses the first, second and fourth; ``env_permuted`` is
offline-only (the live series is read as a trend, but an offline number has to
be interpretable in isolation, which is exactly where it is needed).
"""


def transition_cosine(
    states: jnp.ndarray,
    goals: jnp.ndarray,
    horizon: int,
    done: Optional[jnp.ndarray] = None,
    detach_states: bool = True,
):
    """The manager's transition policy gradient objective, ``d_cos(s_{t+c}-s_t, g_t)``.

    FuN does not treat the goal as an action to be reinforced by the worker's
    behaviour; it trains the manager on whether the state *actually moved* in the
    direction it asked for, `c` steps later. That is this quantity — multiply it
    by the manager's advantage to get the loss.

    `detach_states` implements the paper's explicit rule: *"the dependence of `s`
    on θ is ignored when computing ∇_θ d_cos — this avoids trivial solutions."*
    The observed direction is **data**; only `g_t(θ)` carries gradient. Without
    it the manager owns both arguments of the cosine and can maximize the
    objective by rotating the measuring stick instead of issuing achievable
    goals — see the "Latent collapse" note in the module docstring. `f_Mspace` is
    still trained by this loss, through the *goal* arm, because the core consumes
    `s`. Set ``False`` only to reproduce that failure deliberately.

    Args:
        states: ``(T, ..., d)`` latent states `s`.
        goals: ``(T, ..., d)`` unit goals, same shape.
        horizon: `c`.
        done: optional ``(T, ...)`` terminal mask.
        detach_states: stop gradient through the ``s_{t+c} - s_t`` arm (default).

    Returns:
        ``(cos, valid)``, both ``(T, ...)``. Entries with ``t > T-1-c`` have no
        real `s_{t+c}`; they are computed against the clamped last state and
        flagged ``valid=False``. Steps whose horizon crosses an episode boundary
        are also invalid.
    """
    _check_done(done, states.shape[:-1], "transition_cosine")

    T = states.shape[0]
    steps = jnp.arange(T)
    dst = jnp.minimum(steps + horizon, T - 1)

    target = jax.lax.stop_gradient(states) if detach_states else states
    delta = jnp.take(target, dst, axis=0) - target
    cos = cosine_similarity(delta, goals)

    in_range = (steps + horizon <= T - 1).astype(jnp.float32)
    valid = _align(in_range, cos) * _same_episode(done, steps, dst, T)
    # Always hand back `valid` at the FULL shape of `cos`. Without this it is
    # (T,1,1) whenever `done is None`, which multiplies correctly but makes the
    # masked-mean denominator `valid.sum()` a factor of n_envs*n_agents too
    # small — the same silent-denominator hazard `_check_done` rejects a
    # (T,E,1) mask for. Broadcasting is free (no copy until written to).
    return cos, jnp.broadcast_to(valid, cos.shape)


def worker_intrinsic_reward(
    states: jnp.ndarray,
    goals: jnp.ndarray,
    horizon: int,
    done: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """FuN's intrinsic reward, ``r^I_t = 1/c * sum_{i=1..c} d_cos(s_t - s_{t-i}, g_{t-i})``.

    The worker's own reward for having followed the manager's recent directives.
    Averaged over the `i` that are actually in range (and in the same episode), so
    the first steps of an episode are not diluted; ``r^I_0`` is 0 by construction.

    **Both arguments are detached, unconditionally**: this is a reward, and a
    reward is data. Leaving `s` or `g` attached here would let the manager raise
    the worker's reward by editing the yardstick rather than by issuing useful
    goals, and would additionally backpropagate the worker's objective into the
    manager — which FuN rules out because it *"would deprive Manager's goals `g`
    of any semantic meaning, making them just internal latent variables."*

    Shapes match :func:`transition_cosine`; returns ``(T, ...)``.
    """
    _check_done(done, states.shape[:-1], "worker_intrinsic_reward")

    states = jax.lax.stop_gradient(states)
    goals = jax.lax.stop_gradient(goals)

    T = states.shape[0]
    steps = jnp.arange(T)

    def one_offset(i):
        src = jnp.maximum(steps - i, 0)
        delta = states - jnp.take(states, src, axis=0)
        cos = cosine_similarity(delta, jnp.take(goals, src, axis=0))
        in_range = (steps >= i).astype(jnp.float32)
        mask = _align(in_range, cos) * _same_episode(done, src, steps, T)
        return cos * mask, mask

    cos, mask = jax.vmap(one_offset)(jnp.arange(1, horizon + 1))
    return jnp.sum(cos, axis=0) / jnp.maximum(jnp.sum(mask, axis=0), 1.0)


# ---------------------------------------------------------------------------
# Self-check:  uv run python -m algorithms.feudal_mappo_jax.manager
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from algorithms.feudal_mappo_jax.network import sample_action
    from algorithms.feudal_mappo_jax.worker import bind_goal, init_worker

    E, N, OBS_DIM = 4, 6, 40
    GOAL_DIM, HIDDEN, HORIZON = 16, 32, 5
    GLOBAL_DIM = N * OBS_DIM

    rng = jax.random.PRNGKey(0)
    manager, params, carry = init_manager(
        rng,
        global_state_dim=GLOBAL_DIM,
        n_agents=N,
        goal_dim=GOAL_DIM,
        hidden_dim=HIDDEN,
        core="dilated_lstm",
        horizon=HORIZON,
        batch_shape=(E,),
    )

    flat = jax.tree.leaves(params)
    print(f"FeudalManager  params: {sum(p.size for p in flat):,}")
    for path, p in jax.tree_util.tree_flatten_with_path(params)[0]:
        print(f"  {'/'.join(str(k.key) for k in path if hasattr(k, 'key')):<40} {p.shape}")

    gs = jax.random.normal(jax.random.PRNGKey(1), (E, GLOBAL_DIM))

    # [1] shapes + unit-norm goals -------------------------------------------
    new_carry, goal, s = manager.apply(params, carry, gs)
    assert goal.shape == (E, N, GOAL_DIM), goal.shape
    assert s.shape == (E, N, GOAL_DIM), s.shape
    norms = jnp.linalg.norm(goal, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5), norms
    print("[1] shapes + per-agent unit goals      OK")

    # [2] dilation: one group per step, all r groups after r steps ------------
    c_pool, h_pool = carry.cell
    assert c_pool.shape == (E, HORIZON, HIDDEN), c_pool.shape
    touched = jnp.zeros(HORIZON)
    st, prev = carry, carry
    for t in range(HORIZON):
        st, _, _ = manager.apply(params, st, gs)
        changed = jnp.any(st.cell[1] != prev.cell[1], axis=(0, 2))  # (radius,)
        assert changed.sum() == 1, f"step {t}: {changed.sum()} groups changed"
        assert bool(changed[t % HORIZON]), f"step {t} wrote the wrong group"
        assert st.cell[0].shape == c_pool.shape  # carry shape is invariant
        touched = touched + changed
        prev = st
    assert jnp.all(touched == 1), touched
    print("[2] dilated LSTM writes group t%r only OK")

    # [3] jit + lax.scan over T > radius --------------------------------------
    T = 3 * HORIZON

    @jax.jit
    def roll(params, carry, states):
        def body(c, x):
            c, g, s = manager.apply(params, c, x)
            return c, (g, s)

        return jax.lax.scan(body, carry, states)

    seq = jax.random.normal(jax.random.PRNGKey(2), (T, E, GLOBAL_DIM))
    end_carry, (goals, latents) = roll(params, carry, seq)
    assert goals.shape == (T, E, N, GOAL_DIM), goals.shape
    assert latents.shape == (T, E, N, GOAL_DIM), latents.shape
    assert int(end_carry.t) == T, end_carry.t
    print("[3] jit + lax.scan carry threading     OK")

    # [4] vmap over a leading batch axis --------------------------------------
    B = 3
    vgs = jax.random.normal(jax.random.PRNGKey(3), (B, E, GLOBAL_DIM))
    vcarry = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), carry)
    _, vgoal, _ = jax.vmap(manager.apply, in_axes=(None, 0, 0))(params, vcarry, vgs)
    assert vgoal.shape == (B, E, N, GOAL_DIM), vgoal.shape
    print("[4] vmap                               OK")

    # [5] mlp core: stateless, same output shapes, no LSTM params -------------
    m_mlp, p_mlp, c_mlp = init_manager(
        jax.random.PRNGKey(4),
        global_state_dim=GLOBAL_DIM,
        n_agents=N,
        goal_dim=GOAL_DIM,
        hidden_dim=HIDDEN,
        core="mlp",
        horizon=HORIZON,
        batch_shape=(E,),
    )
    assert c_mlp is None
    c_out, g_mlp, s_mlp = m_mlp.apply(p_mlp, c_mlp, gs)
    assert c_out is None
    assert g_mlp.shape == goal.shape and s_mlp.shape == s.shape
    names = [
        "/".join(str(k.key) for k in path if hasattr(k, "key"))
        for path, _ in jax.tree_util.tree_flatten_with_path(p_mlp)[0]
    ]
    assert not any("LSTMCell" in n for n in names), names
    try:
        FeudalManager(n_agents=N, goal_dim=GOAL_DIM, core="gru").init(rng, None, gs)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown core must raise")
    print("[5] mlp core stateless + bad core raises  OK")

    # [6] goal semantics ------------------------------------------------------
    d = 4
    Ts = 8
    key = jax.random.PRNGKey(5)
    g_fix = _unit(jax.random.normal(key, (Ts, d)))

    # A trajectory whose achieved delta over `c` is exactly the goal.
    c = 3
    s_seq = jnp.cumsum(
        jnp.concatenate([jnp.zeros((1, d)), jnp.tile(g_fix[0] / c, (Ts - 1, 1))]),
        axis=0,
    )
    g_seq = jnp.tile(g_fix[0], (Ts, 1))
    cos, valid = transition_cosine(s_seq, g_seq, c)
    assert jnp.allclose(cos[valid > 0], 1.0, atol=1e-5), cos
    assert jnp.array_equal(valid, (jnp.arange(Ts) + c <= Ts - 1).astype(jnp.float32))
    cos_opp, _ = transition_cosine(s_seq, -g_seq, c)
    assert jnp.allclose(cos_opp[valid > 0], -1.0, atol=1e-5), cos_opp

    r_i = worker_intrinsic_reward(s_seq, g_seq, c)
    assert r_i.shape == (Ts,)
    assert float(r_i[0]) == 0.0, r_i[0]
    assert jnp.all(jnp.abs(r_i) <= 1.0 + 1e-5), r_i
    assert jnp.allclose(r_i[1:], 1.0, atol=1e-5), r_i  # moving exactly on-goal

    pooled = pool_goals(g_seq, c)
    assert jnp.allclose(pooled[0], g_fix[0], atol=1e-5)
    assert jnp.allclose(pooled[c - 1], c * g_fix[0], atol=1e-5)
    assert jnp.allclose(pooled[-1], c * g_fix[0], atol=1e-5)

    # done at t=3 must sever every pair spanning it.
    done = jnp.zeros(Ts).at[3].set(1.0)
    _, valid_d = transition_cosine(s_seq, g_seq, c, done=done)
    assert float(valid_d[1]) == 0.0 and float(valid_d[3]) == 0.0, valid_d
    assert float(valid_d[4]) == 1.0, valid_d
    pooled_d = pool_goals(g_seq, c, done=done)
    assert jnp.allclose(pooled_d[4], g_fix[0], atol=1e-5), pooled_d[4]
    r_d = worker_intrinsic_reward(s_seq, g_seq, c, done=done)
    assert float(r_d[4]) == 0.0, r_d  # nothing in-episode to look back at
    assert jnp.all(jnp.isfinite(r_d))
    print("[6] pool/transition/intrinsic semantics   OK")

    # [7] handshake with the worker ------------------------------------------
    for discrete, act_dim in ((False, 2), (True, 4)):
        worker, w_params = init_worker(
            jax.random.PRNGKey(6),
            obs_dim=OBS_DIM,
            goal_dim=GOAL_DIM,
            action_dim=act_dim,
            hidden_dim=HIDDEN,
            discrete=discrete,
        )
        obs = jax.random.normal(jax.random.PRNGKey(7), (E, N, OBS_DIM))
        action, log_prob = sample_action(
            jax.random.PRNGKey(8),
            bind_goal(worker.apply, goal),
            w_params,
            obs,
            discrete,
        )
        want = (E, N) if discrete else (E, N, act_dim)
        assert action.shape == want, (discrete, action.shape)
        assert log_prob.shape == (E, N), log_prob.shape
    print("[7] manager goal -> worker action      OK")

    # [8] gradient routing under the paper's detach rule ----------------------
    # "the dependence of s on theta is ignored when computing grad d_cos --
    #  this avoids trivial solutions." (Vezhnevets et al., 2017)
    def _leaf(tree, *keys):
        for k in keys:
            tree = tree[k]
        return tree

    # (a) the detach really detaches: no gradient reaches the target arm...
    s_raw = jax.random.normal(jax.random.PRNGKey(9), (Ts, d))
    g_raw = _unit(jax.random.normal(jax.random.PRNGKey(10), (Ts, d)))

    def obj(st, gl, detach):
        cos, valid = transition_cosine(st, gl, c, detach_states=detach)
        return jnp.sum(cos * valid)

    gs_det, gg_det = jax.grad(obj, argnums=(0, 1))(s_raw, g_raw, True)
    assert jnp.all(gs_det == 0.0), "detach_states=True must zero the target arm"
    assert jnp.any(gg_det != 0.0), "the goal arm must still carry gradient"

    # ...and detach_states=False is genuinely different (the gameable mode).
    gs_att, _ = jax.grad(obj, argnums=(0, 1))(s_raw, g_raw, False)
    assert jnp.any(gs_att != 0.0), "detach_states=False must expose the target arm"

    # (b) the intrinsic reward is data: nothing flows back through it at all.
    gi = jax.grad(lambda st, gl: jnp.sum(worker_intrinsic_reward(st, gl, c)),
                  argnums=(0, 1))(s_raw, g_raw)
    assert all(jnp.all(x == 0.0) for x in gi), "r^I must be fully detached"

    # (c) the load-bearing property: f_Mspace is STILL trained under the detach,
    #     via the goal arm, because the core consumes `s`. Wire the core to `z`
    #     instead and this grad is exactly zero and the latent never learns.
    seq_gs = jax.random.normal(jax.random.PRNGKey(11), (T, E, GLOBAL_DIM))

    def manager_pg(params):
        _, (gl, st) = roll(params, carry, seq_gs)
        cos, valid = transition_cosine(st, gl, HORIZON, detach_states=True)
        return jnp.sum(cos * valid)

    grads = jax.grad(manager_pg)(params)
    g_mspace = _leaf(grads, "params", "f_Mspace", "kernel")
    g_goal = _leaf(grads, "params", "goal_head", "kernel")
    assert jnp.any(g_mspace != 0.0), (
        "f_Mspace got no gradient under the detach — the core is not consuming s"
    )
    assert jnp.any(g_goal != 0.0)
    print(
        "[8] detach routing (|dL/df_Mspace| = "
        f"{float(jnp.linalg.norm(g_mspace)):.3f}, "
        f"|dL/dgoal_head| = {float(jnp.linalg.norm(g_goal)):.3f})   OK"
    )

    # [9] the shapes the TRAINER actually passes -------------------------------
    # Everything above exercises 1-D (T, d) sequences. The trainer's trajectory is
    # (T, n_envs, n_agents, goal_dim) with a (T, n_envs) done, and both of those
    # used to go wrong silently — see `_check_done`. This group pins the batched
    # path against the 1-D path that groups [6]/[8] already validate.
    Tb, Eb, Nb, Db, cb = 12, 3, 4, 5, 3
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(12), 3)
    s_b = jax.random.normal(k1, (Tb, Eb, Nb, Db))
    g_b = _unit(jax.random.normal(k2, (Tb, Eb, Nb, Db)))
    # Random terminals, per (env, agent) — enough of them to straddle horizons.
    done_b = (jax.random.uniform(k3, (Tb, Eb, Nb)) < 0.15).astype(jnp.float32)

    cos_b, valid_b = transition_cosine(s_b, g_b, cb, done=done_b)
    assert cos_b.shape == (Tb, Eb, Nb), cos_b.shape
    assert valid_b.shape == (Tb, Eb, Nb), valid_b.shape  # was silently (T,T,E)
    pooled_b = pool_goals(g_b, cb, done=done_b)
    assert pooled_b.shape == g_b.shape, pooled_b.shape
    ri_b = worker_intrinsic_reward(s_b, g_b, cb, done=done_b)
    assert ri_b.shape == (Tb, Eb, Nb), ri_b.shape

    # Every (env, agent) stream must equal the 1-D result on that slice.
    for e in range(Eb):
        for n in range(Nb):
            cos_1d, valid_1d = transition_cosine(
                s_b[:, e, n], g_b[:, e, n], cb, done=done_b[:, e, n]
            )
            assert jnp.allclose(cos_b[:, e, n], cos_1d, atol=1e-6), (e, n)
            assert jnp.allclose(valid_b[:, e, n], valid_1d, atol=1e-6), (e, n)
            assert jnp.allclose(
                pooled_b[:, e, n],
                pool_goals(g_b[:, e, n], cb, done=done_b[:, e, n]),
                atol=1e-5,
            ), (e, n)
            assert jnp.allclose(
                ri_b[:, e, n],
                worker_intrinsic_reward(
                    s_b[:, e, n], g_b[:, e, n], cb, done=done_b[:, e, n]
                ),
                atol=1e-6,
            ), (e, n)

    # An under-shaped mask must RAISE now, not broadcast into nonsense. (T, E) is
    # exactly what `Transition.done` is, so this is the realistic mistake; (T,E,1)
    # broadcasts correctly but would make masked-mean denominators N times small.
    for bad in (done_b[:, :, 0], done_b[:, :, :1]):
        for fn, args in (
            (transition_cosine, (s_b, g_b, cb)),
            (worker_intrinsic_reward, (s_b, g_b, cb)),
            (pool_goals, (g_b, cb)),
        ):
            try:
                fn(*args, done=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"{fn.__name__} accepted a {bad.shape} done mask"
                )

    # done=None still works batched (the mask degenerates to the float 1.0).
    assert transition_cosine(s_b, g_b, cb)[1].shape == (Tb, Eb, Nb)
    assert pool_goals(g_b, cb).shape == g_b.shape
    assert worker_intrinsic_reward(s_b, g_b, cb).shape == (Tb, Eb, Nb)
    print("[9] trainer shapes (T,E,N,D) + guards  OK")

    # ---------------------------------------------------------------- [10]
    # latent="local": the structural guarantees, asserted EXACTLY.
    #
    # Why exactly: under latent="centralized" the diagonal share of the block
    # Jacobian d s[i]/d obs_j measures 0.0631 on 12 trained arms against a
    # uniform 1/16 = 0.0625 (init 0.0623) — no localization at all, where a
    # block-diagonal manager reads 1.0 and a half-local one 0.128. So "roughly
    # local" is exactly the state this mode exists to escape, and a tolerance
    # here would let a partial reintroduction of joint-state mixing pass.
    for core_name, latent_name in [
        (c, l)
        for c in ("mlp", "dilated_lstm")
        for l in ("local", "local_global")
    ]:
        lm, lp, lc = init_manager(
            jax.random.PRNGKey(0),
            global_state_dim=N * OBS_DIM,
            n_agents=N,
            goal_dim=GOAL_DIM,
            hidden_dim=HIDDEN,
            core=core_name,
            horizon=HORIZON,
            latent=latent_name,
            obs_dim=OBS_DIM,
        )
        lobs = jax.random.normal(jax.random.PRNGKey(11), (N, OBS_DIM))

        # (a) exact locality
        jac = jax.jacrev(lambda o: lm.apply(lp, lc, o.reshape(-1), o)[2])(lobs)
        blocks = np.asarray(jnp.sqrt((jac ** 2).sum(axis=(1, 3))))
        off = blocks - np.diag(np.diag(blocks))
        assert np.abs(off).max() == 0.0, (core_name, np.abs(off).max())
        assert np.allclose(np.diag(blocks) / blocks.sum(1), 1.0)

        # (b) the encoder is SHARED -> permuting agents permutes s rows
        perm = np.roll(np.arange(N), 1)
        s_ref = lm.apply(lp, lc, lobs.reshape(-1), lobs)[2]
        s_perm = lm.apply(lp, lc, lobs[perm].reshape(-1), lobs[perm])[2]
        assert jnp.array_equal(s_ref[perm], s_perm), core_name

        # (c) g and s share one basis: the final projection is (D, D), not
        #     per-agent. Without this the cosine compares incommensurate axes.
        assert lp["params"]["f_goalhead"]["kernel"].shape == (GOAL_DIM, GOAL_DIM)

        # (d) gradient routing under the detach rule. Same property [8] checks
        #     on the centralized branch: the core consumes `s`, so f_enc and
        #     f_Mspace keep a signal through the GOAL arm even though the target
        #     arm is detached. Wire the core elsewhere and these go to exactly
        #     0.0 while every loss stays finite.
        Tl, El = 6, 2
        obs_seq = jax.random.normal(
            jax.random.PRNGKey(12), (Tl, El, N, OBS_DIM)
        )

        def _loss(pp, core_name=core_name):
            carry = lm.initialize_carry(jax.random.PRNGKey(0), (El,))
            _, gg, ss = lm.apply(pp, carry, obs_seq.reshape(Tl, El, -1), obs_seq)
            cc, vv = transition_cosine(ss, gg, HORIZON, detach_states=True)
            return -(cc * vv).sum() / jnp.maximum(vv.sum(), 1.0)

        gr = jax.grad(_loss)(lp)["params"]
        names = ("f_enc_0", "f_enc_1", "f_Mspace")
        if latent_name == "local_global":
            # f_percept feeds the GOAL arm only, which is the un-detached one, so
            # it must train too. If it ever reads 0.0 the extra input is dead
            # weight and the arm is silently plain `local`.
            names = names + ("f_percept_0", "f_percept_1")
        for nm in names:
            gsum = float(jnp.abs(gr[nm]["kernel"]).sum())
            assert gsum > 0.0, f"{core_name}/{latent_name}/{nm} got no gradient"

        # (e) `s` is blind to the global state in BOTH variants — that is what
        #     keeps r^I clean under local_global. The goal, however, must read it
        #     under local_global and must NOT under local.
        gs_a = jax.random.normal(jax.random.PRNGKey(15), (N * OBS_DIM,))
        gs_b = jax.random.normal(jax.random.PRNGKey(16), (N * OBS_DIM,))
        assert jnp.array_equal(
            lm.apply(lp, lc, gs_a, lobs)[2], lm.apply(lp, lc, gs_b, lobs)[2]
        ), f"{latent_name}: s moved when only the global state changed"
        goal_moved = not jnp.allclose(
            lm.apply(lp, lc, gs_a, lobs)[1], lm.apply(lp, lc, gs_b, lobs)[1]
        )
        assert goal_moved == (latent_name == "local_global"), (
            f"{latent_name}: goal-vs-global_state dependence is wrong "
            f"(moved={goal_moved})"
        )

    # (f) the centralized default is untouched by the new argument.
    cm, cp, cc_ = init_manager(
        jax.random.PRNGKey(0), N * OBS_DIM, N, GOAL_DIM, HIDDEN, "mlp", HORIZON
    )
    cgs = jax.random.normal(jax.random.PRNGKey(13), (N * OBS_DIM,))
    cobs = jax.random.normal(jax.random.PRNGKey(14), (N, OBS_DIM))
    a0 = cm.apply(cp, cc_, cgs)
    a1 = cm.apply(cp, cc_, cgs, cobs)
    assert jnp.array_equal(a0[1], a1[1]) and jnp.array_equal(a0[2], a1[2])

    # (g) an unknown latent raises rather than failing open.
    try:
        FeudalManager(n_agents=N, goal_dim=GOAL_DIM, latent="bogus").init(
            jax.random.PRNGKey(0), None, jnp.zeros(N * OBS_DIM)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown latent did not raise")
    print("[10] local latents: locality/shared/grad/gs-routing  OK")

    print("\nall manager checks passed")
