"""PPO update mirroring ``mappo_vanilla.mappo.MAPPOAgent.update`` in JAX.

Parity notes (vs. the torch implementation):
- GAE runs once per env on the team reward + shared critic value; vanilla tiles
  the reward per agent, but with identical rewards and a shared value the
  per-agent advantages are identical, so env-level GAE broadcast to agents is
  the same computation.
- Advantages are normalized per (env) trajectory over the rollout steps with an
  unbiased std (torch ``.std()``), matching vanilla's per-(env, agent) stream
  normalization.
- Minibatches are timestep-centric like ``update_shared``: one sample is one
  (step, env) element carrying all agents, the critic runs once per element,
  and the timestep minibatch size is ``(batch // n_minibatches) // n_agents``.
  (Deviation: the trailing partial minibatch is dropped — jit needs static
  shapes; torch's DataLoader keeps it.)
- Actor and critic use separate Adam optimizers; since they share no
  parameters this is equivalent to vanilla's single Adam over the combined
  loss ``policy + val_coef * value + ent_coef * entropy``.
- ``explained_variance`` is the same pre-update diagnostic vanilla computes.
"""

import warnings
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from algorithms.feudal_mappo_jax.types import MAPPOConfig, Transition
from algorithms.feudal_mappo_jax.network import (
    MAPPOCritic,
    evaluate_action,
)
from algorithms.feudal_mappo_jax.manager import (
    LOCAL_LATENTS,
    WORKER_ENCODERS,
    FeudalManager,
)
from algorithms.feudal_mappo_jax.worker import bind_goal, encode_obs, init_worker


class FeudalTrainState(NamedTuple):
    """Immutable container for the learned components.

    ``actor_ts`` / ``critic_ts`` keep their flat-MAPPO names (and their position
    first) so ``ppo_update`` and the stats plumbing read unchanged; the worker IS
    the actor here, just goal-conditioned.

    Each reward stream gets its own value function, because each regresses a
    different target: ``critic_ts`` the worker's *extrinsic* return under
    `gamma`, ``manager_critic_ts`` the manager's under `manager_gamma`, and
    ``intrinsic_critic_ts`` the *intrinsic* return under `gamma`. (What
    ``manager.py`` rules out is writing another critic *class* — all three reuse
    ``MAPPOCritic``.)

    ``intrinsic_critic_ts`` is ``None`` unless ``intrinsic_coef != 0``. `None` is
    a valid empty JAX pytree — the same trick the manager's `mlp` core uses for
    its carry — so at alpha=0 the slot costs nothing, no extra critic is built,
    and the msgpack checkpoint format is byte-for-byte what it was before the
    intrinsic stream existed (existing `feudal_a0` runs still resume).
    """

    actor_ts: TrainState  # FeudalWorker (goal-conditioned)
    critic_ts: TrainState  # worker value (extrinsic), always per-agent
    manager_ts: TrainState  # FeudalManager
    manager_critic_ts: TrainState  # V^M
    intrinsic_critic_ts: TrainState | None = None  # V^I, only when alpha != 0


# Back-compat alias: `run.py` and `trainer.py` refer to the train state by the
# flat-stack name in a few places.
ActorCriticTrainState = FeudalTrainState


def build_manager(config: MAPPOConfig, n_agents: int) -> FeudalManager:
    """The manager module, built from config alone.

    A free function because the module must be reconstructible *without* params
    in three places that never call ``create_train_state``: the rollout scan, the
    eval scan, and ``run.py:view()``. Flax modules are frozen dataclasses, so two
    calls with the same config produce interchangeable instances.
    """
    return FeudalManager(
        n_agents=n_agents,
        goal_dim=config.goal_dim,
        hidden_dim=config.manager_hidden_dim,
        core=config.manager_core,
        horizon=config.goal_horizon,
        latent=config.manager_latent,
    )


def create_train_state(
    rng: jax.Array,
    config: MAPPOConfig,
    obs_dim: int,
    global_state_dim: int,
    action_dim: int,
    discrete: bool,
    n_agents: int,
    n_critic_outputs: int = 1,
    n_manager_outputs: int = 1,
) -> FeudalTrainState:
    """Initialize worker/critic/manager/manager-critic params and optimizers.

    Worker hidden = ``hidden_dim``, centralized critic hidden = ``2 *
    hidden_dim`` (as in flat MAPPO). ``n_critic_outputs`` > 1 gives the worker
    critic a per-agent value head; for the feudal stack the trainer always passes
    ``n_agents`` there, because the intrinsic reward is per-agent.

    The worker/critic RNG split is left exactly as the flat stack had it, and the
    manager keys are folded in separately, so worker and critic init are
    bit-identical to ``mappo_jax`` at the same seed — any divergence from the flat
    baseline is then attributable to the goal columns, not to reseeding. The
    intrinsic critic's key is folded in under a *different* index for the same
    reason: adding it must not perturb any pre-existing init.
    """
    rng_actor, rng_critic = jax.random.split(rng)
    rng_manager, rng_manager_critic = jax.random.split(jax.random.fold_in(rng, 1))
    rng_intrinsic_critic = jax.random.fold_in(rng, 2)

    # The worker takes the raw goal width: `goal_embed_dim` (if set) is applied by
    # an internal bias-free Dense, so the module's input signature is goal_dim.
    #
    # Under `worker_encoder="shared"` the worker's OBSERVATION input is the
    # manager's encoder output, not the raw obs, so its first Dense is
    # manager_hidden_dim wide. `init_worker` infers that width purely from the
    # shape of its dummy, so this one line is the whole change — and putting it
    # here keeps both msgpack load paths in lockstep automatically, since each
    # rebuilds its target tree through this function.
    #
    # ⚠ It also makes the ACTOR tree depend on `manager_hidden_dim`, which
    # previously had no effect on the actor at all. A resume with a drifted
    # manager_hidden_dim now fails on `actor` as well as `manager` — loudly, which
    # is what we want, but it is new.
    worker_obs_dim = (
        config.manager_hidden_dim if config.worker_encoder != "none" else obs_dim
    )
    worker, actor_params = init_worker(
        rng_actor,
        obs_dim=worker_obs_dim,
        goal_dim=config.goal_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        discrete=discrete,
        goal_embed_dim=config.goal_embed_dim,
        normalize_pooled_goal=config.normalize_pooled_goal,
        zero_goal=config.zero_goal,
        worker_fusion=config.worker_fusion,
    )
    # One agent still needs an agent axis: squeezing (E, 1) to (E,) would
    # broadcast the truncation bootstrap against (E, 1) rewards into (E, E).
    keep_worker_axis = n_critic_outputs == n_agents
    critic = MAPPOCritic(
        hidden_dim=2 * config.hidden_dim, n_outputs=n_critic_outputs,
        keep_output_axis=keep_worker_axis,
    )
    critic_params = critic.init(rng_critic, jnp.zeros(global_state_dim))

    manager = build_manager(config, n_agents)
    manager_carry = manager.initialize_carry(rng_manager, ())
    # The local branches read per-agent observations, so their init pass needs a
    # correctly-shaped (n_agents, obs_dim) dummy; the centralized branch ignores
    # the argument entirely, so passing None there keeps its init bit-identical.
    manager_dummy_obs = (
        jnp.zeros((n_agents, obs_dim))
        if config.manager_latent in LOCAL_LATENTS
        else None
    )
    manager_params = manager.init(
        rng_manager, manager_carry, jnp.zeros(global_state_dim), manager_dummy_obs
    )
    manager_critic = MAPPOCritic(
        hidden_dim=2 * config.hidden_dim, n_outputs=n_manager_outputs,
        keep_output_axis=config.per_agent_rewards,
    )
    manager_critic_params = manager_critic.init(
        rng_manager_critic, jnp.zeros(global_state_dim)
    )

    # V^I: same shape as the worker's extrinsic critic (r^I is per-agent), but
    # its own params and its own Adam — it regresses a different return.
    # Built only when the intrinsic stream is live, so alpha=0 is a static no-op.
    intrinsic_critic = None
    intrinsic_critic_params = None
    if config.intrinsic_coef != 0.0:
        intrinsic_critic = MAPPOCritic(
            hidden_dim=2 * config.hidden_dim, n_outputs=n_critic_outputs,
            keep_output_axis=keep_worker_axis,
        )
        intrinsic_critic_params = intrinsic_critic.init(
            rng_intrinsic_critic, jnp.zeros(global_state_dim)
        )

    def _tx(lr):
        return optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adam(lr),
        )

    return FeudalTrainState(
        actor_ts=TrainState.create(
            apply_fn=worker.apply, params=actor_params, tx=_tx(config.lr)
        ),
        critic_ts=TrainState.create(
            apply_fn=critic.apply, params=critic_params, tx=_tx(config.lr)
        ),
        manager_ts=TrainState.create(
            apply_fn=manager.apply, params=manager_params, tx=_tx(config.manager_lr)
        ),
        manager_critic_ts=TrainState.create(
            apply_fn=manager_critic.apply,
            params=manager_critic_params,
            tx=_tx(config.manager_lr),
        ),
        intrinsic_critic_ts=(
            None
            if intrinsic_critic is None
            else TrainState.create(
                apply_fn=intrinsic_critic.apply,
                params=intrinsic_critic_params,
                tx=_tx(config.lr),
            )
        ),
    )


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and returns via reverse scan.

    Shape-agnostic in the trailing axis: with a scalar team reward everything is
    (n_steps, n_envs); with per-agent rewards everything carries a trailing agent
    axis (n_steps, n_envs, n_agents) and the identical recursion runs per agent.

    Args:
        rewards:    (n_steps, n_envs) team | (n_steps, n_envs, n_agents) per-agent
        values:     same shape as rewards
        dones:      (n_steps, n_envs) — episode-level, shared by all agents
        last_value: (n_envs,) | (n_envs, n_agents) — bootstrap value
        gamma, gae_lambda: scalars

    Returns:
        advantages, returns — both shaped like rewards
    """
    # `done` is per-env; add the agent axis so it broadcasts against per-agent
    # values instead of colliding with them.
    if rewards.ndim > dones.ndim:
        dones = dones[..., None]

    values_with_bootstrap = jnp.concatenate(
        [values, last_value[None]], axis=0
    )  # (n_steps+1, ...)

    def _scan_fn(gae, t):
        # t counts backward: 0 = last step, 1 = second-to-last, ...
        step = rewards.shape[0] - 1 - t
        not_done = 1.0 - dones[step]
        delta = (
            rewards[step]
            + gamma * values_with_bootstrap[step + 1] * not_done
            - values_with_bootstrap[step]
        )
        gae = delta + gamma * gae_lambda * not_done * gae
        return gae, gae

    _, advantages_reversed = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(last_value),
        jnp.arange(rewards.shape[0]),
    )
    advantages = advantages_reversed[::-1]  # reverse to chronological order
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


WORKER_OBJECTIVES = ("mixed", "intrinsic_only")


def validate_worker_objective(config: MAPPOConfig) -> None:
    """Check `worker_objective` and the settings it makes incoherent.

    ONE copy, called from both ``trainer.make_train`` (so any path that builds a
    config directly — tests included — is covered) and ``run.py`` (so a launched
    run fails at construction rather than at the first update). Two copies of
    these rules is exactly how a guard rots.

    Raises on the three combinations that would train happily and log healthy
    numbers while measuring nothing; *warns* on the fourth, which is a legitimate
    deliberate ablation but an expensive accident.
    """
    if config.worker_objective not in WORKER_OBJECTIVES:
        raise ValueError(
            f"worker_objective={config.worker_objective!r} is not one of "
            f"{WORKER_OBJECTIVES}. Caught at build time so a typo fails before a "
            "run is launched."
        )
    if config.worker_objective != "intrinsic_only":
        return

    # `intrinsic_coef != 0` is the STATIC gate that builds V^I, captures
    # `next_state_latent` and computes r^I at all. At 0 the worker's entire
    # objective would be a zero advantage.
    if config.intrinsic_coef == 0.0:
        raise ValueError(
            "worker_objective='intrinsic_only' requires intrinsic_coef != 0: at "
            "alpha=0 no intrinsic stream is built, so the worker would train on "
            "a zero advantage. Set intrinsic_coef=1.0 — its VALUE is inert here, "
            "it only has to be nonzero."
        )
    # alpha does not appear in this objective's advantage (adv_int is already
    # unit-std), so a schedule on it is not merely redundant: the default
    # "linear" reaching 0 reads as "the worker's objective annealed away", which
    # is precisely what it does NOT do.
    if config.intrinsic_anneal != "none":
        raise ValueError(
            "worker_objective='intrinsic_only' requires intrinsic_anneal='none', "
            f"got {config.intrinsic_anneal!r}. alpha is not a coefficient under "
            "this objective (adv = adv_int, no alpha), so an anneal schedule "
            "would be inert while reading as if it were decaying the worker's "
            "objective to zero."
        )

    # A warning, not a raise: this combination is the direct contrast that tests
    # whether locality is what matters, so it must stay runnable on purpose.
    if config.manager_latent not in LOCAL_LATENTS:
        warnings.warn(
            "worker_objective='intrinsic_only' with "
            f"manager_latent={config.manager_latent!r}: r^I is the worker's "
            "ENTIRE objective here, but this latent is not agent-local — "
            "measured over 12 trained arms, the diagonal share of d s[i]/d obs_j "
            "is 0.0631 against a uniform 1/N of 0.0625, i.e. agent i's own "
            "intrinsic reward moves as much when a TEAMMATE moves as when it "
            "does. Every worker would then be optimizing a team-aggregate signal "
            f"it does not control. Prefer a latent in {LOCAL_LATENTS} "
            "(e.g. 'local_private').",
            RuntimeWarning,
            stacklevel=2,
        )


def validate_worker_encoder(config: MAPPOConfig) -> None:
    """Check `worker_encoder` and the settings it makes incoherent.

    ONE copy, called from both ``trainer.make_train`` and ``run.py``, for exactly
    the reason :func:`validate_worker_objective` is — two copies of a guard is how
    a guard rots.

    Two raises are structural (the arm cannot be built), one is about gradient
    staleness, and two warnings mark combinations that must stay runnable because
    they are the direct contrasts the ablation exists to draw.
    """
    if config.worker_encoder not in WORKER_ENCODERS:
        raise ValueError(
            f"worker_encoder={config.worker_encoder!r} is not one of "
            f"{WORKER_ENCODERS}. Caught at build time so a typo fails before a "
            "run is launched."
        )
    if config.worker_encoder == "none":
        return

    # Only the local latents build `f_enc`. `centralized` has `f_percept` over the
    # flattened GLOBAL state — not per-agent, not observation-width — so there is
    # literally no tensor to hand the worker.
    if config.manager_latent not in LOCAL_LATENTS:
        raise ValueError(
            f"worker_encoder={config.worker_encoder!r} requires a manager_latent "
            f"in {LOCAL_LATENTS}, got {config.manager_latent!r}: only those build "
            "the per-agent encoder `f_enc` that the worker shares. The "
            "centralized branch has `f_percept` over the flattened global state, "
            "which is neither per-agent nor observation-width."
        )

    # The worker's contribution to `f_enc` is ONE full-batch gradient, taken at the
    # actor params the rollout used. `manager_update` adds it to the transition
    # PG's gradient and takes a single Adam step. With more manager epochs that
    # same cotangent would be re-added at epoch 1, 2, ... — parameter points it was
    # never computed at. Silent: every logged loss stays finite and healthy.
    if config.n_manager_epochs != 1:
        raise ValueError(
            f"worker_encoder={config.worker_encoder!r} requires "
            f"n_manager_epochs=1, got {config.n_manager_epochs}. The worker's "
            "encoder gradient is computed once per update at one parameter point; "
            "replaying it across manager epochs applies a gradient of a function "
            "evaluated where the parameters no longer are. (n_manager_epochs > 1 "
            "is already uncorrected off-policy for the transition PG itself.)"
        )

    # A warning, not a raise: the arm runs and is interpretable on its own terms,
    # it just stops being comparable to the rest of the goal-influence series.
    if config.worker_fusion == "concat":
        warnings.warn(
            f"worker_encoder={config.worker_encoder!r} with "
            "worker_fusion='concat': the goal is one block of a concatenated "
            "input, so widening that input from obs_dim to manager_hidden_dim "
            f"({config.hidden_dim} -> {config.manager_hidden_dim}) cuts the "
            "goal's share of layer-1 preactivation variance at init from ~9.8% "
            "to ~1%. normalize_pooled_goal's whole calibration was derived on the "
            "narrow geometry, and `worker_goal_column_ratio` cannot be read "
            "across the two. Prefer worker_fusion='film', where the goal reaches "
            "the policy through bias-free zero-init FiLM layers whose input is "
            "the goal, so the observation's width is irrelevant to it.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Also a warning: "does true perceptual sharing rescue the intrinsic-only
    # contract?" is a question the ablation is for, and a raise would foreclose it.
    if config.worker_objective == "intrinsic_only":
        warnings.warn(
            f"worker_encoder={config.worker_encoder!r} with "
            "worker_objective='intrinsic_only': `transition_cosine` detaches the "
            "cosine's TARGET arm precisely to stop gradient reaching `s` (FuN: "
            "'the dependence of s on theta is ignored... this avoids trivial "
            "solutions'). The worker's gradient into `f_enc` is subject to no "
            "such rule, and under this objective the worker's entire update "
            "direction is proportional to adv_int — correlated with the very "
            "quantity that detach protects. The trivial-solution channel is "
            "re-opened from the other side, across updates rather than within "
            "one. Watch d_cos_var and goal_direction_count across the WHOLE run.",
            RuntimeWarning,
            stacklevel=2,
        )


def _annealed_alpha(config: MAPPOConfig, progress: jnp.ndarray) -> jnp.ndarray:
    """alpha at this point in training. `progress` is a traced scalar in [0, 1].

    "linear" decays to exactly 0 at the end of training so the converged policy
    optimizes the true objective — with a constant alpha, that fraction of the
    worker's gradient would permanently point at a task-irrelevant target.

    `progress` MUST be traced (a jnp scalar). A python float would make it a
    compile-time constant and retrigger jit compilation of the caller on every
    update.
    """
    if config.intrinsic_anneal == "none":
        return jnp.float32(config.intrinsic_coef)
    if config.intrinsic_anneal == "linear":
        return jnp.float32(config.intrinsic_coef) * jnp.clip(
            1.0 - progress, 0.0, 1.0
        )
    raise ValueError(
        f"unknown intrinsic_anneal: {config.intrinsic_anneal!r} "
        "(expected 'linear' or 'none')"
    )


def ppo_update(
    train_state: ActorCriticTrainState,
    rng: jax.Array,
    trajectory: Transition,
    last_value: jnp.ndarray,
    config: MAPPOConfig,
    discrete: bool,
    last_value_int: jnp.ndarray | None = None,
    progress: jnp.ndarray | None = None,
) -> Tuple[ActorCriticTrainState, dict]:
    """Full PPO update: GAE → multi-epoch timestep-centric minibatch steps.

    This is the WORKER's update. It touches ``actor_ts``/``critic_ts`` (and
    ``intrinsic_critic_ts`` when the intrinsic stream is live); the two manager
    states ride through untouched (the manager is trained by ``manager_update``,
    which cannot share this machinery — its objective needs the time axis in
    order, which the shuffled minibatches destroy).

    **The two reward streams meet here, at the advantage level, and nowhere
    else.** Extrinsic and intrinsic each get their own GAE and their own
    normalization to unit std, and are then mixed as
    ``adv = adv_ext + alpha_t * adv_int``. That makes alpha a gradient fraction
    rather than a reward coefficient — the distinction the whole intrinsic path
    turns on, since the raw streams differ in magnitude by 300-600x during early
    training (see ``types.Params.intrinsic_coef``).

    Under ``config.worker_objective == "intrinsic_only"`` the extrinsic stream is
    dropped from the ACTOR's objective entirely (``adv = adv_int``, no alpha) so
    the worker's only job is to follow the manager's goals. The extrinsic GAE,
    the worker critic's regression and ``explained_variance`` all still run — the
    critic is kept trained so the param tree stays shape-identical to the
    ``"mixed"`` arm (checkpoints remain interchangeable) and so the extrinsic
    return stays a live diagnostic. See ``types.Model_Params.worker_objective``.

    Args:
        train_state: current FeudalTrainState
        rng: PRNG key
        trajectory: Transition with leading dim n_steps
        last_value: (n_envs, n_agents) worker bootstrap (GAE masks dones internally)
        config: hyperparameters
        discrete: action space type
        last_value_int: (n_envs, n_agents) intrinsic bootstrap; required iff
            ``config.intrinsic_coef != 0``
        progress: traced scalar in [0, 1], the fraction of training elapsed, used
            to anneal alpha. Defaults to 0.0 (= full alpha).

    Returns:
        updated train_state, loss metrics dict
    """
    # Static python flag: at alpha == 0 none of the intrinsic machinery below is
    # traced at all, so that path is byte-identical to the pre-intrinsic code.
    use_intrinsic = config.intrinsic_coef != 0.0
    # Static python flag: "intrinsic_only" drops the extrinsic advantage from the
    # ACTOR's objective, so the worker's only job is to follow the manager's
    # goals. Everything else (the extrinsic GAE, the worker critic's regression,
    # `explained_variance`) is untouched — see the branch at the mixing site.
    # Validated once at build time by `validate_worker_objective`, called from
    # both `trainer.make_train` and `run.py` — not re-checked here.
    intrinsic_only = config.worker_objective == "intrinsic_only"
    if progress is None:
        progress = jnp.float32(0.0)
    alpha_t = _annealed_alpha(config, progress)
    n_steps, n_envs, n_agents = trajectory.obs.shape[:3]
    obs_dim = trajectory.obs.shape[3]
    goal_dim = trajectory.pooled_goal.shape[-1]
    # Static: (n_steps, n_envs, n_agents) rewards => per-agent credit path.
    per_agent = trajectory.reward.ndim == 3
    # Static: a real (n_steps, n_envs, n_agents, action_dim) mask => masked
    # categorical. Envs without `avail_actions` store a scalar placeholder, so this
    # is False and the update is byte-identical to the pre-masking code.
    use_mask = trajectory.action_mask.ndim == 4

    dones = trajectory.done.astype(jnp.float32)
    advantages, returns = compute_gae(
        trajectory.reward,
        trajectory.value,
        dones,
        last_value,
        config.gamma,
        config.gae_lambda,
    )

    # Pre-update explained variance of the stored critic predictions
    explained_variance = 1.0 - jnp.var(returns - trajectory.value, ddof=1) / (
        jnp.var(returns, ddof=1) + 1e-8
    )

    def _normalize(a):
        """Center and scale to unit std over the rollout steps.

        Per-env for the team reward, per-(env, agent) under per-agent rewards —
        which is exactly vanilla's per-(env, agent) normalization.
        """
        return (a - a.mean(axis=0)) / (a.std(axis=0, ddof=1) + 1e-8)

    adv = _normalize(advantages)

    # --- Intrinsic stream: its own GAE, its own normalization, mixed in at the
    # advantage level. `compute_gae` is shape-agnostic, so it is reused as-is.
    int_metrics = {}
    if use_intrinsic:
        int_advantages, int_returns = compute_gae(
            trajectory.intrinsic_reward,
            trajectory.value_int,
            dones,
            last_value_int,
            config.gamma,
            config.gae_lambda,
        )
        int_explained_variance = 1.0 - jnp.var(
            int_returns - trajectory.value_int, ddof=1
        ) / (jnp.var(int_returns, ddof=1) + 1e-8)
        adv_int = _normalize(int_advantages)
        if intrinsic_only:
            # The worker's ONLY objective is the manager's goals. No alpha
            # factor: adv_int is already unit-std, so alpha would be a uniform
            # rescale of the entire actor gradient — and under the default
            # `intrinsic_anneal: "linear"` it would reach exactly 0, deleting the
            # objective outright while every logged loss stayed healthy. run.py
            # requires intrinsic_anneal="none" here so nothing reads as if a
            # schedule were running.
            adv = adv_int
        else:
            # Both terms are unit-std here, so alpha_t is exactly the mixing
            # ratio of the two gradient directions.
            adv = adv + alpha_t * adv_int
        int_metrics = {
            "alpha_current": alpha_t,
            # The coefficients that ACTUALLY multiply the two normalized streams
            # in the line above. `alpha_current` alone is misleading under
            # "intrinsic_only", where it is inert — logging the pair means the
            # stats say what the objective was without having to know the rule.
            "adv_ext_weight": jnp.float32(0.0 if intrinsic_only else 1.0),
            "adv_int_weight": jnp.float32(1.0) if intrinsic_only else alpha_t,
            # The RAW (pre-normalization) advantage scales of the two streams.
            # This pair is the diagnostic that reads the defect this whole path
            # exists to fix: it shows the magnitude gap that the per-stream
            # normalization is correcting. If they are within an order of
            # magnitude of each other, normalization is not doing much work; if
            # they differ by ~100x or more, a reward-level fold would be handing
            # essentially the entire gradient to the intrinsic term.
            "adv_ext_std_raw": advantages.std(ddof=1),
            "adv_int_std_raw": int_advantages.std(ddof=1),
            "intrinsic_explained_variance": int_explained_variance,
        }

    # --- Timestep-centric flattening: one sample per (step, env) ---
    total_ts = n_steps * n_envs
    obs_ts = trajectory.obs.reshape(total_ts, n_agents, obs_dim)
    gs_ts = trajectory.global_state.reshape(total_ts, -1)
    act_ts = trajectory.action.reshape(
        total_ts, n_agents, *trajectory.action.shape[3:]
    )
    lp_ts = trajectory.log_prob.reshape(total_ts, n_agents)
    # Per-agent activity mask (1.0 for real decisions, 0.0 for offline agents under
    # staggered starts). All-ones for every ordinary run — the masked means below
    # then reduce to plain means, keeping the update byte-identical.
    active_ts = trajectory.active_mask.reshape(total_ts, n_agents)
    # Legal-action mask, flattened like obs so it stays paired with its own agent.
    mask_ts = (
        trajectory.action_mask.reshape(total_ts, n_agents, -1) if use_mask else None
    )
    # The conditioning the worker ACTED on. Flattened agent-major below, exactly
    # like obs, so goal row k = m*n_agents + i pairs with obs row k.
    pg_ts = trajectory.pooled_goal.reshape(total_ts, n_agents, goal_dim)
    if per_agent:
        adv_ts = adv.reshape(total_ts, n_agents)
        ret_ts = returns.reshape(total_ts, n_agents)
    else:
        adv_ts = adv.reshape(total_ts)
        ret_ts = returns.reshape(total_ts)
    # V^I's regression target. Always per-agent (r^I is), independently of
    # `per_agent` — which in this stack is True anyway, since the worker's reward
    # is broadcast to the agent axis in the collector.
    int_ret_ts = (
        int_returns.reshape(total_ts, n_agents) if use_intrinsic else None
    )

    # minibatch_size agent-samples => minibatch_size // n_agents timesteps
    ts_minibatch_size = max(
        1, (total_ts // config.n_minibatches) // n_agents
    )
    n_minibatches = total_ts // ts_minibatch_size

    # --- shared-encoder plumbing -------------------------------------------------
    # Hoisted out of BOTH scans on purpose. `ppo_update` never moves the manager —
    # `_epoch_step` only `_replace`s the actor/critic states, and `manager_update`
    # runs afterwards — so these are provably constant for the whole function.
    # Reading them here rather than off the scan carry makes that syntactically
    # obvious and keeps a traced value out of the inner scan.
    share_encoder = config.worker_encoder != "none"  # static
    manager_params = train_state.manager_ts.params
    manager_apply = train_state.manager_ts.apply_fn

    def _worker_obs(manager_p, raw_obs):
        """The worker's observation input: raw, or the manager's encoding of it.

        Encoded at the ALREADY-FLATTENED ``(rows, obs_dim)`` rank, which is why
        ``trainer._actor_forward`` also flattens before it encodes: a different
        leading shape can select a different XLA matmul kernel, and the two paths
        would then agree to ~1e-6 rather than bitwise — silently weakening "the
        PPO ratio is exactly 1" from an equality into a tolerance, for nothing.
        """
        return encode_obs(manager_apply, manager_p, raw_obs) if share_encoder else raw_obs

    def _batch(ids):
        """Gather one (mini)batch, flattened agent-major.

        Row ``k = m*n_agents + i`` is agent ``i`` of timestep ``m`` in EVERY
        component — obs, goal, action, log-prob, advantage, active and legal-action
        masks all use the identical reshape, which is what keeps each agent paired
        with its own directive.
        """
        n_flat = ids.shape[0] * n_agents
        active_pa = active_ts[ids]
        return dict(
            obs=obs_ts[ids].reshape(n_flat, obs_dim),
            goal=pg_ts[ids].reshape(n_flat, goal_dim),
            actions=act_ts[ids].reshape(n_flat, *act_ts.shape[2:]),
            old_lp=lp_ts[ids].reshape(n_flat),
            # Same mask the actions were sampled under — without it the ratio
            # compares two different distributions and PPO's weight is meaningless.
            mask=mask_ts[ids].reshape(n_flat, -1) if use_mask else None,
            active_pa=active_pa,
            active=active_pa.reshape(n_flat),
            # per_agent: each agent carries its own advantage, agent-major like
            # obs. Otherwise the env-level advantage is broadcast (identical per
            # agent).
            adv=(
                adv_ts[ids].reshape(n_flat)
                if per_agent
                else jnp.repeat(adv_ts[ids], n_agents)
            ),
            gs=gs_ts[ids],
            returns=ret_ts[ids],
        )

    def _surrogate(actor_params, manager_p, actor_apply, mb):
        """PPO clipped surrogate + entropy over one batch.

        ONE body, called by the per-minibatch actor step AND by the encoder-
        gradient pass below, so the two provably optimize the same objective. Two
        copies would let the encoder be trained on a gradient of something the
        worker is not actually maximizing — and nothing logged would say so.

        `bind_goal` freezes the stored conditioning into the worker's apply_fn,
        restoring the flat (params, obs) signature `evaluate_action` expects — no
        forked evaluation path. Using the STORED pooled goal (not a recomputed
        one) is what keeps the importance ratio valid.
        """
        log_probs, entropy = evaluate_action(
            bind_goal(actor_apply, mb["goal"]),
            actor_params,
            _worker_obs(manager_p, mb["obs"]),
            mb["actions"],
            discrete,
            action_mask=mb["mask"],
        )
        ratio = jnp.exp(log_probs - mb["old_lp"])
        surr1 = ratio * mb["adv"]
        surr2 = jnp.clip(
            ratio, 1.0 - config.eps_clip, 1.0 + config.eps_clip
        ) * mb["adv"]
        # Mask offline agents out of the policy gradient (their proposed skill was
        # never executed). All-ones => plain mean.
        denom = jnp.maximum(mb["active"].sum(), 1.0)
        policy_loss = -(jnp.minimum(surr1, surr2) * mb["active"]).sum() / denom
        entropy_loss = -(entropy * mb["active"]).sum() / denom
        total = policy_loss + config.ent_coef * entropy_loss
        return total, (policy_loss, entropy_loss)

    # The worker's contribution to the SHARED encoder: one gradient of the same
    # surrogate, w.r.t. the manager's params, taken at the actor params the rollout
    # acted under — i.e. BEFORE any PPO step. `manager_update` adds it to the
    # transition PG's gradient and takes a single Adam step.
    #
    # Three properties make this the right form, and each rules out an alternative
    # that looks simpler:
    #
    #   * At the pre-PPO actor params the importance ratio is exactly 1, so
    #     `surr1 == surr2`, the clip is provably inactive and `min` is a no-op.
    #     The gradient reduces to `-mean(adv * dlog pi)` — plain REINFORCE with a
    #     GAE baseline, structurally the SAME form as the manager's own
    #     `-mean(adv * d d_cos)`. Two full-batch means, same batch, same advantage,
    #     same parameter point, summed. Accumulating across PPO's own minibatch
    #     steps instead would mix gradients taken at 40-odd different actor
    #     iterates, which is not the gradient of anything.
    #   * It is scale-commensurate with the manager's term by construction, with no
    #     dependence on `n_minibatches` — which would otherwise become a silent
    #     weight on how much authority the worker has over the shared encoder.
    #   * `ppo_update` does NOT apply it. The manager staying frozen here is what
    #     keeps `manager_update`'s recompute equal to the rollout's goals, and what
    #     keeps the epoch-0 ratio exactly 1. Both are pinned by seam tests.
    #
    # Computed in CHUNKS at fixed parameters, not as one call: chunk size matches
    # PPO's own minibatch, so peak activation memory is unchanged (a single
    # full-batch forward+backward through a manager_hidden_dim-wide encoder over
    # T*E*N rows is ~n_minibatches times the peak of one PPO step). Equal-size
    # chunks make the mean of the per-chunk means exactly the full-batch mean, so
    # this is the same number, not an approximation. The trailing partial batch is
    # dropped, as PPO's own partition already does.
    if share_encoder:
        a_params, a_apply = train_state.actor_ts.params, train_state.actor_ts.apply_fn

        def _enc_chunk(acc, mb_idx):
            ids = mb_idx * ts_minibatch_size + jnp.arange(ts_minibatch_size)
            g = jax.grad(
                lambda mp: _surrogate(a_params, mp, a_apply, _batch(ids))[0]
            )(manager_params)
            return jax.tree.map(jnp.add, acc, g), None

        enc_grads, _ = jax.lax.scan(
            _enc_chunk,
            jax.tree.map(jnp.zeros_like, manager_params),
            jnp.arange(n_minibatches),
        )
        enc_grads = jax.tree.map(lambda x: x / n_minibatches, enc_grads)
    else:
        # A valid empty pytree, the same idiom `intrinsic_critic_ts` uses at
        # alpha=0 — it rides every downstream signature for free.
        enc_grads = None

    def _epoch_step(carry, _epoch_idx):
        train_state, rng = carry
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, total_ts)

        def _minibatch_step(carry, mb_idx):
            actor_ts, critic_ts, int_critic_ts = carry
            start = mb_idx * ts_minibatch_size
            mb_ids = jax.lax.dynamic_slice(perm, (start,), (ts_minibatch_size,))

            mb = _batch(mb_ids)
            mb_active_pa = mb["active_pa"]
            mb_gs = mb["gs"]
            mb_returns = mb["returns"]

            # --- Actor loss ---
            def actor_loss_fn(actor_params):
                return _surrogate(actor_params, manager_params, actor_ts.apply_fn, mb)

            (_, (policy_loss, entropy_loss)), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(actor_ts.params)
            actor_ts = actor_ts.apply_gradients(grads=actor_grads)

            # --- Critic loss (once per timestep; shared value vs team return) ---
            def critic_loss_fn(critic_params):
                values = critic_ts.apply_fn(critic_params, mb_gs)
                if per_agent:
                    # Per-agent value head: mask offline agents' heads out (their
                    # return is a masked-out 0). Team critic (scalar) is always
                    # valid, so it stays a plain mean. All-ones => plain mean.
                    sq = (values - mb_returns) ** 2
                    value_loss = (sq * mb_active_pa).sum() / jnp.maximum(
                        mb_active_pa.sum(), 1.0
                    )
                else:
                    value_loss = jnp.mean((values - mb_returns) ** 2)
                return config.val_coef * value_loss, value_loss

            (_, value_loss), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(critic_ts.params)
            critic_ts = critic_ts.apply_gradients(grads=critic_grads)

            # --- V^I loss: same regression, different stream. Its target is the
            # intrinsic return, so it needs its own params and its own Adam.
            if use_intrinsic:
                mb_int_returns = int_ret_ts[mb_ids]

                def int_critic_loss_fn(int_critic_params):
                    values = int_critic_ts.apply_fn(int_critic_params, mb_gs)
                    sq = (values - mb_int_returns) ** 2
                    v_loss = (sq * mb_active_pa).sum() / jnp.maximum(
                        mb_active_pa.sum(), 1.0
                    )
                    return config.val_coef * v_loss, v_loss

                (_, int_value_loss), int_grads = jax.value_and_grad(
                    int_critic_loss_fn, has_aux=True
                )(int_critic_ts.params)
                int_critic_ts = int_critic_ts.apply_gradients(grads=int_grads)
            else:
                int_value_loss = jnp.float32(0.0)

            # Stats mirror vanilla: raw component losses + the combined total
            losses = {
                "total_loss": (
                    policy_loss
                    + config.val_coef * value_loss
                    + config.ent_coef * entropy_loss
                ),
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "intrinsic_value_loss": int_value_loss,
            }
            return (actor_ts, critic_ts, int_critic_ts), losses

        (actor_ts, critic_ts, int_critic_ts), mb_losses = jax.lax.scan(
            _minibatch_step,
            (
                train_state.actor_ts,
                train_state.critic_ts,
                train_state.intrinsic_critic_ts,
            ),
            jnp.arange(n_minibatches),
        )
        # `_replace` rather than a fresh construction: the two manager states are
        # not part of this update and must ride through untouched. At alpha=0
        # `intrinsic_critic_ts` is None on both sides — an empty pytree that rides
        # the scan carry for free.
        new_ts = train_state._replace(
            actor_ts=actor_ts,
            critic_ts=critic_ts,
            intrinsic_critic_ts=int_critic_ts,
        )
        return (new_ts, rng), mb_losses

    (train_state, rng), epoch_losses = jax.lax.scan(
        _epoch_step,
        (train_state, rng),
        jnp.arange(config.n_epochs),
    )

    # Average losses across epochs and minibatches
    mean_losses = jax.tree.map(lambda x: x.mean(), epoch_losses)
    mean_losses["explained_variance"] = explained_variance
    # Goal-influence diagnostics, routed by fusion: the concat metric reads a
    # goal block that FiLM's first Dense does not have, and used to return a
    # silent NaN for it (84/84 feudal trials). Different keys per fusion is
    # deliberate — one number cannot mean the same thing for an input column and
    # for a multiplicative gain, and pretending otherwise is what produced the
    # two documented misreadings of `worker_goal_column_ratio`.
    if config.worker_fusion == "film":
        mean_losses.update(
            _film_goal_metrics(
                train_state.actor_ts,
                trajectory.obs,
                trajectory.pooled_goal,
                # These metrics apply the WORKER, so under a shared encoder they
                # must feed it the same thing the policy eats. Left on raw obs
                # this is not a wrong number, it is a trace-time shape error at
                # the default fusion of both shared-encoder arms.
                encode=(lambda o: _worker_obs(manager_params, o)) if share_encoder else None,
            )
        )
    else:
        mean_losses["worker_goal_column_ratio"] = _goal_column_ratio(
            train_state.actor_ts.params,
            # The width of the worker's OBSERVATION block, which under sharing is
            # the encoder's, not the env's. Passing the raw obs_dim here would not
            # raise: the kernel is then WIDER than obs_dim (manager_hidden_dim +
            # goal_width), so the concat-only guard does not fire and the function
            # silently slices an "obs block" and a "goal block" that are neither.
            config.manager_hidden_dim if share_encoder else obs_dim,
        )
    if not use_intrinsic:
        # The per-minibatch placeholder carries no information at alpha=0; drop
        # it so the stats keys stay exactly what they were.
        mean_losses.pop("intrinsic_value_loss", None)
    mean_losses.update(int_metrics)
    if share_encoder:
        # Who is actually shaping the shared representation. The structural risk of
        # this arm is that the worker's PPO gradient swamps the manager's transition
        # PG — which reaches `f_enc` only through the cosine's GOAL arm, the target
        # arm being detached — leaving `f_enc` as the worker's trunk that the
        # manager happens to read. The companion `manager_encoder_grad_norm` and
        # `worker_manager_encoder_grad_cos` are logged by `manager_update`, which is
        # where the manager's own gradient exists.
        mean_losses["worker_encoder_grad_norm"] = _encoder_grad_norm(enc_grads)

    return train_state, mean_losses, enc_grads


def _encoder_subtree(grads):
    """The `f_enc_*` leaves of a manager gradient/param tree, flattened.

    The worker only ever reads `f_enc`, so `jax.grad` returns exact zeros on every
    other manager leaf (it instantiates a full cotangent). Restricting the norm to
    `f_enc` anyway keeps the diagnostic meaningful if that ever stops being true —
    and makes the "and nothing else" seam test cheap to write.
    """
    params = grads.get("params", grads)
    return [v for k, v in params.items() if k.startswith("f_enc")]


def _encoder_grad_norm(grads) -> jnp.ndarray:
    """L2 norm of the `f_enc` block of a manager gradient tree."""
    if grads is None:
        return jnp.float32(0.0)
    leaves = jax.tree.leaves(_encoder_subtree(grads))
    if not leaves:
        return jnp.float32(0.0)
    return jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))


def _encoder_grad_cosine(a, b) -> jnp.ndarray:
    """Cosine between two gradient trees, restricted to the `f_enc` block.

    Restricted on purpose: `b` (the worker's) is exactly zero outside `f_enc`, so a
    whole-tree cosine would be diluted by the manager's own goal-path gradient and
    would drift toward 0 for reasons that say nothing about the shared encoder.
    """
    la = jax.tree.leaves(_encoder_subtree(a))
    lb = jax.tree.leaves(_encoder_subtree(b))
    if not la or not lb:
        return jnp.float32(0.0)
    dot = sum(jnp.sum(x * y) for x, y in zip(la, lb))
    na = jnp.sqrt(sum(jnp.sum(x**2) for x in la))
    nb = jnp.sqrt(sum(jnp.sum(y**2) for y in lb))
    return dot / (na * nb + 1e-12)


def _clip_tree(grads, max_norm: float):
    """Scale `grads` down so its global L2 norm is at most `max_norm`.

    The same rule as ``optax.clip_by_global_norm``, applied by hand because it has
    to run on ONE addend before the sum rather than on the manager's whole
    gradient — see the call site for why that distinction is load-bearing.
    Stateless, so it needs no slot in the train state.
    """
    leaves = jax.tree.leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda x: x * scale, grads)


def _goal_column_ratio(actor_params, obs_dim: int) -> jnp.ndarray:
    """How hard the worker's first layer listens to the goal vs the observation.

    ``FeudalWorker`` fuses by **concatenation**, so the worker can learn to ignore
    the manager entirely by driving the goal columns of layer 1 to zero — the
    degeneracy FuN avoids with a bias-free bilinear ``U(obs) @ phi(g)``, where a
    zero goal expresses no preference and cannot be tuned out. Nothing else
    logged would reveal it: the goals stay unit-norm and diverse, the manager's
    own loss keeps improving, and the hierarchy is simply disconnected.

    Per-input-dimension RMS ratio, so the two blocks are compared fairly despite
    their different widths; ~1.0 at orthogonal init. A decay toward 0 is the
    trigger to switch fusion (set ``goal_embed_dim``, or go bilinear).

    ⚠ TWO WAYS TO MISREAD THIS SERIES, both of which have already happened.

    1. It is a ratio of WEIGHTS, blind to the scale of the inputs they multiply.
       Under the pre-2026-08-28 raw-sum fusion the goal arrived at ~5x the
       observation's per-dimension RMS, so a 1.0 weight ratio meant the goal
       owned ~92% of the layer's preactivation variance. With
       ``normalize_pooled_goal=True`` the goal enters at unit scale and the
       weight ratio and the contribution ratio coincide. The series is therefore
       NOT like-for-like across that flag.
    2. **A falling ratio is not evidence that the goal columns shrank.** It has
       a denominator. Measured on the trained ``feudal_a0``/``n01``/``n05``
       checkpoints, the ratio fell to 0.36-0.65 while the goal block **GREW
       2.2-4.7x** from init — the obs block simply grew 6-10x, faster. Combined
       with the 5x oversized input, the goal still held **59-78%** of layer-1
       variance in those runs, i.e. the worker had not disconnected from the
       manager at all; it was relatively downweighting an input that still
       dominated it. The zero-goal ablation makes the point cleanly: with the
       goal columns provably frozen, the logged ratio still drifts 1.005 ->
       0.813. To claim goal-blindness, check ``goal_rms`` against its
       shape-determined init (0.109109 at hidden_dim=168), not this ratio.
    """
    kernel = actor_params["params"]["MAPPOActor_0"]["Dense_0"]["kernel"]
    if kernel.shape[0] <= obs_dim:
        # FiLM: the goal never enters this layer, so `kernel[obs_dim:]` is an
        # empty (0, hidden) slice and the ratio below is `mean of nothing` = NaN.
        # It used to return that NaN silently: measured, `worker_goal_column_ratio`
        # is NaN at every one of ~3000 logged points in 84 of 84 feudal trials
        # across both 12a batches, i.e. there was NO logged signal at all about
        # whether the manager->worker channel was connected. Raise instead —
        # `ppo_update` routes FiLM arms to `_film_goal_metrics`, so reaching here
        # means the routing is wrong, and a NaN series is not a diagnostic.
        raise ValueError(
            f"_goal_column_ratio is concat-only: the worker's first Dense has "
            f"input width {kernel.shape[0]} <= obs_dim {obs_dim}, so there is no "
            f"goal block. Use `_film_goal_metrics` for worker_fusion='film'."
        )
    obs_block, goal_block = kernel[:obs_dim], kernel[obs_dim:]
    obs_rms = jnp.sqrt(jnp.mean(obs_block**2))
    goal_rms = jnp.sqrt(jnp.mean(goal_block**2))
    return goal_rms / (obs_rms + 1e-8)


_SATURATED = 0.95
"""|tanh(z)| above which a hidden unit counts as saturated."""

_FILM_METRIC_ROWS = 8192
"""Rows subsampled for the FiLM diagnostics.

The full flattened batch is n_steps*n_envs*n_agents (402k at the 12a config),
and these metrics need two extra actor forwards over it. A strided subsample is
plenty for four RMS-style statistics and keeps the cost negligible; STRIDED
rather than the first N so the sample spans the whole rollout instead of only
its first timesteps (the goal ring fills over the first `goal_horizon` steps, so
a head slice would systematically over-weight the ramp).
"""


def _film_goal_metrics(actor_ts, obs, goal, encode=None) -> dict:
    """How hard does the manager's directive actually drive a FiLM worker?

    The FiLM counterpart of :func:`_goal_column_ratio`, and the series that stops
    a zero behavioural gap being misread as a disconnected channel. Measured on
    the trained `mjx_12a_3o_trunc_1024` arms, `eval_gap_zeroed` ~ 0 on every
    centralized arm — which reads as "the worker ignores the goal" — while
    `worker_film_gain_rms` is 0.78-0.84 and swapping an agent's goal moves its
    action by 87-114% of the action's own magnitude. The channel is wide open;
    the behaviour it induces is simply orthogonal to return. Those two readings
    call for opposite fixes (change the fusion vs. change the objective), so the
    distinction is not cosmetic.

    Metrics, and why each is shaped the way it is:

    ``worker_film_gain_rms``
        RMS of ``gamma`` over batch x units x both layers. The modulation is
        ``h <- (1 + gamma)*h + beta``, so this is the RMS fractional swing of
        each unit's gain around 1. **Dimensionless, so it needs no denominator**
        — which is precisely what made `worker_goal_column_ratio` misreadable
        twice (see its docstring: a ratio of weights is blind to input scale,
        and a falling ratio can mean the denominator grew). **Exactly 0.0 at
        init** by FiLM's zero-init, so the series starts at a known floor and
        any rise is influence that was earned.

    ``worker_film_shift_ratio``
        Mean over layers of RMS(beta_i) / RMS(pre-modulation preactivation_i).
        `beta` is ADDITIVE, so unlike gamma it does need the scale of what it is
        added to — and per-layer rather than pooled, see the comment at the
        computation. Also 0.0 at init.

    ``worker_tanh_saturation``
        Fraction of modulated preactivations with ``|tanh(z)| > 0.95``. FiLM
        sits BEFORE the nonlinearity — the placement that moves units into and
        out of saturation rather than rescaling an already-squashed value — so a
        large gain can change the trunk's operating regime outright. On
        `feudal_film_n01_local` the modulated trunk runs at 0.75-0.85 against
        0.16-0.20 for the same weights with the goal zeroed (layer-0
        preactivation RMS 7.2-9.1 vs ~1.4). That is the mechanism behind that
        arm's collapse to ~1.5 return when the goal is removed: zeroing does not
        merely delete a directive, it relocates the trunk to a never-trained
        operating point. ⚠ A HIGH value is not on its own a pathology — the
        goal-free `feudal_film_zerogoal` control sits at 0.56-0.61. What marks
        the pathology is the gap between the modulated and unmodulated trunk.

    ``worker_goal_action_delta``
        RMS(mu(obs, w) - mu(obs, roll(w))) / RMS(mu(obs, w)): the END-TO-END
        behavioural sensitivity. The three above are internal and could in
        principle be absorbed downstream; this one asks whether the goal changes
        what the agent does. The re-pairing is a HALF-BATCH roll, i.e. a goal
        from an unrelated (timestep, env, agent) — NOT the agent-axis roll of
        `eval_gap_permuted`, which degenerates to a no-op on a manager that has
        collapsed to one team direction. That difference is the point: this
        series stays meaningful on exactly the arms where the permutation nulls
        stop being informative.

    All four are FORWARD-ONLY on arrays `ppo_update` already holds: one apply
    with the `diagnostics` collection mutable, plus one more on a rolled goal.
    Ungated, like every other manager diagnostic — gating would make future runs
    non-comparable with these.

    REFERENCE VALUES, measured on the trained `mjx_12a_3o_trunc_1024` arms
    (3 seeds each, on-policy rollout, this exact function)::

        arm                     gain_rms   shift_ratio  saturation  act_delta
        feudal_film   (a=0)    0.72-0.76    0.164-0.166  0.50-0.52  0.91-1.09
        feudal_film_n05        0.90-1.06    0.187-0.211  0.53-0.54  1.22-1.33
        feudal_film_local      2.30-2.57    0.174-0.198  0.49-0.66  0.99-1.14
        feudal_film_n01_local  3.89-5.22    0.249-0.306  0.75-0.85  0.63-1.28
        feudal_film_zerogoal   0.000        0.000        0.56-0.61  0.000

    The last row is a POSITIVE CONTROL and it is exact, not approximate: on a
    `zero_goal` arm the worker zeroes the goal inside the module and FiLM's
    gamma/beta Dense layers are bias-free, so `gamma(0) = beta(0) = 0` and all
    three goal-driven metrics must read **exactly 0.0**. Anything else means the
    metrics are reading something other than the live modulation. (`saturation`
    is correctly nonzero there — it is a property of the trunk, not of the goal.)

    Read against those: every centralized arm has `eval_gap_zeroed` ~ 0 while
    running a gain that swings +-75% and an action delta near 1.0. "The worker
    ignores the goal" is not available as an explanation for those runs.

    Args:
        actor_ts: the worker's TrainState (`apply_fn` + `params`).
        obs: ``(..., obs_dim)``, flattened agent-major by the caller.
        goal: ``(..., goal_dim)`` pooled goals, paired row-for-row with `obs`.
        encode: under ``worker_encoder="shared"``, the manager's encoder applied to
            the subsampled rows. These metrics apply the WORKER, so they must feed
            it what the policy eats; on raw obs the first Dense's width would not
            even match and this would be a trace-time error, not a wrong number.
            ``None`` (default) leaves every non-shared arm byte-identical.
    """
    obs = obs.reshape(-1, obs.shape[-1])
    goal = goal.reshape(-1, goal.shape[-1])
    stride = max(1, obs.shape[0] // _FILM_METRIC_ROWS)
    obs, goal = obs[::stride], goal[::stride]
    # After the subsample, so the encoder runs on _FILM_METRIC_ROWS rows rather
    # than the whole trajectory.
    if encode is not None:
        obs = encode(obs)

    out, state = actor_ts.apply_fn(
        actor_ts.params, obs, goal, mutable=["diagnostics"]
    )
    sown = state["diagnostics"]
    # Each FiLM layer sows a 1-tuple per call.
    def _layer(i, key):
        return jnp.concatenate(sown[f"film_{i}"][key], axis=-1)

    rms = lambda x: jnp.sqrt(jnp.mean(x**2))
    n_films = 2

    # gamma is dimensionless and `post` feeds a fraction, so both pool across
    # layers cleanly. `shift_ratio` does NOT: the two layers' preactivations
    # differ in scale (measured ~4.5x on the trained arms, layer 1 being the
    # larger), so a ratio of RMS over the concatenated blocks is dominated by
    # whichever layer is bigger — the same denominator trap documented twice for
    # `_goal_column_ratio`. Average the per-layer ratios instead, so each layer
    # is compared against its own scale and weighted equally.
    gain = jnp.concatenate([_layer(i, "gain") for i in range(n_films)], axis=-1)
    post = jnp.concatenate([_layer(i, "post") for i in range(n_films)], axis=-1)
    shift_ratio = jnp.mean(
        jnp.stack(
            [
                rms(_layer(i, "shift")) / (rms(_layer(i, "pre")) + 1e-8)
                for i in range(n_films)
            ]
        )
    )

    # Continuous heads return (mean, log_std); discrete return logits alone.
    mean_of = lambda y: y[0] if isinstance(y, tuple) else y
    # HALF-BATCH roll on the strided rows: each observation is re-paired with a
    # goal from an unrelated (timestep, env, agent). Deliberately NOT the
    # agent-axis roll `eval_gap_permuted` uses — that one degenerates when the
    # manager has collapsed to one team direction (swapping teammates' goals
    # then changes nothing), which would make a low reading ambiguous between
    # "the worker is insensitive" and "the goals were already identical". A
    # half-batch shift is maximally decorrelated and is the same fixed,
    # rng-free rearrangement every update, so the series is comparable over time.
    other = jnp.roll(goal, goal.shape[0] // 2, axis=0)
    rolled = mean_of(actor_ts.apply_fn(actor_ts.params, obs, other))
    real = mean_of(out)

    return {
        "worker_film_gain_rms": rms(gain),
        "worker_film_shift_ratio": shift_ratio,
        "worker_tanh_saturation": jnp.mean(jnp.abs(jnp.tanh(post)) > _SATURATED),
        "worker_goal_action_delta": rms(real - rolled) / (rms(real) + 1e-8),
    }


# ---------------------------------------------------------------------------
# Manager update (FuN's transition policy gradient)
# ---------------------------------------------------------------------------


def _masked_mean(x, mask):
    return (x * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def _effective_rank(s: jnp.ndarray) -> jnp.ndarray:
    """Entropy-based effective rank of the latent state's covariance, in [1, d].

    The headline collapse diagnostic. If ``f_Mspace`` degenerates to rank 1 —
    ``s_t = phi(x_t) * u`` for a fixed direction ``u`` — then every ``s_t - s_{t-i}``
    is parallel to ``u``, the goal head can emit ``g = u``, and the cosine pins at
    +-1 for every state and every action the worker could take. The intrinsic
    reward becomes a constant, which advantage centering annihilates: the whole
    mechanism goes inert while the loss curves still look healthy. Nothing else
    logged here would show it, because the metric that would expose the failure
    is the one that collapsed.
    """
    d = s.shape[-1]
    flat = s.reshape(-1, d)
    flat = flat - flat.mean(axis=0, keepdims=True)
    cov = (flat.T @ flat) / jnp.maximum(flat.shape[0], 1)
    # Symmetric-eigenvalue round-off can produce tiny negatives; floor them.
    lam = jnp.clip(jnp.linalg.eigvalsh(cov), 0.0)
    p = lam / (lam.sum() + 1e-12)
    entropy = -jnp.sum(p * jnp.log(p + 1e-12))
    return jnp.exp(entropy)


def _agent_gram(v: jnp.ndarray) -> jnp.ndarray:
    """Pairwise cosines between the agent-axis rows of ``v`` (..., n_agents, d).

    The three agent-diversity diagnostics below are all functions of this one
    (..., N, N) matrix, so it is computed once per quantity rather than per
    metric.
    """
    u = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)
    return jnp.einsum("...id,...jd->...ij", u, u)


def _mean_pairwise_cosine(gram: jnp.ndarray) -> jnp.ndarray:
    """Mean off-diagonal entry of ``gram``, averaged over the leading axes.

    Detects the residual `manager.py` flags as unguarded: `s` and `g` are each
    one Dense reshaped to (N, goal_dim), so nothing *structurally* forces the N
    rows to differ. Under uniformity pressure per-agent goals silently degrade to
    a single team goal — every shape and assertion still passes — and this is the
    only thing that would say so.

    Read it together with `_agent_direction_count`: the SIGNED mean cannot tell
    "diverse" from "two antipodal clusters", which averages to ~0 while every
    goal lies on a single line.
    """
    n = gram.shape[-1]
    if n < 2:
        return jnp.float32(0.0)
    off_diag_sum = gram.sum(axis=(-2, -1)) - jnp.trace(gram, axis1=-2, axis2=-1)
    return (off_diag_sum / (n * (n - 1))).mean()


def _agent_direction_count(gram: jnp.ndarray) -> jnp.ndarray:
    """Effective number of DISTINCT agent directions, in [1, min(N, d)].

    The headline "are the goals collapsing?" metric: computed per (timestep,
    env) across the agent axis and then averaged, so it answers "how many
    different directives does the manager issue at a single moment" — 1.0 means
    one shared team goal, N means N mutually orthogonal ones. Unlike the mean
    pairwise cosine it is sign-blind, so it also catches a collapse onto a
    single *line* (goals split into +u / -u clusters, whose signed cosines
    cancel to ~0 and read as healthy diversity).

    This is the participation ratio of the Gram's eigenvalues,
    ``(sum lambda)^2 / sum lambda^2``. The rows are unit-norm, so ``trace = N``
    and the numerator is exactly ``N^2`` — no eigendecomposition needed, only
    the Frobenius norm, which matters because this runs on a (T, E, N, N) stack.
    """
    n = gram.shape[-1]
    if n < 2:
        return jnp.float32(1.0)
    frob_sq = jnp.sum(gram**2, axis=(-2, -1))
    return (n**2 / (frob_sq + 1e-12)).mean()


def manager_cosine_metrics(
    s: jnp.ndarray,
    goal: jnp.ndarray,
    horizon: int,
    done_a: jnp.ndarray,
    active: jnp.ndarray,
    shift: int = 1,
    cos: jnp.ndarray = None,
    valid: jnp.ndarray = None,
) -> dict:
    """``d_cos`` and the two permutation nulls that make it interpretable.

    ``d_cos_mean`` alone says nothing about whether the manager's goals are
    useful, because the manager owns BOTH arguments of the cosine: it picks the
    measuring stick (`s`) and the target (`g`). Its absolute level is therefore
    a property of the geometry of `s` as much as of directive-following. What
    carries information is the gap against a null in which the goals are
    re-paired but their distribution is untouched.

    Two nulls, and they answer different questions (see the block comment above
    ``permute_agent_goals`` in ``manager.py``):

    * ``d_cos_null_agent`` — agent `i` scored against agent `i-shift`'s goal.
      ``d_cos_gap_agent = d_cos_mean - d_cos_null_agent`` is the value of the
      **assignment**.
    * ``d_cos_null_env`` — agent `i` scored against its own slot's goal from an
      unrelated env. ``d_cos_gap_env`` is the value of **state-conditioning**.

    READ ``d_cos_gap_env`` FIRST. If it is ~0 the goals are not functions of the
    state, and ``d_cos_gap_agent`` is uninterpretable however large it is — a
    manager emitting a fixed per-agent code scores well on the agent gap while
    doing nothing.

    ``goal_perm_cos`` is the assumption check that belongs beside both: it is the
    mean cosine between a goal and the one it was swapped with, so ~1 means the
    permutation changed nothing and a zero gap says only that the goals were
    already identical (cross-check against ``goal_direction_count``).

    Both nulls are forward-only on arrays the caller has already materialized.
    A singleton agent or env axis has no permutation null: its metrics are NaN
    (unavailable), while the real cosine and the other axis are still measured.
    Invalid shifts on axes with multiple entries continue to raise.

    Args:
        s: ``(T, n_envs, n_agents, goal_dim)`` latent states.
        goal: ``(T, n_envs, n_agents, goal_dim)`` unit goals, same shape.
        horizon: `c`.
        done_a: ``(T, n_envs, n_agents)`` terminal mask, pre-broadcast.
        active: ``(T, n_envs, n_agents)`` per-agent decision mask.
        shift: cyclic-permutation shift for both nulls.
        cos, valid: the REAL ``transition_cosine`` output, when the caller has
            already computed it (``manager_update`` has it as loss aux). Passing
            them avoids a third pass over a ``(T, E, N, D)`` tensor; omitting
            them recomputes, which is what the offline probe does since it never
            evaluates the loss.
    """
    from algorithms.feudal_mappo_jax.manager import (
        permute_agent_goals,
        permute_env_goals,
        transition_cosine,
    )

    if cos is None or valid is None:
        cos, valid = transition_cosine(
            s, goal, horizon, done=done_a, detach_states=True
        )
    # ONE mask for all three, so real and null are provably averaged over the
    # same entries. This is sound because `transition_cosine`'s `valid` is a
    # function of `states` and `done` only, never of `goals` — the single
    # invariant the whole null rests on, pinned by
    # `test_transition_cosine_valid_is_independent_of_goals`.
    mask = valid * active

    cos_mean = _masked_mean(cos, mask)
    null_agent = null_env = perm_cos_mean = jnp.asarray(jnp.nan, dtype=cos.dtype)

    if goal.shape[-2] > 1:
        goal_agent = permute_agent_goals(goal, shift)
        cos_null_agent, _ = transition_cosine(
            s, goal_agent, horizon, done=done_a, detach_states=True
        )
        null_agent = _masked_mean(cos_null_agent, mask)
        unit = goal / (jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-6)
        perm_cos = jnp.sum(unit * permute_agent_goals(unit, shift), axis=-1)
        perm_cos_mean = _masked_mean(perm_cos, mask)

    # env axis is 1 here: the manager path's goals are (T, n_envs, n_agents, D).
    if goal.shape[1] > 1:
        goal_env = permute_env_goals(goal, 1, shift)
        cos_null_env, _ = transition_cosine(
            s, goal_env, horizon, done=done_a, detach_states=True
        )
        null_env = _masked_mean(cos_null_env, mask)

    return {
        "d_cos_mean": cos_mean,
        # A CONSTANT cosine is annihilated by advantage centering, so a high
        # flat d_cos reads as success while the mechanism is dead. Always read
        # d_cos_var next to d_cos_mean, never the mean alone.
        "d_cos_var": _masked_mean((cos - cos_mean) ** 2, mask),
        "valid_fraction": valid.mean(),
        "d_cos_null_agent": null_agent,
        "d_cos_null_env": null_env,
        "d_cos_gap_agent": cos_mean - null_agent,
        "d_cos_gap_env": cos_mean - null_env,
        "goal_perm_cos": perm_cos_mean,
    }


def manager_update(
    train_state: FeudalTrainState,
    trajectory: Transition,
    last_manager_value: jnp.ndarray,
    config: MAPPOConfig,
    manager_module,
    n_agents: int,
    enc_grads=None,
) -> Tuple[FeudalTrainState, dict]:
    """FuN's transition policy gradient for the manager, plus V^M's regression.

    Deliberately NOT a variant of ``ppo_update``, for two structural reasons:

    * ``transition_cosine`` needs ``s_{t+c}``, i.e. the time axis **in order**.
      ``ppo_update`` flattens to (T*E, N, ...) and shuffles with a random
      permutation, which destroys exactly that.
    * The transition PG has **no importance ratio** — FuN reinforces the observed
      state *transition*, not the likelihood of a goal — so there is nothing to
      clip and extra epochs would be uncorrected off-policy. Hence a full-batch
      pass, and ``n_manager_epochs`` defaults to 1.

    The manager's own advantage comes from the extrinsic stream under
    ``manager_gamma``; the worker's intrinsic reward never enters here.

    Returns the updated train state and a metrics dict whose keys are all
    scalars (``run.py`` casts them with ``float()``).
    """
    from algorithms.feudal_mappo_jax.manager import transition_cosine

    dones = trajectory.done.astype(jnp.float32)
    horizon = config.goal_horizon

    # --- manager advantage on the extrinsic stream -------------------------
    m_adv, m_ret = compute_gae(
        trajectory.manager_reward,
        trajectory.manager_value,
        dones,
        last_manager_value,
        config.manager_gamma,
        config.gae_lambda,
    )
    m_explained_variance = 1.0 - jnp.var(
        m_ret - trajectory.manager_value, ddof=1
    ) / (jnp.var(m_ret, ddof=1) + 1e-8)

    # Same per-stream normalization the worker uses (axis 0 = time).
    m_adv = (m_adv - m_adv.mean(axis=0)) / (m_adv.std(axis=0, ddof=1) + 1e-8)
    # A scalar-reward manager has one advantage per env; broadcast it over the
    # agent axis so each agent's cosine is weighted by the team's advantage.
    manager_per_agent = trajectory.manager_reward.ndim == 3  # static
    if not manager_per_agent:
        m_adv = m_adv[..., None]
    m_adv = jax.lax.stop_gradient(m_adv)

    # `done` must carry the FULL leading shape of the cosine — a (T, E) mask
    # broadcasts into a wrong shape silently (see manager._check_done).
    done_a = jnp.broadcast_to(dones[..., None], trajectory.goal.shape[:-1])
    active = trajectory.active_mask

    # --- V^M regression -----------------------------------------------------
    # Its own loop, with its own epoch count. The targets `m_ret` are fixed, so
    # extra passes are ordinary supervised optimization — none of the
    # off-policy caution that pins `n_manager_epochs` at 1 applies here. Sharing
    # that count instead would give V^M ONE gradient step per update against the
    # worker critic's n_epochs*n_minibatches (48 at the defaults), which is
    # enough on its own to keep `manager_explained_variance` negative.
    def _critic_epoch(manager_critic_ts, _i):
        def manager_critic_loss_fn(params):
            values = manager_critic_ts.apply_fn(params, trajectory.global_state)
            sq = (values - m_ret) ** 2
            if manager_per_agent:
                value_loss = _masked_mean(sq, active)
            else:
                value_loss = jnp.mean(sq)
            return config.manager_val_coef * value_loss, value_loss

        (_, m_value_loss), mc_grads = jax.value_and_grad(
            manager_critic_loss_fn, has_aux=True
        )(manager_critic_ts.params)
        return manager_critic_ts.apply_gradients(grads=mc_grads), m_value_loss

    manager_critic_ts, critic_losses = jax.lax.scan(
        _critic_epoch,
        train_state.manager_critic_ts,
        jnp.arange(config.n_manager_critic_epochs),
    )

    # --- differentiable recompute of (goal, s) ------------------------------
    # The transition PG needs `g_t(theta)` to carry gradient, so the stored
    # rollout goals cannot be used; they are re-derived from the stored global
    # states under the *current* params. Whichever branch runs, it MUST agree
    # with what `trainer._env_step` actually emitted at rollout time — otherwise
    # the manager is optimized for a policy that never acted, silently. That
    # equality is what `test_goals_are_reproducible_from_stored_states` pins,
    # and it is the load-bearing check for the recurrent branch.
    recurrent = config.manager_core != "mlp"  # static
    n_envs = trajectory.global_state.shape[1]

    # `obs` is passed at every recompute site because latent="local" builds `s`
    # from each agent's own observation. It is ALREADY stored on `Transition`
    # (types.py), so this costs no extra rollout memory. The centralized branch
    # ignores it, so those runs are unaffected.
    def _recompute(apply_fn, params):
        if not recurrent:
            # Stateless core: a pure function of the global state, so it
            # vectorizes over (T, E) with no scan and no carry bookkeeping.
            _, goal, s = apply_fn(
                params, None, trajectory.global_state, trajectory.obs
            )
            return goal, s

        def _step(carry, xs):
            gs_t, obs_t, done_t = xs
            carry, goal_t, s_t = apply_fn(params, carry, gs_t, obs_t)
            # Zero the finished envs' sub-state pools AFTER emitting this step's
            # goal — the same order as `_env_step`, and the same semantics as
            # `pool_goals`' episode masking. `DilatedLSTMState.t` is a single
            # shared counter with no env axis, so it is NOT reset per env; it
            # simply keeps incrementing, exactly as it does in the rollout.
            carry = carry._replace(
                cell=tuple(
                    jnp.where(done_t[:, None, None], 0.0, p) for p in carry.cell
                )
            )
            return carry, (goal_t, s_t)

        # No `jax.checkpoint` here, deliberately — MEASURED, not assumed. The
        # BPTT was expected to need rematerialization (a naive estimate put the
        # residuals of a T=1024, E=32, r=10, H=256 scan at ~738 MB), but peak
        # memory is dominated by the scan's own (T, E, N, goal_dim) `goal`/`s`
        # outputs, not by the carry, and XLA already avoids storing the latter
        # naively. Measured over 5 grad calls at T=1024, E=32:
        #     H=256   remat 129.3 ms / 793.5 MiB   vs  no-remat 118.8 ms / 848.2 MiB
        #     H=1024  remat 315.4 ms / 2783.9 MiB  vs  no-remat 314.2 ms / 2789.3 MiB
        # i.e. ~9% slower for ~6% memory at the shipped width, and a wash on both
        # axes at 4x the width. Re-measure before adding it back if `n_steps` or
        # the carry size grows a lot.
        #
        # `initialize_carry` must be deterministic for this to line up with the
        # rollout's carry (it zeroes the pools and ignores the key);
        # `test_dilated_lstm_carry_is_deterministic` pins that contract.
        init_carry = manager_module.initialize_carry(
            jax.random.PRNGKey(0), (n_envs,)
        )
        _, (goal, s) = jax.lax.scan(
            _step, init_carry, (trajectory.global_state, trajectory.obs, dones)
        )
        return goal, s

    def _epoch_step(carry, _epoch_idx):
        manager_ts = carry

        # --- transition policy gradient ---
        def manager_loss_fn(params):
            goal, s = _recompute(manager_ts.apply_fn, params)
            cos, valid = transition_cosine(
                s, goal, horizon, done=done_a, detach_states=True
            )
            mask = valid * active
            # Maximize A^M * d_cos(s_{t+c} - s_t, g_t)  =>  minimize its negation.
            pg_loss = -_masked_mean(cos * m_adv, mask)
            return pg_loss, (cos, valid, mask, goal, s)

        (pg_loss, (cos, valid, mask, goal, s)), m_grads = jax.value_and_grad(
            manager_loss_fn, has_aux=True
        )(manager_ts.params)

        # --- the worker's share of the SHARED encoder (worker_encoder="shared") ---
        # Two losses write to `f_enc`, so their gradients are summed at the one
        # parameter point both were evaluated at and take a single Adam step.
        #
        # ⚠ The encoder term is clipped SEPARATELY first, and that is not a detail.
        # The manager's optimizer is `chain(clip_by_global_norm(grad_clip), adam)`
        # and that clip is GLOBAL over the whole manager tree: summing an unclipped
        # encoder gradient in would make the clip fire more often and, when it
        # fires, rescale the manager's own transition PG by a factor set by how
        # hard the worker happened to push this update. The arm would quietly
        # become "manager PG, attenuated by the worker" while `manager_pg_loss`
        # kept logging a healthy number — the failure mode this file catalogues
        # twice already for `worker_goal_column_ratio`. Clipping the addition to
        # the same budget on its own leaves the PG's own norm untouched whenever
        # the PG alone is under the clip, which is the common case.
        #
        # A second `apply_gradients` instead of a sum would be worse still: optax's
        # Adam MOVES a leaf with a zero gradient, because `mu` is nonzero from the
        # previous update and `count` increments for the whole tree — so every
        # non-encoder manager parameter would drift.
        if enc_grads is not None:
            enc_scaled = _clip_tree(enc_grads, config.grad_clip)
            enc_norm = _encoder_grad_norm(enc_scaled)
            pg_enc_norm = _encoder_grad_norm(m_grads)
            enc_cos = _encoder_grad_cosine(m_grads, enc_scaled)
            m_grads = jax.tree.map(jnp.add, m_grads, enc_scaled)
        manager_ts = manager_ts.apply_gradients(grads=m_grads)

        # --- collapse diagnostics (see the helpers above) ---
        # `goal`/`s` here are concrete forward values under the PRE-update
        # params — the right ones — and this is outside the differentiated
        # region, so the two permutation nulls cost no backward pass.
        goal_gram = _agent_gram(goal)
        metrics = {
            "manager_pg_loss": pg_loss,
            "manager_adv_std": m_adv.std(),
            # d_cos_mean / d_cos_var / valid_fraction plus the agent- and
            # env-axis nulls that make the level interpretable at all.
            **manager_cosine_metrics(
                s, goal, horizon, done_a, active, config.goal_permute_shift,
                cos=cos, valid=valid,
            ),
            "goal_pairwise_cos": _mean_pairwise_cosine(goal_gram),
            # Sign-blind companions to the signed mean: goals collapsed onto one
            # LINE give cos ~ 0 but |cos| ~ 1 and a direction count ~ 1.
            "goal_pairwise_cos_abs": _mean_pairwise_cosine(jnp.abs(goal_gram)),
            "goal_direction_count": _agent_direction_count(goal_gram),
            "state_pairwise_cos": _mean_pairwise_cosine(_agent_gram(s)),
            "state_latent_erank": _effective_rank(s),
        }
        if enc_grads is not None:
            # Who shapes `f_enc`. Read the COSINE first: norms alone are
            # misleading under Adam, whose step is scale-invariant in the limit,
            # so "the worker's norm is 10x" does not by itself mean the worker
            # wins. A persistently negative cosine means the two objectives are
            # pulling the shared representation apart, which is the thing this arm
            # can fail at while every other series stays healthy.
            metrics["manager_encoder_grad_norm"] = pg_enc_norm
            metrics["worker_encoder_grad_norm_clipped"] = enc_norm
            metrics["worker_manager_encoder_grad_cos"] = enc_cos
        return manager_ts, metrics

    manager_ts, epoch_metrics = jax.lax.scan(
        _epoch_step,
        train_state.manager_ts,
        jnp.arange(config.n_manager_epochs),
    )

    metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)
    metrics["manager_value_loss"] = critic_losses.mean()
    # Pre-update EV, like the worker's: measures the critic that PRODUCED the
    # advantages this update used, not the one left behind after fitting.
    metrics["manager_explained_variance"] = m_explained_variance

    return (
        train_state._replace(
            manager_ts=manager_ts, manager_critic_ts=manager_critic_ts
        ),
        metrics,
    )
