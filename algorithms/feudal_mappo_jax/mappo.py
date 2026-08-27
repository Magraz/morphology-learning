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
from algorithms.feudal_mappo_jax.manager import FeudalManager
from algorithms.feudal_mappo_jax.worker import bind_goal, init_worker


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
    worker, actor_params = init_worker(
        rng_actor,
        obs_dim=obs_dim,
        goal_dim=config.goal_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        discrete=discrete,
        goal_embed_dim=config.goal_embed_dim,
    )
    critic = MAPPOCritic(hidden_dim=2 * config.hidden_dim, n_outputs=n_critic_outputs)
    critic_params = critic.init(rng_critic, jnp.zeros(global_state_dim))

    manager = build_manager(config, n_agents)
    manager_carry = manager.initialize_carry(rng_manager, ())
    manager_params = manager.init(
        rng_manager, manager_carry, jnp.zeros(global_state_dim)
    )
    manager_critic = MAPPOCritic(
        hidden_dim=2 * config.hidden_dim, n_outputs=n_manager_outputs
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
            hidden_dim=2 * config.hidden_dim, n_outputs=n_critic_outputs
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
    if progress is None:
        progress = jnp.float32(0.0)
    alpha_t = _annealed_alpha(config, progress)
    n_steps, n_envs, n_agents = trajectory.obs.shape[:3]
    obs_dim = trajectory.obs.shape[3]
    goal_dim = trajectory.pooled_goal.shape[-1]
    # Static: (n_steps, n_envs, n_agents) rewards => per-agent credit path.
    per_agent = trajectory.reward.ndim == 3

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
        # Both terms are unit-std here, so alpha_t is exactly the mixing ratio of
        # the two gradient directions.
        adv = adv + alpha_t * adv_int
        int_metrics = {
            "alpha_current": alpha_t,
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

    def _epoch_step(carry, _epoch_idx):
        train_state, rng = carry
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, total_ts)

        def _minibatch_step(carry, mb_idx):
            actor_ts, critic_ts, int_critic_ts = carry
            start = mb_idx * ts_minibatch_size
            mb_ids = jax.lax.dynamic_slice(perm, (start,), (ts_minibatch_size,))

            n_flat = ts_minibatch_size * n_agents
            mb_obs = obs_ts[mb_ids].reshape(n_flat, obs_dim)
            mb_goal = pg_ts[mb_ids].reshape(n_flat, goal_dim)
            mb_actions = act_ts[mb_ids].reshape(n_flat, *act_ts.shape[2:])
            mb_old_lp = lp_ts[mb_ids].reshape(n_flat)
            # Agent-major flattening matches mb_obs; (mb, n_agents) form for the
            # per-agent critic. All-ones => the masked means are exact plain means.
            mb_active_pa = active_ts[mb_ids]
            mb_active = mb_active_pa.reshape(n_flat)
            if per_agent:
                # Each agent carries its own advantage. `.reshape(-1)` is
                # agent-major within a timestep, matching mb_obs' flattening.
                mb_adv = adv_ts[mb_ids].reshape(n_flat)
            else:
                # env-level advantage broadcast to each agent (identical per agent)
                mb_adv = jnp.repeat(adv_ts[mb_ids], n_agents)
            mb_gs = gs_ts[mb_ids]
            mb_returns = ret_ts[mb_ids]

            # --- Actor loss ---
            def actor_loss_fn(actor_params):
                # `bind_goal` freezes the stored conditioning into the worker's
                # apply_fn, restoring the flat (params, obs) signature
                # `evaluate_action` expects — no forked evaluation path. Using
                # the STORED pooled goal (not a recomputed one) is what keeps the
                # importance ratio valid.
                log_probs, entropy = evaluate_action(
                    bind_goal(actor_ts.apply_fn, mb_goal),
                    actor_params,
                    mb_obs,
                    mb_actions,
                    discrete,
                )
                ratio = jnp.exp(log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = jnp.clip(
                    ratio, 1.0 - config.eps_clip, 1.0 + config.eps_clip
                ) * mb_adv
                # Mask offline agents out of the policy gradient (their proposed
                # skill was never executed). All-ones => plain mean.
                denom = jnp.maximum(mb_active.sum(), 1.0)
                policy_loss = -(jnp.minimum(surr1, surr2) * mb_active).sum() / denom
                entropy_loss = -(entropy * mb_active).sum() / denom
                total = policy_loss + config.ent_coef * entropy_loss
                return total, (policy_loss, entropy_loss)

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
    mean_losses["worker_goal_column_ratio"] = _goal_column_ratio(
        train_state.actor_ts.params, obs_dim
    )
    if not use_intrinsic:
        # The per-minibatch placeholder carries no information at alpha=0; drop
        # it so the stats keys stay exactly what they were.
        mean_losses.pop("intrinsic_value_loss", None)
    mean_losses.update(int_metrics)

    return train_state, mean_losses


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
    """
    kernel = actor_params["params"]["MAPPOActor_0"]["Dense_0"]["kernel"]
    obs_block, goal_block = kernel[:obs_dim], kernel[obs_dim:]
    obs_rms = jnp.sqrt(jnp.mean(obs_block**2))
    goal_rms = jnp.sqrt(jnp.mean(goal_block**2))
    return goal_rms / (obs_rms + 1e-8)


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


def manager_update(
    train_state: FeudalTrainState,
    trajectory: Transition,
    last_manager_value: jnp.ndarray,
    config: MAPPOConfig,
    manager_module,
    n_agents: int,
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

    def _recompute(apply_fn, params):
        if not recurrent:
            # Stateless core: a pure function of the global state, so it
            # vectorizes over (T, E) with no scan and no carry bookkeeping.
            _, goal, s = apply_fn(params, None, trajectory.global_state)
            return goal, s

        def _step(carry, xs):
            gs_t, done_t = xs
            carry, goal_t, s_t = apply_fn(params, carry, gs_t)
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
            _step, init_carry, (trajectory.global_state, dones)
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
        manager_ts = manager_ts.apply_gradients(grads=m_grads)

        # --- collapse diagnostics (see the helpers above) ---
        cos_mean = _masked_mean(cos, mask)
        cos_var = _masked_mean((cos - cos_mean) ** 2, mask)
        goal_gram = _agent_gram(goal)
        metrics = {
            "manager_pg_loss": pg_loss,
            "manager_adv_std": m_adv.std(),
            "d_cos_mean": cos_mean,
            # A CONSTANT cosine is annihilated by advantage centering, so a high
            # flat d_cos reads as success while the mechanism is dead. Always
            # read d_cos_var next to d_cos_mean, never the mean alone.
            "d_cos_var": cos_var,
            "valid_fraction": valid.mean(),
            "goal_pairwise_cos": _mean_pairwise_cosine(goal_gram),
            # Sign-blind companions to the signed mean: goals collapsed onto one
            # LINE give cos ~ 0 but |cos| ~ 1 and a direction count ~ 1.
            "goal_pairwise_cos_abs": _mean_pairwise_cosine(jnp.abs(goal_gram)),
            "goal_direction_count": _agent_direction_count(goal_gram),
            "state_pairwise_cos": _mean_pairwise_cosine(_agent_gram(s)),
            "state_latent_erank": _effective_rank(s),
        }
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
