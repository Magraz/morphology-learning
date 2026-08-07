"""Jitted collect/update/eval functions for MAPPO over a functional MJX env.

The env contract is the gymnax-style API of ``MultiBoxPushMJX``:
``reset(key) -> (obs, state)`` and ``step(state, actions) -> (obs, state,
reward, terminated, truncated, info)`` with obs ``(n_agents, obs_dim)``,
continuous actions ``(n_agents, action_dim)`` in [-1, 1], a scalar team reward,
and **no auto-reset** — this module supplies the resets.

The structure deliberately mirrors ``mappo_vanilla``'s trainer components:
- ``collect_fn`` == ``RolloutCollector.collect``: resets all envs at the top of
  every rollout (vanilla re-seeds its vec env each collect), scans ``n_steps``,
  restarts any env that finishes mid-rollout (vanilla's gymnasium vec env
  auto-resets), and bootstraps the final value from the last observation.
- ``update_fn`` == ``MAPPOAgent.update`` (see ``mappo.ppo_update``).
- ``eval_fn`` == ``PolicyEvaluator.evaluate``: deterministic parallel episodes,
  per-episode returns accumulated until each episode first finishes.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from algorithms.feudal_mappo_jax.types import Bootstrap, MAPPOConfig, Transition
from algorithms.feudal_mappo_jax.network import sample_action
from algorithms.feudal_mappo_jax.manager import (
    goal_ring_pool,
    goal_ring_reset,
    goal_ring_write,
    worker_intrinsic_reward,
)
from algorithms.feudal_mappo_jax.mappo import (
    FeudalTrainState,
    build_manager,
    create_train_state,
    manager_update,
    ppo_update,
)
from algorithms.feudal_mappo_jax.worker import bind_goal

# Back-compat alias for `run.py`, which imports the flat-stack name.
ActorCriticTrainState = FeudalTrainState


class RunnerState(NamedTuple):
    """Carries all mutable state between update iterations.

    No env state: like vanilla, every rollout starts from freshly reset envs.
    """

    train_state: ActorCriticTrainState
    rng: jax.Array


def make_train(config: MAPPOConfig, env):
    """Build jitted train functions for a functional MJX env.

    Returns:
        init_fn(rng) -> RunnerState
        collect_fn(runner_state) -> (RunnerState, trajectory, last_value, rollout_stats)
        update_fn(runner_state, trajectory, last_value) -> (RunnerState, losses)
        eval_fn(train_state, rng) -> mean episode return over n_eval_episodes
        num_updates: total number of update iterations to run
    """
    n_agents = env.n_agents
    obs_dim = env.observation_dim
    action_dim = env.action_dim
    # The base MJX suite is continuous force control; the hierarchical macro env
    # (SyncMacroMJX) selects among discrete skills. The env declares which.
    discrete = getattr(env, "discrete", False)
    # Decision-aligned windowed difference reward (async only): the env's per-step
    # reward is a placeholder (global-window D); the true per-agent D is computed
    # post-collect from logged snapshots + the *next* window's proposals.
    from environments.mjx_suite.macro_wrapper import ALIGNED_WINDOWED_DIFFERENCE_REWARDS
    aligned = getattr(env, "reward_mode", "") == ALIGNED_WINDOWED_DIFFERENCE_REWARDS

    if not config.parameter_sharing:
        raise NotImplementedError(
            "feudal_mappo_jax implements the shared-actor path only "
            "(parameter_sharing=true); use mappo_vanilla for independent actors"
        )

    num_updates = int(config.n_total_steps) // (config.n_steps * config.n_envs)

    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    # The manager module, rebuilt from config (frozen dataclass, so this instance
    # is interchangeable with the one inside create_train_state).
    manager = build_manager(config, n_agents)
    goal_dim = config.goal_dim
    horizon = config.goal_horizon

    def _actor_forward(train_state, obs, pooled_goal, rng, deterministic):
        """Shared goal-conditioned worker over (batch, n_agents, obs_dim).

        `pooled_goal` is flattened agent-major with EXACTLY the same reshape as
        obs, so row k = b_i*n_agents + a pairs agent a's obs with agent a's goal.
        Getting these two out of step would silently train every agent on a
        neighbour's directive.
        """
        b = obs.shape[0]
        actions_flat, log_probs_flat = sample_action(
            rng,
            bind_goal(
                train_state.actor_ts.apply_fn,
                pooled_goal.reshape(b * n_agents, goal_dim),
            ),
            train_state.actor_ts.params,
            obs.reshape(b * n_agents, obs_dim),
            discrete,
            deterministic=deterministic,
        )
        # Discrete actions are integer skill indices (no trailing action_dim
        # axis); continuous actions are (action_dim,) force vectors.
        action_shape = (b, n_agents) if discrete else (b, n_agents, action_dim)
        return (
            actions_flat.reshape(action_shape),
            log_probs_flat.reshape(b, n_agents),
        )

    def _values(train_state, global_state):
        return train_state.critic_ts.apply_fn(
            train_state.critic_ts.params, global_state
        )

    def _manager_values(train_state, global_state):
        return train_state.manager_critic_ts.apply_fn(
            train_state.manager_critic_ts.params, global_state
        )

    def _manager_forward(train_state, m_carry, global_state):
        return train_state.manager_ts.apply_fn(
            train_state.manager_ts.params, m_carry, global_state
        )

    # ------------------------------------------------------------------ init

    # The WORKER critic is always per-agent: the intrinsic reward is per-agent, so
    # the reward the worker learns from is (n_envs, n_agents) regardless of the
    # env's reward mode. (This is why the intrinsic_coef=0 arm is still not
    # numerically identical to mappo_jax — the flat baseline is mappo_jax itself.)
    n_critic_outputs = n_agents
    # V^M regresses the EXTRINSIC return. Under a team scalar reward that is one
    # target, so N heads would be N copies differing only by init noise; give it
    # per-agent heads only when the env reward is genuinely per-agent.
    n_manager_outputs = n_agents if config.per_agent_rewards else 1

    @jax.jit
    def init_fn(rng: jax.Array) -> RunnerState:
        rng, init_rng = jax.random.split(rng)
        train_state = create_train_state(
            init_rng, config, obs_dim, obs_dim * n_agents, action_dim, discrete,
            n_agents=n_agents,
            n_critic_outputs=n_critic_outputs,
            n_manager_outputs=n_manager_outputs,
        )
        return RunnerState(train_state=train_state, rng=rng)

    # ------------------------------------------------------------------ collect

    def _env_step(carry, t):
        train_state, env_state, obs, rng, m_carry, goal_hist = carry
        rng, action_rng, reset_rng = jax.random.split(rng, 3)

        global_state = obs.reshape(config.n_envs, -1)

        # --- Manager: one read of the joint state -> one directive per agent ---
        m_carry, goal, state_latent = _manager_forward(
            train_state, m_carry, global_state
        )
        goal_hist = goal_ring_write(goal_hist, goal, t)
        pooled_goal = goal_ring_pool(goal_hist)

        values = _values(train_state, global_state)
        manager_values = _manager_values(train_state, global_state)
        actions, log_probs = _actor_forward(
            train_state, obs, pooled_goal, action_rng, deterministic=False
        )

        next_obs, next_env_state, reward, terminated, truncated, info = v_step(
            env_state, actions
        )
        # `reward` is what the learner optimizes (team scalar, or per-agent
        # difference rewards); `task_reward` is always the team scalar, so logged
        # returns stay comparable across reward modes.
        team_reward = info["task_reward"]
        done = jnp.logical_or(terminated, truncated)

        # Truncation bootstrap. A time-limit `truncated` (vs a true `terminated`)
        # does not end the MDP, so its return should carry `gamma * V(s_next)`
        # forward instead of being cut to 0. `done` (used below and by GAE) still
        # fires on truncation so advantages don't bleed across the boundary AND
        # the reset restarts the env — but that reset overwrites `next_obs`, so we
        # must value the *real* successor here, before the reset, and fold the
        # bootstrap into the reward (SB3-style). GAE's own `not_done` bootstrap
        # term is then correctly 0 at this step, avoiding a double count.
        # Both learners need this, against their OWN value function and discount:
        # the worker critic predicts the intrinsic-augmented return, V^M the
        # extrinsic one, so bootstrapping the manager stream off the worker's
        # critic would be simply the wrong number.
        next_gs = next_obs.reshape(config.n_envs, -1)
        trunc_f = truncated.astype(jnp.float32)
        next_value = _values(train_state, next_gs)  # (E, N), always per-agent
        next_m_value = _manager_values(train_state, next_gs)

        # The manager's stream is the RAW extrinsic reward: scalar under a dense
        # env, per-agent under difference rewards — which is exactly the condition
        # `n_manager_outputs` keys off, so V^M's head width always matches the
        # target it regresses. It never sees the worker's intrinsic term.
        m_trunc_f = trunc_f[:, None] if next_m_value.ndim > 1 else trunc_f
        manager_reward = reward + config.manager_gamma * m_trunc_f * next_m_value

        # The worker's stream is per-agent because the intrinsic term (added
        # post-scan) is. A scalar env reward must be broadcast EXPLICITLY:
        # (E,) + (E,N) right-aligns E against N and RAISES at E=32/N=16.
        if reward.ndim == 1:
            reward = jnp.broadcast_to(reward[:, None], (config.n_envs, n_agents))
        reward = reward + config.gamma * trunc_f[:, None] * next_value

        # The MJX env does not auto-reset; restart finished envs so the rollout
        # continues with fresh episodes (vanilla's gymnasium vec env does this).
        # lax.cond skips the reset work on the common no-done step.
        def _restart_done(operand):
            cur_obs, cur_state = operand
            reset_obs, reset_state = v_reset(
                jax.random.split(reset_rng, config.n_envs)
            )

            def _select(r, c):
                d = done.reshape((-1,) + (1,) * (c.ndim - 1))
                return jnp.where(d, r, c)

            return _select(reset_obs, cur_obs), jax.tree.map(
                _select, reset_state, cur_state
            )

        next_obs, next_env_state = jax.lax.cond(
            done.any(), _restart_done, lambda operand: operand,
            (next_obs, next_env_state),
        )

        # The manager's memory is per-episode too. Clearing the ring AFTER this
        # step's `pooled_goal` was consumed reproduces `pool_goals`'s semantics
        # exactly: `_same_episode` counts dones in [src, t), so g_src still
        # contributes to w_t on the step where done[t] fires, and not after.
        goal_hist = goal_ring_reset(goal_hist, done)
        if m_carry is not None:
            # Dilated-LSTM pools are (E, radius, features); zero the finished
            # envs' rows. `t` is a single shared scalar counter with no env axis,
            # so a mid-rollout reset leaves that env at an arbitrary dilation
            # phase — harmless (the phase is arbitrary anyway) but it MUST be
            # reproduced identically by the manager update's rescan.
            m_carry = m_carry._replace(
                cell=tuple(
                    jnp.where(done[:, None, None], 0.0, p) for p in m_carry.cell
                )
            )

        # Per-agent activity mask. SyncMacroMJX's staggered-starts mode emits
        # info["active"] (0 for agents offline this window); every other env omits
        # it, so default to all-ones — which leaves the PPO loss byte-identical.
        active_mask = info.get("active", jnp.ones((config.n_envs, n_agents)))

        transition = Transition(
            obs=obs,
            global_state=global_state,
            action=actions,
            reward=reward,
            done=done,
            log_prob=log_probs,
            value=values,
            team_reward=team_reward,
            active_mask=active_mask,
            goal=goal,
            pooled_goal=pooled_goal,
            state_latent=state_latent,
            manager_value=manager_values,
            manager_reward=manager_reward,
        )
        carry = (train_state, next_env_state, next_obs, rng, m_carry, goal_hist)
        if aligned:
            # Extra per-window data for the post-collect decision-aligned D pass:
            # the compact PRE-step state (to fork from), plus the pieces needed to
            # re-apply the truncation bootstrap to the overwritten reward. `reward`
            # above is a placeholder (global-window D); it is discarded post-collect.
            aux = {
                "snapshot": env.snapshot(env_state),
                "truncated": truncated,
                "next_value": next_value,  # pre-reset V(s_next), per-agent head
            }
            return carry, (transition, aux)
        return carry, transition

    def _apply_aligned_rewards(trajectory, aux):
        """Overwrite the placeholder reward with the decision-aligned windowed D.

        Post-collect (the aligned D needs each window's *next* proposals, unknown
        during the scan): for every (window, env) fork the logged compact snapshot
        and run ``env.decision_aligned_D`` with ``proposed = action[w]`` and
        ``proposed_next = action[w+1]`` (the last window reuses its own action — a
        one-window boundary approximation; its overshoot is only reached if the
        episode ran past the rollout). The truncation bootstrap is re-applied to
        the new reward exactly as ``_env_step`` did to the placeholder.
        """
        proposed = trajectory.action  # (T, n_envs, A)
        proposed_next = jnp.concatenate([proposed[1:], proposed[-1:]], axis=0)

        def _one(snap_e, p, pn):
            mstate = env.state_from_snapshot(snap_e)
            return env.decision_aligned_D(mstate, p, pn)  # (A,)

        # vmap over time (outer) and env (inner) — snapshots are (T, n_envs, ...).
        aligned_D = jax.vmap(jax.vmap(_one))(
            aux["snapshot"], proposed, proposed_next
        )  # (T, n_envs, A)
        trunc_f = aux["truncated"].astype(jnp.float32)[..., None]  # (T, n_envs, 1)
        reward = aligned_D + config.gamma * trunc_f * aux["next_value"]
        return trajectory._replace(reward=reward)

    def _apply_intrinsic_reward(trajectory):
        """Fold FuN's intrinsic reward into the worker's stream, post-scan.

        Follows the same post-collect-rewrite pattern as
        ``_apply_aligned_rewards``: r^I_t looks *backwards* over `c` steps, so it
        is a whole-trajectory quantity that cannot be produced inside the scan.
        ``worker_intrinsic_reward`` detaches both arguments — this is a reward,
        i.e. data, and leaving it attached would also backprop the worker's
        objective into the manager, which FuN explicitly rules out.
        """
        # `done` is (T, E); the helpers require the FULL leading shape of the
        # cosine, (T, E, N) — a partially-shaped mask used to broadcast into
        # nonsense silently, which is why `_check_done` now rejects it.
        done_a = jnp.broadcast_to(
            trajectory.done[..., None].astype(jnp.float32),
            trajectory.state_latent.shape[:-1],
        )
        r_int = worker_intrinsic_reward(
            trajectory.state_latent, trajectory.goal, horizon, done=done_a
        )  # (T, E, N)
        return trajectory._replace(
            reward=trajectory.reward + config.intrinsic_coef * r_int
        ), r_int

    @jax.jit
    def collect_fn(runner_state: RunnerState):
        train_state, rng = runner_state
        rng, reset_rng, carry_rng = jax.random.split(rng, 3)

        # Fresh episodes every rollout, matching vanilla's per-collect reset
        obs, env_state = v_reset(jax.random.split(reset_rng, config.n_envs))
        # The manager's memory resets with them. `m_carry` is None for the mlp
        # core — a valid empty pytree, so the carry slot costs nothing there.
        m_carry = manager.initialize_carry(carry_rng, (config.n_envs,))
        goal_hist = jnp.zeros((horizon, config.n_envs, n_agents, goal_dim))

        (train_state, _, last_obs, rng, _, _), scan_out = jax.lax.scan(
            _env_step,
            (train_state, env_state, obs, rng, m_carry, goal_hist),
            # The scan's xs is the step index, which the goal ring needs to pick
            # its slot (and which the flat stack had no use for).
            jnp.arange(config.n_steps),
        )
        if aligned:
            trajectory, aux = scan_out
            trajectory = _apply_aligned_rewards(trajectory, aux)
        else:
            trajectory = scan_out

        # Static python guard: alpha=0 skips the work entirely and leaves the
        # worker's stream exactly as the scan built it.
        mean_intrinsic = jnp.float32(0.0)
        abs_intrinsic = jnp.float32(0.0)
        if config.intrinsic_coef != 0.0:
            trajectory, r_int = _apply_intrinsic_reward(trajectory)
            mean_intrinsic = r_int.mean()
            # The SIGNED mean cancels (cosines are symmetric about 0) and can sit
            # near zero while the per-step term is large. Scale alpha against
            # this magnitude, not the mean.
            abs_intrinsic = jnp.abs(r_int).mean()

        # Bootstrap values for GAE, one per learner (different reward streams,
        # different discounts). Done masking happens inside compute_gae.
        last_gs = last_obs.reshape(config.n_envs, -1)
        last_value = Bootstrap(
            worker=_values(train_state, last_gs),
            manager=_manager_values(train_state, last_gs),
        )

        rollout_stats = {
            # Team reward, not the learner's signal — otherwise this would report
            # mean D and be incomparable to the dense baseline.
            "mean_reward": trajectory.team_reward.mean(),
            "episode_count": trajectory.done.sum(),
            # Raw r^I (NOT scaled by alpha), so the scale probe reads the same
            # number at any alpha — r^I depends only on (s, g).
            "intrinsic_reward": mean_intrinsic,
            "intrinsic_reward_abs": abs_intrinsic,
        }
        return (
            RunnerState(train_state=train_state, rng=rng),
            trajectory,
            last_value,
            rollout_stats,
        )

    # ------------------------------------------------------------------ update

    @jax.jit
    def update_fn(runner_state: RunnerState, trajectory, last_value: Bootstrap):
        train_state, rng = runner_state
        rng, update_rng = jax.random.split(rng)
        # Worker (PPO) and manager (transition PG) touch disjoint parameters and
        # both read the frozen trajectory, so the order is immaterial. The
        # manager runs second so its diagnostics describe the goals the worker
        # was actually just updated against.
        train_state, losses = ppo_update(
            train_state, update_rng, trajectory, last_value.worker, config, discrete
        )
        train_state, m_losses = manager_update(
            train_state,
            trajectory,
            last_value.manager,
            config,
            manager,
            n_agents,
        )
        return RunnerState(train_state=train_state, rng=rng), {**losses, **m_losses}

    # ------------------------------------------------------------------ eval

    @jax.jit
    def eval_fn(train_state: ActorCriticTrainState, rng: jax.Array):
        """Deterministic parallel-episode evaluation (PolicyEvaluator parity)."""
        keys = jax.random.split(rng, config.n_eval_episodes)
        obs, env_state = jax.vmap(env.reset)(keys)
        finished = jnp.zeros(config.n_eval_episodes, dtype=bool)
        episode_rewards = jnp.zeros(config.n_eval_episodes)
        # Eval must run the full hierarchy: the worker is goal-conditioned, so
        # without the manager it would be evaluated on a goal it never sees.
        m_carry = manager.initialize_carry(rng, (config.n_eval_episodes,))
        goal_hist = jnp.zeros(
            (horizon, config.n_eval_episodes, n_agents, goal_dim)
        )

        def _eval_step(carry, t):
            obs, env_state, finished, episode_rewards, m_carry, goal_hist = carry
            gs = obs.reshape(config.n_eval_episodes, -1)
            m_carry, goal, _ = _manager_forward(train_state, m_carry, gs)
            goal_hist = goal_ring_write(goal_hist, goal, t)
            actions, _ = _actor_forward(
                train_state,
                obs,
                goal_ring_pool(goal_hist),
                jax.random.PRNGKey(0),
                deterministic=True,
            )
            next_obs, next_env_state, _, terminated, truncated, info = jax.vmap(
                env.step
            )(env_state, actions)
            # Always score eval on the team reward: under difference rewards the
            # env's `reward` is per-agent, and a policy must still be judged by
            # what the team achieved.
            reward = info["task_reward"]
            episode_rewards = episode_rewards + jnp.where(finished, 0.0, reward)
            finished = finished | terminated | truncated
            return (
                next_obs,
                next_env_state,
                finished,
                episode_rewards,
                m_carry,
                goal_hist,
            ), None

        # Fixed-length scan (jit needs static bounds); finished episodes keep
        # stepping but their rewards are masked out, like vanilla's `finished`.
        # No done-reset of the ring here: eval never restarts an episode, it just
        # masks finished ones out.
        (_, _, _, episode_rewards, _, _), _ = jax.lax.scan(
            _eval_step,
            (obs, env_state, finished, episode_rewards, m_carry, goal_hist),
            jnp.arange(env.max_steps),
        )
        return episode_rewards.mean()

    return init_fn, collect_fn, update_fn, eval_fn, num_updates
