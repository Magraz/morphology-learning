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

from algorithms.mappo_jax.types import MAPPOConfig, Transition
from algorithms.mappo_jax.network import sample_action
from algorithms.mappo_jax.mappo import (
    ActorCriticTrainState,
    create_train_state,
    ppo_update,
)


class RunnerState(NamedTuple):
    """Carries all mutable state between update iterations.

    No env state: like vanilla, every rollout starts from freshly reset envs.
    """

    train_state: ActorCriticTrainState
    rng: jax.Array


def global_state_dim(env) -> int:
    """Width of the centralized critic's input for `env`.

    An env may publish a real global state (SMAX's world state — absolute unit features,
    and much narrower than N egocentric views); otherwise the global state is the
    concatenation of the per-agent observations, which is what the MJX envs use.

    Shared by `make_train` and `run.py`'s checkpoint reload so the critic built at resume
    cannot disagree with the one that was trained — `flax.serialization.from_bytes`
    needs an exactly-shaped target tree, so a mismatch here is a load failure.
    """
    if hasattr(env, "global_state"):
        return int(env.global_state_dim)
    return int(env.observation_dim) * int(env.n_agents)


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
            "mappo_jax implements the shared-actor path only "
            "(parameter_sharing=true); use mappo_vanilla for independent actors"
        )

    num_updates = int(config.n_total_steps) // (config.n_steps * config.n_envs)

    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    # --- Optional env hooks. Both are static (presence is decided at trace time), so an
    # env that has neither takes byte-identical code paths to before they existed.

    # Legal-action masking (SMAX). Without it a categorical actor spends probability on
    # illegal actions and a deterministic argmax can select one outright.
    use_action_mask = hasattr(env, "avail_actions")
    if use_action_mask:
        _v_avail = jax.vmap(env.avail_actions)
        # Placeholder stored in Transition when masking is off — see types.Transition.
        _mask_placeholder = None
    else:
        _v_avail = None
        _mask_placeholder = jnp.zeros(())

    def _avail(env_state):
        return _v_avail(env_state) if use_action_mask else None

    # Centralized-critic input. SMAX ships a real world state (absolute unit features);
    # the MJX envs do not, and there the global state is the concatenation of the
    # per-agent observations — which is what this used to do unconditionally.
    if hasattr(env, "global_state"):
        _v_gs = jax.vmap(env.global_state)

        def _global_state(obs, env_state):
            return _v_gs(env_state)

    else:

        def _global_state(obs, env_state):
            return obs.reshape(obs.shape[0], -1)

    gs_dim = global_state_dim(env)

    def _actor_forward(train_state, obs, rng, deterministic, action_mask=None):
        """Shared actor over (batch, n_agents, obs_dim) in one fused pass."""
        b = obs.shape[0]
        # The mask flattens agent-major exactly like the obs, so row i of the flattened
        # batch keeps its own agent's legal actions.
        mask_flat = (
            action_mask.reshape(b * n_agents, action_dim)
            if action_mask is not None
            else None
        )
        actions_flat, log_probs_flat = sample_action(
            rng,
            train_state.actor_ts.apply_fn,
            train_state.actor_ts.params,
            obs.reshape(b * n_agents, obs_dim),
            discrete,
            deterministic=deterministic,
            action_mask=mask_flat,
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

    # ------------------------------------------------------------------ init

    # Per-agent rewards need a value per agent to run GAE against.
    n_critic_outputs = n_agents if config.per_agent_rewards else 1

    @jax.jit
    def init_fn(rng: jax.Array) -> RunnerState:
        rng, init_rng = jax.random.split(rng)
        train_state = create_train_state(
            init_rng, config, obs_dim, gs_dim, action_dim, discrete,
            n_critic_outputs=n_critic_outputs,
        )
        return RunnerState(train_state=train_state, rng=rng)

    # ------------------------------------------------------------------ collect

    def _env_step(carry, _):
        train_state, env_state, obs, rng = carry
        rng, action_rng, reset_rng = jax.random.split(rng, 3)

        global_state = _global_state(obs, env_state)
        values = _values(train_state, global_state)
        # Fetched from the PRE-step state: these are the actions legal for the state the
        # policy is acting from, and the same array is stored in the transition so the
        # update re-evaluates the identical masked distribution.
        action_mask = _avail(env_state)
        actions, log_probs = _actor_forward(
            train_state, obs, action_rng, deterministic=False, action_mask=action_mask
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
        trunc_f = truncated.astype(jnp.float32)
        # NOTE: computed from the successor state BEFORE the restart below overwrites
        # it — the whole point is to value the real successor, not the reset state.
        next_value = _values(train_state, _global_state(next_obs, next_env_state))
        if next_value.ndim > trunc_f.ndim:  # per-agent critic head
            trunc_f = trunc_f[:, None]
        reward = reward + config.gamma * trunc_f * next_value

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
            action_mask=action_mask if use_action_mask else _mask_placeholder,
        )
        carry = (train_state, next_env_state, next_obs, rng)
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

    @jax.jit
    def collect_fn(runner_state: RunnerState):
        train_state, rng = runner_state
        rng, reset_rng = jax.random.split(rng)

        # Fresh episodes every rollout, matching vanilla's per-collect reset
        obs, env_state = v_reset(jax.random.split(reset_rng, config.n_envs))

        # The final env state is bound (not discarded): with a state-derived global
        # state the bootstrap below needs it, not just the last observation.
        (train_state, last_env_state, last_obs, rng), scan_out = jax.lax.scan(
            _env_step,
            (train_state, env_state, obs, rng),
            None,
            length=config.n_steps,
        )
        if aligned:
            trajectory, aux = scan_out
            trajectory = _apply_aligned_rewards(trajectory, aux)
        else:
            trajectory = scan_out

        # Bootstrap value for GAE (done masking happens inside compute_gae)
        last_value = _values(
            train_state, _global_state(last_obs, last_env_state)
        )

        rollout_stats = {
            # Team reward, not the learner's signal — otherwise this would report
            # mean D and be incomparable to the dense baseline.
            "mean_reward": trajectory.team_reward.mean(),
            "episode_count": trajectory.done.sum(),
        }
        return (
            RunnerState(train_state=train_state, rng=rng),
            trajectory,
            last_value,
            rollout_stats,
        )

    # ------------------------------------------------------------------ update

    @jax.jit
    def update_fn(runner_state: RunnerState, trajectory, last_value):
        train_state, rng = runner_state
        rng, update_rng = jax.random.split(rng)
        train_state, losses = ppo_update(
            train_state, update_rng, trajectory, last_value, config, discrete
        )
        return RunnerState(train_state=train_state, rng=rng), losses

    # ------------------------------------------------------------------ eval

    @jax.jit
    def eval_fn(train_state: ActorCriticTrainState, rng: jax.Array):
        """Deterministic parallel-episode evaluation (PolicyEvaluator parity)."""
        keys = jax.random.split(rng, config.n_eval_episodes)
        obs, env_state = jax.vmap(env.reset)(keys)
        finished = jnp.zeros(config.n_eval_episodes, dtype=bool)
        episode_rewards = jnp.zeros(config.n_eval_episodes)

        def _eval_step(carry, _):
            obs, env_state, finished, episode_rewards = carry
            # The mask matters more here than in collection: an unmasked argmax picks
            # the single highest logit even when that action is illegal, so eval would
            # score a policy the env never actually runs.
            actions, _ = _actor_forward(
                train_state,
                obs,
                jax.random.PRNGKey(0),
                deterministic=True,
                action_mask=_avail(env_state),
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
            return (next_obs, next_env_state, finished, episode_rewards), None

        # Fixed-length scan (jit needs static bounds); finished episodes keep
        # stepping but their rewards are masked out, like vanilla's `finished`.
        (_, _, _, episode_rewards), _ = jax.lax.scan(
            _eval_step,
            (obs, env_state, finished, episode_rewards),
            None,
            length=env.max_steps,
        )
        return episode_rewards.mean()

    return init_fn, collect_fn, update_fn, eval_fn, num_updates
