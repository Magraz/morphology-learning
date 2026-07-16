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
    discrete = False  # the MJX suite is continuous force control

    if not config.parameter_sharing:
        raise NotImplementedError(
            "mappo_jax implements the shared-actor path only "
            "(parameter_sharing=true); use mappo_vanilla for independent actors"
        )

    num_updates = int(config.n_total_steps) // (config.n_steps * config.n_envs)

    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    def _actor_forward(train_state, obs, rng, deterministic):
        """Shared actor over (batch, n_agents, obs_dim) in one fused pass."""
        b = obs.shape[0]
        actions_flat, log_probs_flat = sample_action(
            rng,
            train_state.actor_ts.apply_fn,
            train_state.actor_ts.params,
            obs.reshape(b * n_agents, obs_dim),
            discrete,
            deterministic=deterministic,
        )
        return (
            actions_flat.reshape(b, n_agents, action_dim),
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
            init_rng, config, obs_dim, obs_dim * n_agents, action_dim, discrete,
            n_critic_outputs=n_critic_outputs,
        )
        return RunnerState(train_state=train_state, rng=rng)

    # ------------------------------------------------------------------ collect

    def _env_step(carry, _):
        train_state, env_state, obs, rng = carry
        rng, action_rng, reset_rng = jax.random.split(rng, 3)

        global_state = obs.reshape(config.n_envs, -1)
        values = _values(train_state, global_state)
        actions, log_probs = _actor_forward(
            train_state, obs, action_rng, deterministic=False
        )

        next_obs, next_env_state, reward, terminated, truncated, info = v_step(
            env_state, actions
        )
        # `reward` is what the learner optimizes (team scalar, or per-agent
        # difference rewards); `task_reward` is always the team scalar, so logged
        # returns stay comparable across reward modes.
        team_reward = info["task_reward"]
        done = jnp.logical_or(terminated, truncated)

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

        transition = Transition(
            obs=obs,
            global_state=global_state,
            action=actions,
            reward=reward,
            done=done,
            log_prob=log_probs,
            value=values,
            team_reward=team_reward,
        )
        return (train_state, next_env_state, next_obs, rng), transition

    @jax.jit
    def collect_fn(runner_state: RunnerState):
        train_state, rng = runner_state
        rng, reset_rng = jax.random.split(rng)

        # Fresh episodes every rollout, matching vanilla's per-collect reset
        obs, env_state = v_reset(jax.random.split(reset_rng, config.n_envs))

        (train_state, _, last_obs, rng), trajectory = jax.lax.scan(
            _env_step,
            (train_state, env_state, obs, rng),
            None,
            length=config.n_steps,
        )

        # Bootstrap value for GAE (done masking happens inside compute_gae)
        last_value = _values(train_state, last_obs.reshape(config.n_envs, -1))

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
            actions, _ = _actor_forward(
                train_state, obs, jax.random.PRNGKey(0), deterministic=True
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
