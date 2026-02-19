from typing import NamedTuple, Tuple, Callable

import jax
import jax.numpy as jnp

from algorithms.mappo_jax.types import MAPPOConfig, Transition
from algorithms.mappo_jax.network import MAPPOActor, MAPPOCritic, sample_action
from algorithms.mappo_jax.mappo import (
    ActorCriticTrainState,
    create_train_state,
    ppo_update,
)


class RunnerState(NamedTuple):
    """Carries all mutable state through each update step."""

    train_state: ActorCriticTrainState
    env_state: object  # JaxMARL environment State pytree
    last_obs: dict  # {agent_name: (n_envs, obs_dim)}
    rng: jax.random.PRNGKey


def make_train(config: MAPPOConfig, env) -> Tuple[Callable, Callable, int]:
    """Build JIT-compiled init and update_step functions for JAX MAPPO.

    Returns:
        init_fn(rng) -> RunnerState
        update_step_fn(runner_state) -> (RunnerState, metrics_dict)
        num_updates: total number of update steps to run
    """
    agents = env.agents
    n_agents = env.num_agents

    first_agent = agents[0]
    obs_dim = env.observation_spaces[first_agent].shape[0]
    global_state_dim = obs_dim * n_agents

    from jaxmarl.environments.spaces import Discrete as JaxDiscrete

    discrete = isinstance(env.action_spaces[first_agent], JaxDiscrete)
    action_dim = (
        env.action_spaces[first_agent].n
        if discrete
        else env.action_spaces[first_agent].shape[0]
    )

    has_action_mask = hasattr(env, "get_avail_actions")
    num_updates = config.n_total_steps // (config.n_steps * config.n_envs)

    # -------------------------------------------------------------------------
    # init: initialize train state and vectorized envs
    # -------------------------------------------------------------------------
    @jax.jit
    def init_fn(rng: jax.random.PRNGKey) -> RunnerState:
        rng, init_rng, env_rng = jax.random.split(rng, 3)

        train_state = create_train_state(
            init_rng, config, obs_dim, global_state_dim, action_dim, discrete
        )

        env_rngs = jax.random.split(env_rng, config.n_envs)
        obs, env_state = jax.vmap(env.reset)(env_rngs)

        return RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obs,
            rng=rng,
        )

    # -------------------------------------------------------------------------
    # update_step: one trajectory collection + one PPO update
    # -------------------------------------------------------------------------
    def _env_step(carry, _):
        train_state, env_state, last_obs, rng = carry
        rng, step_rng, action_rng = jax.random.split(rng, 3)

        obs_stacked = jnp.stack(
            [last_obs[agent] for agent in agents], axis=1
        )  # (n_envs, n_agents, obs_dim)
        global_state = obs_stacked.reshape(config.n_envs, -1)

        values = train_state.critic_ts.apply_fn(
            train_state.critic_ts.params, global_state
        )

        if has_action_mask:
            avail_actions = jax.vmap(env.get_avail_actions)(env_state)
            masks_stacked = jnp.stack(
                [avail_actions[agent] for agent in agents], axis=1
            )
        else:
            masks_stacked = jnp.ones(
                (config.n_envs, n_agents, action_dim), dtype=jnp.float32
            )

        if config.parameter_sharing:
            obs_flat = obs_stacked.reshape(-1, obs_dim)
            masks_flat = masks_stacked.reshape(-1, action_dim)
            actions_flat, log_probs_flat = sample_action(
                action_rng,
                train_state.actor_ts.apply_fn,
                train_state.actor_ts.params,
                obs_flat,
                discrete,
                deterministic=False,
                action_mask=masks_flat if has_action_mask else None,
            )
            actions_arr = actions_flat.reshape(config.n_envs, n_agents)
            log_probs_arr = log_probs_flat.reshape(config.n_envs, n_agents)
        else:
            obs_per_agent = jnp.transpose(obs_stacked, (1, 0, 2))
            masks_per_agent = jnp.transpose(masks_stacked, (1, 0, 2))
            agent_rngs = jax.random.split(action_rng, n_agents)

            def _per_agent_act(agent_obs, agent_mask, agent_rng):
                return sample_action(
                    agent_rng,
                    train_state.actor_ts.apply_fn,
                    train_state.actor_ts.params,
                    agent_obs,
                    discrete,
                    deterministic=False,
                    action_mask=agent_mask if has_action_mask else None,
                )

            actions_per_agent, log_probs_per_agent = jax.vmap(_per_agent_act)(
                obs_per_agent, masks_per_agent, agent_rngs
            )
            actions_arr = jnp.transpose(actions_per_agent, (1, 0))
            log_probs_arr = jnp.transpose(log_probs_per_agent, (1, 0))

        actions_dict = {agent: actions_arr[:, i] for i, agent in enumerate(agents)}

        step_rngs = jax.random.split(step_rng, config.n_envs)
        next_obs, next_env_state, reward, done, _ = jax.vmap(env.step)(
            step_rngs, env_state, actions_dict
        )

        rewards_stacked = jnp.stack(
            [reward[agent] for agent in agents], axis=1
        )
        done_all = done["__all__"]

        transition = Transition(
            obs=obs_stacked,
            global_state=global_state,
            action=actions_arr,
            reward=rewards_stacked,
            done=done_all,
            log_prob=log_probs_arr,
            value=values,
            action_mask=masks_stacked,
        )

        carry = (train_state, next_env_state, next_obs, rng)
        return carry, transition

    @jax.jit
    def update_step_fn(runner_state: RunnerState):
        # Collect n_steps of experience
        initial_carry = (
            runner_state.train_state,
            runner_state.env_state,
            runner_state.last_obs,
            runner_state.rng,
        )
        final_carry, trajectory = jax.lax.scan(
            _env_step, initial_carry, None, length=config.n_steps
        )
        train_state, env_state, last_obs, rng = final_carry
        # last_done: whether each env's last collected step ended the episode
        last_done = trajectory.done[-1]

        # Bootstrap value
        last_obs_stacked = jnp.stack(
            [last_obs[agent] for agent in agents], axis=1
        )
        last_gs = last_obs_stacked.reshape(config.n_envs, -1)
        last_value = train_state.critic_ts.apply_fn(
            train_state.critic_ts.params, last_gs
        )
        last_value = last_value * (1.0 - last_done)

        # PPO update
        rng, update_rng = jax.random.split(rng)
        train_state, losses = ppo_update(
            train_state, update_rng, trajectory, last_value, config, discrete
        )

        # Metrics: mean reward per step across all envs and agents
        mean_reward = trajectory.reward.sum(axis=-1).mean()
        # Fraction of steps where an episode ended (proxy for episode rate)
        episode_done_rate = trajectory.done.mean()

        metrics = {
            "mean_reward": mean_reward,
            "episode_done_rate": episode_done_rate,
            **losses,
        }

        new_runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=last_obs,
            rng=rng,
        )
        return new_runner_state, metrics

    return init_fn, update_step_fn, num_updates
