"""Jitted collect / update / eval for `rmappo_jax`, on-policy's runner loop.

Rollouts follow on-policy's shared runner (`runner/shared/*_runner.py`):
- Envs are reset ONCE, at `init_fn` (`warmup`). Every rollout continues from
  where the previous one stopped: the env states, observations, both GRU hidden
  states and the episode mask ride in `RunnerState`, as `after_update` copies
  the buffer's last row to its first. An episode that ends mid-rollout is
  restarted at once (the MJX envs have no auto-reset; on-policy's vec env does
  this itself), its hidden states are zeroed and its next mask is 0 (`insert`).
- The centralized critic's input is the env's centralized state, the same for
  every agent of an env (`mappo_jax.trainer.global_state_fn`: the env's own
  state when it publishes one, else the concatenated observations, which is
  what on-policy's MPE runner builds). The critic still emits one value per
  (env, agent) row, and every agent receives the shared team reward.
- A time-limit truncation is a terminal (on-policy's default
  `use_proper_time_limits=False`). With the flag on, this repo's bootstrap is
  used instead: the real successor is valued before the restart, through the
  critic's advanced hidden state, and `gamma * V` is added to the reward.

`make_train` returns the same `(init_fn, collect_fn, update_fn, eval_fn,
num_updates)` contract as `mappo_jax`, so `MAPPO_JAX_Runner.train` drives it.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from algorithms.mappo_jax.trainer import (
    EvalPolicy,
    global_state_dim,
    global_state_fn,
    make_eval_fn,
    per_agent_rewards,
    restart_done_envs,
)
from algorithms.rmappo_jax import value_norm as vn_lib
from algorithms.rmappo_jax.mappo import (
    RMAPPOTrainState,
    create_train_state,
    ppo_update,
    zero_hidden,
)
from algorithms.rmappo_jax.network import sample_action
from algorithms.rmappo_jax.types import RMAPPOConfig, Transition


class RunnerState(NamedTuple):
    """Everything that persists between updates.

    `train_state` and `rng` are the two fields `MAPPO_JAX_Runner.train` reads;
    the rest is the persistent rollout (on-policy's buffer row 0).
    """

    train_state: RMAPPOTrainState
    rng: jax.Array
    env_state: object
    obs: jax.Array  # (n_envs, n_agents, obs_dim)
    actor_h: jax.Array  # (n_envs, n_agents, recurrent_N, hidden) | placeholder
    critic_h: jax.Array
    mask: jax.Array  # (n_envs,) 0.0 where `obs` starts a new episode


def validate(config: RMAPPOConfig, env):
    """Reject the configurations this port does not implement."""
    if not config.parameter_sharing:
        raise NotImplementedError(
            "rmappo_jax implements on-policy's shared policy only "
            "(parameter_sharing=true)"
        )
    if per_agent_rewards(env):
        raise NotImplementedError(
            f"reward_mode={getattr(env, 'reward_mode', None)!r} emits a reward "
            "per agent, but every "
            "agent row of rmappo_jax's critic sees the same centralized state (as "
            "in on-policy's MPE runner), so the critic could only learn the "
            "agents' mean return. Use mappo_jax, whose critic has a per-agent head."
        )
    if hasattr(env, "avail_actions") and not getattr(env, "discrete", False):
        raise ValueError("avail_actions is only supported for discrete actions")
    if config.n_steps % config.chunk_length:
        # make_config rounds n_steps up; a config built another way must not
        # leave steps that no training chunk covers.
        raise ValueError(
            f"n_steps={config.n_steps} is not a multiple of "
            f"data_chunk_length={config.data_chunk_length}"
        )


def actor_step(
    actor_apply, params, obs, h, mask, rng, discrete, deterministic, action_mask=None
):
    """One decision of the shared actor for a batch of envs.

    `obs` `(B, N, obs_dim)`, `h` `(B, N, recurrent_N, H)` or the feedforward
    placeholder, `mask` `(B,)`. Rows are flattened agent-major, as on-policy's
    `np.concatenate(buffer.obs[step])`. Returns `(actions, log_probs, h)`.
    """
    b, n = obs.shape[:2]
    flat_h = h if h.ndim == 0 else h.reshape(b * n, *h.shape[2:])
    out, new_h = actor_apply(
        params, obs.reshape(b * n, -1), flat_h, jnp.repeat(mask, n)
    )
    flat_action_mask = None if action_mask is None else action_mask.reshape(b * n, -1)
    actions, log_probs = sample_action(
        rng, out, discrete, deterministic, flat_action_mask
    )
    action_shape = (b, n) if discrete else (b, n, -1)
    new_h = new_h if h.ndim == 0 else new_h.reshape(h.shape)
    return actions.reshape(action_shape), log_probs.reshape(b, n), new_h


def critic_step(critic_apply, params, cent_obs, h, mask, n_agents):
    """The critic's value for every (env, agent) row. `cent_obs` `(B, gs_dim)`
    is repeated for each agent. Returns `(values (B, N), h)`, values normalized."""
    b = cent_obs.shape[0]
    flat_h = h if h.ndim == 0 else h.reshape(b * n_agents, *h.shape[2:])
    values, new_h = critic_apply(
        params,
        jnp.repeat(cent_obs, n_agents, axis=0),
        flat_h,
        jnp.repeat(mask, n_agents),
    )
    new_h = new_h if h.ndim == 0 else new_h.reshape(h.shape)
    return values.reshape(b, n_agents), new_h


def make_train(config: RMAPPOConfig, env):
    validate(config, env)
    n_agents = env.n_agents
    obs_dim = env.observation_dim
    action_dim = env.action_dim
    discrete = getattr(env, "discrete", False)
    gs_dim = global_state_dim(env)
    _global_state = global_state_fn(env)
    recurrent = config.use_recurrent_policy

    num_updates = int(config.n_total_steps) // (config.n_steps * config.n_envs)

    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    use_action_mask = hasattr(env, "avail_actions")
    _v_avail = jax.vmap(env.avail_actions) if use_action_mask else None

    def _avail(env_state):
        return _v_avail(env_state) if use_action_mask else None

    def _denormalize(value_norm, values):
        if config.use_valuenorm:
            return vn_lib.denormalize(value_norm, values)
        return values

    def _zero_done(h, done):
        """`insert`: the hidden states of an env whose episode ended are zeroed."""
        if not recurrent:
            return h
        return h * (1.0 - done.astype(jnp.float32))[:, None, None, None]

    # ------------------------------------------------------------------ init

    @jax.jit
    def init_fn(rng: jax.Array) -> RunnerState:
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        train_state = create_train_state(
            init_rng, config, obs_dim, gs_dim, action_dim, discrete, num_updates
        )
        obs, env_state = v_reset(jax.random.split(reset_rng, config.n_envs))
        hidden = zero_hidden(config, (config.n_envs, n_agents))
        return RunnerState(
            train_state=train_state,
            rng=rng,
            env_state=env_state,
            obs=obs,
            actor_h=hidden,
            critic_h=hidden,
            # warmup: the buffer's masks start as ones
            mask=jnp.ones((config.n_envs,)),
        )

    # ------------------------------------------------------------------ collect

    def _env_step(carry, _):
        train_state, env_state, obs, actor_h, critic_h, mask, rng = carry
        rng, action_rng, reset_rng = jax.random.split(rng, 3)

        global_state = _global_state(obs, env_state)
        action_mask = _avail(env_state)
        actions, log_probs, next_actor_h = actor_step(
            train_state.actor_ts.apply_fn,
            train_state.actor_ts.params,
            obs,
            actor_h,
            mask,
            action_rng,
            discrete,
            deterministic=False,
            action_mask=action_mask,
        )
        values, next_critic_h = critic_step(
            train_state.critic_ts.apply_fn,
            train_state.critic_ts.params,
            global_state,
            critic_h,
            mask,
            n_agents,
        )

        next_obs, next_env_state, _, terminated, truncated, info = v_step(
            env_state, actions
        )
        team_reward = info["task_reward"]
        done = jnp.logical_or(terminated, truncated)
        # Every agent receives the shared team reward (on-policy's buffer holds
        # one reward per agent).
        reward = jnp.broadcast_to(team_reward[:, None], (config.n_envs, n_agents))
        if config.use_proper_time_limits:
            # Value the real successor BEFORE the restart below overwrites it,
            # continuing the episode (mask 1) from the advanced hidden state.
            next_value, _ = critic_step(
                train_state.critic_ts.apply_fn,
                train_state.critic_ts.params,
                _global_state(next_obs, next_env_state),
                next_critic_h,
                jnp.ones((config.n_envs,)),
                n_agents,
            )
            reward = reward + config.gamma * truncated[:, None] * _denormalize(
                train_state.value_norm, next_value
            )

        next_obs, next_env_state = restart_done_envs(
            v_reset, reset_rng, config.n_envs, done, next_obs, next_env_state
        )

        transition = Transition(
            obs=obs,
            global_state=global_state,
            actor_h=actor_h,
            critic_h=critic_h,
            mask=mask,
            action=actions,
            log_prob=log_probs,
            value=values,
            reward=reward,
            done=done,
            active_mask=info.get("active", jnp.ones((config.n_envs, n_agents))),
            action_mask=action_mask if use_action_mask else jnp.zeros(()),
            team_reward=team_reward,
        )
        carry = (
            train_state,
            next_env_state,
            next_obs,
            _zero_done(next_actor_h, done),
            _zero_done(next_critic_h, done),
            1.0 - done.astype(jnp.float32),
            rng,
        )
        return carry, transition

    @jax.jit
    def collect_fn(runner_state: RunnerState):
        rs = runner_state
        (_, env_state, obs, actor_h, critic_h, mask, rng), trajectory = jax.lax.scan(
            _env_step,
            (
                rs.train_state,
                rs.env_state,
                rs.obs,
                rs.actor_h,
                rs.critic_h,
                rs.mask,
                rs.rng,
            ),
            None,
            length=config.n_steps,
        )
        # `compute`: the bootstrap value of the step after the rollout.
        last_value, _ = critic_step(
            rs.train_state.critic_ts.apply_fn,
            rs.train_state.critic_ts.params,
            _global_state(obs, env_state),
            critic_h,
            mask,
            n_agents,
        )
        rollout_stats = {
            "mean_reward": trajectory.team_reward.mean(),
            "episode_count": trajectory.done.sum(),
        }
        new_state = rs._replace(
            rng=rng,
            env_state=env_state,
            obs=obs,
            actor_h=actor_h,
            critic_h=critic_h,
            mask=mask,
        )
        return new_state, trajectory, last_value, rollout_stats

    # ------------------------------------------------------------------ update

    @jax.jit
    def update_fn(runner_state: RunnerState, trajectory, last_value):
        rng, update_rng = jax.random.split(runner_state.rng)
        train_state, losses = ppo_update(
            runner_state.train_state,
            update_rng,
            trajectory,
            last_value,
            config,
            discrete,
        )
        return runner_state._replace(train_state=train_state, rng=rng), losses

    # ------------------------------------------------------------------ eval

    def _eval_act(train_state, hidden, obs, env_state):
        # One episode per env and no restarts, so the mask is always 1.
        actions, _, hidden = actor_step(
            train_state.actor_ts.apply_fn,
            train_state.actor_ts.params,
            obs,
            hidden,
            jnp.ones((obs.shape[0],)),
            jax.random.PRNGKey(0),
            discrete,
            deterministic=True,
            action_mask=_avail(env_state),
        )
        return hidden, actions

    eval_fn = make_eval_fn(
        env,
        config.n_eval_episodes,
        EvalPolicy(
            init=lambda train_state, n: zero_hidden(config, (n, n_agents)),
            act=_eval_act,
        ),
    )

    return init_fn, collect_fn, update_fn, eval_fn, num_updates
