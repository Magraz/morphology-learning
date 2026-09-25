"""Jitted collect/update/eval for the two-level waypoint hierarchy.

Two PPO policies on two timescales, both trained by the UNMODIFIED
`mappo_jax.mappo.ppo_update`:

* **Manager** — acts once per window of `c = goal_horizon` env steps. Its action
  is one 2-D Gaussian sample per agent, turned into a waypoint
  `w_i = s_i + R * clip(a_i, -1, 1)` (see `waypoints.waypoint_from_action`). Its
  reward is the discounted TEAM reward over the window,
  `sum_{k<c} gamma^k r_{t+k}`, with discount `gamma^c` per decision, so it is
  trained on exactly "which waypoints led to high env return".
* **Worker** — acts every env step on `(obs_i, (w_i - s_i)/R, time left)`. Its
  ONLY reward is the distance to its waypoint closed by the step
  (`waypoints.intrinsic_reward`). Each commitment is one worker episode: `done`
  fires on the window's last step, since the goal expires there.

Rollout layout: `n_steps / c` windows, an outer `lax.scan` over windows and an
inner one over the `c` steps of each. An env that finishes mid-window is FROZEN
(its state held, its worker steps masked out through `active_mask`) and reset at
the next window boundary. That keeps every manager decision at a fixed index, so
both levels' data are plain `(time, env, ...)` arrays that `ppo_update` takes
as-is. The price is the frozen steps, which only occur on early termination.

Like `mappo_jax`, every rollout starts from freshly reset envs.
"""

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from algorithms.mappo_jax.mappo import (
    ActorCriticTrainState,
    create_train_state,
    ppo_update,
)
from algorithms.mappo_jax.network import sample_action
from algorithms.mappo_jax.trainer import (
    RunnerState,
    global_state_dim,
    global_state_fn,
)
from algorithms.mappo_jax.types import Transition
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.types import (
    FeudalConfig,
    LastValues,
    Rollout,
)

# A waypoint counts as reached when the agent's closest approach during its
# commitment comes within this fraction of R.
WAYPOINT_REACHED_TOL = 0.1


class HierTrainState(NamedTuple):
    worker: ActorCriticTrainState
    manager: ActorCriticTrainState


def validate_env(env) -> None:
    """The hierarchy needs agent positions and continuous primitive actions."""
    if not hasattr(env, "goal_state"):
        raise ValueError(
            f"simplified_feudal_mappo_jax needs an env with a `goal_state(state)` "
            f"hook (each agent's normalized position); {type(env).__name__} has "
            "none. Supported: multi_box_push_mjx, multi_box_multi_goal_push_mjx."
        )
    if getattr(env, "discrete", False) or hasattr(env, "avail_actions"):
        raise ValueError(
            "simplified_feudal_mappo_jax supports continuous-action envs only "
            f"(got {type(env).__name__}, which is discrete / action-masked)."
        )


def create_hier_train_state(rng, config: FeudalConfig, env) -> HierTrainState:
    """Worker + manager train states, each built by `mappo_jax`'s factory.

    The worker critic has one value head per agent (each agent has its own
    waypoint and hence its own return) and keeps that axis at one agent. The
    manager critic is a scalar team value.
    """
    dims = wp.input_dims(
        env.observation_dim, global_state_dim(env), env.n_agents, env.goal_state_dim
    )
    worker_rng, manager_rng = jax.random.split(rng)
    worker = create_train_state(
        worker_rng,
        config.worker,
        dims["worker_actor"],
        dims["worker_critic"],
        env.action_dim,
        discrete=False,
        n_critic_outputs=env.n_agents,
        keep_critic_output_axis=True,
    )
    manager = create_train_state(
        manager_rng,
        config.manager,
        dims["manager_actor"],
        dims["manager_critic"],
        int(env.goal_state_dim),
        discrete=False,
    )
    return HierTrainState(worker=worker, manager=manager)


def _act(ts: ActorCriticTrainState, x, rng, deterministic):
    """Shared actor over `(B, n_agents, in_dim)` in one fused pass."""
    b, n, d = x.shape
    action, log_prob = sample_action(
        rng,
        ts.actor_ts.apply_fn,
        ts.actor_ts.params,
        x.reshape(b * n, d),
        discrete=False,
        deterministic=deterministic,
    )
    return action.reshape(b, n, -1), log_prob.reshape(b, n)


def _value(ts: ActorCriticTrainState, x):
    return ts.critic_ts.apply_fn(ts.critic_ts.params, x)


class Policy(NamedTuple):
    """The hierarchy's forward pass, shared by training, eval and `view()`.

    `observe(obs, env_state) -> (global_state, pos)`
    `decide(manager_ts, global_state, pos, rng, deterministic)
        -> (waypoint, actor_in, critic_in, action, log_prob)`
    `act(worker_ts, obs, pos, waypoint, k, rng, deterministic)
        -> (action, log_prob, actor_in)`  — `k` is the step within the window.

    All batched over a leading env axis.
    """

    observe: object
    decide: object
    act: object


def make_policy(config: FeudalConfig, env) -> Policy:
    horizon, radius = config.goal_horizon, config.waypoint_radius
    v_pos = jax.vmap(env.goal_state)
    global_state = global_state_fn(env)

    def observe(obs, env_state):
        return global_state(obs, env_state), v_pos(env_state)

    def decide(manager_ts, gs, pos, rng, deterministic=False):
        actor_in = wp.manager_actor_input(gs, pos)
        critic_in = wp.manager_critic_input(gs, pos)
        action, log_prob = _act(manager_ts, actor_in, rng, deterministic)
        waypoint = wp.waypoint_from_action(pos, action, radius)
        return waypoint, actor_in, critic_in, action, log_prob

    def act(worker_ts, obs, pos, waypoint, k, rng, deterministic=False):
        error = wp.goal_error(waypoint, pos, radius)
        actor_in = wp.worker_actor_input(
            obs, error, wp.remaining_fraction(k, horizon)
        )
        action, log_prob = _act(worker_ts, actor_in, rng, deterministic)
        return action, log_prob, actor_in

    return Policy(observe=observe, decide=decide, act=act)


def _select(mask, new, old):
    """Per-env `where` over a pytree whose leaves lead with the env axis."""

    def pick(n, o):
        return jnp.where(mask.reshape((-1,) + (1,) * (n.ndim - 1)), n, o)

    return jax.tree.map(pick, new, old)


def make_train(config: FeudalConfig, env):
    """Build jitted train functions.

    Returns:
        init_fn(rng) -> RunnerState
        collect_fn(runner_state) -> (RunnerState, Rollout, LastValues, rollout_stats)
        update_fn(runner_state, Rollout, LastValues) -> (RunnerState, losses)
        eval_fn(train_state, rng) -> mean deterministic team return
        num_updates

    The same signature as `mappo_jax.trainer.make_train`, which is what lets
    the runner inherit `MAPPO_JAX_Runner.train` unchanged.
    """
    validate_env(env)
    wcfg, mcfg = config.worker, config.manager
    horizon, radius = config.goal_horizon, config.waypoint_radius
    if wcfg.n_steps % horizon:
        raise ValueError(
            f"n_steps ({wcfg.n_steps}) must be a multiple of goal_horizon "
            f"({horizon}); the runner rounds it up for you."
        )
    n_windows = wcfg.n_steps // horizon
    if n_windows < 2:
        # ppo_update normalizes advantages with an unbiased std over the time
        # axis, which is NaN over a single manager decision.
        raise ValueError(
            f"a rollout needs at least 2 manager windows; n_steps={wcfg.n_steps} "
            f"and goal_horizon={horizon} give {n_windows}"
        )

    n_envs, n_agents = wcfg.n_envs, env.n_agents
    gamma = wcfg.gamma
    num_updates = int(wcfg.n_total_steps) // (wcfg.n_steps * n_envs)

    policy = make_policy(config, env)
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)
    # No action masking: `ppo_update` switches on `action_mask.ndim == 4`, and a
    # scalar placeholder stacks to a 1-D array.
    no_mask = jnp.zeros(())

    # ------------------------------------------------------------------ init

    @jax.jit
    def init_fn(rng):
        rng, init_rng = jax.random.split(rng)
        return RunnerState(
            train_state=create_hier_train_state(init_rng, config, env), rng=rng
        )

    # ------------------------------------------------------------------ collect

    def _worker_step(train_state, waypoint, carry, k):
        """One env step inside a window, under a fixed set of waypoints."""
        env_state, obs, alive, m_reward, m_done, closest, rng = carry
        rng, action_rng = jax.random.split(rng)
        alive_f = alive.astype(jnp.float32)

        gs, pos = policy.observe(obs, env_state)
        critic_in = wp.worker_critic_input(
            gs, wp.goal_error(waypoint, pos, radius), wp.remaining_fraction(k, horizon)
        )
        value = _value(train_state.worker, critic_in)  # (E, N)
        action, log_prob, actor_in = policy.act(
            train_state.worker, obs, pos, waypoint, k, action_rng
        )

        next_obs, next_state, _, terminated, truncated, info = v_step(
            env_state, action
        )
        team_reward = info["task_reward"]
        done = terminated | truncated
        # No reset happens inside a window, so this is the TRUE successor.
        next_gs, next_pos = policy.observe(next_obs, next_state)
        time_limit = alive & truncated & ~terminated

        # --- Worker: the distance to the waypoint closed by this step, nothing else.
        r_int = wp.intrinsic_reward(waypoint, pos, next_pos, radius) * alive_f[:, None]
        # The window's last step is a true terminal (the goal expires). A time
        # limit that cuts the commitment short is a truncation, so it bootstraps
        # (SB3-style, as in mappo_jax) instead of being treated as worth 0.
        last = k == horizon - 1
        next_value = _value(
            train_state.worker,
            wp.worker_critic_input(
                next_gs,
                wp.goal_error(waypoint, next_pos, radius),
                wp.remaining_fraction(k + 1, horizon),
            ),
        )
        w_bootstrap = (time_limit & ~last).astype(jnp.float32)[:, None]
        reward = r_int + gamma * w_bootstrap * next_value
        w_done = last | done | ~alive

        # --- Manager: discounted team reward over the window, plus the same
        # truncation bootstrap with its own critic.
        m_reward = m_reward + alive_f * jnp.power(gamma, k) * team_reward
        m_next_value = _value(
            train_state.manager, wp.manager_critic_input(next_gs, next_pos)
        )
        m_reward = m_reward + (
            time_limit.astype(jnp.float32) * jnp.power(gamma, k + 1) * m_next_value
        )
        m_done = m_done | (alive & done)

        closest = jnp.where(
            alive[:, None],
            jnp.minimum(closest, wp.distance_to_waypoint(waypoint, next_pos, radius)),
            closest,
        )

        transition = Transition(
            obs=actor_in,
            global_state=critic_in,
            action=action,
            reward=reward,
            done=w_done,
            log_prob=log_prob,
            value=value,
            team_reward=team_reward * alive_f,
            active_mask=jnp.broadcast_to(alive_f[:, None], (n_envs, n_agents)),
            action_mask=no_mask,
        )

        # Freeze envs that have finished until the window boundary.
        env_state = _select(alive, next_state, env_state)
        obs = _select(alive, next_obs, obs)
        alive = alive & ~done
        carry = (env_state, obs, alive, m_reward, m_done, closest, rng)
        return carry, (transition, r_int)

    def _window(carry, _):
        """One manager decision followed by `horizon` worker steps."""
        train_state, env_state, obs, rng = carry
        rng, manager_rng, reset_rng = jax.random.split(rng, 3)

        gs, pos = policy.observe(obs, env_state)
        waypoint, m_actor_in, m_critic_in, m_action, m_log_prob = policy.decide(
            train_state.manager, gs, pos, manager_rng
        )
        m_value = _value(train_state.manager, m_critic_in)  # (E,)

        inner = (
            env_state,
            obs,
            jnp.ones(n_envs, dtype=bool),  # alive
            jnp.zeros(n_envs),  # manager reward
            jnp.zeros(n_envs, dtype=bool),  # episode ended in this window
            wp.distance_to_waypoint(waypoint, pos, radius),  # closest approach
            rng,
        )
        (env_state, obs, _, m_reward, m_done, closest, rng), (w_traj, r_int) = (
            jax.lax.scan(
                partial(_worker_step, train_state, waypoint),
                inner,
                jnp.arange(horizon),
            )
        )

        # Frozen envs hold their last live state, so this is where each
        # commitment actually ended.
        _, end_pos = policy.observe(obs, env_state)
        diagnostics = {
            "offset": wp.distance_to_waypoint(waypoint, pos, radius),
            "final_error": wp.distance_to_waypoint(waypoint, end_pos, radius),
            "reached": (closest <= WAYPOINT_REACHED_TOL).astype(jnp.float32),
            "intrinsic_sum": r_int.sum(),
        }

        m_transition = Transition(
            obs=m_actor_in,
            global_state=m_critic_in,
            action=m_action,
            reward=m_reward,
            done=m_done,
            log_prob=m_log_prob,
            value=m_value,
            team_reward=w_traj.team_reward.sum(axis=0),
            active_mask=jnp.ones((n_envs, n_agents)),
            action_mask=no_mask,
        )

        # Reset the envs whose episode ended during this window.
        def _restart(operand):
            cur_obs, cur_state = operand
            reset_obs, reset_state = v_reset(jax.random.split(reset_rng, n_envs))
            return _select(m_done, reset_obs, cur_obs), _select(
                m_done, reset_state, cur_state
            )

        obs, env_state = jax.lax.cond(
            m_done.any(), _restart, lambda operand: operand, (obs, env_state)
        )
        return (train_state, env_state, obs, rng), (w_traj, m_transition, diagnostics)

    @jax.jit
    def collect_fn(runner_state: RunnerState):
        train_state, rng = runner_state
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = v_reset(jax.random.split(reset_rng, n_envs))

        (_, env_state, obs, rng), (w_traj, m_traj, diag) = jax.lax.scan(
            _window, (train_state, env_state, obs, rng), None, length=n_windows
        )
        # (n_windows, horizon, E, ...) -> (n_steps, E, ...)
        w_traj = jax.tree.map(
            lambda x: x.reshape((n_windows * horizon,) + x.shape[2:]), w_traj
        )

        gs, pos = policy.observe(obs, env_state)
        last_values = LastValues(
            # The rollout's last step is always a window end, i.e. a worker
            # terminal, so the worker bootstrap is masked out by GAE.
            worker=jnp.zeros((n_envs, n_agents)),
            manager=_value(train_state.manager, wp.manager_critic_input(gs, pos)),
        )

        active_agent_steps = jnp.maximum(w_traj.active_mask.sum(), 1.0)
        active_env_steps = jnp.maximum(w_traj.active_mask[..., 0].sum(), 1.0)
        diagnostics = {
            "intrinsic_reward": diag["intrinsic_sum"].sum() / active_agent_steps,
            "waypoint_reached_frac": diag["reached"].mean(),
            "waypoint_final_error": diag["final_error"].mean(),
            "waypoint_offset": diag["offset"].mean(),
            "manager_window_return": m_traj.reward.mean(),
            "rollout_team_reward": w_traj.team_reward.sum() / active_env_steps,
        }
        rollout_stats = {
            "mean_reward": diagnostics["rollout_team_reward"],
            "episode_count": m_traj.done.sum(),
        }
        return (
            RunnerState(train_state=train_state, rng=rng),
            Rollout(worker=w_traj, manager=m_traj, diagnostics=diagnostics),
            last_values,
            rollout_stats,
        )

    # ------------------------------------------------------------------ update

    @jax.jit
    def update_fn(runner_state: RunnerState, rollout: Rollout, last_values):
        train_state, rng = runner_state
        rng, worker_rng, manager_rng = jax.random.split(rng, 3)
        worker, worker_losses = ppo_update(
            train_state.worker, worker_rng, rollout.worker, last_values.worker,
            wcfg, discrete=False,
        )
        manager, manager_losses = ppo_update(
            train_state.manager, manager_rng, rollout.manager, last_values.manager,
            mcfg, discrete=False,
        )
        losses = {f"worker_{k}": v for k, v in worker_losses.items()}
        losses.update({f"manager_{k}": v for k, v in manager_losses.items()})
        losses.update(rollout.diagnostics)
        return (
            RunnerState(
                train_state=HierTrainState(worker=worker, manager=manager), rng=rng
            ),
            losses,
        )

    # ------------------------------------------------------------------ eval

    n_eval_windows = math.ceil(env.max_steps / horizon)
    # Deterministic, so the key is never consumed.
    no_rng = jax.random.PRNGKey(0)

    @jax.jit
    def eval_fn(train_state: HierTrainState, rng):
        """Deterministic manager + worker; mean team return over episodes."""
        n_eps = wcfg.n_eval_episodes
        obs, env_state = v_reset(jax.random.split(rng, n_eps))

        def _eval_step(waypoint, carry, k):
            obs, env_state, finished, returns = carry
            _, pos = policy.observe(obs, env_state)
            action, _, _ = policy.act(
                train_state.worker, obs, pos, waypoint, k, no_rng, True
            )
            obs, env_state, _, terminated, truncated, info = v_step(env_state, action)
            returns = returns + jnp.where(finished, 0.0, info["task_reward"])
            finished = finished | terminated | truncated
            return (obs, env_state, finished, returns), None

        def _eval_window(carry, _):
            obs, env_state, _, _ = carry
            gs, pos = policy.observe(obs, env_state)
            waypoint = policy.decide(train_state.manager, gs, pos, no_rng, True)[0]
            carry, _ = jax.lax.scan(
                partial(_eval_step, waypoint), carry, jnp.arange(horizon)
            )
            return carry, None

        # Finished episodes keep stepping, but their rewards are masked out.
        init = (obs, env_state, jnp.zeros(n_eps, dtype=bool), jnp.zeros(n_eps))
        (_, _, _, returns), _ = jax.lax.scan(
            _eval_window, init, None, length=n_eval_windows
        )
        return returns.mean()

    return init_fn, collect_fn, update_fn, eval_fn, num_updates
