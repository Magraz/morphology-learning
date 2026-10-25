"""Jitted collect/update/eval for the two-level waypoint hierarchy.

Two PPO policies on two timescales, both trained by the UNMODIFIED
`mappo_jax.mappo.ppo_update`:

* **Manager** — acts once per window of `c = goal_horizon` env steps. Its action
  is one 2-D Gaussian sample per agent, turned into a waypoint
  `w_i = s_i + R * b(a_i)`, where `b` is `clip(., -1, 1)` or `tanh` per
  `manager_action_bound` (see `waypoints.waypoint_from_action`). Its
  reward is the discounted TEAM reward over the window,
  `sum_{k<c} gamma_M^k r_{t+k}`, with discount `gamma_M^c` per decision, so it
  is trained on exactly "which waypoints led to high env return". `gamma_M` is
  the manager's per-step discount (`manager_step_gamma`), the worker's `gamma`
  unless set. The actor reads the team view (`manager_input: global`), each
  agent's own relative view (`relative`), or each agent's own observation
  (`local`, the information-matched mode); the critic always reads the team view.
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

Manager credit (`manager_credit`): under `team` (the default) every agent's
waypoint gets the team advantage. Under `counterfactual` each agent's advantage
is corrected by a baseline from a learned goal-conditioned advantage model,
averaged over counterfactual goals for that agent alone. Under `dpp` the same
model supplies a D++ term instead: the gain had the agent's nearest teammates
been sent toward it (`counterfactual.py`).

Training forks (`interventions: true`, `interventions.py`): at every
`intervention_interval`-th decision each env is also forked once, around a
random focal agent i, with teammates teleported next to agent i and given its
waypoint offset. Each fork runs one window as its own episode under the same
frozen policies, and its transitions join both levels' batches as extra env
columns. `worker.n_envs` is then the MAIN env count, already reduced by
`run.make_feudal_config` so that main plus fork steps fill the parent's batch.
The main rollout's random keys are unchanged by the forks, and evaluation is
unchanged.

Fork scheduling (`intervention_scheduler`): under "bandit" / "random" a scheduler
decides at every decision whether to fork, and `adaptive_collection.py` collects
instead of `collect_fn` below, from the same window and fork primitives
(`_decide_window`, `_run_window`, `_fork_place`, `_fork_rollout`). Its update
packs only the collected rows into the PPO batch and fits the bandit
(`intervention_bandit.py`). Under "heuristic" `collect_fn` below runs the main
rollout unchanged while each env samples the steps where an agent stands at an
under-staffed box (`_heuristic_window`); those forks run after the rollout and
are packed into fork columns (`_heuristic_forks`, `intervention_heuristic.py`).
Under "anytime" / "anytime_random" `anytime_collection.py` collects instead, from
`_decide_window`, `_run_window` (its tracker queries the scheduler at every step)
and `_place`; its update is the adaptive one plus the scheduler's own PPO
(`anytime_scheduler.py`).
"""

import math
import warnings
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from algorithms.mappo_jax.mappo import (
    ActorCriticTrainState,
    compute_gae,
    create_train_state,
    flatten_samples,
    masked_explained_variance,
    masked_mean_std,
    optimize_samples,
    ppo_update,
    prepare_advantages,
)
from algorithms.mappo_jax.network import sample_action
from algorithms.mappo_jax.trainer import (
    RunnerState,
    global_state_dim,
    global_state_fn,
)
from algorithms.mappo_jax.types import Transition
from algorithms.simplified_feudal_mappo_jax import adaptive_collection as ac
from algorithms.simplified_feudal_mappo_jax import anytime_collection as anc
from algorithms.simplified_feudal_mappo_jax import anytime_scheduler as asch
from algorithms.simplified_feudal_mappo_jax import counterfactual as cf
from algorithms.simplified_feudal_mappo_jax import intervention_bandit as ib
from algorithms.simplified_feudal_mappo_jax import intervention_heuristic as ih
from algorithms.simplified_feudal_mappo_jax import interventions as iv
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.types import (
    FeudalConfig,
    Intervention,
    LastValues,
    ManagerGoal,
    Rollout,
)

# A waypoint counts as reached when the agent's closest approach during its
# commitment comes within this fraction of R.
WAYPOINT_REACHED_TOL = 0.1
# A manager action component counts as saturated when its bounded value is
# within 1% of +-1, i.e. the waypoint offset sits on the edge of the R-square.
SATURATION_TOL = 0.99


class HierTrainState(NamedTuple):
    worker: ActorCriticTrainState
    manager: ActorCriticTrainState
    # The joint-goal advantage model (a flax TrainState), built only under
    # `manager_credit: counterfactual` or `dpp`. None (an empty pytree)
    # otherwise, so the default train state, and every checkpoint written from
    # it, is unchanged.
    manager_adv: object = None
    # The fork scheduler's bandit (a flax TrainState, `intervention_bandit.py`),
    # built only under `intervention_scheduler: bandit`; None otherwise.
    bandit: object = None


class Decision(NamedTuple):
    """One manager decision for a batch of envs (`_decide_window`)."""

    pos: jax.Array  # (B, N, 2) goal_state positions the waypoints start from
    waypoint: jax.Array  # (B, N, 2)
    actor_in: jax.Array  # (B, N, actor_dim) the manager actor's input
    critic_base: jax.Array  # (B, critic_dim) critic input without fork context
    action: jax.Array  # (B, N, 2) the raw Gaussian sample
    log_prob: jax.Array  # (B, N)
    critic_in: jax.Array  # (B, critic_dim [+ context]) the main row's critic input
    value: jax.Array  # (B,)


class Placement(NamedTuple):
    """A fork's teleport (`_fork_place`), one row per fork lane."""

    obs: jax.Array
    state: object
    valid: jax.Array  # (L,) bool: every recruit was placed
    n_recruits: jax.Array  # (L,) int
    recruit: jax.Array  # (L, N) bool
    offset: jax.Array  # (L, 2) agent i's realized waypoint offset


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


def validate_manager_input(config: FeudalConfig, env) -> None:
    """`manager_input: relative` needs the env's per-entity readout.

    `local` needs no hook. It warns when the waypoint's arena clip would reveal
    more of the agent's position than its own lidar does: the clip shortens the
    error within `R * world_width` of a wall, and the axial lidar rays report a
    wall only within `lidar_range`.
    """
    wp.validate_manager_input(config.manager_input)
    if config.manager_input == "relative" and not hasattr(env, "entity_state"):
        raise ValueError(
            "manager_input='relative' needs an env with an `entity_state(state)` "
            f"hook (per-agent and per-box blocks); {type(env).__name__} has none. "
            "Supported: multi_box_push_mjx."
        )
    world_width = getattr(env, "world_width", None)
    lidar_range = getattr(env, "lidar_range", None)
    if (
        config.manager_input == "local"
        and world_width is not None
        and lidar_range is not None
        and config.waypoint_radius * world_width >= lidar_range
    ):
        warnings.warn(
            f"manager_input='local' with waypoint_radius={config.waypoint_radius}: "
            f"the arena clip acts within {config.waypoint_radius * world_width:.2f} "
            f"world units of a wall, beyond lidar_range={lidar_range:.2f}, so the "
            "worker's error reveals wall distances its observation does not. The "
            "arm is no longer information-matched to the flat baseline.",
            stacklevel=2,
        )


def manager_actor_dim(config: FeudalConfig, env):
    """Manager actor input width. None under `global` (input_dims' formula);
    the observation width under `local`; under `relative`, read off the builder
    by abstract evaluation (no FLOPs), so the declared width is by construction
    the width the builder emits."""
    if config.manager_input == "global":
        return None
    if config.manager_input == "local":
        return int(env.observation_dim)

    def build(key):
        _, state = env.reset(key)
        agents, boxes = env.entity_state(state)
        return wp.manager_actor_input_relative(agents[None], boxes[None])

    return jax.eval_shape(build, jax.random.PRNGKey(0)).shape[-1]


def create_hier_train_state(rng, config: FeudalConfig, env) -> HierTrainState:
    """Worker + manager train states, each built by `mappo_jax`'s factory.

    The worker critic has one value head per agent (each agent has its own
    waypoint and hence its own return) and keeps that axis at one agent. The
    manager critic is a scalar team value. Under `manager_credit:
    counterfactual` or `dpp` a third network, the advantage model, reads the manager
    critic's input plus every agent's goal offset; its key comes from `fold_in`,
    so the worker and manager initializations are the same in either mode.
    Under `interventions: true` the manager critic also reads the fork context
    block (`interventions.critic_context`); the actor parameters are unchanged.
    """
    validate_manager_input(config, env)
    dims = wp.input_dims(
        env.observation_dim,
        global_state_dim(env),
        env.n_agents,
        env.goal_state_dim,
        manager_actor_dim=manager_actor_dim(config, env),
        manager_context_dim=iv.context_dim(config, env.n_agents),
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
    manager_adv = None
    if config.manager_credit in cf.GOAL_MODEL_CREDITS:
        manager_adv = cf.create_adv_model(
            jax.random.fold_in(rng, 2),
            config.manager,
            dims["manager_critic"] + int(env.goal_state_dim) * env.n_agents,
        )
    bandit = None
    if config.intervention_scheduler == "bandit":
        # It reads the manager actor's input; fold_in, so no other init moves.
        bandit = ib.create_bandit_state(
            jax.random.fold_in(rng, 3),
            dims["manager_actor"],
            config.intervention_bandit_hidden_dim,
            config.intervention_bandit_lr,
        )
    elif config.intervention_scheduler in asch.ANYTIME_SCHEDULERS:
        # The anytime scheduler takes the bandit's slot (frozen under
        # `anytime_random`). It reads every agent's manager actor input.
        bandit = asch.create_scheduler_state(
            jax.random.fold_in(rng, 3),
            dims["manager_actor"],
            env.n_agents,
            asch.fork_budget(config),
            config.intervention_bandit_hidden_dim,
            config.intervention_bandit_lr,
            asch.skip_bias(config, env.n_agents),
        )
    return HierTrainState(
        worker=worker, manager=manager, manager_adv=manager_adv, bandit=bandit
    )


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
    `decide(manager_ts, obs, global_state, pos, env_state, rng, deterministic)
        -> (waypoint, actor_in, critic_in, action, log_prob)`
        — the actor reads `obs` only under `manager_input: local` and
        `env_state` only under `relative`.
    `act(worker_ts, obs, pos, waypoint, k, rng, deterministic)
        -> (action, log_prob, actor_in)`  — `k` is the step within the window.
    `actor_input(obs, global_state, pos, env_state) -> actor_in`  — the manager
        actor's input alone, as `decide` builds it.

    All batched over a leading env axis.
    """

    observe: object
    decide: object
    act: object
    actor_input: object = None


def make_policy(config: FeudalConfig, env) -> Policy:
    horizon, radius = config.goal_horizon, config.waypoint_radius
    bound = config.manager_action_bound
    wp.validate_action_bound(bound)
    validate_manager_input(config, env)
    mode = config.manager_input
    v_pos = jax.vmap(env.goal_state)
    v_entity = jax.vmap(env.entity_state) if mode == "relative" else None
    global_state = global_state_fn(env)

    def observe(obs, env_state):
        return global_state(obs, env_state), v_pos(env_state)

    def actor_input(obs, gs, pos, env_state):
        if mode == "relative":
            return wp.manager_actor_input_relative(*v_entity(env_state))
        if mode == "local":
            return wp.manager_actor_input_local(obs)
        return wp.manager_actor_input(gs, pos)

    def decide(manager_ts, obs, gs, pos, env_state, rng, deterministic=False):
        actor_in = actor_input(obs, gs, pos, env_state)
        critic_in = wp.manager_critic_input(gs, pos)
        action, log_prob = _act(manager_ts, actor_in, rng, deterministic)
        waypoint = wp.waypoint_from_action(pos, action, radius, bound)
        return waypoint, actor_in, critic_in, action, log_prob

    def act(worker_ts, obs, pos, waypoint, k, rng, deterministic=False):
        error = wp.goal_error(waypoint, pos, radius)
        actor_in = wp.worker_actor_input(obs, error, wp.remaining_fraction(k, horizon))
        action, log_prob = _act(worker_ts, actor_in, rng, deterministic)
        return action, log_prob, actor_in

    return Policy(observe=observe, decide=decide, act=act, actor_input=actor_input)


def _source_stats(traj, last_value, config):
    """`(raw advantage std, explained variance)` of one source's batch, over its
    active rows only: the pre-update critic's fit per source, for logging."""
    adv, ret = compute_gae(
        traj.reward,
        traj.value,
        traj.done.astype(jnp.float32),
        last_value,
        config.gamma,
        config.gae_lambda,
    )
    active = traj.active_mask.astype(jnp.float32)
    weight = active if traj.reward.ndim == 3 else active.max(axis=-1)
    _, std, _ = masked_mean_std(
        adv.reshape(-1), jnp.broadcast_to(weight, adv.shape).reshape(-1)
    )
    return std, masked_explained_variance(ret, traj.value, weight)


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
    cf.validate_manager_credit(config, env.n_agents)
    wcfg, mcfg = config.worker, config.manager
    horizon, radius = config.goal_horizon, config.waypoint_radius
    # Static: store each decision's goals and correct the manager's per-agent
    # advantages (counterfactual.py). False is the original code path.
    credit_on = config.manager_credit in cf.GOAL_MODEL_CREDITS
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
    # Static: training forks (interventions.py). False is the original code path.
    iv.validate_interventions(config, env, n_windows)
    forks_on = config.interventions
    # Static: the adaptive fork collector (`adaptive_collection.py`).
    adaptive_on = iv.adaptive(config)
    # Static: the anytime fork collector (`anytime_collection.py`). It shares the
    # adaptive update (persistent streams, padded buffers).
    anytime_on = iv.anytime(config)
    interval = config.intervention_interval
    # Static: who decides a fork's goals (interventions.REDECIDE_MODES). "none"
    # is the original fork, which credits the goal chosen before the teleport.
    redecide = config.intervention_redecide if forks_on else "none"
    gamma = wcfg.gamma
    # The manager's per-step discount: it sums the team reward within a window
    # and bootstraps a truncation, and `ppo_update` discounts decisions with
    # `mcfg.gamma`. Both must describe one horizon, or the window return and
    # the GAE recursion value the future differently.
    m_gamma = gamma if config.manager_step_gamma is None else config.manager_step_gamma
    if not math.isclose(mcfg.gamma, m_gamma**horizon, rel_tol=1e-6):
        raise ValueError(
            f"manager.gamma ({mcfg.gamma}) must equal the manager's per-step "
            f"discount ({m_gamma}) ** goal_horizon ({horizon}) = {m_gamma**horizon}"
        )
    # `n_total_steps` caps SIMULATOR steps: under forks an update also steps the
    # fork lanes, so it consumes more of the budget than its main rollout.
    num_updates = int(wcfg.n_total_steps) // iv.simulator_steps_per_update(config)

    policy = make_policy(config, env)
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)
    v_teleport = (
        jax.vmap(partial(env.teleport_agents, radius=config.intervention_radius))
        if forks_on
        else None
    )
    # Fork diagnostics report recruit distance in world units (square arenas).
    world_scale = float(getattr(env, "world_width", 1.0))
    # Static: the heuristic fork rule (`intervention_heuristic.py`) instead of the
    # window-start forks. False is every other code path.
    heuristic_on = iv.heuristic(config)
    budget = config.intervention_heuristic_budget
    if heuristic_on:
        v_contacts = jax.vmap(env.box_contacts)
        v_candidate = jax.vmap(
            partial(ih.candidate, margin=config.intervention_heuristic_trigger_margin)
        )

    # Static: the manager critic reads the fork context block (forks on, except
    # under the anytime schedulers; `interventions.context_dim`).
    context_on = iv.context_dim(config, n_agents) > 0

    def m_critic_input(critic_base, context=None):
        """The manager critic's input. Under forks it carries the context block,
        all zeros for main rows (`interventions.critic_context`)."""
        if not context_on:
            return critic_base
        if context is None:
            context = iv.critic_context(n_agents, critic_base.shape[0])
        return jnp.concatenate([critic_base, context], axis=-1)
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

    def _worker_step(train_state, waypoint, carry, k, bootstrap=True):
        """One env step inside a window, under a fixed set of waypoints.

        Batch-width agnostic (main envs or fork lanes). `bootstrap=False` (the
        forks under `intervention_redecide` none / focal) drops both time-limit
        value additions: such a fork is its own episode and ends with the window,
        so no value may be added to its rewards. The `all` forks bootstrap like
        the main rollout.
        """
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

        next_obs, next_state, _, terminated, truncated, info = v_step(env_state, action)
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
        if bootstrap:
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
        else:
            reward = r_int
        w_done = last | done | ~alive

        # --- Manager: discounted team reward over the window, plus the same
        # truncation bootstrap with its own critic.
        m_reward = m_reward + alive_f * jnp.power(m_gamma, k) * team_reward
        if bootstrap:
            m_next_value = _value(
                train_state.manager,
                m_critic_input(wp.manager_critic_input(next_gs, next_pos)),
            )
            m_reward = m_reward + (
                time_limit.astype(jnp.float32)
                * jnp.power(m_gamma, k + 1)
                * m_next_value
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
            active_mask=jnp.broadcast_to(alive_f[:, None], (alive.shape[0], n_agents)),
            action_mask=no_mask,
        )

        # Freeze envs that have finished until the window boundary.
        env_state = _select(alive, next_state, env_state)
        obs = _select(alive, next_obs, obs)
        alive = alive & ~done
        carry = (env_state, obs, alive, m_reward, m_done, closest, rng)
        return carry, (transition, r_int)

    def _fork_place(env_state, waypoint, pos, focal, k_recruit, k_place):
        """The fork's teleport, for any number of lanes L (`focal.shape[0]`):
        draw N and the recruits, and place them next to agent i with agent i's
        realized waypoint offset (after the bound and arena clip). A failed
        placement returns the state unchanged with `valid` False."""
        n_recruits, recruit = iv.sample_recruits(k_recruit, focal, n_agents)
        lane = jnp.arange(focal.shape[0])
        offset = waypoint[lane, focal] - pos[lane, focal]  # (L, 2)
        return _place(env_state, focal, n_recruits, recruit, offset, k_place)

    def _place(env_state, focal, n_recruits, recruit, offset, k_place):
        """Teleport the given recruits of every lane next to its focal agent
        (`env.teleport_agents`, which also checks `pos + offset` stays in the
        arena)."""
        n_l = focal.shape[0]
        f_obs, f_state, valid = v_teleport(
            env_state, focal, recruit, offset, jax.random.split(k_place, n_l)
        )
        return Placement(
            obs=f_obs, state=f_state, valid=valid, n_recruits=n_recruits,
            recruit=recruit, offset=offset,
        )

    def _fork_rollout(
        train_state, waypoint, m_actor_in, m_critic_base, m_action, m_log_prob,
        focal, placement, key, k_scan, extra_fn=None,
    ):
        """One window of every placed fork lane (any number of lanes L).

        Returns `(worker Transition (H, L, ...), manager Transition (L, ...),
        focal (L,), N (L,), valid (L,), recruit distance (L,), delivered (L,))`
        (delivered: the fork delivered at least one box), plus, unless
        `redecide` is "none", a dict of `(L,)` per-fork diagnostics. Under
        "none" the manager record is the decision sampled in the ORIGINAL state
        (stored actor input, action and log-prob); under "focal" agent i's
        entries are a fresh decision in the post-teleport state; under either,
        only agent i carries actor weight. Under "all" every agent's entries are
        a fresh post-teleport decision, every agent carries actor weight, and
        the fork bootstraps like the main rollout. The critic reads the fork
        context in every mode. `extra_fn(start_state, end_state) -> dict of
        (L,)` adds per-fork series to that dict (the heuristic's outcomes).
        """
        f_obs, f_state, valid = placement.obs, placement.state, placement.valid
        n_recruits, recruit = placement.n_recruits, placement.recruit
        offset = placement.offset
        n_l = focal.shape[0]
        lane = jnp.arange(n_l)
        f_gs, new_pos = policy.observe(f_obs, f_state)
        target = new_pos + offset[:, None, :]  # the recruits' waypoints
        extras = ()
        if redecide != "none":
            # Fresh goals, sampled where the recruits now stand. fold_in leaves
            # the four splits above, and so the placement, identical to the
            # default fork's.
            k_decide = jax.random.fold_in(key, 1)
            r_waypoint, r_actor_in, r_critic_base, r_action, r_log_prob = (
                policy.decide(
                    train_state.manager, f_obs, f_gs, new_pos, f_state, k_decide
                )
            )
            # How far the help moved agent i's MEAN goal, in units of R (bounded
            # action space, before the arena clip): 0 = the manager ignores it.
            mean_pre, _ = _act(train_state.manager, m_actor_in, k_decide, True)
            mean_post, _ = _act(train_state.manager, r_actor_in, k_decide, True)
            mean_shift = jnp.linalg.norm(
                wp.bounded_action(mean_post, config.manager_action_bound)
                - wp.bounded_action(mean_pre, config.manager_action_bound),
                axis=-1,
            )[lane, focal]
            m_critic_base = r_critic_base
            if redecide == "focal":
                is_focal = jax.nn.one_hot(focal, n_agents, dtype=bool)  # (L, N)

                def focal_only(new, old):
                    mask = is_focal.reshape(is_focal.shape + (1,) * (new.ndim - 2))
                    return jnp.where(mask, new, old)

                waypoint = focal_only(r_waypoint, waypoint)
                m_actor_in = focal_only(r_actor_in, m_actor_in)
                m_action = focal_only(r_action, m_action)
                m_log_prob = focal_only(r_log_prob, m_log_prob)
                # Agent i did not move, so this is its new offset in either frame.
                offset = waypoint[lane, focal] - new_pos[lane, focal]
                target = new_pos + offset[:, None, :]
                # The placement filter checked the OLD offset, so a recruit's
                # copy of the new one can leave the arena: clip it, as a manager
                # waypoint is clipped.
                clipped = (
                    jnp.any(jnp.abs(target) > wp.ARENA_HALF_EXTENT, axis=-1) & recruit
                )
                target = jnp.clip(target, -wp.ARENA_HALF_EXTENT, wp.ARENA_HALF_EXTENT)
                extras = (
                    {
                        "intervention_redecide_mean_shift": mean_shift,
                        "intervention_redecide_recruit_clip_frac": (
                            clipped.sum(-1) / n_recruits
                        ),
                    },
                )
            else:  # "all": every agent follows its own fresh goal; none imposed.
                waypoint, m_actor_in, m_action, m_log_prob = (
                    r_waypoint, r_actor_in, r_action, r_log_prob,
                )
                target = waypoint
                gap = jnp.linalg.norm(
                    waypoint - waypoint[lane, focal][:, None, :], axis=-1
                ) / radius  # (L, N): each agent's waypoint to agent i's
                extras = (
                    {
                        "intervention_redecide_mean_shift": mean_shift,
                        "intervention_redecide_recruit_waypoint_gap": (
                            (gap * recruit).sum(-1) / n_recruits
                        ),
                    },
                )
        f_waypoint = jnp.where(recruit[..., None], target, waypoint)
        inner = (
            f_state,
            f_obs,
            valid,  # a failed placement never runs: masked from the start
            jnp.zeros(n_l),
            jnp.zeros(n_l, dtype=bool),
            wp.distance_to_waypoint(f_waypoint, new_pos, radius),
            k_scan,
        )
        (f_end_state, f_end_obs, f_alive, f_m_reward, _, _, _), (f_w_traj, _) = (
            jax.lax.scan(
                partial(
                    _worker_step, train_state, f_waypoint, bootstrap=redecide == "all"
                ),
                inner,
                jnp.arange(horizon),
            )
        )
        if redecide == "all":
            # The fork is an on-policy window, so its manager target is the true
            # return: add the continuing value (zero context, as for main rows)
            # of a lane still alive at the window end. A terminated lane gets
            # none, and a time limit inside the window was already bootstrapped
            # in `_worker_step` (the lane is no longer alive).
            end_gs, end_pos = policy.observe(f_end_obs, f_end_state)
            end_value = _value(
                train_state.manager,
                m_critic_input(wp.manager_critic_input(end_gs, end_pos)),
            )
            f_m_reward = f_m_reward + (
                f_alive.astype(jnp.float32) * jnp.power(m_gamma, horizon) * end_value
            )
        f_critic_in = m_critic_input(
            m_critic_base, iv.critic_context(n_agents, n_l, focal, n_recruits)
        )
        valid_f = valid.astype(jnp.float32)
        # Manager actor weight: only agent i's sample, unless every goal is one.
        weight = (
            jnp.ones((n_l, n_agents))
            if redecide == "all"
            else jax.nn.one_hot(focal, n_agents)
        )
        f_m = Transition(
            obs=m_actor_in,
            global_state=f_critic_in,
            action=m_action,
            reward=f_m_reward,
            done=jnp.ones(n_l, dtype=bool),
            log_prob=m_log_prob,
            value=_value(train_state.manager, f_critic_in),
            team_reward=f_w_traj.team_reward.sum(axis=0),
            active_mask=weight * valid_f[:, None],
            action_mask=no_mask,
        )
        moved = jnp.linalg.norm(new_pos - new_pos[lane, focal][:, None], axis=-1)
        recruit_distance = world_scale * (moved * recruit).sum(-1) / n_recruits
        delivered = iv.window_delivered(f_state, f_end_state)
        if extra_fn is not None:
            extras = (dict(extras[0] if extras else {}, **extra_fn(f_state, f_end_state)),)
        return (
            f_w_traj, f_m, focal, n_recruits, valid, recruit_distance, delivered
        ) + extras

    def _fork(operand):
        """One window of every fork lane of the fixed scheduler, from the main
        decision's start state: lane e forks env e around a focal agent drawn
        uniformly (`_fork_place`, then `_fork_rollout`)."""
        (train_state, env_state, obs, waypoint, pos, m_actor_in, m_critic_base,
         m_action, m_log_prob, key) = operand
        k_focal, k_recruit, k_place, k_scan = jax.random.split(key, 4)
        focal = iv.sample_focal(k_focal, n_envs, n_agents)  # (L,)
        placement = _fork_place(env_state, waypoint, pos, focal, k_recruit, k_place)
        return _fork_rollout(
            train_state, waypoint, m_actor_in, m_critic_base, m_action, m_log_prob,
            focal, placement, key, k_scan,
        )

    def _skip(operand):
        """`_fork`'s output for a window the interval skips: fully masked
        placeholder rows (`done`, zero weight, zero reward and value)."""
        out = jax.tree.map(
            lambda x: jnp.zeros(x.shape, x.dtype), jax.eval_shape(_fork, operand)
        )
        f_w, f_m = out[0], out[1]
        return (
            f_w._replace(done=jnp.ones_like(f_w.done)),
            f_m._replace(done=jnp.ones_like(f_m.done)),
        ) + out[2:]

    def _decide_window(train_state, env_state, obs, manager_rng):
        """The manager's decision at a window start, for any batch width, with
        the main row's critic input (zero fork context) and value."""
        gs, pos = policy.observe(obs, env_state)
        waypoint, m_actor_in, m_critic_base, m_action, m_log_prob = policy.decide(
            train_state.manager, obs, gs, pos, env_state, manager_rng
        )
        m_critic_in = m_critic_input(m_critic_base)
        m_value = _value(train_state.manager, m_critic_in)  # (E,)
        return Decision(
            pos=pos, waypoint=waypoint, actor_in=m_actor_in,
            critic_base=m_critic_base, action=m_action, log_prob=m_log_prob,
            critic_in=m_critic_in, value=m_value,
        )

    def _run_window(train_state, env_state, obs, d, rng, reset_rng, tracker=None):
        """`horizon` worker steps under decision `d` (any batch width), the
        window's records, and the reset of the envs whose episode ended in it.

        Returns `(env_state, obs, rng, (worker Transition (H, B, ...), manager
        Transition (B, ...), diagnostics, goal))`; the returned state is the
        next window's start state. `tracker`, a `(fn, carry)` pair, is called as
        `carry = fn(carry, env_state, obs, alive, k)` on every step's PRE-step
        state, and its final carry is appended to the return.
        """
        n = obs.shape[0]
        start_state = env_state
        inner = (
            env_state,
            obs,
            jnp.ones(n, dtype=bool),  # alive
            jnp.zeros(n),  # manager reward
            jnp.zeros(n, dtype=bool),  # episode ended in this window
            wp.distance_to_waypoint(d.waypoint, d.pos, radius),  # closest approach
            rng,
        )
        step = partial(_worker_step, train_state, d.waypoint)
        if tracker is None:
            (env_state, obs, _, m_reward, m_done, closest, rng), (w_traj, r_int) = (
                jax.lax.scan(step, inner, jnp.arange(horizon))
            )
        else:
            track, tracked = tracker

            def tracked_step(carry, k):
                inner, tracked = carry
                tracked = track(tracked, inner[0], inner[1], inner[2], k)
                inner, out = step(inner, k)
                return (inner, tracked), out

            ((env_state, obs, _, m_reward, m_done, closest, rng), tracked), (
                w_traj, r_int
            ) = jax.lax.scan(tracked_step, (inner, tracked), jnp.arange(horizon))

        # Frozen envs hold their last live state, so this is where each
        # commitment actually ended.
        _, end_pos = policy.observe(obs, env_state)
        diagnostics = {
            "offset": wp.distance_to_waypoint(d.waypoint, d.pos, radius),
            "final_error": wp.distance_to_waypoint(d.waypoint, end_pos, radius),
            "reached": (closest <= WAYPOINT_REACHED_TOL).astype(jnp.float32),
            "intrinsic_sum": r_int.sum(),
        }
        if forks_on:
            # The paired baseline for the forks' delivery share: a frozen env
            # holds its last live state, so this covers a mid-window ending.
            diagnostics["delivered"] = iv.window_delivered(start_state, env_state)

        m_transition = Transition(
            obs=d.actor_in,
            global_state=d.critic_in,
            action=d.action,
            reward=m_reward,
            done=m_done,
            log_prob=d.log_prob,
            value=d.value,
            team_reward=w_traj.team_reward.sum(axis=0),
            active_mask=jnp.ones((n, n_agents)),
            action_mask=no_mask,
        )
        # The decision's goals, for the counterfactual advantage model: the
        # positions the waypoints are measured from and the realized offsets
        # (equal to `wp.waypoint_offset(pos, m_action, ...)`).
        goal = (
            ManagerGoal(pos=d.pos, offset=wp.goal_error(d.waypoint, d.pos, radius))
            if credit_on
            else None
        )

        # Reset the envs whose episode ended during this window.
        def _restart(operand):
            cur_obs, cur_state = operand
            reset_obs, reset_state = v_reset(jax.random.split(reset_rng, n))
            return _select(m_done, reset_obs, cur_obs), _select(
                m_done, reset_state, cur_state
            )

        obs, env_state = jax.lax.cond(
            m_done.any(), _restart, lambda operand: operand, (obs, env_state)
        )
        out = (env_state, obs, rng, (w_traj, m_transition, diagnostics, goal))
        return out if tracker is None else out + (tracked,)

    def _window(carry, w):
        """One manager decision followed by `horizon` worker steps (and, on an
        intervention window, the forks)."""
        if heuristic_on:
            return _heuristic_window(carry, w)
        train_state, env_state, obs, rng = carry
        rng, manager_rng, reset_rng = jax.random.split(rng, 3)
        d = _decide_window(train_state, env_state, obs, manager_rng)

        fork = None
        if forks_on:
            operand = (
                train_state, env_state, obs, d.waypoint, d.pos, d.actor_in,
                d.critic_base, d.action, d.log_prob,
                # fold_in, so the main path's splits are unchanged.
                jax.random.fold_in(manager_rng, iv.FORK_KEY_NAMESPACE),
            )
            if interval == 1:
                fork = _fork(operand)
            else:
                # The window scan is not vmapped, so a skipped window runs no
                # fork physics.
                fork = jax.lax.cond(w % interval == 0, _fork, _skip, operand)

        env_state, obs, rng, records = _run_window(
            train_state, env_state, obs, d, rng, reset_rng
        )
        return (train_state, env_state, obs, rng), records + (fork,)

    def _heuristic_window(carry, w):
        """`_window` under the heuristic fork rule: the same main window, while
        every step's pre-step state is offered to each env's sample of fork
        moments (`intervention_heuristic.reservoir_update`). The forks run after
        the rollout (`_heuristic_forks`)."""
        train_state, env_state, obs, rng, reservoir = carry
        rng, manager_rng, reset_rng = jax.random.split(rng, 3)
        d = _decide_window(train_state, env_state, obs, manager_rng)
        # fold_in, so the main path's splits are unchanged.
        h_key = jax.random.fold_in(manager_rng, ih.HEURISTIC_KEY_NAMESPACE)

        def track(res, state, ob, alive, k):
            k_focal, k_priority = jax.random.split(jax.random.fold_in(h_key, k))
            moment = v_candidate(
                v_contacts(state), jax.random.split(k_focal, ob.shape[0])
            )
            return ih.reservoir_update(
                res, moment, state, ob, alive, w * horizon + k, k_priority
            )

        env_state, obs, rng, records, reservoir = _run_window(
            train_state, env_state, obs, d, rng, reset_rng, (track, reservoir)
        )
        return (train_state, env_state, obs, rng, reservoir), records + (None,)

    def _heuristic_forks(train_state, reservoir, w_traj, m_traj, key):
        """Run every stored fork moment of the rollout and pack the
        transitions into fork columns (`intervention_heuristic.py`).

        Each lane is one reservoir slot (env-major). It teleports the moment's
        recruits next to its focal agent, every agent re-decides its goal
        (`_fork_rollout` under `intervention_redecide: all`), and the fork runs
        `horizon` steps. Empty slots and failed placements are masked lanes.
        Returns `(Intervention, per-update scalars)`.
        """
        # `events` is per env, not per slot.
        lanes = ih.flatten_slots(reservoir._replace(events=None))
        n_l = lanes.priority.shape[0]
        lane = jnp.arange(n_l)
        moment = lanes.moment
        k_decide, k_place, k_fork, k_scan = jax.random.split(key, 4)
        # What the manager would choose at the moment without the help. Only
        # the re-decision diagnostics read it: every goal is re-decided after
        # the teleport.
        d = _decide_window(train_state, lanes.env_state, lanes.obs, k_decide)
        placement = _place(
            lanes.env_state,
            moment.focal,
            # At least 1, so per-recruit means stay finite on empty (masked) slots.
            jnp.maximum(moment.n_recruits, 1),
            moment.recruit,
            # No goal is copied (every agent re-decides), so the arena check on
            # `pos + offset` reduces to the recruit's own position.
            jnp.zeros((n_l, 2)),
            k_place,
        )
        placement = placement._replace(
            valid=placement.valid & jnp.isfinite(lanes.priority)
        )
        box = moment.box

        def outcomes(start, end):
            """The focal box after the teleport and at the fork's end."""
            c0, c1 = v_contacts(start), v_contacts(end)
            on_box = c0.touching[lane, :, box].sum(axis=-1)
            return {
                "intervention_start_met_frac": (
                    on_box >= c0.coupling[lane, box]
                ).astype(jnp.float32),
                "intervention_box_delivered_frac": (
                    c1.delivered[lane, box] & ~c0.delivered[lane, box]
                ).astype(jnp.float32),
                "intervention_box_progress": (
                    c0.goal_distance[lane, box] - c1.goal_distance[lane, box]
                ),
            }

        f_w, f_m, focal, _, valid, recruit_distance, delivered, extras = _fork_rollout(
            train_state, d.waypoint, d.actor_in, d.critic_base, d.action,
            d.log_prob, moment.focal, placement, k_fork, k_scan, extra_fn=outcomes,
        )
        worker, manager, order, kept, dropped = ih.pack(f_w, f_m, valid, n_windows)
        n_cols = ih.n_columns(n_l, n_windows)

        def packed(x):
            return jnp.swapaxes(x[order].reshape(n_cols, n_windows), 0, 1)

        intervention = Intervention(
            worker=worker,
            manager=manager,
            focal=packed(focal),
            n_recruits=packed(moment.n_recruits),
            valid=jnp.swapaxes(kept.reshape(n_cols, n_windows), 0, 1),
            step=packed(lanes.step),
            env=packed(lane // budget),
        )
        live_main = w_traj.active_mask[..., 0]
        g_main, complete = ih.paired_main_return(
            w_traj.team_reward, live_main, m_traj.done, lanes.step, lane // budget,
            horizon, m_gamma,
        )
        g_fork = ih.window_return(f_w.team_reward, m_gamma)
        stats = ih.diagnostics(
            reservoir, f_w, f_m, valid, recruit_distance, (g_fork, g_main, complete),
            live_main.sum(), n_agents, horizon, dropped,
        )
        stats.update(iv.redecide_diagnostics(valid, extras))
        stats.update(iv.return_shares("intervention", f_m.team_reward, delivered, valid))
        return intervention, stats

    @jax.jit
    def collect_fn(runner_state: RunnerState):
        train_state, rng = runner_state
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = v_reset(jax.random.split(reset_rng, n_envs))

        init = (train_state, env_state, obs, rng)
        if heuristic_on:
            init = init + (ih.reservoir_init(env_state, obs, budget),)
        carry, (w_traj, m_traj, diag, goal, fork) = jax.lax.scan(
            _window,
            init,
            jnp.arange(n_windows) if forks_on else None,
            length=n_windows,
        )
        env_state, obs, rng = carry[1:4]

        def flatten_windows(traj):
            # (n_windows, horizon, B, ...) -> (n_steps, B, ...)
            return jax.tree.map(
                lambda x: x.reshape((n_windows * horizon,) + x.shape[2:]), traj
            )

        w_traj = flatten_windows(w_traj)

        gs, pos = policy.observe(obs, env_state)
        last_values = LastValues(
            # The rollout's last step is always a window end, i.e. a worker
            # terminal, so the worker bootstrap is masked out by GAE.
            worker=jnp.zeros((n_envs, n_agents)),
            manager=_value(
                train_state.manager, m_critic_input(wp.manager_critic_input(gs, pos))
            ),
        )

        active_agent_steps = jnp.maximum(w_traj.active_mask.sum(), 1.0)
        active_env_steps = jnp.maximum(w_traj.active_mask[..., 0].sum(), 1.0)
        diagnostics = {
            "intrinsic_reward": diag["intrinsic_sum"].sum() / active_agent_steps,
            "waypoint_reached_frac": diag["reached"].mean(),
            "waypoint_final_error": diag["final_error"].mean(),
            "waypoint_offset": diag["offset"].mean(),
            "manager_window_return": m_traj.reward.mean(),
            "manager_action_saturation": (
                jnp.abs(wp.bounded_action(m_traj.action, config.manager_action_bound))
                >= SATURATION_TOL
            ).mean(),
            "rollout_team_reward": w_traj.team_reward.sum() / active_env_steps,
        }
        intervention = None
        if heuristic_on:
            intervention, fork_stats = _heuristic_forks(
                train_state, carry[4], w_traj, m_traj,
                # fold_in, so the main path's splits are unchanged.
                jax.random.fold_in(reset_rng, ih.FORK_RUN_NAMESPACE),
            )
            diagnostics.update(fork_stats)
        elif forks_on:
            (
                f_w_traj, f_m_traj, focal, n_recruits, valid, recruit_distance,
                fork_delivered, *extras,
            ) = fork
            intervention = Intervention(
                worker=flatten_windows(f_w_traj),
                manager=f_m_traj,
                focal=focal,
                n_recruits=n_recruits,
                valid=valid,
            )
            diagnostics.update(
                iv.fork_diagnostics(intervention, recruit_distance, interval, horizon)
            )
            if redecide != "none":
                diagnostics.update(iv.redecide_diagnostics(valid, extras[0]))
            # Both shares over the same (window, env) pairs: the windows whose
            # fork was placed, each fork against the main window it was cut from.
            diagnostics.update(
                iv.return_shares(
                    "intervention", f_m_traj.team_reward, fork_delivered, valid
                )
            )
            diagnostics.update(
                iv.return_shares(
                    "manager_window", m_traj.team_reward, diag["delivered"], valid
                )
            )
        if forks_on:
            diagnostics.update(
                {
                    "worker_eligible_main": w_traj.active_mask.sum(),
                    "worker_eligible_fork": intervention.worker.active_mask.sum(),
                    "manager_eligible_main": m_traj.active_mask.sum(),
                    "manager_eligible_fork": intervention.manager.active_mask.sum(),
                }
            )
        # Main rollout only: forks are not episodes of the task.
        rollout_stats = {
            "mean_reward": diagnostics["rollout_team_reward"],
            "episode_count": m_traj.done.sum(),
        }
        return (
            RunnerState(train_state=train_state, rng=rng),
            Rollout(
                worker=w_traj,
                manager=m_traj,
                diagnostics=diagnostics,
                manager_goal=goal,
                intervention=intervention,
            ),
            last_values,
            rollout_stats,
        )

    # ------------------------------------------------------------------ update

    def _manager_credit(train_state, rollout, last_values, rng):
        """Per-agent correction of the manager's team advantage, from the
        joint-goal advantage model `Â_φ` (counterfactual.py).

        `counterfactual`: `beta * c`, a baseline (never reads the agent's own goal).
        `dpp`: `-eta * max(0, D++_i)`, shaping (`ppo_update` subtracts the
        correction, so the D++ term is ADDED to the agent's advantage).

        Returns `(correction, refitted advantage model, logged scalars)`. Order
        matters: the correction is scored by the PRE-update model, fitted on
        earlier batches, so it cannot fit this batch's noise; under
        `counterfactual` the counterfactual goals are also drawn from the
        PRE-update manager actor (the policy that acted). The model is refitted
        on this batch last, in both modes.
        """
        m, goal = rollout.manager, rollout.manager_goal
        # The same team advantage `ppo_update` computes internally.
        a_team, _ = compute_gae(
            m.reward,
            m.value,
            m.done.astype(jnp.float32),
            last_values.manager,
            mcfg.gamma,
            mcfg.gae_lambda,
        )
        inputs = cf.adv_model_input(m.global_state, goal.offset)
        fitted = train_state.manager_adv.apply_fn(
            train_state.manager_adv.params, inputs
        )
        if config.manager_credit == "counterfactual":
            cf_offsets = cf.counterfactual_offsets(
                train_state.manager.actor_ts,
                m.obs,
                goal.pos,
                jax.random.fold_in(rng, 1),
                config.counterfactual_samples,
                radius,
                config.manager_action_bound,
                config.counterfactual_default,
            )
            c, sensitivity = cf.correction(
                train_state.manager_adv, m.global_state, goal.offset, cf_offsets
            )
            beta = cf.fit_beta(a_team, c)
            correction = beta * c
            stats = {
                f"manager_cf_{k}": v
                for k, v in cf.credit_diagnostics(
                    a_team, c, beta, sensitivity, fitted
                ).items()
            }
        else:  # dpp
            dpp, best_n = cf.dpp_credit(
                train_state.manager_adv,
                m.global_state,
                goal.offset,
                goal.pos,
                radius,
                config.dpp_max_recruits,
            )
            correction = cf.dpp_correction(dpp, config.dpp_coef)
            stats = {
                f"manager_dpp_{k}": v
                for k, v in cf.dpp_diagnostics(
                    a_team, dpp, best_n, config.dpp_coef
                ).items()
            }
            # The critic's own fit, under the same key as the counterfactual arm.
            stats["manager_cf_model_ev"] = cf.model_explained_variance(a_team, fitted)
        manager_adv, model_loss = cf.fit_adv_model(
            train_state.manager_adv,
            inputs,
            a_team,
            jax.random.fold_in(rng, 2),
            mcfg.n_epochs,
            mcfg.n_minibatches,
        )
        stats["manager_cf_model_loss"] = model_loss
        return correction, manager_adv, stats

    def _adaptive_update(train_state, rng, rollout):
        """The update under the adaptive and anytime collectors: GAE per source
        on the padded buffers (`adaptive_collection.adaptive_batches`), then
        exactly the collected rows into PPO, then the bandit or the anytime
        scheduler (last, on its own loss)."""
        rng, worker_rng, manager_rng = jax.random.split(rng, 3)
        worker_batch, worker_rows, manager_batch, manager_rows, sources = (
            ac.adaptive_batches(rollout, config, no_mask)
        )
        stats = {}
        for (level, source), traj in sources.items():
            cfg = wcfg if level == "worker" else mcfg
            # Every source ends in a recursion cut (the main manager's boundary
            # bootstrap is already folded in), so no last value is read.
            std, ev = _source_stats(traj, jnp.zeros(traj.reward.shape[1:]), cfg)
            stats[f"{level}_adv_std_{source}"] = std
            stats[f"{level}_explained_variance_{source}"] = ev
            stats[f"{level}_eligible_{source}"] = traj.active_mask.sum()
        slots = ac.window_slots(config)
        updated = {}
        for level, ts, batch, rows, size, cfg, key in (
            ("worker", train_state.worker, worker_batch, worker_rows,
             slots * horizon, wcfg, worker_rng),
            ("manager", train_state.manager, manager_batch, manager_rows,
             slots, mcfg, manager_rng),
        ):
            adv, ret, ev = prepare_advantages(
                batch, jnp.zeros(batch.reward.shape[1:]), cfg, masked_statistics=True
            )
            samples = ac.pack(flatten_samples(batch, adv, ret), rows, size)
            new_ts, level_losses = optimize_samples(
                ts, key, samples, cfg, discrete=False,
                squash=level == "manager" and config.manager_action_bound == "tanh",
                masked_statistics=True,
            )
            level_losses["explained_variance"] = ev
            updated[level] = (new_ts, level_losses)
        bandit = train_state.bandit
        if config.intervention_scheduler == "bandit":
            ad = rollout.adaptive
            bandit, loss, steps = ib.fit_bandit(
                bandit, ad.label_x, ad.label_y, ad.label_count,
                jax.random.fold_in(manager_rng, ac.NS_BANDIT),
            )
            stats["bandit_loss"] = loss
            stats["bandit_steps"] = steps.astype(jnp.float32)
        elif config.intervention_scheduler == "anytime":
            # `anytime_random` keeps its scheduler frozen at initialization.
            bandit, scheduler_stats = asch.fit_scheduler(
                bandit, rollout.scheduler,
                jax.random.fold_in(manager_rng, anc.NS_FIT),
                config.intervention_scheduler_ent_coef,
            )
            stats.update(scheduler_stats)
        losses = {f"worker_{k}": v for k, v in updated["worker"][1].items()}
        losses.update({f"manager_{k}": v for k, v in updated["manager"][1].items()})
        losses.update(rollout.diagnostics)
        losses.update(stats)
        train_state = HierTrainState(
            worker=updated["worker"][0],
            manager=updated["manager"][0],
            manager_adv=train_state.manager_adv,
            bandit=bandit,
        )
        return RunnerState(train_state=train_state, rng=rng), losses

    @jax.jit
    def update_fn(runner_state: RunnerState, rollout: Rollout, last_values):
        train_state, rng = runner_state
        if adaptive_on or anytime_on:
            return _adaptive_update(train_state, rng, rollout)
        rng, worker_rng, manager_rng = jax.random.split(rng, 3)
        worker_batch, worker_last = rollout.worker, last_values.worker
        manager_batch, manager_last = rollout.manager, last_values.manager
        source_stats = {}
        if forks_on:
            # Fork columns join the main batch along the env axis. Each fork ends
            # in a terminal, so GAE never crosses into another fork or into main,
            # and their last values are never read.
            fork = rollout.intervention
            # One fork column per env (fixed forks) or the heuristic's packed
            # columns (`intervention_heuristic.pack`).
            n_fork_cols = fork.manager.reward.shape[1]
            for level, main, f, last, cfg in (
                ("worker", rollout.worker, fork.worker, last_values.worker, wcfg),
                ("manager", rollout.manager, fork.manager, last_values.manager, mcfg),
            ):
                for source, traj, lv in (
                    ("main", main, last),
                    ("fork", f, jnp.zeros((n_fork_cols,) + last.shape[1:])),
                ):
                    std, ev = _source_stats(traj, lv, cfg)
                    source_stats[f"{level}_adv_std_{source}"] = std
                    source_stats[f"{level}_explained_variance_{source}"] = ev
            worker_batch = iv.append_columns(rollout.worker, fork.worker)
            worker_last = jnp.concatenate(
                [worker_last, jnp.zeros((n_fork_cols, n_agents))], axis=0
            )
            manager_batch = iv.append_columns(rollout.manager, fork.manager)
            manager_last = jnp.concatenate([manager_last, jnp.zeros(n_fork_cols)])
        worker, worker_losses = ppo_update(
            train_state.worker,
            worker_rng,
            worker_batch,
            worker_last,
            wcfg,
            discrete=False,
            masked_statistics=forks_on,
        )
        correction, manager_adv, credit_stats = None, train_state.manager_adv, {}
        if credit_on:
            # Keys come from fold_in, so the default path's splits are unchanged.
            correction, manager_adv, credit_stats = _manager_credit(
                train_state, rollout, last_values, manager_rng
            )
        manager, manager_losses = ppo_update(
            train_state.manager,
            manager_rng,
            manager_batch,
            manager_last,
            mcfg,
            discrete=False,
            squash=config.manager_action_bound == "tanh",
            advantage_correction=correction,
            masked_statistics=forks_on,
        )
        losses = {f"worker_{k}": v for k, v in worker_losses.items()}
        losses.update({f"manager_{k}": v for k, v in manager_losses.items()})
        losses.update(rollout.diagnostics)
        losses.update(credit_stats)
        losses.update(source_stats)
        return (
            RunnerState(
                train_state=HierTrainState(
                    worker=worker,
                    manager=manager,
                    manager_adv=manager_adv,
                    bandit=train_state.bandit,
                ),
                rng=rng,
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
            waypoint = policy.decide(
                train_state.manager, obs, gs, pos, env_state, no_rng, True
            )[0]
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

    def _final_value(train_state, env_state, obs):
        """The manager's value of each env's state (zero fork context), as the
        fixed collector's `last_values.manager`."""
        gs, pos = policy.observe(obs, env_state)
        return _value(
            train_state.manager, m_critic_input(wp.manager_critic_input(gs, pos))
        )

    if anytime_on:

        def _context(obs, env_state):
            """Every agent's manager actor input and goal-state position."""
            gs, pos = policy.observe(obs, env_state)
            return policy.actor_input(obs, gs, pos, env_state), pos

        collect_fn = anc.make_collect(
            config,
            env,
            anc.Primitives(
                decide_window=_decide_window,
                run_window=_run_window,
                place=_place,
                manager_value=_final_value,
                context=_context,
            ),
            m_gamma,
            SATURATION_TOL,
        )

    if adaptive_on:
        collect_fn = ac.make_collect(
            config,
            env,
            ac.Primitives(
                decide_window=_decide_window,
                run_window=_run_window,
                fork_place=_fork_place,
                fork_rollout=_fork_rollout,
                final_value=_final_value,
            ),
            m_gamma,
            SATURATION_TOL,
        )

    return init_fn, collect_fn, update_fn, eval_fn, num_updates
