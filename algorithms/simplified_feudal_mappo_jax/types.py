from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax

from algorithms.mappo_jax.types import MAPPOConfig
from algorithms.mappo_jax.types import Params as MAPPOParams


@dataclass
class Params(MAPPOParams):
    """The `mappo_jax` PPO surface (it drives the worker) plus the manager's own
    optimization knobs.

    The manager decides once per `goal_horizon` steps, so it sees ~c x fewer
    samples per update than the worker; sharing the worker's epoch/minibatch
    counts would give it the same number of gradient steps on a tiny batch.
    Every other PPO hyperparameter (clip, entropy, value coef, lambda, grad clip)
    is shared. The manager's discount per decision is `manager_gamma **
    goal_horizon`, where `manager_gamma` is a PER-ENV-STEP discount.
    """

    manager_lr: float = 3e-4
    manager_n_epochs: int = 4
    manager_n_minibatches: int = 2
    # Per-ENV-STEP discount of the manager. None (the default) uses the worker's
    # `gamma`, i.e. both levels share one ~100-step horizon. Set higher (e.g.
    # 0.997, a ~333-step horizon) to give the manager a longer horizon than the
    # worker, as in FeUdal Networks. Per step rather than per decision so one
    # value means the same horizon at any goal_horizon.
    manager_gamma: Optional[float] = None
    # Entropy coefficient of the MANAGER only. None (the default) shares the
    # worker's `ent_coef`. Under `manager_action_bound: tanh` the bonus is the
    # squashed entropy, which peaks at mean 0 (a zero waypoint offset), so in
    # states that pay nothing it pulls the manager toward standing still; a
    # smaller value weakens that pull.
    manager_ent_coef: Optional[float] = None


@dataclass
class Model_Params:
    hidden_dim: int
    # Both REQUIRED (no default): `model=mlp` must fail here rather than train
    # into experiments/results/<env>/mlp/, the mappo_jax baseline's directory.
    goal_horizon: int  # c: steps a waypoint stays in force
    waypoint_radius: float  # R: max waypoint offset per axis, in goal_state units
    # "clip" | "tanh": how the manager's Gaussian action is bounded to [-1, 1]
    # (waypoints.ACTION_BOUNDS). "clip" is the original behaviour.
    manager_action_bound: str = "clip"
    # "global" | "relative" | "local": what the manager ACTOR reads
    # (waypoints.MANAGER_INPUTS). "global" is the original behaviour; "local"
    # (own observation only) is the one information-matched to flat mappo_jax.
    manager_input: str = "global"
    # "team" | "counterfactual" | "dpp": how each agent's waypoint is credited
    # (counterfactual.MANAGER_CREDITS). "team" is the original behaviour: every
    # agent gets the team advantage. "counterfactual" subtracts a per-agent
    # baseline from a learned goal-conditioned advantage model. "dpp" adds a
    # per-agent D++ term from the same model (shaping, not a baseline).
    manager_credit: str = "team"
    # "sampled" | "hold": the counterfactual goals that baseline averages over
    # (counterfactual.COUNTERFACTUAL_DEFAULTS). Read only under "counterfactual".
    counterfactual_default: str = "sampled"
    # K, the number of draws per agent under "sampled" (ignored by "hold").
    counterfactual_samples: int = 16
    # eta, the weight of the D++ term: A_i = A_team + eta * max(0, D++_i). Read
    # only under "dpp". 1.0 credits each goal with the better of the realized
    # joint outcome and the outcome had teammates come to the agent.
    dpp_coef: float = 1.0
    # Largest number of recruits the D++ search tries. None = every teammate
    # (N - 1). Read only under "dpp".
    dpp_max_recruits: Optional[int] = None
    # Training-time simulator forks (`interventions.py`). At every
    # `intervention_interval`-th manager decision each env is forked once around
    # a random focal agent i: N ~ U{1..n_agents-1} teammates are teleported within
    # `intervention_radius` WORLD units of agent i and given i's waypoint offset.
    # Each fork is a one-window episode whose transitions join both levels' PPO
    # batches. The fork steps count toward the parent's per-update batch: the
    # main rollout runs fewer envs (`interventions.main_envs`). Execution and
    # evaluation are unchanged. False is the original code path. Set all three
    # in a model group, never on the CLI: the interval
    # and radius leave checkpoints shape-identical, so the model group name in
    # the results path is the only record of them.
    interventions: bool = False
    intervention_radius: float = 1.5
    intervention_interval: int = 1
    # Who decides a fork's goals (`interventions.REDECIDE_MODES`):
    #   none  — the original fork: agent i's goal as chosen BEFORE the teleport,
    #           copied to the recruits; only agent i's manager row trains.
    #   focal — agent i re-samples its goal from the post-teleport state and the
    #           recruits copy it; only agent i's manager row trains.
    #   all   — every agent re-samples its own goal from the post-teleport state,
    #           every manager row trains, and the fork bootstraps like the main
    #           rollout (an on-policy window from a teleported start state).
    # Checkpoint-identical across modes: set it in a model group, never on the CLI.
    intervention_redecide: str = "none"
    # Who schedules the forks (`intervention_bandit.SCHEDULERS`):
    #   fixed  — the original fork at every `intervention_interval`-th decision;
    #   bandit — a learned contextual bandit decides at each decision whether to
    #            fork (`intervention_bandit.py`), and skipped forks are replaced
    #            by extra main windows (`adaptive_collection.py`);
    #   random — the same adaptive collector, forks proposed with probability
    #            `intervention_random_rate` (0 never forks: the all-skip control);
    #   heuristic — a fixed rule forks at any step where an agent is near or
    #            touching its closest undelivered box and that box's coupling is
    #            not met, teleporting exactly the missing agents
    #            (`intervention_heuristic.py`). Needs `intervention_redecide: all`;
    #   anytime — a learned scheduler may fork at any step of a main window and
    #            chooses the focal agent, the recruits and the fork's length in
    #            windows, from a per-stream budget (`anytime_scheduler.py`,
    #            `anytime_collection.py`). Needs `intervention_redecide: all`;
    #   anytime_random — the same collector with that scheduler frozen at its
    #            initialization (its matched control).
    # The adaptive and anytime schedulers need `interventions: true`, interval
    # 1, team credit and an even env-group `n_envs` (half of it are the main
    # streams). Plan: plans/intervention_contextual_bandit_2026-10-16.md.
    intervention_scheduler: str = "fixed"
    intervention_bandit_hidden_dim: int = 32
    intervention_bandit_epsilon: float = 0.1
    intervention_bandit_lr: float = 3e-4
    # Task-return units, per executed fork.
    intervention_bandit_cost_coef: float = 0.0
    # Task-return units, subtracted from the target of a fork whose every raw
    # step earned exactly 0 reward. 0 turns the penalty off.
    intervention_bandit_zero_reward_penalty: float = 5.0
    intervention_random_rate: float = 0.5
    # `heuristic` scheduler only (`intervention_heuristic.py`): at most this
    # many forks per main env per update, a uniform sample of the qualifying
    # moments of its rollout.
    intervention_heuristic_budget: int = 5
    # World units beyond touch range: an agent qualifies as a focal agent once
    # its closest undelivered box is within touch range plus this margin.
    intervention_heuristic_trigger_margin: float = 0.5
    # `anytime` / `anytime_random` only (`anytime_scheduler.py`). The budget is
    # in goal windows per main stream per update (None = n_windows, so forks
    # can take at most half the batch); unspent budget becomes main windows.
    intervention_fork_budget: Optional[int] = None
    # The scheduler's own PPO over its decisions within a rollout (`anytime`
    # only): discount and GAE lambda per decision (gamma 0 is a one-step
    # bandit), and its entropy coefficient.
    intervention_scheduler_gamma: float = 1.0
    intervention_scheduler_lambda: float = 0.95
    intervention_scheduler_ent_coef: float = 0.01
    # Expected launches per main stream per update at initialization (sets the
    # scheduler's initial skip probability).
    intervention_scheduler_init_forks: float = 2.0


@dataclass
class Experiment:
    device: str
    model_params: Model_Params
    params: Params


@dataclass
class FeudalConfig:
    """Resolved config consumed by `trainer.make_train`.

    `worker` and `manager` are ordinary `MAPPOConfig`s, each handed to the
    unmodified `mappo_jax.mappo.ppo_update`. `worker.n_steps` is the rollout
    length in env steps and is a multiple of `goal_horizon`.
    """

    worker: MAPPOConfig
    manager: MAPPOConfig
    goal_horizon: int
    waypoint_radius: float
    manager_action_bound: str = "clip"
    manager_input: str = "global"
    # Per-env-step discount used to sum the team reward within a window and to
    # bootstrap a truncation. None = `worker.gamma` (the original code path).
    # `manager.gamma` must equal this ** goal_horizon; `make_train` checks it.
    manager_step_gamma: Optional[float] = None
    # Manager credit assignment (see Model_Params and `counterfactual.py`).
    # "team" is the original code path.
    manager_credit: str = "team"
    counterfactual_default: str = "sampled"
    counterfactual_samples: int = 16
    dpp_coef: float = 1.0
    dpp_max_recruits: Optional[int] = None
    # Training-time forks (see Model_Params and `interventions.py`). False is the
    # original code path.
    interventions: bool = False
    intervention_radius: float = 1.5
    intervention_interval: int = 1
    intervention_redecide: str = "none"
    # Fork scheduling (see Model_Params and `intervention_bandit.py`). "fixed" is
    # the original code path. `intervention_batch_envs` is the env group's batch
    # width, set only for the adaptive schedulers, whose `worker.n_envs` is the
    # number of persistent main streams (half of it).
    intervention_scheduler: str = "fixed"
    intervention_batch_envs: Optional[int] = None
    intervention_bandit_hidden_dim: int = 32
    intervention_bandit_epsilon: float = 0.1
    intervention_bandit_lr: float = 3e-4
    intervention_bandit_cost_coef: float = 0.0
    intervention_bandit_zero_reward_penalty: float = 5.0
    intervention_random_rate: float = 0.5
    intervention_heuristic_budget: int = 5
    intervention_heuristic_trigger_margin: float = 0.5
    # The anytime scheduler (see Model_Params and `anytime_scheduler.py`).
    intervention_fork_budget: Optional[int] = None
    intervention_scheduler_gamma: float = 1.0
    intervention_scheduler_lambda: float = 0.95
    intervention_scheduler_ent_coef: float = 0.01
    intervention_scheduler_init_forks: float = 2.0


class ManagerGoal(NamedTuple):
    """Per-decision goal record, stored only under `manager_credit:
    counterfactual` or `dpp` (the advantage model's goal input and the
    counterfactual goals' reference point). Leading dims
    `(n_windows, n_envs, n_agents)`."""

    pos: jax.Array  # (..., 2) each agent's goal_state position at the decision
    offset: jax.Array  # (..., 2) realized waypoint offset (w - s) / R


class Rollout(NamedTuple):
    """What `collect_fn` hands to `update_fn`.

    The inherited `MAPPO_JAX_Runner.train` loop passes this through untouched,
    which is how the waypoint diagnostics reach the stats file without changing
    that loop.
    """

    worker: object  # mappo_jax Transition, leading dim n_steps
    manager: object  # mappo_jax Transition, leading dim n_windows
    diagnostics: dict  # scalar rollout diagnostics, merged into the losses
    manager_goal: object = None  # ManagerGoal under counterfactual / dpp credit, else None
    intervention: object = None  # Intervention under `interventions: true`, else None
    # AdaptiveBatch under an adaptive or anytime `intervention_scheduler`, else None.
    adaptive: object = None
    # SchedulerBatch under `intervention_scheduler: anytime`, else None.
    scheduler: object = None


class AdaptiveBatch(NamedTuple):
    """What the adaptive fork collector adds to a `Rollout` (`adaptive_collection.py`).

    The main and fork buffers (`Rollout.worker` / `.manager` and
    `Rollout.intervention`) have a fixed number of round rows, `2 * n_windows`,
    and one column per main stream; rows that were never collected are masked
    placeholders. Exactly `n_windows * batch_envs` window slots are collected per
    update: `main_collected.sum() + fork_executed.sum()`.

    The anytime collector (`anytime_collection.py`) fills the first four fields
    the same way (a fork row there is one window of a fork, in the column of the
    stream it was cut from) and leaves the bandit's labels None.
    """

    main_collected: jax.Array  # (rounds, streams) bool: a main window was collected
    fork_executed: jax.Array  # (rounds, streams) bool: a fork ran from that window
    last_row: jax.Array  # (streams,) int: each stream's last collected round
    final_value: jax.Array  # (streams,) the manager's value of each stream's final state
    label_x: jax.Array = None  # (capacity, actor_dim) bandit inputs of the attempted forks
    label_y: jax.Array = None  # (capacity,) their targets (0 for a failed placement)
    label_count: jax.Array = None  # () int: labelled rows; the rest is padding
    # Where each label came from: the round and stream of its main window, its
    # focal agent, and whether its placement succeeded (its fork ran).
    label_round: jax.Array = None  # (capacity,) int
    label_stream: jax.Array = None  # (capacity,) int
    label_focal: jax.Array = None  # (capacity,) int
    label_placed: jax.Array = None  # (capacity,) bool
    # The bandit's prediction at the decision, before this update's fit (0 under
    # the random scheduler).
    label_q: jax.Array = None  # (capacity,)


class SchedulerBatch(NamedTuple):
    """The anytime scheduler's decisions of one rollout (`anytime_collection.py`),
    on the main buffers' grid `(rounds, streams, goal_horizon, ...)`: one cell
    per main step. `weight` is 1 on the cells where the scheduler decided and 0
    elsewhere (no decision, or an uncollected window). `advantage` / `returns`
    come from GAE over each stream's decisions in time order."""

    x: jax.Array  # (..., N, actor_dim) every agent's manager actor input
    pair: jax.Array  # (..., N, N, 2) pos_j - pos_i, goal_state units
    glob: jax.Array  # (..., G) budget, slots, episode time, step in window
    choice: jax.Array  # (...,) int: 0 skip, i + 1 a fork around agent i
    recruit: jax.Array  # (..., N) bool
    duration: jax.Array  # (...,) int: the fork's windows minus 1
    kmax: jax.Array  # (...,) int: the longest allowed fork, in windows
    log_prob: jax.Array  # (...,) at the decision
    value: jax.Array  # (...,) the scheduler's value at the decision
    reward: jax.Array  # (...,) the fork's paired gain at a placed launch, else 0
    advantage: jax.Array  # (...,)
    returns: jax.Array  # (...,)
    weight: jax.Array  # (...,) float 0/1


class Intervention(NamedTuple):
    """The training forks of one rollout (`interventions.py`), laid out as extra
    env COLUMNS with the main buffers' time length, so `update_fn` can append
    them to the main batch along the env axis. Column e holds env e's forks, one
    per forked window, each around that window's focal agent; a window skipped by
    `intervention_interval` holds fully masked placeholder rows.

    Under `intervention_scheduler: heuristic` the columns are packed instead
    (`intervention_heuristic.pack`): row w of column c holds one fork, whatever
    env and step it came from, recorded in `step` / `env`; unused rows are
    masked."""

    worker: object  # mappo_jax Transition, (n_steps, n_envs, ...)
    manager: object  # mappo_jax Transition, (n_windows, n_envs, ...)
    focal: jax.Array  # (n_windows, L) int — agent i of each fork, 0 on skipped windows
    n_recruits: jax.Array  # (n_windows, L) int — N, 0 on skipped windows
    valid: jax.Array  # (n_windows, L) bool — placement succeeded (False if skipped)
    # Heuristic only (None otherwise): each fork's main-rollout step (window *
    # goal_horizon + k) and main env.
    step: object = None
    env: object = None


class LastValues(NamedTuple):
    worker: jax.Array  # (n_envs, n_agents)
    manager: jax.Array  # (n_envs,)
