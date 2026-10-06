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
    # `intervention_interval`-th manager decision each env is forked once per
    # agent i; in fork i, N ~ U{1..n_agents-1} teammates are teleported within
    # `intervention_radius` WORLD units of agent i and given i's waypoint offset.
    # Each fork is a one-window episode whose transitions join both levels' PPO
    # batches. Execution and evaluation are unchanged. False is the original
    # code path. Set all three in a model group, never on the CLI: the interval
    # and radius leave checkpoints shape-identical, so the model group name in
    # the results path is the only record of them.
    interventions: bool = False
    intervention_radius: float = 1.5
    intervention_interval: int = 1


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


class Intervention(NamedTuple):
    """The training forks of one rollout (`interventions.py`), laid out as extra
    env COLUMNS with the main buffers' time length, so `update_fn` can append
    them to the main batch along the env axis. Fork (window w, env e, focal
    agent i) is column `e * n_agents + i`; a window skipped by
    `intervention_interval` holds fully masked placeholder rows."""

    worker: object  # mappo_jax Transition, (n_steps, n_envs * n_agents, ...)
    manager: object  # mappo_jax Transition, (n_windows, n_envs * n_agents, ...)
    focal: jax.Array  # (n_windows, L) int — agent i of each fork lane
    n_recruits: jax.Array  # (n_windows, L) int — N, 0 on skipped windows
    valid: jax.Array  # (n_windows, L) bool — placement succeeded (False if skipped)


class LastValues(NamedTuple):
    worker: jax.Array  # (n_envs, n_agents)
    manager: jax.Array  # (n_envs,)
