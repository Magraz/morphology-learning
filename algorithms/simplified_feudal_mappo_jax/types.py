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
    # "global" | "relative": what the manager ACTOR reads
    # (waypoints.MANAGER_INPUTS). "global" is the original behaviour.
    manager_input: str = "global"


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


class Rollout(NamedTuple):
    """What `collect_fn` hands to `update_fn`.

    The inherited `MAPPO_JAX_Runner.train` loop passes this through untouched,
    which is how the waypoint diagnostics reach the stats file without changing
    that loop.
    """

    worker: object  # mappo_jax Transition, leading dim n_steps
    manager: object  # mappo_jax Transition, leading dim n_windows
    diagnostics: dict  # scalar rollout diagnostics, merged into the losses


class LastValues(NamedTuple):
    worker: jax.Array  # (n_envs, n_agents)
    manager: jax.Array  # (n_envs,)
