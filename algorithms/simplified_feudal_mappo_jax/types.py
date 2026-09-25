from dataclasses import dataclass
from typing import NamedTuple

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
    is shared. The manager's discount is DERIVED as `gamma ** goal_horizon`, so
    both levels see the same effective horizon in env steps.
    """

    manager_lr: float = 3e-4
    manager_n_epochs: int = 4
    manager_n_minibatches: int = 2


@dataclass
class Model_Params:
    hidden_dim: int
    # Both REQUIRED (no default): `model=mlp` must fail here rather than train
    # into experiments/results/<env>/mlp/, the mappo_jax baseline's directory.
    goal_horizon: int  # c: steps a waypoint stays in force
    waypoint_radius: float  # R: max waypoint offset per axis, in goal_state units


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
