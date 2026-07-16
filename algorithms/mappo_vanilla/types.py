from dataclasses import dataclass
from typing import Optional


@dataclass
class MAPPO_Params:
    # Training Params
    n_epochs: int
    n_total_steps: int
    n_minibatches: int
    # Rollout length per parallel env; the per-update batch is
    # n_steps * env.n_envs env-steps, so it scales with parallelism.
    n_steps: int
    parameter_sharing: bool
    random_seeds: list

    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    eps_clip: float = 0.2
    ent_coef: float = 0.01
    val_coef: float = 0.5
    grad_clip: float = 0.5


@dataclass
class Model_Params:
    hidden_dim: int


@dataclass
class Experiment:
    device: str
    model_params: Model_Params
    params: MAPPO_Params
