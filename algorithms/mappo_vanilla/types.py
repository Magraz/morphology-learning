from dataclasses import dataclass
from typing import Optional


@dataclass
class MAPPO_Params:
    # Training Params
    n_epochs: int
    n_total_steps: int
    n_minibatches: int
    # Per-update batch, in TOTAL env-steps (summed over the parallel envs).
    # RolloutCollector.collect gathers this many steps regardless of how many
    # envs run in parallel, so the batch — and hence num_updates — is fixed by
    # config and independent of env.n_envs / the core count.
    batch_size: int
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
