from dataclasses import dataclass
from typing import Optional


@dataclass
class Params:
    # Training Params
    n_epochs: int
    n_total_steps: int
    n_minibatches: int
    batch_size: int
    parameter_sharing: bool
    random_seeds: list

    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    eps_clip: float = 0.2
    ent_coef: float = 0.01
    val_coef: float = 0.5
    std_coef: float = 0.0
    grad_clip: float = 0.5


@dataclass
class Experiment:
    device: str
    model: str
    params: Params
