from dataclasses import dataclass


@dataclass
class Params:

    # Training Params
    n_epochs: int
    n_total_steps: int
    n_minibatches: int
    batch_size: int
    parameter_sharing: bool
    random_seeds: list

    # PPO Params
    eps_clip: float
    gamma: float
    lr: float
    grad_clip: float
    ent_coef: float
    val_coef: float
    std_coef: float
    lmbda: float


@dataclass
class Experiment:
    device: str = ""
    model: str = ""
    params: Params = None
