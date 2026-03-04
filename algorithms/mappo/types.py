from dataclasses import dataclass
from typing import Optional


@dataclass
class MAPPO_Params:
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
class Model_Params:
    model_name: str
    critic_type: str
    # For HGNN
    n_hyperedge_types: int = 0
    # Entropy conditioning of HGNN critics
    entropy_conditioning: bool = False
    # Auxiliary LSTM entropy predictor
    entropy_pred_seq_len: int = 32
    entropy_pred_coef: float = 0.01


@dataclass
class Experiment:
    device: str
    model_params: Model_Params
    params: MAPPO_Params
