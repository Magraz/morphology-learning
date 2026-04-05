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
    hidden_dim: int
    critic_type: str
    # For HGNN
    n_hyperedge_types: int = 1
    # Standalone episodic intrinsic reward
    use_intrinsic_reward: bool = False
    intrinsic_reward_mode: str = "agent"  # "team" | "agent"
    intrinsic_reward_coef: float = 1.0
    intrinsic_reward_use_encoder: bool = True
    intrinsic_reward_encoder_type: str = "local"  # "local" | "hypergraph"
    intrinsic_reward_encoder_dim: int = 64
    intrinsic_reward_k: int = 6
    intrinsic_reward_memory_capacity: int = 1025
    # Entropy conditioning of HGNN critics
    entropy_conditioning: bool = False
    # Auxiliary LSTM entropy predictor
    entropy_pred_seq_len: int = 32
    entropy_pred_coef: float = 0.01
    # HYGMA dynamic spectral clustering mode
    hypergraph_mode: str = "predefined"  # "predefined" | "hygma" | "learned_affinity"
    hygma_history_len: int = 50
    hygma_clustering_interval: int = 100  # rollout steps
    hygma_min_clusters: int = 2
    hygma_max_clusters: int = 0  # 0 = auto (n_agents - 1)
    hygma_stability_threshold: float = 0.6
    affinity_loss_coef: float = 0.01


@dataclass
class Experiment:
    device: str
    model_params: Model_Params
    params: MAPPO_Params
