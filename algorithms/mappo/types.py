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
    critic_type: str  # "mlp" | "multi_hgnn" | "hg_cross_attention"
    # Predefined hyperedge builders (names resolved via HYPEREDGE_FN_REGISTRY).
    # Determines the number of hyperedge types processed by HGNN critics when
    # hypergraph_mode="predefined". Ignored for hygma / learned_affinity
    # (always 1 type) and combined_affinities (one per source).
    hyperedge_fn_names: list[str] | None = None
    critic_seq_len: int = 32
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
    hypergraph_mode: str = (
        "predefined"  # "predefined" | "hygma" | "learned_affinity" | "combined_affinities"
    )
    hygma_history_len: int = 50
    hygma_clustering_interval: int = 100  # rollout steps
    hygma_min_clusters: int = 2
    hygma_max_clusters: int = 0  # 0 = auto (n_agents - 1)
    hygma_stability_threshold: float = 0.6
    affinity_loss_coef: float = 0.01
    # Combined affinities: load pre-trained AffinityTransformers from other experiments
    combined_affinity_sources: list[str] | None = (
        None  # batch names, e.g. ["contact_12a", "scatter_12a"]
    )
    combined_affinity_trial_id: str = "0"

    @property
    def n_hyperedge_types(self) -> int:
        """Number of hyperedge types the critic will process.

        Derived from `hypergraph_mode`:
          - "hygma" / "learned_affinity": always 1 (single spectral grouping).
          - "combined_affinities": one per entry in `combined_affinity_sources`.
          - "predefined": one per entry in `hyperedge_fn_names`.
        """
        if self.hypergraph_mode in ("hygma", "learned_affinity"):
            return 1
        if self.hypergraph_mode == "combined_affinities":
            return len(self.combined_affinity_sources or [])
        return len(self.hyperedge_fn_names or [])


@dataclass
class Experiment:
    device: str
    model_params: Model_Params
    params: MAPPO_Params
