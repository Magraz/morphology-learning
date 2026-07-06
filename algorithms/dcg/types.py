from dataclasses import dataclass, field


@dataclass
class DCG_Params:
    """Training hyperparameters for DCG (off-policy episodic Q-learning)."""

    # Total number of environment steps to train for.
    n_total_steps: int
    random_seeds: list

    # Replay buffer / sampling (measured in *episodes*).
    buffer_size: int = 32
    batch_size: int = 8

    # Optimisation (PyMARL uses RMSprop for the value learners).
    lr: float = 5e-4
    optim_alpha: float = 0.99
    optim_eps: float = 1e-5
    gamma: float = 0.99
    grad_norm_clip: float = 10.0
    double_q: bool = True

    # Target network refresh cadence (in episodes).
    target_update_interval: int = 200

    # Epsilon-greedy exploration schedule (linear decay over env steps).
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = 50000

    # Max steps stored per episode (buffer sizing). Must be >= the env horizon.
    episode_limit: int = 512

    # Evaluation / checkpoint / logging cadence (in env steps).
    test_interval: int = 20000
    test_nepisode: int = 20
    save_model_interval: int = 200000
    log_interval: int = 10000
    learner_log_interval: int = 10000


@dataclass
class DCG_Model_Params:
    """Network + coordination-graph architecture knobs."""

    rnn_hidden_dim: int = 64

    # Coordination-graph topology and message passing.
    cg_edges: str = "full"  # 'vdn'|'line'|'cycle'|'star'|'full'|int|list
    cg_utilities_hidden_dim: object = None  # None|int|list
    cg_payoffs_hidden_dim: object = None  # None|int|list
    cg_payoff_rank: object = None  # None => full-rank payoff matrices
    msg_iterations: int = 8
    msg_normalized: bool = True
    msg_anytime: bool = True

    # Duelling state-value bias (uses the global state).
    duelling: bool = False
    mixing_embed_dim: int = 32

    # Agent input augmentation.
    obs_last_action: bool = True
    obs_agent_id: bool = True

    # Optional value mixer (None keeps pure DCG; 'vdn'/'qmix' available).
    mixer: object = None


@dataclass
class Experiment:
    device: str
    params: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)
