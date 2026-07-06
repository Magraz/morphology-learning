"""Translate the framework's dataclasses into the flat ``args`` namespace that
the vendored PyMARL DCG modules read.

This is the single point of contact between the framework config
(``DCG_Params`` / ``DCG_Model_Params`` + env-derived dims) and PyMARL's
attribute-bag convention. Every attribute any vendored module touches
(controller, learner, agent, action selector) is set here.
"""

from types import SimpleNamespace

from algorithms.dcg.types import DCG_Model_Params, DCG_Params


def build_args(
    params: DCG_Params,
    model_params: DCG_Model_Params,
    *,
    n_agents: int,
    n_actions: int,
    obs_dim: int,
    state_dim: int,
    device: str,
    batch_size_run: int,
) -> SimpleNamespace:
    args = SimpleNamespace()

    # --- Component selection (kept fixed for the DCG stack) ---
    args.agent = "rnn_feat"
    args.agent_output_type = "q"
    args.action_selector = "epsilon_greedy"
    args.mac = "dcg_mac"
    args.learner = "dcg_learner"
    args.mixer = model_params.mixer

    # --- Env-derived dimensions ---
    args.n_agents = n_agents
    args.n_actions = n_actions
    args.obs_shape = obs_dim
    args.state_shape = state_dim

    # --- Agent / coordination-graph architecture ---
    args.rnn_hidden_dim = model_params.rnn_hidden_dim
    args.cg_edges = model_params.cg_edges
    args.cg_utilities_hidden_dim = model_params.cg_utilities_hidden_dim
    args.cg_payoffs_hidden_dim = model_params.cg_payoffs_hidden_dim
    args.cg_payoff_rank = model_params.cg_payoff_rank
    args.msg_iterations = model_params.msg_iterations
    args.msg_normalized = model_params.msg_normalized
    args.msg_anytime = model_params.msg_anytime
    args.duelling = model_params.duelling
    args.mixing_embed_dim = model_params.mixing_embed_dim
    args.obs_last_action = model_params.obs_last_action
    args.obs_agent_id = model_params.obs_agent_id

    # --- Optimisation ---
    args.lr = params.lr
    args.optim_alpha = params.optim_alpha
    args.optim_eps = params.optim_eps
    args.gamma = params.gamma
    args.grad_norm_clip = params.grad_norm_clip
    args.double_q = params.double_q
    args.target_update_interval = params.target_update_interval

    # --- Exploration schedule ---
    args.epsilon_start = params.epsilon_start
    args.epsilon_finish = params.epsilon_finish
    args.epsilon_anneal_time = params.epsilon_anneal_time
    args.test_greedy = True

    # --- Logging / runtime ---
    args.learner_log_interval = params.learner_log_interval
    args.batch_size_run = batch_size_run
    args.device = device
    args.use_cuda = str(device).startswith("cuda")

    return args
