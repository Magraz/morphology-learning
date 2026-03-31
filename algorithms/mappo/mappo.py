import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from algorithms.mappo.types import MAPPO_Params, Model_Params
from algorithms.mappo.networks.encoders import LocalStateEncoder
from algorithms.mappo.networks.models import MAPPONetwork
from algorithms.mappo.hg_cache import HypergraphCache
from algorithms.mappo.entropy_helpers import EntropyPredictorHelper


class MAPPOAgent:
    """Multi-Agent PPO with centralized critic"""

    def __init__(
        self,
        observation_dim: int,
        global_state_dim: int,
        action_dim: int,
        n_agents: int,
        params: MAPPO_Params,
        device: str,
        discrete: bool,
        n_parallel_envs: int,
        model_params: Model_Params,
        hyperedge_fns: list[tuple] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else device
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.global_state_dim = global_state_dim
        self.discrete = discrete
        self.share_actor = params.parameter_sharing
        self.n_parallel_envs = n_parallel_envs
        self.critic_type = model_params.critic_type
        self.n_hyperedge_types = model_params.n_hyperedge_types
        # Hyperedge builder specs: list of (fn, source) where source is "obs"
        # or an info-dict key name (e.g. "agents_2_objects")
        self.hyperedge_fns = hyperedge_fns or []
        # Entropy conditioning of HGNN critics
        self.entropy_conditioning = model_params.entropy_conditioning
        # Auxiliary LSTM entropy predictor config
        self.entropy_pred_seq_len = model_params.entropy_pred_seq_len
        self.entropy_pred_coef = model_params.entropy_pred_coef
        # Standalone intrinsic reward config
        self.use_intrinsic_reward = model_params.use_intrinsic_reward
        self.intrinsic_reward_encoder_dim = model_params.intrinsic_reward_encoder_dim
        self.intrinsic_reward_k = model_params.intrinsic_reward_k
        self.intrinsic_reward_memory_capacity = (
            model_params.intrinsic_reward_memory_capacity
        )
        self.intrinsic_reward_obs_dim = (
            self.n_agents * self.intrinsic_reward_encoder_dim
        )

        # PPO hyperparameters
        self.gamma = params.gamma
        self.gae_lambda = params.lmbda
        self.clip_epsilon = params.eps_clip
        self.entropy_coef = params.ent_coef
        self.value_coef = params.val_coef
        self.grad_clip = params.grad_clip

        # Create network
        self.network = MAPPONetwork(
            observation_dim=observation_dim,
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=(
                round(model_params.hidden_dim * 1.09)
                if self.critic_type == "mlp"
                else model_params.hidden_dim
            ),
            discrete=discrete,
            share_actor=self.share_actor,
            critic_type=self.critic_type,
            n_hyperedge_types=self.n_hyperedge_types,
            entropy_conditioning=self.entropy_conditioning,
        ).to(self.device)

        # Create old network for PPO
        self.network_old = MAPPONetwork(
            observation_dim=observation_dim,
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=(
                round(model_params.hidden_dim * 1.09)
                if self.critic_type == "mlp"
                else model_params.hidden_dim
            ),
            discrete=discrete,
            share_actor=self.share_actor,
            critic_type=self.critic_type,
            n_hyperedge_types=self.n_hyperedge_types,
            entropy_conditioning=self.entropy_conditioning,
        ).to(self.device)

        self.network_old.load_state_dict(self.network.state_dict())

        self.local_state_encoder = None
        if self.use_intrinsic_reward:
            self.local_state_encoder = LocalStateEncoder(
                observation_dim, self.intrinsic_reward_encoder_dim
            ).to(self.device)
            self.local_state_encoder.eval()

        # Optimizer for all parameters
        self.optimizer = optim.Adam(self.network.parameters(), lr=params.lr)

        # Hypergraph cache (owns signature maps, edge lists, dhg objects, entropy values)
        self.hg_cache = HypergraphCache(
            n_agents=n_agents,
            n_parallel_envs=n_parallel_envs,
        )

        # Entropy predictor helper (owns obs history, sequence building, loss)
        self.entropy_helper = EntropyPredictorHelper(
            n_parallel_envs=n_parallel_envs,
            n_agents=n_agents,
            observation_dim=observation_dim,
            seq_len=self.entropy_pred_seq_len,
        )

        # Buffers for each agent
        self.reset_buffers()

    def reset_buffers(self):
        """Reset experience buffers - separate for each parallel environment"""
        # Buffers indexed by [env_idx][agent_idx]
        self.observations = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.global_states = [[] for _ in range(self.n_parallel_envs)]
        self.actions = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.rewards = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.log_probs = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.values = [[] for _ in range(self.n_parallel_envs)]
        self.dones = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.action_masks = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        # Clear hypergraph caches and per-trajectory buffers
        self.hg_cache.reset()

    def reset_obs_history(self, env_mask=None):
        """Zero out obs history. If env_mask given, only reset those envs."""
        self.entropy_helper.reset_obs_history(env_mask)

    def encode_team_observations(self, obs_batch: np.ndarray) -> np.ndarray:
        """Encode and concatenate per-agent observations for each environment."""
        if self.local_state_encoder is None:
            raise RuntimeError(
                "Intrinsic reward encoder is not initialized for this agent."
            )

        with torch.no_grad():
            obs_tensor = torch.from_numpy(
                np.ascontiguousarray(obs_batch, dtype=np.float32)
            ).to(self.device)
            n_envs = obs_tensor.shape[0]
            obs_flat = obs_tensor.reshape(n_envs * self.n_agents, self.observation_dim)
            encoded = torch.stack(
                [
                    self.local_state_encoder.embedding(obs_flat[idx])
                    for idx in range(obs_flat.shape[0])
                ],
                dim=0,
            )
            encoded = encoded.reshape(
                n_envs, self.n_agents, self.intrinsic_reward_encoder_dim
            )
            return encoded.reshape(n_envs, self.intrinsic_reward_obs_dim).cpu().numpy()

    def get_actions(
        self, observations, global_state, deterministic=False, hypergraphs=None
    ):
        """
        Get actions for all agents

        Args:
            observations: List of observations, one per agent
            global_state: Concatenated global state (all agent observations)
            deterministic: Whether to use deterministic actions
            hypergraphs: list of dhg.Hypergraph (one per type) or None.
                         Required when critic_type="multi_hgnn".

        Returns:
            actions: List of actions for each agent
            log_probs: List of log probabilities
            value: Single value from centralized critic
        """
        with torch.no_grad():
            # Convert observations to tensors
            obs_tensors = [
                torch.FloatTensor(obs).to(self.device) for obs in observations
            ]

            # Convert global state to tensor
            global_state_tensor = torch.FloatTensor(global_state).to(self.device)

            # Get actions from each actor
            actions = []
            log_probs = []

            for agent_idx, obs_tensor in enumerate(obs_tensors):
                action, log_prob = self.network_old.act(
                    obs_tensor, agent_idx, deterministic
                )
                actions.append(action)
                log_probs.append(log_prob)

            # Get value from centralized critic
            if self.critic_type == "multi_hgnn":
                obs_full = torch.stack(obs_tensors)  # (n_agents, obs_dim)
                value = self.network_old.get_value(obs_full, hypergraphs=hypergraphs)
            else:
                value = self.network_old.get_value(global_state_tensor)

        return torch.stack(actions), torch.cat(log_probs), value

    def get_actions_batched(
        self,
        observations_batch,
        global_states_batch,
        deterministic=False,
        action_masks=None,
        hypergraphs=None,
        entropies=None,
    ):
        """
        Get actions for all agents in all environments in a single batched forward pass.

        Args:
            observations_batch: numpy (n_envs, n_agents, obs_dim)
            global_states_batch: numpy (n_envs, global_state_dim)
            deterministic: Whether to use deterministic actions
            action_masks: numpy (n_envs, n_agents, n_actions) or None
            hypergraphs: list[dhg.Hypergraph] of shape [n_hyperedge_types] or None.
                         Each is a block-diagonal batched hypergraph over n_envs.
                         Required when critic_type="multi_hgnn".
            entropies: Optional (n_envs, n_types) tensor of structural entropy
                       values for critic conditioning.

        Returns:
            actions:   tensor (n_envs, n_agents, action_dim) on self.device
            log_probs: tensor (n_envs, n_agents)             on self.device
            values:    tensor (n_envs, 1)                    on self.device
        """
        n_envs = observations_batch.shape[0]

        with torch.no_grad():
            obs_tensor = torch.from_numpy(
                np.ascontiguousarray(observations_batch, dtype=np.float32)
            ).to(
                self.device
            )  # (n_envs, n_agents, obs_dim)
            gs_tensor = torch.from_numpy(
                np.ascontiguousarray(global_states_batch, dtype=np.float32)
            ).to(
                self.device
            )  # (n_envs, global_state_dim)

            masks_tensor = (
                torch.from_numpy(
                    np.ascontiguousarray(action_masks, dtype=np.float32)
                ).to(self.device)
                if action_masks is not None
                else None
            )  # (n_envs, n_agents, n_actions) or None

            # Predict entropy conditioning for actor (if predictor exists)
            pred_entropy_flat = None  # (n_envs * n_agents, 2*n_types) or None
            if self.network_old.entropy_predictor is not None:
                if n_envs == self.n_parallel_envs:
                    # Training collection — use persistent rolling buffer
                    obs_seqs = self.entropy_helper.update_obs_history(obs_tensor)
                else:
                    # Eval/render — use obs as single-step sequence (zero-padded)
                    obs_seqs = self.entropy_helper.make_eval_sequences(obs_tensor)
                obs_seqs_flat = obs_seqs.reshape(
                    n_envs * self.n_agents, self.entropy_pred_seq_len, -1
                ).to(self.device)
                agent_ids = torch.arange(self.n_agents, device=self.device).repeat(
                    n_envs
                )
                pred_mean, pred_log_var = (
                    self.network_old.entropy_predictor.forward_batch(
                        obs_seqs_flat, agent_ids
                    )
                )
                pred_entropy_flat = torch.cat([pred_mean, pred_log_var], dim=-1)

            if self.share_actor:
                # Fuse envs and agents into one batch: single forward pass total
                obs_flat = obs_tensor.reshape(n_envs * self.n_agents, -1)
                if pred_entropy_flat is not None:
                    obs_flat = self.entropy_helper.condition_obs(
                        obs_flat, pred_entropy_flat
                    )
                masks_flat = (
                    masks_tensor.reshape(n_envs * self.n_agents, -1)
                    if masks_tensor is not None
                    else None
                )
                actions_flat, log_probs_flat = self.network_old.act(
                    obs_flat,
                    agent_idx=0,
                    deterministic=deterministic,
                    action_mask=masks_flat,
                )
                actions = actions_flat.reshape(n_envs, self.n_agents, -1)
                log_probs = log_probs_flat.reshape(n_envs, self.n_agents, -1).squeeze(
                    -1
                )
            else:
                # One batched pass per agent (n_agents passes, each over all envs)
                pred_entropy_per_agent = (
                    pred_entropy_flat.reshape(n_envs, self.n_agents, -1)
                    if pred_entropy_flat is not None
                    else None
                )
                actions_list = []
                log_probs_list = []
                for agent_idx in range(self.n_agents):
                    agent_obs = obs_tensor[:, agent_idx, :]  # (n_envs, obs_dim)
                    if pred_entropy_per_agent is not None:
                        agent_obs = self.entropy_helper.condition_obs(
                            agent_obs, pred_entropy_per_agent[:, agent_idx, :]
                        )
                    agent_mask = (
                        masks_tensor[:, agent_idx, :]
                        if masks_tensor is not None
                        else None
                    )
                    a, lp = self.network_old.act(
                        agent_obs, agent_idx, deterministic, action_mask=agent_mask
                    )
                    actions_list.append(a)  # (n_envs, action_dim)
                    log_probs_list.append(lp)  # (n_envs, 1)
                actions = torch.stack(
                    actions_list, dim=1
                )  # (n_envs, n_agents, action_dim)
                log_probs = torch.stack(log_probs_list, dim=1).squeeze(
                    -1
                )  # (n_envs, n_agents)

            # Centralized critic
            if self.critic_type == "multi_hgnn":
                obs_flat = obs_tensor.reshape(n_envs * self.n_agents, -1)
                values = self.network_old.get_value_batched(
                    obs_flat, hypergraphs, n_envs, entropies=entropies
                )  # (n_envs, 1)
            else:
                values = self.network_old.get_value(gs_tensor)  # (n_envs, 1)

        return actions, log_probs, values

    def store_transition(
        self,
        env_idx,
        observations,
        global_state,
        actions,
        rewards,
        log_probs,
        value,
        dones,
        action_masks=None,
    ):
        """Store transition for a specific environment (CPU buffers to avoid small GPU transfers)"""
        # Store global state (shared)
        self.global_states[env_idx].append(torch.FloatTensor(global_state))

        # Store value (shared)
        self.values[env_idx].append(torch.tensor(value, dtype=torch.float32))

        # Store per-agent data
        for agent_idx in range(self.n_agents):
            # Observation
            self.observations[env_idx][agent_idx].append(
                torch.FloatTensor(observations[agent_idx])
            )

            # Action
            self.actions[env_idx][agent_idx].append(
                torch.FloatTensor(actions[agent_idx])
            )

            # Reward, log_prob, done
            self.rewards[env_idx][agent_idx].append(
                torch.tensor(rewards[agent_idx], dtype=torch.float32)
            )
            self.log_probs[env_idx][agent_idx].append(
                torch.tensor(log_probs[agent_idx], dtype=torch.float32)
            )
            self.dones[env_idx][agent_idx].append(
                torch.tensor(dones[agent_idx], dtype=torch.float32)
            )
            if action_masks is not None:
                self.action_masks[env_idx][agent_idx].append(
                    torch.FloatTensor(action_masks[agent_idx])
                )

    def store_transitions_batch(
        self,
        obs,  # np (n_envs, n_agents, obs_dim)
        global_states,  # np (n_envs, gs_dim)
        actions,  # np (n_envs, n_agents) discrete or (n_envs, n_agents, action_dim)
        log_probs,  # np (n_envs, n_agents)
        values,  # np (n_envs, 1)
        rewards,  # np (n_envs,)
        dones,  # np (n_envs,)
        infos,  # dict
        action_masks=None,  # np (n_envs, n_agents, n_actions) or None
        hg_signature_ids=None,  # list[int] of length n_envs or None
        entropies=None,  # torch tensor (n_envs, n_types) or None
    ):
        """Store transitions for all environments in one vectorized call.

        Reduces tensor-creation overhead from n_envs * n_agents * 7 calls
        down to 7 calls (one per array), then indexes into the resulting tensors.
        """
        n_envs = obs.shape[0]

        # Per-agent rewards: (n_envs, n_agents)
        if "local_rewards" in infos:
            per_agent_rewards = infos["local_rewards"].astype(np.float32) + rewards[
                :, None
            ].astype(np.float32)
        else:
            per_agent_rewards = np.tile(
                rewards.astype(np.float32)[:, None], (1, self.n_agents)
            )

        # Ensure trailing action dim for storage: (n_envs, n_agents, 1) for discrete
        if self.discrete and actions.ndim == 2:
            actions_store = actions[:, :, None].astype(np.float32)
        else:
            actions_store = actions.astype(np.float32)

        # Expand dones: (n_envs,) -> (n_envs, n_agents)
        dones_expanded = np.tile(dones.astype(np.float32)[:, None], (1, self.n_agents))

        # Zero-copy numpy→tensor conversion (from_numpy shares memory when array
        # is already C-contiguous float32, which gym outputs typically are)
        obs_t = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
        gs_t = torch.from_numpy(np.ascontiguousarray(global_states, dtype=np.float32))
        act_t = torch.from_numpy(actions_store)  # float32 C-contiguous from .astype()
        lp_t = torch.from_numpy(np.ascontiguousarray(log_probs, dtype=np.float32))
        val_t = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        rew_t = torch.from_numpy(
            per_agent_rewards
        )  # float32 C-contiguous from arithmetic
        done_t = torch.from_numpy(dones_expanded)  # float32 C-contiguous from np.tile()
        masks_t = (
            torch.from_numpy(np.ascontiguousarray(action_masks, dtype=np.float32))
            if action_masks is not None
            else None
        )

        for env_idx in range(n_envs):
            self.global_states[env_idx].append(gs_t[env_idx])
            self.values[env_idx].append(val_t[env_idx])
            if hg_signature_ids is not None:
                self.hg_cache.store_transition(
                    env_idx=env_idx,
                    sig_id=hg_signature_ids[env_idx],
                    entropy_conditioning=self.entropy_conditioning,
                    entropy_tensor=entropies[env_idx] if entropies is not None else None,
                )
            for agent_idx in range(self.n_agents):
                self.observations[env_idx][agent_idx].append(obs_t[env_idx, agent_idx])
                self.actions[env_idx][agent_idx].append(act_t[env_idx, agent_idx])
                self.rewards[env_idx][agent_idx].append(rew_t[env_idx, agent_idx])
                self.log_probs[env_idx][agent_idx].append(lp_t[env_idx, agent_idx])
                self.dones[env_idx][agent_idx].append(done_t[env_idx, agent_idx])
                if masks_t is not None:
                    self.action_masks[env_idx][agent_idx].append(
                        masks_t[env_idx, agent_idx]
                    )

    def compute_returns_and_advantages(self, next_values):
        """
        Compute GAE returns and advantages, vectorized over all envs and agents.

        Replaces three nested loops (n_envs × n_agents × T Python iterations)
        with a single reversed loop over T, operating on (n_envs, n_agents) tensors.
        Output list is dense: index = env_idx * n_agents + agent_idx.
        """
        T = len(self.values[0])

        # --- (n_envs, T+1) critic values ---
        env_vals_list = []
        for env_idx in range(self.n_parallel_envs):
            nv = next_values[env_idx]
            if not torch.is_tensor(nv):
                nv = torch.tensor(nv, dtype=torch.float32)
            # self.values[env_idx]: list of T tensors of shape (1,)
            env_vals = torch.cat(self.values[env_idx])  # (T,)
            env_vals_list.append(torch.cat([env_vals, nv.flatten()[:1]]))  # (T+1,)
        values_t = torch.stack(env_vals_list)  # (n_envs, T+1)

        # --- (n_envs, n_agents, T) rewards and dones ---
        rewards_t = torch.stack(
            [
                torch.stack(
                    [torch.stack(self.rewards[e][a]) for a in range(self.n_agents)]
                )
                for e in range(self.n_parallel_envs)
            ]
        )  # (n_envs, n_agents, T)

        dones_t = torch.stack(
            [
                torch.stack(
                    [torch.stack(self.dones[e][a]) for a in range(self.n_agents)]
                )
                for e in range(self.n_parallel_envs)
            ]
        )  # (n_envs, n_agents, T)

        # --- Vectorized GAE: one reversed loop over T ---
        # Broadcast values over agent dim: (n_envs, 1, T+1)
        vals = values_t.unsqueeze(1)
        advantages = torch.zeros(self.n_parallel_envs, self.n_agents, T)
        gae = torch.zeros(self.n_parallel_envs, self.n_agents)

        for step in reversed(range(T)):
            not_done = 1.0 - dones_t[:, :, step]  # (n_envs, n_agents)
            delta = (
                rewards_t[:, :, step]
                + self.gamma * vals[:, :, step + 1] * not_done
                - vals[:, :, step]
            )
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[:, :, step] = gae

        returns = advantages + values_t[:, :-1].unsqueeze(1)  # (n_envs, n_agents, T)

        # Flatten to list indexed by env_idx * n_agents + agent_idx
        all_returns = []
        all_advantages = []
        for env_idx in range(self.n_parallel_envs):
            for agent_idx in range(self.n_agents):
                all_returns.append(returns[env_idx, agent_idx].detach())
                all_advantages.append(advantages[env_idx, agent_idx].detach())

        return all_returns, all_advantages

    def update_shared(
        self,
        all_advantages,
        all_returns,
        minibatch_size,
        epochs,
        train_device,
    ):

        # Training statistics
        stats = {
            "total_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "entropy_loss": 0,
            "entropy_pred_loss": 0,
        }
        num_updates = 0

        # Combine all agent data for shared actor update
        all_obs = []
        all_global_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns_combined = []
        all_advantages_combined = []
        all_masks = []
        all_ts_indices = []  # maps each sample to its (env, timestep) index
        all_raw_obs_for_seq = []  # (T, obs_dim) per (env, agent) — batched later
        all_entropy_targets = []  # (T, n_types) per (env, agent)
        all_agent_indices = []  # (T,) int per (env, agent)

        has_masks = len(self.action_masks[0][0]) > 0
        use_hgnn = self.critic_type == "multi_hgnn"
        use_entropy_pred = (
            self.network.entropy_predictor is not None
            and len(self.hg_cache.entropies[0]) > 0
        )

        if use_hgnn:
            ts_to_signature_ids = self.hg_cache.get_flat_signature_ids()
            ts_to_entropies = self.hg_cache.get_flat_entropies(
                train_device, self.entropy_conditioning
            )
            hgnn_batch_cache = {}

        # Iterate through each environment
        ts_offset = 0  # running offset for timestep indices
        for env_idx in range(self.n_parallel_envs):
            if len(self.values[env_idx]) == 0:
                continue  # Skip empty environments

            T = len(self.values[env_idx])  # timesteps for this env

            # For this environment, iterate through each agent
            for agent_idx in range(self.n_agents):
                if len(self.observations[env_idx][agent_idx]) == 0:
                    continue  # Skip if no data for this agent

                data_idx = env_idx * self.n_agents + agent_idx

                if data_idx >= len(all_advantages):
                    continue

                # Normalize advantages for this agent in this environment
                advantages = all_advantages[data_idx]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Stack data for this agent in this environment
                obs = torch.stack(self.observations[env_idx][agent_idx])
                actions = torch.stack(self.actions[env_idx][agent_idx])
                old_log_probs = torch.stack(self.log_probs[env_idx][agent_idx])
                returns = all_returns[data_idx]

                global_states = torch.stack(self.global_states[env_idx])

                all_obs.append(obs)
                all_global_states.append(global_states)
                all_actions.append(actions)
                all_old_log_probs.append(old_log_probs)
                all_returns_combined.append(returns)
                all_advantages_combined.append(advantages)

                if has_masks:
                    all_masks.append(torch.stack(self.action_masks[env_idx][agent_idx]))

                if use_hgnn:
                    all_ts_indices.append(torch.arange(ts_offset, ts_offset + T))

                if use_entropy_pred:
                    all_raw_obs_for_seq.append(obs)  # (T, obs_dim)
                    entropy_t = torch.tensor(
                        np.stack(self.hg_cache.entropies[env_idx]), dtype=torch.float32
                    )  # (T, n_types)
                    all_entropy_targets.append(entropy_t)
                    all_agent_indices.append(
                        torch.full((obs.shape[0],), agent_idx, dtype=torch.long)
                    )

            ts_offset += T

        # Concatenate all data and move to training device (may be CUDA)
        combined_obs = torch.cat(all_obs, dim=0).detach().to(train_device)
        combined_global_states = (
            torch.cat(all_global_states, dim=0).detach().to(train_device)
        )
        combined_actions = torch.cat(all_actions, dim=0).detach().to(train_device)
        combined_old_log_probs = (
            torch.cat(all_old_log_probs, dim=0).detach().to(train_device)
        )
        combined_returns = torch.cat(all_returns_combined, dim=0).to(train_device)
        combined_advantages = torch.cat(all_advantages_combined, dim=0).to(train_device)
        combined_masks = (
            torch.cat(all_masks, dim=0).detach().to(train_device) if has_masks else None
        )
        combined_ts_indices = (
            torch.cat(all_ts_indices).to(train_device) if use_hgnn else None
        )
        combined_obs_sequences = (
            self.entropy_helper.build_obs_sequences_batched(
                torch.stack(all_raw_obs_for_seq)
            )
            .detach()
            .to(train_device)
            if use_entropy_pred
            else None
        )
        combined_entropy_targets = (
            torch.cat(all_entropy_targets, dim=0).detach().to(train_device)
            if use_entropy_pred
            else None
        )
        combined_agent_indices = (
            torch.cat(all_agent_indices).to(train_device) if use_entropy_pred else None
        )

        # Create dataset
        dataset_tensors = [
            combined_obs,
            combined_global_states,
            combined_actions,
            combined_old_log_probs,
            combined_returns,
            combined_advantages,
        ]
        if combined_masks is not None:
            dataset_tensors.append(combined_masks)
        if combined_ts_indices is not None:
            dataset_tensors.append(combined_ts_indices)
        if combined_obs_sequences is not None:
            dataset_tensors.append(combined_obs_sequences)
            dataset_tensors.append(combined_entropy_targets)
            dataset_tensors.append(combined_agent_indices)

        dataset = TensorDataset(*dataset_tensors)

        dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

        # Train for multiple epochs
        for epoch in range(epochs):
            for batch in dataloader:
                # Unpack batch — order matches dataset_tensors above
                (
                    batch_obs,
                    batch_global_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_returns,
                    batch_advantages,
                    *extra,
                ) = batch
                batch_masks = extra.pop(0) if combined_masks is not None else None
                batch_ts_idx = extra.pop(0).long() if use_hgnn else None
                if use_entropy_pred:
                    batch_obs_seq = extra.pop(0)
                    batch_entropy_targets = extra.pop(0)
                    batch_agent_idx = extra.pop(0).long()

                # Predict entropy and condition actor observations
                pred_mean = pred_log_var = None
                if use_entropy_pred:
                    pred_mean, pred_log_var = (
                        self.network.entropy_predictor.forward_batch(
                            batch_obs_seq, batch_agent_idx
                        )
                    )
                    pred_entropy = torch.cat([pred_mean, pred_log_var], dim=-1)
                    batch_obs_conditioned = self.entropy_helper.condition_obs(
                        batch_obs, pred_entropy.detach()
                    )
                else:
                    batch_obs_conditioned = batch_obs

                # Actor evaluation (batched over individual agent observations)
                actor = self.network.get_actor(0)
                log_probs, entropy = actor.evaluate(
                    batch_obs_conditioned, batch_actions, action_mask=batch_masks
                )

                # Critic evaluation
                if use_hgnn:
                    values = self.hg_cache.compute_hgnn_critic_values(
                        network_critic=self.network.critic,
                        batch_global_states=batch_global_states,
                        batch_ts_indices=batch_ts_idx,
                        ts_to_signature_ids=ts_to_signature_ids,
                        hgnn_batch_cache=hgnn_batch_cache,
                        observation_dim=self.observation_dim,
                        device=self.device,
                        ts_to_entropies=ts_to_entropies,
                    )
                else:
                    values = self.network.critic(batch_global_states).squeeze(-1)

                # PPO objective
                ratio = torch.exp(log_probs.squeeze(-1) - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Auxiliary entropy prediction loss (Gaussian NLL)
                if use_entropy_pred:
                    entropy_pred_loss = self.entropy_helper.compute_prediction_loss(
                        pred_mean=pred_mean,
                        pred_log_var=pred_log_var,
                        targets=batch_entropy_targets,
                    )
                    loss = loss + self.entropy_pred_coef * entropy_pred_loss

                # Optimize
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.grad_clip
                )
                self.optimizer.step()

                # Update statistics
                stats["total_loss"] += loss.item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy_loss"] += entropy_loss.item()
                if use_entropy_pred:
                    stats["entropy_pred_loss"] += entropy_pred_loss.item()
                num_updates += 1

        return stats, num_updates

    def update_independent_actors(
        self,
        all_advantages,
        all_returns,
        minibatch_size,
        epochs,
        train_device,
    ):

        # Training statistics
        stats = {
            "total_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "entropy_loss": 0,
            "entropy_pred_loss": 0,
        }
        num_updates = 0

        has_masks = len(self.action_masks[0][0]) > 0
        use_hgnn = self.critic_type == "multi_hgnn"
        use_entropy_pred = (
            self.network.entropy_predictor is not None
            and len(self.hg_cache.entropies[0]) > 0
        )

        if use_hgnn:
            ts_to_signature_ids = self.hg_cache.get_flat_signature_ids()
            ts_to_entropies = self.hg_cache.get_flat_entropies(
                train_device, self.entropy_conditioning
            )
            hgnn_batch_cache = {}

        # Update each agent separately (independent actors)
        for agent_idx in range(self.n_agents):
            # Collect data for this agent across all environments
            agent_obs = []
            agent_global_states = []
            agent_actions = []
            agent_old_log_probs = []
            agent_returns = []
            agent_advantages = []
            agent_masks = []
            agent_ts_indices = []
            agent_raw_obs_for_seq = []
            agent_entropy_targets = []

            ts_offset = 0
            for env_idx in range(self.n_parallel_envs):
                T = len(self.values[env_idx]) if len(self.values[env_idx]) > 0 else 0

                if len(self.observations[env_idx][agent_idx]) == 0:
                    ts_offset += T
                    continue

                # Get the index in the flattened advantages/returns list
                data_idx = env_idx * self.n_agents + agent_idx

                if data_idx >= len(all_advantages):
                    ts_offset += T
                    continue

                # Normalize advantages for this agent in this environment
                advantages = all_advantages[data_idx]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Stack data
                obs = torch.stack(self.observations[env_idx][agent_idx])
                actions = torch.stack(self.actions[env_idx][agent_idx])
                old_log_probs = torch.stack(self.log_probs[env_idx][agent_idx])
                returns = all_returns[data_idx]
                global_states = torch.stack(self.global_states[env_idx])

                # Append to agent-specific lists
                agent_obs.append(obs)
                agent_global_states.append(global_states)
                agent_actions.append(actions)
                agent_old_log_probs.append(old_log_probs)
                agent_returns.append(returns)
                agent_advantages.append(advantages)

                if has_masks:
                    agent_masks.append(
                        torch.stack(self.action_masks[env_idx][agent_idx])
                    )

                if use_hgnn:
                    agent_ts_indices.append(torch.arange(ts_offset, ts_offset + T))

                if use_entropy_pred:
                    agent_raw_obs_for_seq.append(obs)  # (T, obs_dim)
                    entropy_t = torch.tensor(
                        np.stack(self.hg_cache.entropies[env_idx]), dtype=torch.float32
                    )  # (T, n_types)
                    agent_entropy_targets.append(entropy_t)

                ts_offset += T

            if len(agent_obs) == 0:
                continue  # No data for this agent

            # Concatenate data for this agent from all environments and move to training device
            obs_combined = torch.cat(agent_obs, dim=0).detach().to(train_device)
            global_states_combined = (
                torch.cat(agent_global_states, dim=0).detach().to(train_device)
            )
            actions_combined = torch.cat(agent_actions, dim=0).detach().to(train_device)
            old_log_probs_combined = (
                torch.cat(agent_old_log_probs, dim=0).detach().to(train_device)
            )
            returns_combined = torch.cat(agent_returns, dim=0).to(train_device)
            advantages_combined = torch.cat(agent_advantages, dim=0).to(train_device)
            masks_combined = (
                torch.cat(agent_masks, dim=0).detach().to(train_device)
                if has_masks
                else None
            )
            ts_indices_combined = (
                torch.cat(agent_ts_indices).to(train_device) if use_hgnn else None
            )
            obs_sequences_combined = (
                self.entropy_helper.build_obs_sequences_batched(
                    torch.stack(agent_raw_obs_for_seq)
                )
                .detach()
                .to(train_device)
                if use_entropy_pred
                else None
            )
            entropy_targets_combined = (
                torch.cat(agent_entropy_targets, dim=0).detach().to(train_device)
                if use_entropy_pred
                else None
            )

            # Create dataset for this agent
            dataset_tensors = [
                obs_combined,
                global_states_combined,
                actions_combined,
                old_log_probs_combined,
                returns_combined,
                advantages_combined,
            ]
            if masks_combined is not None:
                dataset_tensors.append(masks_combined)
            if ts_indices_combined is not None:
                dataset_tensors.append(ts_indices_combined)
            if obs_sequences_combined is not None:
                dataset_tensors.append(obs_sequences_combined)
                dataset_tensors.append(entropy_targets_combined)

            dataset = TensorDataset(*dataset_tensors)
            dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

            # Train for multiple epochs
            for epoch in range(epochs):
                for batch in dataloader:
                    # Unpack batch — order matches dataset_tensors above
                    (
                        batch_obs,
                        batch_global_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_returns,
                        batch_advantages,
                        *extra,
                    ) = batch
                    batch_masks = extra.pop(0) if masks_combined is not None else None
                    batch_ts_idx = extra.pop(0).long() if use_hgnn else None
                    if use_entropy_pred:
                        batch_obs_seq = extra.pop(0)
                        batch_entropy_targets = extra.pop(0)

                    # Predict entropy and condition actor observations
                    pred_mean = pred_log_var = None
                    if use_entropy_pred:
                        pred_mean, pred_log_var = self.network.entropy_predictor(
                            batch_obs_seq, agent_idx
                        )
                        pred_entropy = torch.cat([pred_mean, pred_log_var], dim=-1)
                        batch_obs_conditioned = self.entropy_helper.condition_obs(
                            batch_obs, pred_entropy.detach()
                        )
                    else:
                        batch_obs_conditioned = batch_obs

                    # Actor evaluation
                    actor = self.network.get_actor(agent_idx)
                    log_probs, entropy = actor.evaluate(
                        batch_obs_conditioned, batch_actions, action_mask=batch_masks
                    )

                    # Critic evaluation
                    if use_hgnn:
                        values = self.hg_cache.compute_hgnn_critic_values(
                            network_critic=self.network.critic,
                            batch_global_states=batch_global_states,
                            batch_ts_indices=batch_ts_idx,
                            ts_to_signature_ids=ts_to_signature_ids,
                            hgnn_batch_cache=hgnn_batch_cache,
                            observation_dim=self.observation_dim,
                            device=self.device,
                            ts_to_entropies=ts_to_entropies,
                        )
                    else:
                        values = self.network.critic(batch_global_states).squeeze(-1)

                    # PPO objective
                    ratio = torch.exp(log_probs.squeeze(-1) - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        * batch_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(values, batch_returns)

                    # Entropy loss
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss
                    )

                    # Auxiliary entropy prediction loss (Gaussian NLL)
                    if use_entropy_pred:
                        entropy_pred_loss = self.entropy_helper.compute_prediction_loss(
                            pred_mean=pred_mean,
                            pred_log_var=pred_log_var,
                            targets=batch_entropy_targets,
                        )
                        loss = loss + self.entropy_pred_coef * entropy_pred_loss

                    # Optimize
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.grad_clip
                    )
                    self.optimizer.step()

                    # Update statistics
                    stats["total_loss"] += loss.item()
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy_loss"] += entropy_loss.item()
                    if use_entropy_pred:
                        stats["entropy_pred_loss"] += entropy_pred_loss.item()
                    num_updates += 1

        return stats, num_updates

    def update(self, next_value=0, minibatch_size=128, epochs=10):
        """Update all agents using shared critic"""

        # Compute returns and advantages
        all_returns, all_advantages = self.compute_returns_and_advantages(next_value)

        # Update each agent (or all at once if sharing actor)
        if self.share_actor:
            stats, num_updates = self.update_shared(
                all_advantages, all_returns, minibatch_size, epochs, self.device
            )

        else:
            stats, num_updates = self.update_independent_actors(
                all_advantages, all_returns, minibatch_size, epochs, self.device
            )

        # Update old network
        self.network_old.load_state_dict(self.network.state_dict())

        # Reset buffers
        self.reset_buffers()

        # Average statistics
        for key in stats:
            stats[key] /= max(1, num_updates)

        return stats
