import dhg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from hypergraphs.hgnn_conv_layer import HGNNConv

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MAPPOActor(nn.Module):
    """Decentralized actor - each agent has its own or they share one"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        discrete: bool = False,
        entropy_pred_dim: int = 0,
    ):
        super(MAPPOActor, self).__init__()

        self.discrete = discrete
        self.action_dim = action_dim
        self.entropy_pred_dim = entropy_pred_dim
        input_dim = observation_dim + entropy_pred_dim

        if not discrete:
            # For continuous actions
            self.log_action_std = nn.Parameter(
                torch.full((action_dim,), -0.5, requires_grad=True)
            )

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def forward(self, obs):
        """Get action logits or means"""
        if self.discrete:
            return self.actor(obs)  # logits
        else:
            return self.actor(obs)  # means

    def get_action_dist(self, action_params):
        """Get action distribution"""
        if self.discrete:
            return Categorical(logits=action_params)
        else:
            log_std = self.log_action_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
            action_std = torch.exp(log_std)
            return Normal(action_params, action_std)

    def act(self, obs, deterministic=False, action_mask=None):
        """Sample action from policy"""
        action_params = self.forward(obs)
        if action_mask is not None and self.discrete:
            # Zero-out unavailable actions by pushing their logits to -inf
            action_params = action_params + (1.0 - action_mask) * (-1e9)
        dist = self.get_action_dist(action_params)

        if self.discrete:
            if deterministic:
                action = action_params.argmax(dim=-1, keepdim=True)
            else:
                action = dist.sample().unsqueeze(-1)

            logprob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)

        else:
            if deterministic:
                action = action_params
            else:
                action = dist.sample()

            logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        return action, logprob

    def evaluate(self, obs, action, action_mask=None):
        """Evaluate actions for training"""
        action_params = self.forward(obs)
        if action_mask is not None and self.discrete:
            action_params = action_params + (1.0 - action_mask) * (-1e9)
        dist = self.get_action_dist(action_params)

        if self.discrete:
            action_squeezed = action.squeeze(-1) if action.dim() > 1 else action
            logprob = dist.log_prob(action_squeezed).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
        else:
            logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
            entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)

        return logprob, entropy


class MAPPO_Hybrid_Actor(nn.Module):
    """Decentralized actor - each agent has its own or they share one"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super(MAPPO_Hybrid_Actor, self).__init__()

        self.action_dim = action_dim

        # Actor network
        self.actor_layer1 = layer_init(nn.Linear(observation_dim, hidden_dim))
        self.actor_layer2 = layer_init(nn.Linear(hidden_dim, hidden_dim))

        # Movement action (continuous 2D)
        self.movement_dim = 2
        self.movement_mean = layer_init(
            nn.Linear(hidden_dim, self.movement_dim), std=0.01
        )
        self.movement_log_std = nn.Parameter(
            torch.full((self.movement_dim,), -0.5, requires_grad=True)
        )

        # Attach action (discrete binary)
        self.attach_logits = layer_init(nn.Linear(hidden_dim, 1), std=0.01)

        # Detach action (discrete binary)
        self.detach_logits = layer_init(nn.Linear(hidden_dim, 1), std=0.01)

    def forward(self, state):
        """
        Forward pass through shared layers and all action heads

        Args:
            state: Observation tensor of shape (batch_size, observation_dim)

        Returns:
            Dictionary with action parameters for each component
        """
        # Shared feature extraction
        x = torch.tanh(self.actor_layer1(state))
        x = torch.tanh(self.actor_layer2(x))

        # Movement action parameters
        movement_mean = self.movement_mean(x)
        movement_log_std = self.movement_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        movement_log_std = movement_log_std.expand_as(movement_mean)

        # Discrete action logits
        attach_logits = self.attach_logits(x)
        detach_logits = self.detach_logits(x)

        return {
            "movement": (movement_mean, movement_log_std),
            "attach": attach_logits,
            "detach": detach_logits,
        }

    def act(self, state, deterministic=False):
        """
        Sample actions from policy

        Args:
            state: Observation tensor
            deterministic: If True, use mean/mode instead of sampling

        Returns:
            action_dict: Dictionary with 'movement', 'attach', 'detach' keys
            log_prob: Combined log probability (scalar per batch element)
        """
        action_params = self.forward(state)

        # Movement action (continuous)
        movement_mean, movement_log_std = action_params["movement"]
        movement_std = torch.exp(movement_log_std)

        if deterministic:
            movement = movement_mean
        else:
            movement_dist = Normal(movement_mean, movement_std)
            movement = movement_dist.sample()

        # Attach action (discrete binary)
        attach_logits = action_params["attach"]
        attach_probs = torch.sigmoid(attach_logits)

        if deterministic:
            attach = (attach_probs > 0.5).float()
        else:
            attach = torch.bernoulli(attach_probs)

        # Detach action (discrete binary)
        detach_logits = action_params["detach"]
        detach_probs = torch.sigmoid(detach_logits)

        if deterministic:
            detach = (detach_probs > 0.5).float()
        else:
            detach = torch.bernoulli(detach_probs)

        # Create action dictionary
        action_dict = {
            "movement": movement,
            "attach": attach,
            "detach": detach,
        }

        # Calculate combined log probability
        log_prob = self._compute_log_prob(action_dict, action_params)

        action_tensor = torch.cat(list(action_dict.values()), dim=-1)

        return action_tensor, log_prob

    def evaluate(self, state, action):
        """
        Evaluate actions for training

        Args:
            state: Observation tensor
            action: tensor with 'movement', 'attach', 'detach' actions

        Returns:
            log_prob: Combined log probability
            entropy: Combined entropy
        """
        action_params = self.forward(state)
        log_prob = self._compute_log_prob(
            self.tensor_to_action_dict(action), action_params
        )
        entropy = self._compute_entropy(action_params)

        return log_prob, entropy

    def _compute_log_prob(self, action_dict, action_params):
        """
        Compute combined log probability for all action components

        Args:
            action_dict: Dictionary with sampled actions
            action_params: Dictionary with action distribution parameters

        Returns:
            log_prob: Combined log probability of shape (batch_size, 1)
        """
        # Movement log probability
        movement_mean, movement_log_std = action_params["movement"]
        movement_std = torch.exp(movement_log_std)
        movement_dist = Normal(movement_mean, movement_std)
        movement_log_prob = movement_dist.log_prob(action_dict["movement"]).sum(
            dim=-1, keepdim=True
        )

        # Attach log probability (Bernoulli)
        attach_logits = action_params["attach"]
        attach_dist = torch.distributions.Bernoulli(logits=attach_logits)
        attach_log_prob = (
            attach_dist.log_prob(action_dict["attach"]).squeeze(-1).unsqueeze(-1)
        )

        # Detach log probability (Bernoulli)
        detach_logits = action_params["detach"]
        detach_dist = torch.distributions.Bernoulli(logits=detach_logits)
        detach_log_prob = (
            detach_dist.log_prob(action_dict["detach"]).squeeze(-1).unsqueeze(-1)
        )

        # Combined log probability (sum of independent log probs)
        total_log_prob = movement_log_prob + attach_log_prob + detach_log_prob

        return total_log_prob

    def _compute_entropy(self, action_params):
        """
        Compute combined entropy for exploration bonus

        Args:
            action_params: Dictionary with action distribution parameters

        Returns:
            entropy: Combined entropy of shape (batch_size, 1)
        """
        # Movement entropy
        movement_mean, movement_log_std = action_params["movement"]
        movement_std = torch.exp(movement_log_std)
        movement_dist = Normal(movement_mean, movement_std)
        movement_entropy = movement_dist.entropy().sum(dim=-1, keepdim=True)

        # Attach entropy
        attach_logits = action_params["attach"]
        attach_dist = torch.distributions.Bernoulli(logits=attach_logits)
        attach_entropy = attach_dist.entropy()

        # Detach entropy
        detach_logits = action_params["detach"]
        detach_dist = torch.distributions.Bernoulli(logits=detach_logits)
        detach_entropy = detach_dist.entropy()

        # Combined entropy
        total_entropy = movement_entropy + attach_entropy + detach_entropy

        return total_entropy

    def tensor_to_action_dict(self, action_tensor):
        """
        Convert concatenated action tensor back to dictionary

        Args:
            action_tensor: Tensor of shape (batch_size, action_dim)

        Returns:
            Dictionary with 'movement', 'attach', 'detach' keys
        """
        movement = action_tensor[..., : self.movement_dim]
        attach = action_tensor[..., self.movement_dim : self.movement_dim + 1]
        detach = action_tensor[..., self.movement_dim + 1 : self.movement_dim + 2]

        return {
            "movement": movement,
            "attach": attach,
            "detach": detach,
        }


class MAPPOCritic(nn.Module):
    """Centralized critic - observes global state"""

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int = 256,
    ):
        super(MAPPOCritic, self).__init__()

        # Larger network for centralized critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, global_state):
        """Get value estimate from global state"""
        return self.critic(global_state)


class HGNNCritic(nn.Module):
    """Centralized critic that uses hypergraph neural network convolution
    to aggregate information across agents before producing value estimates.

    Input X has shape (n_agents, observation_dim) — one feature vector per agent node.
    A dhg.Hypergraph encodes the relational structure between agents.

    When ``entropy_conditioning=True`` the first convolution layer accepts
    ``observation_dim + 1`` features so that a per-type structural entropy
    scalar can be broadcast to every agent node and concatenated with X.
    """

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 256,
        n_hgnn_layers: int = 2,
        drop_rate: float = 0.5,
        value_mode: str = "shared",
        entropy_conditioning: bool = True,
    ):
        super(HGNNCritic, self).__init__()

        assert value_mode in (
            "shared",
            "per_agent",
        ), f"value_mode must be 'shared' or 'per_agent', got '{value_mode}'"
        self.value_mode = value_mode
        self.entropy_conditioning = entropy_conditioning

        # First conv input dim is larger when entropy-conditioned
        input_dim = observation_dim + 1 if entropy_conditioning else observation_dim

        # HGNN convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(HGNNConv(input_dim, hidden_dim, drop_rate=drop_rate))
        for _ in range(n_hgnn_layers - 1):
            self.convs.append(HGNNConv(hidden_dim, hidden_dim, drop_rate=drop_rate))

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, X: torch.Tensor, hg: dhg.Hypergraph, entropy: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            X: Node feature matrix of shape (n_agents, observation_dim).
            hg: Hypergraph structure over the n_agents vertices.
            entropy: Optional scalar tensor of shape ``()`` or ``(1,)``.
                     When provided (and ``entropy_conditioning`` is True),
                     it is broadcast to every agent node and concatenated
                     with X before the first convolution.

        Returns:
            If value_mode == "shared":    shape (1,)
            If value_mode == "per_agent": shape (n_agents, 1)
        """
        if self.entropy_conditioning:
            n_agents = X.shape[0]
            if entropy is not None:
                ent_col = entropy.reshape(1, 1).expand(n_agents, 1)
            else:
                ent_col = torch.zeros(n_agents, 1, device=X.device)
            X = torch.cat([X, ent_col], dim=-1)

        return self._forward_conv(X, hg)

    def forward_unconditioned(
        self, X: torch.Tensor, hg: dhg.Hypergraph
    ) -> torch.Tensor:
        """Forward pass where X already has entropy concatenated (used by
        ``MultiHGNNCritic.forward_batched`` which handles concatenation
        externally for the block-diagonal case)."""
        return self._forward_conv(X, hg)

    def _forward_conv(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        for conv in self.convs:
            X = conv(X, hg)

        if self.value_mode == "shared":
            # Mean-pool over agent nodes, then project to scalar
            pooled = X.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            return self.value_head(pooled).squeeze(0)  # (1,)
        else:
            # Per-agent value estimate
            return self.value_head(X)  # (n_agents, 1)


class MultiHGNNCritic(nn.Module):
    """Combines multiple HGNNCritics (one per hyperedge type) with an
    additional type-feature input vector through a linear MLP.

    Each hyperedge type gets its own HGNNCritic that processes the same
    agent features X but with a different hypergraph structure. The scalar
    outputs from all critics and mapped to a final value estimate.

    Each critic is entropy-conditioned: the structural entropy of the
    corresponding hypergraph type is broadcast to every agent node and
    concatenated with X before the first HGNN convolution.
    """

    def __init__(
        self,
        n_hyperedge_types: int,
        n_agents: int,
        observation_dim: int,
        hidden_dim: int = 128,
        n_hgnn_layers: int = 2,
        drop_rate: float = 0.5,
        entropy_conditioning: bool = True,
    ):
        super(MultiHGNNCritic, self).__init__()

        self.n_hyperedge_types = n_hyperedge_types

        # One HGNNCritic per hyperedge type
        self.critics = nn.ModuleList(
            [
                HGNNCritic(
                    observation_dim,
                    hidden_dim=hidden_dim,
                    n_hgnn_layers=n_hgnn_layers,
                    drop_rate=drop_rate,
                    value_mode="per_agent",
                    entropy_conditioning=entropy_conditioning,
                )
                for _ in range(n_hyperedge_types)
            ]
        )

        self.mixer = nn.Sequential(
            layer_init(nn.Linear(n_hyperedge_types * n_agents, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1)),
        )

    def forward(
        self,
        X: torch.Tensor,
        hypergraphs: list,
        entropies: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            X: Node feature matrix of shape (n_agents, observation_dim).
            hypergraphs: List of dhg.Hypergraph, one per hyperedge type.
            entropies: Optional tensor of shape (n_types,) with structural
                       entropy for each hyperedge type.

        Returns:
            Scalar value estimate of shape (1,).
        """
        critic_values = torch.cat(
            [
                critic(
                    X, hg, entropy=entropies[i] if entropies is not None else None
                ).flatten()
                for i, (critic, hg) in enumerate(zip(self.critics, hypergraphs))
            ]
        )  # (n_hyperedge_types * n_agents,)

        return self.mixer(critic_values)  # (1,)

    def forward_batched(
        self,
        X: torch.Tensor,
        batched_hgs: list,
        n_graphs: int,
        entropies: torch.Tensor = None,
    ) -> torch.Tensor:
        """Batched forward pass over multiple environments/timesteps.

        Args:
            X: Node features of shape (n_graphs * n_agents, obs_dim).
            batched_hgs: List of block-diagonal dhg.Hypergraph (one per
                         hyperedge type), each with n_graphs * n_agents vertices.
            n_graphs: Number of graphs batched together.
            entropies: Optional tensor of shape (n_graphs, n_types) with
                       structural entropy per graph per type.

        Returns:
            Value estimates of shape (n_graphs, 1).
        """
        n_agents = X.shape[0] // n_graphs

        per_type_values = []
        for type_idx, (critic, hg) in enumerate(zip(self.critics, batched_hgs)):
            if critic.entropy_conditioning:
                # Build per-node entropy column for this type: (n_graphs * n_agents, 1)
                if entropies is not None:
                    ent_per_graph = entropies[:, type_idx]  # (n_graphs,)
                    ent_col = ent_per_graph.repeat_interleave(n_agents).unsqueeze(-1)
                else:
                    ent_col = torch.zeros(X.shape[0], 1, device=X.device)
                X_cond = torch.cat([X, ent_col], dim=-1)
                v = critic.forward_unconditioned(X_cond, hg)
            else:
                v = critic(X, hg)
            v = v.view(n_graphs, n_agents)  # (n_graphs, n_agents)
            per_type_values.append(v)

        # (n_graphs, n_types * n_agents)
        combined = torch.cat(per_type_values, dim=1)
        return self.mixer(combined)  # (n_graphs, 1)


class HypergraphEntropyPredictor(nn.Module):
    """Shared LSTM predictor of hypergraph structural entropy.

    A single LSTM with learnable agent embeddings replaces per-agent
    LSTMs, allowing the full mixed-agent batch to be processed in one
    forward pass.  Each prediction uses only one agent's observation
    sequence — the agent embedding encodes which agent is being
    predicted for.
    """

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        n_hyperedge_types: int,
        hidden_dim: int = 64,
        n_lstm_layers: int = 1,
        agent_embed_dim: int = 8,
    ):
        super().__init__()
        self.agent_embedding = nn.Embedding(n_agents, agent_embed_dim)
        self.lstm = nn.LSTM(
            observation_dim + agent_embed_dim,
            hidden_dim,
            n_lstm_layers,
            batch_first=True,
        )
        self.mean_head = layer_init(
            nn.Linear(hidden_dim, n_hyperedge_types), std=0.01
        )
        self.log_var_head = layer_init(
            nn.Linear(hidden_dim, n_hyperedge_types), std=0.01
        )

    def _forward_impl(
        self, obs_sequences: torch.Tensor, agent_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core forward: concat agent embedding to obs, run LSTM.

        Args:
            obs_sequences: (batch, seq_len, obs_dim)
            agent_ids:     (batch,) long tensor of agent indices.

        Returns:
            mean:    (batch, n_hyperedge_types)
            log_var: (batch, n_hyperedge_types)
        """
        embed = self.agent_embedding(agent_ids)  # (batch, agent_embed_dim)
        embed = embed.unsqueeze(1).expand(-1, obs_sequences.shape[1], -1)
        x = torch.cat([obs_sequences, embed], dim=-1)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]  # last layer: (batch, hidden)
        return self.mean_head(h), self.log_var_head(h)

    def forward(
        self, obs_sequences: torch.Tensor, agent_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict entropy for a single agent.

        Args:
            obs_sequences: (batch, seq_len, obs_dim)
            agent_idx:     Which agent to predict for (int).

        Returns:
            mean:    (batch, n_hyperedge_types)
            log_var: (batch, n_hyperedge_types)
        """
        agent_ids = torch.full(
            (obs_sequences.shape[0],),
            agent_idx,
            dtype=torch.long,
            device=obs_sequences.device,
        )
        return self._forward_impl(obs_sequences, agent_ids)

    def forward_batch(
        self,
        obs_sequences: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict entropy for a mixed-agent batch in a single LSTM call.

        Args:
            obs_sequences: (batch, seq_len, obs_dim)
            agent_indices: (batch,) int tensor of agent IDs.

        Returns:
            mean:    (batch, n_hyperedge_types)
            log_var: (batch, n_hyperedge_types)
        """
        return self._forward_impl(obs_sequences, agent_indices)


class MAPPONetwork(nn.Module):
    """Combined MAPPO network with shared/individual actors and centralized critic"""

    def __init__(
        self,
        observation_dim: int,
        global_state_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        discrete: bool = False,
        share_actor: bool = True,  # Whether to share actor parameters
        critic_type: str = "mlp",  # "mlp" or "multi_hgnn"
        n_hyperedge_types: int = 0,  # Required when critic_type="multi_hgnn"
        entropy_conditioning: bool = True,
    ):
        super(MAPPONetwork, self).__init__()

        self.n_agents = n_agents
        self.discrete = discrete
        self.share_actor = share_actor
        self.critic_type = critic_type
        # Extra actor input features from predicted entropy: [mean, log_var] per type
        self.entropy_pred_dim = 2 * n_hyperedge_types if n_hyperedge_types > 0 else 0

        if share_actor:
            # Single shared actor for all agents
            if self.discrete:
                self.actor = MAPPOActor(
                    observation_dim, action_dim, hidden_dim, self.discrete,
                    entropy_pred_dim=self.entropy_pred_dim,
                )
            else:
                self.actor = MAPPOActor(
                    observation_dim, action_dim, hidden_dim,
                    entropy_pred_dim=self.entropy_pred_dim,
                )

        else:
            # Separate actor for each agent
            if self.discrete:
                self.actors = nn.ModuleList(
                    [
                        MAPPOActor(
                            observation_dim, action_dim, hidden_dim, discrete,
                            entropy_pred_dim=self.entropy_pred_dim,
                        )
                        for _ in range(n_agents)
                    ]
                )
            else:
                self.actors = nn.ModuleList(
                    [
                        MAPPO_Hybrid_Actor(observation_dim, action_dim, hidden_dim)
                        for _ in range(n_agents)
                    ]
                )

        # Centralized critic (always shared)
        if critic_type == "multi_hgnn":
            assert (
                n_hyperedge_types > 0
            ), "n_hyperedge_types must be > 0 for multi_hgnn critic"
            self.critic = MultiHGNNCritic(
                n_hyperedge_types,
                n_agents,
                observation_dim,
                hidden_dim=hidden_dim * 2,
                entropy_conditioning=entropy_conditioning,
            )
        else:
            self.critic = MAPPOCritic(global_state_dim, hidden_dim * 2)

        # Auxiliary LSTM predictor of hypergraph structural entropy
        self.entropy_predictor = (
            HypergraphEntropyPredictor(
                n_agents, observation_dim, n_hyperedge_types, hidden_dim=64
            )
            if n_hyperedge_types > 0
            else None
        )

    def get_actor(self, agent_idx):
        """Get the actor for a specific agent"""
        if self.share_actor:
            return self.actor
        else:
            return self.actors[agent_idx]

    def act(self, obs, agent_idx, deterministic=False, action_mask=None):
        """Get action for a specific agent"""
        actor = self.get_actor(agent_idx)
        return actor.act(obs, deterministic, action_mask=action_mask)

    def evaluate_actions(
        self,
        obs,
        global_states,
        actions,
        agent_idx,
        action_mask=None,
        hypergraphs=None,
        entropies=None,
    ):
        """Evaluate actions for training

        Args:
            obs: Agent observations.
            global_states: Global state for MLP critic.
            actions: Actions to evaluate.
            agent_idx: Agent index for actor selection.
            action_mask: Optional action mask for discrete actions.
            hypergraphs: List of dhg.Hypergraph (required for multi_hgnn critic).
            entropies: Optional (n_types,) tensor for entropy conditioning.
        """
        actor = self.get_actor(agent_idx)
        log_probs, entropy = actor.evaluate(obs, actions, action_mask=action_mask)

        # Get values from centralized critic
        if self.critic_type == "multi_hgnn":
            values = self.critic(obs, hypergraphs, entropies=entropies).squeeze(-1)
        else:
            values = self.critic(global_states).squeeze(-1)

        return log_probs, values, entropy

    def get_value(self, global_state, hypergraphs=None, entropies=None):
        """Get value from centralized critic

        Args:
            global_state: Global state for MLP critic, or per-agent obs (n_agents, obs_dim) for multi_hgnn.
            hypergraphs: List of dhg.Hypergraph (required for multi_hgnn critic).
            entropies: Optional (n_types,) tensor for entropy conditioning.
        """
        if self.critic_type == "multi_hgnn":
            return self.critic(global_state, hypergraphs, entropies=entropies)
        return self.critic(global_state)

    def get_value_batched(self, obs_flat, batched_hgs, n_graphs, entropies=None):
        """Batched value estimation for multi_hgnn critic.

        Args:
            obs_flat: (n_graphs * n_agents, obs_dim) concatenated observations.
            batched_hgs: List of block-diagonal dhg.Hypergraph, one per type.
            n_graphs: Number of graphs batched together.
            entropies: Optional (n_graphs, n_types) tensor for entropy conditioning.

        Returns:
            Value estimates of shape (n_graphs, 1).
        """
        return self.critic.forward_batched(
            obs_flat, batched_hgs, n_graphs, entropies=entropies
        )
