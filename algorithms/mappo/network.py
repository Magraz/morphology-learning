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
    ):
        super(MAPPOActor, self).__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        if not discrete:
            # For continuous actions
            self.log_action_std = nn.Parameter(
                torch.full((action_dim,), -0.5, requires_grad=True)
            )

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, hidden_dim)),
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
    """

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 256,
        n_hgnn_layers: int = 2,
        drop_rate: float = 0.5,
        value_mode: str = "shared",
    ):
        super(HGNNCritic, self).__init__()

        assert value_mode in (
            "shared",
            "per_agent",
        ), f"value_mode must be 'shared' or 'per_agent', got '{value_mode}'"
        self.value_mode = value_mode

        # HGNN convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(HGNNConv(observation_dim, hidden_dim, drop_rate=drop_rate))
        for _ in range(n_hgnn_layers - 1):
            self.convs.append(HGNNConv(hidden_dim, hidden_dim, drop_rate=drop_rate))

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        """
        Args:
            X: Node feature matrix of shape (n_agents, observation_dim).
            hg: Hypergraph structure over the n_agents vertices.

        Returns:
            If value_mode == "shared":    shape (1,)
            If value_mode == "per_agent": shape (n_agents, 1)
        """
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
    """

    def __init__(
        self,
        n_hyperedge_types: int,
        n_agents: int,
        observation_dim: int,
        hidden_dim: int = 128,
        n_hgnn_layers: int = 2,
        drop_rate: float = 0.5,
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
                )
                for _ in range(n_hyperedge_types)
            ]
        )

        # Combination layer: critic values + processed type features -> scalar
        self.mixer = nn.Linear(n_hyperedge_types * n_agents, 1)

        # Larger network for centralized critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_hyperedge_types * n_agents, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1)),
        )

    def forward(
        self,
        X: torch.Tensor,
        hypergraphs: list,
    ) -> torch.Tensor:
        """
        Args:
            X: Node feature matrix of shape (n_agents, observation_dim).
            hypergraphs: List of dhg.Hypergraph, one per hyperedge type.

        Returns:
            Scalar value estimate of shape (1,).
        """
        # Get a value from each type-specific critic and flatten
        critic_values = torch.cat(
            [critic(X, hg).flatten() for critic, hg in zip(self.critics, hypergraphs)]
        )  # (n_hyperedge_types * n_agents,)

        return self.mixer(critic_values)  # (1,)

    def forward_batched(
        self,
        X: torch.Tensor,
        batched_hgs: list,
        n_graphs: int,
    ) -> torch.Tensor:
        """Batched forward pass over multiple environments/timesteps.

        Args:
            X: Node features of shape (n_graphs * n_agents, obs_dim).
            batched_hgs: List of block-diagonal dhg.Hypergraph (one per
                         hyperedge type), each with n_graphs * n_agents vertices.
            n_graphs: Number of graphs batched together.

        Returns:
            Value estimates of shape (n_graphs, 1).
        """
        n_agents = X.shape[0] // n_graphs

        per_type_values = []
        for critic, hg in zip(self.critics, batched_hgs):
            # Single forward pass through block-diagonal hypergraph
            v = critic(X, hg)  # (n_graphs * n_agents, 1)
            v = v.view(n_graphs, n_agents)  # (n_graphs, n_agents)
            per_type_values.append(v)

        # (n_graphs, n_types * n_agents)
        combined = torch.cat(per_type_values, dim=1)
        return self.mixer(combined)  # (n_graphs, 1)


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
    ):
        super(MAPPONetwork, self).__init__()

        self.n_agents = n_agents
        self.discrete = discrete
        self.share_actor = share_actor
        self.critic_type = critic_type

        if share_actor:
            # Single shared actor for all agents
            if self.discrete:
                self.actor = MAPPOActor(
                    observation_dim, action_dim, hidden_dim, self.discrete
                )
            else:
                # self.actor = MAPPO_Hybrid_Actor(observation_dim, action_dim, hidden_dim)
                self.actor = MAPPOActor(observation_dim, action_dim, hidden_dim)

        else:
            # Separate actor for each agent
            if self.discrete:
                self.actors = nn.ModuleList(
                    [
                        MAPPOActor(observation_dim, action_dim, hidden_dim, discrete)
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
            )
        else:
            self.critic = MAPPOCritic(global_state_dim, hidden_dim * 2)

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
    ):
        """Evaluate actions for training

        Args:
            obs: Agent observations.
            global_states: Global state for MLP critic.
            actions: Actions to evaluate.
            agent_idx: Agent index for actor selection.
            action_mask: Optional action mask for discrete actions.
            hypergraphs: List of dhg.Hypergraph (required for multi_hgnn critic).
        """
        actor = self.get_actor(agent_idx)
        log_probs, entropy = actor.evaluate(obs, actions, action_mask=action_mask)

        # Get values from centralized critic
        if self.critic_type == "multi_hgnn":
            values = self.critic(obs, hypergraphs).squeeze(-1)
        else:
            values = self.critic(global_states).squeeze(-1)

        return log_probs, values, entropy

    def get_value(self, global_state, hypergraphs=None):
        """Get value from centralized critic

        Args:
            global_state: Global state for MLP critic, or per-agent obs (n_agents, obs_dim) for multi_hgnn.
            hypergraphs: List of dhg.Hypergraph (required for multi_hgnn critic).
        """
        if self.critic_type == "multi_hgnn":
            return self.critic(global_state, hypergraphs)
        return self.critic(global_state)

    def get_value_batched(self, obs_flat, batched_hgs, n_graphs):
        """Batched value estimation for multi_hgnn critic.

        Args:
            obs_flat: (n_graphs * n_agents, obs_dim) concatenated observations.
            batched_hgs: List of block-diagonal dhg.Hypergraph, one per type.
            n_graphs: Number of graphs batched together.

        Returns:
            Value estimates of shape (n_graphs, 1).
        """
        return self.critic.forward_batched(obs_flat, batched_hgs, n_graphs)
