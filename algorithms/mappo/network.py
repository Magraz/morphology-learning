import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

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
        self.movement_dim = 2

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

    def act(self, obs, deterministic=False):
        """Sample action from policy"""
        action_params = self.forward(obs)
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

    def evaluate(self, obs, action):
        """Evaluate actions for training"""
        action_params = self.forward(obs)
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
    ):
        super(MAPPONetwork, self).__init__()

        self.n_agents = n_agents
        self.discrete = discrete
        self.share_actor = share_actor

        if share_actor:
            # Single shared actor for all agents
            if self.discrete:
                self.actor = MAPPOActor(
                    observation_dim, action_dim, hidden_dim, self.discrete
                )
            else:
                self.actor = MAPPO_Hybrid_Actor(observation_dim, action_dim, hidden_dim)

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
        self.critic = MAPPOCritic(global_state_dim, hidden_dim * 2)

    def get_actor(self, agent_idx):
        """Get the actor for a specific agent"""
        if self.share_actor:
            return self.actor
        else:
            return self.actors[agent_idx]

    def act(self, obs, agent_idx, deterministic=False):
        """Get action for a specific agent"""
        actor = self.get_actor(agent_idx)
        return actor.act(obs, deterministic)

    def evaluate_actions(self, obs, global_states, actions, agent_idx):
        """Evaluate actions for training"""
        # Get log probs and entropy from actor
        actor = self.get_actor(agent_idx)
        log_probs, entropy = actor.evaluate(obs, actions)

        # Get values from centralized critic
        values = self.critic(global_states).squeeze(-1)

        return log_probs, values, entropy

    def get_value(self, global_state):
        """Get value from centralized critic"""
        return self.critic(global_state)
