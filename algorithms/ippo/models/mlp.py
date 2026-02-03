import torch
import torch.nn as nn
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(MLP, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy network)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 0.01)
            torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        shared_features = self.shared(state)

        # Actor outputs
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))

        # Critic output
        value = self.critic(shared_features)

        return action_mean, action_std, value

    def act(self, state):
        action_mean, action_std, value = self.forward(state)

        # Create normal distribution
        dist = Normal(action_mean, action_std)

        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Clamp action to [-1, 1]
        action = torch.tanh(action)

        return action, log_prob, value

    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)

        # Create distribution
        dist = Normal(action_mean, action_std)

        # Calculate log probability
        unbounded_action = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(unbounded_action).sum(dim=-1)

        # Calculate entropy
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value, entropy
