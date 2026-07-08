import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from algorithms.mappo_vanilla.networks.utils import layer_init, LOG_STD_MIN, LOG_STD_MAX


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
        input_dim = observation_dim

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
