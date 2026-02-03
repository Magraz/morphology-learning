import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # clamp for stability


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# To this:
class Hybrid_MLP_AC(nn.Module):
    """Actor-critic network that supports Dict action spaces with movement, link_openness, and detach components."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super(Hybrid_MLP_AC, self).__init__()

        # Shared feature extraction
        self.actor_layer1 = layer_init(nn.Linear(state_dim, hidden_dim))
        self.actor_layer2 = layer_init(nn.Linear(hidden_dim, hidden_dim))

        # Movement action (continuous 2D)
        movement_dim = 2
        self.movement_mean = layer_init(nn.Linear(hidden_dim, movement_dim), std=0.01)
        self.movement_log_std = nn.Parameter(
            torch.full((movement_dim,), -0.5, requires_grad=True)
        )

        # Attach action (discrete binary)
        self.attach_action_logits = layer_init(nn.Linear(hidden_dim, 1), std=0.01)

        # Detach action (discrete binary)
        self.detach_action_logits = layer_init(nn.Linear(hidden_dim, 1), std=0.01)

        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, state):
        # Shared feature extraction
        x = F.tanh(self.actor_layer1(state))
        x = F.tanh(self.actor_layer2(x))

        # Movement action distribution
        movement_mean = self.movement_mean(x)
        movement_log_std = self.movement_log_std.expand_as(movement_mean)

        # Link openness logits
        attach_action_logits = self.attach_action_logits(x)

        # Detach action distribution
        detach_action_logits = self.detach_action_logits(x)

        value = self.critic(state)

        return {
            "movement": (movement_mean, movement_log_std),
            "attach": attach_action_logits,
            "detach": detach_action_logits,
        }, value

    def act(self, state, deterministic=False):
        action_params, value = self.forward(state)

        # Movement action (continuous)
        movement_mean, movement_log_std = action_params["movement"]

        # Link openness (discrete binary)
        attach_logits = action_params["attach"]
        attach_probs = torch.sigmoid(attach_logits)  # Convert to probability

        # Detach action (discrete binary)
        detach_logits = action_params["detach"]
        detach_probs = torch.sigmoid(detach_logits)  # Convert to probability

        if deterministic:
            movement = movement_mean
            attach_action = (attach_probs > 0.5).int()
            detach_action = (detach_probs > 0.5).int()

        else:
            movement_std = torch.exp(movement_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
            movement = torch.normal(movement_mean, movement_std)
            attach_action = torch.bernoulli(attach_probs).int()
            detach_action = torch.bernoulli(detach_probs).int()

        action_dict = {
            "movement": movement,
            "attach": attach_action,
            "detach": detach_action,
        }

        # Calculate log probability
        log_prob, _ = self._get_log_prob_and_entropy(action_dict, action_params)

        action_tensor = torch.cat(list(action_dict.values()), dim=-1)

        return action_tensor, log_prob, value

    def evaluate(self, state, action):
        action_params, value = self.forward(state)
        action_dict = self._split_by_indices(
            action, list(action_params.keys()), [2, 3], dim=1
        )
        log_prob, entropy = self._get_log_prob_and_entropy(action_dict, action_params)
        return log_prob, value, entropy

    # TODO check how log prob is being calculated and simplify distribution calculation no need to sample from two different distributions for every action

    def _get_log_prob_and_entropy(self, action, action_params):
        """Calculate combined log probability for all action components"""
        # Movement log prob
        movement_mean, movement_log_std = action_params["movement"]
        movement_std = torch.exp(movement_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        movement_dist = torch.distributions.Normal(movement_mean, movement_std)
        movement_log_prob = movement_dist.log_prob(action["movement"]).sum(-1)
        movement_entropy = movement_dist.entropy().sum(-1)

        # Link openness log prob (binary)
        attach_logits = action_params["attach"]
        attach_dist = torch.distributions.Bernoulli(logits=attach_logits.squeeze(-1))
        attach_log_prob = attach_dist.log_prob(action["attach"].float().squeeze(-1))
        attach_entropy = attach_dist.entropy()

        # Detach log prob
        detach_logits = action_params["detach"]
        detach_dist = torch.distributions.Bernoulli(logits=detach_logits.squeeze(-1))
        detach_log_prob = detach_dist.log_prob(action["detach"].float().squeeze(-1))
        detach_entropy = detach_dist.entropy()

        # Combined log probability
        return (movement_log_prob + attach_log_prob + detach_log_prob), (
            movement_entropy + attach_entropy + detach_entropy
        )

    def _split_by_indices(self, t, keys, indices, dim=0):
        # indices = cumulative cut points (exclude endpoints), e.g., [2, 7]
        parts = torch.tensor_split(t, indices, dim=dim)
        return {k: p for k, p in zip(keys, parts)}
