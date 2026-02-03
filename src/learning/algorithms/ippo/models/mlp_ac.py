import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # clamp for stability


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP_AC(nn.Module):
    def __init__(
        self,
        observation_space: int,
        action_dim: int,
        hidden_dim: int = 128,
        discrete: bool = False,  # New parameter
    ):
        super(MLP_AC, self).__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        if not discrete:
            # For continuous actions, we need learnable std
            self.log_action_std = nn.Parameter(
                torch.full((action_dim,), -0.5, requires_grad=True)
            )

        # Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_space, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_space, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, state):
        """Forward pass through actor and critic"""
        if self.discrete:
            # For discrete actions, output logits
            action_logits = self.actor(state)
            value = self.critic(state)
            return value, action_logits
        else:
            # For continuous actions, output mean
            action_mean = self.actor(state)
            value = self.critic(state)
            return value, action_mean

    def get_value(self, state: torch.Tensor):
        """Get value estimate from critic"""
        return self.critic(state)

    def get_action_dist(self, action_params):
        """Get action distribution (Categorical for discrete, Normal for continuous)"""
        if self.discrete:
            # action_params are logits for discrete actions
            return Categorical(logits=action_params)
        else:
            # action_params are means for continuous actions
            log_std = self.log_action_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
            action_std = torch.exp(log_std)
            return Normal(action_params, action_std)

    def act(self, state, deterministic=False):
        """Sample action from policy"""
        value, action_params = self.forward(state)
        dist = self.get_action_dist(action_params)

        if self.discrete:
            if deterministic:
                # For discrete, take argmax
                action = action_params.argmax(dim=-1, keepdim=True)
            else:
                # Sample from categorical distribution
                action = dist.sample().unsqueeze(-1)

            logprob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)

            return action, logprob, value
        else:
            # Continuous action case
            if deterministic:
                action = action_params
            else:
                action = dist.sample()

            logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

            return action, logprob, value

    def evaluate(self, state, action):
        """Evaluate actions for training"""
        value, action_params = self.forward(state)
        dist = self.get_action_dist(action_params)

        if self.discrete:
            # For discrete actions
            # action might have an extra dimension, squeeze it
            action_squeezed = action.squeeze(-1) if action.dim() > 1 else action

            logprob = dist.log_prob(action_squeezed).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)

            return logprob, value, entropy
        else:
            # For continuous actions
            logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
            entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)

            return logprob, value, entropy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test continuous actions
    print("Testing Continuous Actions:")
    model_continuous = MLP_AC(
        observation_space=18,
        action_dim=2,
        hidden_dim=64,
        discrete=False,
    ).to(device)

    state = torch.randn(4, 18).to(device)  # Batch of 4
    action, logprob, value = model_continuous.act(state)
    print(
        f"Continuous - Action shape: {action.shape}, Logprob shape: {logprob.shape}, Value shape: {value.shape}"
    )

    # Test discrete actions
    print("\nTesting Discrete Actions:")
    model_discrete = MLP_AC(
        observation_space=18,
        action_dim=5,  # 5 discrete actions (e.g., no-op, up, down, left, right)
        hidden_dim=64,
        discrete=True,
    ).to(device)

    action, logprob, value = model_discrete.act(state)
    print(
        f"Discrete - Action shape: {action.shape}, Logprob shape: {logprob.shape}, Value shape: {value.shape}"
    )
    print(f"Discrete - Action values (should be 0-4): {action.squeeze()}")

    # Test evaluation
    logprob_eval, value_eval, entropy = model_discrete.evaluate(state, action)
    print(
        f"Discrete - Eval Logprob shape: {logprob_eval.shape}, Entropy shape: {entropy.shape}"
    )

    # Count parameters
    pytorch_total_params = sum(
        p.numel() for p in model_discrete.parameters() if p.requires_grad
    )
    print(f"\nTotal trainable parameters: {pytorch_total_params}")
