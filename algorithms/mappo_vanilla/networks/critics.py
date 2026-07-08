import torch.nn as nn

from algorithms.mappo_vanilla.networks.utils import layer_init


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
