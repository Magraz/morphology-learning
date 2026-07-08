import torch
import torch.nn as nn

from algorithms.mappo_vanilla.networks.actors import MAPPOActor
from algorithms.mappo_vanilla.networks.critics import (
    MAPPOCritic,
)


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
        self.observation_dim = observation_dim
        self.discrete = discrete
        self.share_actor = share_actor

        if share_actor:
            # Single shared actor for all agents
            if self.discrete:
                self.actor = MAPPOActor(
                    observation_dim,
                    action_dim,
                    hidden_dim,
                    self.discrete,
                )
            else:
                self.actor = MAPPOActor(
                    observation_dim,
                    action_dim,
                    hidden_dim,
                )

        else:
            # Separate actor for each agent
            self.actors = nn.ModuleList(
                [
                    MAPPOActor(
                        observation_dim,
                        action_dim,
                        hidden_dim,
                        discrete,
                    )
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
    ):
        """Evaluate actions for training

        Args:
            obs: Agent observations.
            global_states: Global state for MLP critic.
            actions: Actions to evaluate.
            agent_idx: Agent index for actor selection.
            action_mask: Optional action mask for discrete actions.
        """

        actor = self.get_actor(agent_idx)
        log_probs, entropy = actor.evaluate(obs, actions, action_mask=action_mask)

        # Get values from centralized critic
        values = self.critic(global_states).squeeze(-1)

        return log_probs, values, entropy

    def get_value(self, global_state):
        """Get value from the centralized MLP critic."""
        return self.critic(global_state)


if __name__ == "__main__":
    obs_dim = 22
    n_agents = 5
    action_dim = 5
    hidden_dim = 168

    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    net = MAPPONetwork(
        obs_dim,
        obs_dim * n_agents,
        action_dim,
        n_agents,
        hidden_dim=hidden_dim,
    )
    print(f"Actor:  {count_params(net.actor):>10,}")
    print(f"Critic: {count_params(net.critic):>10,}")
    print(f"Total:  {count_params(net):>10,}")
