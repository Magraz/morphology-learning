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

    def get_value(self, global_state, hypergraphs=None, entropies=None):
        """Get value from centralized critic

        Args:
            global_state: Global state for MLP critic, or per-agent obs (n_agents, obs_dim) for multi_hgnn.
        """

        return self.critic(global_state)

    def _to_obs_grid(self, global_state):
        """Reshape a flat all-agent global state (..., n_agents * obs_dim) into
        the per-agent observation grid (..., n_agents, obs_dim) the GNN critic
        consumes. global_state is built as obs.reshape(batch, -1) upstream."""
        return global_state.view(
            *global_state.shape[:-1], self.n_agents, self.observation_dim
        )


if __name__ == "__main__":
    import numpy as np

    obs_dim = 22
    n_agents = 5
    action_dim = 5
    hidden_dim = 168

    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    def print_breakdown(net, label):
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        actor = net.actor if net.share_actor else net.actors
        print(f"  Actor:              {count_params(actor):>10,}")
        print(f"  Critic:             {count_params(net.critic):>10,}")
        if net.entropy_predictor is not None:
            print(f"  Entropy Predictor:  {count_params(net.entropy_predictor):>10,}")
        print(f"  {'─'*40}")
        print(f"  Total:              {count_params(net):>10,}")

    # 1) MLP critic, no hypergraph
    net1 = MAPPONetwork(
        obs_dim,
        obs_dim * n_agents,
        action_dim,
        n_agents,
        hidden_dim=round(hidden_dim * 1.09),
    )
    print_breakdown(net1, "Condition 1: MLP Critic (no hypergraph)")
