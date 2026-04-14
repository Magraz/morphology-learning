import torch
import torch.nn as nn

from algorithms.mappo.networks.actors import MAPPOActor, MAPPO_Hybrid_Actor
from algorithms.mappo.networks.critics import (
    MAPPOCritic,
    MultiHGNNCritic,
    HGNNCrossAttentionCritic,
)
from algorithms.mappo.networks.encoders import (
    AffinityTransformer,
    HypergraphEntropyPredictor,
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
        critic_type: str = "mlp",  # "mlp" | "multi_hgnn" | "hg_cross_attention"
        n_hyperedge_types: int = 0,  # Required when critic_type uses hypergraphs
        critic_seq_len: int = 1,
        entropy_conditioning: bool = False,
        hypergraph_mode: str = "predefined",
        history_len: int = 0,
    ):
        super(MAPPONetwork, self).__init__()

        self.n_agents = n_agents
        self.discrete = discrete
        self.share_actor = share_actor
        self.critic_type = critic_type
        # Extra actor input features from predicted entropy: [mean, log_var] per type
        self.entropy_pred_dim = (
            2 * n_hyperedge_types
            if entropy_conditioning and n_hyperedge_types > 0
            else 0
        )

        if share_actor:
            # Single shared actor for all agents
            if self.discrete:
                self.actor = MAPPOActor(
                    observation_dim,
                    action_dim,
                    hidden_dim,
                    self.discrete,
                    entropy_pred_dim=self.entropy_pred_dim,
                )
            else:
                self.actor = MAPPOActor(
                    observation_dim,
                    action_dim,
                    hidden_dim,
                    entropy_pred_dim=self.entropy_pred_dim,
                )

        else:
            # Separate actor for each agent
            if self.discrete:
                self.actors = nn.ModuleList(
                    [
                        MAPPOActor(
                            observation_dim,
                            action_dim,
                            hidden_dim,
                            discrete,
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
        elif critic_type == "hg_cross_attention":
            assert (
                n_hyperedge_types > 0
            ), "n_hyperedge_types must be > 0 for hg_cross_attention critic"
            self.critic = HGNNCrossAttentionCritic(
                n_hyperedge_types,
                n_agents,
                observation_dim,
                hidden_dim=hidden_dim,
                seq_len=critic_seq_len,
                entropy_conditioning=entropy_conditioning,
            )
        else:
            self.critic = MAPPOCritic(global_state_dim, hidden_dim * 2)

        # Auxiliary LSTM predictor of hypergraph structural entropy
        self.entropy_predictor = (
            HypergraphEntropyPredictor(
                n_agents, observation_dim, n_hyperedge_types, hidden_dim=64
            )
            if entropy_conditioning and n_hyperedge_types > 0
            else None
        )

        # Learned affinity transformer for dynamic grouping
        self.affinity_transformer = (
            AffinityTransformer(
                n_agents=n_agents,
                observation_dim=observation_dim,
                history_length=history_len,
                d_model=hidden_dim,
            )
            if hypergraph_mode == "learned_affinity" and history_len > 0
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
        if self.critic_type in ("multi_hgnn", "hg_cross_attention"):
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
        if self.critic_type in ("multi_hgnn", "hg_cross_attention"):
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


if __name__ == "__main__":
    import numpy as np

    obs_dim = 21
    n_agents = 5
    action_dim = 5
    hidden_dim = 168
    n_hyperedge_types = 2

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
        critic_type="mlp",
    )
    print_breakdown(net1, "Condition 1: MLP Critic (no hypergraph)")

    # 2) Multi-HGNN critic, no entropy conditioning
    net2 = MAPPONetwork(
        obs_dim,
        obs_dim * n_agents,
        action_dim,
        n_agents,
        hidden_dim=hidden_dim,
        critic_type="multi_hgnn",
        n_hyperedge_types=n_hyperedge_types,
        entropy_conditioning=False,
    )
    print_breakdown(net2, "Condition 2: Multi-HGNN Critic (no entropy conditioning)")

    # 3) Multi-HGNN critic + entropy conditioning + entropy predictor
    net3 = MAPPONetwork(
        obs_dim,
        obs_dim * n_agents,
        action_dim,
        n_agents,
        hidden_dim=hidden_dim,
        critic_type="multi_hgnn",
        n_hyperedge_types=n_hyperedge_types,
        entropy_conditioning=True,
    )
    print_breakdown(
        net3, "Condition 3: Multi-HGNN Critic + Entropy Conditioning + Predictor"
    )

    # 4) HG Cross-Attention critic
    net4 = MAPPONetwork(
        obs_dim,
        obs_dim * n_agents,
        action_dim,
        n_agents,
        hidden_dim=round(80),
        critic_type="hg_cross_attention",
        n_hyperedge_types=n_hyperedge_types,
    )
    print_breakdown(net4, "Condition 4: HG Cross-Attention Critic")

    print()
