import dhg
import torch
import torch.nn as nn

from algorithms.mappo.networks.utils import layer_init
from algorithms.mappo.networks.hgnn_cross_attention import (
    CosinePositionalEncoding,
    CrossAttentionLayer,
)
from hypergraphs.hgnn_conv_layer import HGNNConv


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

    When ``entropy_conditioning=True`` the first convolution layer accepts
    ``observation_dim + 1`` features so that a per-type structural entropy
    scalar can be broadcast to every agent node and concatenated with X.
    """

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 256,
        n_hgnn_layers: int = 2,
        drop_rate: float = 0.5,
        value_mode: str = "shared",
        entropy_conditioning: bool = False,
    ):
        super(HGNNCritic, self).__init__()

        assert value_mode in (
            "shared",
            "per_agent",
        ), f"value_mode must be 'shared' or 'per_agent', got '{value_mode}'"
        self.value_mode = value_mode
        self.entropy_conditioning = entropy_conditioning

        # First conv input dim is larger when entropy-conditioned
        input_dim = observation_dim + 1 if entropy_conditioning else observation_dim

        # HGNN convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(HGNNConv(input_dim, hidden_dim, drop_rate=drop_rate))
        for _ in range(n_hgnn_layers - 1):
            self.convs.append(HGNNConv(hidden_dim, hidden_dim, drop_rate=drop_rate))

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, X: torch.Tensor, hg: dhg.Hypergraph, entropy: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            X: Node feature matrix of shape (n_agents, observation_dim).
            hg: Hypergraph structure over the n_agents vertices.
            entropy: Optional scalar tensor of shape ``()`` or ``(1,)``.
                     When provided (and ``entropy_conditioning`` is True),
                     it is broadcast to every agent node and concatenated
                     with X before the first convolution.

        Returns:
            If value_mode == "shared":    shape (1,)
            If value_mode == "per_agent": shape (n_agents, 1)
        """
        if self.entropy_conditioning:
            n_agents = X.shape[0]
            if entropy is not None:
                ent_col = entropy.reshape(1, 1).expand(n_agents, 1)
            else:
                ent_col = torch.zeros(n_agents, 1, device=X.device)
            X = torch.cat([X, ent_col], dim=-1)

        return self._forward_conv(X, hg)

    def forward_unconditioned(
        self, X: torch.Tensor, hg: dhg.Hypergraph
    ) -> torch.Tensor:
        """Forward pass where X already has entropy concatenated (used by
        ``MultiHGNNCritic.forward_batched`` which handles concatenation
        externally for the block-diagonal case)."""
        return self._forward_conv(X, hg)

    def _forward_conv(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
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

    Each critic is entropy-conditioned: the structural entropy of the
    corresponding hypergraph type is broadcast to every agent node and
    concatenated with X before the first HGNN convolution.
    """

    def __init__(
        self,
        n_hyperedge_types: int,
        n_agents: int,
        observation_dim: int,
        hidden_dim: int = 128,
        n_hgnn_layers: int = 2,
        drop_rate: float = 0.5,
        entropy_conditioning: bool = False,
    ):
        super(MultiHGNNCritic, self).__init__()

        self.n_hyperedge_types = n_hyperedge_types

        # One HGNNCritic per hyperedge type
        self.critics = nn.ModuleList(
            [
                HGNNCritic(
                    observation_dim,
                    hidden_dim=round(hidden_dim * 0.85),
                    n_hgnn_layers=n_hgnn_layers,
                    drop_rate=drop_rate,
                    value_mode="per_agent",
                    entropy_conditioning=entropy_conditioning,
                )
                for _ in range(n_hyperedge_types)
            ]
        )

        self.mixer = nn.Sequential(
            layer_init(nn.Linear(n_hyperedge_types * n_agents, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1)),
        )

    def forward(
        self,
        X: torch.Tensor,
        hypergraphs: list,
        entropies: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            X: Node feature matrix of shape (n_agents, observation_dim).
            hypergraphs: List of dhg.Hypergraph, one per hyperedge type.
            entropies: Optional tensor of shape (n_types,) with structural
                       entropy for each hyperedge type.

        Returns:
            Scalar value estimate of shape (1,).
        """
        critic_values = torch.cat(
            [
                critic(
                    X, hg, entropy=entropies[i] if entropies is not None else None
                ).flatten()
                for i, (critic, hg) in enumerate(zip(self.critics, hypergraphs))
            ]
        )  # (n_hyperedge_types * n_agents,)

        return self.mixer(critic_values)  # (1,)

    def forward_batched(
        self,
        X: torch.Tensor,
        batched_hgs: list,
        n_graphs: int,
        entropies: torch.Tensor = None,
    ) -> torch.Tensor:
        """Batched forward pass over multiple environments/timesteps.

        Args:
            X: Node features of shape (n_graphs * n_agents, obs_dim).
            batched_hgs: List of block-diagonal dhg.Hypergraph (one per
                         hyperedge type), each with n_graphs * n_agents vertices.
            n_graphs: Number of graphs batched together.
            entropies: Optional tensor of shape (n_graphs, n_types) with
                       structural entropy per graph per type.

        Returns:
            Value estimates of shape (n_graphs, 1).
        """
        n_agents = X.shape[0] // n_graphs

        per_type_values = []
        for type_idx, (critic, hg) in enumerate(zip(self.critics, batched_hgs)):
            if critic.entropy_conditioning:
                # Build per-node entropy column for this type: (n_graphs * n_agents, 1)
                if entropies is not None:
                    ent_per_graph = entropies[:, type_idx]  # (n_graphs,)
                    ent_col = ent_per_graph.repeat_interleave(n_agents).unsqueeze(-1)
                else:
                    ent_col = torch.zeros(X.shape[0], 1, device=X.device)
                X_cond = torch.cat([X, ent_col], dim=-1)
                v = critic.forward_unconditioned(X_cond, hg)
            else:
                v = critic(X, hg)
            v = v.view(n_graphs, n_agents)  # (n_graphs, n_agents)
            per_type_values.append(v)

        # (n_graphs, n_types * n_agents)
        combined = torch.cat(per_type_values, dim=1)
        return self.mixer(combined)  # (n_graphs, 1)


class HGNNCrossAttentionCritic(nn.Module):
    """Critic that fuses multiple hypergraph types via HGNN encoding
    followed by cross-attention.

    For each hyperedge type, HGNN convolution produces per-agent embeddings
    (n_agents, hidden_dim).  These N sequences are treated as separate
    token streams and fused with a stack of cross-attention layers where
    each type attends to the concatenation of all other types.

    The fused representations are mean-pooled and projected to a scalar
    value estimate.
    """

    def __init__(
        self,
        n_hyperedge_types: int,
        n_agents: int,
        observation_dim: int,
        hidden_dim: int = 128,
        seq_len: int = 1,
        n_heads: int = 4,
        n_cross_attn_layers: int = 2,
        n_hgnn_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        entropy_conditioning: bool = False,
    ):
        super().__init__()
        self.n_hyperedge_types = n_hyperedge_types
        self.n_agents = n_agents
        self.seq_len = seq_len
        self.entropy_conditioning = entropy_conditioning
        self.uses_temporal_sequences = True

        input_dim = observation_dim + 1 if entropy_conditioning else observation_dim

        # One set of HGNN conv layers per hyperedge type
        self.convs = nn.ModuleList()
        for _ in range(n_hyperedge_types):
            layers = nn.ModuleList()
            layers.append(HGNNConv(input_dim, hidden_dim, drop_rate=dropout))
            for _ in range(n_hgnn_layers - 1):
                layers.append(HGNNConv(hidden_dim, hidden_dim, drop_rate=dropout))
            self.convs.append(layers)

        # Cosine positional encoding + cross-attention stack
        self.pos_encoding = CosinePositionalEncoding(hidden_dim)
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    n_hyperedge_types, hidden_dim, n_heads, d_ff, dropout
                )
                for _ in range(n_cross_attn_layers)
            ]
        )

        # Value projection
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        X: torch.Tensor,
        hypergraphs: list,
        entropies: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            X: (n_agents, observation_dim).
            hypergraphs: List of dhg.Hypergraph, one per hyperedge type.
            entropies: Optional (n_types,) tensor for entropy conditioning.

        Returns:
            Scalar value estimate of shape (1,).
        """
        X_seq = X.unsqueeze(0).unsqueeze(0)  # (1, 1, n_agents, obs_dim)
        sequences = self._encode_sequence(X_seq, hypergraphs)
        return self._fuse_and_pool(sequences).squeeze(0)  # (1,)

    def forward_batched(
        self,
        X: torch.Tensor,
        batched_hgs: list,
        n_graphs: int,
        entropies: torch.Tensor = None,
    ) -> torch.Tensor:
        """Batched forward pass over multiple environments/timesteps.

        Args:
            X:
                Either (n_graphs * n_agents, obs_dim) for single-step batched
                evaluation, or (n_graphs, seq_len, n_agents, obs_dim) for
                temporal evaluation.
            batched_hgs:
                List of block-diagonal dhg.Hypergraph (one per type). For the
                temporal case each hypergraph batches ``n_graphs * seq_len``
                graphs together.
            n_graphs: Number of batch elements.
            entropies:
                Optional (n_graphs, n_types) for single-step conditioning, or
                (n_graphs, seq_len, n_types) for temporal conditioning.

        Returns:
            (n_graphs, 1).
        """
        if X.dim() == 2:
            n_agents = X.shape[0] // n_graphs
            X = X.view(n_graphs, 1, n_agents, -1)

        sequences = self._encode_sequence(X, batched_hgs)

        return self._fuse_and_pool(sequences)  # (n_graphs, 1)

    def _encode_sequence(self, X_seq, batched_hgs):
        """Run HGNN encoding for each type, return per-type token sequences."""
        batch_size, seq_len, n_agents, obs_dim = X_seq.shape
        x_flat = X_seq.reshape(batch_size * seq_len * n_agents, obs_dim)

        sequences = []
        for type_idx, conv_layers in enumerate(self.convs):
            x = x_flat

            for conv in conv_layers:
                x = conv(x, batched_hgs[type_idx])

            # Mean-pool agents at each timestep into a temporal token stream.
            tokens = x.view(batch_size, seq_len, n_agents, -1).mean(dim=2)
            sequences.append(tokens)  # (batch, seq_len, hidden)
        return sequences

    def _fuse_and_pool(self, sequences):
        """Apply positional encoding, cross-attention, and pool to value."""
        sequences = [self.pos_encoding(seq) for seq in sequences]

        for layer in self.cross_attn_layers:
            sequences = layer(sequences)

        # Mean over types and agents
        combined = torch.stack(sequences, dim=0).mean(
            dim=0
        )  # (batch, n_agents, hidden)
        pooled = combined.mean(dim=1)  # (batch, hidden)
        return self.value_head(pooled)  # (batch, 1)
