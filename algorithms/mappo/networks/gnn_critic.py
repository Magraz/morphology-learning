import torch
import torch.nn as nn

from algorithms.mappo.networks.utils import layer_init
from algorithms.mappo.networks.multi_head_attention import MultiHeadAttentionEncoder


class DenseGCNConv(nn.Module):
    """Graph convolution over a dense weighted adjacency matrix.

    Implements the symmetric-normalized propagation rule from Kipf & Welling:

        H' = act( D^{-1/2} (A + I) D^{-1/2} H W )

    where ``A`` is a (batch of) dense, non-negative adjacency matrix — exactly
    the kind produced by ``MultiHeadAttentionEncoder`` from attention scores.
    Self-loops are added so each node keeps its own features, and the degree
    used for normalization includes those self-loops.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.lin = layer_init(nn.Linear(in_channels, out_channels, bias=bias))
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.add_self_loops = add_self_loops

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: node features (..., n_nodes, in_channels)
            A: non-negative adjacency (..., n_nodes, n_nodes)
        Returns:
            (..., n_nodes, out_channels)
        """
        n = A.size(-1)
        if self.add_self_loops:
            eye = torch.eye(n, device=A.device, dtype=A.dtype)
            A = A + eye

        deg = A.sum(dim=-1).clamp_min(1.0)  # (..., n_nodes)
        d_inv_sqrt = deg.rsqrt()
        # D^{-1/2} A D^{-1/2}, broadcasting the degree over rows and columns.
        A_norm = A * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)

        X = self.lin(X)
        X = torch.matmul(A_norm, X)
        return self.drop(self.act(X))


class GNNCritic(nn.Module):
    """Centralized critic that processes one graph per attention head with a
    GCN and produces a single scalar value estimate.

    Input
    -----
    * ``node_features``: (batch, n_agents, node_feat_dim) — the per-agent
      tokens emitted by ``MultiHeadAttentionEncoder``.
    * ``adjacency``: (batch, n_heads, n_agents, n_agents) — one symmetric
      adjacency matrix per attention head, also from the attention encoder.

    Pipeline
    --------
    1. Each head has its own GCN stack that convolves the shared node features
       over that head's graph, yielding per-head node embeddings.
    2. Node embeddings are mean-pooled into one graph embedding per head.
    3. The per-head graph embeddings are concatenated and mixed down to a
       single scalar value.
    """

    def __init__(
        self,
        node_feat_dim: int,
        n_heads: int,
        hidden_dim: int = 128,
        n_gcn_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads

        # One GCN stack per attention head (per-head graph processing).
        self.gcn_stacks = nn.ModuleList()
        for _ in range(n_heads):
            stack = nn.ModuleList()
            stack.append(DenseGCNConv(node_feat_dim, hidden_dim, dropout=dropout))
            for _ in range(n_gcn_layers - 1):
                stack.append(DenseGCNConv(hidden_dim, hidden_dim, dropout=dropout))
            self.gcn_stacks.append(stack)

        # Mix the per-head graph embeddings into a single value.
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(n_heads * hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch, n_agents, node_feat_dim). A 2D input
                           (n_agents, node_feat_dim) is also accepted.
            adjacency:     (batch, n_heads, n_agents, n_agents), or the
                           unbatched (n_heads, n_agents, n_agents).
        Returns:
            value: (batch, 1) value estimate, or (1,) if the input was
                   unbatched.
        """
        squeeze = node_features.dim() == 2
        if squeeze:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)

        assert adjacency.size(1) == self.n_heads, (
            f"expected {self.n_heads} heads, got {adjacency.size(1)}"
        )

        graph_embeddings = []
        for head, stack in enumerate(self.gcn_stacks):
            h = node_features
            a = adjacency[:, head]  # (batch, n_agents, n_agents)
            for conv in stack:
                h = conv(h, a)
            graph_embeddings.append(h.mean(dim=1))  # mean-pool nodes → (batch, hidden)

        combined = torch.cat(graph_embeddings, dim=-1)  # (batch, n_heads * hidden)
        value = self.value_head(combined)  # (batch, 1)

        if squeeze:
            value = value.squeeze(0)  # (1,)
        return value


class AttentionGNNCritic(nn.Module):
    """Centralized critic that turns raw per-agent observations into a value.

    Couples a ``MultiHeadAttentionEncoder`` (obs -> per-agent tokens + one
    adjacency matrix per attention head) with a ``GNNCritic`` (per-head GCN +
    pooling -> scalar value). This exposes the single-module ``forward(obs)``
    contract every other centralized critic uses, so the rest of the MAPPO
    pipeline can treat it like the MLP critic — the only difference is the input
    is the all-agent observation grid ``(batch, n_agents, obs_dim)`` instead of a
    flat global-state vector.
    """

    def __init__(
        self,
        observation_dim: int,
        n_agents: int,
        d_model: int = 128,
        n_heads: int = 4,
        hidden_dim: int = 128,
        n_gcn_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.encoder = MultiHeadAttentionEncoder(
            observation_dim=observation_dim,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.gnn = GNNCritic(
            node_feat_dim=d_model,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            n_gcn_layers=n_gcn_layers,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, n_agents, obs_dim) per-agent observations. A 2D input
                 (n_agents, obs_dim) is also accepted (handled by the encoder
                 and GNN squeeze logic).
        Returns:
            value: (batch, 1) value estimate, or (1,) if the input was unbatched.
        """
        tokens, adjacency = self.encoder(obs)
        return self.gnn(tokens, adjacency)

    def coordination_descriptor(
        self,
        obs: torch.Tensor,
        mode: str = "team",
        source: str = "adjacency",
    ) -> torch.Tensor:
        """Extract a coordination-graph descriptor used for novelty-based
        exploration. Reuses the same attention encoder as the value path, so the
        descriptor reflects the *grounded* coordination structure the critic
        learns.

        Args:
            obs:    (batch, n_agents, obs_dim) per-agent observations. A 2D input
                    (n_agents, obs_dim) is also accepted.
            mode:   "team"  -> one descriptor per graph.
                    "agent" -> one descriptor per agent.
            source: "adjacency"      -> derive from the per-head coordination
                                        graph (who-coordinates-with-whom).
                    "node_embedding" -> derive from the per-agent attended tokens.

        Returns:
            mode="team":  (batch, team_dim)            (or (team_dim,) if 2D in)
            mode="agent": (batch, n_agents, agent_dim) (or (n_agents, agent_dim))
        """
        assert mode in ("team", "agent"), f"invalid mode {mode!r}"
        assert source in ("adjacency", "node_embedding"), f"invalid source {source!r}"

        squeeze = obs.dim() == 2
        if squeeze:
            obs = obs.unsqueeze(0)

        # tokens: (B, N, d_model); adjacency: (B, n_heads, N, N)
        tokens, adjacency = self.encoder(obs)

        if source == "node_embedding":
            if mode == "team":
                desc = tokens.mean(dim=1)  # (B, d_model)
            else:
                desc = tokens  # (B, N, d_model)
        else:  # adjacency
            B, H, N, _ = adjacency.shape
            if mode == "team":
                # Upper triangle of each head's symmetric adjacency, per head.
                iu, ju = torch.triu_indices(N, N, offset=1, device=adjacency.device)
                # (B, H, n_pairs) -> (B, H * n_pairs)
                desc = adjacency[:, :, iu, ju].reshape(B, -1)
            else:
                # Agent i's coordination row across heads: (B, N, H * N).
                desc = adjacency.permute(0, 2, 1, 3).reshape(B, N, H * N)

        if squeeze:
            desc = desc.squeeze(0)
        return desc


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    n_agents = 6
    observation_dim = 41
    d_model = 64
    n_heads = 4

    attn_encoder = MultiHeadAttentionEncoder(
        observation_dim=observation_dim,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=128,
    )
    critic = GNNCritic(
        node_feat_dim=d_model,
        n_heads=n_heads,
        hidden_dim=128,
        n_gcn_layers=2,
    )
    attn_encoder.eval()
    critic.eval()

    observations = torch.randn(batch_size, n_agents, observation_dim)
    with torch.no_grad():
        tokens, adjacency = attn_encoder(observations)
        value = critic(tokens, adjacency)

    print(f"observations shape: {tuple(observations.shape)}")
    print(f"tokens shape:       {tuple(tokens.shape)}")
    print(f"adjacency shape:    {tuple(adjacency.shape)}")
    print(f"value shape:        {tuple(value.shape)}  (batch, 1)")
    print(f"values:             {value.squeeze(-1).tolist()}")

    n_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"\nGNNCritic trainable parameters: {n_params:,}")
