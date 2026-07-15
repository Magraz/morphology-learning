import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.mappo.networks.utils import layer_init


class ObservationEncoder(nn.Module):
    """Per-agent MLP that maps a raw observation to a d_model token.

    The same weights are applied independently to every agent's observation,
    so a set of N observations becomes a set of N tokens.
    """

    def __init__(self, observation_dim: int, d_model: int):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(observation_dim, d_model))
        self.fc2 = layer_init(nn.Linear(d_model, d_model))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., observation_dim)
        Returns:
            (..., d_model)
        """
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return self.ln(h)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention over a set of tokens.

    Returns both the attended token features and the per-head attention
    scores so callers can inspect which agents attend to which.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = layer_init(nn.Linear(d_model, d_model))
        self.W_k = layer_init(nn.Linear(d_model, d_model))
        self.W_v = layer_init(nn.Linear(d_model, d_model))
        self.W_o = layer_init(nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (batch, n_agents, d_model)
            mask: optional bool tensor, either
                  - (batch, n_agents): a *padding* mask. True marks valid agents,
                    False marks padding that no query may attend to.
                  - (batch, n_agents, n_agents): an *edge* mask. ``mask[b, i, j]``
                    is True if query i may attend to key j, so each agent can be
                    restricted to its own neighbourhood (e.g. a proximity
                    communication graph). Every row must contain at least one
                    True — a fully-masked row softmaxes to NaN. Adjacencies with
                    self-loops satisfy this by construction.
        Returns:
            out:  (batch, n_agents, d_model) — attended features.
            attn: (batch, n_heads, n_agents, n_agents) — attention scores,
                  rows sum to 1 over the key dimension.
        """
        B, N, _ = x.shape

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 2:
                # (B, 1, 1, N) broadcast over heads and query positions.
                attn_mask = mask.view(B, 1, 1, N)
            elif mask.dim() == 3:
                # (B, 1, N, N) broadcast over heads only — per-query neighbourhoods.
                attn_mask = mask.unsqueeze(1)
            else:
                raise ValueError(
                    f"mask must be (B, N) or (B, N, N), got shape {tuple(mask.shape)}"
                )
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, n_heads, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_o(out), attn

    @staticmethod
    def scores_to_adjacency(attn: torch.Tensor) -> torch.Tensor:
        """Merge the directed ij / ji attention scores into one undirected
        edge weight per agent pair.

        For agents i and j there are two scores — ``attn[..., i, j]`` (how much
        i attends to j) and ``attn[..., j, i]`` — which are averaged into a
        single symmetric weight ``(attn[i, j] + attn[j, i]) / 2``. The result is
        a symmetric adjacency matrix per attention head, suitable for building
        one graph per head.

        Args:
            attn: (..., n_agents, n_agents) attention scores, e.g. the
                  (batch, n_heads, n_agents, n_agents) tensor returned by
                  :meth:`forward`.
        Returns:
            (..., n_agents, n_agents) symmetric adjacency matrix.
        """
        return (attn + attn.transpose(-2, -1)) / 2


class MultiHeadAttentionEncoder(nn.Module):
    """Encode a set of per-agent observations, then apply multi-head
    self-attention to produce attention scores between agents.

    Pipeline
    --------
    1. An ``ObservationEncoder`` maps each agent's observation to a d_model
       token (shared weights across agents).
    2. Multi-head self-attention lets every agent attend to every other,
       producing updated tokens and an attention-score matrix.
    3. A pre-norm residual + feed-forward block refines the tokens.
    """

    def __init__(
        self,
        observation_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = ObservationEncoder(observation_dim, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            layer_init(nn.Linear(d_model, d_ff)),
            nn.GELU(),
            nn.Dropout(dropout),
            layer_init(nn.Linear(d_ff, d_model)),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        observations: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_scores: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            observations: (batch, n_agents, observation_dim) — one observation
                          per agent. A 2D input (n_agents, observation_dim) is
                          also accepted and a batch dim is added/removed.
            mask:         optional (batch, n_agents) bool — True = valid agent.
            return_scores: if True, also return the raw *directed* attention
                          scores (before symmetrization), so callers can inspect
                          asymmetric (who-attends-to-whom) coordination.
        Returns:
            tokens:    (batch, n_agents, d_model) — attended per-agent features.
            adjacency: (batch, n_heads, n_agents, n_agents) — one symmetric
                       adjacency matrix per attention head, used to build one
                       graph per head.
            attn:      (batch, n_heads, n_agents, n_agents) — only when
                       ``return_scores=True``. The raw directed attention scores
                       (rows sum to 1 over the key dimension).
        """
        squeeze = observations.dim() == 2
        if squeeze:
            observations = observations.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        tokens = self.encoder(observations)

        attended, attn = self.attn(self.norm1(tokens), mask)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))

        # Merge directed ij / ji scores into one undirected edge weight per pair.
        adjacency = self.attn.scores_to_adjacency(attn)

        if squeeze:
            tokens = tokens.squeeze(0)
            adjacency = adjacency.squeeze(0)
            attn = attn.squeeze(0)
        if return_scores:
            return tokens, adjacency, attn
        return tokens, adjacency


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    n_agents = 6
    observation_dim = 41
    d_model = 64
    n_heads = 4

    model = MultiHeadAttentionEncoder(
        observation_dim=observation_dim,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=128,
    )
    model.eval()

    observations = torch.randn(batch_size, n_agents, observation_dim)
    with torch.no_grad():
        tokens, adjacency = model(observations)

    print(f"observations shape: {tuple(observations.shape)}")
    print(f"tokens shape:       {tuple(tokens.shape)}")
    print(f"adjacency shape:    {tuple(adjacency.shape)}  (batch, n_heads, N, N)")

    is_symmetric = torch.allclose(adjacency, adjacency.transpose(-2, -1))
    print(f"adjacency symmetric: {is_symmetric}")

    # One graph per attention head: adjacency matrix for head 0, batch 0.
    print("\nadjacency matrix (batch 0, head 0):")
    print(adjacency[0, 0])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")
