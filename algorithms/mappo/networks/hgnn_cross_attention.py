import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

import dhg

from hypergraphs.hgnn_conv_layer import HGNNConv


class CosinePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (sin/cos) from 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        return x + self.pe[:, : x.size(1)]


class HGNNEncoder(nn.Module):
    """
    Encodes a sequence of (hypergraph, node_features) pairs into token embeddings.

    At each timestep the HGNN processes the hypergraph structure and node features,
    then mean-pools over nodes to produce a single d_model-dimensional token.
    Over T timesteps this yields a sequence of T tokens.
    """

    def __init__(
        self,
        node_feat_dim: int,
        d_model: int,
        n_layers: int = 2,
        drop_rate: float = 0.5,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_ch = node_feat_dim if i == 0 else d_model
            self.layers.append(HGNNConv(in_ch, d_model, drop_rate=drop_rate))

    def forward(
        self,
        node_features_seq: List[torch.Tensor],
        hypergraphs_seq: List[dhg.Hypergraph],
    ) -> torch.Tensor:
        """
        Args:
            node_features_seq: List of T tensors, each (n_nodes, node_feat_dim).
            hypergraphs_seq:   List of T dhg.Hypergraph objects.

        Returns:
            (1, T, d_model) — one token per timestep, with a batch dim.
        """
        tokens = []
        for x, hg in zip(node_features_seq, hypergraphs_seq):
            for layer in self.layers:
                x = layer(x, hg)
            tokens.append(x.mean(dim=0))  # mean-pool over nodes → (d_model,)
        return torch.stack(tokens).unsqueeze(0)  # (1, T, d_model)


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: Q from one sequence, K/V from another."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:     (batch, T_q,  d_model)
            key_value: (batch, T_kv, d_model)
        Returns:
            (batch, T_q, d_model)
        """
        B, T_q, _ = query.shape
        T_kv = key_value.size(1)

        Q = self.W_q(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key_value).view(B, T_kv, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(key_value).view(B, T_kv, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, n_heads, T_q, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.W_o(out)


class CrossAttentionBlock(nn.Module):
    """Cross-attention + feed-forward with pre-norm residual connections."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:     (batch, T_q,  d_model)
            key_value: (batch, T_kv, d_model)
        Returns:
            (batch, T_q, d_model)
        """
        x = query + self.cross_attn(self.norm1(query), key_value)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionLayer(nn.Module):
    """
    One layer of N-way cross-attention.

    Each of the N sequences produces queries, while the keys/values come from
    the concatenation of all *other* sequences.  This yields N updated
    sequences per layer.
    """

    def __init__(
        self,
        n_sequences: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_sequences)
            ]
        )

    def forward(self, sequences: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            sequences: List of N tensors, each (batch, T, d_model).
        Returns:
            List of N tensors, each (batch, T, d_model).
        """
        outputs = []
        for i, block in enumerate(self.blocks):
            query = sequences[i]
            others = [sequences[j] for j in range(len(sequences)) if j != i]
            kv = torch.cat(others, dim=1)  # (batch, (N-1)*T, d_model)
            outputs.append(block(query, kv))
        return outputs


class HGNNCrossAttentionTransformer(nn.Module):
    """
    Transformer that fuses N hypergraph-derived token sequences via
    cross-attention.

    Pipeline
    --------
    1. For each hypergraph type *i* and each timestep *t*, an ``HGNNEncoder``
       maps (node_features, hypergraph) → a single token  (mean-pool over nodes).
       This produces N sequences of length T.
    2. Cosine (sinusoidal) positional encoding is added to each sequence.
    3. A stack of ``CrossAttentionLayer``s lets every sequence attend to all
       others.
    4. The N output sequences are averaged, mean-pooled over time, and
       projected to ``output_dim``.
    """

    def __init__(
        self,
        n_sequences: int,
        node_feat_dims: int | List[int],
        d_model: int = 128,
        n_heads: int = 4,
        n_cross_attn_layers: int = 2,
        n_hgnn_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        """
        Args:
            n_sequences:        Number of hypergraph types / encoder streams.
            node_feat_dims:     Per-node feature dimensionality.  Pass a single
                                int if all types share the same dim, or a list
                                of length ``n_sequences`` for heterogeneous dims.
            d_model:            Token / hidden dimensionality throughout the
                                transformer.
            n_heads:            Number of attention heads.
            n_cross_attn_layers: Depth of the cross-attention stack.
            n_hgnn_layers:      Number of HGNNConv layers inside each encoder.
            d_ff:               Feed-forward inner dimensionality.
            dropout:            Dropout rate used everywhere.
            output_dim:         Dimensionality of the final output.
        """
        super().__init__()
        self.n_sequences = n_sequences
        self.d_model = d_model

        # Normalise node_feat_dims to a list
        if isinstance(node_feat_dims, int):
            node_feat_dims = [node_feat_dims] * n_sequences

        # One HGNN encoder per hypergraph type
        self.encoders = nn.ModuleList(
            [
                HGNNEncoder(node_feat_dims[i], d_model, n_hgnn_layers, drop_rate=dropout)
                for i in range(n_sequences)
            ]
        )

        # Shared positional encoding
        self.pos_encoding = CosinePositionalEncoding(d_model)

        # Cross-attention stack
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionLayer(n_sequences, d_model, n_heads, d_ff, dropout)
                for _ in range(n_cross_attn_layers)
            ]
        )

        # Output projection
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self,
        node_features_per_type: List[List[torch.Tensor]],
        hypergraphs_per_type: List[List[dhg.Hypergraph]],
    ) -> torch.Tensor:
        """
        Full forward pass: HGNN encoding → positional encoding → cross-attention → value.

        Args:
            node_features_per_type:
                List of N lists, each containing T tensors of shape
                ``(n_nodes, node_feat_dim)``.
            hypergraphs_per_type:
                List of N lists, each containing T ``dhg.Hypergraph`` objects.

        Returns:
            (1, output_dim)
        """
        sequences = []
        for i in range(self.n_sequences):
            tokens = self.encoders[i](
                node_features_per_type[i], hypergraphs_per_type[i]
            )  # (1, T, d_model)
            tokens = self.pos_encoding(tokens)
            sequences.append(tokens)

        for layer in self.cross_attn_layers:
            sequences = layer(sequences)

        # Aggregate across sequences and time
        combined = torch.stack(sequences, dim=0).mean(dim=0)  # (1, T, d_model)
        pooled = combined.mean(dim=1)  # (1, d_model)
        return self.output_head(pooled)

    def forward_from_tokens(
        self,
        token_sequences: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass from pre-computed token sequences (skips HGNN encoding).

        Useful when the HGNN embeddings are computed externally or when
        the encoder input format differs from the default.

        Args:
            token_sequences: List of N tensors, each (batch, T, d_model).

        Returns:
            (batch, output_dim)
        """
        sequences = [self.pos_encoding(seq) for seq in token_sequences]

        for layer in self.cross_attn_layers:
            sequences = layer(sequences)

        combined = torch.stack(sequences, dim=0).mean(dim=0)  # (batch, T, d_model)
        pooled = combined.mean(dim=1)  # (batch, d_model)
        return self.output_head(pooled)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ---- Config ----
    N_SEQUENCES = 2  # two hypergraph types
    N_NODES = 6
    NODE_FEAT_DIM = 8
    D_MODEL = 64
    T = 5  # timesteps (sequence length)

    # ---- Build random hypergraphs + features for each type and timestep ----
    node_features_per_type: List[List[torch.Tensor]] = []
    hypergraphs_per_type: List[List[dhg.Hypergraph]] = []

    for _ in range(N_SEQUENCES):
        feats = [torch.randn(N_NODES, NODE_FEAT_DIM, device=device) for _ in range(T)]
        hgs = [dhg.random.hypergraph_Gnm(N_NODES, 4).to(device) for _ in range(T)]
        node_features_per_type.append(feats)
        hypergraphs_per_type.append(hgs)

    # ---- Instantiate model ----
    model = HGNNCrossAttentionTransformer(
        n_sequences=N_SEQUENCES,
        node_feat_dims=NODE_FEAT_DIM,
        d_model=D_MODEL,
        n_heads=4,
        n_cross_attn_layers=2,
        n_hgnn_layers=2,
        d_ff=128,
        dropout=0.1,
        output_dim=1,
    ).to(device)

    # ---- Forward pass (full pipeline) ----
    model.eval()
    with torch.no_grad():
        out = model(node_features_per_type, hypergraphs_per_type)
    print(f"Full pipeline output shape: {out.shape}")  # (1, 1)
    print(f"Full pipeline output value: {out.item():.4f}\n")

    # ---- Forward pass (from pre-computed tokens) ----
    token_seqs = [torch.randn(2, T, D_MODEL, device=device) for _ in range(N_SEQUENCES)]
    with torch.no_grad():
        out2 = model.forward_from_tokens(token_seqs)
    print(f"Token-input output shape:   {out2.shape}")  # (2, 1)
    print(f"Token-input output values:  {out2.squeeze().tolist()}\n")

    # ---- Parameter count ----
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
