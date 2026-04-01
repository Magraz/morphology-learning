import dhg
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.mappo.networks.utils import layer_init
from hypergraphs.hgnn_conv_layer import HGNNConv


class LocalStateEncoder(nn.Module):
    def __init__(self, obs_size: int, num_outputs: int):
        super().__init__()
        self.obs_size = obs_size
        self.num_outputs = num_outputs

        self.init = nn.Linear(obs_size, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_outputs)  # per-agent embedding
        self.ln0 = nn.LayerNorm(64)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)

        # Head over concatenated embeddings [e(x1)||e(x2)]
        self.last = nn.Linear(num_outputs * 2, 2)  # logits for 2-way decision

        self._init_weights()

        # Freeze if you truly want random features
        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    def _init_weights(self):
        for m in [self.init, self.fc1, self.fc2, self.fc3, self.last]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)  # good random projection
                nn.init.zeros_(m.bias)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = self.init(x)
        x = self.ln0(x)
        x = F.silu(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.silu(x)
        x = self.fc3(x)
        # Keep embeddings scale-stable for kNN / dot products
        x = F.normalize(x, dim=-1)  # unit-norm features
        if squeeze:
            x = x.squeeze(0)
        return x


class HypergraphStateEncoder(nn.Module):
    """Frozen random encoder that produces a fixed-size embedding from
    a set of hypergraphs and node features.

    Mirrors the LocalStateEncoder API but operates on hypergraph-structured
    inputs.  Each hyperedge type gets its own randomly-initialized HGNNConv
    stack.  Per-type agent representations are mean-pooled and concatenated,
    then projected to a unit-norm embedding via an MLP.
    """

    def __init__(
        self,
        n_hyperedge_types: int,
        observation_dim: int,
        num_outputs: int,
        hidden_dim: int = 64,
        n_hgnn_layers: int = 2,
    ):
        super().__init__()
        self.n_hyperedge_types = n_hyperedge_types
        self.num_outputs = num_outputs

        # Per-type HGNN convolution stacks (drop_rate=0 for deterministic encoding)
        self.conv_stacks = nn.ModuleList()
        for _ in range(n_hyperedge_types):
            stack = nn.ModuleList()
            stack.append(HGNNConv(observation_dim, hidden_dim, drop_rate=0.0))
            for _ in range(n_hgnn_layers - 1):
                stack.append(HGNNConv(hidden_dim, hidden_dim, drop_rate=0.0))
            self.conv_stacks.append(stack)

        # MLP over concatenated per-type pooled vectors
        pooled_dim = n_hyperedge_types * hidden_dim
        self.fc1 = nn.Linear(pooled_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)

        # Head over concatenated embeddings [e(s1)||e(s2)]
        self.last = nn.Linear(num_outputs * 2, 2)

        self._init_weights()

        # Freeze all weights for random feature extraction
        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    def _init_weights(self):
        for stack in self.conv_stacks:
            for conv in stack:
                nn.init.orthogonal_(conv.theta.weight)
                nn.init.zeros_(conv.theta.bias)
        for m in [self.fc1, self.fc2, self.last]:
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def embedding(
        self, X: torch.Tensor, hypergraphs: list[dhg.Hypergraph]
    ) -> torch.Tensor:
        """Produce a unit-norm embedding from node features and hypergraphs.

        Args:
            X: Node features of shape (n_agents, observation_dim).
            hypergraphs: List of dhg.Hypergraph, one per hyperedge type.

        Returns:
            Unit-norm embedding of shape (num_outputs,).
        """
        pooled = []
        for conv_stack, hg in zip(self.conv_stacks, hypergraphs):
            h = X
            for conv in conv_stack:
                h = conv(h, hg)
            pooled.append(h.mean(dim=0))  # mean-pool over agents → (hidden_dim,)

        z = torch.cat(pooled, dim=-1)  # (n_types * hidden_dim,)
        z = F.silu(self.ln1(self.fc1(z)))
        z = self.fc2(z)
        return F.normalize(z, dim=-1)  # unit-norm embedding


class HypergraphEntropyPredictor(nn.Module):
    """Shared LSTM predictor of hypergraph structural entropy.

    A single LSTM with learnable agent embeddings replaces per-agent
    LSTMs, allowing the full mixed-agent batch to be processed in one
    forward pass.  Each prediction uses only one agent's observation
    sequence — the agent embedding encodes which agent is being
    predicted for.
    """

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        n_hyperedge_types: int,
        hidden_dim: int = 64,
        n_lstm_layers: int = 1,
        agent_embed_dim: int = 8,
    ):
        super().__init__()
        self.agent_embedding = nn.Embedding(n_agents, agent_embed_dim)
        self.lstm = nn.LSTM(
            observation_dim + agent_embed_dim,
            hidden_dim,
            n_lstm_layers,
            batch_first=True,
        )
        self.mean_head = layer_init(nn.Linear(hidden_dim, n_hyperedge_types), std=0.01)
        self.log_var_head = layer_init(
            nn.Linear(hidden_dim, n_hyperedge_types), std=0.01
        )

    def _forward_impl(
        self, obs_sequences: torch.Tensor, agent_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core forward: concat agent embedding to obs, run LSTM.

        Args:
            obs_sequences: (batch, seq_len, obs_dim)
            agent_ids:     (batch,) long tensor of agent indices.

        Returns:
            mean:    (batch, n_hyperedge_types)
            log_var: (batch, n_hyperedge_types)
        """
        embed = self.agent_embedding(agent_ids)  # (batch, agent_embed_dim)
        embed = embed.unsqueeze(1).expand(-1, obs_sequences.shape[1], -1)
        x = torch.cat([obs_sequences, embed], dim=-1)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]  # last layer: (batch, hidden)
        return self.mean_head(h), self.log_var_head(h)

    def forward(
        self, obs_sequences: torch.Tensor, agent_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict entropy for a single agent.

        Args:
            obs_sequences: (batch, seq_len, obs_dim)
            agent_idx:     Which agent to predict for (int).

        Returns:
            mean:    (batch, n_hyperedge_types)
            log_var: (batch, n_hyperedge_types)
        """
        agent_ids = torch.full(
            (obs_sequences.shape[0],),
            agent_idx,
            dtype=torch.long,
            device=obs_sequences.device,
        )
        return self._forward_impl(obs_sequences, agent_ids)

    def forward_batch(
        self,
        obs_sequences: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict entropy for a mixed-agent batch in a single LSTM call.

        Args:
            obs_sequences: (batch, seq_len, obs_dim)
            agent_indices: (batch,) int tensor of agent IDs.

        Returns:
            mean:    (batch, n_hyperedge_types)
            log_var: (batch, n_hyperedge_types)
        """
        return self._forward_impl(obs_sequences, agent_indices)
