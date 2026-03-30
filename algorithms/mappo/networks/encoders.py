import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.mappo.networks.utils import layer_init


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
        # Shape guard
        if x.dim() == 1:
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

        return x[0]

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1 = self.embedding(x1)
        e2 = self.embedding(x2)
        z = torch.cat([e1, e2], dim=-1)
        return self.last(z)  # logits (use CrossEntropyLoss outside if training)


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
