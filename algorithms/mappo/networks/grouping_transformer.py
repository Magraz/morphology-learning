import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.mappo.networks.utils import layer_init
from algorithms.mappo.networks.hgnn_cross_attention import CosinePositionalEncoding


class GroupingTransformer(nn.Module):
    """Autoregressive transformer that produces a list of hyperedges from
    per-agent observation histories.

    Output token vocabulary (size n_agents + 2):
        0 .. n_agents-1  : agent indices ("append agent i to the current edge")
        EOE = n_agents   : end of current hyperedge (close it, start a new one)
        EOS = n_agents+1 : terminate generation

    A canonical sequence for hyperedges [[3, 2], [4, 1, 5], [0, 6]] is:
        3, 2, EOE, 4, 1, 5, EOE, 0, 6, EOE, EOS

    Two grouping modes (selected at construction time):
        - allow_overlap=False : each agent appears in at most one hyperedge.
          EOS is gated until every agent has been emitted.
        - allow_overlap=True  : an agent may appear in multiple hyperedges.
          EOS is gated until every agent has been covered at least once
          ("coverage requirement"). max_hyperedges acts as a hard cap.
    """

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        history_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        allow_overlap: bool = False,
        max_hyperedges: Optional[int] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.d_model = d_model
        self.allow_overlap = allow_overlap
        self.max_hyperedges = max_hyperedges if max_hyperedges is not None else n_agents
        self.eoe_id = n_agents
        self.eos_id = n_agents + 1
        self.vocab_size = n_agents + 2

        # Per-agent temporal encoder.
        self.input_proj = layer_init(nn.Linear(observation_dim, d_model))
        self.temporal_pos = nn.Parameter(torch.randn(1, history_length, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Identity embedding so the decoder can distinguish agents whose
        # encoded histories happen to look similar.
        self.agent_index_embed = nn.Embedding(n_agents, d_model)

        # Delimiter / start-of-sequence embeddings.
        self.eoe_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.eos_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.bos_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Tells the decoder which hyperedge it is currently building.
        self.edge_index_embed = nn.Embedding(self.max_hyperedges + 1, d_model)

        self.dec_pos = CosinePositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Pointer-style scoring projection.
        self.out_proj = layer_init(nn.Linear(d_model, d_model))

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def encode_agents(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Encode each agent's observation history into a single d_model vector.

        Args:
            obs_history: (B, N, T, obs_dim)

        Returns:
            agent_emb: (B, N, d_model)
        """
        B, N, T, _ = obs_history.shape
        x = obs_history.reshape(B * N, T, -1)
        h = self.input_proj(x) + self.temporal_pos[:, :T]
        h = self.temporal_encoder(h)
        h = h.mean(dim=1)
        h = h.reshape(B, N, self.d_model)
        idx = torch.arange(N, device=obs_history.device)
        return h + self.agent_index_embed(idx).unsqueeze(0)

    def _build_memory(self, agent_emb: torch.Tensor) -> torch.Tensor:
        """Append delimiter slot vectors to agent embeddings.

        Returns memory of shape (B, N + 2, d_model). Index N is the EOE slot,
        index N+1 is the EOS slot — letting the pointer head score them via
        the same dot-product mechanism as agent slots.
        """
        B = agent_emb.size(0)
        eoe = self.eoe_embed.view(1, 1, -1).expand(B, 1, -1)
        eos = self.eos_embed.view(1, 1, -1).expand(B, 1, -1)
        return torch.cat([agent_emb, eoe, eos], dim=1)

    # ------------------------------------------------------------------
    # State scan (legality bookkeeping)
    # ------------------------------------------------------------------
    def _scan_state(self, tokens: torch.Tensor) -> dict:
        """For each step t, return the state BEFORE consuming tokens[:, t].

        That is, the state used to mask the prediction of tokens[:, t].
        """
        B, L = tokens.shape
        N = self.n_agents
        device = tokens.device

        edge_indices = torch.zeros(B, L, dtype=torch.long, device=device)
        used_anywhere = torch.zeros(B, L, N, dtype=torch.bool, device=device)
        used_in_current = torch.zeros(B, L, N, dtype=torch.bool, device=device)
        current_edge_empty = torch.ones(B, L, dtype=torch.bool, device=device)
        coverage_complete = torch.zeros(B, L, dtype=torch.bool, device=device)
        edges_closed = torch.zeros(B, L, dtype=torch.long, device=device)

        cur_any = torch.zeros(B, N, dtype=torch.bool, device=device)
        cur_in_cur = torch.zeros(B, N, dtype=torch.bool, device=device)
        cur_edge_idx = torch.zeros(B, dtype=torch.long, device=device)
        cur_closed = torch.zeros(B, dtype=torch.long, device=device)
        cur_empty = torch.ones(B, dtype=torch.bool, device=device)

        for t in range(L):
            edge_indices[:, t] = cur_edge_idx
            used_anywhere[:, t] = cur_any
            used_in_current[:, t] = cur_in_cur
            current_edge_empty[:, t] = cur_empty
            coverage_complete[:, t] = cur_any.all(dim=-1)
            edges_closed[:, t] = cur_closed

            tok = tokens[:, t]
            is_agent = tok < N
            is_eoe = tok == self.eoe_id
            agent_safe = tok.clamp(max=N - 1)
            agent_oh = F.one_hot(agent_safe, num_classes=N).bool() & is_agent.unsqueeze(
                -1
            )

            cur_any = cur_any | agent_oh
            cur_in_cur = cur_in_cur | agent_oh
            zero_n = torch.zeros_like(cur_in_cur)
            cur_in_cur = torch.where(
                is_eoe.unsqueeze(-1).expand_as(cur_in_cur), zero_n, cur_in_cur
            )
            cur_edge_idx = cur_edge_idx + is_eoe.long()
            cur_closed = cur_closed + is_eoe.long()
            true_b = torch.ones_like(cur_empty)
            false_b = torch.zeros_like(cur_empty)
            cur_empty = torch.where(
                is_eoe, true_b, torch.where(is_agent, false_b, cur_empty)
            )

        return {
            "edge_indices": edge_indices,
            "used_anywhere": used_anywhere,
            "used_in_current": used_in_current,
            "current_edge_empty": current_edge_empty,
            "coverage_complete": coverage_complete,
            "edges_closed": edges_closed,
        }

    def _legal_mask(
        self,
        used_anywhere: torch.Tensor,  # (*, N)
        used_in_current: torch.Tensor,  # (*, N)
        current_edge_empty: torch.Tensor,  # (*,)
        coverage_complete: torch.Tensor,  # (*,)
        edges_closed: torch.Tensor,  # (*,)
    ) -> torch.Tensor:
        """Boolean mask of shape (*, vocab_size). True = allowed."""
        N = self.n_agents
        prefix = used_anywhere.shape[:-1]
        allowed = torch.ones(
            *prefix, self.vocab_size, dtype=torch.bool, device=used_anywhere.device
        )
        if self.allow_overlap:
            allowed[..., :N] = ~used_in_current
        else:
            allowed[..., :N] = ~used_anywhere

        at_cap = edges_closed >= self.max_hyperedges
        # Once at cap, agents may not open a new edge.
        opening_blocked = at_cap & current_edge_empty
        allowed[..., :N] = allowed[..., :N] & (~opening_blocked.unsqueeze(-1))

        allowed[..., self.eoe_id] = (~current_edge_empty) & (~at_cap)
        # Normal termination: coverage met and the current edge is closed.
        # at_cap is a hard backstop — once we've emitted max_hyperedges edges,
        # EOS becomes available unconditionally, otherwise overlap-mode runs
        # could deadlock when no further EOE is permitted.
        allowed[..., self.eos_id] = (coverage_complete & current_edge_empty) | at_cap
        return allowed

    # ------------------------------------------------------------------
    # Decoder input embedding
    # ------------------------------------------------------------------
    def _embed_tokens(
        self,
        prev_tokens: torch.Tensor,  # (B, L) long, -1 marks the BOS slot
        memory: torch.Tensor,  # (B, N+2, d_model)
        edge_indices: torch.Tensor,  # (B, L) long
    ) -> torch.Tensor:
        B, L = prev_tokens.shape
        d = self.d_model
        is_bos = prev_tokens.eq(-1)
        safe = prev_tokens.clamp(min=0)
        idx = safe.unsqueeze(-1).expand(-1, -1, d)
        token_embed = memory.gather(dim=1, index=idx)
        bos = self.bos_embed.view(1, 1, d).expand(B, L, d)
        token_embed = torch.where(is_bos.unsqueeze(-1), bos, token_embed)

        edge_e = self.edge_index_embed(edge_indices.clamp(max=self.max_hyperedges))
        return self.dec_pos(token_embed + edge_e)

    # ------------------------------------------------------------------
    # Pointer head
    # ------------------------------------------------------------------
    def _decode_logits(
        self, hidden: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        q = self.out_proj(hidden)
        return torch.bmm(q, memory.transpose(1, 2)) / math.sqrt(self.d_model)

    @staticmethod
    def _apply_mask(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~allowed, float("-inf"))

    # ------------------------------------------------------------------
    # Teacher-forced forward
    # ------------------------------------------------------------------
    def forward(
        self, obs_history: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced training pass.

        Args:
            obs_history:    (B, N, T, obs_dim)
            target_tokens:  (B, L) long — the sequence the model should produce,
                            including EOE/EOS terminators.

        Returns:
            logits: (B, L, n_agents + 2) with illegal tokens masked to -inf.
        """
        B, L = target_tokens.shape
        device = obs_history.device

        agent_emb = self.encode_agents(obs_history)
        memory = self._build_memory(agent_emb)

        bos = torch.full((B, 1), -1, dtype=torch.long, device=device)
        prev = torch.cat([bos, target_tokens[:, :-1]], dim=1)

        state = self._scan_state(target_tokens)

        h_in = self._embed_tokens(prev, memory, state["edge_indices"])

        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
        )
        hidden = self.decoder(h_in, memory, tgt_mask=causal)
        logits = self._decode_logits(hidden, memory)

        allowed = self._legal_mask(
            state["used_anywhere"],
            state["used_in_current"],
            state["current_edge_empty"],
            state["coverage_complete"],
            state["edges_closed"],
        )
        return self._apply_mask(logits, allowed)

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        obs_history: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> list[list[tuple[int, ...]]]:
        """Greedy or sampled autoregressive decoding.

        Args:
            obs_history: (B, N, T, obs_dim)
            sample:      if True, draw from softmax(logits/temperature);
                         else argmax.
            temperature: only used when sample=True.

        Returns:
            A list of length B; each element is a list of hyperedges
            (each hyperedge a tuple of agent indices), in the same format
            as algorithms/mappo/hypergraph.py and accepted by dhg.Hypergraph.
        """
        B = obs_history.size(0)
        N = self.n_agents
        device = obs_history.device

        agent_emb = self.encode_agents(obs_history)
        memory = self._build_memory(agent_emb)

        used_any = torch.zeros(B, N, dtype=torch.bool, device=device)
        used_cur = torch.zeros(B, N, dtype=torch.bool, device=device)
        edge_idx = torch.zeros(B, dtype=torch.long, device=device)
        edges_closed = torch.zeros(B, dtype=torch.long, device=device)
        cur_empty = torch.ones(B, dtype=torch.bool, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        emitted: list[torch.Tensor] = []
        max_len = self.max_hyperedges * (N + 1) + 1

        for _ in range(max_len):
            if finished.all():
                break

            bos = torch.full((B, 1), -1, dtype=torch.long, device=device)
            if emitted:
                emitted_t = torch.stack(emitted, dim=1)
                prev = torch.cat([bos, emitted_t], dim=1)
                sc = self._scan_state(emitted_t)
                edge_indices_for_dec = torch.cat(
                    [sc["edge_indices"], edge_idx.unsqueeze(1)], dim=1
                )
            else:
                prev = bos
                edge_indices_for_dec = torch.zeros(
                    B, 1, dtype=torch.long, device=device
                )

            L = prev.size(1)
            h_in = self._embed_tokens(prev, memory, edge_indices_for_dec)
            causal = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
            )
            hidden = self.decoder(h_in, memory, tgt_mask=causal)
            logits = self._decode_logits(hidden[:, -1:], memory).squeeze(1)

            allowed = self._legal_mask(
                used_any,
                used_cur,
                cur_empty,
                used_any.all(dim=-1),
                edges_closed,
            )
            logits = self._apply_mask(logits, allowed)

            if sample:
                probs = F.softmax(logits / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                tok = logits.argmax(dim=-1)

            tok = torch.where(finished, torch.full_like(tok, self.eos_id), tok)
            emitted.append(tok)

            is_agent = tok < N
            is_eoe = tok == self.eoe_id
            is_eos = tok == self.eos_id
            agent_safe = tok.clamp(max=N - 1)
            agent_oh = F.one_hot(agent_safe, num_classes=N).bool() & is_agent.unsqueeze(
                -1
            )
            used_any = used_any | agent_oh
            used_cur = used_cur | agent_oh
            zero_n = torch.zeros_like(used_cur)
            used_cur = torch.where(
                is_eoe.unsqueeze(-1).expand_as(used_cur), zero_n, used_cur
            )
            edge_idx = edge_idx + is_eoe.long()
            edges_closed = edges_closed + is_eoe.long()
            true_b = torch.ones_like(cur_empty)
            false_b = torch.zeros_like(cur_empty)
            cur_empty = torch.where(
                is_eoe, true_b, torch.where(is_agent, false_b, cur_empty)
            )
            finished = finished | is_eos

        if not emitted:
            return [[] for _ in range(B)]
        seq = torch.stack(emitted, dim=1).cpu().tolist()
        return [self.tokens_to_edge_list(s, self.n_agents) for s in seq]

    @torch.no_grad()
    def generate_with_tokens(
        self,
        obs_history: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> tuple[list[list[tuple[int, ...]]], list[list[int]]]:
        """Same as :meth:`generate` but also returns the per-env token sequence
        (truncated at the first EOS, inclusive) so callers can recompute
        log-probabilities in the training loop.
        """
        B = obs_history.size(0)
        N = self.n_agents
        device = obs_history.device

        agent_emb = self.encode_agents(obs_history)
        memory = self._build_memory(agent_emb)

        used_any = torch.zeros(B, N, dtype=torch.bool, device=device)
        used_cur = torch.zeros(B, N, dtype=torch.bool, device=device)
        edge_idx = torch.zeros(B, dtype=torch.long, device=device)
        edges_closed = torch.zeros(B, dtype=torch.long, device=device)
        cur_empty = torch.ones(B, dtype=torch.bool, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        emitted: list[torch.Tensor] = []
        max_len = self.max_hyperedges * (N + 1) + 1

        for _ in range(max_len):
            if finished.all():
                break

            bos = torch.full((B, 1), -1, dtype=torch.long, device=device)
            if emitted:
                emitted_t = torch.stack(emitted, dim=1)
                prev = torch.cat([bos, emitted_t], dim=1)
                sc = self._scan_state(emitted_t)
                edge_indices_for_dec = torch.cat(
                    [sc["edge_indices"], edge_idx.unsqueeze(1)], dim=1
                )
            else:
                prev = bos
                edge_indices_for_dec = torch.zeros(
                    B, 1, dtype=torch.long, device=device
                )

            L = prev.size(1)
            h_in = self._embed_tokens(prev, memory, edge_indices_for_dec)
            causal = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
            )
            hidden = self.decoder(h_in, memory, tgt_mask=causal)
            logits = self._decode_logits(hidden[:, -1:], memory).squeeze(1)

            allowed = self._legal_mask(
                used_any, used_cur, cur_empty,
                used_any.all(dim=-1), edges_closed,
            )
            logits = self._apply_mask(logits, allowed)

            if sample:
                probs = F.softmax(logits / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                tok = logits.argmax(dim=-1)

            tok = torch.where(finished, torch.full_like(tok, self.eos_id), tok)
            emitted.append(tok)

            is_agent = tok < N
            is_eoe = tok == self.eoe_id
            is_eos = tok == self.eos_id
            agent_safe = tok.clamp(max=N - 1)
            agent_oh = F.one_hot(agent_safe, num_classes=N).bool() & is_agent.unsqueeze(-1)
            used_any = used_any | agent_oh
            used_cur = used_cur | agent_oh
            zero_n = torch.zeros_like(used_cur)
            used_cur = torch.where(
                is_eoe.unsqueeze(-1).expand_as(used_cur), zero_n, used_cur
            )
            edge_idx = edge_idx + is_eoe.long()
            edges_closed = edges_closed + is_eoe.long()
            true_b = torch.ones_like(cur_empty)
            false_b = torch.zeros_like(cur_empty)
            cur_empty = torch.where(
                is_eoe, true_b, torch.where(is_agent, false_b, cur_empty)
            )
            finished = finished | is_eos

        if not emitted:
            empty_edges = [[] for _ in range(B)]
            empty_toks = [[self.eos_id] for _ in range(B)]
            return empty_edges, empty_toks

        seq = torch.stack(emitted, dim=1).cpu().tolist()
        edge_lists = [self.tokens_to_edge_list(s, self.n_agents) for s in seq]
        # Truncate each per-env token sequence at (and including) its first EOS.
        truncated = []
        for s in seq:
            cut = len(s)
            for k, t in enumerate(s):
                if t == self.eos_id:
                    cut = k + 1
                    break
            truncated.append(s[:cut])
        return edge_lists, truncated

    # ------------------------------------------------------------------
    # Token <-> edge-list conversion
    # ------------------------------------------------------------------
    @staticmethod
    def tokens_to_edge_list(tokens: list[int], n_agents: int) -> list[tuple[int, ...]]:
        """Parse a token sequence into the list-of-tuples format used by
        algorithms/mappo/hypergraph.py and dhg.Hypergraph.
        """
        edges: list[tuple[int, ...]] = []
        cur: list[int] = []
        eoe = n_agents
        eos = n_agents + 1
        for t in tokens:
            t = int(t)
            if t < n_agents:
                cur.append(t)
            elif t == eoe:
                if cur:
                    edges.append(tuple(cur))
                    cur = []
            elif t == eos:
                break
        if cur:
            edges.append(tuple(cur))
        return edges


if __name__ == "__main__":
    torch.manual_seed(0)

    n_agents = 6
    observation_dim = 21
    history_length = 32
    batch_size = 2

    model = GroupingTransformer(
        n_agents=n_agents,
        observation_dim=observation_dim,
        history_length=history_length,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        allow_overlap=True,
        max_hyperedges=10,
    )
    model.eval()

    obs_history = torch.randn(batch_size, n_agents, history_length, observation_dim)

    print("=" * 60)
    print("Input")
    print("=" * 60)
    print(f"obs_history shape: {tuple(obs_history.shape)}")
    print(
        f"  (batch={batch_size}, n_agents={n_agents}, "
        f"history={history_length}, obs_dim={observation_dim})"
    )
    for i in range(n_agents):
        print(f"agent {i} sequence:\n{obs_history[0, i]}")

    edge_lists = model.generate(obs_history, sample=False)

    with torch.no_grad():
        agent_emb = model.encode_agents(obs_history)
        memory = model._build_memory(agent_emb)
        B = obs_history.size(0)
        device = obs_history.device

        used_any = torch.zeros(B, n_agents, dtype=torch.bool, device=device)
        used_cur = torch.zeros(B, n_agents, dtype=torch.bool, device=device)
        edge_idx = torch.zeros(B, dtype=torch.long, device=device)
        edges_closed = torch.zeros(B, dtype=torch.long, device=device)
        cur_empty = torch.ones(B, dtype=torch.bool, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        emitted: list[torch.Tensor] = []
        max_len = model.max_hyperedges * (n_agents + 1) + 1

        for _ in range(max_len):
            if finished.all():
                break
            bos = torch.full((B, 1), -1, dtype=torch.long, device=device)
            if emitted:
                emitted_t = torch.stack(emitted, dim=1)
                prev = torch.cat([bos, emitted_t], dim=1)
                sc = model._scan_state(emitted_t)
                edge_indices_for_dec = torch.cat(
                    [sc["edge_indices"], edge_idx.unsqueeze(1)], dim=1
                )
            else:
                prev = bos
                edge_indices_for_dec = torch.zeros(
                    B, 1, dtype=torch.long, device=device
                )
            L = prev.size(1)
            h_in = model._embed_tokens(prev, memory, edge_indices_for_dec)
            causal = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
            )
            hidden = model.decoder(h_in, memory, tgt_mask=causal)
            logits = model._decode_logits(hidden[:, -1:], memory).squeeze(1)
            allowed = model._legal_mask(
                used_any,
                used_cur,
                cur_empty,
                used_any.all(dim=-1),
                edges_closed,
            )
            logits = model._apply_mask(logits, allowed)
            tok = logits.argmax(dim=-1)
            tok = torch.where(finished, torch.full_like(tok, model.eos_id), tok)
            emitted.append(tok)

            is_agent = tok < n_agents
            is_eoe = tok == model.eoe_id
            is_eos = tok == model.eos_id
            agent_safe = tok.clamp(max=n_agents - 1)
            agent_oh = F.one_hot(
                agent_safe, num_classes=n_agents
            ).bool() & is_agent.unsqueeze(-1)
            used_any = used_any | agent_oh
            used_cur = used_cur | agent_oh
            zero_n = torch.zeros_like(used_cur)
            used_cur = torch.where(
                is_eoe.unsqueeze(-1).expand_as(used_cur), zero_n, used_cur
            )
            edge_idx = edge_idx + is_eoe.long()
            edges_closed = edges_closed + is_eoe.long()
            true_b = torch.ones_like(cur_empty)
            false_b = torch.zeros_like(cur_empty)
            cur_empty = torch.where(
                is_eoe, true_b, torch.where(is_agent, false_b, cur_empty)
            )
            finished = finished | is_eos

        token_seq = torch.stack(emitted, dim=1).cpu().tolist()

    print()
    print("=" * 60)
    print("Output")
    print("=" * 60)
    print(f"agent embeddings shape: {tuple(agent_emb.shape)}")
    print(f"raw token sequence (per batch): {token_seq}")
    print(
        f"  vocab: agent ids 0..{n_agents - 1}, "
        f"EOE={model.eoe_id}, EOS={model.eos_id}"
    )
    print(f"decoded hyperedges: {edge_lists}")
