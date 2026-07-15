"""Decentralized skill refinement by message passing over a proximity graph.

Each agent independently proposes a skill (the actor's per-agent logits over
``SKILL_ORDER``), then *refines* that choice by exchanging its intended skill
with the agents inside its communication radius. After ``n_rounds`` of message
passing the refined logits are returned, and the usual categorical policy takes
over.

The point is coordination without a coordinator. VO-MASD reaches the same goal
with a centralized "grouper" that reads global state and partitions agents into
groups that share a joint skill code; here the equivalent structure emerges from
local message passing over the ``info["adjacency"]`` graph published by the env.
Nothing outside an agent's neighbourhood can influence its decision, so execution
stays decentralized.

Round ``k`` (for agent ``i``, with neighbourhood ``N(i)`` given by the adjacency):

    e_k^i    = codebook[c_k^i]                  # embedding of i's current pick
    m_k^i    = AGG_{j in N(i)} f(h^j, e_k^j)    # what the neighbours intend
    logits   = logits_0 + head([h^i, e_k^i, m_k^i])
    c_{k+1}^i = argmax(logits)                  # straight-through

The **codebook is a learned communication vocabulary over grounded behaviours**:
the skills themselves are fixed, pre-trained actors (`skills.py`), so a symbol
denotes a concrete macro-behaviour rather than an arbitrary emergent code. Only
the embedding used to *talk about* a skill is learned.

Two properties are deliberate:

- **Refinement is residual and zero-initialized.** ``head`` ends in a zero-init
  layer, so at initialization the refiner is an exact identity on ``logits_0``.
  Training therefore starts byte-for-byte at the independent-selection baseline
  and has to *learn* to deviate — which also makes the skill-change rate a clean
  read on whether message passing is doing anything at all.
- **``n_rounds=0`` is an exact no-op**, returning ``logits_0`` untouched without
  consuming any parameters. That is the "no communication" ablation.

Information travels exactly one hop per round, so ``n_rounds`` is the coordination
horizon: an agent's refined logits depend only on its ``n_rounds``-hop
neighbourhood.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.mappo.networks.gnn_critic import DenseGCNConv
from algorithms.mappo.networks.multi_head_attention import (
    MultiHeadAttention,
    ObservationEncoder,
)
from algorithms.mappo.networks.utils import layer_init


def straight_through_onehot(logits: torch.Tensor) -> torch.Tensor:
    """Hard one-hot in the forward pass, softmax gradient in the backward pass.

    Lets an agent broadcast a genuinely discrete symbol ("I intend to run skill
    #3") while keeping the message differentiable — the same trick VQ-VAE uses to
    push gradients through ``argmin`` quantization.
    """
    probs = F.softmax(logits, dim=-1)
    index = probs.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(probs).scatter_(-1, index, 1.0)
    return hard + probs - probs.detach()


class SkillRefiner(nn.Module):
    """K rounds of message passing over a proximity graph to refine skill logits.

    Args:
        obs_dim:         per-agent observation dim.
        n_skills:        size of the skill vocabulary (``len(SKILL_ORDER)``).
        n_rounds:        K, the number of message-passing rounds (0 = no-op).
        d_model:         hidden width of the message/refinement network.
        skill_embed_dim: width of the learned per-skill communication embedding.
        aggregator:      ``"gcn"`` (degree-normalized mean over neighbours, via
                         the existing :class:`DenseGCNConv`) or ``"attention"``
                         (multi-head attention restricted to graph edges, so an
                         agent can weight *which* neighbours matter).
        message_mode:    ``"hard"`` broadcasts a straight-through one-hot symbol
                         (discrete, bandwidth-limited); ``"soft"`` broadcasts the
                         softmax-weighted embedding (lower-variance gradients —
                         the fallback if straight-through proves too noisy).
        n_heads:         attention heads, when ``aggregator="attention"``.
        share_round_weights: reuse one set of round weights for all K rounds
                         (fewer params, and K becomes a pure inference-time knob).
    """

    def __init__(
        self,
        obs_dim: int,
        n_skills: int,
        n_rounds: int = 2,
        d_model: int = 64,
        skill_embed_dim: int = 16,
        aggregator: str = "gcn",
        message_mode: str = "hard",
        n_heads: int = 4,
        share_round_weights: bool = False,
    ):
        super().__init__()
        if aggregator not in ("gcn", "attention"):
            raise ValueError(
                f"aggregator must be 'gcn' or 'attention', got {aggregator!r}"
            )
        if message_mode not in ("hard", "soft"):
            raise ValueError(
                f"message_mode must be 'hard' or 'soft', got {message_mode!r}"
            )

        self.n_skills = n_skills
        self.n_rounds = int(n_rounds)
        self.aggregator = aggregator
        self.message_mode = message_mode

        # n_rounds=0 is an exact no-op: build nothing, so the ablation carries no
        # dead parameters into the optimizer.
        if self.n_rounds == 0:
            return

        self.obs_encoder = ObservationEncoder(obs_dim, d_model)
        # The communication vocabulary: one learned embedding per grounded skill.
        self.codebook = nn.Embedding(n_skills, skill_embed_dim)

        n_blocks = 1 if share_round_weights else self.n_rounds
        self.share_round_weights = share_round_weights

        # What an agent puts on the wire: its own features + its intended skill.
        self.message_proj = nn.ModuleList(
            [
                layer_init(nn.Linear(d_model + skill_embed_dim, d_model))
                for _ in range(n_blocks)
            ]
        )

        if aggregator == "gcn":
            # add_self_loops=False: the env's adjacency already carries self-loops,
            # so adding I again would double-count the agent's own message.
            self.agg = nn.ModuleList(
                [
                    DenseGCNConv(d_model, d_model, add_self_loops=False)
                    for _ in range(n_blocks)
                ]
            )
        else:
            self.agg = nn.ModuleList(
                [MultiHeadAttention(d_model, n_heads) for _ in range(n_blocks)]
            )

        # Refinement head: [own features, own intent, neighbours' intent] -> delta.
        # Zero-init the last layer so the refiner starts as an exact identity.
        self.head = nn.ModuleList(
            [
                nn.Sequential(
                    layer_init(
                        nn.Linear(d_model + skill_embed_dim + d_model, d_model)
                    ),
                    nn.GELU(),
                    layer_init(nn.Linear(d_model, n_skills), std=0.0, bias_const=0.0),
                )
                for _ in range(n_blocks)
            ]
        )

    def _block(self, k: int) -> int:
        return 0 if self.share_round_weights else k

    def _messages(self, logits: torch.Tensor) -> torch.Tensor:
        """Embed each agent's current skill decision into a message symbol."""
        if self.message_mode == "hard":
            onehot = straight_through_onehot(logits)  # (B, N, n_skills)
            return onehot @ self.codebook.weight  # (B, N, skill_embed_dim)
        return F.softmax(logits, dim=-1) @ self.codebook.weight

    def forward(
        self,
        obs: torch.Tensor,
        logits: torch.Tensor,
        adjacency: torch.Tensor,
        return_info: bool = False,
    ):
        """
        Args:
            obs:       (B, N, obs_dim) per-agent observations.
            logits:    (B, N, n_skills) the actor's independent skill proposals.
            adjacency: (B, N, N) 0/1 proximity graph with self-loops, as published
                       on ``info["adjacency"]``.
            return_info: also return a diagnostics dict.
        Returns:
            refined logits (B, N, n_skills), and optionally a dict with
            ``skill_change_rate`` — the fraction of agents whose argmax skill was
            flipped by message passing. If this is ~0 the refiner has collapsed to
            the identity and communication is buying nothing.
        """
        if self.n_rounds == 0:
            if return_info:
                return logits, {
                    "skill_change_rate": torch.zeros((), device=logits.device)
                }
            return logits

        initial_logits = logits
        h = self.obs_encoder(obs)  # (B, N, d_model)

        # Edge mask for attention: True where agent i may attend to agent j.
        edge_mask = adjacency > 0 if self.aggregator == "attention" else None

        cur_logits = initial_logits
        for k in range(self.n_rounds):
            b = self._block(k)

            e = self._messages(cur_logits)  # (B, N, skill_embed_dim)
            outgoing = self.message_proj[b](torch.cat([h, e], dim=-1))

            if self.aggregator == "gcn":
                m = self.agg[b](outgoing, adjacency)  # (B, N, d_model)
            else:
                m, _ = self.agg[b](outgoing, edge_mask)

            # Residual on the *initial* proposal, not the running one: every round
            # re-decides from the actor's proposal in light of richer neighbourhood
            # context, which keeps the K-round map from drifting far from the
            # policy's own preference.
            delta = self.head[b](torch.cat([h, e, m], dim=-1))
            cur_logits = initial_logits + delta

        if return_info:
            changed = (
                cur_logits.argmax(-1) != initial_logits.argmax(-1)
            ).float().mean()
            return cur_logits, {"skill_change_rate": changed.detach()}
        return cur_logits
