"""Visualize cross-attention scores from a trained HGNNCrossAttentionCritic.

Loads a checkpoint from the multi_box_push_12a_6o_dense/hg_cross_attention
experiment, runs one episode, and produces a figure with 4 environment
snapshots alongside heatmaps of the attention weights between the two
hypergraph-type token sequences (proximity and object).

Usage:
    python -m algorithms.tests.test_cross_attention_visualization
"""

import math
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.mappo.networks.models import MAPPONetwork
from algorithms.mappo.hypergraph import (
    batch_hypergraphs,
    canonicalize_edge_lists,
    distance_based_hyperedges,
    object_contact_hyperedges,
)
from algorithms.mappo.hg_cache import HypergraphCache
from algorithms.mappo.entropy_helpers import update_left_padded_history
from environments.box2d_suite.multi_box_push import MultiBoxPushEnv

# ── Experiment configuration (matches hg_cross_attention.yaml + _env.yaml) ──

N_AGENTS = 12
N_OBJECTS = 6
OBSERVATION_DIM = 21
ACTION_DIM = 2
HIDDEN_DIM = 80
N_HYPEREDGE_TYPES = 2
CRITIC_SEQ_LEN = 32  # default from Model_Params
REWARD_MODE = "dense"
DEVICE = "cpu"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments/results/multi_box_push_12a_6o_dense"
    / "hg_cross_attention/0/models/models_checkpoint.pth"
)

HYPEREDGE_TYPE_NAMES = ["proximity", "object"]
N_SNAPSHOTS = 4
OUTPUT_PATH = PROJECT_ROOT / "algorithms/tests/cross_attention_heatmaps.png"


# ── Attention hook ───────────────────────────────────────────────────────────

class AttentionCapture:
    """Register forward hooks on MultiHeadCrossAttention modules to capture
    attention weights (post-softmax, pre-dropout)."""

    def __init__(self):
        self.scores = {}  # {(layer_idx, block_idx): (B, n_heads, T_q, T_kv)}
        self._handles = []

    def register(self, critic):
        for layer_idx, layer in enumerate(critic.cross_attn_layers):
            for block_idx, block in enumerate(layer.blocks):
                attn_module = block.cross_attn
                handle = attn_module.register_forward_hook(
                    self._make_hook(layer_idx, block_idx)
                )
                self._handles.append(handle)

    def _make_hook(self, layer_idx, block_idx):
        def hook(module, inputs, output):
            query, key_value = inputs
            B, T_q, _ = query.shape
            T_kv = key_value.size(1)
            d_k = module.d_k

            Q = module.W_q(query).view(B, T_q, module.n_heads, d_k).transpose(1, 2)
            K = module.W_k(key_value).view(B, T_kv, module.n_heads, d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            attn = F.softmax(scores, dim=-1)
            self.scores[(layer_idx, block_idx)] = attn.detach().cpu()

        return hook

    def clear(self):
        self.scores.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Environment helpers ──────────────────────────────────────────────────────

def build_hypergraphs(obs, info, n_agents, device):
    """Build batched block-diagonal hypergraphs and signature for one env."""
    hyperedge_fns = [
        (partial(distance_based_hyperedges, threshold=1.0), "obs"),
        (object_contact_hyperedges, "agents_2_objects"),
    ]

    edge_lists_per_type = []
    for fn, source in hyperedge_fns:
        data = obs if source == "obs" else info.get(source)
        if data is None:
            continue
        edge_lists_per_type.append(fn(data, n_agents))

    batched_hgs = []
    for type_edges in edge_lists_per_type:
        hg = batch_hypergraphs([type_edges], n_agents, device=device)
        batched_hgs.append(hg)

    sig = canonicalize_edge_lists(edge_lists_per_type)
    return batched_hgs, edge_lists_per_type, sig


def capture_frame(env):
    """Grab the current pygame surface as an RGB numpy array."""
    import pygame

    surface = pygame.display.get_surface()
    if surface is None:
        return None
    frame = pygame.surfarray.array3d(surface)
    return np.transpose(frame, (1, 0, 2)).copy()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Build network and load checkpoint ──
    global_state_dim = OBSERVATION_DIM * N_AGENTS
    network = MAPPONetwork(
        observation_dim=OBSERVATION_DIM,
        global_state_dim=global_state_dim,
        action_dim=ACTION_DIM,
        n_agents=N_AGENTS,
        hidden_dim=HIDDEN_DIM,
        discrete=False,
        share_actor=True,
        critic_type="hg_cross_attention",
        n_hyperedge_types=N_HYPEREDGE_TYPES,
        critic_seq_len=CRITIC_SEQ_LEN,
        entropy_conditioning=False,
        hypergraph_mode="predefined",
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    network.load_state_dict(checkpoint["network"])
    network.eval()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    critic = network.critic

    # ── Register attention hooks ──
    attn_capture = AttentionCapture()
    attn_capture.register(critic)

    # ── Set up environment ──
    env = MultiBoxPushEnv(
        render_mode="human",
        n_agents=N_AGENTS,
        n_objects=N_OBJECTS,
        reward_mode=REWARD_MODE,
    )
    obs, info = env.reset(seed=42)
    env.render()

    # ── Hypergraph cache for temporal windows ──
    hg_cache = HypergraphCache(n_agents=N_AGENTS, n_parallel_envs=1)

    # ── Temporal history buffers ──
    critic_obs_history = torch.zeros(
        1, CRITIC_SEQ_LEN, N_AGENTS, OBSERVATION_DIM, dtype=torch.float32
    )
    critic_sig_history = torch.zeros(1, CRITIC_SEQ_LEN, dtype=torch.long)
    critic_history_counts = torch.zeros(1, dtype=torch.long)

    # ── Run episode ──
    frames = []
    all_attention_scores = []  # list of dicts per step
    cum_reward = 0.0
    rewards_over_time = []

    step = 0
    with torch.no_grad():
        while True:
            # Build hypergraphs
            batched_hgs, edge_lists, _ = build_hypergraphs(
                obs, info, N_AGENTS, DEVICE
            )

            # Cache signature
            sig_id = hg_cache.intern(edge_lists)

            # Update temporal history
            obs_step = torch.from_numpy(
                np.ascontiguousarray(obs, dtype=np.float32)
            ).unsqueeze(0)
            sig_step = torch.tensor([sig_id], dtype=torch.long)
            prev_counts = critic_history_counts.clone()
            critic_obs_history, critic_history_counts = update_left_padded_history(
                critic_obs_history, obs_step, critic_history_counts
            )
            critic_sig_history, _ = update_left_padded_history(
                critic_sig_history, sig_step, prev_counts
            )

            # Build temporal hypergraphs for critic
            critic_hgs = hg_cache.build_sequence_batched_hypergraphs(
                critic_sig_history, device=DEVICE
            )

            # Forward pass through critic (triggers attention hooks)
            attn_capture.clear()
            _ = critic.forward_batched(
                critic_obs_history.clone(),
                critic_hgs,
                n_graphs=1,
            )

            # Capture frame
            frame = capture_frame(env)
            frames.append(frame)

            # Store attention scores
            all_attention_scores.append(
                {k: v.clone() for k, v in attn_capture.scores.items()}
            )

            # Get actions
            obs_tensor = torch.from_numpy(
                np.ascontiguousarray(obs, dtype=np.float32)
            ).to(DEVICE)
            obs_flat = obs_tensor.reshape(1 * N_AGENTS, -1)
            actions_flat, _ = network.act(obs_flat, agent_idx=0, deterministic=True)
            actions = actions_flat.reshape(N_AGENTS, ACTION_DIM).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(actions)
            env.render()

            cum_reward += float(reward)
            rewards_over_time.append(cum_reward)

            step += 1
            if terminated or truncated:
                break

    env.close()
    attn_capture.remove_hooks()

    total_steps = len(frames)
    print(f"Episode finished after {total_steps} steps, cumulative reward: {cum_reward:.2f}")

    # ── Select 4 snapshot indices ──
    snap_indices = np.linspace(0, total_steps - 1, N_SNAPSHOTS, dtype=int)

    # ── Build figure ──
    # Layout: 4 columns (one per snapshot)
    # Row 0: environment frame
    # Row 1: Layer 0 — proximity attends to object  (block 0)
    # Row 2: Layer 0 — object attends to proximity  (block 1)
    # Row 3: Layer 1 — proximity attends to object  (block 0)
    # Row 4: Layer 1 — object attends to proximity  (block 1)
    n_layers = len(critic.cross_attn_layers)
    n_blocks = N_HYPEREDGE_TYPES
    n_rows = 1 + n_layers * n_blocks

    fig = plt.figure(figsize=(5 * N_SNAPSHOTS, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, N_SNAPSHOTS, hspace=0.35, wspace=0.3)

    for col, t in enumerate(snap_indices):
        # Row 0: environment snapshot
        ax_frame = fig.add_subplot(gs[0, col])
        if frames[t] is not None:
            ax_frame.imshow(frames[t])
        ax_frame.set_title(f"t = {t}", fontsize=13, fontweight="bold")
        ax_frame.axis("off")

        # Attention heatmaps
        scores = all_attention_scores[t]
        row = 1
        for layer_idx in range(n_layers):
            for block_idx in range(n_blocks):
                ax = fig.add_subplot(gs[row, col])
                attn = scores[(layer_idx, block_idx)]
                # attn shape: (1, n_heads, T_q, T_kv)
                # Average over heads for display
                attn_avg = attn[0].mean(dim=0).numpy()  # (T_q, T_kv)

                # Only show the filled portion of the window
                valid_len = min(t + 1, CRITIC_SEQ_LEN)
                attn_display = attn_avg[-valid_len:, -valid_len:]

                im = ax.imshow(
                    attn_display,
                    aspect="auto",
                    cmap="viridis",
                    vmin=0,
                    vmax=attn_display.max(),
                    interpolation="nearest",
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                query_type = HYPEREDGE_TYPE_NAMES[block_idx]
                kv_type = HYPEREDGE_TYPE_NAMES[1 - block_idx]
                if col == 0:
                    ax.set_ylabel(
                        f"L{layer_idx}: {query_type}\n-> {kv_type}",
                        fontsize=10,
                    )
                ax.set_xlabel("KV token (timestep)")
                if col == 0:
                    ax.set_ylabel(
                        f"L{layer_idx}: {query_type} -> {kv_type}\nQ token (timestep)",
                        fontsize=10,
                    )

                row += 1

    fig.suptitle(
        "HGNNCrossAttentionCritic — Attention Scores Across Episode Snapshots",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
