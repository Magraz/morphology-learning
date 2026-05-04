"""Counterfactual visualization of a trained MultiHGNNCritic.

Loads the policy + critic from the multi_box_push_12a_6o_heavy / hgnn_shared
experiment, rolls out a single episode, and at N_SNAPSHOTS evenly spaced
timesteps captures the rendered frame plus one hypergraph drawing per
hyperedge type. Each hypergraph is labeled with its type name and the
critic value produced when that type's hypergraph is the only relational
structure provided to the critic.

For each (snapshot, type) we also build a counterfactual hypergraph by
greedily removing the single multi-vertex hyperedge whose deletion causes
the largest drop in critic value, and draw it next to the original.

Usage:
    python -m algorithms.tests.test_counterfactual_critic
"""

import io
import sys
from pathlib import Path

import dhg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.mappo.hypergraph import (
    distance_based_hyperedges,
    object_contact_hyperedges,
)
from algorithms.mappo.networks.models import MAPPONetwork
from environments.box2d_suite.multi_box_push import MultiBoxPushEnv

# ── Experiment configuration (matches hgnn_shared.yaml + _env.yaml) ──────────

N_AGENTS = 12
N_OBJECTS = 6
OBSERVATION_DIM = 21
ACTION_DIM = 2
HIDDEN_DIM = 168
N_HYPEREDGE_TYPES = 2
REWARD_MODE = "dense"
DEVICE = "cpu"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments/results/multi_box_push_12a_6o_heavy"
    / "hgnn_shared/0/models/models_checkpoint.pth"
)

# Order must match hgnn_shared.yaml `hyperedge_fn_names`: ["proximity", "contact"]
HYPEREDGE_TYPE_NAMES = ["proximity", "contact"]
N_SNAPSHOTS = 4
OUTPUT_PATH = PROJECT_ROOT / "algorithms/tests/counterfactual_critic.png"


# ── Helpers ──────────────────────────────────────────────────────────────────


def build_per_type_edge_lists(obs, info, n_agents):
    """Return one hyperedge list per type, in HYPEREDGE_TYPE_NAMES order."""
    proximity_edges = distance_based_hyperedges(obs, n_agents, threshold=1.0)
    contact_edges = object_contact_hyperedges(
        info["agents_2_objects"], n_agents
    )
    return [proximity_edges, contact_edges]


def make_hypergraph(edges, n_agents, device):
    """Build a single dhg.Hypergraph; ensure it has at least one edge."""
    if len(edges) == 0:
        edges = [(i,) for i in range(n_agents)]
    return dhg.Hypergraph(n_agents, edges, device=device)


def capture_frame(env):
    """Grab the current pygame surface as an RGB numpy array."""
    import pygame

    surface = pygame.display.get_surface()
    if surface is None:
        return None
    frame = pygame.surfarray.array3d(surface)
    return np.transpose(frame, (1, 0, 2)).copy()


def find_max_drop_edge(critic, X, edges, original_value, n_agents, device):
    """Return the multi-vertex hyperedge whose removal causes the largest
    drop in critic value, alongside the resulting counterfactual edge list
    and value.

    Returns (cf_edges, cf_value, removed_edge) or (None, None, None) if no
    multi-vertex hyperedge exists.
    """
    candidate_indices = [i for i, e in enumerate(edges) if len(e) > 1]
    if not candidate_indices:
        return None, None, None

    best = None  # (drop, idx, cf_value, cf_edges)
    for i in candidate_indices:
        cf_edges = [e for j, e in enumerate(edges) if j != i]
        if not cf_edges:
            cf_edges = [(k,) for k in range(n_agents)]
        cf_hg = dhg.Hypergraph(n_agents, cf_edges, device=device)
        cf_value = critic(X, [cf_hg] * N_HYPEREDGE_TYPES).item()
        drop = original_value - cf_value
        if best is None or drop > best[0]:
            best = (drop, i, cf_value, cf_edges)

    _, removed_idx, cf_value, cf_edges = best
    return cf_edges, cf_value, edges[removed_idx]


def draw_hypergraph_to_array(edges, n_agents, device):
    """Render a dhg.Hypergraph to an RGB numpy array via an off-screen figure.

    dhg's force_layout chokes on hypergraphs that only contain self-loops
    (it produces NaN edge centers), so we keep only multi-vertex edges for
    visualization and add isolated self-loops for unconnected agents.
    Returns None when there is nothing meaningful to draw.
    """
    multi_edges = [e for e in edges if len(e) > 1]
    if not multi_edges:
        return None

    grouped = set()
    for e in multi_edges:
        grouped.update(e)
    viz_edges = list(multi_edges) + [(i,) for i in range(n_agents) if i not in grouped]

    viz_hg = dhg.Hypergraph(n_agents, viz_edges, device=device)
    viz_hg.draw(e_style="circle")
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return mpimg.imread(buf)


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
        critic_type="multi_hgnn",
        n_hyperedge_types=N_HYPEREDGE_TYPES,
        entropy_conditioning=False,
        hypergraph_mode="predefined",
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    network.load_state_dict(checkpoint["network"])
    network.eval()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    critic = network.critic

    # ── Set up environment ──
    env = MultiBoxPushEnv(
        render_mode="human",
        n_agents=N_AGENTS,
        n_objects=N_OBJECTS,
        reward_mode=REWARD_MODE,
    )
    obs, info = env.reset(seed=42)
    env.render()

    # ── Run episode, recording everything we need for per-step counterfactuals ──
    frames = []
    per_step_edge_lists = []  # list[step] of list[type] of edges
    per_step_obs = []         # list[step] of (n_agents, obs_dim)
    cum_reward = 0.0

    with torch.no_grad():
        while True:
            obs_np = np.ascontiguousarray(obs, dtype=np.float32)
            edge_lists = build_per_type_edge_lists(obs_np, info, N_AGENTS)

            frame = capture_frame(env)
            frames.append(frame)
            per_step_edge_lists.append(edge_lists)
            per_step_obs.append(obs_np)

            # Step the policy (deterministic).
            obs_tensor = torch.from_numpy(obs_np).to(DEVICE)
            obs_flat = obs_tensor.reshape(N_AGENTS, -1)
            actions_flat, _ = network.act(obs_flat, agent_idx=0, deterministic=True)
            actions = actions_flat.reshape(N_AGENTS, ACTION_DIM).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(actions)
            env.render()
            cum_reward += float(reward)

            if terminated or truncated:
                break

    env.close()

    total_steps = len(frames)
    print(
        f"Episode finished after {total_steps} steps, "
        f"cumulative reward: {cum_reward:.2f}"
    )

    # ── Pick snapshot indices ──
    snap_indices = np.linspace(0, total_steps - 1, N_SNAPSHOTS, dtype=int)

    # ── Compute counterfactual critic values + render hypergraphs ──
    # For each snapshot and each hyperedge type, replicate that type's
    # hypergraph into every type slot of the MultiHGNNCritic. The resulting
    # scalar is the value the critic predicts when *only* that relational
    # structure is available.
    snapshot_data = []
    with torch.no_grad():
        for t in snap_indices:
            edge_lists = per_step_edge_lists[t]
            X = torch.from_numpy(per_step_obs[t]).to(DEVICE)

            type_entries = []
            for type_idx, type_name in enumerate(HYPEREDGE_TYPE_NAMES):
                edges = edge_lists[type_idx]
                hg = make_hypergraph(edges, N_AGENTS, DEVICE)
                value = critic(X, [hg] * N_HYPEREDGE_TYPES).item()
                hg_image = draw_hypergraph_to_array(edges, N_AGENTS, DEVICE)

                cf_edges, cf_value, removed_edge = find_max_drop_edge(
                    critic, X, edges, value, N_AGENTS, DEVICE
                )
                cf_image = (
                    draw_hypergraph_to_array(cf_edges, N_AGENTS, DEVICE)
                    if cf_edges is not None
                    else None
                )

                type_entries.append(
                    {
                        "name": type_name,
                        "value": value,
                        "image": hg_image,
                        "n_edges": len(edges),
                        "cf_value": cf_value,
                        "cf_image": cf_image,
                        "cf_n_edges": len(cf_edges) if cf_edges is not None else None,
                        "removed_edge": removed_edge,
                    }
                )

            snapshot_data.append((int(t), type_entries))

    # ── Build figure ──
    # Layout per row: frame | (original | counterfactual) for each type.
    n_cols = 1 + 2 * N_HYPEREDGE_TYPES
    fig = plt.figure(figsize=(5 * n_cols, 5 * N_SNAPSHOTS))
    gs = gridspec.GridSpec(N_SNAPSHOTS, n_cols, hspace=0.35, wspace=0.2)

    for row, (t, type_entries) in enumerate(snapshot_data):
        ax_frame = fig.add_subplot(gs[row, 0])
        if frames[t] is not None:
            ax_frame.imshow(frames[t])
        ax_frame.set_title(f"t = {t}", fontsize=13, fontweight="bold")
        ax_frame.axis("off")

        for type_idx, entry in enumerate(type_entries):
            col_orig = 1 + 2 * type_idx
            col_cf = col_orig + 1

            # Original hypergraph.
            ax_orig = fig.add_subplot(gs[row, col_orig])
            if entry["image"] is not None:
                ax_orig.imshow(entry["image"])
            else:
                ax_orig.text(
                    0.5, 0.5, "(no edges)", ha="center", va="center",
                    transform=ax_orig.transAxes,
                )
            ax_orig.set_title(
                f"{entry['name']} (original)  |  V = {entry['value']:.3f}\n"
                f"({entry['n_edges']} hyperedges)",
                fontsize=12,
            )
            ax_orig.axis("off")

            # Counterfactual hypergraph (one max-drop edge removed).
            ax_cf = fig.add_subplot(gs[row, col_cf])
            if entry["cf_image"] is not None:
                ax_cf.imshow(entry["cf_image"])
                drop = entry["value"] - entry["cf_value"]
                title = (
                    f"{entry['name']} (counterfactual)  |  V = {entry['cf_value']:.3f}\n"
                    f"removed {tuple(entry['removed_edge'])}  Δ = {drop:+.3f}"
                )
            else:
                ax_cf.text(
                    0.5, 0.5,
                    "(no multi-vertex edge\nto remove)",
                    ha="center", va="center", transform=ax_cf.transAxes,
                )
                title = f"{entry['name']} (counterfactual)"
            ax_cf.set_title(title, fontsize=12)
            ax_cf.axis("off")

    fig.suptitle(
        "MultiHGNNCritic — Per-Type Counterfactual Values",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
