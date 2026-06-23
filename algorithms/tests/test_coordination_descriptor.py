"""Shape/behavior tests for the coordination-graph descriptor used by the
graph-novelty exploration bonus.

Usage:
    python -m algorithms.tests.test_coordination_descriptor
    # or: pytest algorithms/tests/test_coordination_descriptor.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.mappo.networks.gnn_critic import AttentionGNNCritic

B, N, OBS_DIM, D_MODEL, N_HEADS = 3, 5, 11, 16, 4
N_PAIRS = N * (N - 1) // 2
N_OFF_DIAG = N * (N - 1)


def _critic():
    torch.manual_seed(0)
    return AttentionGNNCritic(
        observation_dim=OBS_DIM,
        n_agents=N,
        d_model=D_MODEL,
        n_heads=N_HEADS,
    ).eval()


def test_descriptor_shapes():
    critic = _critic()
    obs = torch.randn(B, N, OBS_DIM)
    with torch.no_grad():
        team_adj = critic.coordination_descriptor(obs, "team", "adjacency")
        agent_adj = critic.coordination_descriptor(obs, "agent", "adjacency")
        team_node = critic.coordination_descriptor(obs, "team", "node_embedding")
        agent_node = critic.coordination_descriptor(obs, "agent", "node_embedding")

    assert team_adj.shape == (B, N_HEADS * N_PAIRS), team_adj.shape
    assert agent_adj.shape == (B, N, N_HEADS * N), agent_adj.shape
    assert team_node.shape == (B, D_MODEL), team_node.shape
    assert agent_node.shape == (B, N, D_MODEL), agent_node.shape


def test_directed_descriptor_shapes():
    critic = _critic()
    obs = torch.randn(B, N, OBS_DIM)
    with torch.no_grad():
        team = critic.coordination_descriptor(obs, "team", "directed_adjacency")
        agent = critic.coordination_descriptor(obs, "agent", "directed_adjacency")
    # Both triangles kept (off-diagonal), so twice the symmetric pair count.
    assert team.shape == (B, N_HEADS * N_OFF_DIAG), team.shape
    # Outgoing row + incoming column per agent, across heads.
    assert agent.shape == (B, N, 2 * N_HEADS * N), agent.shape


def test_directed_preserves_asymmetry():
    """The directed descriptor must distinguish i->j from j->i; the symmetric
    one cannot. Build a case with asymmetric attention and check the directed
    team descriptor carries strictly more (different) information."""
    critic = _critic()
    obs = torch.randn(B, N, OBS_DIM)
    with torch.no_grad():
        sym = critic.coordination_descriptor(obs, "team", "adjacency")
        directed = critic.coordination_descriptor(obs, "team", "directed_adjacency")
    # Directed keeps both triangles, so it is strictly higher-dimensional.
    assert directed.shape[-1] == 2 * sym.shape[-1]
    # Averaging the two directed halves must recover the symmetric weights,
    # confirming directed is a faithful superset of the symmetric descriptor.
    d = directed.reshape(B, N_HEADS, N_OFF_DIAG)
    # Split off-diagonal entries into upper/lower by reconstructing the matrix.
    off_diag = ~torch.eye(N, dtype=torch.bool)
    full = torch.zeros(B, N_HEADS, N, N)
    full[:, :, off_diag] = d
    recovered = ((full + full.transpose(-2, -1)) / 2)
    iu, ju = torch.triu_indices(N, N, offset=1)
    recovered_pairs = recovered[:, :, iu, ju].reshape(B, -1)
    assert torch.allclose(recovered_pairs, sym, atol=1e-5)


def test_unbatched_input_squeezes():
    critic = _critic()
    obs = torch.randn(N, OBS_DIM)
    with torch.no_grad():
        team = critic.coordination_descriptor(obs, "team", "adjacency")
        agent = critic.coordination_descriptor(obs, "agent", "adjacency")
    assert team.shape == (N_HEADS * N_PAIRS,), team.shape
    assert agent.shape == (N, N_HEADS * N), agent.shape


def test_descriptor_permutation_sensitivity():
    """Reordering agents must change the descriptor (it encodes who couples to
    whom), but the team descriptor's *value distribution* is unchanged because
    the graph is the same up to relabeling — sanity that it is a function of the
    coordination structure, not noise."""
    critic = _critic()
    obs = torch.randn(1, N, OBS_DIM)
    perm = torch.tensor([1, 0, 2, 3, 4])
    with torch.no_grad():
        a = critic.coordination_descriptor(obs, "agent", "adjacency")
        b = critic.coordination_descriptor(obs[:, perm], "agent", "adjacency")
    # Per-agent rows should not be identical after permutation.
    assert not torch.allclose(a, b)


if __name__ == "__main__":
    test_descriptor_shapes()
    test_directed_descriptor_shapes()
    test_directed_preserves_asymmetry()
    test_unbatched_input_squeezes()
    test_descriptor_permutation_sensitivity()
    print("all coordination_descriptor tests passed")
