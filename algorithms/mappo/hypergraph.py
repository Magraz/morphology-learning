import numpy as np
import dhg
from collections import Counter
from typing import Callable


def distance_based_hyperedges(
    obs: np.ndarray, threshold: float
) -> list[tuple]:
    """
    Build a hyperedge list from agent observations using a distance threshold.

    Extracts agent positions from obs[:, :2] (indices 0=x, 1=y), then groups
    agents into hyperedges based on pairwise Euclidean distance. Each agent i
    defines one hyperedge containing itself and every agent within `threshold`.
    Duplicate hyperedges from symmetric neighbourhoods are removed. Falls back
    to isolated self-loops when no multi-agent hyperedges exist.

    Args:
        obs:       Agent observations of shape (n_agents, obs_dim).
                   Positions are at obs_dim indices 0 (x) and 1 (y).
        threshold: Euclidean distance threshold for hyperedge membership.

    Returns:
        List of tuples, each tuple being a hyperedge (sequence of agent indices).
    """
    n_agents = obs.shape[0]
    pos = obs[:, :2]  # (n_agents, 2)

    # Vectorised pairwise distance matrix: (n_agents, n_agents)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 2)
    dist_matrix = np.linalg.norm(diff, axis=-1)            # (N, N)

    hyperedge_list = []
    for i in range(n_agents):
        members = tuple(np.where(dist_matrix[i] <= threshold)[0].tolist())
        if len(members) > 1:
            hyperedge_list.append(members)

    # Deduplicate: symmetric neighbourhoods produce identical hyperedges
    hyperedge_list = list(set(hyperedge_list))

    # Fallback: if no group hyperedges exist, every agent is isolated
    if not hyperedge_list:
        hyperedge_list = [(i,) for i in range(n_agents)]

    return hyperedge_list


def build_hypergraph_from_obs(
    obs: np.ndarray,
    hyperedge_fn: Callable[[np.ndarray], list[tuple]],
) -> list[dhg.Hypergraph]:
    """
    Build hypergraphs from vectorised environment observations.

    Args:
        obs:          Observations of shape (n_envs, n_agents, obs_dim).
        hyperedge_fn: Callable that accepts a full observation slice of shape
                      (n_agents, obs_dim) and returns a list of hyperedge tuples.
                      The function is responsible for extracting whatever fields
                      it needs from the observation. Use functools.partial to
                      bind extra parameters, e.g.:
                        partial(distance_based_hyperedges, threshold=1.0)

    Returns:
        A list of dhg.Hypergraph objects, one per environment.
    """
    n_envs, n_agents, _ = obs.shape

    hypergraphs = []
    for env_idx in range(n_envs):
        hyperedge_list = hyperedge_fn(obs[env_idx])  # (n_agents, obs_dim)
        hg = dhg.Hypergraph(n_agents, hyperedge_list)
        hypergraphs.append(hg)

    return hypergraphs


def compute_hyperedge_structural_entropy_batch(
    hypergraphs: list,
) -> np.ndarray:
    """
    Compute hyperedge structural entropy for a batch of dhg.Hypergraph objects.

    Args:
        hypergraphs: List of dhg.Hypergraph instances (length n_hypergraphs).

    Returns:
        entropies: np.ndarray of shape (n_hypergraphs, 2).
                   Column 0: S_e          — raw Shannon entropy (nats).
                   Column 1: S_normalized — entropy normalised by ln(E_total).
    """
    results = []
    for hg in hypergraphs:
        S_e, S_norm, _ = compute_hyperedge_structural_entropy(hg)
        results.append([S_e, S_norm])
    return np.array(results, dtype=np.float64)  # (n_hypergraphs, 2)


def compute_hyperedge_structural_entropy(hg: dhg.Hypergraph):
    """
    Compute hyperedge structural entropy S_e for a dhg.Hypergraph.

    Args:
        hg: dhg.Hypergraph instance.

    Returns:
        S_e:          float, raw Shannon entropy (nats).
        S_normalized: float, entropy normalised by ln(E_total).
        weights:      np.ndarray of per-hyperedge weights.
    """
    # A: (E_total, |V|) incidence matrix
    A = hg.H_T.to_dense().numpy().astype(np.float64)  # H_T shape is (num_e, num_v)
    E_total = hg.num_e

    # Step 1: Node hyperdegrees D(i) = column sums of A
    D = A.sum(axis=0)  # shape (|V|,)

    # Step 2: Hyperedge weights w(e) = O(e) * sum_{i in n_e} D(i)
    O = A.sum(axis=1)  # hyperedge orders, shape (E_total,)
    sum_D = A @ D  # sum of node hyperdegrees per edge, shape (E_total,)
    weights = O * sum_D  # shape (E_total,)

    # Step 3: Probability distribution P(w) = n_w / E_total
    weight_counts = Counter(weights.tolist())
    unique_weights = np.array(list(weight_counts.keys()))
    counts = np.array(list(weight_counts.values()), dtype=np.float64)
    P = counts / E_total

    # Step 4: Shannon entropy S_e = -sum P(w) ln P(w)
    S_e = -np.sum(P * np.log(P))

    # Normalization: S_max = ln(E_total)
    S_max = np.log(E_total) if E_total > 1 else 1.0
    S_normalized = S_e / S_max if S_max > 0 else 0.0

    return S_e, S_normalized, weights
