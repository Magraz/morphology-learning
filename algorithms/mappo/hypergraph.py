import numpy as np
import dhg
from collections import Counter
from typing import Callable


def canonicalize_edge_lists(edge_lists: list[list[tuple]]) -> tuple:
    """Return a hashable, order-invariant signature for a list of edge-lists.

    Each element of *edge_lists* corresponds to one hyperedge type and is
    itself a list of hyperedge tuples.  The returned tuple can be used as a
    dictionary key for deduplication / caching.
    """
    return tuple(
        tuple(sorted(tuple(sorted(int(v) for v in edge)) for edge in type_edges))
        for type_edges in edge_lists
    )


def distance_based_hyperedges(
    obs: np.ndarray, n_agents: int, threshold: float
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
    pos = obs[:, :2]  # (n_agents, 2)

    # Vectorised pairwise distance matrix: (n_agents, n_agents)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 2)
    dist_matrix = np.linalg.norm(diff, axis=-1)  # (N, N)

    hyperedge_list = []
    grouped_agents = set()

    for i in range(n_agents):
        members = tuple(np.where(dist_matrix[i] <= threshold)[0].tolist())
        if len(members) > 1:
            hyperedge_list.append(members)
            grouped_agents.update(members)

    # Deduplicate: symmetric neighbourhoods produce identical hyperedges
    hyperedge_list = list(set(hyperedge_list))

    # Isolated self-loops for agents not in any multi-agent hyperedge
    for i in range(n_agents):
        if i not in grouped_agents:
            hyperedge_list.append((i,))

    return hyperedge_list


def object_contact_hyperedges(
    agents_2_objects: list[list[int]],
    n_agents: int,
) -> list[tuple]:
    """
    Build hyperedges from agent-object contact information.

    Each object that has 2+ agents touching it produces one hyperedge
    containing those agent indices. Agents not touching any object
    (or alone on an object) get isolated self-loop hyperedges.

    Args:
        n_agents:        Total number of agents.
        agents_2_objects: List of lists where agents_2_objects[obj_idx]
                         contains the agent indices touching that object.

    Returns:
        List of tuples, each tuple being a hyperedge (sequence of agent indices).
    """
    hyperedge_list = []
    grouped_agents = set()

    for agents_per_object in agents_2_objects:
        if len(agents_per_object) > 0:
            edge = tuple(sorted(agents_per_object))
            hyperedge_list.append(edge)
            grouped_agents.update(agents_per_object)

    # Deduplicate
    hyperedge_list = list(set(hyperedge_list))

    # Fallback: if no group hyperedges exist, every agent is isolated
    if not hyperedge_list:
        hyperedge_list = [()]

    return hyperedge_list


def build_hypergraph(
    n_envs: int, n_agents: int, data, hyperedge_fn: Callable, device: str = "cpu"
) -> list[dhg.Hypergraph]:
    """
    Build hypergraphs from per-environment data.

    Args:
        n_envs:       Number of environments.
        n_agents:     Number of agents per environment.
        data:         Any indexable object where data[env_idx] yields the
                      per-environment slice passed to hyperedge_fn.
                      E.g. an obs array of shape (n_envs, n_agents, obs_dim),
                      or an info list/array indexed by env.
        hyperedge_fn: Callable that accepts data[env_idx] and returns a list
                      of hyperedge tuples. Use functools.partial to bind
                      extra parameters, e.g.:
                        partial(distance_based_hyperedges, threshold=1.0)

    Returns:
        A list of dhg.Hypergraph objects, one per environment.
    """
    hypergraphs = []
    for env_idx in range(n_envs):
        hyperedge_list = hyperedge_fn(data[env_idx], n_agents)
        hg = dhg.Hypergraph(n_agents, hyperedge_list, device=device)
        hypergraphs.append(hg)

    return hypergraphs


def batch_hypergraphs(
    edge_lists: list[list[tuple]],
    n_vertices_per_graph: int,
    device: str = "cpu",
) -> dhg.Hypergraph:
    """Merge multiple hypergraphs into a single block-diagonal hypergraph.

    Each graph's vertex indices are offset by ``i * n_vertices_per_graph`` so
    that the resulting incidence / Laplacian matrices are block-diagonal.
    A single ``smoothing_with_HGNN`` call on the merged graph is equivalent
    to running it on each sub-graph independently, but executes as one sparse
    matmul.

    Args:
        edge_lists:  List of hyperedge lists, one per graph.  Each hyperedge
                     list is a sequence of tuples of vertex indices (0-based
                     within their own graph).
        n_vertices_per_graph:  Number of vertices in each sub-graph (typically
                               ``n_agents``).
        device:      Torch device string.

    Returns:
        A single ``dhg.Hypergraph`` with
        ``len(edge_lists) * n_vertices_per_graph`` vertices.
    """
    merged_edges: list[tuple] = []
    for i, edges in enumerate(edge_lists):
        offset = i * n_vertices_per_graph
        for edge in edges:
            merged_edges.append(tuple(v + offset for v in edge))

    total_vertices = len(edge_lists) * n_vertices_per_graph
    return dhg.Hypergraph(total_vertices, merged_edges, device=device)


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
    A = (
        hg.H_T.to_dense().cpu().numpy().astype(np.float64)
    )  # H_T shape is (num_e, num_v)
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
