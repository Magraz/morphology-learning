import numpy as np
import dhg
from collections import Counter
from functools import partial
from typing import Callable

from environments.types import EnvironmentEnum


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


def smaclite_ally_visibility_hyperedges(
    obs: np.ndarray,
    n_agents: int,
    n_enemies: int,
    enemy_feat_size: int,
    ally_feat_size: int,
) -> list[tuple]:
    """Build hyperedges from ally visibility in SMACLITE observations.

    SMACLITE per-agent observation layout:
        [0:4]                                   — movement feasibility (4 dirs)
        [4 : 4 + n_enemies * enemy_feat_size]   — enemy features
            per enemy: attackable, distance, dx, dy, hp, [shield], [unit_type]
        [... : ... + (n_agents-1) * ally_feat_size] — ally features
            per ally:  visible, distance, dx, dy, hp, [shield], [unit_type]
        [... : obs_dim]                         — own features

    The ally block for agent i skips i's own slot, so slot ``s`` in agent i's
    block refers to ally id ``s if s < i else s + 1``.

    For each agent i, build a hyperedge of i plus every ally it currently sees
    (``visible`` flag == 1). Ally visibility is symmetric by construction
    (same sight radius on both sides) and dead agents produce all-zero obs, so
    no explicit mutual-visibility filter is needed. Duplicate hyperedges are
    removed; agents not in any multi-member edge get isolated self-loops.

    Args:
        obs:              Agent observations, shape (n_agents, obs_dim).
        n_agents:         Total number of agents.
        n_enemies:        Number of enemies in the map.
        enemy_feat_size:  Size of each enemy's feature sub-block.
        ally_feat_size:   Size of each ally's feature sub-block.

    Returns:
        List of hyperedge tuples (sequences of agent indices).
    """
    if n_agents < 2:
        return [(i,) for i in range(n_agents)]

    ally_block_start = 4 + n_enemies * enemy_feat_size
    ally_block_len = (n_agents - 1) * ally_feat_size
    ally_block = obs[:, ally_block_start : ally_block_start + ally_block_len]
    ally_block = ally_block.reshape(n_agents, n_agents - 1, ally_feat_size)
    visible = ally_block[:, :, 0]  # (n_agents, n_agents - 1)

    hyperedge_list = []
    grouped_agents = set()

    for i in range(n_agents):
        visible_slots = np.flatnonzero(visible[i] > 0.5)
        if visible_slots.size == 0:
            continue
        real_ids = np.where(visible_slots < i, visible_slots, visible_slots + 1)
        members = tuple(sorted([i, *real_ids.tolist()]))
        hyperedge_list.append(members)
        grouped_agents.update(members)

    hyperedge_list = list(set(hyperedge_list))

    for i in range(n_agents):
        if i not in grouped_agents:
            hyperedge_list.append((i,))

    return hyperedge_list


def smaclite_shared_targets_hyperedges(
    obs: np.ndarray,
    n_agents: int,
    n_enemies: int,
    enemy_feat_size: int,
) -> list[tuple]:
    """Build hyperedges from shared enemy targets in SMACLITE observations.

    SMACLITE per-agent observation layout (see
    ``smaclite_ally_visibility_hyperedges`` for the full spec). The first
    feature of each enemy sub-block is the ``attackable`` flag — set when the
    enemy is visible, alive, and within the agent's attack range. Healers
    contribute 0 to every enemy slot since their avail-action mask covers
    allies, not enemies, so they are naturally excluded from shared-target
    edges without an explicit role filter.

    For each enemy, collect the set of agents with ``attackable == 1``. Sets
    with 2+ members become hyperedges (focus-fire groups). Duplicate
    hyperedges are removed; agents in no group get isolated self-loops.

    Args:
        obs:             Agent observations, shape (n_agents, obs_dim).
        n_agents:        Total number of agents.
        n_enemies:       Number of enemies in the map.
        enemy_feat_size: Size of each enemy's feature sub-block.

    Returns:
        List of hyperedge tuples (sequences of agent indices).
    """
    if n_enemies == 0:
        return [(i,) for i in range(n_agents)]

    enemy_block = obs[:, 4 : 4 + n_enemies * enemy_feat_size]
    attackable = enemy_block.reshape(n_agents, n_enemies, enemy_feat_size)[:, :, 0]

    hyperedge_list = []
    grouped_agents = set()

    for e in range(n_enemies):
        attackers = np.flatnonzero(attackable[:, e] > 0.5)
        if attackers.size > 1:
            edge = tuple(attackers.tolist())
            hyperedge_list.append(edge)
            grouped_agents.update(edge)

    hyperedge_list = list(set(hyperedge_list))

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
    return np.array(results, dtype=np.float32)  # (n_hypergraphs, 2)


def compute_soft_hyperedge_structural_entropy_batch(
    hypergraphs: list,
    n_bins: int = 30,
    sigma: float | None = None,
    use_log_binning: bool = True,
) -> np.ndarray:
    """
    Compute smooth surrogate hyperedge structural entropy for a batch of hypergraphs.

    Args:
        hypergraphs: List of dhg.Hypergraph instances (length n_hypergraphs).
        n_bins:      Number of histogram bins (passed to soft entropy fn).
        sigma:       Bandwidth of the Gaussian soft assignment (None = auto).
        use_log_binning: If True, bin in log-space.

    Returns:
        entropies: np.ndarray of shape (n_hypergraphs, 2).
                   Column 0: S_soft      — smoothed Shannon entropy (nats).
                   Column 1: S_soft_norm — entropy normalised by ln(B).
    """
    results = []
    for hg in hypergraphs:
        S_soft, S_soft_norm, _ = calculate_soft_hyperedge_structural_entropy(
            hg, n_bins=n_bins, sigma=sigma, use_log_binning=use_log_binning
        )
        results.append([S_soft, S_soft_norm])
    return np.array(results, dtype=np.float32)  # (n_hypergraphs, 2)


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
        hg.H_T.to_dense().cpu().numpy().astype(np.float32)
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
    counts = np.array(list(weight_counts.values()), dtype=np.float32)
    P = counts / E_total

    # Step 4: Shannon entropy S_e = -sum P(w) ln P(w)
    S_e = -np.sum(P * np.log(P))

    # Normalization: S_max = ln(E_total)
    S_max = np.log(E_total) if E_total > 1 else 1.0
    S_normalized = S_e / S_max if S_max > 0 else 0.0

    return S_e, S_normalized, weights


def calculate_soft_hyperedge_structural_entropy(
    hg: dhg.Hypergraph,
    n_bins: int = 30,
    sigma: float | None = None,
    use_log_binning: bool = True,
    log_delta: float = 1.0,
    eps: float = 1e-12,
) -> tuple[float, float, np.ndarray]:
    """
    Smooth surrogate for Xian et al. hyperedge structural entropy.

    Replaces the hard weight-category counts with soft (expected) counts
    via a Gaussian RBF soft histogram, yielding a smooth entropy signal
    suitable for RL reward/goal shaping.

    Args:
        hg:              dhg.Hypergraph instance.
        n_bins:          Number of histogram bins B (recommended 20–50).
        sigma:           Bandwidth of the Gaussian soft assignment. If None,
                         defaults to 0.75 × bin spacing.
        use_log_binning: If True, bin in log-space x_e = log(w_e + delta)
                         for better handling of heavy-tailed weight ranges.
        log_delta:       Offset added before log transform for numerical
                         stability (only used when use_log_binning=True).
        eps:             Small constant for numerical stability in ln.

    Returns:
        S_soft:          float, smoothed Shannon entropy (nats).
        S_soft_norm:     float, entropy normalised by ln(B).
        weights:         np.ndarray of per-hyperedge weights w(e).
    """
    # --- Step 1: Compute hyperedge weights (unchanged from paper) ---
    A = hg.H_T.to_dense().cpu().numpy().astype(np.float32)  # (E, |V|)
    E_total = hg.num_e

    D = A.sum(axis=0)  # node hyperdegrees, shape (|V|,)
    O = A.sum(axis=1)  # hyperedge orders,  shape (E,)
    sum_D = A @ D  # sum of node hyperdegrees per edge, shape (E,)
    weights = O * sum_D  # shape (E,)

    if E_total <= 1:
        return 0.0, 0.0, weights

    # --- Step 2: Choose bin centers ---
    if use_log_binning:
        values = np.log(weights + log_delta)
    else:
        values = weights

    v_min, v_max = values.min(), values.max()
    # Avoid degenerate case where all weights are identical
    if v_max - v_min < eps:
        return 0.0, 0.0, weights

    centers = np.linspace(v_min, v_max, n_bins)  # (B,)
    bin_spacing = centers[1] - centers[0] if n_bins > 1 else 1.0

    if sigma is None:
        sigma = 0.75 * bin_spacing

    # --- Step 3: Soft-assign each weight to bins (RBF) ---
    # values shape (E,), centers shape (B,) → diffs shape (E, B)
    diffs = values[:, np.newaxis] - centers[np.newaxis, :]
    log_assignments = -(diffs**2) / (2.0 * sigma**2)
    # Numerically stable softmax along bin axis
    log_assignments -= log_assignments.max(axis=1, keepdims=True)
    assignments = np.exp(log_assignments)
    assignments /= assignments.sum(axis=1, keepdims=True)  # (E, B)

    # --- Step 4: Soft counts and smoothed probability ---
    soft_counts = assignments.sum(axis=0)  # (B,)
    P_soft = soft_counts / E_total  # (B,)

    # --- Step 5: Smoothed hyperedge structural entropy ---
    S_soft = -np.sum(P_soft * np.log(P_soft + eps))

    # Normalise by ln(B) — maximum entropy over B bins
    S_max = np.log(n_bins)
    S_soft_norm = S_soft / S_max if S_max > 0 else 0.0

    return S_soft, S_soft_norm, weights


def soft_entropy_from_edges(
    edge_list: list[tuple],
    n_vertices: int,
    n_bins: int = 30,
    sigma: float | None = None,
    use_log_binning: bool = True,
    log_delta: float = 1.0,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Fast soft structural entropy computed directly from an edge list.

    Equivalent to ``calculate_soft_hyperedge_structural_entropy`` but avoids
    constructing a ``dhg.Hypergraph`` and the expensive sparse→dense
    incidence-matrix conversion.

    Args:
        edge_list:       List of hyperedge tuples (vertex indices).
        n_vertices:      Number of vertices in the hypergraph.
        n_bins:          Number of histogram bins B.
        sigma:           Gaussian RBF bandwidth (None = auto).
        use_log_binning: Bin in log-space if True.
        log_delta:       Offset before log transform.
        eps:             Numerical stability constant.

    Returns:
        (S_soft, S_soft_norm) — smoothed entropy and normalised version.
    """
    E_total = len(edge_list)
    if E_total <= 1:
        return 0.0, 0.0

    # Node hyperdegrees
    D = np.zeros(n_vertices, dtype=np.float32)
    for edge in edge_list:
        for v in edge:
            D[v] += 1.0

    # Per-edge weights: w(e) = |e| * sum_{v in e} D(v)
    weights = np.empty(E_total, dtype=np.float32)
    for i, edge in enumerate(edge_list):
        weights[i] = len(edge) * sum(D[v] for v in edge)

    # Bin centers
    if use_log_binning:
        values = np.log(weights + log_delta)
    else:
        values = weights

    v_min, v_max = values.min(), values.max()
    if v_max - v_min < eps:
        return 0.0, 0.0

    centers = np.linspace(v_min, v_max, n_bins)
    bin_spacing = centers[1] - centers[0] if n_bins > 1 else 1.0

    if sigma is None:
        sigma = 0.75 * bin_spacing

    # Soft-assign via Gaussian RBF
    diffs = values[:, np.newaxis] - centers[np.newaxis, :]
    log_assignments = -(diffs**2) / (2.0 * sigma**2)
    log_assignments -= log_assignments.max(axis=1, keepdims=True)
    assignments = np.exp(log_assignments)
    assignments /= assignments.sum(axis=1, keepdims=True)

    # Smoothed entropy
    soft_counts = assignments.sum(axis=0)
    P_soft = soft_counts / E_total
    S_soft = -np.sum(P_soft * np.log(P_soft + eps))

    S_max = np.log(n_bins)
    S_soft_norm = S_soft / S_max if S_max > 0 else 0.0

    return S_soft, S_soft_norm


# ── Hyperedge function registry ───────────────────────────────────────────────
#
# Maps (environment, hyperedge_fn_name) -> factory(env_ctx) -> (fn, source).
# `source` is either "obs" (per-agent observations) or a key in the env `info`
# dict (e.g. "agents_2_objects"). `env_ctx` provides environment-specific
# parameters like `n_enemies` / `enemy_feat_size` for SMACLITE builders.

_PROXIMITY_FACTORY = lambda ctx: (
    partial(distance_based_hyperedges, threshold=1.0),
    "obs",
)

HYPEREDGE_FN_REGISTRY: dict[str, dict[str, Callable]] = {
    EnvironmentEnum.MULTI_BOX: {
        "proximity": _PROXIMITY_FACTORY,
        "contact": lambda ctx: (object_contact_hyperedges, "agents_2_objects"),
    },
    EnvironmentEnum.SMACLITE: {
        "ally_visibility": lambda ctx: (
            partial(
                smaclite_ally_visibility_hyperedges,
                n_enemies=ctx["n_enemies"],
                enemy_feat_size=ctx["enemy_feat_size"],
                ally_feat_size=ctx["ally_feat_size"],
            ),
            "obs",
        ),
        "shared_targets": lambda ctx: (
            partial(
                smaclite_shared_targets_hyperedges,
                n_enemies=ctx["n_enemies"],
                enemy_feat_size=ctx["enemy_feat_size"],
            ),
            "obs",
        ),
    },
    EnvironmentEnum.SCATTER: {"proximity": _PROXIMITY_FACTORY},
    EnvironmentEnum.RENDEZVOUZ: {"proximity": _PROXIMITY_FACTORY},
    EnvironmentEnum.CONTACT: {"proximity": _PROXIMITY_FACTORY},
}


def build_hyperedge_fns_from_names(
    env_name: str,
    fn_names: list[str],
    env_ctx: dict | None = None,
) -> list[tuple]:
    """Resolve `fn_names` to a list of (hyperedge_fn, source) tuples.

    Args:
        env_name: EnvironmentEnum value identifying the environment.
        fn_names: Hyperedge function names declared in the experiment config.
        env_ctx:  Environment-specific context required by some factories
                  (e.g. SMACLITE needs n_enemies / feat sizes).

    Raises:
        ValueError: If the environment has no registered hyperedge functions,
                    or any name in `fn_names` is not registered for that env.
    """
    if env_name not in HYPEREDGE_FN_REGISTRY:
        raise ValueError(
            f"No hyperedge functions are registered for environment "
            f"'{env_name}'. Registered environments: "
            f"{sorted(HYPEREDGE_FN_REGISTRY.keys())}."
        )

    registry = HYPEREDGE_FN_REGISTRY[env_name]
    missing = [n for n in fn_names if n not in registry]
    if missing:
        raise ValueError(
            f"hyperedge_fn_names {missing} are not registered for environment "
            f"'{env_name}'. Available names: {sorted(registry.keys())}."
        )

    ctx = env_ctx or {}
    return [registry[name](ctx) for name in fn_names]
