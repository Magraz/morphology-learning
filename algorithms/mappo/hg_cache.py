"""Hypergraph cache and HGNN critic helper for MAPPOAgent.

Owns all hypergraph structure caches (signature maps, edge lists, dhg objects,
entropy values) and the per-trajectory signature/entropy buffers.  Provides
the batched HGNN critic evaluation used during PPO updates.
"""

import numpy as np
import torch

from algorithms.mappo.hypergraph import batch_hypergraphs, soft_entropy_from_edges


class HypergraphCache:
    """Agent-side owner of hypergraph caches and HGNN critic utilities."""

    def __init__(self, *, n_agents: int, n_parallel_envs: int):
        self.n_agents = n_agents
        self.n_parallel_envs = n_parallel_envs

        # Structure caches (persist within a trajectory, cleared each reset)
        self.signature_to_id: dict[tuple, int] = {}
        self.unique_edge_lists: list[list[list[tuple]]] = []
        self.object_cache: dict[tuple, object] = {}
        self.entropy_cache: dict[int, np.ndarray] = {}

        # Per-(env, timestep) buffers populated during collection
        self.signature_ids: list[list[int]] = [[] for _ in range(n_parallel_envs)]
        self.entropies: list[list[np.ndarray]] = [[] for _ in range(n_parallel_envs)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all caches and per-trajectory buffers."""
        self.signature_to_id.clear()
        self.unique_edge_lists.clear()
        self.object_cache.clear()
        self.entropy_cache.clear()
        self.signature_ids = [[] for _ in range(self.n_parallel_envs)]
        self.entropies = [[] for _ in range(self.n_parallel_envs)]

    # ------------------------------------------------------------------
    # Entropy computation
    # ------------------------------------------------------------------

    def get_or_compute_entropy(self, sig_id: int) -> np.ndarray:
        """Return cached soft structural entropy for a hypergraph signature.

        On a cache miss, computes soft entropy directly from edge lists
        (avoids dhg.Hypergraph construction and sparse->dense conversion).

        Args:
            sig_id: Index into unique_edge_lists.

        Returns:
            np.ndarray of shape (n_types,) with S_soft_norm per hyperedge type.
        """
        cached = self.entropy_cache.get(sig_id)
        if cached is not None:
            return cached

        edge_lists = self.unique_edge_lists[sig_id]
        n_types = len(edge_lists)
        result = np.empty(n_types, dtype=np.float64)

        for type_idx, edges in enumerate(edge_lists):
            _, S_soft_norm = soft_entropy_from_edges(edges, self.n_agents)
            result[type_idx] = S_soft_norm

        self.entropy_cache[sig_id] = result
        return result

    # ------------------------------------------------------------------
    # Transition storage helpers
    # ------------------------------------------------------------------

    def store_transition(
        self,
        env_idx: int,
        sig_id: int,
        entropy_conditioning: bool,
        entropy_tensor=None,
    ):
        """Record signature ID and (optionally) entropy for one (env, timestep).

        Args:
            env_idx: Environment index.
            sig_id: Hypergraph signature ID for this timestep.
            entropy_conditioning: Whether entropy conditioning is enabled.
            entropy_tensor: Optional pre-computed entropy (torch tensor for one env)
                            or None to compute on-the-fly.
        """
        self.signature_ids[env_idx].append(sig_id)
        if entropy_conditioning:
            if entropy_tensor is not None:
                self.entropies[env_idx].append(entropy_tensor.cpu().numpy())
            else:
                self.entropies[env_idx].append(self.get_or_compute_entropy(sig_id))

    # ------------------------------------------------------------------
    # Flatten helpers (for update phase)
    # ------------------------------------------------------------------

    def get_flat_signature_ids(self) -> list[int]:
        """Return a flat list of signature IDs for all (env, timestep) pairs."""
        result = []
        for env_idx in range(self.n_parallel_envs):
            result.extend(self.signature_ids[env_idx])
        return result

    def get_flat_entropies(self, device, entropy_conditioning: bool):
        """Return a flat (N_total_ts, n_types) tensor of structural entropy.

        Returns None if entropy conditioning is disabled or buffer is empty.
        """
        if not entropy_conditioning or len(self.entropies[0]) == 0:
            return None
        all_ent = []
        for env_idx in range(self.n_parallel_envs):
            all_ent.extend(self.entropies[env_idx])
        return torch.tensor(np.stack(all_ent), dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # HGNN critic batched evaluation
    # ------------------------------------------------------------------

    def compute_hgnn_critic_values(
        self,
        network_critic,
        batch_global_states,
        batch_ts_indices,
        ts_to_signature_ids,
        hgnn_batch_cache,
        observation_dim: int,
        device: str,
        ts_to_entropies=None,
    ):
        """Evaluate MultiHGNNCritic for a minibatch using batched forward pass.

        Args:
            network_critic: The critic network (e.g. network.critic).
            batch_global_states: (B, n_agents * obs_dim) flattened all-agent obs.
            batch_ts_indices: (B,) int tensor mapping each sample to an index
                              in ts_to_signature_ids.
            ts_to_signature_ids: list mapping timestep index -> signature id.
            hgnn_batch_cache: dict mapping tuple(signature ids in minibatch) to
                              pre-built list[dhg.Hypergraph] batched by type.
            observation_dim: Single-agent observation dimensionality.
            device: Torch device string.
            ts_to_entropies: Optional (N_total_ts, n_types) tensor of structural
                             entropy values aligned with ts_to_signature_ids.

        Returns:
            values: (B,) tensor of critic values.
        """
        B = batch_global_states.shape[0]
        unique_ts, inverse = batch_ts_indices.unique(return_inverse=True)
        n_unique = unique_ts.shape[0]

        # Vectorized first-occurrence gathering
        first_occ = torch.full(
            (n_unique,), B, dtype=torch.long, device=batch_ts_indices.device
        )
        arange = torch.arange(B, device=batch_ts_indices.device)
        first_occ.scatter_reduce_(0, inverse, arange, reduce="amin")
        obs_unique = batch_global_states[first_occ]
        obs_flat = obs_unique.reshape(n_unique * self.n_agents, observation_dim)

        unique_ts_list = [idx.item() for idx in unique_ts]
        unique_sig_ids = [ts_to_signature_ids[ts] for ts in unique_ts_list]

        # Cache by structural signature ids
        cache_key = tuple(unique_sig_ids)
        batched_hgs = hgnn_batch_cache.get(cache_key)
        if batched_hgs is None:
            n_types = len(self.unique_edge_lists[0])
            batched_hgs = []
            for type_idx in range(n_types):
                edge_lists = [
                    self.unique_edge_lists[sig_id][type_idx]
                    for sig_id in unique_sig_ids
                ]
                batched_hgs.append(
                    batch_hypergraphs(edge_lists, self.n_agents, device=device)
                )
            hgnn_batch_cache[cache_key] = batched_hgs

        # Gather per-unique-timestep entropy values for conditioning
        unique_entropies = None
        if ts_to_entropies is not None:
            unique_entropies = ts_to_entropies[unique_ts]

        # Single batched forward pass
        unique_values = network_critic.forward_batched(
            obs_flat, batched_hgs, n_unique, entropies=unique_entropies
        )

        return unique_values.squeeze(-1)[inverse]
