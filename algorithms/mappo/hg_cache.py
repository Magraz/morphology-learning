"""Hypergraph cache and HGNN critic helper for MAPPOAgent.

Owns all hypergraph structure caches (signature maps, edge lists, dhg objects,
entropy values) and the per-trajectory signature/entropy buffers.  Provides
the batched HGNN critic evaluation used during PPO updates.
"""

import numpy as np
import torch

from algorithms.mappo.hypergraph import (
    batch_hypergraphs,
    canonicalize_edge_lists,
    remove_agent_from_edge_lists,
    soft_entropy_from_edges,
)


class HypergraphCache:
    """Agent-side owner of hypergraph caches and HGNN critic utilities."""

    def __init__(self, *, n_agents: int, n_parallel_envs: int):
        self.n_agents = n_agents
        self.n_parallel_envs = n_parallel_envs

        # Structure caches (persist within a trajectory, cleared each reset)
        self.signature_to_id: dict[tuple, int] = {}
        self.unique_edge_lists: list[list[list[tuple]]] = []
        # Built dhg.Hypergraph objects. Inference entries persist across rollout
        # steps; minibatch entries are cleared at the start of each PPO update.
        self.inference_object_cache: dict[tuple, object] = {}
        self.minibatch_object_cache: dict[tuple, object] = {}
        self.entropy_cache: dict[int, np.ndarray] = {}

        # Per-(env, timestep) buffers populated during collection
        self.signature_ids: list[list[int]] = [[] for _ in range(n_parallel_envs)]
        self.entropies: list[list[np.ndarray]] = [[] for _ in range(n_parallel_envs)]

        # COMA-style counterfactual edge-list cache, keyed by (sig_id, agent_idx).
        # Entries are deterministic functions of the original edge lists, so we
        # never invalidate them during training.
        self._cf_edge_lists: dict[tuple[int, int], list[list[tuple]]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all caches and per-trajectory buffers."""
        self.signature_to_id.clear()
        self.unique_edge_lists.clear()
        self.inference_object_cache.clear()
        self.minibatch_object_cache.clear()
        self.entropy_cache.clear()
        self._cf_edge_lists.clear()
        self.signature_ids = [[] for _ in range(self.n_parallel_envs)]
        self.entropies = [[] for _ in range(self.n_parallel_envs)]

    def clear_minibatch_cache(self):
        """Drop cached batched hypergraphs built during a PPO update pass."""
        self.minibatch_object_cache.clear()

    # ------------------------------------------------------------------
    # Structure interning / batched-object building
    # ------------------------------------------------------------------

    def intern(self, edge_lists: list[list[tuple]]) -> int:
        """Return (inserting if new) the signature id for these per-type edge lists."""
        sig = canonicalize_edge_lists(edge_lists)
        sig_id = self.signature_to_id.get(sig)
        if sig_id is None:
            sig_id = len(self.unique_edge_lists)
            self.signature_to_id[sig] = sig_id
            self.unique_edge_lists.append(edge_lists)
        return sig_id

    def get_or_build_batched_by_type(
        self,
        sig_ids: tuple[int, ...],
        type_idx: int,
        device: str,
        cache_scope: str | None = None,
    ) -> object:
        """Return a block-diagonal batched hypergraph for one hyperedge type.

        cache_scope: "inference" or "minibatch" to read/write the matching cache;
        None skips caching entirely.
        """
        cache = None
        if cache_scope == "inference":
            cache = self.inference_object_cache
        elif cache_scope == "minibatch":
            cache = self.minibatch_object_cache

        key = (type_idx, sig_ids)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                return cached

        edge_lists = [self.unique_edge_lists[sid][type_idx] for sid in sig_ids]
        hg = batch_hypergraphs(edge_lists, self.n_agents, device=device)
        if cache is not None:
            cache[key] = hg
        return hg

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
        result = np.empty(n_types, dtype=np.float32)

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

    def build_sequence_batched_hypergraphs(
        self,
        signature_sequences: torch.Tensor,
        device: str,
    ) -> list:
        """Build block-diagonal hypergraphs for a batch of temporal windows.

        Temporal windows generate a very large number of unique signature tuples,
        so the built objects are deliberately not cached — keeping them alive
        would grow GPU memory unboundedly across rollout steps and minibatches.

        Args:
            signature_sequences: (batch, seq_len) tensor of signature IDs.
            device: Torch device string.

        Returns:
            list[dhg.Hypergraph]: One block-diagonal hypergraph per hyperedge type,
            batching ``batch * seq_len`` graphs together.
        """
        if not self.unique_edge_lists:
            raise RuntimeError("No hypergraph signatures are cached yet.")

        flat_sig_ids = tuple(
            int(sig_id) for sig_id in signature_sequences.reshape(-1).tolist()
        )
        n_types = len(self.unique_edge_lists[0])

        return [
            self.get_or_build_batched_by_type(flat_sig_ids, type_idx, device)
            for type_idx in range(n_types)
        ]

    def get_entropies_for_signature_sequences(
        self,
        signature_sequences: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Return structural entropies aligned with a signature window tensor."""
        flat_entropies = [
            self.get_or_compute_entropy(int(sig_id))
            for sig_id in signature_sequences.reshape(-1).tolist()
        ]
        entropies = torch.tensor(np.stack(flat_entropies), dtype=torch.float32).to(
            device
        )
        return entropies.view(*signature_sequences.shape, -1)

    @staticmethod
    def build_temporal_window_indices(
        timestep_indices: torch.Tensor,
        trajectory_lengths: list[int],
        seq_len: int,
    ) -> torch.Tensor:
        """Build left-padded temporal windows over flattened per-env trajectories."""
        if not trajectory_lengths:
            raise RuntimeError("Cannot build temporal windows without trajectories.")

        device = timestep_indices.device
        lengths = torch.tensor(trajectory_lengths, dtype=torch.long, device=device)
        starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                lengths.cumsum(dim=0)[:-1],
            ]
        )
        boundaries = lengths.cumsum(dim=0)
        env_indices = torch.bucketize(timestep_indices, boundaries, right=True)
        local_t = timestep_indices - starts[env_indices]

        offsets = torch.arange(seq_len, device=device) - (seq_len - 1)
        local_window = (local_t.unsqueeze(1) + offsets).clamp(min=0)
        return starts[env_indices].unsqueeze(1) + local_window

    # ------------------------------------------------------------------
    # HGNN critic batched evaluation
    # ------------------------------------------------------------------

    def compute_hgnn_critic_values(
        self,
        network_critic,
        batch_global_states,
        batch_ts_indices,
        ts_to_signature_ids,
        observation_dim: int,
        device: str,
        ts_to_entropies=None,
        ts_to_global_states=None,
        trajectory_lengths: list[int] | None = None,
    ):
        """Evaluate MultiHGNNCritic for a minibatch using batched forward pass.

        Built batched hypergraphs are memoized in ``minibatch_object_cache`` —
        callers should invoke :meth:`clear_minibatch_cache` once per PPO update.

        Args:
            network_critic: The critic network (e.g. network.critic).
            batch_global_states: (B, n_agents * obs_dim) flattened all-agent obs.
            batch_ts_indices: (B,) int tensor mapping each sample to an index
                              in ts_to_signature_ids.
            ts_to_signature_ids: list mapping timestep index -> signature id.
            observation_dim: Single-agent observation dimensionality.
            device: Torch device string.
            ts_to_entropies: Optional (N_total_ts, n_types) tensor of structural
                             entropy values aligned with ts_to_signature_ids.
            ts_to_global_states: Optional (N_total_ts, n_agents * obs_dim) tensor
                                 of flattened team observations. Required for
                                 temporal critics.
            trajectory_lengths: Optional list of per-env trajectory lengths in
                                flat timestep order. Required for temporal
                                critics.

        Returns:
            values: (B,) tensor of critic values.
        """
        B = batch_global_states.shape[0]
        unique_ts, inverse = batch_ts_indices.unique(return_inverse=True)
        n_unique = unique_ts.shape[0]

        if getattr(network_critic, "uses_temporal_sequences", False):
            if ts_to_global_states is None or trajectory_lengths is None:
                raise RuntimeError(
                    "Temporal HGNN critics require flat global states and trajectory lengths."
                )

            window_indices = self.build_temporal_window_indices(
                unique_ts, trajectory_lengths, network_critic.seq_len
            )
            obs_sequences = ts_to_global_states[window_indices].reshape(
                n_unique, network_critic.seq_len, self.n_agents, observation_dim
            )

            sig_tensor = torch.tensor(
                ts_to_signature_ids, dtype=torch.long, device=batch_ts_indices.device
            )
            signature_sequences = sig_tensor[window_indices]
            batched_hgs = self.build_sequence_batched_hypergraphs(
                signature_sequences, device=device
            )

            entropy_sequences = None
            if ts_to_entropies is not None:
                entropy_sequences = ts_to_entropies[window_indices]

            unique_values = network_critic.forward_batched(
                obs_sequences, batched_hgs, n_unique, entropies=entropy_sequences
            )
            return unique_values.squeeze(-1)[inverse]

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

        # Cache by structural signature ids (per-type, scoped to this PPO update)
        cache_key = tuple(unique_sig_ids)
        n_types = len(self.unique_edge_lists[0])
        batched_hgs = [
            self.get_or_build_batched_by_type(
                cache_key, type_idx, device, cache_scope="minibatch"
            )
            for type_idx in range(n_types)
        ]

        # Gather per-unique-timestep entropy values for conditioning
        unique_entropies = None
        if ts_to_entropies is not None:
            unique_entropies = ts_to_entropies[unique_ts]

        # Single batched forward pass
        unique_values = network_critic.forward_batched(
            obs_flat, batched_hgs, n_unique, entropies=unique_entropies
        )

        return unique_values.squeeze(-1)[inverse]

    # ------------------------------------------------------------------
    # COMA-style per-agent counterfactual baseline
    # ------------------------------------------------------------------

    def get_cf_edge_lists(
        self, sig_id: int, agent_idx: int
    ) -> list[list[tuple]]:
        """Return the per-type edge lists for sig_id with agent_idx removed.

        Cached on the instance — counterfactual edges are a deterministic
        function of (original sig_id, agent_idx).
        """
        key = (sig_id, agent_idx)
        cached = self._cf_edge_lists.get(key)
        if cached is None:
            cached = remove_agent_from_edge_lists(
                self.unique_edge_lists[sig_id], agent_idx, self.n_agents
            )
            self._cf_edge_lists[key] = cached
        return cached

    def _get_or_build_cf_batched_by_type(
        self,
        cache_key: tuple,
        unique_sig_ids: tuple[int, ...],
        agent_idx: int,
        type_idx: int,
        device: str,
    ):
        """Block-diagonal batched hypergraph for one type, with `agent_idx`
        removed from each contributing timestep's edge list. Cached in the
        minibatch object cache (cleared per PPO update)."""
        cached = self.minibatch_object_cache.get(cache_key)
        if cached is not None:
            return cached

        type_edge_lists = [
            self.get_cf_edge_lists(sid, agent_idx)[type_idx]
            for sid in unique_sig_ids
        ]
        hg = batch_hypergraphs(type_edge_lists, self.n_agents, device=device)
        self.minibatch_object_cache[cache_key] = hg
        return hg

    def compute_per_agent_counterfactual_values(
        self,
        network_critic,
        batch_global_states: torch.Tensor,
        batch_ts_indices: torch.Tensor,
        ts_to_signature_ids,
        observation_dim: int,
        device: str,
    ) -> torch.Tensor:
        """COMA-style counterfactual values V_cf(i, s) for every (sample, agent).

        For each agent i, evaluates the critic on the same observations but
        with all multi-vertex hyperedges containing i removed from the
        per-type hypergraphs.

        Args:
            network_critic: MultiHGNNCritic instance.
            batch_global_states: (B, n_agents * obs_dim) flattened all-agent obs.
            batch_ts_indices: (B,) int tensor mapping each sample to an index
                              in ts_to_signature_ids.
            ts_to_signature_ids: list mapping global timestep index -> signature id.
            observation_dim: Single-agent observation dimensionality.
            device: Torch device string.

        Returns:
            (B, n_agents) tensor of counterfactual values.
        """
        B = batch_global_states.shape[0]
        unique_ts, inverse = batch_ts_indices.unique(return_inverse=True)
        n_unique = unique_ts.shape[0]

        # Vectorized first-occurrence gathering (mirrors compute_hgnn_critic_values).
        first_occ = torch.full(
            (n_unique,), B, dtype=torch.long, device=batch_ts_indices.device
        )
        arange = torch.arange(B, device=batch_ts_indices.device)
        first_occ.scatter_reduce_(0, inverse, arange, reduce="amin")
        obs_unique = batch_global_states[first_occ]
        obs_flat = obs_unique.reshape(n_unique * self.n_agents, observation_dim)

        unique_ts_list = [idx.item() for idx in unique_ts]
        unique_sig_ids = tuple(ts_to_signature_ids[ts] for ts in unique_ts_list)
        n_types = len(self.unique_edge_lists[0])

        cf_unique = torch.zeros(
            n_unique, self.n_agents, device=obs_flat.device, dtype=obs_flat.dtype
        )
        for agent_idx in range(self.n_agents):
            batched_hgs = [
                self._get_or_build_cf_batched_by_type(
                    cache_key=("cf", type_idx, agent_idx, unique_sig_ids),
                    unique_sig_ids=unique_sig_ids,
                    agent_idx=agent_idx,
                    type_idx=type_idx,
                    device=device,
                )
                for type_idx in range(n_types)
            ]
            unique_v = network_critic.forward_batched(
                obs_flat, batched_hgs, n_unique
            ).squeeze(-1)
            cf_unique[:, agent_idx] = unique_v

        return cf_unique[inverse]
