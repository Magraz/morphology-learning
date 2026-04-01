from functools import partial

import numpy as np
import torch

import dhg

from algorithms.mappo.hypergraph import (
    batch_hypergraphs,
    canonicalize_edge_lists,
    distance_based_hyperedges,
    object_contact_hyperedges,
)


class HypergraphRuntime:
    """Owns hypergraph runtime behavior for rollout/eval/render inference."""

    def __init__(
        self,
        *,
        agent,
        device: str,
        n_agents: int,
        n_parallel_envs: int,
        critic_type: str,
        model_params,
    ):
        self.agent = agent
        self.device = device
        self.n_agents = n_agents
        self.n_parallel_envs = n_parallel_envs
        self.critic_type = critic_type
        self.hypergraph_mode = model_params.hypergraph_mode

        self.hyperedge_fns = [
            (partial(distance_based_hyperedges, threshold=1.0), "obs"),
            (object_contact_hyperedges, "agents_2_objects"),
        ]

        if self.hypergraph_mode == "hygma":
            from algorithms.mappo.dynamic_grouping import DynamicSpectralGrouping

            assert (
                self.critic_type == "multi_hgnn"
            ), "HYGMA mode requires critic_type='multi_hgnn'"
            assert (
                model_params.n_hyperedge_types == 1
            ), "HYGMA mode requires n_hyperedge_types=1"

            max_k = model_params.hygma_max_clusters or (self.n_agents - 1)
            self._dynamic_grouping = DynamicSpectralGrouping(
                n_agents=self.n_agents,
                n_envs=self.n_parallel_envs,
                obs_dim=agent.observation_dim,
                history_len=model_params.hygma_history_len,
                clustering_interval=model_params.hygma_clustering_interval,
                min_clusters=model_params.hygma_min_clusters,
                max_clusters=max_k,
                stability_threshold=model_params.hygma_stability_threshold,
            )
            self._entropy_type_names = ["proximity"]
        elif self.hypergraph_mode == "predefined":
            self._dynamic_grouping = None
            self._entropy_type_names = ["proximity", "object"]
        else:
            raise ValueError(
                f"Unknown hypergraph_mode: {self.hypergraph_mode!r}. "
                "Expected 'predefined' or 'hygma'."
            )

    @property
    def entropy_type_names(self) -> list[str]:
        return self._entropy_type_names

    def on_rollout_reset(self) -> None:
        if self._dynamic_grouping is not None:
            self._dynamic_grouping.reset_history()

    def on_rollout_step(self, obs: np.ndarray) -> None:
        if self._dynamic_grouping is not None:
            self._dynamic_grouping.update_history(obs)
            self._dynamic_grouping.maybe_update_groups()

    def on_env_done_mask(self, dones: np.ndarray) -> None:
        if self._dynamic_grouping is not None:
            self._dynamic_grouping.reset_history(env_mask=dones)

    def _compute_all_type_edge_lists(self, obs, infos, n_envs: int):
        """Compute per-type, per-env edge lists from observations and infos.

        Returns:
            all_type_edge_lists: [type_idx][env_idx] -> edge_list
        """
        if self.hypergraph_mode == "hygma":
            spectral_edges = self._dynamic_grouping.get_edge_lists(n_envs)
            return [spectral_edges]

        all_type_edge_lists = []
        for fn, source in self.hyperedge_fns:
            if source == "obs":
                data = obs
            else:
                data = infos.get(source) if isinstance(infos, dict) else None
                if data is None:
                    continue
            all_type_edge_lists.append(
                [fn(data[e], self.n_agents) for e in range(n_envs)]
            )
        return all_type_edge_lists

    def build_inference_hypergraphs(self, obs, infos, n_envs: int):
        """Build batched block-diagonal hypergraphs for critic inference."""
        if self.critic_type != "multi_hgnn":
            return None, None

        all_type_edge_lists = self._compute_all_type_edge_lists(obs, infos, n_envs)

        n_types = len(all_type_edge_lists)
        per_env_edge_lists = [
            [all_type_edge_lists[t][e] for t in range(n_types)] for e in range(n_envs)
        ]

        per_env_sig_ids = []
        hg_cache = self.agent.hg_cache
        for edge_lists in per_env_edge_lists:
            sig = canonicalize_edge_lists(edge_lists)
            sig_id = hg_cache.signature_to_id.get(sig)
            if sig_id is None:
                sig_id = len(hg_cache.unique_edge_lists)
                hg_cache.signature_to_id[sig] = sig_id
                hg_cache.unique_edge_lists.append(edge_lists)
            per_env_sig_ids.append(sig_id)

        batched_hgs = []
        for type_idx in range(n_types):
            type_edge_lists = [
                hg_cache.unique_edge_lists[sid][type_idx] for sid in per_env_sig_ids
            ]
            cache_key = tuple(per_env_sig_ids)
            cached = hg_cache.object_cache.get((type_idx, cache_key))
            if cached is not None:
                batched_hgs.append(cached)
            else:
                hg = batch_hypergraphs(type_edge_lists, self.n_agents, device=self.device)
                hg_cache.object_cache[(type_idx, cache_key)] = hg
                batched_hgs.append(hg)

        return batched_hgs, per_env_sig_ids

    def build_per_env_hypergraphs(self, obs, infos, n_envs: int):
        """Build individual per-env hypergraphs (not batched).

        Returns:
            list[list[dhg.Hypergraph]]: [env_idx][type_idx]
        """
        all_type_edge_lists = self._compute_all_type_edge_lists(obs, infos, n_envs)
        n_types = len(all_type_edge_lists)

        result = []
        for e in range(n_envs):
            env_hgs = [
                dhg.Hypergraph(
                    self.n_agents, all_type_edge_lists[t][e], device=self.device
                )
                for t in range(n_types)
            ]
            result.append(env_hgs)
        return result

    def compute_entropies_for_critic(self, per_env_sig_ids):
        """Compute structural entropy values for critic conditioning."""
        if per_env_sig_ids is None:
            return None

        entropies = [
            self.agent.hg_cache.get_or_compute_entropy(sig_id) for sig_id in per_env_sig_ids
        ]
        return torch.tensor(np.stack(entropies), dtype=torch.float32).to(self.agent.device)
