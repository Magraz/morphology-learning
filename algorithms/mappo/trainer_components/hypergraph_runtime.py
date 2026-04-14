from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch

import dhg

from algorithms.mappo.hypergraph import (
    batch_hypergraphs,
    canonicalize_edge_lists,
    distance_based_hyperedges,
    object_contact_hyperedges,
)


def load_pretrained_affinity_transformers(
    results_root: Path,
    batch_names: list[str],
    trial_id: str,
    n_agents: int,
    observation_dim: int,
    history_len: int,
    hidden_dim: int,
    device: str,
) -> list[tuple[str, "AffinityTransformer"]]:
    """Load frozen AffinityTransformers from completed learned_affinity experiments."""
    from algorithms.mappo.networks.encoders import AffinityTransformer

    named_transformers = []
    for batch_name in batch_names:
        ckpt_path = (
            results_root / batch_name / "learned_affinity" / trial_id
            / "models" / "models_finished.pth"
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for source '{batch_name}': {ckpt_path}"
            )

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("affinity_transformer")
        if state_dict is None:
            raise KeyError(
                f"Checkpoint for '{batch_name}' does not contain 'affinity_transformer'"
            )

        loaded_obs_dim = state_dict["input_proj.weight"].shape[1]
        if loaded_obs_dim != observation_dim:
            raise ValueError(
                f"obs_dim mismatch for '{batch_name}': checkpoint has {loaded_obs_dim}, "
                f"current env has {observation_dim}"
            )

        transformer = AffinityTransformer(
            n_agents=n_agents,
            observation_dim=observation_dim,
            history_length=history_len,
            d_model=hidden_dim,
        )
        transformer.load_state_dict(state_dict)
        transformer.eval()
        transformer.requires_grad_(False)
        transformer.to(device)

        named_transformers.append((batch_name, transformer))

    return named_transformers


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
        batch_dir: Path = None,
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

        # Plural list used only by combined_affinities mode
        self._dynamic_groupings: list | None = None

        if self.hypergraph_mode in ("hygma", "learned_affinity"):
            from algorithms.mappo.dynamic_grouping import DynamicSpectralGrouping

            assert self.critic_type in (
                "multi_hgnn", "hg_cross_attention"
            ), f"{self.hypergraph_mode} mode requires critic_type='multi_hgnn' or 'hg_cross_attention'"
            assert (
                model_params.n_hyperedge_types == 1
            ), f"{self.hypergraph_mode} mode requires n_hyperedge_types=1"

            affinity_fn = None
            if self.hypergraph_mode == "learned_affinity":
                transformer = agent.network.affinity_transformer
                assert transformer is not None, (
                    "learned_affinity mode requires AffinityTransformer on MAPPONetwork"
                )
                affinity_fn = self._make_affinity_fn(transformer, agent.device)

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
                affinity_fn=affinity_fn,
            )
            self._entropy_type_names = ["proximity"]

        elif self.hypergraph_mode == "combined_affinities":
            from algorithms.mappo.dynamic_grouping import DynamicSpectralGrouping

            sources = model_params.combined_affinity_sources
            assert self.critic_type in (
                "multi_hgnn", "hg_cross_attention"
            ), "combined_affinities mode requires critic_type='multi_hgnn' or 'hg_cross_attention'"
            assert sources and len(sources) > 0, (
                "combined_affinities mode requires non-empty combined_affinity_sources"
            )
            assert model_params.n_hyperedge_types == len(sources), (
                f"n_hyperedge_types ({model_params.n_hyperedge_types}) must equal "
                f"len(combined_affinity_sources) ({len(sources)})"
            )
            assert batch_dir is not None, (
                "combined_affinities mode requires batch_dir for checkpoint resolution"
            )

            results_root = Path(batch_dir).parents[1] / "results"
            named_transformers = load_pretrained_affinity_transformers(
                results_root=results_root,
                batch_names=sources,
                trial_id=model_params.combined_affinity_trial_id,
                n_agents=self.n_agents,
                observation_dim=agent.observation_dim,
                history_len=model_params.hygma_history_len,
                hidden_dim=model_params.hidden_dim,
                device=self.device,
            )

            max_k = model_params.hygma_max_clusters or (self.n_agents - 1)
            self._dynamic_groupings = []
            for _name, transformer in named_transformers:
                afn = self._make_affinity_fn(transformer, self.device)
                self._dynamic_groupings.append(
                    DynamicSpectralGrouping(
                        n_agents=self.n_agents,
                        n_envs=self.n_parallel_envs,
                        obs_dim=agent.observation_dim,
                        history_len=model_params.hygma_history_len,
                        clustering_interval=model_params.hygma_clustering_interval,
                        min_clusters=model_params.hygma_min_clusters,
                        max_clusters=max_k,
                        stability_threshold=model_params.hygma_stability_threshold,
                        affinity_fn=afn,
                    )
                )

            self._dynamic_grouping = None
            self._entropy_type_names = [name for name, _ in named_transformers]

        elif self.hypergraph_mode == "predefined":
            self._dynamic_grouping = None
            self._entropy_type_names = ["proximity", "object"]
        else:
            raise ValueError(
                f"Unknown hypergraph_mode: {self.hypergraph_mode!r}. "
                "Expected 'predefined', 'hygma', 'learned_affinity', or 'combined_affinities'."
            )

    @staticmethod
    def _make_affinity_fn(transformer, device):
        """Create a closure that maps observation history to a numpy affinity matrix."""
        @torch.no_grad()
        def affinity_fn(agg_tensor: torch.Tensor) -> np.ndarray:
            x = agg_tensor.to(device)
            return transformer(x).cpu().numpy()
        return affinity_fn

    @property
    def entropy_type_names(self) -> list[str]:
        return self._entropy_type_names

    @contextmanager
    def render_grouping_context(self):
        """Temporarily swap in a single-env DynamicSpectralGrouping for render.

        The training grouping (n_envs=n_parallel_envs) is restored on exit.
        """
        if self._dynamic_grouping is None and not self._dynamic_groupings:
            yield
            return

        from algorithms.mappo.dynamic_grouping import DynamicSpectralGrouping

        def _make_render_copy(g):
            return DynamicSpectralGrouping(
                n_agents=self.n_agents,
                n_envs=1,
                obs_dim=g.obs_dim,
                history_len=g.history_len,
                clustering_interval=g.clustering_interval,
                min_clusters=g._min_clusters,
                max_clusters=g._max_clusters,
                stability_threshold=g.stability_threshold,
                affinity_fn=g.affinity_fn,
            )

        if self._dynamic_groupings:
            saved = self._dynamic_groupings
            self._dynamic_groupings = [_make_render_copy(g) for g in saved]
            try:
                yield
            finally:
                self._dynamic_groupings = saved
        else:
            training_grouping = self._dynamic_grouping
            self._dynamic_grouping = _make_render_copy(training_grouping)
            try:
                yield
            finally:
                self._dynamic_grouping = training_grouping

    def on_rollout_reset(self) -> None:
        if self._dynamic_groupings:
            for g in self._dynamic_groupings:
                g.reset_history()
        elif self._dynamic_grouping is not None:
            self._dynamic_grouping.reset_history()

    def on_rollout_step(self, obs: np.ndarray) -> None:
        if self._dynamic_groupings:
            for g in self._dynamic_groupings:
                g.update_history(obs)
                g.maybe_update_groups()
        elif self._dynamic_grouping is not None:
            self._dynamic_grouping.update_history(obs)
            updated, agg_history = self._dynamic_grouping.maybe_update_groups()
            if updated and agg_history is not None and self.hypergraph_mode == "learned_affinity":
                self.agent.store_affinity_snapshot(agg_history)

    def on_env_done_mask(self, dones: np.ndarray) -> None:
        if self._dynamic_groupings:
            for g in self._dynamic_groupings:
                g.reset_history(env_mask=dones)
        elif self._dynamic_grouping is not None:
            self._dynamic_grouping.reset_history(env_mask=dones)

    def _compute_all_type_edge_lists(self, obs, infos, n_envs: int):
        """Compute per-type, per-env edge lists from observations and infos.

        Returns:
            all_type_edge_lists: [type_idx][env_idx] -> edge_list
        """
        if self.hypergraph_mode in ("hygma", "learned_affinity"):
            spectral_edges = self._dynamic_grouping.get_edge_lists(n_envs)
            return [spectral_edges]

        if self.hypergraph_mode == "combined_affinities":
            return [g.get_edge_lists(n_envs) for g in self._dynamic_groupings]

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
        if self.critic_type not in ("multi_hgnn", "hg_cross_attention"):
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
