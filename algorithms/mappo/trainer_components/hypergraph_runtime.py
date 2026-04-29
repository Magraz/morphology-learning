from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import dhg


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
        hyperedge_fns: list[tuple] = None,
    ):
        self.agent = agent
        self.device = device
        self.n_agents = n_agents
        self.n_parallel_envs = n_parallel_envs
        self.critic_type = critic_type
        self.hypergraph_mode = model_params.hypergraph_mode

        self.hyperedge_fns = hyperedge_fns or []

        # Plural list used only by combined_affinities mode
        self._dynamic_groupings: list | None = None

        # Learned-grouping rolling obs history and last-emitted token sequences.
        # Populated only when hypergraph_mode == "learned_grouping".
        self._grouping_history_len: int = 0
        self._grouping_obs_history: torch.Tensor | None = None
        self._last_grouping_tokens: list[list[int]] | None = None

        if self.hypergraph_mode in ("hygma", "learned_affinity"):
            from algorithms.mappo.dynamic_grouping import DynamicSpectralGrouping

            assert self.critic_type in (
                "multi_hgnn", "hg_cross_attention"
            ), f"{self.hypergraph_mode} mode requires critic_type='multi_hgnn' or 'hg_cross_attention'"

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
            self._entropy_type_names = list(model_params.hyperedge_fn_names or [])
        elif self.hypergraph_mode == "learned_grouping":
            assert self.critic_type in (
                "multi_hgnn", "hg_cross_attention"
            ), "learned_grouping mode requires critic_type='multi_hgnn' or 'hg_cross_attention'"
            assert agent.network_old.grouping_transformer is not None, (
                "learned_grouping mode requires GroupingTransformer on MAPPONetwork"
            )
            self._grouping_history_len = model_params.grouping_history_len
            self._grouping_obs_history = torch.zeros(
                self.n_parallel_envs,
                self.n_agents,
                self._grouping_history_len,
                agent.observation_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self._last_grouping_tokens = [None] * self.n_parallel_envs
            self._dynamic_grouping = None
            self._entropy_type_names = ["learned_grouping"]
        else:
            raise ValueError(
                f"Unknown hypergraph_mode: {self.hypergraph_mode!r}. "
                "Expected 'predefined', 'hygma', 'learned_affinity', "
                "'combined_affinities', or 'learned_grouping'."
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
        if self._grouping_obs_history is not None:
            self._grouping_obs_history.zero_()

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
        if self._grouping_obs_history is not None:
            obs_t = torch.from_numpy(
                np.ascontiguousarray(obs, dtype=np.float32)
            ).to(self._grouping_obs_history.device)
            self._grouping_obs_history = torch.roll(
                self._grouping_obs_history, shifts=-1, dims=2
            )
            self._grouping_obs_history[:, :, -1, :] = obs_t

    def on_env_done_mask(self, dones: np.ndarray) -> None:
        if self._dynamic_groupings:
            for g in self._dynamic_groupings:
                g.reset_history(env_mask=dones)
        elif self._dynamic_grouping is not None:
            self._dynamic_grouping.reset_history(env_mask=dones)
        if self._grouping_obs_history is not None:
            self._grouping_obs_history[dones] = 0.0

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

        if self.hypergraph_mode == "learned_grouping":
            edges, tokens = self._sample_learned_groupings(n_envs)
            self._last_grouping_tokens = tokens
            return [edges]

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

    @torch.no_grad()
    def _sample_learned_groupings(self, n_envs: int):
        """Run the GroupingTransformer once on the current obs history window.

        Returns:
            edge_lists: list[list[tuple]] of length n_envs.
            token_seqs: list[list[int]] of length n_envs (the sampled token
                        sequence per env, including its terminal EOS).
        """
        gt = self.agent.network_old.grouping_transformer
        history = self._grouping_obs_history[:n_envs]
        edges, tokens = gt.generate_with_tokens(history, sample=True)
        return edges, tokens

    def get_last_grouping_tokens(self) -> list[list[int]] | None:
        """Return the per-env token sequences emitted by the most recent
        learned-grouping inference call, or None if not in that mode."""
        return self._last_grouping_tokens

    def build_inference_hypergraphs(self, obs, infos, n_envs: int):
        """Build batched block-diagonal hypergraphs for critic inference."""
        if self.critic_type not in ("multi_hgnn", "hg_cross_attention"):
            return None, None

        all_type_edge_lists = self._compute_all_type_edge_lists(obs, infos, n_envs)

        n_types = len(all_type_edge_lists)
        per_env_edge_lists = [
            [all_type_edge_lists[t][e] for t in range(n_types)] for e in range(n_envs)
        ]

        hg_cache = self.agent.hg_cache
        per_env_sig_ids = [hg_cache.intern(e) for e in per_env_edge_lists]

        sig_key = tuple(per_env_sig_ids)
        batched_hgs = [
            hg_cache.get_or_build_batched_by_type(
                sig_key, type_idx, self.device, cache_scope="inference"
            )
            for type_idx in range(n_types)
        ]

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
