import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


class DynamicSpectralClustering:
    """HYGMA spectral grouping helper with small-sample hardening."""

    def __init__(
        self,
        min_clusters: int,
        max_clusters: int,
        n_agents: int,
    ):
        self.min_clusters = int(min_clusters)
        self.max_clusters = int(max_clusters)
        self.n_agents = int(n_agents)
        self.current_groups = None

    def cluster(
        self,
        state_history: torch.Tensor,
        affinity_matrix: np.ndarray | None = None,
    ) -> list[list[int]]:
        batch_size, _, _ = state_history.shape
        reshaped_states = state_history.view(batch_size, -1).cpu().numpy()
        best_n_clusters, best_labels = self._find_best_clustering(
            reshaped_states, affinity_matrix
        )

        new_groups = [[] for _ in range(best_n_clusters)]
        for i in range(self.n_agents):
            group = int(best_labels[i])
            new_groups[group].append(i)

        return [group for group in new_groups if group]

    def _find_best_clustering(
        self,
        data: np.ndarray,
        affinity_matrix: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        best_score = -1.0
        best_n_clusters = self.min_clusters
        best_labels = None

        use_precomputed = affinity_matrix is not None
        n_samples = int(data.shape[0])
        n_neighbors = max(1, min(10, self.n_agents - 1, n_samples - 1))

        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            try:
                sc_kwargs = dict(
                    n_clusters=n_clusters,
                    affinity="precomputed" if use_precomputed else "nearest_neighbors",
                    random_state=0,
                )
                if not use_precomputed:
                    sc_kwargs["n_neighbors"] = n_neighbors
                spectral_clustering = SpectralClustering(**sc_kwargs)
                fit_data = affinity_matrix if use_precomputed else data
                labels = spectral_clustering.fit_predict(fit_data)
                if np.unique(labels).shape[0] < 2:
                    continue
                score = silhouette_score(data, labels)
            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels

        if best_labels is None:
            fallback_k = max(1, min(self.max_clusters, self.n_agents))
            fallback_labels = np.arange(self.n_agents, dtype=np.int64) % fallback_k
            return fallback_k, fallback_labels

        return best_n_clusters, best_labels

    def update_groups(
        self,
        state_history: torch.Tensor,
        stability_threshold: float,
        affinity_matrix: np.ndarray | None = None,
    ) -> tuple[bool, list[list[int]], int]:
        new_groups = self.cluster(state_history, affinity_matrix)

        if self.current_groups is None:
            self.current_groups = new_groups
            return True, new_groups, self.n_agents

        num_moved = self._count_moved_agents(self.current_groups, new_groups)
        if num_moved == 0:
            return False, self.current_groups, num_moved
        if num_moved / self.n_agents < stability_threshold:
            return False, self.current_groups, num_moved

        self.current_groups = new_groups
        return True, new_groups, num_moved

    def _count_moved_agents(
        self, old_groups: list[list[int]], new_groups: list[list[int]]
    ) -> int:
        old_set = {frozenset(group) for group in old_groups}
        new_set = {frozenset(group) for group in new_groups}
        if old_set == new_set:
            return 0

        moved = 0
        old_group_map = {
            agent: frozenset(group) for group in old_groups for agent in group
        }
        new_group_map = {
            agent: frozenset(group) for group in new_groups for agent in group
        }
        for agent in range(self.n_agents):
            if old_group_map[agent] != new_group_map[agent]:
                moved += 1
        return moved


class DynamicSpectralGrouping:
    def __init__(
        self,
        n_agents: int,
        n_envs: int,
        obs_dim: int,
        history_len: int,
        clustering_interval: int,
        min_clusters: int,
        max_clusters: int,
        stability_threshold: float,
        affinity_fn=None,
    ):
        """
        Args:
            affinity_fn: Optional callable (torch.Tensor) -> np.ndarray.
                         Receives aggregated history (n_agents, history_len, obs_dim)
                         and returns an (n_agents, n_agents) affinity matrix.
        """
        assert history_len > 0, f"hygma_history_len must be > 0, got {history_len}"
        assert (
            clustering_interval > 0
        ), f"hygma_clustering_interval must be > 0, got {clustering_interval}"

        self.n_agents = int(n_agents)
        self.n_envs = int(n_envs)
        self.obs_dim = int(obs_dim)
        self.history_len = int(history_len)
        self.clustering_interval = int(clustering_interval)
        self.stability_threshold = float(stability_threshold)
        self.affinity_fn = affinity_fn

        self._clustering_possible = self.n_agents > 2
        if self._clustering_possible:
            max_valid_k = self.n_agents - 1
            self._min_clusters = min(max(2, int(min_clusters)), max_valid_k)
            self._max_clusters = min(int(max_clusters), max_valid_k)
            self._max_clusters = max(self._max_clusters, self._min_clusters)
        else:
            self._min_clusters = 0
            self._max_clusters = 0

        self._history = np.zeros(
            (self.n_envs, self.n_agents, self.history_len, self.obs_dim),
            dtype=np.float32,
        )
        self._env_steps = np.zeros(self.n_envs, dtype=np.int32)
        self._rollout_steps = 0
        self._last_cluster_rollout_step = -self.clustering_interval

        self._clusterer = (
            DynamicSpectralClustering(
                self._min_clusters, self._max_clusters, self.n_agents,
            )
            if self._clustering_possible
            else None
        )
        self._current_groups = None

    def update_history(self, obs: np.ndarray) -> None:
        if obs.shape != (self.n_envs, self.n_agents, self.obs_dim):
            raise ValueError(
                f"obs shape must be {(self.n_envs, self.n_agents, self.obs_dim)}, "
                f"got {obs.shape}"
            )
        self._history = np.roll(
            self._history, -1, axis=2
        )  # FIFO sliding window buffer, contains only the most recent observations
        self._history[:, :, -1, :] = obs
        self._env_steps = np.minimum(self._env_steps + 1, self.history_len)
        self._rollout_steps += 1

    def reset_history(self, env_mask: np.ndarray = None) -> None:
        if env_mask is None:
            self._history.fill(0.0)
            self._env_steps[:] = 0
            self._rollout_steps = 0
            self._last_cluster_rollout_step = -self.clustering_interval
            self._current_groups = None
            if self._clusterer is not None:
                self._clusterer.current_groups = None
            return

        env_mask = np.asarray(env_mask, dtype=bool)
        if env_mask.shape != (self.n_envs,):
            raise ValueError(
                f"env_mask shape must be {(self.n_envs,)}, got {env_mask.shape}"
            )
        self._history[env_mask] = 0.0
        self._env_steps[env_mask] = 0

    def maybe_update_groups(self) -> tuple[bool, np.ndarray | None]:
        """Attempt re-clustering if the interval has elapsed.

        Returns:
            (updated, agg_history) where agg_history is the (n_agents,
            history_len, obs_dim) array used for clustering when it fired,
            or None when clustering was skipped.
        """
        if not self._clustering_possible:
            return False, None
        if self._env_steps.max() < self.history_len:
            return False, None

        if (
            self._rollout_steps // self.clustering_interval
            == self._last_cluster_rollout_step // self.clustering_interval
        ):
            return False, None

        valid = self._env_steps >= self.history_len
        agg = self._history[valid].mean(axis=0)
        agg_tensor = torch.from_numpy(agg)

        affinity_matrix = None
        if self.affinity_fn is not None:
            affinity_matrix = self.affinity_fn(agg_tensor)

        updated, new_groups, _ = self._clusterer.update_groups(
            agg_tensor, self.stability_threshold, affinity_matrix
        )
        if updated:
            self._current_groups = new_groups

        self._last_cluster_rollout_step = self._rollout_steps
        return updated, agg

    def get_edge_lists(self, n_envs: int) -> list[list[tuple]]:
        if self._current_groups is None:
            edge_list = [tuple(range(self.n_agents))]
        else:
            edge_list = [tuple(sorted(g)) for g in self._current_groups if g]
            if not edge_list:
                edge_list = [tuple(range(self.n_agents))]

        return [list(edge_list) for _ in range(n_envs)]
