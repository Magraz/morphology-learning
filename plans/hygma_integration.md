# Plan: Integrate HYGMA Dynamic Spectral Clustering into Hypergraph Pipeline

## Context

The MAPPO trainer creates hypergraphs using two predefined edge functions (distance-based proximity and object-contact). HYGMA proposes an alternative: dynamically computing agent groups via spectral clustering on state history, then turning each group into a hyperedge. The goal is a clean `hypergraph_mode` switch between the existing predefined approach and the HYGMA spectral approach.

No changes needed to `mappo.py` or `hypergraph.py` — integration is entirely at the trainer level.

---

## Files

| File | Action |
|---|---|
| `algorithms/mappo/dynamic_grouping.py` | CREATE |
| `algorithms/mappo/types.py` | MODIFY |
| `algorithms/mappo/vec_trainer.py` | MODIFY |
| `pyproject.toml` | MODIFY |

---

## 1. NEW: `algorithms/mappo/dynamic_grouping.py`

Copy the `DynamicSpectralClustering` class directly from `HYGMA/src/utils/dynamic_clustering.py` into this file (it's ~89 lines, self-contained, only depends on sklearn). This avoids the brittle `sys.path.insert` approach and eliminates `utils.*` name collision risks.

Harden the copied class: set `n_neighbors=min(10, self.n_agents - 1)` in `SpectralClustering(affinity='nearest_neighbors', n_neighbors=...)` to prevent sklearn failures with small agent counts.

```python
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


class DynamicSpectralClustering:
    """Copied from HYGMA/src/utils/dynamic_clustering.py, hardened for small n_agents."""

    def _find_best_clustering(self, data):
        # Bound n_neighbors to avoid sklearn failure with few samples
        n_neighbors = min(10, self.n_agents - 1)
        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            try:
                sc = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=n_neighbors,
                )
                labels = sc.fit_predict(data)
                score = silhouette_score(data, labels)
                # ... update best if score improved
            except Exception:
                continue  # skip this k on failure
        # If no valid clustering found, return None, None
        # ... rest unchanged


class DynamicSpectralGrouping:
    def __init__(self, n_agents, n_envs, obs_dim, history_len,
                 clustering_interval, min_clusters, max_clusters,
                 stability_threshold):
        # Validate positive values
        assert history_len > 0, f"hygma_history_len must be > 0, got {history_len}"
        assert clustering_interval > 0, f"hygma_clustering_interval must be > 0, got {clustering_interval}"

        self.n_agents = n_agents
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.clustering_interval = clustering_interval
        self.stability_threshold = stability_threshold

        # Clamp cluster bounds to valid range for silhouette_score:
        # Both bounds must be in [2, n_agents - 1] (silhouette is undefined when k == n)
        self._min_clusters = min(max(2, min_clusters), max(2, n_agents - 1))
        self._max_clusters = min(max_clusters, max(2, n_agents - 1))
        # Normalize: ensure max >= min after clamping
        self._max_clusters = max(self._max_clusters, self._min_clusters)
        # Edge case: if n_agents <= 2 or bounds are degenerate, skip clustering
        self._clustering_possible = n_agents > 2 and self._max_clusters >= self._min_clusters

        # numpy buffer: (n_envs, n_agents, history_len, obs_dim)
        self._history = np.zeros(
            (n_envs, n_agents, history_len, obs_dim), dtype=np.float32
        )
        # Per-env valid-step counters (for weighted aggregation after partial resets)
        self._env_steps = np.zeros(n_envs, dtype=np.int32)
        # Dedicated rollout-step counter (increments by 1 per call, not by n_envs)
        self._rollout_steps = 0
        self._last_cluster_rollout_step = -clustering_interval

        self._clusterer = (
            DynamicSpectralClustering(
                self._min_clusters, self._max_clusters, n_agents
            )
            if self._clustering_possible
            else None
        )
        self._current_groups = None  # None = cold start
```

### Methods

**`update_history(obs: np.ndarray)`**
- `obs` shape: `(n_envs, n_agents, obs_dim)`
- Rolling shift: `self._history = np.roll(self._history, -1, axis=2)`
- Assign: `self._history[:, :, -1, :] = obs`
- Increment per-env counters: `self._env_steps += 1` (capped at `history_len`)
- Increment `self._rollout_steps += 1`

**`reset_history(env_mask: np.ndarray = None)`**
- If `env_mask` is None: zero entire buffer, `self._env_steps[:] = 0`, `self._rollout_steps = 0`
- If `env_mask` is bool array of shape `(n_envs,)`: zero `self._history[env_mask]`, set `self._env_steps[env_mask] = 0`

**`maybe_update_groups() -> bool`** (no arguments — uses internal `_rollout_steps`)
- Guard: `not self._clustering_possible` → return False
- Guard: no env has a full window (`self._env_steps.max() < history_len`) → return False (cold start)
- Guard: `_rollout_steps // clustering_interval == self._last_cluster_rollout_step // clustering_interval` → return False
- Aggregate only envs with full windows:
  ```python
  valid = self._env_steps >= self.history_len
  agg = self._history[valid].mean(axis=0)  # (n_agents, history_len, obs_dim)
  ```
- Convert: `agg_tensor = torch.from_numpy(agg)`
- Wrap clustering in try/except to handle degenerate data gracefully:
  ```python
  try:
      updated, new_groups, _ = self._clusterer.update_groups(agg_tensor, self.stability_threshold)
  except Exception as e:
      import warnings
      warnings.warn(f"Spectral clustering failed, keeping previous groups: {e}")
      return False
  ```
- If updated: `self._current_groups = new_groups`
- Update `self._last_cluster_rollout_step = self._rollout_steps`
- Return `updated`

**`get_edge_lists(n_envs: int) -> list[list[tuple]]`**
- Cold start (`_current_groups` is None): return `[[tuple(range(self.n_agents))]] * n_envs`
  - Single all-agent hyperedge per env; valid DHG, degrades gracefully
- Otherwise: convert `_current_groups` to edge list once, replicate n_envs times
  ```python
  edge_list = [tuple(sorted(g)) for g in self._current_groups if g]
  return [edge_list] * n_envs
  ```

---

## 2. MODIFY: `algorithms/mappo/types.py`

Add to `Model_Params` dataclass:

```python
# HYGMA dynamic spectral clustering mode
hypergraph_mode: str = "predefined"       # "predefined" | "hygma"
hygma_history_len: int = 50               # rolling obs window length
hygma_clustering_interval: int = 100      # rollout steps between cluster runs
hygma_min_clusters: int = 2
hygma_max_clusters: int = 0               # 0 = auto (n_agents - 1)
hygma_stability_threshold: float = 0.6
```

---

## 3. MODIFY: `algorithms/mappo/vec_trainer.py`

### 3a. `__init__` — after existing `hyperedge_fns` block

Config validation + construction:

```python
self.hypergraph_mode = model_params.hypergraph_mode

if self.hypergraph_mode == "hygma":
    from algorithms.mappo.dynamic_grouping import DynamicSpectralGrouping

    # Validate config
    assert self.critic_type == "multi_hgnn", (
        "HYGMA mode requires critic_type='multi_hgnn'"
    )
    assert model_params.n_hyperedge_types == 1, (
        "HYGMA mode requires n_hyperedge_types=1"
    )

    max_k = model_params.hygma_max_clusters or (self.n_agents - 1)
    self._dynamic_grouping = DynamicSpectralGrouping(
        n_agents=self.n_agents,
        n_envs=self.n_parallel_envs,
        obs_dim=observation_dim,
        history_len=model_params.hygma_history_len,
        clustering_interval=model_params.hygma_clustering_interval,
        min_clusters=model_params.hygma_min_clusters,
        max_clusters=max_k,
        stability_threshold=model_params.hygma_stability_threshold,
    )
elif self.hypergraph_mode == "predefined":
    self._dynamic_grouping = None
else:
    raise ValueError(
        f"Unknown hypergraph_mode: {self.hypergraph_mode!r}. "
        f"Expected 'predefined' or 'hygma'."
    )
```

### 3b. `_build_inference_hypergraphs` — replace the edge-list generation block

Replace the `for fn, source in self.hyperedge_fns:` loop with:

```python
if self.hypergraph_mode == "hygma":
    spectral_edges = self._dynamic_grouping.get_edge_lists(n_envs)
    all_type_edge_lists = [spectral_edges]   # one type: [env0_edges, env1_edges, ...]
else:
    # existing predefined logic (unchanged)
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
```

Everything after this (transpose, canonicalize, cache, `batch_hypergraphs`) is **unchanged**.

### 3c. `collect_trajectory` — three additions

**After `obs, infos = self.vec_env.reset(...)`:**
```python
if self._dynamic_grouping is not None:
    self._dynamic_grouping.reset_history()
```

**Inside step loop, after `global_states = obs.reshape(...)`, before `_build_inference_hypergraphs`:**
```python
if self._dynamic_grouping is not None:
    self._dynamic_grouping.update_history(obs)
    self._dynamic_grouping.maybe_update_groups()
```

**Inside `if dones.any():` block:**
```python
if self._dynamic_grouping is not None:
    self._dynamic_grouping.reset_history(env_mask=dones)
```

### 3d. `render()` — make entropy logging dynamic over `n_hyperedge_types`

Replace the hardcoded `entropies[0]`/`entropies[1]` indexing (lines 664-673) with a loop driven by a type-names list. This prevents IndexError when HYGMA mode produces only 1 type.

Store `self._entropy_type_names` in `__init__`:
```python
# In __init__, after hypergraph_mode setup:
if self.hypergraph_mode == "hygma":
    self._entropy_type_names = ["proximity"]  # compatibility key for single type
else:
    self._entropy_type_names = ["proximity", "object"]
```

Replace hardcoded entropy collection in `render()`:

Before (current):
```python
entropy_proximity_log = []
entropy_object_log = []
soft_entropy_proximity_log = []
soft_entropy_object_log = []
```

After:
```python
n_types = len(self._entropy_type_names)
entropy_type_logs = [[] for _ in range(n_types)]
soft_entropy_type_logs = [[] for _ in range(n_types)]
```

Replace the inner loop:
```python
if render_hgs is not None:
    entropies = compute_hyperedge_structural_entropy_batch(render_hgs)
    soft_entropies = compute_soft_hyperedge_structural_entropy_batch(render_hgs)
    for t in range(len(render_hgs)):
        entropy_type_logs[t].append(entropies[t])
        soft_entropy_type_logs[t].append(soft_entropies[t])
```

Build `entropy_logs` dict at end:
```python
entropy_logs = {}
for t, name in enumerate(self._entropy_type_names):
    entropy_logs[name] = np.array(entropy_type_logs[t])
    entropy_logs[f"soft_{name}"] = np.array(soft_entropy_type_logs[t])
entropy_logs["predicted_per_agent"] = (
    np.array(predicted_entropy_log) if predicted_entropy_log else None
)
```

This keeps `run.py` working unchanged: both modes produce `"proximity"` / `"soft_proximity"` keys. Predefined mode additionally produces `"object"` / `"soft_object"`. The `run.py` plotting iterates dynamically over available keys (line 112), and its `type_names` fallback (line 176: `type_names[t] if t < len(type_names) else f"type_{t}"`) handles any count.

### 3e. Checkpoint serialization of grouping state

Add grouping state to `save_agent` / `load_agent` so that `view` mode (which loads from checkpoint without training) has the discovered groups available instead of falling back to cold-start.

**In `save_agent()`**, after saving model state dicts:
```python
if self._dynamic_grouping is not None:
    state["dynamic_grouping"] = {
        "current_groups": self._dynamic_grouping._current_groups,
        "env_steps": self._dynamic_grouping._env_steps.copy(),
        "rollout_steps": self._dynamic_grouping._rollout_steps,
        "last_cluster_rollout_step": self._dynamic_grouping._last_cluster_rollout_step,
    }
```

**In `load_agent()`**, after loading model state dicts:
```python
if self._dynamic_grouping is not None and "dynamic_grouping" in checkpoint:
    dg_state = checkpoint["dynamic_grouping"]
    self._dynamic_grouping._current_groups = dg_state["current_groups"]
    self._dynamic_grouping._env_steps[:] = dg_state["env_steps"]
    self._dynamic_grouping._rollout_steps = dg_state["rollout_steps"]
    self._dynamic_grouping._last_cluster_rollout_step = dg_state["last_cluster_rollout_step"]
```

This ensures `view` mode uses the groups discovered during training. History buffer is **not** serialized (large, not needed — groups are already computed).

### 3f. Eval/render grouping policy: frozen from training (explicit)

During eval and render, `_build_inference_hypergraphs` uses whatever `_current_groups` was last computed during training. No `update_history` or `maybe_update_groups` calls are made. This is intentional: eval measures the policy's performance under the discovered grouping structure without further adaptation. Document this in a comment at the top of `evaluate()` and `render()`.

---

## 4. MODIFY: `pyproject.toml`

Add `scikit-learn` to dependencies:

```toml
dependencies = [
    "dhg>=0.9.5",
    "dill>=0.4.1",
    "gymnasium[box2d]>=1.2.3",
    "imageio>=2.37.3",
    "imageio-ffmpeg>=0.6.0",
    "notebook>=7.5.3",
    "pandas>=2.3.3",
    "scikit-learn>=1.3.0",
]
```

---

## Key Design Decisions

- **No `sys.path` hacks**: Copy `DynamicSpectralClustering` directly into `dynamic_grouping.py`. It's small (~89 lines), self-contained, avoids `utils.*` name collisions and cwd-dependent imports. Hardened with explicit `n_neighbors=min(10, n_agents - 1)` for small agent counts.
- **Dedicated rollout-step counter**: `_rollout_steps` increments by 1 per `update_history` call (one per env-step in the collection loop), not by `n_envs` or `batch_size`. This makes `hygma_clustering_interval` mean what it says: N rollout steps between cluster runs. `maybe_update_groups()` is no-arg — uses the internal counter, no private-field access from outside.
- **Per-env history validity**: `_env_steps[i]` tracks how many valid obs each env has contributed. On `maybe_update_groups`, only envs with `_env_steps >= history_len` are included in the mean aggregation. This prevents freshly-reset env histories (full of zeros) from diluting the clustering signal.
- **Cluster bounds clamping + normalization**: Both `min_clusters` and `max_clusters` are clamped to `[2, n_agents - 1]`, then `max_clusters = max(max_clusters, min_clusters)` to prevent empty ranges. If `n_agents <= 2` or bounds are still degenerate, clustering is disabled entirely and the cold-start fallback is used permanently.
- **Runtime clustering fallback**: Spectral clustering and silhouette scoring are wrapped in try/except. On failure (degenerate data, edge-case label distributions), the previous stable groups are kept and a warning is logged once per interval. This prevents a single bad clustering call from crashing rollout collection.
- **Checkpoint serialization**: `_current_groups`, `_env_steps`, `_rollout_steps`, and `_last_cluster_rollout_step` are saved/restored in `save_agent`/`load_agent`. This ensures `view` mode (checkpoint-only) uses discovered groups instead of cold-start fallback. History buffer is not serialized (large, not needed).
- **Init validation**: `assert history_len > 0` and `assert clustering_interval > 0` to prevent division/modulo issues.
- **Cold-start fallback**: `[tuple(range(n_agents))]` — single all-agents hyperedge. Valid DHG, degrades gracefully to "everyone shares info."
- **Render key compatibility**: HYGMA single-type mode uses `"proximity"` / `"soft_proximity"` as key names so `run.py` plotting works unchanged. No consumer updates needed.
- **Eval/render policy**: frozen groups from training. Explicitly documented.

---

## Verification Steps

1. `hypergraph_mode: "predefined"` → `_dynamic_grouping` is None, existing behavior fully unchanged
2. `hypergraph_mode: "hygma"` + `n_hyperedge_types: 1` → spectral groups used after cold start
3. Cold start: first `history_len` rollout steps use fallback single hyperedge; no crash
4. After `clustering_interval` rollout steps: `_current_groups` is set and edge lists reflect actual clusters
5. Assertion fires when `hypergraph_mode: "hygma"` but `n_hyperedge_types != 1`
6. Assertion fires when `hypergraph_mode: "hygma"` but `critic_type != "multi_hgnn"`
7. ValueError fires for unknown `hypergraph_mode` values
8. Render works with 1 hyperedge type (no IndexError); `run.py` plots correctly via `"proximity"` key
9. Partial env resets: `reset_history(env_mask=dones)` zeros only done envs; clustering aggregation skips envs with insufficient history
10. `n_agents == 2`: clustering is disabled, permanent cold-start fallback, no crash
11. `scikit-learn` installs cleanly from `pyproject.toml`
12. `hygma_max_clusters=1` with `hygma_min_clusters=2`: bounds normalize to `min=max=2`, clustering proceeds normally
13. `hygma_history_len=0` or `hygma_clustering_interval=0`: assertion fires at init
14. Small `n_agents` (3-4): `SpectralClustering` runs without `n_neighbors` errors
15. `hygma_min_clusters=10` with `n_agents=5`: `min_clusters` clamped to `4` (`n_agents - 1`), no invalid silhouette range
16. Degenerate clustering data (e.g., all-zero history): try/except catches failure, previous groups kept, warning logged
17. `view` mode after checkpoint load: `_current_groups` restored from checkpoint, render uses discovered groups (not cold-start fallback)
