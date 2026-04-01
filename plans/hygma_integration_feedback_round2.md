# Feedback (Round 2): Updated HYGMA Integration Plan

## Summary
The updated plan is much stronger and resolves most of the original high-risk issues. It is close to implementation-ready. A few guardrails should still be added to prevent edge-case runtime failures and key-name mismatches.

## Remaining findings (ordered by severity)

### 1) Small-`n_agents` safety in copied `DynamicSpectralClustering` (High)
The plan now copies `DynamicSpectralClustering` into `algorithms/mappo/dynamic_grouping.py` unchanged. In the source implementation, `SpectralClustering` is created with `affinity='nearest_neighbors'` and default neighbor settings.

Why this matters:
- With low sample counts, sklearn nearest-neighbor spectral clustering can fail unless `n_neighbors` is explicitly bounded.
- Even if clustering is disabled for `n_agents <= 2`, low-but-valid counts can still be fragile.

Recommendation:
- In the copied class, set `n_neighbors=min(10, self.n_agents - 1)` (and at least 1).
- Keep this as an explicit constructor argument or harden internally.

### 2) Cluster bound normalization can still produce invalid ranges (Medium)
The plan clamps:
- `min_clusters >= 2`
- `max_clusters <= n_agents - 1`

But if user config sets `hygma_max_clusters` very low (e.g., 1), `_min_clusters` can end up greater than `_max_clusters`.

Why this matters:
- Range loops for candidate `k` may be empty or inconsistent.
- Behavior may silently degrade or crash depending on implementation details.

Recommendation:
- Normalize after clamping: enforce `_max_clusters = max(_max_clusters, _min_clusters)`.
- If normalization is impossible or meaningless, disable clustering and use fallback edge policy.

### 3) Missing validation for positive interval/history (Medium)
The plan uses interval arithmetic in `maybe_update_groups` and rolling history windows but does not explicitly require positive values.

Why this matters:
- `clustering_interval <= 0` can cause division/modulo issues.
- `history_len <= 0` breaks history logic.

Recommendation:
- Add init assertions:
  - `hygma_history_len > 0`
  - `hygma_clustering_interval > 0`

### 4) Entropy log key naming should stay consumer-compatible (Medium)
The updated plan suggests HYGMA render keys like `spectral` / `soft_spectral`. Current plotting in `algorithms/mappo/run.py` still expects compatibility patterns like `soft_{name}` where `name` is from `['proximity', 'object']`.

Why this matters:
- Swapping key names without updating consumers can break prediction-vs-actual plotting paths.

Recommendation:
- Lock one strategy explicitly. Preferred for minimal churn:
  - Keep compatibility keys in HYGMA single-type mode (`proximity` + `soft_proximity`) so `run.py` continues to work unchanged.
- If using spectral keys instead, explicitly include required `run.py` updates in the plan.

### 5) API cleanliness: avoid external access to private counter (Low)
The plan calls:
- `maybe_update_groups(self._dynamic_grouping._rollout_steps)`

Why this matters:
- Exposes internals and couples trainer to grouping implementation details.

Recommendation:
- Make `maybe_update_groups()` no-arg and use internal `_rollout_steps`.

## Final recommendation
After adding the five adjustments above, the plan is decision-complete and safe to implement.
