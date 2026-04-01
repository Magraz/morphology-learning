# Feedback: HYGMA Integration Plan

## Summary
The proposed integration is directionally good, but there are several concrete implementation risks that should be addressed before coding to avoid runtime failures and misleading training behavior.

## Findings (ordered by severity)

### 1) Render path likely breaks when `n_hyperedge_types=1` (High)
The plan requires HYGMA mode to use a single hyperedge type. Current `render()` logic in `algorithms/mappo/vec_trainer.py` assumes two hyperedge types and indexes `entropies[1]` and `soft_entropies[1]` unconditionally.

Why this matters:
- In HYGMA mode, the entropy arrays will likely have length 1.
- This can trigger an index error during rendering/logging.

Recommendation:
- Make entropy logging in `render()` dynamic over `len(render_hgs)` (or `model_params.n_hyperedge_types`) instead of hardcoded proximity/object indexing.

### 2) Clustering interval uses mismatched step units (High)
The plan passes `total_step_count` into `maybe_update_groups()`, but in `collect_trajectory()` this counter increments by `batch_size` (number of parallel envs), not by one environment step.

Why this matters:
- `hygma_clustering_interval` is documented as "env steps between cluster runs".
- Actual trigger cadence becomes ~`n_envs` times faster than intended.

Recommendation:
- Track a dedicated per-loop env-step counter (increment by 1 each rollout step), or explicitly redefine interval semantics to sampled transitions.

### 3) Cluster count bounds can violate silhouette constraints (High)
The plan sets `max_clusters` to `n_agents` by default. HYGMA code evaluates silhouette score across that range.

Why this matters:
- `silhouette_score` is invalid when `n_clusters == n_samples`.
- Can raise runtime exceptions depending on `n_agents` and labels.

Recommendation:
- Clamp to valid range: `min_clusters >= 2`, `max_clusters <= n_agents - 1`.
- Add robust fallback for tiny agent counts (e.g., no clustering / one all-agent edge).

### 4) Missing dependency in project config (High)
HYGMA clustering imports `sklearn` (`SpectralClustering`, `silhouette_score`), but `scikit-learn` is not listed in `pyproject.toml`.

Why this matters:
- Fresh environments may fail immediately at import/runtime.

Recommendation:
- Add `scikit-learn` to dependencies and pin a compatible version range.

### 5) `sys.path.insert("HYGMA/src")` is brittle (Medium)
The proposed import approach depends on current working directory and may resolve ambiguously.

Why this matters:
- Running from a different cwd can break imports.
- Name collisions with generic modules (e.g., `utils`) become more likely.

Recommendation:
- Resolve paths via `Path(__file__)` to repo root, or package HYGMA utilities with explicit module paths.
- Avoid broad `utils.*` imports when possible.

### 6) Partial env resets + global history counter are under-specified (Medium)
Plan resets history per done env, but uses one global `_steps_observed` gate for cold start.

Why this matters:
- Clustering may run while some env histories are newly zeroed.
- Mean aggregation across envs can be skewed after asynchronous resets.

Recommendation:
- Track valid-history lengths per env and only aggregate envs with full windows, or weight by valid timesteps.

### 7) Eval/render staleness behavior should be explicit (Medium)
Plan states eval/render should use last training groups. Current eval/render do not update grouping state themselves.

Why this matters:
- If render/eval are run without prior training updates in process, grouping may remain cold-start fallback.
- Reported behavior may differ from expected HYGMA dynamics.

Recommendation:
- Choose one policy and encode it explicitly:
  - A) frozen groups from training (documented), or
  - B) live grouping updates during eval/render.

## Additional suggestions

1. Add config validation early in trainer init:
- `hypergraph_mode in {"predefined", "hygma"}`
- HYGMA mode requires `critic_type == "multi_hgnn"`
- HYGMA mode requires `n_hyperedge_types == 1`
- Cluster bounds validity checks

2. Add focused smoke tests:
- HYGMA cold start path builds valid batched hypergraphs.
- First clustering update occurs at intended cadence.
- Render/eval paths run without indexing assumptions.
- Done-mask resets do not crash or produce malformed edge lists.

3. Keep fallback hyperedge canonical and explicit:
- Prefer `[tuple(range(n_agents))]` for cold start and empty-group fallback.

## Conclusion
The integration concept is good and can stay trainer-centric, but it needs guardrails around cadence, cluster bounds, dependency management, and rendering assumptions to be production-safe.
