# Feedback (Round 3): Updated HYGMA Integration Plan

## Summary
The updated `hygma_integration.md` is close to implementation-ready. Most prior concerns are addressed, and only a few high-value guardrails remain before coding.

## Remaining findings (ordered by severity)

### 1) `min_clusters` still needs upper clamping to `n_agents - 1` (High)
Current plan clamps `max_clusters` to `n_agents - 1`, but `min_clusters` can still exceed the valid silhouette range if configured too high.

Why this matters:
- Invalid `k` ranges can still reach clustering/scoring code and cause runtime issues.

Recommendation:
- Clamp both bounds into valid silhouette domain for each run:
  - `min_clusters = min(max(2, min_clusters), n_agents - 1)`
  - `max_clusters = min(max_clusters, n_agents - 1)`
- Then normalize to guarantee `max_clusters >= min_clusters`, or disable clustering if impossible.

### 2) Need runtime fallback around spectral/silhouette failures (Medium)
Even with better bounds and neighbor settings, sklearn can still fail on degenerate label outcomes (e.g., effectively one cluster).

Why this matters:
- A single bad clustering call can crash rollout collection.

Recommendation:
- Wrap clustering/score selection in `try/except` and fallback safely:
  - Prefer previous stable groups if available.
  - Otherwise fallback to single all-agent hyperedge.
- Log a concise warning once per interval to aid debugging without flooding output.

### 3) Eval/render frozen-group policy conflicts with checkpoint-only `view` path (Medium)
Plan says eval/render use frozen groups from training state. But `run.py view` loads network weights from checkpoint and renders immediately, while dynamic grouping state is not currently serialized in trainer checkpoint payloads.

Why this matters:
- In checkpoint-only sessions, HYGMA may remain in cold-start grouping and not reflect intended discovered structure.

Recommendation:
- Choose one explicit fix and document it in plan:
  - A) Serialize/restore grouping state (`_current_groups`, counters, possibly history) in `save_agent`/`load_agent`, or
  - B) Warm/update groups at eval/render start before first hypergraph build.
- Keep behavior consistent between training-time eval and offline `view`.

## Final recommendation
After these three adjustments, the plan is decision-complete and safe to implement.
