# Make GroupingTransformer-produced hypergraphs persistent across timesteps

## Context

The predefined hypergraph constructors in [algorithms/mappo/hypergraph.py](algorithms/mappo/hypergraph.py) (e.g. `distance_based_hyperedges`, `smaclite_ally_visibility_hyperedges`) achieve persistence and gradualness **passively** — continuous observation signals + hard thresholds mean consecutive hypergraphs are usually byte-identical, and rare threshold crossings flip only one or two edges at a time.

The learned `GroupingTransformer` ([algorithms/mappo/networks/grouping_transformer.py](algorithms/mappo/networks/grouping_transformer.py)) by contrast is invoked at **every rollout step** in [`HypergraphRuntime._sample_learned_groupings`](algorithms/mappo/trainer_components/hypergraph_runtime.py#L324-L335) via `gt.generate_with_tokens(history, sample=True)`, sampling a fresh sequence from scratch with no bias toward the previous timestep's structure. The result is high-frequency, often discontinuous changes in team structure that are noisy as critic input and as a REINFORCE target, and that waste compute.

We will make the learned hypergraph **hold for N consecutive timesteps** before regenerating, mirroring the existing `hygma_clustering_interval` pattern from `DynamicSpectralGrouping`. This addresses the persistence requirement directly and reduces GT calls by a factor of N. Per-decision changes can still be drastic — that is a known limitation of this approach and is accepted; gradualness *between* decisions is out of scope for this plan and can be layered on later (e.g. token-KL regularizer or previous-structure conditioning) if needed.

## Approach

Run the GT only once every `grouping_interval` steps; on hold-steps, reuse the cached `(edges, tokens)` from the last decision step. To keep downstream consumers and the per-step indexing in `compute_grouping_loss` working unchanged, *replicate* the cached tokens into `_grouping_tokens` on every step. In the loss, filter to decision steps and use the **mean** of the next `grouping_interval` advantages as the credit signal for that decision.

## Concrete change set

### 1. Config
[algorithms/mappo/types.py](algorithms/mappo/types.py) — add to the learned-grouping config block (near `hygma_clustering_interval` at line 56):

```python
grouping_interval: int = 4  # hold the learned hypergraph for this many env steps
```

### 2. Runtime caching (skip regeneration on hold-steps)
[algorithms/mappo/trainer_components/hypergraph_runtime.py](algorithms/mappo/trainer_components/hypergraph_runtime.py) — in `_sample_learned_groupings`:

- Add per-env counters `self._grouping_step_counter: torch.Tensor (n_envs,)` and per-env caches `self._cached_grouping_edges: list[list[tuple[int,...]]]`, `self._cached_grouping_tokens: list[list[int]]`.
- On entry, for envs where `step_counter % grouping_interval == 0` **or** the cache is empty (start of episode / post-reset), call `gt.generate_with_tokens(history, sample=True)` *only on those envs* (slice the history tensor) and overwrite their cache entries.
- For all other envs, return the cached values.
- Increment `step_counter`; reset to 0 on episode-end (hook into the existing episode-reset path that already zeros `_grouping_obs_history`).

This keeps the function's external interface identical — same shape of returned `edges` and `tokens` per call.

### 3. Token bookkeeping (no-op for downstream)
[algorithms/mappo/mappo.py](algorithms/mappo/mappo.py) at the existing `_grouping_tokens.append(...)` site (around line 699 in `store_transitions_batch`) — **no code change needed**, because step 2 already returns the cached tokens on hold-steps. The buffer continues to receive one (possibly repeated) token sequence per step, so all per-step indexing and downstream consumers (critic, intrinsic reward encoder, hg_cache) remain unchanged.

### 4. Loss change — decision-step filtering + windowed-mean advantage
[`compute_grouping_loss` in algorithms/mappo/mappo.py:182-268](algorithms/mappo/mappo.py#L182-L268):

- Read `grouping_interval` from `self` (plumb through from the config in `__init__`).
- In the loop over `(env_idx, t)` at line 216, additionally skip samples where `t % grouping_interval != 0`. (Token sequences on hold-steps are duplicates and would inflate the gradient on the same decision.)
- Replace the `valid_advantages.append(advantages[k])` line at 242 with the windowed mean:
  ```python
  end = min(k + grouping_interval, len(advantages))
  valid_advantages.append(advantages[k:end].mean())
  ```
  (Indexing assumes `advantages` is aligned with the same `(env_indices, ts_indices)` minibatch; if it is not, gather by `(env_idx, t')` for `t' in [t, t+grouping_interval)` from the per-env advantage buffer instead.)
- Everything else in the loss (token padding, teacher-forced forward, log-prob gather, REINFORCE) stays as-is.

## Verification

- **Smoke test**: run the existing `if __name__ == "__main__"` block at [grouping_transformer.py:549](algorithms/mappo/networks/grouping_transformer.py#L549) — confirms the network is unchanged.
- **Cache correctness**: add a short ad-hoc print in `_sample_learned_groupings` (or a debug counter) confirming `gt.generate_with_tokens` is called roughly `total_steps / grouping_interval` times per env per rollout. Remove before commit.
- **Persistence metric**: log per-step Jaccard between consecutive timesteps' edge lists during a short training run; expect ≈1.0 on hold-steps and a sharp drop only at decision boundaries.
- **Loss sanity**: in a short MAPPO run, confirm `compute_grouping_loss` returns a nonzero value and that the number of valid samples is ≈ `1/grouping_interval` of the previous baseline (since hold-steps are skipped).
- **Learning regression**: run one short box2d_suite training (~50k steps) with `grouping_interval=4` vs. master; verify final return is not worse and that GT wall-clock per rollout drops by ~4x.

## Critical files to modify

- [algorithms/mappo/types.py](algorithms/mappo/types.py) — new `grouping_interval` config knob.
- [algorithms/mappo/trainer_components/hypergraph_runtime.py](algorithms/mappo/trainer_components/hypergraph_runtime.py) — interval gate + per-env cache in `_sample_learned_groupings`.
- [algorithms/mappo/mappo.py](algorithms/mappo/mappo.py) — `compute_grouping_loss`: decision-step filter + windowed-mean advantage; plumb `grouping_interval` through `__init__`.

## Files inspected (no change required)

- [algorithms/mappo/networks/grouping_transformer.py](algorithms/mappo/networks/grouping_transformer.py) — network unchanged.
- [algorithms/mappo/hypergraph.py](algorithms/mappo/hypergraph.py) — predefined constructors (behavioral reference).
- [algorithms/mappo/trainer_components/rollout_collector.py](algorithms/mappo/trainer_components/rollout_collector.py) — confirmed downstream consumers recompute from edge structure each call, so repeated edges work without changes.

## Out of scope (deferred)

- Gradual change *between* decision steps (token-KL regularizer, previous-structure conditioning, soft Jaccard). Easy to layer on later if persistence alone proves insufficient.
- Adaptive intervals (stability-gate / change-detector). Could replace the fixed `grouping_interval` in a follow-up.
