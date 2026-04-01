# MAPPO Canonical-Core Refactor (Hypergraph/Entropy Extraction)

## Summary
- Keep `MAPPOAgent` public constructor and public method signatures unchanged.
- Keep canonical MAPPO flow in `mappo.py`: buffer lifecycle, GAE, PPO objective, optimizer/update orchestration.
- Extract hypergraph/entropy-specific runtime and data handling into dedicated internal helper modules, with behavior preserved exactly.

## Implementation Changes

### 1. Hypergraph helper module (`algorithms/mappo/hg_cache.py`)
- **Owns** all hypergraph caches currently embedded in `MAPPOAgent`:
  - `hg_signature_to_id` — canonical edge-list signatures → integer IDs
  - `hg_unique_edge_lists` — representative edge lists per signature ID
  - `hg_object_cache` — pre-built `dhg.Hypergraph` objects keyed by type and signature tuple
  - `hg_entropy_cache` — pre-computed normalized soft entropy per signature ID
  - `hg_signature_ids` — per-(env, timestep) signature ID buffer
  - `hg_entropies` — per-(env, timestep) entropy value buffer
- Owns flatten helpers: `get_flat_signature_ids()` and `get_flat_entropies()`.
- Owns `get_or_compute_entropy()` (currently `MAPPOAgent.get_or_compute_entropy`).
- Owns HGNN minibatch critic value path (`compute_hgnn_critic_values()`), including batch-hypergraph cache semantics and keying. **Receives the network as a method argument** — does not hold a reference.
- Owns `reset()` method that clears all caches and buffers (called from `MAPPOAgent.reset_buffers()`).
- Keep dtype/device behavior unchanged (`np.float64` cached entropy, `float32` tensors on target device).

**Relationship to existing `HypergraphRuntime`**: `HypergraphRuntime` (trainer_components) handles *collection-time* hypergraph construction and entropy computation. It currently mutates agent caches directly (`agent.hg_signature_to_id`, etc.). After refactor, `HypergraphRuntime` will access these through `agent.hg_cache.<attr>` instead. `HypergraphRuntime` remains a trainer-side orchestrator; `hg_cache` is the agent-side data owner.

### 2. Entropy predictor helper module (`algorithms/mappo/entropy_helpers.py`)
- Owns rolling observation history (`_obs_history`) and its reset/update logic used during collection/eval.
- Owns sequence-building utilities for predictor training batches (`build_obs_sequences()`, `build_obs_sequences_batched()`).
- Owns actor-observation conditioning helper: given predictor output (mean, log_var), concatenates to obs.
- Owns predictor loss computation helper (Gaussian NLL over entropy targets).
- Preserves current branch behavior (`n_envs == n_parallel_envs` uses persistent history; eval/render uses zero-padded single-step history).
- **Does not own the predictor network itself** — that stays in `MAPPONetwork`. Helpers receive the predictor as an argument.

**Cut point for entropy data flow**: The hypergraph helper *produces* entropy values (`get_or_compute_entropy`, `get_flat_entropies`). The entropy helper *consumes* them as prediction targets during training. The boundary is the flat entropy tensor: hypergraph helper outputs it, entropy helper uses it for loss computation.

### 3. Slim `mappo.py` to canonical orchestration
- Replace embedded hypergraph/entropy blocks with helper calls.
- Keep PPO math and returns/advantages logic in `mappo.py`.
- Keep `update_shared` and `update_independent_actors` structure intact (targeted extraction, no aggressive training-loop unification).
- **Deduplication via helpers**: both update methods call the same helper interface (e.g., `self.hg_cache.compute_hgnn_critic_values(...)`, `self.entropy_helpers.build_obs_sequences_batched(...)`). The interleaved per-minibatch structure stays in each update method, but the heavy logic lives in one place.
- Remove dead internal helper duplication where safe, without changing outputs.

### 4. Align trainer-side integration with new ownership boundaries
- Update `HypergraphRuntime` to access caches through `agent.hg_cache` instead of mutating raw agent attributes directly.
- Preserve rollout/eval/render call flow and tensor shapes.

## File Layout
All new modules under `algorithms/mappo/` (agent-internal helpers, not trainer orchestration):
- `algorithms/mappo/hg_cache.py` — hypergraph cache + critic helper
- `algorithms/mappo/entropy_helpers.py` — entropy predictor utilities

Existing files modified:
- `algorithms/mappo/mappo.py` — delegates to helpers, thins out
- `algorithms/mappo/trainer_components/hypergraph_runtime.py` — accesses `agent.hg_cache` instead of raw attributes

## Public API / Interface Impact
- Public API change: none (constructor + public methods remain the same).
- Internal additions: helper classes instantiated in `MAPPOAgent.__init__`, thin delegation/accessor methods.
- Compatibility: existing `VecMAPPOTrainer` usage remains valid.
- `agent.hg_cache` exposed as an attribute for `HypergraphRuntime` access (replaces direct cache mutation).

## Test Plan
- Static safety:
  - `python -m py_compile` on all modified MAPPO modules.
- Behavioral parity (fixed seed, before vs after, ≥5 update steps):
  - `critic_type="mlp"` (no hypergraph path).
  - `critic_type="multi_hgnn", entropy_conditioning=False`.
  - `critic_type="multi_hgnn", entropy_conditioning=True`.
- Parity assertions (**bit-exact match** on fixed seed):
  - Same output values for actions, log_probs, values (not just shapes/dtypes).
  - Same buffer population behavior for `hg_signature_ids` and entropy targets.
  - Same update stats keys and finite values (`total_loss`, `policy_loss`, `value_loss`, `entropy_loss`, `entropy_pred_loss`).
  - No change to checkpoint load/save compatibility.
- Regression: verify `HypergraphRuntime` works through `agent.hg_cache` accessor.

## Assumptions and Defaults
- Selected depth: targeted extraction only.
- Selected compatibility: keep `MAPPOAgent` public signatures unchanged.
- No intended algorithmic or numerical behavior changes.
- Default plan file path for implementation turn: `plans/mappo_canonical_refactor_plan.md`.
