# Refactor Plan: Make `VecMAPPOTrainer` Readable and Reusable

## Summary
- Refactor with a VecTrainer-first scope: keep runtime behavior unchanged while decomposing `VecMAPPOTrainer` into focused components.
- Make `vec_trainer.py` an orchestration facade (target: ~250–350 lines).
- Delete and migrate away from legacy `MAPPOTrainer` (`algorithms/mappo/trainer.py`).
- Keep external entrypoint behavior stable (`MAPPO_Runner` still constructs `VecMAPPOTrainer` and calls the same methods).

## Implementation Changes
- Introduce `algorithms/mappo/trainer_components/` with focused modules:
  - `hypergraph_runtime.py`
    - Own hypergraph mode config/state (`predefined` vs `hygma`).
    - Build batched inference hypergraphs.
    - Compute per-env entropy tensors for critic conditioning.
    - Manage dynamic grouping hooks: `on_rollout_reset()`, `on_rollout_step(obs)`, `on_env_done_mask(dones)`.
    - Public class: `HypergraphRuntime`.
  - `rollout_collector.py`
    - Own rollout collection loop currently in `collect_trajectory()`.
    - Handle mask flow, env stepping, done handling, transition storage, and final bootstrap values.
    - Accept pluggable hypergraph runtime hooks.
    - Public class: `RolloutCollector`; return dataclass `RolloutResult(step_count, episode_count, final_values)`.
  - `evaluator.py`
    - Own evaluation loop currently in `evaluate()`.
    - No training stats mutation; pure reward aggregation.
    - Uses `HypergraphRuntime` for inference graph/entropy inputs.
    - Public class: `PolicyEvaluator`.
  - `renderer.py`
    - Own render loop and entropy logging currently in `render()` and `_make_render_env()`.
    - Keep entropy key compatibility behavior exactly as now (`proximity`/`soft_proximity` in hygma mode).
    - Public class: `PolicyRenderer`.
  - `checkpoint_io.py`
    - Own agent save/load and training-stats load/save logic.
    - Keep checkpoint schema unchanged in this refactor.
    - Public class: `CheckpointIO`.
  - `stats_tracker.py`
    - Own in-memory training/timing accumulation and summary formatting.
    - Public class: `TrainingStatsTracker`.

- Rewrite `algorithms/mappo/vec_trainer.py` as coordinator:
  - Constructor creates envs/agent once, then instantiates component objects.
  - Public methods remain unchanged:
    - `collect_trajectory(max_steps)`
    - `train(...)`
    - `evaluate()`
    - `render(capture_video=False)`
    - `save_agent/load_agent/load_checkpoint_progress/save_training_stats`
    - `close_environments/__del__`
  - Each method delegates to one component and only orchestrates.
  - Remove dead imports (for example `build_hypergraph` in `vec_trainer.py`).

- Delete `algorithms/mappo/trainer.py` and migrate:
  - Remove file.
  - Confirm no internal imports remain.
  - Add a short note in MAPPO docs/readme (or `vec_trainer.py` module docstring) that vector trainer is now the sole supported trainer path.

- Small supporting cleanup in `mappo`:
  - Move seed helper from `run.py` to `algorithms/mappo/utils.py` as `set_global_seeds(seed)` and import from there.
  - Keep `run.py` plotting logic unchanged in this refactor.

## Migration Sequence
1. Add `trainer_components` modules and dataclasses with unit-testable boundaries.
2. Port hypergraph helpers from `vec_trainer.py` into `HypergraphRuntime`.
3. Port rollout loop into `RolloutCollector` and wire into `VecMAPPOTrainer.collect_trajectory`.
4. Port `evaluate` and `render` into `PolicyEvaluator` and `PolicyRenderer`.
5. Port checkpoint and stats methods into `CheckpointIO` and `TrainingStatsTracker`.
6. Slim `vec_trainer.py` to orchestration only; preserve method signatures and return types.
7. Delete `trainer.py`; run repo-wide import search to ensure no dangling references.
8. Run validation suite and smoke runs.

## Test Plan
- Static checks:
  - `python3 -m py_compile` over modified MAPPO modules.
  - Import smoke for `MAPPO_Runner` and `VecMAPPOTrainer` construction.
- Behavioral parity smoke tests:
  - One short training run in `hypergraph_mode="predefined"` and one in `"hygma"`.
  - Verify step collection/update/eval/checkpoint cycle completes.
  - Verify render returns entropy logs with expected keys and shapes in both modes.
- Regression checks:
  - Checkpoint save/load round-trip works with RNG restoration.
  - `run.py view` still runs and plots without key errors.
  - Confirm no references to deleted `algorithms/mappo/trainer.py`.

## Assumptions
- This is a structure-first refactor with no intended algorithmic behavior changes.
- Public `VecMAPPOTrainer` API and `MAPPO_Runner` call flow remain stable.
- Checkpoint format remains unchanged in this pass.
- Legacy non-vector trainer is intentionally removed.
