# When user gives instructions, push back if you think the user is wrong. Do not accept everything the user says as source truth. Use your best judgement but share your reasoning with the user and provide both options. Always go with what the user chooses after this.

Whenever building new code, try to reuse as much code as possible. If the new functionality overlaps heavily with other parts of the code, find a way to abstract and reuse the logic instead of duplicating the functionality.

Always keep the CLAUDE.md file up to date to reflect the current functionality and architecture of the code.

Avoid abbreviating terms, and if you are going to use an abbreviation, explain it don't assume I know what an abbreviated term is.

When analyzing results, explain to me the quantities that are being logged, their meaning as well as why use them.

## Experiment config: Hydra is the sole path (`conf/` + `train.py`)

Runs are launched **only** through the Hydra entry point `train.py`. The legacy
yaml loader (`run_algorithm` in `algorithms/algorithms.py`) and its argparse CLI
(`run_trial.py`) have been **retired** — the `experiments/yamls/<batch>/` files
remain on disk as source material to migrate into `conf/`, but nothing loads them
at runtime any more.

`train.py` composes a run from orthogonal groups (**algorithm × env × model ×
seeds**), resolves it, and hands the result to the shared dispatch tail
`_dispatch(algorithm, exp_dict, env_config, batch_dir, results_dir, trial_id,
...)` in `algorithms/algorithms.py`. `_dispatch` builds the per-algo
`Experiment(**exp_dict)` → constructs the Runner → `train()`/`view()`/
`evaluate()`. `batch_dir` (`experiments/yamls/<batch>`) is only used by runners
for `combined_affinities` checkpoint resolution (`batch_dir.parents[1]/results`);
`results_dir` is the runner's `trials_dir` (`results/<batch>/<name>`).

- `conf/config.yaml` — defaults list (`algorithm: mappo`, `env: ...`, `model:
  ...`, `seeds: standard`, `_self_`) + top-level `device`/`trial_id`/`view`/
  `checkpoint`/`evaluate`. Group order sets precedence (later wins): algorithm
  supplies base `params`/`model_params`; env overrides env-scoped `params`
  (`val_coef`, `n_total_steps`) and publishes a `hyperedges` map; model overrides
  `model_params`; seeds injects `params.random_seeds`. `hydra.job.chdir=false` +
  `output_subdir=null` + null log handlers keep cwd/paths/results untouched.
- `conf/algorithm/{mappo,...}.yaml`, `conf/env/<batch>.yaml`,
  `conf/model/<variant>.yaml`, `conf/seeds/{standard,...}.yaml`. **Env/model
  filenames equal the old batch/variant names** so `results/<batch>/<name>/
  <trial_id>/` and existing checkpoints resolve. Model files hold only the
  `model_params` delta (env-specific `hyperedge_fn_names` interpolate the env's
  map, e.g. `${hyperedges.mix}`), mirroring the legacy variant's keys exactly.
- `train.py` — `@hydra.main`; `OmegaConf.to_container(resolve=True)` →
  `_build_dispatch_args(cfg, choices)` → `_dispatch`. `choices` (env/model) come
  from `HydraConfig.get().runtime.choices` and preserve the output layout.
  Run: `uv run python train.py env=multi_box_push_9a_3o model=hgnn_mix trial_id=0`;
  sweep: `uv run python train.py -m model=mlp_shared,gnn_critic trial_id=0,1,2`
  (add `hydra/launcher=joblib_auto` for local parallelism).
- **Hardware must not change the optimization trajectory.** Governing invariant:
  the machine decides *how fast* data is gathered, never *what* the optimizer
  sees. Two rules enforce it.
  1. **The batch is config, not hardware.** For the torch stacks
     (`mappo`, `mappo_vanilla`) the per-update batch is an explicit
     `params.batch_size` in **total env-steps** (`conf/algorithm/{mappo,
     mappo_vanilla}.yaml`, default `32768`), read straight through by
     `run.py` → `VecMAPPOTrainer.train`. It used to be derived as
     `n_steps * env.n_envs`, which made the **core count** set the batch and
     hence `num_updates` — the same nominal run optimized differently on a
     4-core and a 32-core node. `RolloutCollector.collect` already loops on a
     *total* step count (`while total_step_count <= max_steps`), so it gathers
     `batch_size` steps however many envs run in parallel; `n_envs` is now purely
     a speed knob there. (`params.n_steps` no longer exists for these two
     stacks.) `mappo_jax` keeps `params.n_steps` — it is the static length of the
     jitted collect scan and must stay per-env — but its `n_envs` is a literal,
     so its batch is likewise fixed by config alone.
  2. **`n_envs` lives in the env group, never in `_self_`.** `conf/config.yaml`
     deliberately does **not** set `env.n_envs`; being last in the defaults list
     it would override every group. Each `conf/env/*.yaml` declares its own, and
     the right value depends on what `n_envs` *means* for that env:
     - **Subprocess envs** (box2d `multi_box_push`, `hrl_skill`, `smaclite`):
       `n_envs: ${envs_per_job:${n_jobs}}` — a genuine hardware knob (one OS
       process per env), safe to autoscale now that the batch is decoupled.
     - **MJX envs** (`macro_mjx`, `multi_box_push_mjx`): a **literal** `n_envs:
       32` — this is a vmap width on one device, not a core budget, and it *is*
       a hyperparameter (batch = `n_steps * n_envs` → `num_updates`). Keep it
       identical across arms being compared; lower it only for GPU memory, and
       then for every arm at once.
  The collector can only move in whole **rows** of `n_envs` steps (the vector env
  steps every env together; GAE stacks per-env trajectories and needs a uniform
  length, so cuts must land on a row boundary). Its loop condition is therefore
  `while total_step_count < max_steps` — with `<=` it took one extra full row
  even when `max_steps` was hit exactly, making the batch a function of `n_envs`
  (32800 at `n_envs=32` vs 32776 at 8). With `<`, a batch that `n_envs` divides
  is collected **exactly**: verified identical `[1024, 2048, 3072]` step grids at
  `n_envs` ∈ {2, 4, 8, 32}. This matters for analysis, not optimization —
  `plotting/plot_training_stats.ipynb` averages seeds with
  `groupby(["plot_group", "total_steps"])`, an exact-value match, so trials whose
  grids differ stop aggregating (`n_runs` → 1, SEM band becomes one run).
  Residuals (accepted, not fixed): (1) when `n_envs` does **not** divide
  `batch_size` a rollout still overshoots by up to `n_envs - 1` — measured
  `[1032, 2064, 3072]` at `n_envs=12`, i.e. intermediate x points shift though
  the final total self-corrects (`steps_to_collect = min(batch_size, remaining)`
  shrinks the last request). Prefer power-of-2 `n_envs`. (2) equal-batch runs at
  different `n_envs` are not *bit*-identical — advantage normalization is
  per-env-stream, so 32×128 and 8×512 partition the same transitions
  differently. Hyperparameters and batch size are invariant; the exact gradient
  sequence is not.
- **Parallelism autoscaling (two nested layers).** Layer 1 is the sweep: with
  the joblib launcher each cross-product job runs in its own loky worker. Layer 2
  is per-job rollout collection: `make_vec_env` forks `env.n_envs` box2d
  subprocesses. The two multiply, so the budget is `n_jobs × n_envs ≲ cores`.
  The top-level knob `n_jobs` (in `conf/config.yaml`, default `1`) drives both:
  subprocess env groups set `n_envs: ${envs_per_job:${n_jobs}}` →
  `usable_cores // n_jobs`, and the `hydra/launcher=joblib_auto` group
  (`conf/hydra/launcher/joblib_auto.yaml`, wraps the plugin's `joblib` and sets
  `n_jobs: ${n_jobs}`) makes joblib run that many at once. Resolvers `cores` /
  `envs_per_job` are registered at `train.py` import; `_usable_cores()` reads the
  CPU-affinity mask (`os.sched_getaffinity`) so it respects `taskset` / cgroup /
  SLURM quotas. So `n_jobs=1` (default) → one run using all cores for envs;
  `-m ... n_jobs=4 hydra/launcher=joblib_auto` → 4 concurrent jobs × `cores//4`
  envs each. Override `env.n_envs=<N>` on the CLI to opt out of autoscaling.
  **Fork context is pinned** in `make_vec_env` (`context="fork"` for non-HRL,
  `"forkserver"` for HRL) rather than the ambient default: inside a loky worker
  the default start method is `"loky"` (spawn-like), which forces
  `AsyncVectorEnv` to pickle its `shared_memory` buffers and crashes with
  `cannot pickle 'mmap.mmap'`. `Runner.__init__` floors torch threads at
  `max(1, get_num_threads()//2)` since a loky worker can start with 1 thread.
- **⚠ Nothing validates the `env:` block, so a misindented key is SILENT.**
  `mappo_jax`/`feudal_mappo_jax` `run.py` pull named keys out of `env:` and warn
  on nothing (an earlier note here claimed a `_base_env_kwargs` helper warned on
  unrecognized keys — **it does not exist**). Live example, fixed 2026-08-28:
  `conf/env/mjx_16a_4o.yaml` carried `n_steps: 512` indented **under `env:`**
  (commit `1fbb3f3`), i.e. `env.params.n_steps`, which nothing reads — so that
  group ran at the algorithm default of **1024** while `_trunc` / `_partition`
  ran at 512, i.e. twice their per-update batch. Diagnose it from the stats:
  `total_steps[1] - total_steps[0]` is `n_steps * n_envs` (32768 vs 16384 here).
  `params:` belongs at **column 0** in an env group (`# @package _global_`).
  Any `mjx_16a_4o` result predating the fix is not batch-comparable to those
  arms; within-group `mlp`-vs-`feudal` is unaffected (both inherited the same
  inert override).
- **Migration status:** `multi_box_push_9a_3o` (MAPPO) and `dcg_smaclite_2s3z`
  (DCG) are currently ported into `conf/`. Other batches under
  `experiments/yamls/` must be migrated to `conf/env` + `conf/model` before they
  can run. To add a batch: create `conf/env/<batch>.yaml` (from `_batch.yaml`'s
  `env:` block + `params` overrides + `hyperedges` map) and one
  `conf/model/<variant>.yaml` per experiment yaml (the `model_params` delta); the
  env/model filenames must equal the old batch/variant names to preserve the
  `results/<batch>/<name>` layout. The DCG port also added
  `conf/algorithm/dcg.yaml` (the `params` block); DCG's env group must expose
  `environment`/`n_agents`/`env_variant` under `env:` (the trainer reads
  `env_params.get("environment")`, not `name`), and the default `seeds: standard`
  list already matches the old DCG seed list for `trial_id` indexing. Non-box2d
  batches (smac, dcg, ippo/jax) may also need `vec_trainer`
  `self.env_name`/`self.env_variant` wiring. The convenience wrappers `train.sh` /
  `scripts/evaluate.sh` translate `(BATCH, ALGORITHM, ENVIRONMENT, TRIAL_ID,
  EXP_NAME)` positional args into Hydra overrides (`env=$BATCH model=$EXP_NAME
  algorithm=$ALGORITHM ...`); `$ENVIRONMENT` is vestigial. The `scripts/hpc/*`
  launchers still reference the removed `run_trial.py` and must be updated to
  `train.py` before use.

## Plotting (`plotting/plot_training_stats.ipynb` + `plotting/config.yaml`)

`config.yaml` selects runs as `batches` × `experiments` × `trials` under
`base_path`. Cell 0 defines `load_reward_runs(batches)` and
`summarize_rewards(df, group_cols)`, which give the mean ± standard error of the
mean over trials at each exact `total_steps`. Both reward figures use them.
- **Single reward plot:** every batch in `batches` goes on one axis.
- **Multi-batch reward grid:** one subplot per batch in `multi_batches`. The
  grid has up to 3 columns, and each subplot shows the same `experiments` and
  `trials`. The figure has one shared legend. Each experiment keeps one color
  across subplots, assigned by its position in `experiments`. `plot_colors` can
  override that color by experiment name, but a `batch/experiment` key has no
  effect here. Y axes are not shared, because reward scales differ between
  batches. The cell prints a table of the trials found for each batch and
  experiment. If `multi_batches` is absent, the cell skips the plot.

## Single-agent FeUdal runs

`mjx_1a_3o_111_1024` is a valid sequential-delivery task: its one agent can
push each box with `coupling_def: [1, 1, 1]`. The coupling sum warning is
informational. FeUdal training skips the agent-permuted eval block when
`n_agents == 1`, retaining the real, constant and zeroed blocks.
`manager.training_goal_variants` supplies the same selection to the trainer and
runner. The offline `goal_dependence_probe` uses `manager.offline_goal_variants`,
which drops only `permuted` at `n_agents == 1` and keeps the offline-only
`env_permuted` (until 2026-09-24 the probe hardcoded all five and raised on
every single-agent checkpoint). Agent-permutation cosine metrics (`d_cos_null_agent`,
`d_cos_gap_agent`, `goal_perm_cos`) are NaN because there is no other agent to
swap with. Environment-permutation metrics remain available when `n_envs > 1`;
they likewise become NaN for a singleton env axis. Direct requests for an
identity permutation still raise, including invalid shifts on larger axes.
The worker and intrinsic critics retain their trailing agent axis for one
agent (`MAPPOCritic.keep_output_axis`), as does the manager critic under
per-agent rewards. Otherwise the singleton output was squeezed, causing
truncation bootstraps to broadcast `(n_envs, 1)` rewards to `(n_envs, n_envs)`
and fail at the first advantage calculation. Team manager values stay scalar.
At `n_agents == 1`, `manager_latent: centralized` and `local_private` are the
**same network** unless the env publishes `global_state`. Without that hook,
`global_state` is `obs.reshape(E, -1)`, which is agent 0's observation.
`f_percept` then matches `f_enc`, `f_Mspace` matches `f_Mspace_agent`, and
`goal_head` matches `goal_head_agent` in shape and init distribution. The
parameter counts match. With transplanted weights, `goal` and `s` differ by
exactly 0.0 on both cores (CPU check, 2026-09-23). Only the parameter names
differ, so initial draws differ and checkpoints do not load across them. On
`mjx_1a_3o_111_1024`, an arm pair differs only by seed. On the `_gs` twin,
the difference is informational: `centralized` reads the 22-dimensional compact
state, while `local_private` reads the 40-dimensional egocentric observation.
`manager_latent: local` is the same except in the goal path. Its `s` is exactly
`local_private`'s after transplanting weights, with a 0.0 difference. Its goal
path adds one 32-wide tanh layer, `f_gpre -> tanh -> f_goalhead`, with 1,056
more parameters. `worker_encoder: shared` is the one `*_local_shared` change
that is real at any N: the worker reads the 256-wide `f_enc(obs)`, not raw obs,
and its PPO gradient trains `f_enc`.
`debug=true` disables JAX compilation; use `debug=false` for training speed.

## Box2D suite observations

All `environments/box2d_suite` envs share `ObservationManager.get_observation`
(in `observation.py`). The per-agent observation vector is, in order:

- `own_velocity` (2) — linear velocity normalized by `velocity_norm`
- `density_sensors` (16) — 8-sector centroid distance to agents (0-7) and objects (8-15)
- `is_touching_object` (1)
- `neighbor_fraction` (1) — fraction of agents within `neighbor_detection_range` (incl. self)
- `contact_force` (1) — per-agent contact force / `force_multiplier`
- `nearest_box_vec` (2) — relative (dx, dy) to the nearest **sensed** object,
  per axis normalized by `world_width`. An object is sensed when it is
  undelivered **and within `sector_sensor_radius`** — the same strict `<` range
  cap the density sensors apply, so the two object channels switch on at exactly
  the same distance. Zero vector when the env has no objects, and **per-agent**
  zero for any agent with no object in range (which subsumes the
  every-object-delivered case). Already-delivered objects
  (`env.delivered_objects` in Box2D, the `delivered` mask in MJX) are excluded
  from the search, so an agent stops being drawn to a box parked in the goal
  band. Egocentric (no absolute world anchor).
  - **⚠ The range cap was added 2026-08-31 and it changes every env's
    observation semantics — results before that date are not comparable.**
    Until then this was an *unrestricted global argmin*: the ONLY unlimited-range
    channel in an otherwise entirely local vector (density sectors, lidar and
    `neighbor_fraction` are all capped). Measured at `mjx_16a_4o` (40 seeds,
    spawn): mean agent-to-box distance **22.1** against R = **15.67** in a
    47-wide arena, so an average agent locally perceives only **21.9%** of the
    boxes and **47.8%** perceive none at all — yet every one was handed an exact
    bearing and range. That single feature collapsed the task's partial
    observability and made "approach your nearest undelivered box" a *fully
    local* near-optimal policy, which is why a shared-parameter flat policy had
    no role-assignment problem left to solve. Capped, one agent knows ~22% of
    the boxes while the joint state (what a centralized manager reads) contains
    **95.6%** — a 73.7-point gap that did not previously exist.
  - The cap is **not** configurable and there is no opt-out flag: it is applied
    directly in `MJXObservationBuilder._nearest_box` and, in parity, in the
    Box2D `ObservationManager._calculate_nearest_box_vectors`. **Both engines
    must apply it or their observations diverge.**
  - Anything keyed to the sensed box must gate on `nearest_box_sensed` — the
    companion mask — because `nearest_box_indices` is an argmin over all-`+inf`
    when nothing is in range and its index is then meaningless. Live consumers:
    `MJXObservationBuilder.build` zeroes `goal_distance` **only when
    `goal_from_pos` is not None** (i.e. the per-box goal-ring envs, where the
    feature is measured from the sensed box); it must NOT be gated when the goal
    is the agent's own distance to a single shared band, which is always well
    defined. `MultiBoxMultiGoalPushMJX._coupling_fractions` takes the mask too.
    Verified: `nearest_box_vec`, `goal_distance` and `coupling_fraction` gate
    together on all 320 checked (seed, agent) rows of the multi-goal env.
  - Verified surgical: over 20 seeds at 16a/4o exactly columns **[21, 22]** of
    the 40-dim vector differ from the pre-change observation, and for every one
    of the 165 agents that *does* have a box in range the vector is bit-identical
    to the old global argmin (nearest-in-range == nearest-globally whenever the
    global nearest is in range). Max nonzero magnitude is now bounded by
    `R / world_width` (measured 0.3331 against the 0.3333 cap).
- `goal_distance` (1) — signed relative distance from the agent to the target
  region center, measured along the env's **goal axis**: the y axis by default
  (normalized by `world_height`), or the x axis (normalized by `world_width`)
  when the env sets `goal_axis == "x"` (read via `getattr`, default `"y"`). 0
  when the env has no `target_areas`. Egocentric goal-grounding for the
  box-push/grab tasks; `push_box` uses the x axis when its goal band is on the
  left/right wall.
- `lidar` (`N_LIDAR_RAYS`, default 16) — nearest-obstacle distance along evenly
  spaced world-frame rays via Box2D raycast; normalized to [0, 1], 1.0 == clear

Note: absolute `own_pos` is intentionally **not** in the vector — the
observation is egocentric. `nearest_box_vec` + `goal_distance` restore goal
grounding (where to push, and how far) without reintroducing an absolute
world-frame anchor.

⚠ **Consequence of the egocentric design for any centralized reader** (the
feudal manager, whose input is literally `obs.reshape(n_envs, -1)`): it receives
N egocentric views with **no shared frame**, so exploiting the union of partial
views requires it to *learn to localize agents first*. The anchors available are
`goal_distance` (an exact coordinate on the goal axis) and the lidar wall
returns (in a 47-wide arena with R = 15.67, ~89% of positions are within range
of at least one wall, giving a partial cross-axis fix). Sufficient in principle,
but a real representation-learning burden — and in `feudal_mappo_jax` the whole
manager bottleneck is `goal_dim` (16 by default), i.e. 16 numbers to encode
where every agent and box is.
  - **⚠ MEASURED 2026-09-04, and it largely REFUTES the "burden" framing above.**
    A decoding probe (`algorithms/feudal_mappo_jax/global_state_probe.py`, 8192
    on-policy states from the trained `mjx_16a_4o_trunc_512/mlp/0` policy) fits
    agent world coordinates from the 640-dim concat to a held-out mean error of
    **3.36 world units in a 47-wide arena** — against 17.95 for a
    predict-the-mean baseline and 0.01 for the compact global state (exact by
    construction). That is **81% of the gap closed, by a LINEAR readout**, so the
    joint state is not merely present but essentially unentangled. Per axis it
    splits exactly as the layout predicts: **y is free** (0.01 — `goal_distance`
    is an affine function of the agent's own y) and **x costs 3.43** against
    14.54 knowing nothing. Box positions decode to 2.62. So the manager's input
    is **not information-poor about where things are**, and an `env.global_state`
    hook would buy width and exact features, NOT information — do not justify one
    on the frame argument. The honest residual: a probe is *trained to decode*,
    whereas the manager is trained on a task gradient, so this shows the
    information is available and cheap, not that it is used; and 3.4 units is
    ~22% of the sensor radius, so the recovered resolution is coarse. Raising `goal_dim` also raises the ceiling on the
`goal_direction_count` diagnostic, whose healthy random-direction baseline is
`N^2 / (N + N(N-1)/goal_dim)` — 8.26 at N=16/goal_dim=16 — so that series stops
being comparable across runs with different widths.

### Sensor overlay (debug rendering)

`Renderer._draw_sensor_overlay` (`renderer.py`) draws the observation of **one
focus agent** on top of the world: the 8+8 density sectors (`A:` agents / `O:`
objects), the lidar scan (rays to their hit points, red dot on a hit, faint when
clear), a magenta `nearest_box_vec` arrow, a green `goal_distance` segment along
the env's goal axis, and a HUD legend with the scalar values. The focus agent is
`env.render_sensor_agent` (default 0) — drawing every agent is unreadable past a
handful, and costs a raycast pass per agent per frame.

Values come from `ObservationManager.get_sensor_readout(agent_idx)`, which calls
the **same** `_calculate_*` paths as `get_observation` (verified equal to the
corresponding obs slices), so the overlay cannot drift from what the policy sees.
`get_sensor_readout` calls `_refresh_caches()` itself, so it is safe outside a
`get_observation` step. The lidar scan is factored into a per-agent
`_calculate_lidar` (`_calculate_lidar_all` loops it) so the overlay raycasts only
the focus agent, and `ObservationManager.lidar_directions` is shared by the scan
and the drawing. Envs with no `objects` / no `target_areas` (scatter,
rendezvouz) simply skip the box/goal arrows. The old scalar
`calculate_density_sensors` — a duplicate of the vectorized math, used only by
the renderer — was deleted.

The total dimension is exported as `OBS_DIM` (= `BASE_OBS_DIM + N_LIDAR_RAYS`) from
`observation.py`; every env's `observation_space` must use `OBS_DIM` so the layout
stays in sync. Per-env overrides `n_lidar_rays` / `lidar_range` are read via
`getattr` (defaults: `N_LIDAR_RAYS`, `sector_sensor_radius`).

## Push-box environment (`push_box.py`)

`PushBoxEnv` (`EnvironmentEnum.PUSH_TO_TOP` case, key `"push_box"`) is a
single-box cooperative pushing task built by reusing the `multi_box_push`
machinery (boundary, observation, renderer, contact listener, target band).

- **Variable goal wall.** Each episode `reset` samples one of the four walls
  (`_GOAL_SIDES`: top/bottom/left/right) and sets `self.goal_side`,
  `self.goal_axis` (`"x"`/`"y"`), and `self.goal_sign` (+1 toward the high end
  of that axis). `_create_target_areas` builds the band spanning that wall
  (full inner length, `band`-thick). `__init__` defaults to `"top"` so a valid
  target/observation exists before the first reset.
- **Spawn layout (band → box → agents).** `reset` builds the goal band first,
  then the box, then the agents, so each step can reference the previous. Both
  the box and every agent start at least `self.min_goal_spawn_distance` from the
  goal band along the goal axis (= `_MIN_GOAL_SPAWN_FRACTION` (0.4) × world
  extent; the world is square). The shared line is
  `_goal_axis_spawn_limit()` — the goal-axis coordinate exactly that far from
  the band's inner edge.
  - `_create_dynamic_objects` places the box **at** that line (goal axis) with a
    randomized perpendicular coordinate, independent of the agents — so it never
    starts inside the band (no instant delivery) and is far from the goal.
  - `_scatter_agent_positions` scatters agents on the **far side** of that line
    (away from the goal), spaced `min_sep` apart, rejecting any sample that
    would overlap the box (`_overlaps_box`, a disc-vs-rect test using
    `_AGENT_RADIUS`). Uses the seeded `np_random`; falls back to an even spread
    if rejection sampling fails. Replaces the old `get_scatter_positions` call,
    which ignored the goal side and clustered agents in the bottom third.
- Box size **varies per episode**: square half-extent sampled uniformly in
  `[1.5, 1.8]` (1.5 is the minimum, +20%) via the seeded `np_random`.
- **Coupling mechanic** (shared `utils.update_object_mass_from_contacts`): the
  box's `userData["coupling"]` is `n_agents`. Base density `20.0` keeps it
  nearly immovable until **all** agents are touching it; once the requirement is
  met density drops to `0.05 * coupling`, making it far lighter. Same helper now
  used by `multi_box_push`.
- Reward (`_calculate_goal_push_reward`) is the **per-step displacement of the
  box toward the goal wall** (`(box_coord - prev_box_coord) * goal_sign`, where
  `box_coord` is the box's position on `goal_axis`), plus a one-time `+100`
  completion bonus that terminates the episode when the box enters the band.
  `reward_mode="dense"` keeps the shaping term; `"sparse"` pays only the bonus.
- Wired into `algorithms/create_env.py` `make_vec_env` (reads `reward_mode` from
  `env_params`). Run the manual debugger with
  `SDL_VIDEODRIVER=dummy python -m environments.box2d_suite.push_box`.

## MJX suite

### Shared observations (`environments/mjx_suite/observation.py`)

`MJXObservationBuilder` is the JAX counterpart of the Box2D suite's
`ObservationManager`: it owns the sensor math and the 40-dim `OBS_DIM` base
layout. `include_agent_sector_counts=True` appends eight normalized teammate
counts after lidar, making `builder.obs_dim = 48` with the default rays.
No env currently enables it — `MultiBoxPushMJX` passes
`include_agent_sector_counts=False` (commit `b6bf344`), so its observation is the
40-dim base layout.
Pure and `jit`/`vmap`-able; the env passes plain arrays (agent positions/
velocities, box poses) plus the `mjx.Data` (needed for lidar raycasts and the
efc contact-force decode).

- Construct with the `mjx.Model` + world/normalization constants
  (`world_width/height`, `velocity_norm`, `neighbor_detection_range`,
  `agent_radius`, `force_multiplier`; `sector_sensor_radius` defaults to
  `world_width/3` and `lidar_range` to the sector radius, as in Box2D). Contact
  attribution needs the geom→entity maps from the helper `geom_index_maps(mj_model,
  n_agents, n_objects)` (naming convention `g_agent_{i}` / `g_box_{j}`).
- `build(data, agent_pos, agent_vel, box_pos=, box_yaw=, box_half=,
  goal_coord=, goal_axis=, delivered=)` returns `(A, builder.obs_dim)`; the components
  are also exposed individually (`touch_matrix`, `density_sensors`,
  `neighbor_fractions`, `pairwise_agent_distances`, `nearest_box_vectors`,
  `goal_distances`, `lidar`, `contact_forces`) — `_touch_matrix` (coupling) and
  the renderer reuse them. The optional `delivered` (O,) bool mask (threaded
  from `EnvState.delivered` by `MultiBoxPushMJX._get_obs`) drops delivered boxes
  from `nearest_box_vectors` only — an agent stops being drawn to a box parked
  in the goal band; all delivered → zero vector. The Box2D
  `ObservationManager._calculate_nearest_box_vectors` does the same via
  `env.delivered_objects`, keeping the two engines in parity.
- **Agent-sector counts:** `obs[:, 40:48]` in `MultiBoxPushMJX` contains the
  number of other agents in each of the eight density sectors, divided by
  `AGENT_SECTOR_COUNT_SCALE = 4.0`. The sector assignment and strict `< radius`
  mask are shared with agent proximity; self/zero-distance entries are excluded.
  Counts are independent of total team size and are not clipped (eight
  teammates in one sector gives 2.0). The existing centroid calculation already
  needs these counts, so `_density_sensors_and_counts` computes both in one pass.
  The original 40 fields, including lidar at `24:40`, retain their positions.
- **Generalizes past multi_box_push**, mirroring the Box2D fallbacks: `n_objects=0`
  (scatter/rendezvouz) zeros the object density block, `is_touching_object`,
  `nearest_box_vec` and the contact force; `goal_coord=None` (contact/scatter/
  rendezvouz have no `target_areas`) zeros `goal_distance`; and `goal_axis`
  takes a **traced** axis index (0=x, 1=y) as well as the static `"x"`/`"y"`,
  so push_box's per-episode goal wall stays jit/vmap-able.
- Verified bit-identical to the pre-extraction inline implementation across a
  150-step rollout, all 40 dims. Note when checking such things: MJX rollouts
  are **not reproducible across processes** (`mjx.ray` differs ~3e-4 run to run,
  which chaos amplifies) — compare both implementations on the *same* states in
  one process instead.

### MJX multi-box-push (`environments/mjx_suite/multi_box_push_mjx.py`)

`MultiBoxPushMJX` is a MuJoCo-MJX port of the Box2D `multi_box_push` env with a
functional, fully `jit`/`vmap`-able gymnax-style API: `reset(key) -> (obs,
EnvState)`, `step(state, actions) -> (obs, state, reward, terminated,
truncated, info)`; no auto-reset (caller's job). `EnvState` is a registered
dataclass holding `mjx.Data` + step counter + per-box `prev_box_goal_dist` /
`delivered`.

- **2D by construction.** Bodies own only planar DOFs (agents: slide-x/y;
  boxes: slide-x/y + hinge-yaw), gravity is zero, walls are four inward-facing
  planes — there is no z DOF, so MJX never computes out-of-plane dynamics.
  Options: `integrator="implicitfast"` (implicit joint damping — the same
  semantics as Box2D's `v /= 1 + d*dt`) and the default **pyramidal** friction
  cone (elliptic NaNs out on GPU/f32 when a light coupled box is crushed
  against a wall by many agents).
- **Solver settings and contact cap (changed 2026-09-25)**, defined once in
  `environments/mjx_suite/physics_options.py` and emitted by `physics_xml()`
  into both `MultiBoxPushMJX._build_xml` and `MultiBoxMultiGoalPushMJX._build_xml`:
  - Solver: `iterations="20" ls_iterations="10"`, down from MuJoCo's 100 / 50.
  - Contact cap: `max_contact_points = max(64, 4 * (n_agents + n_objects))`.

  **Why.** A step is ~all constraint solver: observations and lidar take ~0.05 ms
  of a 3.4-5.9 ms step, and collection is ~93% of a `mappo_jax` iteration. Under
  `vmap` two defaults make the solver expensive:
  - The Newton `while_loop` runs to the slowest env in the batch.
  - The line search's early exit becomes a select, so every Newton iteration
    pays all `ls_iterations`.
  - Separately, MJX gave the solver one contact slot per candidate geom pair: 210
    slots (840 rows) at 12a/3o, against ~9 contacts touching on average.

  **Fidelity, measured.** Identical states are stepped once under each setting,
  since rollouts are chaotic.
  - Relative error of the velocity update: median ~1e-6, p99 <= 2e-2, including
    all 16 agents crowding one box.
  - The contact-force obs column matches (median difference 0).
  - Trained `mlp` policies score the same, paired over 128 episodes:
    `mjx_12a_3o_trunc_1024` +2.6 ± 6.8, `mjx_16a_4o_partition_1024` −5.1 ± 4.0.
  - `--check-drift` reproduces its recorded numbers.

  **Speed.** Collection with a trained policy (1048 steps × 32 envs) went from
  7.64 to 2.72 s at 12a/3o and from 11.52 to 2.79 s at 16a/4o. Throughput now
  scales with `n_envs` (16a/4o: 21k -> 97k steps/s from 32 -> 256 envs), where
  the old settings saturated the GPU at ~8.5k. `n_envs` is still a
  hyperparameter, per the rules at the top.

  **Pre-change throughput numbers in this file were measured under 100 / 50 with
  no cap.** Results before this date ran under the old settings. The difference
  is below the existing cross-process nondeterminism, so they stay comparable.
  - ⚠ **Do not go below ~8 iterations.** At 4 iterations the velocity update is
    off by 50-200%, while scripted deliveries still succeed, so a
    return-based check will not catch it. 1 iteration NaNs. The CG solver is 5x
    slower.
  - ⚠ **The cap is exact only while fewer contacts touch (`contact.dist < 0`)
    than the cap.** MJX keeps the most-penetrating contacts, so a binding cap
    silently drops real ones. Measured maximum touching, scripted
    partition/swarm, 1024 steps × 32 envs, both arenas: 26/27/36 at
    9a3o/12a3o/16a4o, against caps of 64/64/80. Re-check before a much more
    crowded config: `touching == cap` means the cap bound.
- **Observation width:** **40** per agent — the agent-sector counts were
  turned OFF again in commit `b6bf344` (2026-09-21), so the concatenated-obs
  critic input is `40 * n_agents` (verified: `global_state_dim` = 40 at 1a/3o).
  While they were on (the stretch before `b6bf344`) it was 48 per agent, and
  checkpoints from that window are 48-input — incompatible with the current
  40-input networks in both directions. Consumers use `env.observation_dim`,
  not the shared 40-dim `OBS_DIM` constant, so they follow either setting.
  Checks (the builder option itself, still tested): `JAX_PLATFORMS=cpu .venv/bin/python -m pytest
  algorithms/tests/test_mjx_sector_counts.py -q` covers multiplicity, range,
  sector alignment, fixed scaling, preserved fields, and JIT/vmap/wrapper shapes.
- **Parity with Box2D.** Same world sizing, spawn regions, coupling list, box
  sizing, target band, reward (shaping + one-time +100/box, dense/sparse),
  boundary-contact termination, and the original 40-dim observation prefix
  (built by the shared `MJXObservationBuilder` above). Previously verified by
  posing both engines identically and diffing all 40 dims: everything equal to
  f32 precision, lidar within 4e-4. Box2D damping/mass constants are emulated
  with joint damping = coeff × mass (× inertia for the hinge).
- **Coupling mechanic** is a per-step override of `body_mass` / `body_inertia`
  / `dof_damping` on the `mjx.Model` pytree (`_model_for`) — jit-safe because
  the model is an argument to `mjx.step`. Touch detection is the same
  rotated-box surface-distance test as Box2D's. `_model_for(data, active=None)`
  takes an optional traced (A,) mask of *cooperating* agents; masked agents are
  dropped from the touch count (used by the difference-reward counterfactual).
- **`reward_mode="difference_rewards"`** makes `step` return a **(n_agents,)**
  per-agent reward instead of the team scalar: the exact single-step difference
  reward `D_i = G - G_-i`, from forking the pre-step state once per agent
  (`_difference_rewards`, vmapped) and re-running the same step with agent i
  contributing nothing — zero force *and* dropped from the coupling count.
  `info["task_reward"]` still carries the team scalar in **every** mode, so
  logging/eval stay comparable. The team reward is shaped exactly as `"dense"`
  (a sparse base would leave D zero except on delivery steps). `step` is
  factored into `_advance` (physics) + `_task_reward` (pure in
  `(state, data)`), so counterfactual branches reuse both with no recursion.
  Costs A extra `mjx.step` calls, but they vmap onto spare GPU: measured **1.17x**
  wall-clock (332 -> 284 FPS at 9a/3o, n_envs=8), not the ~9x the step count
  suggests.
  - **Known property, important before using it:** single-step D is *additive
    force attribution*, not coalition credit — `sum_i D_i / G ~ 1.1`. Box mass
    affects acceleration, not instantaneous velocity, so a heavy box still
    coasts and one step cannot reveal the coupling mechanic. The coalition
    structure (each of the 3 required agents individually necessary, so
    `sum_i D_i / G -> ~3.2`) only appears at counterfactual windows of **n >= 30**
    steps (measured: n=1 -> 1.12, n=15 -> 2.41, n=30 -> 3.26, n=60 -> 3.22,
    saturating at the coupling number). A windowed counterfactual costs the
    *same* compute when amortized (A*K extra steps per K steps == A per step)
    but must roll forward with the **policy**, so it belongs in the trainer, not
    the env. Single-step D is still a legitimate learnability signal (agent i's
    gradient stops being polluted by teammates' noise).
- **Sensors in JAX**: density sectors / neighbor fraction / nearest-box /
  goal distance are direct jnp ports; lidar is one vmapped `mjx.ray` call with
  ray origins offset just past the caster's own surface (`bodyexclude` is
  static numpy inside mjx, so self-exclusion can't be traced); per-agent
  contact normal force = sum of the 4 pyramidal facet rows at each contact's
  `efc_address` (verified ≈100 N steady-state for a 100 N push). A dummy
  `<material>` asset works around an mjx.ray crash on material-less models.
- Spawns use shuffled jittered grids (same regions/min separations as the
  Box2D rejection sampling — jit needs static shapes). Reward shaping is live
  from step 1 (Box2D pays 0 on its first step); box sizes fixed per instance
  (as in Box2D). `info` carries `adjacency`, `agents_2_objects` as a dense
  (O, A) 0/1 matrix, positions, `delivered`.
- **Renderer** (`environments/mjx_suite/renderer.py`): the env stays pure JAX;
  `MJXRenderer(env)` is a host-side subclass of the Box2D suite `Renderer`
  that consumes an `EnvState` — it inherits the walls / target-band / sensor-
  overlay drawing (the env exposes a real `ObjectTargetArea`; a
  `SimpleNamespace` shim supplies `observation_manager.lidar_directions`) and
  reimplements only the body drawing from a numpy snapshot. `render(state,
  obs=obs, focus_agent=i)` returns an (H, W, 3) uint8 frame in the default
  `rgb_array` mode (headless-safe) or draws to a window with `mode="human"`;
  the overlay is sliced from the actual observation vector. Extras over
  Box2D: green outline + live `touching/coupling` counter on each box,
  delivered boxes washed out. `save_video(frames, "out.mp4"|".gif")` via
  imageio. Vmapped states: index one env with `jax.tree.map(lambda x: x[i],
  state)`. The same module also has `MuJoCoNativeRenderer(env, camera="iso"|
  "top")` — native MuJoCo OpenGL rendering via `mujoco.Renderer` against a
  cosmetic **visual twin** model (`env._build_xml(..., visual=True)`: same
  bodies/joints so the MJX qpos copies straight into a host `MjData` +
  `mj_forward`; adds contype-0 floor/walls/target-band/skybox/light, never
  stepped). Coupled boxes tint green, delivered fade translucent. Needs
  `MUJOCO_GL=egl` headless. Demo (writes mp4 + png, scripted delivery via the
  shared `scripted_push_action`): `MUJOCO_GL=egl SDL_VIDEODRIVER=dummy uv run
  python -m environments.mjx_suite.renderer [--native iso|top]`.
- **Keyboard control** (`renderer.manual_control(env)`, `--manual`) — the MJX
  counterpart of the box2d suite's per-env `manual_debug`, same control scheme:
  `[ARROWS]` move the controlled agent, `[SPACE]` switches it, `[G]` toggles
  group control, `[R]` resets, `[TAB]` moves the sensor overlay, `[ESC]` quits.
  **Group control drives every agent within the controlled agent's sensing
  radius** (`env.sector_sensor_radius`, = `world_width/3`) with the same force,
  recomputed from live positions each step so membership tracks the
  neighbourhood rather than freezing when `G` was pressed. It exists because
  `coupling` agents must touch a box *simultaneously* to move it — single-agent
  control cannot exercise the coupling mechanic at all. Grouped agents get an
  orange ring (`MJXRenderer.render(..., highlight=[...])`, a general hook; empty
  `highlight` is a no-op, so recorded rollouts are unchanged), and since the
  sensor overlay already draws the sector-radius circle, every ring should sit
  inside it. The window caption carries step / return / group size / delivered.
  Needs a **real display** — do not set `SDL_VIDEODRIVER=dummy`. No `MUJOCO_GL`
  needed (that is only for `MuJoCoNativeRenderer`; this path is pygame and MJX
  is pure compute). `manual_control` draws the reset state **before** entering
  the loop: `MJXRenderer` creates its screen lazily inside `render()`, and
  `pygame.event.get()` / `key.get_pressed()` raise `video system not
  initialized` until `pygame.init()` has run.
  ```
  uv run python -m environments.mjx_suite.renderer --manual \
      [--n-agents 16 --n-objects 4 --variant drift|trunc --seed N]
  ```
  **`--env {square,circular}`** picks which env the whole CLI drives (scripted
  recording, `--native`, and `--manual` alike): `square` = `MultiBoxPushMJX`
  (default), `circular` = `MultiBoxMultiGoalPushMJX`. It also selects that
  module's `scripted_push_action`. `manual_control` itself is env-agnostic —
  the square-only `variant` / `box_drift_speed` / `boundary_ends_episode` in
  its banner are read via `getattr`, and `--variant` errors on `--env circular`
  (that env has no presets) rather than being silently ignored.
  A `.vscode/launch.json` entry must use `"module":
  "environments.mjx_suite.renderer"` + `"cwd": "${workspaceFolder}"` — the
  module uses absolute imports, so `"program": ".../renderer.py"` fails with
  `ModuleNotFoundError: No module named 'environments'`.
- Demo/sanity check (scripted delivery rollout + vmapped throughput):
  `uv run python -m environments.mjx_suite.multi_box_push_mjx`. Step-matched
  wall-clock shootout vs Box2D (`profile_multi_box_push.py`, Box2D on all
  cores via AsyncVectorEnv, MJX vmapped on GPU): at 30a/6o, 100 eps x 1024
  steps, MJX 15.0s (6.8k steps/s) vs Box2D 56.5s (1.8k steps/s) — 3.8x, plus
  a one-time ~7s MJX compile. Not wired into `create_env` (torch MAPPO gains
  nothing from it), but it **is** the training env of the fully-jitted
  `mappo_jax` stack (below) via `EnvironmentEnum.MULTI_BOX_MJX =
  "multi_box_push_mjx"`.

#### Compact global state (`use_global_state`, off by default)

`MultiBoxPushMJX` can publish a real centralized state instead of leaving the
critic (and, under `feudal_mappo_jax`, the manager) to read
`obs.reshape(n_envs, -1)` — N egocentric 40-dim views with no shared frame. The
flag is **`env.use_global_state: true`**, forwarded by the `MULTI_BOX_MJX` branch
of both `mappo_jax/run.py` and `feudal_mappo_jax/run.py`; nothing else changes —
the actor still reads its own observation, only the critic (and the feudal
manager) switch. Turn it on with a **twin env group** differing in exactly that
one key, never with a CLI override on the base group: the results path is
`results/<env group>/<model>/<trial>`, so an override would write into — and a
`checkpoint=true` resume would try to load — the baseline's directory, whose
critic has a different input width. Current arms:
**`mjx_6a_4o_1024_gs`** (twin of the untracked `mjx_6a_4o_1024`) and
**`mjx_1a_3o_111_1024_gs`** (twin of the single-agent `mjx_1a_3o_111_1024`;
gs_dim 22 vs 40). The `mjx_16a_4o_trunc_1024_gs` group this section used to name
no longer exists in `conf/env/`.

```
uv run python train.py algorithm=mappo_jax env=mjx_1a_3o_111_1024_gs \
    model=mlp trial_id=0
```

- **Single agent: here the hook adds INFORMATION, not just exactness** (the ⚠
  below is about the multi-agent case). With one agent the concat-obs critic
  input is that agent's own egocentric view. Measured at 1a/3o (256 envs,
  sensor radius W/4 = 7.5, 30-wide arena), it senses **5.7%** of boxes at spawn,
  **6.3%** over 300 random-action steps, and **none 84%** of the time. The compact
  state carries every box's position and `delivered` flag. Verified end-to-end
  with `mappo_jax` (train, `checkpoint=true` resume, `evaluate=true`): the
  checkpoint's critic `Dense_0/kernel` is `(22, 336)` and the actor's
  `(40, 168)`.

- **Layout** (`_compact_global_state`), agent-major then box-major, width
  `n_agents*4 + n_objects*6` — **88 at 16a/4o against 640 for concat-obs**:

  | block | dims | contents | normalization |
  |---|---|---|---|
  | agents | `(A, 4)` | `pos - centre`, `vel` | `/(world_width, world_height)`; `/velocity_norm` |
  | boxes | `(O, 6)` | `pos - centre`, `target_y - box_y`, `delivered`, `touch.sum(0)`, `coupling` | `/(W, H)`; `/world_height`; —; `/n_agents`; `/n_agents` |

  Every block is normalized with the env's own constants so none enters the first
  Tanh at a different scale — the defect measured for `normalize_pooled_goal`,
  where a 5x-scaled block took 91.6% of layer-1 preactivation variance. Measured:
  `max |gs| = 1.557` over 100 steps at 16a/4o. `GLOBAL_STATE_AGENT_FEATURES` /
  `GLOBAL_STATE_BOX_FEATURES` **are** the layout — `compact_global_state_dim`
  derives from them and the builder uses them, so the declared and emitted widths
  cannot disagree; `__init__` also checks it via `jax.eval_shape` (abstract, so no
  FLOPs and no MJX compile). The two blocks themselves come from
  `entity_state(state) -> (agents (A,4), boxes (O,6))`, and
  `_compact_global_state` is their flattened concatenation (refactored
  2026-09-27, bit-identical). `entity_state` is an unconditional method with no
  `hasattr` consumer; `simplified_feudal_mappo_jax`'s `manager_input: relative`
  reads it.

- **⚠ What it buys is EXACTNESS AND WIDTH, NOT INFORMATION.** A *linear* readout
  already recovers agent world coordinates from the 640-dim concat to **3.36**
  world units in a 47-wide arena, against 17.95 for predict-the-mean and 0.006
  from the compact state (`global_state_probe.py`, results pickled next to it).
  Do not write this up as "the manager finally sees where things are." What the
  observation genuinely does **not** carry is `delivered` and the live touch
  count.
  - `delivered` is the load-bearing one: delivery **latches**, so a box shoved
    back out of the band stays delivered and stops paying while its position says
    otherwise. Without it, "already banked its +100" and "about to pay +100" are
    the same state and `V(s)` must predict a 100-point discontinuity from a state
    that does not contain it.
  - `touch.sum(0)` is not derivable from the rest of the vector — the touch test
    needs box **yaw**, which is deliberately omitted.
  - ⚠ `goal_dist` is **exactly redundant here**: the band is fixed, so
    `target_y - box_y` is affine in the box-y feature already present. It is kept
    because it costs `O` dims, spares the critic a constant offset, and keeps the
    layout portable to the per-box-goal envs where it is not redundant. Do not
    report it as new information in this env.

- **The gating is instance-binding, and it has to be.** `hasattr(env,
  "global_state")` is the trainers' only switch and a class method always exists,
  so an unconditional method would flip **every** MJX arm from `gs_dim = 640` to
  88 — every saved `models_*.msgpack` failing `from_bytes`, every past result
  incomparable. `__init__` therefore binds `self.global_state =
  self._compact_global_state` into the **instance** dict only when the flag is
  set. Read `env.global_state_enabled` in tests/logging, never `hasattr`. ⚠ The
  ctor arg is `use_global_state`, **not** `global_state`: the latter invites
  `self.global_state = bool(...)`, which makes `hasattr` true with a bool and
  explodes inside `jax.vmap` far from the line that caused it.
  - Verified: the hook is off by default, `gs_dim` stays 640 on every existing
    group, and a **trained 1e8-step `mjx_16a_4o_trunc_1024/mlp/0` checkpoint still
    loads and evaluates (282.42)** after the change.

- **⚠ The layout is FROZEN once an arm trains.** `gs_dim` is baked into every
  checkpoint, so adding box yaw later means a **new env group**, not an edit.
  `GLOBAL_STATE_VERSION` exists to say so.

- **`trainer.global_state_fn(env)`** is now the single definition of how the
  centralized input is built (twin copies in both stacks, as CLAUDE.md requires),
  replacing three hand-copied `if hasattr(...)` predicates. `make_train`,
  `feudal run.py:view()` and the probes all route through it. This matters because
  **`view()` used to hardcode `obs.reshape(-1)`** as the manager's input — with the
  hook on that is an 88-vs-640 mismatch, a bare `ScopeParamShapeError` at render
  time, and under `manager_latent: centralized` that vector is the manager's *only*
  input.

- **`SyncMacroMJX` REFUSES the hook** (`NotImplementedError` in `__init__`). It
  re-declares its metadata explicitly and has no `__getattr__`, so a hook on the
  base env does not propagate — the macro arm would silently train on concat-obs
  while its config claimed otherwise. Forwarding it later is ~5 lines over the
  existing `base_state()` staticmethod.

- **The latent probes refuse a hook-on arm** (`latent_locality_probe.
  unsupported_env_reason`): `collect_states` derives the manager input from obs,
  i.e. assumes `global_state == concat(obs)`. The message now points at the
  hook-**off** twin group, which the opt-in design guarantees exists and differs
  in exactly one key. Making them work on a hook-on arm means storing `obs`
  alongside `env.global_state(state)` rather than deriving one from the other.

- **Not carried over**: `MultiBoxMultiGoalPushMJX` (a small port — same constants,
  pass `_goal_dist(box_pos) / world_width` as the goal block; ⚠ that is the env
  where the hook adds *real* information, since `MJXObservationBuilder.build`
  zeroes `goal_distance` for any agent whose nearest box is out of
  `sector_sensor_radius`).

- Checks: `uv run python -m environments.mjx_suite.multi_box_push_mjx
  --check-global-state [--n-agents 16 --n-objects 4]` (4 assertions: default-off,
  width, layout round-trip, scale) and `uv run pytest
  algorithms/tests/test_mjx_global_state.py -q` (8 tests, CPU-pinned).
- **⚠ No training result.** Everything above is a mechanism check; whether the
  compact state helps return needs full-length paired runs against
  `mjx_16a_4o_trunc_1024` at the same seeds.

#### Larger arena (`arena_scale`, default 1.0, added 2026-10-01)

`env.arena_scale` (>= 1.0) makes the boxes harder to find. The arena width is
`round(base_world_width * arena_scale)`, where `base_world_width` is the usual
`int(30 * max(1, (A+O)/8) ** 0.5)`. Everything laid out relative to the width
scales with it: the agent spawn grid (bottom third), the box spawn band (middle
30% of the height, 80% of the width), the goal band (`max(5, 5H/30)` tall), the
wall planes, and position normalization (`_centre`/`_extent`, so `goal_state`
and the compact global state stay in [-0.5, 0.5]).

The sensing ranges do **not** scale. `sector_sensor_radius`, `lidar_range` and
`comm_radius` stay at `base_world_width / 4`. Agent radius, box sizes, forces and
speeds are unchanged.

- **`arena_scale: 1.0` is bit-identical to the pre-flag env.** Verified over
  1a/3o, 2a/4o, 6a/4o, 9a/3o, 12a/3o and 16a/4o, each with `variant` None and
  `trunc`: 24 fields per config (geometry, both MuJoCo XMLs, spawn tables,
  reset `qpos`, global state) have 0 differences.
- **Plumbing:** forwarded at all four `MultiBoxPushMJX` construction sites
  (bare + macro base, in `mappo_jax/run.py` and the `feudal_mappo_jax` copy),
  so `simplified_feudal_mappo_jax` gets it through `make_env`. MJX-only; the
  Box2D env and `MultiBoxMultiGoalPushMJX` have no equivalent. Both the renderer
  CLI and the env demo take `--arena-scale`, e.g.
  `uv run python -m environments.mjx_suite.renderer --manual --arena-scale 2`.
- **Use a twin env group, never a CLI override.** Observation and global-state
  widths do not change, so a `checkpoint=true` resume into the baseline's results
  directory would load without error and keep training a different task. The
  example arm is `conf/env/mjx_2a_4o_1122_1024_gs_arena1p5.yaml`, which
  differs from `mjx_2a_4o_1122_1024_gs` only in `arena_scale: 1.5`.
- **Measured effect** (2048 resets for sensing; 64 rollouts of a scripted team
  that reads true box positions for travel time; `trunc`):

  | config | scale | width / sensing radius | agents sensing no box at spawn | whole team blind at spawn | median step of first delivery (scripted) |
  |---|---|---|---|---|---|
  | 1a/3o | 1.0 / 1.5 / 2.0 | 30 / 45 / 60, R 7.5 | 83.1 / 95.1 / 98.8% | same | 139 / 211 / 299 |
  | 2a/4o | 1.0 / 1.5 / 2.0 | 30 / 45 / 60, R 7.5 | 79.5 / 94.0 / 98.4% | 63.0 / 88.5 / 96.8% | 154 / 247 / 324 |
  | 6a/4o | 1.0 / 1.5 / 2.0 | 33 / 50 / 66, R 8.25 | 73.6 / 94.9 / 98.3% | 18.3 / 74.4 / 90.8% | 171 / 271 / 365 |

- **⚠ Travel time grows with the arena, and `max_steps` is not reachable from
  Hydra.** None of the `run.py` branches forward `max_steps`, so every group runs
  at the constructor default of 1024. A scripted team that knows every box
  position takes ~1.6x / ~2.1x as long to its first delivery at scale 1.5 / 2.0,
  before any search cost. A return gap against the scale-1 twin can therefore be
  time as well as search. Raising `max_steps` also requires `params.n_steps >=
  max_steps`, because `collect_fn` resets every env at the top of every rollout,
  and that changes the per-update batch.
- **Feature-scale side effects:** `nearest_box_vec` is divided by
  `world_width` but capped at the fixed sensing radius, so its maximum magnitude
  falls from 0.25 to `0.25 / arena_scale`. `goal_distance` keeps its [-1, 1]
  range. Simplified feudal's `waypoint_radius` is a fraction of the width, so its
  reach in world units grows with the scale. R = 0.15 was calibrated to the
  32-step reach in a 30-wide arena; rescale it to `0.15 / arena_scale` to keep
  the same reach. `validate_manager_input`'s `R*W >= lidar_range` warning fires
  sooner, correctly.
- Smoke-verified 2026-10-01 on the 1.5 group: `mappo_jax` train,
  `checkpoint=true` resume and `evaluate=true` all run. **No training result.**

#### `reward_mode="sparse"` (implemented, arm added 2026-09-04)

One branch, no plumbing: `task_reward = completion + (shaping if self._dense
else 0.0)` (`multi_box_push_mjx.py`, and the identical line in
`multi_box_multi_goal_push_mjx.py`). `sparse` drops the per-step displacement
shaping and pays **only** the one-time `+100` per delivered box; delivery
latching, the `prev_box_goal_dist` bookkeeping, the observation and the physics
are untouched. Measured (9a/3o, scripted balanced partition, 600 steps): dense
return 331.09 with the first nonzero reward at step 66 (+9e-4), sparse 300.00
with the first at step 245 (+100.0), both delivering 3/3.

- Accepted by `MULTI_BOX_MJX` and `MULTI_BOX_MULTI_GOAL_MJX` (both validate
  `dense|sparse|difference_rewards` and `run.py` forwards it in `mappo_jax` and
  `feudal_mappo_jax` alike). `MACRO_MJX` passes it to the base env *and* to
  `SyncMacroMJX`, which accumulates it over the window — structurally fine,
  **untested**. Not combinable with the windowed DR modes (they raise unless the
  base is dense); SMAX hardcodes `"dense"`.
- **The arm is `conf/env/mjx_16a_4o_trunc_1024_sparse.yaml`** — a copy of
  `mjx_16a_4o_trunc_1024` differing in exactly one key, so it is the controlled
  sparse-vs-dense comparison against *that* group. Every MJX env group already
  declares `reward_mode`, so `env.reward_mode=sparse` also works as a plain CLI
  override under Hydra struct mode.
- **⚠ Do not pair sparse with `mjx_16a_4o_1024`.** That group has
  `boundary_ends_episode: true` and `boundary_hit` is `any()` over 16 agents, so
  episodes die at ~43 steps on incidental wall contact. With shaping removed
  there is nothing in the gradient before the first delivery and a 43-step
  episode essentially never reaches one — the signal is all-zero. The `trunc`
  base (inert walls, full 1024 steps) is what makes a delivery reachable.
- **⚠ Do not pair sparse with `variant: drift`.** The decay's only channel into
  the reward is the shaping term; with shaping gone an unattended box sinks for
  free until it makes delivery outright impossible, so that combination is
  closer to plain sparse than to "sparse with coalition pressure".
- **No sparse run has been done** — every number above is a scripted-oracle
  mechanism check, not a training result.

#### Box drift / "decay" (`box_drift_speed`, off by default)

Every box whose coupling requirement is **not currently met** sinks toward the
bottom wall at a constant speed, so progress decays on any box the team is not
working on. Added because `mappo_jax` fully solves `mjx_16a_4o`: a 16-agent
swarm can deliver boxes one at a time, so no coalition structure has to be
discovered. With the drift a sequential schedule arithmetically cannot reach its
last box. Env groups `conf/env/mjx_16a_4o_drift.yaml` (`variant: drift`) and
`conf/env/mjx_16a_4o_trunc.yaml` (`variant: trunc`, the boundary-semantics
control — see below; the drift arm changes two things at once, so this arm is
what makes the comparison attributable).

- **Config surface is `env.variant`, a preset, and it is the WHOLE surface** —
  every knob is a constant of the preset, so an arm is fully identified by its
  name and there are no `box_drift_speed` / `box_drift_floor` kwargs to pass:
  `"drift"` → `box_drift_speed=_DEFAULT_DRIFT_SPEED` (currently `0.5`, the knee
  of the calibration table below) + inert walls; `"trunc"` → inert walls
  only; absent/`None` (baseline `mjx_16a_4o`, every `macro_mjx_*`,
  `multi_box_push_mjx_*`) → neither, i.e. stock Box2D-parity behavior, verified
  **bit-identical** (obs/reward/qpos/qvel over a fixed-seed 200-step rollout, in
  both `dense` and `difference_rewards`). `run.py` forwards `variant` to
  `MultiBoxPushMJX` at both construction sites (bare and macro-wrapped).
  - **Regression to know about (fixed 2026-07-31, was live in 32cf60d and
    7db37b3):** `VARIANTS` was a plain `Enum` read through a side table
    `_VARIANT_MAP = {"trunc": 1, "drift": 2}`, so the guards compared a raw
    `int` to an enum member — `2 == VARIANTS.DRIFT` is silently `False`. Drift
    and the boundary flag were therefore **off in every run**, and
    `variant=None` raised `KeyError: None`, breaking the baseline/macro groups
    outright. `VARIANTS` is now a `StrEnum` (the repo idiom, cf.
    `EnvironmentEnum`) parsed via `VARIANTS(variant)`, which also rejects an
    unknown name instead of failing open. **Any `mjx_16a_4o_drift` /
    `mjx_16a_4o_trunc` result produced before this fix is really a baseline
    run and must be rediscarded/retrained.** The `--check-drift` suite did not
    catch it because the same commit dropped the `box_drift_speed` kwarg the
    suite constructs with, so the suite could not run at all — it now
    constructs via `variant=` and needs no kwargs, so it cannot desync again.

- **Wall contact is inert in both drift arms** (`boundary_ends_episode` is True
  only for the baseline), and the crash step pays its **real** reward rather
  than the Box2D-parity 0. This replaces the earlier `boundary_truncates`
  approach, which **did not work** — policies trained on `_drift` learned to
  crash into a wall on purpose to end the episode. Why bootstrapping cannot fix
  it: the stored return for crashing is `0 + γ·V̂(s_wall)` against
  `r_t + γ·V̂(s_{t+1})` for continuing, so crashing wins by `−r_t` plus the
  critic's own error — with `V̂ ≈ 0` early that is exactly the per-step drift
  bleed, and the old `reward = where(boundary_hit, 0, task_reward)` handed back
  that same bleed *unconditionally*, independent of any critic. Worse, it is
  self-sealing: once the policy crashes at step k no data past k is collected,
  so `V̂` at the bootstrapped states is trained only against other bootstrap
  targets — self-consistent with **any** value — and never learns that drifting
  states are worth `< 0`. Compounding it, `boundary_hit` is `jnp.any()` over
  agents, so one of 16 ends the episode for the team from ~6 steps out of spawn.
  The escape is therefore **removed, not priced**: the walls are real
  inward-facing planes and agents cannot leave regardless, so boundary
  *termination* was parity, not physics. Both arms share the change, so the
  ladder still isolates one thing per step: baseline (wall ends it) → `trunc`
  (wall inert) → `drift` (wall inert + decay).

- **Mechanism**: a generalized force on each box's world-y slide DOF via
  `data.qfrc_applied`, set in `_advance`. The y slide axis is world-fixed
  regardless of box yaw (mjx rotates a joint axis by the quat accumulated from
  *preceding* joints only, and the box joint order is slide-x, slide-y,
  hinge-yaw). Since `_model_for` sets `dof_damping[box y] = _BOX_LIN_DAMPING *
  mass`, sizing the force as `F = -v_d * _BOX_LIN_DAMPING * mass` puts the fixed
  point at exactly `-v_d` **independent of box mass**, with a mass-independent
  time constant `tau = 1 / _BOX_LIN_DAMPING = 0.2 s` (12 steps). Verified:
  terminal `v_y = -0.79995` for `v_d = 0.8`, `|v_x|`/`|v_yaw| < 1e-4`, identical
  across masses 180–627 kg, and the transient at 12 steps is 0.617 == `1 - rho^12`
  with `rho = 1/(1 + k*dt)`.
- **Gate**: `~met & ~delivered & (box_y > box_drift_floor)`. `met` comes from the
  new `_coupling_met`, extracted from `_model_for` so the mass override and the
  drift share one notion of "working together" — and so the drift is masked by
  `active` and is therefore automatically part of every difference-reward
  counterfactual.
- **Floor** (`5 * boundary_thickness + 2 * box_half_extent` = 5.7 at 16a/4o): a
  box resting on the bottom wall is hard to recover — to push it up an agent
  centre must get under it. The clearance keeps that geometry feasible (verified:
  a box spawned exactly at the floor is pushed out and delivered by a *minimum*
  coalition in 260 steps). It costs the mechanic nothing — a passive box needs
  ~20 s to reach it versus a 17 s episode. Implemented as a force gate, **not**
  an mjx joint limit: a limit is static model structure (an extra constraint row
  on every step of every arm, so the drift-off graph would change), MJX limits
  are soft, and it would obstruct legitimate downward pushing. Two known
  properties: the floor guarantees the geometry for a minimum coalition, not
  that any crowd survives; and because it is a *gate*, a box coasts past it by
  its stopping distance `v_d * tau = v_d / k` before the damping kills the
  carried velocity — **0.6 of the 5.7 floor at the current `v_d = 3.0`**
  (measured resting y ≈ 5.06), so the effective clearance is smaller than the
  nominal floor. `--check-drift` [3] ties its tolerance to that formula rather
  than a constant, so raising `v_d` cannot silently eat the margin.
- **`variant=None` is a strict no-op**: every branch is a Python-level `if`, so
  the graph and the numbers are unchanged. Verified **bit-identical** qpos /
  qvel / obs / reward over a fixed-seed 200-step rollout against the pre-change
  code, in both `dense` and `difference_rewards`.
- **Calibration** (16a/4o, 4 seeds, scripted swarm vs balanced partition). Mean y
  of the boxes the swarm ignores / boxes delivered by the partition:

  | `v_d`        | 0.0  | 0.2  | 0.3  | 0.5  | 0.8  |
  |--------------|------|------|------|------|------|
  | ignored-box y| 22.8 | 19.5 | 17.7 | 14.3 | 12.3 |
  | partition box| 3.75 | 3.50 | 3.50 | 3.50 | 3.25 |

  Decay pressure is monotone in `v_d`; `0.5` is the knee — near-maximal decay
  without extra degradation of a *correct* strategy. At `0.8` the 819 N force
  exceeds a full 4-agent coalition's 400 N of thrust, so exactly-coupling
  coalitions become fragile *while forming* and even the scripted partition
  starts dropping boxes.
- **Measured, and contrary to the design prediction:** the partition-over-swarm
  return *gap* does **not** widen with drift (16a/4o, 4 seeds: +318 at `v_d=0` vs
  +284 at 0.8; it stays roughly flat). With a scripted oracle both arms pay drift
  cost. What was robust *within the swept range* is that the partition still wins
  at every `v_d` (the mechanic never inverts the preference) and that the swarm's
  ignored boxes decay monotonically. Whether it changes what a *learned* policy
  does is a training question, not a scripted-probe one.
- **`_DEFAULT_DRIFT_SPEED` is now `0.5`, the knee of the table above, and all 10
  `--check-drift` checks pass** (re-verified 2026-08-13). An earlier value of
  `3.0` was outside the calibrated range and broke the mechanic — it put 3072 N
  against a full coalition's ~400 N, and check [8] failed there (the scripted
  balanced partition delivered 0.5 of 4 boxes with drift on vs 4.0 with it off,
  *losing* to the swarm, i.e. the drift inverted the preference it exists to
  create). At 0.5, [8] reports swarm 91 (1.0 boxes) -> partition 461 (4.0) with
  drift on, i.e. the partition still wins. Keep [8] as the canary if the speed
  is ever changed again.
- **⚠ But the mechanic does not do its job on a *learned* policy.** Measured on
  the trained `mjx_16a_4o_drift` mlp arm (deterministic eval, 24 episodes/seed,
  3 seeds): **3.92–4.00 of 4 boxes delivered**, against 3.42–4.00 on the
  `mjx_16a_4o` baseline; final eval return 458.1 ± 1.4 vs 421.8 ± 8.3, and the
  per-step reward rate is only ~7% lower. Drift was introduced to kill the
  sequential swarm-one-box-at-a-time strategy; at `v_d = 0.5` MAPPO still
  delivers everything. The coupling gate turns the drift off on whatever box the
  team is working, and the floor keeps sunk boxes recoverable, so sequencing
  pays only a bounded one-time cost. **Do not use this arm as the coalition-
  pressure manipulation** — see `conf/env/mjx_16a_4o_partition.yaml` for the
  structural alternative, and note a time budget (cutting `max_steps` below what
  sequencing needs) is the untried knob that would actually forbid it.
  ⚠ The comparison is also not attributable: `mjx_16a_4o_trunc`, the middle rung
  that isolates the inert-wall change, is **stale** — its runs terminate well
  before `max_steps` (episode lengths 12–2000), which is impossible with
  `boundary_ends_episode: false`, so they predate the `97a7dfc` StrEnum fix and
  are really baseline runs; its group also sets `n_steps: 512` against 1024
  everywhere else. Re-run it before attributing anything to drift.
- **Difference rewards are structurally blind to this pressure.** An unattended
  box's drift cost does not depend on agent *i*, so it appears identically in `G`
  and `G_-i` and cancels exactly. Measured: pivotal `D_i` (exactly `coupling`
  agents touching) rises only ~5% (1.332e-2 -> 1.395e-2, re-measured 2026-09-25
  under both the old and new solver settings), and the pile-on case
  (`2*coupling` touching) is unchanged to 4 significant figures. **Use the drift
  arms in the dense/team-reward study, not as a fix for the DR magnitude gap.**
- **Config plumbing**: `algorithms/mappo_jax/run.py` used to `.get()` a hard-coded
  key list at each of its two `MultiBoxPushMJX` construction sites (bare, and as
  the macro wrapper's base), so an `env:` yaml key that only one site forwarded
  was silently ignored — which is why `coupling_def` / `max_steps` /
  `comm_radius` were unreachable from Hydra. ⚠ **This note used to claim a
  `_base_env_kwargs(env_config)` helper (`_BASE_ENV_KEYS` / `_RUNNER_ENV_KEYS`)
  warns on unrecognized `env:` keys. NO SUCH HELPER EXISTS** — a repo-wide grep
  finds it only in this file. Each `run.py` branch still pulls a hard-coded key
  list inline, so a key is reachable only where a branch names it and a typo is
  silent (the correct note is the one above at "Nothing validates the `env:`
  block"). Nothing validates it — no dataclass guards it
  (`environments/types.py:EnvironmentParams` is never instantiated). The one
  mitigation is per-key: both runners print
  `centralized input: gs_dim=... (env hook: ...)` at launch, so a
  `use_global_state` that did not land shows up as the concat width.
- **`scripted_push_action` now takes a per-agent assignment** (`box_idx` scalar,
  or `(A,)` e.g. `jnp.arange(A) % O` for a balanced partition) and gives agents
  sharing a box **distinct slots** along its bottom face. With the old single
  shared staging point they collided and only ~2 ever reached the surface, so a
  coalition of exactly `coupling` agents could never satisfy the requirement —
  the swarm demo only worked because 9 agents crowding one box got 3 in contact
  by accident. Surplus agents clamp to the face edges and crowd in as before.
- Assertion suite (10 checks: no-op, terminal velocity/axis purity/mass
  independence/transient, floor settling, coupling gate, reward semantics,
  inert walls, recoverability, efficacy, vmapped stability, DR structure):
  `uv run python -m environments.mjx_suite.multi_box_push_mjx --check-drift
  [--n-agents 16 --n-objects 4]` (needs jit, so it ignores `--debug`). It
  constructs via `variant="drift"`, so it exercises the arm training actually
  runs and cannot desync from the shipped constants. **All 10 pass** at the
  shipped `_DEFAULT_DRIFT_SPEED = 0.5` (re-verified 2026-08-13). [8] is a config
  canary, not a code defect: it prints its numbers before asserting, and the
  message names the speed — it is what catches an uncalibrated drift speed.

### MJX circular arena / per-box concentric goal rings (`multi_box_multi_goal_push_mjx.py`)

`MultiBoxMultiGoalPushMJX` is a copy of `MultiBoxPushMJX` with the **geometry**
changed and nothing else: same physics constants, coupling mechanic, 40-dim
`OBS_DIM` layout, reward structure (`dense`/`sparse`/`difference_rewards`),
`EnvState`, and functional API. **Trainable with `mappo_jax`**
(`EnvironmentEnum.MULTI_BOX_MULTI_GOAL_MJX = "multi_box_multi_goal_push_mjx"`,
its own branch in `mappo_jax/run.py`, env group
`conf/env/mjx_16a_4o_multi_goal.yaml`); not wired into `create_env` (the torch
stacks) — everything downstream of `run.py` is duck-typed on the functional API,
so the trainer, `view()` and `evaluate()` needed no changes. The box-drift
mechanic and the `variant` preset are deliberately **not** carried over — do not
set `env.variant` on this group.

- **Walls are inert here, and that is the default** (`boundary_ends_episode:
  false`, a plain bool kwarg rather than a `variant` preset — this env has no
  preset surface). It adopts the square env's `trunc` treatment for the same
  reason and then some: the wall segments are real inward-facing planes so
  agents are already confined (terminating on contact was Box2D parity, not
  physics); `boundary_hit` is `any()` over agents, so one of 16 ended the
  episode for the team; and agents spawn in the **outer annulus** here, i.e.
  right against the wall. Decisively, the crash step used to pay `0` instead of
  its real reward while ~1/3 of episodes carry net-negative shaping — a standing
  bonus for touching a wall, the same escape hatch the square env's
  `boundary_truncates` attempt hit. Set `boundary_ends_episode: true` to restore
  the old terminate-on-contact semantics (the ablation arm). Verified: physics
  is untouched (qpos/qvel/obs bit-identical over a 200-step fixed-action
  rollout, both settings); the scripted oracle is **unaffected** (977.9 steps /
  return 279.6 / 2.5-of-4 delivered under both — a competent policy never
  touches the wall before delivering); a random policy goes from ep_len 141 with
  100% wall-termination to the full 1024 with 0%. Plumbed at both `run.py`
  construction sites (`mappo_jax` and `feudal_mappo_jax`).
  - ⚠ This is now a **third** uncontrolled difference from the `mjx_16a_4o`
    baseline it is ablated against (after arena shape and per-box goals), which
    already was not a controlled geometry ablation. Note the measured
    wall-termination rate was *not* what separated the two arms — under trained
    policies both sat at 0.22–0.31, and the square baseline reaches 400+ anyway.
    This change removes a hazard, it does not explain the failure.

- **Arena is a disc** of `arena_radius = world_width/2 - boundary_thickness`
  about the world center (now the *geometric* center, `W/2` not `W//2`). MuJoCo
  has no concave primitive, so the wall is `_N_WALL_SEGMENTS` (32)
  **inward-facing planes tangent to that circle** — the free region is the
  intersection of their half-spaces, a regular N-gon with apothem
  `arena_radius`, whose corners stick out by `1/cos(pi/N) - 1` = 0.5% at 32.
  Same construction as the square arena's four wall planes, just more of them;
  measured **no throughput cost** (32 envs, 9a/3o: 20.2k vs 13.9k steps/s for
  the square env — i.e. within run-to-run noise, not slower). `N` is the one
  fidelity/cost knob (it multiplies candidate collision pairs and lidar ray
  tests).
- **Goal is one concentric ring per box.** The `[0, goal_outer_radius]` disc is
  cut into `n_objects` rings of equal width and **box j belongs in ring j
  counted from the center out** — box 0 in the central disc, box 1 in the
  annulus around it, and so on — so the boxes are not interchangeable: each has
  its own stopping radius, and the outer ones must be left in place while the
  inner ones are pushed past them. Boxes and rings are **color-coded to match**
  (`env.box_colors`, the `COLORS_LIST[n_agents + j]` scheme, now the single
  source of truth for the box geoms, the `CircularTargetArea`s, the native
  discs, and `MJXRenderer._draw_boxes`), and each ring is labelled `BOX j`.
  - Ring width is the box's **side** (`2*max(box_half_extents)`) where the arena
    affords it, so a box square-on fits its ring; `_max_goal_radius` caps the
    whole structure at the largest rim that still leaves a usable agent spawn
    annulus and the rings shrink uniformly if that binds. At the shipped configs
    it does not bind: **9a/3o -> 3 rings of 3.73 (rim 11.19), 16a/4o -> 4 of
    4.80 (rim 19.2)** (measured; the 3.00/9.0 and 3.20/12.8 recorded here
    earlier are stale). The goal block therefore sits *below* the
    coupling/box-size block in `__init__` (it needs `box_half_extents`).
    ⚠ The rim matters for difficulty: boxes spawn at r ~ 21.6 at 16a/4o, so with
    the rim at 19.2 the **outermost box needs only 2.4 units of travel while box
    0 needs 17.0** ([17.0, 12.1, 7.2, 2.4]). Deliveries by a trained policy track
    that ordering exactly ([2, 2, 6, 10] over 80 episodes), i.e. essentially all
    of the learnable signal is the outermost box.
  - `_BOX_RING_FRAC` is **0.25**, not the 0.40 the single-goal version used:
    the goal structure now grows with `n_objects`, and pulling the box spawn
    ring inward is what buys the radial room for full-width rings (at 0.40 the
    cap bit and 9a/3o rings came out 2.23 wide against a 3.0 box). Boxes end up
    at nearly the same radius either way, since the rim they are measured from
    moved out by about as much as their offset shrank. The spawn-layout
    constants live at module level precisely because `_max_goal_radius` inverts
    `_agent_annulus_inner` to derive that cap — one copy, so the cap cannot
    drift from the layout it protects.
  Everything keyed to the goal axis becomes radial **and per-box**:
  - delivery is `ring_inner[j] <= |box j - center| <= ring_outer[j]`; a box
    parked in someone else's ring is *not* delivered and keeps bleeding shaping;
  - shaping is the reduction in `_goal_dist`, now indexed **by box** — entry j
    is box j's distance to ring j (`clip(| |box-center| - ring_mid[j] | -
    half_width, 0)`), 0 inside its ring and growing on **both** sides, so
    approaching from either side pays and burrowing past pays nothing. Ring 0 is
    a disc and the same formula degenerates correctly for it (`|r - w/2| - w/2
    <= 0` for all `r <= w`). `in_goal` is literally `dist <= 0`, one source of
    truth for the two;
  - the `goal_distance` obs is a **per-agent** offset from the centerline of the
    ring belonging to the box that agent is sensing — the same nearest
    undelivered box `nearest_box_vec` points at, so the two features describe
    one box. A single global goal radius would say nothing about where *this*
    box has to go. The lookup uses the new shared
    `MJXObservationBuilder.nearest_box_indices` (factored out of
    `nearest_box_vectors`, same search, one copy); `goal_radius` accepts a
    per-agent `(A,)` array and just broadcasts;
  - **the obs is `MULTI_GOAL_OBS_DIM = OBS_DIM + 1 = 41`**, not the shared 40 —
    this env appends a task-specific **extras tail** after lidar. Currently one
    scalar, `[40] coupling_fraction` = `coupling[nearest] / n_agents` in [0, 1]:
    the share of the whole team that must touch the agent's nearest undelivered
    box *simultaneously* before it drops to its light mass. It is keyed to the
    same nearest box as `nearest_box_vec` [21:23] and `goal_distance` [23], so
    the three say "that box is there, it must go this far, it takes this share of
    the team to move it"; it is 0 when every box is delivered or `n_objects == 0`,
    matching `nearest_box_vec`'s convention.
    - **Extras APPEND, they never insert.** Every index into the shared 40 dims
      stays valid — in particular `mjx_suite/renderer.py`'s `_DENSITY_SLICE` /
      `_BOX_VEC_SLICE` / `_GOAL_IDX` / `_LIDAR_SLICE = slice(BASE_OBS_DIM,
      OBS_DIM)`, which the sensor overlay slices straight out of the policy
      input. Inserting the scalar before lidar would silently shift that slice
      and the overlay would render a different vector than the policy sees.
      Verified: `_get_obs(...)[:, :40]` is bit-identical to the bare
      `obs_builder.build(...)` (compare both **under jit** — `mjx.ray`'s reduction
      order differs between compilations and a jitted-vs-eager diff shows a
      spurious ~1.6e-4 in the lidar block).
    - The **shared** `MJXObservationBuilder` is untouched, so no other MJX env,
      no Box2D env and no existing checkpoint changes width. Everything
      downstream reads `env.observation_dim` (`mappo_jax/trainer.py:54`,
      `run.py:373`), so the network width follows automatically.
    - ⚠ **At the shipped config the feature is a constant.** `coupling_def:
      "even"` gives `coupling = [4,4,4,4]` at 16a/4o, so `coupling_fraction` is
      0.25 for every agent, box and step — zero information, just a bias the
      first layer absorbs. It only varies under an explicit unequal
      `coupling_def` list (e.g. `[2, 3, 5, 6]`). See the `coupling_def` note
      below for what switching costs.
  - boundary contact is `|agent - center| >= arena_radius - agent_radius` —
    measured on the circle, so in the N-gon's corner directions it trips ~0.5% of
    R early (conservative).
  Helpers `_radius` / `_outward` / `_goal_dist` own the polar math. **Delivery
  still latches** (`delivered | newly_delivered`), so a box shoved out of its
  ring stays delivered — matching the square env's semantics rather than
  re-paying/revoking the +100.
- **Spawn layout keeps the square env's ordering** (agents behind the boxes,
  boxes between agents and goal) mapped onto the radius: goal rings in the
  middle -> boxes on a ring `_BOX_RING_FRAC` (0.25) of the way from the
  outermost goal ring's rim to the wall (shuffled *angular* slots + radial
  jitter), so every box spawns outside *every* goal ring and none starts in
  another's target -> agents on concentric rings of cells in the outer annulus,
  whose inner radius clears the outermost box surface. Same
  jitter-safety rule as the square grid (jitter <= half the smallest cell gap).
- **`scripted_push_action` needed two fixes** that are easy to re-break:
  1. **It orbits, it does not beeline.** Agents spawn at *every* bearing here,
     so a straight line to the staging point runs through the arena middle and
     into the box's **inner** face — an agent arriving that way pushes the box
     *outward*. Measured before the fix: the swarm shoved its box from r=10 to
     r=14.5 and into the wall. Agents now circle at the docking radius toward
     the staging bearing and only close in once they are outside the box and
     roughly behind it. That bearing tolerance must include the box's **own**
     angular half-width (`20 deg + asin(half / box_r)`), which grows as the box
     nears the center: under a fixed cone the outer lateral slots fall outside
     it once the box is close in (at 16a/4o a slot 1.35 off-axis subtends
     19 deg at r=4), so those agents orbit forever and the coalition stalls one
     agent short of `coupling` — measured as `touch` pinned at 3/4 for hundreds
     of steps while the box crawled.
  2. **Stand-off must be `half + 0.6`** (= agent radius + touch eps), so an
     agent that reaches its staging point is *already touching*. Standing off
     far enough to clear a rotated box's `sqrt(2)*half` corner reach makes the
     agent hover: it pushes in, fails the `close` test, and is pulled back out,
     so a minimum coalition never gets all `coupling` members on the box (16a/4o
     partition: 0 boxes delivered vs 3 after the fix).
  3. **Agents go limp once their box is delivered** (`state.delivered[idx]` ->
     zero action). With a ring, pushing does not stop being right *and then
     wrong* — keep pushing and the box exits through the inner edge into the
     next box's ring. Delivery triggers at the ring's *outer* edge and the box
     then coasts ~1.4 units before the damping kills its speed, which lands it
     about the centerline of a box-wide ring. Chasing the centerline explicitly
     instead overshoots by that same coast and drops the box a ring too far in
     (measured: box 1 parked at r=2.80 against a `[3.00, 6.00]` ring). Delivery
     latches, so this cannot oscillate.
  Sanity numbers, balanced partition (`arange(A) % O`), 1024 steps: 9a/3o
  delivers **3/3** by step ~355 (return ~316), each box parked inside its own
  ring (r = 2.98 / 5.01 / 8.96 for rings `[0,3] / [3,6] / [6,9]`); 16a/4o
  delivers 3/4 (return ~330) — the innermost box has the longest trip and ends
  a few tenths short of its ring at the step limit. 3/4 at 16a/4o is this
  controller's standing result, not a regression from the per-box goals.
- **Shared code extended, not copied** (all changes inert for existing envs):
  `CircularTargetArea` in `box2d_suite/utils.py` (disc/annulus drop zone, no
  `width`/`height` — that is what the renderer keys off — with an optional
  `inner_radius`, default 0 = plain disc, validated `0 <= inner < radius`, plus
  `color` and a `label`); `MJXObservationBuilder.goal_distances(...,
  goal_axis="radial", goal_radius=)`, where `goal_coord` is the center `(2,)`
  and the feature is `(|agent - center| - goal_radius) / world_width` —
  **unchanged**, per-box rings are expressed purely by passing a per-agent
  `goal_radius`; `MJXObservationBuilder.nearest_box_indices` (the
  nearest-undelivered-box search, factored out of `nearest_box_vectors` so an
  env with a per-box goal can look up *which* box an agent senses); and four
  branches in the shared box2d `Renderer` — a ring for `_draw_boundary_walls`
  when `env.arena_radius` exists, a disc/annulus branch in `_draw_target_areas`
  (the hole is punched by drawing it `(0,0,0,0)` on the SRCALPHA surface —
  pygame *replaces* pixels rather than blending — with an outline and label
  darkened from the zone's own color, and concentric zones labelled at the
  middle of their own band rather than all stacked on the shared center), and a
  `goal_axis == "radial"` case in `_draw_goal_distance` (segment points at the
  goal center). `MJXRenderer._draw_boxes` prefers `env.box_colors` when present
  and washes delivered boxes only 30% toward white (was 65%) — in this env the
  hue is what ties a box to its ring, so washing it out hid the assignment.
- Both renderers work unchanged otherwise: `MJXRenderer(env)` (pygame, verified
  by rendering a delivery rollout) and `MuJoCoNativeRenderer(env)` — the visual
  twin builds the wall as a ring of tangential slabs (half-length
  `R*tan(pi/N)`, so they meet corner-to-corner) and the goal rings as nested
  cosmetic cylinders in the boxes' colors, outermost lowest, each inner one
  stacked just above and covering the previous one's middle (MuJoCo has no
  annulus primitive). They must be **opaque** — translucent discs blend with the
  ones below instead of covering them, turning every ring into a mix of the
  colors outside it — and the whole stack has to stay between the floor
  (z=-0.41) and the bottom of the boxes (z=-0.4), hence the `0.008/n_objects`
  z-step, or the discs cut through the boxes.
  Keyboard control works too, via the shared CLI's env switch (verified: reset
  draw + step loop + auto-reset on truncation):
  `uv run python -m environments.mjx_suite.renderer --env circular --manual`.
- **Training** (verified end-to-end: train writes the usual
  `training_stats_*.pkl` / `models_*.msgpack` under
  `experiments/results/mjx_16a_4o_multi_goal/mlp/<trial>/`, and `evaluate=true`
  reloads them):
  ```
  uv run python train.py algorithm=mappo_jax env=mjx_16a_4o_multi_goal \
      model=mlp trial_id=0
  ```
  ⚠ The group ships `params.n_steps: 512` against the env's `max_steps: 1024`.
  `collect_fn` **resets every env at the top of every rollout** and then scans
  exactly `n_steps`, so at 512 training never sees the second half of *any*
  episode — and in this task deliveries land late (the scripted oracle needs
  ~350–900 steps), so the +100 bonuses would be almost entirely outside the
  training distribution. Use `n_steps: 1024` to cover a full episode, which also
  makes the per-update batch (1024 x 32) identical to the `mjx_16a_4o` baseline
  arm it is meant to be compared against.
- Demo: `uv run python -m environments.mjx_suite.multi_box_multi_goal_push_mjx`.

## JAX MAPPO (`algorithms/mappo_jax/`)

`algorithm=mappo_jax` (`AlgorithmEnum.MAPPO_JAX`) is a fully-jitted MAPPO that
trains **directly on the functional MJX envs** (`MultiBoxPushMJX` and its
hierarchical macro wrapper `SyncMacroMJX`; the old JaxMARL dict-API path was
removed). It is a deliberate logic mirror of `mappo_vanilla` so runs are
drop-in comparable:

- **Same per-iteration cadence** (`run.py` ≙ `VecMAPPOTrainer.train`): jitted
  `collect_fn` (≙ `RolloutCollector.collect` — resets all envs at the top of
  every rollout, scans `params.n_steps` (per-update batch = `n_steps * n_envs`
  env-steps here; the vanilla stack reaches the same total via an explicit
  `params.batch_size` — see the hardware-invariance rules above), restarts envs that
  finish mid-rollout since MJX has no auto-reset, bootstraps the final value) →
  jitted `update_fn` (≙ `MAPPOAgent.update`) → jitted deterministic `eval_fn`
  (≙ `PolicyEvaluator`, 5 parallel episodes → the `reward` stat). Deviation:
  eval runs every 10 updates (+ the last), not every iteration — it scans a
  full `env.max_steps` sequentially, which would dominate wall-clock — and the
  `reward` stat carries the last eval forward in between.
- **Same PPO update semantics** (`mappo.py`): env-level GAE on the scalar team
  reward with the shared critic (vanilla tiles it per agent — identical math),
  per-env-stream advantage normalization (unbiased std), timestep-centric
  minibatches (`(batch // n_minibatches) // n_agents` timesteps each, critic
  once per timestep), combined loss `policy + val_coef*value +
  ent_coef*entropy` (actor/critic use separate Adams — equivalent, no shared
  params), pre-update `explained_variance`. Known deviations: the trailing
  partial minibatch is dropped (jit needs static shapes), shared-actor only
  (`parameter_sharing=false` raises). Both continuous (base `MULTI_BOX_MJX`
  force control) **and discrete** (the hierarchical `MACRO_MJX` skill-selection
  env, `SyncMacroMJX`) action spaces are supported: the env declares
  `env.discrete` and `run.py`/`trainer.py` thread it into the actor head
  (categorical logits vs diagonal Gaussian) and the `_actor_forward` reshape —
  the shape-agnostic `ppo_update` handles integer-index actions unchanged. The
  `MACRO_MJX` group (`conf/env/macro_mjx_9a_3o.yaml`, `model=mlp`) trains a
  hierarchical policy that picks among 4 scripted skills every `macro_len`
  low-level steps; verified end-to-end (train + checkpoint resume) on GPU.
- **Truncation vs termination bootstrap** (all three MAPPO stacks: `mappo_jax`,
  `mappo_vanilla`, `mappo`). The MJX and box2d envs already return `terminated`
  (true episode end: boundary hit / all delivered) and `truncated` (time-limit
  `t >= max_steps`) as *separate* flags, but GAE needs the
  `done = terminated | truncated` mask for **two** different jobs and they
  diverge on truncation: cutting the advantage recursion (want it on *both*, so
  returns don't bleed across the episode boundary) vs masking the value bootstrap
  (want it *only* on true termination — a time-limit cut-off should still carry
  `gamma * V(s_next)` forward, not be treated as a value-0 terminal). Fix
  (SB3-style, in the **collectors**, not the envs): at a truncated step add
  `gamma * V(s_next)` into the stored reward and keep `done` for the recursion,
  so GAE's own bootstrap term is 0 there (no double count). The catch is the
  auto-reset overwriting the successor obs, handled differently per stack:
  mappo_jax (`trainer.py:_env_step`) resets in the *same* step, so it values the
  real `next_obs` **before** the reset cond; mappo_vanilla / mappo
  (`trainer_components/rollout_collector.py`) ride gymnasium 1.x `NEXT_STEP`
  autoreset, where the truncated step's `next_obs` already *is* the true terminal
  successor (the reset obs only appears on the following step), so
  `_state_values(next_obs)` values it directly. Shared helper `_state_values`
  (also the body of `_compute_final_values`) uses `network_old` so the bootstrap
  matches the stored `values`; the `mappo` copy additionally handles the
  hypergraph critics (builds inference hypergraphs from `next_obs`) and is called
  **after** the loop's `get_last_grouping_tokens()` read, since building
  hypergraphs mutates `_last_grouping_tokens` under `learned_grouping`. Per-agent
  (difference-rewards) path in `mappo_jax`: `next_value` is the per-agent critic
  head and the truncation mask broadcasts over the agent axis. Without this,
  episodes that run to the time limit (the common case in box2d/MJX push tasks)
  systematically teach the critic that the final state is worth 0.
- **Same networks** (`network.py`, flax): 2-layer Tanh MLPs with the same
  orthogonal init, actor hidden = `model_params.hidden_dim`, critic hidden =
  `2*hidden_dim`, learned state-independent `log_action_std` (init -0.5, clamp
  [-5, 2]). Distributions are hand-rolled diagonal-Gaussian/categorical
  (no distrax; `flax`+`optax` are deps, `distrax`/`chex`/`jaxmarl` are not).
- **Same outputs**: reuses `TrainingStatsTracker`, writing
  `training_stats_{checkpoint,finished}.pkl` with the exact vanilla key set
  (plotting notebooks read them unchanged) under `results/<env>/<model>/
  <trial_id>/logs`. Params are flax msgpack (`models_{checkpoint,finished}
  .msgpack`), not torch `.pth`. **Checkpoint resume works** (`checkpoint=true`):
  the stats checkpoint restores the progress counters (vanilla flow) and
  `models/train_checkpoint.msgpack` restores the full training state — actor/
  critic params, optimizer states, step counters, and both RNG chains — saved
  at every log point *and* at finish, so re-running with a larger
  `n_total_steps` extends a finished run. (`load_from_dict` in the shared
  `TrainingStatsTracker` now also restores the agent-loss series, so resumed
  stats stay index-aligned — this fixed a latent vanilla resume flaw too.)
  `view()` renders 10 deterministic episodes via `MJXRenderer`
  (video + reward plot, like vanilla) and, when a GL context is available
  (`MUJOCO_GL=egl` headless), also saves a `MuJoCoNativeRenderer` video per
  episode (`episode_<i>_native.mp4`); `evaluate()` prints the mean eval return.
  For the `MACRO_MJX` env `view()` renders at **low-level** granularity — it
  holds each high-level skill choice fixed for `macro_len` steps but drives and
  draws the base env one physics step at a time (via `render_env.step` +
  `SyncMacroMJX._skill_actions`), so the video is smooth (1024 frames, not
  ~103); the high-level policy re-decides at each macro boundary off the base
  obs there, exactly as `SyncMacroMJX.step` does internally.
- **Per-agent rewards / difference rewards.** When the env's
  `reward_mode="difference_rewards"` (env group `multi_box_push_mjx_9a_3o_dr`),
  `run.py` sets `MAPPOConfig.per_agent_rewards=True` and the stack switches to a
  per-agent credit path; otherwise **nothing changes** (the scalar path is
  byte-for-byte the original). What the flag switches:
  `Transition.reward` `(n_envs,)` -> `(n_envs, n_agents)` and `value` likewise;
  `MAPPOCritic(n_outputs=n_agents)` grows a **per-agent value head** (one value
  per agent off the same global state — each agent now has its own return to
  predict); `compute_gae` broadcasts `done` over a trailing agent axis and runs
  the identical recursion per agent (verified: feeding per-agent rewards that are
  identical reproduces the team result exactly); the minibatch advantage stops
  being `jnp.repeat`'d from the env level and is taken per agent. Advantage
  normalization then becomes per-(env, agent) — which is exactly what vanilla
  does. `Transition.team_reward` (`info["task_reward"]`) is carried purely for
  logging so `mean_reward`, `eval_fn` and `view()` always report **team**
  performance and stay comparable to the dense baseline. Stats keys are
  unchanged, so the plotting notebooks read either arm.
- **Config**: `conf/algorithm/mappo_jax.yaml` (same params surface as
  `mappo_vanilla`), `conf/env/multi_box_push_mjx_9a_3o.yaml` (dense team reward)
  and `conf/env/multi_box_push_mjx_9a_3o_dr.yaml` (difference rewards), model
  group `mlp` (plain `hidden_dim`; `mlp_shared` carries full-MAPPO keys like
  `critic_type` that `Model_Params` rejects). Every MJX env group carries a
  literal `n_envs: 32`, so no CLI pin is needed (override only to experiment):
  ```
  uv run python train.py algorithm=mappo_jax env=multi_box_push_mjx_9a_3o \
      model=mlp trial_id=0
  # difference-rewards arm (same command, _dr env group):
  uv run python train.py algorithm=mappo_jax env=multi_box_push_mjx_9a_3o_dr \
      model=mlp trial_id=0
  # circular arena, one concentric goal ring per box:
  uv run python train.py algorithm=mappo_jax env=mjx_16a_4o_multi_goal \
      model=mlp trial_id=0
  ```
  `run.make_env(env_config)` has one `elif` per supported env group
  (`MULTI_BOX_MJX` / `MULTI_BOX_MULTI_GOAL_MJX` / `MACRO_MJX` / `SMAX`), each passing its
  constructor arguments explicitly — deliberately not a shared kwargs helper.
  It is module-level (moved out of `MAPPO_JAX_Runner.__init__` on 2026-09-25) so
  `simplified_feudal_mappo_jax` builds envs through the same code; the feudal
  stack still has its own copy in its `run.py`.
  - **Subclass hooks** (same date, behaviour-preserving): `MAPPO_JAX_Runner.
    _init_trial(...)` (results dirs + seeding) and `_make_train()` (default:
    `make_train(self.config, self.env)`, called by `train()` and `evaluate()`).
    `train()` treats the trajectory and bootstrap value opaquely, so a subclass
    can return any structure its own `update_fn` accepts.
  - `MAPPOCritic.keep_output_axis` / `create_train_state(...,
    keep_critic_output_axis=)` (default `False`) keep a width-1 per-agent value
    head's trailing axis. It only skips the squeeze, so the param tree is
    unchanged and every existing checkpoint loads.
  This means an `env:` key is only reachable where a branch names it.

- **`env.coupling_def` is wired** (all three branches — bare `MultiBoxPushMJX`,
  `MultiBoxMultiGoalPushMJX`, and the macro wrapper's base env — in **both**
  `mappo_jax/run.py` and its `feudal_mappo_jax` copy, which must stay in sync).
  It accepts `"even"` (== `n_agents // n_objects` per box, the default and what
  every existing arm uses) **or an explicit per-box list** of ints, e.g.
  `coupling_def: [2, 3, 5, 6]` at 16a/4o. Parsing lives in the single shared
  helper `box2d_suite/utils.py:resolve_coupling`, called by all three envs
  (box2d `multi_box_push` included) so the copies cannot drift; it validates
  length and range, and **warns** when the list sums to more than `n_agents`
  (no simultaneous partition exists, so boxes must be delivered sequentially —
  a difficulty change, not just a structural one). Verified: `"even"` and an
  explicit `[4,4,4,4]` are **bit-identical** (obs/reward/qpos over a fixed-seed
  100-step rollout, both MJX envs), so **no existing arm changes**; an unknown
  value raises (`ValueError: unknown coupling_def: 'bogus'`). A Hydra list
  override needs the key to exist in the env group (struct mode) — declare it in
  the group rather than passing `+env.coupling_def=...`.
  - **The `"random"` branch was removed.** It drew from a fixed
    `np.random.default_rng(42)`, so one arbitrary draw defined the arm, and at
    16a/4o it changed three things at once (couplings `[2,7,6,5]` summing to
    20 > 16, hence forced sequencing; box sizes, hence multi-goal ring geometry;
    and no per-seed resampling). An explicit list expresses any of that
    deliberately. The `--check-drift` mass-independence check [2b], its only
    consumer, now constructs with an ascending list `[2+j ...]`.
  - **Why the explicit form exists** — `conf/env/mjx_16a_4o_partition.yaml`
    (`[2, 3, 5, 6]`, summing to exactly 16). Under `even` every box needs an
    identical crew, so agents are interchangeable and there is no particular
    assignment to discover; measured, the trained baseline just sequences the
    boxes with a swarm. Unequal-but-summing-to-`n_agents` keeps a simultaneous
    partition feasible while making the assignment *specific* — the structure a
    per-agent goal mechanism (the feudal manager) is supposed to exploit. ⚠ Box
    size follows coupling (`max(1.5, coupling*0.4)`), so that arm's boxes are
    `[1.5, 1.5, 2.0, 2.4]` against the baseline's uniform 1.6: it changes the
    assignment structure **and** the geometry, and the group's docstring says so.
  - Under `even`, `coupling_fraction` (the multi-goal obs extra) is the constant
    0.25; an explicit unequal list is now what makes it informative.

## Simplified Feudal MAPPO (`algorithms/simplified_feudal_mappo_jax/`)

A deliberately small (~1170 lines with docstrings, added 2026-09-25) waypoint hierarchy.
- **Relation to `feudal_mappo_jax`:** it is not a copy. It keeps none of that
  stack's variants: latents, FiLM, LSTM, alpha mixing, permutation nulls, probes.
- **Relation to `mappo_jax`:** built on top of it, reusing its code unmodified.
- **Levels:** two PPO policies on two timescales, both trained by the
  **unmodified** `mappo_jax.mappo.ppo_update`.

```
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_12a_3o_trunc_1024 model=simplified_feudal trial_id=0
# squashed-Gaussian manager (see manager_action_bound below):
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_1a_3o_111_1024_gs model=simplified_feudal_tanh trial_id=0
# the two one-key arms on top of it (added 2026-09-27, see the 2026-09-27 block):
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_1a_3o_111_1024_gs model=simplified_feudal_tanh_relative_input trial_id=0
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_1a_3o_111_1024_gs model=simplified_feudal_tanh_manager_gamma trial_id=0
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_1a_3o_111_1024_gs model=simplified_feudal_tanh_low_manager_entropy trial_id=0
# c=16 with R rescaled to the 16-step reach (one per gh16 parent):
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_1a_3o_111_1024_gs model=simplified_feudal_tanh_gh16_scaled_radius trial_id=0
# information-matched to mappo_jax: the manager reads only its own obs (2026-09-28):
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_1a_3o_111_1024_gs model=simplified_feudal_tanh_local_input trial_id=0
# counterfactual-goal credit for the manager (2026-09-29, see its block below):
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_2a_4o_1122_1024_gs model=simplified_feudal_tanh_relative_input_cf trial_id=0
# D++ credit for the manager (2026-10-03, see its block below):
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_2a_4o_1122_1024_gs model=simplified_feudal_tanh_relative_input_dpp trial_id=0
# training forks with teleported recruits (2026-10-05, see its block below):
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_2a_4o_1122_1024_gs \
    model=simplified_feudal_tanh_relative_input_interventions trial_id=0
```

- **Manager: PPO, one decision per window of `c = goal_horizon` env steps**
  (a semi-Markov decision).
  - Actor: shared across agents. Under `manager_input: global` (the default) it
    reads `concat(M, one_hot(i))`, where
    `M = concat(global_state, every agent's goal_state)`. Under `relative` it
    reads each agent's own view measured from itself (see Config below).
  - Critic: scalar `V^M(M)`, under either input.
  - Action: a 2-D Gaussian per agent, mapped to `w_i = clip(s_i + R*clip(a_i, -1, 1), -0.5, 0.5)`,
    where `s_i = env.goal_state(state)[i]` (normalized position) and
    `R = waypoint_radius`.
  - **The manager picks direction AND distance** (up to R per axis), so it can
    learn reachable waypoints. The feudal waypoint arm used a unit direction ×
    fixed R, which no policy could reach at c=10.
  - Reward: `sum_{k<c} gamma_M^k r_{t+k}` of the TEAM reward
    (`info["task_reward"]`, so any `reward_mode` works).
  - Discount: **`gamma_M^c` per decision**, where `gamma_M` is the manager's
    per-env-step discount: `params.manager_gamma`, or the worker's `gamma` when
    that is `null` (the default, and the original behaviour). The same
    `gamma_M` sums the window reward and bootstraps a truncation;
    `make_train` raises if `manager.gamma != gamma_M ** c`.
  - Chosen over FeUdal's transition gradient (`-A^M * progress`) because PPO
    optimizes "waypoints that led to env return" directly, without assuming the
    worker follows its goals.
- **Worker: PPO every step, and its ONLY reward is following the waypoint.**
  - Actor input: `concat(obs_i, (w_i - s_i)/R, time_left)`, with the error
    recomputed live each step.
  - Reward: `(||w - s_t|| - ||w - s_{t+1}||)/R`, the distance closed. It
    telescopes over a commitment, so it cannot be farmed.
  - Episode: **each commitment is one worker episode.** `done` fires on the
    window's last step (a true terminal, since the goal expires). A time limit
    mid-window bootstraps.
  - Critic: centralized `MAPPOCritic(n_outputs=N, keep_output_axis=True)` on
    `concat(global_state, all errors, time_left)`. The per-agent reward makes
    `ppo_update` take its existing per-agent path.
- **Rollout layout: freeze-to-boundary.** `n_steps / c` windows: an outer scan
  over windows, an inner scan over `c` steps.
  - An env that finishes mid-window is **frozen**: its state is held, its worker
    steps are masked via `active_mask` with reward 0, and it is reset at the
    window boundary.
  - This keeps every manager decision at a fixed index, so both levels are plain
    `(time, env, ...)` `Transition`s and `ppo_update` is reused as-is.
  - Cost: wasted steps on **early termination only**. That is zero on `trunc`
    groups (`max_steps=1024` is 32 windows of 32). It is heavy on
    boundary-terminating groups (e.g. `mjx_16a_4o`, ~43-step episodes), which are
    a poor fit.
  - Residual: advantage normalization in `ppo_update` is not masked, so frozen
    steps enter the per-stream mean/std.
- **Reuse:**
  - From `mappo_jax`: `ppo_update` ×2 (the fork arm adds its default-off
    `masked_statistics` flag), `create_train_state`,
    `sample_action`/`evaluate_action`, `global_state_fn`/`global_state_dim`
    (so `_gs` twin groups work), `Transition` and `make_env`.
  - The runner **subclasses `MAPPO_JAX_Runner`**, so the train loop, stats,
    checkpoint cadence, resume and `evaluate()` are inherited.
  - It overrides only `_make_train`, the four-train-state msgpack I/O
    (`worker_*` / `manager_*` keys, plus `manager_adv*` under counterfactual
    credit), and `view()`.
  - `collect_fn` returns a `Rollout(worker, manager, diagnostics)` bundle that the
    inherited loop passes through untouched. `update_fn` merges the diagnostics
    into the losses dict, which is how they reach the stats pickle.
  - `trainer.make_policy` is the one forward pass (observe / decide / act) shared
    by training, eval and `view()`.
- **Config:**
  - Algorithm group: `conf/algorithm/simplified_feudal_mappo_jax.yaml`, with the
    `mappo_jax` PPO block for the worker plus `manager_lr` / `manager_n_epochs` /
    `manager_n_minibatches` / `manager_gamma` / `manager_ent_coef`. The manager
    sees ~c x fewer samples. `run.make_feudal_config(params, model_params,
    n_envs)` is the one place these are resolved into the `FeudalConfig`; it is
    pure (no env) so the seam tests check it, and every existing arm was
    verified to resolve to the same config as the runner's old inline code.
    - `manager_ent_coef` (default `null` = the shared `ent_coef`) replaces the
      entropy coefficient for the manager's PPO update only. The logged
      `manager_entropy_loss` is coefficient-free, so it stays comparable.
    - `manager_gamma` (default `null`) is a **per-env-step** discount, so one
      value means the same horizon at any `goal_horizon`. Set it in a model
      group, never on the CLI (the results path carries only the model group).
      Checkpoints are shape-identical across values, so the path is the only
      record of which discount trained a run.
  - Model group: `conf/model/simplified_feudal.yaml`, with `hidden_dim: 168`,
    `goal_horizon: 32`, `waypoint_radius: 0.15`.
    - These two keys are **required** in `Model_Params`, so `model=mlp` raises a
      `TypeError` before any results directory is created.
    - The defaults come from the measured agent speed: flat out, ~4.3 world units
      in 32 steps, ~0.145 of a 30-wide arena. Re-measured 2026-09-27 per axis
      (both 30-wide `_gs` arenas, 64 resets): 32 steps 0.145 from rest,
      ~0.175 moving; **16 steps 0.058 from rest, 0.086 moving**. Speed ramps up
      from rest (6-step time constant), so halving c more than halves the
      from-rest reach. R = 0.15 at c=32 sits at the from-rest reach; the
      `*_gh16_scaled_radius` arms apply that calibration at c=16 (R = 0.06).
      The original `*_gh16` arms kept R = 0.15, i.e. ~2.6x the reach.
    - `manager_action_bound: clip | tanh` (default `clip`, the original) sets how
      the manager's raw Gaussian action is bounded to [-1, 1] before scaling by R.
      `conf/model/simplified_feudal_tanh.yaml` is the one-key `tanh` arm. Switch it
      by model group, never by CLI override: the parameter trees are identical, so
      a checkpoint cannot tell you which bound trained it, and the model group
      name in the results path is the only record.
    - `tanh` also switches the manager's entropy bonus to the squashed-action
      entropy, via a default-off `squash=` argument on the shared
      `mappo_jax.mappo.ppo_update` / `network.evaluate_action`. The stored action
      is the pre-tanh sample, so the Jacobian cancels in the PPO ratio and the
      log-prob is unchanged. The expectation uses 16-point Gauss–Hermite
      quadrature. `squash=False` is the original code path; the 222
      `test_feudal_seams` / `test_smax_seams` / `test_mjx_global_state` tests pass.
      `manager_entropy_loss` is therefore not comparable between the two arms.
    - `manager_input: global | relative | local` (default `global`, the
      original) sets what the manager ACTOR reads. `relative`
      (`waypoints.manager_actor_input_relative`) gives each agent, in order:
      own position + velocity (4); every box as `box - own position` (2), goal
      distance, touch fraction, coupling fraction and a 1.0 (6 per box), sorted
      nearest first, with **delivered boxes zeroed and sorted last**; teammates
      as relative position + velocity (4 each), nearest first, self excluded;
      the one-hot index. Width `4 + 6*O + 4*(N-1) + N` (23 at 1a/3o, 34 at
      2a/4o, against 25 / 38 for `global` on the `_gs` groups), read off the
      builder by `jax.eval_shape` (`trainer.manager_actor_dim`). The critic
      input is unchanged. Residual: the zeroed tail still reveals how many
      boxes are delivered.
    - `relative` needs the env's `entity_state(state) -> (agents (A,4), boxes
      (O,6))` hook, the per-entity blocks that `_compact_global_state` now
      flattens (refactored 2026-09-27; verified bit-identical over a 30-step
      rollout at 1a/3o and 2a/4o). Only `MultiBoxPushMJX` has it;
      `trainer.validate_manager_input` raises elsewhere. It is an unconditional
      method, unlike `global_state`, because nothing tests for it with
      `hasattr` to size a critic.
    - **`local` is the only information-matched mode** (added 2026-09-28, arm
      `simplified_feudal_tanh_local_input`, one key from `simplified_feudal_tanh`
      and from `_relative_input`). The actor reads agent i's own observation,
      the 40-dim vector the flat `mappo_jax` actor reads, with **no one-hot**:
      the baseline shares parameters without an agent index, and `obs_i`
      already differs per agent (`waypoints.manager_actor_input_local`). Both
      critics keep their centralized inputs, which are used only in training, as
      the baseline critic's is. Width 40 at every N, so checkpoints do not load
      across modes.
      - **Why: `global` and `relative` act on state the baseline's policy never
        sees.** Both read every box and teammate at any range; the baseline sees
        that state only in its critic, during training. Measured on the trained
        `mlp` baseline's own trajectories (both `_gs` batches, 5 seeds × 64
        deterministic episodes, steps with an undelivered box left):

        | | 1a/3o | 2a/4o |
        |---|---|---|
        | undelivered boxes within `sector_sensor_radius` (7.5), on-policy | 56.7% | 53.8% |
        | steps with no box sensed, on-policy | 24.4% | 21.7% |
        | the same two at spawn | 5.6% / 84.4% | 5.9% / 81.1% |
        | teammate within range | — | 77.1% (sector centroid only) |

        The `global`/`relative` managers see 100%, plus `delivered`, touch counts,
        coupling and teammate velocities, none of which is in the observation. So
        an arm reading them beating `mlp` is not attributable to the hierarchy.
        Read `local` vs `mlp` for that; `relative` vs `local` measures what
        privileged manager information buys. The probe script is not in the repo.
      - **What the hierarchy still has, none of it teammate or global state:**
        (1) c-step memory: the latched waypoint encodes `obs_i` at the window
        start, and the worker's error `b(a) − (s_t − s_start)/R` carries its own
        displacement since then (odometry, not absolute position). Deliberately
        not controlled: it is what temporal abstraction is. (2) The arena clip in
        `waypoint_from_action` shortens the error within `R·world_width` of a
        wall. That is 4.5 units at R=0.15 in the 30-wide `_gs` arenas, inside
        `lidar_range` 7.5, whose axial rays already report the wall.
        `trainer.validate_manager_input` warns under `local` when
        `R·world_width >= lidar_range`. (3) Two actors, ~2x the baseline's actor
        parameters (capacity, not information).
      - Note the observation's `nearest_box_vec` is already identity-free and
        drops delivered boxes, i.e. the mechanism credited to `relative` below,
        but only within range.
      - Smoke-verified 2026-09-28 at 2a/4o (2e5 steps): train, resume,
        `evaluate=true` and `view()` run. The checkpoint's manager actor
        `Dense_0` is `(40, 168)`, the manager critic `(36, 336)`. **No training
        result.**
    - `Policy.decide(manager_ts, obs, gs, pos, env_state, ...)`: the actor reads
      `obs` only under `local` and `env_state` only under `relative`; all three
      callers (collect, eval, `view()`) pass both.
    - Both knobs are verified inert at their defaults: rollout fields, losses,
      post-update parameters and eval returns are **bit-identical** to the
      pre-change code on the stub env (N=1 and 3, `clip` and `tanh`).
  - `n_steps` is rounded **up** to a multiple of `c` (1048 -> 1056), with a
    printed notice.
  - A rollout needs >= 2 windows, because `ppo_update`'s unbiased std over one
    decision is NaN.
- **Scope:** envs with a `goal_state` hook and continuous actions, i.e.
  `multi_box_push_mjx` and `multi_box_multi_goal_push_mjx`, at any `n_agents`
  including 1. `validate_env` rejects SMAX (no positions) and the macro env
  (discrete). `manager_input: relative` additionally needs `entity_state`, so
  `multi_box_push_mjx` only; `local` needs no extra hook.
- **Logged series (per update):**
  - `worker_*` / `manager_*` `{policy,value,entropy,total}_loss` and
    `explained_variance`: the PPO health signals at each level.
  - `intrinsic_reward`: mean fraction of R closed per live step.
  - `waypoint_offset`: mean initial distance `||w - s_start||/R`, i.e. how far the
    manager asks agents to go.
  - `waypoint_final_error`: `||w - s_end||/R` when the waypoint expires. It equals
    `waypoint_offset` for a worker that ignores its goal, so the **gap between the
    two is the worker's goal-following**.
  - `waypoint_reached_frac`: closest approach within 0.1 R.
  - `manager_window_return`: mean manager reward per decision.
  - `manager_action_saturation`: share of manager action components whose
    bounded value is within 1% of +-1, i.e. waypoint offsets on the edge of the
    R-square. Comparable across `clip` and `tanh` (added 2026-09-26; older runs
    lack it).
  - `rollout_team_reward`: stochastic-policy team reward per step.
  - `reward`: deterministic eval return, the same series as `mappo_jax`.
- **`view()`:** 10 deterministic episodes to `logs/episode_<i>.mp4`, drawing a
  violet line and × per agent at its current waypoint, **without** the sensor
  overlay (it buried the marks). Reward plots mark the manager's decision steps.
  - Early in training the deterministic manager's mean offset is ~0, so each ×
    sits on its agent. That is correct, not a bug.
- **Measured (smoke only, 2026-09-25, ~2e5 steps):** at 1a/3o and 12a/3o the
  worker is learning to follow within ~5 updates.
  - `waypoint_final_error` fell 0.69 -> 0.49 against a flat `waypoint_offset` of
    ~0.68.
  - `worker_explained_variance` rose from -3.4 to 0.57.
  - `manager_explained_variance` is strongly negative (to -22). That is expected
    while task reward is ~0, the same reading as the feudal stack's `V^M`.
- **MEASURED 2026-09-26, `mjx_1a_3o_111_1024_gs`, 5 seeds at 1e8 steps: the
  MANAGER fails, and the interface does not.** Eval return is 29–110 (at most
  one box), against 322–330 for `mlp` and `feudal_film_zerogoal`.
  - **The interface and the worker are sufficient.** A scripted manager that
    stages below the nearest undelivered box and then pushes up, issuing
    ordinary R-limited waypoints every c=32 steps to each seed's TRAINED worker,
    delivers **3.00/3** boxes (return ~329) on every seed. It still delivers
    2.92–3.00 with the stochastic worker. Re-deciding every 8 steps adds nothing.
  - **The manager's actions saturate.** `waypoint_from_action` clips a raw
    Gaussian sample to [-1, 1]. The trained manager's mean lies far outside
    that range: median |mu| is 1–4 per axis, p90 is 2–7, and 86–99% of decisions
    have a clipped axis. Only **14–51%** of sampled actions per axis land inside
    the clip; the rest all give the same corner waypoint and carry no signal.
    The training-time symptom is `waypoint_offset` rising 0.69 → 1.07–1.29,
    toward the corner value sqrt(2) ≈ 1.41.
  - **What it learned: push up only.** When the agent is below its target box,
    waypoints go up (+0.86–0.92 R). Otherwise they do not go down: mean dy is
    ≈0 on seeds 0–2, and on seeds 3–4 only 27–36% of those waypoints go down.
    Agents spawn below every box, so the first box is delivered and the agent is
    then stuck above the rest. Of 320 episodes, one delivered 2 boxes.
  - **⚠ Saturation is NOT irreversible, so it does not explain the failure on
    its own.** An earlier version of this note said a clipped mean "cannot come
    back". A toy check with the real `ppo_update` disproves that: one 2-D action,
    manager-sized batches (33 x 32), reward `+b(a)_y` for 400 updates and then
    `-b(a)_y`. Updates after the flip until the mean turned around, at reward
    noise std 0 / 1 / 3:

    | bound | updates to turn around | mean at the flip |
    |---|---|---|
    | `clip` | 110 / 40 / 66 | 5.2 / 2.2 / 2.0 |
    | `tanh` + squashed entropy | **22 / 36 / 33** | 9.0 / 2.7 / 2.0 |
    | `tanh` + plain Gaussian entropy | 116 / 242 / 167 | 11.7 / 3.6 / 2.9 |

    Per-stream advantage normalization amplifies the few in-range samples, which
    is how `clip` recovers. The real manager had ~3000 updates. The leading
    untested cause is therefore the **delayed payoff of repositioning**: getting
    back below the next box takes several windows with no shaping reward before
    any +100.
  - The worker saturates too: median |mu| is 5–29 and sigma reaches the
    `LOG_STD_MAX` clamp (7.39) on two seeds. This is harmless for waypoint
    reaching (bang-bang force), as the scripted-manager runs show.
  - The `tanh` arm (`simplified_feudal_tanh`, above) is the fix the toy favours.
    It has since been trained; it does not fix the failure (next block). Probe
    scripts are not in the repo.
- **MEASURED 2026-09-27: every learned manager caps at ONE box, on both `_gs`
  batches, in all four arms** (`simplified_feudal`, `_tanh`, `_gh16`,
  `_tanh_gh16`; 5 seeds each, 64 deterministic episodes per seed). Mean boxes
  per episode: 0.31–0.70 of 3 at `mjx_1a_3o_111_1024_gs` and 0.44–0.98 of 4 at
  `mjx_2a_4o_1122_1024_gs`, against 2.95 / 3.87 for `mlp`. At most 3.1% of
  episodes reach 2 boxes in any arm; `mlp` reaches 2 in 99%.
  - **The worker and the interface are ruled out on every arm.** The scripted
    manager above, driving each arm's own trained worker, delivers 2.92–3.00 of
    3 and 3.03–3.97 of 4 (it sends both agents to one box, so < 4 at 2a/4o is
    the script's limit), still 3.81–3.98 with the stochastic worker.
  - **Not time:** when a first box is delivered it lands at step ~125–160
    (`mlp`: ~130), leaving ~870 steps.
  - **Cause 1, a reward desert after the first box.** In training-time
    (stochastic) rollouts a second delivery happens in 0–3.1% of episodes, and
    windows after the first delivery earn ≈0 whether the manager sends agents
    down or up (6-window discounted return ≈0). Nothing in the manager's
    gradient says "go to the next box".
  - **Cause 2, the global input lets the manager learn a separate policy there.**
    Before the first delivery its waypoints track the box direction
    (correlation +0.4–0.5); after it, ≈0. Editing the input at spawn states:
    the approach survives moving the agent to the top or marking a box
    delivered, and is lost only in the states actually visited after a
    delivery. The flat policy escapes this because `nearest_box_vec` simply
    drops a delivered box, so its first-box reflex fires on the next one.
    ⇒ `simplified_feudal_tanh_relative_input`.
  - **Cause 3, the `tanh` entropy parks the agents.** With zero advantage the
    only pull on the mean is the squashed entropy, which peaks at mean 0, i.e.
    offset 0: post-delivery offsets fall to 0.08–0.36 R (from ~1 R), agents sit
    at the top wall and are below an undelivered box < 2% of the time (`mlp`:
    20–36%). `clip` managers keep moving (~1.2 R) but not toward boxes.
  - **Cause 4, the zero-box seeds only ever push up** (pre-delivery x-tracking
    0.04–0.17), so they deliver only a box in their spawn column. At 2a/4o the
    first box needs one agent in 97–100% of episodes and the two agents never
    split up.
  - **Structural, untested: the manager's horizon equals the worker's** (γ^c
    per decision is ~100 env steps, ~3.6 windows at c=32). `mlp`'s next box is
    ~125–165 steps away, so its +100 is worth ~0.2 when repositioning should
    start and GAE credit 5 windows back is 0.16.
    ⇒ `simplified_feudal_tanh_manager_gamma` (0.997/step: 0.64 and 0.48).
  - ⚠ The `gh16` arms halve c without scaling R, so their waypoints are mostly
    unreachable (`clip` reaches 0.3–0.6% of them). Not the cap (the scripted
    manager still works with those workers), but not a clean test of c either.
  - ⇒ Cause 3: `simplified_feudal_tanh_low_manager_entropy`
    (`manager_ent_coef` 0.001, 10x below the shared 0.01). Not 0: that would
    leave nothing against a collapsing spread, and the toy check above found the
    squashed entropy is what lets a saturated `tanh` mean turn around.
  - ⇒ `gh16` confound: `simplified_feudal_gh16_scaled_radius` and
    `simplified_feudal_tanh_gh16_scaled_radius` (R = 0.06, see the reach
    measurement under Config).
  - **The new arms.** Each is one key on its parent (`simplified_feudal_tanh`,
    or the matching `*_gh16`). Smoke (2e5 steps): train, resume and evaluate
    work; at 2a/4o the checkpoint's manager actor `Dense_0` is `(34, 168)` under
    `relative` and `(38, 168)` under the discount arm, the critic `(36, 336)` in
    both.
- **MEASURED 2026-09-28: the four new arms that ran do not break the one-box
  cap.** `_gh16_scaled_radius` (clip and tanh), `_low_manager_entropy` and
  `_manager_gamma`, 5 seeds × both `_gs` batches, 1e8 steps, 64 deterministic +
  64 stochastic episodes per seed. At 1a/3o, 0 of 2560 deterministic and 0 of
  2560 stochastic episodes deliver 2 boxes; at 2a/4o, at most 1.3%. No `reward`
  eval point of any seed of any simplified arm ever exceeded 150.
  (`simplified_feudal_tanh_relative_input` arrived later; see the next block.)
  Figure: `plotting/feudal_goal_analysis/simplified_feudal_boxes_2026-09-28.png`.
  - **First-box reliability did improve.** Mean boxes, 1a/3o / 2a/4o:
    `tanh` 0.70 / 0.84, `_manager_gamma` 0.79 / **0.99 (5/5 seeds)**,
    `_tanh_gh16_scaled_radius` 0.78 / 0.92, `_low_manager_entropy` 0.36 / 0.34,
    `_gh16_scaled_radius` (clip) 0.42 / 0.38. `_manager_gamma` learns the first
    box sooner (median first eval >= 100: 27M vs 84M steps at 1a/3o, 46M vs 84M
    at 2a/4o) and `V^M` at pre-delivery states doubles (29 -> 75). `mlp` reaches
    one box at 12M and three at 14.8M (1a/3o): its later boxes cost ~1.3M steps
    each, i.e. they transfer.
  - **Discount (cause "structural horizon"): refuted as the binding cause.** A
    second delivery never occurs in training-distribution rollouts, so there is
    no reward for a longer discount to propagate. `V^M` after the first delivery
    is 0.0 in every arm (correct, on-policy).
  - **Entropy parking (cause 3): symptom, not cause.** At manager entropy 0.001,
    post-delivery offsets return to 0.82 R (from 0.14), but the waypoints point
    AWAY from the remaining boxes (cosine with the nearest-box direction -0.79),
    and fewer seeds learn even the first box.
  - **Scaled radius:** R = 0.06 at c = 16 makes waypoints reachable
    (`waypoint_reached_frac` 0.95 / 0.84), so c = 16 with reachable waypoints
    performs like c = 32. The window length is not the cap.
  - **Hand-off test (the generalization cause, measured):** the learned manager
    runs to the first delivery, a scripted manager puts the team under the next
    undelivered box, then the learned manager resumes. It adds 0.01-0.17
    boxes/episode, against 2.5-3.2 when the scripted manager keeps control. The
    same hand-off BEFORE any delivery ends in a delivery in 94-100% of episodes
    (the placed-under box itself only ~31%; the manager re-targets its preferred
    box). Per-window traces: after the hand-back the manager issues one "up"
    waypoint (+0.9 R), the agent slides off within one window, and it never
    re-acquires the box, while before a delivery it keeps re-approaching until
    delivery. So after a delivery the manager has neither the push nor the
    approach competence. This is the region problem `relative_input` targets.
  - **2a/4o has a second cap: the two-agent boxes.** The first delivered box
    needs one agent in 100% of episodes. Placed under a coupling-2 box, the
    learned manager delivers it 0% before a delivery and 0-3% after. Even a fixed
    post-delivery policy would stop at the two coupling-1 boxes.
  - Probe scripts (deterministic/stochastic rollouts, hand-off) are not in the
    repo.
- **MEASURED 2026-09-28: `simplified_feudal_tanh_relative_input` BREAKS the
  one-box cap.** 5 seeds per batch; at the time of measurement 1a/3o seeds 0-2
  were at 36-73M steps and 2a/4o seed 4 at 33M (probed from
  `models_checkpoint.msgpack`), the rest finished. Deterministic boxes per
  episode: 1a/3o 2.92 / 2.86 / 2.83 / 2.92 / **0.95** (seed 4), 2a/4o
  2.73 / 2.97 / 2.67 / 2.58 / 1.95. Sampled episodes reaching two boxes:
  83-98% (0% for every global arm). Two-agent boxes at 2a/4o are delivered in
  50-69% of episodes (0% before). Mean eval return at 30M steps: 268 at 1a/3o
  (`mlp` 245, best global arm 65) and 134 at 2a/4o (`mlp` 365, best global 17).
  At 1a/3o the second box appears 0.3-1.4M steps after the first, the same
  transfer signature as `mlp`. At 2a/4o the fourth box lands at a median step
  of ~870 of 1024, so part of the gap to `mlp` is time.
  ⚠ **The margin over `mlp` is not information-matched**: this manager reads
  every box at any range, while `mlp`'s observation senses ~55% of them (see
  `manager_input: local` above). `simplified_feudal_tanh_local_input` is the
  control.
  - **Why the global managers cap at one box (measured): each seed learns
    "fetch box k", with k a fixed input index.** Every global-input seed
    delivers one fixed box index first in 96-100% of episodes, whatever that
    box's spawn position. The nearest box at spawn is chosen at chance (25-44%
    at 1a/3o, 17-36% at 2a/4o). Once box k is delivered the rule has no target;
    the other boxes sit in input blocks the policy never learned to read. This
    is also why the earlier pre-delivery hand-off pushed the placed box only
    ~31% (≈ 1/3) of the time. The relative input has no box identity: slot 0 is
    always the nearest undelivered box, so the simplest learnable rule
    ("waypoint toward slot 0, then push up") is "nearest box". Delivering a box
    removes it from slot 0 and the same rule re-targets. The relative arm's
    first box is spread evenly over indices, and it is the nearest one in 86-97%
    of episodes at 1a/3o.
  - After a delivery: waypoint cosine with the direction to the nearest
    remaining box is +0.80-0.89 (global arms -0.79 to -0.01); the agent is
    under a remaining box 40-53% of the time (0%); `V^M` is 24-43 (0.0). The
    hand-off test after a delivery succeeds 88-100% (global 0-21%), and the
    placed-under box itself is delivered 97% at 2a/4o, 91% for two-agent boxes.
  - Two-agent boxes need no dedicated mechanism: when the agents are near each
    other their slot 0 is the same box, and both apply the same rule. Sharing a
    target is not new (global arms share one 35-82% of the time); the change is
    that the relative manager actually pushes the shared box.
  - **The residual: the blanked tail slot leaks the delivery count, and one
    seed uses it.** 1a/3o seed 4 fails exactly like the global arms (cosine
    -0.71 after a delivery with the box below the agent, +0.46 before). Input
    edits on its trained manager: moving its own position to mid-arena changes
    nothing (-0.74 -> -0.67), while refilling the blanked slot with a
    live-looking box flips it to +0.68. The four working seeds are insensitive
    to both edits. Hiding the count (e.g. only the K nearest undelivered boxes,
    or blanking random far boxes before any delivery) would close that channel.
- **MEASURED 2026-09-29: eval returns per seed** (mean of the last 10 `reward`
  evals, seeds 0-4; `*` = still a checkpoint, steps in brackets).
  - **Dense, 1a/3o:** `relative_input` matches `mlp` on 4 of 5 seeds.
    - `relative_input`: 315 / 313* (45M) / 307* (41M) / 328 / **107**. Seed 4
      is the delivery-count leak above.
    - `mlp`: 322-329.
  - **Dense, 2a/4o:** `relative_input` does **not** match `mlp`; the ranges do
    not overlap.
    - `relative_input`: 309 / 298 / 291 / 299 / 181* (33M).
    - `mlp`: 412-429.
  - **Sparse twins:** `mjx_1a_3o_111_1024_gs_sparse` and
    `mjx_2a_4o_1122_1024_gs_sparse` are the `_gs` groups with the one key
    `reward_mode: sparse`. The maximum return is 300 / 400.
  - **Sparse, 1a/3o:**
    - `relative_input`: 291 / 290 / 291 / **0** / 284.
    - `local_input`: 287 / 267 / 265 / 253 / 277. Every seed works, and it is
      information-matched to `mlp`.
    - `mlp`: 297 / 300 / 297 / **16** / 297.
    - Every other simplified arm gets at most 100, i.e. one box.
  - **Sparse, 2a/4o:** `mlp` reaches 343-390 at 73-81M steps (still running).
    `relative_input` and `local_input` have not been run there, and
    `local_input` has not been run on either dense group.
- **Counterfactual-goal credit (`manager_credit: counterfactual`, added
  2026-09-29): wired, verified, and a measured NULL on return at 2a/4o
  (2026-10-02, see the end of this block).**
  - **The problem it targets.** Under `team` credit (the default) `ppo_update`
    copies the manager's one team advantage to every agent's waypoint
    (`jnp.repeat`). That reinforces a free rider's waypoint whenever a teammate
    delivers, and under tight coupling it punishes an agent waiting at a
    two-agent box when its partner never comes.
  - **The mechanism** (`counterfactual.py`):
    ```
    A_i = A_team − β·c_i
    c_i = mean over K draws o'_i of Â_φ(x, o with only slot i replaced by o'_i)
    ```
    - `Â_φ` is a learned estimate of the team advantage from the manager
      critic's input `x` and every agent's realized waypoint offset `o`
      (`(w − s)/R`, from `waypoints.waypoint_offset`).
    - `o'_i` are counterfactual goals for agent i alone, with teammates' goals
      held fixed: K draws of its own rollout policy (`sampled`, the aristocrat
      utility) or the zero offset (`hold`, the wonderful life utility with
      "stay where you are" as the default action).
    - β is a per-batch control-variate coefficient `Cov(A_team, c)/Var(c)`,
      clipped to [0, 1] and 0 when `Var(c) < 1e-8`.
  - **Why it is safe.**
    - `c_i` never reads agent i's own goal, and the manager policy factors over
      agents. So `c_i` is a baseline: it changes the variance of agent i's
      gradient, never its expectation, however wrong `Â_φ` is (Wu et al., ICLR
      2018). COMA uses its critic for the whole advantage instead.
    - `Â_φ` is queried only on goal joints the policy could have produced.
    - `Â_φ`'s head starts at exactly 0, so the arm starts as exact team credit.
    - Both critics keep regressing the team return; only the actor's advantage
      changes.
  - **Knobs** (`Model_Params` → `FeudalConfig`, defaults = the original):
    `manager_credit: team | counterfactual`, `counterfactual_default: sampled |
    hold`, `counterfactual_samples` (K, default 16).
  - **Arms** (one key each on their parents):
    - `simplified_feudal_tanh_relative_input_cf`;
    - `..._relative_input_cf_hold` (the `hold` ablation);
    - `simplified_feudal_tanh_local_input_cf` (the information-matched twin).

    Testbed: the `_gs` groups at N = 1 (negative control), 2 and 6
    (`mjx_6a_4o_1024_gs`, couplings [4,3,3,2]). `mlp` exists on all three; the
    team-credit parent still needs its 6a/4o seeds.
  - **Wiring:**
    - `ppo_update(advantage_correction=None)` in the SHARED `mappo_jax.mappo`
      (the `squash=` precedent), and `MAPPOCritic.head_init_scale` (default
      1.0; 0.0 = zero head). Both defaults are byte-identical.
    - `HierTrainState.manager_adv` (None unless enabled) and
      `Rollout.manager_goal` (`ManagerGoal(pos, offset)`).
    - The branch `update_fn._counterfactual_credit`.
    - Checkpoints gain `manager_adv` / `manager_adv_ts` only when enabled.
      flax's `from_bytes` raises only when the TARGET has a key the file lacks,
      so default checkpoints are unchanged.
  - **Load-bearing details** (each pinned by a seam test):
    - **Normalization.** Centre per (env, agent) stream but divide by the TEAM
      advantage's per-env std. A per-(env, agent) std would rescale a free
      rider's near-zero residual back to unit variance, re-amplifying the noise
      the correction removed.
    - **Ordering.** The counterfactual goals come from the PRE-update manager
      actor, and are scored by the PRE-update `Â_φ` (fitted on earlier batches,
      so c cannot fit this batch's noise). `Â_φ` is refitted last, on the raw
      `A_team` (the same `compute_gae` call `ppo_update` makes), at the
      manager's epoch and minibatch counts.
    - **Random numbers.** The model's init and sampling keys come from
      `fold_in`, so the default path's splits are unchanged.
    - **Memory.** `correction` runs `jax.lax.map` over agents, so memory is
      O(T·E·K·N), not O(T·E·K·N²).
  - **Logged per update:**
    - `manager_cf_beta` (read first: 0 = no correction applied);
    - `manager_cf_correction_std`;
    - `manager_cf_adv_var_ratio` (`Var(A_team − βc)/Var(A_team)`, below 1 =
      variance removed);
    - `manager_cf_model_ev` (explained variance of the pre-update `Â_φ` on
      `A_team`);
    - `manager_cf_goal_sensitivity`;
    - `manager_cf_model_loss`.
  - **⚠ β is ill-conditioned while the correction is tiny.** In the smoke run
    below it jumped 0 → 0.55 → 0.09 → 0 → 0.75 → 1 → 1 → 0 with a correction
    std of ~0.007, while `adv_var_ratio` stayed at 0.97–1.0. Read β together
    with `adv_var_ratio`, not alone.
  - **Verified 2026-09-29:**
    - **Default paths.** A before/after snapshot on the CPU stub is
      bit-identical: 0.0 max difference over simplified feudal init → collect →
      update → eval (N ∈ {1, 3} × clip/tanh × all three `manager_input`s),
      `ppo_update` (scalar/per-agent × continuous/discrete) and `MAPPOCritic`
      init.
    - **Tests.** 19 new seam tests (below). 102 pass across
      `test_simplified_feudal.py` + `test_smax_seams.py` +
      `test_mjx_global_state.py`.
    - **Toy check** (team advantage = f(agent 0's goal) + g(agent 1's goal) +
      noise, equal variances). After fitting:
      - correlation with the agent's own term: 0.69–0.70 → 0.97;
      - correlation with the teammate's term: 0.69–0.70 → 0.04;
      - variance ratio 0.53–0.55;
      - β = 0.95;
      - model explained variance 0.98.
    - **MJX smoke** (`mjx_2a_4o_1122_1024_gs`, `trial_id=smoke_cf`, 8 updates):
      - train, `checkpoint=true` resume (the first five history entries kept)
        and `evaluate=true` all work;
      - the checkpoint's advantage model `Dense_0` is `(40, 336)`: 32 global
        state + 4 positions + 4 offsets;
      - a warm update takes 0.03–0.04 s against 0.6–0.9 s of collection.
    - ⚠ A numeric `trial_id` indexes `conf/seeds/standard.yaml` (30 seeds), so
      use a non-numeric id (seed 118) for smoke runs.
  - **⚠ No training result.** Acceptance:
    - `_cf` ≥ its parent at matched seeds with a paired confidence interval at
      N ≥ 2, and no difference at N = 1;
    - `manager_cf_beta` > 0 and `adv_var_ratio` < 1 once the model has signal.

    Expect little at N = 2 (one teammate to remove). In a reward desert `Â_φ`
    learns nothing and this is team credit. The fork-based audit (learned `c_i`
    vs the simulator's exact counterfactual) is deliberately not built yet.
  - **MEASURED 2026-10-02: no return change at 2a/4o, although the correction is
    active.** 5 seeds per arm at 1e8 steps, mean of the last 10 evals. At the end
    of training `manager_cf_beta` is 0.93–0.99 (`hold`: 0.27–0.79),
    `adv_var_ratio` 0.74–0.87 (`hold`: 0.94–0.97) and `model_ev` 0.13–0.27, on
    all three groups.

    | group | parent `_relative_input` | `_cf` | `_cf_hold` | `mlp` |
    |---|---|---|---|---|
    | `mjx_2a_4o_1111_1024_gs` | 399–431 | 423–435 | 375–432 | 437–438 |
    | `mjx_2a_4o_1122_1024_gs` | 281–320 | 248–320 | 175–319 | 418–429 |
    | `mjx_2a_4o_1122_1024_gs_sparse` | 116–160 | 118–163 | 123–275 | 370–387 |

    `_local_input_cf` likewise matches `_local_input`. A baseline changes the
    variance of the gradient, not its expectation, so it cannot create the
    coordinated goal pairs that tight coupling needs. The 1111-vs-1122 pair
    differs only in `coupling_def` (box sizes are identical), and coupling costs
    the hierarchy 122 (dense) / 250 (sparse) points against 16 / 20 for `mlp`.
    The follow-up design is `plans/goal_recruitment_critic_2026-10-02.md`.
- **D++ credit (`manager_credit: dpp`, added 2026-10-03): wired, verified,
  and its offline critic check FAILS. No training run.** Plan:
  `plans/goal_recruitment_critic_2026-10-02.md`.
  - **The term.** It reuses the counterfactual arm's joint-goal advantage model
    `Â_φ` unchanged (same network, training, update order and checkpoint entry).
    ```
    A_i      = A_team + dpp_coef · max(0, D++_i)
    D++_i(n) = [Â(x, o++(i, n)) − Â(x, o)] / n,   D++_i = max over n = 1..N−1
    ```
    `o++(i, n)` keeps agent i's goal and replaces its n nearest teammates' goals
    with an R-bounded waypoint toward agent i's position
    (`counterfactual.support_offsets`, built by `waypoints.waypoint_offset` so it
    is a legal manager output). The state value cancels in the difference, so
    `Â_φ` serves as the Q critic. Only critic forward passes, no simulator fork,
    and execution is unchanged. The `/n` and the search over `n` follow D++
    (Rahmattalabi, Chung, Colby and Tumer, IROS 2016); the clip mirrors D++'s
    fall-back to the difference reward. With `dpp_coef: 1`, `A_i ≈ max(Q(x, o),
    Q(x, o++)) − V(x)`.
  - **It is shaping, not a baseline.** The term reads agent i's own goal, so it
    changes the expected gradient on purpose (the true baseline of `_cf` changed
    nothing). `ppo_update`'s `advantage_correction` docstring now says so. The
    trainer passes `counterfactual.dpp_correction` = `−coef · max(0, D++)`
    because that argument is subtracted.
  - **Knobs** (`Model_Params` → `FeudalConfig`): `manager_credit: dpp`,
    `dpp_coef` (default 1.0) and `dpp_max_recruits` (default None = N − 1).
    `validate_manager_credit` rejects a negative coefficient and a zero recruit
    cap, and warns at N = 1, where the term is exactly 0.
  - **Arms**, one key each on their parents:
    `simplified_feudal_tanh_relative_input_dpp` and
    `simplified_feudal_tanh_local_input_dpp`. The checkpoint tree equals the
    `_cf` arm's (both carry `manager_adv`, `(40, 336)` at 2a/4o), so only the
    model group name records which credit trained it.
  - **Wiring.** `trainer._manager_credit` (formerly `_counterfactual_credit`)
    computes the team advantage, scores with the pre-update model, branches on
    the mode, and refits the model last. `counterfactual.GOAL_MODEL_CREDITS`
    decides when the model and `ManagerGoal` exist.
  - **Logged per update:** `manager_dpp_{mean, std, positive_frac,
    mean_best_n, adv_shift}` (`adv_shift` = std of the applied term / std of the
    team advantage), plus `manager_cf_model_ev` and `manager_cf_model_loss` for
    the critic's own fit. Read `positive_frac` and `adv_shift` before return.
  - **Verified 2026-10-03:**
    - `team` and `counterfactual` are bit-identical to the pre-change code: 0.0
      maximum difference over 36 CPU-stub configurations (N ∈ {1, 3}, both
      bounds, all three manager inputs, both counterfactual defaults), covering
      the rollout, two updates, every logged loss and eval.
    - 14 new seam tests; 116 pass across `test_simplified_feudal.py` +
      `test_smax_seams.py` + `test_mjx_global_state.py`.
    - Smoke on `mjx_2a_4o_1122_1024_gs`, both arms: train to 2e5, `checkpoint=true`
      resume to 3e5 (all 8 history entries kept) and `evaluate=true` work. The
      term is exactly 0 on the first update (zero-initialized head) and its
      `positive_frac` is 0.2–0.6 afterwards. One local-arm resume attempt exited
      1 with its log lost; an identical rerun succeeded.
  - **⚠ The offline critic check FAILS (`dpp_probe.py`, 2026-10-03).** It loads
    each finished `_cf` checkpoint's trained `Â_φ`, rolls out that trial's
    manager (stochastic, 64 episodes), and compares the clipped term when the
    agent waits alone (touching an undelivered box whose coupling ≥ 2 is unmet)
    with all other decisions. Intervals bootstrap over episodes.

    | 1122 group | relative `_cf` (alone − other) | local `_cf` |
    |---|---|---|
    | dense | −0.70 / −0.80 / +0.05 / −0.91 / −1.04 | −0.30 / −0.47 / −0.49 / −0.29 / −0.56 |
    | sparse | −0.13 / −0.96 / −1.02 / −0.14 / −2.06 | −0.34 / −0.39 / −0.26 / −0.33 / −0.34 |

    Seeds 0–4 per cell. 0 of 20 intervals lie above zero; 19 lie below it. The
    existing critics rate sending a teammate toward a waiting agent LOWER than
    at other decisions. The control mirrors the same move away from the agent:
    when an agent waits alone, "toward minus away" is negative on 17 of 20
    (16 intervals below zero). The exceptions are relative dense seed 1
    (interval includes zero) and relative sparse seeds 0 and 3, which have the
    fewest waiting cases (151 and 72). So the critics do not merely penalize
    leaving the policy's own goals; they actively prefer the teammate not to
    come.
    - **Unresolved, and it decides what to fix.** The probe cannot tell whether
      the critic is wrong or right. It may be right that a ONE-window detour
      toward a waiting agent costs the teammate its own task while the manager
      abandons the coalition at the next boundary. Heading for the agent's
      position may also approach the box from the wrong side. Or the critic
      may encode the parent's own failure, since it is trained on on-policy
      advantages from policies that rarely complete coalitions. Separating these
      needs a simulator audit of the same counterfactual, or another recruit
      target.
    - **Do not launch the full `_dpp` runs on the strength of this mechanism**
      until the check passes. The plan's rule is to fix the critic first.
    ```
    uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_probe \
        --batch mjx_2a_4o_1122_1024_gs \
        --model simplified_feudal_tanh_relative_input_cf --trials 0,1,2,3,4
    ```
    MJX is not reproducible across processes, so a rerun moves the numbers
    slightly (seed 0 dense relative read −0.77 in a first run); the signs held.
- **Training forks (`interventions: true`, added 2026-10-05): wired and
  smoke-tested. No training result.** Plan:
  `plans/simplified_feudal_interventions_2026-10-05.md`; code:
  `interventions.py` plus branches in `trainer.py`.
  - **The mechanism.** At every `intervention_interval`-th manager decision
    (default 1), each env is forked once per agent i, before the window's first
    worker step:
    - N ~ U{1..n_agents−1} teammates, drawn without replacement, are teleported
      within `intervention_radius` (default 1.5 world units, centre to centre)
      of agent i.
    - Each recruit gets agent i's realized waypoint offset `w_i − s_i` (after
      the bound and arena clip), measured from its own new position. Agent i
      and the other agents keep their positions and waypoints.
    - The fork runs one window under the same frozen policies and is then
      discarded. Its worker steps end with `done` at the window end. Its one
      manager transition is a terminal whose return is the discounted team
      reward of its live steps. Neither level adds a time-limit bootstrap
      (`_worker_step(bootstrap=False)`).
    - Execution, `eval_fn` and `view()` are unchanged. `episode_count` and
      `rollout_team_reward` count the main rollout only.
  - **`n_total_steps` caps SIMULATOR steps (changed 2026-10-05, at the author's
    request).** One update charges its main `n_steps * n_envs` plus every fork
    lane on every intervention window for `goal_horizon` steps. Frozen and
    failed lanes count because they are stepped; skipped windows do not
    (`iv.simulator_steps_per_update`).
    - `make_train`'s `num_updates` and the runner's `_steps_per_update` (a new
      default-identity hook on `MAPPO_JAX_Runner`, i.e. the logged `total_steps`
      and the resume index) both call that function, so they cannot disagree.
    - At 2a/4o, interval 1, an update is 101,376 steps (33,792 main), so a 1e8
      budget gives **986 updates against the parent's 2959**.
    - Curves of this arm against the parent at equal `total_steps` are
      therefore equal-SIMULATOR-step comparisons. For equal main steps, divide
      this arm's `total_steps` by `1 + A * ceil(W/k) / W`.
  - **Who is trained on what.**
    - Manager: the fork's record is the main decision itself (stored actor
      input, action and log-prob, so the PPO ratio is 1). Only agent i carries
      actor weight (`active_mask` = one-hot(i) × valid), because the recruits'
      goals were imposed.
    - Worker: every live agent's fork action is a real sample.
    - The manager critic reads a context block appended to its input,
      `[is_intervention, one_hot(i), N/(n_agents−1)]`, all zeros on main rows,
      so it can tell a continuing return from a one-window fork return. N is
      drawn independently of the action, so the baseline does not depend on it.
      The critic is `n_agents + 2` inputs wider: `(40, 336)` at 2a/4o against 36
      for the parent, so checkpoints do not load across.
  - **Layout: forks are extra env COLUMNS** (`types.Intervention`). Fork (w, e,
    i) is column `e·A + i` in the rows of its source window, so the fork buffers
    keep the main buffers' time length. `update_fn` appends them along the env
    axis (`iv.append_columns`) and calls `ppo_update` once per level. Every fork
    ends in a terminal, so GAE never crosses between forks or into main (pinned
    column for column). Windows skipped by the interval hold fully masked
    placeholder rows; the fork branch sits under `lax.cond`, so they run no
    physics.
  - **Shared change: `ppo_update(masked_statistics=False)`** in
    `mappo_jax/mappo.py`. When on, `active_mask` also governs the per-stream
    advantage normalization (a stream with fewer than two active rows is left
    unnormalized), the explained variance and the scalar critic loss. The
    variant turns it on for both levels. Without it, interval-k placeholder rows
    would shrink each fork column's std by ~√k, and failed placements would
    train the scalar critic. It also excludes frozen MAIN rows from the
    statistics, a small difference from the parent: on `trunc` groups a main row
    freezes only after a mid-window all-delivered termination. Helpers
    `masked_mean_std` / `masked_normalize` / `masked_explained_variance`. Off is
    byte-identical.
  - **Env hook `MultiBoxPushMJX.teleport_agents(state, focal, recruit_mask,
    offset, key, radius) -> (obs, state, valid)`**:
    - Recruits go in the annulus between 0.85 and `radius` around agent i, in
      index order, each at the first valid one of 32 candidates.
    - A candidate must clear the walls, every box surface and every agent that
      stays put or is already placed, each by a 0.05 margin.
    - It must also keep its translated waypoint inside `[-0.5, 0.5]`.
    - If any recruit fails, `valid` is False and the state is returned
      unchanged. The radius is never enlarged and N is never reduced.
    - Only recruit `qpos` changes. Velocities, box poses, `t`, `delivered` and
      `prev_box_goal_dist` are kept, so teleporting pays nothing.
    - One `mjx.forward` under `_model_for(_coupling_met(.))` refreshes contacts
      and the contact-force observation, and the warm start is set from it.
    - The rotated-box distance is shared with `touch_matrix` through
      `observation.box_surface_distance` (refactor verified bit-identical).
    - No other env has the hook; the validator raises for them.
  - **Validation** (`iv.validate_interventions`):
    - `manager_credit: team` only;
    - `n_agents >= 2`, radius > 0, the env hook present;
    - integer interval >= 1 with `ceil(n_windows/interval) >= 2`;
    - an interval other than 1 without `interventions` raises, rather than
      silently training the parent.
  - **Arm** `simplified_feudal_tanh_relative_input_interventions`: one key on
    its parent, and it declares the radius and interval defaults. Interval and
    radius leave checkpoints shape-identical, so set them in a child model
    group, never on the CLI.
  - **Extra-data control** `conf/env/mjx_2a_4o_1122_1024_gs_n96.yaml` (96 envs,
    same `n_total_steps`), run under the PARENT model. At the same budget it
    matches interval 1 on three counts: updates (986 at 1e8), worker rows per
    update and simulator steps. For interval k use about
    `32·(1 + 2·ceil(33/k)/33)` envs.
  - **Logged per update:**
    - `intervention_attempted`, `_valid_frac`, and both per N (`_n{N}`);
    - `intervention_recruit_distance` (world units), `_fork_length`,
      `_window_return`, `_worker_progress`;
    - `intervention_sim_steps` (live) and `_stepped_lanes`;
    - `{worker,manager}_eligible_{main,fork}`;
    - `{worker,manager}_{adv_std,explained_variance}_{main,fork}`, from
      separate per-source GAE.

    Read placement success per N first: the geometric filter can reshape the
    realized N distribution.
  - **Verified 2026-10-05:**
    - The default path is bit-identical to the pre-change code: 0.0 maximum
      difference over 6,498 arrays. That covers 36 CPU-stub configurations
      (N ∈ {1, 3} × both bounds × three manager inputs × three credits; rollout,
      two updates, losses, parameters, eval) and four direct `ppo_update` cases.
    - With forks on, the main rollout is bit-identical to the parent's at the
      same seed, at intervals 1 and 2, including when every placement fails.
    - 45 seam tests in `algorithms/tests/test_simplified_feudal_interventions.py`,
      including the budget arithmetic and its agreement with the runner hook.
    - MJX placement on random-action states, 150 steps in: 93% valid at 2a/4o;
      83% at 6a/4o (N=1..5: 92 / 88 / 83 / 75 / 77%).
    - Smoke on `mjx_2a_4o_1122_1024_gs`: train, `checkpoint=true` resume and
      `evaluate=true` all work, and every placement was valid. Interval 2 forks
      at 17 of 33 windows.
    - Under the simulator-step budget, 3e5 gives 2 updates of 101,376 steps.
      The resume to 5e5 picked up at update 2, and every `total_steps`
      increment equals 33,792 main steps plus the logged
      `intervention_stepped_lanes`.
    - **Measured cost:** warm collection 1.5–2.1 s per update at interval 1,
      1.0–1.4 s at interval 2, against 0.6–0.8 s for the parent. The update
      takes 0.06 against 0.03 s. Physics alone at 3× lanes in one batch costs
      only ~1.06×, so the gap most likely comes from the fork scan running
      after the main scan in every window (not separately measured). Merging
      both into one lane batch is the optimization if it matters.
  - ⚠ **On sparse groups a one-window fork carries almost no task reward.** A
    box at spawn height needs at least ~40 steps of pushing to reach the band.
    Pilot on dense groups.
- **Checks:** `uv run pytest algorithms/tests/test_simplified_feudal.py -q` runs
  90 tests on a CPU stub env, each at N=1 and N=3 where relevant. They cover:
  - waypoints are fixed within a window and redrawn between windows;
  - the worker reward telescopes;
  - freeze + reset at the boundary;
  - both truncation bootstraps are exact;
  - the PPO ratio is exactly 1 at both levels;
  - end-to-end collect/update/eval;
  - the ratio and end-to-end tests under both `manager_action_bound` values;
  - `tanh` moves the waypoint where `clip` is flat;
  - the squashed entropy matches Monte Carlo, pulls a saturated mean toward 0,
    and leaves the log-prob unchanged;
  - an unknown bound is rejected;
  - `keep_output_axis` is inert by default;
  - `model=mlp` is rejected;
  - the ratio and end-to-end tests under all three `manager_input` values;
  - the local input: the stored manager input is the observation the worker
    read at the window start, and agent 0's decision is unchanged by editing
    its teammates or its absolute position and changed by editing its own
    observation (`global` is the positive control; in the stub the observation
    contains the position, so this pins the plumbing, not partial
    observability);
  - the relative input: nearest undelivered box first, delivered boxes zeroed,
    invariant to box order and to where a delivered box is, teammates nearest
    first without self, and the stored manager input equals the relative view
    of the true state (a box delivered mid-rollout drops out);
  - `relative` without `entity_state`, or an unknown mode, is rejected; `local`
    builds without it;
  - `manager_step_gamma` is used for the window return and the truncation
    bootstrap while the worker keeps its own gamma, and a `manager.gamma` that
    disagrees with it is rejected;
  - `make_feudal_config` wires `manager_gamma` / `manager_ent_coef` to the
    manager only, and a different manager entropy coefficient leaves the
    worker's update bit-identical while changing the manager's;
  - counterfactual credit (19 tests):
    - it is off by default;
    - a zero `advantage_correction` reproduces the team update, and a
      correction on per-agent rewards is rejected;
    - the counterfactual arm starts as exact team credit: the same rollout, a
      zero model output, β = 0, a bit-identical worker update and a manager
      update within 1e-5;
    - agent i's correction is bit-identical when its own goal is edited and
      moves for its teammates (the unbiasedness precondition);
    - `own_slot_joint` changes only slot i;
    - `sampled` draws come from the given actor, and `hold` is zero;
    - `fit_beta` scales, clips and handles a constant correction;
    - the toy mechanism check;
    - end-to-end collect/update/eval under both defaults;
    - config validation and the one-agent warning;
    - `make_feudal_config` wiring;
    - checkpoint trees carry the model only when it is enabled.
  - D++ credit (14 tests):
    - support offsets step toward the focal agent, within R per axis and the
      arena;
    - `dpp_joint` keeps the focal goal and changes exactly the n nearest
      teammates;
    - a hand-built indicator critic pins the `/n` division, the maximum over n,
      `dpp_max_recruits`, and the clip of a harmful recruitment to 0;
    - the correction's sign, both directly and through the real `ppo_update`
      (crediting an agent's goals raises their log-probability against debiting
      them);
    - the term is 0 at N = 1 and with a zero-initialized critic, and the arm
      starts as exact team credit (bit-identical worker update, manager within
      1e-5);
    - editing agent i's own goal moves D++_i, the shaping property that
      separates it from `counterfactual`;
    - end-to-end collect/update/eval at N = 1 and 3, config validation and the
      one-agent warning, `make_feudal_config` wiring, and the checkpoint carrying
      `manager_adv`.

## Feudal MAPPO (`algorithms/feudal_mappo_jax/`)

A FeUdal-Networks (Vezhnevets et al. 2017) hierarchy on top of a **copy** of
`mappo_jax`: a centralized manager emits one unit-norm latent goal per agent, a
goal-conditioned worker emits the primitive action. **Wired end-to-end and
runnable** (`algorithm=feudal_mappo_jax`); the manager's transition policy
gradient trains, the intrinsic-reward path is implemented but ships **off**
(`intrinsic_coef: 0.0`).

```
uv run python train.py algorithm=feudal_mappo_jax env=mjx_16a_4o \
    model=feudal trial_id=0
```

- **Model-group hierarchy: every arm is `defaults: [<nearest ancestor>, _self_]`
  plus the ONE key that makes it distinct** (normalized 2026-09-22). The forks,
  outermost first, are fusion (`feudal` / `feudal_film`) → alpha (`_n001` /
  `_n01` / `_n05`) → latent (`_local`, `_local_private`, then `_local_global*`
  chained off those) → the extra axis (`_shared`, `_dilated`). So
  `feudal_film_n01_local_global_private` resolves through
  `feudal_film_n01_local_private` → `feudal_film_n01` → `feudal_film`, and the
  file itself contains exactly `manager_latent: local_global_private`.
  - **`feudal_n01_c50` / `feudal_n05_c50`** (added 2026-09-23) are the concat
    alpha arms with `params.goal_horizon` 10 → 50 (cosine lag, pooling window
    and r^I window all move together). Only meaningful on long-episode groups
    (`valid_fraction` measured 0.951 at `mjx_16a_4o_trunc_1024`; on a
    wall-terminating ~43-step group almost no horizon is valid). ⚠ r^I's
    `_intrinsic_window` vmaps over all c offsets, so **GPU memory scales
    linearly in c**: measured peak 4.7 GB at c=10 vs 10.8 GB at c=50 on 16a/4o
    (incl. ~1.2 GB desktop baseline). No training result yet.
  - **Why it matters, not style:** the flat files hand-copied `hidden_dim: 168`
    and `worker_fusion: film` into 17 places, so a change to the base arm reached
    the alpha arms only if you remembered all 17 — the same silent-divergence
    shape as the `# @package _global_` omission that cost 12 runs. Verified
    value-neutral: resolving all 43 feudal groups before and after the
    restructure gives **0 differences** across every `params` / `model_params`
    key, so no checkpoint or result is invalidated.
  - ⚠ **A duplicate top-level key does NOT merge — OmegaConf rejects the file**
    (`ConstructorError: found duplicate key model_params`), and it dies during
    Hydra composition, before any banner names the arm you launched. Two
    `model_params:` blocks in one file is the way this is written by accident;
    `uv run python -c "from omegaconf import OmegaConf; OmegaConf.load(p)"` over
    `conf/**/*.yaml` is the lint (PyYAML's `SafeLoader` is permissive and will
    NOT catch it).
  - **`feudal_n05_shared` was renamed to `feudal_n05_local_shared`** and given
    `manager_latent: local`. Under its old name it set `worker_encoder: shared`
    while inheriting the default `centralized` latent, which
    `validate_worker_encoder` rejects — the arm could not run at all, and it had
    no results on disk. It is now the exact alpha=0.5 twin of
    `feudal_n01_local_shared`. All 43 groups now pass both
    `validate_worker_encoder` and `validate_worker_objective`; the only arms that
    warn are the four concat+shared-encoder ones and `feudal_intrinsic`, both
    deliberate contrasts.
  - ⚠ **`feudal_film_dilated` WAS INERT AND ITS RESULTS ARE STALE (found
    2026-09-22).** Commit `9c0c1b3` created it by copying `feudal_film.yaml`
    wholesale and never added `manager_core: dilated_lstm`, so the arm was a
    byte-for-byte duplicate of `feudal_film` under a name claiming recurrence —
    the same silent-inert shape as the `# @package _global_` omission, and again
    nothing warned. Every sibling (`feudal_film_n0*_dilated`,
    `feudal_film_zerogoal_dilated`) set the key correctly, so it was the lone
    miss. **Diagnose these from the checkpoint, not the yaml**:
    `manager/params/core` is a single `Dense(384, 256)` for the mlp core and
    `LSTMCell_0/{ii,if,ig,io,hi,hf,hg,ho}` gates for `dilated_lstm`. The 6
    affected trials (`mjx_12a_3o_trunc_1024` + `mjx_12a_3o_partition_1024`,
    seeds 0-2) are valid `feudal_film` runs and worthless as recurrence
    evidence; the key is now set, so a `checkpoint=true` resume fails loudly on
    the core shape mismatch instead of continuing. **Re-run them.**
  - ⚠ **`intrinsic_anneal` is `none`, not `linear`,** for every `_n0*` arm
    (changed in `eddd654`). The `feudal_n01`/`feudal_n05` headers documented the
    anneal for a while after that and were simply wrong; they now say so. Alpha
    is held for the whole run, so these arms measure a **sustained** `r^I`, not
    an early-exploration bonus — `alpha_current` is flat in the stats.

- **⚠ Always pair with `model=feudal`, never `model=mlp`.** `train.py` builds
  `experiments/results/<env>/<model>/` and the **algorithm is not in that path**,
  so `model=mlp` writes into the `mappo_jax` baseline's directory — and a
  `checkpoint=true` resume would try to load a 2-state mappo_jax checkpoint into
  the 4-state feudal train state. `conf/model/feudal.yaml` exists as much for
  that separation as for its knobs.
- **The flat baseline is `algorithm=mappo_jax`, not `intrinsic_coef=0`.** The
  feudal worker critic is **always per-agent** (the intrinsic reward is
  inherently `(T,E,N)`), so even with alpha=0 the arm differs from flat MAPPO by
  the critic head width and the goal conditioning.
- **Registration** is the usual three points: `AlgorithmEnum.FEUDAL_MAPPO_JAX`,
  a `_dispatch` case mirroring `MAPPO_JAX`, and
  `conf/algorithm/feudal_mappo_jax.yaml`. The runner is
  `Feudal_MAPPO_JAX_Runner`; `run.py` was re-synced with the
  `MULTI_BOX_MULTI_GOAL_MJX` branch it had drifted behind (commit 617cb62), so
  its only diffs vs `algorithms/mappo_jax/run.py` are imports, the class name and
  one error string. **`algorithms/mappo_jax/` is untouched.**
- **Equivalence cannot be checked by comparing training stats.** MJX rollouts are
  not reproducible across processes (CLAUDE.md documents this above), and it
  shows at the top level: two runs of *unmodified* `mappo_jax` at the same seed
  gave 953 vs 897 episodes. Verify structural equivalence instead by
  constructing both packages' train states **in one process** and diffing the
  parameter leaves (max abs diff 0.0 across 13 leaves).
- **`worker.py` — `FeudalWorker`** (done). Goal-conditioned low-level policy:
  `__call__(obs, goal)` concatenates the goal onto the obs and runs the flat
  `MAPPOActor` body **reused verbatim** (same 2-layer Tanh MLP, orthogonal init,
  and return contract), so `sample_action`/`evaluate_action` work on it
  unmodified via the `bind_goal(apply_fn, goal)` closure — no forked sampling
  path. The goal broadcasts over the obs's leading axes, so a team goal
  `(n_envs, goal_dim)` and a per-agent goal `(n_envs, n_agents, goal_dim)` both
  pair with `(n_envs, n_agents, obs_dim)`. Optional `goal_embed_dim` puts the
  goal through a **bias-free** Dense first (FuN's `phi`). `init_worker(...)`
  returns `(module, params)`. Verified: shapes/log-prob agreement between sample
  and evaluate, goal-sensitivity, jit + vmap, both action-space types.
  - **Design caveat to revisit:** with concat fusion the worker *can* learn to
    ignore the goal (zero the goal columns of layer 1) — the degeneracy FuN
    avoids with a bilinear `logits = U(obs) @ phi(g)` and no bias, so a zero
    goal expresses no preference. If the manager's goals turn out not to steer
    the worker, swap the fusion; the module interface stays the same.
  - **⚠ `normalize_pooled_goal` (default `True`, `Model_Params` /
    `MAPPOConfig`) — a MEASURED defect in the concat fusion, fixed 2026-08-28.**
    The worker eats FuN's `w_t = sum_{i=t-c+1..t} g_i`, a sum of `c` unit goals.
    Measured on trained `mjx_16a_4o` / `_trunc` checkpoints, **consecutive goals
    are 0.95-0.99 collinear**, so that sum pools nothing — it is a near-exact
    `c`x multiplier on one slowly-varying direction (`||w_t||` = 9.95 of a max
    10 at `c=10`). Concatenated raw it entered the first Dense at **~5x** the
    obs's per-dim RMS (2.49 vs 0.48), so 16 goal dims took **91.6%** of the
    layer's preactivation variance from 40 obs dims — into a Tanh. Decomposed on
    the trained checkpoints, the goal block still held **59-78%** of layer-1
    variance at the end of training. `||w_t||` also ramps `1 -> c` over
    the first `c` steps of **every** episode, so the scale is non-stationary in
    episode phase. FuN never hits this: its bilinear fusion makes `||w_t||` a
    pure logit rescale that never competes with the obs inside a saturating
    nonlinearity. The fix L2-normalizes `w_t` in `FeudalWorker.__call__` (before
    `goal_embed_dim`), discarding **only the magnitude** — the direction is the
    whole content of a FuN goal and the only thing the scale-invariant `d_cos`
    objective scores, so it costs the mechanism nothing. Verified:
    `normalize_pooled_goal=False` is **bit-identical** to the pre-fix worker on
    the same params (so old behaviour is exactly reproducible), `True` drops the
    goal's variance share 90.7% -> 8.8% and makes the output invariant to
    `||g||`; 21 seam tests + all 9 manager self-checks + a smoke train pass.
    **Every `feudal_*` result before this date used the raw sum.**
    - ⚠ **`worker_goal_column_ratio` is easy to misread, in two ways.** (a) It
      is a ratio of *weights*, blind to the scale of the inputs they multiply,
      so it is not like-for-like across `normalize_pooled_goal`. (b) **A falling
      ratio does not mean the goal columns shrank** — it has a denominator.
      Measured on the trained `feudal_*` checkpoints the ratio fell to 0.36-0.65
      while the goal block **GREW 2.2-4.7x** from init; the obs block just grew
      6-10x, faster. So the worker was *not* disconnecting from the manager, and
      an earlier reading of that decay as goal-blindness was wrong. The
      `zero_goal` arm demonstrates it: with the goal columns provably frozen the
      ratio still drifts 1.005 -> 0.813. To claim goal-blindness compare
      `goal_rms` to its shape-determined init (**0.109109** at `hidden_dim=168`,
      identical across seeds under `orthogonal(sqrt(2))`), not this ratio.
- **`conf/model/feudal_zerogoal.yaml` — the ISOLATE ARM** (`model_params.
  zero_goal`, default `False`). `FeudalWorker` receives `jnp.zeros_like(goal)`,
  so the policy is provably independent of the manager while the manager net,
  its PG, its critic, its diagnostics and the worker's per-agent critic head all
  still train. **It exists because `feudal_a0` is not an isolate of "hierarchy
  vs flat"** — it differs from `algorithm=mappo_jax` in three ways at once:
  goal conditioning; an **unconditionally** `n_agents`-wide worker critic head
  (`trainer.py`, because `r^I` is inherently per-agent — under a dense team
  scalar that is N heads regressing N identical targets); and manager training.
  The measured ~250-point gap vs `mlp` is therefore unattributable, and the
  three have opposite implications (nuisance / implementation bug / actual
  result). The ladder is `mappo_jax+mlp` < `feudal_zerogoal` < `feudal_a0` <
  `feudal_n01|n05`, each rung adding one thing. Read it as: `zerogoal ~ a0`,
  both << `mlp` ⇒ the goals are not the cause (bug hunt); `zerogoal ~ mlp`,
  `a0` << `mlp` ⇒ goal conditioning is what hurts (the reportable result).
  - Zeroing is at the **input**, so the param tree is shape-identical and
    checkpoints stay interchangeable; the goal columns just get zero gradient.
  - `zero_goal=True` with `intrinsic_coef != 0` **raises** in `run.py` — the
    worker cannot see the goals it would be rewarded for reaching, and that arm
    would still train and log healthy numbers, which is how an uninterpretable
    run gets mistaken for a result.
  - Pinned by 4 seam tests (policy invariant to arbitrary goals for both
    fusions, off by default, param tree unchanged, manager still moves + goal
    columns frozen). Launch:
    ```
    uv run python train.py algorithm=feudal_mappo_jax env=mjx_16a_4o \
        model=feudal_zerogoal trial_id=0
    ```
- **`manager.py` — `FeudalManager`** (done, network only; nothing calls it).
  Centralized manager emitting **one unit-norm goal per agent**. `s` (the latent
  state space) and `g` share a space — in FuN a goal is a *direction in the state
  embedding*, not a separate code — so `goal_dim` is the only width knob:
  ```
  global_state (E, N*obs_dim)      # trainer.py: obs.reshape(n_envs, -1)
    -> f_percept 2-layer Tanh MLP   -> z    (E, hidden_dim)   team embedding
    -> f_Mspace  Dense(N*goal_dim)  -> s    (E, N, goal_dim)
    -> f_Mrnn (dilated LSTM | MLP)  -> y    (E, hidden_dim)   consumes s
    -> goal head Dense(N*goal_dim)  -> ghat (E, N, goal_dim)
                   per-agent L2     -> g    (E, N, goal_dim)
  ```
  **`s` is a bottleneck, not a side head** — the core consumes `s`, matching the
  paper's `h^M_t, ghat_t = f^Mrnn(s_t, h^M_{t-1})`. This is load-bearing, not
  cosmetic: see the detach rule below. It also means `goal_dim` is the manager's
  entire information channel, so shrinking it throttles the goal RNN too.
  Layers are explicitly named (`f_percept_0/1`, `f_Mspace`, `core`, `goal_head`)
  so gradient-routing assertions don't depend on flax's `Dense_N` ordering.
  `__call__(carry, global_state) -> (carry, goal, s)`; `init_manager(...)` returns
  `(module, params, carry)`. Goals are per-agent (not per-team) so the manager can
  assign a **division of labour**; `g` drops straight into `FeudalWorker`
  (`_broadcast_goal` is a no-op on an already-per-agent goal). Recurrence is
  **team-level** — one LSTM over `z`, expanded to agents only at the goal head —
  so the carry has no agent axis.
  - **`DilatedLSTM`** (FuN §3): state is a pool of `radius` sub-states
    `(..., r, features)`; at step `t` only group `t % r` is read and written, so a
    group's gradient path spans `r`× more real time. Gate params are **shared
    across groups** (one `nn.LSTMCell`, reused for the gate math — the repo's only
    other recurrence is torch). Group selection is a one-hot gather + masked
    write-back, **not** a dynamic index, so `t` may be a tracer and the module is
    `jit`/`scan`-safe. `r` defaults to the goal horizon `c`. Carry is
    `DilatedLSTMState(cell=(c_pool, h_pool), t)`, built by the module-level
    `dilated_lstm_carry(...)` — *not* inside `initialize_carry`, because flax wraps
    every public module method in its scope machinery and an unbound module cannot
    construct submodules there (this raises a bare `AssertionError` if reintroduced).
  - **`core="mlp"`** is the stateless alternative: carry is `None` and passes
    through, same signature and output shapes. It exists because **nothing in the
    JAX stack is recurrent** — `trainer.py`'s rollout scan carry is
    `(train_state, env_state, obs, rng)` — so the MLP core can be wired in without
    touching the scan, and the dilated LSTM added after.
  - **Deliberate deviations from the paper**, both documented in-file: `f_Mspace`
    uses the repo's 2-layer Tanh body rather than FuN's `Dense + ReLU` (bounded
    latents are better conditioned for a cosine objective); and the goal head uses
    `orthogonal(1.0)`, not the actor head's `0.01` — a near-zero `ghat` normalizes
    to a direction set entirely by init noise. FuN also shares one `f_percept`
    between manager and worker; here the worker eats the raw local obs, so
    `f_Mspace` is manager-only.
  - **Goal-semantics helpers** (pure, jittable, static shapes, all in the same
    module): `pool_goals(goals, c)` = `sum_{i=t-c+1..t} g_i`, what the worker is
    actually conditioned on so directives persist over the horizon;
    `transition_cosine(states, goals, c) -> (cos, valid)` = `d_cos(s_{t+c}-s_t,
    g_t)`, the manager's transition policy gradient objective (multiply by the
    manager advantage); `worker_intrinsic_reward(states, goals, c)` = `1/c *
    sum_{i=1..c} d_cos(s_t-s_{t-i}, g_{t-i})`. All three take an **optional `done`
    mask** that severs any pair straddling an episode boundary (a `cumsum`
    comparison) — FuN's env never terminated, ours do, and omitting it silently
    mixes latents from different episodes. `_unit` is zero-safe (`+ eps`), the same
    convention as `environments/mjx_suite/macro_skills.py:_unit`.
  - **Latent collapse and the detach rule (the reason for the topology).**
    `d_cos` is scale-invariant, so the risk is **directional**, not about the
    magnitude of `s` (an earlier version of this note said "shrinking `s`" — that
    is a no-op). If `f_Mspace` collapses to rank 1 (`s_t = phi(x_t) * u`), every
    `s_t - s_{t-i}` is parallel to `u`, the goal head emits `g = u`, and the
    cosine pins at ±1 for **every** state and action: the intrinsic reward becomes
    a constant, which advantage centering annihilates. The mechanism goes inert
    while the losses look healthy — self-sealing, since the metric that would
    expose it is the one that collapsed (same trap as the `boundary_truncates`
    bug above). The manager is pushed there because it owns *both* arguments of
    the cosine: it picks the measuring stick (`s`) *and* the target (`g`), and
    rotating the yardstick is far cheaper than learning what the worker can
    achieve. The guard is FuN's own and is **explicit in the paper**: *"the
    dependence of `s` on θ is ignored when computing ∇_θ d_cos — this avoids
    trivial solutions."* Implemented as `transition_cosine(..., detach_states=True)`
    (default; `False` reproduces the failure deliberately) plus an unconditional
    detach in `worker_intrinsic_reward` — a reward is data, and leaving it
    attached would also backprop the worker's objective into the manager, which
    FuN rules out because it *"would deprive Manager's goals `g` of any semantic
    meaning, making them just internal latent variables."*
    **Why the core must consume `s`:** the detach hits only the *target* arm;
    `g_t(θ)` still depends on θ **through `s`**, so `f_Mspace` keeps a learning
    signal via the *goal* arm. Wire the core to `z` instead (as the first draft
    did) and the detach starves `f_Mspace` to its random init — measured
    `|dL/df_Mspace| = 0.0` exactly under the old wiring vs `45.2` now, which is
    what check [8] asserts.
    **Unguarded residual, per-agent-specific:** `s` and `g` are each one `Dense`
    reshaped to `(N, goal_dim)`, so nothing structurally forces the `N` rows to
    differ — under uniformity pressure per-agent goals silently degrade to one
    team goal and every shape/assertion still passes. Log the mean pairwise cosine
    between agents' goals, the effective rank of `s`, and `Var_t[d_cos]`; a high
    **flat** intrinsic reward is the pathology, so its mean alone reads as success.
  - **Two shape bugs fixed when the helpers met real trainer data** (both were
    invisible to the 1-D self-check): `pool_goals` multiplied an unaligned `(T,)`
    `in_range` against `_same_episode`'s `(T,E,N)` before aligning either, so it
    **raised** for any `goals.ndim > 2` with a `done` mask; and
    `transition_cosine` returned `valid` of shape **`(T,T,E)`** — silently, no
    exception — when handed the trainer's actual `(T, n_envs)` `done`, because
    `(T,1,1) * (T,E)` right-aligns. `done` must now be pre-broadcast to the full
    leading shape of the cosine `(T, n_envs, n_agents)`; `_check_done` enforces it
    (Python-level, so jit-free) and rejects `(T,E,1)` too — that one broadcasts
    *correctly* but makes a masked-mean denominator `n_agents` times too small.
    `transition_cosine` now also returns `valid` at the full shape of `cos` for
    the same reason. Self-check group **[9]** pins the batched path against the
    1-D one per `(env, agent)` slice and asserts the bad masks raise.

### Wiring (what the hierarchy adds to the flat stack)

- **Four train states**, not two: `FeudalTrainState(actor_ts, critic_ts,
  manager_ts, manager_critic_ts)`. `V^M` **reuses `MAPPOCritic`** (writing a
  second critic *class* is what `manager.py` rules out) but keeps its **own
  params and Adam**: the worker critic predicts the intrinsic-augmented return
  under `gamma`, `V^M` the extrinsic one under `manager_gamma`. Its head width
  follows the env (`n_agents` under difference rewards, else 1) so it never
  regresses N copies of one target. All **four msgpack sites** in `run.py`
  (`save_params`, `_save_train_checkpoint`, and both `from_bytes` targets) must
  move in lockstep — `from_bytes` needs an exactly-shaped target tree.
- **The goal ring is the training path; `pool_goals` is only the oracle.** The
  worker acts on `w_t` = sum of the last `c` goals, but `pool_goals` is a
  whole-trajectory function that cannot be called inside the scan, so the scan
  carries a `(c, E, N, D)` ring written with a one-hot slot (`t % c`, not a
  dynamic index, so `t` may be a tracer). The pooled goal **must be stored**, not
  recomputed: the PPO ratio is only valid if `evaluate_action` sees exactly the
  vector `sample_action` saw. Scan carry is now
  `(train_state, env_state, obs, rng, m_carry, goal_hist)` with `xs=arange(n_steps)`.
  - The ring lives in `manager.py` as `goal_ring_write` / `goal_ring_pool` /
    `goal_ring_reset` and is **shared by all three consumers** — the training
    scan, the eval scan, and `run.py:view()`. They are rank-agnostic, so the
    batched `(c, E, N, D)` and unbatched `(c, N, D)` layouts are one code path.
    This matters because a drifted copy of the convention (wrong slot index,
    stale pool) **renders perfectly happily** — it just silently shows a
    different policy than the one that trained. `view()` had its own copy until
    that was consolidated; `test_ring_helpers_agree_across_batched_and_unbatched_layouts`
    now pins the two layouts against each other across a full wrap-around.
- **Two truncation bootstraps.** `Transition` gains a `manager_reward` field
  because `_env_step`'s bootstrap uses the *worker* critic, which is the wrong
  number for the manager stream. Also: a scalar env reward must be broadcast to
  `(E, N)` **explicitly** before the worker bootstrap — `(E,) + (E,N)`
  right-aligns `E` against `N` and *raises* at E=32/N=16.
- **`manager_update` is a separate full-batch pass** (`mappo.py`), not a variant
  of `ppo_update`: `transition_cosine` needs the time axis **in order**, which
  the flatten to `(T*E, N, …)` plus `jax.random.permutation` destroys. It has no
  importance ratio (FuN reinforces the observed state *transition*, not a goal
  likelihood), so extra policy epochs are uncorrected off-policy —
  `n_manager_epochs: 1`.
  - **`n_manager_critic_epochs` is separate from it, and needs to be.** `V^M`
    regresses **fixed** targets, so extra passes are ordinary supervised fitting.
    Sharing the count gave `V^M` *one* gradient step per update against the
    worker critic's `n_epochs * n_minibatches` (48 at the defaults). Measured at
    131k steps: manager value loss 3.9e-3 → 1.1e-3 → 2.8e-4 at 1 / 8 / 48 epochs.
    Default 8 — 48 costs 6x for a further 3.9x and the returns on EV are sharply
    diminishing.
- **Goal flattening must be agent-major in both places** — `_actor_forward`'s
  `pooled_goal.reshape(b*n_agents, D)` and `ppo_update`'s
  `pg_ts[mb_ids].reshape(n_flat, goal_dim)` — matching the obs reshape exactly,
  or every agent silently trains on a neighbour's directive.
- **`view()` runs the full hierarchy** (unbatched, Python-loop ring over the
  shared `goal_ring_*` helpers). Rendering the worker without the manager would
  drive it with a goal it never saw. Verified end-to-end on **both** cores —
  10 episodes, pygame + native MuJoCo videos + reward plots:
  ```
  MUJOCO_GL=egl SDL_VIDEODRIVER=dummy uv run python train.py \
      algorithm=feudal_mappo_jax env=mjx_16a_4o model=feudal trial_id=0 view=true
  ```

#### Goal-following video + raster (`view()`, on by default)

Per episode, `view()` also writes `goal_following_episode_<i>.png` and (MJX only)
`episode_<i>_goals.mp4`; the plain `episode_<i>.mp4` / `_native.mp4` /
`task_vs_alignment*` outputs are unchanged. Code: `goal_visualization.py`
(`frame_alignment`, `goal_following_figure`, the shared color scale) and
`goal_video.py` (overlay, panel, writer).

- **What the marks mean** (`s` = the goal-space state: learned latent, or the
  env's `goal_state` readout on grounded arms; `c` = `goal_horizon`):
  - **ring** (outer band) = horizon alignment `cos(g_(d-c), s_d - s_(d-c))`, i.e.
    whether the goal issued `c` steps ago was followed. This is the manager's own
    objective. A thin gray outline means undefined (the first `c` steps,
    inactive, or a zero vector);
  - **dot** (centre) = one-step `cos(w_(d-1), s_d - s_(d-1))`: did the last move
    follow the goal the worker held?
  - **black arrow** = the goal-induced action,
    `clip(mu(obs, w)) - clip(mu(obs, 0))`, drawn at a FIXED scale (full force = 3
    ring radii). Continuous-action envs only (macro/SMAX pick discrete skills).
    The null is the zero goal, the per-step analogue of the probe's `zeroed`
    variant: exact for FiLM (the unmodulated trunk), off-distribution for concat,
    and "arrived" for waypoints. So it shows sensitivity, not causal value. It is
    **exactly 0** on a `zero_goal` arm (the positive control; `view()` prints
    `max |action change|` per episode);
  - **violet** (grounded arms only) = the goal in the arena: a heading arrow for
    `position_direction`; for `position_waypoint`, the latched waypoint
    `w = s + R·pooled` (x marker), a line to it, and the path since the latch.
    Mapped by `env.goal_state_to_world`, the inverse kept beside `goal_state` in
    both MJX envs. A **latent goal gets no arena mark**: any latent-to-world
    decode would be a fabricated picture;
  - **side panel** = reward plus agents × time heatmaps of the ring and dot
    values, with a cursor at the current frame; the top band shows the per-frame
    numbers (mean cosines, number of agents above +0.5).
- **Scale**: diverging blue (+1 follows) ↔ orange (−1 opposes) around a neutral
  gray. Orange rather than red because MJX agents are red discs; both poles are
  at OKLCH L ≈ 0.576. `alignment_rgb` samples the same colormap the heatmaps use,
  so a color means the same thing in the video and in the PNG.
- **Why post hoc**: a horizon score exists only `c` steps after its goal. So
  `view()` records per-frame agent positions (`MJXRenderer.agent_positions`) and
  `frame_decision` (constant over a macro window), then draws on the buffered
  frames via `MJXRenderer.annotate` after the episode. Every mark reads one
  `frame_alignment` array, which is **causal**: frame `d` uses only
  `s_0..s_d`, the state it shows. On waypoint arms the ring holds the last
  completed latch `(d // c) * c - c`, since waypoints are scored only at latches.
- ⚠ **Following ≠ useful.** A high ring is exactly what the measured decorative
  goal channel looks like (`feudal_film_n01_local`: the goals are followed and
  worth nothing). Read the video for *where and when* goals are followed, and
  settle usefulness with between-arm returns.
- ⚠ pygame rejects numpy **float32** scalars as coordinates (JAX actions are
  f32), so `draw_goal_overlay` casts its inputs to float64. Keep that cast.
- SMAX (`_view_with_env_renderer`) gets the raster only: jaxmarl's visualizer
  draws the GIF and has no hook for per-agent marks.
- The goals video is drawn on a **second, overlay-free render** of each step
  (`goal_frames`). The focus agent's lidar, sector wedges and arrows otherwise
  buried its goal marks, and with one agent that agent is the whole picture. The
  cost is up to ~1.5 GB extra RAM on a 1024-step episode (700² RGB frames).
  Arena marks carry a surface-colored halo so the violet arrows read over the
  colored boxes.
- When another job holds the GPU, `JAX_PLATFORMS=cpu` runs `view` fine: it steps
  one unbatched env. This took ~2–3 min per arm for 10 episodes at 1a/3o.
- **Verified 2026-09-24 on `mjx_1a_3o_111_1024_gs`, trial 0, 7 arms** (all exit 0,
  10 goal videos + rasters each):
  - `feudal_film_zerogoal` reads `max |action change|` **0** in every episode
    (the positive control); every other arm reaches **2**, the full −1 → +1 swing;
  - `goal_horizon: 50` (`feudal_n01_c50`) hatches exactly the first 50 steps;
  - both grounded arms (`feudal_film_waypoint`, `feudal_film_direction`) show
    **long stretches of horizon cos ≈ −1**. The agent drives against its goal for
    60–120 steps at a time while fetching a box: e.g. the latched waypoint sits
    in the drop zone while the agent heads down to the box. These alternate with
    stretches of ≈ +1. Returns are ~327–333 on all 7 arms, `zerogoal` included,
    so none of this goal-following shows up in return at 1a/3o.

### Diagnostics (load-bearing — both failure modes are self-sealing)

`manager_update` returns these in its metrics dict, so they ride `run.py`'s
existing `append_agent_stats` path with **no stats-plumbing change**:
`state_latent_erank` (entropy-based effective rank of `cov(s)`; **≲1.5 ⇒ rank-1
latent collapse**, the mechanism inert while losses look healthy),
`goal_pairwise_cos` / `state_pairwise_cos` (**≳0.9 ⇒ per-agent goals degenerated
to one team goal** — the residual `manager.py` flags as structurally unguarded),
`goal_direction_count` + `goal_pairwise_cos_abs` (the sign-blind companions to
that signed mean — see below), `d_cos_var` (**≲1e-3 ⇒ the cosine is constant**,
annihilated by advantage
centering — never read `d_cos_mean` alone, a high *flat* value reads as success),
`valid_fraction`, `manager_pg_loss`, `manager_value_loss`,
`manager_explained_variance`.

⚠ **Every metric in that paragraph is a COLLAPSE detector — none of them tests
whether the goals are USEFUL.** They answer "are the goals well-formed?", which
is a necessary condition and nothing more, and two of them mislead if read as
usefulness: `d_cos_mean`'s level is uninterpretable because the manager owns
*both* arguments of the cosine, and `worker_goal_column_ratio` already misled
once (it fell while the goal block *grew* 2.2–4.7x). The usefulness tests are
the permutation nulls below.

### Permutation nulls — the "are the goals useful?" diagnostics

The method: permute goals along one axis, which preserves the goal distribution
**exactly** and destroys exactly one property, so the real-minus-null gap
isolates that property. Two axes, and they answer different questions:

| variant | agent pairing | state conditioning | goal present | a gap measures |
|---|---|---|---|---|
| `real` | ✓ | ✓ | ✓ | (reference) |
| `permuted` (agent axis) | ✗ | ✓ | ✓ | value of the **assignment** |
| `env_permuted` (env axis) | ✓ | ✗ | ✓ | value of **state-conditioning** |
| `constant` (one frozen vector) | ✗ | ✗ | ✓ | value of the goal's **content** |
| `zeroed` | ✗ | ✗ | ✗ | value of goal conditioning at all |

⚠ **`constant` is NOT a permutation, and that is the point.** The other three
rearrange or delete the manager's own goals; `constant` substitutes **one fixed
direction** for every agent, env and timestep (`manager.constant_goals`,
direction from `manager.mean_goal_direction` over a rollout's `pooled_goal`). It
exists because `constant` and `zeroed` differ in exactly one bit — whether the
worker still receives a goal-shaped vector of the usual magnitude — and that bit
separates the two readings of a large `gap_zeroed`:

* the goal carries information the worker uses, or
* the goal is a near-constant vector whose **presence** the worker co-adapted
  to, so removing it is a large off-distribution perturbation carrying no
  information at all.

Under the second the manager is **decorative** while every collapse metric and
the zeroed gap read as a healthy, strongly-used goal channel — i.e. the
degenerate case presents as the headline success condition. `real ≈ constant ≫
zeroed` is that case; `real > constant` is the first evidence in this suite that
the goal's content does work. The per-row **norm is preserved** and only the
direction replaced, so the variant stays a pure direction intervention under
`normalize_pooled_goal=False` too. A *random* direction is a fourth thing
(robustness to noise, not "is the manager doing anything") and is not a
substitute — measured on `feudal_film_n01_local` trial 0, a fixed random
direction returns **5.8** against 291 real and 263 constant.
Read `goal_concentration` (offline probe; ‖mean of the row-normalized pooled
goals‖, 1.0 = one frozen direction) next to `gap_constant`: a small gap at a
*low* concentration is the interesting finding (varied goals that nonetheless do
not matter), while a small gap at ≈1.0 only says the manager had already
collapsed onto the constant it is being compared with.

⚠ **READ THE ENV NULL FIRST — the ordering is the finding.** A manager that has
degenerated into a fixed per-agent code (an agent-ID label with no dependence on
`s_t`) scores a *large* agent-permutation gap while doing nothing the hierarchy
exists for — and the collapse suite calls that state healthy, since
`goal_direction_count` reads ≈8.25 against a random-direction baseline of 8.26.
If `d_cos_gap_env ≈ 0` the goals are not functions of the state, and
`d_cos_gap_agent` is uninterpretable however large it is.

**Latent (live, every update, unconditional).** `mappo.manager_cosine_metrics`
adds `d_cos_null_agent`, `d_cos_null_env`, `d_cos_gap_agent`, `d_cos_gap_env`,
`goal_perm_cos` next to `d_cos_mean`/`d_cos_var`/`valid_fraction`. Forward-only
on arrays `manager_update` has already materialized (the loss aux `cos`/`valid`
is passed in, so the real cosine is not recomputed) — no network forward, no
backward pass. Ungated, matching every other manager diagnostic; gating would
make future runs non-comparable. The one invariant it rests on:
`transition_cosine`'s `valid` is a function of `states` + `done` **only**, never
of `goals`, so real and null share one mask — pinned by
`test_transition_cosine_valid_is_independent_of_goals`.

**Behavioural (live, at the eval cadence).** `trainer.eval_fn` gained static
`variants=` / `detail=` args and runs every variant in **one scan** at
`V × n_eval_episodes` vmapped width, with `jnp.tile`d reset keys so episode *j*
of every block starts from a bit-identical state (the gap is therefore a
**paired** statistic that carries no reset variance).
Series: `eval_reward_{permuted,constant,zeroed}`,
`eval_gap_{permuted,constant,zeroed}`, `eval_len_{real,permuted,constant,zeroed}`.
The live default is `("real", "permuted", "constant", "zeroed")`;
`env_permuted` stays offline-only. The `reward` series is
still exactly the `real` block, so every existing plot and pickle is unaffected
(`test_real_eval_variant_is_unchanged` pins the bit-equality).
`eval_len_*` exists because boundary contact *terminates* in these envs — without
it a return gap conflates "less reward per step" with "shorter episode".

⚠ **COST, measured 2026-09-06 — vmap width is free only up to a point, and the
batched scan is NOT ~free.** Clean A/B on `mjx_16a_4o_512/feudal` (983k steps,
7 evals each, warm median excluding the first-call compile):

| | warm median eval | vs. 1 variant |
|---|---|---|
| `eval_goal_variants: false` (32 envs) | 9.58 s | 1.00x |
| `eval_goal_variants: true` (3 x 32 = 96 envs) | 18.26 s | **1.91x** |

⚠ **The 4th block (`constant`, added 2026-09-14) is FREE — measured, and it is
not a rounding error.** Re-measured on `mjx_12a_3o_trunc_1024/feudal_film_n01_local/0`
at `n_eval_episodes=32`, 8 warm reps each, median: **1 variant 7.76 s, 3
variants 12.40 s (1.60x), 4 variants 11.54 s (1.49x)** — four blocks are
consistently *cheaper* than three. Almost certainly because 4 x 32 = **128** is a
power of two and 96 is not, so XLA picks a better tiling. Two things follow: the
`constant` diagnostic costs nothing to keep on, and if the batched eval ever
needs trimming, drop to 2 blocks (64 envs) rather than 3. Do not extrapolate the
1.91x figure linearly in the number of variants — it is not linear.

So three variants cost **1.91x** one eval — better than the ~3x of three
sequential scans, but far from free. Against the production reference
(`mjx_16a_4o_1024/feudal/0`: 612 evals x 9.63 s = 9.3% of a 22.3 h run) that is
**~+8.5% wall**, not the ~+2% the batching argument predicted. The premise
("cost is the fixed sequential scan, not env width") holds only while the GPU is
underutilized: at 16a/4o it is *exactly* true from 5 to 32 envs (9.63 s at
`n_eval_episodes=5` vs 9.58 s at 32 — raising the episode count really is free)
and has broken by 96. Budget the extra ~8.5%, or add a coarser cadence for the
variant blocks; `eval_goal_variants: false` turns them off entirely.

⚠ **The eval gap's reading is ASYMMETRIC.** A permuted rollout is off-policy
twice (mispaired input, and it then visits different states), so the gap
*overstates* the causal value of correct assignment. **gap ≈ 0 is a STRONG
negative**; **gap > 0 is WEAK evidence** and its magnitude is *not* "the value of
hierarchy". A collapsed-goal manager also gives gap ≈ 0 (the permutation is then
near-identity) — `goal_perm_cos` (mean cosine between a goal and the one it was
swapped with; ≈1 ⇒ the permutation changed nothing) is what separates the two.

**Positive control, and the stop-the-line check:** on a `feudal_zerogoal` arm the
worker zeroes the goal *inside* the module, so all variants coincide and every
gap must be **exactly 0.0 bitwise** — `constant` included, and the probe's
control now iterates `GOAL_VARIANTS[1:]` so a newly-added variant cannot slip
past it. Anything else means the harness is wrong and no other number is worth
reading. (Verified: all 6 zerogoal arms measured so far report exactly 0.0, and
re-verified for all 5 variants on the three `mjx_12a_3o_trunc_1024/
feudal_film_zerogoal` seeds.) This works because `_unit(0) == 0` and `goal_embed_dim`'s
Dense is bias-free, so an externally-zeroed pooled goal *is* `zero_goal=True` —
which is also why the ablation is applied to the goal outside the module: the
param tree stays shape-identical and checkpoints remain interchangeable.

**Permutation is a deterministic `jnp.roll`** (`manager.permute_agent_goals` /
`permute_env_goals`), shift from `params.goal_permute_shift` (default 1). A
cyclic shift is a guaranteed **derangement**; a uniform random permutation has
`E[fixed points] = 1`, i.e. on average 1 agent in 16 silently keeps its own goal.
`manager_update` also has no rng in scope, and determinism is what lets the
offline probe and the live series produce the same number. Applied to the
**pooled** `w_t`, never the raw per-step goals: `goal_ring_pool` is a plain sum
so `roll(pool(h)) == pool(roll(h))` bit-identically for a fixed shift
(`test_agent_permutation_commutes_with_the_goal_ring`), whereas permuting raw
goals then pooling would hand agent *i* a sum of *different agents'* goals at
different times — `‖w‖` drops from ≈`c` to ≈`√c`, i.e. a silent ~3.2x scale
intervention, and the marginal is no longer preserved. A shift that is a multiple
of `n_agents` (or `n_agents == 1`) **raises** in `make_train`: the roll would be
the identity, every gap exactly 0.0, and the diagnostic would report "the goals
make no difference" having measured nothing — the self-sealing failure again.

**Offline probe (`algorithms/feudal_mappo_jax/goal_dependence_probe.py`).** Runs
both diagnostics on **already-trained checkpoints**, so the question is
answerable without a new 1e8-step run per arm. Composes each arm's config through
`train._build_dispatch_args` (Hydra) rather than rebuilding the env from CLI
flags as `global_state_probe.py` does — a hand-copied yaml value is how you
silently measure a different network than the one that trained.
⚠ It then **reads `goal_dim`/`hidden_dim`/`manager_hidden_dim` back off the
checkpoint's param shapes and overrides the config**, because the yaml moves
while checkpoints do not: commit `44c3af0` changed `goal_dim` **16 → 32** after
every existing feudal arm was trained, so composing today's config for those runs
fails to load outright. (For an *unparameterized* setting like
`normalize_pooled_goal` the same drift would load fine and silently evaluate a
different function — the probe prints the resolved config and checkpoint mtime
per arm for that reason.) It reports paired-bootstrap CIs over episodes, adds the
offline-only `env_permuted` variant, and records `goal_direction_count` on both
the raw goal and the **pooled** `w_t` (the existing series is raw-only, which
leaves a gap between the collapse metric and the thing actually permuted).
```
MUJOCO_GL=egl uv run python -m algorithms.feudal_mappo_jax.goal_dependence_probe \
    --batches mjx_16a_4o_trunc_1024 --models feudal,feudal_zerogoal \
    --trials 0,1,2 --n-eval-episodes 64 --shifts 1
```

#### MEASURED 2026-09-17 — `mjx_12a_4o_4444_512`: goals can be COHERENCE-CRITICAL and WORTHLESS at once

The finding that added `gap_zeroed > 0` as a third acceptance condition. Env
group: 12 agents / 4 objects, `variant: trunc`, `coupling_def: [4,4,4,4]`
(sums to 16 > 12, so deliveries are forced sequential), and `params.n_steps: 512`
against the env's `max_steps: 1024`. `feudal_n01_local_private` — concat fusion,
`manager_latent: local_private`, alpha=0.1 — *looks* like it beats the baselines
there. It does not, twice over.

**(a) The matched control ties it exactly.** `plotting/config.yaml` listed
`feudal_film_zerogoal` but not `feudal_zerogoal`; the arm is **concat**, so
`feudal_zerogoal` is its fusion-matched goal-free control. Probe, 64 paired
deterministic episodes, 3 seeds:

| arm | real | permuted | env_permuted | constant | zeroed |
|---|---|---|---|---|---|
| `feudal_n01_local_private` | **281.4 ± 72.1** | 257.3 | 260.3 | 240.3 | 283.8 |
| `feudal_zerogoal` (matched) | **281.1 ± 67.2** | — | — | — | — |
| `feudal_film_zerogoal` (plotted) | 208.8 ± 52.5 | — | — | — | — |
| `feudal_n01` (centralized latent) | 159.9 ± 6.7 | 162.9 | 168.1 | 159.7 | 170.6 |

**(b) The `_1024` sibling inverts the ranking, because 512 BREAKS THE BASELINES.**
`mjx_12a_4o_4444_1024` is the identical arm set at `n_steps: 1048`:

| arm | @1024 | @512 | Δ |
|---|---|---|---|
| `mlp` | **433.0 ± 13.1** | 230.0 ± 51.6 | −203 |
| `feudal_zerogoal` | **426.3 ± 9.4** | 278.8 ± 115.5 | −148 |
| `feudal_film_zerogoal` | **438.0 ± 4.7** | 216.7 ± 101.6 | −221 |
| `feudal_n01_local_private` | **144.8 ± 20.2** | 276.5 ± 117.1 | **+132** |

Seed SD explodes 5–20x at 512 (4.7 → 101.6). ⚠ **It is NOT mainly the truncated
episode window**, which was the obvious hypothesis (`collect_fn` resets every env
at the top of every rollout and scans exactly `n_steps`, so at 512 training never
sees steps 512–1023 — the trap this file records for `mjx_16a_4o_multi_goal`).
Measured by running both trained `mlp` policies on identical reset keys (32
episodes) and splitting the return by segment:

| | steps 0–511 (both trained here) | steps 512–1023 (512-policy never trained here) | total |
|---|---|---|---|
| trained @ `n_steps=1048` | 309.2 | 126.9 | 436.1 |
| trained @ `n_steps=512` | 179.1 | 52.6 | 231.7 |
| deficit | **130.1 (64%)** | 74.3 (36%) | 204.4 |

**64% of the deficit is inside the window the 512 policy trained on** — it is a
worse policy everywhere, not a good one falling off a cliff at step 512. The
window effect is real but minority (the 512 policy keeps 58% of the 1024
policy's return in the trained half, 41% in the untrained half). That leaves the
halved per-update batch (512x32 = 16384 env-steps vs 1048x32 = 33536, hence 6103
updates instead of 2980) as the remaining candidate. **Unseparated:** the clean
test is `n_steps=512` with `env.n_envs=64`, which restores the batch while
keeping the truncated window.

**(c) The goals are USED, state-conditioned, agent-specific — and worth zero
return.** Every collapse detector on this arm is the healthiest in the codebase:
the env null gate **passes** (`d_cos_gap_env` +0.081 against `d_cos_mean` +0.110,
i.e. 73%), `d_cos_gap_agent` ≈ `d_cos_mean`, `goal_direction_count` 8.98 against
the random-direction baseline of **8.93** at N=12/`goal_dim`=32 (so
`local_private` really did fix the row collapse that `local` caused),
`goal_concentration` **0.129** (nowhere near the 0.78 of the decorative
`mjx_12a_3o` case), `goal_perm_cos` −0.013, and the `feudal_zerogoal` positive
control reports exactly 0.0 on all five variants. Read the returns instead:

* **`zeroed` 283.8 ≥ `real` 281.4** — the correct goal buys **nothing** over no
  goal at all (`gap_zeroed` = −2.4);
* `permuted` 257.3 / `env_permuted` 260.3 — any *incoherent* goal costs ≈20–24;
* `constant` 240.3 — a frozen direction costs 41.

Both nulls cost about the same, which is the signature of a **coherence
requirement, not an information channel**: `local_private` builds `s_i` from
`obs_i`, so `g_i` partly re-encodes what the worker already sees, and
contradicting it is an off-distribution hit. So `gap_permuted` here measures
**damage from incoherence**, not value from the assignment — and this arm
satisfies the old two-condition acceptance test (`gap_zeroed ≈ 0` **and**
`gap_permuted > 0`, positive in all 3 seeds) while being exactly as good as its
goal-free control. That is why the test now requires `gap_zeroed > 0`.

⚠ `feudal_n05_local_private` seed 1 is **missing** from the 512 batch (n=2).

#### MEASURED 2026-09-14 — `mjx_12a_3o_trunc_1024`: the goal channel can be LIVE and DECORATIVE at once

The finding that produced the `constant` variant. `feudal_film_n01_local` is the
best feudal arm in that batch and the one that looks like it is still climbing
(late-run slopes +30.8 / +10.8 / −8.9 return per 1e7 steps across seeds, against
−3.6 / −7.3 / +0.5 for `mlp`). It is neither of the two obvious explanations:

* **not "the goal channel is off"** — `zeroed` drops it from ~206 to **1.5**, the
  largest goal-dependence in the batch. Under FiLM `zeroed` is *exactly* the
  unmodulated trunk (γ/β bias-free), so the trained policy lives entirely in the
  modulated regime;
* **not "good per-agent goals"** — all 12 agents get one direction
  (`goal_direction_count` 1.45 against the random-direction baseline **8.93**,
  inter-agent cos 0.58–0.94), `permuted` is free, `env_permuted` is free, and
  replacing the goal with the per-env mean over agents is free.

**`constant` is what named it**: one frozen vector recovers **95%** of the return
(206.4 → 195.3 over 3 seeds; per-trial gaps +17.2 / +22.2 / −6.3, only one CI
excluding 0). `goal_concentration` is **0.78** — the manager had already very
nearly collapsed to that constant. So the manager's entire time- and
agent-varying output is worth ~nothing, while every collapse metric plus a
+280 `gap_zeroed` read as a strongly-used goal channel. A dose–response confirms
the dependence is co-adaptation to that *specific* direction rather than
goal-following (trial 0, return vs cosine-to-real): `1.00 → 291`, `0.95 → 288`,
`0.78 → 287`, `0.49 → 250`, `0.20 → 78`, random `0.01 → 5.8`, zeroed `1.5`.

**Mechanism: goal-dependence needs `local` AND `intrinsic_coef > 0`.** Under
`centralized` the worker ignores the goal at every alpha; under `local` the
dependence is monotone in alpha, and so is the collapse
(`real` / `zeroed` / `goal_direction_count`, means over 3 seeds):

| α | centralized | local |
|---|---|---|
| 0.00 | 151.9 / 151.6 / 8.99 | 105.2 / 172.3 / 1.79 |
| 0.01 | 169.5 / 173.3 / 8.94 | 106.3 / 58.3 / 1.45 |
| 0.10 | 140.0 / 148.3 / 9.02 | **203.7 / 1.4 / 1.45** |
| 0.50 | 83.9 / 82.7 / 9.20 | 65.7 / 0.1 / 1.95 |

`local` makes `s_i` a pure function of `obs_i`, so `d_cos(s_i(t)−s_i(t−k), g_i)`
is finally something agent *i* controls — but manager and worker share that
cosine as an objective, and the cheapest joint solution is degenerate: the
manager freezes on one direction and the worker drives its own observation along
it. At α=0 the goal is actively harmful (zeroing **helps** by +67); at α=0.5 the
intrinsic gradient eats the task. α=0.1 is where the lock-in is strong and the
task survives — which is why that arm looks best and why it is not evidence for
hierarchy.

⚠ **It still loses to every goal-free control**: `feudal_film_zerogoal_dilated`
293.7, `mlp` 272.3, `feudal_film_zerogoal` 248.9, `feudal_film_n01_local` 203.7.
The best arm in the batch is the one whose goals are provably disconnected.

⚠ **Consequence for the acceptance criterion in `conf/model/feudal_film.yaml`:**
a large `gap_zeroed` is *not* progress on its own. `gap_zeroed` large +
`gap_permuted ≈ 0` + `gap_constant ≈ 0` is the degenerate outcome, and only
`constant` separates it from the target one. (This is one half of why that
criterion is now **three** conditions rather than two; the other half is the
`mjx_12a_4o_4444_512` measurement below.)

#### MEASURED 2026-09-06 — all 20 trained arms (5 env groups × 4 models × 3 trials, 64 episodes)

Results pickled at `algorithms/feudal_mappo_jax/goal_dependence_probe.results.pkl`.
**The two sides disagree, and that disagreement is the finding.**

**Latent: the manager's goals are well-formed, agent-specific AND
state-conditioned.** `d_cos_gap_agent ≈ d_cos_mean` in *every* arm (e.g. 0.233 vs
0.227) — i.e. `d_cos_null_agent ≈ 0`, so handing agent *i* a teammate's goal
destroys the cosine completely. `d_cos_gap_env` is 60–90% of `d_cos_mean`
(0.184/0.233), so the env null gate **passes** and the agent gap is interpretable.
`goal_perm_cos` is −0.016…+0.044 everywhere, so the permutation really is
disruptive and a zero *behavioural* gap cannot be blamed on collapsed goals.
`goal_direction_count` is the same on the raw goal and the pooled `w_t`
(8.4/8.4, 10.9/10.9), so pooling does not collapse the directions either.

**Behavioural: the worker does not use any of it.** `gap_permuted ≈ 0` — only
**5 of 45** non-control trials have a paired CI excluding 0, and those go in both
directions. Per the asymmetry above this is the STRONG direction: mispairing the
goals maximally changes the return by nothing.

**And conditioning on the goals actively COSTS return.** `gap_zeroed` is
systematically **negative** — 27 of 45 trials significant, essentially all
negative — i.e. feeding the same trained worker a zero goal *improves* it, e.g.
`mjx_16a_4o_1024/feudal_n05` 43.7 → 133.3 (3.0x), `partition_512/feudal_n05`
83.6 → 143.7, `partition_512/feudal` 107.7 → 148.1. Consistent across 15 of the
16 non-control arms; `mjx_16a_4o_trunc_1024` is the exception (gaps ≈ 0).
Separately, the `feudal_zerogoal` *training* arm outscores `feudal` outright
(trunc_1024: 368.9 vs 242.3; 512: 185.0 vs 162.1; partition_512: 152.1 vs 107.7).

⚠ **Two readings of `gap_zeroed < 0`, and this measurement does not separate
them.** (a) the directives are genuinely misleading; (b) the goal block is a
large input perturbation and zeroing it merely de-saturates the first Tanh —
which is mechanistically plausible given the measurement already recorded here
that the goal block held **59–78%** of layer-1 variance even after
`normalize_pooled_goal`. The assumption-free claim is the *pairing* one:
`gap_permuted ≈ 0` with `goal_perm_cos ≈ 0`.

#### The fix: `worker_fusion: film` (`conf/model/feudal_film.yaml`)

`FeudalWorker` gained a `worker_fusion` knob — `"concat"` (default, unchanged)
or `"film"`, zero-initialized **Feature-wise Linear Modulation** (Perez et al.
2018) on both hidden preactivations: `h ← (1 + γ(w_t)) ⊙ h + β(w_t)`.
`MAPPOActor` gained an optional `modulate(x, layer)` hook called *before* each
Tanh (FiLM's own placement, and the only one that can move units into and out of
saturation rather than rescaling an already-squashed value); `modulate=None` is
byte-identical to the pre-hook module, so every other caller is untouched.

It addresses the two **structural** properties of concat, neither of which is
about scale (scale is spent — `normalize_pooled_goal` already took the goal
block from 59–78% of layer-1 variance to a measured **14–19%**, while the
worker's goal columns are actually *larger* than its obs columns per input dim,
ratio 1.11–1.22):

1. **The default is influence, not no-influence.** Measured at init on real
   `mjx_16a_4o_trunc_1024` observations: swapping the goal moves concat's action
   mean by **1.40e-03** against an action scale of **1.65e-03** — i.e. **~85% of
   the untrained policy's output is goal-driven before any learning**, so
   goal-*agnosticism* is what the worker must learn. FiLM measures **exactly
   0.000e+00** for both a swapped and a zeroed goal. Zero-init makes identity the
   default and influence the thing that must be earned.
2. **Concat can only translate the policy.** `∂z₁/∂obs = W_obs` contains no
   goal term, so a concatenated goal cannot change *which* observation features
   matter — only where the operating point sits. FiLM is multiplicative, so the
   goal appears in the Jacobian and can gate features up/down/off.

Details that are load-bearing rather than stylistic:
- **`1 + γ`, not `γ`.** A bare `γ ⊙ h` with a zero-init kernel outputs zeros and
  kills the forward pass. Same trick as adaLN-Zero / ReZero / LoRA's zero-init B.
  It does **not** stall learning — `dL/dW_γ = (dL/dz ⊙ z) gᵀ` is nonzero from the
  first update, so zero-init sets the default, it does not disconnect.
- **γ/β Dense layers are bias-free**, so `γ(0) = β(0) = 0` *forever*, not just at
  init: "zero goal ⇒ exactly the flat policy" survives training. That is what
  keeps the probe's `zeroed` variant interpretable — with a bias, a trained
  `γ(0) ≠ 0` would make that arm "flat policy plus a learned constant
  modulation" and the control would silently drift. FuN makes `φ` bias-free for
  the same reason. It also makes `zero_goal=True` under FiLM a genuinely exact
  isolate (flat policy + per-agent critic + manager training).
- ⚠ **Not checkpoint-compatible with the concat arms**: concat's first Dense has
  input width `obs_dim + goal_dim` (56), FiLM's is `obs_dim` (40). The arm trains
  from scratch into `results/<env>/feudal_film/`; the 15 existing `feudal`
  checkpoints cannot warm-start it. Pinned by
  `test_film_and_concat_are_not_checkpoint_compatible`.
- ⚠ **FuN's own bilinear `U(obs) @ φ(g)` was NOT used**, deliberately: its "zero
  goal ⇒ no preference" property holds for *discrete logits*, but these envs are
  continuous force control, where a zero goal would give a zero mean action —
  "stand still", a specific and consequential action, not neutrality. A naive
  port would look principled and be wrong on exactly the arms measured. For the
  discrete arms (`macro_mjx`, SMAX) it is the right shape; for continuous it
  needs to be residual on the mean.
- ⚠ **Watch saturation.** The trunk already runs at 36–43% of units with
  |tanh| > 0.95 (the goal itself contributes ~5 points of that). A gain before
  the nonlinearity can worsen it; zero-init means it *starts* at the goal-free
  36%, and `1 + tanh(γ)` ∈ [0, 2] is the fallback bound if it drifts.

**Acceptance test is the probe, and it needs ALL THREE** (every gap is
`real − variant`; each needs a paired CI excluding 0):

| condition | what it captures | what its failure looks like |
|---|---|---|
| `gap_zeroed > 0` | the goal channel **earns** return against no goal at all | `≈ 0`: deleting the channel is free, so it is worth nothing |
| `gap_constant > 0` | that return comes from the goal's **content**, not a frozen vector | `≈ 0`: the worker co-adapted to a bias; the manager is decorative |
| `gap_permuted > 0` | and specifically from the per-agent **assignment** | `≈ 0`: the worker uses content but not who-gets-what |

The concat arms fail all three today. Expect FiLM alone to move `gap_zeroed` off
its negative value at best — at `intrinsic_coef=0` nothing in the worker's
objective asks it to follow the goal, and the `n01`/`n05` data says the current
intrinsic reward is not the answer either.

⚠ **`gap_zeroed > 0` is the THIRD condition, added 2026-09-17, and it replaces
the weaker `gap_zeroed ≈ 0` this file asked for before.** That waypoint was
written against the 2026-09-06 failure where the goal *cost* return (zeroing it
improved 15 of 16 arms), so "stopped costing" was the thing to reach — but
`≈ 0` means `real ≈ zeroed`, i.e. **deleting the channel is free**, which a
useful channel cannot be. The counterexample that passes the old two-condition
test and is still worth nothing is measured below
(`mjx_12a_4o_4444_512/feudal_n01_local_private`).

⚠ **Three necessary conditions are still not sufficient.** Every variant is an
off-distribution perturbation of an already-trained policy, so by the asymmetry
above a positive gap is **weak** evidence and a zero gap is **strong negative**
evidence. The probe reports what the trained policy is *sensitive to*; it cannot
show the mechanism bought anything. The decisive test is the **between-arm** one
— the arm's return against `feudal_zerogoal` trained from scratch on the same
env group and seeds. Use the probe to rule arms **out** cheaply; cite the
between-arm gap to rule one **in**. Verified end-to-end
(train + resume + aligned stats); 6 new seam tests, 52 passing overall.

⚠ **A `conf/model/*.yaml` WITHOUT `# @package _global_` is SILENTLY INERT — it
cost 12 full runs (found 2026-09-08).** Every model group's keys must land at the
top level (`model_params:` / `params:` at column 0 under that header); without it
Hydra nests them under the `model` node, where nothing reads them, and the run
proceeds at the **algorithm defaults** with no warning. The six
`conf/model/feudal_film_{n01,n05,zerogoal}{,_dilated}.yaml` shipped without it, so
every arm of `mjx_12a_3o_trunc_1024` and `mjx_12a_3o_partition_1024` trained as
plain **concat** feudal with `zero_goal=False`, `manager_core=mlp` and
`intrinsic_coef=0.0` — i.e. **6 identical configurations x 3 seeds per batch, not
6 arms.** Those 36 runs answer no FiLM / intrinsic / recurrence question and must
be re-run (header added 2026-09-08; composition re-verified for all six).
Diagnose it from the CHECKPOINT, which cannot lie about what trained:
`actor/params/MAPPOActor_0/Dense_0/kernel` is `(obs+goal, hidden)` = `(72, 168)`
for concat and `(obs, hidden)` = `(40, 168)` for FiLM (whose tree also carries
`film_0`/`film_1`); `manager/params/core` is a single Dense for the mlp core and
LSTM gates for `dilated_lstm`; and `intrinsic_reward_abs` is identically 0.0 in
the stats when `intrinsic_coef` never arrived. The three existing header-carrying
groups (`feudal`, `feudal_film`, `feudal_zerogoal`, and the `feudal_n0*` files
that inherit via `defaults:`) were unaffected.

⚠ **The `eval_gap_*` positive control FAILED on those runs, and that is the same
bug, not a harness fault** — worth knowing because the failure mode is generic:
`feudal_film_zerogoal` reported max |gap| of 48-92 where the invariant demands
exactly 0.0 (offline probe on trial 0: perm -12.99, **env -26.21 with a paired CI
excluding 0**, zeroed -0.06). With `zero_goal` silently False those arms were
ordinary goal-conditioned workers, so the gaps were real. Two things this
established that survive the bug: the MJX env is **bit-identical across vmap
lanes** (96 lanes, tiled reset keys, identical actions, 1024 steps -> `max|obs
diff| = 0.0`), so a nonzero gap can only come from the policy and there is **no
chaotic noise floor** to blame; and `goal_dependence_probe.py`'s positive-control
assertion used to key on the literal model name `feudal_zerogoal`, so every
renamed `*_zerogoal*` arm slipped past it. Since 2026-09-24 it matches any model
name containing `zerogoal` (e.g. `feudal_film_zerogoal`).

⚠ **The eval-level pre-flight "all variants agree at init" is VACUOUS — do not
use it.** At init the actor head is `orthogonal(0.01)`, so the policy is inert:
returns are 0 and (on `trunc`, with inert walls) every episode runs the full
1024 steps, for **concat too**. The init-identity property is only observable at
the network level — compare action means directly, as
`test_film_is_identity_at_init` does against the flat `MAPPOActor`.

**So the manager side works and the manager→worker channel is where it fails** —
which locates the concat-fusion degeneracy `worker.py` warns about. Note the
worker has *not* simply ignored the goal (zeroing it changes the return a lot);
it has entangled with it in a way that does not help. The thing to change was
therefore the **fusion**, not the manager, the goal width, or the intrinsic
reward — implemented as `worker_fusion: film` (see above). This also gives the
previously-unattributable ~250-point `feudal` vs `mlp` gap a direction: it is the
goal conditioning.

⚠ **`n_eval_episodes` was a silent `5`.** `MAPPOConfig.n_eval_episodes` defaulted
to 5 and neither `run.py` nor any `conf/algorithm/*.yaml` ever set it, in
**both** jax stacks — so every `reward` point in every plot predating this is a
5-episode deterministic mean, far too noisy to resolve a goal-ablation gap on a
~470-return env. Now wired through `Params` and defaulted to **32** in
`feudal_mappo_jax` *and* `mappo_jax` (kept equal so the feudal-vs-flat comparison
does not also compare two different eval noise floors). It buys width, not depth,
so it is nearly free; it reduces variance without biasing the mean, so runs at 5
and at 32 stay comparable in expectation.

⚠ **Any newly-added stats series is a silent resume hazard**, now fixed at the
source. `append_agent_stats` is a bare `defaultdict(list)` append, so a key that
did not exist when a run started — or that is absent from the checkpoint a run
resumed from — is created on its first append and stays permanently *shorter*
than `total_steps`; the notebook then plots it against `range(1, len+1)`, i.e.
shifted left and averaged against other runs' wrong iterations.
`TrainingStatsTracker.to_dict()` (`algorithms/mappo_vanilla/trainer_components/`)
now **left**-pads every short non-empty series with NaN to `len(total_steps)`.
Left, not right: element 0 of a late series belongs to the iteration it first
existed for. Fixed at save time rather than in `load_from_dict` because at load
time the key is simply *absent* — there is nothing to pad, and padding it there
would need a hardcoded key list that rots on the next new metric. Empty lists are
skipped (`action_distribution` is legitimately length 0 for continuous runs, and
padding it would make it ragged). Pinned by `test_to_dict_left_pads_short_series`.

#### Grounded goals (`goal_space`) — wired, UNRUN

`model_params.goal_space` replaces the manager's **learned** latent `s` with a
**zero-parameter readout of the environment**, so a goal stops being a direction
in an unobservable space and becomes a physical quantity.

```
goal_space: latent              # default, BIT-IDENTICAL to the pre-flag code
goal_space: position_waypoint   # g = a target (x,y), latched for `goal_horizon`
goal_space: position_direction  # g = a unit direction in world (x,y)
```

- **Why.** Under `latent` the manager owns **both** arguments of its own
  objective — it picks the measuring stick `s` *and* the target `g`. That is why
  `d_cos_mean`'s level is uninterpretable, why the collapse detectors exist, and
  why every measurement says the channel is decorative (`gap_permuted ≈ 0` on 40
  of 45 non-control trials; `gap_zeroed` systematically negative; the best arm in
  `mjx_12a_3o_trunc_1024` is `feudal_film_zerogoal_dilated`, whose goals are
  provably disconnected). With `s = env.goal_state(state)` the yardstick has no
  parameters: the detach rule is satisfied **structurally** (verified: gradient
  into `Transition.state_latent` is exactly 0.0), rank-1 latent collapse is
  **impossible** rather than guarded, `d s[i]/d obs_j` is exactly 0 for `j != i`
  for **every** `manager_latent`, and progress is measurable in metres.
- **The env hook is `env.goal_state(state) -> (n_agents, 2)` + `goal_state_dim`,
  and it is an UNCONDITIONAL method** — unlike `global_state`, whose instance-dict
  binding exists because `hasattr(env, "global_state")` is the *trainers'* switch
  and a class method would flip every arm's critic width. `goal_state` has **no
  `hasattr` consumer**: the only switch is `config.goal_space`, which defaults to
  `latent`. Pinned by `test_goal_state_hook_presence_is_inert` (hooked vs
  hookless env, 0.0 diff on all 18 `Transition` fields, every loss key and every
  post-update param leaf). Published by `MultiBoxPushMJX` and
  `MultiBoxMultiGoalPushMJX`, **forwarded** by `SyncMacroMJX` via `base_state()`
  (forwarded, not refused like `global_state`, because a missing `goal_state`
  *raises* rather than silently changing a width). **SMAX does not have one** —
  `_inner(state.env_state).unit_positions[:n_allies]` is reachable but the
  normalization is a separate question, so the validator raises.
- **⚠ `goal_dim` is DERIVED, not configured.** `resolve_goal_space(config, env)`
  overwrites it with `env.goal_state_dim`. A grounded arm inheriting
  `goal_dim: 32` from `feudal_film` would make the manager emit a 32-vector
  "direction in R²"; that *does* raise, but inside `cosine_similarity` in
  `manager_update` — **after `ppo_update` already trained the worker on a 32-wide
  error vector for a full update**. Under `latent` the function returns the
  **identical object**, so the default path is a python-level no-op.
- **⚠ `manager_latent_dim` is mandatory on a grounded arm, and it is not a
  tuning knob.** `manager.py`'s core reads `s.reshape(..., n_agents*latent_dim)`,
  so that width is the manager's *entire information channel* — the module
  docstring already said so. Fused with `goal_dim: 2` the core would see
  `2*n_agents` numbers (24 at N=12) as its only view of the world. `None`
  resolves to `goal_dim`, which is **byte-identical** to the pre-split module
  (verified: 0.0 max diff on every param leaf, all five latents).
- **`GoalChannel` (`manager.goal_channel`) is the one definition of how a
  directive reaches the worker**, with `init/write/pool/reset` closures. The ring
  branch wraps the **unmodified** `goal_ring_*` helpers; the waypoint branch
  latches. ⚠ **This fixed a live bug**: `view()` called `goal_ring_*` *directly*,
  so under a waypoint goal it would have rendered a policy that never trained —
  the exact failure the ring consolidation exists to prevent. All three consumers
  (training scan, eval scan, both `view()` paths) now route through
  `build_goal_channel`.
- **`position_waypoint`** — `w_i = stop_grad(s_t,i) + R*unit(ghat_i)`, latched for
  `goal_horizon` steps; the worker eats the **live error** `(w - s_t)/R`, stored
  as `pooled_goal` (it must be stored, or the PPO ratio is invalid).
  `Transition.goal` keeps the **raw** `unit(ghat)`, since `w` is exactly
  reconstructible as `state_latent + R*pooled_goal`.
  `R = model_params.waypoint_radius`, default `1/3` = exactly
  `sector_sensor_radius / world_width` in **both** MJX envs.
  - `latch_waypoints` is the whole-trajectory oracle (the analogue of
    `pool_goals` for the ring) and is **closed-form, no scan**, so
    `manager_update` stays scan-free on the mlp core. Verified in-scan vs oracle:
    **exactly 0.0**.
  - **Manager PG** `(‖w−s_t‖ − ‖w−s_{t+c}‖)/R`, **scored at LATCH STEPS ONLY**
    (folded into `valid`, so the permutation nulls share one mask). At a
    non-latch `t` the window runs past the waypoint's own expiry. At a latch step
    `‖w−s_t‖ ≡ R`, so it reduces to `1 − ‖w−s_{t+c}‖/R`. **⚠ `valid_fraction`
    therefore falls to ~1/c** (measured 0.300 at c=3, vs 0.700 latent) — correct,
    but it cuts the manager's effective sample count, so `manager_lr` /
    `n_manager_critic_epochs` may need revisiting.
  - **`r^I` TELESCOPES, and that is the reason this mode leads.**
    `‖pg_t‖ − ‖pg_t − (s⁺_t−s_t)/R‖`, transition-aligned by construction (a
    function of `next_state_latent`), needing no episode mask and paying no
    boundary zero. Summed over a latch block it collapses to `1 − ‖w−s_end‖/R ≤ 1`
    — verified exactly, block sums ≤ 1. CLAUDE.md names non-saturation as one of
    three compounding causes of the α>0 absorbing state; `d_cos` has no such
    bound. **⚠ Consequently `intrinsic_reward_abs` reads much smaller** (measured
    0.0018 vs the cosine's 0.119). α is a gradient fraction over unit-std
    advantages so the mixing is unaffected, but read as a magnitude it looks dead.
  - **⚠ `normalize_pooled_goal` RAISES here.** The error vector's magnitude *is*
    the distance to go — the entire content of a waypoint. Normalizing it leaves a
    latched direction, i.e. a worse `position_direction`, while every shape, loss
    and diagnostic stays healthy. The default is `True` and every `feudal_film*`
    ancestor inherits it, so the arm must set it false explicitly.
  - **⚠ `gap_zeroed` does not grade a waypoint arm.** A zero error vector is not
    "no directive" — it is the specific, **in-distribution** directive "you have
    arrived". Of the three acceptance conditions in `conf/model/feudal_film.yaml`,
    read `gap_constant` as the content test here.
- **⚠ `position_direction` is MORE exposed to the shared-objective degeneracy
  than `latent`, not less** — it is the **attribution rung and test fixture**,
  not the arm expected to win. Manager and worker still climb one objective
  *through the environment*. Under `latent` the degenerate fixed point was
  "freeze on one direction and drive `s` along it" (measured: `dir_count` 1.45 at
  return 1.4); the physical analogue "everyone go north-east" is *easier*,
  because a force-controlled agent achieves any heading trivially and no bounded
  resource is consumed. Its uses are real but narrow: it isolates "grounded
  objective" from "latch + saturating reward", and it exercises the grounded path
  without the latch. ⚠ Also note **near-stationary agents dilute `d_cos`** —
  `_unit` is zero-safe, so a mechanically dead agent scores `cos ≈ 0` and is
  indistinguishable from a disobedient one.
- **Diagnostics.** `goal_direction_count` is **bounded by `min(N, goal_dim) = 2`
  and SATURATES**: N evenly-spaced headings score exactly 2.0, and so does "half
  at 0°, half at 90°" — it cannot tell a uniform fan from two perpendicular
  clusters. Replaced under a grounded goal space by **`goal_heading_dispersion`**
  = `1 − ‖mean unit goal over agents‖` (and `_axial`, the sign-blind companion).
  **Its baseline depends only on N, not on `goal_dim`** — exactly the property
  the participation ratio lacked: `E[R̄] ≈ 0.886/√N`, so **0.744 at N=12** and
  **0.778 at N=16** (measured 0.740 / 0.774). On the discriminating pair it reads
  1.000 vs 0.293. Watch it trend toward 0 across the **whole** run.
  - ⚠ **Emitted whenever the goal is 2-wide, NOT only when grounded.** The
    saturation is a property of the geometry: a random 2-wide *latent* goal at
    N=12 also scores 1.85 against its ceiling of 2. Gating on `grounded` left
    `feudal_film_narrow` reading only the saturated metric — blind on the one
    thing it exists to show.
  - ⚠ **`state_pairwise_cos` / `state_latent_erank` keep reading `s_learned`**, the
    manager's internal bottleneck, not the objective's `s` — on a 2-d readout an
    erank is bounded by 2 and would read as permanently collapsed.
  - ⚠ **`manager_cosine_metrics` takes a `score_fn`** so the nulls are computed
    with the **same objective and the same mask** as the loss. Real and null
    scored by different functions is a difference of two quantities in different
    spaces: finite, plausible, meaningless.
  - **Waypoint achievement (`position_waypoint` only)**: `waypoint_error_norm` =
    distance left when a waypoint EXPIRES, `‖w − s_{τ+c}‖/R` (1.0 = no net
    progress, 0 = arrived, >1 = ended further away), and
    `waypoint_reached_frac` = share of commitments whose **closest approach** came
    within `WAYPOINT_REACHED_TOL` (0.1) × R — closest approach, so an agent that
    passes through its waypoint and overshoots still counts. Both come from
    `manager.waypoint_achievement` over the **stored** rollout
    (`Transition.goal` + `state_latent`, via `latch_waypoints`), averaged over the
    same complete commitments as the manager's objective (the shared
    `_complete_commitments` mask, factored out of `waypoint_progress`). In units
    of R, not world units, so they compare across arena sizes. No other goal space
    gains a column.
    - ⚠ **`waypoint_error_norm` is exactly `1 − d_cos_mean` on a waypoint arm**
      (at a latch step `‖w − s_τ‖ = R`, so progress = 1 − error). It adds a
      readable name and a stored-data computation, not new information;
      `waypoint_reached_frac` is the one that does add something.
    - ⚠ **At the shipped `goal_horizon: 10` / `waypoint_radius: 1/3` NO POLICY
      CAN REACH A WAYPOINT (measured 2026-09-23).** Driving every agent flat out
      from reset on `MultiBoxPushMJX` covers only **6–12% of R in 10 steps**
      (6a/4o: 0.080 axis / 0.113 diagonal; 12a/3o: 0.064 / 0.091; 2a/4o:
      0.088 / 0.125). Terminal speed is `F/(damping·mass)` = 10 units/s, i.e.
      0.167/step, with a 6-step velocity time constant, while R is 10–13.7 world
      units. Reaching R takes **~50–90 steps** (diagonal is fastest, the larger
      12a arena slowest). So `waypoint_error_norm` cannot
      fall below ~0.85, `waypoint_reached_frac` is 0 for every policy, and the
      waypoint's magnitude (the distance to go, the content that separates it
      from `position_direction`) is effectively constant. R = 1/3 was chosen to
      match `sector_sensor_radius`, a perception scale, not a reachability one.
      Making the waypoint reachable from rest along an axis means
      `waypoint_radius ≲ 0.02` (12a/3o) to `0.03` (2a/4o) at c=10, or
      `goal_horizon` of ~70–90 at R = 1/3.
    - **Confirmed on `mjx_1a_3o_111_1024_gs` (2026-09-24):** every trained
      waypoint seed logged `waypoint_reached_frac` = 0.0 and
      `waypoint_error_norm` ≈ 1.0. Measured there (flat-out agent, 64 resets,
      8 directions; R = 10 world units in the 30-wide arena), 0.9 R takes **60
      steps along an axis and 45 on a diagonal** (1.0 R: 66 / 49), so c=50 still
      misses axis waypoints (0.73 R) and c=80 covers 1.21 / 1.66 R. Hence
      **`feudal_film_waypoint_c80`** (α=0 control) and
      **`feudal_film_n01_waypoint_c80`** (primary), each `goal_horizon: 80` on
      its c=10 parent and nothing else. At c=80 `valid_fraction` is ~1/c
      (0.0115 measured), about 400 manager samples per update at 32 envs; GPU
      memory does not grow with c on this goal space (the waypoint r^I has no
      c-offset window). The latent `_c50` arms do NOT test this — they are
      `goal_space: latent`, where a cosine objective has no reachability.
      `goal_heading_dispersion` is NaN at every logged point on 1-agent arms.
- **⚠ `goal_dependence_probe._dims_from_checkpoint` now reads `goal_dim` off the
  GOAL HEAD**, never off `f_Mspace` (which emits the bottleneck after the split),
  and returns `manager_latent_dim`. Left stale it would rebuild a wrongly-shaped
  target tree and fail `from_bytes` **at the arm being measured**. ⚠ The two
  grounded modes are **indistinguishable from the checkpoint** — same param trees
  — so `goal_space` / `waypoint_radius` are recorded in the probe's provenance
  dict, like `worker_objective`.
- **The latent probes REFUSE a grounded arm**
  (`latent_locality_probe.unsupported_goal_space_reason`, checked in the shared
  `collect_states` choke point): locality there is exact *by construction*, so
  reporting it would be reporting the definition, and the `s` they would measure
  is the bottleneck, not comparable to the recorded 8.9–9.2 / 1.0–2.9 figures.
- **Arms**, all descending from `feudal_film` directly and each setting
  `manager_latent_dim: 32` (REQUIRED — see below; `validate_goal_space` raises
  without it, which is how four of them were found not to run at all):
  `feudal_film_waypoint.yaml`, **`feudal_film_n01_waypoint.yaml` (the primary
  arm)**, `feudal_film_n05_waypoint.yaml`, `feudal_film_direction.yaml`,
  `feudal_film_n01_direction.yaml`.
  ⚠ They deliberately do **not** descend from `feudal_film_narrow`: that arm
  cannot hold the bottleneck fixed (see the confound note below), so inheriting
  from it would imply a control relationship its own header disclaims.
  - ⚠ **THE GOAL-WIDTH CONFOUND IS IRREDUCIBLE — an earlier version of this note
    claimed `feudal_film_narrow` controlled for it, and that arm DID NOT EVEN
    RUN.** A grounded arm changes two things against `feudal_film`: the goal
    becomes physical *and* 2-wide, which under FiLM drops the γ/β modulation from
    rank-32 to **rank-2** (γ = `Dense(hidden)(goal)` over a 2-d input, so
    achievable gains span a 2-d subspace of `R^hidden`). The intended control was
    a latent arm at `goal_dim: 2, manager_latent_dim: 32`. **That configuration
    is ill-posed**: under FuN a goal is a *direction in* the latent state space,
    so `s` and `g` are the same space by construction and `transition_cosine`
    contracts them on the last axis — `s ∈ R^32` against `g ∈ R^2` is not a
    narrow goal, it is a broken objective. It died as a bare `TypeError: mul got
    incompatible shapes for broadcasting: (T,E,N,32), (T,E,N,2)` inside
    `manager_update`, naming neither key. `validate_goal_space` now **raises** on
    it at build time.
    - So under `latent`, `goal_dim` controls the goal width **and** the manager's
      whole bottleneck as one knob (`manager.py`: "shrinking it throttles the
      goal RNN too"), and a 2-wide-goal/32-wide-bottleneck arm is reachable
      **only** by grounding. No width control exists in the latent mode. Measured
      on the corrected arm: `state_latent_erank` 1.91, i.e. the bottleneck really
      is strangled to 2.
    - `goal_embed_dim` does not rescue it either — a bias-free linear map out of
      `R^2` still has rank ≤ 2. **State the limitation; do not manufacture a
      control that cannot exist.** Rank-2 is part of what "a grounded (x, y)
      goal" *means*, not a nuisance to subtract.
    - `feudal_film_narrow` survives as a coherent arm at `goal_dim: 2` alone
      (bottleneck narrows with it). It answers "how much width does the feudal
      channel need?" — a real question — but it is **not** the control for the
      grounded arms.
  - ⚠ **FiLM is NOT required — and the width argument for it is one I got wrong
    and then measured.** `goal_space` and `worker_fusion` are orthogonal;
    `validate_goal_space` *warns* on concat, it does not raise. The goal's share
    of the worker's layer-1 preactivation variance is **9.8% at goal_dim 32, 16
    and 2 alike**, because `normalize_pooled_goal` fixes `‖w‖ = 1` and that share
    is set by the block's NORM, not its width (same measurement reproduces the
    recorded **91.6%** for the pre-normalization raw sum, which validates it). So
    narrowing the goal does **not** starve it of scale, and an earlier claim here
    that the share collapses to ~1% was false. The real concat cost is the
    structural one this file already records: swapping the goal moves the
    **untrained** action mean by 47.3% of its own RMS at `goal_dim=32` and 20.3%
    at 2, against **exactly 0.000** for FiLM at both — influence is concat's
    default and goal-agnosticism must be learned. Plus concat can only
    *translate* the policy (`∂z₁/∂obs = W_obs` carries no goal term), so a 2-d
    directive cannot change *which* observation features matter, only where the
    operating point sits.
  - ⚠ **Under `position_waypoint` + concat the goal's scale is non-stationary
    within a latch block**, because `normalize_pooled_goal` is forbidden there
    and `‖pooled_goal‖` decays from 1 at latch to 0 on arrival: measured goal
    share 9.8% → 3.8% (at `‖w‖ = 0.6`) → 0%. That "the directive fades as you
    arrive" is arguably correct behaviour, but it is exactly the kind of
    episode-phase-dependent scale `normalize_pooled_goal` was introduced to
    remove for the ring. FiLM gets the same fade *cleanly*, since bias-free γ/β
    give `γ(0) = β(0) = 0` exactly.
  - ⚠ **For waypoint the α=0.1 arm is PRIMARY and α=0 is its control.** At α=0
    there is no `r^I`, so the saturation property that justifies the mode is
    inactive and it differs from the direction arm only in the manager's PG.
- Checks: `uv run pytest algorithms/tests/test_feudal_seams.py -q` (**196**
  collected, of which 31 are the grounded-goal seams, incl. 5 pinning the
  waypoint-achievement metrics), plus
  `test_feudal_goal_visualization.py` (20 — three of which were failing at HEAD,
  a `SimpleNamespace` config stub missing `worker_encoder`).
- **⚠ No training result.** Everything above is a mechanism check. Acceptance is
  the **between-arm** return against `feudal_film` (the wide-goal latent arm),
  `feudal_film_zerogoal` (the goal-free floor) and `mlp` on the same env group
  and seeds — NOT against `feudal_film_narrow`, which varies the bottleneck too — not the
  goal-dependence probe, which rules an arm out cheaply but never in.

#### Shared perceptual encoder (`worker_encoder: shared`) — wired, UNRUN

FuN feeds **one** `z_t = f_percept(x_t)` to both the manager and the worker, and
both gradients shape it. This stack never did: the worker ate the raw local
observation and built its own features, a deviation `manager.py` documents but
which had never been tested. `model_params.worker_encoder` closes it.

```
worker_encoder: "none"    # default, and a STATIC no-op (byte-identical)
worker_encoder: "shared"  # worker reads f_enc(obs_i); its PPO gradient trains f_enc
```

Under `"shared"` the worker's observation input is **replaced** by
`f_enc(obs_i)` — the manager's shared per-agent encoder output,
`manager_hidden_dim` wide — and the worker's gradient flows back into `f_enc`
with **no** `stop_gradient`. Arms:
`conf/model/feudal_film_local_shared_enc.yaml` and
`conf/model/feudal_film_local_private_shared_enc.yaml`, each one key apart from
its own matched control.

```
uv run python train.py algorithm=feudal_mappo_jax env=mjx_12a_3o_trunc_1024 \
    model=feudal_film_local_private_shared_enc trial_id=0
```

- **Why**: every measurement in this file says the goals are decorative
  (`eval_gap_permuted` ≈ 0 on 40 of 45 non-control trials, `gap_zeroed`
  systematically negative, best arm in `mjx_12a_3o_trunc_1024` is
  `feudal_film_zerogoal_dilated` whose goals are provably disconnected). One live
  explanation is that the worker must learn, from a scalar PPO gradient alone,
  what directions in a space built by a network it shares **nothing** with mean.

- **THE GRADIENT ROUTING IS THE WHOLE DESIGN, and three obvious ways to do it are
  wrong.** `ppo_update` computes the worker's gradient w.r.t. the manager's
  params and **returns it unapplied**; `update_fn` threads it into
  `manager_update`, which clips it separately and adds it to the transition PG's
  gradient before the single `apply_gradients`.
  - **It must not be applied inside `ppo_update`.** The manager staying frozen
    there is what keeps the epoch-0 PPO ratio exactly 1 **and** keeps
    `manager_update`'s recompute equal to the goals the rollout emitted. Apply it
    inline and the manager is optimized for a policy that never acted — silently,
    every loss finite. Pinned by
    `test_manager_ts_does_not_move_during_ppo_update`.
  - **It must not be accumulated across PPO's minibatch steps.** That would mix
    gradients taken at ~40 different actor iterates (the gradient of nothing) and
    make `n_minibatches` a silent weight on the worker's authority over the shared
    encoder. It is ONE gradient at the pre-PPO actor params, merely computed in
    equal-size chunks to cap activation memory — a single full-batch
    forward+backward through a 256-wide encoder over `T*E*N` rows is
    `n_minibatches`× the peak of one PPO step. Pinned by
    `test_the_encoder_gradient_is_one_full_batch_gradient`.
  - **A second `apply_gradients` would corrupt the manager.** optax's Adam *moves*
    a leaf with a zero gradient (`mu` is nonzero from the previous update and
    `count` increments tree-wide), so every non-encoder manager parameter drifts.
  - At the pre-PPO actor params the ratio is exactly 1, so the clip is provably
    inactive and `min` is a no-op: the encoder gradient reduces to
    `-mean(adv·∇log π)`, structurally the same form as the manager's own
    `-mean(adv·∇d_cos)`. Two full-batch means, same batch, same advantage, same
    parameter point, summed.

- **⚠ The separate pre-clip BOUNDS the coupling, it does not remove it — measured,
  and an earlier version of this note overclaimed.** The manager's optimizer is
  `chain(clip_by_global_norm(grad_clip), adam)` and that clip is **global**, so
  the encoder addend still contributes to the norm that rescales every leaf, even
  though it is exactly zero on all of them. Pre-clipping it to `grad_clip` buys
  **saturation**, not isolation. Worst relative displacement of a non-`f_enc`
  parameter on the CPU stub:

  | `\|enc grad\|` | 0 | 1e-3 | 1.0 | 1e6 |
  |---|---|---|---|---|
  | displacement | **0.0 exactly** | 1.9e-07 | 1.2e-05 | **1.2e-05** |

  Six further orders of magnitude buy no further displacement. *Unclipped* it
  would grow without bound and a large worker gradient would attenuate the
  manager's transition PG by a factor set by how hard the worker pushed — while
  `manager_pg_loss` logged a healthy number. That saturation is the real
  guarantee and is what
  `test_the_worker_gradient_cannot_scale_the_managers_own_pg_step` pins; do not
  restate it as exact isolation.

- **Guards** (`mappo.validate_worker_encoder`, called from **both**
  `trainer.make_train` and `run.py`, like `validate_worker_objective`). Raises on
  an unknown value; on a **non-local `manager_latent`** (only the locals build
  `f_enc`; `centralized` has `f_percept` over the *global* state, which is
  neither per-agent nor obs-width); and on **`n_manager_epochs != 1`** (the
  encoder gradient is computed once at one parameter point, so replaying it
  across manager epochs applies it where the parameters no longer are). Warns —
  runnable on purpose, both are direct contrasts — on `worker_fusion: concat` and
  on `worker_objective: intrinsic_only`.

- **⚠ Pair with FiLM, not concat.** The goal is one block of a concatenated
  input, so widening that input from `obs_dim` to `manager_hidden_dim` cuts the
  goal's share of layer-1 preactivation variance at init from ~9.8% to
  **~0.6–1.5%** — `normalize_pooled_goal`'s whole calibration was derived on the
  narrow geometry. FiLM is immune: the goal reaches the policy only through
  bias-free zero-init layers whose input is the goal, so the observation's width
  is irrelevant to it. Separately, `_goal_column_ratio` is passed the **encoder**
  width when sharing — with the raw `obs_dim` its concat-only guard
  (`kernel.shape[0] <= obs_dim`) would **not** fire (the kernel is *wider*, 288 vs
  40) and it would silently slice an "obs block" and a "goal block" that are
  neither, a third documented misreading of that series.

- **New diagnostics**: `worker_encoder_grad_norm` (from `ppo_update`),
  `manager_encoder_grad_norm`, `worker_encoder_grad_norm_clipped` and
  **`worker_manager_encoder_grad_cos`** (from `manager_update`). **Read the
  COSINE first** — Adam is scale-invariant in the limit, so "the worker's norm is
  10× larger" does not by itself mean the worker wins; a persistently negative
  cosine means the two objectives are pulling the shared representation apart.
  None of these keys exists at `worker_encoder: "none"`, so no pre-existing run
  gains a column.

- **⚠ NOT checkpoint-compatible with any existing feudal arm.** The worker's
  `MAPPOActor_0/Dense_0/kernel` is `(manager_hidden_dim, hidden_dim)` = `(256,
  168)` under sharing against `(obs_dim, hidden_dim)` = `(40, 168)` for FiLM.
  That kernel shape is the tell — unlike the yaml, a checkpoint cannot lie about
  what trained. The **manager's** tree is unchanged, so manager checkpoints stay
  interchangeable. Note this also makes the ACTOR tree depend on
  `manager_hidden_dim`, which previously had no effect on the actor at all.

- **⚠ THE ARM CHANGES TWO THINGS AT ONCE**, so a return gap is not attributable to
  perceptual sharing alone: (a) the worker's input becomes a 256-d encoding of its
  own obs, and (b) the worker now trains `f_enc`. Same objection this file raises
  against `feudal_a0` as an isolate of "hierarchy vs flat". The separating rung is
  one `jax.lax.stop_gradient` (a `"shared_detached"` third value in
  `manager.WORKER_ENCODERS`), deliberately not shipped; add it before attributing
  anything. It also buys an exact test — its encoder gradient must be bitwise 0.

- **⚠ The worker compresses 256 → 168.** `manager_hidden_dim` and the worker's
  `hidden_dim` are one knob in FuN and two unrelated defaults here. Left as-is
  rather than setting `manager_hidden_dim: 168`, which would change a second key
  versus the control. Documented, not accidental.

- **The single home of the shared forward is `worker.encode_obs`**, used by the
  rollout, the eval scan, `ppo_update`, both `view()` policy fns and the FiLM
  diagnostics. Callers that also flatten agent-major must **flatten first, then
  encode** the `(rows, obs_dim)` array: `ppo_update` encodes at that rank, and
  matching it keeps the two bitwise equal rather than merely close — which is what
  keeps "the PPO ratio is exactly 1" an equality instead of a tolerance (verified
  at `atol=1e-6`).

- **The largest scientific risk, and it is specific to this arm**: if `f_enc`
  collapses the way `local`'s `s` did (participation ratio 1.03–2.89 of 12), the
  worker's **entire observation** collapses with it and a return regression is
  uninterpretable. Run `latent_diversity_probe` on `h`, not only on `s`/`g`.

- **⚠ No training result.** Everything above is a mechanism check. Acceptance is
  the **between-arm** return comparison against `feudal_film_local` /
  `feudal_film_local_private` on the same env group and seeds — not the
  goal-dependence probe, which can rule an arm out but never in.

#### The manager's latent is NOT agent-local — measured 2026-09-09, and it is what makes `r^I` unusable

`worker_intrinsic_reward` scores `d_cos(s_t[i] - s_{t-k}[i], g_{t-k}[i])`, so
whatever `s[i]` responds to is what agent *i* gets paid for. Under the original
`manager_latent: centralized`, `s` is one `Dense(n_agents*goal_dim)` over a
2-layer MLP of the **full joint state**, reshaped to `(N, goal_dim)` — i.e.
`s_i = W_i z + b_i`. The agent axis is a **slice index, not a factorization**.

`algorithms/feudal_mappo_jax/latent_locality_probe.py` measures the block
Jacobian `B[i,j] = ||d s[i,:] / d obs_block_j||_F` (input columns scaled by
on-policy std) at on-policy states under the trained hierarchy, and reports
`diag_share[i] = B[i,i] / sum_j B[i,j]`. Over **12 trained arms** (4 env groups
x {`feudal`, `feudal_n01`, `feudal_n05`}, trial 0, 256 states each, N=16 so
uniform = 0.0625):

| | trained | same net at init | uniform (1/N) |
|---|---|---|---|
| `s` (the space `r^I` is measured in) | **0.0631** (0.0621-0.0645) | 0.0623 | 0.0625 |
| `g` (the assigned goal) | **0.0628** (0.0621-0.0636) | 0.0624 | 0.0625 |

Paired trained-minus-init is **+0.00086** for `s` (positive in 10/12 arms). The
metric is **not** blind — its built-in control (`--control`) reads **1.0000** for
a surgically block-diagonal manager and **0.128** for a half-local one, so these
arms moved **~1.3% of the way to *half* localized**. `best-perm` (Hungarian
assignment over the row-normalized `B`, which catches localization stored under a
*relabeling*) is 0.0655 trained vs 0.0650 at init — there is no permutation under
which the rows are agent-local either.

**So agent *i*'s "own" intrinsic reward moves as much when a TEAMMATE moves as
when it does.** `r^I` is per-agent in **indexing**, not in **causation**: it has
the same credit-assignment structure as the team scalar it was supposed to
decompose. This is the mechanism behind
`plans/feudal_goal_reward_diagnosis_2026-09-09.md`: intrinsic coefficient 0.1 ->
145.7 and 0.5 -> 91.0 against a zero-goal control at 252-278 and MAPPO at 264.5,
monotone in alpha, **while** `d_cos_mean` rises to 0.258 — workers farming a
team-aggregate alignment signal that no individual worker controls. Note the
dual-advantage-stream normalization did **not** rescue those arms, so the
magnitude story alone does not explain them.

- **⚠ Every logged diagnostic is blind to this.** `state_latent_erank` and
  `state_pairwise_cos` are computed on the rows **alone** and read healthy here —
  correctly, because the rows *are* distinct from one another. Distinct-but-
  non-local is a **third** failure mode, separate from the rank-1 collapse and
  the row-uniformity residual `manager.py` already flags. Nothing in
  `manager_update` looks at the rows' relationship to the *inputs*.
- **The fix is `model_params.manager_latent: local`** (arm:
  `conf/model/feudal_film_local.yaml`, which differs from `feudal_film` in
  exactly that one key). A **shared** per-agent encoder builds `s_i` from agent
  *i*'s own observation:
  ```
  obs_i --f_enc (SHARED, per agent)--> f_Mspace (SHARED) --> s_i
  [s_1..s_N] --core--> y --f_gpre--> y_i --f_goalhead (SHARED)--> g_i
  ```
  This fixes **two** independently-identified defects at once: `d s[i]/d obs_j`
  is structurally **zero** for `j != i` (verified exactly: max off-diagonal block
  norm `0.0`, diag share `1.0`), and sharing `f_enc`/`f_Mspace` puts every row in
  **one basis** — the diagnosis note's point that each `W_i` otherwise gives
  agent *i* its own private coordinate system. `f_goalhead` **must** be shared
  too, or `g` keeps N private bases and the cosine compares incommensurate axes.
  - The core still consumes `s`, so the bottleneck that keeps `f_Mspace`
    trainable under FuN's detach rule survives (self-check `[10](d)` asserts
    `f_enc_*`/`f_Mspace` all receive gradient).
  - **Centralization is preserved** — the core mixes all N local latents, so goal
    assignment still reads the whole team. For the MJX envs `global_state` **is**
    `obs.reshape(E, -1)` (`trainer._global_state` falls back to exactly that when
    the env has no `global_state` hook), so nothing is lost and `local` is a pure
    refactoring of the same input.
  - **⚠ `local` FIXED THE JACOBIAN AND BROKE THE GOALS — measured 2026-09-14 over
    all 24 trained 12a arms (2 env groups x 4 alphas x 3 seeds).** It is worse
    than `centralized` on return by **−25.4** (95% paired-bootstrap CI
    [−48.6, −0.7], better in 9/24), and on the *unconfounded* `intrinsic_coef=0`
    cut — the only one the still-unfixed `r^I` timing misalignment cannot touch —
    by **−64.6** (CI [−97.0, −27.1], better in 1 of 6 seeds). Both latents remain
    far under flat MAPPO (`mlp` 271.6 / 294.6 on trunc / partition).
    The mechanism, from `algorithms/feudal_mappo_jax/latent_diversity_probe.py`
    (participation ratio over the N agent rows on on-policy states; 1.0 = all
    rows identical, N = mutually orthogonal):

    | stage | `local` | `centralized` |
    |---|---|---|
    | `obs_i` (**the same input**) | 3.85–4.19 | 3.63–4.07 |
    | `s` | **1.03–2.89** | **8.92–9.22** |
    | `g` | **1.46–1.67** | **8.73–9.10** |
    | `s_t − s_{t−1}` (what `r^I` scores) | 2.00–3.66 | 8.99–9.01 |

    So 12 agents under `local` receive ~1.5 distinct directives; the live
    `goal_direction_count` reads 1.39–1.95 against the 8.93 random baseline,
    where `centralized` reads 8.87–9.16. **It is NOT a weight-rank collapse** —
    on those checkpoints `f_Mspace` has effective rank 31.3/32, `f_goalhead`
    25.6–30.9/32, and `f_gpre`'s per-agent blocks are near-orthogonal
    (mean |cos| 0.04–0.15). It is structural: **a shared projection can only
    TRANSMIT the row diversity its input already has, and N homogeneous agents'
    egocentric observations supply only ~4 of 12.** `centralized` escapes that
    ceiling because its diversity lives in the *weights* — one team vector read
    through N near-orthogonal blocks manufactures distinctness that was never in
    the input. Secondary collapse point: the shared `f_goalhead` squashes goals
    that `f_gpre` had just re-diversified (3.26 → 1.62), its top singular
    direction having grown to carry 15.7–27.9% of its energy against 3.1% at
    orthogonal init.
  - **`manager_latent: local_private` is the fix for that** (arm
    `conf/model/feudal_film_local_private.yaml`, plus `_n001/_n01/_n05`
    variants): `local`'s shared encoder is kept, and **both** shared projections
    are replaced by per-agent ones — `s_i = W_i · f_enc(obs_i)` and
    `g_i = W^g_i · y`, stacked `(n_agents, manager_hidden, goal_dim)` params
    `f_Mspace_agent_*` / `goal_head_agent_*` initialized by `block_orthogonal`
    (ONE orthogonal matrix split into blocks, so they start mutually
    near-orthogonal — independently drawn orthogonal blocks are *not*).
    Locality is untouched: it is a property of *where the encoder runs*, not of
    whether the projection is shared. Verified on a smoke checkpoint —
    `latent_locality_probe.py --wrt obs` reads diag share **exactly 1.0000** for
    `s`, and the diversity probe reads `s` **8.72**, `g` **9.24**, differences
    **8.95** against the same ~3-of-12 observation ceiling, i.e. back at the
    centralized branch's numbers.
    - **What it gives up, deliberately.** `local`'s rationale is that a shared
      final projection puts `g_i` and `s_i` in one basis. That is real but
      weaker than stated: the objective contracts `d_cos(s_t[i] − s_{t−k}[i],
      g_{t−k}[i])` **per agent** and never compares agent i's axes to agent j's,
      so it needs `s_i` and `g_i` to agree *for each i* — which a **matched
      pair** of per-agent projections gives. Genuinely lost is cross-agent
      commensurability: "goal coordinate 3" means something different per agent
      again, so the shared worker must learn N goal-to-action mappings, as under
      `centralized`. Row diversity traded for basis sharing; that is the
      hypothesis the arm tests.
    - ⚠ **Checkpoint-incompatible with every existing feudal arm** (no
      `f_Mspace`/`f_gpre`/`f_goalhead` at all).
      `goal_dependence_probe._dims_from_checkpoint` keys off
      `f_Mspace_agent_kernel` and must check it **first** — `f_enc_0` is shared
      with the other local variants, so the `(f_enc_0, f_percept_0)` pair that
      separates `local` from `local_global` does not separate this one.
    - ⚠ **The numbers above are from a 200k-step smoke run, i.e. near init.**
      `local`'s collapse happened by **10M** steps and was permanent, so
      `goal_direction_count` staying near 8.93 across a full run is the thing to
      watch, not its value at the start. And restoring goal diversity is
      *necessary* for the mechanism to be testable, not sufficient for return:
      `eval_gap_permuted` is ≈0 on every arm measured so far.
    - ✅ **RESOLVED 2026-09-14 — it WAS float noise, and the `local` arms are
      fine.** This file previously recorded
      `test_goals_are_reproducible_from_stored_states[mlp-local]` failing at
      5.2e-4 against `atol=1e-5` and argued that failing for `local` but not
      `centralized` was "more than float noise would explain". Measured, that
      reasoning was wrong on both halves. (1) It is not `local`-specific: on GPU
      it fails for **all four** local latents on the `mlp` core (`local` 5.2e-4,
      `local_global` 4.0e-4, `local_private` 4.6e-4, `local_global_private`
      6.8e-4), ordered by the depth of the goal path — `centralized` has the
      shallowest and merely stays under the tolerance. (2) It is **exactly 0.0 in
      float64**, every latent, eager and jitted alike, and it passes on CPU at
      f32. So the recompute is mathematically identical and the manager IS
      optimized for the policy that acted; XLA is just associating the f32
      reductions differently between the rollout's `collect_fn` compilation and
      the test's standalone one — the same artifact this file already records for
      `mjx.ray` (~3e-4, "compare both under jit"). Adding `jax.jit` to the
      recompute fixes the two private latents exactly and leaves the shared-
      projection ones, because the enclosing program still differs.
      **Fix: `test_feudal_seams.py` now has the autouse `_run_on_cpu` fixture**
      that `test_smax_seams.py` already had (CLAUDE.md previously flagged this
      suite as the unpinned one that "does show this"). The tolerance was
      deliberately NOT loosened — a real convention bug (wrong carry, wrong reset
      order) is O(1), so a widened atol would have hidden a precision artifact
      behind a number instead of removing it.
  - **`manager_latent: local_global` is the third value, and it is the SMAX arm**
    (`conf/model/feudal_film_local_global.yaml`). It is `local` plus
    `core_in = concat(s_flat, z)` with `z = f_percept(global_state)`: `s` stays a
    pure function of `obs` — locality and `r^I` unaffected — while **goal
    generation** regains the env's own global state. It exists because on SMAX
    `global_state` is **not** recoverable from the observations, so plain `local`
    would make the manager strictly blinder than the centralized branch:
    `SMAX.get_obs` zeroes unit *j* out of unit *i*'s observation entirely unless
    `‖pos_j − pos_i‖ < sight_range_i`, while `get_world_state` applies no such
    gate; and obs positions are relative/sight-normalized against the world
    state's absolute ones. (Allies are recoverable either way — `get_self_features`
    carries each unit's own absolute position — so the loss is specifically about
    **enemies**.) Measured, 64 envs x 100 steps, random legal actions, fraction of
    alive enemies invisible to **every** ally: **100% at t=0** on all of `3m` /
    `5m_vs_6m` / `2s3z` / `3s5z`, falling to 28–42% over the first 10 steps.
    ⚠ Read the t=0 column only — under random actions the teams never engage, so
    the mid/late figures are pessimistic and a trained policy would push them far
    lower; but t=0 is a **spawn** property, not a policy artifact, and with
    `goal_horizon: 10` that is exactly the window the first pooled `w_t` is built
    in. **Do not use it on an MJX env**: `z` would be computed from numbers the
    encoder already saw, it costs an extra `f_percept`, and it would make the arm
    differ from `feudal_film` in two things instead of one.
  - **`manager_latent: local_global_private` is the fourth value — `local_global`
    and `local_private` at once, and it is the arm to run on SMAX** (arm
    `conf/model/feudal_film_local_global_private.yaml`, plus `_n001/_n01/_n05`).
    The local family is a **2×2 over two independent axes** on top of the shared
    per-agent encoder, and the code now reads them as such (`PRIVATE_LATENTS` /
    `GLOBAL_LATENTS` in `manager.py`, membership tests rather than four
    hand-written string equalities):

    | model group | *private* (per-agent `W_i` on `s` and `g`) | *global* (`f_percept` on the goal path) |
    |---|---|---|
    | `feudal_film_local` | – | – |
    | `feudal_film_local_private` | ✓ | – |
    | `feudal_film_local_global` | – | ✓ |
    | `feudal_film_local_global_private` | ✓ | ✓ |

    It exists because on SMAX **both** defects are live and each single-axis arm
    fixes only one: `local_private` there leaves the manager blind to enemies no
    ally can see (100% of them at t=0), and `local_global` there leaves the N goal
    rows collapsing onto ~1.5 directions. **Yes, `local_private` *runs* on SMAX**
    (verified end-to-end: train + resume at `smax_3m`) — nothing crashes, which is
    exactly why this needed a separate value rather than a guard.
    - **Locality is untouched by either axis.** `s` is a pure function of `obs` in
      all four, so `d s[i]/d obs_j` is exactly 0 for `j != i` and `r^I` stays
      clean. Only goal *generation* reads `z`.
    - **Pick the narrowest variant the env needs.** On MJX the `global` axis is
      dead weight (`global_state` **is** `obs.reshape(E, -1)` there), so compare
      `local` vs `local_private`; on SMAX compare `local_global` vs
      `local_global_private`. Crossing the two families changes two things at once.
    - ⚠ **`_dims_from_checkpoint` needed a second bit.** Its
      `f_Mspace_agent_kernel` early-return hardcoded `"local_private"`, which for
      a `local_global_private` checkpoint (that key **and** `f_percept_*`) would
      build a target tree missing `f_percept_*` — a loud `from_bytes` failure, but
      at the arm you were trying to measure. It now splits that branch on
      `f_percept_0` too. **No single key separates the five latents**: `f_enc_0`
      is in all four locals, `f_percept_0` is in `centralized`/`local_global`/
      `local_global_private`, `f_Mspace_agent_kernel` is in both private ones.
      `test_every_latent_is_distinguishable_from_the_checkpoint_alone` now calls
      the **real** function on serialized trees for all five rather than
      reimplementing its rule — a copy of the rule in the test is precisely what
      would have kept passing through this change.
    - ⚠ **`latent_diversity_probe.py` had a live bug this surfaced**: its
      `local_global` branch built `core_in` **without** `z`, so `y` and every goal
      number it printed for those arms came from a forward pass the checkpoint
      never computed. Fixed (shared `with_global` helper) for both global variants.
    - ⚠ **Neither latent probe works on SMAX arms, at any latent — now declared
      rather than discovered.** Both are MJX-only in *two* undeclared ways, and
      the shared `collect_states` (`latent_locality_probe.py`) is where both
      live: it builds the manager's centralized input as `obs.reshape(b, -1)`,
      i.e. it **assumes** `global_state == concat(obs)`, and it hardcodes
      `discrete=False` when sampling. The callers then invert the same assumption
      to recover per-agent observations (`gs.reshape(N, obs_dim)`). On SMAX the
      real global state is 72 dims at `3m` against 195 of concatenated obs, so a
      `local_global*` arm died on a raw `ScopeParamShapeError` inside
      `f_percept_0` and a `centralized` one would have been silently measured on
      an input it never trained on. New helper
      `latent_locality_probe.unsupported_env_reason(env)` names the reason;
      `collect_states` **raises** on it (single choke point, so neither probe can
      bypass it) and `latent_diversity_probe.py` checks it *before* the rollout
      and skips with a message. Verified: the SMAX arm now prints
      `SKIPPED — env has its own global_state hook (72 dims vs 195 …)`, and the
      MJX arms are unaffected (`mjx_12a_3o_trunc_1024` reproduces the recorded
      1.04/1.62 for `local` and 9.12/8.76 for `centralized`). Measuring SMAX arms
      needs `collect_states` to store `obs` alongside `env.global_state(state)`
      rather than deriving one from the other, and to thread
      `env.discrete`/`avail_actions` through the rollout.
    - **Verified**: the four pre-existing latents are **bit-identical** to the
      pre-change code (0.0 max diff on every param leaf, `goal` and `s`, both
      cores — checked against a reverted copy of the module loaded side by side);
      all 10 manager self-checks pass with `[10]` now sweeping all four local
      latents × both cores; 103 seam tests pass
      (`test_feudal_seams.py` + `test_smax_seams.py`), with the two end-to-end
      latent tests (goal reproducibility, full update) widened from
      `["centralized", "local"]` to every latent.
    - ⚠ **No training result** — every number here is a mechanism check. And note
      the `private` axis's motivation was measured at **N=12 homogeneous** agents
      (`obs_i` participation ratio only ~4 of 12); `smax_3m` is N=3 with a
      random-direction baseline of 2.82/3, and `2s3z`/`3s5z` have heterogeneous
      unit types, so a shared projection may not bind there. Read
      `goal_direction_count` on a `local_global` run before assuming it does —
      and read it **across the whole run**, since `local`'s collapse on the 12a
      arms happened by ~10M steps.
  - **All three critics keep reading `global_state` under every latent**
    (`trainer._values` / `_manager_values` / `_values_int`), so CTDE is intact in
    every mode. Only the manager's *directives* change.
  - **`centralized` is the default and is bit-identical to pre-change code** —
    verified against a `git worktree` at HEAD: 0.0 max diff on every param leaf,
    `goal` and `s`, for **both** cores. The extra `obs` argument is ignored there.
  - ⚠ **Not checkpoint-compatible** with any existing `feudal*` arm: the manager
    tree carries `f_enc_0`/`f_enc_1`/`f_gpre`/`f_goalhead` instead of
    `f_percept_0`/`f_percept_1`/`goal_head`, and `f_Mspace` is
    `(manager_hidden, goal_dim)` rather than `(manager_hidden, n_agents*goal_dim)`.
    `goal_dependence_probe._dims_from_checkpoint` keys off `f_enc_0` to tell them
    apart — without that it would infer `goal_dim // n_agents` for a local run.
- **Residual the shared encoder does NOT remove.** `obs_i` is egocentric but not
  *proprioceptive*: density sensors, `neighbor_fraction` and lidar all respond to
  teammates inside `sector_sensor_radius`. So `s_i` is agent *i*'s **view**, which
  teammates still perturb through channels agent *i* can perceive. Arguably the
  right notion of own-progress under partial observability, and a large reduction
  from uniform 1/N mixing, but it is not literally own-physical-state. Measure it
  with `--wrt positions` (finite-difference, because the lidar's `mjx.ray` is not
  usefully differentiable); even a perfectly obs-local manager scores < 1.0 there.
- ⚠ **This is necessary, not sufficient, and it is only observable at `alpha > 0`.**
  `intrinsic_coef` ships at 0.0, so nothing currently running depends on `r^I`.
- **`r^I` is TRANSITION-ALIGNED — fixed 2026-09-16 (plan:
  `plans/feudal_intrinsic_reward_timing_fix_2026-09-16.md`). The measurements
  below are what the fix was worth, and they are the reason it is hygiene rather
  than a result.** The training path is now
  `manager.worker_intrinsic_reward_aligned`:
  `r^I_t = mean_{k=0..c−1} d_cos(s⁺_t − s_{t−k}, g_{t−k})`, where `s⁺_t` =
  `Transition.next_state_latent` is the latent of the successor `a_t` actually
  produced, captured in `_env_step` **before** `_restart_done` rebinds
  `next_obs`. So `r^I_t` scores `a_t`, as `reward[t]` on that transition already
  did. ⚠ Encoding that latent *after* the reset cond is the one mistake that
  leaves every other check passing — it scores the teleport; pinned by
  `test_next_state_latent_is_the_true_successor_not_the_reset`, which asserts
  equality with `state_latent[t+1]` off done steps and inequality on them.
  - **Until then** `trainer._env_step` stored the *pre-action* `state_latent[t]`
    and `r^I_t = 1/c Σ_i d_cos(s_t − s_{t−i}, g_{t−i})` was built entirely from
    quantities fixed **before `a_t` was sampled** — exactly independent of the
    action on its own transition (verified bitwise), while `reward[t]` was that
    action's consequence. The two streams scored different actions
    (`plans/feudal_goal_reward_diagnosis_2026-09-09.md` §4). The paper's literal
    form survives as `manager.worker_intrinsic_reward` for the diagnostic only —
    **do not wire it back into the trainer.** Both are one call into the shared
    `_intrinsic_window`, differing in the endpoint and the offset range, so the
    episode-masking algebra cannot drift between them.
  - **Why it was worth doing anyway, given the table below: the BOUNDARY, not the
    magnitude.** Any endpoint derived by *shifting* the stored latents is exact
    in the interior (`r_new[t] == r_old[t+1]`, verified to 0.0) but must pay
    `r^I = 0` on the action that **ends** an episode — a standing bonus for
    terminating, the same shape as the `boundary_truncates` failure the MJX env
    removed. ~0.2% of transitions on a 1024-step `trunc` arm, ~2.4% on the
    ~43-step boundary-terminating baseline. Explicit successor capture is what
    covers those; a shift cannot.
  - **Cost, measured** (`mjx_12a_3o_trunc_1024`, n_steps=1048, n_envs=32, α=0.1,
    warm median of 7): collect **2.581 s → 2.562 s**, i.e. nothing — the extra
    encoder is a latent-only forward (`FeudalManager.__call__(...,
    latent_only=True)`, which skips the core and the goal head because `s` is
    upstream of the core in **every** latent variant, so it touches no carry and
    consumes no RNG). Peak GPU **1652 → 1751 MiB** (+6%), the new
    `(T,E,N,goal_dim)` buffer.
  - **α=0 is a STATIC no-op and is verified bit-identical to the pre-change
    code** (git-worktree A/B on the CPU stub env: all 18 rollout fields, all 23
    loss/metric keys and all 95 post-update param/optimizer leaves at 0.0 max
    diff). The parameter tree is unchanged, so **existing checkpoints load** —
    re-verified by loading trained `feudal_film` and `feudal_film_n01_local`
    manager params and confirming `latent_only` is bitwise the full forward's
    `s`. `next_state_latent` is a scalar placeholder at α=0 (the `action_mask`
    idiom).
  - **Post-fix probe reading** (`feudal_film_n01`/`n05` trial 0, T=128): the
    legacy-vs-corrected gradient cosine is **0.999951 / 0.998870** (≈0.57° and
    2.7°) and legacy-vs-shift is 0.999934 / 0.998776 — i.e. the fix landed where
    the table predicted, which is the implementation check. `corr(r^I legacy,
    corrected)` = **0.7213** against a lag-1 autocorrelation of **0.7210**, the
    mechanism stated below reproduced exactly. Causal dependence now demonstrated
    by **stepping the env** under two actions from one state (Δr^I ≈ 0.72); the
    old check edited a stored action while holding the stored states fixed, which
    could not show dependence — the legacy reward is a function of `(s, g)`
    alone. ⚠ At `--n-steps 128` against `max_steps: 1024` **no episode ends**, so
    the boundary arm is not exercised; use `--n-steps 1100`.
  - ⚠ **And with the boundary exercised, the terminal-credit correction is
    invisible in the gradient** (`--n-steps 1100 --n-envs 4`, 4 dones = 5 of 4400
    transitions): shift-vs-corrected reads **0.999998**, legacy-vs-corrected
    0.999947. That is the honest reading and it does **not** undercut the reason
    for the change — a per-transition gradient cosine measures how much the
    update moves *now*, not the incentive a systematic zero on the
    episode-ending action creates over a run, which is what the
    `boundary_truncates` episode showed can be self-sealing. Expect a larger
    share on the ~43-step boundary-terminating baseline arm (~2.4% of
    transitions vs 0.11% here); **unmeasured there.**

  **What the fix was worth**, measured 2026-09-14 *before* it landed.
  `algorithms/feudal_mappo_jax/intrinsic_timing_probe.py` measures what the
  misalignment cost the **update**, which is the decision-relevant quantity: it
  rebuilds the actor's first-epoch gradient (where the PPO ratio is exactly 1, so
  the gradient is exactly `Σ_t A_t ∇log π`) on one trajectory, same params and
  same critics, under both indexings, and reports the angle between them. Over 13
  trained arms (both 12a env groups × α ∈ {0.01, 0.1, 0.5} × 3 seeds):

  | α | gradient rotation from **fixing the timing** | rotation the **intrinsic term itself** causes |
  |---|---|---|
  | 0.01 | **0.08°** | 0.68° |
  | 0.1 | **1.2°** | 6.7° |
  | 0.5 | **3.6°** | 26.3° |

  So the fix was ~13% of the effect `r^I` is already having, and **it is not why
  the α>0 arms fail** — do not expect the correction to move return, and do not
  read a change in the α>0 arms as its consequence. The mechanism is measurable, not hand-waved: GAE
  integrates over `1/(1 − γλ) = 16.8` steps at γ=0.99/λ=0.95, and `r^I` has lag-1
  autocorrelation **0.53–0.73**, so shifting it one step leaves the advantage
  **0.986–0.998** correlated even though the raw reward streams correlate only
  0.53–0.73. Two corollaries: (1) it applies identically to every arm, so it can
  **never** bias a paired local-vs-centralized comparison — the α>0 pairs in the
  `local` measurement above are valid, not confounded; (2) `V^I` is not the
  problem either — `intrinsic_explained_variance` reads **0.96–0.99** on these
  arms, often above the extrinsic critic's, so the history-dependence of `r^I` is
  recoverable from the current state — which is why the intrinsic critic's inputs
  were deliberately left alone by the timing fix. ⚠ Measured at *trained*
  checkpoints at the *configured* (pre-anneal) α, i.e. the strongest case for it
  mattering; a full-trajectory claim would need the same probe early in training,
  and that is the one regime where the correction could matter more than recorded
  (the whole mechanism rests on an autocorrelation measured on trained nets).
  **Still unrun.** Acceptance for the latent change itself
  is **structural** (diag share 1.0); a return claim needs competitive extrinsic
  return against matched active-goal/alpha=0 and zero-goal controls, and per the
  diagnosis note a high cosine or a positive permutation gap does not meet that bar.
- Probe (its **positive control is mandatory** — a flat reading is otherwise
  indistinguishable from a dead metric):
  ```
  uv run python -m algorithms.feudal_mappo_jax.latent_locality_probe --control
  MUJOCO_GL=egl uv run python -m algorithms.feudal_mappo_jax.latent_locality_probe \
      --batches mjx_16a_4o_trunc_1024 --models feudal --trial 0 [--wrt positions]
  ```
  Train the fixed arm:
  ```
  uv run python train.py algorithm=feudal_mappo_jax env=mjx_16a_4o_trunc_1024 \
      model=feudal_film_local trial_id=0
  ```

**Goal collapse: read `goal_direction_count`, not `goal_pairwise_cos` alone.**
The headline "are the manager's goals unique per agent?" series is
`goal_direction_count` — the effective number of *distinct* goal directions
emitted at a single (timestep, env), averaged over the batch, in
`[1, min(n_agents, goal_dim)]`. **1.0 ⇒ one shared team goal; n_agents ⇒
mutually orthogonal per-agent directives.** It exists because the signed mean
cosine cannot distinguish genuine diversity from a collapse onto a single
**line**: goals splitting into `+u` / `-u` clusters average to ≈0 and read as
perfectly healthy while carrying one bit of information. `goal_pairwise_cos_abs`
(mean `|cos|`; ≳0.9 ⇒ collinear) is the cheap cross-check that says which of the
two a `goal_pairwise_cos` ≈ 0 is. All three are computed off one shared
`_agent_gram` (`mappo.py`); the count is the participation ratio
`(Σλ)²/Σλ² = N²/‖G‖_F²` of that Gram — unit-norm rows make the numerator exact,
so it costs a Frobenius norm, not an eigendecomposition, over the (T, E, N, N)
stack. Pinned by `test_goal_direction_count_reads_the_collapse_cases`.

⚠ **The healthy value is not `n_agents`** — compare against the random-direction
baseline `N² / (N + N(N−1)/goal_dim)`, since `N` unit vectors in `goal_dim`
dimensions are only near-orthogonal when `goal_dim >> N`. At `mjx_16a_4o`
(`n_agents=16`, `goal_dim=16`) that baseline is **8.26**, and a 65k-step run
measures 8.25 (with `goal_pairwise_cos` ≈ −0.02, `|cos|` ≈ 0.204 ≈ the random
`sqrt(2/(pi*goal_dim))`) — i.e. maximally diverse for this width, *not* half
collapsed. Read the series as a **trend**: a drift downward toward 1 is the
collapse. Because the ceiling is set by `goal_dim`, the count is only comparable
across runs that share it.

Measured at 131k steps on `mjx_16a_4o` (`goal_dim=16`, `c=10`): erank 14.5 → 11.8,
`goal_pairwise_cos` ≈ ±0.02, `d_cos_var` ≈ 0.063, `valid_fraction` 0.88 → 0.76 —
all healthy. **`manager_explained_variance` is negative (≈ −3) and that is not
yet diagnosable**: `m_ret - manager_value` is identically the advantage, so the
metric is `1 - var(adv)/var(ret)`, and at this budget the task reward is ~0 (the
env's own baseline needs 1e8 steps), leaving `var(ret)` tiny and the ratio noise.
For calibration the flat `mappo_jax` baseline also starts at EV −1.34 and only
reaches 0.97 after 1e8 steps. Judge `V^M` on a real-length run, not a smoke.

`valid_fraction` interacts with episode length: horizons straddling a boundary
are masked out, and `mjx_16a_4o` episodes are short (~43 steps, boundary contact
terminates), so a large `c` starves the manager. Watch it when changing `c`.

- **Seam tests**: `algorithms/tests/test_feudal_seams.py` (123 tests, CPU stub
  env, no MJX — fast and *deterministic*, unlike an MJX rollout). They pin the
  joints where a mistake is silent: the in-scan ring equals the `pool_goals`
  oracle including done-masking; the stored goals are reproducible by re-scanning
  the manager over the stored global states (the property `manager_update` relies
  on); agent-major flattening pairs each agent's obs with its own goal; and the
  **PPO ratio is exactly 1** before any update. Run:
  `uv run pytest algorithms/tests/test_feudal_seams.py -q`
### Intrinsic reward (`intrinsic_coef`) — DUAL ADVANTAGE STREAMS, shipped OFF

**alpha is a GRADIENT FRACTION, not a reward coefficient.** The extrinsic and
intrinsic streams each get their own critic, their own GAE and their own
normalization to unit std, and meet only in `ppo_update` as
`adv = adv_ext + alpha_t * adv_int`. `intrinsic_coef` still defaults to 0.0
(the no-intrinsic control); the live arms are `conf/model/feudal_n01.yaml`
(alpha=0.1) and `feudal_n05.yaml` (0.5), both with `intrinsic_anneal: linear`.

- **⚠ The `feudal_a01` / `feudal_a05` arms are DEAD RUNS — do not cite them as
  evidence about hierarchy.** They used the old reward-level fold
  (`reward += alpha * r^I`) and scored **0–4** against a ~470 ceiling on all
  three of `mjx_16a_4o` / `_trunc` / `_partition`, 3 seeds each. Cause, measured:
  `r^I` is ~0.155/step and near-flat while the extrinsic reward is
  **~5e-05/step** until boxes start moving, so even alpha=0.1 put **300–600×**
  (alpha=0.5: **1100–1700×**) more magnitude on the intrinsic term through
  exactly the window where learning had to start — the alpha=0 arm's
  `train_reward` climbs 7.9e-05 → 3.8e-03 → 0.122 over 3e5 → 1e6 → 5e6 steps,
  and the alpha>0 arms sat frozen at ~5e-05 straight through it.
- **The failure was an absorbing state, not a transient exploration phase.**
  `r^I = d_cos(s_t − s_{t−i}, g_{t−i})` uses the manager's own outputs, and the
  manager's objective (`mappo.py`, `-mean(cos * m_adv)`) is the *same cosine*, so
  worker and manager raise it together **through the environment** — no gradient
  crosses the detach, which is why FuN's detach rule does not block it. Measured
  over the full 1e8 steps: `d_cos_mean` 0.0003 → 0.125 and `intrinsic_reward`
  0.0002 → 0.138, monotone, with 8× headroom still unspent at the end, while task
  reward stayed at ~1e-3. Three effects compound: advantage normalization makes
  it a **variance-share** contest (not an additive bonus); `r^I` is trivially
  climbable where box delivery is not; and it never saturates.
- **The old calibration was wrong in kind, and no value in a 0/0.1/0.5 sweep
  could have worked.** It divided by the *converged* baseline rate (~0.4/step)
  and read a reassuring 17–19%. The rate that governs whether learning *starts*
  is ~5e-05/step, against which the same alpha is ~1500:1 — so the smallest
  nonzero member of that sweep was already ~300× too large.
- **Anneal (`intrinsic_anneal`)**: `"linear"` (default) decays alpha to exactly 0
  across `n_total_steps`; `"none"` holds it. Required for asymptotic correctness
  — at constant alpha that fraction of the worker's gradient points at a
  task-irrelevant objective forever. `progress` must reach `update_fn` as a
  **traced** jnp scalar (a python float recompiles the jitted update every
  iteration); it is derived from `update / num_updates`, so resume is correct for
  free. `alpha_current` is logged.
  - ⚠ The schedule is relative to the **current** `n_total_steps`, so *extending*
    a finished run restarts the decay partway rather than continuing it. Verified:
    a 196608-step run annealed 0.1 → 0.00104, and resuming it with
    `n_total_steps=393216` picked up at `0.1*(1 − 96/192) = 0.05` and decayed to
    0.00052. Each run's schedule is internally exact; the seam is a step. Only
    matters if you extend an intrinsic arm — set `intrinsic_anneal: none` for the
    extension if you need a continuous schedule.
- **alpha=0 is a STATIC no-op**, gated on a python-level `config.intrinsic_coef
  != 0.0`: no `intrinsic_critic_ts` is built (it is `None`, a valid empty JAX
  pytree, like the mlp core's carry), no `r^I` is computed, and the msgpack
  format is unchanged — **existing `feudal_a0` checkpoints still resume**
  (verified against `mjx_16a_4o/feudal_a0/1`). Verified **bit-identical** to the
  pre-change code across the rollout, both bootstraps, all 18 loss keys and the
  post-update actor/critic/manager params (StubEnv, one seed; the StubEnv is pure
  CPU JAX so unlike MJX it *is* reproducible across processes — the comparison
  was run in two git worktrees).
- **New diagnostics** in `ppo_update`'s metrics: `alpha_current`,
  **`adv_ext_std_raw` / `adv_int_std_raw`** (the raw, pre-normalization advantage
  scales — the pair that reads the defect this path exists to fix),
  `intrinsic_explained_variance`, `intrinsic_value_loss`. Note the *advantage*
  gap is much smaller than the *reward* gap (a smoke run measured 0.30 vs 0.79 at
  init) because the critic centers the extrinsic stream — but it widens as the
  critic improves (0.0067 vs 0.78 by the end of that run, ~116:1), which is
  precisely why the normalization has to be per-stream and permanent.
- `MAPPOCritic` is reused for V^I (writing another critic *class* is what
  `manager.py` rules out); it is keyed off `jax.random.fold_in(rng, 2)` so adding
  it perturbs no pre-existing init. There are now **five** msgpack sites in
  `run.py`, all routed through `_intrinsic_tree` so they cannot disagree.
- **Scale against `intrinsic_reward_abs`, never `intrinsic_reward`.** The signed
  mean is ~-4e-4 — cosines cancel about zero — so it reads as "no intrinsic
  signal" while the per-step term is 0.152. Both are logged; the raw (unscaled)
  `r^I` is recorded, so the number is comparable across alphas.
- **`worker_goal_column_ratio`** (in `ppo_update`'s metrics) is the per-input-dim
  RMS of the goal columns of the worker's first Dense over the obs columns, ~1.0
  at orthogonal init. `FeudalWorker` fuses by **concatenation**, so the worker
  can disconnect the hierarchy simply by driving those columns to zero — the
  degeneracy FuN avoids with a bias-free bilinear `U(obs) @ phi(g)`. Nothing else
  logged would show it: goals stay unit-norm and diverse and the manager's own
  loss keeps improving. A decay toward 0 is the trigger to set `goal_embed_dim`
  or change the fusion. (0.9998 → 0.951 over 64 updates — far too short to read
  anything into.)
  - ⚠ **It is CONCAT-ONLY, and for a year it silently logged NaN on every FiLM
    run.** It slices the goal block as `kernel[obs_dim:]`; under FiLM the first
    Dense has input width `obs_dim` exactly, so that is an empty `(0, hidden)`
    array and `jnp.mean` of nothing is NaN. Measured: NaN at **every** logged
    point in **84 of 84** feudal trials across `mjx_12a_3o_trunc_1024` and
    `mjx_12a_3o_partition_1024` — i.e. the whole default arm family trained for
    1e8 steps with no logged signal whatsoever about whether the manager→worker
    channel was connected. `ppo_update` now routes on `config.worker_fusion`
    (concat → this ratio; film → `_film_goal_metrics` below) and the concat
    function **raises** on a FiLM kernel rather than returning NaN, because
    reaching it means the routing is wrong.
- **FiLM goal-influence metrics** (`mappo._film_goal_metrics`, `worker_fusion:
  film` only): `worker_film_gain_rms`, `worker_film_shift_ratio`,
  `worker_tanh_saturation`, `worker_goal_action_delta`. Forward-only on arrays
  `ppo_update` already holds — one apply with the `diagnostics` collection
  mutable plus one on a re-paired goal — and ungated like every other manager
  diagnostic.
  - **What they are.** The modulation is `h ← (1+γ(w))·h + β(w)` before each
    Tanh. `gain_rms` = RMS(γ) over batch × units × both layers: the fractional
    swing of each unit's gain around 1, **dimensionless so it needs no
    denominator** — precisely what made the concat ratio misreadable twice.
    `shift_ratio` = mean over layers of RMS(β_i)/RMS(pre_i); β is *additive* so
    it does need the scale it is added to, and **per-layer rather than pooled**
    because the two layers' preactivations differ ~4.5x in scale and a pooled
    ratio is dominated by the larger. `tanh_saturation` = fraction of modulated
    preactivations with |tanh| > 0.95. `goal_action_delta` =
    RMS(μ(obs,w) − μ(obs,w′))/RMS(μ(obs,w)) with w′ a **half-batch roll** (a
    goal from an unrelated (timestep, env, agent)) — deliberately NOT the
    agent-axis roll of `eval_gap_permuted`, which degenerates to a no-op once
    the manager has collapsed to one team direction, so this series stays
    meaningful exactly where the permutation nulls stop being.
  - **`gain_rms`/`shift_ratio`/`action_delta` are exactly 0.0 at init** (FiLM's
    zero-init) and exactly 0.0 on a `zero_goal` arm forever (γ/β Dense are
    bias-free, so `γ(0)=β(0)=0`). That is a **built-in positive control**:
    verified 0.000 on all three `feudal_film_zerogoal` seeds. `saturation` is
    correctly nonzero there — it is a property of the trunk, not of the goal.
  - **Reference values**, trained `mjx_12a_3o_trunc_1024`, 3 seeds each:

    | arm | gain_rms | shift_ratio | saturation | action_delta |
    |---|---|---|---|---|
    | `feudal_film` (α=0) | 0.72–0.76 | 0.164–0.166 | 0.50–0.52 | 0.91–1.09 |
    | `feudal_film_n05` | 0.90–1.06 | 0.187–0.211 | 0.53–0.54 | 1.22–1.33 |
    | `feudal_film_local` | 2.30–2.57 | 0.174–0.198 | 0.49–0.66 | 0.99–1.14 |
    | `feudal_film_n01_local` | 3.89–5.22 | 0.249–0.306 | 0.75–0.85 | 0.63–1.28 |
    | `feudal_film_zerogoal` | **0.000** | **0.000** | 0.56–0.61 | **0.000** |

  - ⚠ **THEY REFUTE "the worker ignores the goal".** Every centralized arm has
    `eval_gap_zeroed` ≈ 0 — which reads as a disconnected channel — while running
    a gain that swings ±75% and moving its action by ~100% of the action's own
    magnitude when handed a different goal. The channel is wide open; the
    behaviour it induces is orthogonal to return. **The two readings call for
    opposite fixes** (change the fusion / add goal-following pressure, vs change
    the objective or the task), and with only the eval gaps logged the wrong one
    is the natural inference — `feudal_film_n05` is the counterexample: forced
    goal-dependence (`real − zeroed` = +35.4, `dir_count` 9.10) at a return of
    **36.2** against 151.9 for α=0.
  - They also give `feudal_film_n01_local`'s collapse a mechanism: at γ RMS ~4.6
    the trunk runs **75–85% saturated** against 16–20% for the same weights with
    the goal zeroed (layer-0 preactivation RMS 7.2–9.1 vs ~1.4). Zeroing that
    goal does not delete a directive, it relocates the trunk to a never-trained
    operating point — which is why return falls to ~1.5. A high saturation is
    not itself the pathology (`zerogoal` sits at 0.56–0.61); the **gap** between
    modulated and unmodulated is.
  - ⚠ **Implementation note that is easy to re-break:** γ/β are exposed by
    `self.sow("diagnostics", ...)` inside `FiLM.__call__`, **guarded on
    `is_initializing()`**. `sow` fires under `init` too, and `init_worker`'s
    return *is* the train state's `params`, so without the guard a
    `"diagnostics"` collection reaches `create_train_state`, the optimizer and
    every msgpack site — incompatible with all 84 existing FiLM checkpoints.
    Sown rather than recomputed from the kernels so the diagnostic cannot drift
    from the forward pass it describes, which is how the concat ratio managed to
    report NaN for 84 runs unnoticed. Pinned by 6 seam tests.
- Rollout-level series (`train_reward`, `intrinsic_reward`,
  `intrinsic_reward_abs`) are appended to the same per-update stats dict as the
  losses. Note `train_reward` is the **rollout** team reward; the `reward` series
  remains the deterministic **eval** return.

### Pure-intrinsic worker (`worker_objective: intrinsic_only`) — wired, UNRUN

`model_params.worker_objective` selects what the **worker's** policy gradient
optimizes. `"mixed"` (default) is the original and is **byte-identical** to the
pre-change code; `"intrinsic_only"` drops the extrinsic advantage from the actor
entirely, so the worker's only job is to follow the manager's goals:

```
mixed (every other arm):  adv = adv_ext + alpha_t * adv_int
intrinsic_only:           adv = adv_int
```

All task pressure then sits with the **manager**, whose transition PG is already
weighted by the extrinsic advantage under `manager_gamma` — so task return can
only be reached *through* the goal channel. That is the classic Dayan–Hinton
feudal contract and the one configuration in which the hierarchy is load-bearing
by construction. It exists because under `mixed` every measurement says the goals
are decorative (`eval_gap_permuted` ≈ 0 on 40 of 45 non-control trials,
`gap_zeroed` systematically **negative**, and the best arm in
`mjx_12a_3o_trunc_1024` is `feudal_film_zerogoal_dilated` at 293.7, whose goals
are provably disconnected). The opposite extreme had never been run.

- **Only the actor's advantage changes.** The extrinsic GAE, the worker critic's
  regression and `explained_variance` all still run. The critic is deliberately
  kept trained: the param tree stays **shape-identical** to the matched `mixed`
  control (so checkpoints remain interchangeable — the same reason `zero_goal`
  zeroes at the *input*), and the extrinsic return stays a live diagnostic
  (measured on the smoke run: EV climbs 0.22 → 0.94 while the actor never sees it).
- **⚠ alpha is NOT a coefficient here, and that is a trap with a guard.**
  `adv_int` is already unit-std, so an alpha factor would be a uniform rescale of
  the whole actor gradient — which the shipped `intrinsic_anneal: linear` would
  drive to **exactly 0**, silently deleting the worker's entire objective over the
  second half of a run while every logged loss stayed healthy (the self-sealing
  shape of the `boundary_truncates` and `VARIANTS`-enum bugs). The expression
  therefore carries no alpha at all, and the arm **raises** unless
  `intrinsic_anneal: none`. `intrinsic_coef` must still be **nonzero** — it is the
  static gate that builds V^I, captures `next_state_latent` and computes `r^I` at
  all — but its *value* is inert, so the group ships `1.0`.
- **All four rules live in one function**, `mappo.validate_worker_objective`,
  called from **both** `trainer.make_train` (so the seam tests and any direct-
  config path are covered) and `run.py.__init__` (so a launched run fails at
  construction). Three **raise** (unknown value; `intrinsic_coef == 0`;
  `intrinsic_anneal != "none"`); the fourth **warns** — a non-local
  `manager_latent`, membership tested against the existing `manager.LOCAL_LATENTS`
  tuple rather than a hardcoded list. The warning is **latent-goal-space only**
  (skipped for any `GROUNDED_GOAL_SPACES` member since 2026-09-24): under a
  grounded goal `r^I` is scored on `env.goal_state`, agent *i*'s own position,
  which is agent-local for every `manager_latent`, so its premise is false there.
- **⚠ The latent is a PRECONDITION, not a preference.** `r^I` scores
  `d_cos(s_t[i] − s_{t−k}[i], g_{t−k}[i])`, so whatever `s[i]` responds to is what
  agent *i* is paid for. Under `centralized` that is **not agent-local at all**
  (measured diag share of `d s[i]/d obs_j` = 0.0631 against a uniform 1/N of
  0.0625), which is a confound under `mixed` and **fatal** here — the worker's
  whole loss would be a team-aggregate signal it does not control.
  `local_private` is the only latent measured to have both locality (diag share
  exactly 1.0) and restored row diversity (`s` 8.72, `g` 9.24). Hence the single
  shipped arm, `conf/model/feudal_film_intrinsic_only_local_private.yaml`
  (`defaults: [feudal_film_local_private, _self_]`, i.e. one key apart from its
  own control). The combination stays *runnable* on a non-local latent because it
  is the direct contrast that tests whether locality is what matters.
- **⚠ THE PRIMARY RISK, and it is self-sealing.** Manager and worker share `d_cos`
  as an objective and can climb it **through the environment** (no gradient
  crosses FuN's detach). Measured on `feudal_film_n01_local` they do exactly that:
  the manager freezes on one direction and the worker drives its own observation
  along it — `goal_direction_count` 1.45 against a random baseline of 8.93, task
  return 1.4. With `intrinsic_only` there is nothing else in the worker's
  gradient, so that degenerate joint solution is **optimal for the worker**. The
  counter-pressure is that the manager's PG is weighted by the extrinsic
  advantage, so the manager is not free to collapse; the residual feedback risk is
  that a perfectly obedient worker drives `d_cos → 1` everywhere, `d_cos_var → 0`,
  and the manager's own gradient flattens. **Watch, in this order, across the
  WHOLE run** (`local`'s collapse landed by ~10M steps and was permanent):
  `d_cos_var` (≲1e-3 ⇒ cosine constant) → `goal_direction_count` (vs 8.93 at
  N=12/`goal_dim`=32) → `state_latent_erank` (≲1.5) → only then return.
- **⚠ THE GOAL-DEPENDENCE PROBE CANNOT GRADE THIS ARM.** `gap_zeroed`,
  `gap_constant` and `gap_permuted` will all be large **by construction** (the
  worker is *defined* to depend on the goals), and so will `d_cos_mean`. The
  three-condition acceptance test in `conf/model/feudal_film.yaml` is therefore
  vacuous here. Acceptance is the **between-arm** return comparison, which this
  file already names as the only test that can rule an arm *in*: vs
  `feudal_film_local_private` (the matched one-key control), vs
  `feudal_film_zerogoal` (the goal-free floor), vs `mlp` (flat MAPPO, 272 / 295 on
  the 12a groups). `worker_objective` is recorded in
  `evaluate_goal_dependence`'s provenance dict for that reason.
- **New stats keys `adv_ext_weight` / `adv_int_weight`** — the coefficients that
  actually multiply the two normalized streams (`1.0`/`alpha_t` under `mixed`,
  `0.0`/`1.0` under `intrinsic_only`). They exist because `alpha_current` alone is
  **misleading** here: it is logged and inert. (New keys are safe on resume —
  `TrainingStatsTracker.to_dict()` left-pads short series with NaN.)
- **Verified end-to-end** on `mjx_12a_3o_trunc_1024`: composition resolves
  (`worker_objective: intrinsic_only`, `worker_fusion: film`, `manager_latent:
  local_private`), all three raising guards fire before launch, the warning fires
  and still runs, train + **resume** work, and the checkpoint carries `film_0` +
  `f_Mspace_agent_kernel` (i.e. the intended arm trained). Smoke stats at 268k
  steps: `adv_ext_weight` 0.0, `adv_int_weight` 1.0, `intrinsic_reward_abs`
  0.10→0.13 (**not** identically 0.0 — the `# @package _global_` tell),
  `d_cos_var` 0.033→0.051, `goal_direction_count` 8.87→9.00. 11 new seam tests;
  **144 pass** across `test_feudal_seams.py` + `test_smax_seams.py`.
- **⚠ No training result.** Every number above is a mechanism check. Plan:
  `plans/feudal_pure_intrinsic_worker_2026-09-17.md`.
  ```
  uv run python train.py algorithm=feudal_mappo_jax env=mjx_12a_3o_trunc_1024 \
      model=feudal_film_intrinsic_only_local_private trial_id=0
  ```
- **Waypoint arms: `feudal_intrinsic_waypoint` / `feudal_intrinsic_local_private_waypoint`**
  (`defaults: [feudal_film_waypoint, feudal_intrinsic{,_local_private}, _self_]`
  + `goal_horizon: 80`). Here `r^I` is the saturating waypoint reward (≤ 1 per
  latch block), so the worker's only job is to reach the manager's waypoints and
  it cannot farm `r^I` indefinitely the way it can under the latent `d_cos`.
  - **They are FiLM**, although `feudal_intrinsic` is concat: it sets no fusion
    key, so `feudal_film_waypoint`'s `worker_fusion: film` survives.
    `feudal_intrinsic` re-includes `feudal` after the waypoint chain. That is
    harmless while `feudal.yaml` sets only `hidden_dim`.
  - **Not a one-key delta** from the `feudal_film{,_n01}_waypoint_c80` controls:
    the resolved configs also differ in `ent_coef` 0.01 → 0.1 (from
    `feudal_intrinsic`). A gap is objective + entropy, unseparated.
  - **The latent pair is not a locality contrast** on a grounded goal space. It
    changes only what the manager reads. On `mjx_1a_3o_111_1024_gs` that is the
    22-dim compact state for `centralized` against the 40-dim egocentric obs for
    `local_private`.
  - Verified 2026-09-24 on `mjx_1a_3o_111_1024_gs`: both compose, train,
    checkpoint and resume. The checkpoints show FiLM (`film_0`, actor `Dense_0`
    `(40, 168)`), a 2-wide goal head, an `intrinsic_critic` tree, and the
    expected manager tree per latent. Stats show `adv_ext_weight` 0.0 /
    `adv_int_weight` 1.0 and `valid_fraction` 0.0115 ≈ 1/80. **No training
    result.**

### Recurrent manager (`manager_core: dilated_lstm`) — works end-to-end

```
uv run python train.py algorithm=feudal_mappo_jax env=mjx_16a_4o model=feudal \
    trial_id=0 model_params.manager_core=dilated_lstm
```

Verified at production scale (`n_steps=1024`, `n_envs=32`): trains, resumes from
checkpoint, no OOM. `mlp` stays the default so goal-mechanism effects remain
attributable separately from recurrence effects.

- **The load-bearing requirement is that `manager_update`'s recompute reproduces
  the rollout's carry exactly.** The PG needs `g_t(theta)` differentiable, so it
  re-derives `(goal, s)` from the stored global states; with a stateless core
  that is a pure function, but with the LSTM it is a `lax.scan` that must match
  `trainer._env_step` on **both** conventions: start from the same
  zero-initialized pools, and apply the per-env reset *after* emitting the
  step's goal. Get either wrong and the manager is optimized for a policy that
  never acted — silently, with healthy-looking losses.
  `test_goals_are_reproducible_from_stored_states` is parametrized over both
  cores and is what catches it.
  - The two paths build their carries from **different rngs** and only agree
    because `dilated_lstm_carry` zeroes the pools and ignores the key. That
    implicit contract is pinned by `test_dilated_lstm_carry_is_deterministic`.
  - `DilatedLSTMState.t` is a single shared counter with **no env axis**, so it
    is deliberately not reset per env: a mid-rollout reset leaves that env at an
    arbitrary dilation phase. Harmless (the phase is arbitrary anyway), but both
    paths must do the same thing, and they do.
- **No `jax.checkpoint`/remat, and that is a measured decision.** The plan
  assumed the BPTT would need it (~738 MB of f32 residuals at `T=1024, E=32,
  r=10, H=256`). Measured, that estimate is wrong: peak memory is dominated by
  the scan's own `(T, E, N, goal_dim)` `goal`/`s` outputs, not the carry, and
  XLA already avoids storing the latter naively. Over 5 grad calls at `T=1024,
  E=32`:

  | hidden | remat | no remat |
  |---|---|---|
  | 256  | 129.3 ms / 793.5 MiB  | 118.8 ms / 848.2 MiB |
  | 1024 | 315.4 ms / 2783.9 MiB | 314.2 ms / 2789.3 MiB |

  ~9% slower for ~6% memory at the shipped width, and a wash on both axes at 4x
  the width. Re-measure before adding it back if `n_steps` or the carry grows a
  lot.

- **Still open**: no full-length run of any feudal arm has been done, so nothing
  is known about whether the hierarchy *helps* — every number in this section is
  a mechanism check, not a result. The `intrinsic_coef` ablation (0 / 0.1 / 0.5)
  and any `mlp` vs `dilated_lstm` comparison both need full-length runs.
  `view()`'s rendering path has not been exercised since the goal plumbing
  landed.
  - Self-check (8 groups: shapes/unit-norm, dilation writes group `t%r` only and
    covers all `r`, `jit`+`lax.scan` carry threading, `vmap`, mlp core is
    stateless + an unknown core raises, goal-semantics incl. done-masking, a
    manager→worker handshake through `bind_goal`+`sample_action` for both action
    spaces, and gradient routing under the detach rule — target arm zeroed, goal
    arm live, `r^I` fully detached, `f_Mspace` still trained). All pass:
    `uv run python -m algorithms.feudal_mappo_jax.manager`

## SMAX (JaxMARL StarCraft) — `environments/smax/smax_env.py`

**Both JAX stacks run on SMAX** (`EnvironmentEnum.SMAX = "SMAX"`, env groups
`conf/env/smax_3m.yaml` and `conf/env/smax_5m_vs_6m.yaml`):

```
uv run python train.py algorithm=mappo_jax        env=smax_3m model=mlp    trial_id=0
uv run python train.py algorithm=feudal_mappo_jax env=smax_3m model=feudal trial_id=0
```

`SMAXAdapter` presents SMAX under the **same functional array contract** as the MJX
envs, so the trainers were extended rather than forked — no dict ever reaches them. (An
older JaxMARL dict-API path existed in commit `f074a3a` and was deleted in `e78638b`; it
predates the truncation bootstrap, eval, checkpointing and the whole feudal stack, so it
was **not** resurrected.)

- **`jaxmarl` is a dependency again** (`pyproject.toml`), pinned **`>=0.2.0`**, and that
  floor is load-bearing — see the next bullet. `environments/smax/_compat.py` restores
  the `jax.tree_map`/`jax.tree_leaves` aliases removed from JAX's top level in 0.9.x
  (this repo runs 0.10.2). It is **inert under 0.2.0** — a source-wide grep of the
  installed package finds **0** references to any removed `jax.tree_*` alias — but it is
  guarded (`if not hasattr`) and idempotent, so it is kept as insurance and costs
  nothing. Delete it only if the `>=0.2.0` floor is enforced some other way. ⚠ If you
  ever drop back to a 0.0.x jaxmarl, note that **an import smoke test does NOT prove the
  shim is unnecessary**: `import jaxmarl`, `jaxmarl.make(...)` and `env.reset(key)` all
  succeed without it; only a call that walks a pytree (`step_env`, `get_avail_actions`)
  raises. Import `_compat` **before** jaxmarl.
- **⚠ THE `jaxmarl>=0.2.0` FLOOR IS A CORRECTNESS PIN, and `>=0.0.2` silently resolved
  to a 2023 release (found 2026-09-15).** jaxmarl 0.0.3–0.1.0 hard-pin
  `jax==0.4.17.*` / `jaxlib==0.4.17.*`, which jax 0.10.2 cannot satisfy, so uv
  backtracked past all of them to **0.0.2 (Nov 2023)** rather than failing. 0.2.0
  requires only `jax>=0.4.25` and resolves cleanly. The two disagree on SMAX's per-unit
  observation features:

  | | `unit_features` (before the 6 unit-type bits) | per unit |
  |---|---|---|
  | 0.0.2 | `health, position_x, position_y, last_action, weapon_cooldown` | 11 |
  | >=0.0.3 | `health, position_x, position_y, last_movement_x, last_movement_y, last_targeted, weapon_cooldown` | 13 |

  `own_features` is **unchanged**, so `state_size = (own+2)*n_units` and `action_dim`
  match across versions and **only `obs_size` moves**: at `10m_vs_11m` (20 other units)
  270 vs **230**, at `3s5z` 205 vs 175, at `5m_vs_6m` 140 vs 120, at `3m` 75 vs 65.
  - **How it presents**: loading a checkpoint trained under the other version raises a
    bare `flax.errors.ScopeParamShapeError` at the first layer that eats raw `obs` —
    `manager/f_enc_0` for the `local*` latents (which is *before* the actor, so the
    error names the manager and looks like a latent/model-group problem), or the FiLM
    actor's `Dense_0` otherwise. The `global_state_dim` and `action_dim` agreeing
    exactly is the tell that the **scenario is right and only jaxmarl differs**.
  - **The dangerous direction is training, not loading.** From scratch nothing crashes:
    the run trains happily on a narrower observation and its curves are simply not
    comparable to any arm trained elsewhere. Before this was found, `smax_3m/feudal` and
    `smax_3m/feudal_film_local_global` had been trained locally at obs 65 while every
    `smax_10m_vs_11m` / `smax_3s5z` / `smax_5m_vs_6m` arm came off the cluster at
    270 / 205 / 140. **Those two `smax_3m` arms are dead** — retrain them under 0.2.0.
  - Diagnose from the checkpoint, which cannot lie: `actor/params/.../Dense_0/kernel` is
    `(obs_dim, hidden)` under `worker_fusion: film` and `(obs_dim + goal_dim, hidden)`
    under concat; `critic/params/Dense_0/kernel` is `(global_state_dim, 2*hidden)`.
    Solve `obs = u*(n_allies-1+n_enemies) + o` and `gs = (o+2)*n_units` for `u` — 13 is
    a >=0.0.3 checkpoint, 11 is 0.0.2.
  - Unrelated but found the same way: `smax_3s5z/mlp` and `smax_5m_vs_6m/mlp` hold
    **concat-feudal** checkpoints (the actor tree has `MAPPOActor_0` and `Dense_0` is
    `obs+goal_dim` wide), not flat `mappo_jax` actors — i.e. they are
    `algorithm=feudal_mappo_jax model=mlp` runs that overwrote the baseline directory,
    exactly the footgun flagged below. They are **not** baselines. `smax_3m/mlp` is a
    genuine flat actor.
- **Four contract mismatches the adapter reconciles**, each of which is silent if got
  wrong:
  1. **`step_env`, not `step`.** `MultiAgentEnv.step` auto-resets on done; the collector
     already restarts finished envs itself, so using `step` would double-reset *and*
     make the truncation bootstrap value the post-reset observation instead of the true
     successor.
  2. **`terminated` / `truncated` split.** SMAX gives only `dones["__all__"]`, and
     `is_terminal = all_allies_dead | all_enemies_dead | (time >= max_steps)`. The
     adapter splits that flag on `battle_over` rather than recomputing the predicate, so
     `terminated | truncated` is **exactly** the env's own done. The bootstrap design
     in this file depends on the two being distinct.
  3. **The RNG rides in the state.** SMAX's `step` is key-first (the heuristic enemy is
     stochastic); the contract's `step(state, actions)` has nowhere to pass one, so
     `SMAXState` carries a key and splits it each step.
  4. **`info` is empty in SMAX**, so `task_reward`, `active` and `won_episode` are all
     synthesized. All allies share one team scalar (`compute_reward` returns one value
     per team), so any agent's entry *is* the team reward.
- **`SMAXState` must stay a flat, `jnp.where`-able pytree** with a leading `n_envs` axis
  on every leaf — the collector's `_restart_done` does a `jax.tree.map` select of
  reset-vs-current state across it. Nothing static lives on that dataclass.
- **`env.agents` is the allies only** under `HeuristicEnemySMAX`; the enemy team is the
  built-in heuristic. `self.agents` is fixed once at construction and is the single
  ordering for every stack/un-stack (obs, actions, rewards, masks, alive flags) —
  agent-major, matching the trainers' `obs.reshape(b * n_agents, obs_dim)`.

### Three optional env hooks (inert for every other env)

These were added to **both** `mappo_jax` and `feudal_mappo_jax` — the two are
near-copies and must stay in sync. Each is gated on a **static** (trace-time) check, so
an env that has none takes byte-identical code paths to before they existed. Verified:
a hookless CPU stub run through init → collect → update gives **bit-identical** losses,
`param_sq_sum` and eval on both stacks vs the pre-change code (two git worktrees; MJX
itself is not reproducible across processes, hence the stub).

1. **`env.avail_actions(state) -> (n_agents, action_dim)`** — legal-action masking.
   Switched on by `hasattr(env, "avail_actions")`. `network.py` already had the whole
   masking path (`_masked_logits`, the `action_mask=` kwargs on `sample_action` /
   `evaluate_action`) as **dead code**; this wires it, it did not rewrite it.
   - The mask is fetched from the **pre-step** state and **stored** in
     `Transition.action_mask`. The stored mask must be the one that sampled, or the PPO
     ratio compares two different distributions — same class of bug as the feudal
     pooled-goal storage rule.
   - Envs without the hook store a **scalar placeholder** (stacked by `lax.scan` to
     `(n_steps,)`), not a real all-ones array: `ppo_update` switches on
     `action_mask.ndim == 4`, mirroring the `reward.ndim == 3` idiom, and a genuine
     all-ones buffer would cost `(T, E, N, A)` floats for nothing.
   - ⚠ **`eval_fn` and `view()` need the mask too.** A deterministic `argmax` over
     unmasked logits will select an illegal action outright, so eval would score a
     policy the env never runs.
2. **`env.global_state(state)` + `env.global_state_dim`** — the centralized critic's
   input. SMAX ships a real world state (absolute unit features): **72 dims at 3m against
   195 for the concatenated observations**. Without the hook the global state is
   `obs.reshape(n_envs, -1)`, exactly as before. Two things this touches that are easy
   to miss: the truncation bootstrap must read the successor state **before** the reset
   `lax.cond`, and `collect_fn` now **binds** the scan's final env state (it used to
   discard it) because the last-value bootstrap needs it.
   - The rule lives in **one** helper, `trainer.global_state_dim(env)`, used by both
     `make_train` and `run.py`'s checkpoint reload — `from_bytes` needs an
     exactly-shaped target tree, so a disagreement there is a load failure.
   - In the feudal stack this is also the **manager's** joint-state input. SMAX was
     for a long time the only env that supplied one: the egocentric MJX observations
     give the manager *no shared frame*, so it must localize agents first, whereas
     SMAX's world state is already a shared frame. `MultiBoxPushMJX` can now publish
     one too — see "Compact global state" below — though note the measured caveat
     there: on MJX the hook buys exactness and width, not information.
3. **`info["active"]` — dead units.** Nothing new was built: `Transition.active_mask` and
   its masked means in `ppo_update` were added for `SyncMacroMJX`'s staggered starts and
   apply verbatim. Dead units are dropped from the policy loss, the entropy loss and the
   per-agent critic head. All-ones for every other env keeps those reductions exact plain
   means. (Note SMAX itself still *pays* dead agents their team reward — "to allow for
   noble sacrifice" — which is unaffected: this masks the policy gradient, not credit.)
   - The mask is computed from the **pre-step** state, because the transition stores the
     pre-step obs/action: an agent that acted and then died in that same step still
     contributed a real decision and keeps its credit.
   - Measured: a dead unit's `avail_actions` row is `[0,0,0,0,1,0,0,0]` — SMAX leaves it
     exactly one legal action (the no-op). So a mask is **never all-zero**, and the
     `logits + (1-mask)*(-1e9)` path has no degenerate/NaN case. Verified over a full
     episode: no NaN in obs or reward, log-probs finite.

⚠ The seam tests pin themselves to CPU via an autouse `jax.default_device` fixture. They
are tiny, and sharing the GPU with a training or rendering job makes them fail on
cuSolver/OOM errors that read exactly like assertion failures. **`test_feudal_seams.py`
now has the same fixture** (2026-09-14) — it previously did not, which is what produced
the phantom `local`-latent reproducibility failure recorded in the Feudal section. A
module-level `JAX_PLATFORMS=cpu` does not work here: it only takes effect if that module
is imported before JAX initializes, i.e. it depends on pytest collection order.

### Verification

- `uv run pytest algorithms/tests/test_smax_seams.py -q` — 11 tests pinning the silent
  seams: no illegal action is ever sampled or stored; **the PPO ratio is exactly 1
  before any update** under masking (catches a sampling/update mask mismatch); a
  *reversed* mask is shown to produce a different action set, so the pairing test is
  evidence of agent-major flattening and not merely of masking; masked `argmax`; the
  global-state hook is used and sizes the critic, and falls back to concat without it;
  a hookless env stores the placeholder; and a dead agent's transition demonstrably
  changes the actor update.
- `uv run python -m environments.smax.smax_env` — adapter smoke test: shapes, jit+vmap,
  `terminated`/`truncated` mutual exclusivity, and that the state pytree survives the
  collector's `tree.map` reset-select.
- End-to-end (train / resume / evaluate / view) verified on both stacks at `3m`, plus
  `5m_vs_6m` on the flat stack.

### Rendering (`view=true`) — colors, cost, and the ffmpeg writer

`SMAXAdapter.render_episode` hands the episode to jaxmarl's `SMAXVisualizer`, which
draws **strictly by unit index**: allies `0 .. num_allies-1` are **blue**
(`cornflowerblue` while being shot at), enemies `num_allies ..` are **green**
(`limegreen` likewise); a gray square is a bullet interpolated shooter -> target, the
white letter in each circle is the unit-type shorthand (`m`/`M`/`s`/`Z`/`z`/`h`), and
the circle radius is the unit type's, not its health. Under `HeuristicEnemySMAX`
`self.agents` is the **allies only**, so **blue is the policy being rendered** and green
is the built-in scripted enemy. ⚠ The shot-highlighting is one-sided: `init_render`
builds `attacked_agents` by looping `self.agents`, which on the wrapper is allies only,
so the lighter shade and the bullets only ever show **your** team's fire. Positions,
deaths and the blue/green split are correct regardless.

- **`view=true` is slow, and the encoder is not why.** `SMAXVisualizer.expand_state_seq`
  multiplies the frame count by `world_steps_per_env_step` (**8**), so one 200-step
  episode is **1600** frames, and every frame re-runs `init_render` — `ax.clear()`, a
  fresh `Circle` + `text` per unit, then `figure.savefig(buff, format="raw")`. Measured
  **0.23 s/frame** (11.0 s for 48 frames at `3m`), i.e. **~6 min per full episode**, and
  `_view_with_env_renderer` renders 3 of them. Budget ~20 min, and do not wrap it in a
  short `timeout` — a `timeout 1500` cut one off mid-episode-2 and reported the
  misleading exit code 124.
- **`use_bundled_ffmpeg()` (module-level in `smax_env.py`, called by `render_episode`)
  silences `MovieWriter ffmpeg unavailable; using Pillow instead.`** jaxmarl's
  `Visualizer.animate` calls `ani.save(fname)` with **no `writer=`**, so matplotlib uses
  `rcParams["animation.writer"]` (default `"ffmpeg"`) resolved via
  `rcParams["animation.ffmpeg_path"]` (default the bare name `"ffmpeg"`, looked up on
  `PATH`). There is no system ffmpeg here, so it fell back to Pillow. The helper points
  that rcParam at the static binary **`imageio-ffmpeg` already vendors** — that package
  is a hard dependency in `pyproject.toml`, so this needs no new package and no root,
  which is what makes it work on the HPC nodes too. Idempotent, best-effort (returns
  False and leaves the Pillow fallback in place if anything is missing), and it defers
  to a real system ffmpeg when one is on `PATH`. ⚠ It changes the **encoder only** —
  per the measurement above it does **not** make rendering meaningfully faster.

### Config notes

`n_agents` is a property of the SMAX scenario, so the env groups do **not** set it — the
adapter reads it back off the env, and `run.py` does not pass it. Unlike the MJX
branches, `max_steps` is **not** `params.n_steps`: the benchmark owns its own horizon
(100). `params.n_steps` is 128, i.e. **>= `max_steps`** — `collect_fn` resets every env
at the top of every rollout, so a shorter rollout would never train on the back half of
any episode, and in SMAX the win bonus is paid at the very end (the same trap this file
records for `mjx_16a_4o_multi_goal`). `n_envs` is a **literal** (a vmap width, i.e. a
hyperparameter), as for every MJX group.

⚠ `model=feudal` is mandatory with `algorithm=feudal_mappo_jax`: results go to
`results/<env>/<model>/` and the algorithm is **not** in that path, so `model=mlp` would
write into the `mappo_jax` baseline's directory.

**Not yet done:** no full-length run of either stack on SMAX, so nothing is known about
final win rates or whether the hierarchy helps — every number above is a mechanism
check. `use_self_play_reward=True` raises (the adapter reports one shared scalar and
sets `per_agent_rewards=False`).

## Coordination-graph novelty exploration (gnn critic)

When `critic_type="gnn"`, the `AttentionGNNCritic` (`networks/gnn_critic.py`)
emits a per-head coordination graph from its attention encoder. Setting
`use_intrinsic_reward=True` (in `Model_Params`) turns that graph into an
exploration bonus: agents are rewarded for reaching states whose **coordination
graph is novel** within the current episode.

- The encoder is **dual-purpose** — shared with the value path and trained by the
  value loss, so the graph is grounded. The bonus reads it under `no_grad` via
  `network_old` (`MAPPOAgent.compute_coordination_features`).
- Descriptor (`AttentionGNNCritic.coordination_descriptor`, exposed via
  `MAPPONetwork.coordination_descriptor`):
  - `intrinsic_reward_mode="team"` → upper-triangle of each head's adjacency,
    one bonus per env tiled to all agents.
  - `intrinsic_reward_mode="agent"` → each agent's coordination row across heads,
    a per-agent bonus.
  - `intrinsic_descriptor_source` is `"adjacency"` (symmetric graph structure,
    default), `"directed_adjacency"` (raw directed attention scores — keeps the
    who-attends-to-whom asymmetry that symmetrization discards; team = all
    off-diagonal entries per head, agent = outgoing row + incoming column per
    agent), or `"node_embedding"` (attended tokens) for ablation. The directed
    scores are exposed by `MultiHeadAttentionEncoder.forward(..., return_scores=True)`;
    averaging the two directed halves recovers the symmetric descriptor exactly.
- Novelty is episodic k-NN distance (`intrinsic_reward.py`,
  `BatchedIntrinsicReward`). One batched rewarder scores all streams at once —
  one stream per env (`team`) or per (env, agent) (`agent`) — using a
  preallocated ring buffer `(n_streams, capacity, feat_dim)` and a single
  `cdist`/`sort`, instead of a per-stream deque that restacked its full memory
  every step (quadratic in episode length). Streams reset on episode done.
  Reward is `log(d_k + 1)` for the `min(k, count)`-th nearest stored point;
  empty-memory/done streams score 0. Plumbed in `RolloutCollector`
  (`_get_team_intrinsic_rewards` / `_get_agent_intrinsic_rewards`) and folded into
  per-agent rewards in `MAPPOAgent.store_transitions_batch` (single-stream, scaled
  by `intrinsic_reward_coef`).
- Config knobs in `Model_Params`: `intrinsic_reward_coef`, `intrinsic_reward_k`,
  `intrinsic_reward_memory_capacity`. Asserts `critic_type=="gnn"`; fully inert
  when `use_intrinsic_reward=False`.
- Logging: `RolloutCollector.collect` returns per-rollout means
  `mean_intrinsic_reward` (coef-scaled bonus exactly as it enters the agents'
  reward; 0 when intrinsic is off) and `mean_extrinsic_reward` (raw env reward
  over the same steps) on `RolloutResult`. `vec_trainer` records these into
  `training_stats["intrinsic_reward"]` / `["extrinsic_reward"]` and prints them
  on the log line when `use_intrinsic_reward`. Use the two curves to diagnose
  per-seed divergence: a failing trial with high sustained intrinsic but flat
  extrinsic is farming graph novelty (reward hacking) rather than just unlucky
  exploration.

### Visualizing the coordination graphs

`algorithms/tests/visualize_coordination_graph.py` runs one deterministic
episode with a trained policy and plots, at evenly spaced snapshot timesteps,
the env frame next to the critic's per-head coordination graphs. Nodes (agents)
sit on a **fixed circular layout** that carries no meaning, so the focus is the
edges: their weights are the symmetric attention adjacency read from
`network_old.critic.encoder` under `no_grad`, mapped to both color (shared
viridis colorbar over all heads/snapshots) and line width. `--show-labels`
annotates each edge with its weight; `--edge-threshold` hides weak edges. Frames
are captured headlessly via the pygame dummy SDL driver. Defaults target the
`cg_team_novelty` trial-2 model; run with:

```
SDL_VIDEODRIVER=dummy python -m algorithms.tests.visualize_coordination_graph \
    [--model ...pth] [--config ...yaml] [--env _env.yaml] \
    [--seed N] [--snapshots K] [--edge-threshold T] [--show-labels] [--out fig.png]
```

Plan: `plans/coordination_graph_novelty.md`.

## Hierarchical macro-action controller (`algorithms/hierarchical/`)

A high-level policy that, instead of emitting low-level forces, **selects which
of 4 frozen pre-trained skills to run** as a fixed-duration macro-action. The
controller is trained with the ordinary MAPPO stack — the trick is a gym wrapper
that makes "pick a skill, run it for K steps" look like one discrete env step.

- **Skills** (`skills.py`). A skill is a frozen, eval-mode `MAPPOActor`. The 4
  box2d tasks share `ObservationManager`, so every skill actor takes the same
  `obs_dim=40` local obs and emits the same `action_dim=2` force; only their
  critics differ (unused — we run actors only). `load_skill_actor` reads
  `checkpoint["network"]`, keeps the `"actor."`-prefixed keys (strip one prefix:
  `actor.actor.0.weight -> actor.0.weight`) and loads them into a fresh
  `MAPPOActor`. **Architecture (in/hidden/out) is inferred from the weights, not
  the yaml** — the pre-trained `mlp_shared` actors use hidden=183, not
  `Model_Params.hidden_dim=168`. `SKILL_ORDER = [contact, scatter, push_box,
  rendezvouz]` fixes the discrete action index → skill mapping;
  `resolve_skill_checkpoint` prefers `models_finished.pth`, falling back to
  `models_checkpoint.pth` (e.g. `scatter_9a` only ships the checkpoint).
- **Wrapper** (`hrl_env.py`). `HierarchicalSkillEnv(gym.Env)` builds a base env
  via the shared `make_single_env` factory and loads the 4 skills once.
  `decision_scope`:
  - `"agent"`: each agent picks its own skill. Obs `(n_agents, 40)`, action
    `MultiDiscrete([4]*n_agents)`.
  - `"team"`: one skill for all agents. Obs `(1, n_agents*40)` (flattened team
    state), action `MultiDiscrete([4])` → a single high-level agent.
  Each `step` runs the chosen skill(s) for `macro_len` (default 10) low-level
  steps — agents sharing a skill are batched through that actor in one forward —
  accumulating reward and stopping early on done. `torch.set_num_threads(1)` per
  worker.
- **Wiring.** `EnvironmentEnum.HRL_SKILL = "hrl_skill"`; `make_single_env`
  (factored out of `make_vec_env`'s closure so the wrapper can reuse it) builds
  the wrapper. `make_vec_env` launches HRL workers with the **`forkserver`** MP
  start method — the default `fork` deadlocks when each worker `torch.load`s
  models (inherited OpenMP/thread state); other envs keep `fork`. `forkserver`
  re-imports the entry module, so HRL training **must** be launched under an
  `if __name__ == '__main__':` guard (Hydra's `train.py` already is).
  `vec_trainer` adds `HRL_SKILL` to its discrete list and derives the *learning*
  agent count from `obs_space.shape[0]` (1 for team scope, n_agents otherwise) —
  behavior-preserving for normal envs where that equals `env_params["n_agents"]`.
- **Config.** Batches `experiments/yamls/hrl_{agent,team}_multi_box_push_9a/`
  carry the macro knobs in the `_batch.yaml` `env:` block (`base_environment`,
  `decision_scope`, `macro_len`, `skill_experiment`, `skill_trial`); the
  `mlp_shared.yaml` is a standard discrete-MAPPO config. Once migrated into
  `conf/` (`conf/env/hrl_agent_multi_box_push_9a.yaml` + the model file), run with:
  ```
  uv run python train.py env=hrl_agent_multi_box_push_9a model=mlp_shared \
      algorithm=mappo trial_id=0
  ```
- **Skill-selection logging.** `RolloutCollector` tallies the chosen discrete
  actions over each rollout (`np.bincount`) and returns a normalized
  `action_distribution` on `RolloutResult`; `vec_trainer` records it into
  `training_stats["action_distribution"]` (one fractions-vector per iteration,
  recorded only for discrete runs) and prints it on the log line — labeled with
  `SKILL_ORDER` names for HRL (`Skills: contact=0.19 scatter=0.25 ...`) via
  `_format_action_distribution`. Use it to watch the controller specialize off a
  uniform `1/n` split. Generic to any discrete env (shown as `Actions: i:p`).
- Plan: `plans/now-i-want-you-wise-graham.md`.

## Hypergraph backend: `dhg` shim (`hypergraphs/hg_compat.py`)

The upstream `dhg` (DeepHypergraph) package pins `torch<2`, which blocked
upgrading PyTorch. The runtime only ever used `dhg` as a thin container that
turns `(num_v, edge_list)` into the sparse incidence matrices `H` / `H_T` — the
HGNN smoothing math is already reimplemented in
`hypergraphs/hgnn_conv_layer.py:smoothing_with_hgnn_factors`. So `dhg` was
replaced by a small drop-in shim, `hypergraphs/hg_compat.py`, imported
everywhere as `import hypergraphs.hg_compat as dhg`.

- Implements exactly the surface the code consumes: `dhg.Hypergraph(num_v,
  e_list, device=...)` with `.H`, `.H_T`, `.num_e`, `.num_v`, `.device`,
  `.to(device)`, `.e`, `.draw(...)`, plus `dhg.random.hypergraph_Gnm` /
  `graph_Gnm` (demo/test helpers).
- Semantics matched against `dhg` 0.9.x and verified numerically (incidence
  `H @ Hᵀ`, HGNN smoothing output, and structural-entropy edge-size multiset
  all equal): `H` is `(num_v, num_e)` float32 with unit entries; identical
  hyperedges (order-independent) are merged so `num_e` counts unique edges;
  duplicate vertices within an edge accumulate. Edge/column ordering is not
  guaranteed to match dhg's (irrelevant to every consumer — smoothing is
  `H Hᵀ`, entropy is permutation-invariant).
- `.draw()` is best-effort matplotlib (circular node layout, hyperedges as
  blobs/lines/rings), not pixel-faithful to dhg's renderer; it raises
  `ValueError` on an empty hypergraph like dhg (the renderer catches that).
- `dhg` is removed from `pyproject.toml` and `torch` is now `>=2.0`.
- NOT ported: `hypergraphs/hypegraph_training.py`, a standalone Cora/GCN demo
  that uses `dhg.models.GCN` / `dhg.data.Cora` / `dhg.metrics`. It is not part
  of the MAPPO runtime and still requires the real `dhg` to run.

## DCG coordination-graph algorithm (`algorithms/dcg/`)

DCG (Deep Coordination Graph, Böhmer et al. 2020) is integrated as a first-class
algorithm alongside MAPPO/IPPO: launch it with `train.py algorithm=dcg`
(`AlgorithmEnum.DCG`, dispatched in `algorithms/algorithms.py`). Unlike MAPPO's
on-policy vectorized PPO, DCG is **off-policy episodic Q-learning** — RNN feature
agents → per-agent utility `f_i` and per-edge payoff `f_ij` nets → max-sum
message passing over a coordination graph → double-Q TD targets from an episode
replay buffer.

- **Vendored core + adapter.** The upstream PyMARL project lives unmodified
  under `algorithms/dcg/src` (controller `controllers/dcg_controller.py`,
  learner `learners/dcg_learner.py`, `components/episode_buffer.py`, action
  selectors, `modules/agents/rnn_feature_agent.py`, mixers). The framework
  adapter wraps it: `types.py` (dataclasses `DCG_Params` / `DCG_Model_Params` /
  `Experiment`), `args_builder.py` (translates the dataclasses + env dims into
  the flat `args` namespace the vendored modules read — the single config
  bridge), `logger_shim.py` (a `log_stat`/`console_logger` stand-in for Sacred),
  `trainer.py` (`DCGTrainer`), and `run.py` (`DCG_Runner(Runner)`). `_vendor.py`
  puts `src/` on `sys.path` so `from controllers.dcg_controller import ...`
  resolves (no repo-root name collisions). The `controllers/__init__.py` and
  `learners/__init__.py` registries were trimmed to the DCG stack; the alt
  controllers/learners (`cg_mac`, `low_rank_q`, `coma`, `qtran`) are still
  vendored but unregistered (out of scope, and some need `torch_scatter`).
- **Discrete envs only.** DCG requires a `MultiDiscrete` action space +
  available-action masks. It targets the SMAC-style envs (`smaclite`,
  `smacv2`), whose gym wrappers already surface `info["avail_actions"]`
  `(n_envs, n_agents, n_actions)`. Continuous box2d envs are unsupported
  (`DCGTrainer.__init__` raises on a non-discrete action space). Global state is
  the concatenation of per-agent obs (`obs_dim * n_agents`), matching MAPPO;
  only the optional duelling bias / mixers consume it.
- **`torch_scatter` removed.** `dcg_controller.py`'s 3 `scatter_add` sites now
  use a native-torch `_scatter_add` helper (`out.scatter_add_` with a broadcast
  index), dropping the compiled, version-pinned dependency — same motivation as
  the `dhg` shim. Verified numerically against a reference. The vendored
  `episode_buffer._parse_slices` now returns a tuple (torch>=2 deprecates
  list-of-slices tensor indexing).
- **Collection loop** (`DCGTrainer._collect`) replaces PyMARL's
  `parallel_runner`: it drives a gym `AsyncVectorEnv` (built by the shared
  `make_vec_env`) in lockstep, packing transitions into DCG's `EpisodeBatch`
  (shape `(n_envs, episode_limit+1)`). Each env is **frozen on its first
  done**; the stored `terminated` field is the gym `terminated` flag only, so a
  time-limit `truncated` keeps `terminated=0` and its TD target still bootstraps
  (PyMARL semantics). Under Gymnasium's default `NEXT_STEP` autoreset the
  terminal observation is returned at the done step, so the stored next state is
  correct. Frozen envs still get stepped (the vector API requires it) with a
  valid fallback action (`cur_avail.argmax`) — a dummy `0` would be an illegal
  action once a frozen env auto-resets into a fresh live episode.
- **Checkpoint / stats.** `save_agent`/`load_agent` write a single `.pth`
  (agent + utility/payoff nets + optimiser + RNG) as `models_finished.pth` /
  `models_checkpoint.pth`; `_StatsBook` pickles `training_stats_*.pkl`.
  `checkpoint=true` resumes from the last saved step (verified).
- **Config.** Ported into `conf/`: `conf/algorithm/dcg.yaml` (`params`),
  `conf/model/dcg.yaml` (`model_params`), `conf/env/dcg_smaclite_2s3z.yaml`
  (`env:` block — `environment`/`n_agents`/`env_variant`). Source material stays
  at `experiments/yamls/dcg_smaclite_2s3z/` (`_batch.yaml`, `dcg.yaml`, and the
  tiny fast-run `dcg_test.yaml`). Launch:
  ```
  uv run python train.py env=dcg_smaclite_2s3z model=dcg algorithm=dcg trial_id=0
  ```
- Plan: `plans/okay-the-following-is-composed-truffle.md`.

## DCG over macro-actions (`algorithms/dcg_macro/`)

`dcg_macro` (`AlgorithmEnum.DCG_MACRO = "dcg_macro"`) runs the **unmodified DCG
core** over the hierarchical macro-action interface, so DCG's discrete
coordination-graph Q-learning drives a continuous box2d task (e.g.
`multi_box_push`) by **selecting frozen skills** instead of low-level forces. It
is the DCG analogue of the hierarchical MAPPO controller.

- **No DCG code change; the env supplies the macro mechanism.** DCG needs a
  `MultiDiscrete` action space, which `HierarchicalSkillEnv`
  (`algorithms/hierarchical/hrl_env.py`) already provides: the discrete action
  picks one of 4 frozen skills (`SKILL_ORDER`) and runs it for `macro_len`
  low-level steps. So `dcg_macro` is a **thin package** — `algorithms/dcg_macro/
  run.py` defines `DCG_Runner` but imports the trainer/types straight from
  `algorithms.dcg` (`from algorithms.dcg.trainer import DCGTrainer`); the vendored
  PyMARL core is reused, not duplicated. (The rest of the `algorithms/dcg_macro/`
  copy — `trainer.py`/`args_builder.py`/`src/` etc. — is currently unused dead
  weight; delete if the package need not diverge from `dcg`.)
- **Wiring.** `algorithms/types.py` adds the enum; `algorithms/algorithms.py`
  `_dispatch` adds a `case AlgorithmEnum.DCG_MACRO` mirroring `DCG` but importing
  `algorithms.dcg_macro.run.DCG_Runner` (and reusing `algorithms.dcg.types.
  Experiment`). `make_vec_env` already pins the `forkserver` start method for
  `HRL_SKILL` (each worker `torch.load`s the skill actors), so DCG's two vec
  envs build correctly. The box2d/HRL envs surface no `info["avail_actions"]`, so
  DCG's `_get_avail` falls back to an all-ones mask — correct, since all 4 skills
  are always selectable.
- **Config.** `conf/algorithm/dcg_macro.yaml` (`params`, same as `dcg` but with
  `episode_limit: 200` ≥ the ~103 macro-steps of a 1024-step base env at
  `macro_len=10`), `conf/model/dcg_macro.yaml` (DCG `model_params`, verbatim from
  `dcg`), and `conf/env/dcg_macro_multi_box_push_9a.yaml` — the HRL-wrapped env
  block: `environment: hrl_skill` (DCG reads this), `base_environment:
  multi_box_push`, `decision_scope: agent` (each of 9 agents picks a skill →
  9-node coordination graph, `MultiDiscrete([4]*9)`, obs `(9, 40)`), `macro_len`,
  `skill_experiment: mlp_shared`, `skill_trial: "0"`. The 4 skills load from
  `experiments/results/{contact,scatter,push_box,rendezvouz}_9a/mlp_shared/0/`.
  Launch:
  ```
  uv run python train.py env=dcg_macro_multi_box_push_9a model=dcg_macro \
      algorithm=dcg_macro trial_id=0
  ```

## Oracle difference rewards (`algorithms/difference_rewards/`)

A measurement stack (not wired into training) for computing **exact** difference
rewards `D_i = G(z) - G(z_-i + c_i)` by forking the pure functional MJX env.

**Status:** the research direction it was built for — *difference rewards under
asynchronous macro-actions* — was explored and **abandoned as tautological** (an
estimator fed a knowingly-wrong commitment state produces wrong credit; and the
non-tautological rescue, an async-specific counterfactual-scope ambiguity, was
measured and falsified: sync 0.626 vs async 0.680 cross-horizon stability). See
`plans/async_difference_rewards.md`. The **oracle itself is the reusable asset** and
does not depend on asynchrony: it enables auditing what learned counterfactual
baselines (vendored COMA, `algorithms/mappo/hg_cache.py:414`) actually recover —
normally uncheckable, since most envs cannot rewind.

- **`environments/mjx_suite/macro_skills.py`** — 4 scripted, deterministic JAX skills
  (`SKILL_ORDER = [contact, push, scatter, rendezvous]`, index = discrete action) plus
  `null_action`, the counterfactual default `c_i` (not policy-selectable). Skills are
  scripted rather than the frozen torch actors of `algorithms/hierarchical/skills.py`:
  being deterministic they make the forked counterfactual exact. **They sense only
  within `env.sector_sensor_radius`** — a global centroid gives a distant, physically
  irrelevant agent a causal channel into every teammate and manufactures false credit.
  `skill_scatter` carries `_wall_repulsion`; without it agents walk into the boundary,
  which `MultiBoxPushMJX` terminates with zero reward.
- **`environments/mjx_suite/macro_wrapper.py`** — `AsyncMacroMJX` + `MacroState`
  (`EnvState` + per-agent `skill_idx`/`elapsed`/`remaining`), pure and jit/vmap-able.
  One `step` is one **low-level** step: the policy is queried every step but only
  agents whose commitment expired adopt the proposed skill, so decision points
  decouple while shapes stay static. Obs = `MACRO_OBS_DIM` (the shared 40-dim `OBS_DIM`
  + one-hot skill + remaining/elapsed). Conditions: `d_min == d_max, stagger=False`
  reproduces the `HierarchicalSkillEnv` lockstep exactly (the control); `stagger=True`
  offsets phases; `d_min < d_max` varies durations. `commit`/`step_committed` are split
  out so the oracle can fork *after* commitment.
  - **`SyncMacroMJX` (same module) is the active hierarchical-training env**, not
    part of the abandoned async study: it is the JAX analogue of the box2d
    `HierarchicalSkillEnv`, where **one `step` is one macro decision** — all
    agents adopt their proposed skill in lockstep, the skills roll out for
    `macro_len` low-level physics steps (reactive actions re-derived each step),
    reward is **accumulated** over the window, and the episode freezes at the
    first low-level done. So the `mappo_jax` rollout scan stores exactly one
    transition per genuine decision (the SMDP/options view) — correct PPO credit
    assignment, unlike stepping `AsyncMacroMJX` every low-level step where most
    proposals are discarded mid-commitment. The macro state is just the base
    `EnvState` (no commitment bookkeeping under lockstep), so it plugs into the
    collector's `v_reset`/`tree.map` auto-reset unchanged; obs = the shared
    40-dim `OBS_DIM` (no commitment features), action = a per-agent **discrete**
    skill index (`action_dim=N_SKILLS`, `discrete=True`), `max_steps =
    ceil(base.max_steps / macro_len)` decisions. Reward summed over the window
    with no intra-option discounting, mirroring the box2d wrapper for parity.
    Skills are the **scripted** JAX skills of `macro_skills.py` (not frozen
    torch actors). Smoke test (interface + one-step==macro_len accumulation +
    vmap): `MUJOCO_GL=egl SDL_VIDEODRIVER=dummy uv run python -m
    environments.mjx_suite.macro_wrapper`. **Training is wired into `mappo_jax`**
    (see that section): `EnvironmentEnum.MACRO_MJX = "macro_mjx"`, three env
    groups, all reusing the `mlp` model group and flipping `mappo_jax` between the
    scalar and per-agent (per-agent critic head + per-agent GAE) paths while
    `info["task_reward"]` always logs the team scalar:
    - `conf/env/macro_mjx_9a_3o.yaml` — **dense** team reward (scalar).
    - `conf/env/macro_mjx_9a_3o_dr.yaml` — **single-step difference rewards**: the
      per-macro-window reward is the sum of the base env's exact single-step `D_i`
      over `macro_len` steps. This is *additive force attribution* (`sum_i D_i/G ~
      1.1`): each agent credited for its instantaneous force share, NOT coalition
      necessity — a single step can't reveal the coupling (box mass affects
      acceleration, which needs many steps to integrate into a displacement).
    - `conf/env/macro_mjx_9a_3o_wdr.yaml` — **windowed difference rewards**
      (`reward_mode="windowed_difference_rewards"`, `macro_len=30`): the exact
      *windowed* counterfactual `D_i = G(window) - G_{-i}(window)`, where `G_{-i}`
      re-rolls the **same** macro window with agent i absent (zero force + dropped
      from the coupling count via the `active` mask threaded into `env.step`) for
      the WHOLE window. Per the difference-reward formulation, the counterfactual
      changes **only agent i's** contribution: the teammates **replay the exact
      low-level forces they applied in the factual window** (recorded from the
      factual roll and fed back as `replay_actions`), open-loop — they do NOT
      re-derive their skills from the counterfactual state and react to i's absence.
      So `D_i` isolates i's physical effect and does not absorb teammates'
      behavioral compensation. Holding an agent absent across the window still lets
      the coupling stall the box if i was required (the mass/coupling physics acts
      regardless of whether teammates react), so the credit reflects coalition
      necessity. Computed by `SyncMacroMJX._step_windowed` — the factual window
      (which records the per-step `(macro_len, A, 2)` action sequence) + an `A`-way
      `vmap` of counterfactual windows replaying it from the same start state;
      **exact** because the recorded forces + MJX step are deterministic (verified
      against a manual fork). Costs `(A+1)×macro_len` base steps/decision (vmapped). The coupling
      reveals only as the window grows (smoke test measured `sum_i D_i/G` climbing
      `+0.04→+0.30→+0.53→+0.75` at macro_len `1→5→15→30` for one state; the
      saturated `~coupling` ratio needs a *tight* coalition state). It is a
      **single-macro-window** counterfactual — it does NOT span future decisions
      (that needs the policy re-deciding, i.e. a trainer-level windowed D). The
      fine-control (small `macro_len`) vs coalition-credit (window ≥ ~30) tension
      is real; the `_wdr` group picks `macro_len=30` for the credit signal.

    **Budget scaling: every `macro_mjx_*` group must divide `params.n_total_steps`
    and `params.n_steps` by `macro_len`.** Both are counted in *env steps*, and one
    env step of `SyncMacroMJX` is one macro **decision** = `macro_len` low-level
    `mjx.step`s (plus a 4-skill `all_skill_actions` evaluation each), so inheriting
    the `mappo_jax` defaults (`n_total_steps: 1e8`, `n_steps: 1024`) silently made
    the macro arms cost `macro_len`× the base env for the same nominal config — at
    `macro_len=20` that was 2e9 low-level steps vs `mjx_16a_4o`'s 1e8, i.e. ~20×
    the wall-clock, plus rollouts spanning ~20 episodes per stream (the macro
    horizon is only `ceil(1024/20) = 52`) against exactly 1 for the base env. The
    `macro_mjx_16a_4o*` groups therefore pin `n_total_steps: 5e6` (= 1e8/20, so the
    **low-level physics budget matches `mjx_16a_4o` exactly**) and `n_steps: 256`
    (~5 episodes/stream, batch 8192 decisions, 610 updates). Note `n_steps` alone
    is *not* a wall-clock knob — `num_updates = n_total_steps // (n_steps*n_envs)`,
    so lowering it only trades batch size for update count; only `n_total_steps`
    moves total compute. Rescale both if you change `macro_len`, and keep them
    identical across arms being compared. The `_wdr`/`_stagger_wdr*` counterfactual
    forks are *extra* compute on top of that factual budget.

    All three arms verified end-to-end (train + resume + evaluate); the `_dr`/
    `_wdr` critic heads are `n_agents`-wide, the actor a 4-way categorical. Launch:
    ```
    uv run python train.py algorithm=mappo_jax env=macro_mjx_9a_3o \
        model=mlp trial_id=0
    # single-step difference-rewards arm (_dr) / windowed arm (_wdr):
    uv run python train.py algorithm=mappo_jax env=macro_mjx_9a_3o_dr \
        model=mlp trial_id=0
    uv run python train.py algorithm=mappo_jax env=macro_mjx_9a_3o_wdr \
        model=mlp trial_id=0
    ```
  - **Staggered starts (async-onset study).** `SyncMacroMJX(stagger_starts=True,
    max_start_delay=D)` makes each agent come online at a random **low-level** step
    in `[0, D]` (sampled per episode) and thereafter re-decide on its **own phase**
    — every `macro_len` steps counted from *its* onset. Because onsets are not
    multiples of `macro_len`, the agents' decision phases stay decoupled the whole
    episode (persistent asynchrony), unlike the lockstep options view; the design
    question it probes is whether a setup that records **one transition per global
    macro window** copes with agents deciding out of phase. Mechanism: the policy is
    still queried once per window (`proposed` off the window-start obs), but inside
    `_step_staggered`'s low-level scan each agent adopts `proposed[i]` only at *its*
    boundary (`t >= onset & (t-onset)%macro_len==0`) and flies its previous
    `skill_idx` until then — so `skill_idx` must persist across the window boundary
    (state is a registered `StaggeredMacroState(env_state, skill_idx, onset)`; the
    absolute low-level step is read from `env_state.t`). Since period == window ==
    `macro_len`, each online agent hits exactly one boundary per window, so the
    trainer still stores one transition per agent per window. Before its onset an
    agent is **offline**: `online` is threaded as the base env's per-step `active`
    mask (null force + dropped from the coupling count), and it is masked out of the
    PPO loss. That masking is a new `Transition.active_mask` `(n_envs, n_agents)`
    field — `SyncMacroMJX` emits `info["active"]` (who decided this window; the
    final truncated window can be `<` the onset schedule), the trainer defaults it
    to **all-ones** for every other env, and `ppo_update` applies it as a masked
    mean to the actor policy/entropy loss (+ the per-agent critic head). All-ones
    reduces the masked mean to a plain mean, so **every non-stagger run is
    byte-identical**. Config `conf/env/macro_mjx_16a_4o_stagger.yaml` (dense;
    `max_start_delay: 50` low-level steps). Verified: onset→active-mask schedule,
    decoupled phases, vmap, `tree.map` auto-reset, and train (collect→update→eval).
    Launch:
    ```
    uv run python train.py algorithm=mappo_jax env=macro_mjx_16a_4o_stagger \
        model=mlp trial_id=0
    ```
  - **Difference rewards under asynchrony (global-window baseline).** Stagger now
    also supports `reward_mode="windowed_difference_rewards"`: the per-agent reward
    is `D_i = G(window) - G_{-i}(window)` over the **global** recording window
    `[W, W+macro_len)`, computed by `SyncMacroMJX._step_staggered_windowed`. The
    scan is refactored into `_staggered_window(mstate, proposed, drop_agent,
    replay_actions)` — the factual run (`drop_agent=-1`, which records the per-step
    `(macro_len, A, 2)` action sequence) plus an `A`-way `vmap` of counterfactual
    runs each nulling one agent for the whole window (its `active` mask is `online &
    (arange != drop_agent)`, like the oracle's `override_agent`). As in the sync
    path, the teammates **replay their factual low-level forces** (`replay_actions`)
    open-loop instead of reacting to the counterfactual state, so only the dropped
    agent's contribution changes (the difference-reward requirement). **Forking
    `StaggeredMacroState` resumes teammates' in-flight commitments automatically**
    — carrying `skill_idx` + `onset` continues the scan from every partial
    commitment; the replayed forces then flow open-loop, nothing else to restore.
    Exact (recorded forces + deterministic MJX, common random numbers).
    Cross-validated: with `onset` all-zero and a macro-boundary-aligned start state
    the D is **bit-identical** to the independent sync `_step_windowed` path (0.0
    gap); a never-online agent gets `D_i=0`/`active_i=0`. Still raises for the base
    single-step `"difference_rewards"` (its counterfactual ignores the outer online
    mask). Config `conf/env/macro_mjx_16a_4o_stagger_wdr.yaml`; verified train +
    checkpoint resume. **Known limitation (the baseline's whole point):** agent i's
    own decision window `[φ_i, φ_i+L)` (φ_i = `onset_i % L`, its fixed phase) is
    phase-**offset** from `[W, W+L)`, so removing i over the global window blends
    the tail of i's *previous* commitment with the head of its *new* one and pays
    that D to the transition labelled with the new action (~φ/L of the window
    misattributed) — mirroring the dense reward's own phase smear. The
    **decision-aligned** variant (below / trainer-level) corrects it; the gap
    between the two measures the misattribution cost. Launch:
    ```
    uv run python train.py algorithm=mappo_jax env=macro_mjx_16a_4o_stagger_wdr \
        model=mlp trial_id=0
    ```
  - **Decision-aligned difference rewards under asynchrony (the phase-corrected
    arm).** `reward_mode="aligned_windowed_difference_rewards"` credits each agent
    over **its own** decision window `[W+φ_i, W+φ_i+L)` instead of the global
    `[W, W+L)`, so `D_i` pairs with the action i chose at `W+φ_i` and the `[W,
    W+φ_i)` tail (i still flying its *previous* skill) lands on i's previous
    transition — fixing the global-window baseline's ~φ/L phase misattribution.
    The core primitive is `SyncMacroMJX.decision_aligned_D(mstate, proposed,
    proposed_next)`: per agent (vmapped) it rolls `2L` steps from the window start,
    factual and with i nulled **only during its own window**, and diffs the team
    reward over `[W+φ_i, W+φ_i+L)`; the factual roll is shared, teammates **replay**
    the recorded factual forces open-loop, and boundaries in the overshoot
    `[W+L, W+φ_i+L)` adopt `proposed_next`. Because that window spills into the next
    global window (whose proposals are unknown during collection), it is computed
    **post-collect in the trainer** (`mappo_jax/trainer.py:_apply_aligned_rewards`):
    `_env_step` logs a compact per-window `snapshot` (qpos/qvel + EnvState scalars +
    skill_idx/onset — `SyncMacroMJX.snapshot`, reconstructed via
    `state_from_snapshot` + `mjx.forward`; D is exact w.r.t. the reconstruction
    since both branches share it), `truncated`, and the pre-reset `next_value`; the
    post-collect pass `vmap`s `decision_aligned_D` over (window, env) with
    `proposed_next = action` shifted by one (last window reuses its own action, a
    boundary approximation), then re-applies the truncation bootstrap
    `reward = D + γ·truncated·next_value`. In-collect the env returns the
    global-window D as a **placeholder** (overwritten post-collect). Gated by a
    static `aligned` flag in `make_train`, so **every non-aligned run is
    unchanged**; the aligned mode requires `stagger_starts` and a dense base env.
    Verified: **phase-0 parity** — with all onsets at phase 0 (decision window ==
    global window) `decision_aligned_D` is bit-identical to
    `_step_staggered_windowed` (0.0 gap) and independent of `proposed_next`; at
    nonzero phase D genuinely **depends on `proposed_next`** (the credit reaches
    into the next window); and train + checkpoint resume end-to-end. Config
    `conf/env/macro_mjx_16a_4o_stagger_wdr_aligned.yaml`. Costs ~2× the
    global-window arm (a shared factual + `A` replayed counterfactuals of `2L` each
    per window). Compare its learning curve against the global-window arm — the gap
    is the cost of the phase misattribution. Launch:
    ```
    uv run python train.py algorithm=mappo_jax \
        env=macro_mjx_16a_4o_stagger_wdr_aligned model=mlp trial_id=0
    ```
- **`algorithms/difference_rewards/oracle.py`** — exact `D_i` by **forking the
  simulator** (`MultiBoxPushMJX` is pure, so a state can be replayed under a
  counterfactual — no learned model, unlike COMA/Dr.Reinforce). One vmap over
  `[-1, 0..A-1]` runs the factual + every counterfactual in one compiled call under
  **common random numbers**. `aligned_belief` collapses commitment phase to the joint
  mean = the synchronous estimator's belief. Two invariants that are easy to get
  wrong: the counterfactual rollout **must let agents re-decide** (frozen skills make
  `remaining`/`elapsed` inert and the sync/async estimators coincide identically), and
  `aligned_belief` must collapse to the **mean** phase, not reset to nominal `L`, or
  the control shows spurious bias.
- **`algorithms/difference_rewards/bias_study.py`** — the (abandoned) falsification,
  no training. Compares `D_oracle` vs `D_sync` from the same physical state under the
  same rollout key. Result: sync estimator exact under synchrony (pearson 1.0, bias
  0.0), collapses under any asynchrony (pearson ~0.3, `norm_bias ~1.0`, sign wrong
  ~25%) — but this is **near-tautological**, see the status note above. Two reusable
  methodology points survive: compute metrics **per state then aggregate** (credit
  scale varies ~100x across states, so pooled ratios are meaningless — a first attempt
  produced a garbage `norm_bias=29.9` from a small denominator), and measure only at
  **engaged** states (at reset every `D_i` is 0, so attribution tests pass vacuously —
  the first verification run was a false pass for exactly this reason).
  ```
  MUJOCO_GL=egl uv run python -m algorithms.difference_rewards.bias_study \
      --n-states 24 --horizon 60
  ```
- **`algorithms/difference_rewards/reward_magnitude_study.py`** — one-plot
  diagnostic (no training) for *why* the `macro_mjx_16a_4o` DR arms (`_dr`,
  `_wdr`) learn worse than the dense baseline: it compares the **magnitude of the
  reward actually stored per transition** in each arm (what the critic/actor learn
  from). It rolls out the **dense-trained** policies (`macro_mjx_16a_4o/mlp/
  <trial>`, argmax skills) and at each macro decision reads all three stored
  rewards off the *same* pre-step state / skills: dense team scalar `G`, the
  `_dr` per-agent `D_i` (sum of the base env's single-step `D_i` over the window),
  and the `_wdr` per-agent `D_i` (`_step_windowed`) — these are literally
  `Transition.reward` in each arm. Exact/fair because base physics is independent
  of `reward_mode` (only the read-out changes), so one canonical trajectory feeds
  all three; all share `macro_len=20`. Uses three `SyncMacroMJX` views (dense
  driver, `difference_rewards` base, windowed) and vmaps over rollouts × the
  per-agent counterfactual forks; `--chunk` caps peak GPU memory (windowed forks
  `chunk * n_agents` concurrent MJX sims — 32 rollouts unchunked OOMs a 16 GB
  GPU). Caches the pooled arrays to `<out>.data.pkl`; `--from-cache` re-plots
  without recomputing. **Finding** (11 trials, signed means): the per-agent DR
  signal stored per transition is far weaker than the dense team reward each agent
  learns from — dense `G ≈ 6.8`, timestep `D_i ≈ 0.095` (**~72x smaller**),
  windowed `D_i ≈ 0.80` (**~8.5x smaller**). (Note: the earlier ~1.1 single-step
  `sum_i D_i/G` ratio quoted elsewhere came from a hand-crafted tight-coalition
  9a/3o *state*; averaged over the learned 16a/4o policy most of the 16 agents are
  redundant per step, so single-step credit is even weaker.) Writes one bar chart
  `algorithms/difference_rewards/reward_magnitude.png`.
  ```
  MUJOCO_GL=egl uv run python -u -m \
      algorithms.difference_rewards.reward_magnitude_study \
      --n-rollouts 5 --chunk 8
  ```
