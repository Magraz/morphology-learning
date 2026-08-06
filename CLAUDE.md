# When user gives instructions, push back if you think the user is wrong. Do not accept everything the user says as source truth. Use your best judgement but share your reasoning with the user and provide both options. Always go with what the user chooses after this. 

Whenever building new code, try to reuse as much code as possible. If the new functionality overlaps heavily with other parts of the code, find a way to abstract and reuse the logic instead of duplicating the functionality.

Always keep the CLAUDE.md file up to date to reflect the current functionality and architecture of the code.

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

## Box2D suite observations

All `environments/box2d_suite` envs share `ObservationManager.get_observation`
(in `observation.py`). The per-agent observation vector is, in order:

- `own_velocity` (2) — linear velocity normalized by `velocity_norm`
- `density_sensors` (16) — 8-sector centroid distance to agents (0-7) and objects (8-15)
- `is_touching_object` (1)
- `neighbor_fraction` (1) — fraction of agents within `neighbor_detection_range` (incl. self)
- `contact_force` (1) — per-agent contact force / `force_multiplier`
- `nearest_box_vec` (2) — relative (dx, dy) to the nearest **undelivered**
  object, per axis normalized by `world_width`; zero vector when the env has no
  objects or when every object has been delivered. Already-delivered objects
  (`env.delivered_objects` in Box2D, the `delivered` mask in MJX) are excluded
  from the nearest-object search, so an agent stops being drawn to a box parked
  in the goal band. Egocentric (no absolute world anchor).
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
`ObservationManager`: it owns the sensor math and the 40-dim `OBS_DIM` layout
for **every** MJX port, so a new port only supplies its own qpos layout and
goal. Pure and `jit`/`vmap`-able; the env passes plain arrays (agent positions/
velocities, box poses) plus the `mjx.Data` (needed for lidar raycasts and the
efc contact-force decode).

- Construct with the `mjx.Model` + world/normalization constants
  (`world_width/height`, `velocity_norm`, `neighbor_detection_range`,
  `agent_radius`, `force_multiplier`; `sector_sensor_radius` defaults to
  `world_width/3` and `lidar_range` to the sector radius, as in Box2D). Contact
  attribution needs the geom→entity maps from the helper `geom_index_maps(mj_model,
  n_agents, n_objects)` (naming convention `g_agent_{i}` / `g_box_{j}`).
- `build(data, agent_pos, agent_vel, box_pos=, box_yaw=, box_half=,
  goal_coord=, goal_axis=, delivered=)` returns `(A, OBS_DIM)`; the components
  are also exposed individually (`touch_matrix`, `density_sensors`,
  `neighbor_fractions`, `pairwise_agent_distances`, `nearest_box_vectors`,
  `goal_distances`, `lidar`, `contact_forces`) — `_touch_matrix` (coupling) and
  the renderer reuse them. The optional `delivered` (O,) bool mask (threaded
  from `EnvState.delivered` by `MultiBoxPushMJX._get_obs`) drops delivered boxes
  from `nearest_box_vectors` only — an agent stops being drawn to a box parked
  in the goal band; all delivered → zero vector. The Box2D
  `ObservationManager._calculate_nearest_box_vectors` does the same via
  `env.delivered_objects`, keeping the two engines in parity.
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
- **Parity with Box2D.** Same world sizing, spawn regions, coupling list, box
  sizing, target band, reward (shaping + one-time +100/box, dense/sparse),
  boundary-contact termination, and the exact 40-dim `OBS_DIM` observation
  layout (built by the shared `MJXObservationBuilder` above). Verified by
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
  `"drift"` → `box_drift_speed=_DEFAULT_DRIFT_SPEED` (read the constant — it is
  being tuned, and is set well past the 0.5 knee of the calibration table below,
  which has not been re-swept there) + inert walls; `"trunc"` → inert walls
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
- **⚠ `_DEFAULT_DRIFT_SPEED = 3.0` is outside the calibrated range and breaks the
  mechanic.** The sweep above stops at 0.8, already the point where the drift
  force exceeds a full coalition's thrust; at 3.0 it is 3072 N against ~400 N.
  `--check-drift` [8] **fails** there (16a/4o, seeds 7/23): the scripted balanced
  partition — the strategy the mechanic exists to reward — delivers **0.5 of 4
  boxes** with drift on vs **4.0** with it off, and its return *loses* to the
  swarm (21 vs 60), i.e. the drift **inverts** the preference. The task is close
  to infeasible at this speed, so a flat learning curve on `_drift` says nothing
  about coalition discovery. Re-sweep and pick from the table (0.5 was the knee)
  before running the arm; the check is the canary. Note the inversion is mostly
  the speed, not the inert-wall change: measured partition − swarm at `v_d=3.0`
  is **+8** under the old episode-ending walls and **−39** under inert walls
  (against **+345** with no drift), because the swarm no longer has its episode
  cut short at ~615 steps by an incidental wall hit and finishes its one box.
- **Difference rewards are structurally blind to this pressure.** An unattended
  box's drift cost does not depend on agent *i*, so it appears identically in `G`
  and `G_-i` and cancels exactly. Measured: pivotal `D_i` (exactly `coupling`
  agents touching) rises only ~7% (1.33e-2 -> 1.43e-2), and the pile-on case
  (`2*coupling` touching) is unchanged to 4 significant figures. **Use the drift
  arms in the dense/team-reward study, not as a fix for the DR magnitude gap.**
- **Config plumbing**: `algorithms/mappo_jax/run.py` used to `.get()` a hard-coded
  key list at each of its two `MultiBoxPushMJX` construction sites (bare, and as
  the macro wrapper's base), so an `env:` yaml key that only one site forwarded
  was silently ignored — which is why `coupling_def` / `max_steps` /
  `comm_radius` were unreachable from Hydra. Both sites now go through
  `_base_env_kwargs(env_config)` (`_BASE_ENV_KEYS` / `_RUNNER_ENV_KEYS`), which
  also **warns** on unrecognized `env:` keys. Nothing else validates the `env:`
  block — no dataclass guards it (`environments/types.py:EnvironmentParams` is
  never instantiated).
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
  runs and cannot desync from the shipped constants. **Checks 1–7 and 9–10 pass;
  [8] currently fails** — see the `_DEFAULT_DRIFT_SPEED` warning above. [8] is a
  config canary, not a code defect: it prints its numbers before asserting, and
  the message names the speed.

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
mechanic and the `variant` (`drift`/`trunc`) presets are deliberately **not**
carried over — do not set `env.variant` on this group; a wall touch ends the
episode as in the square-arena baseline.

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
    it does not bind: 9a/3o -> 3 rings of 3.00 (rim 9.0), 16a/4o -> 4 of 3.20
    (rim 12.8). The goal block therefore sits *below* the coupling/box-size
    block in `__init__` (it needs `box_half_extents`).
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
  `run.py` has one `elif` per supported env group
  (`MULTI_BOX_MJX` / `MULTI_BOX_MULTI_GOAL_MJX` / `MACRO_MJX`), each passing its
  constructor arguments explicitly — deliberately not a shared kwargs helper.
  Note this means an `env:` key is only reachable where a branch names it:
  `coupling_def` and `max_steps` are currently forwarded by no branch.

## Feudal MAPPO (`algorithms/feudal_mappo_jax/`) — WORK IN PROGRESS

A FeUdal-Networks-style (Vezhnevets et al. 2017) hierarchy being built on top of
a **copy** of `mappo_jax`: a manager emits a latent goal vector, a goal-conditioned
worker emits the primitive action. Not wired into `algorithms/types.py` /
`_dispatch` / `conf/` yet — nothing launches it.

- **The copy is now self-contained.** `trainer.py`, `run.py` and `mappo.py`
  originally imported `Transition`/`MAPPOConfig`/`sample_action`/
  `create_train_state`/`make_train` `from algorithms.mappo_jax...`, which left
  the local `network.py`/`mappo.py`/`types.py` inert — edits to them changed
  nothing. All 8 import sites now point at `algorithms.feudal_mappo_jax`, so the
  package no longer reads any `mappo_jax` code and diverging it is safe.
  Behavior-preserving: the only diffs against `algorithms/mappo_jax/*` are those
  import lines plus two error strings that named the wrong package (the files
  are otherwise byte-identical, and nothing outside the package imports it yet).
  The runner class is still called `MAPPO_JAX_Runner` — rename when it is wired
  into `_dispatch`.
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
  - **Still open, for whoever wires it up** (also in the module docstring):
    manager value `V^M` should **reuse `MAPPOCritic` on the global state with
    `n_outputs=n_agents`** — do not add another critic; `create_train_state`
    returns a 2-field `ActorCriticTrainState` and needs a third manager
    `TrainState` (plus the two checkpoint-restore sites in `run.py`);
    `trainer._actor_forward` flattens obs to `(b*n_agents, obs_dim)` for the fused
    shared-actor pass, so the goal must be flattened the same way *there*
    (`worker._broadcast_goal` handles the un-flattened layout only).
  - Self-check (8 groups: shapes/unit-norm, dilation writes group `t%r` only and
    covers all `r`, `jit`+`lax.scan` carry threading, `vmap`, mlp core is
    stateless + an unknown core raises, goal-semantics incl. done-masking, a
    manager→worker handshake through `bind_goal`+`sample_action` for both action
    spaces, and gradient routing under the detach rule — target arm zeroed, goal
    arm live, `r^I` fully detached, `f_Mspace` still trained). All pass:
    `uv run python -m algorithms.feudal_mappo_jax.manager`

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
