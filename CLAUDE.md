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
- **Parallelism autoscaling (two nested layers).** Layer 1 is the sweep: with
  the joblib launcher each cross-product job runs in its own loky worker. Layer 2
  is per-job rollout collection: `make_vec_env` forks `env.n_envs` box2d
  subprocesses. The two multiply, so the budget is `n_jobs × n_envs ≲ cores`.
  A single top-level knob `n_jobs` (in `conf/config.yaml`, default `1`) drives
  both: `conf/config.yaml`'s `_self_` block sets `env.n_envs:
  ${envs_per_job:${n_jobs}}` → `usable_cores // n_jobs` **centrally** for every
  env group (it wins over the selected `conf/env/*` because `_self_` is last;
  pin a fixed count on the CLI with `env.n_envs=<N>`), and the
  `hydra/launcher=joblib_auto` group
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
- Demo/sanity check (scripted delivery rollout + vmapped throughput):
  `uv run python -m environments.mjx_suite.multi_box_push_mjx`. Step-matched
  wall-clock shootout vs Box2D (`profile_multi_box_push.py`, Box2D on all
  cores via AsyncVectorEnv, MJX vmapped on GPU): at 30a/6o, 100 eps x 1024
  steps, MJX 15.0s (6.8k steps/s) vs Box2D 56.5s (1.8k steps/s) — 3.8x, plus
  a one-time ~7s MJX compile. Not wired into `create_env` (torch MAPPO gains
  nothing from it), but it **is** the training env of the fully-jitted
  `mappo_jax` stack (below) via `EnvironmentEnum.MULTI_BOX_MJX =
  "multi_box_push_mjx"`.

## JAX MAPPO (`algorithms/mappo_jax/`)

`algorithm=mappo_jax` (`AlgorithmEnum.MAPPO_JAX`) is a fully-jitted MAPPO that
trains **directly on the functional MJX envs** (`MultiBoxPushMJX` and its
hierarchical macro wrapper `SyncMacroMJX`; the old JaxMARL dict-API path was
removed). It is a deliberate logic mirror of `mappo_vanilla` so runs are
drop-in comparable:

- **Same per-iteration cadence** (`run.py` ≙ `VecMAPPOTrainer.train`): jitted
  `collect_fn` (≙ `RolloutCollector.collect` — resets all envs at the top of
  every rollout, scans `params.n_steps` (the per-update batch is `n_steps *
  n_envs` env-steps in both stacks, scaling with parallelism), restarts envs that
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
  `critic_type` that `Model_Params` rejects). The central `env.n_envs` autoscale
  targets subprocess envs — for vmapped MJX pin it on the CLI:
  ```
  uv run python train.py algorithm=mappo_jax env=multi_box_push_mjx_9a_3o \
      model=mlp trial_id=0 env.n_envs=32
  # difference-rewards arm (same command, _dr env group):
  uv run python train.py algorithm=mappo_jax env=multi_box_push_mjx_9a_3o_dr \
      model=mlp trial_id=0 env.n_envs=32
  ```

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

    All three arms verified end-to-end (train + resume + evaluate); the `_dr`/
    `_wdr` critic heads are `n_agents`-wide, the actor a 4-way categorical. Launch:
    ```
    uv run python train.py algorithm=mappo_jax env=macro_mjx_9a_3o \
        model=mlp trial_id=0 env.n_envs=32
    # single-step difference-rewards arm (_dr) / windowed arm (_wdr):
    uv run python train.py algorithm=mappo_jax env=macro_mjx_9a_3o_dr \
        model=mlp trial_id=0 env.n_envs=32
    uv run python train.py algorithm=mappo_jax env=macro_mjx_9a_3o_wdr \
        model=mlp trial_id=0 env.n_envs=32
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
        model=mlp trial_id=0 env.n_envs=32
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
        model=mlp trial_id=0 env.n_envs=32
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
        env=macro_mjx_16a_4o_stagger_wdr_aligned model=mlp trial_id=0 env.n_envs=32
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
