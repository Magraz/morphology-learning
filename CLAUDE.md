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
  (`batch_size`, `val_coef`) and publishes a `hyperedges` map; model overrides
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
- `nearest_box_vec` (2) — relative (dx, dy) to the nearest object, per axis
  normalized by `world_width`; zero vector when the env has no objects.
  Egocentric (no absolute world anchor).
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
