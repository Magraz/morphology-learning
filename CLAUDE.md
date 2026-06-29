Whenever building new code, try to reuse as much code as possible. If the new functionality overlaps heavily with other parts of the code, find a way to abstract and reuse the logic instead of duplicating the functionality.

Always keep the CLAUDE.md file up to date to reflect the current functionality and architecture of the code.

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