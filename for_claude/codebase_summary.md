# Morphology Learning — Codebase Summary

## Overview

A multi-agent reinforcement learning framework built around **Box2D** physics environments. The project provides a modular pipeline: define an environment, pick an algorithm, configure via YAML, and run experiments with reproducible seeds and logging. The core research focus is on agents learning to coordinate (e.g., forming chains, pushing objects) in 2D physics worlds.

---

## Project Structure

```
run_trial.py          # CLI entry point — dispatches to the correct algorithm runner
train.sh / evaluate.sh # Shell wrappers for batch training / evaluation
algorithms/           # RL algorithm implementations (IPPO, MAPPO, CEM-RL)
builders/             # Python scripts that generate experiment YAML configs
environments/         # Gymnasium environments (Box2D Salp, Multi-Box Push, MPE wrappers)
experiments/          # YAML configs + results (models, logs, plots, videos)
plotting/             # Notebook for plotting training curves
utils/                # Misc utilities (alignment loss)
```

---

## Entry Point (`run_trial.py`)

CLI args: `--batch`, `--name`, `--algorithm`, `--environment`, `--trial_id`, `--view`, `--checkpoint`, `--evaluate`.

1. Loads `_env.yaml` (environment params) and `{name}.yaml` (experiment/hyperparams) from `experiments/yamls/{batch}/`.
2. Instantiates the appropriate `Runner` subclass (IPPO or MAPPO).
3. Calls `runner.train()`, `runner.view()`, or `runner.evaluate()`.

---

## Algorithms (`algorithms/`)

### Shared Infrastructure

| File | Purpose |
|---|---|
| `algorithms.py` | `run_algorithm()` — master dispatcher. Loads YAML configs, creates the correct Runner. |
| `runner.py` | `Runner` base class. Sets up directories (`logs/`, `models/`, `videos/`), configures PyTorch threads. Subclassed by each algorithm. |
| `create_env.py` | `create_env()` factory — returns `(env, state_dim, action_dim)`. `make_vec_env()` — returns `AsyncVectorEnv` or `SyncVectorEnv` for parallel training. Also contains `PettingZooToGymWrapper` for MPE environments. |
| `env_wrapper.py` | `EnvWrapper` — thin adapter that normalizes the `step()` return values across environment types (MPE vs Box2D). |
| `types.py` | `AlgorithmEnum` — string enum with values: `ccea`, `ippo`, `ppo`, `mappo`, `ppo_parallel`, `td3`, `none`. |
| `plotter.py` | Standalone script to load `.pkl` training stats and generate matplotlib plots per metric. |

### IPPO (`algorithms/ippo/`)

Independent PPO — each agent has its own policy (or shares one via `parameter_sharing` flag).

| File | Purpose |
|---|---|
| `ippo.py` | `PPOAgent` class. Owns a policy + old policy, optimizer, and rollout buffer. Implements GAE advantage computation and clipped PPO update with minibatch DataLoader. |
| `trainer.py` | `IPPOTrainer`. Orchestrates trajectory collection across agents, handles parameter sharing (merges buffers into agent 0 then does a single update), periodic evaluation, checkpointing. |
| `run.py` | `IPPO_Runner(Runner)`. Wires up env creation, seed setting, trainer init. Implements `train()`, `view()`. |
| `types.py` | `Params` dataclass (PPO hyperparams) and `Experiment` dataclass (device, model name, params). |
| `network.py` | `PPONetwork` — supports Dict action spaces (movement continuous + link_openness discrete + detach continuous). Legacy fallback for simple Box actions. |

#### Models (`algorithms/ippo/models/`)

| Model | Description |
|---|---|
| `mlp_ac.py` | `MLP_AC` — standard Actor-Critic MLP. Supports both **discrete** (Categorical) and **continuous** (Normal) action distributions. Orthogonal weight init. Separate actor/critic networks (not shared). |
| `hybrid_mlp_ac.py` | `Hybrid_MLP_AC` — hybrid action head: continuous movement (2D Normal), discrete attach (Bernoulli), discrete detach (Bernoulli). Combined log-prob across all three heads. Used for the Salp environment. |
| `mlp.py` | `MLP` — simpler shared-trunk actor-critic with tanh squashing. Continuous actions only. |

### MAPPO (`algorithms/mappo/`)

Multi-Agent PPO with a **centralized critic** (takes global state = concatenation of all agent observations) and decentralized actors.

| File | Purpose |
|---|---|
| `mappo.py` | `MAPPOAgent`. Single network with N actor heads + 1 centralized critic. Buffers indexed by `[env_idx][agent_idx]` for vectorized training. GAE computed per-environment. |
| `network.py` | `MAPPONetwork` containing `MAPPOActor` (or `MAPPO_Hybrid_Actor`) + `MAPPOCritic`. Actors can be shared or independent. Hybrid actor has same movement/attach/detach structure as IPPO hybrid. Critic input is the full global state. |
| `trainer.py` | `MAPPOTrainer` — single-env trainer. Same collect→update loop as IPPO but with global state construction. |
| `vec_trainer.py` | `VecMAPPOTrainer` — **vectorized** trainer using `AsyncVectorEnv`. Collects trajectories from N parallel envs simultaneously. Tracks timing stats (collection, update, eval times, FPS). |
| `run.py` | `MAPPO_Runner(Runner)`. Currently uses `VecMAPPOTrainer` by default. |
| `types.py` | Same structure as IPPO types. |

### CEM-RL (`algorithms/cem_rl/`) — *Work in Progress*

Cross-Entropy Method + RL hybrid (evolutionary + gradient-based). Originally built for a PettingZoo MuJoCo walker environment.

| File | Purpose |
|---|---|
| `cem_rl.py` | `CEM_RL` class. Maintains a population of flat parameter vectors. First half evaluated directly, second half trained with TD3 gradient steps. Updates population mean/covariance from elites. Includes `rollout_actor()` which tracks foot-ground contacts for behavior descriptors (QD-style). |
| `td3.py` | `TD3` — Twin Delayed DDPG. `Actor` (3-layer MLP) + `Critic` (twin Q-networks). Standard TD3 update logic. |
| `trainer.py` | `CEMRL_Trainer` — scaffolding for the CEM-RL loop. Creates TD3 + replay buffer, runs `create_population → evaluate → train_rl → update` loop. |
| `run.py` | `CEMRL_Runner(Runner)` — only wired for `BOX2D_SALP` so far. |

---

## Environments (`environments/`)

### Shared Types

| File | Purpose |
|---|---|
| `types.py` | `EnvironmentEnum` (`mpe_spread`, `mpe_simple`, `box2d_salp`, `multi_box_push`). `EnvironmentParams` dataclass (`environment`, `n_envs`, `n_agents`, `state_representation`). |
| `make_vec_env.py` | Alternative vec env factory (only for `BOX2D_SALP`). |

### Box2D Salp Chain (`environments/box2d_salp/`)

A 2D multi-agent environment where circular agents move in a bounded arena, can **form joints** (chains) with nearby agents, and navigate toward **target areas** that require a minimum chain size (coupling requirement).

**`domain.py` — `SalpChainEnv(gym.Env)`**
- **World**: 40×40 Box2D, zero gravity, boundary walls.
- **Agents**: Dynamic circles (radius 0.4), high linear damping.
- **Actions** (per agent, shape `(n_agents, 4)`): `[force_x, force_y, attach_signal, detach_signal]`.
- **Joints**: `b2RevoluteJoint` created via `_join_on_proximity()` when two open agents are within 1.5 units. Max 2 joints per agent. Broken when detach signal fires.
- **UnionFind**: Tracks connected components efficiently.
- **Observations** (per agent, dim 18): own position (centered) + 4-sector agent densities + 4-sector target densities + relative coords to nearest non-connected agent + relative coords to nearest target.
- **Density sensors**: 4 quadrant sectors with inverse-distance weighting. Skips connected agents.
- **Rewards**: Target area coupling reward (agents in a chain ≥ coupling_req inside a target radius get reward) + proximity shaping reward (getting closer to other agents and targets).
- **Termination**: Boundary collision → terminated. Max steps → truncated.
- **Rendering**: Pygame. Agents colored green (open) / red (closed). Draws joints, force vectors, density sector arcs, target areas, agent indices.
- **Interactive `__main__`**: Arrow keys to move active agent, Space to switch agent, G for group control.

**`utils.py`**
- `UnionFind` — path compression + union by rank.
- `TargetArea` — circular target with coupling requirement. `calculate_reward()` checks which connected components have enough agents inside.
- `BoundaryContactListener` — detects agent-wall collisions.
- `get_scatter_positions()` — random positions with min distance constraint.
- `get_linear_positions()` — horizontal line centered on world.
- `fixed_position_target_area()` / `dynamic_position_target_area()` — target placement strategies.
- Color constants and `add_dictionary_values()` helper.

**`wrapper.py`** — JAX2D wrapper (references a `domain_jax.py` that wraps the env for JAX compatibility).

### Multi-Box Push (`environments/multi_box_push/`)

A variant of the Salp environment focused on **cooperative object pushing**. Agents must push dynamic objects into a rectangular drop zone.

**`domain.py` — `MultiBoxPushEnv(gym.Env)`**
- **World**: 30×30 Box2D, zero gravity, boundary walls.
- **Agents**: Same circular agents as Salp (radius 0.4), high damping. Can collide with `AGENT_CATEGORY`, `BOUNDARY_CATEGORY`, and `OBJECT_CATEGORY`.
- **Dynamic Objects** (`_create_dynamic_objects()`): Currently 1 square (box shape 1.5×1.5). Configured with density=1.0, linearDamping=5.0. Supports squares, triangles, and circles.
- **Target Area**: `ObjectTargetArea` — a rectangle at the top of the world (`world_width-1` wide, 5 tall). `contains_object(body)` checks AABB containment.
- **Actions** (per agent, shape `(n_agents, 4)`): `[force_x, force_y, attach_signal, detach_signal]`. Force multiplier = 100.
- **Observations** (per agent, dim 22): own position (2) + 8-sector agent densities + 8-sector object densities + relative coords to nearest agent (2) + relative coords to nearest object (2).
- **Density Sensors**: 8 sectors (45° each), shifted 22.5° counter-clockwise. Agent densities summed, object densities take max per sector.
- **Rewards** (`_calculate_box_push_reward()`): Distance shaping — reward = `(prev_dist - curr_dist) * 10.0` for Y-distance to target. +10.0 bonus when object enters the drop zone. Shared across all agents.
- **Termination**: Boundary collision → terminated. Object in drop zone → terminated. Max steps → truncated.
- **Rendering**: Same Pygame stack as Salp. Draws rectangular drop zone ("DROP ZONE" label), dynamic objects with outlines, 8-sector density visualization with two-line text (A:/T: values).
- **Interactive `__main__`**: Same controls as Salp + G key for group control (replicates forces to agents within 5-unit radius).

**`utils.py`**
- Same `UnionFind`, `BoundaryContactListener`, position generators as Salp utils.
- `OBJECT_CATEGORY = 0x0004` — collision filter for dynamic objects.
- `ObjectTargetArea` — rectangular target with `contains_object(body)` AABB check.

### MPE Wrappers

PettingZoo MPE environments (`simple_spread_v3`, `simple_v3`) wrapped via `PettingZooToGymWrapper` in `create_env.py`. Converts dict-based PettingZoo API to stacked numpy arrays compatible with the training pipeline. Supports discrete and continuous action spaces.

---

## Experiment Management

### Builders (`builders/`)

Python scripts that programmatically generate YAML experiment configs.

- `ippo.py` — generates configs for IPPO experiments on `BOX2D_SALP`.
- `ppo.py` — generates configs for PPO experiments (references `VMAS_SALP_PASSAGE`).
- `experiment_builder.ipynb` — notebook that imports a builder, runs it, and writes YAML files to `experiments/yamls/`.
- `builder.yaml` — points to which builder module to use.

### YAML Config Structure

Each experiment batch has a folder under `experiments/yamls/{batch_name}/`:
- `_env.yaml` — environment params: `n_agents`, `n_envs`, `state_representation`.
- `{experiment_name}.yaml` — algorithm hyperparams: `device`, `model`, `params` (learning rate, gamma, clip, batch size, epochs, seeds, etc.). `parameter_sharing: true/false` controls whether agents share a single policy.

### Results

Stored under `experiments/results/{batch_name}/{experiment_name}/{trial_id}/`:
- `models/` — `models_checkpoint.pth`, `models_finished.pth`
- `logs/` — `training_stats_checkpoint.pkl` (dict of lists: rewards, losses, step counts, timing)
- `plots/` — auto-generated from plotter
- `videos/` — (placeholder)

---

## Key Design Patterns

1. **Runner Pattern**: Each algorithm has a `Runner` subclass that handles env creation, trainer wiring, and train/view/evaluate dispatch.
2. **Parameter Sharing**: Both IPPO and MAPPO support a `parameter_sharing` flag. When true, all agents share a single network — IPPO merges buffers before updating, MAPPO uses a single actor network for all agents.
3. **Hybrid Action Spaces**: The Box2D environments use 4D actions (movement_x, movement_y, attach, detach). The `Hybrid_MLP_AC` and `MAPPO_Hybrid_Actor` models handle mixed continuous+discrete action distributions with combined log-probs.
4. **Density Sensors**: Both environments use sector-based density sensing. Agents observe how many other agents/objects are in each directional sector around them, providing a scalable local observation that doesn't depend on agent count.
5. **UnionFind for Chains**: Connected agent groups are tracked with a Union-Find data structure, enabling efficient queries like "are agents i and j in the same chain?" and "how big is the largest chain?".
6. **Vectorized Training**: MAPPO's `VecMAPPOTrainer` uses Gymnasium's `AsyncVectorEnv` to run multiple environments in parallel, with timing breakdowns logged.

---

## Current Experiment Batches

| Batch | Env | Agents | Envs | Algorithms |
|---|---|---|---|---|
| `box2d_salp_test` | `box2d_salp` | 2 | 4 | IPPO, MAPPO |
| `mpe_simple_test` | `mpe_simple` | — | — | IPPO |
| `mpe_spread_novec` | `mpe_spread` | — | — | IPPO |
| `mpe_spread_vec` | `mpe_spread` | — | — | IPPO |

---

## Dependencies

- `gymnasium[box2d]`, `pygame`, `Box2D`, `torch`, `numpy`, `matplotlib`, `pickle`, `pyyaml`
- Optional: `mpe2` (PettingZoo MPE), `jax` (JAX wrapper), `dhg` (hypergraph viz)
- Python 3.12
