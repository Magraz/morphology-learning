# Observation Space Improvement Plan

## Context
The current observation space in `MultiBoxPushEnv` is 19 dims per agent: position (2), density sensors (16), and a binary touching-any-object flag (1). Agents lack velocity, coupling requirements, object progress, and delivery status — all of which are critical for coordination in the cooperative box-pushing task.

## Proposed Observation: 19 → 25 dims

| Indices | Content | Dims | Notes |
|---------|---------|------|-------|
| 0-1 | Agent position (relative to center) | 2 | **Unchanged** — `hypergraph.py:40` reads `obs[:, :2]` |
| 2-3 | Agent velocity (normalized) | 2 | **New** — `[vx, vy] / velocity_norm` |
| 4-11 | Agent density sensors (8 sectors) | 8 | Unchanged |
| 12-19 | Object density sensors (8 sectors) | 8 | Unchanged |
| 20 | Nearest object coupling requirement | 1 | **New** — `coupling / n_agents`, [0,1] |
| 21 | Agents touching nearest object | 1 | **New** — `n_touching / n_agents`, [0,1] |
| 22 | Nearest object progress to target | 1 | **New** — `1.0 - (dy / world_height)`, [0,1] |
| 23 | Is touching nearest object | 1 | **Replaces** old binary "touching any object" |
| 24 | Fraction of objects delivered | 1 | **New** — `len(delivered) / n_objects`, [0,1] |

"Nearest object" = closest non-delivered object by surface distance (`_agent_object_distance`). If all delivered, default to zeros.

## Justification per addition

- **Velocity (2)**: Feedforward MLP policy cannot infer motion from single-frame position. Highest value per dim.
- **Coupling requirement (1)**: Core coordination signal — agents need to know how many partners are needed.
- **Agents touching nearest (1)**: Combined with coupling, agents can compute the "deficit" and decide to join or leave.
- **Progress to target (1)**: Lets agents prioritize objects far from delivery.
- **Touching nearest (replaces old)**: More actionable than "touching any object."
- **Fraction delivered (1)**: Global task progress — helps agents shift focus to remaining objects.

## Files to modify

### Primary: `environments/multi_box_push/domain.py`
1. Update `observation_space` shape from `(n_agents, 19)` to `(n_agents, 25)` (~line 55)
2. Add `self.velocity_norm = self.world_width / 10.0` in `__init__`
3. Add `_get_nearest_object_info_all()` helper — for each agent, find nearest non-delivered object and return `[coupling_norm, n_touching_norm, progress_norm, is_touching]`
4. Update `_get_observation()` (~line 802) to concatenate: `[own_state, own_velocity, density_sensors, nearest_obj_info, fraction_delivered]`

### Verify only (no changes expected):
- `algorithms/mappo/hypergraph.py:40` — reads `obs[:, :2]` for positions, unaffected
- `algorithms/mappo/vec_trainer.py:70` — reads `obs_space.shape[1]` dynamically, auto-adapts
- `algorithms/mappo/mappo.py` — `observation_dim` flows from trainer, auto-adapts
- `algorithms/mappo/network.py` — all networks parameterized by `observation_dim`, auto-adapt

## Verification
1. Run `reset()` and `step()`, confirm obs shape is `(n_agents, 25)`
2. Confirm `obs[:, :2]` still returns valid positions (check hypergraph construction)
3. Run a short training loop to confirm no crashes and non-degenerate rewards
4. Note: saved model checkpoints will be incompatible due to input dim change
