# Add 8-Sector LiDAR Features to MultiBoxPush Observations

## Summary
Add an 8-value LiDAR vector (one value per existing sensing sector) in `environments/multi_box_push/domain.py` by casting Box2D rays from each agent and using closest-hit proximity (`1 - d/r`) over walls and objects, then append it to each agent observation.

## Implementation Changes
- Update observation interface in `MultiBoxPushEnv`:
  - Change observation shape from `(n_agents, 19)` to `(n_agents, 27)`.
  - New per-agent layout: `2 own_state + 16 density + 8 lidar + 1 touching = 27`.
- Add Box2D raycast callback for closest-hit detection:
  - Create a callback class (e.g. `_ClosestHitRayCastCallback(b2RayCastCallback)`) that stores `fixture`, `point`, `normal`, and `fraction`.
  - In `ReportFixture`, filter out non-LiDAR targets and return `fraction` to clip the ray and keep the nearest valid hit.
- Add LiDAR computation methods:
  - `_calculate_lidar_sensors_all(sensor_radius) -> (n_agents, 8)`.
  - `_raycast_lidar_distance(agent_idx, origin, angle, max_range)` that:
    - builds `p1` and `p2` (`p2 = p1 + max_range * direction`),
    - calls `self.world.RayCast(callback, p1, p2)`,
    - converts closest hit to distance via `distance = callback.fraction * max_range`.
- Use fixture/body filtering so LiDAR only considers intended geometry:
  - Include fixtures in `BOUNDARY_CATEGORY` and `OBJECT_CATEGORY`.
  - Exclude the casting agent's own body (and other agents unless intentionally desired).
- Use the same sector convention as existing density sensing:
  - `n_sectors = 8`, `sector_step = 2π/8`, `shift = 22.5°`.
  - Cast one ray at each sector center angle: `theta = shift + (s + 0.5) * sector_step`.
- Integrate in `_get_observation()`:
  - Keep cached positions (`_agent_pos_cache`, `_object_pos_cache`) for density/touch logic.
  - Compute `all_density_sensors` and `all_lidar_sensors` once per step.
  - Concatenate LiDAR before touch flag.
- Keep MAPPO pipeline unchanged:
  - `VecMAPPOTrainer` already reads obs size from env, so no hard-coded update needed.
  - Existing checkpoints trained with obs dim 19 are not load-compatible with dim 27.

## Test Plan
- Shape and sanity:
  - `env.reset()` and `env.step()` return obs shape `(n_agents, 27)`.
  - LiDAR slice is always in `[0, 1]`.
- Raycast correctness:
  - For a ray with no intersection in range, callback has no fixture and LiDAR value is `0.0`.
  - For a hit, LiDAR value equals `1.0 - (callback.fraction * range) / range`.
  - Closest-hit behavior is validated by placing multiple fixtures collinear with the ray.
- Behavioral checks:
  - Agent near left wall: sector(s) pointing left should have high proximity; opposite sectors low.
  - Place object in a known sector: that sector's LiDAR increases vs empty baseline.
  - Empty-object case still gives wall-driven LiDAR signal.
- Regression checks:
  - Density sensor values unchanged.
  - Training startup still works and derives new obs dim automatically.

## Assumptions and Defaults
- LiDAR detects walls and objects (not agents).
- Encoding is nearest-hit proximity from Box2D raycast (`0` means no hit within range).
- LiDAR range uses existing `self.sector_sensor_radius`.
- One ray per sector (sector-center ray), not full angular sweep.
- Geometry intersection is exact at fixture level via Box2D raycasting (no custom wall math or object bounding-circle approximation).
