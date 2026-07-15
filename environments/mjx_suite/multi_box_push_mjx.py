"""MuJoCo-MJX port of the Box2D ``MultiBoxPushEnv`` (multi_box_push.py).

The physics is constrained to 2D by construction: every body only owns planar
DOFs (agents: slide-x + slide-y; boxes: slide-x + slide-y + hinge-yaw), gravity
is zero, and the arena walls are four inward-facing planes. There is no z DOF
anywhere, so MJX literally cannot compute out-of-plane motion — no springs,
no clamping, no wasted 3D dynamics.

Functional JAX API (gymnax-style), fully ``jit``/``vmap``-able::

    env = MultiBoxPushMJX(n_agents=9, n_objects=3)
    obs, state = jax.jit(env.reset)(key)                       # obs (A, 40)
    obs, state, reward, terminated, truncated, info = jax.jit(env.step)(state, actions)

No auto-reset: when ``terminated | truncated`` the caller resets (or wraps with
its own auto-reset logic before ``vmap``).

Parity with the Box2D env (same layout as ``observation.py``'s OBS_DIM = 40):

- observation: own_velocity(2) + density_sensors(16) + is_touching_object(1)
  + neighbor_fraction(1) + contact_force(1) + nearest_box_vec(2)
  + goal_distance(1) + lidar(16), identical normalizations.
- reward: per-step shaping toward the top target band + one-time +100 per
  delivered box (``reward_mode="dense"`` keeps shaping, ``"sparse"`` doesn't);
  terminate when all boxes delivered or any agent touches a wall.
- coupling mechanic: a box keeps its heavy base mass until at least
  ``coupling`` agents touch it, then drops to the light coupled mass. Done by
  overriding ``body_mass``/``body_inertia``/``dof_damping`` on the mjx.Model
  pytree each step (the model is a step argument, so this is jit-safe).
- physics constants: dt = 1/60, agent mass 1 / radius 0.4 / damping 10, box 2D
  density 20 (0.05 * coupling when coupled), box linear/angular damping 5/8.
  Box2D's body damping is emulated with joint damping = coeff * mass (inertia
  for the hinge), which has the same steady state (v_terminal = F / (m * d)).

Differences from Box2D worth knowing:
- Spawns use shuffled jittered grids instead of rejection sampling (jit needs
  static shapes); same regions and min separations.
- Reward shaping is live from the first step (Box2D pays 0 shaping on step 1).
- Box sizes are fixed per instance (they already were in Box2D — derived from
  the coupling list at __init__).

Run the built-in demo / sanity check (no display needed):
    uv run python -m environments.mjx_suite.multi_box_push_mjx
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from environments.box2d_suite.observation import N_LIDAR_RAYS, OBS_DIM
from environments.box2d_suite.utils import COLORS_LIST, ObjectTargetArea

_AGENT_RADIUS = 0.4
_AGENT_MASS = 1.0  # Box2D default-mass fallback for zero-density fixtures
_AGENT_DAMPING = 10.0
_BOX_LIN_DAMPING = 5.0
_BOX_ANG_DAMPING = 8.0
_BOX_BASE_DENSITY_2D = 20.0  # Box2D kg/m^2
_COUPLED_DENSITY_PER_AGENT = 0.05
_TOUCH_EPS = 0.2  # agent counts as touching within radius + eps of a box face
_BOX_HALF_HEIGHT = 0.4  # z half-extent; cosmetic (no z DOF), keeps contacts planar
_FORCE_MULTIPLIER = 100.0
_TIME_STEP = 1.0 / 60.0
_N_SECTORS = 8
_LIDAR_EPS = 1e-3  # ray origin offset past the agent surface (self-hit guard)
_WALL_EPS = 0.01  # boundary-contact slack, ~Box2D contact slop


def _impl(data):
    """mjx 3.10 moved contact/efc arrays behind Data._impl."""
    return getattr(data, "_impl", data)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EnvState:
    data: mjx.Data
    t: jax.Array  # () int32 — steps taken this episode
    prev_box_goal_dist: jax.Array  # (O,) signed y-distance box -> band center
    delivered: jax.Array  # (O,) bool


class MultiBoxPushMJX:
    def __init__(
        self,
        n_agents: int = 3,
        n_objects: int = 3,
        coupling_def: str = "even",
        max_steps: int = 1024,
        reward_mode: str = "dense",
        comm_radius: float | None = None,
    ):
        self.n_agents = n_agents
        self.n_objects = n_objects
        self.max_steps = max_steps
        self.reward_mode = reward_mode

        # --- world geometry (identical to the Box2D env) ---
        total_entities = n_agents + n_objects
        self.world_width = int(30 * max(1.0, total_entities / 8) ** 0.5)
        self.world_height = self.world_width
        self.world_center_x = self.world_width // 2
        self.world_center_y = self.world_height // 2
        self.boundary_thickness = 0.5

        self.velocity_norm = self.world_width / 10.0
        self.neighbor_detection_range = 3.0
        self.sector_sensor_radius = self.world_width / 3.0
        self.lidar_range = self.sector_sensor_radius
        self.comm_radius = (
            self.world_width / 3.0 if comm_radius is None else float(comm_radius)
        )
        self.force_multiplier = _FORCE_MULTIPLIER

        # --- target band spanning the top wall ---
        bt = self.boundary_thickness
        target_h = max(5.0, 5.0 * self.world_height / 30.0)
        self.target_x = self.world_width / 2
        self.target_y = self.world_height - bt - target_h / 2
        self.target_half_w = (self.world_width - 2 * bt) / 2
        self.target_half_h = target_h / 2
        # Box2D-suite target object (numpy-only, never touched by jitted code):
        # lets the shared Renderer machinery draw the band exactly as for Box2D.
        self.target_areas = [
            ObjectTargetArea(
                self.target_x,
                self.target_y,
                2 * self.target_half_w,
                2 * self.target_half_h,
            )
        ]

        # --- coupling requirements and box sizes (fixed per instance) ---
        if coupling_def == "random":
            coupling = np.random.default_rng(42).integers(
                2, (n_agents // 2) + 1, n_objects
            )
        elif coupling_def == "even":
            coupling = np.array([n_agents // n_objects] * n_objects)
        else:
            raise ValueError(f"unknown coupling_def: {coupling_def}")
        self.objects_push_coupling_list = coupling
        self.box_half_extents = np.maximum(1.5, coupling * _AGENT_RADIUS)

        heavy_mass = _BOX_BASE_DENSITY_2D * (2 * self.box_half_extents) ** 2
        light_mass = (
            _COUPLED_DENSITY_PER_AGENT * coupling * (2 * self.box_half_extents) ** 2
        )

        # --- build & compile the planar MuJoCo model ---
        self._heavy_mass_np = heavy_mass  # kept for the visual model rebuild
        self._mj_model = mujoco.MjModel.from_xml_string(self._build_xml(heavy_mass))
        self._model = mjx.put_model(self._mj_model)

        # --- static lookups (numpy at init, jnp where consumed traced) ---
        m = self._mj_model
        agent_bodies = np.array(
            [m.body(f"agent_{i}").id for i in range(n_agents)], dtype=np.int32
        )
        box_bodies = np.array(
            [m.body(f"box_{j}").id for j in range(n_objects)], dtype=np.int32
        )
        self._agent_body_ids = jnp.asarray(agent_bodies)
        self._box_body_ids = jnp.asarray(box_bodies)

        def qadr(body):  # first qpos address of a body's joint chain
            return int(m.jnt_qposadr[m.body_jntadr[body]])

        def dadr(body):
            return int(m.jnt_dofadr[m.body_jntadr[body]])

        self._agent_qadr = jnp.asarray(
            [[qadr(b), qadr(b) + 1] for b in agent_bodies]
        )  # (A, 2)
        self._agent_dadr = jnp.asarray([[dadr(b), dadr(b) + 1] for b in agent_bodies])
        self._box_qadr = jnp.asarray(
            [[qadr(b), qadr(b) + 1, qadr(b) + 2] for b in box_bodies]
        )  # (O, 3) — x, y, yaw
        box_dadr = np.array([[dadr(b), dadr(b) + 1, dadr(b) + 2] for b in box_bodies])
        self._box_dof_lin = jnp.asarray(box_dadr[:, :2].ravel())  # (2O,)
        self._box_dof_ang = jnp.asarray(box_dadr[:, 2])  # (O,)

        # geom id -> agent/object index maps (-1 = neither), for contact forces
        agent_of_geom = -np.ones(m.ngeom, dtype=np.int32)
        object_of_geom = -np.ones(m.ngeom, dtype=np.int32)
        for i in range(n_agents):
            agent_of_geom[m.geom(f"g_agent_{i}").id] = i
        for j in range(n_objects):
            object_of_geom[m.geom(f"g_box_{j}").id] = j
        self._agent_of_geom = jnp.asarray(agent_of_geom)
        self._object_of_geom = jnp.asarray(object_of_geom)

        # heavy/light overrides for the coupling mechanic
        self._coupling = jnp.asarray(coupling, dtype=jnp.int32)
        self._heavy_mass = jnp.asarray(heavy_mass, dtype=jnp.float32)
        self._light_mass = jnp.asarray(light_mass, dtype=jnp.float32)
        self._heavy_inertia = self._model.body_inertia[self._box_body_ids]  # (O, 3)
        self._box_half = jnp.asarray(self.box_half_extents, dtype=jnp.float32)

        self._lidar_dirs = jnp.asarray(
            np.stack(
                [
                    np.cos(np.arange(N_LIDAR_RAYS) * 2 * np.pi / N_LIDAR_RAYS),
                    np.sin(np.arange(N_LIDAR_RAYS) * 2 * np.pi / N_LIDAR_RAYS),
                    np.zeros(N_LIDAR_RAYS),
                ],
                axis=1,
            ),
            dtype=jnp.float32,
        )  # (R, 3)

        self._agent_spawn_grid = self._make_spawn_grid()
        self._box_spawn_slots = self._make_box_slots()

        self.observation_dim = OBS_DIM
        self.action_dim = 2

    # ------------------------------------------------------------------ model

    def _build_xml(self, heavy_mass: np.ndarray, visual: bool = False) -> str:
        """MJCF for the planar model.

        ``visual=False`` (the physics/obs model handed to MJX): walls are four
        inward-facing planes, no floor, no colors that matter. ``visual=True``
        builds the *native-rendering* twin used only by ``MuJoCoNativeRenderer``
        — identical bodies/joints (same nq/qpos layout, so qpos copies across)
        but with contype-0 cosmetic geometry: slim wall boxes instead of the
        giant planes, a checkered floor below the bodies, the target band
        painted on it, a skybox and a shadow-casting light. It is never
        stepped, only mj_forward'd for camera rendering.
        """
        W, H, bt = self.world_width, self.world_height, self.boundary_thickness
        cx, cy = W / 2, H / 2
        # Box colors follow the Box2D scheme: COLORS_LIST offset by n_agents.
        box_rgba = [
            "{:.3f} {:.3f} {:.3f} 1".format(
                *(c / 255 for c in COLORS_LIST[(self.n_agents + j) % len(COLORS_LIST)])
            )
            for j in range(self.n_objects)
        ]
        agent_rgba = "0.78 0.2 0.2 1"  # Box2D agent disc red

        parts = [
            "<mujoco>",
            # implicitfast integrates joint damping implicitly — same semantics
            # as Box2D's v /= (1 + damping * dt). Pyramidal cone (the default):
            # elliptic NaNs out when a light coupled box is crushed against a
            # wall by many agents.
            f'  <option timestep="{_TIME_STEP}" gravity="0 0 0" integrator="implicitfast"/>',
        ]
        if visual:
            parts += [
                '  <visual><global offwidth="1440" offheight="1440"/>'
                '<map znear="0.05"/></visual>',
                "  <asset>",
                '    <texture type="skybox" builtin="gradient" rgb1="0.95 0.97 1" '
                'rgb2="0.55 0.7 0.9" width="256" height="256"/>',
                '    <texture name="floor_tex" type="2d" builtin="checker" '
                'rgb1="0.93 0.93 0.93" rgb2="0.83 0.83 0.86" width="256" height="256"/>',
                '    <material name="floor_mat" texture="floor_tex" '
                f'texrepeat="{W // 3} {H // 3}" reflectance="0.05"/>',
                "  </asset>",
            ]
        else:
            # mjx.ray indexes mat_rgba even when no geom has a material; an
            # empty material table crashes it, so ship one dummy material.
            parts.append(
                '  <asset><material name="_raycast_workaround" rgba="1 1 1 1"/></asset>'
            )
        parts.append("  <worldbody>")
        if visual:
            wall_h, wall_z = 0.8, 0.39  # spans z in [-0.41, 1.19]
            cosmetic = 'contype="0" conaffinity="0"'
            parts += [
                f'    <light directional="true" pos="{cx} {cy - H / 4} 40" '
                'dir="0.1 0.15 -1" diffuse="0.85 0.85 0.85" castshadow="true"/>',
                f'    <geom name="floor" type="plane" pos="{cx} {cy} -0.41" '
                f'size="{cx} {cy} 0.1" material="floor_mat" {cosmetic}/>',
                f'    <geom name="target_band" type="box" '
                f'pos="{self.target_x} {self.target_y} -0.385" '
                f'size="{self.target_half_w} {self.target_half_h} 0.02" '
                f'rgba="0.2 0.78 0.2 0.45" {cosmetic}/>',
                f'    <geom name="wall_left" type="box" pos="{bt / 2} {cy} {wall_z}" '
                f'size="{bt / 2} {cy} {wall_h}" rgba="0.35 0.35 0.38 1" {cosmetic}/>',
                f'    <geom name="wall_right" type="box" pos="{W - bt / 2} {cy} {wall_z}" '
                f'size="{bt / 2} {cy} {wall_h}" rgba="0.35 0.35 0.38 1" {cosmetic}/>',
                f'    <geom name="wall_bottom" type="box" pos="{cx} {bt / 2} {wall_z}" '
                f'size="{cx} {bt / 2} {wall_h}" rgba="0.35 0.35 0.38 1" {cosmetic}/>',
                f'    <geom name="wall_top" type="box" pos="{cx} {H - bt / 2} {wall_z}" '
                f'size="{cx} {bt / 2} {wall_h}" rgba="0.35 0.35 0.38 1" {cosmetic}/>',
            ]
        else:
            parts += [
                # Four inward-facing planes as walls. Low friction so contact
                # friction (elementwise max) stays governed by the dynamic geom,
                # approximating Box2D's sqrt(f1*f2) combine.
                f'    <geom name="wall_left" type="plane" pos="{bt} {cy} 0" zaxis="1 0 0" size="{H} {H} 0.1" friction="0.05"/>',
                f'    <geom name="wall_right" type="plane" pos="{W - bt} {cy} 0" zaxis="-1 0 0" size="{H} {H} 0.1" friction="0.05"/>',
                f'    <geom name="wall_bottom" type="plane" pos="{cx} {bt} 0" zaxis="0 1 0" size="{W} {W} 0.1" friction="0.05"/>',
                f'    <geom name="wall_top" type="plane" pos="{cx} {H - bt} 0" zaxis="0 -1 0" size="{W} {W} 0.1" friction="0.05"/>',
            ]
        for i in range(self.n_agents):
            parts += [
                f'    <body name="agent_{i}" pos="0 0 0">',
                f'      <joint name="agent_{i}_x" type="slide" axis="1 0 0" damping="{_AGENT_DAMPING * _AGENT_MASS}"/>',
                f'      <joint name="agent_{i}_y" type="slide" axis="0 1 0" damping="{_AGENT_DAMPING * _AGENT_MASS}"/>',
                f'      <geom name="g_agent_{i}" type="sphere" size="{_AGENT_RADIUS}" mass="{_AGENT_MASS}" friction="0.2" rgba="{agent_rgba}"/>',
                "    </body>",
            ]
        for j in range(self.n_objects):
            h = self.box_half_extents[j]
            mass = heavy_mass[j]
            izz = mass * (2 * h) ** 2 / 6.0  # thin-box yaw inertia (z extent inert)
            parts += [
                f'    <body name="box_{j}" pos="0 0 0">',
                f'      <joint name="box_{j}_x" type="slide" axis="1 0 0" damping="{_BOX_LIN_DAMPING * mass}"/>',
                f'      <joint name="box_{j}_y" type="slide" axis="0 1 0" damping="{_BOX_LIN_DAMPING * mass}"/>',
                f'      <joint name="box_{j}_yaw" type="hinge" axis="0 0 1" damping="{_BOX_ANG_DAMPING * izz}"/>',
                f'      <geom name="g_box_{j}" type="box" size="{h} {h} {_BOX_HALF_HEIGHT}" mass="{mass}" friction="0.3" rgba="{box_rgba[j]}"/>',
                "    </body>",
            ]
        parts.append("  </worldbody>")
        parts.append("  <actuator>")
        for i in range(self.n_agents):
            for ax in ("x", "y"):
                parts.append(
                    f'    <motor joint="agent_{i}_{ax}" gear="{_FORCE_MULTIPLIER}" '
                    'ctrlrange="-1 1" ctrllimited="true"/>'
                )
        parts.append("  </actuator>")
        parts.append("</mujoco>")
        return "\n".join(parts)

    # ------------------------------------------------------------------ spawns

    def _make_spawn_grid(self) -> jnp.ndarray:
        """Candidate agent spawn cells: bottom third, jitter-safe >=2 apart."""
        margin, min_dist = 2.0, 2.0
        spacing = min_dist + 0.5
        x_lo, x_hi = margin, self.world_width - margin
        y_lo, y_hi = margin, self.world_height / 3 - margin
        n_cols = int((x_hi - x_lo) // spacing) + 1
        n_rows = int((y_hi - y_lo) // spacing) + 1
        if n_cols * n_rows < self.n_agents:
            raise ValueError("spawn region too small for n_agents")
        xs = np.linspace(x_lo, x_hi, n_cols)
        ys = np.linspace(y_lo, y_hi, n_rows)
        gap = min(xs[1] - xs[0] if n_cols > 1 else np.inf,
                  ys[1] - ys[0] if n_rows > 1 else np.inf)
        self._agent_spawn_jitter = max(0.0, (gap - min_dist) / 2)
        gx, gy = np.meshgrid(xs, ys)
        return jnp.asarray(
            np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=jnp.float32
        )  # (n_cells, 2)

    def _make_box_slots(self) -> jnp.ndarray:
        """Candidate box x-slots in the central spawn band, min separation kept."""
        h_max = float(self.box_half_extents.max())
        spacing = max(4.0, 2 * h_max + 1.0)  # Box2D min_separation / min_x_separation
        spawn_w = self.world_width * 0.8
        x_lo = self.world_center_x - spawn_w / 2 + h_max
        x_hi = self.world_center_x + spawn_w / 2 - h_max
        n_slots = int((x_hi - x_lo) // spacing) + 1
        if n_slots < self.n_objects:
            raise ValueError("spawn band too narrow for n_objects")
        return jnp.asarray(np.linspace(x_lo, x_hi, n_slots), dtype=jnp.float32)

    # ------------------------------------------------------------------ helpers

    def _agent_pos(self, data) -> jnp.ndarray:
        return data.qpos[self._agent_qadr]  # (A, 2)

    def _agent_vel(self, data) -> jnp.ndarray:
        return data.qvel[self._agent_dadr]  # (A, 2)

    def _box_pose(self, data) -> tuple[jnp.ndarray, jnp.ndarray]:
        q = data.qpos[self._box_qadr]  # (O, 3)
        return q[:, :2], q[:, 2]  # positions (O, 2), yaws (O,)

    def _touch_matrix(self, agent_pos, box_pos, box_yaw) -> jnp.ndarray:
        """(A, O) bool — agent within radius + eps of a (rotated) box surface.

        Port of ObservationManager._agent_object_distance: rotate into box
        frame, clamp to the half extents, distance to the clamped point.
        """
        rel = agent_pos[:, None, :] - box_pos[None, :, :]  # (A, O, 2)
        c, s = jnp.cos(box_yaw), jnp.sin(box_yaw)  # (O,)
        local = jnp.stack(
            [c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]],
            axis=-1,
        )  # (A, O, 2)
        clamped = jnp.clip(local, -self._box_half[None, :, None],
                           self._box_half[None, :, None])
        dist = jnp.linalg.norm(local - clamped, axis=-1)  # (A, O)
        return dist <= _AGENT_RADIUS + _TOUCH_EPS

    def _model_for(self, data) -> mjx.Model:
        """Per-step model with the coupling mechanic applied.

        A box whose coupling requirement is met (enough agents touching) gets
        the light mass; otherwise the heavy base mass. Inertia and the
        Box2D-style damping (coeff * mass / inertia) scale along with it.
        """
        agent_pos = self._agent_pos(data)
        box_pos, box_yaw = self._box_pose(data)
        n_touch = self._touch_matrix(agent_pos, box_pos, box_yaw).sum(axis=0)
        met = n_touch >= self._coupling  # (O,)

        mass = jnp.where(met, self._light_mass, self._heavy_mass)  # (O,)
        scale = mass / self._heavy_mass
        inertia = self._heavy_inertia * scale[:, None]  # (O, 3)

        body_mass = self._model.body_mass.at[self._box_body_ids].set(mass)
        body_inertia = self._model.body_inertia.at[self._box_body_ids].set(inertia)
        dof_damping = self._model.dof_damping.at[self._box_dof_lin].set(
            jnp.repeat(_BOX_LIN_DAMPING * mass, 2)
        )
        dof_damping = dof_damping.at[self._box_dof_ang].set(
            _BOX_ANG_DAMPING * inertia[:, 2]
        )
        return self._model.replace(
            body_mass=body_mass, body_inertia=body_inertia, dof_damping=dof_damping
        )

    # ------------------------------------------------------------------ obs

    def _density_sensors(self, agent_pos, box_pos) -> jnp.ndarray:
        """(A, 16) sector centroid-proximity sensors, cols 0-7 agents, 8-15 boxes."""
        R = self.sector_sensor_radius
        shift = jnp.radians(22.5)
        step = 2 * jnp.pi / _N_SECTORS

        def sector_block(rel, in_range):
            # rel (A, N, 2), in_range (A, N) -> (A, 8)
            ang = (jnp.arctan2(rel[..., 1], rel[..., 0]) - shift) % (2 * jnp.pi)
            sect = (ang // step).astype(jnp.int32) % _N_SECTORS
            onehot = jax.nn.one_hot(sect, _N_SECTORS) * in_range[..., None]  # (A,N,8)
            count = onehot.sum(axis=1)  # (A, 8)
            sum_rel = jnp.einsum("ans,anc->asc", onehot, rel)  # (A, 8, 2)
            centroid_dist = jnp.linalg.norm(
                sum_rel / jnp.maximum(count, 1.0)[..., None], axis=-1
            )
            return jnp.where(count > 0, 1.0 - centroid_dist / R, 0.0)

        rel_aa = agent_pos[None, :, :] - agent_pos[:, None, :]  # (A, A, 2)
        dist_aa = jnp.linalg.norm(rel_aa, axis=-1)
        agents_block = sector_block(rel_aa, (dist_aa > 0) & (dist_aa < R))

        rel_ao = box_pos[None, :, :] - agent_pos[:, None, :]  # (A, O, 2)
        dist_ao = jnp.linalg.norm(rel_ao, axis=-1)
        objects_block = sector_block(rel_ao, dist_ao < R)

        return jnp.concatenate([agents_block, objects_block], axis=1)

    def _lidar(self, data, agent_pos) -> jnp.ndarray:
        """(A, R) normalized range scan via mjx raycasts.

        mjx.ray's bodyexclude is static (numpy-side), so instead of excluding
        the caster we start each ray just outside its own sphere surface and
        add the offset back — identical ranges, one fully vmapped call.
        """
        origins3 = jnp.pad(agent_pos, ((0, 0), (0, 1)))  # (A, 3), z = 0
        start = (
            origins3[:, None, :]
            + self._lidar_dirs[None, :, :] * (_AGENT_RADIUS + _LIDAR_EPS)
        ).reshape(-1, 3)
        vecs = jnp.broadcast_to(
            self._lidar_dirs[None], (self.n_agents, N_LIDAR_RAYS, 3)
        ).reshape(-1, 3)
        dist, _ = jax.vmap(lambda p, v: mjx.ray(self._model, data, p, v))(start, vecs)
        total = _AGENT_RADIUS + _LIDAR_EPS + dist
        frac = jnp.where(dist < 0, 1.0, jnp.clip(total / self.lidar_range, 0.0, 1.0))
        return frac.reshape(self.n_agents, N_LIDAR_RAYS)

    def _contact_forces(self, data) -> jnp.ndarray:
        """(A,) summed contact normal force of each agent against any box.

        With pyramidal friction cones and condim 3, each contact owns 4 efc
        facet rows starting at efc_address, and the normal force is their
        plain sum (mju_decodePyramid) — the counterpart of Box2D's PostSolve
        normal-impulse sum / dt.
        """
        impl = _impl(data)
        contact = impl.contact
        g1, g2 = contact.geom[:, 0], contact.geom[:, 1]
        a_idx = jnp.maximum(self._agent_of_geom[g1], self._agent_of_geom[g2])
        is_pair = (a_idx >= 0) & (
            jnp.maximum(self._object_of_geom[g1], self._object_of_geom[g2]) >= 0
        )
        addr = jnp.maximum(contact.efc_address, 0)[:, None] + jnp.arange(4)[None, :]
        normal = impl.efc_force[addr].sum(axis=1)
        normal = jnp.where(is_pair & (contact.efc_address >= 0), normal, 0.0)
        seg = jnp.where(is_pair, a_idx, self.n_agents)
        return jax.ops.segment_sum(normal, seg, num_segments=self.n_agents + 1)[
            : self.n_agents
        ]

    def _get_obs(self, data) -> jnp.ndarray:
        agent_pos = self._agent_pos(data)
        box_pos, box_yaw = self._box_pose(data)

        velocity = self._agent_vel(data) / self.velocity_norm  # (A, 2)
        density = self._density_sensors(agent_pos, box_pos)  # (A, 16)
        touch = self._touch_matrix(agent_pos, box_pos, box_yaw)  # (A, O)
        is_touching = touch.any(axis=1).astype(jnp.float32)[:, None]  # (A, 1)

        pair_dist = jnp.linalg.norm(
            agent_pos[:, None, :] - agent_pos[None, :, :], axis=-1
        )
        neighbor_fraction = (
            (pair_dist <= self.neighbor_detection_range).mean(axis=1)
        )[:, None]

        contact_force = (self._contact_forces(data) / self.force_multiplier)[:, None]

        rel = box_pos[None, :, :] - agent_pos[:, None, :]  # (A, O, 2)
        nearest = jnp.argmin(jnp.linalg.norm(rel, axis=-1), axis=1)  # (A,)
        nearest_box_vec = (
            rel[jnp.arange(self.n_agents), nearest] / self.world_width
        )  # (A, 2)

        goal_distance = (
            (self.target_y - agent_pos[:, 1]) / self.world_height
        )[:, None]

        lidar = self._lidar(data, agent_pos)  # (A, 16)

        return jnp.concatenate(
            [
                velocity,
                density,
                is_touching,
                neighbor_fraction,
                contact_force,
                nearest_box_vec,
                goal_distance,
                lidar,
            ],
            axis=1,
        ).astype(jnp.float32)

    # ------------------------------------------------------------------ API

    def reset(self, key: jax.Array) -> tuple[jnp.ndarray, EnvState]:
        k_cells, k_jitter, k_slots, k_boxy = jax.random.split(key, 4)

        # agents: shuffled grid cells + jitter (min separation preserved)
        cells = jax.random.permutation(k_cells, self._agent_spawn_grid.shape[0])[
            : self.n_agents
        ]
        jitter = jax.random.uniform(
            k_jitter,
            (self.n_agents, 2),
            minval=-self._agent_spawn_jitter,
            maxval=self._agent_spawn_jitter,
        )
        agent_pos = self._agent_spawn_grid[cells] + jitter  # (A, 2)

        # boxes: shuffled x-slots, uniform y inside the central band
        slots = jax.random.permutation(k_slots, self._box_spawn_slots.shape[0])[
            : self.n_objects
        ]
        box_x = self._box_spawn_slots[slots]
        band_half = self.world_height * 0.3 / 2
        box_y = jax.random.uniform(
            k_boxy,
            (self.n_objects,),
            minval=self.world_center_y - band_half + self._box_half,
            maxval=self.world_center_y + band_half - self._box_half,
        )

        qpos = jnp.zeros(self._mj_model.nq)
        qpos = qpos.at[self._agent_qadr].set(agent_pos)
        qpos = qpos.at[self._box_qadr[:, 0]].set(box_x)
        qpos = qpos.at[self._box_qadr[:, 1]].set(box_y)

        data = mjx.make_data(self._model).replace(qpos=qpos)
        data = mjx.forward(self._model, data)

        state = EnvState(
            data=data,
            t=jnp.zeros((), dtype=jnp.int32),
            prev_box_goal_dist=self.target_y - box_y,
            delivered=jnp.zeros(self.n_objects, dtype=bool),
        )
        return self._get_obs(data), state

    def step(self, state: EnvState, actions: jnp.ndarray):
        """actions: (n_agents, 2) in [-1, 1]. Returns
        (obs, state, reward, terminated, truncated, info)."""
        ctrl = jnp.clip(actions, -1.0, 1.0).reshape(-1)

        # mass update from pre-step positions, then physics (Box2D ordering)
        model_t = self._model_for(state.data)
        data = state.data.replace(ctrl=ctrl)
        data = mjx.step(model_t, data)

        agent_pos = self._agent_pos(data)
        box_pos, box_yaw = self._box_pose(data)

        # boundary termination: any agent touching a wall plane
        lo = self.boundary_thickness + _AGENT_RADIUS + _WALL_EPS
        boundary_hit = jnp.any(
            (agent_pos < lo) | (agent_pos > self.world_width - lo)
        )

        # reward: shaping toward the band + one-time delivery bonus
        dist = self.target_y - box_pos[:, 1]  # (O,) signed, matches Box2D
        shaping = jnp.sum(
            (state.prev_box_goal_dist - dist) * (~state.delivered)
        )
        in_band = (jnp.abs(box_pos[:, 0] - self.target_x) <= self.target_half_w) & (
            jnp.abs(box_pos[:, 1] - self.target_y) <= self.target_half_h
        )
        newly_delivered = in_band & ~state.delivered
        completion = 100.0 * newly_delivered.sum()
        task_reward = completion + jnp.where(
            self.reward_mode == "dense", shaping, 0.0
        )

        # Box2D skips reward/delivery bookkeeping on a boundary hit
        reward = jnp.where(boundary_hit, 0.0, task_reward)
        delivered = jnp.where(boundary_hit, state.delivered,
                              state.delivered | newly_delivered)

        terminated = boundary_hit | jnp.all(delivered)
        t = state.t + 1
        truncated = t >= self.max_steps

        obs = self._get_obs(data)
        new_state = EnvState(
            data=data, t=t, prev_box_goal_dist=dist, delivered=delivered
        )

        touch = self._touch_matrix(agent_pos, box_pos, box_yaw)
        pair_dist = jnp.linalg.norm(
            agent_pos[:, None, :] - agent_pos[None, :, :], axis=-1
        )
        info = {
            "task_reward": reward,
            "adjacency": (pair_dist <= self.comm_radius).astype(jnp.float32),
            # (O, A) 0/1 matrix — JAX-friendly stand-in for Box2D's ragged
            # agents_2_objects lists (object_contact_hyperedges can consume it)
            "agents_2_objects": touch.T.astype(jnp.float32),
            "agent_positions": agent_pos,
            "box_positions": box_pos,
            "delivered": delivered,
        }
        return obs, new_state, reward, terminated, truncated, info


def scripted_push_action(env: MultiBoxPushMJX, state: EnvState, box_idx: int = 0):
    """Hand-written cooperative controller: everyone converges on a staging
    point just below ``box_idx`` and then pushes straight up. Delivers the box
    through the coupling mechanic — used by the module demos and the renderer
    demo as a non-random rollout.
    """
    agent_pos = env._agent_pos(state.data)
    box = state.data.qpos[env._box_qadr[box_idx, :2]]
    stage_pt = box + jnp.array([0.0, -(env.box_half_extents[box_idx] + 0.6)])
    to_stage = stage_pt - agent_pos
    close = jnp.linalg.norm(to_stage, axis=1, keepdims=True) < 0.7
    approach = to_stage / (jnp.linalg.norm(to_stage, axis=1, keepdims=True) + 1e-6)
    push = jnp.broadcast_to(jnp.array([0.0, 1.0]), approach.shape)
    return jnp.where(close, push, approach)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=9)
    parser.add_argument("--n-objects", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=32, help="vmap batch size")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    env = MultiBoxPushMJX(n_agents=args.n_agents, n_objects=args.n_objects)
    print(f"world {env.world_width}x{env.world_height}, "
          f"coupling {list(env.objects_push_coupling_list)}, "
          f"box half-extents {list(env.box_half_extents)}")

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    key = jax.random.PRNGKey(0)
    obs, state = reset(key)
    assert obs.shape == (args.n_agents, OBS_DIM), obs.shape
    print(f"obs shape OK: {obs.shape}")

    # --- scripted sanity rollout: everyone pushes box 0 into the band ---
    total, done_at = 0.0, None
    t0 = time.time()
    for i in range(1024):
        obs, state, r, term, trunc, info = step(state, scripted_push_action(env, state))
        total += float(r)
        if bool(term) or bool(trunc):
            done_at = i + 1
            break
    print(f"scripted rollout: return {total:.1f}, "
          f"delivered {np.asarray(info['delivered'])}, "
          f"ended at step {done_at} ({time.time() - t0:.1f}s incl. compile)")

    # --- vmapped random-action throughput ---
    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(1), args.n_envs)
    obs, vstate = v_reset(keys)

    acts = jax.random.uniform(
        jax.random.PRNGKey(2),
        (args.steps, args.n_envs, args.n_agents, 2),
        minval=-1,
        maxval=1,
    )
    out = v_step(vstate, acts[0])  # compile
    jax.block_until_ready(out[0])
    t0 = time.time()
    vstate_i = out[1]
    for i in range(1, args.steps):
        o, vstate_i, r, te, tr, _ = v_step(vstate_i, acts[i])
    jax.block_until_ready(o)
    dt = time.time() - t0
    sps = (args.steps - 1) * args.n_envs / dt
    print(f"vmapped throughput: {sps:,.0f} env-steps/s "
          f"({args.n_envs} envs, {dt:.2f}s for {args.steps - 1} steps)")
