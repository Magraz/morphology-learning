"""MuJoCo-MJX port of the Box2D ``MultiBoxPushEnv`` (multi_box_push.py).

2D by construction: bodies own only planar DOFs (agents slide-x/y; boxes
slide-x/y + hinge-yaw), gravity is zero, walls are four inward-facing planes —
with no z DOF anywhere, MJX cannot compute out-of-plane motion at all.

Functional JAX API (gymnax-style), fully ``jit``/``vmap``-able; no auto-reset
(on ``terminated | truncated`` the caller resets)::

    env = MultiBoxPushMJX(n_agents=9, n_objects=3)
    obs, state = jax.jit(env.reset)(key)                       # obs (A, 40)
    obs, state, reward, terminated, truncated, info = jax.jit(env.step)(state, actions)

Parity with the Box2D env:

- obs: the shared 40-dim ``OBS_DIM`` layout from ``mjx_suite/observation.py``
  (``MJXObservationBuilder``); this env supplies only the qpos layout
  (``_agent_pos``/``_agent_vel``/``_box_pose``) and its goal band.
- reward: shaping toward the top target band + one-time +100 per delivered box
  (``"sparse"`` drops the shaping); terminate on all-delivered or a wall touch.
- coupling: a box keeps its heavy base mass until ``coupling`` agents touch it,
  then drops to the light coupled mass — a per-step override of ``body_mass`` /
  ``body_inertia`` / ``dof_damping`` on the mjx.Model pytree (jit-safe: the
  model is a step argument). ``_coupling_met`` owns the "enough agents working
  together" predicate, shared by the mass override and the drift below so the
  two cannot diverge.
- constants: dt=1/60, agent mass 1 / radius 0.4 / damping 10, box 2D density 20
  (0.05*coupling when coupled), box lin/ang damping 5/8. Box2D body damping is
  emulated as joint damping = coeff*mass (inertia for the hinge) — same steady
  state ``v_terminal = F / (m*d)``.

Box drift ("decay"), off unless ``box_drift_speed > 0``: every *uncoupled* box
sinks toward the bottom wall, so progress decays on any box the team is not
working on. This breaks the sequential swarm-one-box-at-a-time strategy that
makes the dense task easy — such a schedule arithmetically cannot reach its last
box, forcing the agents into simultaneous coalitions.

- A generalized force on each box's world-y slide DOF (``qfrc_applied``). That
  axis is world-fixed regardless of box yaw: mjx rotates a joint axis only by
  the quat accumulated from *preceding* joints, and the order is slide-x,
  slide-y, hinge-yaw.
- ``F = -box_drift_speed * _BOX_LIN_DAMPING * mass`` against
  ``dof_damping = _BOX_LIN_DAMPING * mass`` puts the terminal speed at exactly
  ``box_drift_speed`` with ``tau = 1/k = 0.2 s`` (12 steps) — both
  mass-independent, and negligible against a 1024-step episode.
- Gate: ``~met & ~delivered & (box_y > box_drift_floor)``. ``met`` is masked by
  ``active``, so the drift joins every difference-reward counterfactual for
  free: drop a *pivotal* agent and its box starts sinking in that branch.
- The floor keeps a sunk box **recoverable**: pushing a box up means getting
  under it, and the default ``5*boundary_thickness + 2*box_half_extent`` is the
  smallest clearance that keeps that geometry feasible. Insurance, not a cap on
  the pressure: a passive box needs ~20 s to reach it versus a 17 s episode.
- **Wall contact is inert in both drift arms** (``boundary_ends_episode`` is
  True only for the baseline). Drift makes shaping negative while any box is
  uncoupled, so an episode-ending wall hit is an *escape* from the bleeding —
  and a unilateral one, since ``boundary_hit`` is ``any()`` over agents, so one
  of 16 ends it for the team from ~6 steps out of spawn.

  The earlier fix reported the hit as ``truncated`` so the trainers' bootstrap
  would add ``gamma * V(s_next)`` and price the escape at the cost of
  continuing. That is not enough, and the arms trained under it learned to
  crash on purpose. Two reasons. (a) The stored return for crashing is
  ``0 + gamma*V̂(s_wall)`` against ``r_t + gamma*V̂(s_t+1)`` for continuing, so
  crashing wins by ``-r_t`` plus the critic's own error — with ``V̂ ~ 0`` early
  that is exactly the per-step drift bleed. (b) It is self-sealing: once the
  policy crashes at step k, no data past k is ever collected, so ``V̂`` at the
  bootstrapped states is trained only against other bootstrap targets, which is
  self-consistent with *any* value. The critic never learns that drifting
  states are worth ``< 0``, so the bias never corrects.

  So the escape is removed rather than priced. The walls are real inward-facing
  planes and agents cannot leave regardless; boundary *termination* was Box2D
  parity. With it gone, "end the episode" is not in the action space and the
  whole failure mode is unreachable. Both arms share this, so the ablation
  ladder still isolates one change per step: baseline (wall ends it) ->
  ``trunc`` (wall inert) -> ``drift`` (wall inert + decay).
- ``variant=None`` is a strict no-op: every branch is a Python-level ``if``, so
  the baseline graph and numbers are unchanged (verified bit-identical).
- Config surface is the ``variant`` preset (``env.variant`` in ``conf/env/``),
  and it is the *whole* surface — every knob is a constant of the preset, so an
  arm is fully identified by its name: ``"drift"`` -> decay + inert walls,
  ``"trunc"`` -> inert walls only, ``None`` -> stock Box2D-parity behavior.

Differences from Box2D: spawns use shuffled jittered grids rather than rejection
sampling (jit needs static shapes), same regions and min separations; drift and
the inert-wall variants have no counterpart; shaping is live from step 1 (Box2D
pays 0 on its first step); box sizes are fixed per instance (already true there).

Demo / sanity check (no display needed):
    uv run python -m environments.mjx_suite.multi_box_push_mjx
"""

import dataclasses
import math
from enum import StrEnum
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from environments.box2d_suite.utils import (
    COLORS_LIST,
    ObjectTargetArea,
    resolve_coupling,
)
from environments.mjx_suite.observation import (
    OBS_DIM,
    MJXObservationBuilder,
    geom_index_maps,
)

_AGENT_RADIUS = 0.4
_AGENT_MASS = 1.0  # Box2D default-mass fallback for zero-density fixtures
_AGENT_DAMPING = 10.0
_BOX_LIN_DAMPING = 5.0
_BOX_ANG_DAMPING = 8.0
_BOX_BASE_DENSITY_2D = 20.0  # Box2D kg/m^2
_COUPLED_DENSITY_PER_AGENT = 0.05
_BOX_HALF_HEIGHT = 0.4  # z half-extent; cosmetic (no z DOF), keeps contacts planar
_FORCE_MULTIPLIER = 100.0
_TIME_STEP = 1.0 / 60.0
_WALL_EPS = 0.01  # boundary-contact slack, ~Box2D contact slop
_DEFAULT_DRIFT_SPEED = 0.5


class VARIANTS(StrEnum):
    """Preset bundles of the drift / termination-semantics knobs.

    A `StrEnum` (the repo idiom, cf. `EnvironmentEnum`) so `VARIANTS("drift")`
    parses a config string straight into a member. The previous
    `{"trunc": 1, "drift": 2}` side table handed back a raw `int`, and
    `2 == VARIANTS.DRIFT` is silently `False` for a plain `Enum` — so every
    branch below evaluated to "off" and the drift never applied a force.
    """

    TRUNC = "trunc"  # boundary hits are inert (never end the episode); no drift
    DRIFT = "drift"  # the above + uncoupled boxes decay toward the bottom wall


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
        variant: str = None,
    ):
        if reward_mode not in ("dense", "sparse", "difference_rewards"):
            raise ValueError(
                f"reward_mode must be dense|sparse|difference_rewards, got {reward_mode}"
            )
        if variant is not None:
            try:
                variant = VARIANTS(variant)
            except ValueError:
                raise ValueError(
                    f"unknown variant: {variant!r}; expected one of "
                    f"{[v.value for v in VARIANTS]} or None"
                ) from None
        self.variant = variant
        self.n_agents = n_agents
        self.n_objects = n_objects
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        # `difference_rewards` shapes the *team* reward exactly like "dense" and
        # then decomposes it per agent; a sparse base would leave D_i zero on every
        # step except a delivery, which is too thin to learn from.
        self._dense = reward_mode in ("dense", "difference_rewards")
        self._difference = reward_mode == "difference_rewards"

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
        self.comm_radius = self.world_width / 3.0
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
        coupling = resolve_coupling(coupling_def, n_agents, n_objects)
        self.objects_push_coupling_list = coupling
        self.box_half_extents = np.maximum(1.5, coupling * _AGENT_RADIUS)

        heavy_mass = _BOX_BASE_DENSITY_2D * (2 * self.box_half_extents) ** 2
        light_mass = (
            _COUPLED_DENSITY_PER_AGENT * coupling * (2 * self.box_half_extents) ** 2
        )

        # --- box drift ("decay"); see the module docstring ---
        # `variant` is the whole config surface: every knob below is a constant
        # of the preset, so an arm is fully identified by its name.
        box_drift_speed = _DEFAULT_DRIFT_SPEED if variant is VARIANTS.DRIFT else 0.0
        self.box_drift_speed = float(box_drift_speed)
        self._drift_on = self.box_drift_speed > 0.0
        # F = -v_d * k * m has fixed point v = F / (k*m) = -v_d, so the terminal
        # speed is mass-independent. The gate is `~met`, so the force only ever
        # acts on a *heavy* box — no need for the light-mass variant.
        self._drift_force_heavy = jnp.asarray(
            -self.box_drift_speed * _BOX_LIN_DAMPING * heavy_mass, dtype=jnp.float32
        )  # (O,)
        # Floor: an agent must get *under* a box to push it up, but an agent
        # within `boundary_thickness + _AGENT_RADIUS` of the wall ends the
        # episode — so a box resting on the bottom wall is unrecoverable. One
        # box-width of clearance is the smallest floor that keeps it feasible
        # (holds even at 45 deg box yaw, where the box reaches sqrt(2)*half down).
        # Keyed off the half-extent rather than `coupling * _AGENT_RADIUS`
        # directly — the two agree except where the 1.5 minimum box size binds,
        # and there the raw coupling form would under-provide clearance.
        drift_floor = 5 * self.boundary_thickness + 2 * self.box_half_extents

        self.box_drift_floor = np.array(drift_floor)  # (O,) numpy, for logging/tests
        self._drift_floor = jnp.asarray(drift_floor, dtype=jnp.float32)
        # Drift makes shaping negative while any box is uncoupled, which turns a
        # boundary hit into an *escape* from the bleeding. Both drift arms
        # therefore make wall contact inert: it neither ends the episode nor
        # alters the bookkeeping, so "quit early" leaves the action space
        # entirely. See the module docstring for why pricing the escape (the
        # previous `boundary_truncates` approach) cannot work.
        self.boundary_ends_episode = variant is None

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
        self._box_dof_y = jnp.asarray(box_dadr[:, 1])  # (O,) world-y slide, drift axis

        # heavy/light overrides for the coupling mechanic
        self._coupling = jnp.asarray(coupling, dtype=jnp.int32)
        self._heavy_mass = jnp.asarray(heavy_mass, dtype=jnp.float32)
        self._light_mass = jnp.asarray(light_mass, dtype=jnp.float32)
        self._heavy_inertia = self._model.body_inertia[self._box_body_ids]  # (O, 3)
        self._box_half = jnp.asarray(self.box_half_extents, dtype=jnp.float32)

        agent_of_geom, object_of_geom = geom_index_maps(m, n_agents, n_objects)
        self.obs_builder = MJXObservationBuilder(
            self._model,
            n_agents=n_agents,
            n_objects=n_objects,
            world_width=self.world_width,
            world_height=self.world_height,
            velocity_norm=self.velocity_norm,
            neighbor_detection_range=self.neighbor_detection_range,
            agent_radius=_AGENT_RADIUS,
            force_multiplier=self.force_multiplier,
            sector_sensor_radius=self.sector_sensor_radius,
            lidar_range=self.lidar_range,
            agent_of_geom=agent_of_geom,
            object_of_geom=object_of_geom,
        )

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
        gap = min(
            xs[1] - xs[0] if n_cols > 1 else np.inf,
            ys[1] - ys[0] if n_rows > 1 else np.inf,
        )
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
        """(A, O) bool — agent within radius + eps of a (rotated) box surface."""
        return self.obs_builder.touch_matrix(
            agent_pos, box_pos, box_yaw, self._box_half
        )

    def _coupling_met(self, data, active: jnp.ndarray | None = None) -> jnp.ndarray:
        """(O,) bool — is each box's coupling requirement currently satisfied?

        `active` is an optional (A,) bool mask (traced) of agents that count as
        *cooperating*. Masked-out agents are excluded from the touch count, so a
        counterfactual agent does not prop up a box it is no longer pushing. The
        coupling requirement abstracts "enough agents working together", and an
        agent applying zero force is not working — see `_difference_rewards`.

        Shared by the mass override (`_model_for`) and the box drift
        (`_drift_force`) so the two cannot disagree about who is cooperating,
        and so the drift is counterfactual under `active` for free.
        """
        agent_pos = self._agent_pos(data)
        box_pos, box_yaw = self._box_pose(data)
        touch = self._touch_matrix(agent_pos, box_pos, box_yaw)  # (A, O)
        if active is not None:
            touch = touch & active[:, None]
        return touch.sum(axis=0) >= self._coupling  # (O,)

    def _drift_force(self, data, met, delivered) -> jnp.ndarray:
        """(O,) <= 0 generalized force on each box's world-y slide DOF.

        Sized for a mass-independent terminal speed (see the module docstring):
        `F = -box_drift_speed * _BOX_LIN_DAMPING * mass` against
        `dof_damping = _BOX_LIN_DAMPING * mass`. Off where the coupling is met,
        where the box is delivered, and below `box_drift_floor` — a box parked on
        the bottom wall could never be pushed out again.

        A force gate rather than an mjx joint limit on purpose: a limit is static
        model structure (an extra constraint row on every step of *every* arm, so
        the drift-off graph would change), MJX limits are soft so the box would
        sink through and could be crushed against it, and it would also obstruct
        legitimate downward pushing. The gate cannot chatter either: below the
        floor there is no restoring force, so the implicit damping bleeds off the
        residual velocity and the box settles ~`v_d * tau` units low and stays.
        """
        box_y = data.qpos[self._box_qadr[:, 1]]  # (O,)
        drifting = ~met & ~delivered & (box_y > self._drift_floor)
        return jnp.where(drifting, self._drift_force_heavy, 0.0)

    def _model_for(self, met: jnp.ndarray) -> mjx.Model:
        """Per-step model with the coupling mechanic applied.

        A box whose coupling requirement is met (enough agents touching, per
        `_coupling_met`) gets the light mass; otherwise the heavy base mass.
        Inertia and the Box2D-style damping (coeff * mass / inertia) scale along
        with it.
        """
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

    def _get_obs(self, data, delivered=None) -> jnp.ndarray:
        """(A, OBS_DIM) — the shared Box2D-suite layout, built by obs_builder.

        ``delivered`` (O,) bool excludes already-delivered boxes from
        ``nearest_box_vec`` so agents stop being drawn to a box parked in the
        goal band; ``None`` senses every box (the pre-delivery / no-mask case).
        """
        box_pos, box_yaw = self._box_pose(data)
        return self.obs_builder.build(
            data,
            agent_pos=self._agent_pos(data),
            agent_vel=self._agent_vel(data),
            box_pos=box_pos,
            box_yaw=box_yaw,
            box_half=self._box_half,
            goal_coord=self.target_y,  # band spans the top wall: goal axis is y
            goal_axis="y",
            delivered=delivered,
        )

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
        return self._get_obs(data, state.delivered), state

    def _advance(
        self, state: EnvState, actions: jnp.ndarray, active: jnp.ndarray | None = None
    ):
        """One physics step. `active` (A,) bool zeroes masked agents' force *and*
        drops them from the coupling count (see `_model_for`)."""
        ctrl = jnp.clip(actions, -1.0, 1.0)
        if active is not None:
            ctrl = ctrl * active[:, None]
        # mass update from pre-step positions, then physics (Box2D ordering)
        met = self._coupling_met(state.data, active)
        model_t = self._model_for(met)
        data = state.data.replace(ctrl=ctrl.reshape(-1))
        if self._drift_on:  # Python-level: drift-off arms compile unchanged
            # Rebuilt from zeros rather than accumulated, which keeps
            # qfrc_applied a pure function of (qpos, met, delivered). That is
            # what lets SyncMacroMJX.state_from_snapshot stay exact: it restores
            # only qpos/qvel, and the next _advance recomputes this force.
            data = data.replace(
                qfrc_applied=jnp.zeros_like(state.data.qfrc_applied)
                .at[self._box_dof_y]
                .set(self._drift_force(state.data, met, state.delivered))
            )
        return mjx.step(model_t, data)

    def _task_reward(self, state: EnvState, data):
        """Team reward for a post-step `data` against the pre-step `state`.

        Pure in (state, data), so counterfactual branches reuse it directly —
        no recursion through `step`. Returns
        (reward, newly_delivered, boundary_hit, dist).
        """
        agent_pos = self._agent_pos(data)
        box_pos, _ = self._box_pose(data)

        # boundary termination: any agent touching a wall plane
        lo = self.boundary_thickness + _AGENT_RADIUS + _WALL_EPS
        top_right_boundary_hit = jnp.hstack(
            [
                (agent_pos[:, 0] > self.world_width - lo)[:, jnp.newaxis],
                (agent_pos[:, 1] > self.world_height - lo)[:, jnp.newaxis],
            ]
        )
        boundary_hit = jnp.any((agent_pos < lo) | top_right_boundary_hit)

        # reward: shaping toward the band + one-time delivery bonus
        dist = self.target_y - box_pos[:, 1]  # (O,) signed, matches Box2D
        shaping = jnp.sum((state.prev_box_goal_dist - dist) * (~state.delivered))
        in_band = (jnp.abs(box_pos[:, 0] - self.target_x) <= self.target_half_w) & (
            jnp.abs(box_pos[:, 1] - self.target_y) <= self.target_half_h
        )
        newly_delivered = in_band & ~state.delivered
        completion = 100.0 * newly_delivered.sum()
        task_reward = completion + (shaping if self._dense else 0.0)

        # Box2D skips reward/delivery bookkeeping on a boundary hit — but only
        # because the hit *is* the episode's terminal failure there. Where wall
        # contact is inert (the drift arms) the step physically happened like any
        # other and must pay its real reward: zeroing it would hand back exactly
        # the negative drift shaping the agent would otherwise eat, i.e. a
        # standing bonus for touching a wall. Python-level `if`, so the baseline
        # graph and numbers are untouched.
        if self.boundary_ends_episode:
            task_reward = jnp.where(boundary_hit, 0.0, task_reward)
        return task_reward, newly_delivered, boundary_hit, dist

    def _difference_rewards(
        self, state: EnvState, actions: jnp.ndarray, g_factual: jnp.ndarray
    ) -> jnp.ndarray:
        """(A,) exact single-step difference rewards `D_i = G - G_-i`.

        Fork the pre-step state once per agent and re-run *the same* step with
        agent i contributing nothing: zero force **and** dropped from the coupling
        count. Only agent i's participation differs, so D_i is exact, not an
        estimate — this is what the pure functional env buys us.

        The coupling exclusion is what gives the signal its structure. With force
        zeroed alone, a limp agent would still be counted as touching, the box
        would stay light, and D_i would collapse to agent i's marginal force
        share (roughly additive, sum_i D_i ~ G). Dropping it from the count
        instead asks "was agent i *necessary*": if a box needs 3 touchers and
        exactly 3 are on it, removing any one makes the box heavy and it stops
        moving, so all three score D_i ~ G; if 5 are on it, the coupling still
        holds without agent i and it correctly scores ~0. Credit is therefore
        non-additive by design (sum_i D_i can far exceed G).

        Costs A extra `mjx.step` calls per env step, vmapped over the agent axis.
        Single-step by necessity: a multi-step counterfactual would need to know
        what agents do *next*, which is the policy's business, not the env's.
        """
        agent_ids = jnp.arange(self.n_agents)

        def counterfactual(i):
            data = self._advance(state, actions, active=agent_ids != i)
            reward, _, _, _ = self._task_reward(state, data)
            return reward

        return g_factual - jax.vmap(counterfactual)(agent_ids)

    def step(
        self, state: EnvState, actions: jnp.ndarray, active: jnp.ndarray | None = None
    ):
        """actions: (n_agents, 2) in [-1, 1]. Returns
        (obs, state, reward, terminated, truncated, info).

        `reward` is the scalar team reward, except under
        `reward_mode="difference_rewards"` where it is (n_agents,) per-agent
        difference rewards. `info["task_reward"]` always carries the team scalar.

        `active` is an optional (A,) bool mask of *cooperating* agents (traced):
        masked-out agents contribute zero force and are dropped from the coupling
        count for this step (see `_advance`/`_model_for`), so a caller can roll a
        counterfactual "agent i absent" trajectory forward — this is what the
        windowed macro difference reward (`SyncMacroMJX`) forks over. `None` (the
        default, == an all-True mask) is the ordinary full-participation step.
        """
        data = self._advance(state, actions, active)
        reward, newly_delivered, boundary_hit, dist = self._task_reward(state, data)

        if self.boundary_ends_episode:
            delivered = jnp.where(
                boundary_hit, state.delivered, state.delivered | newly_delivered
            )
        else:
            delivered = state.delivered | newly_delivered
        t = state.t + 1
        all_delivered = jnp.all(delivered)
        if self.boundary_ends_episode:
            terminated = boundary_hit | all_delivered
            truncated = t >= self.max_steps
        else:
            # Wall contact is inert. The walls are real inward-facing planes, so
            # agents are already physically confined — ending the episode on
            # contact was Box2D parity, not a physical necessity, and under drift
            # it was the cheapest way out of a negative reward stream.
            terminated = all_delivered
            truncated = t >= self.max_steps

        obs = self._get_obs(data, delivered)
        new_state = EnvState(
            data=data, t=t, prev_box_goal_dist=dist, delivered=delivered
        )

        agent_pos = self._agent_pos(data)
        box_pos, box_yaw = self._box_pose(data)
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

        if self._difference:
            reward = self._difference_rewards(state, actions, reward)
        return obs, new_state, reward, terminated, truncated, info


def scripted_push_action(env: MultiBoxPushMJX, state: EnvState, box_idx=0):
    """Hand-written cooperative controller: each agent converges on a staging
    point just below its assigned box and then pushes straight up. Delivers the
    box through the coupling mechanic — used by the module demos and the renderer
    demo as a non-random rollout.

    ``box_idx`` is a scalar (the whole team swarms one box) or an ``(A,)`` array
    assigning a box per agent, e.g. ``jnp.arange(A) % O`` for a balanced
    partition — the two arms of the swarm-vs-partition comparison.

    Agents sharing a box get *distinct* slots spread along its bottom face. With
    a single shared staging point they collide with each other and only ~2 ever
    reach the surface, so a coalition of exactly ``coupling`` agents could never
    satisfy the coupling requirement. Surplus agents clamp to the face edges and
    crowd in as before.
    """
    agent_pos = env._agent_pos(state.data)
    idx = jnp.broadcast_to(jnp.asarray(box_idx), (env.n_agents,))
    box = state.data.qpos[env._box_qadr[idx, :2]]  # (A, 2)
    # env._box_half, not the numpy box_half_extents: idx may be traced.
    half = env._box_half[idx]  # (A,)
    # Rank within the group sharing this box, and that group's size.
    same = idx[:, None] == idx[None, :]
    order = jnp.arange(env.n_agents)
    rank = (same & (order[None, :] < order[:, None])).sum(axis=-1)
    group = same.sum(axis=-1)
    slot = (rank - (group - 1) / 2) * (2 * _AGENT_RADIUS + 0.1)
    lateral = jnp.clip(slot, -half, half)
    stage_pt = box + jnp.stack([lateral, -(half + 0.6)], axis=-1)
    to_stage = stage_pt - agent_pos
    close = jnp.linalg.norm(to_stage, axis=1, keepdims=True) < 0.7
    approach = to_stage / (jnp.linalg.norm(to_stage, axis=1, keepdims=True) + 1e-6)
    push = jnp.broadcast_to(jnp.array([0.0, 1.0]), approach.shape)
    return jnp.where(close, push, approach)


def _pose(env: MultiBoxPushMJX, state: EnvState, agent_pos=None, box_pos=None):
    """Hand-place agents/boxes in an existing state (test helper).

    Rebuilds qpos and re-runs `mjx.forward` so contacts/positions are consistent,
    and re-bases `prev_box_goal_dist` so the first step's shaping is not a jump.
    """
    qpos = state.data.qpos
    if agent_pos is not None:
        qpos = qpos.at[env._agent_qadr].set(jnp.asarray(agent_pos))
    if box_pos is not None:
        box_pos = jnp.asarray(box_pos)
        qpos = qpos.at[env._box_qadr[:, 0]].set(box_pos[:, 0])
        qpos = qpos.at[env._box_qadr[:, 1]].set(box_pos[:, 1])
    at_rest = state.data.replace(qpos=qpos, qvel=jnp.zeros_like(state.data.qvel))
    data = mjx.forward(env._model, at_rest)
    box_y = data.qpos[env._box_qadr[:, 1]]
    return dataclasses.replace(
        state, data=data, prev_box_goal_dist=env.target_y - box_y
    )


def _face_positions(box_xy, half: float, n: int) -> np.ndarray:
    """`n` agent positions touching a box, spread over its four faces.

    Two slots per face at +-half/2 along it, offset `half + _AGENT_RADIUS - 0.05`
    along the face normal — inside the touch threshold with room to spare, and
    with the slots far enough apart not to overlap each other.
    """
    normals = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    out = []
    for slot in range(n):
        nx, ny = normals[slot % 4]
        along = (-1.0 if slot // 4 == 0 else 1.0) * half / 2
        d = half + _AGENT_RADIUS - 0.05
        out.append(
            (
                box_xy[0] + nx * d + (0.0 if nx else along),
                box_xy[1] + ny * d + (0.0 if ny else along),
            )
        )
    return np.asarray(out)


def _check_drift(n_agents: int, n_objects: int, n_envs: int):  # noqa: C901
    """Assertion suite for the box drift mechanic (see the module docstring)."""
    import time

    # The arms are named presets with no tunable knobs, so the suite checks the
    # `drift` arm exactly as training runs it.
    v_d = _DEFAULT_DRIFT_SPEED
    k = _BOX_LIN_DAMPING
    rho = 1.0 / (1.0 + k * _TIME_STEP)  # per-step velocity decay factor

    def make(**kw):
        return MultiBoxPushMJX(n_agents=n_agents, n_objects=n_objects, **kw)

    off, on = make(), make(variant="drift")
    coupling = np.asarray(off.objects_push_coupling_list)
    print(
        f"drift checks @ {n_agents}a/{n_objects}o | world {on.world_width} | "
        f"coupling {list(coupling)} | half {list(on.box_half_extents)} | "
        f"floor {list(np.round(on.box_drift_floor, 2))} | "
        f"force {list(np.round(np.asarray(on._drift_force_heavy), 1))} N"
    )

    # --- 1. drift off is inert -------------------------------------------------
    assert off._drift_on is False and off.boundary_ends_episode is True
    assert on._drift_on is True and on.boundary_ends_episode is False
    assert make(variant="trunc")._drift_on is False
    assert make(variant="trunc").boundary_ends_episode is False
    _, s_off = jax.jit(off.reset)(jax.random.PRNGKey(0))
    _, s_off, *_ = jax.jit(off.step)(s_off, jnp.zeros((n_agents, 2)))
    assert float(jnp.abs(s_off.data.qfrc_applied).max()) == 0.0
    print("  [1] drift off: no applied force, boundary still terminates   OK")

    # --- 2. terminal velocity, axis purity, mass independence, transient ------
    # Agents parked in the top corners so nothing touches the drifting boxes.
    def clear_state(env, key=0):
        _, st = jax.jit(env.reset)(jax.random.PRNGKey(key))
        top = env.world_height - 3.0
        pos = np.stack(
            [
                np.linspace(3.0, env.world_width - 3.0, env.n_agents),
                np.full(env.n_agents, top),
            ],
            axis=1,
        )
        return _pose(env, st, agent_pos=pos)

    state = clear_state(on)
    step_on = jax.jit(on.step)
    zero = jnp.zeros((n_agents, 2))
    vy_at = {}
    for i in range(120):
        _, state, r, term, trunc, _ = step_on(state, zero)
        vy_at[i + 1] = np.asarray(state.data.qvel[on._box_dof_y])
        assert not bool(term), f"unexpected termination at step {i + 1}"
    vy = vy_at[120]
    assert np.allclose(vy, -v_d, atol=1e-3), vy
    vx = np.asarray(state.data.qvel[on._box_dof_lin])[0::2]
    vyaw = np.asarray(state.data.qvel[on._box_dof_ang])
    assert np.abs(vx).max() < 1e-4 and np.abs(vyaw).max() < 1e-4, (vx, vyaw)
    frac = float(np.mean(vy_at[12] / -v_d))
    assert abs(frac - (1 - rho**12)) < 0.03, (frac, 1 - rho**12)
    print(
        f"  [2] terminal v_y={vy.mean():+.5f} (want {-v_d}); |v_x|,|v_yaw| < 1e-4; "
        f"transient at tau: {frac:.3f} vs 1-rho^12 = {1 - rho**12:.3f}   OK"
    )

    # mass independence: uneven couplings -> different box sizes/masses.
    # An explicit ascending list rather than a random draw: the point is only
    # that the boxes differ in mass, and a fixed list keeps the check
    # reproducible across `n_agents` / `n_objects`.
    uneven = [min(2 + j, n_agents) for j in range(n_objects)]
    rnd = MultiBoxPushMJX(
        n_agents=n_agents,
        n_objects=n_objects,
        coupling_def=uneven,
        variant="drift",
    )
    st_r = clear_state(rnd)
    step_r = jax.jit(rnd.step)
    for _ in range(120):
        _, st_r, *_ = step_r(st_r, zero)
    vy_r = np.asarray(st_r.data.qvel[rnd._box_dof_y])
    assert np.allclose(vy_r, -v_d, atol=1e-3), (rnd.objects_push_coupling_list, vy_r)
    print(
        f"  [2b] mass independence: couplings {list(rnd.objects_push_coupling_list)} "
        f"masses {list(np.round(np.asarray(rnd._heavy_mass), 1))} all reach "
        f"{vy_r.mean():+.5f}   OK"
    )

    # --- 3. floor: the box stops and stays stopped ----------------------------
    # The arm's own floor (no override knob), so this is the floor training sees.
    # Enough steps for the *highest* box to fall to it and settle.
    fl = on
    floor = np.asarray(fl.box_drift_floor)  # (O,)
    st_f = clear_state(fl)
    box_start = np.asarray(st_f.data.qpos[fl._box_qadr[:, 1]])
    n_settle = int((box_start - floor).max() / v_d * 60) + 150
    step_f = jax.jit(fl.step)
    for _ in range(n_settle):
        _, st_f, *_ = step_f(st_f, zero)
    box_end = np.asarray(st_f.data.qpos[fl._box_qadr[:, 1]])
    # The floor is a force *gate*, not a hard limit, so a box coasts past it by
    # its stopping distance v_d*tau = v_d/k before the damping kills the carried
    # velocity. That is real resting clearance the floor does not provide —
    # significant at the current v_d (0.6 of the 5.7 floor), so keep the tolerance
    # tied to the physics rather than a constant that silently absorbs it.
    coast = v_d / k
    assert (box_start > floor).all(), (box_start, floor)
    assert (box_end >= floor - coast - 0.1).all(), (box_end, floor, coast)
    assert np.abs(np.asarray(st_f.data.qvel[fl._box_dof_y])).max() < 1e-3
    print(
        f"  [3] floor {np.round(floor, 2)} after {n_settle} steps: "
        f"y {np.round(box_start, 2)} -> {np.round(box_end, 2)} "
        f"(coast {coast:.2f}), settled (no chatter)   OK"
    )

    # --- 4. coupling gate: met -> that box's force is off, others keep theirs --
    st_c = clear_state(on)
    box_xy = np.asarray(st_c.data.qpos[on._box_qadr[:, :2]])
    touchers = _face_positions(
        box_xy[0], float(on.box_half_extents[0]), int(coupling[0])
    )
    rest = np.stack(
        [
            np.linspace(3.0, on.world_width - 3.0, n_agents - len(touchers)),
            np.full(n_agents - len(touchers), on.world_height - 3.0),
        ],
        axis=1,
    )
    st_c = _pose(on, st_c, agent_pos=np.concatenate([touchers, rest]))
    met = on._coupling_met(st_c.data)
    force = np.asarray(on._drift_force(st_c.data, met, st_c.delivered))
    assert bool(met[0]) and not bool(met[1:].any()), np.asarray(met)
    assert force[0] == 0.0, force
    assert np.allclose(force[1:], np.asarray(on._drift_force_heavy)[1:]), force
    print(
        f"  [4] coupling gate: met {np.asarray(met)} -> "
        f"force {np.round(force, 1)}   OK"
    )

    # --- 5. reward semantics: shaping == negated box displacement -------------
    st_r0 = clear_state(on)
    y0 = np.asarray(st_r0.data.qpos[on._box_qadr[:, 1]])
    cum, st = 0.0, st_r0
    for _ in range(200):
        _, st, r, term, trunc, _ = step_on(st, zero)
        assert float(r) < 0.0, r
        cum += float(r)
    dy = float((y0 - np.asarray(st.data.qpos[on._box_qadr[:, 1]])).sum())
    assert abs(cum + dy) < 2e-3, (cum, -dy)
    assert abs(float(r) + v_d * n_objects / 60.0) < 2e-3, r
    st_o, cum_off = clear_state(off), 0.0
    step_off = jax.jit(off.step)
    for _ in range(200):
        _, st_o, r_o, *_ = step_off(st_o, zero)
        cum_off += float(r_o)
    assert abs(cum_off) < 1e-3, cum_off
    print(
        f"  [5] reward: 200 idle steps -> {cum:.4f} (== -sum box dy {-dy:.4f}); "
        f"per-step {float(r):+.5f} vs -v_d*O/60 = {-v_d * n_objects / 60:+.5f}; "
        f"drift off -> {cum_off:.1e}   OK"
    )

    # --- 6. wall contact: inert in the drift arms, terminal in the baseline ---
    # The failure this guards: under drift, an episode-ending wall hit is an
    # escape from the negative shaping, and the crash step used to be paid 0
    # (Box2D parity) — handing back exactly the bleed it escaped.
    def wall_dive(env, n=30):
        _, st = jax.jit(env.reset)(jax.random.PRNGKey(0))
        pos = np.array(st.data.qpos[env._agent_qadr])
        pos[0] = (env.world_width / 2, env.boundary_thickness + _AGENT_RADIUS + 0.4)
        st = _pose(env, st, agent_pos=pos)
        down = jnp.zeros((n_agents, 2)).at[0, 1].set(-1.0)
        f = jax.jit(env.step)
        lo = env.boundary_thickness + _AGENT_RADIUS + _WALL_EPS
        contact = []  # rewards on the steps where agent 0 is against the wall
        for _ in range(n):
            _, st, r, term, trunc, _ = f(st, down)
            if float(st.data.qpos[env._agent_qadr][0, 1]) < lo:
                contact.append(float(r))
            if bool(term) or bool(trunc):
                return bool(term), bool(trunc), contact
        return False, False, contact

    assert wall_dive(off)[:2] == (True, False), "baseline must still terminate"
    term_on, trunc_on, contact = wall_dive(on)
    assert (term_on, trunc_on) == (False, False), "wall contact must not end it"
    assert contact, "agent never reached the wall"
    assert all(r != 0.0 for r in contact), contact  # not the old zeroed reward
    assert np.mean(contact) < 0.0, contact  # it eats the drift bleed like any step
    print(
        f"  [6] wall contact: baseline -> terminated; drift arm -> episode "
        f"continues, {len(contact)} contact steps pay real reward "
        f"(mean {np.mean(contact):+.4f}, never 0)   OK"
    )

    # --- 7. recoverability: a box sitting *at* the floor is still deliverable --
    # Tested with a *minimum* coalition (the balanced partition, `coupling` agents
    # per box), which is what the floor is sized for. Piling the whole team under
    # a floored box can still shove one of them into the wall — the floor
    # guarantees the geometry, not that any crowd survives it.
    swarm = 0
    partition = jnp.arange(n_agents) % n_objects
    _, st_g = jax.jit(on.reset)(jax.random.PRNGKey(3))
    # Box 0 down at its floor with a clear column above it; its coalition staged
    # just under it; everyone else parked at the top of the arena and idle. The
    # other boxes are moved to the side walls: left where they are, a drifting
    # *heavy* box lands on the rising one and the pair sinks together at
    # (819 - 400) / 1074 ~ 0.4 u/s — correct physics, but not what this checks.
    crew = np.asarray(partition) == 0
    boxes = np.array(st_g.data.qpos[on._box_qadr[:, :2]])
    boxes[0] = (on.world_width / 2, float(on.box_drift_floor[0]))
    for j in range(1, n_objects):
        side = 2.5 + on.box_half_extents[j]
        boxes[j] = (
            side if j % 2 else on.world_width - side,
            on.world_center_y + 2.0 * j,
        )
    half0 = float(on.box_half_extents[0])
    agents = np.stack(
        [
            np.linspace(3.0, on.world_width - 3.0, n_agents),
            np.full(n_agents, on.world_height - 3.0),
        ],
        axis=1,
    )
    n_crew = int(crew.sum())
    agents[crew] = np.stack(
        [
            boxes[0, 0] + np.linspace(-half0, half0, n_crew),
            np.full(n_crew, boxes[0, 1] - half0 - 1.0),
        ],
        axis=1,
    )
    st_g = _pose(on, st_g, agent_pos=agents, box_pos=boxes)
    assert not bool(st_g.delivered.any())
    idle = jnp.asarray(crew)[:, None]
    ended_early, delivered0 = False, False
    for _ in range(on.max_steps):
        act = jnp.where(idle, scripted_push_action(on, st_g, partition), 0.0)
        _, st_g, _, term, trunc, info = step_on(st_g, act)
        delivered0 = bool(np.asarray(info["delivered"])[0])
        if delivered0:
            break
        if bool(term) or bool(trunc):
            ended_early = True
            break
    assert delivered0 and not ended_early, (delivered0, ended_early, int(st_g.t))
    print(
        f"  [7] recoverability: box starting *at* the floor "
        f"({float(on.box_drift_floor[0]):.2f}) pushed out and delivered by a "
        f"minimum coalition of {n_crew} in {int(st_g.t)} steps, no wall hit   OK"
    )

    # --- 8. efficacy: the sequential swarm loses the boxes it ignores ---------
    # Averaged over seeds: single scripted rollouts are noisy at this scale (MJX
    # is not bit-reproducible across processes and the returns are chaotic in it).
    #
    # What is asserted is what is *robust*: (i) the balanced partition still beats
    # the swarm under drift — the mechanic must not invert the preference — and
    # (ii) the swarm's ignored boxes end the episode markedly lower with drift
    # than without, which is the mechanic doing its job. The partition-over-swarm
    # *return gap* does NOT reliably widen (measured roughly flat, 16a/4o 4 seeds:
    # +318 at v_d 0 vs +284 at 0.8) — with a scripted oracle both arms pay drift
    # cost, so the interesting question is what a *learned* policy does, which
    # only a training run answers.
    def scripted_return(env, assign, seeds=(7, 23)):
        rets, boxes, undelivered_y = [], [], []
        f = jax.jit(env.step)
        for seed in seeds:
            _, st = jax.jit(env.reset)(jax.random.PRNGKey(seed))
            total = 0.0
            for _ in range(env.max_steps):
                _, st, r, term, trunc, info = f(
                    st, scripted_push_action(env, st, assign)
                )
                total += float(r)
                if bool(term) or bool(trunc):
                    break
            done = np.asarray(info["delivered"])
            rets.append(total)
            boxes.append(int(done.sum()))
            box_y = np.asarray(st.data.qpos[env._box_qadr[:, 1]])
            undelivered_y.append(box_y[~done].mean() if (~done).any() else np.nan)
        return (
            float(np.mean(rets)),
            float(np.mean(boxes)),
            float(np.nanmean(undelivered_y)),
        )

    r_off_s, d_off_s, y_off_s = scripted_return(off, swarm)
    r_off_p, d_off_p, _ = scripted_return(off, partition)
    r_on_s, d_on_s, y_on_s = scripted_return(on, swarm)
    r_on_p, d_on_p, _ = scripted_return(on, partition)
    # Reported before asserting: when this check fails it is usually diagnosing
    # the *drift speed*, not a code defect, and the numbers are what tell you so.
    print(
        f"  [8] efficacy: swarm {r_off_s:.0f} ({d_off_s:.1f} boxes) -> partition "
        f"{r_off_p:.0f} ({d_off_p:.1f}) with drift off; {r_on_s:.0f} ({d_on_s:.1f}) "
        f"-> {r_on_p:.0f} ({d_on_p:.1f}) with drift on. Swarm's ignored boxes end "
        f"at y {y_off_s:.1f} -> {y_on_s:.1f} (they decay)"
    )
    assert y_on_s < y_off_s - 2.0, (y_on_s, y_off_s)
    # The mechanic must not invert the preference it exists to create. If the
    # partition stops winning, `box_drift_speed` is too high for the coalition to
    # survive: the drift force (v_d * k * mass) is racing `coupling` agents'
    # combined thrust, so past some v_d even a perfectly scripted partition loses
    # its boxes and the "correct" strategy is punished. Compare `d_on_p` against
    # `d_off_p` — a collapse there means the task itself became infeasible.
    assert r_on_p > r_on_s, (
        f"drift INVERTED the strategy preference: partition {r_on_p:.0f} "
        f"({d_on_p:.1f} boxes) <= swarm {r_on_s:.0f} ({d_on_s:.1f}); the partition "
        f"delivers {d_off_p:.1f} boxes with drift off vs {d_on_p:.1f} on. "
        f"box_drift_speed={v_d} is too high."
    )
    print("       partition still beats the swarm under drift   OK")

    # --- 9. stability + throughput under vmapped random actions ---------------
    def vmapped(env, steps=256):
        v_reset, v_step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
        vst = v_reset(jax.random.split(jax.random.PRNGKey(1), n_envs))[1]
        acts = jax.random.uniform(
            jax.random.PRNGKey(2), (steps, n_envs, n_agents, 2), minval=-1, maxval=1
        )
        vst = v_step(vst, acts[0])[1]  # compile
        jax.block_until_ready(vst.data.qpos)
        t0 = time.time()
        for i in range(1, steps):
            o, vst, *_ = v_step(vst, acts[i])
        jax.block_until_ready(o)
        dt = time.time() - t0
        assert jnp.isfinite(vst.data.qpos).all() and jnp.isfinite(vst.data.qvel).all()
        assert jnp.isfinite(o).all() and float(jnp.abs(o).max()) < 1e3, o.max()
        return (steps - 1) * n_envs / dt

    sps_off, sps_on = vmapped(off), vmapped(on)
    assert sps_on > 0.7 * sps_off, (sps_on, sps_off)
    print(
        f"  [9] stability: {n_envs} envs x 256 random steps all finite; "
        f"throughput {sps_on:,.0f} vs {sps_off:,.0f} steps/s "
        f"({100 * sps_on / sps_off:.0f}% of drift-off)   OK"
    )

    # --- 10. difference rewards: pivotal gains, pile-on unchanged -------------
    dr_off = make(reward_mode="difference_rewards")
    dr_on = make(reward_mode="difference_rewards", variant="drift")
    up = jnp.zeros((n_agents, 2)).at[:, 1].set(1.0)

    def d_for(env, n_touch):
        st = clear_state(env, key=5)
        bxy = np.asarray(st.data.qpos[env._box_qadr[:, :2]])
        pos = np.array(st.data.qpos[env._agent_qadr])
        pos[:n_touch] = _face_positions(bxy[0], float(env.box_half_extents[0]), n_touch)
        st = _pose(env, st, agent_pos=pos)
        assert bool(env._coupling_met(st.data)[0])
        return np.asarray(jax.jit(env.step)(st, up)[2])[:n_touch]

    c0 = int(coupling[0])
    piv_off, piv_on = d_for(dr_off, c0), d_for(dr_on, c0)
    pile_off, pile_on = d_for(dr_off, 2 * c0), d_for(dr_on, 2 * c0)
    assert piv_on.mean() > piv_off.mean(), (piv_on, piv_off)
    assert np.allclose(pile_on, pile_off, atol=1e-4), (pile_on, pile_off)
    print(
        f"  [10] difference rewards: pivotal ({c0} touching) D_i "
        f"{piv_off.mean():.3e} -> {piv_on.mean():.3e} (drift helps); "
        f"pile-on ({2 * c0} touching) {pile_off.mean():.3e} -> "
        f"{pile_on.mean():.3e} (unchanged: the drift cancels in G - G_-i)   OK"
    )

    print("drift checks passed.")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=9)
    parser.add_argument("--n-objects", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=32, help="vmap batch size")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="run the box-drift assertion suite instead of the demo rollout "
        "(needs jit, so it ignores --debug)",
    )
    args = parser.parse_args()

    if args.check_drift:
        _check_drift(args.n_agents, args.n_objects, args.n_envs)
        raise SystemExit(0)

    if args.debug:
        jax.config.update("jax_disable_jit", True)

    env = MultiBoxPushMJX(n_agents=args.n_agents, n_objects=args.n_objects)
    print(
        f"world {env.world_width}x{env.world_height}, "
        f"coupling {list(env.objects_push_coupling_list)}, "
        f"box half-extents {list(env.box_half_extents)}"
    )

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
    print(
        f"scripted rollout: return {total:.1f}, "
        f"delivered {np.asarray(info['delivered'])}, "
        f"ended at step {done_at} ({time.time() - t0:.1f}s incl. compile)"
    )

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
    print(
        f"vmapped throughput: {sps:,.0f} env-steps/s "
        f"({args.n_envs} envs, {dt:.2f}s for {args.steps - 1} steps)"
    )
