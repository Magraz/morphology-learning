"""Circular-arena, concentric-goal variant of ``MultiBoxPushMJX``.

Same physics, coupling mechanic, observation layout and reward structure as
``multi_box_push_mjx.py``; what changes is the **geometry**:

- the arena is a **disc** of radius ``arena_radius`` about the world center,
  built as ``_N_WALL_SEGMENTS`` inward-facing planes tangent to that circle (the
  free region is their intersection: a regular N-gon with apothem
  ``arena_radius``, ~0.5% off a true circle at the default 32 segments). MuJoCo
  has no concave primitive, so a polygon of half-spaces is the way to confine
  bodies *inside* a curved boundary — the same construction as the four walls of
  the square arena, just with more of them.
- the goal is **one concentric ring per box**: the ``[0, goal_outer_radius]``
  disc is cut into ``n_objects`` rings of equal width, and **box j belongs in
  ring j counted from the center out** — box 0 in the central disc, box 1 in the
  annulus around it, and so on. So the task is radial *and* per-box: push every
  box inward from whatever bearing it lies, each to a *different* stopping
  radius, leaving the outer ones in place while the inner ones are pushed past
  them. Boxes and rings are color-coded to match (``box_colors``), so a frame
  shows the assignment directly.
  Ring width is the box's side (``2*max(box_half_extents)``) where the arena
  affords it; ``_max_goal_radius`` caps the whole structure at the largest
  radius that still leaves a usable agent spawn annulus, and the rings shrink
  uniformly if that binds.

Everything that referenced the goal axis becomes radial *and* per-box: delivery
is ``ring_inner[j] <= |box j - center| <= ring_outer[j]``, the shaping term is
box j's reduction in ``_goal_dist`` (distance to *its* ring, zero inside it,
growing on *both* sides — so approaching from either side pays and burrowing
past it does not), the ``goal_distance`` observation is the radial offset of the
box an agent is sensing from the centerline of *that box's* ring (via the shared
builder's ``goal_axis="radial"`` mode with a per-agent ``goal_radius`` and
``goal_from_pos``: negative inside that ring, positive outside it, ~0 on it), and
the boundary-contact test is
``|agent - center| >= arena_radius - agent_radius``.

The square-arena env's box-drift mechanic and its ``variant`` presets
(``drift`` / ``trunc``) are deliberately **not** carried over — this env is the
plain task on a disc. What it *does* adopt from the ``trunc`` arm is the
**inert-wall** treatment, and it is the default here (``boundary_ends_episode``
defaults to ``False``; pass ``True`` for the square baseline's semantics):

- the wall segments are real inward-facing planes, so agents are already
  physically confined — ending the episode on contact was Box2D parity, not a
  physical necessity;
- ``boundary_hit`` is ``any()`` over agents, so under the old semantics one of N
  agents ended the episode for the whole team, and agents spawn in the *outer*
  annulus here, i.e. right against the wall;
- worse, the crash step paid ``0`` instead of its real reward. In this env the
  shaping stream is negative whenever boxes drift away from their rings
  (measured: ~1/3 of episodes), so ending the episode on purpose was a standing
  bonus — the same escape hatch the square env's ``boundary_truncates`` attempt
  hit, where policies learned to crash into a wall deliberately.

Every branch is a Python-level ``if``, so ``boundary_ends_episode=True``
reproduces the previous graph and numbers exactly.

2D by construction: bodies own only planar DOFs (agents slide-x/y; boxes
slide-x/y + hinge-yaw), gravity is zero, walls are inward-facing planes — with
no z DOF anywhere, MJX cannot compute out-of-plane motion at all.

Functional JAX API (gymnax-style), fully ``jit``/``vmap``-able; no auto-reset
(on ``terminated | truncated`` the caller resets)::

    env = MultiBoxMultiGoalPushMJX(n_agents=9, n_objects=3)
    obs, state = jax.jit(env.reset)(key)                       # obs (A, 40)
    obs, state, reward, terminated, truncated, info = jax.jit(env.step)(state, actions)

Inherited from the Box2D env / its square-arena MJX port:

- obs: the shared 40-dim ``OBS_DIM`` layout from ``mjx_suite/observation.py``
  (``MJXObservationBuilder``); this env supplies only the qpos layout
  (``_agent_pos``/``_agent_vel``/``_box_pose``) and its goal band — **plus a
  1-scalar extras tail appended after lidar**, so ``observation_dim`` is
  ``MULTI_GOAL_OBS_DIM = 41``. See ``_N_OBS_EXTRAS``. Appending keeps every index
  into the shared 40 dims valid (renderer overlay, analysis scripts) and leaves
  every other env untouched.
- reward: shaping toward the goal band + one-time +100 per delivered box
  (``"sparse"`` drops the shaping); terminate on all-delivered or a wall touch.
- coupling: a box keeps its heavy base mass until ``coupling`` agents touch it,
  then drops to the light coupled mass — a per-step override of ``body_mass`` /
  ``body_inertia`` / ``dof_damping`` on the mjx.Model pytree (jit-safe: the
  model is a step argument). ``_coupling_met`` owns the "enough agents working
  together" predicate.
- constants: dt=1/60, agent mass 1 / radius 0.4 / damping 10, box 2D density 20
  (0.05*coupling when coupled), box lin/ang damping 5/8. Box2D body damping is
  emulated as joint damping = coeff*mass (inertia for the hinge) — same steady
  state ``v_terminal = F / (m*d)``.

Spawn layout follows the square-arena env's *ordering* (agents behind the boxes,
boxes between the agents and the goal) mapped onto the radius: the goal rings in
the middle, boxes on a ring outside *all* of them (so no box starts in anyone's
target), agents in the outer annulus beyond that. Spawns use
shuffled jittered grids rather than rejection sampling (jit needs static
shapes); shaping is live from step 1; box sizes are fixed per instance.

Demo / sanity check (no display needed):
    uv run python -m environments.mjx_suite.multi_box_multi_goal_push_mjx
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from environments.box2d_suite.utils import (
    COLORS_LIST,
    CircularTargetArea,
    resolve_coupling,
)
from environments.mjx_suite.observation import (
    OBS_DIM,
    MJXObservationBuilder,
    geom_index_maps,
)

# --- env-specific observation extras, APPENDED after the shared layout -------
# This env's obs is the shared `OBS_DIM` (40) vector plus a tail of task-specific
# scalars. Appending rather than inserting is deliberate: every index into the
# shared layout (the renderer's `_DENSITY_SLICE` / `_BOX_VEC_SLICE` / `_GOAL_IDX`
# / `_LIDAR_SLICE`, the sensor overlay, any analysis script) stays valid, and no
# other MJX or Box2D env is touched — the extras are the only new width.
#
#   [40] coupling_fraction — the fraction of the *whole team* that must touch the
#        agent's nearest undelivered box simultaneously before it turns light
#        (`coupling[nearest] / n_agents`, in [0, 1]). Pairs with `nearest_box_vec`
#        [21:23] and `goal_distance` [23], which describe the *same* box, so the
#        three together say "that box is there, it must go this far, and it takes
#        this share of the team to move it". 0 when every box is delivered or the
#        env has no objects, matching `nearest_box_vec`'s convention.
_N_OBS_EXTRAS = 1
MULTI_GOAL_OBS_DIM = OBS_DIM + _N_OBS_EXTRAS
_COUPLING_FRAC_IDX = OBS_DIM  # 40

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
# Inward-facing planes tangent to the arena circle. Their intersection is a
# regular N-gon with apothem `arena_radius`; the corners stick out by
# 1/cos(pi/N) - 1, i.e. 0.5% at 32 segments (~0.1 world units at 16a/4o). Higher
# N is rounder but adds N*(A+O) candidate collision pairs and N ray tests per
# lidar beam, so it is the one knob trading fidelity for step cost.
_N_WALL_SEGMENTS = 32

# Spawn layout, radially outward: goal rings -> box ring -> agent annulus -> wall.
# Module-level because `_max_goal_radius` solves them for the largest goal
# structure that still leaves agents somewhere to spawn; keeping one copy is what
# stops that cap from drifting away from the layout it is supposed to protect.
_SPAWN_MARGIN = 2.0  # agent cells stay this far inside the wall
_SPAWN_MIN_DIST = 2.0  # min separation between agent spawn cells
# 0.25 rather than the 0.40 the single-goal version used: the goal structure now
# grows with `n_objects`, and pulling the box spawn ring inward is what buys the
# radial room for every ring to be a full box wide at the shipped configs (at
# 0.40 the cap bit and 9a/3o rings came out 2.23 wide against a 3.0 box). The
# boxes end up at almost the same radius either way, since the goal rim they are
# measured from moved out by about as much as their offset shrank.
_BOX_RING_FRAC = 0.25  # box ring, as a fraction of goal-rim -> wall
_BOX_RING_JITTER_FRAC = 0.10  # +- radial jitter, same units
_MIN_AGENT_ANNULUS = 1.0  # radial width the agent spawn annulus must keep


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EnvState:
    data: mjx.Data
    t: jax.Array  # () int32 — steps taken this episode
    prev_box_goal_dist: jax.Array  # (O,) distance box -> goal band (0 inside)
    delivered: jax.Array  # (O,) bool


class MultiBoxMultiGoalPushMJX:
    def __init__(
        self,
        n_agents: int = 3,
        n_objects: int = 3,
        coupling_def: str = "even",
        max_steps: int = 1024,
        reward_mode: str = "dense",
        boundary_ends_episode: bool = False,
    ):
        if reward_mode not in ("dense", "sparse", "difference_rewards"):
            raise ValueError(
                f"reward_mode must be dense|sparse|difference_rewards, got {reward_mode}"
            )
        self.n_agents = n_agents
        self.n_objects = n_objects
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        # Inert walls by default (the `trunc` treatment) — see the module
        # docstring. `True` restores the square baseline's semantics: a wall
        # touch terminates and that step pays 0.
        self.boundary_ends_episode = boundary_ends_episode
        # `difference_rewards` shapes the *team* reward exactly like "dense" and
        # then decomposes it per agent; a sparse base would leave D_i zero on every
        # step except a delivery, which is too thin to learn from.
        self._dense = reward_mode in ("dense", "difference_rewards")
        self._difference = reward_mode == "difference_rewards"

        # --- world geometry ---
        # The world box is sized exactly as in the square-arena env (so all the
        # normalizations, sensor radii and the renderer's scale carry over); the
        # arena is the inscribed disc. The center is the *geometric* center
        # (W/2, not W//2) so the circle sits symmetrically in that box.
        total_entities = n_agents + n_objects
        self.world_width = int(30 * max(1.0, total_entities / 8) ** 0.75)
        self.world_height = self.world_width
        self.world_center_x = self.world_width / 2
        self.world_center_y = self.world_height / 2
        self.boundary_thickness = 0.5
        self.arena_radius = self.world_width / 2 - self.boundary_thickness
        self._center = jnp.asarray(
            [self.world_center_x, self.world_center_y], dtype=jnp.float32
        )

        self.velocity_norm = self.world_width / 10.0
        self.neighbor_detection_range = 3.0
        self.sector_sensor_radius = self.world_width / 3.0
        self.lidar_range = self.sector_sensor_radius
        self.comm_radius = self.world_width / 3.0
        self.force_multiplier = _FORCE_MULTIPLIER

        # --- coupling requirements and box sizes (fixed per instance) ---
        coupling = resolve_coupling(coupling_def, n_agents, n_objects)
        self.objects_push_coupling_list = coupling
        self.box_half_extents = np.maximum(1.5, coupling * _AGENT_RADIUS)

        # --- one concentric goal ring per box ---
        # The goal is not a single region: the [0, goal_outer_radius] disc is cut
        # into `n_objects` concentric rings of equal width, and **box j must be
        # delivered to ring j counted from the center out** — box 0 to the
        # central disc, box 1 to the annulus around it, and so on. So the boxes
        # cannot be treated interchangeably: each has its own stopping radius,
        # and the outer ones must be left in place while the inner ones are
        # pushed past them.
        #
        # Ring width is the box's side (`2*half`) where the arena affords it, so
        # a box square-on fits inside its ring; `_max_goal_radius` caps the whole
        # structure at the largest radius that still leaves a usable agent spawn
        # annulus, and the rings shrink uniformly if that binds.
        # `box_half_extents` must be known first — hence this block sits below
        # the coupling/size block.
        self.n_goal_rings = n_objects
        preferred = n_objects * 3.0 * float(self.box_half_extents.max())
        cap = self._max_goal_radius()
        if cap <= 0.0:
            raise ValueError(
                f"no room for {n_objects} goal rings in an arena of radius "
                f"{self.arena_radius}"
            )
        self.goal_outer_radius = min(preferred, cap)
        self.goal_ring_width = self.goal_outer_radius / n_objects
        rings = np.arange(n_objects)
        self.goal_ring_inner = rings * self.goal_ring_width  # (O,)
        self.goal_ring_outer = (rings + 1) * self.goal_ring_width  # (O,)
        # Centerline of each ring: the radius its box should come to rest at, and
        # the offset the `goal_distance` observation is measured against. Ring 0
        # is a disc, and the |r - mid| <= half_width test degenerates correctly
        # for it (every r in [0, width] passes), so one formula covers both.
        self.goal_ring_mid = (rings + 0.5) * self.goal_ring_width  # (O,)
        self._ring_mid = jnp.asarray(self.goal_ring_mid, dtype=jnp.float32)
        self._ring_half_width = 0.5 * self.goal_ring_width
        self.target_x = self.world_center_x
        self.target_y = self.world_center_y
        # Read by the shared Renderer (`_draw_goal_distance`) and mirrored by the
        # observation builder's `goal_axis="radial"` mode.
        self.goal_axis = "radial"
        # Each ring carries its box's color (and index), so the renderers can
        # show at a glance which box belongs in which ring. Colors follow the
        # Box2D scheme — COLORS_LIST offset by n_agents — and this attribute is
        # the single source of truth for the box geoms, the rings, and the
        # renderer's box fill.
        self.box_colors = [
            COLORS_LIST[(n_agents + j) % len(COLORS_LIST)] for j in range(n_objects)
        ]
        # Box2D-suite target objects (numpy-only, never touched by jitted code):
        # let the shared Renderer machinery draw the rings.
        self.target_areas = [
            CircularTargetArea(
                self.target_x,
                self.target_y,
                float(self.goal_ring_outer[j]),
                inner_radius=float(self.goal_ring_inner[j]),
                color=(*self.box_colors[j], 110),
                label=f"BOX {j}",
            )
            for j in range(n_objects)
        ]

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

        self._make_box_ring()  # sets _box_ring_r / _box_ring_jitter
        self._agent_spawn_grid = self._make_spawn_grid()
        self._box_spawn_angles = self._make_box_slots()

        # Shared 40-dim layout + this env's extras tail (see _N_OBS_EXTRAS).
        self.observation_dim = MULTI_GOAL_OBS_DIM
        self.action_dim = 2
        # coupling[j] / n_agents, precomputed: the fraction of the team needed to
        # move box j. Constant per instance, so it is a lookup at obs time.
        self._coupling_fraction = jnp.asarray(
            self.objects_push_coupling_list / n_agents, dtype=jnp.float32
        )

    # ------------------------------------------------------------------ model

    def _build_xml(self, heavy_mass: np.ndarray, visual: bool = False) -> str:
        """MJCF for the planar model.

        ``visual=False`` (the physics/obs model handed to MJX): the arena wall is
        ``_N_WALL_SEGMENTS`` inward-facing planes tangent to the arena circle, no
        floor, no colors that matter. ``visual=True`` builds the
        *native-rendering* twin used only by ``MuJoCoNativeRenderer`` — identical
        bodies/joints (same nq/qpos layout, so qpos copies across) but with
        contype-0 cosmetic geometry: slim tangential wall slabs instead of the
        giant planes, a checkered floor below the bodies, the goal disc painted
        on it, a skybox and a shadow-casting light. It is never stepped, only
        mj_forward'd for camera rendering.
        """
        W, H, bt = self.world_width, self.world_height, self.boundary_thickness
        cx, cy = self.world_center_x, self.world_center_y
        R = self.arena_radius
        wall_angles = 2 * np.pi * np.arange(_N_WALL_SEGMENTS) / _N_WALL_SEGMENTS
        # One color per box (see `self.box_colors`); its goal ring gets the same
        # one, so the pairing is readable straight off a frame.
        box_rgba = [
            "{:.3f} {:.3f} {:.3f} 1".format(*(c / 255 for c in color))
            for color in self.box_colors
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
            # Tangential slabs on the outside of the arena circle: half-length
            # R*tan(pi/N) is exactly half the segment chord, so consecutive slabs
            # meet at their corners with no gap.
            seg_half = R * math.tan(math.pi / _N_WALL_SEGMENTS)
            parts += [
                f'    <light directional="true" pos="{cx} {cy - H / 4} 40" '
                'dir="0.1 0.15 -1" diffuse="0.85 0.85 0.85" castshadow="true"/>',
                f'    <geom name="floor" type="plane" pos="{cx} {cy} -0.41" '
                f'size="{cx} {cy} 0.1" material="floor_mat" {cosmetic}/>',
            ]
            # Goal rings: MuJoCo has no annulus primitive, so paint them as
            # nested discs — outermost first at the lowest z, each inner ring
            # stacked just above it (no z-fighting) and covering the previous
            # one's middle. What remains visible of ring j is exactly its
            # annulus, in box j's color. They must be **opaque**: translucent
            # discs blend with the ones below instead of covering them, which
            # turns every ring into a mix of the colors outside it. The whole
            # stack also has to stay between the floor (z=-0.41) and the bottom
            # of the boxes (z=-0.4), or the discs cut through them.
            z_step = 0.008 / self.n_objects
            for j in reversed(range(self.n_objects)):
                r, g, b = (c / 255 for c in self.box_colors[j])
                z = -0.409 + z_step * (self.n_objects - 1 - j)
                parts.append(
                    f'    <geom name="goal_ring_{j}" type="cylinder" '
                    f'pos="{self.target_x} {self.target_y} {z}" '
                    f'size="{self.goal_ring_outer[j]} 0.002" '
                    f'rgba="{r:.3f} {g:.3f} {b:.3f} 1" {cosmetic}/>'
                )
            for k, a in enumerate(wall_angles):
                px, py = cx + (R + bt / 2) * math.cos(a), cy + (R + bt / 2) * math.sin(
                    a
                )
                parts.append(
                    f'    <geom name="wall_{k}" type="box" pos="{px} {py} {wall_z}" '
                    f'euler="0 0 {math.degrees(a)}" '
                    f'size="{bt / 2} {seg_half} {wall_h}" '
                    f'rgba="0.35 0.35 0.38 1" {cosmetic}/>'
                )
        else:
            # The arena wall: `_N_WALL_SEGMENTS` inward-facing planes tangent to
            # the circle of radius R about the world center. Each plane's normal
            # (its frame z axis) points inward, so the free region is the
            # intersection of the half-spaces — a regular N-gon with apothem R.
            # Low friction so contact friction (elementwise max) stays governed
            # by the dynamic geom, approximating Box2D's sqrt(f1*f2) combine.
            for k, a in enumerate(wall_angles):
                ux, uy = math.cos(a), math.sin(a)  # outward normal of segment k
                px, py = cx + R * ux, cy + R * uy
                parts.append(
                    f'    <geom name="wall_{k}" type="plane" pos="{px} {py} 0" '
                    f'zaxis="{-ux} {-uy} 0" size="{W} {W} 0.1" friction="0.05"/>'
                )
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

    def _agent_annulus_inner(self, goal_outer_radius: float) -> float:
        """Inner radius of the agent spawn annulus for a given goal rim.

        Everything outward of the goal is laid out from that rim: the box ring
        sits `_BOX_RING_FRAC` of the way to the wall (plus jitter), and agents
        start beyond the outermost box's surface so one can never spawn inside a
        box. Written as a function of the rim so `_max_goal_radius` can invert
        it instead of restating the chain.
        """
        span = self.arena_radius - goal_outer_radius
        return (
            goal_outer_radius
            + (_BOX_RING_FRAC + _BOX_RING_JITTER_FRAC) * span
            + float(self.box_half_extents.max())
            + _AGENT_RADIUS
            + 0.5
        )

    def _max_goal_radius(self) -> float:
        """Largest goal rim that still leaves a usable agent spawn annulus.

        Solves `arena_radius - _SPAWN_MARGIN - _agent_annulus_inner(G) >=
        _MIN_AGENT_ANNULUS` for G. With one ring per box the goal structure grows
        with `n_objects`, and past this radius it would push the agents' spawn
        region out through the wall — so the rings are capped here and shrink
        uniformly rather than the layout silently failing.
        """
        f = _BOX_RING_FRAC + _BOX_RING_JITTER_FRAC
        clearance = float(self.box_half_extents.max()) + _AGENT_RADIUS + 0.5
        return (
            (1 - f) * self.arena_radius - _SPAWN_MARGIN - clearance - _MIN_AGENT_ANNULUS
        ) / (1 - f)

    def _make_box_ring(self) -> None:
        """Radius (and jitter) of the ring the boxes spawn on.

        The square arena spawned boxes in a central band with the goal beyond it
        and the agents behind them; radially that ordering becomes goal rings ->
        box ring -> agent annulus. The ring sits `_BOX_RING_FRAC` of the way from
        the outermost goal ring's rim to the wall, leaving room for a coalition
        to work its way around a box on any side. Every box spawns outside
        *every* goal ring, so no box starts in another's target.
        """
        span = self.arena_radius - self.goal_outer_radius
        self._box_ring_r = self.goal_outer_radius + _BOX_RING_FRAC * span
        self._box_ring_jitter = _BOX_RING_JITTER_FRAC * span

    def _make_spawn_grid(self) -> jnp.ndarray:
        """Candidate agent spawn cells: the outer annulus, jitter-safe >=2 apart.

        Concentric rings of evenly spaced cells, the radial analogue of the
        square arena's meshgrid over the bottom third. The inner radius clears
        the outermost box surface so an agent can never spawn inside a box.
        """
        min_dist = _SPAWN_MIN_DIST
        spacing = min_dist + 0.5
        r_lo = self._agent_annulus_inner(self.goal_outer_radius)
        r_hi = self.arena_radius - _SPAWN_MARGIN
        if r_hi <= r_lo:
            raise ValueError("agent spawn annulus is empty; arena too small")
        n_rings = int((r_hi - r_lo) // spacing) + 1
        radii = np.linspace(r_lo, r_hi, n_rings)
        radial_gap = (radii[1] - radii[0]) if n_rings > 1 else np.inf

        cells, angular_gap = [], np.inf
        for r in radii:
            n_ang = max(1, int(2 * np.pi * r // spacing))
            ang = np.arange(n_ang) * (2 * np.pi / n_ang)
            angular_gap = min(angular_gap, 2 * np.pi * r / n_ang)
            cells.append(
                np.stack(
                    [
                        self.world_center_x + r * np.cos(ang),
                        self.world_center_y + r * np.sin(ang),
                    ],
                    axis=1,
                )
            )
        grid = np.concatenate(cells)
        if len(grid) < self.n_agents:
            raise ValueError("spawn region too small for n_agents")
        # Same rule as the square grid: jitter can consume at most half the
        # smallest cell gap on each side, so cells stay `min_dist` apart.
        gap = min(radial_gap, angular_gap)
        self._agent_spawn_jitter = max(0.0, (gap - min_dist) / 2)
        return jnp.asarray(grid, dtype=jnp.float32)  # (n_cells, 2)

    def _make_box_slots(self) -> jnp.ndarray:
        """Candidate box angular slots on the spawn ring, min separation kept.

        Radial jitter only ever *increases* the distance between boxes on
        distinct angular slots, so the arc spacing alone bounds the separation.
        """
        h_max = float(self.box_half_extents.max())
        spacing = max(4.0, 2 * h_max + 1.0)  # Box2D min_separation / min_x_separation
        n_slots = max(1, int(2 * np.pi * self._box_ring_r // spacing))
        if n_slots < self.n_objects:
            raise ValueError("spawn ring too short for n_objects")
        angles = np.arange(n_slots) * (2 * np.pi / n_slots)
        return jnp.asarray(angles, dtype=jnp.float32)

    # ------------------------------------------------------------------ helpers

    def _agent_pos(self, data) -> jnp.ndarray:
        return data.qpos[self._agent_qadr]  # (A, 2)

    def _agent_vel(self, data) -> jnp.ndarray:
        return data.qvel[self._agent_dadr]  # (A, 2)

    def _box_pose(self, data) -> tuple[jnp.ndarray, jnp.ndarray]:
        q = data.qpos[self._box_qadr]  # (O, 3)
        return q[:, :2], q[:, 2]  # positions (O, 2), yaws (O,)

    def _radius(self, pos: jnp.ndarray) -> jnp.ndarray:
        """(N,) distance of each (N, 2) position from the goal/arena center."""
        return jnp.linalg.norm(pos - self._center, axis=-1)

    def _goal_dist(self, box_pos: jnp.ndarray) -> jnp.ndarray:
        """(O,) distance from each box to **its own** goal ring.

        Indexed by box: entry j measures box j against ring j, so this is where
        the per-box goal assignment actually lives. Zero everywhere inside the
        ring and growing on **both** sides of it — outward toward the wall and
        inward toward the middle — so the shaping term it feeds pays for
        approaching the ring from either side and pays nothing for burrowing
        past it. (A plain distance-to-center would keep rewarding inward motion
        all the way to the middle, i.e. reward overshooting every ring but the
        innermost.) Ring 0 is a disc, and this degenerates correctly for it:
        `|r - w/2| - w/2 <= 0` for every `r` in `[0, w]`.
        """
        return jnp.clip(
            jnp.abs(self._radius(box_pos) - self._ring_mid) - self._ring_half_width,
            0.0,
        )

    def _outward(self, pos: jnp.ndarray) -> jnp.ndarray:
        """(N, 2) unit vector from the center toward each position.

        Safe at the center itself (returns ~0 there rather than NaN); the only
        callers gate on states where that cannot matter.
        """
        rel = pos - self._center
        return rel / (jnp.linalg.norm(rel, axis=-1, keepdims=True) + 1e-6)

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

        """
        agent_pos = self._agent_pos(data)
        box_pos, box_yaw = self._box_pose(data)
        touch = self._touch_matrix(agent_pos, box_pos, box_yaw)  # (A, O)
        if active is not None:
            touch = touch & active[:, None]
        return touch.sum(axis=0) >= self._coupling  # (O,)

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
        """(A, MULTI_GOAL_OBS_DIM) — the shared Box2D-suite layout built by
        obs_builder, plus this env's extras tail (see ``_N_OBS_EXTRAS``).

        ``delivered`` (O,) bool excludes already-delivered boxes from
        ``nearest_box_vec`` so agents stop being drawn to a box parked in its
        goal ring; ``None`` senses every box (the pre-delivery / no-mask case).
        """
        agent_pos = self._agent_pos(data)
        box_pos, box_yaw = self._box_pose(data)
        # Each box has a *different* target ring, so a single global goal radius
        # would tell an agent nothing about where its box has to go. The feature
        # is therefore keyed to the box that agent is sensing — the same
        # nearest-undelivered-box the `nearest_box_vec` component points at, so
        # the two features describe one box.
        nearest = self.obs_builder.nearest_box_indices(agent_pos, box_pos, delivered)
        # `nearest` is an argmin over +inf for an agent with no box in range, so
        # every feature keyed to it must be gated on this mask (the goal-distance
        # component is gated inside `build`; `_coupling_fractions` takes it).
        sensed = self.obs_builder.nearest_box_sensed(agent_pos, box_pos, delivered)
        base = self.obs_builder.build(
            data,
            agent_pos=agent_pos,
            agent_vel=self._agent_vel(data),
            box_pos=box_pos,
            box_yaw=box_yaw,
            box_half=self._box_half,
            goal_coord=self._center,  # concentric goal: distance is radial
            goal_axis="radial",
            # Offset by that ring's centerline, and measured from the *box*
            # rather than the agent: the quantity that has to be driven to zero
            # is how far the sensed box still is from its own ring — negative
            # inside it, positive outside it, ~0 on its centerline. (The agent's
            # own radial offset said nothing about the box's remaining work, and
            # went to zero when the agent stood on the ring itself.)
            goal_radius=self._ring_mid[nearest],  # (A,)
            goal_from_pos=box_pos[nearest],  # (A, 2)
            delivered=delivered,
        )
        return jnp.concatenate(
            [base, self._coupling_fractions(nearest, sensed)[:, None]], axis=1
        )

    def _coupling_fractions(self, nearest, sensed) -> jnp.ndarray:
        """(A,) share of the whole team needed to move each agent's nearest box.

        ``coupling[nearest] / n_agents`` in [0, 1] — how much of the team has to
        be touching that box *simultaneously* before it drops to its light mass.
        Keyed to the same nearest *sensed* box as ``nearest_box_vec`` and
        ``goal_distance``, so the three features describe one box.

        Zero when the env has no objects, and (like ``nearest_box_vectors``) for
        any agent with no box in range — ``nearest`` is an argmin over +inf
        there, so the index is meaningless and must not be read as a real
        requirement. That subsumes the every-box-delivered case, which is why
        this takes the ``sensed`` mask rather than ``delivered``.
        """
        if self.n_objects == 0:
            return jnp.zeros(self.n_agents)
        return jnp.where(sensed, self._coupling_fraction[nearest], 0.0)

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

        # boxes: shuffled angular slots on the spawn ring, jittered in radius
        slots = jax.random.permutation(k_slots, self._box_spawn_angles.shape[0])[
            : self.n_objects
        ]
        box_ang = self._box_spawn_angles[slots]
        box_r = self._box_ring_r + jax.random.uniform(
            k_boxy,
            (self.n_objects,),
            minval=-self._box_ring_jitter,
            maxval=self._box_ring_jitter,
        )
        box_xy = self._center + box_r[:, None] * jnp.stack(
            [jnp.cos(box_ang), jnp.sin(box_ang)], axis=1
        )  # (O, 2)

        qpos = jnp.zeros(self._mj_model.nq)
        qpos = qpos.at[self._agent_qadr].set(agent_pos)
        qpos = qpos.at[self._box_qadr[:, :2]].set(box_xy)

        data = mjx.make_data(self._model).replace(qpos=qpos)
        data = mjx.forward(self._model, data)

        state = EnvState(
            data=data,
            t=jnp.zeros((), dtype=jnp.int32),
            prev_box_goal_dist=self._goal_dist(box_xy),
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
        return mjx.step(model_t, data)

    def _task_reward(self, state: EnvState, data):
        """Team reward for a post-step `data` against the pre-step `state`.

        Pure in (state, data), so counterfactual branches reuse it directly —
        no recursion through `step`. Returns
        (reward, newly_delivered, boundary_hit, dist).
        """
        agent_pos = self._agent_pos(data)
        box_pos, _ = self._box_pose(data)

        # boundary termination: any agent touching the arena wall. Measured on
        # the circle rather than the wall polygon, so in the N-gon's corner
        # directions it trips ~R*(1/cos(pi/N) - 1) early — 0.5% of R at the
        # default 32 segments, i.e. conservative by a fraction of an agent.
        boundary_hit = jnp.any(
            self._radius(agent_pos) >= self.arena_radius - _AGENT_RADIUS - _WALL_EPS
        )

        # reward: shaping toward each box's own goal ring + delivery bonus
        dist = self._goal_dist(box_pos)  # (O,) distance to box j's ring j
        shaping = jnp.sum((state.prev_box_goal_dist - dist) * (~state.delivered))
        # `_goal_dist` is clipped at 0, so "distance 0" *is* "inside my ring"
        # (ring_inner[j] <= |box j - center| <= ring_outer[j]) — one source of
        # truth for the two. A box parked in someone else's ring is not
        # delivered, and keeps bleeding shaping until it reaches its own.
        in_goal = dist <= 0.0
        newly_delivered = in_goal & ~state.delivered
        completion = 100.0 * newly_delivered.sum()
        task_reward = completion + (shaping if self._dense else 0.0)

        # Box2D skips reward/delivery bookkeeping on a boundary hit — but only
        # because the hit *is* the episode's terminal failure there. With inert
        # walls (the default) the step physically happened like any other and
        # must pay its real reward: zeroing it would hand back exactly the
        # negative shaping the agent would otherwise eat, i.e. a standing bonus
        # for touching a wall. Python-level `if`, so the old graph and numbers
        # are reproduced exactly under `boundary_ends_episode=True`.
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
            # Wall contact is inert: the step counted, so a delivery that landed
            # on it counts too.
            delivered = state.delivered | newly_delivered
        t = state.t + 1
        all_delivered = jnp.all(delivered)
        terminated = (boundary_hit | all_delivered) if self.boundary_ends_episode \
            else all_delivered
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


def scripted_push_action(env: MultiBoxMultiGoalPushMJX, state: EnvState, box_idx=0):
    """Hand-written cooperative controller: each agent converges on a staging
    point just *outside* its assigned box (on the far side from the goal center)
    and then pushes straight inward. Delivers the box through the coupling
    mechanic — used by the module demos and the renderer demo as a non-random
    rollout.

    The radial counterpart of the square-arena controller's "stage below, push
    up": the push direction is the box's inward radial unit vector, and the
    staging offset / lateral slots are expressed in that box-relative frame, so
    it works for a box at any bearing from the goal.

    Unlike the square arena — where every agent spawned *behind* the box and
    could drive straight at its staging point — agents here start at every
    bearing, and a straight line to the staging point runs through the middle of
    the arena and into the box's *inner* face. An agent arriving that way pushes
    the box **outward**, away from the goal (measured: the swarm shoved its box
    from r=10 to r=14.5 and into the wall). So the approach is an orbit: while
    an agent is inside the docking ring or off its bearing it circles at the
    docking radius toward the staging bearing, and only closes in once it is
    outside the box and roughly behind it.

    ``box_idx`` is a scalar (the whole team swarms one box) or an ``(A,)`` array
    assigning a box per agent, e.g. ``jnp.arange(A) % O`` for a balanced
    partition — the two arms of the swarm-vs-partition comparison.

    Agents sharing a box get *distinct* slots spread along its outward face. With
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
    outward = env._outward(box)  # (A, 2) center -> box
    tangent = jnp.stack([-outward[:, 1], outward[:, 0]], axis=-1)
    # Rank within the group sharing this box, and that group's size.
    same = idx[:, None] == idx[None, :]
    order = jnp.arange(env.n_agents)
    rank = (same & (order[None, :] < order[:, None])).sum(axis=-1)
    group = same.sum(axis=-1)
    slot = (rank - (group - 1) / 2) * (2 * _AGENT_RADIUS + 0.1)
    lateral = jnp.clip(slot, -half, half)
    # `half + 0.6` puts the staging point exactly at the touch threshold
    # (agent radius 0.4 + touch eps 0.2), so an agent that reaches it is already
    # in contact. Standing further off (e.g. clearing the box's sqrt(2)*half
    # corner reach) breaks the coupling: the agent hovers at the staging point,
    # pushes in, loses the `close` test, and is pulled back out — measured as a
    # minimum coalition never getting all `coupling` members onto the box.
    stand_off = half + 0.6
    stage_pt = box + outward * stand_off[:, None] + tangent * lateral[:, None]
    to_stage = stage_pt - agent_pos
    stage_dist = jnp.linalg.norm(to_stage, axis=1, keepdims=True)
    approach = to_stage / (stage_dist + 1e-6)

    # Orbit: circle at the docking radius toward the staging bearing, correcting
    # radius as you go. The tangential term is a unit vector and the radial one
    # is clipped to 1, so the motion is never more than 45 deg off tangential —
    # an agent that starts *at* the box's bearing slides around it rather than
    # driving through it.
    u_agent = env._outward(agent_pos)  # (A, 2) agent bearing
    u_stage = env._outward(stage_pt)
    dock_r = env._radius(stage_pt)
    spin = jnp.sign(
        u_agent[:, 0] * u_stage[:, 1] - u_agent[:, 1] * u_stage[:, 0]
    )  # +1 CCW toward the staging bearing
    orbit = (
        spin[:, None] * jnp.stack([-u_agent[:, 1], u_agent[:, 0]], axis=-1)
        + jnp.clip(dock_r - env._radius(agent_pos), -1.0, 1.0)[:, None] * u_agent
    )
    orbit = orbit / (jnp.linalg.norm(orbit, axis=1, keepdims=True) + 1e-6)

    # Behind the box and already out at the docking radius -> stop orbiting and
    # close in. The bearing tolerance has to include the box's *own* angular
    # half-width, which grows as it nears the center: with a fixed cone the
    # agents in the outer lateral slots fall outside it once the box is close in
    # (at 16a/4o a slot 1.35 off-axis subtends 19 deg at r=4), so they orbit
    # forever and the coalition stalls one agent short of `coupling`.
    box_r = jnp.maximum(env._radius(box), half)
    ang_tol = jnp.radians(20.0) + jnp.arcsin(jnp.clip(half / box_r, 0.0, 1.0))
    behind = ((u_agent * u_stage).sum(axis=1) > jnp.cos(ang_tol)) & (
        env._radius(agent_pos) >= dock_r - 0.5
    )
    move = jnp.where(behind[:, None], approach, orbit)
    action = jnp.where(stage_dist < 0.7, -outward, move)
    # Each goal is a *ring*, so pushing does not stop being wrong once the box is
    # in it: keep going and the box leaves through the inner edge into some other
    # box's ring. Agents whose box is delivered go limp, and the ~1.4 units the
    # box coasts before the damping kills its speed carry it from the ring's
    # outer edge (where delivery triggers) to about the centerline of a
    # box-wide ring — so it parks *inside* its ring rather than straddling the
    # edge. Chasing the centerline explicitly instead overshoots by that same
    # coast and drops the box into the next ring in. Delivery latches in
    # `EnvState`, so this never oscillates.
    return jnp.where(state.delivered[idx][:, None], 0.0, action)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=9)
    parser.add_argument("--n-objects", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=32, help="vmap batch size")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--debug", type=bool, default=True)
    args = parser.parse_args()

    if args.debug:
        jax.config.update("jax_disable_jit", True)

    env = MultiBoxMultiGoalPushMJX(n_agents=args.n_agents, n_objects=args.n_objects)
    print(
        f"world {env.world_width}x{env.world_height}, arena r={env.arena_radius}, "
        f"coupling {list(env.objects_push_coupling_list)}, "
        f"box half-extents {list(env.box_half_extents)}"
    )
    print(f"goal rings (width {env.goal_ring_width:.2f}, box j -> ring j):")
    for j in range(env.n_objects):
        print(
            f"  box {j}: r in [{env.goal_ring_inner[j]:.2f}, "
            f"{env.goal_ring_outer[j]:.2f}]  color {env.box_colors[j]}"
        )
    print(f"boxes spawn at r ~ {env._box_ring_r:.2f} +- {env._box_ring_jitter:.2f}")

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    key = jax.random.PRNGKey(0)
    obs, state = reset(key)
    assert obs.shape == (args.n_agents, MULTI_GOAL_OBS_DIM), obs.shape
    assert obs.shape[1] == env.observation_dim
    print(f"obs shape OK: {obs.shape} (shared {OBS_DIM} + {_N_OBS_EXTRAS} extras)")
    frac = np.asarray(obs[:, _COUPLING_FRAC_IDX])
    print(
        f"  coupling_fraction [{_COUPLING_FRAC_IDX}]: {np.unique(frac.round(4))} "
        f"(coupling {list(env.objects_push_coupling_list)} / {env.n_agents} agents)"
    )

    # --- scripted sanity rollout: a balanced partition, each box to its ring ---
    # (Not the swarm: with one ring per box, delivering them all is the task.)
    assign = jnp.arange(args.n_agents) % args.n_objects
    total, done_at = 0.0, None
    t0 = time.time()
    for i in range(1024):
        obs, state, r, term, trunc, info = step(
            state, scripted_push_action(env, state, assign)
        )
        total += float(r)
        if bool(term) or bool(trunc):
            done_at = i + 1
            break
    box_r = np.asarray(env._radius(env._box_pose(state.data)[0]))
    print(
        f"scripted rollout: return {total:.1f}, "
        f"delivered {np.asarray(info['delivered'])}, "
        f"ended at step {done_at} ({time.time() - t0:.1f}s incl. compile)"
    )
    print(
        "  final box radii vs their rings: "
        + ", ".join(
            f"box {j} r={box_r[j]:.2f} in [{env.goal_ring_inner[j]:.2f}, "
            f"{env.goal_ring_outer[j]:.2f}]"
            for j in range(env.n_objects)
        )
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
