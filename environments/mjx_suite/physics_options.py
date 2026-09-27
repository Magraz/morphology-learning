"""MuJoCo solver and contact settings shared by the MJX push envs.

One definition for ``MultiBoxPushMJX`` and ``MultiBoxMultiGoalPushMJX``, so the
two ``_build_xml`` copies cannot drift apart.

Almost all of an MJX env step is the constraint solver: observations and lidar
take ~0.05 ms of a 3.4-5.9 ms step. Two MuJoCo defaults made it expensive under
``vmap``:

- ``iterations=100, ls_iterations=50``. The Newton loop is a ``while_loop``, so
  a vmapped batch runs until its slowest env converges. The line search is a
  fixed-length scan whose early exit becomes a select under vmap, so every
  Newton iteration pays all ``ls_iterations`` line-search steps.
- One contact slot per candidate geom pair. At 12a/3o that is 210 contacts
  (840 constraint rows at 4 pyramidal rows each), while ~9 touch on average and
  35 at most.

``max_contact_points`` keeps the most-penetrating contacts, up to the cap. It is
exact while the number of touching contacts (``dist < 0``) stays below the cap,
because the dropped slots carry zero force. When the cap binds, the
least-penetrating real contacts are dropped, and bodies can pass through each
other. Check any new, more crowded config (see CLAUDE.md, "MJX suite").

Measured 2026-09-25 (RTX 4080 SUPER, 32 envs), single-step, on identical
states, against the old defaults. The relative error of the velocity update has
median ~1e-6 and p99 <= 2e-2, including 16 agents crowding one box. A
``mappo_jax`` iteration went from 2.60 to 1.26 s at 12a/3o and from 4.17 to
1.42 s at 16a/4o. Four or fewer iterations change the physics (50-200% of the
velocity update), even though scripted deliveries still succeed. One iteration
NaNs. The CG solver is 5x slower.
"""

SOLVER_ITERATIONS = 20
SOLVER_LS_ITERATIONS = 10
MIN_CONTACT_CAP = 64
CONTACT_CAP_PER_BODY = 4


def contact_cap(n_agents: int, n_objects: int) -> int:
    """Maximum contacts handed to the solver for an env of this size.

    ``4 * (n_agents + n_objects)``, with a floor of 64. The most touching
    contacts measured were 35 at 12a/3o (scripted deliveries) and 32 at 16a/4o
    (all 16 agents on one box), against caps of 64 and 80.
    """
    return max(MIN_CONTACT_CAP, CONTACT_CAP_PER_BODY * (n_agents + n_objects))


def physics_xml(timestep: float, n_agents: int, n_objects: int) -> list[str]:
    """The ``<option>`` and ``<custom>`` MJCF lines for the planar push envs.

    ``implicitfast`` integrates joint damping implicitly, the same semantics as
    Box2D's ``v /= (1 + damping * dt)``. The friction cone stays the default
    pyramidal one: elliptic NaNs out when a light coupled box is crushed against
    a wall by many agents.
    """
    return [
        f'  <option timestep="{timestep}" gravity="0 0 0" integrator="implicitfast" '
        f'iterations="{SOLVER_ITERATIONS}" ls_iterations="{SOLVER_LS_ITERATIONS}"/>',
        '  <custom><numeric name="max_contact_points" '
        f'data="{contact_cap(n_agents, n_objects)}"/></custom>',
    ]
