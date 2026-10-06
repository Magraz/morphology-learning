**Simplified feudal MAPPO with explicit rollout interventions**

Date: 5 October 2026, revised the same day after a feasibility check against
the code. Status: **implemented and smoke-tested 2026-10-05; no training
result.** CLAUDE.md ("Training forks") records what was verified and measured.
Measured collection cost at 2a/4o: 1.5–2.1 s per update at interval 1 and
1.0–1.4 s at interval 2, against 0.6–0.8 s for the parent (section 2's
physics-only estimate was optimistic). The full runs in section 9 are not
launched.

**Changed after implementation, at the author's request:** `n_total_steps`
caps SIMULATOR steps (main plus every stepped fork lane), and `total_steps`
counts them. At 2a/4o and interval 1 a 1e8 budget gives 986 updates against the
parent's 2959. This supersedes "`total_steps` stays main-only" in section 6.
Under it, the extra-data control runs at the SAME budget as the arm (sections 1
and 9 are updated).
Target: `algorithms/simplified_feudal_mappo_jax`.

Scope: one addition. A named variant collects real simulator forks at manager
decisions during training. Each fork lasts one `goal_horizon`, is an
independent mini episode, and its transitions are added to both levels' PPO
batches. Execution and evaluation are unchanged. The plan delivers code, seam
tests, model groups and a smoke run of at most 2e5 steps; the full runs in
section 9 are for the author to launch.

**1. Requirements and defaults**

The requested behavior is:

- At every `intervention_interval`-th manager decision (default 1, every
  decision), sample the main manager decision, then fork the complete
  environment state before the first worker action executes.
- In the fork, teleport N existing teammates near focal agent i and assign
  each the same goal offset as i, relative to its own new position.
- Advance the main world and fork for `goal_horizon` worker steps, subject to
  earlier environment termination. Only the main world continues afterward.
- Keep main and intervention transitions in separate buffers. Every fork has
  its own terminal boundary, including a `done` on its final valid step.
- Compute generalized advantage estimation (GAE) separately, then mix and
  shuffle the samples for the usual proximal policy optimization (PPO)
  losses. Sources mix within each policy's batch; manager and worker keep
  separate optimizers and their existing timescales.

| Choice | Setting |
|---|---|
| Intervention cadence | Every `intervention_interval`-th manager decision (default 1 = every decision), in every environment stream; section 2 |
| Forks per decision | One per agent (confirmed by the author) |
| Recruitment count | N sampled independently per fork (confirmed), uniform over `1 .. n_agents - 1` |
| Recruit identities | N distinct teammates uniformly without replacement, excluding i |
| Proximity | Recruit centers within 1.5 **world units** of i's center; provisional |
| Recruit velocities | Preserved; teleport changes position and assigned goal only |
| Fork duration | Exactly `goal_horizon`; no separate knob |
| Early endings | Original environment clock; the fork ends early if the environment ends |
| Manager credit | `team` only; `counterfactual` / `dpp` are rejected in version 1 |
| Parent model | `simplified_feudal_tanh_relative_input` (the arm that breaks the one-box cap; the `_cf` / `_dpp` arms use it too) |
| Testbeds | `mjx_2a_4o_1122_1024_gs` first (N is always 1), then `mjx_6a_4o_1024_gs` (N in 1..5) |
| Optimizer steps | Unchanged per update; minibatches grow with the mixed batch (the existing `n_minibatches` rule) |
| Evaluation | Ordinary environments, interventions off (`eval_fn` and `view()` untouched) |

Every `(window, environment, focal agent)` gets its own N draw, independent of
the sampled goals. A one-agent environment cannot run the variant (the config
validator raises); the parent still runs one-agent groups.

The variant learns directly from intervened rollouts. It needs no D++ bonus, no
maximum over recruitment counts and no learned counterfactual model. Real
teammates are relocated, so their lost contribution elsewhere appears in the
fork reward.

**2. Collection**

```text
Main manager samples joint goals at state S_t
                    |
         +----------+-------------------+
         |                              |
  Main state S_t               A forks copied from S_t (E*A lanes)
  Original goals               Fork i: teleport recruits; copy i's offset
         |                              |
  H worker steps (E lanes)      H worker steps (E*A lanes)
         |                              |
  Main worker transitions       Fork worker transitions
  Main manager transition       Fork manager transition, done=True
         |                              |
  Continue / normal reset       Discard fork state
         |
  Next main manager decision

H = goal_horizon
```

Inside `_window`, after the main decision on an intervention window:

1. Tile the start state and observations to `E * A` lanes
   (`jnp.repeat(..., A, axis=0)`, environment-major, focal index = lane mod A).
2. Apply the teleport hook per lane (section 3). Lanes whose placement fails
   start with `alive = False`.
3. Run a second `lax.scan` of the existing `_worker_step` over the fork lanes,
   with the fork waypoints held fixed.

`_worker_step` is reused with two changes: the batch width is read from its
inputs (`active_mask` currently broadcasts to the main `n_envs`), and a static
`bootstrap` flag turns off both time-limit value additions for forks. Setting
`done` alone would not remove value already added to the reward.

**Interval.** The window scan gets `xs = jnp.arange(W)`, and window `w` is an
intervention window when `w % intervention_interval == 0`. The index counts the
rollout's manager decisions. Every rollout starts from reset, so this is also
the episode's decision index, except after an early (all-delivered)
termination mid-rollout. Window 0 of every rollout always forks.

The fork branch sits under `lax.cond` on that predicate. The window scan is not
vmapped, so the skipped branch runs no physics. The other branch returns a
same-shaped placeholder: worker and manager rows with `done = True`,
`active_mask = 0`, and zero reward and value. Every placeholder row is fully
masked in the update (section 5). At `intervention_interval: 1` the cond is
dropped at trace time, so the default compiles the every-decision path.

Random numbers: fork keys come from `jax.random.fold_in` on the window's
manager key with a fork namespace constant, so the main path consumes no extra
splits. With equal parameters and seed, the variant's main actions, states and
rewards therefore equal the parent's (only the manager critic's values differ,
through its wider input). The fork carry and final states are discarded; no
fork successor enters the main carry. Forks never generate further forks.

Measured cost (2026-10-05, idle GPU, `MultiBoxPushMJX` `trunc`, random actions,
one 32-step window, median of 5): 32 vs 96 lanes at 2a/4o took 10.0 vs
10.6 ms; 32 vs 224 lanes at 6a/4o took 12.5 vs 19.9 ms. At interval 1 the
simulated steps grow `(1 + A)`-fold (3x and 7x), but physics wall-clock grows
only ~1.06x and ~1.6x because the GPU is underused at 32 lanes. In general they
grow `(1 + A / interval)`-fold, since skipped windows run no fork physics.
Policy forwards and the update are not included, so re-measure the full
iteration in the smoke run. No chunking is needed at these widths (MJX
throughput was already measured to scale to 256 lanes).

**3. Teleportation and goal semantics**

Use the focal agent's realized offset after the manager action bound and arena
clip:

```text
delta_i = main_waypoint_i - main_position_i
fork_waypoint_j = teleported_position_j + delta_i     for every recruit j
```

In goal-state units this copies `wp.goal_error(waypoint, pos, R)[i]`. Copying
the raw Gaussian action would differ near an arena boundary. The focal agent
and every non-recruited agent keep their original positions and waypoints.
Each worker still receives its own live error; primitive actions are sampled
from the worker policy, not copied from i.

Recruitment is a fixed-length `(A,)` boolean mask: rank teammates by a random
key (excluding i) and take ranks `< N`, the same rank-mask pattern as
`counterfactual.recruit_mask`. Shapes do not depend on N, so nothing
recompiles.

Add `MultiBoxPushMJX.teleport_agents(state, focal, recruit_mask, offset, key,
radius) -> (obs, state, valid)`: pure, vmappable, environment-owned. It must:

- Place each recruit in the annulus between `2 * agent_radius + margin` and
  `radius` around i, with its center at least `boundary_thickness +
  agent_radius + margin` from every wall, at least `agent_radius + margin` from
  every box surface, and at least `2 * agent_radius + margin` from every other
  agent, including recruits already placed. Box clearance uses the rotated-box
  clamp distance in `MJXObservationBuilder.touch_matrix`; factor that distance
  into one helper so the touch test and placement share it.
- Reject a candidate whose translated waypoint `goal_state(candidate) + offset`
  leaves `[-0.5, 0.5]` on either axis. Clipping it would break the equal-offset
  requirement.
- Search K static candidates per recruit, placing recruits in mask order. If any
  recruit has no valid candidate, return `valid = False` and the unmodified
  state. Never enlarge the radius or reduce N.
- Change only recruit `qpos`. Keep all velocities, box poses, `t`, `delivered`
  and `prev_box_goal_dist` (boxes did not move, so no reward rebase is needed).
  Then run `mjx.forward` with `_model_for(_coupling_met(data))`, so the
  contact-force observation sees the same masses a step would, and set
  `qacc_warmstart = qacc` from that pass. Rebuild observations with `_get_obs`.
- Pay no reward: teleportation is not an environment step. Worker progress is
  measured from the post-teleport positions.

`_pose` is reference only: it zeroes every velocity and rebases reward history.

Feasibility of the radius: with 0.8 minimum center spacing, at most 12 agents
fit within 1.5 of i in open space (6 at 0.8, 6 at 1.39), roughly half next to a
box face or wall, and a random candidate search finds fewer. N up to 5 at 6a/4o
is feasible; teams above ~12 agents are not at this radius. Report placement
success per N (section 9).

The `StubEnv` in `test_simplified_feudal.py` gets a minimal hook (place
recruits at fixed offsets from i, with a switch that forces `valid = False`).
Other environments raise when the variant is enabled.

**4. Manager eligibility and the critic context**

The fork manager record is the decision sampled in the **original
pre-teleport state**: store the main actor input, raw action and log
probability. Never re-evaluate that action on post-teleport observations.

Only focal agent i contributes to the fork's manager actor and entropy losses:
its `active_mask` row is the focal one-hot, times `valid`. Recruits' offsets
were imposed, so they are not policy samples. The main manager transition
keeps its all-ones mask. Every live agent's fork **worker** action is a real
policy sample and stays eligible (`active_mask = alive`, as in the main path).

The manager critic must separate a continuing main return from a one-window
fork return. In this variant only, append to its usual pre-decision input
`wp.manager_critic_input(gs, pos)`:

- `is_intervention`;
- a focal one-hot vector;
- the recruit fraction `N / (A - 1)`.

Main rows carry zeros. N is drawn independently of the manager action, so the
context does not depend on the action being credited. The critic does not see
the sampled goal or the post-teleport state, both of which depend on that
action. Every manager critic call (`_window`, the truncation bootstrap in
`_worker_step`, `collect_fn`'s last value) goes through one helper that
appends the context, and `wp.input_dims` adds its width `A + 2`. The actor and
worker widths are unchanged. A parent checkpoint loaded into this variant
fails loudly at the first manager critic call (`ScopeParamShapeError`).

The worker critic needs no context: a fork worker episode has the same
structure as a main one (an H-step waypoint commitment).

These samples optimize goals under supplied support as well as ordinary
execution. Unassisted evaluation decides whether that transfers.

**5. Buffers, terminals and advantages**

Extend `Rollout` with an optional `intervention` field (default `None`) holding
a fork worker `Transition`, a fork manager `Transition`, and metadata (focal,
N, recruit mask, valid). Physics states stay transient.

Lay each fork out as extra environment **columns**, with each fork in the time
rows of its source window:

| Buffer | Leading axes |
|---|---|
| Main worker | `(W * H, E, ...)` |
| Main manager | `(W, E, ...)` |
| Fork worker | `(W * H, E * A, ...)`: fork (w, e, i) fills rows `w*H .. w*H + H - 1` of column `e*A + i` |
| Fork manager | `(W, E * A, ...)`: fork (w, e, i) is row w of column `e*A + i` |

The outer window scan already emits `(W, H, E * A, ...)`, which reshapes to
the fork worker layout exactly as the main buffer does today. Windows skipped by
the interval hold the masked placeholder rows. The fork columns keep the main
buffer's time length, which the concatenation below requires.

The fork-end `done` makes this exact. `compute_gae` cuts both its bootstrap and
its recursion at every `done`, so consecutive forks in one column are
independent episodes. The main worker buffer already has the same structure,
with a `done` at every window end. Fork terminals:

- Worker `done` is true on step H-1 and on any earlier terminating step. After
  an early ending the lane is frozen with zero reward, `done = True` and
  `active_mask = 0`, which is how `_worker_step` already treats frozen main
  lanes.
- Every fork manager row has `done = True`, so its advantage is the discounted
  window reward minus its stored context value, and its target is the window
  reward. The window reward is `sum(gamma_M ** k * team_reward_k)` over live
  fork steps.
- Last values for fork columns are zeros for both levels. The fork's final row
  is always a terminal, so GAE never reads them.

In `update_fn`, concatenate main and fork along the environment axis, at
`(W * H, E * (1 + A))` for the worker and `(W, E * (1 + A))` for the manager.
Then call `ppo_update` once per level. This reuses its GAE, per-column
advantage normalization, shuffling, minibatching, clipping and `active_mask`
handling. The legacy minibatch rule fixes the number of optimizer steps, so
minibatches grow `(1 + A)`-fold. Masked rows are still processed, so the update
cost does not shrink with the interval; only collection does.

One change to shared training code is needed, a static, default-off
`ppo_update` argument `masked_statistics`. Off is byte-identical; this variant
turns it on for both levels. When on, the active mask also governs three
things that are plain means today:

- **Advantage normalization:** mean and standard deviation per column (per
  column and agent on the worker's per-agent path) over active rows only. A
  column with fewer than two active rows is left unnormalized.
- **Explained variance:** over active rows only.
- **Scalar critic loss:** a mean over rows with any active agent. The per-agent
  worker critic is already masked this way.

The flag is required, not cosmetic. At interval k, (k - 1)/k of every fork
column is placeholder. Counted as zeros, those rows would shrink each fork
column's standard deviation by roughly √k and inflate its advantages by the
same factor. Invalid forks would also train the scalar manager critic. With
the flag, frozen, invalid and skipped rows enter no statistic. That includes
frozen main rows, a small difference from the parent's update: on `trunc`
groups a main row freezes only after a mid-window all-delivered termination.

Each fork column normalizes over its `ceil(W / k)` intervention windows (33
at `n_steps` 1056 and k = 1, 17 at k = 2).

Fork-to-main sample ratios at interval k, with every fork valid and full
length:

- worker actions: A/k;
- eligible manager actions: 1/k;
- manager critic rows: A/k.

Log the realized counts, because early endings and failed placements change
them.

**6. Implementation sequence and files**

1. **Configuration.** Add `interventions: bool = False`,
   `intervention_radius: float = 1.5` and `intervention_interval: int = 1` to
   `Model_Params` / `FeudalConfig`, wired through `run.make_feudal_config`.
   Validate:
   - `manager_credit == "team"`, `n_agents >= 2` and a positive radius;
   - that the env has `teleport_agents`;
   - `intervention_interval >= 1`, and `ceil(n_windows / intervention_interval)
     >= 2`, so every fork column has two intervention windows to normalize over
     (the same reason `make_train` already requires two windows);
   - an `intervention_interval` other than 1 raises when `interventions` is
     false, rather than silently running the parent.

   Add `conf/model/simplified_feudal_tanh_relative_input_interventions.yaml`
   (one key on its parent). Each other interval is its own one-key child, e.g.
   `..._interventions_every2`. Set the interval in a model group, never on the
   CLI: checkpoints are shape-identical across intervals, so the model group
   name is the only record of which interval trained a run. An optional
   information-matched twin is
   `simplified_feudal_tanh_local_input_interventions`.
2. **Environment hook.** `teleport_agents` plus the shared box-distance helper
   in `observation.py`, and the stub hook.
3. **Collection.** Generalize `_worker_step` (input-derived width, static
   `bootstrap`), add the interval-gated fork branch and its placeholder to
   `_window`, the manager critic context helper and `input_dims`, and the
   `Rollout.intervention` field. Keep the
   recruitment/offset logic in a small
   `simplified_feudal_mappo_jax/interventions.py`; geometry stays in the env.
4. **Update.** Concatenate along the env axis and add `masked_statistics` to
   `mappo_jax.mappo.ppo_update`.
5. **Runner and logging.** Fork diagnostics into the losses dict (section 8).
   `total_steps`, `episode_count`, `rollout_team_reward` and eval stay main-only.
   Resume needs nothing new: the fork keys derive from the restored runner key.
   Document the variant in `CLAUDE.md`.
6. **Tests and smoke run** (section 7), then a 2e5-step
   train / `checkpoint=true` resume / `evaluate=true` run on
   `mjx_2a_4o_1122_1024_gs` with a non-numeric `trial_id`.

**7. Acceptance tests**

Add `algorithms/tests/test_simplified_feudal_interventions.py` on the CPU stub,
plus MJX tests of the hook.

- **Legacy parity:** with `interventions: false` every existing test passes, and
  `ppo_update` with `masked_statistics=False` is bit-identical to the current
  code.
- **Main-path isolation:** at equal seed and parameters, the variant's main
  actions, states and rewards equal the parent's at every interval, and
  changing fork rewards or terminations changes no main advantage.
- **Interval:** forks run exactly at windows with `w % k == 0`, and every other
  window's fork rows are the fully masked placeholder. k = 1 forks at every
  window. k = 2 at W = 33 forks at 17 windows. The validator rejects k < 1,
  `ceil(W / k) < 2`, and an interval set without `interventions`.
- **Column separation:** GAE on the concatenated buffer equals GAE on main and
  fork computed separately, per column. A large reward in one fork changes no
  other fork's or main's advantages. Cover H = 1, full-length forks, early
  termination, and a time limit at the window end (no bootstrap survives in
  fork rewards).
- **Geometry and goals (MJX):** exactly N distinct recruits, excluding i; recruit
  waypoints minus recruit positions equal i's realized offset; the first fork
  worker input reflects the teleport. Cover a wall, a box face, a forced
  failure, and the waypoint-bound rejection.
- **Physics preservation (MJX):** non-recruit poses and velocities, box poses,
  `t`, `delivered` and `prev_box_goal_dist` are unchanged. The first
  post-teleport step is finite, and its shaping equals that of an untouched
  state with the same box motion.
- **Recruitment:** N lies in `1 .. A-1`, masks have exactly N entries, draws are
  reproducible, shapes do not depend on N, A = 2 always gives 1, and the critic
  context carries each fork's own N.
- **PPO provenance:** pre-update ratios are 1 for every eligible action. Recruit
  and non-focal manager actions get zero actor and entropy gradient. Every live
  fork worker action is eligible. Main and fork critic contexts differ.
- **Masking:** with `masked_statistics` on, editing frozen, invalid or
  placeholder rows changes no normalized advantage, explained variance or
  gradient. That covers both levels, including the scalar manager critic. A
  column with fewer than two active rows is left unnormalized and stays
  finite.
- **End to end:** stub collect / update / eval at A = 2 and 3, plus the MJX
  smoke run, give finite losses.

**8. Logging**

Per update:

- fork lanes attempted (`ceil(W / k) * E * A` per update), valid and failed,
  overall and per N;
- recruit distance and fork length;
- fork window return (team reward, discounted as the manager's);
- fork worker progress;
- eligible samples per source and level;
- advantage scale and value explained variance per source;
- fork simulator steps (live, and stepped including frozen lanes).

Also log collection and update time. Read placement success per N before
anything else: geometric filtering can silently change the realized N
distribution, and with it which focal offsets receive extra training.

**9. Feasibility notes and runs for the author**

- **One-window fork returns carry little task reward on sparse groups.** A box at
  spawn height needs 6.5–12.5 world units of pushing to reach the band. At the
  agents' 0.167 units/step terminal speed that is at least ~40 steps, longer
  than one 32-step window. The support audit (test 4, 2026-10-05) saw no
  selected-box delivery within 128 steps of support. Under dense reward the
  fork return still pays the box's progress inside the window. Pilot on the
  dense group; on `_sparse` groups the fork manager rows will mostly carry an
  advantage of `0 - V`.
- **The extra-data control needs a twin env group, not a CLI override.**
  At interval k the control is the parent group with about
  `n_envs: 32 * (1 + A * ceil(W/k) / W)` at the same `n_total_steps` as the
  arm. Run under the parent model, it matches the arm on three counts: number
  of updates (986 at 1e8 for k = 1), worker rows per update, and total
  simulator steps. For k = 1 at 2a/4o that is `mjx_2a_4o_1122_1024_gs_n96`
  (`n_envs: 96`). An `env.n_envs` override would write into the parent's
  results directory.

Runs, after the smoke run passes (5 seeds each, `n_total_steps` 1e8 simulator
steps):

```
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_2a_4o_1122_1024_gs \
    model=simplified_feudal_tanh_relative_input_interventions trial_id=0
uv run python train.py algorithm=simplified_feudal_mappo_jax \
    env=mjx_2a_4o_1122_1024_gs_n96 \
    model=simplified_feudal_tanh_relative_input trial_id=0
```

Compare with the existing `simplified_feudal_tanh_relative_input` and `mlp`
seeds on `mjx_2a_4o_1122_1024_gs` at equal `total_steps`, which is now equal
simulator steps. Compare with the `_n96` control as well, which also matches
the arm's updates and rows per update. Run `mjx_6a_4o_1024_gs` only after
placement success per N and transfer to unassisted evaluation hold at 2a/4o.
The parent still needs its 6a/4o seeds.
