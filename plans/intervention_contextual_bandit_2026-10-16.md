# Contextual bandit for training-time interventions

Date: 2026-10-16. Status: implemented 2026-10-19 (see "Implementation notes"
at the end); no training run.

## Objective and agreed requirements

Learn whether to create the existing training-time simulator fork at a manager
decision. Improve unassisted learning under a fixed simulator-step budget.

1. A skipped intervention runs no fork steps and consumes no collection capacity.
   Fill the freed capacity with ordinary main-rollout experience before updating.
2. Intervention selection includes exploration noise.
3. The bandit receives exactly the manager actor input for the focal agent.
   It gets no additional critic state, sampled goals, progress features, or
   training-time features. It has its own parameters and optimizer.

This plan adopts the paired fork-minus-main reward proposed in the discussion,
with a penalty for forks that earn no task reward (added 2026-10-19).
It is a proxy for useful training data, not a measurement of downstream learning
improvement. Keep unassisted evaluation as the scientific success criterion.

## Current implementation and boundaries

- `run.make_feudal_config` reduces main environment count using a fixed fork
  interval. `interventions.simulator_steps_per_update` assumes that interval.
- `trainer._window` runs one fork per main environment on eligible windows.
  `_fork` currently samples the focal agent internally and assumes a fixed width.
- `policy.decide` returns `m_actor_in` with shape `(env, agent, features)`;
  its construction already handles `global`, `relative`, and `local` modes.
- `_worker_step` records raw task reward separately from intrinsic reward and
  value-bootstrap additions. Use that raw task reward for bandit targets.
- PPO currently computes GAE on rectangular trajectory columns before updating.
- The inherited runner assumes a constant number of simulator steps per update.

Add an opt-in `intervention_scheduler: bandit` path. Keep `fixed` as the default
for existing intervention groups. Preserve their collection, random streams,
checkpoint format, and results. The working tree already contains edits to
intervention code and tests; implement on top of those edits without reverting
them. Do not use the older plan's one-fork-per-agent design.

Version 1 learns only whether to intervene. Continue sampling one focal agent
uniformly per main environment/window, and retain the existing recruitment,
placement, horizon, and `intervention_redecide` semantics. Support `none`,
`focal`, and `all`; use the pre-teleport manager input in every mode. Require
`manager_credit: team` and `intervention_interval: 1` for the adaptive path.

## Bandit model, decisions, and learning

Implement a small shared MLP in `intervention_bandit.py`: two hidden layers of
32 units, tanh activations, and one unconstrained scalar output. The output
predicts the expected utility of intervening; the utility of skipping is
defined to be zero. Share the bandit across focal agents, without adding an
agent-ID feature. Reuse any identity already present in the manager input.

For each candidate main decision:

1. Sample focal agent `i` independently of the manager's sampled action.
2. Take `x = stop_gradient(m_actor_in[env, i])` from the actual manager call.
3. Compute `q(x)`. Greedily intervene iff `q(x) > 0`; break ties by skipping.
4. With probability epsilon, replace the greedy action with a uniformly random
   skip/intervene action. Otherwise use the greedy action.

Start with constant epsilon = 0.10. Thus even a predicted-negative state has a
5% chance of intervention when resources permit, and a predicted-positive one
has a 95% chance. Zero-initialize the output layer. Do not add PPO, a critic,
discounted bandit returns, or GAE to this one-step prediction problem.

Use named `fold_in` RNG namespaces for focal selection, exploration, recruitment,
and placement. Derive keys from stable main-stream and decision identities so
compacting/reordering selected lanes does not change their random draws.
Separate these keys from the main policy RNG stream.

For every executed, valid fork, compute:

    G_main = sum(k=0..H-1, manager_step_gamma**k * main_task_reward[k])
    G_fork = sum(k=0..H-1, manager_step_gamma**k * fork_task_reward[k])
    zero_fork = 1 if fork_task_reward[k] == 0 for every k < H, else 0
    target = G_fork - G_main
             - zero_reward_penalty * zero_fork
             - cost_coef * (fork_simulator_steps / H)

Use the matching original main window, not a later rollout or critic prediction.
Both rewards exclude intrinsic rewards and all value-bootstrap additions,
including those used by `redecide: all`. After an environment terminates,
remaining rewards in that window are zero. Count actual simulator calls for
the cost term, including calls on frozen lanes if retained for compatibility.

The zero-reward penalty prices a fork whose window carries no task-reward
signal at all, which spends a budget slot on a manager row with nothing to
learn from. `zero_fork` reads the same raw per-step task reward as `G_fork`,
before discounting. A step with negative shaping is a nonzero reward, and steps
after a termination count as zero. Its meaning depends on the reward mode: on
dense groups an all-zero window is one in which no box moved, and on sparse
groups one with no delivery. It applies to executed forks only, never to failed
placements, which run no physics and use no slot.

Proposed defaults: Adam at 3e-4, gradient clipping at 0.5, one pass over each
update's executed-attempt data, minibatches of 128, Huber loss with delta 1.
Use `cost_coef: 0.0` initially: enforce scarcity through the fixed collection
budget, and expose a nonnegative cost coefficient for a named follow-up arm.
A positive coefficient is in task-return units and should be calibrated to
observed return differences, rather than silently choosing an arbitrary scale.
Use `zero_reward_penalty: 5.0`, also in task-return units, calibrated offline
(below).

**Offline evidence (2026-10-19, `intervention_headroom_probe.py`).** On the
existing fork collector's pairs (2a/4o, 5 seeds, trained policies) the
unpenalized target fails the probe's pre-registered rule: dense 2/5 seeds, sparse
1/5. Its predictable part is loss avoidance. The bottom decile is windows in
which the main rollout was about to deliver. Elsewhere the gain is about 0.
Results: `plotting/feudal_goal_analysis/intervention_headroom_20261019/`.
The penalty was then fitted the same way (the plan's network, held-out seed,
parent policies):

| group / fork mode | penalty | share forked (greedy) | gain of chosen forks | chosen forks with zero reward |
|---|---|---|---|---|
| dense, `none` | 0 / 1 / 5 | 63% / 43% / 38% | +0.09 / +0.14 / +0.22 | 19% / 10% / 2% |
| dense, `all` | 0 / 1 / 5 | 65% / 52% / 40% | +0.15 / +0.18 / +0.30 | 18% / 8% / 1% |
| sparse, `none` | 0 / 1 / 5 | 37% / 2% / 0.2% | +0.02 / -0.29 / -1.31 | 98% / 97% / 92% |

The gain is `G_fork - G_main` per window; main windows average ~8 on dense.
- **Dense:** at 5 the penalty removes nearly all zero-reward forks and roughly
  doubles the gain of the chosen forks, which is still small.
- **Sparse:** 97% of forks earn nothing, so any positive penalty makes the
  greedy bandit skip almost everywhere, and forks come only from exploration.
- **Unpenalized fork rate is unstable:** it moved between 34% and 65% across
  fits, because most contexts sit near zero gain. The penalty moves the
  threshold off that flat region.

Do not train the intervention predictor toward zero on skipped decisions:
their intervention outcome is unobserved. Log their decisions and propensities
for diagnostics. If placement fails, run no fork physics, assign utility zero
to that failed attempt (no zero-reward penalty: no fork ran), and train on that
outcome; this estimates utility of
the complete attempt, including placement reliability. Do not retry the same
candidate. Resource-denied actions get no outcome target. With no labeled
attempts, skip the optimizer entirely, including its step counter/momentum.

Freeze worker, manager, and bandit parameters throughout collection. Update
the bandit only after the full batch is collected, using its own loss. Start
without replay so old-policy outcomes do not dominate the changing target.
No gradient from bandit training enters either PPO policy.

## Collection: skipped forks must be replaced by main experience

Let `H = goal_horizon`, `W = rounded_n_steps / H`, and `E_batch` be the original
environment group's batch width. Preserve the parent's per-update budget:

    B = rounded_n_steps * E_batch simulator steps
    K = B / H collected window slots

For version 1 require even `E_batch >= 2` and use `E_main = E_batch / 2`
persistent streams within each collection call. Reset them at the start of
collection, as today, and reset terminated streams at window boundaries.
Continue those streams until the batch is full; do not reset them merely
because a proposed fork was skipped.

Collect rounds of main windows and their optional forks:

1. Advance up to `min(E_main, remaining_slots)` main streams, rotating fairly
   if only a partial round fits. Retain the pre-step states, decisions, and
   exact actor inputs needed for candidate forks.
2. Each collected main window consumes one slot and H simulator calls.
3. Sample bandit actions on those candidates. Admit at most the remaining slot
   count; if too many choose intervention, select a uniform random subset.
   Record other proposals as budget-denied, not bandit skips. Decisions made
   when no fork slot remains are forced skips and excluded from bandit fitting.
4. Attempt placements only for admitted intervention proposals. Compact valid
   placements and run only those fork lanes. Each executed fork consumes one
   slot and H simulator calls. Invalid placements consume neither.
5. Repeat with more main windows until exactly K slots have been collected.

A skipped fork is never inserted as a collected window or sampled PPO row.
All-skip therefore spends B steps on main experience. All-intervene with valid
placements spends B/2 on main and B/2 on forks. With 32 batch environments and
W=33, this means 16 main streams run 66 windows under all-skip, versus 33
windows plus 33 forks each under all-intervene. Both collect 33,792 steps for
H=32. Extra main windows can include new episodes after normal terminations.

This choice changes trajectory lengths and independent reset counts compared
with the parent's 32-stream collector. Add an adaptive-collector all-skip
control to separate that effect from intervention timing. Do not claim
trajectory-identical parity between these two collectors.

Keep the current convention that a launched H-step main/fork window counts
all H simulator calls, including calls whose transitions are masked after
early termination. Report valid transitions separately. Eliminating those
calls is a separate optimization; skipped/invalid forks must execute none.

### JAX execution strategy

Use a host-orchestrated adaptive collector around jitted manager/window,
placement, and fork primitives. Compact selected indices before launching fork
physics. Decompose each nonempty lane set into exact power-of-two widths
(e.g. 13 = 8 + 4 + 1), with cached compiled functions per width. This bounds
compilation variants without simulating padded fork lanes. Zero selected lanes
means no call to the fork rollout primitive.

Do not implement selection as `vmap(cond)` or step every lane and mask afterward:
neither is an adequate guarantee of skipped physics. Host synchronization and
smaller GPU batches may reduce throughput; benchmark them explicitly before
considering a more complex device-side collector. Keep the existing fixed
collector fully jitted and unchanged in behavior.

The inherited train loop can still call `collect_fn` as an ordinary callable.
The adaptive path reports constant B steps per update and `floor(total_budget/B)`
updates, so resume indexing and step-based schedules remain well-defined.
Report any unused final remainder smaller than B; do not exceed the run budget.

## Trajectories, GAE, and PPO integration

Separate collection capacity from temporary array storage. It is acceptable
to allocate padded storage; padding must not consume collection slots, counted
steps, statistics, or sampled PPO records.

- Retain main trajectories in temporal order for each stream, with actual
  lengths and final manager values. Preserve continuing manager GAE across
  adjacent main windows and cut it at real episode boundaries.
- At each stream's collection boundary, bootstrap from its final main state.
  If using padded GAE arrays, encode that boundary with an explicit bootstrap
  contribution and a recursion cut; never let following padding erase it or
  bootstrap a terminal twice. Workers remain terminal at every goal boundary.
- Compute fork GAE separately with the existing `redecide` semantics. A fork
  must never inherit continuation from a different fork or the main stream.
- Preserve masked normalization over temporal samples per source stream, as
  used by the current intervention path. For fork manager records, group by
  originating stream, rather than normalizing every singleton separately.
- After GAE and normalization, pack only collected window records into fixed
  PPO arrays: B worker environment rows and K manager environment rows.
  Retain their existing per-agent eligibility masks. Frozen rows from actual
  simulator calls remain masked; uncollected storage padding is removed.

Refactor `mappo_jax.mappo` narrowly into trajectory preparation and optimization
of prepared samples. Let the adaptive collector supply prepared advantages and
returns to the shared optimization helper; keep the existing `ppo_update`
entry point and math for all legacy callers. Preserve masked critic statistics,
source diagnostics, shuffle semantics, and configured optimizer-step counts.
Validate minibatch divisibility for B and K or use masked optimizer-only padding
when necessary; such padding is not rollout experience.

## Configuration, state, and checkpoints

Add fields to `Model_Params` / `FeudalConfig` and wire them through
`make_feudal_config`:

- `intervention_scheduler: fixed | bandit` (default `fixed`).
- `intervention_bandit_hidden_dim: 32`.
- `intervention_bandit_epsilon: 0.10`.
- `intervention_bandit_lr: 3e-4`.
- `intervention_bandit_cost_coef: 0.0`.
- `intervention_bandit_zero_reward_penalty: 5.0`.

Keep the initial optimizer pass count, batch size, and loss defaults in one
documented bandit configuration location. Validate epsilon in [0, 1], positive
dimensions/rate, nonnegative cost and zero-reward penalty, interventions enabled, interval 1, team
credit, and the supported batch-width constraint. Preserve the original batch
width explicitly instead of trying to recover it from a reduced worker config.

Add an optional bandit TrainState to `HierTrainState` and bandit decision/outcome
metadata to the adaptive rollout. Save parameters, optimizer state, and RNG
state for exact update-boundary resume. Add checkpoint metadata identifying
scheduler/config semantics; fail clearly on incompatible resumes. Legacy trees
must omit new keys when the feature is disabled, including all load/eval paths.

Add a named model group
`simplified_feudal_tanh_relative_input_interventions_bandit.yaml`, inheriting
the existing intervention group. Give variants for redecision modes their own
names if used. Evaluation runs ordinary environments with interventions off;
the bandit is never consulted during evaluation or view playback.

## Files and implementation order

1. `types.py`, `run.py`, and the new model group: validated configuration,
   explicit batch target, optional state/checkpoint wiring.
2. New `intervention_bandit.py`: network, epsilon-greedy selection, paired-return
   targets, masked fitting, and diagnostics. Verify this independently first.
3. `trainer.py` / `interventions.py`: factor focal selection and placement out
   of fixed-width `_fork`; expose width-independent rollout primitives and
   preserve existing fixed-mode behavior.
4. New `adaptive_collection.py`: budget ledger, exact-width dispatch, ordinary
   window refill, candidate IDs, variable main lengths, and raw reward pairing.
5. `mappo_jax/mappo.py` and the simplified trainer: preparation/optimization seam,
   source-correct GAE, compact sample assembly, and adaptive update dispatch.
6. Tests, diagnostics, CPU integration run, then a bounded MJX smoke/resume/eval
   run. Update `CLAUDE.md` and this plan with measured results after implementation.

## Logging and acceptance tests

Log candidates, policy skips, exploratory decisions, proposed interventions,
budget denials, invalid placements, executed forks, predicted utility, paired
main/fork return, gain, the share of executed forks with zero reward (and of
their paired main windows), the penalty applied, cost, and bandit loss. Use distinct denominators for
placement validity and policy intervention rate. Log main/fork simulator steps,
valid transition counts, PPO eligible samples, completed ordinary episodes,
collection/update wall time, and steps per second. Retain existing diagnostics.

Add `test_intervention_bandit.py` and adaptive collector tests using the CPU
stub, alongside the existing simplified-feudal suites:

- Exact input equality for global/relative/local modes and each focal agent;
  pre-teleport input is used even when goals are re-decided afterward.
- Epsilon 0 is greedy, epsilon 1 is uniform, seeded draws reproduce, and
  repeated draws have the expected exploration frequency.
- Synthetic positive/negative utility contexts learn different actions;
  unobserved skips do not become negative labels. No-label updates do not
  change parameters or optimizer state.
- Raw paired returns ignore intrinsic and bootstrap rewards, match the correct
  main window, and handle early termination and failed placement correctly.
- The zero-reward penalty applies exactly when every raw fork step is 0:
  - an all-zero fork is penalized;
  - a fork with a single nonzero step, including a negative one, is not;
  - bootstrap and intrinsic additions never change the indicator;
  - a fork that terminates after a nonzero step is not penalized;
  - a failed placement is never penalized;
  - a zero penalty reproduces the unpenalized target bit for bit.
- Forced all-skip invokes no placement/fork stepping and collects B main steps;
  forced all-intervene with valid placement collects B/2 main and B/2 fork steps.
  Mixed, all-invalid, and near-full cases finish exactly at B with no fake rows.
- Test invocation/accounting with an instrumented host-dispatched stub, not
  Python trace-time counters inside JIT. Verify zero-sized dispatch is omitted
  and each nonempty chunk's executed width equals its admitted lane count.
- Check GAE/targets against independently computed ragged trajectories,
  including unequal main lengths, collection-boundary bootstraps, early ends,
  and all redecision modes. Test no cross-source advantage propagation.
- Test prepared-batch PPO against the original path on identical data, including
  masked scalar critic rows and worker per-agent masks.
- Check fixed-mode and disabled-mode regression parity, and exact adaptive
  save/resume behavior for subsequent decisions, samples, and updates.

Run the relevant existing suites plus new tests. Then run at most 2e5 simulator
steps on `mjx_2a_4o_1122_1024_gs` with a nonnumeric trial ID, including checkpoint
resume and evaluation. Measure actual collection throughput, memory, and compile
variants; do not infer wall-time savings from reduced fork counts alone.

## Experiment to decide whether the scheduler helps

After correctness and smoke validation, compare multiple seeds at equal total
simulator budgets: the ordinary parent, adaptive all-skip, adaptive random
selection, and the learned bandit. Match the random arm's realized fork spending
to the learned arm as closely as practical, including its training-stage rate
profile; log residual differences. Include all hyperparameter exploration and
bandit-generated forks in training budgets. Report evaluation costs separately
and hold the evaluation protocol fixed across arms.

Compare unassisted evaluation return versus simulator steps and wall time,
completed main episodes, and intervention spending. Positive paired gain alone
does not establish success. If the bandit improves assisted return but not
unassisted learning, reconsider the reward proxy or intervention semantics
before increasing model complexity.

## Implementation notes (2026-10-19)

Code: `algorithms/simplified_feudal_mappo_jax/intervention_bandit.py` (network,
schedulers, target, fit), `adaptive_collection.py` (host collector, buffers,
batches), `trainer.py` (factored `_decide_window` / `_run_window` /
`_fork_place` / `_fork_rollout`, bandit state, adaptive update), `run.py`
(config, checkpoints, banner), `interventions.py` (validation, budget), `types.py`;
the shared `mappo_jax/mappo.py` split of `ppo_update` into `prepare_advantages`
/ `flatten_samples` / `optimize_samples`. Tests:
`algorithms/tests/test_intervention_bandit.py`.

Where the code goes beyond or settles the text:

- **`intervention_scheduler: random`** with `intervention_random_rate` was added.
  The experiment's "adaptive all-skip" and "adaptive random selection" arms had
  no config surface under `fixed | bandit`. Rate 0 is the all-skip control.
- **The zero-reward penalty** is `intervention_bandit_zero_reward_penalty`
  (default 5; 0 turns it off). The zero-penalty option is its own arm,
  `..._bandit_no_zero_penalty`.
- **Model groups:** `..._interventions_bandit`, `..._bandit_no_zero_penalty`,
  `..._adaptive_skip` (random, rate 0) and `..._adaptive_random` (rate 0.5, a
  placeholder to match to the bandit's realized `scheduler_proposal_rate`).
  Redecide variants are not shipped; each would be a one-key child group.
- **A partial round** can only be the final one: it occurs only when fewer
  slots remain than streams, so it fills the budget and all its proposals are
  forced skips. Streams are chosen by a random permutation from the
  collection key.
- **Packing:** "pack only collected window records" is implemented as GAE and
  normalization on the padded `(2 * n_windows, streams)` buffers. Exactly K
  manager rows and `K * H` worker rows are then gathered (`jnp.nonzero(...,
  size=...)`) into the optimizer. The minibatch arithmetic therefore equals the
  fixed fork arm's.
- **Testing hooks:**
  - Labels carry `(round, stream, focal, placed)`, so a test can match every
    label to its main decision.
  - `adaptive_collection.DISPATCH_HOOK` wraps the host-dispatched calls, for the
    instrumented dispatch tests.
- **Validation:** scheduler knobs that cannot act under the chosen scheduler
  raise (for example a random rate under `bandit`).
- **Checkpoints:** adaptive train checkpoints carry `scheduler_meta`, and a
  resume under other scheduler settings, or across fixed and adaptive, raises.
  Fixed checkpoints are byte-identical to before.

Verification:

- **Before/after snapshot** on the CPU stub: 3,611 arrays, 0.0 maximum
  difference. It covers every legacy `ppo_update` mode, plus collect/update/
  eval of the parent, every fork mode at intervals 1 and 2, and both credits.
- **New tests:** 56 in `test_intervention_bandit.py`, all passing.
- **Existing suites:** 504 pass across the simplified-feudal, interventions,
  SMAX, global-state, D++ probe, FeUdal seam and visualization suites.
  `test_mjx_sector_counts.py` has 4 failures that also fail on a clean `HEAD`
  (the sector counts were switched off in `b6bf344`).
- **MJX smoke** (`mjx_2a_4o_1122_1024_gs`, `trial_id=smoke_bandit`): trained to
  2e5, resumed with `checkpoint=true` to 3e5, and evaluated with
  `evaluate=true`. Every update collected exactly 33,792 simulator steps (main
  plus fork).
  - **Warm collection per update** (median of updates 2–7): parent ~0.76 s,
    fixed forks ~1.3 s, `_adaptive_skip` ~1.3 s, bandit ~1.8 s,
    `_adaptive_random` (0.5) ~2.3 s. The update itself takes 0.06 s, so at 1e8
    a run spends roughly 1–2 h collecting, against ~0.6 h for the parent.
  - **Compiles:** the first update compiles in ~70 s (fixed: 20 s), and each new
    lane width adds ~20 s later.
  - **Peak GPU memory:** ~630 MiB above idle.

**Early-training behaviour to expect (measured in the smoke).** With an
untrained policy no box moves, so 92–97% of forks earn exactly 0 reward. With
the zero-reward penalty the target is then about −4.8 for nearly every fork,
and the bandit's predictions drift negative from the first updates: −0.011
after 5. It therefore forks only through exploration (about 5% of decisions)
until the policy starts moving boxes. The bandit also gets one optimizer step
per update from about 50 labels. Compare against `..._bandit_no_zero_penalty`
before attributing anything to the bandit.

