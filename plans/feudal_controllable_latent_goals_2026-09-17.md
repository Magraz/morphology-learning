# Anchoring Feudal latent goals to controllable outcomes

Date: 17 September 2026. **Status: PROPOSED; not implemented or validated.**
Revised 17 September 2026 (revision 2) against the working-tree implementation
and the measurements recorded in CLAUDE.md. No training was run for either
version; every number cited below is an existing measurement, attributed inline.

This plan retains learned directional goals and gives the progress encoder an
explicit learning signal from interventions on worker commands. The proposed
multiagent contrastive objective is an experiment, not an established extension
of FuN or a guarantee of useful coordination. Writing this plan does not launch
training or change the current implementation.

## 0. What revision 2 changed, and why

Revision 1 was structurally sound: the identifiability argument, the control set
and the update-ordering discipline all survive. Seven things changed.

1. **The pilot arm was the known-degenerate one.** Revision 1's pilot
   (`mjx_12a_3o_trunc_1024`, `manager_latent: local`, FiLM, `intrinsic_coef 0.1`)
   is exactly `feudal_film_n01_local`, the arm CLAUDE.md records as collapsed:
   `goal_direction_count` 1.45 against a random-direction baseline of 8.93, a
   frozen constant vector recovering 95% of its return, and `local` measured
   **-64.6** return against `centralized` on the unconfounded `alpha=0` cut.
   §6 now runs `local_private` as the primary and demotes `local` to a
   *repair* arm whose collapse is a prediction the objective is supposed to
   prevent.
2. **The auxiliary data budget was ~2 orders of magnitude too small** for the
   thing it has to fit, while the branches it was rationing are nearly free.
   §4/§6 rescale it and name the real binding constraint (device memory, not
   simulator steps).
3. **The main leakage channel was unnamed.** `own_velocity` is observation dims
   `[0:2]` and is close to an integral of the agent's own applied force, so a
   held command is identifiable from the agent's own kinematics with no effect
   on the world at all. Existing measurements say this is the *likely* outcome,
   not a remote one. §3 makes the channel-ablation report a required output
   rather than a follow-up.
4. **The implementation route was more invasive than necessary.** The encoder is
   already isolated by `FeudalManager.__call__(..., latent_only=True)`. §5 now
   proposes a parameter-label partition over the existing manager tree instead
   of a module extraction, which keeps the checkpoint tree, the five-way
   `_dims_from_checkpoint` discrimination and all four `run.py` msgpack sites
   unchanged.
5. **The measurement vocabulary was parallel to the repo's.** §7 now routes the
   behavioural claim through the existing three-condition
   `goal_dependence_probe` acceptance test and keeps the new metrics as the
   mechanism layer beneath it.
6. **The causal chain from `L_anchor` to behaviour was implicit.** §1 states it,
   and states the consequence: `intrinsic_coef > 0` is load-bearing for the
   hypothesis, not an incidental setting, so `alpha=0` is a mechanism ablation
   with a pre-registered prediction rather than a neutral control.
7. **Step 1 is answerable today on trained checkpoints**, with no new training
   and no objective change. §8 elevates it to a decision gate with
   pre-registered readings, because one outcome of it retires the rest of the
   plan.

## 1. Question and hypothesis

Can a learned goal space become a reliable interface for control when its
directions are trained to distinguish the outcomes of different commands issued
from the same starting situation?

The desired relationship is:

```text
latent command -> worker behavior -> repeatable environmental effect
```

The manager must subsequently learn which effects improve team return. Reliable
goal following and task usefulness are separate requirements.

### The defect this targets

The manager currently owns **both** arguments of its own objective. It picks the
measuring stick `s` and the target `g`, and `manager.py` already documents the
degenerate fixed point that follows. The intrinsic reward inherits the same
problem from the other side: `r^I` scores `d_cos(s^+ - s, g)`, where both terms
are manager outputs, so worker and manager can raise it jointly through the
environment without the task moving. That is the measured outcome on the
`local` arms (CLAUDE.md, 2026-09-14): `d_cos_mean` rises, `goal_direction_count`
falls to 1.45, one frozen vector recovers 95% of the return.

`L_anchor` introduces an **exogenous label**: a command `u` sampled independently
of the state, which the encoder cannot influence. An encoder cannot lower the
loss by rotating its own yardstick, because the yardstick is now scored against
a vector it did not produce. This is the one structural property that separates
the proposal from every diagnostic already in the repository, all of which are
necessary-only collapse detectors.

### The causal chain, stated so each link is separately falsifiable

```text
(a) anchored phi  =>  latent directions correspond to distinguishable outcomes
(b)               =>  r^I stops being self-referential and becomes a reward for
                      ACHIEVING the commanded outcome
(c)               =>  goal-following becomes learnable by the worker
(d)               =>  the manager's per-agent assignment can start to matter
```

Link (b) is where `L_anchor` reaches behaviour. **Nothing in the worker's
objective asks it to follow a goal when `intrinsic_coef == 0`**; the anchored
encoder would then only change the manager's cosine target and the intrinsic
critic that nothing consumes. So the hypothesis requires `alpha > 0`, and the
`alpha = 0` anchored arm is an ablation with the pre-registered prediction that
it moves held-out identification (a) and leaves behaviour (c)/(d) roughly where
the unanchored `alpha = 0` arm sits. If that prediction fails in either
direction it is informative, which is why it is worth running.

Replacing the latent space with physical coordinates would provide a diagnostic
reference, but would not itself train useful latent goals. This plan instead
keeps the learned progress representation inside both the auxiliary objective
and the existing manager/worker objectives.

## 2. Starting point in the repository

The current implementation already contains the FuN-style manager loss:

```text
L_manager = -mean(A_external * cos(stopgrad(z[t+c] - z[t]), g[t]))
```

See [manager_update](../algorithms/feudal_mappo_jax/mappo.py) and
[transition_cosine](../algorithms/feudal_mappo_jax/manager.py). The manager's
extrinsic advantage connects goals to return, but the loss assumes a useful
relationship between commands and achieved transitions.

In the local variants, `f_enc` (shared, per agent) and `f_Mspace` /
`f_Mspace_agent` compute the per-agent progress representation from local
observations. Those features also feed the manager's goal-generation core, which
currently supplies their learning signal even when the displacement target is
detached. Input locality does not establish physical controllability: teammates
can change an agent's observations, and `obs_i` is egocentric but not
proprioceptive (density sensors, `neighbor_fraction` and lidar all respond to
teammates inside `sector_sensor_radius`).

The worker uses stored pooled goals and separate extrinsic/intrinsic advantage
streams. Preserve the implemented successor-state intrinsic reward timing,
including pre-reset terminal successors.

### Measurements that constrain the design

These are recorded in CLAUDE.md and in the referenced notes. They are the reason
several revision-1 choices changed.

| Fact | Source | Consequence for this plan |
|---|---|---|
| `worker_goal_action_delta` is **0.91-1.09** on trained `feudal_film` arms: swapping in an unrelated goal moves the mean action by ~100% of its own RMS | CLAUDE.md, FiLM goal-influence metrics | Commands already produce large *action* differences. The increment `L_anchor` must supply is **systematic alignment**, not mere distinguishability. Do not report "commands are distinguishable" as progress. |
| `local` scores **-64.6** return vs `centralized` on the `intrinsic_coef=0` cut (CI [-97.0, -27.1]); its goal rows collapse to a participation ratio of 1.46-1.67 | CLAUDE.md, 2026-09-14 | Do not make plain `local` the primary pilot latent. |
| `local_private` restores row diversity (`s` 8.72, `g` 9.24 against a 8.93 random baseline) at diag share exactly 1.0 | CLAUDE.md smoke measurement | Primary pilot latent, with the caveat that its numbers are from a 200k-step smoke run. |
| On `mjx_12a_3o_trunc_1024` the best arm is `feudal_film_zerogoal_dilated` at **293.7**, then `mlp` 272.3, then `feudal_film_zerogoal` 248.9; the best goal-*using* arm is 203.7 | CLAUDE.md, 2026-09-14 | The goal-free control currently **wins** on the pilot env. Name it as the number to beat; a within-feudal improvement is not a result. |
| Acceptance is three conditions (`gap_zeroed`, `gap_constant`, `gap_permuted` all > 0 with paired CIs), and the decisive test is between-arm return against `feudal_zerogoal`/`feudal_film_zerogoal` | `conf/model/feudal_film.yaml` | Reuse this, do not invent a parallel criterion. |
| The `r^I` timing fix rotates the actor gradient by 0.08/1.2/3.6 deg at alpha 0.01/0.1/0.5 — ~13% of what the intrinsic term itself does | `intrinsic_timing_probe.py`, 2026-09-14 | Timing is settled and is not a confound for this study. Preserve it; do not relitigate it. |

Background:

- [Goal/reward diagnosis](feudal_goal_reward_diagnosis_2026-09-09.md).
- [Intrinsic reward timing fix](feudal_intrinsic_reward_timing_fix_2026-09-16.md).
- [Current method and baseline assessment](feudal_novelty_and_baselines_2026-09-14.md).

## 3. Proposed representation objective

Use the existing local progress encoder:

```text
z_i = f_phi(obs_i)      # exactly FeudalManager.__call__(..., latent_only=True)
```

The encoder receives environment observations only. Goals, actions, branch
indices, random keys, and policy hidden states carrying the command are not
encoder inputs. Adding those inputs would permit goal identification without an
observable environmental consequence.

For one starting simulator state and one selected agent `i`, sample `K` distinct
unit commands `u[0:K]`. Run one branch per command, holding the selected command
for `H` steps. Define:

```text
delta_z[k] = f_phi(obs_i_end[k]) - f_phi(obs_i_start)
score[k, l] = cosine(delta_z[k], stopgrad(u[l])) / temperature
L_anchor = mean_k(-score[k, k] + logsumexp_l(score[k, l]))
```

Average losses over valid start-state/agent groups. Candidate goals in each
denominator belong to the same starting state and teammate-command context.
Use a stable log-softmax implementation and the existing zero-norm convention.
Log near-zero displacements rather than silently filtering them out.

This loss trains the actual cosine geometry used by FuN. There is no additional
flexible decoder in the primary experiment. A decoder that can recover physical
information from the latent does not establish that latent directions are
usable commands.

Unlike the manager loss, `L_anchor` differentiates through both encoder calls.
Commands and collected environment observations are data. No gradient passes
through the simulator or worker policy during this encoder update.

### The condition the label distribution must satisfy

The `log(K)` floor argument requires only that the joint law of `u[0:K]` is
**independent of the starting state and of the group**, and that the correct
index is uniform within the group. Isotropic unit samples satisfy this, and are
the default. A fixed *empirical* marginal — for example a distribution fit once
to the manager's recent goals and then held constant for a training phase — also
satisfies it, and is worth offering as an option: isotropic sampling fits the
geometry over the whole sphere while the manager occupies a small part of it
(measured `goal_concentration` ranges from 0.129 to 0.78 across arms), so the
anchored geometry may be fit largely where the manager never goes. What is **not**
permitted is resampling the marginal per state or per group, which reintroduces
exactly the shortcut the matched start state exists to remove. Record which
sampler produced each run.

### The leakage channel that decides whether this measures anything

The observation is 40-dimensional and its first two entries are `own_velocity`.
Under joint damping, an agent's velocity over `H` steps is close to an integral
of its own applied force, and the FiLM-modulated action mean already moves ~100%
of its RMS under a different goal (§2). **So an encoder can very plausibly
achieve near-perfect held-out command identification by reading two observation
dimensions, while nothing in the world changes.** That is the degenerate outcome
of this objective and it is the same failure documented for state-marginal skill
discovery, where a discriminator latches onto trivially distinguishable state
components.

This is not a footnote; it is the crux. Two consequences:

1. **Channel ablations are a required output, reported with every anchor
   metric,** not a follow-up. Compute held-out identification from `f_phi`
   restricted to (a) `own_velocity` only `[0:2]`, (b) everything except
   `own_velocity` `[2:40]`, (c) exteroception only (density sectors `[2:18]` +
   `nearest_box_vec` `[21:23]` + lidar `[24:40]`), and (d) the full vector. The
   quantity of interest is (b)/(c), not (d). Report the *increment* of (d) over
   (a): if it is ~0, the objective is fitting the agent's own kinematics.
2. **Decide in advance what to do about it, and record the decision as a method
   change.** The options, in increasing invasiveness: report and accept (the
   agent moving *is* a real effect, just the weakest one); restrict the encoder's
   input to the exteroceptive channels, which makes "own progress" mean "change
   in what the agent perceives of the world" and is defensible but is a
   different method; or keep the full input and add a magnitude criterion, which
   §7 already warns changes the method. Do not pick after seeing the result.

### What this objective establishes and what it leaves open

With identical outcomes in every branch, the encoder cannot identify a uniformly
chosen command. Across the complete candidate group the minimum average loss is
`log(K)` and classification accuracy is at chance, `1/K`. An arbitrary encoder
can have a larger loss; identical outcomes do not force uniform logits.

When goals produce different observations, the loss encourages those differences
to align with the commands. Starting from matched states removes the shortcut of
predicting the manager's command from the starting situation alone.

**Useful side property: the label pins the frame.** Because `u` is sampled in the
goal space's standard basis and used directly as the target, any global rotation
of `f_phi`'s output breaks the alignment. `L_anchor` therefore fixes the
representation's frame absolutely rather than up to a rotation. This bounds the
between-version drift §5 is otherwise exposed to: the manager's goal head does
not chase a rotating frame, only a moving conditional mean. Measure the drift
anyway; the argument gives a reason to expect it small, not a guarantee.

The loss does not guarantee full-rank representations, large physical effects,
predictable outcomes under new teammate behavior, or task usefulness. Tiny
repeatable movements may suffice for classification. Held-out behavioral
measurements are therefore required alongside representation metrics.

### Reporting: use the continuous statistic, not only accuracy

At `K = 4` in a 32-dimensional goal space the negatives are near-orthogonal to
the positive, so top-1 accuracy saturates on very little alignment and is a
coarse statistic. Report `cos(delta_z[k], u[k])` — an effect size on a fixed
scale — and the correct-minus-best-wrong margin, alongside accuracy against
`1/K`. Use a larger `K` (8-16) for the held-out *metric* even if training uses a
smaller one; extra candidates in the denominator cost branches linearly, and
§4 shows branches are not the binding constraint.

## 4. Intervention collector

Implement a bounded auxiliary collector before changing training objectives.

### Snapshots are cheap; say so and stop rationing them

Revision 1 warned against "storing an entire MJX physics state at every ordinary
rollout step". That is correct about `mjx.Data`, but the repository already has
the compact form:
[`SyncMacroMJX.snapshot` / `state_from_snapshot`](../environments/mjx_suite/macro_wrapper.py)
stores `qpos`/`qvel` plus the `EnvState` scalars and rebuilds via
`mjx.make_data(...).replace(...)` + `mjx.forward`. At 12 agents / 3 objects that
is `nq = nv = 33` floats per env. Snapshotting **every** step of a production
rollout is `1048 x 32 x ~70` floats, under 10 MB. So:

- store the compact snapshot at every step and sample groups uniformly from the
  rollout, rather than building machinery to pre-select "a small number of
  on-policy snapshots";
- the teammates' held commands come from `Transition.pooled_goal[t]`, which is
  already stored, so no extra recording is needed for them.

⚠ **Reconstruction is not the original `mjx.Data`.** Warm-start and contact
state are recomputed by `mjx.forward`, so a branch is not a bit-exact
continuation of the sampled trajectory. All `K` branches share one
reconstruction, so the *comparison* is exact and `L_anchor` is unaffected. Any
claim that a trial measures "what would have happened on-policy" is not, and
should not be made.

### Protocol

1. Sample groups uniformly from the stored snapshots, with an active agent for
   each. Reconstruct once per group and share that state across the group's
   branches.
2. Copy the initial state into `K` branches. Freeze all network parameters for
   the entire collection batch.
3. Replace the selected agent's effective worker command with `u[k]` throughout
   its branch. Set each teammate's command to the same prescribed value across
   all branches, initially its command at the snapshot.
4. Use the same per-time/per-agent action-noise streams in corresponding
   branches. Candidate-command sampling uses a separate random stream. Common
   noise reduces variance; repeat with independent streams to assess reliability.
5. Let worker policies react to their branch's observations. This measures the
   total effect of changing a command under fixed teammate policies and command
   schedules, including teammate reactions. It is not an effect with teammates'
   primitive action sequences held fixed.
6. Record starting and ending observations, candidate commands, selected agent,
   branch validity, and physical outcome summaries. Store per-step transitions
   and behavior log-probabilities only when worker training uses these trials.

### Two intervention designs; pick deliberately, and note `local_private`

**(A) Single-agent (the clean isolate).** One selected agent per group, `K`
branches. Agent `i`'s outcome is confounded only by teammates reacting to `i`.
Cost: `K x H` steps per group.

**(B) All-agent (cheaper by `N`, and closer to deployment).** Intervene on every
agent simultaneously with independently sampled `u_i`, and form one contrastive
group per agent using its own `u_i[0:K]`. The `log(K)` argument is unchanged —
`u_i` is still independent of the start state — but agent `i`'s outcome now also
varies with teammates' randomized commands, which adds variance and measures
controllability under a *randomized* team context. §7 asks for exactly that
robustness check anyway. Cost: `K x H` steps for `N` groups.

Design (B) matters specifically for `manager_latent: local_private`, where the
per-agent projections `f_Mspace_agent` are only trained for agents that were
selected. Under (A) each `W_i` sees `1/N` of the data; under (B) all of it. Run
(A) for the primary mechanism claim and (B) for budget and for the randomized-
context robustness measurement, and report which produced which number.

**Goal pooling:** intervene on the delivered command, not just one newly written
raw goal in the history ring. With the shipped `normalize_pooled_goal: True` the
worker L2-normalizes `w_t`, so magnitude is discarded and setting `w_i = u[k]`
matches magnitudes automatically; under `normalize_pooled_goal: False` the
override must reproduce the pooled magnitude explicitly. Record the exact
`pooled_goal` supplied to the actor either way. For worker training, begin a new
intrinsic-reward history at the branch start; do not invent earlier state/goal
pairs to populate the history window.

**Boundaries:** never reset inside a trial and use the reset observation as an
endpoint. Initially choose snapshots with at least `H` steps before the time
limit. Use complete groups for the fixed-horizon encoder objective; if any
branch terminates early, omit that group from this objective and report its
outcomes and censoring separately. Goal-dependent termination can bias the
complete-case sample, so log exclusion rates by candidate and context. A later
terminal-aware objective requires its own specification.
⚠ On the `trunc` env groups the walls are inert and episodes run the full 1024
steps, so censoring is near-zero and this machinery is nearly inert — implement
the logging, defer any terminal-aware objective. It becomes live on the
boundary-terminating baseline, whose episodes are ~43 steps.

The existing pure simulator and
[difference-reward branching](../environments/mjx_suite/multi_box_push_mjx.py)
provide useful mechanics, but their `active` intervention is different:
excluding an agent changes both its force and its contribution to box coupling.
Keep agents active for goal interventions. Reuse state-copying and paired-noise
patterns, not the agent-removal semantics.

**Memory, not steps, is the constraint.** Branches vmap, and the repository has
already hit the ceiling: the windowed-difference-reward study needed a `--chunk`
flag because `n_rollouts x n_agents` concurrent MJX simulations exhausted a
16 GB device. Size the auxiliary batch by concurrent branch count and chunk over
groups, reusing that idiom.

## 5. Parameter ownership and update order

### Own the encoder by parameter label, not by module extraction

The encoder is already isolated at the call level: `FeudalManager.__call__(...,
latent_only=True)` returns `s` and skips the core and goal head entirely, and it
is exact rather than an approximation because `s` is feedforward and upstream of
the core in every latent variant. The parameters it reads are exactly
`f_enc_0`, `f_enc_1`, and `f_Mspace` (shared latents) or `f_Mspace_agent_*`
(private latents).

So prefer a **label partition over the existing manager parameter tree** to a
module extraction:

- an `anchor_encoder: bool` model parameter that (a) wraps `s` in
  `stop_gradient` before `core_in` and (b) routes the two label groups to two
  `optax.masked` transforms;
- `optax.masked` skips masked-out leaves entirely, which is required: handing
  Adam exactly-zero gradients on the encoder leaves is **not** a no-op, since it
  still advances the step count and decays the moments.

What this buys, all of which a module extraction would spend:

- the checkpoint parameter tree is **unchanged**, so existing `feudal_film*`
  checkpoints stay loadable and comparable;
- `goal_dependence_probe._dims_from_checkpoint` keeps working. CLAUDE.md records
  that no single key separates the five existing latents and that its rule has
  already had to be split twice; a sixth variant with a new tree is a good way
  to break the tool you need for the acceptance test;
- the four/five `run.py` msgpack sites that must move in lockstep do not move.
  Keeping one optimizer-state pytree for the manager subtree means the auxiliary
  optimizer adds **no** new save site.

If a separate module is preferred for clarity, that is a defensible choice — but
then budget the migration explicitly: a new `from_bytes` target tree, a
`_dims_from_checkpoint` discriminator, an explicit legacy load path, and a
re-verification that `view`/probe/eval entry points build the same tree.

### Invariants that change in the anchored arm, and the tests that will fail

Under anchoring the manager loss must not reach the encoder, which **inverts two
assertions that are currently load-bearing**:

- `manager.py` self-check **[8]** asserts `|dL/df_Mspace| > 0` under the detach,
  with the message "the core is not consuming s". In the anchored arm this must
  be exactly 0.
- self-check **[10](d)** asserts `f_enc_*` / `f_Mspace` all receive gradient.
  Same inversion.

Make both arm-conditional rather than relaxing them; the unanchored arm must
keep the original assertion, because it is what pins the "core consumes `s`"
topology.

Note the consequence: in the anchored arm the FuN detach rule is satisfied
*structurally* (the only path from the manager loss to `f_Mspace` is severed),
so the "core must consume `s`" topology stops being load-bearing there. State
this in the code comment, or the next person to see self-check [8] fail will
rewire the core back to `z` and silently starve the unanchored arm.

| Parameters | Anchored arm's update | Data |
|---|---|---|
| Progress encoder `phi` (`f_enc_*`, `f_Mspace*`) | `L_anchor` only | Randomized intervention trials |
| Worker actor | Existing PPO with extrinsic and intrinsic advantages | Ordinary rollouts; later, correctly collected trial rollouts |
| Manager goal network (core, goal head) | Extrinsic-advantage transition cosine, on `stop_gradient(s)` | Manager-generated ordinary rollouts only |
| Critics | Their corresponding return targets | Matching collection regime |

No parameter should have two independent Adam owners. In the anchored arm,
manager and worker updates must leave encoder parameters and its optimizer state
unchanged. The manager's progress target stays detached.

One outer iteration should use the following order:

1. Freeze encoder version `phi_v` and collect ordinary rollouts and any
   scheduled auxiliary trials under a recorded worker snapshot.
2. Compute intrinsic rewards and advantages using `phi_v`. Keep stored goals,
   rewards, old log-probabilities, and bootstrap values fixed through PPO.
3. Complete worker updates and the ordinary manager update while `phi_v` remains
   fixed. `manager_update` re-derives `(goal, s)` from the stored global states
   and must reproduce the goals it issued; with `phi` fixed across the iteration
   that property is preserved unchanged. Externally sampled goals are excluded
   from manager training.
4. Fit the encoder on the collected intervention observations, producing
   `phi_v+1`. Collect fresh policy data before using that new representation.

This prevents encoder fitting from invalidating the rollout being optimized.
It does not eliminate changes in the reward definition and manager inputs
between iterations: `r^I` is defined *in* `phi`, so every encoder version
redefines the worker's intrinsic reward. Watch `adv_int_std_raw` and
`intrinsic_explained_variance` across encoder versions — the latter currently
reads 0.96-0.99 and a drop is the signal that `V^I` is chasing a moving target.
Log encoder displacement/direction drift on a fixed validation set, and
behaviour before and after encoder updates. Start with few encoder passes; slow
updates or a lagged encoder are follow-up ablations if measured drift is
disruptive.

### Worker learning on randomized commands

Fitting an encoder alone cannot create command effects when the worker ignores
all goals. Alternate representation fitting with goal-conditioned worker
learning. Implement this in two stages:

- First use auxiliary trials only for encoder fitting and measurement; ordinary
  positive-intrinsic rollouts continue training the worker. This isolates the
  new representation signal with minimal PPO changes.
- Then add worker learning on trial trajectories if broader command practice is
  needed. This is a separate experimental factor, shared by its controls.

For the second stage, store the exact issued goals, primitive actions, old
log-probabilities, rewards, and masks. Use the same worker snapshot for collection
and the start of PPO optimization. Specify the trial return objective explicitly:
either a finite `H`-step trial with no continuation value, or a continuation
under the same fixed-command regime with an appropriate goal-conditioned value
estimate. The current manager-policy critic is not automatically a valid
bootstrap for that different regime. Never concatenate branches for GAE or mix
their boundaries with ordinary trajectories. Validate this path before using
trial data for PPO; representation fitting does not require it.

## 6. Pilot settings and controls

Initial scope: continuous MJX box pushing, FiLM worker, MLP manager, on
`mjx_12a_3o_trunc_1024`. Assess the partition task after the mechanism check.
Preserve the chosen configuration's extrinsic reward definition and
execution-time information. Serialize the resolved configuration because
existing model groups and local changes can override defaults.

**Primary latent is `local_private`, not `local`.** Revision 1's choice is the
measured-degenerate arm (§0, §2). `local_private` is the current best-structured
local latent: exact obs-locality (diag share 1.0), and restored row diversity.
Its diversity numbers come from a 200k-step smoke run, and `local`'s collapse
happened by ~10M steps, so `goal_direction_count` must be read across the whole
run, not at the start.

Keep `local` as a **secondary repair arm**: it is the one configuration where a
collapse has already been measured, so "does anchoring prevent a collapse we
know occurs?" is a cheap, sharp, pre-registered question. Predict in advance that
anchoring raises its `goal_direction_count` off 1.45; a failure there is
informative about `L_anchor` itself rather than about the task.

Suggested initial settings, subject to collector profiling:

- `H = goal_horizon` (currently 10), temperature `0.1`. `K = 8` for training,
  `K = 16` for the held-out metric.
- **Auxiliary batch: 128-256 groups per scheduled batch, chunked to fit device
  memory, every 10 ordinary updates.** Revision 1 proposed 8 groups x `K=4`
  = 32 examples per batch; over a ~2980-update run at `n_steps=1048`,
  `n_envs=32` that is ~300 batches and ~10^4 examples total, to fit a 2-layer
  256-wide encoder plus projection whose output also defines the intrinsic
  reward. That is far too little. At 256 groups x `K=8` x `H=10` the branch cost
  is 20,480 steps per batch against 33,536 steps per ordinary update, i.e.
  **~6% overhead** at a schedule of one batch per 10 updates, and ~6x10^5
  examples over the run. Chunk over groups (for example 32 groups x 8 branches
  = 256 concurrent simulations per chunk) and profile the first pass before
  committing.
- Count and report simulation steps and wall time separately for ordinary and
  auxiliary collection. Expect wall-clock to be dominated by the second scan's
  compilation and the per-group `mjx.forward` reconstruction, not by the steps.
- Several encoder gradient passes per scheduled batch over a small replay of
  recent trials, with the existing manager learning rate as an initial value,
  not a tuned choice. Revision 1's "one pass per batch" is under-powered at any
  batch size worth collecting.
- `intrinsic_coef` at the existing `0.1` with the matched anneal, because §1
  makes it load-bearing. `alpha = 0` is a mechanism ablation with a
  pre-registered prediction, not a neutral control.
- Keep current goal width and manager capacity matched across the primary arms.
  If a smaller progress space is investigated, decouple its width from manager
  information capacity and compare all relevant arms at that width.

| Arm | Encoder learning | Auxiliary worker practice | Purpose |
|---|---|---|---|
| Existing reference | Existing manager-derived gradients | None | Current method under the same reward/timing configuration |
| Matched manager-trained encoder | Existing manager-derived gradients | Same as anchored arm | Control for extra interactions and command exposure |
| Anchored encoder | Correct-label `L_anchor` only | Chosen stage, recorded explicitly | Proposed method |
| Anchored, `alpha = 0` | `L_anchor` only | Same | Mechanism ablation for link (b) of §1 |
| Frozen encoder | None; matched initialization | Same as anchored arm | Separate learned anchoring from random-feature progress |
| Shuffled-label encoder | Same auxiliary update, labels freshly shuffled within groups | Same as anchored arm | Check that command/outcome pairing matters |
| **Goal-free reference** | n/a | n/a | `feudal_film_zerogoal` on the same env group and seeds |

The shuffled control may actively damage a representation; beating it is not
sufficient evidence. The goal-free reference is not optional and is not a
formality: on `mjx_12a_3o_trunc_1024` the goal-free arms currently **win**
(`feudal_film_zerogoal_dilated` 293.7, `mlp` 272.3, `feudal_film_zerogoal` 248.9
against 203.7 for the best goal-using arm). That is the number to beat.

Evaluate all methods under their normal execution policy. Use three matched
training seeds for the initial pilot, then increase replication for a
performance claim. Compare at equal total training simulator interactions,
including branches, and report ordinary and auxiliary counts separately. Keep
evaluation counts matched — note `conf/algorithm/feudal_mappo_jax.yaml`
currently sets `n_eval_episodes: 16` while `mappo_jax` sets 32, so any
feudal-vs-flat comparison also compares two eval noise floors until that is
aligned. Count evaluation cost separately. Do not choose a favorable checkpoint
after observing the test results.

## 7. Measurements and decisions

Split fitting and evaluation by starting trajectories/episodes, with independent
evaluation noise streams. Branches from one starting state stay in one split.
If repeated online evaluation influences design decisions, retain a separate
final test set. Report uncertainty across independent training seeds, with
paired comparisons across starting-state groups inside each seed.

### Behavioural acceptance is the existing probe

Route the behavioural claim through
[`goal_dependence_probe.py`](../algorithms/feudal_mappo_jax/goal_dependence_probe.py)
and its established three-condition test, each gap `real - variant` with a
paired CI excluding 0:

| condition | what it captures |
|---|---|
| `gap_zeroed > 0` | the goal channel earns return against no goal at all |
| `gap_constant > 0` | that return comes from the goal's content, not a frozen vector |
| `gap_permuted > 0` | and specifically from the per-agent assignment |

Read `d_cos_gap_env` first, as the probe's documentation requires: a zero env gap
makes the agent gap uninterpretable. Keep the `zerogoal` positive control, which
must report **exactly 0.0** on all five variants; anything else means the harness
is wrong and no other number is worth reading.

Two notes specific to this plan. First, the three conditions are *necessary, not
sufficient* — `mjx_12a_4o_4444_512/feudal_n01_local_private` passes two of three
while tying its goal-free control outright. The decisive test remains the
between-arm return against `feudal_film_zerogoal` trained from scratch on the
same env group and seeds. Second, **anchoring changes what `d_cos_mean` means**:
today the manager owns the semantics of both arguments, so its level is
uninterpretable; with `phi` pinned by an exogenous label it becomes a genuine
goal-following measure. If the anchored arms are accepted, that reinterpretation
should be recorded explicitly, because every prior run's `d_cos` series does not
have it.

### Mechanism layer

| Question | Measurement |
|---|---|
| Can the latent identify the command on unseen states? | Contrastive loss, candidate accuracy versus `1/K`, correct-minus-best-wrong cosine margin, and the continuous `cos(delta_z, u)` |
| **Is that identification trivial?** | The §3 channel ablations: full-observation identification minus `own_velocity`-only identification. A near-zero increment means the objective fitted the agent's own kinematics |
| Does changing a command affect behavior? | Action response and paired physical endpoint differences, repeated under independent noise |
| Are effects substantial? | Agent displacement/velocity, contact changes, box displacement; near-zero displacement rate |
| Are effects reliable? | Within-command variability versus between-command differences |
| Is useful coordination developing? | Contact counts per box, coupling satisfaction and duration, deliveries, task return |
| Is geometry stable and usable? | `goal_direction_count` against the `N^2/(N + N(N-1)/goal_dim)` random baseline (8.93 at N=12, `goal_dim`=32), `state_latent_erank`, `goal_concentration`, encoder drift across versions, normal-rollout goal alignment |
| Is the intrinsic critic keeping up with a moving `phi`? | `intrinsic_explained_variance`, `adv_int_std_raw` across encoder versions |
| Does the hierarchy help the task? | Between-arm return versus matched manager-trained and goal-free controls |

Log contact and free-space contexts separately. Repeat across teammate command
configurations (design (B) of §4 supplies one such context directly):
predictability with one fixed team context is not general controllability.
Evaluate both held-command trials and ordinary rolling-goal execution; success
in one does not establish success in the other.

Decision criteria:

- Chance-level held-out identification and no physical response: inspect goal
  delivery and worker goal-following learning before increasing encoder fitting.
- **High held-out identification carried entirely by `own_velocity`:** this is
  the predicted degenerate outcome, not a success. Apply the §3 decision that was
  recorded in advance; do not reinterpret the metric afterwards.
- Better training classification without held-out improvement: investigate
  overfitting or input leakage; do not interpret this as controllability.
- Better held-out identification with only tiny movements: the code is
  distinguishable but the outcomes are weak. Measure effect size before adding
  a magnitude objective, since that would change the method.
- Repeatable individual effects without better task progress: investigate
  manager selection, horizon, and coalition structure. Do not call this useful
  coordination on the basis of latent metrics.
- Improved held-out response and improved matched task return against the
  goal-free reference: supports the anchoring hypothesis for the tested
  environments and interaction budget.

Set any minimum physical-effect or practical return threshold after calibrating
measurement resolution and before examining comparative test outcomes. Do not
invent universal thresholds from cosine values.

## 8. Implementation sequence

### Step 1 is a gate, and it needs no training

1. **Frozen-checkpoint probe.** Add `goal_intervention_probe.py` with paired
   branches, outcome logging, held-out groups, the §3 channel ablations, and
   positive/negative controls. Run it on the **existing** trained `feudal_film*`
   checkpoints. This answers "do the current commands already produce
   systematically alignable outcomes?" for the price of a probe, before any
   objective, optimizer or config work.

   Reuse, per the repository's standing instruction: compose each arm's config
   through `train._build_dispatch_args` and override dims from the checkpoint,
   as `goal_dependence_probe.py` does (the yaml moves, checkpoints do not —
   `goal_dim` already changed 16 -> 32 under existing arms); reuse
   `latent_locality_probe.collect_states` and its
   `unsupported_env_reason` choke point rather than rolling a second rollout
   helper; reuse `SyncMacroMJX.snapshot` / `state_from_snapshot` for the
   branching; reuse the `--chunk` idiom from `reward_magnitude_study.py`.

   **Pre-register the readings:**

   | Probe outcome on existing checkpoints | Reading | Action |
   |---|---|---|
   | Identification already near-ceiling **and** driven by exteroception | The latent already distinguishes commands by their world effect; the bottleneck is usefulness, not distinguishability | `L_anchor` is largely redundant. Redirect to the manager's *selection* problem and to the worker's incentive, and do not build steps 2-5 |
   | Identification already near-ceiling but carried by `own_velocity` | Commands change the agent and nothing else | The plan's real target is effect *magnitude and exteroceptive consequence*, not identifiability. Revisit §3's input restriction before proceeding |
   | Identification near chance | Commands do not produce systematically alignable outcomes | Proceed as written; this is the case the plan was designed for |

   Existing evidence (`worker_goal_action_delta` 0.91-1.09) says commands already
   change *actions* a lot but says nothing about whether the resulting
   displacements align *systematically* with the command across states, which is
   the strictly stronger property `L_anchor` needs. That is exactly the gap this
   probe measures, and it is why it is worth running first.

2. **Gradient ownership refactor.** Add `anchor_encoder`, the `stop_gradient`
   on `s` into the core, and the `optax.masked` label partition (§5). Make
   self-checks [8] and [10](d) arm-conditional. Verify forward, reward, and
   goal-recomputation parity, and verify the unanchored arm is **bit-identical**
   to the pre-change code — the repository's standing bar for a gated change,
   established with the `intrinsic_coef` and `latent` work.
3. **Encoder-only anchoring.** Add the objective and auxiliary optimizer, update
   order, logging, configuration, and serialization. Implement the first stage
   without trial PPO updates.
4. **Bounded pilot and controls.** Run matched short experiments and inspect
   held-out response, channel ablations, physical outcomes, and drift. Establish
   command learning before scaling the training budget.
5. **Optional trial worker training.** Implement the separately specified return
   and value treatment if broader command practice is necessary. Match this
   additional data and optimization in controls.
6. **Task and coalition evaluation.** Extend duration and seeds only after the
   mechanism checks; test partition and different teammate contexts.

Likely code surfaces:

- [manager.py](../algorithms/feudal_mappo_jax/manager.py): the `latent_only`
  path, the `stop_gradient` boundary, the arm-conditional self-checks.
- [mappo.py](../algorithms/feudal_mappo_jax/mappo.py): parameter ownership,
  auxiliary optimization, manager target and PPO update boundaries.
- [trainer.py](../algorithms/feudal_mappo_jax/trainer.py): compact snapshots,
  trial collection, versioned rewards and update scheduling.
- [types.py](../algorithms/feudal_mappo_jax/types.py): explicit auxiliary data,
  configuration and encoder train state; avoid overloading ordinary transitions.
- [run.py](../algorithms/feudal_mappo_jax/run.py): logs, all parameter and
  training checkpoint save/restore paths, evaluation and visualization
  construction.
- `conf/model/`: a new experimental group and explicit matched controls, using
  the repository's `# @package _global_` convention. ⚠ A model group without
  that header is silently inert and runs at algorithm defaults; that has already
  cost 36 runs in this repository. Verify composition before launching.
- [test_feudal_seams.py](../algorithms/tests/test_feudal_seams.py): cross-component
  invariants; focused intervention tests may live in a separate test module.

Use a distinct results/configuration identity. Preserve legacy loading; if §5's
label-partition route is taken the parameter tree is unchanged and this is free,
which is most of the argument for it. Save encoder parameters, optimizer state,
auxiliary RNG, schedule counters, and any state needed for reproducible resume.
Ensure all view/probe/eval entry points construct the same representation as
training.

## 9. Required verification

Use small controlled dynamics for structural tests, plus a bounded MJX smoke
check. Pin these in the CPU-only seam suite where possible — the suite's autouse
`jax.default_device` fixture exists because sharing a GPU with a training job
produces failures that read exactly like assertion failures. The key tests should
demonstrate behavior, not merely repeat the loss implementation:

- A controllable toy system whose command determines displacement is identified
  on held-out starting states after encoder fitting.
- Identical outcomes for all commands cannot beat the chance loss, even when
  those outcomes depend on the starting state or shared noise.
- A nuisance that changes identically across branches cannot identify the
  randomized command. Shuffled labels remove held-out identification.
- **The channel ablation is faithful**: an encoder restricted to a subset of
  observation dimensions provably reads no others.
- When goal conditioning is explicitly disabled, matched branches coincide.
  FiLM's zero-initialized modulation supplies an additional initial-state check.
- Encoder inputs exclude commands/branch metadata; the auxiliary loss updates
  only encoder parameters, and ordinary anchored-arm updates leave them **and
  their optimizer moments** fixed. Assert the moments explicitly — a masked
  optimizer is the specific thing that is easy to get wrong here, because
  zero-gradient Adam steps are not no-ops.
- The unanchored arm is bit-identical to the pre-change code across the rollout,
  every loss key and every post-update parameter leaf.
- Self-checks [8] and [10](d) hold in their original form for the unanchored arm
  and in their inverted form for the anchored one.
- Goal overrides reach the actor throughout the horizon with matching magnitude,
  under both `normalize_pooled_goal` settings; paired keys and teammate command
  schedules remain matched.
- A terminal transition never reads a reset endpoint; one branch's rewards,
  history and GAE cannot cross into another branch.
- `state_from_snapshot` reconstruction is shared by all branches of a group, so
  the within-group comparison is exact; the reconstruction is *not* asserted
  equal to the original `mjx.Data`.
- Enabling measurement alone does not consume the ordinary policy RNG or change
  its updates. Count its simulation cost even when it does not train anything.
- For trial PPO, the initial importance ratios are one under stored conditioning,
  and bootstrap/termination semantics match the declared trial objective.
- Saving/resuming reproduces the next command samples, encoder update, and
  ordinary rollout under the same supported numerical environment.

## 10. Coalition extension and limits

Individual controllability is conditional on the teammates. A goal may produce
box progress only when enough agents already maintain contact. The initial
experiment should expose this dependence rather than force a conclusion that
every useful direction must be independently achievable. Design (B) of §4 gives
one cheap handle on it: identification under randomized teammate commands versus
under a frozen team context.

If coordinated interventions reveal useful effects that individual commands do
not, design a second experiment with a coalition command and a joint or
object-centered outcome representation. Its command-to-worker mapping,
representation inputs, and manager objective must be specified separately.
Applying the individual loss unchanged does not implement coalition control.

Local partial observability can also hide real effects. Failure to identify a
command from `obs_i` may reflect insufficient observations or an unsuitable
horizon. A recurrent progress encoder or centralized outcome representation is
a separate information/architecture ablation, not an automatic fix.

Avoid claims of guaranteed causal credit assignment, an exact transition policy
gradient for coupled dynamics, automatic task relevance, or learned human-readable
skills. The proposed evidence concerns measured command effects and task return
under explicitly defined intervention and execution policies.

## 11. Research context

- [FuN, Vezhnevets et al. (2017), §§3.2-3.3](https://proceedings.mlr.press/v70/vezhnevets17a/vezhnevets17a.pdf):
  directional goals, intrinsic goal-following rewards, and an approximate
  transition-gradient argument based on goal-centered transition directions.
- [DIAYN, Eysenbach et al. (2019)](https://arxiv.org/pdf/1802.06070): a latent
  code trained to be recoverable from visited states. The relevant lesson here
  is the failure mode, not the method: a discriminator objective is satisfied by
  whatever state component is easiest to distinguish, which is why §3 treats the
  `own_velocity` channel as the crux rather than as a caveat.
- [DADS, Sharma et al. (2020), §3](https://arxiv.org/pdf/1907.01657):
  skill learning based on future-state information conditional on the starting
  state, with predictable skill dynamics. This motivates distinguishing command
  effects from state-only predictability; it does not supply the proposed
  multiagent branching protocol or its cosine classifier.
- [METRA, Park et al. (2024), §4](https://arxiv.org/html/2310.08887v2):
  joint skill-policy and representation learning through directional latent
  progress constrained by a temporal metric. It is a relevant alternative if
  metric learning becomes the main approach, and its temporal-distance constraint
  is one principled answer to the magnitude problem §7 defers. Its results do not
  validate this proposed contrastive multiagent adaptation.

The primary intervention here is an auxiliary learning signal for the same
latent geometry used to issue and score goals. Its value must be established
against the specified controls; physical-coordinate goals are an optional
reference experiment, not a prerequisite or a promised transfer mechanism.
