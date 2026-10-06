# Joint-goal team-return critic and D++-inspired recruitment

Date: 2 October 2026. Status: proposed; no implementation or training run.
Target: `algorithms/simplified_feudal_mappo_jax`, as selected by the user.

## 1. Recommended first result

Build a frozen-policy experiment that answers this question: from an actual
manager decision state, can a joint-goal critic predict which reassignment of
existing teammates improves environmental team return?

Implement the simulator evaluator first, train a separate scalar return critic,
and validate its recruitment rankings before using it to choose goals. Start
with the trained relative-input, tanh waypoint hierarchy and the square-arena
`mjx_2a_4o_1122_1024_gs` environment. Keep the existing worker, manager PPO, and
counterfactual credit model as comparison arms.

The original [D++ paper](https://jenjenchung.github.io/anthropomorphic/Papers/Rahmattalabi2016dppIROS.pdf)
evaluates adding identical counterfactual agents. This experiment reallocates
existing agents, including their travel and abandoned tasks, so call it
**D++-inspired goal recruitment**. The proposed design and validation gates below
are implementation choices, not claims established by that paper.

## 2. Define the action and continuation precisely

At a manager window boundary, let `g = waypoint` be the complete `(N, 2)` array
of absolute waypoints proposed by the current manager. The worker follows this
assignment for `H = goal_horizon` primitive environment steps. At the next
boundary, the original manager chooses a fresh assignment from the new state.

For fixed manager and worker parameters:

\[
Q^{\mu,\pi}(s,\mathbf g)
=\mathbb E\left[R_H+\gamma_M^H V^{\mu,\pi}(s_H)\mid s,\mathbf g\right],
\qquad R_H=\sum_{k=0}^{H-1}\gamma_M^k r_k^{\mathrm{team}}.
\]

Use `manager_step_gamma`, falling back to `worker.gamma`, for the per-step
discount. The manager decision discount remains `gamma_M ** H`. Store and
predict one environmental team return per joint assignment.

Initial evaluation uses the actual finite episode: both success/failure and the
environment's time limit end return accumulation, and terminal continuation is
zero. Include time remaining in critic features. This makes the objective
explicit and matches episode-return evaluation. The current PPO manager's
timeout bootstrap follows a continuing-task convention; keep that training
path intact, but do not copy its decorated rewards into this auxiliary critic.
If a continuing-task critic is desired later, a timeout inside a window requires
continuation under the *retained goals for the remaining commitment*, rather
than an immediate manager value as though the goals had expired.

Use stochastic policies for both labels and critic queries initially. A
deterministic manager/worker evaluation defines a different continuation policy;
report it separately or fit a critic for that policy.

For a focal agent `i`, a coalition mask `C`, and supporting waypoints `u`:

```python
recruited = where(C[..., None], u, base_waypoints)
B = Q(state_features, recruited) - Q(state_features, base_waypoints)
```

Enforce `C[i] = False`. The focal agent and every unselected teammate retain
their exact baseline waypoints. Both Q calls receive identical state features.
`C = empty` is always a candidate, with exactly zero benefit.

This initial action lasts one window. If recruitment needs persistent roles
across several windows, introduce that commitment into the manager action and
continuation definition in a later experiment. A short travel waypoint followed
by the original manager does not promise continued assistance.

## 3. What can be reused, and what needs to change

| Existing component | Reuse or change |
|---|---|
| `trainer.make_policy`, `Policy.observe/decide/act` | Reuse the actual manager and worker forward passes in forked rollouts. |
| `waypoints.goal_error`, `waypoint_from_action`, `waypoint_offset` | Reuse the goal frame, radius, and arena limits. |
| `counterfactual.adv_model_input` | Reuse the flattened joint-goal layout concept. |
| `HierTrainState.manager_adv` | Keep its present meaning; add a separate optional `manager_q`. |
| `ManagerGoal(pos, offset)` | Extend or introduce a separate return-critic record, independent of `manager_credit`. |
| `counterfactual.fit_adv_model` | Reuse batching/optimizer structure through a small shared regression helper if useful. |
| `run.py` checkpoint helpers | Add Q parameters, optimizer state, target parameters if enabled, and policy-version metadata. |

The current advantage model already sees the complete joint goal assignment. It
regresses raw team GAE advantage, and its counterfactual replaces the focal
agent's own goal while teammates retain theirs. The requested critic instead
regresses return, and recruitment preserves the focal goal while replacing
selected teammates' goals.

For an exact advantage function at the same state,
`A(s, recruited) - A(s, base) = Q(s, recruited) - Q(s, base)`, because the state
value cancels. That makes the existing model a useful comparison, but its fit to
on-policy GAE targets does not establish accuracy on coordinated reassignments.
The single-agent counterfactual baseline in
[COMA](https://arxiv.org/abs/1705.08926) is also a different use of a joint-action
critic from the coalition-selection experiment proposed here.

## 4. Critic features and records

Use a scalar MLP initially, with the existing critic's width and optimizer
conventions. Input:

```text
concat(physical_state_features, flatten((waypoints - start_positions) / R))
                                                        -> scalar Q
```

Keep the absolute waypoints in records. Construct offsets from each candidate's
actual starting positions, after applying waypoint bounds. The full physical
state is the mathematical `s`; the MLP consumes a feature representation of it.
Do not describe an incomplete observation vector as an exact Markov state.

The square environment's compact `global_state` includes agent positions and
velocities, box positions, delivery flags, touch counts, and coupling levels.
It omits box orientation and box linear/angular velocity. Add normalized box
motion, `sin/cos(yaw)`, and episode time remaining through a dedicated optional
critic feature builder. Include reward-history fields if they are independently
needed to reproduce the next reward. Preserve the complete `EnvState` for
simulator forks, including physics data and reward caches.

Proposed auxiliary decision record:

```text
state_features, start_positions, executed_waypoints
raw_discounted_team_reward, actual_duration
next_state_features, next_manager_actor_input, next_positions
terminated, truncated, manager_version, worker_version
```

Store full simulator snapshots only for a bounded sample of decisions used by
the fork evaluator. Avoid storing every MJX physics state for every rollout
window. Both raw rewards and true successors must be captured before resets.

Coalition metadata additionally records focal agent, changed-agent mask,
task/box ID, per-agent slot/role, original goals, and supporting goals. In the
first version roles determine waypoints; workers receive their existing waypoint
interface. If a role later changes worker behavior beyond its waypoint, that
role becomes part of the critic's action input too.

## 5. Simulator evaluator and training data

Implement `recruitment_probe.py` around a frozen checkpoint pair `(mu, pi)`:

1. Collect actual manager boundary snapshots, observations, and base goals.
2. Construct valid recruitment candidates from each snapshot.
3. Fork the *same complete EnvState* for the base and each candidate.
4. Run each branch's workers under its assignment for H steps, or until the
   episode ends. Workers recompute actions from branch-specific observations.
5. After H steps, resume the original manager and worker in each branch until
   episode end. The manager recomputes goals from that branch's state.
6. Accumulate discounted `info["task_reward"]`, using the same per-step policy
   noise sequence across paired branches. Repeat with multiple independent
   sequences to estimate means and uncertainty.

Sharing random numbers pairs the policy noise; it does not force identical
actions or future goals once states diverge. Agents physically travel and all
tasks continue contributing to team return in every branch.

Full-tail Monte Carlo returns give labels without relying on the existing
manager V at counterfactual endpoint states. A cheap H-step rollout plus a value
bootstrap can be an additional diagnostic, but report its shared bootstrap
error instead of treating it as independent ground truth.

Train Q on both base and recruitment assignments. Split by source episode and
snapshot: all candidates and repeats from a snapshot belong to one partition.
Reserve independently seeded states and some coalition/role arrangements for
testing. Record which checkpoints produced every label; stale data from a
different worker policy estimate a different Q unless relabeled.

Start with modest configurable budgets, for example 128 snapshots, at most 8
candidate assignments per snapshot, and 4 paired repetitions. Measure runtime
and uncertainty before scaling. These are pilot settings, not accuracy claims.
Count simulator steps and device memory as experimental costs.

If branches never produce useful task progress, first check goal following and
support geometry. A flat return dataset cannot teach the critic the payoff of
unobserved cooperation. Include deliberately coordinated candidate assignments
in the data collection; a learned critic alone does not solve exploration.

## 6. Supporting roles and coalition search

Initially support the square-arena box task only. Its geometry is simpler and
the 2-agent environment has exactly one possible teammate for each focal agent.

- Associate the focal goal with an undelivered box only when its waypoint/path
  is compatible with approaching or pushing that box. Flag ambiguous cases and
  return the baseline rather than inventing a task commitment.
- Generate distinct staging/contact slots on the useful pushing side of the
  box, accounting for box pose and occupied slots. Other teammates' retained
  goals remain part of the candidate context.
- Express a support target from the recruit's own actual position. Restrict it
  to the same per-axis radius R and arena bounds as manager waypoints. A distant
  recruit receives a legal travel waypoint toward its slot; the worker has H
  steps to realize it. Do not move the recruit's starting position.
- Preserve the focal goal even when it appears suboptimal; rejecting a focal
  proposal is a separate manager action from this counterfactual.

For scaling, shortlist nearby eligible teammates by travel cost and enumerate
small subsets jointly, including the empty coalition. At 6 agents with
coupling `[4, 3, 3, 2]`, include coalitions large enough to fill the threshold:
one focal agent may need three recruits. Independent singleton scores or a
greedy positive-singleton rule can miss a benefit that appears only after the
required coalition forms. Shortlisting is a compute heuristic and can miss an
optimal distant teammate; report that limitation.

Score complete realized assignments. Compute base Q once, batch candidate Q
calls, and chunk simulator branches to bound memory. Use total benefit for
selection, with fewer changed agents as a tie breaker. A score divided by
coalition size changes the objective and should be an explicit ablation.
Travel and abandoned-task costs are already in environmental return; additional
recruitment penalties represent a new preference rather than recovered costs.

Make one coalition decision per state initially. Applying several individually
positive reassignments can invalidate each other's benefits. Multiple
recruitments need evaluation of their combined final assignment.

## 7. Validation gates before executing recruitment

Report held-out Q error, but make paired benefit prediction the main criterion:

- Predicted versus observed B: error, correlation, and sign accuracy outside
  the simulator estimate's uncertainty band.
- Candidate ranking and regret against the best candidate in the evaluated
  set, including the empty coalition.
- Mean observed benefit among selected candidates and the fraction that hurt
  team return. Use intervals clustered by episode/snapshot, not candidate-level
  pseudo-replication.
- Breakdown by recruit distance, coalition size, missing coupling partners,
  and the value of abandoned tasks.

The first acceptance gate is a positive held-out mean benefit for the learned
selector with a paired confidence interval above zero, plus comparison against
nearest-support and random-candidate selection. Also report simulator-selected
recruitment as the upper reference within the same candidate set. Choose score
thresholds on validation data, then evaluate the locked selector on test data.
Do not use test labels to tune abstention thresholds.

An ensemble can supply a conservative disagreement heuristic later. It does
not replace paired simulator calibration, especially when all members share
the same data gap.

## 8. Safe progression into training

First run a **single intervention per evaluation episode**, then return to the
original manager. This exactly matches Q's stated continuation. Next evaluate
recruitment at every boundary as a separate policy `mu_recruit`; refresh labels
and critic continuation for that policy before claiming Q evaluates it.

When integrating auxiliary Q fitting into live training, freeze the behavior
policy versions during collection and target construction. An optional target
critic supports expected-SARSA updates at manager boundaries:

\[
y = R_{\mathrm{raw},\ell}
  + \gamma_M^{\ell}(1-d)\,
    \frac{1}{K}\sum_{k=1}^{K}
      Q_{\bar\theta}(s_{\mathrm{end}},\mathbf g'_k),
\quad \mathbf g'_k\sim\mu_{\mathrm{behavior}}(\cdot\mid s_{\mathrm{end}}).
\]

Here `d = terminated | truncated` under the chosen finite-episode objective;
live windows have `ell = H`. Stop gradients through targets and sampled goals.
This derives continuation V from the goal critic under the stated manager,
without depending on the existing PPO V's different timeout convention. Fit
with a scalar regression loss, using explicit auxiliary optimizer settings.
Include executed counterfactual windows, not fabricated labels from Q's own
predictions, and refresh forked data as workers change.

Keep Q fitting separate from PPO's real-return advantage. Recruitment benefit
depends on the focal goal and is not the action-independent baseline used by
the existing credit correction; substituting B as an advantage or adding it to
reward is a new optimization objective.

Training-time execution also requires an explicit behavior-policy design. Do
not rewrite goals and treat the resulting assignment as an unchanged sample
from the factorized Gaussian manager. Options are a stochastic recruiter with
logged decision probabilities, or a documented fixed transformation of sampled
manager proposals whose raw proposals remain the policy actions. The latter
must freeze its transformation during PPO updates and changes what manager Q
is conditioned on. Distilling validated selected goals into the manager with a
separate supervised loss is another later experiment. These choices should
follow evidence from the frozen-policy selector.

A centralized recruitment selector uses privileged joint state at execution.
Compare it to the relative/global manager arms accordingly; it changes the
execution information available to the `manager_input: local` arm.

## 9. Implementation sequence and file map

| Step | Concrete change | Completion check |
|---|---|---|
| 1 | New `recruitment.py`: assignment records, masks, box slots, bounded support waypoints. | Focal and unselected goals preserved; slots legal and distinct; empty case exact. |
| 2 | New `recruitment_probe.py`: frozen checkpoint loading, boundary snapshots, paired branch returns, dataset/report output. | Identical assignments give identical paired rollouts; branches use full original state and recompute all policies. |
| 3 | New `goal_q.py`: state-feature builder, joint-goal scalar critic, offline fitting and held-out evaluation. | Benefits and ranking assessed on unseen episodes; policy versions and objective stored with model. |
| 4 | `trainer.py`, `types.py`, `run.py`: optional Q state/records, inference loading, single-intervention evaluation. | Default modes and existing checkpoint layouts preserved; selector meets the held-out gate. |
| 5 | Optional online Q targets and refreshed fork collection; new experiment YAMLs. | Explicit raw reward/terminal masks, consistent discount, reproducible resume, bounded extra cost. |
| 6 | Recruitment as a trained manager decision or supervised distillation. | Correct behavior likelihoods/objective, matched-budget multi-seed return comparison. |

Use optional fields and mode guards. Proposed configuration names are
`goal_q_enabled` and `recruitment_mode: off | diagnose | eval_once`; the default
is off. Use named Hydra model groups for distinct checkpoint/result paths,
and retain `# @package _global_` in new model YAMLs. Enabling Q on an existing
checkpoint should explicitly load the old policy trees and initialize the new
Q tree; resume of a Q-enabled run restores its own complete state.

Add meaningful seam tests to `algorithms/tests/test_simplified_feudal.py` or a
new `test_goal_recruitment.py`: raw task reward selection; per-step versus
per-window discount; terminal and timeout masking; pre-reset successor capture;
radius/arena bounds; same-state preservation; coalition thresholds requiring
multiple recruits; policy reactivity under divergent branches; and optional Q
checkpoint save/load. Keep broader training validation in the experiment probe.

Pilot comparisons: no intervention, nearest-support heuristic, random valid
candidate, existing advantage-model ranking, learned-Q ranking, and
simulator-selected ranking. Use 1 agent as the exact no-recruitment control,
2 agents as the smallest useful case, and 6 agents to test non-additive
coalitions. Compare matched seeds and account for auxiliary simulator work.

The first implementation deliverable is steps 1–3: an auditable dataset and a
report showing whether predicted benefits identify real, useful recruitment.
