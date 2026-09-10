# Diagnosis of the 12-agent goal experiments

The new results support a harmful intrinsic-learning effect: increasing the
intrinsic coefficient from 0.1 to 0.5 lowers late task return in all 12 matched
environment × manager-core × seed comparisons. They do not establish that goals
are inherently harmful, or isolate the entire gap against zero-goal controls:
those controls remove both goal input and intrinsic learning. The missing arm is
an active-goal FiLM worker with intrinsic coefficient zero.

This analysis reads the current local logs and checkpoints, not the older runs
documented in CLAUDE.md. It does not change training code or launch training.

## What the results establish

All 36 feudal checkpoints have FiLM layers and a `(40, 168)` actor input kernel;
the six recurrent arms have LSTM parameters; intrinsic critics appear only in
the nonzero-coefficient arms. Logged initial coefficients are 0.1 and 0.5. All
zero-goal checkpoints have exactly zero FiLM kernels. The previous missing-Hydra-
package bug therefore does not explain these results.

For a fair comparison with MAPPO, use **70–80 million environment steps**. All
42 local runs cover that window. Some MAPPO copies are incomplete: truncation
seed 0 stops at 83.59M and seed 1 at 99.22M; partition seed 2 stops at 98.30M.
Do not silently average a different number of seeds at the right-hand edge.

| Worker / manager | Truncation return | Partition return |
|---|---:|---:|
| MAPPO | 264.5 ± 64.3 | 267.3 ± 61.1 |
| Zero goal / MLP | 252.0 ± 45.8 | 267.3 ± 35.9 |
| Zero goal / dilated LSTM | 278.7 ± 50.8 | 271.2 ± 55.5 |
| Intrinsic 0.1 / MLP | 145.7 ± 11.0 | 140.7 ± 13.2 |
| Intrinsic 0.1 / dilated LSTM | 168.6 ± 5.8 | 119.3 ± 29.7 |
| Intrinsic 0.5 / MLP | 91.0 ± 38.6 | 30.9 ± 20.5 |
| Intrinsic 0.5 / dilated LSTM | 74.1 ± 3.0 | 56.2 ± 13.8 |

Values are means ± sample standard deviations across three training seeds.
First average actual evaluation events within each seed/window, then average
the seeds equally. `reward` is deterministic evaluation return, not rollout
reward. The logs carry evaluations forward between evaluation events; filter
with `eval_time > 0`. Temporal points are not independent training seeds.

All feudal arms finish at 99.975M. In their **90–100M** window, partition returns
are 282.7 / 279.8 for the two zero-goal controls, 139.5 / 119.2 for coefficient
0.1, and 38.4 / 64.2 for coefficient 0.5. Recurrence does not consistently fix
the problem. With zero goals it cannot causally help the worker through goals;
differences between these controls are not evidence of useful manager memory.

The strongest diagnostic is the partition MLP / coefficient 0.5 arm:

- Task return: **38.4**, compared with **282.7** for its zero-goal control.
- `d_cos_mean`: **0.258**, compared with **−0.066** for that control. This is
  the mean cosine between a goal and the ensuing ten-step latent displacement.
  Larger means more agreement with the learned direction, not more box delivery.
- Zeroing goals at evaluation reduces its return to **0.77**. The worker
  depends on goals, yet the learned policy remains very poor. Removing goals
  at inference does not undo the policy learned under intrinsic rewards.
- `eval_gap_permuted`, real return minus return with goals exchanged between
  agents, averages **9.52**, with seed means **22.08, 6.22, 0.27**. This is
  evidence of dependence in some seeds, not proof of beneficial role assignment.

At coefficient 0.1 on partition, the MLP permutation gap averages only **0.16**
(seed means −4.60, 4.43, 0.66), despite a large deficit to the control.

Gross representation collapse and missing training windows are weak explanations:
late intrinsic-arm latent effective ranks are approximately **18–28** of 32,
goal direction counts approximately **9** for 12 agents, and valid manager
windows approximately **99%**. Intrinsic critic explained variance is about
**0.96–0.99**. These are diagnostics of geometry and fitting bootstrapped
targets; none certifies controllability, reward relevance, or causal goal use.

One positive-control anomaly remains: truncation/zero-goal/seed 2 has a single
evaluation at 29.20M with gaps about 0.15. All other evaluation events in that
run, and all other zero-goal runs, have zero gaps. Its source is unresolved;
it does not affect either reported comparison window. Current configuration
also requests 16 evaluation episodes for feudal versus 32 for MAPPO; the logs
do not serialize the original resolved episode count. Match counts prospectively.

## Why the objective can hurt

**1. The intrinsic reward measures a different outcome from the task.**
`manager.py:worker_intrinsic_reward` scores the direction of change in a learned
latent space. `multi_box_push_mjx.py:_task_reward` pays upward box displacement
and a one-time delivery bonus. Cosine alignment does not require moving a box,
making sufficient physical progress, or maintaining a viable pushing coalition.
The partition task needs contact counts [3, 5, 4]; independent agent rewards
can favor changes that fail those collective requirements. Motion without
productive pushing is a plausible attractor, not a behavior verified by these
aggregate logs. Verify it with contact counts, box displacement, agent travel,
deliveries, and action interventions before naming a particular behavior.

**2. Per-agent goals have no enforced shared or controllable meaning.**
`FeudalManager.__call__` makes one team embedding `z`, then computes
`s = Dense(n_agents * goal_dim)(z)` and reshapes it. Equivalently,
`s_i = W_i z + b_i`. Each agent's latent displacement can reflect any agent or
box, and each output block has its own learned coordinate system. The shared
worker receives local observations and a goal, with no explicit common decoder
for these coordinate systems. This increases the difficulty of learning one
consistent goal-to-action mapping; it does not make such a mapping impossible.

Consequently, a large latent agent-permutation gap can partly reflect different
output-head coordinates, rather than useful division of labor. Even the
environment-permutation gap establishes statistical association, not that
following a goal caused a beneficial transition. Earlier comments declaring
the manager healthy based on these gaps overstate the evidence.

The manager uses the extrinsic advantage, so it is inaccurate to say it
unconditionally maximizes intrinsic reward. However, its update is the surrogate
`−mean(A_manager * cosine(displacement, goal))`, not a likelihood-based policy
gradient on the goal actually selected. The original FuN derivation assumes
goal-conditioned transition directions approximately follow a von Mises–Fisher
distribution centered on the goal. Whether that approximation holds for these
coupled agents is unmeasured. See [Vezhnevets et al., 2017, §4](https://proceedings.mlr.press/v70/vezhnevets17a/vezhnevets17a.pdf).

**3. Advantage normalization does not protect the task gradient.**
`mappo.py:ppo_update` computes `A = normalize(A_ext) + alpha * normalize(A_int)`.
This fixes the earlier raw-reward scale problem. It does **not** make alpha an
exact fraction of the policy gradient, contrary to comments in the code.
At the initial PPO ratio of one, the two gradient contributions are proportional
to `E[A_ext * grad(log pi)]` and `alpha * E[A_int * grad(log pi)]`. Equal
advantage variance does not imply equal correlation with the policy score or
equal gradient norms. PPO clipping and Adam add further differences. Small,
poorly estimated intrinsic advantages can also be rescaled to unit variance.

A numerical counterexample with equal-unit-variance advantage streams produced
an intrinsic/extrinsic gradient norm ratio of **10** at alpha **0.1**. This
demonstrates the mathematical problem with the claimed guarantee; it is **not**
a measured gradient ratio from these experiments. Actual gradient norms,
directional conflicts, and update sizes are not in the current logs.

The linear schedule applies its largest coefficient before workers are
competent and tapers across essentially the entire 100M-step run. Turning it
almost off at the end does not reset visited states, network weights, learned
goal dependence, or optimizer state. Annealing removes ongoing objective bias
in the limit; it does not guarantee convergence or recovery of lost learning.

**4. Two implementation details add avoidable credit noise.**
The collector stores pre-action `state_latent[t]`. The intrinsic calculation
uses `cos(s[t] − s[t−i], g[t−i])` and writes it onto the transition for action
`a[t]`; the extrinsic reward on that transition comes from `env.step(a[t])`.
Thus the intrinsic stream is delayed by one action relative to the outcome
being scored. With horizon one, a two-action east-then-north example gives the
current stream `[0, 1, 1]`: the first achieved goal is credited one entry later.
Future-return terms retain some learning signal, so this is not proof that
the delay causes the large deficit. It does add past-dependent immediate reward,
and the final action's outcome is omitted from the stored latent sequence.

Score the actual successor state against the goals active for the action,
including the true pre-reset terminal successor and the rollout's last
successor. Simply shifting the existing array is insufficient across resets
and at its last entry. A meaningful regression test must associate a controlled
action with its outcome; the existing seam test mirrors the current helper.

Also, the intrinsic critic reads only flattened current observations, while
its reward depends on previous latent states/goals. The policy itself consumes
a goal history, and the dilated manager has recurrent state. Current observations
alone need not be sufficient for the intrinsic value function. Include the
goal/phase and adequate history or recurrent context in that critic. A pooled
goal alone is only an approximation for the current historical cosine reward.

## Recommended sequence

**First isolate the cause with a small, matched experiment.** Use the MLP
manager initially; keep the observation, critic, optimizer, seeds and evaluation
budget identical. Compare zero goals/alpha 0, active goals/alpha 0, active
goals/alpha 0.01, and the existing active goals/alpha 0.1 setup. Evaluate
learning curves and seed-level task return before spending another complete
recurrence sweep. The existing `feudal_film` model is the missing alpha-zero
MLP arm; `feudal_film_n001` is alpha 0.01. Neither appears in these result sets.

For the recurrent alpha-zero arm, explicitly set
`model_params.manager_core=dilated_lstm`: despite its name,
`conf/model/feudal_film_dilated.yaml` currently does not set `manager_core`.
Audit composed configs and checkpoint shapes before interpreting names.

Correct reward/transition alignment as its own comparison, keeping all other
choices fixed. For the intrinsic actor contribution, log the norm ratio and
cosine of the separately computed extrinsic and intrinsic gradients before
clipping. Also record extrinsic surrogate change under the actual proposed
optimizer step. Advantage correlation alone cannot establish gradient conflict.

**Then protect task learning.** Treat alpha 0.01 as an empirical starting point,
not a safe threshold. Test enabling a small intrinsic contribution after a
worker warm-up, or removing it early enough to leave a substantial period of
task-only learning. A stronger safeguard caps its measured gradient norm and
suppresses/project-outs components opposed to the extrinsic gradient. Such a
projection protects a local first-order surrogate; Adam, finite steps, noisy
estimates and future distribution changes prevent a return guarantee. Inspect
the actual update and compare paired held-out task evaluations.

FiLM already supplies an identity starting point. Making goal influence stronger
or increasing its width is not a solution to a misdirected objective. Preserve
an extrinsic-trained policy path, but remember that an identity FiLM input does
not preserve a baseline policy after its backbone has been trained differently.

**For goals that actually help, ground the goal space.** Start with a shared
agent-centered representation of controllable, task-relevant outcomes: reaching
a pushing position for a selected box, assembling its required crew, and then
moving that box toward delivery. Keep the meaning of the reward representation
fixed or slowly updated while learning goal execution. A shared encoder applied
to each agent/context makes a goal coordinate mean the same thing across agents.
Use role-appropriate subgoals so approaching and holding contact are not punished
for failing to move a heavy box alone. These are proposed design choices, not
facts established by the present logs.

Before restoring the learned manager, check a fixed or scripted goal assignment
against the same worker. If that worker cannot execute coherent goals while
retaining task progress, the manager cannot rescue the interface. Test local
goal changes from matched simulator states while holding other agents' actions
fixed; measure the resulting controllable displacement, coalition formation,
and task progress. Held-goal intervals can simplify semantics and the critic,
but are a separate departure from the current overlapping FuN goal scheme.

If the transition-policy-gradient approximation remains poor, use an explicit
stochastic goal/role policy trained on extrinsic returns over its execution
interval, with the corresponding likelihood ratios and duration discount.
That is a larger algorithm change and should follow, rather than obscure,
the worker controllability test.

For a principled reward-shaping alternative, a fixed bounded potential has
`F(x,x') = gamma * Phi(x') − Phi(x)`. Added at the reward level with a fixed
coefficient and correct boundary handling, its discounted sum telescopes;
this is the standard policy-invariance result from
[Ng, Harada and Russell, 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).
For a changing goal, the augmented state `x` must include goal/phase/history
needed to make the process Markov; include goal-switch transitions and use
zero potential at true terminal states (bootstrap time limits correctly).
Do not independently normalize a shaping-advantage stream, change coefficients
arbitrarily, or omit goal-switch terms and claim the same guarantee. Even valid
potential shaping preserves the objective, not finite-sample learning speed.

Acceptance should require competitive extrinsic return against the matched
active-goal/alpha-zero and zero-goal controls, plus interpretable execution of
intervened goals. A high cosine, a positive permutation gap, or dependence on
goals alone does not meet that standard.

## Reproduction

Run `.venv/bin/python plotting/analyze_feudal_goal_runs.py` from the repository
root. It produces per-seed/window metrics and the task-versus-alignment figure
in `plotting/feudal_goal_analysis/`. The script asserts complete three-seed
coverage for the comparison and figure windows. Checkpoint architecture and
the small reward-timing/gradient counterexamples were checked separately on CPU;
no new simulator rollouts or training-gradient measurements were performed.
