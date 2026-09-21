# A reconstructable goal space for Feudal MAPPO

Date: 17 September 2026. Status: design proposal; not implemented or tested.

The hypothesis is that a reconstruction objective can make the representation
used to measure goal progress retain useful information about observations,
instead of being shaped only by the manager's reinforcement learning objective.
The first experiment should retain directional goals, pretrain a representation,
and freeze it during policy training. Compare a deterministic autoencoder with
a variational autoencoder (VAE) to separate reconstruction from prior
regularization. This is a representation experiment, not yet a reachability model.

## 1. What changes in the current algorithm

`FeudalManager` currently computes both the state embedding `s` and unit goal `g`.
The goal-generation core consumes `s`; worker rewards compare latent state
displacements with previously issued goals. The state arm of the manager's
cosine is detached, but the encoder still receives gradients through the goal
arm. See [manager.py](../algorithms/feudal_mappo_jax/manager.py) and
[`manager_update`](../algorithms/feudal_mappo_jax/mappo.py).

Replace the learning signal for that encoder with observation reconstruction:

```text
observation_i -> encoder -> posterior mean s_i -> manager core -> unit goal g_i
                    |
                    +-> posterior sample -> decoder -> reconstructed observation_i

worker(local observation_i, sum of recent goals_i) -> action_i
intrinsic reward <- alignment of observed latent displacement with issued goals
```

Use the posterior mean for every policy, reward, evaluation and manager-recompute
call. Sample only when training the VAE reconstruction objective. The decoder
need not run during action selection.

For local variants, the reconstructable object is the **local observation**,
not the complete simulator state. The observation can omit hidden objects,
absolute positions and other agents' information. No decoder can uniquely
recover information that the input does not identify. A privileged-state
encoder or a history-based belief representation would be a separate design.

The shared MJX observation already has 40 engineered features, including
velocity, density, contact, box-relative position, goal distance and lidar
([observation.py](../environments/mjx_suite/observation.py)). The current
32-dimensional goal space is therefore only a modest compression per agent.
Representation learning here must justify its cost through better control,
not through the image-compression argument alone.

## 2. Objective and goal semantics

For observation `x`, let the encoder return a diagonal Gaussian:

```text
q_phi(z | x) = Normal(mu_phi(x), diag(exp(logvar_phi(x))))
z_sample = mu_phi(x) + exp(0.5 * logvar_phi(x)) * epsilon
epsilon ~ Normal(0, I)

L_rep = E[-log p_psi(x | z_sample)]
        + beta * KL(q_phi(z | x) || Normal(0, I))

KL = 0.5 * sum(mu**2 + exp(logvar) - 1 - logvar)
s = mu_phi(x)
```

This follows the VAE construction in
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
`beta` controls the information/compression tradeoff; it does not guarantee
interpretable or independently controllable coordinates.

For continuous features use a fixed-variance Gaussian likelihood with explicit
feature scales; for a binary contact flag use a Bernoulli likelihood. Specify
whether losses sum or average over feature and latent dimensions so `beta` has
a reproducible meaning. Report errors by sensor group: sixteen density channels
or many lidar channels can dominate a few task-relevant coordinates. Fit any
additional normalization only on training data, and freeze/checkpoint it.

The initial manager objective remains:

```text
L_manager = -mean_valid(A_external * cosine(stopgrad(s[t+c] - s[t]), g[t]))
```

Retain `worker_intrinsic_reward_aligned`, including its pre-reset successor
endpoint and episode masks. A sampled latent would introduce apparent motion
even between identical observations and would break deterministic goal
recomputation. Do not normalize `s` to unit length: only `g` is a unit direction.

**A direction is not an encoded state.** `decoder(g)` has no intended semantics.
For interpretation, examine a candidate displacement from a particular start:

```text
candidate_observation = decoder(mu(x_t) + rho * g_t)
decoded_change = candidate_observation - decoder(mu(x_t))
```

Choose diagnostic `rho` values from observed c-step latent displacement norms;
the existing manager learns no goal magnitude. Label this a candidate change,
not a predicted future state. A VAE models observations, not dynamics. Also,
the worker receives a sum of recent directions, not a single persistent target.

An absolute-goal variant could instead emit `z_goal` and reward proximity to it.
That is a later experiment requiring a goal lifetime, distance-based reward,
worker interface, and a compatible high-level learning objective. Keeping the
current transition-cosine objective would train only the direction toward
`z_goal`, not its distance. Summing absolute goals would also be inappropriate.

## 3. What this can and cannot establish

1. **Reconstruction is not controllability.** Other agents can change density,
   lidar and contact observations. Encoding these accurately does not give
   worker i control over them. Even a physically possible decoded state may be
   unreachable from this start within horizon c, or require coordinated action
   from several workers. Local inputs reduce mixing across observation blocks
   but do not isolate physical causes.

2. **Reconstruction does not determine useful geometry.** An ordinary
   autoencoder can change its latent coordinates with an invertible transform
   and compensate in its decoder without changing reconstruction. Such a
   transform can badly distort distances and directions. The VAE prior limits
   some distortions, but neither linear controllability nor a meaningful cosine
   metric follows from it. Small progress in an easy coordinate can still earn
   high cosine reward while objects remain unmoved.

3. **A decoder is not a validity constraint on manager outputs.** Reconstruction
   constrains encodings of training observations. `s + rho*g` can leave those
   regions; a neural decoder still returns numbers there. Prior regularization
   is a soft distributional preference, not a physical feasibility test. Even
   low decode/re-encode error is not a reachability certificate. Sampling goals
   from observed encodings improves data support but still does not establish
   current-state reachability.

4. **The VAE introduces another form of collapse.** Excessive information
   pressure, decoder behavior and optimization can leave posterior means nearly
   constant, destroying displacement rewards. Monitor variance of posterior
   means, per-coordinate KL and held-out reconstruction; consider a gradual KL
   ramp if needed. A large posterior sampling variance does not establish useful
   state information. Collapse is not solely a large-beta phenomenon; see
   [Dai et al.](https://proceedings.mlr.press/v119/dai20c.html).

5. **Good reconstruction can favor the wrong information.** Background sensors,
   easy velocity changes, or redundant channels may consume the available
   representation. Full reconstruction and a compact representation of only
   controllable effects can conflict. A later control/context split would need
   action or intervention supervision; naming two latent blocks does not make
   them separate automatically.

6. **Joint learning changes the worker's reward coordinates.** If the encoder
   changes during a rollout or between collection and manager recomputation,
   stored goals, observed transitions and recomputed goals can cease to agree.
   Freezing during one batch solves that consistency problem, but online updates
   still change semantics across batches. A slow target encoder helps only if
   goal generation and reward measurements use the same coordinate system.

7. **Per-agent bases complicate interpretation.** The existing `local_private`
   encoder gives each agent its own projection. Its decoder needs the agent
   identity, or a private decoder head, to interpret those coordinates. Keep
   that conditioning limited to identity: giving the decoder the original
   observation or a rich state bypass would weaken the bottleneck. Independently
   decoded local goals may also describe mutually inconsistent team states.
   High diversity of vectors across private bases is not itself evidence of
   diverse physical behaviors.

8. **Freezing trades drift for coverage limits.** Pretraining only on random
   behavior may miss pushing, contact and delivered-object configurations.
   Use a fixed, documented mixture of exploratory and available competent
   behavior; hold out entire episodes/seeds. Track reconstruction on new
   on-policy data. A failed frozen encoder with poor coverage does not settle
   whether reconstruction is useful with adequate coverage.

These are design deductions for this system, not claims that a VAE has already
failed here. [RIG](https://papers.neurips.cc/paper_files/paper/2018/file/7ec69dd44416c46745f6edd947b470cd-Paper.pdf)
is a relevant precedent: it uses VAE mean encodings and latent distances for
goal-conditioned visual control. Its off-policy goal relabeling is not directly
portable to this on-policy worker update. Changing stored goals in a PPO batch
would invalidate the existing behavior-policy probability comparison.

## 4. First implementation

Start with the existing `local_private`/MLP/FiLM family on an MJX task, keeping
the current goal dimension, horizon, worker objective and fusion fixed. Use
`local` as a later shared-basis comparison, not an unannounced simultaneous
architecture change. For SMAX, retain the corresponding global-state input to
goal generation; local reconstruction cannot recover unobserved enemies.

Reuse the existing manager encoder and `latent_only=True` path. Its projection
becomes the posterior-mean head. Add a parallel log-variance head and a decoder
without an observation bypass. In VAE mode, detach the mean at the manager-core
input as well as retaining the existing transition-target detach. The manager
then learns how to use the representation but cannot reshape it through the
goal arm. Pretrain with reconstruction, then freeze **all** encoder parameters
and observation-normalization statistics, not just the last projection.

Use an explicit optimizer partition: encoder/variance/decoder parameters belong
to representation optimization; core/goal-head parameters belong to manager
optimization. Frozen parameters must receive zero optimizer updates, including
momentum or weight-decay updates. Do not rely on zero gradients alone when
resuming an optimizer with nonzero momentum. A frozen parameter subtree in the
existing manager tree avoids extracting a second live encoder or changing the
worker input contract.

Proposed configuration names (not available yet):

```yaml
model_params:
  manager_latent: local_private
  manager_representation: vae       # learned | autoencoder | vae
  manager_core: mlp
  worker_fusion: film
  goal_dim: 32
params:
  representation_update: frozen    # pretrained; online is a later experiment
  representation_checkpoint: <pretrained artifact>
  intrinsic_coef: 0.1
```

Keep the existing intrinsic annealing schedule matched across all main arms.
The base algorithm defaults `intrinsic_coef` to zero; explicitly enable it to
test whether reconstruction improves the worker's progress reward. An alpha=0
ablation is useful but can still change behavior through manager inputs and
goals, so it is not a guarantee of unchanged policy behavior.

| Location | Planned change |
|---|---|
| `algorithms/feudal_mappo_jax/manager.py` | Reuse mean encoder; optional variance/decoder paths; representation-only gradient ownership; preserve full/latent-only mean equality. |
| New representation helper module | Reconstruction likelihood, KL, pretraining update, feature normalization and scalar diagnostics. |
| `mappo.py` | Parameter partitions, pretrained-encoder initialization, freeze enforcement; unchanged directional manager and worker objectives. |
| `trainer.py` | Same deterministic encoder for collection, real successors, evaluation and recomputation; no VAE sampling in those paths. |
| `types.py`, `run.py`, `conf/model/` | Validated opt-in config, Hydra pretraining entry path, distinct results directory, representation metadata and checkpoint support. |
| `run.py` and checkpoint probes | Update all four save/load sites together; record representation kind, architecture and normalization. Do not infer VAE mode only from old projection shapes. |

Keep the default learned representation parameter structure and behavior intact.
VAE checkpoints add parameters and need an explicit format/metadata distinction;
fail clearly on incompatible artifacts. Document the implemented architecture
in `CLAUDE.md` when implementation happens.

If online representation training is later needed, update it only **after**
worker and manager updates finish consuming the old rollout. Use one frozen
encoder/normalizer version for each complete collect-and-update cycle. Keep
stored pooled goals fixed throughout PPO optimization. A reconstruction replay
buffer may store raw observations; it does not authorize replaying old PPO
trajectories. Measure coordinate drift on a fixed held-out observation set.

## 5. Experiment and verification gates

First validate reconstruction and gradients without a long policy run:

- Posterior mean and goals are deterministic; only reconstruction sampling uses
  a random key. Full and latent-only paths produce the same mean.
- Reconstruction updates reach encoder and decoder, not the goal head. Manager
  updates reach the core/head and leave every frozen representation parameter
  unchanged, including after checkpoint resume.
- Existing successor timing, terminal/reset masks, recurrent recomputation and
  checkpoint round trips remain correct. Missing observation channels cannot
  leak through a decoder bypass.
- Held-out reconstruction improves over a per-feature training-set mean
  predictor, including individual task-relevant groups. Check `decoder(mu(x))`
  as well as sampled reconstruction, since control uses means.

Then compare these arms with matched training/evaluation seeds and budgets:

| Arm | Question |
|---|---|
| Current learned representation | Reference for the actual proposed change. |
| Frozen random encoder | Does merely fixing the reward coordinate system help? |
| Pretrained frozen deterministic autoencoder | Does reconstruction add useful information? |
| Pretrained frozen VAE | Does stochastic training plus KL improve on reconstruction alone? |
| Fusion-matched zero-goal worker and flat baseline | Does the hierarchy improve task return? |

The deterministic autoencoder must use `z=mu(x)` during training too. A VAE
with `beta=0` that still samples noise is not that control. Use identical
pretraining data for both learned reconstruction arms; report simulator data
collection and representation compute separately. Fix loss reductions before
choosing beta values, and sweep reconstruction/KL balance on held-out data
rather than selecting it from the final task-return test set.

Retain the current horizon, rollout length and environment count across arms.
Existing repository measurements show that changing rollout length can reverse
the baseline ranking; do not shorten it only for VAE runs. Existing
`local_private` results also show that healthy latent diagnostics and damage
from permuted goals can coexist with no improvement over zero goals
([CLAUDE.md](../CLAUDE.md), measurement dated 2026-09-17).

Measure three distinct outcomes:

1. **Information retained:** held-out per-group reconstruction, variance/rank of
   posterior means, active coordinates and per-coordinate KL.
2. **Control correspondence:** from matched simulator starts, vary commands and
   measure physical changes over horizon c. Report object motion/task progress
   separately from own velocity and sensor changes. Compare decoded candidate
   changes with actual outcomes without treating them as dynamics predictions.
3. **Task usefulness:** real-goal return against zeroed, constant and permuted
   goal evaluations with paired confidence intervals, plus separately trained
   fusion-matched zero-goal and flat baselines. Reuse `goal_dependence_probe`.

Reconstruction improving without better physical control is evidence that the
representation objective needs control information. Better cosine reward alone
does not establish success. A later action-conditioned prediction or matched
command-intervention objective can address that gap; the existing
[controllable latent goals plan](feudal_controllable_latent_goals_2026-09-17.md)
is complementary. Even prediction accuracy is not causal attribution in a
multiagent system, so retain matched interventions when making control claims.
