# Difference rewards under asynchronous macro-actions

Status: **ABANDONED as a contribution.** The measurement stack works and the results
below are real, but they do not support a paper. Kept as (a) a record of why, and
(b) documentation of the oracle machinery, which is the reusable asset.

## Why this was abandoned (read this first)

**The headline result is near-tautological.** "Assuming synchrony in an asynchronous
setting produces the wrong difference reward" is true by inspection — the estimator is
fed a commitment state known to be wrong, so of course its output is wrong. The
control returning exactly 0.0 is arithmetic (`aligned_belief` is provably the identity
under synchrony), not evidence. And `D_sync` is a **strawman**: nobody deploys "assume
lockstep when agents demonstrably are not"; real baselines (COMA, Dr.Reinforce) learn
from data and would absorb some async structure implicitly.

**The non-tautological rescue was tested and falsified.** The salvageable claim was
that under synchrony there is a *canonical* counterfactual scope (every commitment
spans [t, t+L)) while under asynchrony no window contains "the current joint
commitment", so defensible scope choices would disagree about who mattered — an
ambiguity with no wrong answer, hence not tautological. Measured
(`scratchpad/scope_ambiguity.py`, cross-horizon agreement of `D_oracle`, 10 states,
H in {10,20,40,80}):

| | sync | async |
|---|---|---|
| mean cross-horizon pearson | 0.626 | **0.680** |
| widest gap (H=10 vs H=80)  | 0.396 | **0.423** |

Async credit is *slightly more* scope-stable than sync. There is **no async-specific
scope ambiguity** — credit is horizon-sensitive in both, which is the ordinary fact
that different horizons measure different things (true of single-step synchronous
MARL too). The framing does not survive.

**What is worth keeping:** the oracle (`oracle.py`) — exact difference rewards by
forking a pure functional simulator. That is the real asset and it does not depend on
asynchrony. The open, non-tautological question it enables is *how well do learned
counterfactual baselines (COMA's, Dr.Reinforce's, `hg_cache.py:414`'s) actually
approximate the true difference reward* — nobody can normally check, because their
envs cannot rewind.

---

## Original framing (superseded — retained for context)

## The question

Do difference rewards `D_i = G(z) - G(z_-i + c_i)` break when agents' macro-action
commitments are **asynchronous** — when agent *i* commits at *t* for 7 steps while
*j* committed at *t-3* for 9?

They should, in principle: `D_i` assumes you can swap agent *i*'s action while
holding others' fixed, but under asynchrony "others' actions" is not a fixed object
— it is a set of partially-elapsed commitments with differing remaining times. The
reward is meanwhile an *integral* over a window (box displacement per step), smeared
across asymmetrically overlapping commitments.

This was worth *measuring* before building anything, because the honest alternative
was that the gap is negligible and the whole idea is a re-skin of Dr.Reinforce
(Castellini et al. 2021) / COMA (already vendored at
`algorithms/dcg/src/learners/coma_learner.py`; a first-party counterfactual baseline
already exists at `algorithms/mappo/hg_cache.py:414`).

## Why this repo can answer it exactly

`MultiBoxPushMJX` is a **pure functional env** — `EnvState` is a registered dataclass
threaded through `step`. So a state can be *forked* and replayed under a
counterfactual, giving the **exact** `D_i` with no learned model. Prior
difference-reward work approximates precisely because their envs cannot rewind.

## What was built

- `environments/mjx_suite/macro_skills.py` — 4 scripted, deterministic JAX skills
  (`contact`, `push`, `scatter`, `rendezvous`) + `null_action` (the counterfactual
  default `c_i`, deliberately not policy-selectable).
- `environments/mjx_suite/macro_wrapper.py` — `AsyncMacroMJX`: per-agent
  `(skill, elapsed, remaining)` commitment state over the base env. One `step` is one
  *low-level* step; only agents whose commitment expired adopt the proposed skill, so
  decision points decouple while shapes stay static for jit. `d_min == d_max` +
  `stagger=False` reproduces the box2d `HierarchicalSkillEnv` lockstep exactly (the
  control).
- `algorithms/difference_rewards/oracle.py` — exact `D_i` by forking + vmapping over
  `[-1, 0..A-1]` in one compiled call, under common random numbers.
- `algorithms/difference_rewards/bias_study.py` — the falsification.

## Results

Control validity (`aligned_belief` is provably the identity under true synchrony, so
this must be exact, not merely small): **pearson 1.0000, sign agreement 100.0%,
norm_bias 0.0000 at every horizon tested.** Harness valid.

9 agents / 3 objects, L=10, 24 states, per-state metrics, noise floor 5e-3:

| condition      | pearson | sign agree | norm_bias |
|----------------|---------|-----------|-----------|
| sync (control) | 1.0000  | 100.0%    | 0.0000    |
| stagger only   | 0.2041  | 71.6%     | 1.21      |
| spread 8-12    | 0.3183  | 72.6%     | 0.89      |
| spread 5-15    | 0.3536  | 76.1%     | 1.01      |
| spread 2-18    | 0.3069  | 76.7%     | 0.98      |

Horizon sweep (12 states):

| H   | pearson (async) | sign agree | norm_bias |
|-----|-----------------|-----------|-----------|
| 20  | 0.43 – 0.54     | 70 – 79%  | 0.65 – 0.84 |
| 60  | 0.20 – 0.35     | 72 – 77%  | 0.89 – 1.21 |
| 120 | 0.06 – 0.38     | 57 – 71%  | 0.70 – 1.52 |

### Findings

> **Caveat added after review: finding 1 is near-tautological** (a knowingly-wrong
> commitment state produces wrong credit) and the control is arithmetic, not
> evidence. See "Why this was abandoned" at the top. The findings are reported
> accurately; they just do not amount to a contribution.

1. **Any asynchrony breaks the synchronous estimator.** Correlation with ground
   truth falls from exactly 1.0 to ~0.2-0.35; `norm_bias ~ 1.0` means the error is
   *the same magnitude as the quantity being estimated*; the sign of an agent's
   credit is wrong ~25% of the time.
2. **It is phase misalignment, not duration variance.** The original hypothesis
   ("bias grows with duration variance") is **falsified**: bias is flat (~1.0) across
   every async condition. `stagger only` — identical durations, offset phases — is
   just as broken as `spread 2-18`. Sync -> async is a *step function*. This is a
   sharper claim than the one planned: the contribution does not depend on exotic
   duration distributions.
3. **The damage compounds with the counterfactual horizon.** Longer windows contain
   more re-decisions, so the believed-aligned schedule drifts further from the true
   staggered one. At H=120 sign agreement is 56.6% — near chance. Horizon (=
   counterfactual scope) is the real monotone axis.

### Caveats (do not overstate these results)

- Measured under **scripted** skills and a **uniform-random** behaviour policy, not a
  trained one. On-policy state/action distributions may differ.
- Shows the estimator is **biased**; does **not** yet show this *degrades learning*.
  That is the Phase 3 claim and is untested.
- One env, one team size, one L. `pearson ~0.3` is degraded, not zero — some signal
  survives.

## Non-obvious things learned (traps)

- **Re-decisions are load-bearing.** With skills frozen for the counterfactual
  window, `remaining`/`elapsed` never touch the dynamics, `D_sync == D_oracle`
  *identically*, and the study reports zero bias for a reason unrelated to the
  question. Durations only bite through *when agents re-decide*.
- **`aligned_belief` must collapse phase to the joint mean**, not reset it to a
  nominal `L`. Otherwise the control (where all phases already agree but are mid-
  window) shows spurious non-zero bias.
- **Measure at engaged states.** At reset every `D_i` is 0 (nothing has touched a
  box), so any attribution assertion passes vacuously. First verification run was a
  false pass for exactly this reason. Warm up under `contact` first.
- **Skills must sense locally.** A global `_others_centroid` gave an agent parked
  20 units away (outside `sector_sensor_radius`=12, physically unable to reach
  anything) a real causal channel into every teammate: oracle reported |D| = 0.16,
  ~5% of max and larger than genuinely engaged agents. Local sensing dropped it to
  2e-4. It also contradicted the observation model. After the fix, `scatter` also
  stopped dragging agents off the boxes and deliveries began (G: 5.9 -> 107).
- **`scatter` used to suicide into the walls** (episodes ended at ~21 steps, reward
  0 via `boundary_hit`). A skill that ends the episode makes `D_i` measure "did agent
  i end the episode" rather than credit. `_wall_repulsion` fixed it; `push`'s return
  also went 2.6 -> 31.3.
- **vmap vs. un-vmapped rollouts drift ~1e-3** (float reduction order, chaos-
  amplified over ~120 steps). Not a bug and not how `D` is computed — inside
  `difference_rewards` all branches share one vmap, so `G[0]` is a batching-
  consistent reference for `G[1:]`. It does set the noise floor that `|D_i|` must
  clear (~5e-3 used).
- Confirms CLAUDE.md's warning: MJX is **not reproducible across processes** (the
  same script gave G=5.0 vs G=5.9 on two runs). In-process it is bitwise
  reproducible, which is exactly why the forked oracle is valid.
- `jax.jit(env.step)` on a bound method rebuilds the wrapper each call -> jit cache
  miss -> full recompile per sampled state. Cache it (`bias_study._jitted`).

## Reproduce

```
MUJOCO_GL=egl uv run python -m environments.mjx_suite.macro_wrapper       # Phase 0
MUJOCO_GL=egl uv run python -m algorithms.difference_rewards.bias_study \
    --n-states 24 --horizon 60                                            # Phase 2
```

## Where to go instead

The async framing is dead; the **oracle is the asset**. It does not depend on
asynchrony at all. The non-tautological questions it unlocks — no strawman required,
because the comparison is against methods people actually deploy:

- **(a) Audit learned counterfactual baselines against ground truth.** Everyone
  assumes COMA's counterfactual baseline / Dr.Reinforce's reward model approximate the
  true difference reward. How well, actually? Nobody normally checks, because their
  envs cannot rewind. This repo has the oracle *and* vendored COMA
  (`algorithms/dcg/src/learners/coma_learner.py`) *and* a first-party hyperedge
  counterfactual (`algorithms/mappo/hg_cache.py:414`). Needs a literature check first
  — some COMA analysis exists — but a direct oracle comparison in a rewindable
  simulator appears rare.
- **(b) Value of perfect credit.** Train with oracle `D_i` (affordable: MJX vmaps the
  forked counterfactuals) to get an *upper bound* on what credit assignment can buy.
  How much performance is approximate credit leaving on the table?
- **(c) Non-additivity under strong coupling.** Observed here: with 3-agent coupling,
  two agents each had `D_i ~ 102` for the same +100 delivery, and `sum_i D_i = 306`
  vs `G = 107`. Each is individually necessary. Known in principle (Wolpert-Tumer);
  measurable exactly here.

If per-agent rewards are ever needed in `mappo_jax`: `Transition.reward` `(n_envs,)`
-> `(n_envs, n_agents)` (`algorithms/mappo_jax/types.py:76`), `compute_gae` is
shape-agnostic (`mappo.py:116-126`), drop the broadcast at `mappo.py:219`. The real
cost is the **centralized single-scalar critic** (`mappo.py:62,244-247`), which needs a
per-agent value head.
