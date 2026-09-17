# FeUdal intrinsic reward timing fix

Date: 16 September 2026. **Status: IMPLEMENTED 2026-09-16.** Revised before
implementation after reading `intrinsic_timing_probe.py`, `trainer.py` and
`manager.py` — the scope and acceptance criteria changed (see §0). Outcomes are
recorded in §11; the body below is the plan as executed.

Shipped: `manager._intrinsic_window` + `worker_intrinsic_reward_aligned`,
`FeudalManager.__call__(..., latent_only=True)`, `Transition.next_state_latent`,
the capture site in `trainer._env_step`, a three-arm `intrinsic_timing_probe`,
and 20 new seam tests. No config knob, no training A/B — as decided in §0.

## 0. What this is worth, decided before anything is built

**The interior of this fix is already measured, and it is small.**
`intrinsic_timing_probe.aligned_intrinsic_reward` shifts states *and* goals
forward by one step. Working the indices through, that is **identically** the
corrected reward of §2 everywhere except at episode boundaries and the final
rollout step: `r_corrected[t] == r_current[t+1]` in the interior. So the numbers
CLAUDE.md records are not an approximation of this change, they are essentially
the change itself:

| alpha | gradient rotation from fixing the timing | rotation the intrinsic term itself causes |
|---|---|---|
| 0.01 | 0.08 deg | 0.68 deg |
| 0.1  | 1.2 deg  | 6.7 deg  |
| 0.5  | 3.6 deg  | 26.3 deg |

At `n_steps=1048` against 1024-step `trunc` episodes, the exact fix and the
shift differ on roughly **2 of 1048** transitions per env per rollout.

**Do it anyway, for one reason that is not "the indexing is wrong".** The shift
approximation, and every implementation that cannot see past the reset, pays
`r^I = 0` on the action that ENDS an episode. That is the same shape as the
`boundary_truncates` failure CLAUDE.md documents for the env: a standing bonus
for terminating, invisible in the loss. It is 0.2% of transitions on a `trunc`
arm but about **2.4%** on the ~43-step boundary-terminating baseline arm, and it
is a bias with a direction rather than noise. Removing an incentive is worth a
bounded, well-tested change; chasing a 1.2 deg rotation is not.

**Scope consequences, and these are the point of this section:**

* This is **correctness hygiene**. It is not a hypothesis about why positive-alpha
  arms lose to alpha=0 and to the goal-free controls. Per CLAUDE.md the timing is
  ~13% of the effect `r^I` already has, and `r^I`'s own effect is negative.
* **No dedicated training A/B.** The previous revision asked for matched-seed
  old-vs-corrected runs; at 1e8 steps x 3 seeds x 2 arms that is a large budget to
  resolve ~1 deg. Acceptance is structural (§7) plus a probe reading (§8). Ship
  the corrected timing on by default and let the next intrinsic experiment, if
  there is one, run on it.
* **No config knob for the legacy timing.** CLAUDE.md records three separate
  silent-config failures in this repo (the `VARIANTS` int-vs-enum guard, the
  missing `# @package _global_` that cost 12 runs, the misindented `n_steps`). A
  `reward_timing: legacy|aligned` field in `Model_Params`/`MAPPOConfig` buys a
  fourth opportunity and, with the A/B dropped, has no consumer. The legacy
  function stays in the probe, clearly labelled, where it cannot reach training.

## 1. Objective and scope

Make transition `t`'s intrinsic reward score the outcome of action `a_t`,
including the last action before termination, reset, or rollout end.

Hold alpha, goal conditioning, fusion, critic architecture, advantage
normalization and the manager objective fixed, so the timing change is the only
difference. Improving the intrinsic critic's history inputs is a separate change
and is explicitly out of scope — `intrinsic_explained_variance` already reads
0.96-0.99 on the trained arms, so the non-Markov history dependence of `r^I` is
empirically recoverable from the current state and is not the binding problem.

Relevant files:

- [trainer.py](../algorithms/feudal_mappo_jax/trainer.py)
- [manager.py](../algorithms/feudal_mappo_jax/manager.py)
- [types.py](../algorithms/feudal_mappo_jax/types.py)
- [test_feudal_seams.py](../algorithms/tests/test_feudal_seams.py)
- [intrinsic_timing_probe.py](../algorithms/feudal_mappo_jax/intrinsic_timing_probe.py)

## 2. The defect and the corrected reward

`trainer._env_step` stores the pre-action manager latent as `state_latent[t]`
(trainer.py:263, :391). `manager.worker_intrinsic_reward` builds

```text
r^I_t = 1/c * sum_{i=1..c} d_cos(s_t - s_{t-i}, g_{t-i})
```

onto the transition carrying `a_t`. Every term is fixed before `a_t` is sampled,
while `reward[t]` on that same transition IS `a_t`'s consequence. The two streams
score different actions.

Let `s_plus[t]` be the latent of the **actual successor** produced by `a_t`,
read before any reset. The corrected reward is

```text
r_int[t] = sum_{k=0..c-1} mask[t,k] * d_cos(s_plus[t] - s[t-k], g[t-k])
           / max(1, sum_{k=0..c-1} mask[t,k])
```

* `mask[t,k]` requires the origin to exist (`t - k >= 0`) and to lie in the
  episode `a_t` was selected in: no done in `done[t-k : t]`, i.e. `done[t]` is
  **excluded**. The action that ends an episode still earns credit for its own
  terminal successor. `k = 0` has an empty interval, so it is always valid — which
  is what lets the first action of an episode earn a nonzero reward.
* The endpoint needs no separate episode test: `s_plus[t]` is by construction the
  pre-reset successor of transition `t`'s own episode. Capturing it before the
  reset (§3) is what makes that true; it is the crux of the reset-isolation test.
* Keep the existing denominator (average over valid terms, not a fixed `c`) and
  the existing zero-displacement stabilization in `cosine_similarity`, so only the
  indexing changes.
* At `c = 1` this reduces to `d_cos(s_plus[t] - s[t], g[t])`.
* Gradients are stopped through all three inputs, as `worker_intrinsic_reward`
  already does unconditionally. Note this is belt-and-braces: the trajectory is a
  constant to `update_fn`, so there is no gradient path to cut.

**Logged-series note.** `intrinsic_reward` / `intrinsic_reward_abs` will shift
level slightly (the `k=0` term is new at episode starts). Historical series stay
readable but are not point-comparable across the change; record the date.

## 3. Capture the successor latent before the reset

In `_env_step`, between `v_step` and the `_restart_done` `lax.cond`:

1. Read the pre-reset successor pair. `next_gs` is already computed there for the
   truncation bootstrap; `next_obs` is in scope and still pre-reset.
2. Encode it with the manager params into `s_plus`, and store as
   `Transition.next_state_latent`.

**Use a latent-only path through the manager, not a full forward.** In all five
latent variants `s` is computed strictly upstream of the core (`manager.py`:
centralized `f_percept -> f_Mspace`; local family `f_enc -> f_Mspace` or
`f_Mspace_agent`) — feedforward, no carry, no goal head. Add a static
`latent_only: bool = False` argument to `FeudalManager.__call__` that returns `s`
immediately after that block. Under `apply` with an existing param dict the
later params simply go unread; `init` always runs the full path, so the tree is
unchanged. This is not cosmetic:

* it removes the "must not advance the live carry / consume action RNG" hazard
  structurally rather than by discipline (and that hazard was overstated — JAX is
  functional, so a discarded carry was never a mutation);
* it roughly halves the added compute (no LSTM/MLP core, no goal head, no
  `_unit`);
* it makes the `local*` branches read only `next_obs`, and `centralized` /
  `local_global` only `next_gs`, matching what each actually needs.

**The one real trap:** `_restart_done` rebinds `next_obs` and `next_env_state` in
place. Encoding after the `lax.cond` silently yields the **reset** latent and
manufactures exactly the artificial reward test §7.5 exists to catch. Put the
encode above the cond and comment why, in the style of the existing `next_gs`
comment block.

**Leave `state_latent` alone.** It must stay the pre-action latent:
`manager_update` recomputes exactly it, differentiably, from the stored
`global_state`/`obs`, and `test_goals_are_reproducible_from_stored_states`
(test_feudal_seams.py:254) pins that equality. `next_state_latent` is a new
field, not a replacement.

**Gate on the existing static `use_intrinsic` flag** (trainer.py:96). When alpha
is 0, skip the encode and store the repo's scalar-placeholder idiom
(`jnp.zeros(())`, as `action_mask` does) rather than a real `(E, N, D)` buffer.

**Budget, to be confirmed not assumed.** Compute: one 2-layer encoder + one
projection per step, negligible against an `mjx.step`. Memory: a third
`(T, E, N, goal_dim)` f32 buffer — at `T=1048, E=32, N=12, goal_dim=32` that is
~51 MB, on top of the `goal`/`state_latent` pair CLAUDE.md already names as the
peak-memory driver of the manager BPTT. Measure peak memory and collect time
before and after; do not refactor the parameter tree to save it.

## 4. Wire the transition-aligned helper into training

Add `worker_intrinsic_reward_aligned(states, next_states, goals, horizon, done)`
in `manager.py` alongside the existing helper, and call it from
`trainer._apply_intrinsic_reward`. The existing helper stays — it is still the
correct object for the manager-side diagnostics and for the probe's legacy arm —
but production training must use the aligned one.

Bootstrap semantics are unchanged and must stay exactly once each:

| Transition | Intrinsic target handling |
|---|---|
| Ordinary step | Corrected reward; GAE supplies the successor value. |
| True termination | Corrected terminal reward; no continuation value. |
| Time-limit truncation | Corrected reward plus `intrinsic_bootstrap` (`gamma * V^I(s_next)`, already computed in-scan against the pre-reset successor), once. |
| Nonterminal rollout end | Corrected final reward; GAE uses `Bootstrap.worker_int`. |

Update the `Transition` field comments in `types.py` to distinguish pre-action
state, actual pre-reset successor, and post-reset observation — the three are now
all present in the same function and the names do not separate them.

## 5. What deliberately does NOT change, and why

**The manager's transition objective.** `transition_cosine` scores
`d_cos(s_{t+c} - s_t, g_t)` against the manager advantage at `t`. That window
**contains** the effects of `a_t .. a_{t+c-1}`, every one of which `g_t`
influenced through the pooled goal; the worker's window `[t-i, t]` contained
**none** of `a_t`. So the manager side is aligned in the sense this plan is
fixing, and the residual there (whether the endpoint should be `s_{t+c+1}`) is a
1/c effect against the worker's 1/1. Named here so it is a recorded decision
rather than an unexamined scope boundary.

**`V^I`'s inputs**, per §1. **Fusion, recurrence, goal representation, alpha and
its schedule**, so this change can be attributed.

## 6. Correct the timing diagnostic

`aligned_intrinsic_reward` currently shifts arrays and zeros the last entry.
Replace it with the production helper driven by real `next_state_latent` values,
so boundaries are covered. Keep the shifted version as an explicitly labelled
`legacy_shift_approximation` third arm — the gap between it and the exact fix is
precisely the terminal-action credit this change exists to restore, and it is the
only quantity that isolates that.

Two corrections to the probe's method:

* **Action-dependence must step the env.** Re-running the stored trajectory with
  a substituted action while holding stored states fixed cannot show causal
  dependence. Step from the same state under two different actions and compare
  the resulting successor-based rewards.
* **Match production when comparing gradients**: same bootstrap, same
  per-(env, agent) normalization, same masks, same alpha semantics. Label legacy
  and corrected results explicitly in the output.

**Self-check with a prediction, not a discovery.** On a trained checkpoint the
corrected-vs-legacy gradient angle should land near the table in §0 (0.08 / 1.2 /
3.6 deg at alpha 0.01 / 0.1 / 0.5), plus a small boundary term. A materially
larger number means the implementation is wrong — most likely the reset-ordering
trap of §3 — not that the bug was bigger than measured.

**Also measure at an early checkpoint.** The reason the shift is nearly harmless
is `r^I`'s lag-1 autocorrelation of 0.53-0.73 combined with GAE integrating over
~16.8 steps, both measured at *trained* checkpoints at the pre-anneal alpha.
Early in training the manager moves faster and that autocorrelation may be lower,
which is the one regime where the timing could matter more than recorded. This is
the cheapest open question the probe can answer and it should be answered here.

## 7. Tests

Use a tiny deterministic environment where the action moves position directly and
the latent is a known function of it. **Compute expected values independently of
the production helper.** The existing
`test_intrinsic_stream_is_separate_and_exact` (test_feudal_seams.py:617) checks
the collector against the same `worker_intrinsic_reward` the collector calls —
that is a wiring test, not a semantics test, and it must not be the model for the
new ones.

1. **Immediate action credit.** `c=1`, eastward goal: east / north / west actions
   give approximately `+1 / 0 / -1` on that same transition.
2. **First action.** Step 0 of an episode earns nonzero reward (`k=0` is valid).
3. **Longer horizon.** A hand-computed trajectory verifies goal indices,
   displacement origins, and averaging over valid terms only.
4. **Terminal successor.** The episode-ending action is paid against its real
   terminal successor, not zero. This is the regression that motivates the change.
5. **Reset isolation.** A deliberately distant reset position produces no reward;
   the new episode cannot reach old origins or old goals.
6. **Final rollout step.** Its actual outcome is included.
7. **Truncation.** `intrinsic_bootstrap` is added exactly once against the true
   successor; a true termination carries no continuation value.
8. **Latent-only path.** For every latent variant and both cores,
   `__call__(..., latent_only=True)` returns `s` bit-identical to the full
   forward's third output, and leaves the caller's carry untouched.
9. **Alpha-zero is bit-identical.** Same seeds, same rollout and update arrays
   exactly (the repo's idiom, and stronger than "numerics preserved"): the encode
   is statically skipped and consumes no RNG. Extend the existing alpha-zero
   comparison rather than adding a parallel one.
10. **No gradient into the manager** from the reward inputs. Cheap, and it
    documents the invariant even though the trajectory is already a constant to
    `update_fn`.

Not a test: **checkpoint compatibility.** `Transition` is rollout-local and never
serialized — only `FeudalTrainState` reaches the five msgpack sites in `run.py`.
It is preserved iff the parameter tree is untouched, which §3's `latent_only`
early return guarantees by construction. Assert the param-tree invariance
instead, and state the reason in the plan rather than paying for a load test that
cannot fail for this change.

Retain the existing goal-pooling, goal-recomputation, agent-major pairing and
PPO-ratio-equals-1 checks unchanged.

## 8. Validation

1. `uv run pytest algorithms/tests/test_feudal_seams.py -q` on CPU (the autouse
   `_run_on_cpu` fixture is already there) plus the new cases.
2. `uv run python -m algorithms.feudal_mappo_jax.manager` self-checks.
3. Short positive-alpha smoke train + checkpoint resume: finite rewards, updates
   apply, collect time and peak memory recorded against the pre-change numbers.
4. Probe run per §6 on a trained arm and on an early checkpoint, reporting
   legacy / shift-approximation / corrected.

## 9. Completion criteria

Every intrinsic reward is attached to its causing action, terminal, truncated and
final-rollout actions included. No reset leakage, no duplicate bootstrap, no
parameter-tree change. Alpha-zero runs are bit-identical. The probe's
corrected-vs-legacy angle reproduces §0's table within a small boundary term.

Improved task return is **not** a criterion and is not expected: per §0 this
moves the update by ~1 deg at alpha=0.1, and every positive-alpha arm measured
loses to alpha=0 and to the goal-free controls. Claiming otherwise afterwards
would be reading noise.

## 10. Documentation to update when this lands

* **CLAUDE.md**, FeUdal section: the paragraph beginning "The `r^I` timing
  misalignment is REAL but SMALL" describes the *current* behaviour and becomes
  false. Rewrite it as the rationale for the fix (keeping the measured table,
  which stays the best estimate of what the fix did) and record that pre-fix
  `intrinsic_reward` series are not point-comparable to post-fix ones.
* `algorithms/feudal_mappo_jax/ALGORITHM_DIAGRAM.md` — the intrinsic stream.
* `types.py` `Transition` docstring/comments, per §4.
* `intrinsic_timing_probe.py` module docstring: it currently describes the
  defect in the present tense.

## 11. Outcomes (2026-09-16)

**Correctness.** 123/123 `test_feudal_seams.py` (was 103; +20, of which 10 are
the `latent_only` sweep over 5 latents x 2 cores), 11/11 `test_smax_seams.py`,
all 10 `manager.py` self-check groups. The reward helpers were checked against a
straight python transcription of the definition (max err 3e-08), not against
each other.

**alpha=0 is bit-identical to the pre-change code.** Git-worktree A/B on the CPU
stub env: all 18 rollout fields, all 23 loss/metric keys and all 95 post-update
param/optimizer leaves at 0.0 max diff. Parameter tree unchanged, so existing
checkpoints load — re-verified by loading trained `feudal_film` (centralized)
and `feudal_film_n01_local` manager params and confirming `latent_only` returns
bitwise the full forward's `s`.

**Cost** (`mjx_12a_3o_trunc_1024`, n_steps=1048, n_envs=32, alpha=0.1, warm
median of 7): collect **2.581 s -> 2.562 s**, i.e. no measurable time cost; peak
GPU **1652 -> 1751 MiB** (+6%, the new `(T,E,N,goal_dim)` buffer — about 2x the
51 MB estimated in §3, which counted the stored buffer and not its transient).

**Smoke train.** `feudal_film_n01_local` at alpha=0.1, 67k steps then resumed to
101k with `checkpoint=true`: trains, resumes, all stats finite and aligned to
`total_steps`, `intrinsic_reward_abs` 0.13 (in line with the ~0.155 on record).

**Probe, and the §6 prediction held.** `feudal_film_n01`/`n05` trial 0:
legacy-vs-corrected gradient cosine **0.999951 / 0.998870** (~0.57 deg, 2.7 deg)
against the 1.2 / 3.6 deg predicted from the pre-fix measurement, and
legacy-vs-shift 0.999934 / 0.998776 — the two agree, as §0 argued they must.
`corr(r^I legacy, corrected)` = **0.7213** against a lag-1 autocorrelation of
**0.7210**: the mechanism reproduced to three decimals. Causal dependence on
`a_t` now demonstrated by stepping the env under two actions from one state
(delta r^I ~ 0.72).

⚠ **One finding that qualifies the motivation.** Re-run at `--n-steps 1100` so
episodes actually end (4 dones, 5 boundary transitions of 4400), the
shift-vs-corrected gradient cosine is **0.999998** — the terminal-credit
correction, which §0 named as the whole reason to do this, is invisible in the
update on a `trunc` arm. That does not retract the argument: a gradient cosine
measures how much the update moves now, not the incentive a systematic zero on
the episode-ending action creates over a run, and `boundary_truncates` is this
repo's own record of that class of incentive being self-sealing. But it does
mean the boundary claim is **structural, not measured**. The arm where it would
be visible is the ~43-step boundary-terminating baseline (~2.4% of transitions
against 0.11% here), and it has not been measured there.

**Not done, deliberately:** no training A/B (§0), no config knob (§0), no early-
checkpoint probe run (§6 — still the cheapest open question here), no measurement
on a boundary-terminating arm.
