# Pure-intrinsic worker arm (`worker_objective: intrinsic_only`)

## Context

Today the feudal worker optimizes a **mixture** of two normalized advantage
streams ([mappo.py:369-389](algorithms/feudal_mappo_jax/mappo.py#L369-L389)):

```
A_t_i = A_ext_norm_t_i  +  alpha_t * A_int_norm_t_i
```

So the worker always carries its own task gradient, and the manager's goals are
at best an auxiliary pressure on top of it. Every measurement in CLAUDE.md is
consistent with the goals being *decorative*: `eval_gap_permuted ~ 0` on 40 of
45 non-control trials (the 5 with a CI excluding 0 go in both directions),
`gap_zeroed` systematically **negative** (zeroing the goal
*improves* 15 of 16 trained arms), and the best arm in the `mjx_12a_3o` batch is
`feudal_film_zerogoal_dilated`, whose goals are provably disconnected.

**What is untested is the other extreme**: a worker whose *only* objective is to
follow the manager. That is the classic Dayan–Hinton feudal contract and it
makes the hierarchy load-bearing by construction — the manager holds all task
pressure (its PG is already weighted by the extrinsic advantage under
`manager_gamma`, [mappo.py:1052-1071](algorithms/feudal_mappo_jax/mappo.py#L1052-L1071)),
and task performance can only be reached *through* the goal channel. It also
converts the probe's "are the goals used?" question from an open one into a
tautology, which changes the acceptance criterion (see below).

Requested outcome: a new arm where the worker's loss signal is purely `r^I`.

---

## Design

### The one-line core change

`ppo_update` gains a static, python-level branch on a new
`MAPPOConfig.worker_objective`:

| value | advantage fed to the actor | notes |
|---|---|---|
| `"mixed"` (default) | `A_ext_norm + alpha_t * A_int_norm` | **byte-identical to today** |
| `"intrinsic_only"` | `A_int_norm` | no alpha factor — see below |

The extrinsic GAE, `returns`, the worker critic's regression and
`explained_variance` are **all kept**. Only the *actor's* advantage changes.

### Five decisions and why

**1. `alpha` must be removed from the expression, not set to 1.**
`A_int_norm` is already unit-std, so multiplying by `alpha_t` is a uniform
rescale of the whole actor gradient — nearly a no-op under Adam, *except* that
the shipped `intrinsic_anneal: linear` decays it to **exactly 0** at the end of
training. The worker's entire objective would silently vanish over the second
half of the run while every logged loss stayed healthy. This is the same
self-sealing shape as the `boundary_truncates` and `VARIANTS`-enum bugs. So:
`intrinsic_only` uses `adv = adv_int` with no coefficient, and `run.py`
**raises** on `intrinsic_anneal != "none"` for this objective.

**2. `intrinsic_only` requires `intrinsic_coef != 0.0`.**
`use_intrinsic` is the static gate that builds `intrinsic_critic_ts`, captures
`next_state_latent` and computes `r^I` at all
([trainer.py:96](algorithms/feudal_mappo_jax/trainer.py#L96)).
At alpha=0 none of it exists, so `adv = adv_int` would be a zero array — the
actor would train on nothing. `run.py` raises. (This also means the existing
`zero_goal && intrinsic_coef != 0` guard already covers
`zero_goal` + `intrinsic_only` for free — pinned by a test rather than a new guard.)

**3. The worker's extrinsic critic keeps training.** It costs
`n_epochs * n_minibatches` gradient steps per update on a value function the
actor no longer reads, and buys three things: the param tree stays
**shape-identical** to the matched control (so checkpoints remain
interchangeable, the same reason `zero_goal` zeroes at the *input*);
`explained_variance` stays a live, cross-arm-comparable readout of whether task
return is even predictable during the run; and the extrinsic truncation
bootstrap at [trainer.py:369](algorithms/feudal_mappo_jax/trainer.py#L369)
keeps working unchanged.

**4. The entropy term stays.** `ent_coef * entropy_loss` is exploration
regularization, orthogonal to which reward is being maximized.

**5. Two unambiguous weight metrics replace a misleading `alpha_current`.**
`alpha_current` would read as the intrinsic weight while being inert. Log the
coefficients that actually appear in the expression:

| metric | `mixed` | `intrinsic_only` |
|---|---|---|
| `adv_ext_weight` | 1.0 | 0.0 |
| `adv_int_weight` | `alpha_t` | 1.0 |

`alpha_current` is kept as-is for `mixed` so no existing plot changes.

---

## Why this arm is risky, and what to watch

Three concerns, in order. None of them is a reason not to run it, but the first
is a reason to distrust a *good-looking* result.

**(a) The shared-cosine fixed point becomes the worker's global optimum.**
CLAUDE.md records this measured failure: manager and worker share `d_cos` as an
objective and climb it *through the environment* (no gradient crosses FuN's
detach), so under `local` + alpha>0 they jointly collapse — manager freezes on one
direction, worker drives its own observation along it, `goal_direction_count`
1.45, task return 1.4. With `intrinsic_only` there is nothing else in the
worker's gradient, so that degenerate joint solution is not merely reachable, it
is *optimal for the worker*. The counter-pressure is that the manager's PG is
weighted by the extrinsic advantage, so the manager is not free to collapse —
but there is a feedback risk: a perfectly obedient worker drives
`d_cos -> 1` everywhere, `d_cos_var -> 0`, and the manager's own gradient
flattens (the documented `d_cos_var <~ 1e-3` detector).

**Watch, in this order:** `d_cos_var` (must stay off the floor),
`goal_direction_count` (against the random baseline
`N^2/(N + N(N-1)/goal_dim)` = 8.93 at N=12/goal_dim=32), `state_latent_erank`,
then return. Read them *across the whole run* — `local`'s collapse happened by
~10M steps.

**(b) `r^I` must be agent-local or the worker farms a team signal it does not
control.** Measured 2026-09-09 over 12 arms: under `manager_latent:
centralized` the diagonal share of `d s[i]/d obs_j` is **0.0631** against a
uniform `1/N` of 0.0625 — i.e. no localization at all. Under the current mixed
objective that is a confound; under `intrinsic_only` it is fatal, because that
non-local signal would be the worker's *entire* loss. `local_private` is the
only latent measured to have both locality (diag share exactly 1.0) and restored
row diversity (`s` 8.72, `g` 9.24). **This is a precondition, not a preference**,
and it is why `local_private` is the arm being built.

It is enforced as a **loud warning, not a raise** — the combination stays
available as a deliberate ablation, but an accidental one is obvious in the log.
Membership reuses the existing
`manager.LOCAL_LATENTS` tuple ([manager.py:124](algorithms/feudal_mappo_jax/manager.py#L124)),
not a hardcoded list, so it cannot drift as latents are added — the same reason
that tuple was introduced for `PRIVATE_LATENTS` / `GLOBAL_LATENTS`.

**(c) The goal-dependence probe stops being the acceptance test.**
`gap_zeroed` / `gap_constant` / `gap_permuted` will all be large by
construction — the worker is *defined* to depend on the goals. So will
`d_cos_mean`. The probe can no longer rule this arm in or out. Acceptance
becomes the **between-arm return comparison**, which CLAUDE.md already names as
the only test that can rule an arm *in*:

| comparison | question |
|---|---|
| vs `feudal_film_local_private` (alpha-matched, `mixed`) | does removing the worker's own task gradient help or hurt? |
| vs `feudal_film_zerogoal` | does it beat the goal-free floor? |
| vs `mlp` (`algorithm=mappo_jax`) | does it beat flat MAPPO? (272 / 295 on the 12a groups) |

---

## Files to change

| file | change |
|---|---|
| `algorithms/feudal_mappo_jax/types.py` | `Model_Params.worker_objective: str = "mixed"` (next to `zero_goal`/`worker_fusion` — it is an arm-defining axis, like those, not a shared-across-arms diagnostic knob like `goal_permute_shift`); mirror onto `MAPPOConfig` |
| `algorithms/feudal_mappo_jax/mappo.py` | in `ppo_update`, branch the `adv` expression (~[L369-L389](algorithms/feudal_mappo_jax/mappo.py#L369-L389)); add `adv_ext_weight` / `adv_int_weight` to `int_metrics` |
| `algorithms/feudal_mappo_jax/run.py` | thread `worker_objective` into `MAPPOConfig` (~L268); guards next to the existing `worker_fusion` / `zero_goal` ones (~L275-L289) — **raise** on an unknown value, on `intrinsic_only` + `intrinsic_coef == 0`, and on `intrinsic_only` + `intrinsic_anneal != "none"`; **warn** on `intrinsic_only` + `manager_latent not in manager.LOCAL_LATENTS`. Add to the banner print and to `evaluate_goal_dependence`'s provenance dict (~L1047) |
| `conf/model/feudal_film_intrinsic_only_local_private.yaml` | **new, and the only new config** — `defaults: [feudal_film_local_private, _self_]`, `model_params.worker_objective: intrinsic_only`, `params.intrinsic_coef: 1.0`, `intrinsic_anneal: none`. **Must carry `# @package _global_`** (the header whose absence cost 12 runs). A centralized-latent contrast is a 4-line copy if wanted later; not built, per the locality precondition above |
| `algorithms/tests/test_feudal_seams.py` | new tests (below) |
| `CLAUDE.md` | new subsection under "Intrinsic reward (`intrinsic_coef`)" recording the objective, the three guards, the collapse risk and the changed acceptance criterion |

No change to `manager.py`, `worker.py`, `trainer.py` or the checkpoint format.

---

## Tests (`algorithms/tests/test_feudal_seams.py`)

Mirroring the existing `_config()` / `_collect()` StubEnv fixtures:

1. `test_mixed_objective_is_unchanged` — `worker_objective="mixed"` reproduces
   today's combined advantage bitwise (the no-op guarantee, same shape as
   `test_real_eval_variant_is_unchanged`).
2. `test_intrinsic_only_drops_the_extrinsic_advantage` — scaling the stored
   extrinsic reward by 1000x leaves the actor's post-update params unchanged,
   while scaling the intrinsic stream does not. This is the direct analogue of
   the existing `test_alpha_is_a_gradient_fraction_not_a_reward_coefficient`.
3. `test_intrinsic_only_ignores_alpha` — post-update actor params are identical
   at `intrinsic_coef=0.1` and `1.0` under `intrinsic_only` (same rng), proving
   the anneal cannot silently zero the objective.
4. `test_intrinsic_only_still_trains_the_worker_critic_and_the_manager` — the
   extrinsic critic params and both manager states still move, and the param
   tree is shape-identical to the `mixed` arm.
5. `test_intrinsic_only_requires_a_live_intrinsic_stream` /
   `test_intrinsic_only_rejects_annealing` / `test_unknown_worker_objective_raises`
   — the three raising `run.py` guards (tested at the `MAPPOConfig` validation
   site so they need no env).
6. `test_intrinsic_only_warns_on_a_non_local_latent_but_still_runs` — asserts
   the non-local case emits the warning (`pytest.warns`) **and** proceeds, and
   that a `LOCAL_LATENTS` member emits none. This is what keeps the "warn, do
   not block" decision from silently degrading into either extreme.
7. `test_zero_goal_is_still_rejected_under_intrinsic_only` — no new guard; pins
   that the existing `zero_goal && intrinsic_coef != 0` check already covers it,
   so a future refactor of that guard cannot open the hole.

---

## Verification

```bash
# 1. Seam tests (CPU-pinned by the autouse fixture; fast, deterministic)
uv run pytest algorithms/tests/test_feudal_seams.py -q

# 2. Manager self-checks unaffected
uv run python -m algorithms.feudal_mappo_jax.manager

# 3. Guards fire before a run is launched
uv run python train.py algorithm=feudal_mappo_jax env=mjx_12a_3o_trunc_1024 \
    model=feudal_film_intrinsic_only_local_private trial_id=0 \
    params.intrinsic_coef=0.0                        # must RAISE
uv run python train.py ... params.intrinsic_anneal=linear          # must RAISE
uv run python train.py ... model_params.manager_latent=centralized # must WARN, then run

# 4. Composition check — the `# @package _global_` footgun.
#    Confirm from the RESOLVED config, not the yaml.
uv run python train.py --cfg job algorithm=feudal_mappo_jax \
    env=mjx_12a_3o_trunc_1024 model=feudal_film_intrinsic_only_local_private \
  | grep -E "worker_objective|intrinsic_coef|intrinsic_anneal|manager_latent|worker_fusion"

# 5. Smoke train + resume (short n_total_steps), then confirm from the
#    CHECKPOINT that the intended arm trained: manager tree carries
#    f_Mspace_agent_kernel (local_private), actor Dense_0 is (obs, hidden) (film),
#    and `intrinsic_reward_abs` in the stats is NOT identically 0.0.
uv run python train.py algorithm=feudal_mappo_jax env=mjx_12a_3o_trunc_1024 \
    model=feudal_film_intrinsic_only_local_private trial_id=0 \
    params.n_total_steps=200000

# 6. Structural preconditions BEFORE reading any return (per (b) above)
uv run python -m algorithms.feudal_mappo_jax.latent_locality_probe --control
MUJOCO_GL=egl uv run python -m algorithms.feudal_mappo_jax.latent_locality_probe \
    --batches mjx_12a_3o_trunc_1024 \
    --models feudal_film_intrinsic_only_local_private --trial 0
MUJOCO_GL=egl uv run python -m algorithms.feudal_mappo_jax.latent_diversity_probe \
    --batches mjx_12a_3o_trunc_1024 \
    --models feudal_film_intrinsic_only_local_private --trial 0
```

Full-length runs (3 seeds) on `mjx_12a_3o_trunc_1024` against
`feudal_film_local_private`, `feudal_film_zerogoal` and `mlp` are the actual
experiment; the above is the mechanism check.
