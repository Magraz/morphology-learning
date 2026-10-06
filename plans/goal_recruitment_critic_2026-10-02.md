# D++ counterfactual-goal credit from a joint-goal critic

Date: 2 October 2026, refocused 3 October 2026. Status: implemented and
smoke-tested 3 October 2026 (sections 3 to 5). The section 6 check FAILED, so the
section 8 full runs are on hold. Target: `algorithms/simplified_feudal_mappo_jax`. The original
broader draft is kept at `plans/goal_recruitment_critic_2026-10-02.v1.md`.

## 1. Scope

This plan adds exactly one thing: a per-agent D++ term in the manager's
advantage, computed only from the joint-goal critic. Nothing else changes:

- the MDP, the manager's waypoint action space and execution are untouched;
- no goal is ever overridden at execution, and no simulator fork is used;
- the worker, the manager critic and every existing mode stay bit-identical.

The plan runs no full training runs. It delivers code, tests, smoke runs of at
most about 2e5 steps, and model groups. The author launches the full runs in
section 8.

## 2. Motivation, measured

`mjx_2a_4o_1111_1024_gs` and `mjx_2a_4o_1122_1024_gs` differ only in
`coupling_def`; box sizes are identical. Mean of the last 10 evaluations, 5 seeds,
1e8 steps:

| env group | `mlp` | relative manager, team credit | relative manager, counterfactual credit |
|---|---|---|---|
| 1111 dense | 437–438 | 399–431 | 423–435 |
| 1122 dense | 418–429 | 281–320 | 248–320 |
| 1111 sparse | 398 | 383–392 | 347–394 |
| 1122 sparse | 370–387 | 116–160 | 118–163 |

Tight coupling costs the hierarchy 122 points on dense reward and 250 on sparse
reward. Flat MAPPO loses 16 and 20. On the coupled groups the hierarchy delivers
53% of two-agent boxes on dense reward and 0.2% on sparse, against 93% and 95% for
flat MAPPO (review probe, 5 seeds × 64 episodes).

The existing counterfactual credit did not change return. It is a true baseline:
agent `i`'s correction never reads agent `i`'s own goal, so it changes the
variance of the gradient but not its expected value. The term below is
deliberately different.

## 3. The term

### 3.1 Joint-goal critic

The critic is the existing advantage model `Â_φ(x, o)` in `counterfactual.py`.
It reads the joint state `x` (the manager critic's input: global state plus every
agent's position) and the joint goal `o` (every agent's waypoint offset
`(w − s) / R`). It is trained by `fit_adv_model` on the raw team advantage.

A separate return critic `Q(x, o)` is not needed, because the state value cancels
in every difference this plan takes:

```text
Q(x, o') − Q(x, o) = A(x, o') − A(x, o)
```

Reusing `Â_φ` keeps everything already verified for the counterfactual-credit
arm: its zero-initialized head, the scoring-before-refit order, its checkpoint
entry and its diagnostics. Section 7 lists the alternative.

### 3.2 The D++ counterfactual joint goal

For focal agent `i` and `n` recruits:

1. Take the realized joint goal `o`.
2. Keep agent `i`'s own goal unchanged.
3. Choose the `n` teammates nearest to agent `i`.
4. Replace each recruit `j`'s goal with a waypoint toward agent `i`'s current
   position, bounded exactly like a manager waypoint:
   ```text
   w_j++ = clip(s_j + R · clip((s_i − s_j) / R, −1, 1), arena)
   o_j++ = (w_j++ − s_j) / R
   ```
5. Leave every other teammate's goal unchanged.

Call the result `o++(i, n)`. Every entry is a legal manager output, so the critic
is queried inside its input range. The query is, however, off the policy's usual
distribution; section 6 checks how the critic behaves there.

This is the goal-space analogue of D++. D++ adds `n` hypothetical copies of agent
`i` at agent `i`'s location and asks how much the team objective would rise. Here
`n` real teammates are sent toward agent `i`, so the value also counts what they
stop doing elsewhere.

### 3.3 The value

```text
D++_i(n) = [ Â(x, o++(i, n)) − Â(x, o) ] / n
D++_i    = max over n = 1 .. N−1 of D++_i(n)
```

At two agents `n` can only be 1, and this is exactly the requested value
`Q(x, o++) − Q(x, o)`. The division by `n` and the search over `n` follow the
original D++ (Rahmattalabi, Chung, Colby and Tumer, IROS 2016). The search costs
`N × (N − 1)` critic forward passes per decision and no environment steps. At
`N = 1` there are no teammates and the term is exactly 0.

### 3.4 How it enters the advantage

```text
A_i = A_team + η · max(0, D++_i)
```

It is passed to the unmodified shared `ppo_update` as
`advantage_correction = −η · max(0, D++_i)`, because that argument is subtracted.
The existing normalization then centres each agent's stream and scales by the
team advantage's standard deviation.

With `η = 1` and an accurate critic, `A_i ≈ max(Q(x, o), Q(x, o++)) − V(x)`. Each
agent's goal is credited with the better of two outcomes: what actually
happened, or what would have happened if teammates had come to it. A goal that
only pays when supported, such as waiting at a two-agent box, is therefore
reinforced before any partner arrives. That is D++'s stepping-stone signal.

### 3.5 This is shaping, not a baseline

`D++_i` depends on agent `i`'s own goal: it is kept in `o++`, and it is part of
`o`. So the term changes the expected gradient. That is intended, because the
true baseline was measured to do nothing. Two consequences:

- **The objective changes.** As with D++ itself, the policy is optimized for
  supportable goals, not purely for team return. The uncoupled 1111 group is the
  check that this does not cost return where no support is needed.
- **The `ppo_update` docstring must be updated.** It currently says the caller
  must keep the correction independent of the agent's own action. Under this
  mode that is false by design. The docstring should state that such a
  correction is shaping, and that the D++ mode accepts the resulting bias.

## 4. Decisions for the author

Each has a default the plan uses unless you choose otherwise.

1. **Sign.** The default adds the term, as D++ does. Subtracting the raw value as
   a literal "baseline" would penalize goals that become valuable with support.
   That pushes agents away from two-agent boxes, the opposite of the intent.
2. **Clip at zero.** The default credits only positive potential, mirroring the
   original algorithm's fallback to the plain difference reward when adding
   agents does not help. The signed version also penalizes goals that support
   would make worse, which mixes in the quality of teammates' actual goals.
3. **Recruit target.** The default is the focal agent's position, as you
   specified. The variant is the focal agent's current waypoint ("following the
   same goal" in `paper/AGENTS.md`). That variant can bring a recruit to the
   wrong side of a box being pushed.
4. **Coefficient.** The default is `η = 1.0`, the interpretable setting in
   section 3.4. Use `η = 0.5` only if the 1111 control regresses.

## 5. Implementation

| step | change | check |
|---|---|---|
| 1 | `counterfactual.py`: `support_offsets(pos, i, recruits, radius)`, `dpp_joint(offsets, pos, i, n)` and `dpp_credit(adv_ts, critic_in, offsets, pos, n_max) -> (dpp, best_n)`, batched over `(T, E)`, with `jax.lax.map` over focal agents as `correction` already does. | Unit tests in the table below. |
| 2 | `trainer.py`: factor the shared part of `_counterfactual_credit` (team advantage, pre-update scoring, refit of `Â_φ`) into one helper with a pluggable correction, and add the D++ branch. Store `ManagerGoal` and build `manager_adv` when the mode is `counterfactual` or `dpp`. | `team` and `counterfactual` modes bit-identical to the current code on the CPU stub. |
| 3 | `types.py`, `run.py`: `manager_credit: team / counterfactual / dpp`, plus `dpp_coef` (`η`, default 1.0) and `dpp_max_recruits` (default `N − 1`), wired through `make_feudal_config`; the launch banner prints them. `validate_manager_credit` rejects a negative `η` and warns at `N = 1`. | `make_feudal_config` test. |
| 4 | `mappo_jax/mappo.py`: docstring of `advantage_correction` only (section 3.5). No code change. | — |
| 5 | Model groups, each one key on its parent: `simplified_feudal_tanh_relative_input_dpp` and `simplified_feudal_tanh_local_input_dpp`. | Compose, smoke train, resume, evaluate at about 2e5 steps with a non-numeric `trial_id`. |
| 6 | `CLAUDE.md`: a short section in the simplified feudal block. | — |

Ordering inside an update is the same as the counterfactual-credit path: the D++
term is scored with the pre-update critic, the manager is updated, and the critic
is refitted last on this batch's team advantage. The critic's zero head makes the
term exactly 0 at the start, so the arm begins as team credit.

Checkpoints: a D++ run has the same tree as a counterfactual-credit run, because
both carry `manager_adv` with the same input width. Only the model group name
records which mode trained it, as for the other credit knobs.

Logged per update, prefixed `manager_dpp_`:

- `mean`, `std` and `positive_frac` of the clipped term;
- `mean_best_n`, the average chosen number of recruits;
- `adv_shift`, the ratio of the term's standard deviation to the team
  advantage's standard deviation, i.e. how much of each agent's signal it
  supplies.

The model's explained variance stays under the existing `manager_cf_model_ev` key.

Seam tests, added to `algorithms/tests/test_simplified_feudal.py`:

- off by default, with `team` and `counterfactual` bit-identical;
- `dpp_joint` keeps the focal slot exactly and changes only the `n` nearest
  teammates;
- support offsets point toward the focal agent, lie within `[−1, 1]` per axis
  and respect the arena;
- the term is exactly 0 at `N = 1` and with a zero-initialized critic, and the
  update then equals team credit;
- the clipped term is never negative, and the `n` normalization and maximum
  behave as specified on a hand-built critic;
- a positive term raises the agent's advantage, so the sign is pinned;
- end-to-end collect, update and evaluate under `dpp` at `N = 1` and `N = 3`;
- the checkpoint carries `manager_adv` under `dpp`.

## 6. Check before any full run: does the critic see coalitions?

The term is only as good as `Â_φ`'s sensitivity to teammates coming to a focal
agent. The counterfactual-credit arms explained only 13–25% of the team
advantage from the goals. Before the author launches full runs, check this
offline with critics that already exist. The `_relative_input_cf` checkpoints on
the 1122 groups contain a trained `manager_adv`.

1. Roll out each seed's trained manager on `mjx_2a_4o_1122_1024_gs`.
2. Compute `D++_i` at every decision with that seed's critic.
3. Compare the term when agent `i` is touching an undelivered two-agent box
   alone with its value at all other decisions.

**Pass:** the term is clearly larger when the agent waits alone, consistently
across seeds. **Fail:** no difference. The term would then be noise of the
critic's own making. Fix the critic, with more capacity, epochs or a return
target, before running the arm.

This needs only the critic, no simulator forks. Put it in a small script beside
the trainer, as the other probes are.

**Result, 3 October 2026: FAIL** (`algorithms/simplified_feudal_mappo_jax/dpp_probe.py`,
5 seeds × 64 stochastic episodes per arm, intervals bootstrapped over episodes).
Clipped term when waiting alone minus elsewhere:

| 1122 group | relative `_cf` critic | local `_cf` critic |
|---|---|---|
| dense | −0.70 / −0.80 / +0.05 / −0.91 / −1.04 | −0.30 / −0.47 / −0.49 / −0.29 / −0.56 |
| sparse | −0.13 / −0.96 / −1.02 / −0.14 / −2.06 | −0.34 / −0.39 / −0.26 / −0.33 / −0.34 |

No interval lies above zero, and 19 of 20 lie below it. The probe's control sends
the same recruit away from the agent instead. When an agent waits alone, the
critics prefer the away move on 17 of 20 combinations, so this is not just a
penalty for leaving the policy's own goals.

What the probe cannot say is whether the critic is wrong. Three explanations
fit, and they call for different fixes:

1. **The critic is right about a one-window detour.** The teammate pays for
   leaving its own task, and the manager abandons the coalition at the next
   boundary, so the payoff never arrives.
2. **The target is wrong.** Heading for the waiting agent's position can
   approach the box from the wrong side.
3. **The critic encodes the parent's failure.** It is trained on on-policy
   advantages from policies that rarely complete coalitions.

Deciding among them needs either a simulator audit of the same counterfactual
(measurement only, not training) or a different recruit target. Both are the
author's call.

## 7. Alternative critic (not the default)

A dedicated `Q(x, o)` trained on the discounted window return plus a bootstrap
would make the "Q" in the definition literal. It costs a new network,
checkpoint entries and a second target with its own variance. The default reuse
of `Â_φ` gives the same differences when both are accurate. Switch only if
section 6 fails for reasons a return target would fix.

## 8. Full runs, launched by the author

Already available at 1e8 steps, 5 seeds: `mlp`, the relative and local
parents, and their `_cf` twins on all four 2-agent groups.

After the plan is done, the author launches 5 seeds each of:

| env group | arms | role |
|---|---|---|
| `mjx_2a_4o_1122_1024_gs` | `_relative_input_dpp`, `_local_input_dpp` | main comparison |
| `mjx_2a_4o_1122_1024_gs_sparse` | the same | stepping-stone regime |
| `mjx_2a_4o_1111_1024_gs` | the same | uncoupled control; must not regress |
| `mjx_6a_4o_1024_gs` | the same, plus the parents and `mlp` seeds 3–4 | larger coalitions |

The 6-agent parents (`simplified_feudal_tanh_relative_input` and
`..._local_input`) have no runs yet, and the arms there need them. Compare each
`_dpp` arm with its parent and its `_cf` twin at matched seeds, with paired
confidence intervals. The local-input pair is the information-matched comparison
with flat MAPPO. On the dense coupled group the local parent scores 188–224
against flat MAPPO's 418–429, so expect to close part of that gap, not to pass
flat MAPPO.

## 9. Known limitations

- **The term credits the waiting agent, not the partner.** The dense failure is
  mostly the partner not coming: 45% of the hierarchy's waits are never
  completed, against 6% for flat MAPPO. The partner gets only the team
  advantage. The term can help only by making agents go to two-agent boxes more
  often, so that partners meet them more often.
- **The critic learns only from what the policy does.** Under sparse reward, if
  coupled deliveries never happen in training, `Â_φ` cannot learn that support
  pays. The term then stays near 0 and the arm reduces to team credit. Read
  `manager_dpp_positive_frac` and `adv_shift` before reading return.
- **The objective is biased by design** (section 3.5). The uncoupled 1111 group
  is the regression check.
