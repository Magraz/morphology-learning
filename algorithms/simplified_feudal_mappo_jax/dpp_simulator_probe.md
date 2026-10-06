This is test 3: compare the joint-goal critic's predicted recruitment benefit
with actual environmental return from paired simulator forks.

Start with the dense two-agent arm, where reward changes are easier to measure:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_simulator_probe \
  --batch mjx_2a_4o_1122_1024_gs \
  --model simplified_feudal_tanh_relative_input_dpp \
  --trials 0,1,2,3,4
```

Then audit the sparse six-agent arm:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_simulator_probe \
  --batch mjx_6a_4o_1024_gs_sparse \
  --model simplified_feudal_tanh_relative_input_dpp \
  --trials 0,1,2,3,4
```

The local-input DPP arm and the counterfactual-credit arms can also be loaded:
they all contain the required `manager_adv` model. Defaults are 32 source
episodes, 64 sampled (manager decision, focal agent) states per trial, eight
Monte Carlo continuations per candidate, and four states per simulation chunk.
Full MJX collection and continuation are intended for GPU execution. Simulation
width is `chunk-size * mc-samples`, independent of the number of candidates;
candidates run sequentially to bound memory usage.

The checkpoint's worker, manager and joint-goal critic remain frozen. Source
states come from stochastic policy episodes and preserve complete simulator
data, original elapsed time, delivery flags, previous reward distances,
observations and sampled original waypoints. Nothing is trained or overwritten.

Sampling uses four contact strata, restricted to undelivered boxes requiring
at least two agents:

- `waiting_alone`: the focal agent touches a box with exactly one agent present;
- `understaffed`: more than one agent is present, but fewer than required;
- `coalition`: the focal agent touches a box whose coupling requirement is met;
- `other`: all remaining decisions, including travel and light-box activity.

Quotas are approximately equal. If a stratum is missing, its quota is filled
from observed states. `report.json` records both availability and sampled counts.
The combined `all_stratified` summaries describe this sample, not population
frequencies under the policy. A missing waiting/coalition stratum is a coverage
limitation, not a fabricated positive example.

For each state and every recruit count permitted by `dpp_max_recruits`, evaluate:

1. The exact original joint waypoints (the common factual reference).
2. The exact `counterfactual.dpp_joint` assignment, sending nearest teammates
   toward the focal agent's current position.
3. The corresponding assignment away from the focal agent.
4. A random feasible rotation/reflection of each toward step, preserving its
   Euclidean travel distance and the same recruited teammate identities.

The focal goal and all unrecruited teammates' goals are preserved exactly.
Random directions are chosen from eight feasible rotations/reflections. Near
arena boundaries the feasible set can be small; a random control may coincide
with toward recruitment. The away control uses the production geometry and
arena clipping, so its travel distance may differ. Candidate displacement
columns expose these cases. Random draws and candidate selection are recorded
through the source seed and source indices.

Every branch uses the same simulator state and matched worker/manager random
draws for each MC replicate. Only the first goal window is overridden. At later
boundaries the original frozen manager chooses fresh goals in that branch's
new state. Early termination and the original environment time limit end the
trajectory; episodes are never reset or extended.

Predicted gain is `(adv_model(candidate) - adv_model(original)) / recruits`.
Actual gain is the paired difference in discounted environmental team return,
also divided by recruits. The primitive discount is the configured manager
step gamma, falling back to the worker gamma. Physical returns contain no
value bootstrap and stop at the episode's actual endpoint. The probe reports
one-window, four-window and remaining-episode returns, plus undiscounted reward,
deliveries, coupled contacts, focal support, box progress and live steps.
`coupled_box_steps` counts box-step pairs with met coupling after the physical
step; `focal_supported_steps` counts primitive steps where the focal agent is
touching such a box. A delivery step is included before its flag is excluded.

The critic was trained on GAE, with a continuing-task timeout bootstrap, so its
predictions need not numerically equal finite-episode return differences. To
expose that target mismatch, `training_gae_gain` separately recomputes the full
continuation's initial GAE advantage with training discounts, lambda, critic
values and timeout bootstrap. This field is the same across a candidate's
return-horizon rows. A gain only in this field, with no physical return gain,
depends on learned value predictions rather than measured task reward.

Outputs go to a new timestamped directory under `plotting/feudal_goal_analysis`:

- `candidates.csv`: per-source, per-candidate and per-return-horizon predictions,
  MC mean realized gains, paired-bootstrap intervals and physical diagnostics.
- `summary.csv`: per-training-seed and per-stratum sign/ranking correlations,
  positive predictions with harmful or tied outcomes, and mean gain intervals
  clustered by source episode. `dpp_selected` restricts toward candidates to the
  count selected by DPP where a positive raw bonus would actually be applied.
- `rankings.csv`: within-state rankings for toward recruit counts and for all
  interventions. Rankings exclude measured ties. Selection regret includes
  the option of retaining the original goals; it is an MC estimate using the
  same samples, not a known oracle.
- `trial_*_sources.npz`: complete simulator source states and checkpoint
  parameters, with resolved configs and checkpoint hash.
- `trial_*_chunk_*.npz`: all paired MC outcomes, raw predicted deltas,
  continuation keys, candidate labels/counts and horizon steps.
- `report.json`: definitions, sampling coverage, provenance and summary rows.

`gain-tolerance` defaults to `1e-4` per-recruit realized return units. Realized
gains below it are ties. Predicted positivity uses strictly greater than zero,
matching production DPP even when the predicted bonus is tiny. All-zero sparse
outcomes yield undefined sign accuracy/correlations;
they do not count as accurate predictions or prove that recruitment never
helps. Candidate intervals resample paired MC continuations. Summary intervals
resample whole source episodes within each trained seed. Five trained seeds
remain five independent training replicates regardless of how many states or
continuations are sampled.

Read the result by asking whether positive DPP-selected predictions actually
increase return, whether toward beats the away/random controls, and whether
the critic selects a useful recruit count. If support helps physically but
the critic misranks it, investigate critic data/targets. If the exact support
assignment is physically harmful, investigate geometry and opportunity cost.
If all candidates tie at zero, use dense checkpoints or more continuations to
obtain evidence that can distinguish these explanations.

Replay saved source states and weights without recollecting them:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_simulator_probe \
  --snapshot PATH/trial_0_sources.npz --mc-samples 16
```

Replay still runs the simulator. It restores the environment's original time
limit and validates the simulator state schema. It needs compatible installed
simulator/JAX versions. Continuation random streams are identical across
branches and independent of chunk size. Source snapshots from the normalization
probe contain no complete physics state and cannot serve as fork snapshots.
Keep the same chunk size/backend for closest numerical replay: changing vector
width or backend can alter floating-point reductions and slightly change MJX
trajectories. Near-zero dense gains should be checked with a larger gain
tolerance rather than treated as reliable sign changes.

For a short smoke test:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_simulator_probe \
  --trials 0 --n-envs 1 --states 2 --mc-samples 2 --chunk-size 1 \
  --source-steps 64 --eval-steps 64
```

Artificial continuation limits must contain whole goal windows and are labeled
`cutoff`, not completed-episode returns. Short limits are useful for debugging
and immediate reward effects; they can miss delayed coalition payoffs.
