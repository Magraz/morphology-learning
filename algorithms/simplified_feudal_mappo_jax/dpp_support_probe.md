This is test 4: isolate the effects of support target and commitment duration
using the same saved simulator states and frozen policies from test 3.

Run on the completed dense two-agent audit:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_support_probe \
  --source-dir /home/magraz/morphology-learning/plotting/feudal_goal_analysis/dpp_simulator_20261005T200820570157Z
```

Defaults select `waiting_alone,understaffed` sources, the states where the focal
agent needs support. In the two-agent audit this selects 73 lone-waiting sources
across five seeds; the understaffed stratum is empty because all coupled boxes
require exactly two agents. Use `--groups all` to also test travel, light-box
activity and existing coalitions. No new episodes or source states are needed.

The experiment includes the original manager policy and this 3 × 3 matrix:

| Target for recruited agents | 1 window | 2 windows | 4 windows |
|---|---|---|---|
| Current focal position | Yes | Yes | Yes |
| Current focal waypoint | Yes | Yes | Yes |
| Box staging controller | Yes | Yes | Yes |

Each condition uses the same complete source state, sampled initial goals,
focal agent, recruit count, recruited teammate identities, and paired random
streams. The recruit count is production DPP's best count at the source state;
it is chosen once, regardless of whether the bonus is positive. Use `--recruits
N` to fix a different count across sources. Identities are the production
nearest-teammate mask at the source and remain fixed throughout the condition.

Focal and unrecruited agents retain their exact original initial waypoints.
At subsequent boundaries they follow the frozen learned manager. Only recruited
agents' goals are overridden. Support is replanned at each manager boundary
for the requested duration, then the learned manager resumes full control.
Thus persistent support follows moving targets rather than holding an obsolete
absolute point. Every waypoint retains the original radius and arena bounds;
worker inputs and their per-window remaining-time fraction are unchanged.

Target definitions:

- `focal_position`: move toward the focal agent's current position at each
  boundary. Its one-window condition is the existing DPP recruitment geometry.
- `focal_waypoint`: move toward the focal agent's learned waypoint for the
  current window. This tests whether aiming at where it is going helps.
- `box_staging`: prefer a live coupled box touched by the focal agent at the
  source, otherwise select the nearest live coupled box. Keep that box identity
  fixed. Assign separated rear staging slots, route helpers around the box
  when approaching from above, and choose an upward pushing goal when already
  near the staging point. Replan only at boundaries. If the selected box is
  delivered, release recruits at the next boundary.

Box staging is a privileged diagnostic controller, adapted from the repo's
scripted manager. Its source box choice, slot assignment, routing, and pushing
phase constitute a stronger intervention than changing a target coordinate.
Its success would establish that these frozen workers can benefit from this
particular assistance strategy; it would not prove that a decentralized manager
can learn it or that the controller is an optimal coalition planner. Slots are
heuristic and may crowd when too many agents are recruited. The focal agent is
assigned a reserved staging slot for spacing but is never forced to occupy it.

Defaults are eight paired continuations per source/condition and four sources
per chunk. Conditions run sequentially so the simulation width remains
`chunk-size * mc-samples`. The default one-window conditions are anchors even
if custom `--commitments 2,4,8` are requested. Original time limits remain in
force: late source states can have less time than the requested commitment.
`remaining_episode_steps` exposes this limitation. `--eval-steps` allows a
shorter whole-window cutoff for smoke tests and immediate effects.

The probe records physical returns at one window, four windows and the actual
episode endpoint, or a labeled artificial cutoff. The task reward is discounted
using the manager's primitive-step discount and contains no value bootstrap.
Training-style GAE, including timeout bootstraps, is a separate diagnostic.

For the selected source box it also measures:

- focal-agent contact duration;
- recruited-agent contact duration, and whether any/all recruits arrive;
- coupling duration and first coupling step;
- delivery probability and first delivery step;
- whether recruits reach their assigned worker waypoint.

These metrics keep the same monitored recruits and box in the original-policy
reference, even though that reference does not override any goals. First event
times are post-step times relative to intervention start, conditional on the
event occurring. Nonarrival is recorded separately, not averaged as a long
arrival time. Contact counts exclude already-delivered target boxes and include
the delivery step. Reaching a waypoint does not itself imply helpful contact.

The learned critic scores only the first joint goal assignment. That score is
the same for different commitments with the same target. It is interpretable
as the existing one-window prediction only when `prediction_matches_commitment`
is true. The critic was not trained for the new persistent continuation policy,
so do not treat its unchanged score as a prediction of two/four-window support.

Outputs are in a new `plotting/feudal_goal_analysis/dpp_support_*` directory:

- `conditions.csv`: per-source target/duration gains versus the factual policy,
  paired MC intervals, selected-box contact/delivery metrics and event times.
- `summary.csv`: per-training-seed and source-stratum means and intervals
  clustered by original source episode.
- `contrasts.csv` and `contrast_summary.csv`: paired factorial comparisons:
  `target(K) - focal_position(K)` isolates target changes at a fixed duration;
  `target(K) - target(1)` isolates commitment within a target strategy;
  their difference-in-differences measures target × commitment interaction.
- `trial_*_chunk_*.npz`: all raw paired returns, contacts, event times and keys.
- `trial_*_sources.npz`: filtered complete source snapshots and frozen weights,
  retaining original source IDs for reproducible random streams.
- `report.json`: definitions, provenance, source filtering and summary rows.

Gains are divided by the fixed recruit count per source. Physical gains within
`--gain-tolerance` (default 1e-4) are ties. Invalid staging conditions without a
live coupled source box retain manager goals and are excluded from staging
summaries. Samples are stratified and filtered; pooled results are not training
state prevalence estimates. Repeated continuations are not additional training
seeds. Keep chunk size/backend fixed for closest numerical replay.

Interpret the factorial comparisons together with contacts:

- Better targets at one window support a geometry/routing explanation.
- Better results only with two/four windows support a travel/commitment
  explanation for that target strategy.
- Recruits arrive but focal contact disappears: the focal manager may abandon
  the coalition. This test holds the focal policy fixed, so it detects rather
  than solves that role-persistence problem.
- More coupling without more return/deliveries: contact alone is insufficient
  to establish useful cooperative pushing.
- Reliably beneficial interventions that the one-window critic misranks
  motivate fitting the critic on intervention examples. Persistent results
  require revising the critic's action/continuation definition as well.

Replay a filtered trial with more continuations:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_support_probe \
  --snapshot PATH/trial_0_sources.npz --mc-samples 16
```

For a short command check on the existing snapshots:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_support_probe \
  --source-dir /home/magraz/morphology-learning/plotting/feudal_goal_analysis/dpp_simulator_20261005T200820570157Z \
  --trials 0 --mc-samples 2 --eval-steps 128
```
