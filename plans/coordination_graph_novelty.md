# Coordination-Graph Novelty Exploration

## Idea

Repurpose the multi-head attention encoder (currently feeding `AttentionGNNCritic`)
as the source of an **exploration signal**. The encoder emits a per-head
coordination graph — a symmetric adjacency `(n_heads, N, N)` describing
who-coordinates-with-whom. We reward agents for steering the team into states
whose **coordination graph is novel** relative to graphs already seen this
episode.

Hypothesis: the coordination graph is a low-dimensional, *coordination-relevant*
projection of the joint state. Novelty over this projection focuses exploration
on **new ways of coordinating** rather than new positions — a signal that
state-novelty methods (RND, pseudo-counts) and influence-based methods
(EITI/EDTI) do not directly provide.

## Key decisions (settled)

- **Dual-purpose encoder.** The attention encoder stays inside the critic and is
  trained by the value loss (so the graph is *grounded* in something meaningful).
  The bonus reads the encoder output with `no_grad` via `network_old` (the
  behavior network) — no separate representation-learning objective.
- **Both bonus granularities supported**, selected by `intrinsic_reward_mode`:
  - `"team"`: one bonus per env from the whole-graph descriptor, tiled to all agents.
  - `"agent"`: per-agent bonus from each agent's own coordination-row descriptor.
- **Novelty estimator:** episodic k-NN distance (restored `IntrinsicReward`).
  Count-free, continuous-friendly, and robust to the encoder being
  non-stationary because episodic memory resets every episode.
- **Single-stream reward** to start: bonus is added into the per-agent reward
  before GAE, reusing the existing advantage pipeline. Two-stream (separate
  intrinsic value head) is a fallback only if this destabilizes the critic.
- **Only valid with `critic_type="gnn"`** (asserted). No effect when
  `use_intrinsic_reward=False` — baseline behavior is byte-for-byte unchanged.

## Descriptor (the defining choice)

Derived from the **adjacency** by default (the hypothesis is about graph
structure), with a `node_embedding` alternative for ablation
(`intrinsic_descriptor_source`):

- **team / `adjacency`:** flatten upper triangle of each head's symmetric
  adjacency → `n_heads * N*(N-1)/2`.
- **agent / `adjacency`:** agent *i*'s row across heads → `n_heads * N`.
- **`node_embedding`:** use the encoder's per-agent tokens instead (team = mean
  pooled, agent = own token). Lets us ablate structure-novelty vs feature-novelty.

## File changes

1. `algorithms/mappo/intrinsic_reward.py` *(restore)* — k-NN `IntrinsicReward`
   from commit `7d33748~1`.
2. `algorithms/mappo/networks/gnn_critic.py` — add
   `AttentionGNNCritic.coordination_descriptor(obs, mode, source)`.
3. `algorithms/mappo/networks/models.py` — add
   `MAPPONetwork.coordination_descriptor(global_state, mode, source)` (uses
   existing `_to_obs_grid`; guarded to `critic_type=="gnn"`).
4. `algorithms/mappo/mappo.py` — config fields on `MAPPOAgent`,
   `compute_coordination_features(obs)` (no_grad, `network_old`), restore the
   `per_agent_intrinsic_rewards` arg in `store_transitions_batch` and fold it into
   `per_agent_rewards`.
5. `algorithms/mappo/trainer_components/rollout_collector.py` — restore the
   rewarder setup / reset-on-done / `_get_team_intrinsic_rewards` /
   `_get_agent_intrinsic_rewards`, but feed them
   `compute_coordination_features(next_obs)`.
6. `algorithms/mappo/types.py` + config yaml — new `Model_Params` fields, gated.

## Build order

1. `coordination_descriptor` on critic + network, with a shape unit test.
2. Restore `IntrinsicReward` + test.
3. Agent config + `compute_coordination_features` + storage arg.
4. Rollout wiring.
5. Smoke test: `intrinsic_reward_coef=0` must match baseline exactly; then `coef>0`.

## Risks

- **Cold-start noise:** random encoder early → noisy bonus. Mitigate with coef
  warmup/anneal if needed.
- **Single-stream coupling:** critic predicts returns that include novelty of its
  own encoder's output. If unstable, switch to a separate intrinsic value head.
- **Descriptor scale:** post-softmax adjacency rows are small; the `log(d+1)`
  estimator may over-compress. May need a scale factor — verify in step 5.

## Experiments

- Headline: graph-novelty vs RND-on-state vs no-bonus on a sparse-reward
  coordination task.
- Descriptor ablation: `adjacency` vs `node_embedding`.
- Grounding sanity: frozen-random encoder must do *worse* (else the graph is
  meaningless).
