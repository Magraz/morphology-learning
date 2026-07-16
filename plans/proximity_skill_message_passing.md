# Decentralized Skill Selection by Proximity Message Passing

## Idea

Replace VO-MASD's **centralized grouper** with **local message passing over a
proximity graph**. Each agent independently proposes a skill from a learned
discrete codebook, then *refines* that choice by exchanging its intended skill
with agents inside a fixed communication radius `R`. After `K` rounds the agents
commit, and the chosen skill runs as a macro-action for `macro_len` steps.

Hypothesis: **coordination does not need to live in the code, and it does not
need a coordinator.** VO-MASD encodes coordination *structurally* — a k-agent
joint codebook entry, sliced among group members, selected by a grouper that
reads global state. If coordination is instead produced by *message passing*, the
joint codebooks, the group partition, the intra-group role assignment, and the
grouper all become unnecessary — and execution becomes genuinely decentralized.

The framing that makes this a contribution rather than "a GNN on a proximity
graph": **the learned codebook is a grounded, discrete communication protocol.**
An agent broadcasts `log2(n_codes)` bits meaning *"I intend to run skill #3"*.
Unlike emergent-communication methods, the vocabulary is not arbitrary — it is
pre-grounded in behavior by offline skill discovery, so every symbol denotes a
concrete, inspectable macro-behavior.

## What this buys us over VO-MASD

VO-MASD's grouper (`algorithms/VOMASD/VO-MASD/onpolicy/algorithms/vomasd/vomasd_module.py`,
`Grouper`) is centralized **at execution**, not just training: it reads
`share_obs` (global state) and runs a Multi-Agent Transformer jointly over all
agents, and its output is what decides each agent's skill. Its group-shared
codes then force three further mechanisms to exist. Our substitution deletes all
of them:

| VO-MASD mechanism | Why it exists | Fate here |
| --- | --- | --- |
| `Grouper` (MAT over global state) | partition agents into groups | **deleted** — proximity graph replaces it |
| Grouper PPO loop (`pretrain_rl`, reward `rec_rwd + emb_rwd`) | train the grouper | **deleted** — refinement GNN is differentiable, trains on the VAE loss |
| `codebooks[k-1]` tower, `max_group_size` / `max_skill_size` | one joint code per group size | **deleted** — a single codebook |
| slice/role assignment within a joint code | who gets which slice of a k-code | **never arises** |
| `collect_codes` frequency thresholding | prune the group codebooks | **deleted** |

Net effect: strictly simpler, count-invariant (any team size), and decentralized
with radius-limited communication instead of global state + a central arbiter.

## Positioning (the honest novelty risk)

This is adjacent to work that exists, **including in this repo**. The claim must
be the combination, not any single ingredient.

- **DGN / Graph Convolutional RL** (Jiang et al.) — GNN over a proximity graph
  for MARL. If we pass *continuous* messages we largely collapse to this. Passing
  **discrete codebook symbols** is the differentiator.
- **DCG** (`algorithms/dcg/`) — max-sum message passing over a coordination graph
  to select joint discrete actions. Message passing to agree on discrete actions
  *is* DCG's core.
- **`algorithms/dcg_macro/`** — already does DCG message passing to select frozen
  macro-skills. Our idea differs by (a) a **proximity-sparsified** graph rather
  than DCG's complete graph, and (b) skills from **offline VQ discovery** rather
  than hand-trained.
- **ROMA / RODE** — role-based MARL; our subgroup→skill map is a role mechanism.
  The fresh angle is that roles are *emergent and decentralized*.

**Headline result to target:** match or beat the centralized grouper using only
radius-limited local communication, and transfer zero-shot to team sizes the
grouper (with its baked-in `max_group_size`) cannot handle.

## Key decisions (settled)

- **Graph is geometric, not learned.** Edge `(i, j)` iff `dist(i, j) < R`. `R` is
  a config knob. Grouping is a fixed function of geometry, so there is nothing to
  train and nothing to go unstable.
- **Graph is frozen per macro-step.** Agents move, so the radius graph drifts.
  Build it once at each macro-decision boundary; hold it for the whole
  `macro_len` window. Message passing is treated as instantaneous negotiation
  *before* acting (standard in comms-MARL).
- **Fixed `K` rounds, not iterate-to-convergence.** `K` is small (2–3). This makes
  the refiner literally a K-layer GNN, sidestepping the oscillation /
  flip-flopping risk of iterative re-quantization entirely. `K` sets the
  coordination horizon (K-hop receptive field).
- **Messages are the chosen skill's code embedding**, selected by
  straight-through (hard forward, soft backward) — the same STE trick VQ already
  uses. Discrete in effect, but still a vector the GNN can consume, and it
  preserves the "shared vocabulary" story and the gradient path.
- **Single codebook.** Each agent picks its own code. Coordination comes from the
  message passing, not from joint codes.
- **Refiner is differentiable and trains on the existing objective.** No separate
  PPO-on-intrinsic-reward stage (contrast: VO-MASD's grouper).
- **Two-phase delivery.** Phase A isolates the *coordination* contribution on
  existing skills; Phase B adds the *skill-discovery* contribution. If local
  message passing does not beat independent selection in Phase A, the VQ part
  will not save it — so Phase A is the go/no-go gate.

## Algorithm

Per macro-decision step, for agent `i`:

1. **Propose.** Actor emits `z_e^i` from local obs (decentralized, unchanged).
2. **Quantize.** `c^i_0 = argmin_c ||z_e^i - codebook[c]||` → code embedding
   `e^i_0 = codebook[c^i_0]`, taken with STE.
3. **Refine, K rounds.** With `N(i) = {j : dist(i,j) < R}`:
   ```
   m^i_k   = AGG_{j in N(i)} ( MSG(e^j_{k-1}, edge_feat(i,j)) )
   z^i_k   = z_e^i + GNN_k( [h^i, e^i_{k-1}, m^i_k] )     # h^i = local obs encoding
   c^i_k   = argmin_c || z^i_k - codebook[c] ||           # re-quantize (STE)
   e^i_k   = codebook[c^i_k]
   ```
   `AGG` = attention or mean (attention preferred; reuse
   `MultiHeadAttentionEncoder` from `networks/gnn_critic.py`, masked by the
   adjacency). `edge_feat(i,j)` = relative position / distance, so the message
   is spatially grounded.
4. **Commit.** `c^i_K` is the skill. Run its actor for `macro_len` low-level steps
   (existing `HierarchicalSkillEnv` machinery).

Everything after step 1 uses **only** neighbors within `R`. No global state, no
coordinator.

## Phase A — prototype on existing skills (go/no-go)

Reuse the 4 hand-trained skills (`SKILL_ORDER = [contact, scatter, push_box,
rendezvouz]`) as a *fixed, pre-grounded codebook*. This isolates the coordination
mechanism from skill discovery and gets a signal fast on infrastructure we own.

**The proximity graph must come from the env, not the obs.** `ObservationManager`
is deliberately egocentric — `observation.py` has no absolute `own_pos`, so an
adjacency cannot be recovered from the observation vector. The env has the Box2D
bodies and their positions, so `HierarchicalSkillEnv` computes the adjacency and
surfaces it. Preferred: return it in `info["adjacency"]` `(n_agents, n_agents)`
each macro-step, leaving `observation_space` untouched (no obs-layout churn, no
`OBS_DIM` change).

### File changes (Phase A)

1. `environments/box2d_suite/*` — expose agent world positions (they already
   exist as Box2D bodies); no observation-vector change.
2. `algorithms/hierarchical/hrl_env.py` — in `HierarchicalSkillEnv.step`/`reset`,
   build the radius adjacency from agent positions and return it in `info`. New
   env knob `comm_radius`. Only meaningful for `decision_scope="agent"`.
3. `algorithms/hierarchical/skill_refiner.py` *(new)* — the K-round message-passing
   refiner. Consumes `(skill_logits | z_e, adjacency, edge_feats)`, returns
   refined per-agent skill logits. Reuse `MultiHeadAttentionEncoder`
   (`networks/gnn_critic.py`) with an adjacency mask rather than writing a new
   attention block.
4. `algorithms/mappo/` — thread the adjacency from `info` through
   `RolloutCollector` into the actor forward, and store it in the buffer so the
   PPO update can recompute refined logits. This is the only invasive plumbing.
5. `conf/model/<variant>.yaml` — `use_skill_refiner`, `comm_radius`,
   `refine_rounds` (K), `refine_agg`. Fully inert when `use_skill_refiner=False`
   so the baseline is byte-for-byte unchanged.
6. Logging — reuse the existing `action_distribution` tally in `vec_trainer`
   (already prints `Skills: contact=.. scatter=..`). Add mean graph degree and
   the **pre- vs post-refinement skill-change rate** (how often message passing
   actually flips a decision) — the single most diagnostic number for whether the
   mechanism is doing anything at all.

### Baselines (they bracket the space cleanly, and we already have all of them)

| Run | Graph | Comms | Purpose |
| --- | --- | --- | --- |
| `decision_scope="agent"` | none | none | independent selection — **the control** |
| `decision_scope="team"` | — | — | one shared skill (fully homogeneous) |
| `dcg_macro` | complete | full | does *locality* cost anything vs. all-to-all? |
| **ours** | radius `R` | local | the contribution |
| ours, `R = ∞` | complete | full | ablate locality inside our own architecture |
| ours, `K = 0` | radius | none | ablate message passing (≈ independent) |

Sweep `R` and `K`. Scale test: train at 9 agents, evaluate at 15/30 (the repo
already has a 30-agent experiment) — the count-invariance claim.

**Go/no-go:** ours must beat `decision_scope="agent"` on `multi_box_push`. If it
does not, stop.

## Phase B — learned VQ codebook (the full contribution)

Swap the 4 fixed skills for a codebook discovered offline, VO-MASD-style, but
with the grouper replaced by the Phase-A refiner:

- **Data.** Same recipe as VO-MASD: roll out trained policies to produce offline
  trajectory datasets. We already have the behavior policies (the frozen
  `mlp_shared` checkpoints under `experiments/results/{contact,scatter,push_box,
  rendezvouz}_9a/`), so this is a collection script, not new training.
- **Encoder.** Trajectory encoder over a `macro_len` window → `z_e` (VO-MASD's
  `TrajEncoder`, minus the SMAC entity decomposition, which does not port —
  our obs is a flat egocentric vector, not an entity set, and there are no
  "enemies").
- **Quantize + refine.** Single codebook; the refiner runs over the proximity
  graph exactly as in Phase A.
- **Loss.** Reconstruction (decoder log-prob of the offline actions given the
  *refined* code) + VQ commitment (`beta`). The refiner is inside the gradient
  path, so it trains on this loss — **no grouper, no `pretrain_rl`**.
- **Online.** Freeze codebook + decoder; MAPPO trains the high-level actor over
  the skill space, with the refiner in the loop.

## Risks / open questions

- **Does message passing actually change any decisions?** If the refiner learns
  the identity map, we have an expensive no-op. The pre/post skill-change-rate
  metric catches this immediately; if it is ~0, the coordination signal is too
  weak or `R` is too small.
- **Is proximity the right graph?** A radius captures spatial locality only. For
  tasks where coordination is *not* spatial it is a bad prior. Our box2d tasks are
  spatial, so this is a well-motivated inductive bias here — but it is a scope
  limit to state, not hide.
- **Codebook collapse.** Standard VQ failure (all `z_e` map to one code). Watch
  code-usage entropy; the existing `action_distribution` logging generalizes.
- **Refiner + PPO interaction.** The refiner sits inside the actor, so its
  parameters are updated by the policy gradient. Straight-through through
  *two* quantization steps (propose + refine) may be high-variance. Fallback: soft
  (softmax-weighted) messages during training, hard at execution.
- **Novelty vs. `dcg_macro`.** If proximity-sparsified DCG over macro-actions
  performs identically, the VQ-discovered-codebook half of the story has to carry
  the contribution. Phase B is what distinguishes us; Phase A alone is an
  ablation of DCG.

## Sequencing

1. Phase A plumbing (env adjacency → collector → refiner → buffer).
2. Phase A go/no-go on `multi_box_push_9a`, vs. `decision_scope="agent"`.
3. `R` / `K` sweeps + scale test (9 → 15/30 agents).
4. Phase B: offline data collection, VQ codebook + trajectory encoder.
5. Phase B evaluation; port to SMAC only if a cross-benchmark comparison against
   VO-MASD's published numbers is needed.
