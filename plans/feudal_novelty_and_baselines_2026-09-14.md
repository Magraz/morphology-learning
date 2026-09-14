# Feudal MAPPO: novelty assessment and baseline selection

Assessment: 14 September 2026. Based on the current working-tree implementation and a targeted search of primary papers. This updates the earlier positioning note for the newly added local/private latent variants. No training or evaluation was run, and numerical claims in existing code comments were not independently reproduced. The search does not establish an exhaustive priority claim.

## Recommendation

The broad contribution “extend FeUdal Networks to multiagent continuous control” is insufficient to establish novelty. The exact combination in this repository was not identified in the papers checked, but combining established mechanisms is a weaker contribution than demonstrating a new solution to a specific multiagent problem.

Use **HiMPPO as the main external feudal comparator**, and **a minimally adapted multiworker FuN with PPO as the mechanism control**. Include ordinary MAPPO and a flat actor with equivalent execution-time information. FLE is a particularly relevant older continuous-action comparator. FMH is the historical multiagent feudal reference; use it experimentally when meaningful task-defined subgoals exist.

## What is actually implemented

- A centralized manager emits a distinct unit-norm directional goal for each agent. Goals are generated every environment step and pooled over a rolling horizon; the manager does not simply act once every horizon steps. See [manager.py](../algorithms/feudal_mappo_jax/manager.py) and [trainer.py](../algorithms/feudal_mappo_jax/trainer.py).
- The manager uses an extrinsic-advantage-weighted transition cosine, detaching the latent displacement. Its update is not PPO over sampled goals. See `manager_update` in [mappo.py](../algorithms/feudal_mappo_jax/mappo.py).
- Shared workers use local observations plus goals, with Gaussian continuous-action policies or categorical discrete-action policies. Goal fusion is concatenation or FiLM. See [worker.py](../algorithms/feudal_mappo_jax/worker.py) and [network.py](../algorithms/feudal_mappo_jax/network.py).
- With intrinsic rewards enabled, separate critics and separately normalized advantages produce `A = normalize(A_ext) + alpha * normalize(A_int)`. Alpha can anneal.
- The base [algorithm configuration](../conf/algorithm/feudal_mappo_jax.yaml) uses `intrinsic_coef: 0.0` and `manager_core: mlp`. This is not a faithful full-FuN reference configuration.
- Local variants compute the progress representation from each worker's observation, while goal selection mixes all workers' representations. `local_global` additionally supplies global state to goal selection. Private variants introduce agent-specific projections. These are materially different candidate methods and should be named explicitly in an experiment table.

Schematically, for agent i:

```
centralized progress: s_i = f_i(global_state)
local progress:       s_i = f(obs_i)
private local:        s_i = W_i f(obs_i) + b_i
goal assignment:      (g_1, ..., g_N) = manager(s_1, ..., s_N, [global_state])
manager objective:   maximize mean_i A_i^M cosine(stopgrad(s_i[t+c]-s_i[t]), g_i[t])
worker policy:       pi(a_i | obs_i, pooled_goal_i)
```

Under shared rewards, the manager broadcasts a team advantage across goal slots. This does not itself provide counterfactual agent credit assignment.

## Closest precedents

| Work | Relevant overlap | Distinction from this implementation |
|---|---|---|
| [FeUdal Networks, ICML 2017](https://proceedings.mlr.press/v70/vezhnevets17a/vezhnevets17a.pdf) | Directional latent goals, transition policy gradients, intrinsic progress rewards, goal pooling and dilated recurrence. | Your extension introduces multiple simultaneous workers, centralized goal assignment and continuous primitive-action heads, with a different optimization/conditioning recipe. |
| [Feudal Multi-Agent Hierarchies, 2019](https://arxiv.org/pdf/1901.08492) | A manager rewards multiple concurrent workers for assigned subgoals. | FMH experiments use supplied goal/reward mappings and DDPG. Your goals inhabit a learned directional representation. |
| [Feudal Latent-space Exploration, ALA 2020 version](https://ala2020.vub.ac.be/papers/ALA2020_paper_22.pdf) | Learned centralized latent coordination, tested on continuous Waterworld and Multi-Walker. | FLE uses a stochastic shared latent code and MADDPG-based optimization; your manager uses per-agent directional goals and a transition objective. |
| [Hierarchical Message-Passing Policies / HiMPPO, 2025 preprint, revised June 2026](https://arxiv.org/html/2507.23604v2) | Feudal multiagent hierarchy, PPO and continuous control in VMAS Sampling. | It uses hierarchical graphs and rewards derived from upper-level advantages; your design uses a centralized manager and latent-transition cosine rewards. |
| [Feudal Graph Reinforcement Learning, TMLR 2024](https://re.public.polimi.it/retrieve/c920f960-fa2b-4617-a11a-0728a0aa556c/3451_Feudal_Graph_Reinforcemen.pdf) | Feudal control of composable physical systems, including MuJoCo locomotion. | Its actuator hierarchy is a closer match to modular-body control than to independently moving cooperative robots. |

The FLE workshop paper alone is sufficient to refute a broad first claim for learned feudal coordination in continuous multiagent environments. Its [author publication list](https://shawnlue.github.io/publications/) also lists a later TNNLS version; the architectural comparison above specifically uses the accessible workshop version.

## Where a contribution could lie

The most specific candidate already visible in the code is **separating centralized goal assignment from worker-local progress measurement**. The hypothesis is that scoring each worker against an unrestricted global latent can reward transitions produced by other agents, while an appropriately structured progress space yields goals workers can use more reliably.

This is a hypothesis to establish, not a verified novel result. Show the failure in a matched baseline, explain why the new constraint addresses it, and demonstrate improved behavior and task return across multiple cooperative tasks. Check related representation and credit-assignment work before claiming priority for the remedy.

FiLM, PPO, JAX, goal normalization and annealing are not individually strong novelty claims. They can support a substantive empirical contribution when their necessity and effect are demonstrated. A careful study of why FuN-style transition alignment fails in cooperative continuous control could also be valuable without claiming an entirely new hierarchy.

## Baseline design

1. **MAPPO:** the existing flat implementation, with matched reward definitions, training interactions and tuning budget.
2. **Flat actor with equivalent information:** allow the policy access to the information the feudal manager uses during evaluation. This separates gains from information sharing from gains attributable to hierarchy. Match capacity approximately and report it.
3. **Minimal multiworker FuN-PPO adaptation:** retain the directional manager objective, goal pooling and positive intrinsic reward; use Gaussian primitive actions and the same PPO infrastructure. Keep the basic centralized latent representation, and remove the proposed local/private-space refinements. Label it as your adaptation, not an existing named published algorithm. The current centralized branch is a useful starting point, but document its deviations from original FuN.
4. **HiMPPO:** the strongest external match found for the stated setting. Start with a reported setup that can be reproduced, then port to the target tasks. Report graph construction, communication access and hierarchy depth. No official implementation was verified in this search; account for reproduction effort before scheduling runs.
5. **FLE or FMH where appropriate:** prefer FLE for learned continuous coordination; prefer FMH when your task naturally supplies physical subgoals such as assigned objects or target locations. Count worker pretraining interactions. A PPO rewrite should be labeled as an adaptation rather than presented as the original method.

For an explicitly original-FuN architectural comparison, restore its recurrent components and bilinear goal conditioning before claiming fidelity. A Gaussian mean can be formed from a bilinear observation-goal map; its standard deviation and any residual action branch must be specified. Merely naming the existing `model=feudal` arm “original FuN” would conceal meaningful differences. Hold recurrence fixed in the primary ablation so its effect is identifiable.

FGRL has an [official repository](https://github.com/tommasomarzi/fgrl), making it a practical additional comparator for modular locomotion. Its availability alone does not make it the closest scientific comparator for a robot team.

## Decisive experiments and interpretation

- Compare centralized versus local progress spaces with the same worker fusion, intrinsic coefficient, information access and recurrence. Compare private versus shared projections separately.
- Train a zero-goal control with intrinsic reward disabled. Also intervene on goals at evaluation: these answer different questions, because an evaluation-only perturbation can be out of distribution.
- Swap goals across environments while preserving agent identity. Cross-agent swaps in private coordinate systems mix different semantics, so a return drop does not uniquely demonstrate useful role assignment.
- Measure achieved physical effects and task return alongside cosine alignment. High latent rank or mutually distinct goal vectors can result from agent-specific coordinate choices without producing distinct physical behaviors.
- Evaluate multiple independently trained seeds; five is a reasonable pilot, and increase this if uncertainty prevents a decision. Use paired evaluation seeds for interventions. Report learning curves, confidence intervals and both interaction cost and wall time.
- Match evaluation counts: the checked YAML files currently specify 16 episodes for feudal and 32 for flat MAPPO despite a comment claiming equality.

## Claims to avoid

- **Communication-free decentralized execution:** the evaluation loop still runs a manager that mixes the team observations, and some variants consume privileged global state. Describe this as centralized coordination with local workers, and specify the communication model.
- **Causal credit assignment from local encoding:** `ds_i/dobs_j = 0` is a representation constraint. Teammates can still change `obs_i` through the environment.
- **Guaranteed objective correctness from annealing:** removing an auxiliary term does not prove convergence to an optimal task policy.
- **An exact gradient fraction from advantage normalization:** normalized advantage magnitudes do not fix policy-gradient norm ratios, and PPO clipping adds further dependence.
- **Concatenation cannot condition feature use:** in a nonlinear MLP the full observation-to-action Jacobian can depend on the goal. FiLM supplies explicit multiplicative conditioning; its superiority here requires evidence.

Suggested current description: “A FuN-inspired multiagent actor-critic with centralized per-agent directional goals, goal-conditioned continuous-action workers, and structured worker progress representations.” Reserve stronger claims about improved coordination or novelty of the representation constraint for validated results.
