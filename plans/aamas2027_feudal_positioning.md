**Assessment dated 7 September 2026.** This note combines inspection of the current `feudal_mappo_jax` implementation, its configurations, the saved goal-dependence probe, and a targeted search of primary research sources. It is a research assessment, not an exhaustive novelty certification. Publication status is stated only where verified. No training or evaluation rollouts were run for this assessment; the numerical observations below come from an existing result file. Proposed mechanisms and experiments are not implemented by this note.

**Verdict.** The repository provides a useful experimental implementation of multiagent FeUdal Networks (FuN), but “extending FuN to multiple agents” is not a defensible first-of-its-kind contribution. The current combination of centralized per-agent latent goals, a FuN transition objective, and Multi-Agent Proximal Policy Optimization (MAPPO) appears incremental unless it establishes a new mechanism or a substantial, carefully explained empirical finding. I did not establish that this exact implementation has appeared before; uniqueness of an implementation is a weaker claim than research novelty.

The strongest direction close to the existing code is **learning task-useful, jointly achievable goals for changing coalitions under partial observability**. The central question is whether a goal causes the intended coalition behavior and improves team outcomes. Alignment between a predicted goal and an observed latent transition does not establish either property.

**What the code actually implements.** The following descriptions refer to executable code rather than historical comments, some of which are stale.

| Component | Current behavior and evidence | Implication for the paper |
|---|---|---|
| Manager | `manager.py:FeudalManager` reads centralized input, produces an `N × goal_dim` latent state and an equally shaped array of normalized goals. | One centralized network with a distinct output slot for each agent; not independent decentralized managers. |
| Information | `trainer.py:_global_state` uses the environment's global-state hook where present; otherwise concatenates observations. The evaluation loop calls the same manager. | The joint policy needs centralized information during execution. On SMAX the manager can receive simulator state unavailable to ordinary local actors. |
| Worker | `worker.py:FeudalWorker` is a shared policy conditioned on local observation and pooled goals. It supports concatenation or feature-wise linear modulation (FiLM). | Factorized workers are not sufficient to establish communication-free decentralized execution. |
| Temporal structure | A fresh goal is emitted every environment step; the worker pools a sliding window. The same horizon controls transition scoring and optional recurrent dilation. | This is FuN-style temporal structure, not a manager that communicates only once every horizon steps. |
| Manager learning | `mappo.py:manager_update` weights transition cosine by extrinsic generalized advantage estimates, with detached displacement targets. Under shared rewards, one team advantage is broadcast across agents. | This does not implement agent- or coalition-specific counterfactual credit assignment. |
| Worker learning | PPO uses separately normalized extrinsic and optional intrinsic advantages. The intrinsic weight can anneal. | This changes FuN's original training recipe; it is an implementation choice requiring ablation, not automatically a new contribution. |
| Defaults | `conf/algorithm/feudal_mappo_jax.yaml`: intrinsic coefficient 0, feedforward manager, goal dimension 32, horizon 10. Worker defaults to concatenation. | The default is not the full recurrent, intrinsically motivated configuration. Setting the intrinsic coefficient to zero does not remove the manager or goals. |
| Scalability | Input concatenation and dense `N × goal_dim` output heads fix the agent count and ordering. | Training larger fixed teams is distinct from transferring one trained policy to unseen team sizes. No permutation equivariance is built into this manager. |

Writing `x_t` for the manager input and `s_t^i` for agent slot `i` in its latent state, the current manager approximately optimizes

\[
L_M=-\operatorname{masked\ mean}_{t,i}\left[\widehat A^M_t\,
\cos\big(\operatorname{sg}[s_{t+c}^i-s_t^i],g_t^i\big)\right].
\]

Here `sg` means stop-gradient. The worker receives a normalized sum of recent goals and uses

\[
\widehat A^W_{t,i}=\operatorname{normalize}(A^{ext}_{t,i})+
\alpha_t\operatorname{normalize}(A^{int}_{t,i}).
\]

Per-agent reward modes modify the manager's advantage layout. Their presence in the environment wrappers should not be described as a new credit-assignment mechanism in the feudal algorithm without isolating and evaluating that combination.

**The closest literature and what it rules out.** Read the first six rows before selecting a final contribution claim.

| Work | Established contribution relevant here | Consequence for this project |
|---|---|---|
| [FeUdal Networks, ICML 2017](https://proceedings.mlr.press/v70/vezhnevets17a.html) | Learned directional goals, separate levels, intrinsic goal-following rewards, and temporal abstraction. | These are inherited ingredients. A careful multiagent extension must explain the extra difficulty introduced by interacting workers. |
| [Feudal Multi-Agent Hierarchies, 2019](https://arxiv.org/abs/1901.08492) | A manager communicates subgoals to multiple simultaneous workers; effectiveness depends on an adequate supplied subgoal set. | Rules out the broad “first multiagent feudal hierarchy” claim. Learned latent goals can distinguish your implementation from its specified subgoal repertoire. |
| [Feudal Latent Space Exploration, ALA 2020 version](https://ala2020.vub.ac.be/papers/ALA2020_paper_22.pdf), [journal version, online 2022 / issue 2023](https://pubmed.ncbi.nlm.nih.gov/35167482/) | A global commander generates a stochastic shared latent code; local policies condition on it and observations. The accessible workshop version optimizes through a MADDPG critic. | Rules out “first latent feudal coordination.” Your transition-based directional goals differ from its shared stochastic exploration code. The workshop version explicitly acknowledges the execution centralization issue. |
| [Hierarchical learning with Skill Discovery (HSD), AAMAS 2020](https://aamas.csc.liv.ac.uk/Proceedings/aamas2020/pdfs/p1566.pdf) | A two-level cooperative hierarchy discovers skills and learns strategic selection. | A direct AAMAS precedent for learned hierarchy and teamwork; compare if claiming superior online skill learning. |
| [HAVEN, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26386) | A two-level value-decomposition method addresses coordination between agents and hierarchy levels without domain knowledge or pretraining. | “Hierarchy without predefined skills or pretraining” is already occupied. Important discrete-action comparison. |
| [Hierarchical Message-Passing Policies / HiMPo, 2025 preprint](https://arxiv.org/abs/2507.23604) | Feudal message-passing hierarchies with an advantage-based assignment of lower-level rewards. Its PPO implementation is called HiMPPO. | The most important recent direct comparison found. Simply adding a graph, PPO, or task-aligned reward to a feudal hierarchy is insufficient differentiation. |
| [Feudal Graph Reinforcement Learning, TMLR 2024](https://github.com/tommasomarzi/fgrl) | Hierarchical message passing for modular control, including locomotion and graph clustering. | Morphology-based or graph-based hierarchy alone is not new. Distinguish autonomous team coordination from decomposition of a single physical system. |
| [MAVEN, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f816dc0acface7498e10496222e9db10-Abstract.html) | A shared latent variable supports committed, temporally extended multiagent exploration. | Necessary conceptual comparison for exploration claims; useful empirical comparison on discrete tasks. |
| [ROMA, ICML 2020](https://proceedings.mlr.press/v119/wang20f.html), [RODE, ICLR 2021](https://arxiv.org/abs/2010.01523) | Learned roles condition behavior; RODE discovers restricted role action spaces from action effects. | Distinct per-agent codes and division of labor are not by themselves new. RODE is a useful discrete coordination baseline. |
| [Hierarchical Multi-Agent Skill Discovery (HMASD), NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/c276c3303c0723c83a43b95a44a1fcbf-Abstract-Conference.html) | Joint discovery of individual and team skills, with sequential high-level assignment. | Relevant to reusable skill and sparse-reward claims; account for skill-learning interactions. |
| [Variational Offline Multi-agent Skill Discovery (VO-MASD), IJCAI 2025](https://arxiv.org/abs/2405.16386) | Offline subgroup and temporal abstractions, dynamic grouping, and transfer of skills. | A relevant comparison if pursuing coalition skills or transfer. Its offline data access must be accounted for; merely having its code locally does not make it an equivalent online baseline. |

Additional recent work matters if the scope expands: [HiSSD, 2025 preprint](https://arxiv.org/abs/2503.21200) studies transferable skills from offline multi-task data; [Partner-Aware Skill Discovery, May 2026 preprint](https://arxiv.org/abs/2605.24352) studies partner-conditioned skills and robustness in human–AI collaboration. Neither is an automatic required baseline for a fixed-team online control paper. [Room Clearance with Feudal Hierarchical Reinforcement Learning, 2021](https://arxiv.org/abs/2105.11328) is another explicit multiagent feudal application.

**What is and is not defensibly novel.**

| Candidate claim | Assessment |
|---|---|
| First multiagent FuN / manager with several workers | Not defensible given the precedents above. |
| Learned latent goals for coordinated exploration | Not defensible as a broad first claim. |
| FuN manager with MAPPO workers | A potentially useful combination; presently a weak standalone method contribution. |
| FiLM goal conditioning, goal normalization, or separate advantage streams | Useful engineering and ablations. FiLM itself is established; a new explanation of failure and a general solution could be a contribution. |
| Goals that yield identifiable, controllable coalition effects with appropriate credit | A promising research hypothesis. Requires a precise mechanism and a further focused novelty check. |
| Systematic finding that latent alignment fails to establish useful multiagent hierarchy | Potentially valuable if demonstrated beyond one implementation, with controlled counterexamples and practical consequences. |
| Generalization to unseen agent counts | Not supported by the present fixed-size manager. |

FiLM is a known conditioning layer: [Perez et al.](https://arxiv.org/abs/1709.07871). Claims about the implementation should also avoid saying concatenation cannot express goal-dependent feature use: after a nonlinear activation, the observation Jacobian can depend on the goal. FiLM provides a different inductive bias and optimization path, not that absolute representational separation.

**A gap worth targeting.** I recommend this problem statement:

> Learn temporally coherent coalition directives from online team experience, without a supplied skill library, when agents must repeatedly regroup and achieve effects that no agent can produce alone. Establish that the directives improve task outcomes under matched execution information and are attributable to the commanded workers' behavior.

This is narrower than “solve hierarchical MARL.” Its motivation fits multi-box pushing: enough agents must assemble, maintain contact, move a box toward its assigned destination, and redeploy afterward. Other workers and passive object dynamics can change a global latent state even when a particular worker fails to follow its goal. A collection of individually plausible goals can also be jointly infeasible when they compete for the same workers.

Three candidate mechanisms follow, but choose one central contribution rather than accumulating all three:

1. **Attribute goal achievement to controllable effects.** Learn or estimate the change caused by replacing a worker's or coalition's goal over a fixed interval. Compare against other goals while retaining the same initial state and the other policies' response mechanisms. Train against task-relevant controllable displacement rather than any observed movement. Simulator interventions can validate a learned estimator; count their extra transitions if used in training.
2. **Represent jointly feasible coalition goals.** Condition goals on agent–object or coalition structure, using a shared representation that can express who must act together. Evaluate whether the method coordinates complementary goals and reallocates workers, rather than producing diverse vectors. Existing graph hierarchies and offline grouping methods make a plain graph substitution insufficient.
3. **Learn when directives should change.** Separate the goal-selection interval from the scoring horizon and recurrence radius. Use commitment or learned termination only if the problem requires balancing persistence with reassignment. The current sliding window alone cannot substantiate a communication-efficiency claim.

Counterfactual credit is itself established: [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794) marginalizes one agent's primitive action against a centralized critic. A novel proposal would need to explain what changes at a temporally extended coalition-goal level and why existing credit estimators are inadequate. Do not claim that a goal-level COMA substitution alone settles novelty.

The theoretical issue is concrete. The original [FuN transition-gradient derivation](https://proceedings.mlr.press/v70/vezhnevets17a/vezhnevets17a.pdf) assumes a directional transition distribution centered on the commanded goal. In your multiagent extension, an agent slot's latent transition can depend on all workers and all goals. Summing cosine surrogates is not automatically an unbiased gradient of the team's return. State sufficient assumptions or explicitly treat the update as a surrogate and measure its limitations. With intrinsic learning disabled, the argument that workers will learn to realize this transition model is particularly weak.

**What the saved experiments currently establish.** I loaded `algorithms/feudal_mappo_jax/goal_dependence_probe.results.pkl` without executing its evaluation code. It contains 60 entries: 45 non-zero-goal checkpoints and 15 zero-goal controls, across `feudal`, `feudal_n01`, `feudal_n05`, and `feudal_zerogoal`. There are no FiLM variants in this artifact. For shift 1:

| Stored diagnostic | Count among 45 non-control checkpoints |
|---|---:|
| Real minus agent-permuted return: interval entirely positive | 4 |
| Real minus agent-permuted return: interval entirely negative | 1 |
| Real minus environment-permuted return: interval entirely positive | 3 |
| Real minus environment-permuted return: interval entirely negative | 1 |
| Real minus zeroed return: interval entirely negative | 28 |

For `mjx_16a_4o_trunc_1024`, the average stored real return across three checkpoints is 242.27 for `feudal` and 368.90 for workers trained with zero goals. These are descriptive summaries of saved evaluations, not a new significance analysis or a controlled estimate of the current configuration's performance. They do not support claiming useful learned hierarchy yet.

The first record uses 64 evaluation episodes, goal dimension 16, horizon 10, and normalized pooled goals; current defaults differ. The probe records checkpoint modification time and some settings, but this does not establish full source/configuration provenance. `CLAUDE.md` also records observation and goal-processing changes. Reproduce the observations using frozen training configurations and matching checkpoint code before turning them into paper results.

**Interpretability needs more than a permutation gap.** Agent permutation preserves the pooled marginal goal distribution, but not necessarily the distribution conditional on an agent's observation or identity. Your latent heads are agent-specific and do not impose a shared semantic coordinate system. Goal diversity and poor alignment after swapping slots can therefore reflect incompatible learned coordinates. An evaluation intervention measures sensitivity of a fixed policy to that intervention; it does not measure the training benefit of hierarchy. Likewise, a statistically inconclusive return gap does not prove that a policy never uses its goals.

Use three complementary forms of evidence: retrain without the proposed mechanism; intervene on goals in matched states; and measure actual physical outcomes. For coalition pushing, suitable outcomes include contact-group formation, sustained coalition membership, useful displacement, completion time, and reassignment after a delivery. Freeze workers and the goal representation when evaluating skill reuse; changing the representation also changes what a frozen worker's inputs mean.

**A comparison set that answers the right questions.**

| Priority | Comparison | What it resolves |
|---|---|---|
| Essential | Tuned feedforward and recurrent MAPPO | Improvement over a strong conventional learner; distinguishes hierarchy from added memory. |
| Essential | Flat centralized PPO actor using exactly the manager's execution input, with factorized action heads | Whether access to team information explains the gain. This is an empirical reference, not a guaranteed upper bound. |
| Essential | Nonhierarchical communication policy, such as graph PPO, with matched communication reach and comparable capacity | Whether communication or relational representation is sufficient. |
| Essential | Same encoder/worker interface trained as an ordinary end-to-end latent communication policy | Whether FuN's detached transition objective and goal semantics add value beyond the bottleneck. |
| Essential | Current multiagent FuN implementation, including a trained zero-goal control | Whether a proposed new mechanism improves on the project's starting point. |
| High | HiMPPO and FLE | Direct contemporary hierarchy and latent-feudal alternatives. Preserve original methods and separately label any PPO adaptations. |
| High on discrete tasks | HAVEN; HSD if practical | Whether online learned hierarchy improves beyond established hierarchical MARL. |
| Claim-dependent | MAVEN for exploration; RODE for role allocation; HMASD for team/individual skills | Tests the specific advertised benefit. |
| Claim-dependent | VO-MASD or HiSSD for offline skill transfer | Use only with an explicit offline-data/pretraining comparison protocol. |
| Standard discrete reference | [QMIX](https://arxiv.org/abs/1803.11485) or [QPLEX](https://arxiv.org/abs/2008.01062) | Provides a value-decomposition reference in addition to policy-gradient methods. |

The practical first suite is: MAPPO, recurrent MAPPO, flat centralized PPO, end-to-end latent communication PPO, current multiagent FuN, HiMPPO, and the proposed method. Include a graph communication baseline when the proposed method uses graph structure. Add one strong skill/role/exploration baseline according to the claim. FLE deserves direct comparison for a latent-feudal claim; the full historical list need not appear in every experiment.

The [MAPPO paper](https://arxiv.org/abs/2103.01955) makes implementation quality an important issue for this comparison. Validate your local implementation against an established implementation on at least a small shared benchmark. Identical hyperparameters are useful for controlled ablations, but different published algorithms should receive comparable tuning budgets within their intended training recipes.

**Benchmarks suited to the proposed gap.**

1. **Controlled coalition pushing:** exploit the existing dense/sparse pair, partial sensing, per-box destinations, and coalition requirements. Vary required coalition size, information range, and reward delay independently. Randomize layouts and assignments; include low-coordination controls where a flat policy should suffice. In the multi-goal environment, `coupling_def=random` currently changes object size and ring geometry too, so it is not an isolated coalition-complexity intervention. Some configurations permit sequential delivery instead of simultaneous coalitions; specify which behavior the task actually requires.
2. **An external coalition or continuous-control domain:** Level-Based Foraging with coalition requirements or a VMAS task gives an independent check. Reproducing a HiMPPO benchmark is especially useful for direct comparability; its paper evaluates foraging, VMAS Sampling, and SMACv2. A custom benchmark should explain the mechanism, not carry all external-validity claims. [HiMPo experiment definitions](https://arxiv.org/html/2507.23604v1).
3. **A recognized tactical domain:** use SMAX for fast development if convenient, then a selected SMACv2 suite for broader claims. SMAX and SMACv2 are different benchmarks; report their results separately. SMACv2 was designed to address weaknesses in older SMAC scenarios through procedural variation and stronger partial observability. [JaxMARL](https://arxiv.org/abs/2311.10090), [SMACv2](https://arxiv.org/abs/2212.07489).

Avoid claiming general hierarchical superiority from only easy `3m` or a single box-pushing layout. Continuous tasks do not require forcing discrete value-based baselines into artificial action discretizations; choose compatible comparisons per domain.

**Ablations that would make the mechanism convincing.**

| Change | Hypothesis tested |
|---|---|
| Trained zero goals and fixed per-agent identifiers | Gains require state-dependent directives, not merely extra capacity or symmetry breaking. |
| Single team goal versus per-agent or coalition goals | Coordination needs structured assignment beyond a shared context signal. |
| Feedforward, ordinary recurrent, and dilated managers; recurrent flat reference | Temporal benefit is not simply additional memory. |
| Goal-selection interval 1 versus longer intervals, independently of scoring horizon | Temporal commitment matters beyond smoothing. |
| Intrinsic coefficient zero; normalized intrinsic stream; proposed goal reward | The proposed worker objective explains improvements. |
| Current cosine objective versus proposed manager credit; shared representation held fixed where feasible | The new learning mechanism, rather than a changed encoder, explains improvements. |
| Same information without hierarchy | The advantage is not explained by centralized sensing. |
| Frozen worker and representation, new manager/task | Learned behaviors can be recomposed, if transfer is claimed. |
| Goal interventions at matched simulator states | Goals change commanded physical outcomes in predictable ways. |

Keep execution information in the result tables: local history only, communicating observations, or privileged state. A centralized manager is a legitimate formulation, but ordinary centralized-training/decentralized-execution baselines have weaker execution information. To claim decentralized execution, the manager must become locally implementable or be replaced by an explicitly evaluated communication/distillation mechanism; this is not currently provided.

**Experimental protocol and resource priorities.** Start with a few diagnostic tasks and three seeds to reject failed mechanisms. For final claims, target 8–10 independent training seeds on the central comparisons and enough held-out evaluation episodes to make evaluation noise smaller than training variation. Treat this as a planning recommendation, not an AAMAS requirement. Expand seed counts according to uncertainty and effect size rather than treating any count as universally sufficient.

Report final task success/return, area under the learning curve, steps to a prespecified threshold, seed failure rate, and wall-clock/GPU cost. Count joint environment transitions consistently; also report agent count so agent-transition totals are clear. Charge pretraining, offline dataset generation where applicable, and counterfactual simulator rollouts. Report inference communication separately from training cost.

Use confidence intervals across training seeds. Paired evaluation episodes are useful for checkpoint interventions, but 64 episodes of one checkpoint are not 64 training seeds. Aggregate across diverse tasks using normalized scores with disclosed normalization, interquartile means, and uncertainty estimates; show per-task results so aggregate improvements cannot conceal systematic failures. Avoid choosing the best checkpoint on the final evaluation seeds. These recommendations follow the concerns and tools in [Agarwal et al.](https://arxiv.org/abs/2108.13264) and [rliable](https://github.com/google-research/rliable).

Before final sweeps, match environment code, observation semantics, reset/truncation rules, action masks, reward definitions, step budgets, rollout batches for controlled ablations, and evaluation schedules. Current YAML defaults specify 16 evaluation episodes for feudal versus 32 for flat MAPPO. The feudal worker also always uses per-agent critic outputs, whereas the flat shared-team baseline can use one scalar. The zero-goal arm helps isolate that difference.

Two mathematical wording fixes are advisable in a future documentation pass. Separately normalizing advantages makes alpha their mixing coefficient; it does not make alpha an exact fraction of parameter-gradient magnitude, because policy scores, correlations, and PPO clipping also matter. Annealing the intrinsic term to zero removes that term asymptotically; it does not establish convergence or asymptotic correctness of the whole jointly trained hierarchy.

**Recommended next milestone.** Freeze a reproducible configuration and test current FiLM/intrinsic variants against a trained zero-goal control, an information-matched flat centralized policy, and end-to-end latent communication on one difficult coalition task and one external task. Require a repeatable extrinsic improvement and behaviorally meaningful goal interventions before committing to a large sweep. If these fail, use the failure to motivate a change to goal learning or credit assignment; more horizon/width tuning alone does not establish a contribution.

An eventual abstract could center on a demonstrated failure of transition alignment in coupled systems, a mechanism that learns controllable coalition directives, and an evaluation that separates information, memory, and hierarchy. At present those are proposed contributions, not results. Refresh the literature search before submission, especially around goal-level counterfactual credit, controllable representations, and online coalition skill learning.
