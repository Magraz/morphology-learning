# Literature for waypoint-goal Feudal multiagent RL

Search and verification date: **2 October 2026**. The collection contains **48 papers**, grouped into **7 topics**, for the introduction and background of the AAMAS 2027 paper.

The scope is a manager assigning simple spatial waypoints to workers, with counterfactual goals intended to improve credit assignment and coordination in a tightly coupled continuous-control cooperative object-pushing task. Papers using latent commands are architectural comparisons, not a proposal to change the current waypoint representation.

## How to use the bibliography

[references.bib](references.bib) contains complete author lists, verified titles and publication details, an `abstract` field, and an `annotation` explaining relevance. **Every abstract is an original paraphrase of the author abstract**, not a verbatim copy. The `abstract_url` field links to the original abstract; when there is no separate abstract page, it links to the paper containing it. The standard ACM bibliography style ignores abstracts and research annotations, so they remain useful research metadata without appearing in the submitted reference list.

Entries are separated by topic comments and carry a topic in `keywords`. Within each topic they are ordered by year and citation key. `priority` means: **essential** for central positioning or the implementation; **high** for close follow-up reading; **supporting** for foundations and baseline context; **context** for adjacent settings or methodological guidance. `publication_status` records the form of publication rather than treating all entries as main-track papers. `verification_url` records the primary source used to check the reference, and `urldate` records the verification date.

The bibliography is selected for relevance, not intended as the paper's final reference list. Cite individual entries with, for example, `\\cite{rahmattalabi2016dpp,foerster2018coma,yang2020cm3}`. The manuscript's `root.tex` now points to `references.bib`.

## Start with these comparisons

| Reading order | Paper and source | Verified venue | Why it matters here |
| --- | --- | --- | --- |
| 1 | [D++: Structural Credit Assignment in Tightly Coupled Multiagent Domains](https://jenjenchung.github.io/anthropomorphic/Papers/Rahmattalabi2016dppIROS.pdf)<br>`rahmattalabi2016dpp` | IROS 2016 | The motivating case where useful individual behavior receives poor feedback until others cooperate. |
| 2 | [Counterfactual Multi-Agent Policy Gradients](https://ojs.aaai.org/index.php/AAAI/article/view/11794)<br>`foerster2018coma` | AAAI 2018 | The canonical action-level counterfactual baseline; a key comparison for moving credit to goal decisions. |
| 3 | [Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines](https://openreview.net/forum?id=H1tSsb-AW)<br>`wu2018factorizedbaselines` | ICLR 2018 | The estimator assumptions behind conditioning a baseline on other action or goal factors. |
| 4 | [The Mirage of Action-Dependent Baselines in Reinforcement Learning](https://proceedings.mlr.press/v80/tucker18a.html)<br>`tucker2018mirage` | ICML 2018 | A necessary companion when claiming variance reduction or learning gains from an action-dependent baseline. |
| 5 | [CM3: Cooperative Multi-goal Multi-stage Multi-agent Reinforcement Learning](https://openreview.net/forum?id=S1lEX04tPr)<br>`yang2020cm3` | ICLR 2020 | Close prior work on credit for how one agent affects another agent's goal attainment. |
| 6 | [Feudal Multi-Agent Hierarchies for Cooperative Reinforcement Learning](https://ala2019.vub.ac.be/papers/ALA2019_paper_5.pdf)<br>`ahilan2019fmh` | ALA workshop at AAMAS 2019 | The closest early manager/multiple-worker Feudal architecture; its verified version is a workshop paper. |
| 7 | [FeUdal Networks for Hierarchical Reinforcement Learning](https://proceedings.mlr.press/v70/vezhnevets17a.html)<br>`vezhnevets2017fun` | ICML 2017 | The principal Feudal architectural inspiration, with a different goal representation. |
| 8 | [Hierarchical Message-Passing Policies for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2507.23604)<br>`marzi2025himpo` | arXiv 2025; revised 2026 | Recent Feudal MARL connecting manager goals, graph communication, and rewards based on upper-level advantages. |
| 9 | [MASER: Multi-Agent Reinforcement Learning with Subgoals Generated from Experience Replay Buffer](https://proceedings.mlr.press/v162/jeon22a.html)<br>`jeon2022maser` | ICML 2022 | Multiagent subgoal generation and intrinsic rewards, relevant to the simple waypoint design. |
| 10 | [HAVEN: Hierarchical Cooperative Multi-Agent Reinforcement Learning with Dual Coordination Mechanism](https://ojs.aaai.org/index.php/AAAI/article/view/26386)<br>`xu2023haven` | AAAI 2023 | A direct hierarchical MARL comparison concerned with coordination across agents and levels. |
| 11 | [Aligning Credit for Multi-Agent Cooperation via Model-based Counterfactual Imagination](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p281.pdf)<br>`chai2024macd` | AAMAS 2024 | A recent AAMAS approach extending counterfactual evaluation through imagined trajectories. |
| 12 | [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html)<br>`yu2022mappo` | NeurIPS 2022, Datasets and Benchmarks Track | The main flat PPO baseline and implementation reference. |

Also read [Gaussian Processes as Multiagent Reward Models](https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p330.pdf) for optimistic signals when complementary cooperation is absent, [Difference Rewards Policy Gradients](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1475.pdf) for reward-function counterfactuals, and [Data-Efficient Hierarchical Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2018/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html) for explicit state-space subgoals in continuous control.

## Suggested use in the introduction

1. **Problem and difficulty:** motivate shared-reward attribution and tightly coupled cooperation with D++, COMA, and Gaussian Processes as Multiagent Reward Models. Describe the physical requirement to act together in the pushing task as a property of this paper's environment.
2. **Existing approaches:** briefly cover action/reward counterfactuals, coalition-sensitive credit, and hierarchical coordination. COMA, Dr.Reinforce, SHAQ, HSD, and HAVEN give representative anchors.
3. **Specific gap to investigate:** compare credit for temporally extended manager-generated waypoint assignments with credit for primitive actions, instantaneous rewards, replay-selected subgoals, and discovered skills. Check CM3, FMH, MASER, HiMPo, and MACD carefully before making novelty claims.
4. **Approach and contribution:** explain which goals change in the counterfactual and which remain fixed, why that comparison is informative, and which policy update receives it. Use simple physical waypoints as an interpretable design choice supported by goal-conditioned control literature.
5. **Results:** insert only measured findings from the pushing experiments, including the comparison method, agent counts, reward conditions, and uncertainty. Published gains in other papers are not evidence for this method's performance.

This is a proposed positioning based on the selected literature, not a verified claim that counterfactual goals are entirely absent from prior work.

## Suggested background structure

### Hierarchical Multiagent RL

Define temporally extended decisions and coordinated decomposition with HSD, HAVEN, HMASD, and MASER. Use LDSA, ROMA, and RODE to explain complementary responsibilities and distinguish skills, roles, and subtasks from explicit goal coordinates. OPRE concerns strategic responses to opponents, and VO-MASD concerns offline skill discovery; use them only when those broader settings help the discussion.

Suggested citation cluster: `\\cite{yang2020hsd,xu2023haven,yang2023hmasd,jeon2022maser}`.

### Feudal Multiagent RL

Introduce manager and worker objectives using Dayan and Hinton and FeUdal Networks, then move to the multiworker setup with FMH. HIRO and HAC provide context for spatial subgoals and learning across levels; UVFA supports goal-conditioned value estimates. Carvalho et al. supplies theory under explicit assumptions. FLE uses shared latent exploration, and HiMPo uses hierarchical message passing; explain their different coordination mechanisms while retaining this paper's waypoint definition.

Suggested citation cluster: `\\cite{dayan1992feudal,vezhnevets2017fun,ahilan2019fmh,nachum2018hiro,schaul2015uvfa}`.

### Counterfactuals for credit assignment

Distinguish instantaneous difference rewards, action-value baselines, temporally extended counterfactual effects, and coalition credit. Start with D++, COMA, Dr.Reinforce, and CM3; then use MACD and the Shapley/nucleolus papers where group effects matter. Introduce factorized-baseline assumptions and the limitations identified by Tucker et al. before asserting estimator properties. Potential-based shaping is relevant if the mechanism changes worker rewards rather than only the manager's advantage.

Suggested citation cluster: `\\cite{rahmattalabi2016dpp,foerster2018coma,castellini2021difference,yang2020cm3,chai2024macd}`.

## Design distinctions to keep explicit

| Comparison | What the literature evaluates | What this paper needs to specify |
| --- | --- | --- |
| COMA and factorized baselines | Alternatives for one primitive action factor, conditioned on other factors. | The distribution of alternative waypoints and assumptions about the manager's joint goal policy. |
| D++ and optimistic reward models | Whether useful behavior becomes valuable when hypothetical partners or optimistic joint outcomes supply missing cooperation. | Whether the goal counterfactual changes one worker or also supplies complementary teammate behavior. Changing one goal while holding all teammates fixed is not automatically a D++-style cooperation augmentation. |
| FMH and HiMPo | Manager commands, worker objectives, and coordination across hierarchy levels. | Which level receives counterfactual credit and how worker competence affects goal-value estimates. |
| MASER, HER, HIRO, and HAC | Selecting, relabeling, or stabilizing subgoals and goal-conditioned learning. | How counterfactual goals are evaluated for attribution rather than merely reused for training. |
| Shapley and nucleolus credit | Contributions across coalitions and their interactions. | Whether individual goal comparisons capture complementary group effects or require joint alternatives. |
| Potential-based shaping | Conditions for preserving optimal policies or equilibria. | Whether the actual waypoint-progress reward satisfies those conditions and which objective the guarantee concerns. |

These comparisons are research interpretations, separate from the paraphrased abstracts. A learned critic's predictions for unobserved goal combinations need empirical validation; calling them counterfactuals does not by itself establish causal identification or unbiased gradients.

## Complete topic catalog

Relevance notes and abstract summaries for every entry are in `references.bib`.

### Counterfactual credit, difference rewards, and coalition attribution (16)

Use these papers to explain why shared rewards obscure contributions, how counterfactual comparisons change credit, and why missing complementary behavior creates an additional coordination problem.

| Citation key and primary source | Verified venue / status | Priority |
| --- | --- | --- |
| `devlin2014potentialdifference`<br>[Potential-Based Difference Rewards for Multiagent Reinforcement Learning](https://www.ifaamas.org/Proceedings/aamas2014/aamas/p165.pdf) | AAMAS 2014<br>Conference paper | high |
| `rahmattalabi2016dpp`<br>[D++: Structural Credit Assignment in Tightly Coupled Multiagent Domains](https://jenjenchung.github.io/anthropomorphic/Papers/Rahmattalabi2016dppIROS.pdf) | IROS 2016<br>Conference paper | essential |
| `foerster2018coma`<br>[Counterfactual Multi-Agent Policy Gradients](https://ojs.aaai.org/index.php/AAAI/article/view/11794) | AAAI 2018<br>Conference paper | essential |
| `tucker2018mirage`<br>[The Mirage of Action-Dependent Baselines in Reinforcement Learning](https://proceedings.mlr.press/v80/tucker18a.html) | ICML 2018<br>Conference paper | high |
| `wu2018factorizedbaselines`<br>[Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines](https://openreview.net/forum?id=H1tSsb-AW) | ICLR 2018<br>Conference paper | essential |
| `dixit2020gaussianrewards`<br>[Gaussian Processes as Multiagent Reward Models](https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p330.pdf) | AAMAS 2020<br>Conference paper | essential |
| `yang2020cm3`<br>[CM3: Cooperative Multi-goal Multi-stage Multi-agent Reinforcement Learning](https://openreview.net/forum?id=S1lEX04tPr) | ICLR 2020<br>Conference paper | essential |
| `castellini2021difference`<br>[Difference Rewards Policy Gradients](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1475.pdf) | AAMAS 2021, extended abstract<br>Conference extended abstract | essential |
| `li2021shapleycounterfactual`<br>[Shapley Counterfactual Credits for Multi-Agent Reinforcement Learning](https://kunkuang.github.io/papers/KDD21-ShapleyMARL.pdf) | KDD 2021<br>Conference paper | high |
| `mesnard2021counterfactual`<br>[Counterfactual Credit Assignment in Model-Free Reinforcement Learning](https://proceedings.mlr.press/v139/mesnard21a.html) | ICML 2021<br>Conference paper | high |
| `wang2022shaq`<br>[SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27985d21f0b751b933d675930aa25022-Abstract-Conference.html) | NeurIPS 2022<br>Conference paper | high |
| `chai2024macd`<br>[Aligning Credit for Multi-Agent Cooperation via Model-based Counterfactual Imagination](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p281.pdf) | AAMAS 2024<br>Conference paper | essential |
| `li2025nucleolus`<br>[Nucleolus Credit Assignment for Effective Coalitions in Multi-agent Reinforcement Learning](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1318.pdf) | AAMAS 2025<br>Conference paper | high |
| `liang2025asynchronouscredit`<br>[Asynchronous Credit Assignment for Multi-Agent Reinforcement Learning](https://www.ijcai.org/proceedings/2025/20) | IJCAI 2025<br>Conference paper | context |
| `li2026counterfactualshapley`<br>[Counterfactual Shapley Credit Assignment](https://rlj.cs.umass.edu/2026/papers/Paper45.html) | RLJ / RLC 2026, pre-proceedings<br>Journal/conference paper; official pre-proceedings metadata | context |
| `nigam2026gpat`<br>[Zero-Shot Coordination in Ad Hoc Teams with Generalized Policy Improvement and Difference Rewards](https://ifaamas.org/Proceedings/aamas2026/pdfs/TNEX7143.pdf) | AAMAS 2026<br>Conference paper | context |

### Feudal hierarchies and manager-worker learning (6)

Use these papers to define Feudal control, distinguish manager and worker objectives, and position the architecture against earlier multiworker hierarchies. Latent-command papers are related work; the proposed method uses explicit spatial waypoints.

| Citation key and primary source | Verified venue / status | Priority |
| --- | --- | --- |
| `dayan1992feudal`<br>[Feudal Reinforcement Learning](https://proceedings.neurips.cc/paper/1992/hash/d14220ee66aeec73c49038385428ec4c-Abstract.html) | NIPS 1992<br>Conference paper | high |
| `vezhnevets2017fun`<br>[FeUdal Networks for Hierarchical Reinforcement Learning](https://proceedings.mlr.press/v70/vezhnevets17a.html) | ICML 2017<br>Conference paper | essential |
| `ahilan2019fmh`<br>[Feudal Multi-Agent Hierarchies for Cooperative Reinforcement Learning](https://ala2019.vub.ac.be/papers/ALA2019_paper_5.pdf) | ALA workshop at AAMAS 2019<br>Workshop paper | essential |
| `carvalho2023feudaltheory`<br>[Theoretical Remarks on Feudal Hierarchies and Reinforcement Learning](https://doi.org/10.3233/FAIA230290) | ECAI 2023<br>Conference paper | high |
| `liu2023fle`<br>[Feudal Latent Space Exploration for Coordinated Multi-Agent Reinforcement Learning](https://doi.org/10.1109/TNNLS.2022.3146201) | IEEE TNNLS 2023<br>Journal article | context |
| `marzi2025himpo`<br>[Hierarchical Message-Passing Policies for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2507.23604) | arXiv 2025; revised 2026<br>Preprint; no archival venue verified | essential |

### Hierarchical MARL, skills, roles, and subtask assignment (10)

These papers cover coordinated temporal abstraction, subgoals, skill discovery, and division of labor. Most use discrete benchmark settings or learned skill representations, so task and command semantics matter when making comparisons.

| Citation key and primary source | Verified venue / status | Priority |
| --- | --- | --- |
| `mahajan2019maven`<br>[MAVEN: Multi-Agent Variational Exploration](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f816dc0acface7498e10496222e9db10-Abstract.html) | NeurIPS 2019<br>Conference paper | high |
| `vezhnevets2020opre`<br>[OPtions as REsponses: Grounding behavioural hierarchies in multi-agent reinforcement learning](https://proceedings.mlr.press/v119/vezhnevets20a.html) | ICML 2020<br>Conference paper | context |
| `wang2020roma`<br>[ROMA: Multi-Agent Reinforcement Learning with Emergent Roles](https://proceedings.mlr.press/v119/wang20f.html) | ICML 2020<br>Conference paper | context |
| `yang2020hsd`<br>[Hierarchical Cooperative Multi-Agent Reinforcement Learning with Skill Discovery](https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p1566.pdf) | AAMAS 2020<br>Conference paper | essential |
| `wang2021rode`<br>[RODE: Learning Roles to Decompose Multi-Agent Tasks](https://openreview.net/forum?id=TTUVg6vkNjK) | ICLR 2021<br>Conference paper | context |
| `jeon2022maser`<br>[MASER: Multi-Agent Reinforcement Learning with Subgoals Generated from Experience Replay Buffer](https://proceedings.mlr.press/v162/jeon22a.html) | ICML 2022<br>Conference paper | essential |
| `yang2022ldsa`<br>[LDSA: Learning Dynamic Subtask Assignment in Cooperative Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0b4145b562cc22fb7fa50a2cd17c191d-Abstract-Conference.html) | NeurIPS 2022<br>Conference paper | high |
| `xu2023haven`<br>[HAVEN: Hierarchical Cooperative Multi-Agent Reinforcement Learning with Dual Coordination Mechanism](https://ojs.aaai.org/index.php/AAAI/article/view/26386) | AAAI 2023<br>Conference paper | essential |
| `yang2023hmasd`<br>[Hierarchical Multi-Agent Skill Discovery](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c276c3303c0723c83a43b95a44a1fcbf-Abstract-Conference.html) | NeurIPS 2023<br>Conference paper | high |
| `chen2025vomasd`<br>[Variational Offline Multi-agent Skill Discovery](https://www.ijcai.org/proceedings/2025/538) | IJCAI 2025<br>Conference paper | context |

### Explicit goals, spatial subgoals, and intrinsic reward shaping (5)

Use these references to motivate a simple waypoint interface and goal-conditioned critics. Separate achieved-goal relabeling, hierarchy stabilization, and reward shaping from attribution across agents' goal assignments.

| Citation key and primary source | Verified venue / status | Priority |
| --- | --- | --- |
| `ng1999shaping`<br>[Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | ICML 1999<br>Conference paper | supporting |
| `schaul2015uvfa`<br>[Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html) | ICML 2015<br>Conference paper | supporting |
| `andrychowicz2017her`<br>[Hindsight Experience Replay](https://proceedings.neurips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html) | NIPS 2017<br>Conference paper | supporting |
| `nachum2018hiro`<br>[Data-Efficient Hierarchical Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2018/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html) | NeurIPS 2018<br>Conference paper | essential |
| `levy2019hac`<br>[Learning Multi-Level Hierarchies with Hindsight](https://openreview.net/forum?id=ryzECoAcY7) | ICLR 2019<br>Conference paper | supporting |

### Policy-gradient foundations and continuous-control MARL baselines (5)

These papers support the PPO implementation and comparisons with flat cooperative actor-critic methods. MAPPO and FACMAC are particularly useful baseline candidates for the current task.

| Citation key and primary source | Verified venue / status | Priority |
| --- | --- | --- |
| `lowe2017maddpg`<br>[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html) | NIPS 2017<br>Conference paper | supporting |
| `schulman2017ppo`<br>[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | arXiv 2017<br>Preprint | essential |
| `iqbal2019maac`<br>[Actor-Attention-Critic for Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v97/iqbal19a.html) | ICML 2019<br>Conference paper | supporting |
| `peng2021facmac`<br>[FACMAC: Factored Multi-Agent Centralised Policy Gradients](https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html) | NeurIPS 2021<br>Conference paper | essential |
| `yu2022mappo`<br>[The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html) | NeurIPS 2022, Datasets and Benchmarks Track<br>Conference paper, Datasets and Benchmarks Track | essential |

### Team-value factorization (3)

Use this group to summarize another major approach to cooperative credit. These discrete-action methods provide background; including them does not imply they are suitable direct baselines without adaptation.

| Citation key and primary source | Verified venue / status | Priority |
| --- | --- | --- |
| `rashid2018qmix`<br>[QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v80/rashid18a.html) | ICML 2018<br>Conference paper | supporting |
| `sunehag2018vdn`<br>[Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward](https://www.ifaamas.org/Proceedings/aamas2018/pdfs/p2085.pdf) | AAMAS 2018, extended abstract<br>Conference extended abstract | supporting |
| `wang2021qplex`<br>[QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://openreview.net/forum?id=Rcmk0xxIQV) | ICLR 2021<br>Conference paper | supporting |

### Benchmarks and reliable experimental evaluation (3)

These references support reproducible baseline comparisons, uncertainty reporting, and context for continuous cooperative robot tasks. VMAS is relevant simulator literature, not a description of the current custom environment.

| Citation key and primary source | Verified venue / status | Priority |
| --- | --- | --- |
| `agarwal2021statistical`<br>[Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | NeurIPS 2021<br>Conference paper | supporting |
| `papoudakis2021benchmarking`<br>[Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a8baa56554f96369ab93e4f3bb068c22-Abstract-round1.html) | NeurIPS 2021, Datasets and Benchmarks Track<br>Conference paper, Datasets and Benchmarks Track | supporting |
| `bettini2024vmas`<br>[VMAS: A Vectorized Multi-agent Simulator for Collective Robot Learning](https://link.springer.com/chapter/10.1007/978-3-031-51497-5_4) | DARS 2022; proceedings 2024<br>Conference paper | supporting |

## Verification and publication details

Verification used publisher or proceedings records (PMLR, AAAI, NeurIPS, IFAAMAS, IJCAI, IOS Press, Springer, and RLJ), supplemented by authors' published PDFs, institutional publication pages, and arXiv records. Search results and bibliographic aggregators were discovery aids; the cited evidence is primary. Some OpenReview pages presented browser-verification challenges, so the corresponding published PDFs and author records supplied the accessible evidence. An arXiv abstract can therefore be the abstract source for an entry whose conference venue was checked separately.

Specific details preserved in the BibTeX file:

- [D++: Structural Credit Assignment in Tightly Coupled Multiagent Domains](https://jenjenchung.github.io/anthropomorphic/Papers/Rahmattalabi2016dppIROS.pdf) has **four authors**, including Mitchell Colby, and is an IROS 2016 conference paper. IROS is included because this paper directly motivates the project.
- [FeUdal Networks for Hierarchical Reinforcement Learning](https://proceedings.mlr.press/v70/vezhnevets17a.html) is cited as **ICML 2017**, replacing the less informative preprint-only citation in the research notes.
- [Difference Rewards Policy Gradients](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1475.pdf) and [Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward](https://www.ifaamas.org/Proceedings/aamas2018/pdfs/p2085.pdf) are the verified **AAMAS extended-abstract versions**, not full-length AAMAS papers.
- [Feudal Multi-Agent Hierarchies for Cooperative Reinforcement Learning](https://ala2019.vub.ac.be/papers/ALA2019_paper_5.pdf) is the **Adaptive and Learning Agents workshop at AAMAS 2019** version; no ICLR main-track acceptance is claimed.
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html) is **NeurIPS 2022, Datasets and Benchmarks Track**. [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a8baa56554f96369ab93e4f3bb068c22-Abstract-round1.html) is the **2021 Datasets and Benchmarks** publication, not a conventional volume-34 main-track entry.
- [Feudal Latent Space Exploration for Coordinated Multi-Agent Reinforcement Learning](https://doi.org/10.1109/TNNLS.2022.3146201) is a **2023 journal issue** article whose DOI contains 2022, reflecting the earlier publication process.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) remains an **arXiv technical report**. [Hierarchical Message-Passing Policies for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2507.23604) is an **arXiv preprint**, initially posted in 2025 and revised in June 2026; an archival acceptance was not verified.
- [VMAS: A Vectorized Multi-agent Simulator for Collective Robot Learning](https://link.springer.com/chapter/10.1007/978-3-031-51497-5_4) is a **2024 Springer proceedings chapter for DARS 2022**. Its verified DOI ends in **\_4**, and its publication year is 2024.
- [Counterfactual Shapley Credit Assignment](https://rlj.cs.umass.edu/2026/papers/Paper45.html) is listed on the official **RLJ/RLC 2026 pre-proceedings page**, which still leaves pagination pending. The entry omits page numbers rather than inventing them.
- Some older NeurIPS website records prepend an affiliation to Pieter Abbeel's author name. The HER and MADDPG author lists were normalized against the author/paper records. Empty or unverified page and DOI fields were omitted.

The collection includes a small number of directly relevant IROS, KDD, robotics proceedings, journal, workshop, and preprint references alongside the requested major AI and learning venues. LLM-agent orchestration papers were excluded when their decision model and goals did not match the cooperative control setting. Searches included recent 2026 proceedings; only relevant papers with verifiable publication evidence were added. NeurIPS 2026 proceedings were not assumed to be available before that conference.

This is an extensive targeted search, not a systematic review or a certification of novelty. Update the closest related-work entries before submission, especially provisional records and preprints.

## Bibliography validation

All 48 entries have unique keys, an abstract paraphrase, a source URL, a verification URL, and publication-status metadata. All entries parsed with BibTeX and rendered into a bibliography PDF using the manuscript's `ACM-Reference-Format.bst`, without compilation errors. Some records have no verified publisher, pagination, or publisher location, and the ACM style reports those optional missing fields as warnings. Those values are left unset rather than guessed. AAAI records use the publisher's journal-style proceedings representation so that both volume and issue are retained correctly.
