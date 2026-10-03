Keywords: Feudal Reinforcement Learning, Multiagent Reinforcement Learning, Counterfactuals, Credit Assignment, Difference Rewards, Hierarchical Reinforcement Learning.

This folder called "paper" lies within the codebase of my current research code. The focus of this research paper is to develop an multiagent learning algorithm that uses the Feudal framework to produce goals for worker agents to achieve the environment's task. The key contribution this work aims to produce is a clever use of counterfactual goals to promote better credit assignment as well as better coordination.

Main inspirations for this work:
@article{DBLP:journals/corr/VezhnevetsOSHJS17,
  author       = {Alexander Sasha Vezhnevets and
                  Simon Osindero and
                  Tom Schaul and
                  Nicolas Heess and
                  Max Jaderberg and
                  David Silver and
                  Koray Kavukcuoglu},
  title        = {FeUdal Networks for Hierarchical Reinforcement Learning},
  journal      = {CoRR},
  volume       = {abs/1703.01161},
  year         = {2017},
  url          = {https://doi.org/10.48550/arXiv.1703.01161},
  doi          = {10.48550/ARXIV.1703.01161},
  eprinttype   = {arXiv},
  eprint       = {1703.01161},
  timestamp    = {Thu, 01 Oct 2026 11:29:56 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/VezhnevetsOSHJS17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@INPROCEEDINGS{7759651,
  author={Rahmattalabi, Aida and Chung, Jen Jen and Colby, Mitchell and Tumer, Kagan},
  booktitle={2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={D++: Structural credit assignment in tightly coupled multiagent domains}, 
  year={2016},
  volume={},
  number={},
  pages={4424-4429},
  keywords={Robot kinematics;Robot sensing systems;Neural networks;Environmental monitoring;Multi-robot systems;Training},
  doi={10.1109/IROS.2016.7759651}}


Your job as an AI agent will be to help me write the research paper based on this work. It will be submitted to AAMAS 2027 in the Main Track.

The paper will have 7 sections: an abstract, an introduction, background, method, experiments, results, and a conclusion. These should be broken down into their own .tex files and placed in the "sections" folder.

The main TEX file is called root.tex, it uses the aamas.cls styling.

When looking for sources make sure to verify that the source exists, and that the provided citations are correct.

The following is guidance for writing the sections of the paper:
INTRODUCTION
When writing the introduction and abstract, make sure the content reflects the answers to these 8 questions:

1. What is the problem and why do I care?
2. Why is it important/difficult?
3. What has been done already in this problem area?
4. What particular problem remains unsolved?
5. How did you solve it?
6. What is cool about your approach?
7. What were your key results?
8. What are the contributions of this paper?

BACKGROUND
The background section should include the following subsections:
- Hierarchical Multiagent RL
- Feudal Multiagent RL
- Counterfactuals for credit assignment

Guidance on sections ends here.

Idea:
Here's a thought dump on why i think goals and the feudal architecture are necessary. For D++ calculation we need a G function that is a 1 to 1 mapping of joint state ->reward. Not only that but the majority of the reward structure has to be instantaneous. This makes it hard to calculate D++ in settings where just "being there" is not enough. Not only the state has to be aligned, but also the actions. That's what the box pushing enables, just being there touching the box won't solve the task. You need to take actions together to push the box. The goals are a proxy of action alignment, without caring about all the intermediate actions. 
So the question the new D++ would answer is not just, what if more agents where with me? but what if more agents where taking actions/following goals with me?
So i don´t know if this should be a reward shaping term or part of the advantage. But the goals would enable me to calculate two things, the cost of abandonding my goal and moving towards you, and the benefit of once being there taking the following the same goal.
For this I need to use a critic that can calculate V_i(s, gi, g++) where g_++ is the counterfactual goal all other agents would take.
