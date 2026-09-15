This project functions mainly as a folder structure and experiment manager. The purpose is for it to be general enough to add any RL environment and use any RL algorithm implementation, while retaining the same folder structure.
### To install
Project requires Python 3.11

When using conda
conda create --name hypergraphs python=3.11
conda install -c conda-forge "libstdcxx-ng>=13"
conda install -c conda-forge libspatialindex

uv pip install -r requirements.txt 

### Feudal goal visualization

Calling `view()` with `feudal_mappo_jax` saves one plot per agent as
`goals_episode_<episode>_agent_<agent>.png` (agent IDs start at 1), alongside
each episode's reward plot and video in the run's logs directory.
Each plot shows that agent's manager goals as stars and reached latent states
as circles, with agent 1 in red, agent 2 in blue, and additional agents in
distinct colors. Lines connect each goal/outcome pair. All agent plots use
one shared 2D PCA projection per episode so their coordinates are comparable.

Goals specify directions: a star represents `s_t + g_t`, and its paired circle
represents `s_(t + goal_horizon)`. The star's distance from the starting state
is illustrative, since the manager does not prescribe a travel distance.
Only complete horizons within the episode are plotted; macro environments
count policy decisions. PCA is fitted separately for each episode, and axis
labels report the retained variance.

The plotting function accepts `save_goal_plot(..., data_pairs=4)` to display
four evenly spaced goal/outcome pairs per agent. For a 1000-step episode,
these correspond to reached states at steps 250, 500, 750, and 1000, paired
with goals issued `goal_horizon` steps earlier. Sampling uses only complete
horizons and is capped at the available number of pairs. `data_pairs=None`
(the default) displays all pairs; the PCA fit always uses all complete pairs.
