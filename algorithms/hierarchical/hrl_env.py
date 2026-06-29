"""Hierarchical macro-action environment.

Wraps a base box2d_suite env and exposes a *discrete* high-level interface: the
high-level action selects one of the 4 frozen skills (see ``skills.py``), and the
wrapper runs that skill for ``macro_len`` low-level steps before handing control
back. This makes the high-level problem look like an ordinary discrete-action
gym env, so the existing MAPPO training stack trains the controller unchanged.

Two decision scopes:
- ``"agent"``: each agent picks its own skill. Obs ``(n_agents, obs_dim)``,
  action ``MultiDiscrete([n_skills] * n_agents)``.
- ``"team"``: one skill for the whole team. Obs ``(1, n_agents * obs_dim)``
  (flattened team state), action ``MultiDiscrete([n_skills])``. The single
  high-level agent maps to ``obs_space.shape[0] == 1`` in the trainer.
"""

import gymnasium as gym
import numpy as np
import torch

from environments.types import EnvironmentEnum
from algorithms.hierarchical.skills import SKILL_ORDER, load_skills


class HierarchicalSkillEnv(gym.Env):
    def __init__(self, env_params: dict):
        super().__init__()

        # Each async worker hosts a single env; cap torch to one thread so many
        # parallel workers don't oversubscribe cores (and to avoid OpenMP
        # thread-pool contention inside forked workers).
        torch.set_num_threads(1)

        self.n_agents = env_params.get("n_agents")
        self.macro_len = int(env_params.get("macro_len", 10))
        self.decision_scope = env_params.get("decision_scope", "agent")
        assert self.decision_scope in ("team", "agent"), (
            f"decision_scope must be 'team' or 'agent', got {self.decision_scope}"
        )
        self.device = env_params.get("device", "cpu")
        self.n_skills = len(SKILL_ORDER)

        # Build the base env via the shared factory (imported lazily to avoid a
        # circular import with create_env).
        from algorithms.create_env import make_single_env

        base_env_name = EnvironmentEnum(env_params.get("base_environment"))
        self.base_env = make_single_env(base_env_name, self.n_agents, env_params)

        # Per-agent obs/action dims come straight from the base env.
        base_obs_space = self.base_env.observation_space
        self.obs_dim = base_obs_space.shape[1]
        base_act_space = self.base_env.action_space
        self.skill_action_dim = base_act_space.shape[1]

        # Load the 4 frozen skill actors (index = high-level action).
        self.skills = load_skills(
            obs_dim=self.obs_dim,
            action_dim=self.skill_action_dim,
            hidden_dim=int(env_params.get("skill_hidden_dim", 168)),
            device=self.device,
            experiment=env_params.get("skill_experiment", "mlp_shared"),
            trial=str(env_params.get("skill_trial", "0")),
            skill_batches=env_params.get("skill_batches"),
        )

        # High-level spaces.
        if self.decision_scope == "agent":
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_agents, self.obs_dim),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.MultiDiscrete(
                [self.n_skills] * self.n_agents
            )
        else:  # team
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, self.n_agents * self.obs_dim),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.MultiDiscrete([self.n_skills])

        # Cache of the latest per-agent base observation (n_agents, obs_dim).
        self._last_obs = None

    def _high_level_obs(self) -> np.ndarray:
        """Shape the cached base obs into the high-level observation."""
        if self.decision_scope == "agent":
            return self._last_obs.astype(np.float32)
        return self._last_obs.reshape(1, -1).astype(np.float32)

    def _skill_indices(self, high_level_action) -> np.ndarray:
        """Map a high-level action to a per-agent skill index array (n_agents,)."""
        action = np.asarray(high_level_action).reshape(-1).astype(np.int64)
        if self.decision_scope == "team":
            return np.full(self.n_agents, action[0], dtype=np.int64)
        return action

    def _low_level_actions(self, skill_idx: np.ndarray) -> np.ndarray:
        """Query each agent's chosen skill for its continuous action.

        Agents sharing a skill are batched through that actor in one forward.
        """
        actions = np.zeros((self.n_agents, self.skill_action_dim), dtype=np.float32)
        obs_t = torch.from_numpy(self._last_obs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            for s in np.unique(skill_idx):
                mask = skill_idx == s
                agent_obs = obs_t[mask]
                skill_action, _ = self.skills[int(s)].act(
                    agent_obs, deterministic=True
                )
                actions[mask] = skill_action.cpu().numpy()
        return actions

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = np.asarray(obs)
        return self._high_level_obs(), info

    def step(self, high_level_action):
        skill_idx = self._skill_indices(high_level_action)

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.macro_len):
            low_level_actions = self._low_level_actions(skill_idx)
            obs, reward, terminated, truncated, info = self.base_env.step(
                low_level_actions
            )
            self._last_obs = np.asarray(obs)
            total_reward += reward
            if terminated or truncated:
                break

        return self._high_level_obs(), total_reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
