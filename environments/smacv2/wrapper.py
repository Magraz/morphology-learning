import numpy as np
import gymnasium as gym


class SMACv2ToGymWrapper(gym.Env):
    """
    Wraps SMACv2's StarCraft2Env to expose a Gymnasium-compatible interface,
    matching the (n_agents, obs_dim) observation convention used by the rest
    of the MAPPO pipeline.

    Observation space: Box, shape (n_agents, obs_dim)
    Action space:      MultiDiscrete([n_actions] * n_agents)
    Reward:            scalar team reward (SMACv2 is fully cooperative)
    Truncated:         always False (SMACv2 handles its own episode limit)
    """

    def __init__(self, map_name: str, seed: int = 0):
        from smacv2.env import StarCraft2Env

        self.env = StarCraft2Env(map_name=map_name, seed=seed)
        env_info = self.env.get_env_info()

        self.n_agents = env_info["n_agents"]
        self._obs_dim = env_info["obs_shape"]
        self._n_actions = env_info["n_actions"]

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, self._obs_dim),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([self._n_actions] * self.n_agents)

        # Tracks available actions for each agent; populated after reset()
        self._avail_actions = np.ones(
            (self.n_agents, self._n_actions), dtype=np.float32
        )

    def _get_avail_actions(self):
        return np.array(
            [self.env.get_avail_agent_actions(i) for i in range(self.n_agents)],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):  # noqa: ARG002
        self.env.reset()
        obs = np.array(self.env.get_obs(), dtype=np.float32)  # (n_agents, obs_dim)
        self._avail_actions = self._get_avail_actions()
        return obs, {"avail_actions": self._avail_actions.copy()}

    def step(self, actions):
        """
        actions: np.ndarray shape (n_agents,), dtype int

        Returns: obs, reward, terminated, truncated=False, info
        """
        reward, terminated, info = self.env.step(actions.tolist())
        obs = np.array(self.env.get_obs(), dtype=np.float32)
        self._avail_actions = self._get_avail_actions()
        info["avail_actions"] = self._avail_actions.copy()
        return obs, float(reward), bool(terminated), False, info

    def close(self):
        self.env.close()

    def render(self):
        pass
