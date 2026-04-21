import numpy as np
import gymnasium as gym

from smaclite.smaclite.env import SMACliteEnv
from smaclite.smaclite.env.maps.map import MapPreset


class SmacliteToGymWrapper(gym.Env):
    """
    Wraps a smaclite Gymnasium env to match the (n_agents, obs_dim) observation
    convention and avail_actions info-dict pattern used by the MAPPO pipeline.

    Observation space: Box, shape (n_agents, obs_dim)
    Action space:      MultiDiscrete([n_actions] * n_agents)
    Reward:            scalar team reward
    """

    def __init__(
        self,
        map_name: str,
        use_cpp_rvo2: bool = False,
        render_mode=None,
    ):
        map_info = self._resolve_map_info(map_name)
        self.env = SMACliteEnv(
            map_info=map_info,
            use_cpp_rvo2=use_cpp_rvo2,
            render_mode=render_mode,
        )
        self.max_steps = 512
        self.step_count = 0

        base = self.env.unwrapped
        self.n_agents = base.n_agents
        self.n_enemies = base.n_enemies
        self.enemy_feat_size = base.enemy_feat_size
        self.ally_feat_size = base.ally_feat_size
        n_actions = base.n_actions

        self._obs_dim = int(self.env.observation_space.spaces[0].shape[0])

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, self._obs_dim),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([n_actions] * self.n_agents)

    @staticmethod
    def _resolve_map_info(map_name: str):
        """Resolve flexible map names like ``MMM2`` or ``MAP_MMM2``."""
        normalized = map_name.upper()
        candidate_names = {
            normalized,
            normalized.removeprefix("MAP_"),
        }

        for preset in MapPreset:
            preset_name = preset.name.upper()
            value_name = preset.value.name.upper()
            if preset_name in candidate_names or value_name in candidate_names:
                return preset.value

        known_maps = ", ".join(preset.value.name for preset in MapPreset)
        raise ValueError(
            f"Unknown SMACLite map '{map_name}'. Known maps: {known_maps}"
        )

    def _get_avail_actions(self):
        return np.array(
            self.env.unwrapped.get_avail_actions(), dtype=np.float32
        )  # (n_agents, n_actions)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)
        info["avail_actions"] = self._get_avail_actions()
        return obs, info

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int32).tolist()
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(actions)
        truncated = bool(truncated) or (self.step_count >= self.max_steps)
        obs = np.asarray(obs, dtype=np.float32)
        info["avail_actions"] = self._get_avail_actions()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
