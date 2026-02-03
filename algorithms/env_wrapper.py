from environments.types import EnvironmentEnum
import numpy as np


class EnvWrapper:
    def __init__(self, env, env_name, n_agents):
        self.env = env
        self.env_name = env_name
        self.n_agents = n_agents

    def step(self, actions):
        global_reward = 0

        if (
            self.env_name == EnvironmentEnum.MPE_SPREAD
            or self.env_name == EnvironmentEnum.MPE_SIMPLE
        ):
            next_obs = []
            local_rewards = []

            next_obs, local_rewards, terminated, truncated, info = self.env.step(
                actions.squeeze(-1).numpy()
            )

            local_rewards = [local_rewards]

        else:
            next_obs, global_reward, terminated, truncated, info = self.env.step(
                actions
            )
            local_rewards = np.array(info["local_rewards"])

        return next_obs, global_reward, local_rewards, terminated, truncated, info

    def reset(self):
        if (
            self.env_name == EnvironmentEnum.MPE_SPREAD
            or self.env_name == EnvironmentEnum.MPE_SIMPLE
        ):
            obs, _ = self.env.reset()

        return obs
