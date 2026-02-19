from environments.types import EnvironmentEnum, EnvironmentParams
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from environments.types import EnvironmentEnum
import gymnasium as gym
import numpy as np


class PettingZooToGymWrapper(gym.Env):
    """
    Wrapper to convert PettingZoo parallel_env to Gymnasium Env
    for use with Gymnasium's vectorization
    """

    def __init__(self, pettingzoo_env):
        self.env = pettingzoo_env
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)

        # Get spaces from first agent (assuming homogeneous agents)
        first_agent = self.agents[0]
        single_obs_space = self.env.observation_space(first_agent)
        single_action_space = self.env.action_space(first_agent)

        # Store whether actions are discrete
        self.is_discrete = isinstance(single_action_space, gym.spaces.Discrete)

        # Create stacked observation space: (n_agents, obs_dim)
        if isinstance(single_obs_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=np.tile(single_obs_space.low, (self.n_agents, 1)),
                high=np.tile(single_obs_space.high, (self.n_agents, 1)),
                shape=(self.n_agents, *single_obs_space.shape),
                dtype=single_obs_space.dtype,
            )
        else:
            # For discrete observation spaces
            self.observation_space = gym.spaces.MultiDiscrete(
                [single_obs_space.n] * self.n_agents
            )

        # Create stacked action space
        if isinstance(single_action_space, gym.spaces.Box):
            # Continuous actions: (n_agents, action_dim)
            self.action_space = gym.spaces.Box(
                low=np.tile(single_action_space.low, (self.n_agents, 1)),
                high=np.tile(single_action_space.high, (self.n_agents, 1)),
                shape=(self.n_agents, *single_action_space.shape),
                dtype=single_action_space.dtype,
            )
        elif isinstance(single_action_space, gym.spaces.Discrete):
            # Discrete actions: (n_agents,) where each element is 0 to n-1
            # Use MultiDiscrete which expects shape (n_agents,) with integer actions
            self.action_space = gym.spaces.MultiDiscrete(
                [single_action_space.n] * self.n_agents
            )
        else:
            raise NotImplementedError(
                f"Action space type {type(single_action_space)} not supported"
            )

    def reset(self, seed=None, options=None):
        """Reset environment and return stacked observations"""
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()

        # Get observations for all agents
        obs_dict, _ = self.env.reset()

        # Stack observations: (n_agents, obs_dim)
        obs_list = [obs_dict[agent] for agent in self.agents]
        stacked_obs = np.stack(obs_list, axis=0)

        info = {"agents": self.agents}

        return stacked_obs, info

    def step(self, actions):
        """
        Step environment with stacked actions

        Args:
            actions: numpy array
                - For discrete: shape (n_agents,) with integer values
                - For continuous: shape (n_agents, action_dim) with float values

        Returns:
            obs: (n_agents, obs_dim)
            reward: scalar (sum of all agent rewards)
            terminated: bool
            truncated: bool
            info: dict
        """
        # Validate action shape
        if self.is_discrete:
            # Discrete actions should be 1D array of integers
            if actions.ndim != 1 or actions.shape[0] != self.n_agents:
                raise ValueError(
                    f"Discrete actions should have shape ({self.n_agents},), got {actions.shape}"
                )
            # Ensure integer type
            actions = actions.astype(np.int32)
        else:
            # Continuous actions should be 2D array
            if actions.ndim != 2 or actions.shape[0] != self.n_agents:
                raise ValueError(
                    f"Continuous actions should have shape ({self.n_agents}, action_dim), got {actions.shape}"
                )

        # Convert stacked actions to dict
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}

        # Step environment
        obs_dict, reward_dict, termination_dict, truncation_dict, info_dict = (
            self.env.step(action_dict)
        )

        # Stack observations
        obs_list = [obs_dict[agent] for agent in self.agents]
        stacked_obs = np.stack(obs_list, axis=0)

        # Aggregate rewards (you can change this to mean, max, etc.)
        total_reward = sum(reward_dict.values())

        # Check if any agent is done
        terminated = any(termination_dict.values())
        truncated = any(truncation_dict.values())

        # Collect individual rewards for info
        local_rewards = np.array([reward_dict[agent] for agent in self.agents])

        info = {
            "local_rewards": local_rewards,
            "agents": self.agents,
        }

        return stacked_obs, total_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def get_state_and_action_dims(
    env_name: EnvironmentEnum, n_agents: int, render_mode: str = None, map_name: str = None
):
    match (env_name):
        case EnvironmentEnum.BOX2D_SALP:
            from environments.box2d_salp.domain import SalpChainEnv

            # Environment configuration
            env = SalpChainEnv(n_agents=n_agents, render_mode=render_mode)
            state_dim = env.observation_space.shape[1]
            action_dim = env.action_space.shape[1]

        case EnvironmentEnum.MULTI_BOX:
            from environments.multi_box_push.domain import MultiBoxPushEnv

            # Environment configuration
            env = MultiBoxPushEnv(n_agents=n_agents, render_mode=render_mode)
            state_dim = env.observation_space.shape[1]
            action_dim = env.action_space.shape[1]

        case EnvironmentEnum.MPE_SPREAD:
            from mpe2 import simple_spread_v3

            pz_env = simple_spread_v3.parallel_env(
                N=n_agents,
                local_ratio=0.5,
                max_cycles=25,
                continuous_actions=False,
                dynamic_rescaling=True,
                render_mode=render_mode,
            )
            # Wrap PettingZoo env for Gymnasium compatibility
            env = PettingZooToGymWrapper(pz_env)
            state_dim = env.observation_space.shape[1]
            action_dim = env.action_space.nvec[0]

        case EnvironmentEnum.MPE_SIMPLE:
            from mpe2 import simple_v3

            pz_env = simple_v3.parallel_env(
                max_cycles=25,
                continuous_actions=False,
                render_mode=render_mode,
            )
            # Wrap PettingZoo env for Gymnasium compatibility
            env = PettingZooToGymWrapper(pz_env)
            state_dim = env.observation_space.shape[1]
            action_dim = env.action_space.nvec[0]

        case EnvironmentEnum.SMACV2:
            from environments.smacv2.wrapper import SMACv2ToGymWrapper

            env = SMACv2ToGymWrapper(map_name=map_name)
            state_dim = env.observation_space.shape[1]   # obs_dim per agent
            action_dim = env.action_space.nvec[0]        # n_actions per agent
            env.close()

    return state_dim, action_dim


def make_vec_env(
    env_name: EnvironmentEnum,
    n_agents: int,
    n_envs: int,
    use_async: bool = True,
    map_name: str = None,
):
    """
    Create a vectorized environment using Gymnasium's built-in vectorization.

    Args:
        env_name: Environment type
        n_envs: Number of parallel environments
        n_agents: Number of agents per environment
        use_async: If True, use AsyncVectorEnv (parallel). If False, use SyncVectorEnv (sequential)

    Returns:
        Gymnasium VectorEnv (AsyncVectorEnv or SyncVectorEnv)
    """

    # Create factory functions that return NEW environment instances
    def make_env_fn():
        """Factory function that creates a new environment instance"""
        match env_name:
            case EnvironmentEnum.BOX2D_SALP:
                from environments.box2d_salp.domain import SalpChainEnv

                return SalpChainEnv(n_agents=n_agents, render_mode=None)

            case EnvironmentEnum.MULTI_BOX:
                from environments.multi_box_push.domain import MultiBoxPushEnv

                return MultiBoxPushEnv(n_agents=n_agents, render_mode=None)

            case EnvironmentEnum.MPE_SPREAD:
                from mpe2 import simple_spread_v3

                pz_env = simple_spread_v3.parallel_env(
                    N=n_agents,
                    local_ratio=0.5,
                    max_cycles=25,
                    continuous_actions=False,
                    dynamic_rescaling=True,
                    render_mode=None,
                )
                # Wrap PettingZoo env to make it Gymnasium-compatible
                return PettingZooToGymWrapper(pz_env)

            case EnvironmentEnum.MPE_SIMPLE:
                from mpe2 import simple_v3

                pz_env = simple_v3.parallel_env(
                    max_cycles=25,
                    continuous_actions=False,
                    render_mode=None,
                )
                # Wrap PettingZoo env to make it Gymnasium-compatible
                return PettingZooToGymWrapper(pz_env)

            case EnvironmentEnum.SMACV2:
                from environments.smacv2.wrapper import SMACv2ToGymWrapper

                return SMACv2ToGymWrapper(map_name=map_name)

    # Create list of factory functions (not environment instances!)
    env_fns = [make_env_fn for _ in range(n_envs)]

    # Create vectorized environment using Gymnasium's API
    if use_async and n_envs > 1:
        # AsyncVectorEnv runs environments in parallel using multiprocessing
        return AsyncVectorEnv(env_fns)
    else:
        # SyncVectorEnv runs environments sequentially (useful for debugging)
        return SyncVectorEnv(env_fns)
