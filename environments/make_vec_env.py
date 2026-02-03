import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from environments.box2d_salp.domain import make_salp_env
from environments.types import EnvironmentEnum


def make_vec_env(
    env_name: EnvironmentEnum,
    n_envs: int,
    use_async: bool = True,
):
    """
    Create a vectorized environment using Gymnasium's built-in vectorization.

    Args:
        env_name: Environment type
        n_envs: Number of parallel environments
        n_agents: Number of agents per environment
        n_target_areas: Number of target areas
        sensing_radius: Sensing radius for agents
        use_async: If True, use AsyncVectorEnv (parallel). If False, use SyncVectorEnv (sequential)

    Returns:
        Gymnasium VectorEnv (AsyncVectorEnv or SyncVectorEnv)
    """
    if env_name == EnvironmentEnum.BOX2D_SALP:
        # Create list of environment factory functions
        env_fns = [make_salp_env() for _ in range(n_envs)]

        # Create vectorized environment using Gymnasium's API
        if use_async and n_envs > 1:
            # AsyncVectorEnv runs environments in parallel using multiprocessing
            return AsyncVectorEnv(env_fns)
        else:
            # SyncVectorEnv runs environments sequentially (useful for debugging)
            return SyncVectorEnv(env_fns)

    else:
        raise NotImplementedError(f"Vectorization not implemented for {env_name}")
