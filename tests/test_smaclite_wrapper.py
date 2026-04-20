import numpy as np

from algorithms.create_env import make_vec_env
from environments.smaclite.wrapper import SmacliteToGymWrapper
from environments.types import EnvironmentEnum


def _sample_valid_actions(avail_actions: np.ndarray) -> np.ndarray:
    return avail_actions.argmax(axis=-1).astype(np.int32)


def test_smaclite_wrapper_reset_and_step():
    env = SmacliteToGymWrapper("MMM2")
    try:
        obs, info = env.reset(seed=0)

        assert obs.shape == env.observation_space.shape
        assert info["avail_actions"].shape == (env.n_agents, env.action_space.nvec[0])

        actions = _sample_valid_actions(info["avail_actions"])
        next_obs, reward, terminated, truncated, next_info = env.step(actions)

        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert next_info["avail_actions"].shape == info["avail_actions"].shape
    finally:
        env.close()


def test_smaclite_vector_env_exposes_action_masks():
    vec_env = make_vec_env(
        EnvironmentEnum.SMACLITE,
        n_agents=10,
        n_envs=2,
        use_async=False,
        env_variant="MMM2",
    )
    try:
        obs, infos = vec_env.reset(seed=[0, 1])

        assert obs.shape[:2] == (2, 10)
        assert infos["avail_actions"].shape == (
            2,
            10,
            vec_env.single_action_space.nvec[0],
        )

        actions = np.stack(
            [_sample_valid_actions(mask) for mask in infos["avail_actions"]], axis=0
        )
        next_obs, rewards, terminated, truncated, next_infos = vec_env.step(actions)

        assert next_obs.shape == obs.shape
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert next_infos["avail_actions"].shape == infos["avail_actions"].shape
    finally:
        vec_env.close()
