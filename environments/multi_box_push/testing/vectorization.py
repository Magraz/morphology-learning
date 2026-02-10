# Test script
from environments.make_vec_env import make_vec_env
from environments.types import EnvironmentEnum
import numpy as np

# Create vectorized environment
vec_env = make_vec_env(
    env_name=EnvironmentEnum.BOX2D_SALP,
    n_envs=4,
    n_agents=3,
    n_target_areas=1,
    use_async=True,  # Use parallel processing
)

print(f"Created vectorized env: {type(vec_env)}")
print(f"Number of envs: {vec_env.num_envs}")
print(f"Observation space: {vec_env.single_observation_space}")
print(f"Action space: {vec_env.single_action_space}")

# Test reset
obs, infos = vec_env.reset()
print(f"\nAfter reset:")
print(f"Observation shape: {obs.shape}")  # Should be (4, 3, 18)

# Test step
actions = np.random.uniform(-1, 1, size=(4, 3, 4))  # (n_envs, n_agents, action_dim)
obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

print(f"\nAfter step:")
print(f"Observation shape: {obs.shape}")  # (4, 3, 18)
print(f"Rewards shape: {rewards.shape}")  # (4,)
print(f"Terminateds shape: {terminateds.shape}")  # (4,)
print(f"Info keys: {infos.keys()}")
if "local_rewards" in infos:
    print(f"Local rewards shape: {infos['local_rewards'].shape}")  # (4, 3)

vec_env.close()
