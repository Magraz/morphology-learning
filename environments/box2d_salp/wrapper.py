import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import numpy as np
from environments.box2d_salp.domain_jax import JAX2DSalpChainEnv


class JAX2DSalpChainWrapper(gym.Env):
    """Gym wrapper for JAX2D Salp Chain Environment"""

    def __init__(
        self, n_agents=12, world_size=(40, 30), max_steps=1000, render_mode=None
    ):
        super().__init__()

        self.jax_env = JAX2DSalpChainEnv(
            n_agents=n_agents,
            world_size=world_size,
            max_steps=max_steps,
            render_mode=render_mode,
        )

        self.n_agents = n_agents

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_agents, 17), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_agents, 2), dtype=np.float32
        )

        # Initialize RNG key
        self.key = jax.random.PRNGKey(42)
        self.env_state = None

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)

        self.key, reset_key = jax.random.split(self.key)
        obs, self.env_state = self.jax_env.reset(reset_key)

        # Convert JAX arrays to numpy for Gym compatibility
        obs_np = np.array(obs)

        return obs_np, {}

    def step(self, action):
        """Step the environment"""
        # Convert numpy action to JAX array
        action_jax = jnp.array(action)

        # Step the JAX environment
        obs, self.env_state, reward, terminated, truncated, info = self.jax_env.step(
            self.env_state, action_jax
        )

        # Convert JAX arrays to numpy
        obs_np = np.array(obs)
        reward_np = float(reward)

        return obs_np, reward_np, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment"""
        return self.jax_env.render(self.env_state)

    def close(self):
        """Close the environment"""
        pass
