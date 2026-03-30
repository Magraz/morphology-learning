import math
from collections import deque

import numpy as np
import torch


class IntrinsicReward:
    """Episodic curiosity-based intrinsic reward using k-NN distances."""

    def __init__(
        self,
        *,
        n_agents: int,
        n_parallel_envs: int,
        obs_dim: int,
        k: int = 4,
        memory_capacity: int = 10_000,
    ):
        self.n_agents = n_agents
        self.n_parallel_envs = n_parallel_envs
        self.obs_dim = obs_dim
        self.k = k
        self._memory: deque[torch.Tensor] = deque(maxlen=memory_capacity)

    def on_rollout_step(self, obs: np.ndarray) -> None:
        """Store all agent observations from all envs into global memory.

        Args:
            obs: shape (n_envs, n_agents, obs_dim)
        """
        flat = torch.as_tensor(
            obs.reshape(-1, self.obs_dim), dtype=torch.float32
        )
        self._memory.extend(flat)

    def compute_reward(self, obs: np.ndarray) -> np.ndarray:
        """Compute intrinsic rewards for a batch of observations.

        Args:
            obs: shape (n_envs, n_agents, obs_dim)

        Returns:
            rewards: shape (n_envs, n_agents)
        """
        n_envs, n_agents, _ = obs.shape
        flat = torch.as_tensor(obs.reshape(-1, self.obs_dim), dtype=torch.float32)
        rewards = np.empty(n_envs * n_agents)
        for i, state in enumerate(flat):
            rewards[i] = self._compute_intrinsic_reward(state)
        return rewards.reshape(n_envs, n_agents)

    def _compute_intrinsic_reward(self, current_state: torch.Tensor) -> float:
        if len(self._memory) == 0:
            return 1.0

        memory_tensor = torch.stack(self._memory, dim=0)

        s_dist = (
            torch.cdist(
                current_state.unsqueeze(0),
                memory_tensor,
                p=2,
                compute_mode="use_mm_for_euclid_dist",
            )
            .squeeze(0)
            .sort()[0]
        )

        s_dist = np.array(s_dist)
        idx = min(self.k, len(s_dist)) - 1
        return math.log(s_dist[idx] + 1)
