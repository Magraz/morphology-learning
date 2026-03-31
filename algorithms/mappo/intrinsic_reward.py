import math
from collections import deque

import torch


class IntrinsicReward:
    """Episodic curiosity-based intrinsic reward using k-NN distances."""

    def __init__(
        self,
        *,
        obs_dim: int,
        k: int = 4,
        memory_capacity: int = 10_000,
    ):
        self.obs_dim = obs_dim
        self.k = k
        self._memory: deque[torch.Tensor] = deque(maxlen=memory_capacity)

    def reset(self) -> None:
        """Clear episodic memory."""
        self._memory.clear()

    def on_rollout_step(self, obs) -> None:
        """Store one encoded observation vector for the current episode.

        Args:
            obs: shape (obs_dim,)
        """
        self._memory.append(self._as_tensor(obs))

    def compute_reward(self, obs) -> float:
        """Compute intrinsic reward for one encoded observation vector.

        Args:
            obs: shape (obs_dim,)

        Returns:
            reward: scalar intrinsic reward
        """
        return self._compute_intrinsic_reward(self._as_tensor(obs))

    def _as_tensor(self, obs) -> torch.Tensor:
        tensor = torch.as_tensor(obs, dtype=torch.float32).reshape(-1).cpu()
        if tensor.shape[0] != self.obs_dim:
            raise ValueError(
                f"Expected encoded observation of shape ({self.obs_dim},), "
                f"got {tuple(tensor.shape)}."
            )
        return tensor

    def _compute_intrinsic_reward(self, current_state: torch.Tensor) -> float:
        if len(self._memory) == 0:
            return 1.0

        memory_tensor = torch.stack(list(self._memory), dim=0)

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

        idx = min(self.k, s_dist.numel()) - 1
        return math.log(float(s_dist[idx]) + 1.0)
