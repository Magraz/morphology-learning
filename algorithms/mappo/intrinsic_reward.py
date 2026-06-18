import numpy as np
import torch


class BatchedIntrinsicReward:
    """Episodic k-NN novelty bonus computed for many independent streams at once.

    A "stream" is one rewarder identity: one per env in ``team`` mode, one per
    (env, agent) in ``agent`` mode. Every stream keeps its own episodic memory in
    a shared ring buffer of shape ``(n_streams, capacity, feat_dim)``. Because the
    buffer is preallocated and written in place, there is no per-step restack of a
    growing deque (the old per-stream class rebuilt its full memory tensor every
    step, making the per-episode cost quadratic in episode length). All streams
    are scored in a single batched ``cdist``/``sort`` instead of a Python loop.

    Reward for a stream is ``log(d_k + 1)`` where ``d_k`` is the distance to its
    ``min(k, count)``-th nearest stored point, and ``count`` is the number of
    points already in memory (i.e. excluding the current one). A stream with empty
    memory scores 0. On ``done`` a stream's memory is cleared and the current
    point is not stored — matching the original per-env semantics exactly.
    """

    def __init__(
        self,
        *,
        n_streams: int,
        feat_dim: int,
        k: int = 4,
        memory_capacity: int = 5000,
        device: str = "cpu",
    ):
        self.n_streams = n_streams
        self.feat_dim = feat_dim
        self.k = k
        self.capacity = memory_capacity
        self.device = torch.device(device)

        self._buffer = torch.zeros(
            n_streams, self.capacity, feat_dim, device=self.device
        )
        # Number of valid entries per stream (capped at capacity) and the next
        # write index (ring pointer). While an episode is shorter than capacity
        # these are equal and the valid entries occupy [0:count] contiguously.
        self._count = torch.zeros(n_streams, dtype=torch.long, device=self.device)
        self._ptr = torch.zeros(n_streams, dtype=torch.long, device=self.device)

    def reset(self, mask=None) -> None:
        """Clear episodic memory for all streams, or a boolean-masked subset."""
        if mask is None:
            self._count.zero_()
            self._ptr.zero_()
            return
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        self._count[m] = 0
        self._ptr[m] = 0

    def compute_and_store(self, feats, dones) -> np.ndarray:
        """Score the current point of every stream against its memory, then fold
        the point into memory (clearing streams that are done).

        Args:
            feats: (n_streams, feat_dim) — current descriptor per stream.
            dones: (n_streams,) bool — streams whose episode just ended.

        Returns:
            (n_streams,) float32 novelty rewards (0 for empty-memory / done
            streams), un-scaled by any coefficient.
        """
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=self.device)
        feats_t = feats_t.reshape(self.n_streams, self.feat_dim)
        not_done = ~torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        reward = self._knn_reward(feats_t)  # scored against pre-store memory
        reward = reward * not_done.float()  # done streams contribute nothing

        # Clear memory of done streams, then store the current point for the rest.
        self.reset(mask=~not_done)
        self._store(feats_t, not_done)

        return reward.cpu().numpy()

    def _knn_reward(self, feats_t: torch.Tensor) -> torch.Tensor:
        max_count = int(self._count.max().item())
        if max_count == 0:
            return torch.zeros(self.n_streams, device=self.device)

        # Only the filled prefix can hold valid entries (contiguous until the ring
        # wraps, at which point max_count == capacity and the whole buffer is valid).
        buf = self._buffer[:, :max_count]  # (n, m, feat_dim)
        arange = torch.arange(max_count, device=self.device)
        valid = arange[None, :] < self._count[:, None]  # (n, m)

        dist = torch.cdist(
            feats_t.unsqueeze(1),
            buf,
            p=2,
            compute_mode="use_mm_for_euclid_dist",
        ).squeeze(
            1
        )  # (n, m)
        dist = dist.masked_fill(~valid, float("inf"))
        dist_sorted, _ = torch.sort(dist, dim=1)  # ascending; inf padding last

        # k-th nearest, clamped to available memory (1-indexed -> gather index k-1).
        k_eff = torch.clamp(self._count, max=self.k)  # (n,)
        gather_idx = torch.clamp(k_eff - 1, min=0)
        kth = dist_sorted.gather(1, gather_idx[:, None]).squeeze(1)  # (n,)

        reward = torch.log(kth + 1.0)
        return torch.where(self._count > 0, reward, torch.zeros_like(reward))

    def _store(self, feats_t: torch.Tensor, not_done: torch.Tensor) -> None:
        rows = not_done.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return
        write_idx = self._ptr[rows]
        self._buffer[rows, write_idx] = feats_t[rows]
        self._ptr[rows] = (write_idx + 1) % self.capacity
        self._count[rows] = torch.clamp(self._count[rows] + 1, max=self.capacity)
