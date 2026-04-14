"""Entropy predictor utilities for MAPPOAgent.

Owns rolling observation history for inference, sequence-building utilities
for predictor training, actor-observation conditioning, and predictor loss
computation.
"""

import torch
import torch.nn.functional as F


def update_left_padded_history(
    history: torch.Tensor,
    new_values: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one step to a rolling history with repeat-left padding.

    Args:
        history: (batch, seq_len, ...) tensor storing previous values.
        new_values: (batch, ...) tensor for the current timestep.
        counts: (batch,) tensor with the number of valid steps seen so far.

    Returns:
        Updated ``history`` and ``counts``. If a sequence has fewer than
        ``seq_len`` valid elements, the earliest available timestep is repeated
        on the left to keep the full window dense.
    """
    history = torch.roll(history, shifts=-1, dims=1)
    history[:, -1] = new_values
    counts = torch.clamp(counts + 1, max=history.shape[1])

    for batch_idx in range(history.shape[0]):
        pad_len = history.shape[1] - int(counts[batch_idx].item())
        if pad_len > 0:
            history[batch_idx, :pad_len] = history[batch_idx, pad_len]

    return history, counts


class EntropyPredictorHelper:
    """Agent-side helper for entropy predictor inference and training."""

    def __init__(
        self,
        *,
        n_parallel_envs: int,
        n_agents: int,
        observation_dim: int,
        seq_len: int,
    ):
        self.n_parallel_envs = n_parallel_envs
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.seq_len = seq_len

        # Rolling observation history for entropy predictor during collection.
        # Persists across reset_buffers calls (running inference state).
        self._obs_history = torch.zeros(
            n_parallel_envs, n_agents, seq_len, observation_dim
        )

    # ------------------------------------------------------------------
    # Rolling observation history (collection-time)
    # ------------------------------------------------------------------

    def update_obs_history(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Shift obs history left and append new obs.

        Args:
            obs_tensor: (n_envs, n_agents, obs_dim)
        Returns:
            (n_envs, n_agents, seq_len, obs_dim) current sequences.
        """
        self._obs_history = torch.roll(self._obs_history, -1, dims=2)
        self._obs_history[:, :, -1, :] = obs_tensor
        return self._obs_history.clone()

    def reset_obs_history(self, env_mask=None):
        """Zero out obs history. If env_mask given, only reset those envs."""
        if env_mask is None:
            self._obs_history.zero_()
        else:
            self._obs_history[env_mask] = 0.0

    def make_eval_sequences(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Build zero-padded single-step sequences for eval/render.

        Args:
            obs_tensor: (n_envs, n_agents, obs_dim)
        Returns:
            (n_envs, n_agents, seq_len, obs_dim) with obs at last position.
        """
        n_envs = obs_tensor.shape[0]
        seqs = torch.zeros(
            n_envs,
            self.n_agents,
            self.seq_len,
            self.observation_dim,
            device=obs_tensor.device,
        )
        seqs[:, :, -1, :] = obs_tensor
        return seqs

    # ------------------------------------------------------------------
    # Sequence building (training-time)
    # ------------------------------------------------------------------

    def build_obs_sequences(self, obs_stacked: torch.Tensor) -> torch.Tensor:
        """Build zero-padded sliding-window observation sequences.

        Args:
            obs_stacked: (T, obs_dim) tensor of observations for one
                         (env, agent) trajectory.
        Returns:
            sequences: (T, seq_len, obs_dim) tensor.
        """
        _, obs_dim = obs_stacked.shape

        padding = torch.zeros(self.seq_len - 1, obs_dim, dtype=obs_stacked.dtype)
        padded = torch.cat([padding, obs_stacked], dim=0)

        sequences = padded.unfold(0, self.seq_len, 1)  # (T, obs_dim, seq_len)
        sequences = sequences.permute(0, 2, 1)  # (T, seq_len, obs_dim)

        return sequences

    def build_obs_sequences_batched(self, obs_batched: torch.Tensor) -> torch.Tensor:
        """Build sliding-window sequences for a batch of trajectories at once.

        Args:
            obs_batched: (B, T, obs_dim) — B trajectories stacked.
        Returns:
            (B * T, seq_len, obs_dim) — ready for dataset construction.
        """
        B, T, obs_dim = obs_batched.shape

        padding = torch.zeros(B, self.seq_len - 1, obs_dim, dtype=obs_batched.dtype)
        padded = torch.cat([padding, obs_batched], dim=1)

        sequences = padded.unfold(1, self.seq_len, 1)  # (B, T, obs_dim, seq_len)
        sequences = sequences.permute(0, 1, 3, 2)  # (B, T, seq_len, obs_dim)
        return sequences.reshape(B * T, self.seq_len, obs_dim)

    # ------------------------------------------------------------------
    # Inference conditioning
    # ------------------------------------------------------------------

    @staticmethod
    def condition_obs(obs: torch.Tensor, pred_entropy: torch.Tensor) -> torch.Tensor:
        """Concatenate predicted entropy to observations for actor conditioning.

        Args:
            obs: (..., obs_dim)
            pred_entropy: (..., 2*n_types) concatenated [mean, log_var]
        Returns:
            (..., obs_dim + 2*n_types)
        """
        return torch.cat([obs, pred_entropy], dim=-1)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    @staticmethod
    def compute_prediction_loss(
        pred_mean: torch.Tensor,
        pred_log_var: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian NLL loss for entropy prediction.

        Args:
            pred_mean: (B, n_types) predicted entropy means.
            pred_log_var: (B, n_types) predicted log-variance.
            targets: (B, n_types) ground-truth entropy values.
        Returns:
            Scalar loss tensor.
        """
        return F.gaussian_nll_loss(pred_mean, targets, pred_log_var.exp())
