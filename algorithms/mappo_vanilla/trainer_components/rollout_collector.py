import time
from dataclasses import dataclass
from algorithms.mappo_vanilla.mappo import MAPPOAgent
import numpy as np
import torch


@dataclass
class RolloutResult:
    step_count: int
    episode_count: int
    final_values: list[float]
    # Wall-clock seconds spent inside vec_env reset/step (the actual environment
    # physics + IPC), as opposed to the surrounding main-process work (policy
    # inference, transition storage).
    env_time: float = 0.0
    # Mean per-step environment reward over the rollout.
    mean_extrinsic_reward: float = 0.0
    # Normalized frequency of each discrete action over the rollout (length
    # n_actions, sums to 1). None for continuous-action envs.
    action_distribution: list | None = None


class RolloutCollector:
    def __init__(
        self,
        *,
        vec_env,
        agent: MAPPOAgent,
        device: str,
        n_agents: int,
        n_parallel_envs: int,
        discrete: bool,
    ):
        self.vec_env = vec_env
        self.agent = agent
        self.device = device
        self.n_agents = n_agents
        self.n_parallel_envs = n_parallel_envs
        self.discrete = discrete

        # Number of discrete actions, for the per-rollout action/skill histogram.
        # Read from the env's action space (MultiDiscrete -> nvec[0], else n).
        self.n_actions = None
        if self.discrete:
            act_space = self.vec_env.single_action_space
            nvec = getattr(act_space, "nvec", None)
            if nvec is not None:
                self.n_actions = int(np.asarray(nvec).reshape(-1)[0])
            elif hasattr(act_space, "n"):
                self.n_actions = int(act_space.n)

    def collect(self, max_steps: int) -> RolloutResult:
        """Collect trajectory using Gymnasium vectorized environments."""
        env_time = 0.0  # accumulated wall-clock inside vec_env reset/step

        train_seeds = [
            int(np.random.randint(0, 2**31)) for _ in range(self.n_parallel_envs)
        ]
        _env_t0 = time.perf_counter()
        obs, infos = self.vec_env.reset(seed=train_seeds)
        env_time += time.perf_counter() - _env_t0
        batch_size = obs.shape[0]

        current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
        total_step_count = 0
        episode_count = 0
        current_episode_steps = np.zeros(self.n_parallel_envs, dtype=np.int32)

        # Accumulate per-step reward means for the training log line.
        extrinsic_reward_sum = 0.0
        reward_step_count = 0

        # Per-rollout histogram of selected discrete actions, summed over all
        # envs/agents/steps.
        action_counts = (
            np.zeros(self.n_actions, dtype=np.int64) if self.n_actions else None
        )

        # `<`, not `<=`: the loop can only move in whole rows of `n_envs` steps
        # (the vector env steps every env together), so `<=` took one extra full
        # row even when `max_steps` was hit exactly — making the collected batch
        # a function of n_envs. With `<` the batch is exactly `max_steps`
        # whenever n_envs divides it, and overshoots by at most n_envs-1 when it
        # does not. Cut only at row boundaries: GAE stacks per-env trajectories
        # and requires a uniform length across envs.
        while total_step_count < max_steps:
            global_states = obs.reshape(batch_size, -1)

            actions_t, log_probs_t, values_t = self.agent.get_actions_batched(
                obs,
                global_states,
                deterministic=False,
                action_masks=current_masks,
            )

            actions_array = actions_t.cpu().numpy()
            log_probs_array = log_probs_t.cpu().numpy()
            values_array = values_t.cpu().numpy()

            if self.discrete:
                if actions_array.ndim == 3 and actions_array.shape[-1] == 1:
                    actions_array = actions_array.squeeze(-1)
                actions_array = actions_array.astype(np.int32)
                if action_counts is not None:
                    action_counts += np.bincount(
                        actions_array.ravel(), minlength=self.n_actions
                    )[: self.n_actions]

            _env_t0 = time.perf_counter()
            next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(
                actions_array
            )
            env_time += time.perf_counter() - _env_t0
            dones = np.logical_or(terminateds, truncateds)

            extrinsic_reward_sum += float(np.asarray(rewards).mean())
            reward_step_count += 1

            # Truncation bootstrap (mirror of mappo_jax). A time-limit
            # `truncated` (vs a true `terminated`) does not end the MDP, so its
            # return must carry `gamma * V(s_next)` forward rather than be cut to
            # 0 by GAE's `done` mask. Under gymnasium's NEXT_STEP autoreset,
            # `next_obs` at a truncated step is the *real* terminal successor
            # (the reset obs only appears on the following step), so we value it
            # directly and fold the bootstrap into the stored reward. `dones`
            # still fires on truncation, so GAE cuts the recursion (no bleed
            # across the boundary) and its own bootstrap term is 0 here — no
            # double count. Logged extrinsic reward (above) stays the raw env
            # reward.
            if np.any(truncateds):
                next_values = self._state_values(next_obs, batch_size)
                rewards = np.asarray(rewards, dtype=np.float32) + (
                    self.agent.gamma
                    * truncateds.astype(np.float32)
                    * next_values
                )

            self.agent.store_transitions_batch(
                obs,
                global_states,
                actions_array,
                log_probs_array,
                values_array,
                rewards,
                dones,
                action_masks=current_masks,
            )

            current_masks = (
                infos.get("avail_actions") if isinstance(infos, dict) else None
            )

            obs = next_obs
            total_step_count += batch_size
            current_episode_steps += 1

            episode_count += int(dones.sum())
            current_episode_steps[dones] = 0

        final_values = self._compute_final_values(obs, infos, batch_size)

        action_distribution = None
        if action_counts is not None:
            total = int(action_counts.sum())
            if total > 0:
                action_distribution = (action_counts / total).tolist()

        denom = max(reward_step_count, 1)
        return RolloutResult(
            step_count=total_step_count,
            episode_count=episode_count,
            final_values=final_values,
            env_time=env_time,
            mean_extrinsic_reward=extrinsic_reward_sum / denom,
            action_distribution=action_distribution,
        )

    def _state_values(self, obs, batch_size: int) -> np.ndarray:
        """Critic values ``V(global_state(obs))`` as a ``(batch_size,)`` numpy
        array, using the same (old) network that produced the stored rollout
        values — so bootstraps stay consistent with the collected `values`."""
        global_states = obs.reshape(batch_size, -1)
        with torch.no_grad():
            gs_tensor = torch.from_numpy(
                np.ascontiguousarray(global_states, dtype=np.float32)
            ).to(self.device)
            return (
                self.agent.network_old.get_value(gs_tensor)
                .cpu()
                .squeeze(-1)
                .numpy()
                .astype(np.float32)
            )

    def _compute_final_values(self, obs, infos, batch_size: int) -> list[float]:
        return self._state_values(obs, batch_size).tolist()
