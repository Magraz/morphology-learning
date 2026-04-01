from dataclasses import dataclass
from algorithms.mappo.intrinsic_reward import IntrinsicReward
from algorithms.mappo.trainer_components.hypergraph_runtime import HypergraphRuntime
import numpy as np
import torch


@dataclass
class RolloutResult:
    step_count: int
    episode_count: int
    final_values: list[float]


class RolloutCollector:
    def __init__(
        self,
        *,
        vec_env,
        agent,
        device: str,
        n_agents: int,
        n_parallel_envs: int,
        discrete: bool,
        entropy_conditioning: bool,
        hypergraph_runtime: HypergraphRuntime,
    ):
        self.vec_env = vec_env
        self.agent = agent
        self.device = device
        self.n_agents = n_agents
        self.n_parallel_envs = n_parallel_envs
        self.discrete = discrete
        self.entropy_conditioning = entropy_conditioning
        self.hypergraph_runtime = hypergraph_runtime
        self.intrinsic_rewarders = None
        self.agent_intrinsic_rewarders = None
        self.intrinsic_reward_mode = (
            self.agent.intrinsic_reward_mode
            if self.agent.use_intrinsic_reward
            else None
        )
        self.intrinsic_reward_use_encoder = (
            self.agent.intrinsic_reward_use_encoder
            if self.agent.use_intrinsic_reward
            else False
        )
        self.intrinsic_reward_encoder_type = (
            self.agent.intrinsic_reward_encoder_type
            if self.agent.use_intrinsic_reward
            else "local"
        )
        if self.agent.use_intrinsic_reward:
            obs_dim_per_agent = self.agent.observation_dim
            if self.intrinsic_reward_use_encoder:
                team_obs_dim = self.agent.intrinsic_reward_obs_dim
                agent_obs_dim = self.agent.intrinsic_reward_encoder_dim
            else:
                team_obs_dim = self.n_agents * obs_dim_per_agent
                agent_obs_dim = obs_dim_per_agent

            if self.intrinsic_reward_mode == "agent":
                self.agent_intrinsic_rewarders = [
                    [
                        IntrinsicReward(
                            obs_dim=agent_obs_dim,
                            k=self.agent.intrinsic_reward_k,
                            memory_capacity=self.agent.intrinsic_reward_memory_capacity,
                        )
                        for _ in range(self.n_agents)
                    ]
                    for _ in range(self.n_parallel_envs)
                ]
            else:
                self.intrinsic_rewarders = [
                    IntrinsicReward(
                        obs_dim=team_obs_dim,
                        k=self.agent.intrinsic_reward_k,
                        memory_capacity=self.agent.intrinsic_reward_memory_capacity,
                    )
                    for _ in range(self.n_parallel_envs)
                ]

    def collect(self, max_steps: int) -> RolloutResult:
        """Collect trajectory using Gymnasium vectorized environments."""
        train_seeds = [
            int(np.random.randint(0, 2**31)) for _ in range(self.n_parallel_envs)
        ]
        obs, infos = self.vec_env.reset(seed=train_seeds)
        batch_size = obs.shape[0]

        self.hypergraph_runtime.on_rollout_reset()
        if self.intrinsic_rewarders is not None:
            for rewarder in self.intrinsic_rewarders:
                rewarder.reset()
        if self.agent_intrinsic_rewarders is not None:
            for env_rewarders in self.agent_intrinsic_rewarders:
                for rewarder in env_rewarders:
                    rewarder.reset()

        current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
        total_step_count = 0
        episode_count = 0
        current_episode_steps = np.zeros(self.n_parallel_envs, dtype=np.int32)

        self.agent.reset_obs_history()

        while total_step_count <= max_steps:
            global_states = obs.reshape(batch_size, -1)
            self.hypergraph_runtime.on_rollout_step(obs)

            per_env_hgs, per_env_sig_ids = (
                self.hypergraph_runtime.build_inference_hypergraphs(
                    obs, infos, batch_size
                )
            )

            per_env_entropies = (
                self.hypergraph_runtime.compute_entropies_for_critic(per_env_sig_ids)
                if self.entropy_conditioning
                else None
            )

            actions_t, log_probs_t, values_t = self.agent.get_actions_batched(
                obs,
                global_states,
                deterministic=False,
                action_masks=current_masks,
                hypergraphs=per_env_hgs,
                entropies=per_env_entropies,
            )

            actions_array = actions_t.cpu().numpy()
            log_probs_array = log_probs_t.cpu().numpy()
            values_array = values_t.cpu().numpy()

            if self.discrete:
                if actions_array.ndim == 3 and actions_array.shape[-1] == 1:
                    actions_array = actions_array.squeeze(-1)
                actions_array = actions_array.astype(np.int32)

            next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(
                actions_array
            )
            dones = np.logical_or(terminateds, truncateds)

            per_agent_intrinsic = None
            if self.intrinsic_rewarders is not None:
                encoder_hgs = None
                if (
                    self.intrinsic_reward_use_encoder
                    and self.intrinsic_reward_encoder_type == "hypergraph"
                ):
                    encoder_hgs = self.hypergraph_runtime.build_per_env_hypergraphs(
                        next_obs, infos, batch_size
                    )
                per_agent_intrinsic = self._get_team_intrinsic_rewards(
                    next_obs=next_obs,
                    dones=dones,
                    hypergraphs=encoder_hgs,
                )
            elif self.agent_intrinsic_rewarders is not None:
                per_agent_intrinsic = self._get_agent_intrinsic_rewards(
                    next_obs=next_obs,
                    dones=dones,
                )

            if dones.any():
                self.agent.reset_obs_history(env_mask=dones)
                self.hypergraph_runtime.on_env_done_mask(dones)

            self.agent.store_transitions_batch(
                obs,
                global_states,
                actions_array,
                log_probs_array,
                values_array,
                rewards,
                dones,
                action_masks=current_masks,
                hg_signature_ids=per_env_sig_ids,
                entropies=per_env_entropies,
                per_agent_intrinsic_rewards=per_agent_intrinsic,
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

        return RolloutResult(
            step_count=total_step_count,
            episode_count=episode_count,
            final_values=final_values,
        )

    def _get_team_intrinsic_rewards(
        self, next_obs, dones, hypergraphs=None
    ) -> np.ndarray:
        if self.intrinsic_reward_use_encoder:
            team_features = self.agent.encode_team_observations(
                np.ascontiguousarray(next_obs, dtype=np.float32),
                hypergraphs=hypergraphs,
            )
        else:
            team_features = next_obs.reshape(self.n_parallel_envs, -1).astype(
                np.float32
            )

        intrinsic_rewards = np.zeros(self.n_parallel_envs, dtype=np.float32)
        for env_idx, rewarder in enumerate(self.intrinsic_rewarders):
            if dones[env_idx]:
                rewarder.reset()
            else:
                intrinsic_rewards[env_idx] = rewarder.compute_reward(
                    team_features[env_idx]
                )
                rewarder.on_rollout_step(team_features[env_idx])

        rewards = self.agent.intrinsic_reward_coef * intrinsic_rewards
        return np.tile(rewards[:, None], (1, self.n_agents))

    def _get_agent_intrinsic_rewards(self, next_obs, dones) -> np.ndarray:
        """Compute per-agent intrinsic rewards.

        Returns:
            np.ndarray of shape (n_envs, n_agents) — scaled intrinsic rewards.
        """
        if self.intrinsic_reward_use_encoder:
            agent_features = self.agent.encode_agent_observations(
                np.ascontiguousarray(next_obs, dtype=np.float32)
            )  # (n_envs, n_agents, encoder_dim)
        else:
            agent_features = next_obs.astype(np.float32)  # (n_envs, n_agents, obs_dim)

        intrinsic_rewards = np.zeros(
            (self.n_parallel_envs, self.n_agents), dtype=np.float32
        )
        for env_idx, env_rewarders in enumerate(self.agent_intrinsic_rewarders):
            if dones[env_idx]:
                for rewarder in env_rewarders:
                    rewarder.reset()
            else:
                for agent_idx, rewarder in enumerate(env_rewarders):
                    intrinsic_rewards[env_idx, agent_idx] = rewarder.compute_reward(
                        agent_features[env_idx, agent_idx]
                    )
                    rewarder.on_rollout_step(agent_features[env_idx, agent_idx])

        return self.agent.intrinsic_reward_coef * intrinsic_rewards

    def _compute_final_values(self, obs, infos, batch_size: int) -> list[float]:
        final_global_states = obs.reshape(batch_size, -1)
        with torch.no_grad():
            final_gs_tensor = torch.from_numpy(
                np.ascontiguousarray(final_global_states, dtype=np.float32)
            ).to(self.device)

            if self.agent.critic_type == "multi_hgnn":
                final_batched_hgs, final_sig_ids = (
                    self.hypergraph_runtime.build_inference_hypergraphs(
                        obs, infos, batch_size
                    )
                )
                final_entropies = (
                    self.hypergraph_runtime.compute_entropies_for_critic(final_sig_ids)
                    if self.entropy_conditioning
                    else None
                )
                obs_tensor = torch.from_numpy(
                    np.ascontiguousarray(obs, dtype=np.float32)
                ).to(self.device)
                obs_flat = obs_tensor.reshape(batch_size * self.n_agents, -1)
                final_values = (
                    self.agent.network_old.get_value_batched(
                        obs_flat,
                        final_batched_hgs,
                        batch_size,
                        entropies=final_entropies,
                    )
                    .cpu()
                    .squeeze(-1)
                    .tolist()
                )
            else:
                final_values = (
                    self.agent.network_old.get_value(final_gs_tensor)
                    .cpu()
                    .squeeze(-1)
                    .tolist()
                )

        return final_values
