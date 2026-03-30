from dataclasses import dataclass

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
        hypergraph_runtime,
    ):
        self.vec_env = vec_env
        self.agent = agent
        self.device = device
        self.n_agents = n_agents
        self.n_parallel_envs = n_parallel_envs
        self.discrete = discrete
        self.entropy_conditioning = entropy_conditioning
        self.hypergraph_runtime = hypergraph_runtime

    def collect(self, max_steps: int) -> RolloutResult:
        """Collect trajectory using Gymnasium vectorized environments."""
        train_seeds = [int(np.random.randint(0, 2**31)) for _ in range(self.n_parallel_envs)]
        obs, infos = self.vec_env.reset(seed=train_seeds)
        batch_size = obs.shape[0]

        self.hypergraph_runtime.on_rollout_reset()

        current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
        total_step_count = 0
        episode_count = 0
        current_episode_steps = np.zeros(self.n_parallel_envs, dtype=np.int32)

        self.agent.reset_obs_history()

        while total_step_count <= max_steps:
            global_states = obs.reshape(batch_size, -1)
            self.hypergraph_runtime.on_rollout_step(obs)

            per_env_hgs, per_env_sig_ids = self.hypergraph_runtime.build_inference_hypergraphs(
                obs, infos, batch_size
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

            next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(actions_array)
            dones = np.logical_or(terminateds, truncateds)

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
                infos,
                action_masks=current_masks,
                hg_signature_ids=per_env_sig_ids,
                entropies=per_env_entropies,
            )

            current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
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
                obs_tensor = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).to(
                    self.device
                )
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
