import time
from dataclasses import dataclass
from algorithms.mappo.intrinsic_reward import BatchedIntrinsicReward
from algorithms.mappo.entropy_helpers import update_left_padded_history
from algorithms.mappo.trainer_components.hypergraph_runtime import HypergraphRuntime
from algorithms.mappo.mappo import MAPPOAgent
import numpy as np
import torch


@dataclass
class RolloutResult:
    step_count: int
    episode_count: int
    final_values: list[float]
    # Wall-clock seconds spent inside vec_env reset/step (the actual environment
    # physics + IPC), as opposed to the surrounding main-process work (policy
    # inference, hypergraph build, intrinsic rewards, transition storage).
    env_time: float = 0.0
    # Per-step means over the rollout, for diagnosing intrinsic-vs-extrinsic
    # balance per trial. ``mean_intrinsic_reward`` is the coef-scaled novelty
    # bonus exactly as it enters the agents' reward (0 when intrinsic is off);
    # ``mean_extrinsic_reward`` is the raw environment reward over the same steps.
    mean_intrinsic_reward: float = 0.0
    mean_extrinsic_reward: float = 0.0
    # Normalized frequency of each discrete action over the rollout (length
    # n_actions, sums to 1). For the hierarchical controller these are the
    # skill-selection fractions. None for continuous-action envs.
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

        # Coordination-graph novelty exploration. A single batched episodic k-NN
        # rewarder scores all streams at once: one stream per env ("team") or per
        # (env, agent) ("agent"). Memory holds coordination descriptors extracted
        # from the dual-purpose attention encoder.
        self.intrinsic_rewarder = None
        if self.agent.use_intrinsic_reward:
            feat_dim = self.agent.intrinsic_feature_dim
            if self.agent.intrinsic_reward_mode == "agent":
                n_streams = self.n_parallel_envs * self.n_agents
            else:
                n_streams = self.n_parallel_envs
            self.intrinsic_rewarder = BatchedIntrinsicReward(
                n_streams=n_streams,
                feat_dim=feat_dim,
                k=self.agent.intrinsic_reward_k,
                memory_capacity=self.agent.intrinsic_reward_memory_capacity,
            )

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

        self.hypergraph_runtime.on_rollout_reset()
        if self.intrinsic_rewarder is not None:
            self.intrinsic_rewarder.reset()

        current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
        total_step_count = 0
        episode_count = 0
        current_episode_steps = np.zeros(self.n_parallel_envs, dtype=np.int32)

        # Accumulate per-step reward means to surface the intrinsic-vs-extrinsic
        # balance (see RolloutResult). Both are summed per env-step then divided
        # by the number of env-steps at the end.
        intrinsic_reward_sum = 0.0
        extrinsic_reward_sum = 0.0
        reward_step_count = 0

        # Per-rollout histogram of selected discrete actions (= skills for the
        # hierarchical controller), summed over all envs/agents/steps.
        action_counts = (
            np.zeros(self.n_actions, dtype=np.int64)
            if self.n_actions
            else None
        )

        self.agent.reset_obs_history()
        critic_obs_history = None
        critic_sig_history = None
        critic_history_counts = None
        if self.agent.critic_type == "hg_cross_attention":
            critic_obs_history = torch.zeros(
                batch_size,
                self.agent.critic_seq_len,
                self.n_agents,
                self.agent.observation_dim,
                dtype=torch.float32,
            )
            critic_sig_history = torch.zeros(
                batch_size, self.agent.critic_seq_len, dtype=torch.long
            )
            critic_history_counts = torch.zeros(batch_size, dtype=torch.long)

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

            critic_obs_sequences = None
            critic_signature_sequences = None
            if self.agent.critic_type == "hg_cross_attention":
                obs_step = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
                sig_step = torch.tensor(per_env_sig_ids, dtype=torch.long)
                prev_counts = critic_history_counts.clone()
                critic_obs_history, critic_history_counts = update_left_padded_history(
                    critic_obs_history, obs_step, critic_history_counts
                )
                critic_sig_history, _ = update_left_padded_history(
                    critic_sig_history, sig_step, prev_counts
                )
                critic_obs_sequences = critic_obs_history.clone()
                critic_signature_sequences = critic_sig_history.clone()

            actions_t, log_probs_t, values_t = self.agent.get_actions_batched(
                obs,
                global_states,
                deterministic=False,
                action_masks=current_masks,
                hypergraphs=per_env_hgs,
                entropies=per_env_entropies,
                critic_obs_sequences=critic_obs_sequences,
                critic_signature_sequences=critic_signature_sequences,
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

            per_agent_intrinsic = None
            if self.intrinsic_rewarder is not None:
                if self.agent.intrinsic_reward_mode == "agent":
                    per_agent_intrinsic = self._get_agent_intrinsic_rewards(
                        next_obs, dones
                    )
                else:
                    per_agent_intrinsic = self._get_team_intrinsic_rewards(
                        next_obs, dones
                    )
                intrinsic_reward_sum += float(per_agent_intrinsic.mean())

            extrinsic_reward_sum += float(np.asarray(rewards).mean())
            reward_step_count += 1

            if dones.any():
                self.agent.reset_obs_history(env_mask=dones)
                self.hypergraph_runtime.on_env_done_mask(dones)
                if critic_obs_history is not None:
                    critic_obs_history[dones] = 0.0
                    critic_sig_history[dones] = 0
                    critic_history_counts[dones] = 0

            grouping_tokens = (
                self.hypergraph_runtime.get_last_grouping_tokens()
                if self.agent.hypergraph_mode == "learned_grouping"
                else None
            )

            # Truncation bootstrap (mirror of mappo_jax / mappo_vanilla). A
            # time-limit `truncated` (vs a true `terminated`) does not end the
            # MDP, so its return must carry `gamma * V(s_next)` forward rather
            # than be cut to 0 by GAE's `done` mask. Under gymnasium's NEXT_STEP
            # autoreset, `next_obs` at a truncated step is the real terminal
            # successor, so value it directly and fold the bootstrap into the
            # stored reward; `dones` still fires so GAE cuts the recursion and
            # its own bootstrap term is 0 here (no double count). Placed *after*
            # the grouping-token read so the hypergraph rebuild for `next_obs`
            # can't clobber this step's `_last_grouping_tokens`. Logged extrinsic
            # reward (above) stays the raw env reward.
            if np.any(truncateds):
                next_values = self._state_values(next_obs, infos, batch_size)
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
                hg_signature_ids=per_env_sig_ids,
                entropies=per_env_entropies,
                grouping_tokens=grouping_tokens,
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
            mean_intrinsic_reward=intrinsic_reward_sum / denom,
            mean_extrinsic_reward=extrinsic_reward_sum / denom,
            action_distribution=action_distribution,
        )

    def _get_team_intrinsic_rewards(self, next_obs, dones) -> np.ndarray:
        """One coordination-graph novelty bonus per env (whole-graph descriptor),
        tiled across all agents. Returns coef-scaled (n_envs, n_agents)."""
        team_features = self.agent.compute_coordination_features(
            np.ascontiguousarray(next_obs, dtype=np.float32)
        )  # (n_envs, team_dim)

        intrinsic = self.intrinsic_rewarder.compute_and_store(
            team_features, dones
        )  # (n_envs,)
        rewards = self.agent.intrinsic_reward_coef * intrinsic
        return np.tile(rewards[:, None], (1, self.n_agents))

    def _get_agent_intrinsic_rewards(self, next_obs, dones) -> np.ndarray:
        """Per-agent coordination-graph novelty bonus from each agent's own
        coordination row. Returns coef-scaled (n_envs, n_agents)."""
        agent_features = self.agent.compute_coordination_features(
            np.ascontiguousarray(next_obs, dtype=np.float32)
        )  # (n_envs, n_agents, agent_dim)
        n_envs, n_agents, feat_dim = agent_features.shape

        # Flatten (env, agent) into independent streams; replicate each env's done
        # flag across its agents so all of an env's streams reset together.
        flat_features = agent_features.reshape(n_envs * n_agents, feat_dim)
        flat_dones = np.repeat(np.asarray(dones), n_agents)

        intrinsic = self.intrinsic_rewarder.compute_and_store(
            flat_features, flat_dones
        ).reshape(n_envs, n_agents)
        return self.agent.intrinsic_reward_coef * intrinsic

    def _state_values(self, obs, infos, batch_size: int) -> np.ndarray:
        """Critic values ``V(global_state(obs))`` as a ``(batch_size,)`` numpy
        array, using the (old) network that produced the stored rollout values.
        Handles the hypergraph critics (which need inference hypergraphs built
        from obs/infos) the same way as the end-of-rollout bootstrap. Used for
        both the final-value bootstrap and the per-step truncation bootstrap.

        NOTE for `learned_grouping`: building inference hypergraphs mutates the
        runtime's `_last_grouping_tokens`, so callers inside the rollout loop
        must invoke this only *after* reading `get_last_grouping_tokens()` for
        the current step (the next iteration recomputes them anyway)."""
        global_states = obs.reshape(batch_size, -1)
        with torch.no_grad():
            gs_tensor = torch.from_numpy(
                np.ascontiguousarray(global_states, dtype=np.float32)
            ).to(self.device)

            if self.agent.critic_type in ("multi_hgnn", "hg_cross_attention"):
                batched_hgs, sig_ids = (
                    self.hypergraph_runtime.build_inference_hypergraphs(
                        obs, infos, batch_size
                    )
                )
                entropies = (
                    self.hypergraph_runtime.compute_entropies_for_critic(sig_ids)
                    if self.entropy_conditioning
                    else None
                )
                obs_tensor = torch.from_numpy(
                    np.ascontiguousarray(obs, dtype=np.float32)
                ).to(self.device)
                obs_flat = obs_tensor.reshape(batch_size * self.n_agents, -1)
                values = self.agent.network_old.get_value_batched(
                    obs_flat, batched_hgs, batch_size, entropies=entropies
                )
            else:
                values = self.agent.network_old.get_value(gs_tensor)

            return values.cpu().squeeze(-1).numpy().astype(np.float32)

    def _compute_final_values(self, obs, infos, batch_size: int) -> list[float]:
        return self._state_values(obs, infos, batch_size).tolist()
