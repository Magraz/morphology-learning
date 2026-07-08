import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from algorithms.mappo_vanilla.types import MAPPO_Params, Model_Params
from algorithms.mappo_vanilla.networks.models import MAPPONetwork


class MAPPOAgent:
    """Multi-Agent PPO with centralized critic"""

    def __init__(
        self,
        observation_dim: int,
        global_state_dim: int,
        action_dim: int,
        n_agents: int,
        params: MAPPO_Params,
        device: str,
        discrete: bool,
        n_parallel_envs: int,
        model_params: Model_Params,
    ):
        self.device = "cuda" if torch.cuda.is_available() else device
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.global_state_dim = global_state_dim
        self.discrete = discrete
        self.share_actor = params.parameter_sharing
        self.n_parallel_envs = n_parallel_envs

        # PPO hyperparameters
        self.gamma = params.gamma
        self.gae_lambda = params.lmbda
        self.clip_epsilon = params.eps_clip
        self.entropy_coef = params.ent_coef
        self.value_coef = params.val_coef
        self.grad_clip = params.grad_clip

        # Create network
        network_kwargs = dict(
            observation_dim=observation_dim,
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=model_params.hidden_dim,
            discrete=discrete,
            share_actor=self.share_actor,
        )
        self.network = MAPPONetwork(**network_kwargs).to(self.device)

        # Create old network for PPO
        self.network_old = MAPPONetwork(**network_kwargs).to(self.device)

        self.network_old.load_state_dict(self.network.state_dict())

        # Optimizer for all parameters
        self.optimizer = optim.Adam(self.network.parameters(), lr=params.lr)

        # Buffers for each agent
        self.reset_buffers()

    def reset_buffers(self):
        """Reset experience buffers - separate for each parallel environment"""
        # Buffers indexed by [env_idx][agent_idx]
        self.observations = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.global_states = [[] for _ in range(self.n_parallel_envs)]
        self.actions = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.rewards = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.log_probs = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.values = [[] for _ in range(self.n_parallel_envs)]
        self.dones = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]
        self.action_masks = [
            [[] for _ in range(self.n_agents)] for _ in range(self.n_parallel_envs)
        ]

    def get_actions_batched(
        self,
        observations_batch,
        global_states_batch,
        deterministic=False,
        action_masks=None,
    ):
        """
        Get actions for all agents in all environments in a single batched forward pass.

        Args:
            observations_batch: numpy (n_envs, n_agents, obs_dim)
            global_states_batch: numpy (n_envs, global_state_dim)
            deterministic: Whether to use deterministic actions
            action_masks: numpy (n_envs, n_agents, n_actions) or None

        Returns:
            actions:   tensor (n_envs, n_agents, action_dim) on self.device
            log_probs: tensor (n_envs, n_agents)             on self.device
            values:    tensor (n_envs, 1)                    on self.device
        """
        n_envs = observations_batch.shape[0]

        with torch.no_grad():
            obs_tensor, gs_tensor, masks_tensor = self._to_device_tensors(
                observations_batch, global_states_batch, action_masks
            )
            actions, log_probs = self._run_actors(
                obs_tensor, masks_tensor, n_envs, deterministic
            )
            values = self.network_old.get_value(gs_tensor)

        return actions, log_probs, values

    def _to_device_tensors(self, observations_batch, global_states_batch, action_masks):
        """Convert numpy rollout inputs to device tensors."""
        obs_tensor = torch.from_numpy(
            np.ascontiguousarray(observations_batch, dtype=np.float32)
        ).to(
            self.device
        )  # (n_envs, n_agents, obs_dim)
        gs_tensor = torch.from_numpy(
            np.ascontiguousarray(global_states_batch, dtype=np.float32)
        ).to(
            self.device
        )  # (n_envs, global_state_dim)
        masks_tensor = (
            torch.from_numpy(np.ascontiguousarray(action_masks, dtype=np.float32)).to(
                self.device
            )
            if action_masks is not None
            else None
        )  # (n_envs, n_agents, n_actions) or None
        return obs_tensor, gs_tensor, masks_tensor

    def _run_actors(self, obs_tensor, masks_tensor, n_envs, deterministic):
        """Run actors over all envs/agents; returns (actions, log_probs)."""
        if self.share_actor:
            # Fuse envs and agents into one batch: single forward pass total
            obs_flat = obs_tensor.reshape(n_envs * self.n_agents, -1)

            masks_flat = (
                masks_tensor.reshape(n_envs * self.n_agents, -1)
                if masks_tensor is not None
                else None
            )
            actions_flat, log_probs_flat = self.network_old.act(
                obs_flat,
                agent_idx=0,
                deterministic=deterministic,
                action_mask=masks_flat,
            )
            actions = actions_flat.reshape(n_envs, self.n_agents, -1)
            log_probs = log_probs_flat.reshape(n_envs, self.n_agents, -1).squeeze(-1)
            return actions, log_probs

        # One batched pass per agent (n_agents passes, each over all envs)
        actions_list = []
        log_probs_list = []
        for agent_idx in range(self.n_agents):
            agent_obs = obs_tensor[:, agent_idx, :]  # (n_envs, obs_dim)

            agent_mask = (
                masks_tensor[:, agent_idx, :] if masks_tensor is not None else None
            )
            a, lp = self.network_old.act(
                agent_obs, agent_idx, deterministic, action_mask=agent_mask
            )
            actions_list.append(a)  # (n_envs, action_dim)
            log_probs_list.append(lp)  # (n_envs, 1)
        actions = torch.stack(actions_list, dim=1)  # (n_envs, n_agents, action_dim)
        log_probs = torch.stack(log_probs_list, dim=1).squeeze(-1)  # (n_envs, n_agents)
        return actions, log_probs

    def store_transitions_batch(
        self,
        obs,  # np (n_envs, n_agents, obs_dim)
        global_states,  # np (n_envs, gs_dim)
        actions,  # np (n_envs, n_agents) discrete or (n_envs, n_agents, action_dim)
        log_probs,  # np (n_envs, n_agents)
        values,  # np (n_envs, 1)
        rewards,  # np (n_envs,)
        dones,  # np (n_envs,)
        action_masks=None,  # np (n_envs, n_agents, n_actions) or None
    ):
        """Store transitions for all environments in one vectorized call.

        Reduces tensor-creation overhead from n_envs * n_agents * 7 calls
        down to 7 calls (one per array), then indexes into the resulting tensors.
        """
        n_envs = obs.shape[0]

        # Per-agent rewards: (n_envs, n_agents)
        per_agent_rewards = np.tile(
            rewards.astype(np.float32)[:, None], (1, self.n_agents)
        )

        # Ensure trailing action dim for storage: (n_envs, n_agents, 1) for discrete
        if self.discrete and actions.ndim == 2:
            actions_store = actions[:, :, None].astype(np.float32)
        else:
            actions_store = actions.astype(np.float32)

        # Expand dones: (n_envs,) -> (n_envs, n_agents)
        dones_expanded = np.tile(dones.astype(np.float32)[:, None], (1, self.n_agents))

        # Zero-copy numpy→tensor conversion (from_numpy shares memory when array
        # is already C-contiguous float32, which gym outputs typically are)
        obs_t = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
        gs_t = torch.from_numpy(np.ascontiguousarray(global_states, dtype=np.float32))
        act_t = torch.from_numpy(actions_store)  # float32 C-contiguous from .astype()
        lp_t = torch.from_numpy(np.ascontiguousarray(log_probs, dtype=np.float32))
        val_t = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        rew_t = torch.from_numpy(
            per_agent_rewards
        )  # float32 C-contiguous from arithmetic
        done_t = torch.from_numpy(dones_expanded)  # float32 C-contiguous from np.tile()
        masks_t = (
            torch.from_numpy(np.ascontiguousarray(action_masks, dtype=np.float32))
            if action_masks is not None
            else None
        )

        for env_idx in range(n_envs):
            self.global_states[env_idx].append(gs_t[env_idx])
            self.values[env_idx].append(val_t[env_idx])

            for agent_idx in range(self.n_agents):
                self.observations[env_idx][agent_idx].append(obs_t[env_idx, agent_idx])
                self.actions[env_idx][agent_idx].append(act_t[env_idx, agent_idx])
                self.rewards[env_idx][agent_idx].append(rew_t[env_idx, agent_idx])
                self.log_probs[env_idx][agent_idx].append(lp_t[env_idx, agent_idx])
                self.dones[env_idx][agent_idx].append(done_t[env_idx, agent_idx])
                if masks_t is not None:
                    self.action_masks[env_idx][agent_idx].append(
                        masks_t[env_idx, agent_idx]
                    )

    def compute_returns_and_advantages(self, next_values):
        """
        Compute GAE returns and advantages, vectorized over all envs and agents.

        Replaces three nested loops (n_envs × n_agents × T Python iterations)
        with a single reversed loop over T, operating on (n_envs, n_agents) tensors.
        Output list is dense: index = env_idx * n_agents + agent_idx.
        """
        T = len(self.values[0])

        # --- (n_envs, T+1) critic values ---
        env_vals_list = []
        for env_idx in range(self.n_parallel_envs):
            nv = next_values[env_idx]
            if not torch.is_tensor(nv):
                nv = torch.tensor(nv, dtype=torch.float32)
            # self.values[env_idx]: list of T tensors of shape (1,)
            env_vals = torch.cat(self.values[env_idx])  # (T,)
            env_vals_list.append(torch.cat([env_vals, nv.flatten()[:1]]))  # (T+1,)
        values_t = torch.stack(env_vals_list)  # (n_envs, T+1)

        # --- (n_envs, n_agents, T) rewards and dones ---
        rewards_t = torch.stack(
            [
                torch.stack(
                    [torch.stack(self.rewards[e][a]) for a in range(self.n_agents)]
                )
                for e in range(self.n_parallel_envs)
            ]
        )  # (n_envs, n_agents, T)

        dones_t = torch.stack(
            [
                torch.stack(
                    [torch.stack(self.dones[e][a]) for a in range(self.n_agents)]
                )
                for e in range(self.n_parallel_envs)
            ]
        )  # (n_envs, n_agents, T)

        # --- Vectorized GAE: one reversed loop over T ---
        # Broadcast values over agent dim: (n_envs, 1, T+1)
        vals = values_t.unsqueeze(1)
        advantages = torch.zeros(self.n_parallel_envs, self.n_agents, T)
        gae = torch.zeros(self.n_parallel_envs, self.n_agents)

        for step in reversed(range(T)):
            not_done = 1.0 - dones_t[:, :, step]  # (n_envs, n_agents)
            delta = (
                rewards_t[:, :, step]
                + self.gamma * vals[:, :, step + 1] * not_done
                - vals[:, :, step]
            )
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[:, :, step] = gae

        returns = advantages + values_t[:, :-1].unsqueeze(1)  # (n_envs, n_agents, T)

        # Flatten to list indexed by env_idx * n_agents + agent_idx
        all_returns = []
        all_advantages = []
        for env_idx in range(self.n_parallel_envs):
            for agent_idx in range(self.n_agents):
                all_returns.append(returns[env_idx, agent_idx].detach())
                all_advantages.append(advantages[env_idx, agent_idx].detach())

        return all_returns, all_advantages

    def update_shared(
        self,
        all_advantages,
        all_returns,
        minibatch_size,
        epochs,
        train_device,
    ):

        # Training statistics
        stats = {
            "total_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "entropy_loss": 0,
        }
        num_updates = 0

        # Build timestep-centric tensors so the centralized critic runs once per
        # (env, timestep) batch element rather than once per duplicated agent sample.
        all_obs = []
        all_global_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns_combined = []
        all_advantages_combined = []
        all_masks = []

        has_masks = len(self.action_masks[0][0]) > 0

        for env_idx in range(self.n_parallel_envs):
            if len(self.values[env_idx]) == 0:
                continue

            env_global_states = torch.stack(self.global_states[env_idx])
            env_obs = []
            env_actions = []
            env_old_log_probs = []
            env_returns = []
            env_advantages = []

            for agent_idx in range(self.n_agents):
                if len(self.observations[env_idx][agent_idx]) == 0:
                    continue

                data_idx = env_idx * self.n_agents + agent_idx
                if data_idx >= len(all_advantages):
                    continue

                advantages = all_advantages[data_idx]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                obs = torch.stack(self.observations[env_idx][agent_idx])
                actions = torch.stack(self.actions[env_idx][agent_idx])
                old_log_probs = torch.stack(self.log_probs[env_idx][agent_idx])
                returns = all_returns[data_idx]

                env_obs.append(obs)
                env_actions.append(actions)
                env_old_log_probs.append(old_log_probs)
                env_returns.append(returns)
                env_advantages.append(advantages)

            if not env_obs:
                continue

            all_obs.append(torch.stack(env_obs, dim=1))
            all_global_states.append(env_global_states)
            all_actions.append(torch.stack(env_actions, dim=1))
            all_old_log_probs.append(torch.stack(env_old_log_probs, dim=1))
            all_returns_combined.append(torch.stack(env_returns, dim=1))
            all_advantages_combined.append(torch.stack(env_advantages, dim=1))
            if has_masks:
                env_masks = [
                    torch.stack(self.action_masks[env_idx][agent_idx])
                    for agent_idx in range(self.n_agents)
                ]
                all_masks.append(torch.stack(env_masks, dim=1))

        # Concatenate to timestep-major tensors:
        # obs/actions/... shape (n_total_ts, n_agents, ...)
        combined_obs = torch.cat(all_obs, dim=0).detach().to(train_device)
        combined_global_states = (
            torch.cat(all_global_states, dim=0).detach().to(train_device)
        )
        combined_actions = torch.cat(all_actions, dim=0).detach().to(train_device)
        combined_old_log_probs = (
            torch.cat(all_old_log_probs, dim=0).detach().to(train_device)
        )
        combined_returns = torch.cat(all_returns_combined, dim=0).to(train_device)
        combined_advantages = torch.cat(all_advantages_combined, dim=0).to(train_device)
        combined_masks = (
            torch.cat(all_masks, dim=0).detach().to(train_device) if has_masks else None
        )

        # Create dataset
        dataset_tensors = [
            combined_obs,
            combined_global_states,
            combined_actions,
            combined_old_log_probs,
            combined_returns,
            combined_advantages,
        ]
        if combined_masks is not None:
            dataset_tensors.append(combined_masks)

        dataset = TensorDataset(*dataset_tensors)

        timestep_minibatch_size = max(1, minibatch_size // self.n_agents)
        dataloader = DataLoader(
            dataset, batch_size=timestep_minibatch_size, shuffle=True
        )

        # Train for multiple epochs
        for epoch in range(epochs):
            for batch in dataloader:
                # Unpack batch — order matches dataset_tensors above
                (
                    batch_obs,
                    batch_global_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_returns,
                    batch_advantages,
                    *extra,
                ) = batch
                batch_masks = extra.pop(0) if combined_masks is not None else None

                batch_n_ts = batch_obs.shape[0]
                batch_obs_flat = batch_obs.reshape(batch_n_ts * self.n_agents, -1)
                batch_actions_flat = batch_actions.reshape(
                    batch_n_ts * self.n_agents, *batch_actions.shape[2:]
                )
                batch_old_log_probs_flat = batch_old_log_probs.reshape(-1)
                batch_advantages_flat = batch_advantages.reshape(-1)
                batch_masks_flat = (
                    batch_masks.reshape(batch_n_ts * self.n_agents, -1)
                    if batch_masks is not None
                    else None
                )

                # Actor evaluation (batched over individual agent observations)
                actor = self.network.get_actor(0)
                log_probs, entropy = actor.evaluate(
                    batch_obs_flat,
                    batch_actions_flat,
                    action_mask=batch_masks_flat,
                )

                # Critic evaluation
                values = self.network.critic(batch_global_states).squeeze(-1)

                # PPO objective
                ratio = torch.exp(log_probs.squeeze(-1) - batch_old_log_probs_flat)
                surr1 = ratio * batch_advantages_flat
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages_flat
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Shared critic value is compared against each agent's return.
                value_loss = F.mse_loss(
                    values.unsqueeze(1).expand_as(batch_returns), batch_returns
                )

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.grad_clip
                )
                self.optimizer.step()

                # Update statistics
                stats["total_loss"] += loss.item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy_loss"] += entropy_loss.item()

                num_updates += 1

        return stats, num_updates

    def update_independent_actors(
        self,
        all_advantages,
        all_returns,
        minibatch_size,
        epochs,
        train_device,
    ):

        # Training statistics
        stats = {
            "total_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "entropy_loss": 0,
        }
        num_updates = 0

        has_masks = len(self.action_masks[0][0]) > 0

        # Update each agent separately (independent actors)
        for agent_idx in range(self.n_agents):
            # Collect data for this agent across all environments
            agent_obs = []
            agent_global_states = []
            agent_actions = []
            agent_old_log_probs = []
            agent_returns = []
            agent_advantages = []
            agent_masks = []

            for env_idx in range(self.n_parallel_envs):
                if len(self.observations[env_idx][agent_idx]) == 0:
                    continue

                # Get the index in the flattened advantages/returns list
                data_idx = env_idx * self.n_agents + agent_idx

                if data_idx >= len(all_advantages):
                    continue

                # Normalize advantages for this agent in this environment
                advantages = all_advantages[data_idx]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Stack data
                obs = torch.stack(self.observations[env_idx][agent_idx])
                actions = torch.stack(self.actions[env_idx][agent_idx])
                old_log_probs = torch.stack(self.log_probs[env_idx][agent_idx])
                returns = all_returns[data_idx]
                global_states = torch.stack(self.global_states[env_idx])

                # Append to agent-specific lists
                agent_obs.append(obs)
                agent_global_states.append(global_states)
                agent_actions.append(actions)
                agent_old_log_probs.append(old_log_probs)
                agent_returns.append(returns)
                agent_advantages.append(advantages)

                if has_masks:
                    agent_masks.append(
                        torch.stack(self.action_masks[env_idx][agent_idx])
                    )

            if len(agent_obs) == 0:
                continue  # No data for this agent

            # Concatenate data for this agent from all environments and move to training device
            obs_combined = torch.cat(agent_obs, dim=0).detach().to(train_device)
            global_states_combined = (
                torch.cat(agent_global_states, dim=0).detach().to(train_device)
            )
            actions_combined = torch.cat(agent_actions, dim=0).detach().to(train_device)
            old_log_probs_combined = (
                torch.cat(agent_old_log_probs, dim=0).detach().to(train_device)
            )
            returns_combined = torch.cat(agent_returns, dim=0).to(train_device)
            advantages_combined = torch.cat(agent_advantages, dim=0).to(train_device)
            masks_combined = (
                torch.cat(agent_masks, dim=0).detach().to(train_device)
                if has_masks
                else None
            )

            # Create dataset for this agent
            dataset_tensors = [
                obs_combined,
                global_states_combined,
                actions_combined,
                old_log_probs_combined,
                returns_combined,
                advantages_combined,
            ]
            if masks_combined is not None:
                dataset_tensors.append(masks_combined)

            dataset = TensorDataset(*dataset_tensors)
            dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

            # Train for multiple epochs
            for epoch in range(epochs):
                for batch in dataloader:
                    # Unpack batch — order matches dataset_tensors above
                    (
                        batch_obs,
                        batch_global_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_returns,
                        batch_advantages,
                        *extra,
                    ) = batch
                    batch_masks = extra.pop(0) if masks_combined is not None else None

                    # Actor evaluation
                    actor = self.network.get_actor(agent_idx)
                    log_probs, entropy = actor.evaluate(
                        batch_obs, batch_actions, action_mask=batch_masks
                    )

                    # Critic evaluation
                    values = self.network.critic(batch_global_states).squeeze(-1)

                    # PPO objective
                    ratio = torch.exp(log_probs.squeeze(-1) - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        * batch_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(values, batch_returns)

                    # Entropy loss
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss
                    )

                    # Optimize
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.grad_clip
                    )
                    self.optimizer.step()

                    # Update statistics
                    stats["total_loss"] += loss.item()
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy_loss"] += entropy_loss.item()

                    num_updates += 1

        return stats, num_updates

    def update(self, next_value=0, minibatch_size=128, epochs=10):
        """Update all agents using shared critic"""

        # Compute returns and advantages
        all_returns, all_advantages = self.compute_returns_and_advantages(next_value)

        # Pre-update explained variance: how well the centralized critic's
        # stored predictions explain the GAE returns on this rollout.
        # Ordering of all_returns is env_idx * n_agents + agent_idx, so values
        # for each env are repeated n_agents times contiguously to align.
        with torch.no_grad():
            flat_returns = torch.cat(all_returns)
            flat_values = torch.cat(
                [
                    torch.cat(self.values[e]).repeat(self.n_agents)
                    for e in range(self.n_parallel_envs)
                ]
            )
            var_returns = flat_returns.var()
            explained_variance = 1.0 - (flat_returns - flat_values).var() / (
                var_returns + 1e-8
            )

        # Update each agent (or all at once if sharing actor)
        if self.share_actor:
            stats, num_updates = self.update_shared(
                all_advantages, all_returns, minibatch_size, epochs, self.device
            )

        else:
            stats, num_updates = self.update_independent_actors(
                all_advantages, all_returns, minibatch_size, epochs, self.device
            )

        # Update old network
        self.network_old.load_state_dict(self.network.state_dict())

        # Reset buffers
        self.reset_buffers()

        # Average statistics
        for key in stats:
            stats[key] /= max(1, num_updates)

        stats["explained_variance"] = float(explained_variance.item())

        return stats
