import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from algorithms.mappo.types import Params
from algorithms.mappo.network import MAPPONetwork


class MAPPOAgent:
    """Multi-Agent PPO with centralized critic"""

    def __init__(
        self,
        observation_dim: int,
        global_state_dim: int,
        action_dim: int,
        n_agents: int,
        params: Params,
        device: str,
        discrete: bool,
        n_parallel_envs: int,
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
        self.network = MAPPONetwork(
            observation_dim=observation_dim,
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            discrete=discrete,
            share_actor=self.share_actor,
        ).to(self.device)

        # Create old network for PPO
        self.network_old = MAPPONetwork(
            observation_dim=observation_dim,
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            discrete=discrete,
            share_actor=self.share_actor,
        ).to(self.device)

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

    def get_actions(self, observations, global_state, deterministic=False):
        """
        Get actions for all agents

        Args:
            observations: List of observations, one per agent
            global_state: Concatenated global state (all agent observations)
            deterministic: Whether to use deterministic actions

        Returns:
            actions: List of actions for each agent
            log_probs: List of log probabilities
            value: Single value from centralized critic
        """
        with torch.no_grad():
            # Convert observations to tensors
            obs_tensors = [
                torch.FloatTensor(obs).to(self.device) for obs in observations
            ]

            # Convert global state to tensor
            global_state_tensor = torch.FloatTensor(global_state).to(self.device)

            # Get actions from each actor
            actions = []
            log_probs = []

            for agent_idx, obs_tensor in enumerate(obs_tensors):
                action, log_prob = self.network_old.act(
                    obs_tensor, agent_idx, deterministic
                )
                actions.append(action)
                log_probs.append(log_prob)

            # Get value from centralized critic
            value = self.network_old.get_value(global_state_tensor)

        return torch.stack(actions), torch.cat(log_probs), value

    def get_actions_batched(
        self, observations_batch, global_states_batch, deterministic=False, action_masks=None
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
            obs_tensor = torch.FloatTensor(observations_batch).to(
                self.device
            )  # (n_envs, n_agents, obs_dim)
            gs_tensor = torch.FloatTensor(global_states_batch).to(
                self.device
            )  # (n_envs, global_state_dim)

            masks_tensor = (
                torch.FloatTensor(action_masks).to(self.device)
                if action_masks is not None
                else None
            )  # (n_envs, n_agents, n_actions) or None

            if self.share_actor:
                # Fuse envs and agents into one batch: single forward pass total
                obs_flat = obs_tensor.reshape(n_envs * self.n_agents, -1)
                masks_flat = (
                    masks_tensor.reshape(n_envs * self.n_agents, -1)
                    if masks_tensor is not None
                    else None
                )
                actions_flat, log_probs_flat = self.network_old.act(
                    obs_flat, agent_idx=0, deterministic=deterministic,
                    action_mask=masks_flat,
                )
                actions = actions_flat.reshape(n_envs, self.n_agents, -1)
                log_probs = log_probs_flat.reshape(n_envs, self.n_agents, -1).squeeze(
                    -1
                )
            else:
                # One batched pass per agent (n_agents passes, each over all envs)
                actions_list = []
                log_probs_list = []
                for agent_idx in range(self.n_agents):
                    agent_obs = obs_tensor[:, agent_idx, :]  # (n_envs, obs_dim)
                    agent_mask = (
                        masks_tensor[:, agent_idx, :]
                        if masks_tensor is not None
                        else None
                    )
                    a, lp = self.network_old.act(
                        agent_obs, agent_idx, deterministic, action_mask=agent_mask
                    )
                    actions_list.append(a)  # (n_envs, action_dim)
                    log_probs_list.append(lp)  # (n_envs, 1)
                actions = torch.stack(
                    actions_list, dim=1
                )  # (n_envs, n_agents, action_dim)
                log_probs = torch.stack(log_probs_list, dim=1).squeeze(
                    -1
                )  # (n_envs, n_agents)

            # Centralized critic: single batched pass over all envs
            values = self.network_old.get_value(gs_tensor)  # (n_envs, 1)

        return actions, log_probs, values

    def store_transition(
        self,
        env_idx,
        observations,
        global_state,
        actions,
        rewards,
        log_probs,
        value,
        dones,
        action_masks=None,
    ):
        """Store transition for a specific environment (CPU buffers to avoid small GPU transfers)"""
        # Store global state (shared)
        self.global_states[env_idx].append(torch.FloatTensor(global_state))

        # Store value (shared)
        self.values[env_idx].append(torch.tensor(value, dtype=torch.float32))

        # Store per-agent data
        for agent_idx in range(self.n_agents):
            # Observation
            self.observations[env_idx][agent_idx].append(
                torch.FloatTensor(observations[agent_idx])
            )

            # Action
            self.actions[env_idx][agent_idx].append(
                torch.FloatTensor(actions[agent_idx])
            )

            # Reward, log_prob, done
            self.rewards[env_idx][agent_idx].append(
                torch.tensor(rewards[agent_idx], dtype=torch.float32)
            )
            self.log_probs[env_idx][agent_idx].append(
                torch.tensor(log_probs[agent_idx], dtype=torch.float32)
            )
            self.dones[env_idx][agent_idx].append(
                torch.tensor(dones[agent_idx], dtype=torch.float32)
            )
            if action_masks is not None:
                self.action_masks[env_idx][agent_idx].append(
                    torch.FloatTensor(action_masks[agent_idx])
                )

    def compute_returns_and_advantages(self, next_values):
        """
        Compute returns and advantages using per-environment final values

        Args:
            next_values: List or array of final values, one per environment
        """
        all_returns = []
        all_advantages = []

        # Process each environment separately
        for env_idx in range(self.n_parallel_envs):
            if len(self.values[env_idx]) == 0:
                continue  # Skip if no data for this env

            next_value = next_values[env_idx]
            if not torch.is_tensor(next_value):
                next_value = torch.tensor(next_value, dtype=torch.float32)

            # Use values from this environment's trajectory
            env_values = self.values[env_idx][:]
            env_values.append(next_value.unsqueeze(0))
            env_values = torch.cat(env_values)

            # Compute advantages for each agent in this environment
            for agent_idx in range(self.n_agents):
                if len(self.rewards[env_idx][agent_idx]) == 0:
                    continue

                rewards = torch.stack(self.rewards[env_idx][agent_idx])
                dones = torch.cat(
                    [
                        torch.stack(self.dones[env_idx][agent_idx]),
                        torch.zeros(1),
                    ]
                )

                # Initialize advantages
                advantages = torch.zeros_like(rewards)

                # Compute GAE
                gae = torch.tensor(0.0)
                for step in reversed(range(len(rewards))):
                    delta = (
                        rewards[step]
                        + self.gamma * env_values[step + 1] * (1 - dones[step])
                        - env_values[step]
                    )
                    gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
                    advantages[step] = gae

                # Compute returns
                returns = advantages + env_values[:-1]

                all_returns.append(returns.detach())
                all_advantages.append(advantages.detach())

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
        stats = {"total_loss": 0, "policy_loss": 0, "value_loss": 0, "entropy_loss": 0}
        num_updates = 0

        # Combine all agent data for shared actor update
        all_obs = []
        all_global_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns_combined = []
        all_advantages_combined = []
        all_masks = []

        has_masks = len(self.action_masks[0][0]) > 0

        # Iterate through each environment
        for env_idx in range(self.n_parallel_envs):
            if len(self.values[env_idx]) == 0:
                continue  # Skip empty environments

            # For this environment, iterate through each agent
            for agent_idx in range(self.n_agents):
                if len(self.observations[env_idx][agent_idx]) == 0:
                    continue  # Skip if no data for this agent

                # Get the index in the flattened advantages/returns list
                # The compute_returns_and_advantages returns a flat list combining all envs and agents
                data_idx = env_idx * self.n_agents + agent_idx

                if data_idx >= len(all_advantages):
                    continue

                # Normalize advantages for this agent in this environment
                advantages = all_advantages[data_idx]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Stack data for this agent in this environment
                obs = torch.stack(self.observations[env_idx][agent_idx])
                actions = torch.stack(self.actions[env_idx][agent_idx])
                old_log_probs = torch.stack(self.log_probs[env_idx][agent_idx])
                returns = all_returns[data_idx]

                # Get corresponding global states (repeated for each agent's timestep)
                global_states = torch.stack(self.global_states[env_idx])

                # Append to combined lists
                all_obs.append(obs)
                all_global_states.append(global_states)
                all_actions.append(actions)
                all_old_log_probs.append(old_log_probs)
                all_returns_combined.append(returns)
                all_advantages_combined.append(advantages)

                if has_masks:
                    all_masks.append(torch.stack(self.action_masks[env_idx][agent_idx]))

        # Concatenate all data and move to training device (may be CUDA)
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

        dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

        # Train for multiple epochs
        for epoch in range(epochs):
            for batch in dataloader:
                if combined_masks is not None:
                    (
                        batch_obs,
                        batch_global_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_returns,
                        batch_advantages,
                        batch_masks,
                    ) = batch
                else:
                    (
                        batch_obs,
                        batch_global_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_returns,
                        batch_advantages,
                    ) = batch
                    batch_masks = None

                # We need to know which agent each sample belongs to
                # For shared actor, we can use agent_idx = 0 (same for all)
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch_obs, batch_global_states, batch_actions, agent_idx=0,
                    action_mask=batch_masks,
                )

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
                self.optimizer.zero_grad()
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
        stats = {"total_loss": 0, "policy_loss": 0, "value_loss": 0, "entropy_loss": 0}
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
                    continue  # Skip if no data for this agent in this environment

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
                    if masks_combined is not None:
                        (
                            batch_obs,
                            batch_global_states,
                            batch_actions,
                            batch_old_log_probs,
                            batch_returns,
                            batch_advantages,
                            batch_masks,
                        ) = batch
                    else:
                        (
                            batch_obs,
                            batch_global_states,
                            batch_actions,
                            batch_old_log_probs,
                            batch_returns,
                            batch_advantages,
                        ) = batch
                        batch_masks = None

                    # Forward pass
                    log_probs, values, entropy = self.network.evaluate_actions(
                        batch_obs, batch_global_states, batch_actions, agent_idx,
                        action_mask=batch_masks,
                    )

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
                    self.optimizer.zero_grad()
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

        return stats
