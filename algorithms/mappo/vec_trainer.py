import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
import time
import psutil

from algorithms.mappo.mappo import MAPPOAgent
from environments.types import EnvironmentEnum
from algorithms.create_env import make_vec_env

import torch


class VecMAPPOTrainer:
    def __init__(
        self,
        env_name,
        n_agents,
        observation_dim,
        global_state_dim,
        action_dim,
        params,
        dirs,
        device: str,
        n_parallel_envs: int,
    ):
        self.device = device
        self.dirs = dirs
        self.n_agents = n_agents
        self.params = params
        self.env_name = env_name
        self.n_parallel_envs = n_parallel_envs

        # Create environment for evaluation
        self.eval_env = make_vec_env(
            self.env_name,
            self.n_agents,
            1,
            use_async=False,
        )

        # Create vectorized environment using Gymnasium's API
        self.vec_env = make_vec_env(
            self.env_name,
            self.n_agents,
            self.n_parallel_envs,
            use_async=False,  # Use parallel processing
        )

        # Set action bounds based on environment
        if env_name in [EnvironmentEnum.MPE_SPREAD, EnvironmentEnum.MPE_SIMPLE]:
            self.discrete = True
        else:
            self.discrete = False

        # Create MAPPO agent
        self.agent = MAPPOAgent(
            observation_dim,
            global_state_dim,
            action_dim,
            self.n_agents,
            self.params,
            self.device,
            self.discrete,
            self.n_parallel_envs,
        )

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = defaultdict(list)

        # Timing statistics
        self.training_start_time = None
        self.total_training_time = 0.0

    def collect_trajectory(self, max_steps):
        """
        Collect trajectory using Gymnasium's vectorized environments.

        Gymnasium VectorEnv provides:
        - obs: (n_envs, *obs_shape)
        - reward: (n_envs,)
        - terminated: (n_envs,)
        - truncated: (n_envs,)
        - info: dict with (n_envs,) arrays
        """

        # Reset environment
        obs, infos = self.vec_env.reset()
        batch_size = obs.shape[0]

        total_step_count = 0
        episode_count = 0
        current_episode_steps = np.zeros(self.n_parallel_envs, dtype=np.int32)

        while total_step_count <= max_steps:

            # Construct global states for each environment
            # Shape: (n_envs, n_agents * obs_dim)
            global_states = obs.reshape(batch_size, -1)

            # Get actions for all agents in all environments
            all_actions = []
            all_log_probs = []
            all_values = []

            with torch.no_grad():
                for env_idx in range(batch_size):
                    # Get observations for this environment
                    env_obs = obs[env_idx]  # (n_agents, obs_dim)
                    env_global_state = global_states[env_idx]  # (n_agents * obs_dim)

                    # Get actions for all agents in this environment
                    actions, log_probs, value = self.agent.get_actions(
                        env_obs, env_global_state, deterministic=False
                    )

                    all_actions.append(actions)
                    all_log_probs.append(log_probs)
                    all_values.append(value)

            # Convert to numpy arrays
            # actions shape: (n_envs, n_agents, action_dim)
            actions_array = np.array(all_actions)
            log_probs_array = np.array(all_log_probs)
            values_array = np.array(all_values)

            # IMPORTANT: Reshape actions for discrete environments
            if self.discrete:
                # For discrete actions, squeeze the last dimension
                # Shape: (n_envs, n_agents, 1) -> (n_envs, n_agents)
                if actions_array.ndim == 3 and actions_array.shape[-1] == 1:
                    actions_array = actions_array.squeeze(-1)

                # Ensure integer type
                actions_array = actions_array.astype(np.int32)

            # Step all environments in parallel
            # Gymnasium VectorEnv step returns:
            # - next_obs: (n_envs, n_agents, obs_dim)
            # - rewards: (n_envs,)
            # - terminateds: (n_envs,)
            # - truncateds: (n_envs,)
            # - infos: dict with keys that have (n_envs,) shape
            next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(
                actions_array
            )

            # Compute dones for each environment
            dones = np.logical_or(terminateds, truncateds)

            # Store transitions for all environments
            for env_idx in range(batch_size):
                # Extract individual rewards for each agent from info
                # Your SalpChainEnv returns info['local_rewards']
                individual_rewards = [0.0 for _ in range(self.n_agents)]
                if "local_rewards" in infos:
                    # infos['local_rewards'] has shape (n_envs, n_agents)
                    individual_rewards = infos["local_rewards"][env_idx]

                # For storage, we need to restore the action dimension if discrete
                # because the buffer expects (n_agents, 1) for discrete
                actions_to_store = actions_array[env_idx]
                if self.discrete and actions_to_store.ndim == 1:
                    actions_to_store = actions_to_store.reshape(-1, 1)

                self.agent.store_transition(
                    env_idx,
                    obs[env_idx],
                    global_states[env_idx],
                    actions_to_store,
                    individual_rewards + rewards[env_idx],
                    log_probs_array[env_idx],
                    values_array[env_idx],
                    np.array([dones[env_idx]] * self.n_agents),
                )

            # Update for next iteration
            obs = next_obs
            total_step_count += batch_size
            current_episode_steps += 1

            # Check for episode terminations
            for env_idx in range(batch_size):
                if dones[env_idx]:
                    current_episode_steps[env_idx] = 0
                    episode_count += 1

            # Gymnasium VectorEnv automatically resets terminated environments
            # The obs returned is already the reset observation for terminated envs

        # Get final values for advantage computation
        final_global_states = obs.reshape(batch_size, -1)
        final_values = []

        with torch.no_grad():
            for env_idx in range(batch_size):
                global_state_tensor = (
                    torch.FloatTensor(final_global_states[env_idx])
                    .unsqueeze(0)
                    .to(self.device)
                )
                value = (
                    self.agent.network_old.get_value(global_state_tensor).cpu().item()
                )
                final_values.append(value)

        return total_step_count, episode_count, final_values

    def train(self, total_steps, batch_size, minibatches, epochs, log_every=10000):
        """Train MAPPO agent"""
        print(f"Starting MAPPO training for {total_steps} total environment steps...")

        # Start timing
        self.training_start_time = time.time()

        steps_completed = 0
        episodes_completed = 0

        self.training_stats["total_steps"] = []
        self.training_stats["reward"] = []
        self.training_stats["episodes"] = []
        self.training_stats["training_time"] = []
        self.training_stats["collection_time"] = []
        self.training_stats["update_time"] = []
        self.training_stats["eval_time"] = []

        while steps_completed < total_steps:
            steps_to_collect = min(batch_size, total_steps - steps_completed)

            # Time trajectory collection
            collection_start = time.time()
            step_count, episode_count, final_values = self.collect_trajectory(
                max_steps=int(steps_to_collect)
            )
            collection_time = time.time() - collection_start

            # Time agent update
            update_start = time.time()
            stats = self.agent.update(
                next_value=final_values,
                minibatch_size=batch_size // minibatches,
                epochs=epochs,
            )
            update_time = time.time() - update_start

            # Update tracking
            steps_completed += step_count
            episodes_completed += episode_count

            # Store statistics
            for key, value in stats.items():
                self.training_stats[key].append(value)

            # Time evaluation
            eval_start = time.time()
            rew_per_episode = []
            eval_episodes = 5
            while len(rew_per_episode) < eval_episodes:
                rew_per_episode.append(self.evaluate())
            eval_rewards = np.array(rew_per_episode).mean()
            eval_time = time.time() - eval_start

            # Calculate elapsed time
            elapsed_time = time.time() - self.training_start_time
            self.total_training_time = elapsed_time

            # Store timing statistics
            self.training_stats["total_steps"].append(steps_completed)
            self.training_stats["reward"].append(eval_rewards)
            self.training_stats["episodes"].append(episodes_completed)
            self.training_stats["training_time"].append(elapsed_time)
            self.training_stats["collection_time"].append(collection_time)
            self.training_stats["update_time"].append(update_time)
            self.training_stats["eval_time"].append(eval_time)

            # Calculate throughput
            steps_per_second = steps_completed / elapsed_time if elapsed_time > 0 else 0

            # Log progress
            if steps_completed % log_every < step_count:
                mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                print(
                    f"Steps: {steps_completed}/{total_steps} ({steps_completed/total_steps*100:.1f}%) | "
                    f"Episodes: {episodes_completed} | "
                    f"Reward: {self.training_stats['reward'][-1]:.2f} | "
                    f"Time: {elapsed_time:.1f}s | "
                    f"FPS: {steps_per_second:.1f} | "
                    f"Mem: {mem:.0f}MB | "
                    f"Collection: {collection_time:.2f}s | "
                    f"Update: {update_time:.2f}s | "
                    f"Eval: {eval_time:.2f}s"
                )

                self.save_training_stats(
                    self.dirs["logs"] / "training_stats_checkpoint.pkl"
                )
                self.save_agent(self.dirs["models"] / "models_checkpoint.pth")

        # Close envs
        self.close_environments()

        # Final timing summary
        total_time = time.time() - self.training_start_time
        avg_collection_time = np.mean(self.training_stats["collection_time"])
        avg_update_time = np.mean(self.training_stats["update_time"])
        avg_eval_time = np.mean(self.training_stats["eval_time"])
        final_fps = steps_completed / total_time

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"{'='*60}")
        print(f"Total steps:          {steps_completed}")
        print(f"Total episodes:       {episodes_completed}")
        print(f"Total time:           {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"Final FPS:            {final_fps:.1f} steps/second")
        print(f"\nTime breakdown:")
        print(f"  Avg collection:     {avg_collection_time:.3f}s")
        print(f"  Avg update:         {avg_update_time:.3f}s")
        print(f"  Avg evaluation:     {avg_eval_time:.3f}s")
        print(f"{'='*60}\n")

    def evaluate(self, render=False):
        """Evaluate current policy"""

        # Set policies to eval
        self.agent.network_old.eval()

        with torch.no_grad():

            obs, _ = self.eval_env.reset()

            episode_rew = 0

            while True:

                global_state = np.concatenate(obs.squeeze(0))

                actions, _, _ = self.agent.get_actions(
                    obs, global_state, deterministic=True
                )

                actions = actions.cpu().detach().numpy()

                if self.discrete:
                    # For discrete actions, squeeze the last dimension
                    # Shape: (n_envs, n_agents, 1) -> (n_envs, n_agents)
                    if actions.ndim == 3 and actions.shape[-1] == 1:
                        actions = actions.squeeze(-1)

                    # Ensure integer type
                    actions = actions.astype(np.int32)

                obs, rewards, terminated, truncated, info = self.eval_env.step(actions)

                episode_rew += rewards

                if render:
                    self.eval_env.envs[0].render_mode = "human"
                    self.eval_env.envs[0].render()

                if terminated or truncated:
                    break

        # Set policies to train
        self.agent.network_old.train()

        return episode_rew

    def close_environments(self):
        """Properly close all vectorized environments"""
        self.vec_env.close()
        self.eval_env.close()

    def __del__(self):
        """Destructor to ensure environments are closed"""
        self.close_environments()

    def save_agent(self, path):
        """Save MAPPO agent"""
        torch.save(
            {
                "network": self.agent.network_old.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
            },
            path,
        )

    def load_agent(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)

        self.agent.network_old.load_state_dict(checkpoint["network"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"Agents loaded from {filepath}")

    def save_training_stats(self, path):
        """Save training statistics"""
        with open(path, "wb") as f:
            pickle.dump(dict(self.training_stats), f)
