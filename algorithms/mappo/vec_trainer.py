import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
import time
import random
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
        observation_dim=None,
        global_state_dim=None,
        action_dim=None,
        params=None,
        dirs=None,
        device: str = "cpu",
        n_parallel_envs: int = 1,
        map_name: str = None,
    ):
        self.device = device
        self.dirs = dirs
        self.n_agents = n_agents
        self.params = params
        self.env_name = env_name
        self.n_parallel_envs = n_parallel_envs
        self.map_name = map_name

        # Create environment for evaluation (parallel eval episodes)
        self.n_eval_episodes = 10
        self.eval_env = make_vec_env(
            self.env_name,
            self.n_agents,
            self.n_eval_episodes,
            use_async=True,
            map_name=self.map_name,
        )

        # Create vectorized environment using Gymnasium's API
        self.vec_env = make_vec_env(
            self.env_name,
            self.n_agents,
            self.n_parallel_envs,
            use_async=True,  # Use parallel processing
            map_name=self.map_name,
        )

        # Derive dims from the env when not provided, avoiding a separate probe env
        if observation_dim is None:
            import gymnasium as gym
            obs_space = self.vec_env.single_observation_space   # Box(n_agents, obs_dim)
            act_space = self.vec_env.single_action_space
            observation_dim = obs_space.shape[1]
            global_state_dim = observation_dim * n_agents
            if isinstance(act_space, gym.spaces.MultiDiscrete):
                action_dim = int(act_space.nvec[0])
            else:
                action_dim = act_space.shape[1]

        # Set action bounds based on environment
        if env_name in [
            EnvironmentEnum.MPE_SPREAD,
            EnvironmentEnum.MPE_SIMPLE,
            EnvironmentEnum.SMACV2,
        ]:
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

        # Sync device — agent may have upgraded to CUDA in its __init__
        self.device = self.agent.device

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

        # Grab initial action masks if the env provides them (e.g. SMACv2)
        # Shape: (n_envs, n_agents, n_actions) or None
        current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None

        total_step_count = 0
        episode_count = 0
        current_episode_steps = np.zeros(self.n_parallel_envs, dtype=np.int32)

        while total_step_count <= max_steps:

            # Construct global states for each environment
            # Shape: (n_envs, n_agents * obs_dim)
            global_states = obs.reshape(batch_size, -1)

            # Single batched forward pass for all envs × agents
            actions_t, log_probs_t, values_t = self.agent.get_actions_batched(
                obs, global_states, deterministic=False, action_masks=current_masks
            )

            # Convert to numpy; shapes: (n_envs, n_agents, action_dim), (n_envs, n_agents), (n_envs, 1)
            actions_array = actions_t.cpu().numpy()
            log_probs_array = log_probs_t.cpu().numpy()
            values_array = values_t.cpu().numpy()

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

                env_masks = current_masks[env_idx] if current_masks is not None else None

                self.agent.store_transition(
                    env_idx,
                    obs[env_idx],
                    global_states[env_idx],
                    actions_to_store,
                    individual_rewards + rewards[env_idx],
                    log_probs_array[env_idx],
                    values_array[env_idx],
                    np.array([dones[env_idx]] * self.n_agents),
                    action_masks=env_masks,
                )

            # Update masks for next step.
            # For auto-reset envs (terminated), Gymnasium VecEnv merges the reset()
            # info into the regular infos dict, so avail_actions reflects the new episode.
            current_masks = (
                infos.get("avail_actions") if isinstance(infos, dict) else None
            )

            # Update for next iteration
            obs = next_obs
            total_step_count += batch_size
            current_episode_steps += 1

            # Track episode terminations with vectorized ops
            episode_count += int(dones.sum())
            current_episode_steps[dones] = 0

            # Gymnasium VectorEnv automatically resets terminated environments
            # The obs returned is already the reset observation for terminated envs

        # Compute final values for advantage computation in a single batched pass
        final_global_states = obs.reshape(batch_size, -1)
        with torch.no_grad():
            final_gs_tensor = torch.FloatTensor(final_global_states).to(self.device)
            final_values = (
                self.agent.network_old.get_value(final_gs_tensor)
                .cpu()
                .squeeze(-1)
                .tolist()
            )

        return total_step_count, episode_count, final_values

    def train(
        self,
        total_steps,
        batch_size,
        minibatches,
        epochs,
        log_every=10000,
        resume_steps=0,
        resume_episodes=0,
    ):
        """Train MAPPO agent"""

        # Start timing
        self.training_start_time = time.time()

        # TODO keep track of elapsed time, so that FPS calculation is correct when using checkpoints
        steps_completed = resume_steps
        episodes_completed = resume_episodes

        if resume_steps > 0:
            print(f"Resuming MAPPO training from step {resume_steps}/{total_steps}...")
        else:
            print(
                f"Starting MAPPO training for {total_steps} total environment steps..."
            )

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
            eval_rewards = self.evaluate()
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
                self.save_agent(
                    self.dirs["models"] / "models_checkpoint.pth",
                    steps_completed=steps_completed,
                    episodes_completed=episodes_completed,
                )

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

    def evaluate(self):
        """Evaluate current policy using parallel episodes"""

        self.agent.network_old.eval()
        n_eps = self.n_eval_episodes

        with torch.no_grad():
            obs, infos = self.eval_env.reset()
            current_masks = (
                infos.get("avail_actions") if isinstance(infos, dict) else None
            )
            episode_rewards = np.zeros(n_eps)
            finished = np.zeros(n_eps, dtype=bool)

            while not finished.all():
                global_states = obs.reshape(n_eps, -1)

                actions_t, _, _ = self.agent.get_actions_batched(
                    obs, global_states, deterministic=True, action_masks=current_masks
                )
                actions = actions_t.cpu().numpy()

                if self.discrete:
                    if actions.ndim == 3 and actions.shape[-1] == 1:
                        actions = actions.squeeze(-1)
                    actions = actions.astype(np.int32)

                obs, rewards, terminated, truncated, infos = self.eval_env.step(actions)
                current_masks = (
                    infos.get("avail_actions") if isinstance(infos, dict) else None
                )

                # Only accumulate rewards for episodes not yet finished
                dones = np.logical_or(terminated, truncated)
                episode_rewards[~finished] += rewards[~finished]
                finished |= dones

        self.agent.network_old.train()

        return episode_rewards.mean()

    def close_environments(self):
        """Properly close all vectorized environments"""
        self.vec_env.close()
        self.eval_env.close()

    def __del__(self):
        """Destructor to ensure environments are closed"""
        self.close_environments()

    def save_agent(self, path, steps_completed=0, episodes_completed=0):
        """Save MAPPO agent with RNG states and training progress"""
        torch.save(
            {
                "network": self.agent.network_old.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
                "steps_completed": steps_completed,
                "episodes_completed": episodes_completed,
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch_cpu": torch.random.get_rng_state(),
                "rng_torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            path,
        )

    def load_agent(self, filepath, restore_rng=False):
        checkpoint = torch.load(filepath, map_location=self.device)

        self.agent.network_old.load_state_dict(checkpoint["network"])
        self.agent.network.load_state_dict(checkpoint["network"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])

        if restore_rng and "rng_python" in checkpoint:
            random.setstate(checkpoint["rng_python"])
            np.random.set_state(checkpoint["rng_numpy"])
            torch.random.set_rng_state(checkpoint["rng_torch_cpu"].cpu())
            if torch.cuda.is_available() and checkpoint["rng_torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(
                    [s.cpu() for s in checkpoint["rng_torch_cuda"]]
                )

        steps = checkpoint.get("steps_completed", 0)
        episodes = checkpoint.get("episodes_completed", 0)
        print(f"Agents loaded from {filepath} (steps={steps}, episodes={episodes})")
        return steps, episodes

    def save_training_stats(self, path):
        """Save training statistics"""
        with open(path, "wb") as f:
            pickle.dump(dict(self.training_stats), f)
