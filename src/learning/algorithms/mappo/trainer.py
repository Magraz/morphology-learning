import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle

from learning.algorithms.mappo.mappo import MAPPOAgent
from learning.environments.types import EnvironmentEnum
from learning.algorithms.env_wrapper import EnvWrapper

import torch


class MAPPOTrainer:
    def __init__(
        self,
        env,
        env_name,
        n_agents,
        observation_dim,
        global_state_dim,
        action_dim,
        params,
        dirs=None,
        device="cpu",
        share_actor=True,
    ):
        self.device = device
        self.dirs = dirs
        self.n_agents = n_agents

        # Create environment
        self.wrapped_env = EnvWrapper(env=env, env_name=env_name, n_agents=n_agents)

        # Set action bounds based on environment
        if env_name in [EnvironmentEnum.MPE_SPREAD, EnvironmentEnum.MPE_SIMPLE]:
            self.discrete = True
        else:
            self.discrete = False

        # Create MAPPO agent
        self.agent = MAPPOAgent(
            observation_dim=observation_dim,
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            params=params,
            device=device,
            discrete=self.discrete,
            share_actor=share_actor,
        )

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = defaultdict(list)

    def collect_trajectory(self, max_steps):
        """Collect trajectory using MAPPO"""

        # Reset environment
        obs = self.wrapped_env.reset()

        total_step_count = 0
        episode_count = 0
        steps_per_episode = []
        current_episode_steps = 0

        for step in range(max_steps):
            # Construct global state (concatenate all observations)
            global_state = np.concatenate(obs)

            # Get actions from MAPPO agent
            actions, log_probs, value = self.agent.get_actions(
                obs, global_state, deterministic=False
            )

            # Step environment
            next_obs, global_reward, local_rewards, terminated, truncated, info = (
                self.wrapped_env.step(actions)
            )

            # Store transition
            self.agent.store_transition(
                torch.FloatTensor(obs).to(self.device),
                torch.FloatTensor(global_state).to(self.device),
                actions,
                torch.FloatTensor(local_rewards + global_reward).to(self.device),
                log_probs,
                value,
                torch.FloatTensor(np.logical_or(terminated, truncated)).to(self.device),
            )

            obs = next_obs
            total_step_count += 1
            current_episode_steps += 1

            # If environment terminated or truncated, reset it and continue collecting
            if terminated.all() or truncated.all():
                obs = self.wrapped_env.reset()

                # Keep track of episode count
                steps_per_episode.append(current_episode_steps)
                current_episode_steps = 0
                episode_count += 1

        # Get final value for advantage computation
        final_global_state = np.concatenate(obs)

        with torch.no_grad():
            final_global_state_tensor = (
                torch.FloatTensor(final_global_state).unsqueeze(0).to(self.device)
            )
            final_value = (
                self.agent.network_old.get_value(final_global_state_tensor).cpu().item()
            )

        return total_step_count, episode_count, steps_per_episode, final_value

    def train(self, total_steps, batch_size, minibatches, epochs, log_every=10000):
        """Train MAPPO agent"""
        print(f"Starting MAPPO training for {total_steps} total environment steps...")

        steps_completed = 0
        episodes_completed = 0

        self.training_stats["total_steps"] = []
        self.training_stats["reward"] = []
        self.training_stats["episodes"] = []
        self.training_stats["steps_per_episode"] = []

        while steps_completed < total_steps:
            steps_to_collect = min(batch_size, total_steps - steps_completed)

            # Collect trajectory
            step_count, episode_count, steps_per_episode, final_value = (
                self.collect_trajectory(max_steps=int(steps_to_collect))
            )

            # Update agent
            stats = self.agent.update(
                next_value=final_value,
                minibatch_size=batch_size // minibatches,
                epochs=epochs,
            )

            # Update tracking
            steps_completed += step_count
            episodes_completed += episode_count

            # Store statistics
            for key, value in stats.items():
                self.training_stats[key].append(value)

            # Evaluate
            rew_per_episode = []
            eval_episodes = 10
            while len(rew_per_episode) < eval_episodes:
                rew_per_episode.append(self.evaluate())
            eval_rewards = np.array(rew_per_episode).mean()

            self.training_stats["total_steps"].append(steps_completed)
            self.training_stats["reward"].append(eval_rewards)
            self.training_stats["episodes"].append(episodes_completed)
            self.training_stats["steps_per_episode"].extend(steps_per_episode)

            # Log progress
            if steps_completed % log_every < step_count:
                print(
                    f"Steps: {steps_completed}/{total_steps} ({steps_completed/total_steps*100:.1f}%) | "
                    f"Episodes: {episodes_completed} | "
                    f"Recent Avg Reward: {self.training_stats['reward'][-1]:.2f} | "
                    f"Last Batch Steps: {step_count}"
                )

                self.save_training_stats(
                    self.dirs["logs"] / "training_stats_checkpoint.pkl"
                )
                self.save_agent(self.dirs["models"] / "models_checkpoint.pth")

        print(
            f"Training completed! Total steps: {steps_completed}, Episodes: {episodes_completed}"
        )

    def evaluate(self, render=False):
        """Evaluate current policy"""

        # Set policies to eval
        self.agent.network_old.eval()

        with torch.no_grad():

            obs = self.wrapped_env.reset()

            episode_rew = 0

            while True:

                global_state = np.concatenate(obs)

                actions, _, _ = self.agent.get_actions(
                    obs, global_state, deterministic=True
                )

                obs, global_reward, local_rewards, terminated, truncated, info = (
                    self.wrapped_env.step(actions)
                )

                episode_rew += local_rewards[0] + global_reward

                if render:
                    self.wrapped_env.env.render()

                if terminated.all() or truncated.all():
                    break

        # Set policies to train
        self.agent.network_old.train()

        return episode_rew

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
