import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from algorithms.ippo.ippo import PPOAgent
import pickle  # Add this import at the top of the file

from environments.types import EnvironmentEnum
from algorithms.ippo.types import Params
from algorithms.env_wrapper import EnvWrapper


class IPPOTrainer:
    def __init__(
        self,
        env,
        env_name,
        n_agents,
        state_dim,
        action_dim,
        params: Params,
        dirs=None,
        device="cpu",
    ):
        self.device = device
        self.dirs = dirs
        self.n_agents = n_agents

        # Create environment
        self.wrapped_env = EnvWrapper(env=env, env_name=env_name, n_agents=n_agents)

        self.param_sharing = params.parameter_sharing

        self.agents = []

        if self.param_sharing:
            shared_agent = PPOAgent(
                env_name,
                state_dim,
                action_dim,
                params,
                device,
            )

        for _ in range(self.n_agents):

            agent = PPOAgent(
                env_name,
                state_dim,
                action_dim,
                params,
                device,
            )

            if self.param_sharing:
                agent.policy = shared_agent.policy
                agent.policy_old = shared_agent.policy_old
                agent.optimizer = shared_agent.optimizer

            self.agents.append(agent)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = defaultdict(list)

    def take_actions(self, obs, deterministic=False):
        # Get actions from all agents
        actions = []
        log_probs = []
        values = []

        for i, agent in enumerate(self.agents):

            with torch.no_grad():
                action, log_prob, value = agent.get_action(obs[i], deterministic)

            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

        return np.array(actions), log_probs, values

    def collect_trajectory(self, max_steps):

        obs = self.wrapped_env.reset()

        total_step_count = 0
        current_episode_steps = 0
        steps_per_episode = []
        episode_count = 0

        for step in range(max_steps):

            actions, log_probs, values = self.take_actions(obs)

            # Step environment
            next_obs, global_reward, local_rewards, terminated, truncated, info = (
                self.wrapped_env.step(actions)
            )

            # Store transitions for all agents
            for i, agent in enumerate(self.agents):
                agent.store_transition(
                    state=obs[i],
                    action=actions[i],
                    reward=local_rewards[i] + global_reward,
                    log_prob=log_probs[i],
                    value=values[i],
                    done=terminated[i] or truncated[i],
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

        # Get final values for advantage computation
        final_values = []
        for i, agent in enumerate(self.agents):
            with torch.no_grad():
                _, _, final_value = agent.get_action(obs[i])
            final_values.append(final_value)

        return (
            total_step_count,
            episode_count,
            steps_per_episode,
            final_values,
        )

    def train(self, total_steps, batch_size, minibatches, epochs, log_every=10000):
        """
        Train agents for a specific number of environment steps.

        Args:
            total_steps: Total number of environment steps to train for
            log_every: Log progress every X steps
        """
        print(f"Starting training for {total_steps} total environment steps...")

        # Initialize tracking variables
        steps_completed = 0
        episodes_completed = 0
        self.training_stats["total_steps"] = []
        self.training_stats["reward"] = []
        self.training_stats["episodes"] = []
        self.training_stats["steps_per_episode"] = []

        # Keep training until we reach the desired number of steps
        while steps_completed < total_steps:
            # Determine how many more steps to collect in this iteration
            steps_to_collect = min(batch_size, total_steps - steps_completed)

            # Collect trajectory for a fixed number of steps
            (
                step_count,
                episode_count,
                steps_per_episode,
                final_values,
            ) = self.collect_trajectory(max_steps=int(steps_to_collect))

            # Update all agents
            update_stats = {}

            if self.param_sharing:
                # First agent in the list will be our shared agent
                # Build a singular buffer from all agents experiences
                for i in range(1, self.n_agents):
                    # Append other agents' data to the shared agent
                    self.agents[0].states.extend(self.agents[i].states)
                    self.agents[0].actions.extend(self.agents[i].actions)
                    self.agents[0].rewards.extend(self.agents[i].rewards)
                    self.agents[0].log_probs.extend(self.agents[i].log_probs)
                    self.agents[0].values.extend(self.agents[i].values)
                    self.agents[0].dones.extend(self.agents[i].dones)

                    # Clear the other agents' buffers
                    self.agents[i].reset_buffer()

                # Do a single update with combined data
                stats = self.agents[0].update(
                    next_value=final_values[
                        0
                    ],  # Use any final value (they should be similar)
                    minibatch_size=batch_size // minibatches,
                    epochs=epochs,
                )

                for key, value in stats.items():
                    if f"agent_0_{key}" not in update_stats:
                        update_stats[f"agent_0_{key}"] = []
                    update_stats[f"agent_0_{key}"].append(value)

            else:
                # Update each agent separately with their own buffer
                for i, (agent, final_value) in enumerate(
                    zip(self.agents, final_values)
                ):

                    stats = agent.update(
                        next_value=final_value,
                        minibatch_size=batch_size // minibatches,
                        epochs=epochs,
                    )

                    # Store stats from each agent into their own key
                    for key, value in stats.items():
                        if f"agent_{i}_{key}" not in update_stats:
                            update_stats[f"agent_{i}_{key}"] = []
                        update_stats[f"agent_{i}_{key}"].append(value)

            # Update tracking variables
            steps_completed += step_count
            episodes_completed += episode_count

            # Store training statistics like loss and entropy
            for key, values in update_stats.items():
                self.training_stats[key].extend(values)

            # Evaluate policies
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
                self.save_agents(self.dirs["models"] / "models_checkpoint.pth")

        print(
            f"Training completed! Total steps: {steps_completed}, Episodes: {episodes_completed}"
        )

    def evaluate(self, render=False):

        # Set policies to eval
        for agent in self.agents:
            agent.policy_old.eval()

        with torch.no_grad():

            obs = self.wrapped_env.reset()

            episode_rew = 0

            while True:

                actions, _, _ = self.take_actions(obs, deterministic=True)

                obs, global_reward, local_rewards, terminated, truncated, info = (
                    self.wrapped_env.step(actions)
                )

                episode_rew += local_rewards[0] + global_reward

                if render:
                    self.wrapped_env.env.render()

                if terminated.all() or truncated.all():
                    break

        # Set policies to train
        for agent in self.agents:
            agent.policy_old.train()

        return episode_rew

    def save_agents(self, filepath):
        torch.save(
            {
                "agents": [agent.policy_old.state_dict() for agent in self.agents],
            },
            filepath,
        )
        print(f"Agents saved to {filepath}")

    def load_agents(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)

        for i, agent in enumerate(self.agents):
            agent.policy.load_state_dict(checkpoint["agents"][i])
            agent.policy_old.load_state_dict(checkpoint["agents"][i])

        print(f"Agents loaded from {filepath}")

    def save_training_stats(self, filepath):
        """Save the training statistics to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.training_stats, f)
        print(f"Training statistics saved to {filepath}")

    def load_training_stats(self, filepath):
        """Load the training statistics from a file."""
        with open(filepath, "rb") as f:
            self.training_stats = pickle.load(f)
        print(f"Training statistics loaded from {filepath}")
