import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from learning.environments.box2d_salp.domain import SalpChainEnv
from learning.algorithms.ippo.ippo import PPOAgent
import pickle
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import copy


class IPPOTrainer:
    def __init__(self, env_config, ppo_config, device="cpu", num_workers=None):
        self.device = device
        self.env_config = env_config
        self.ppo_config = ppo_config

        # Set number of worker processes
        self.num_workers = num_workers or min(cpu_count(), 4)  # Default to 4 or fewer

        # Create main environment for observation/action space info
        self.env = SalpChainEnv(**env_config)

        # Create independent PPO agents
        state_dim = self.env.observation_space.shape[1]
        action_dim = self.env.action_space.shape[1]

        self.agents = []
        for i in range(self.env.n_agents):
            agent = PPOAgent(
                state_dim=state_dim, action_dim=action_dim, device=device, **ppo_config
            )
            self.agents.append(agent)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = defaultdict(list)

    def collect_trajectory_worker(self, worker_args):
        """Worker function for multiprocessing"""
        worker_id, agent_states, env_config, max_steps = worker_args

        # Create environment for this worker
        env = SalpChainEnv(**env_config)

        # Create agents for this worker and load states
        state_dim = env.observation_space.shape[1]
        action_dim = env.action_space.shape[1]

        agents = []
        for i in range(env.n_agents):
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                device="cpu",  # Use CPU for workers
                **self.ppo_config,
            )
            # Load the agent state
            agent.network.load_state_dict(agent_states[i])
            agents.append(agent)

        # Collect trajectory
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0

        # Store trajectory data
        trajectory_data = {
            "states": [[] for _ in range(env.n_agents)],
            "actions": [[] for _ in range(env.n_agents)],
            "rewards": [],
            "log_probs": [[] for _ in range(env.n_agents)],
            "values": [[] for _ in range(env.n_agents)],
            "dones": [],
        }

        for step in range(max_steps):
            # Get actions from all agents
            actions = []
            log_probs = []
            values = []

            for i, agent in enumerate(agents):
                action, log_prob, value = agent.get_action(obs[i])
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

                # Store data
                trajectory_data["states"][i].append(obs[i].copy())
                trajectory_data["actions"][i].append(action.copy())
                trajectory_data["log_probs"][i].append(log_prob)
                trajectory_data["values"][i].append(value)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(np.array(actions))

            # Store shared data
            trajectory_data["rewards"].append(reward)
            trajectory_data["dones"].append(terminated or truncated)

            obs = next_obs
            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        # Get final values
        final_values = []
        for i, agent in enumerate(agents):
            _, _, final_value = agent.get_action(obs[i])
            final_values.append(final_value)

        # Clean up environment
        env.close()

        return {
            "trajectory_data": trajectory_data,
            "episode_reward": episode_reward,
            "step_count": step_count,
            "final_values": final_values,
            "worker_id": worker_id,
        }

    def collect_trajectories_parallel(self, num_trajectories=4, max_steps=1000):
        """Collect multiple trajectories in parallel"""

        # Get current agent states for workers
        agent_states = []
        for agent in self.agents:
            agent_states.append(copy.deepcopy(agent.network.state_dict()))

        # Prepare worker arguments
        worker_args = []
        for i in range(num_trajectories):
            worker_args.append((i, agent_states, self.env_config, max_steps))

        # Collect trajectories in parallel
        if self.num_workers == 1:
            # Sequential execution for debugging
            results = [self.collect_trajectory_worker(args) for args in worker_args]
        else:
            # Parallel execution
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(self.collect_trajectory_worker, worker_args)

        return results

    def collect_trajectory(self, max_steps=1000):
        """Original single trajectory collection (kept for compatibility)"""
        obs, _ = self.env.reset()
        episode_reward = 0
        step_count = 0

        for step in range(max_steps):
            # Get actions from all agents
            actions = []
            log_probs = []
            values = []

            for i, agent in enumerate(self.agents):
                action, log_prob, value = agent.get_action(obs[i])
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(
                np.array(actions)
            )

            # Store transitions for all agents
            for i, agent in enumerate(self.agents):
                agent.store_transition(
                    state=obs[i],
                    action=actions[i],
                    reward=reward,
                    log_prob=log_probs[i],
                    value=values[i],
                    done=terminated or truncated,
                )

            obs = next_obs
            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        # Get final values for advantage computation
        final_values = []
        for i, agent in enumerate(self.agents):
            _, _, final_value = agent.get_action(obs[i])
            final_values.append(final_value)

        return episode_reward, step_count, final_values

    def train_episode_parallel(self, num_parallel_trajectories=4):
        """Train using parallel trajectory collection"""

        # Collect multiple trajectories in parallel
        results = self.collect_trajectories_parallel(
            num_trajectories=num_parallel_trajectories
        )

        # Process all trajectories and store in agents
        total_episode_reward = 0
        total_steps = 0

        for result in results:
            trajectory_data = result["trajectory_data"]
            episode_reward = result["episode_reward"]
            step_count = result["step_count"]
            final_values = result["final_values"]

            # Store transitions in agents
            for i, agent in enumerate(self.agents):
                states = trajectory_data["states"][i]
                actions = trajectory_data["actions"][i]
                rewards = trajectory_data["rewards"]
                log_probs = trajectory_data["log_probs"][i]
                values = trajectory_data["values"][i]
                dones = trajectory_data["dones"]

                for j in range(len(states)):
                    agent.store_transition(
                        state=states[j],
                        action=actions[j],
                        reward=rewards[j],
                        log_prob=log_probs[j],
                        value=values[j],
                        done=dones[j],
                    )

            total_episode_reward += episode_reward
            total_steps += step_count

        # Update all agents with collected data
        update_stats = {}
        for i, agent in enumerate(self.agents):
            # Use the final values from the last trajectory
            final_value = results[-1]["final_values"][i]
            stats = agent.update(next_value=final_value)
            for key, value in stats.items():
                if f"agent_{i}_{key}" not in update_stats:
                    update_stats[f"agent_{i}_{key}"] = []
                update_stats[f"agent_{i}_{key}"].append(value)

        # Store episode statistics (average across parallel trajectories)
        avg_episode_reward = total_episode_reward / len(results)
        avg_episode_length = total_steps / len(results)

        self.episode_rewards.append(avg_episode_reward)
        self.episode_lengths.append(avg_episode_length)

        # Store training statistics
        for key, values in update_stats.items():
            self.training_stats[key].extend(values)

        return avg_episode_reward, avg_episode_length

    def train(
        self,
        num_episodes=1000,
        log_every=10,
        use_parallel=True,
        num_parallel_trajectories=4,
    ):
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Using parallel collection: {use_parallel}")
        if use_parallel:
            print(f"Number of parallel trajectories: {num_parallel_trajectories}")
            print(f"Number of worker processes: {self.num_workers}")

        for episode in range(num_episodes):
            if use_parallel:
                episode_reward, episode_length = self.train_episode_parallel(
                    num_parallel_trajectories
                )
            else:
                episode_reward, episode_length = self.train_episode()

            # Logging
            if episode % log_every == 0:
                avg_reward = np.mean(self.episode_rewards[-log_every:])
                avg_length = np.mean(self.episode_lengths[-log_every:])

                print(
                    f"Episode {episode:4d} | "
                    f"Avg Reward: {avg_reward:8.2f} | "
                    f"Avg Length: {avg_length:6.1f}"
                )

        print("Training completed!")

    # ... rest of your methods remain the same ...
