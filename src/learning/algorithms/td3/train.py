import os
import torch


from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.td3.td3 import TD3
from learning.algorithms.td3.utils import ReplayBuffer

import numpy as np
import pickle as pkl
from pathlib import Path

from vmas.simulator.utils import save_video


class TD3_Trainer:
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: int,
        trial_name: str,
        video_name: str,
    ):
        # Directories
        self.device = device
        self.batch_dir = batch_dir
        self.trials_dir = trials_dir
        self.trial_name = trial_name
        self.trial_id = trial_id
        self.video_name = video_name
        self.trial_folder_name = "_".join(("trial", str(self.trial_id)))
        self.trial_dir = self.trials_dir / self.trial_folder_name
        self.logs_dir = self.trial_dir / "logs"
        self.models_dir = self.trial_dir / "models"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def eval_policy(self, policy, seed, env_name, eval_episodes=10):
        eval_env = create_env(
            self.batch_dir,
            1,
            device=self.device,
            env_name=env_name,
            seed=seed + 100,
        )

        avg_reward = 0.0
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                state = torch.stack(state).permute(1, 0, 2).reshape(1, -1).cpu().numpy()
                action = policy.select_action(state)
                action = action.reshape(
                    2,
                    1,
                    2,
                )
                state, reward, done, _ = eval_env.step(torch.tensor(action))
                avg_reward += reward[0].item()

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def process_state(
        self,
        n_envs: int,
        state: list,
        representation: str,
    ):
        match (representation):
            case "global":
                return state[0]
            case _:
                state = torch.stack(state).permute(1, 0, 2).reshape(n_envs, -1)
                return state

        return state

    def train(
        self,
        exp_config,
        env_config: EnvironmentParams,
    ):
        seed = 118
        env = create_env(
            self.batch_dir,
            env_config.n_envs,
            device=self.device,
            env_name=env_config.environment,
            seed=seed,
        )

        # NumPy
        np.random.seed(118)

        # PyTorch (CPU)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        action_dim = env.action_space.spaces[0].shape[0] * 2
        state_dim = env.observation_space.spaces[0].shape[0] * 2

        policy = "TD3"
        start_timesteps = 25e2
        eval_freq = 5e3
        max_timesteps = 1e6
        expl_noise = 0.1
        batch_size = 256
        discount = 0.99
        tau = 0.005
        policy_noise = 0.2
        noise_clip = 0.5
        policy_freq = 2
        save_model = True
        load_model = ""
        max_episode_steps = 500

        if not os.path.exists("./results"):
            os.makedirs("./results")

        if save_model and not os.path.exists("./models"):
            os.makedirs("./models")

        max_action = float(1.0)

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": discount,
            "tau": tau,
        }

        # Initialize policy
        if policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = policy_noise * max_action
            kwargs["noise_clip"] = noise_clip * max_action
            kwargs["policy_freq"] = policy_freq
            policy = TD3(**kwargs)

        replay_buffer = ReplayBuffer(state_dim, action_dim)

        # Evaluate untrained policy
        evaluations = [
            self.eval_policy(policy, env_name=env_config.environment, seed=seed)
        ]

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(max_timesteps)):

            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < start_timesteps:
                action = env.action_space.sample()
                action = np.stack(action)
                action = torch.from_numpy(action)
                action = action.reshape(
                    2,
                    1,
                    2,
                )
            else:
                action = (
                    policy.select_action(
                        torch.stack(state).permute(1, 0, 2).reshape(1, -1).cpu().numpy()
                    )
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                action = action.reshape(
                    2,
                    1,
                    2,
                )
                action = torch.tensor(action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(
                torch.stack(state).permute(1, 0, 2).reshape(1, -1).cpu().numpy(),
                action.permute(1, 0, 2).reshape(1, -1).cpu().numpy(),
                torch.stack(next_state).permute(1, 0, 2).reshape(1, -1).cpu().numpy(),
                reward[0].item(),
                done_bool,
            )

            state = next_state
            episode_reward += reward[0].item()

            # Train agent after collecting sufficient data
            if t >= start_timesteps:
                policy.train(replay_buffer, batch_size)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
                )
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % eval_freq == 0:
                evaluations.append(
                    self.eval_policy(policy, env_name=env_config.environment, seed=seed)
                )
                np.save(f"./results/{self.trial_name}", evaluations)
                if save_model:
                    policy.save(f"./models/{self.trial_name}")
