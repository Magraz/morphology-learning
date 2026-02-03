from learning.environments.types import EnvironmentParams, EnvironmentEnum
from learning.algorithms.ippo.types import Experiment, Params
from learning.algorithms.runner import Runner
from learning.algorithms.create_env import create_env
from pathlib import Path

from learning.algorithms.ippo.trainer import IPPOTrainer

import torch
import numpy as np
import random


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA
    # torch.backends.cudnn.deterministic = True  # Make CUDA deterministic
    # torch.backends.cudnn.benchmark = False  # Disable CUDA benchmarking


class IPPO_Runner(Runner):
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        checkpoint: bool,
        exp_config: Experiment,
        env_config: EnvironmentParams,
    ):
        super().__init__(device, batch_dir, trials_dir, trial_id, checkpoint)

        self.exp_config = exp_config
        self.env_config = env_config

        # Set params
        self.params = Params(**self.exp_config.params)

        # Set seeds
        random_seed = self.params.random_seeds[0]

        if self.trial_id.isdigit():
            random_seed = self.params.random_seeds[int(self.trial_id)]

        # Set all random seeds for reproducibility
        set_seeds(random_seed)

        # Device configuration
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.exp_config.device}")

        # Create environment
        self.env, state_dim, action_dim = create_env(env_config)

        # Create trainer
        self.trainer = IPPOTrainer(
            self.env,
            self.env_config.environment,
            self.env_config.n_agents,
            state_dim,
            action_dim,
            self.params,
            self.dirs,
            self.device,
        )

    def train(self):
        # Train
        self.trainer.train(
            total_steps=self.params.n_total_steps,
            batch_size=self.params.batch_size,
            minibatches=self.params.n_minibatches,
            epochs=self.params.n_epochs,
        )

        self.trainer.save_training_stats(
            self.dirs["logs"] / "training_stats_finished.pkl"
        )

        # Save trained agents
        self.trainer.save_agents(self.dirs["models"] / "models_finished.pth")

        self.trainer.env.close()

    def view(self):
        self.env, _, _ = create_env(self.env_config, render_mode="human")

        # Update the environment in the trainer
        self.trainer.wrapped_env.env = self.env

        # Save trained agents
        self.trainer.load_agents(self.dirs["models"] / "models_checkpoint.pth")

        # Test trained agents with rendering
        print("\nTesting trained agents...")
        for i in range(10):
            rew = self.trainer.evaluate(render=True)
            print(f"REWARD: {rew}")

        self.env.close()

    def evaluate(self):
        pass
