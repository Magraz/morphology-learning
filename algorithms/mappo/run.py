from environments.types import EnvironmentParams, EnvironmentEnum
from algorithms.mappo.types import Experiment, MAPPO_Params, Model_Params
from algorithms.runner import Runner
from algorithms.create_env import get_state_and_action_dims
from pathlib import Path

from algorithms.mappo.vec_trainer import VecMAPPOTrainer

import torch
import numpy as np
import random
import matplotlib.pyplot as plt


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA
    # torch.backends.cudnn.deterministic = True  # Make CUDA deterministic
    # torch.backends.cudnn.benchmark = False  # Disable or enable CUDA benchmarking


class MAPPO_Runner(Runner):
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
        self.params = MAPPO_Params(**self.exp_config.params)
        self.model_params = Model_Params(**self.exp_config.model_params)
        # Set seeds
        random_seed = self.params.random_seeds[0]

        if self.trial_id.isdigit():
            random_seed = self.params.random_seeds[int(self.trial_id)]

        # Set all random seeds for reproducibility
        set_seeds(random_seed)

        # Device configuration
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.exp_config.device}")

        self.trainer = VecMAPPOTrainer(
            self.env_config.environment,
            self.env_config.n_agents,
            self.params,
            self.dirs,
            self.device,
            n_parallel_envs=self.env_config.n_envs,
            env_variant=self.env_config.env_variant,
            critic_type=self.model_params.critic_type,
            n_hyperedge_types=self.model_params.n_hyperedge_types,
        )

    def train(self):
        resume_steps = 0
        resume_episodes = 0

        # Resume from checkpoint if requested and checkpoint exists
        checkpoint_path = self.dirs["models"] / "models_checkpoint.pth"
        if self.checkpoint and checkpoint_path.exists():
            resume_steps, resume_episodes = self.trainer.load_agent(
                checkpoint_path, restore_rng=True
            )

        self.trainer.train(
            total_steps=self.params.n_total_steps,
            batch_size=self.params.batch_size,
            minibatches=self.params.n_minibatches,
            epochs=self.params.n_epochs,
            resume_steps=resume_steps,
            resume_episodes=resume_episodes,
        )

        self.trainer.save_training_stats(
            self.dirs["logs"] / "training_stats_finished.pkl"
        )

        # Save trained agents
        self.trainer.save_agent(self.dirs["models"] / "models_finished.pth")

    def view(self):

        self.trainer.load_agent(self.dirs["models"] / "models_checkpoint.pth")

        # Test trained agents with rendering
        print("\nTesting trained agents...")
        for episode in range(10):
            rewards, entropy_logs = self.trainer.render()

            print(f"REWARD: {rewards[-1]:.4f}")

            if rewards.shape[0] > 0:
                steps = np.arange(len(rewards))
                n_types = len(entropy_logs)
                # 1 row for reward + 2 rows per hyperedge type (S_e and S_norm)
                n_rows = 1 + 2 * n_types
                fig, axes = plt.subplots(
                    n_rows, 1, figsize=(10, 3 * n_rows), sharex=True
                )

                axes[0].plot(steps, rewards)
                axes[0].set_ylabel("Reward")
                axes[0].set_title(
                    f"Episode {episode} — Reward & Hyperedge Structural Entropy"
                )

                row = 1
                for htype, ent in entropy_logs.items():
                    axes[row].plot(steps, ent[:, 0])
                    axes[row].set_ylabel(f"$S_e$ ({htype})")
                    row += 1

                    axes[row].plot(steps, ent[:, 1])
                    axes[row].set_ylabel(f"$S_{{\\mathrm{{norm}}}}$ ({htype})")
                    row += 1

                axes[-1].set_xlabel("Step")

                plt.tight_layout()
                fig_path = self.dirs["logs"] / f"entropy_episode_{episode}.png"
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Plot saved to {fig_path}")

    def evaluate(self):
        pass
