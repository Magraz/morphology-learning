from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio

from algorithms.runner import Runner
from algorithms.utils import set_global_seeds

from algorithms.mappo_vanilla.types import Experiment, MAPPO_Params, Model_Params
from algorithms.mappo_vanilla.vec_trainer import VecMAPPOTrainer


class MAPPO_Vanilla_Runner(Runner):
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        checkpoint: bool,
        exp_config: Experiment,
        env_config: dict,
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
        set_global_seeds(random_seed)

        # Device configuration
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.exp_config.device}")

        self.trainer = VecMAPPOTrainer(
            self.params,
            self.dirs,
            self.device,
            env_params=self.env_config,
            model_params=self.model_params,
        )

    def train(self):

        # Resume from checkpoint if requested and checkpoint exists
        checkpoint_path = self.dirs["models"] / "models_checkpoint.pth"
        stats_checkpoint_path = self.dirs["logs"] / "training_stats_checkpoint.pkl"

        checkpoint_loaded = False
        if (
            self.checkpoint
            and checkpoint_path.exists()
            and stats_checkpoint_path.exists()
        ):
            self.trainer.load_agent(checkpoint_path, restore_rng=True)
            self.trainer.load_checkpoint_progress(stats_checkpoint_path)
            checkpoint_loaded = True

        self.trainer.train(
            total_steps=self.params.n_total_steps,
            batch_size=self.params.n_steps * self.env_config.get("n_envs"),
            minibatches=self.params.n_minibatches,
            epochs=self.params.n_epochs,
            checkpoint=checkpoint_loaded,
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
            rewards, frames = self.trainer.render(capture_video=True)

            print(f"REWARD: {rewards[-1]:.4f}")

            if rewards.shape[0] > 0:
                steps = np.arange(len(rewards))
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(steps, rewards)
                ax.set_ylabel("Reward")
                ax.set_xlabel("Step")
                ax.set_title(f"Episode {episode} — Reward")

                plt.tight_layout()
                fig_path = self.dirs["logs"] / f"reward_episode_{episode}.png"
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Plot saved to {fig_path}")

                # Save video of the episode
                if frames:
                    video_path = self.dirs["logs"] / f"episode_{episode}.mp4"
                    imageio.mimwrite(video_path, frames, fps=30, macro_block_size=1)
                    print(f"Video saved to {video_path}")

    def evaluate(self):
        pass
