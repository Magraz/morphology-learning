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
            batch_size=self.params.batch_size,
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
            rewards, entropy_logs = self.trainer.render()

            print(f"REWARD: {rewards[-1]:.4f}")

            if rewards.shape[0] > 0:
                steps = np.arange(len(rewards))
                plot_types = {
                    k: v
                    for k, v in entropy_logs.items()
                    if k != "predicted_per_agent" and v is not None
                }
                # 1 row for reward + 2 rows per hyperedge type (S_e and S_norm)
                n_rows = 1 + 2 * len(plot_types)
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
                    if htype == "predicted_per_agent" or ent is None:
                        continue
                    is_soft = htype.startswith("soft_")
                    prefix = r"$\tilde{S}_e$" if is_soft else "$S_e$"
                    prefix_norm = (
                        r"$\tilde{S}_{\mathrm{norm}}$"
                        if is_soft
                        else r"$S_{\mathrm{norm}}$"
                    )
                    label = htype.removeprefix("soft_")

                    axes[row].plot(steps, ent[:, 0])
                    axes[row].set_ylabel(f"{prefix} ({label})")
                    row += 1

                    axes[row].plot(steps, ent[:, 1])
                    axes[row].set_ylabel(f"{prefix_norm} ({label})")
                    row += 1

                axes[-1].set_xlabel("Step")

                plt.tight_layout()
                fig_path = self.dirs["logs"] / f"entropy_episode_{episode}.png"
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Plot saved to {fig_path}")

                # Plot predicted vs actual entropy and prediction error
                # predicted_per_agent shape: (n_steps, n_agents, n_types)
                pred_per_agent = entropy_logs.get("predicted_per_agent")
                if pred_per_agent is not None:
                    type_names = ["proximity", "object"]
                    n_agents = pred_per_agent.shape[1]
                    n_types = pred_per_agent.shape[2]
                    pred_mean = pred_per_agent.mean(axis=1)  # (n_steps, n_types)
                    fig2, axes2 = plt.subplots(
                        2, n_types, figsize=(6 * n_types, 6), sharex=True
                    )
                    if n_types == 1:
                        axes2 = axes2[:, np.newaxis]

                    for t in range(n_types):
                        name = type_names[t] if t < len(type_names) else f"type_{t}"
                        # Predictor targets S_soft_norm (index 1)
                        actual = entropy_logs[f"soft_{name}"][:, 1]

                        # Per-agent predictions (thin, transparent)
                        for a in range(n_agents):
                            axes2[0, t].plot(
                                steps,
                                pred_per_agent[:, a, t],
                                alpha=0.3,
                                lw=0.8,
                                color="tab:orange",
                                label=f"Agents" if a == 0 else None,
                            )
                        # Mean prediction and actual
                        axes2[0, t].plot(
                            steps,
                            pred_mean[:, t],
                            lw=2,
                            color="tab:orange",
                            label="Predicted (mean)",
                        )
                        axes2[0, t].plot(
                            steps, actual, lw=2, color="tab:blue", label="Actual"
                        )
                        axes2[0, t].set_ylabel(r"$\tilde{S}_{\mathrm{norm}}$")
                        axes2[0, t].set_title(f"{name} — Predicted vs Actual")
                        axes2[0, t].legend()

                        error = actual - pred_mean[:, t]
                        axes2[1, t].plot(steps, error, color="tab:red")
                        axes2[1, t].axhline(0, color="grey", ls="--", lw=0.8)
                        axes2[1, t].set_ylabel("Error (actual - mean predicted)")
                        axes2[1, t].set_xlabel("Step")
                        axes2[1, t].set_title(f"{name} — Prediction Error")

                    plt.tight_layout()
                    fig2_path = (
                        self.dirs["logs"] / f"entropy_pred_episode_{episode}.png"
                    )
                    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
                    plt.close(fig2)
                    print(f"Prediction plot saved to {fig2_path}")

    def evaluate(self):
        pass
