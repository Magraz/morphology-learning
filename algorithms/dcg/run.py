from pathlib import Path

from algorithms.dcg.trainer import DCGTrainer
from algorithms.dcg.types import DCG_Model_Params, DCG_Params, Experiment
from algorithms.mappo.utils import set_global_seeds
from algorithms.runner import Runner


class DCG_Runner(Runner):
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

        self.params = DCG_Params(**self.exp_config.params)
        self.model_params = DCG_Model_Params(**self.exp_config.model_params)

        # Per-trial seed (same convention as MAPPO_Runner).
        random_seed = self.params.random_seeds[0]
        if self.trial_id.isdigit():
            random_seed = self.params.random_seeds[int(self.trial_id)]
        set_global_seeds(random_seed)

        print(f"Using device: {self.exp_config.device}")

        self.trainer = DCGTrainer(
            self.params,
            self.dirs,
            self.device,
            env_params=self.env_config,
            model_params=self.model_params,
        )

    def train(self):
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
            checkpoint=checkpoint_loaded,
        )

        self.trainer.save_training_stats(
            self.dirs["logs"] / "training_stats_finished.pkl"
        )
        self.trainer.save_agent(self.dirs["models"] / "models_finished.pth")
        self.trainer.close_environments()

    def view(self):
        self.trainer.load_agent(self.dirs["models"] / "models_finished.pth")
        for episode in range(5):
            returns = self.trainer.render(capture_video=False)
            print(f"Episode {episode} return: {float(returns.mean()):.4f}")

    def evaluate(self):
        pass
