from learning.environments.types import EnvironmentParams
from learning.algorithms.runner import Runner
from learning.algorithms.ppo.types import Experiment
from learning.algorithms.ppo.trainer import PPOTrainer
from learning.algorithms.ppo.view import view
from learning.algorithms.ppo.evaluate import evaluate

from pathlib import Path
import torch


class PPO_Runner(Runner):
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

        self.trainer = PPOTrainer(
            self.exp_config,
            self.env_config,
            self.device,
            self.trial_id,
            self.dirs,
            self.checkpoint,
        )

    def train(self):
        self.trainer.train()

    def view(self):
        view(self.exp_config, self.env_config, self.device, self.dirs)

    def evaluate(self):
        evaluate(
            self.exp_config,
            self.env_config,
            self.device,
            self.trial_id,
            self.dirs,
        )
