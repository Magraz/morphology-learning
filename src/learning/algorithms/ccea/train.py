import csv
import os
import pickle
from learning.algorithms.ccea.ccea import CooperativeCoevolutionaryAlgorithm
from learning.algorithms.ccea.types import Checkpoint, Experiment
from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from dataclasses import asdict
from pathlib import Path


class CCEA_Trainer:
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: int,
        trial_name: str,
        video_name: str,
    ):
        self.device = device
        self.batch_dir = batch_dir
        self.trials_dir = trials_dir
        self.trial_name = trial_name
        self.trial_id = trial_id
        self.video_name = video_name
        self.trial_folder_name = "_".join(("trial", str(self.trial_id)))
        self.trial_dir = self.trials_dir / self.trial_folder_name

        self.fitness_file = self.trial_dir / "fitness.csv"
        self.checkpoint_file = self.trial_dir / "checkpoint.pickle"

        self.checkpoint = Checkpoint()

    def train(
        self,
        env_config: EnvironmentParams,
        exp_config: Experiment,
    ):

        # Create directory for saving data
        self.trial_dir.mkdir(parents=True, exist_ok=True)

        if self.checkpoint_file.is_file():
            self.checkpoint = self.load_checkpoint(
                self.checkpoint_file, self.fitness_file, self.trial_dir
            )

        else:
            # Create csv file for saving evaluation fitnesses
            self.create_log_file(self.fitness_file)

        ccea = CooperativeCoevolutionaryAlgorithm(
            self.device,
            env_config,
            exp_config,
        )

        # Load checkpoint
        pop = None

        if not self.checkpoint.exists:
            # Initialize the population
            pop = ccea.toolbox.population()

        else:
            pop = self.checkpoint.population

        # Create environment
        env = create_env(
            self.batch_dir,
            n_envs=ccea.subpop_size,
            env_name=env_config.environment,
            device=self.device,
        )

        # Train
        for n_gen in range(ccea.n_gens + 1):

            # Get loading bar up to checkpoint
            if self.checkpoint.exists and n_gen <= self.checkpoint.generation:
                continue

            pop, best_team, team_fitness, avg_team_fitness = ccea.step(n_gen, pop, env)

            self.write_log_file(
                self.fitness_file, n_gen, avg_team_fitness, team_fitness
            )

            if (n_gen > 0) and (n_gen % exp_config.n_gens_between_save == 0):
                self.checkpoint.exists = True
                self.checkpoint.best_team = best_team
                self.checkpoint.generation = n_gen
                self.checkpoint.population = pop

                self.save_checkpoint(self.checkpoint)

    def create_log_file(self):
        header = ["gen", "avg_team_fitness", "best_team_fitness"]

        with open(self.fitness_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([header])

    def write_log_file(self, gen, avg_fitness, best_fitness):

        # Now save it all to the csv
        with open(self.fitness_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([gen, avg_fitness, best_fitness])

    def save_checkpoint(self, checkpoint: Checkpoint):
        # Save checkpoint
        with open(self.checkpoint_file, "wb") as handle:
            pickle.dump(
                asdict(checkpoint),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_checkpoint(
        self,
        fitness_dir: str,
        trial_dir: str,
    ):
        checkpoint = Checkpoint()

        # Load checkpoint file
        with open(self.checkpoint_file, "rb") as handle:
            checkpoint = pickle.load(handle)
            checkpoint = Checkpoint(**checkpoint)
            checkpoint_gen = checkpoint.generation
            checkpoint.exists = True

        # Set fitness csv file to checkpoint
        new_fit_path = os.path.join(trial_dir, "fitness_edit.csv")
        with open(fitness_dir, "r") as inp, open(new_fit_path, "w") as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                if row[0].isdigit():
                    gen = int(row[0])
                    if gen <= checkpoint_gen:
                        writer.writerow(row)
                else:
                    writer.writerow(row)

        # Remove old fitness file
        os.remove(fitness_dir)
        # Rename new fitness file
        os.rename(new_fit_path, fitness_dir)

        return checkpoint
