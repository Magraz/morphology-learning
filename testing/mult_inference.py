import pickle
import sys
import os
from pathlib import Path
import torch
import yaml

sys.path.insert(0, "./src")

from algorithms.ccea.ccea import CooperativeCoevolutionaryAlgorithm
from algorithms.types import ExperimentConfig, EnvironmentConfig
from environments.create_env import create_env
from dataclasses import asdict

batch_name = "static_spread"
experiment_name = "g_mlp"
trial_id = 0
checkpoint_path = f"./src/learning/testing/checkpoint.pickle"
batch_dir = f"./src/learning/experiments/yamls/{batch_name}"

exp_file = os.path.join(batch_dir, f"{experiment_name}.yaml")

with open(str(exp_file), "r") as file:
    exp_dict = yaml.unsafe_load(file)

env_file = os.path.join(batch_dir, "_env.yaml")

with open(str(env_file), "r") as file:
    env_dict = yaml.safe_load(file)

env_config = EnvironmentConfig(**env_dict)
exp_config = ExperimentConfig(**exp_dict)

best_team = None

with open(checkpoint_path, "rb") as handle:
    checkpoint = pickle.load(handle)
    best_team = checkpoint["best_team"]

ccea = CooperativeCoevolutionaryAlgorithm(
    batch_dir=batch_dir,
    trials_dir=Path(batch_dir).parents[1] / "results" / batch_name / experiment_name,
    trial_id=trial_id,
    trial_name=Path(exp_file).stem,
    video_name=f"{experiment_name}_{trial_id}",
    device="cuda" if torch.cuda.is_available() else "cpu",
    # Environment Data
    map_size=env_config.map_size,
    observation_size=env_config.obs_space_dim,
    action_size=env_config.action_space_dim,
    n_agents=len(env_config.agents),
    n_pois=len(env_config.targets),
    # Experiment Data
    **asdict(exp_config),
)

eval_infos = []
for _ in range(1000):
    eval_info = ccea.evaluateTeams(
        create_env(
            batch_dir=batch_dir,
            device=ccea.device,
            n_envs=1,
            n_agents=4,
            viewer_zoom=1.8,
            benchmark=False,
        ),
        [best_team],
    )
    eval_infos.append(eval_info.team_fitness)

print(eval_infos)
