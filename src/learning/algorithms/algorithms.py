import os
import yaml
import torch
from pathlib import Path

from learning.algorithms.ccea.train import CCEA_Trainer
from learning.algorithms.ccea.types import Experiment as CCEA_Experiment

from learning.algorithms.ppo.run import PPO_Runner
from learning.algorithms.ppo.types import Experiment as PPO_Experiment

from learning.algorithms.td3.train import TD3_Trainer

from learning.algorithms.ippo.run import IPPO_Runner
from learning.algorithms.ippo.types import Experiment as IPPO_Experiment

from learning.algorithms.mappo.run import MAPPO_Runner
from learning.algorithms.mappo.types import Experiment as MAPPO_Experiment

# from learning.algorithms.manual.control import ManualControl

from learning.algorithms.types import AlgorithmEnum

from learning.environments.types import EnvironmentEnum, EnvironmentParams
from learning.environments.rover.types import RoverEnvironmentParams
from learning.environments.salp_navigate.types import SalpNavigateEnvironmentParams
from learning.environments.salp_passage.types import SalpPassageEnvironmentParams


def run_algorithm(
    batch_dir: Path,
    batch_name: str,
    experiment_name: str,
    algorithm: str,
    environment: str,
    trial_id: str,
    view: bool = False,
    checkpoint: bool = False,
    evaluate: bool = False,
):

    # Load environment config
    env_file = batch_dir / "_env.yaml"

    with open(env_file, "r") as file:
        env_dict = yaml.safe_load(file)

    match (environment):
        case EnvironmentEnum.VMAS_ROVER:
            env_config = RoverEnvironmentParams(**env_dict)

        case EnvironmentEnum.VMAS_SALP_NAVIGATE:
            env_config = SalpNavigateEnvironmentParams(**env_dict)

        case EnvironmentEnum.VMAS_SALP_PASSAGE:
            env_config = SalpPassageEnvironmentParams(**env_dict)

        case EnvironmentEnum.VMAS_BALANCE | EnvironmentEnum.VMAS_BUZZ_WIRE:
            env_config = EnvironmentParams(**env_dict)

        case (
            EnvironmentEnum.BOX2D_SALP
            | EnvironmentEnum.MPE_SPREAD
            | EnvironmentEnum.MPE_SIMPLE
        ):
            env_config = EnvironmentParams(**env_dict)

    env_config.environment = environment

    # Load experiment config
    exp_file = batch_dir / f"{experiment_name}.yaml"

    with open(exp_file, "r") as file:
        exp_dict = yaml.unsafe_load(file)

    match (algorithm):

        case AlgorithmEnum.CCEA:
            exp_config = CCEA_Experiment(**exp_dict)
            runner = CCEA_Trainer(
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                trial_name=Path(exp_file).stem,
                video_name=f"{experiment_name}_{trial_id}",
            )

        case AlgorithmEnum.IPPO:
            exp_config = IPPO_Experiment(**exp_dict)
            runner = IPPO_Runner(
                exp_config.device,
                batch_dir,
                (Path(batch_dir).parents[1] / "results" / batch_name / experiment_name),
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

        case AlgorithmEnum.MAPPO:
            exp_config = MAPPO_Experiment(**exp_dict)
            runner = MAPPO_Runner(
                exp_config.device,
                batch_dir,
                (Path(batch_dir).parents[1] / "results" / batch_name / experiment_name),
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

        case AlgorithmEnum.PPO:
            exp_config = PPO_Experiment(**exp_dict)
            runner = PPO_Runner(
                exp_config.device,
                batch_dir,
                (Path(batch_dir).parents[1] / "results" / batch_name / experiment_name),
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

        case AlgorithmEnum.TD3:
            exp_config = None
            runner = TD3_Trainer(
                device="cpu",
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                trial_name=Path(exp_file).stem,
                video_name=f"{experiment_name}_{trial_id}",
            )

        # case AlgorithmEnum.NONE:
        #     exp_config = None

        #     runner = ManualControl(
        #         device="cpu",
        #         batch_dir=batch_dir,
        #         trials_dir=Path(batch_dir).parents[1]
        #         / "results"
        #         / batch_name
        #         / experiment_name,
        #         trial_id=trial_id,
        #         trial_name=Path(exp_file).stem,
        #         video_name=f"{experiment_name}_{trial_id}",
        #     )

    if view:
        runner.view()
    elif evaluate:
        runner.evaluate()
    else:
        runner.train()
