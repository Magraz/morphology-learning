import os
import yaml
import torch
from pathlib import Path

from algorithms.ippo.run import IPPO_Runner
from algorithms.ippo.types import Experiment as IPPO_Experiment

from algorithms.mappo.run import MAPPO_Runner
from algorithms.mappo.types import Experiment as MAPPO_Experiment

# from algorithms.manual.control import ManualControl

from algorithms.types import AlgorithmEnum

from environments.types import EnvironmentEnum, EnvironmentParams


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

        case (
            EnvironmentEnum.BOX2D_SALP
            | EnvironmentEnum.MULTI_BOX
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
