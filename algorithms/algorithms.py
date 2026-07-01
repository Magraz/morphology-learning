import yaml
from pathlib import Path

from algorithms.types import AlgorithmEnum


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

    # Load batch config. The env config is nested under an `env:` block and the
    # batch-wide `random_seeds` live at the top level.
    batch_file = batch_dir / "_batch.yaml"

    with open(batch_file, "r") as file:
        batch_config = yaml.safe_load(file)

    # A batch is tied to the algorithm/environment it was built for, so it may
    # declare them as defaults. Explicit CLI values still win when provided.
    algorithm = algorithm or batch_config.get("algorithm", "")
    environment = environment or batch_config.get("environment", "")

    env_config = batch_config.get("env", {})
    random_seeds = batch_config.get("random_seeds")

    env_config["environment"] = environment

    # Load experiment config
    exp_file = batch_dir / f"{experiment_name}.yaml"

    with open(exp_file, "r") as file:
        exp_dict = yaml.safe_load(file)

    # Random seeds are shared across the whole batch, so they live in the batch
    # config. Inject them into each experiment's params (overriding any
    # per-experiment value) so the runners can keep reading params.random_seeds.
    if random_seeds is not None:
        exp_dict.setdefault("params", {})["random_seeds"] = random_seeds

    match (algorithm):

        case AlgorithmEnum.IPPO:
            from algorithms.ippo.run import IPPO_Runner
            from algorithms.ippo.types import Experiment as IPPO_Experiment

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
            from algorithms.mappo.run import MAPPO_Runner
            from algorithms.mappo.types import Experiment as MAPPO_Experiment

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

        case AlgorithmEnum.MAPPO_JAX:
            from algorithms.mappo_jax.run import MAPPO_JAX_Runner
            from algorithms.mappo_jax.types import Experiment as MAPPO_JAX_Experiment

            exp_config = MAPPO_JAX_Experiment(**exp_dict)
            runner = MAPPO_JAX_Runner(
                exp_config.device,
                batch_dir,
                (Path(batch_dir).parents[1] / "results" / batch_name / experiment_name),
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

        case AlgorithmEnum.DCG:
            from algorithms.dcg.run import DCG_Runner
            from algorithms.dcg.types import Experiment as DCG_Experiment

            exp_config = DCG_Experiment(**exp_dict)
            runner = DCG_Runner(
                exp_config.device,
                batch_dir,
                (Path(batch_dir).parents[1] / "results" / batch_name / experiment_name),
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

    if view:
        runner.view()
    elif evaluate:
        runner.evaluate()
    else:
        runner.train()
