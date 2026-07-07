from pathlib import Path

from algorithms.types import AlgorithmEnum


def _dispatch(
    algorithm: str,
    exp_dict: dict,
    env_config: dict,
    batch_dir: Path,
    results_dir: Path,
    trial_id: str,
    view: bool = False,
    checkpoint: bool = False,
    evaluate: bool = False,
):
    """Build the per-algorithm Experiment + Runner and run it.

    The single dispatch tail behind the Hydra entry point (`train.py`).
    `batch_dir` is only used by the runners for `combined_affinities` checkpoint
    resolution (`batch_dir.parents[1]/results`); `results_dir` is the runner's
    `trials_dir` (`results/<batch>/<name>`), under which it writes
    `<trial_id>/{logs,models,videos}`.
    """

    match (algorithm):

        case AlgorithmEnum.IPPO:
            from algorithms.ippo.run import IPPO_Runner
            from algorithms.ippo.types import Experiment as IPPO_Experiment

            exp_config = IPPO_Experiment(**exp_dict)
            runner = IPPO_Runner(
                exp_config.device,
                batch_dir,
                results_dir,
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
                results_dir,
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

        case AlgorithmEnum.MAPPO_VANILLA:
            from algorithms.mappo_vanilla.run import MAPPO_Vanilla_Runner
            from algorithms.mappo_vanilla.types import Experiment as MAPPO_Experiment

            exp_config = MAPPO_Experiment(**exp_dict)
            runner = MAPPO_Runner(
                exp_config.device,
                batch_dir,
                results_dir,
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
                results_dir,
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
                results_dir,
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

        case AlgorithmEnum.DCG_MACRO:
            # DCG over macro-actions: identical DCG core (reused from
            # algorithms.dcg), but the env group wraps a continuous box2d env in
            # HierarchicalSkillEnv so DCG sees a discrete skill-selection action.
            from algorithms.dcg_macro.run import DCG_Runner
            from algorithms.dcg.types import Experiment as DCG_Experiment

            exp_config = DCG_Experiment(**exp_dict)
            runner = DCG_Runner(
                exp_config.device,
                batch_dir,
                results_dir,
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
