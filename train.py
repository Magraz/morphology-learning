"""Hydra entry point for experiment training.

Composes a run from the `conf/` groups (algorithm × env × model × seeds) and
hands the resolved config to the shared `_dispatch` in `algorithms.algorithms`.
This is the sole entry point; the legacy `run_trial.py` yaml path was retired.

    uv run python train.py env=multi_box_push_9a_3o model=hgnn_mix trial_id=0
    uv run python train.py -m model=mlp_shared,gnn_critic trial_id=0,1,2

Output paths are preserved: results land in
`experiments/results/<env-choice>/<model-choice>/<trial_id>/`, matching the old
`results/<batch>/<name>/<trial_id>` layout so existing checkpoints resolve.
"""

import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def _usable_cores() -> int:
    """Cores this process may actually run on.

    Uses the CPU affinity mask (respects `taskset`, cgroup/container quotas and
    SLURM `--cpus-per-task`) so autoscaling matches the real budget on HPC; the
    affinity call is Linux-only, so fall back to the machine core count.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# `${cores:}` -> usable cores; `${envs_per_job:<n_jobs>}` splits that budget
# across the concurrent sweep jobs (floored at 1). Registered at import so they
# resolve during Hydra composition, in both single-run and multirun. `replace`
# keeps re-imports (forkserver/loky re-import the entry module) from raising.
OmegaConf.register_new_resolver("cores", _usable_cores, replace=True)
OmegaConf.register_new_resolver(
    "envs_per_job",
    lambda n_jobs: max(1, _usable_cores() // max(1, int(n_jobs))),
    replace=True,
)


def _build_dispatch_args(cfg, choices):
    """Resolve a composed config into the `_dispatch` argument dict.

    `choices` is the Hydra group-choice mapping (`{"env": ..., "model": ...}`),
    used to preserve the `results/<batch>/<name>` output layout. Passed in
    explicitly (rather than read from the HydraConfig singleton) so the
    equivalence harness can call this outside a `@hydra.main` job.

    Factored out (and import-light) so the harness can call it without importing
    torch-heavy runner modules.
    """
    c = OmegaConf.to_container(
        cfg, resolve=True
    )  # resolves ${hyperedges.*}; null -> None
    batch, name = choices["env"], choices["model"]

    algorithm = c["algorithm"]

    env_config = dict(c["env"])

    exp_dict = {"device": c["device"], "params": c["params"]}
    if algorithm in ("ippo", "mappo_jax"):
        # IPPO / MAPPO_JAX take `model` as a bare string, not a model_params dict.
        exp_dict["model"] = c.get("model_name", "")
    else:
        exp_dict["model_params"] = c["model_params"]

    # batch_dir keeps the legacy shape (experiments/yamls/<batch>) so runners can
    # resolve combined_affinities checkpoints via batch_dir.parents[1]/results.
    batch_dir = Path("experiments") / "yamls" / batch
    results_dir = Path("experiments") / "results" / batch / name

    return dict(
        algorithm=algorithm,
        exp_dict=exp_dict,
        env_config=env_config,
        batch_dir=batch_dir,
        results_dir=results_dir,
        trial_id=str(c["trial_id"]),
        view=c["view"],
        checkpoint=c["checkpoint"],
        evaluate=c["evaluate"],
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    from algorithms.algorithms import _dispatch

    choices = HydraConfig.get().runtime.choices
    _dispatch(**_build_dispatch_args(cfg, choices))


if __name__ == "__main__":
    main()
