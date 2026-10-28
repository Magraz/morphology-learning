"""Build rover envs from the Hydra ``conf/env/rover_*.yaml`` groups.

Each POI layout is one env group (a "variant"); ``rover_2poi_c2`` is the
default. ``train.py`` builds it through ``mappo_jax.run.make_env`` for the JAX
stacks (``algorithm=mappo_jax`` / ``rmappo_jax``); outside training, build it
from here::

    from environments.rover.config import load_rover_env
    env = load_rover_env("rover_2poi_c2", overrides=["env.n_agents=6"])

``make_rover_env`` rejects unknown keys: nothing else validates an ``env:``
block, and a misindented key would otherwise be silently ignored.
"""

from pathlib import Path

from environments.rover.base import BaseRoverEnv
from environments.rover.observations import SectorDensityObservation
from environments.rover.pois import StaticPOIs
from environments.rover.rewards import CouplingDepletionReward
from environments.types import EnvironmentEnum

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"
VARIANT_PREFIX = "rover_"

# ``type:`` value in a yaml block -> component class. Register new
# implementations here to make them selectable from config.
POI_MODELS = {"static": StaticPOIs}
OBSERVATIONS = {"sector_density": SectorDensityObservation}
REWARDS = {"coupling_depletion": CouplingDepletionReward}

_ENV_KEYS = {
    "environment",
    "n_agents",
    "max_steps",
    "world_size",
    "spawn_region",
    "dynamics",
    "observation",
    "reward",
    "pois",
}
# Keys of a rover `env:` group that the training runners read and the env does
# not: `n_envs` is the vmap width of the JAX collectors (`mappo_jax`,
# `rmappo_jax`). Accepted here so one env group serves both, and ignored.
_RUNNER_KEYS = {"n_envs"}
_DYNAMICS_KEYS = {"max_speed", "max_turn"}


def _build_component(registry, block, default_type, what):
    block = dict(block or {})
    kind = block.pop("type", default_type)
    if kind not in registry:
        raise ValueError(f"unknown {what} type {kind!r}; known: {sorted(registry)}")
    return registry[kind](**block)


def make_rover_env(env_cfg):
    """``BaseRoverEnv`` from a resolved ``env:`` block (a plain dict)."""
    cfg = {k: v for k, v in dict(env_cfg).items() if k not in _RUNNER_KEYS}
    unknown = set(cfg) - _ENV_KEYS
    if unknown:
        raise ValueError(
            f"unknown rover env keys {sorted(unknown)}; "
            f"allowed {sorted(_ENV_KEYS | _RUNNER_KEYS)}"
        )
    if cfg.get("environment") != EnvironmentEnum.ROVER:
        raise ValueError(
            f"environment must be {EnvironmentEnum.ROVER.value!r}, got {cfg.get('environment')!r}"
        )
    for required in ("n_agents", "pois"):
        if required not in cfg:
            raise ValueError(f"rover env config is missing {required!r}")
    dynamics = dict(cfg.get("dynamics") or {})
    unknown = set(dynamics) - _DYNAMICS_KEYS
    if unknown:
        raise ValueError(f"unknown dynamics keys {sorted(unknown)}; allowed {sorted(_DYNAMICS_KEYS)}")

    kwargs = {k: cfg[k] for k in ("world_size", "spawn_region", "max_steps") if k in cfg}
    return BaseRoverEnv(
        n_agents=cfg["n_agents"],
        pois=_build_component(POI_MODELS, cfg["pois"], "static", "pois"),
        observation=_build_component(
            OBSERVATIONS, cfg.get("observation"), "sector_density", "observation"
        ),
        reward=_build_component(REWARDS, cfg.get("reward"), "coupling_depletion", "reward"),
        **kwargs,
        **dynamics,
    )


def list_variants():
    """Names of every rover env group under ``conf/env/``."""
    return sorted(p.stem for p in (CONF_DIR / "env").glob(f"{VARIANT_PREFIX}*.yaml"))


def load_env_config(variant, overrides=()):
    """Compose ``conf/config.yaml`` with ``env=<variant>``; return the env block.

    Same composition as the offline probes (e.g.
    ``feudal_mappo_jax/goal_dependence_probe.py``), so a variant resolves here
    exactly as it would under ``train.py``. ``overrides`` are ordinary Hydra
    overrides such as ``"env.max_steps=200"``.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[f"env={variant}", *overrides])
    return OmegaConf.to_container(cfg.env, resolve=True)


def load_rover_env(variant, overrides=()):
    return make_rover_env(load_env_config(variant, overrides))
