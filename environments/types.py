from enum import StrEnum
from dataclasses import dataclass


@dataclass
class EnvironmentParams:
    environment: str = None
    n_envs: int = 1
    n_agents: int = 1
    n_objects: int = 3
    reward_mode: str = "dense"
    env_variant: str = None


class EnvironmentEnum(StrEnum):
    MPE_SPREAD = "mpe_spread"
    MPE_SIMPLE = "mpe_simple"
    BOX2D_SALP = "box2d_salp"
    MULTI_BOX = "multi_box_push"
    # MuJoCo-MJX port of multi_box_push (functional JAX API, mappo_jax only)
    MULTI_BOX_MJX = "multi_box_push_mjx"
    PUSH_BOX = "push_box"
    SCATTER = "scatter"
    RENDEZVOUZ = "rendezvouz"
    CONTACT = "contact"
    SMACV2 = "smacv2"
    SMACLITE = "smaclite"
    # Hierarchical macro-action controller over frozen low-level skills
    HRL_SKILL = "hrl_skill"
    # JAXMARL
    SMAX = "SMAX"
