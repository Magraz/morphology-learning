from enum import StrEnum
from dataclasses import dataclass


@dataclass
class EnvironmentParams:
    environment: str = None
    n_envs: int = 1
    n_agents: int = 1
    # SMACv2 Specific
    map_name: str = None


class EnvironmentEnum(StrEnum):
    MPE_SPREAD = "mpe_spread"
    MPE_SIMPLE = "mpe_simple"
    BOX2D_SALP = "box2d_salp"
    MULTI_BOX = "multi_box_push"
    SMACV2 = "smacv2"
