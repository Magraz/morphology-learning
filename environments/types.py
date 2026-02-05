from enum import StrEnum
from dataclasses import dataclass


@dataclass
class EnvironmentParams:
    environment: str = None
    n_envs: int = 1
    n_agents: int = 1
    state_representation: str = None


class EnvironmentEnum(StrEnum):
    MPE_SPREAD = "mpe_spread"
    MPE_SIMPLE = "mpe_simple"
    BOX2D_SALP = "box2d_salp"
