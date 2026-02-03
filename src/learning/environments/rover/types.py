from dataclasses import dataclass
from learning.environments.types import EnvironmentParams


@dataclass(frozen=True)
class PositionConfig:
    spawn_rule: str
    coordinates: tuple[int]


@dataclass(frozen=True)
class RoversConfig:
    observation_radius: int
    type: int
    color: str
    position: PositionConfig


@dataclass(frozen=True)
class POIConfig:
    value: float
    coupling: int
    observation_radius: float
    type: int
    order: int
    position: PositionConfig


@dataclass
class RoverEnvironmentParams(EnvironmentParams):
    agents: list[RoversConfig] = None
    targets: list[POIConfig] = None
    use_order: bool = False
