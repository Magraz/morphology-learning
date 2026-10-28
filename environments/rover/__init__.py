"""Pure-JAX 2D rover domain with coupled points of interest (POIs).

Algorithm-independent: functional ``reset``/``step`` usable by evolutionary
algorithms and RL alike. See ``base.BaseRoverEnv`` for the dynamics,
``components`` for the pluggable interfaces, and ``config`` to build a variant
from its Hydra env group (``conf/env/rover_*.yaml``).
"""

from environments.rover.base import BaseRoverEnv, wrap_angle
from environments.rover.components import (
    ObservationModel,
    POIModel,
    RewardModel,
    RewardOutput,
)
from environments.rover.observations import SectorDensityObservation
from environments.rover.pois import StaticPOIs
from environments.rover.rewards import CouplingDepletionReward
from environments.rover.state import POIState, RoverEnvState, RoverState

__all__ = [
    "BaseRoverEnv",
    "CouplingDepletionReward",
    "ObservationModel",
    "POIModel",
    "POIState",
    "RewardModel",
    "RewardOutput",
    "RoverEnvState",
    "RoverState",
    "SectorDensityObservation",
    "StaticPOIs",
    "wrap_angle",
]
