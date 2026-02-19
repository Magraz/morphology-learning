from enum import StrEnum


class AlgorithmEnum(StrEnum):
    IPPO = "ippo"
    PPO = "ppo"
    MAPPO = "mappo"
    MAPPO_JAX = "mappo_jax"
    TD3 = "td3"
