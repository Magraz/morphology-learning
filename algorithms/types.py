from enum import StrEnum


class AlgorithmEnum(StrEnum):
    IPPO = "ippo"
    PPO = "ppo"
    MAPPO = "mappo"
    MAPPO_JAX = "mappo_jax"
    DCG = "dcg"
    TD3 = "td3"
