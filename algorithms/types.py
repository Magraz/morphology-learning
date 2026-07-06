from enum import StrEnum


class AlgorithmEnum(StrEnum):
    IPPO = "ippo"
    PPO = "ppo"
    MAPPO = "mappo"
    MAPPO_JAX = "mappo_jax"
    DCG = "dcg"
    DCG_MACRO = "dcg_macro"
    TD3 = "td3"
