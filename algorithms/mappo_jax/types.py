from dataclasses import dataclass
from typing import NamedTuple

import jax


@dataclass
class Params:
    """Mirrors ``mappo_vanilla.types.MAPPO_Params`` field-for-field so the same
    Hydra ``params`` block drives either algorithm."""

    # Training Params
    n_epochs: int
    n_total_steps: int
    n_minibatches: int
    batch_size: int
    parameter_sharing: bool
    random_seeds: list

    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    eps_clip: float = 0.2
    ent_coef: float = 0.01
    val_coef: float = 0.5
    grad_clip: float = 0.5


@dataclass
class Model_Params:
    hidden_dim: int


@dataclass
class Experiment:
    device: str
    model_params: Model_Params
    params: Params


@dataclass
class MAPPOConfig:
    """Resolved training config consumed by the jitted train functions."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    ent_coef: float = 0.01
    val_coef: float = 0.5
    grad_clip: float = 0.5
    n_epochs: int = 10
    n_minibatches: int = 4
    n_steps: int = 128
    n_envs: int = 8
    n_total_steps: int = 1_000_000
    parameter_sharing: bool = True
    hidden_dim: int = 168
    n_eval_episodes: int = 5


class Transition(NamedTuple):
    """Single timestep of rollout data across all envs.

    When accumulated via jax.lax.scan the leading dim becomes n_steps. The
    reward is the scalar team reward straight from the env (mappo_vanilla tiles
    it per agent; with the shared critic the two are equivalent, so it is kept
    env-level here and broadcast where needed).
    """

    obs: jax.Array  # (n_envs, n_agents, obs_dim)
    global_state: jax.Array  # (n_envs, n_agents * obs_dim)
    action: jax.Array  # (n_envs, n_agents, action_dim)
    reward: jax.Array  # (n_envs,) team reward
    done: jax.Array  # (n_envs,) terminated | truncated
    log_prob: jax.Array  # (n_envs, n_agents)
    value: jax.Array  # (n_envs,)
