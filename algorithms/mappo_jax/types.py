from dataclasses import dataclass
from typing import NamedTuple

import chex


@dataclass
class MAPPOConfig:
    """Config for JAX MAPPO. Plain dataclass — loadable from YAML like Params."""

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


@dataclass
class Params:
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
    std_coef: float = 0.0
    grad_clip: float = 0.5


@dataclass
class Experiment:
    device: str
    model: str
    params: Params


class Transition(NamedTuple):
    """Single timestep of rollout data across all envs.

    When accumulated via jax.lax.scan the leading dim becomes n_steps.
    """

    obs: chex.Array  # (n_envs, n_agents, obs_dim)
    global_state: chex.Array  # (n_envs, global_state_dim)
    action: chex.Array  # (n_envs, n_agents)
    reward: chex.Array  # (n_envs, n_agents)
    done: chex.Array  # (n_envs)
    log_prob: chex.Array  # (n_envs, n_agents)
    value: chex.Array  # (n_envs,)
    action_mask: chex.Array  # (n_envs, n_agents, n_actions)
