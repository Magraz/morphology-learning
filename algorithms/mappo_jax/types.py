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
    # Rollout length per parallel env; the per-update batch is
    # n_steps * env.n_envs env-steps, so it scales with parallelism.
    n_steps: int
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
    # True when the env emits a per-agent reward (reward_mode="difference_rewards").
    # Switches the critic to a per-agent value head and runs GAE on the agent axis;
    # False keeps the exact scalar-team-reward path (mappo_vanilla parity).
    per_agent_rewards: bool = False


class Transition(NamedTuple):
    """Single timestep of rollout data across all envs.

    When accumulated via jax.lax.scan the leading dim becomes n_steps.

    `reward` is what the learner optimizes: the scalar team reward, or a
    per-agent vector under `per_agent_rewards`. `team_reward` is always the
    scalar team reward (`info["task_reward"]`) and is used only for logging, so
    reported returns stay comparable across reward modes.
    """

    obs: jax.Array  # (n_envs, n_agents, obs_dim)
    global_state: jax.Array  # (n_envs, n_agents * obs_dim)
    action: jax.Array  # (n_envs, n_agents, action_dim)
    reward: jax.Array  # (n_envs,) team | (n_envs, n_agents) per-agent
    done: jax.Array  # (n_envs,) terminated | truncated
    log_prob: jax.Array  # (n_envs, n_agents)
    value: jax.Array  # (n_envs,) | (n_envs, n_agents)
    team_reward: jax.Array  # (n_envs,) scalar team reward — logging only
    # (n_envs, n_agents) 1.0/0.0 — did this agent contribute a real decision this
    # step? All 1.0 for ordinary envs (no change); 0.0 for agents that were offline
    # under SyncMacroMJX's staggered-starts mode, or dead under SMAX, whose transition
    # is masked out of the PPO loss. All-ones keeps the loss byte-identical.
    active_mask: jax.Array
    # (n_envs, n_agents, action_dim) 1.0/0.0 legal-action mask, for discrete envs that
    # expose `avail_actions` (SMAX). For every other env this is a **scalar placeholder**
    # and no masking happens: storing a real all-ones array would cost a
    # (n_steps, n_envs, n_agents, action_dim) buffer for nothing. `ppo_update` switches
    # on `action_mask.ndim == 4`, the same static-shape idiom as `reward.ndim == 3`.
    #
    # The stored mask MUST be the one used at sampling time — the PPO ratio is only
    # valid if `evaluate_action` sees the same masked distribution `sample_action` did.
    action_mask: jax.Array
