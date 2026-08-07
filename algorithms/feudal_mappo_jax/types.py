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

    # ---- FeUdal-only knobs (every yaml `params:` key must be a field here) ----
    # Goal horizon `c`: how far ahead the manager's directive is evaluated
    # (transition cosine), how many goals the worker's conditioning pools over,
    # and the dilation radius `r` of the manager's LSTM core.
    goal_horizon: int = 10
    # alpha: weight of the worker's intrinsic reward relative to the env reward.
    # 0.0 makes the whole intrinsic path inert (the goals still condition the
    # worker, but nothing rewards following them).
    intrinsic_coef: float = 0.0
    # The manager is a separate optimization problem (its own params, its own
    # value function, no importance ratio), so it gets its own lr / value coef /
    # discount. `manager_gamma` may exceed `gamma`: FuN's manager operates on a
    # longer timescale than the worker.
    manager_lr: float = 3e-4
    manager_val_coef: float = 0.5
    manager_gamma: float = 0.99
    # The transition policy gradient has NO importance-sampling correction, so
    # extra epochs are uncorrected off-policy. Keep at 1 unless you add one.
    n_manager_epochs: int = 1
    # V^M's regression is ordinary supervised fitting against FIXED targets, so
    # it carries no such constraint and is counted separately. It needs to be:
    # the worker critic gets n_epochs * n_minibatches (48 at the defaults)
    # gradient steps per update, so sharing n_manager_epochs=1 leaves V^M ~50x
    # undertrained and `manager_explained_variance` pinned negative.
    n_manager_critic_epochs: int = 8


@dataclass
class Model_Params:
    hidden_dim: int
    # Width of the manager's latent state space `s` AND of the goal `g` — in FuN
    # a goal is a *direction in the state embedding*, so they share a space and
    # this is the only width knob. It is also the manager's entire information
    # bottleneck (the recurrent core consumes `s`), so shrinking it throttles the
    # goal RNN too.
    goal_dim: int = 16
    manager_hidden_dim: int = 256
    # "mlp" (stateless) or "dilated_lstm" (FuN's dilated recurrence, radius =
    # goal_horizon). The mlp core is the default so that goal-mechanism effects
    # are attributable separately from recurrence effects.
    manager_core: str = "mlp"
    # Optional bias-free Dense on the goal before it meets the obs (FuN's `phi`).
    # None = raw concat fusion; see the degeneracy note in worker.py.
    goal_embed_dim: int | None = None


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
    # Drives the MANAGER's value head width here; the worker's head is always
    # per-agent (see below). Also runs the manager's GAE on the agent axis.
    per_agent_rewards: bool = False

    # ---- FeUdal ----
    # NOTE on the worker critic: unlike mappo_jax it is ALWAYS per-agent, because
    # the intrinsic reward is inherently per-agent ((T,E,N)) and is added to the
    # env reward. So the feudal `intrinsic_coef=0` arm is still not numerically
    # identical to mappo_jax — the flat baseline is `algorithm=mappo_jax`.
    goal_dim: int = 16
    goal_horizon: int = 10
    intrinsic_coef: float = 0.0
    manager_lr: float = 3e-4
    manager_val_coef: float = 0.5
    manager_gamma: float = 0.99
    n_manager_epochs: int = 1
    n_manager_critic_epochs: int = 8
    manager_hidden_dim: int = 256
    manager_core: str = "mlp"
    goal_embed_dim: int | None = None


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
    # under SyncMacroMJX's staggered-starts mode, whose transition is masked out of
    # the PPO loss. All-ones keeps the loss byte-identical to the unmasked path.
    active_mask: jax.Array

    # ---- FeUdal hierarchy ----
    # The manager's raw per-step directive, (n_envs, n_agents, goal_dim), unit
    # norm per agent. Stored because it is *data* for the worker's intrinsic
    # reward (which detaches both of its arguments), and because it is the oracle
    # the "recomputed goals match the rollout" seam test compares against.
    goal: jax.Array
    # The goal the worker ACTUALLY acted on: sum of the last `goal_horizon` goals
    # (FuN's w_t), (n_envs, n_agents, goal_dim). Must be stored rather than
    # recomputed — the PPO ratio is only valid if `evaluate_action` sees exactly
    # the vector `sample_action` saw, and recomputation would drift once the
    # manager's params move.
    pooled_goal: jax.Array
    # The manager's latent state `s`, (n_envs, n_agents, goal_dim). Data for the
    # intrinsic reward; recomputed differentiably inside the manager update.
    state_latent: jax.Array
    # V^M at act time, (n_envs,) | (n_envs, n_agents) — mirrors `value`.
    manager_value: jax.Array
    # The manager's own reward stream: the extrinsic reward plus a truncation
    # bootstrap taken against V^M, NOT against the worker's critic. It is a
    # separate field because `reward` carries the worker's bootstrap (and, once
    # alpha > 0, the intrinsic term), neither of which the manager should see.
    manager_reward: jax.Array


class Bootstrap(NamedTuple):
    """Final-state values for GAE, one per learner.

    Replaces the bare `last_value` of the flat stack: the worker and the manager
    run GAE over different reward streams with different discounts, so each needs
    its own bootstrap.
    """

    worker: jax.Array  # (n_envs, n_agents)
    manager: jax.Array  # (n_envs,) | (n_envs, n_agents)
