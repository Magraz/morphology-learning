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
    # alpha: the worker's intrinsic-reward weight, as a GRADIENT FRACTION.
    #
    # The two streams are combined at the ADVANTAGE level, each normalized to
    # unit std first: `adv = adv_ext + alpha * adv_int`. So alpha is the mixing
    # ratio of the two gradient directions, NOT a reward coefficient.
    #
    # This distinction is the whole point. As a reward coefficient alpha was
    # unusable: r^I is ~0.155/step and near-constant, while the extrinsic reward
    # is ~5e-05/step until the policy starts delivering boxes, so even alpha=0.1
    # put 300-600x more magnitude on the intrinsic term during exactly the window
    # where learning had to start. Both 0.1 and 0.5 then converged to a
    # task-free fixed point (worker and manager climbing each other's cosine)
    # with the env reward pinned at ~1e-3 for 1e8 steps. Normalizing per stream
    # removes the scale gap entirely, so alpha means what it reads as at any
    # point in training.
    #
    # 0.0 makes the whole intrinsic path inert (the goals still condition the
    # worker, but nothing rewards following them) and is a STATIC no-op: no
    # intrinsic critic is built and the checkpoint format is unchanged.
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
    # Schedule for alpha over training: "linear" decays it to 0 across
    # n_total_steps, "none" holds it constant.
    #
    # Annealing exists because alpha is now a GRADIENT FRACTION (see the
    # `intrinsic_coef` note above): a constant alpha would leave that fraction of
    # the worker's gradient permanently pointing at a task-irrelevant objective,
    # biasing the converged policy. Decaying to 0 makes the endpoint optimal for
    # the true objective while keeping the early exploration pressure.
    intrinsic_anneal: str = "linear"


@dataclass
class Model_Params:
    hidden_dim: int
    # Width of the manager's latent state space `s` AND of the goal `g` — in FuN
    # a goal is a *direction in the state embedding*, so they share a space and
    # this is the only width knob. It is also the manager's entire information
    # bottleneck (the recurrent core consumes `s`), so shrinking it throttles the
    # goal RNN too.
    goal_dim: int = 32
    manager_hidden_dim: int = 256
    # "mlp" (stateless) or "dilated_lstm" (FuN's dilated recurrence, radius =
    # goal_horizon). The mlp core is the default so that goal-mechanism effects
    # are attributable separately from recurrence effects.
    manager_core: str = "mlp"
    # Optional bias-free Dense on the goal before it meets the obs (FuN's `phi`).
    # None = raw concat fusion; see the degeneracy note in worker.py.
    goal_embed_dim: int | None = None
    # L2-normalize the pooled goal `w_t` before it is concatenated onto the obs.
    # True (default) is the FIX for a measured defect: `w_t` is a sum of c
    # near-collinear unit goals, so it arrives at ~c x unit scale (measured
    # ||w_t|| = 9.95 at c=10) and takes 91.6% of the worker's first-layer
    # preactivation variance away from the observation. False reproduces the
    # raw-sum behaviour of runs before this change. See worker.py's docstring.
    normalize_pooled_goal: bool = True
    # ABLATION: feed the worker a zero goal, so its policy is independent of the
    # manager while the manager, its critic, its PG, its diagnostics and the
    # worker's per-agent critic head all still run. This is the isolate rung
    # between `algorithm=mappo_jax` and `feudal_a0` — see
    # conf/model/feudal_zerogoal.yaml for why the alpha=0 arm is NOT one.
    # Incoherent with intrinsic_coef != 0 (it would reward the worker for
    # reaching goals it cannot see); run.py raises on that combination.
    zero_goal: bool = False


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
    # the intrinsic reward is inherently per-agent ((T,E,N)) and shares the
    # worker's advantage. So the feudal `intrinsic_coef=0` arm is still not
    # numerically identical to mappo_jax — the flat baseline is
    # `algorithm=mappo_jax`. It regresses the EXTRINSIC return only; the
    # intrinsic return has its own critic (`intrinsic_critic_ts`) whenever
    # alpha != 0.
    goal_dim: int = 32
    goal_horizon: int = 10
    # Gradient fraction, not a reward coefficient — see `Params.intrinsic_coef`.
    intrinsic_coef: float = 0.0
    intrinsic_anneal: str = "linear"
    manager_lr: float = 3e-4
    manager_val_coef: float = 0.5
    manager_gamma: float = 0.99
    n_manager_epochs: int = 1
    n_manager_critic_epochs: int = 8
    manager_hidden_dim: int = 256
    manager_core: str = "mlp"
    goal_embed_dim: int | None = None
    normalize_pooled_goal: bool = True
    zero_goal: bool = False


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
    # separate field because `reward` carries the worker's bootstrap, which the
    # manager should not see.
    manager_reward: jax.Array

    # ---- Intrinsic stream (all zeros when intrinsic_coef == 0) ----
    # FuN's r^I plus its own truncation bootstrap, (n_envs, n_agents). A SEPARATE
    # stream, never folded into `reward`: it gets its own critic, its own GAE and
    # its own advantage normalization, and only meets the extrinsic signal at the
    # advantage level in `ppo_update`. Folding it into the reward instead made
    # the combination a contest of raw magnitudes, which the intrinsic term won
    # by 300-600x — see `Params.intrinsic_coef`.
    #
    # Filled post-scan (r^I looks *backwards* over `goal_horizon` steps, so it is
    # a whole-trajectory quantity the scan cannot produce).
    intrinsic_reward: jax.Array
    # V^I at act time, (n_envs, n_agents) — mirrors `value` for the intrinsic
    # stream. Always per-agent, as r^I is.
    value_int: jax.Array
    # gamma * truncated * V^I(s_next), (n_envs, n_agents), computed IN the scan
    # (it needs the pre-reset successor state) and added to r^I post-scan. Stored
    # as the bootstrap term rather than as `truncated` so the intrinsic stream
    # gets the same treatment the extrinsic one already gets inline, against its
    # OWN value function.
    intrinsic_bootstrap: jax.Array


class Bootstrap(NamedTuple):
    """Final-state values for GAE, one per learner.

    Replaces the bare `last_value` of the flat stack: the worker and the manager
    run GAE over different reward streams with different discounts, so each needs
    its own bootstrap. The worker's intrinsic stream is a third such stream.
    """

    worker: jax.Array  # (n_envs, n_agents)
    manager: jax.Array  # (n_envs,) | (n_envs, n_agents)
    # (n_envs, n_agents) — V^I at the final state. Zeros when alpha == 0.
    worker_int: jax.Array
