"""Config surface and rollout records for `rmappo_jax`.

`rmappo_jax` reimplements the reference MAPPO code, marlbenchmark/on-policy
(`main`, `onpolicy/algorithms/r_mappo`), in JAX over this repo's functional
envs. Every default below is that repo's `config.py` default. Each on-policy flag
maps to one key:

| on-policy           | here (`params`)        | on-policy             | here (`model_params`)       |
|---------------------|------------------------|-----------------------|-----------------------------|
| episode_length      | n_steps                | hidden_size           | hidden_size                 |
| ppo_epoch           | n_epochs               | layer_N               | layer_N                     |
| num_mini_batch      | n_minibatches          | use_ReLU              | use_ReLU                    |
| clip_param          | eps_clip               | use_feature_normalization | use_feature_normalization |
| entropy_coef        | ent_coef               | gain                  | gain                        |
| value_loss_coef     | val_coef               | use_recurrent_policy  | use_recurrent_policy        |
| max_grad_norm       | grad_clip              | recurrent_N           | recurrent_N                 |
| gae_lambda          | lmbda                  |                       |                             |
| eval_episodes       | n_eval_episodes        |                       |                             |

Every other flag keeps its on-policy name. The renamed ones are the keys this
repo's other algorithm groups already use, so an env group's `params.n_steps` /
`params.n_total_steps` overrides reach this algorithm as they reach every other.
"""

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import jax


@dataclass
class Params:
    # Training Params
    n_epochs: int  # ppo_epoch
    n_total_steps: int  # num_env_steps (an experiment budget; env groups set it)
    n_minibatches: int  # num_mini_batch
    # episode_length: the rollout length per env between updates. Envs are NOT
    # reset between rollouts (see trainer.py), so this need not cover an episode.
    n_steps: int
    parameter_sharing: bool  # share_policy; only True is implemented
    random_seeds: list

    lr: float = 5e-4
    critic_lr: float = 5e-4
    opti_eps: float = 1e-5
    gamma: float = 0.99
    lmbda: float = 0.95
    eps_clip: float = 0.2
    ent_coef: float = 0.01
    val_coef: float = 1.0
    use_max_grad_norm: bool = True
    grad_clip: float = 10.0
    use_valuenorm: bool = True
    use_huber_loss: bool = True
    huber_delta: float = 10.0
    use_clipped_value_loss: bool = True
    use_value_active_masks: bool = True
    use_policy_active_masks: bool = True
    # False (on-policy's default): a time-limit truncation ends the episode like
    # a termination, with no bootstrap. True: this repo's bootstrap (value the
    # real successor before the reset, add gamma * V to the reward). NOT
    # on-policy's own True path, which zeroes the truncated step's advantage.
    use_proper_time_limits: bool = False
    use_linear_lr_decay: bool = False
    data_chunk_length: int = 10
    n_eval_episodes: int = 32
    eval_interval: int = 25  # updates between deterministic evals


@dataclass
class Model_Params:
    # No default, so a model group must set it: `model=mlp` (mappo_jax's group)
    # fails here with a TypeError before any results directory is created,
    # instead of writing into mappo_jax's results path.
    use_recurrent_policy: bool
    hidden_size: int = 64
    layer_N: int = 1
    use_ReLU: bool = True
    use_feature_normalization: bool = True
    gain: float = 0.01
    recurrent_N: int = 1


@dataclass
class Experiment:
    device: str
    model_params: Model_Params
    params: Params


@dataclass(frozen=True)
class RMAPPOConfig:
    """Resolved config consumed by the jitted train functions.

    `n_steps` / `n_envs` / `n_total_steps` are the attribute names the inherited
    `MAPPO_JAX_Runner.train` loop reads.
    """

    n_envs: int
    n_steps: int
    n_total_steps: int
    n_epochs: int = 15
    n_minibatches: int = 1
    parameter_sharing: bool = True
    lr: float = 5e-4
    critic_lr: float = 5e-4
    opti_eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    ent_coef: float = 0.01
    val_coef: float = 1.0
    use_max_grad_norm: bool = True
    grad_clip: float = 10.0
    use_valuenorm: bool = True
    use_huber_loss: bool = True
    huber_delta: float = 10.0
    use_clipped_value_loss: bool = True
    use_value_active_masks: bool = True
    use_policy_active_masks: bool = True
    use_proper_time_limits: bool = False
    use_linear_lr_decay: bool = False
    data_chunk_length: int = 10
    n_eval_episodes: int = 32
    eval_interval: int = 25
    use_recurrent_policy: bool = True
    hidden_size: int = 64
    layer_N: int = 1
    use_ReLU: bool = True
    use_feature_normalization: bool = True
    gain: float = 0.01
    recurrent_N: int = 1

    @property
    def chunk_length(self) -> int:
        """Steps per training sequence: `data_chunk_length` for the recurrent
        policy, 1 (independent rows) for the feedforward one."""
        return self.data_chunk_length if self.use_recurrent_policy else 1


def make_config(params: Params, model_params: Model_Params, n_envs: int):
    """Resolve the Hydra blocks into an `RMAPPOConfig`. Pure (no env).

    Under the recurrent policy `n_steps` is rounded UP to a multiple of
    `data_chunk_length`, so every training chunk lies inside one (env, agent)
    trajectory. on-policy does not check this and silently builds chunks that
    run across two trajectories. Compare `.n_steps` with `params.n_steps` to see
    whether it moved.
    """
    n_steps = int(params.n_steps)
    if model_params.use_recurrent_policy:
        chunk = int(params.data_chunk_length)
        if chunk < 1:
            raise ValueError(f"data_chunk_length must be >= 1, got {chunk}")
        n_steps = -(-n_steps // chunk) * chunk
    return RMAPPOConfig(
        n_envs=int(n_envs),
        n_steps=n_steps,
        n_total_steps=int(params.n_total_steps),
        n_epochs=params.n_epochs,
        n_minibatches=params.n_minibatches,
        parameter_sharing=params.parameter_sharing,
        lr=params.lr,
        critic_lr=params.critic_lr,
        opti_eps=params.opti_eps,
        gamma=params.gamma,
        gae_lambda=params.lmbda,
        eps_clip=params.eps_clip,
        ent_coef=params.ent_coef,
        val_coef=params.val_coef,
        use_max_grad_norm=params.use_max_grad_norm,
        grad_clip=params.grad_clip,
        use_valuenorm=params.use_valuenorm,
        use_huber_loss=params.use_huber_loss,
        huber_delta=params.huber_delta,
        use_clipped_value_loss=params.use_clipped_value_loss,
        use_value_active_masks=params.use_value_active_masks,
        use_policy_active_masks=params.use_policy_active_masks,
        use_proper_time_limits=params.use_proper_time_limits,
        use_linear_lr_decay=params.use_linear_lr_decay,
        data_chunk_length=params.data_chunk_length,
        n_eval_episodes=params.n_eval_episodes,
        eval_interval=params.eval_interval,
        **dataclasses.asdict(model_params),
    )


class Transition(NamedTuple):
    """One rollout step across all envs (leading dim n_steps after the scan).

    The fields mirror on-policy's `SharedReplayBuffer`: `actor_h` / `critic_h`
    are its `rnn_states` / `rnn_states_critic` and `mask` its `masks`, all for
    the step's own observation. Unlike the buffer, the centralized state is
    stored once per env, not once per agent; the sampler repeats it per agent.
    """

    obs: jax.Array  # (n_envs, n_agents, obs_dim)
    global_state: jax.Array  # (n_envs, global_state_dim)
    # GRU hidden states the step started from, BEFORE `mask` is applied
    # (n_envs, n_agents, recurrent_N, hidden_size); a scalar placeholder for the
    # feedforward policy.
    actor_h: jax.Array
    critic_h: jax.Array
    # (n_envs,) 0.0 when this observation starts a new episode (the hidden state
    # is reset before the step), else 1.0.
    mask: jax.Array
    action: jax.Array  # (n_envs, n_agents, action_dim) | (n_envs, n_agents)
    log_prob: jax.Array  # (n_envs, n_agents)
    value: jax.Array  # (n_envs, n_agents) NORMALIZED critic output
    reward: jax.Array  # (n_envs, n_agents) the shared team reward, per agent
    done: jax.Array  # (n_envs,) terminated | truncated
    # (n_envs, n_agents) 1.0 when the agent took a real decision this step
    # (info["active"]; all ones for envs that do not emit it).
    active_mask: jax.Array
    # (n_envs, n_agents, action_dim) legal-action mask for envs with an
    # `avail_actions` hook, else a scalar placeholder.
    action_mask: jax.Array
    team_reward: jax.Array  # (n_envs,) info["task_reward"], logging only
