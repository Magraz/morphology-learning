"""Runner for `rmappo_jax`, on-policy's MAPPO over the functional JAX envs.

Subclasses `MAPPO_JAX_Runner`, so env construction (`make_env`), results
layout, the train loop, stats tracking, checkpoint cadence and `evaluate()` are
inherited. This class supplies the config, the train functions, the checkpoint
trees (which add the ValueNorm state) and the recurrent `view()` policy.

Resume (`checkpoint=true`) restores the networks, both Adam states, ValueNorm
and the RNG chains. The rollout itself is not saved: a resumed run starts fresh
episodes with zeroed hidden states.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes, msgpack_restore, to_bytes

from algorithms.mappo_jax.run import MAPPO_JAX_Runner, make_env
from algorithms.mappo_jax.trainer import global_state_dim
from algorithms.rmappo_jax.mappo import (
    RMAPPOTrainState,
    create_train_state,
    zero_hidden,
)
from algorithms.rmappo_jax.trainer import actor_step, make_train, validate
from algorithms.rmappo_jax.types import Experiment, Model_Params, Params, make_config

# Settings that leave every parameter shape unchanged, so a checkpoint alone
# cannot tell which values trained it. They are stored with the train
# checkpoint and a resume under other values raises (cf. the scheduler record
# in simplified_feudal_mappo_jax/run.py).
META_FIELDS = (
    "n_envs", "n_steps", "n_epochs", "n_minibatches", "lr", "critic_lr",
    "opti_eps", "gamma", "gae_lambda", "eps_clip", "ent_coef", "val_coef",
    "use_max_grad_norm", "grad_clip", "use_valuenorm", "use_huber_loss",
    "huber_delta", "use_clipped_value_loss", "use_value_active_masks",
    "use_policy_active_masks", "use_proper_time_limits", "use_linear_lr_decay",
    "data_chunk_length", "use_ReLU", "gain",
)


class RecurrentViewPolicy:
    """`view()`'s deterministic policy, carrying the GRU hidden state across the
    decisions of one rendered episode (the `StatelessViewPolicy` interface)."""

    def __init__(self, train_state, env, config):
        discrete = getattr(env, "discrete", False)
        n_agents, obs_dim = env.n_agents, env.observation_dim
        self._initial_hidden = zero_hidden(config, (1, n_agents))

        @jax.jit
        def act(obs, hidden, mask):
            actions, _, hidden = actor_step(
                train_state.actor_ts.apply_fn,
                train_state.actor_ts.params,
                obs.reshape(1, n_agents, obs_dim),
                hidden,
                jnp.ones((1,)),
                jax.random.PRNGKey(0),
                discrete,
                deterministic=True,
                action_mask=None if mask is None else mask.reshape(1, n_agents, -1),
            )
            return actions[0], hidden

        self._act = act
        self.reset()

    def reset(self):
        self._hidden = self._initial_hidden

    def __call__(self, obs, mask=None):
        actions, self._hidden = self._act(obs, self._hidden, mask)
        return actions


class RMAPPO_JAX_Runner(MAPPO_JAX_Runner):
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        checkpoint: bool,
        exp_config: Experiment,
        env_config: dict,
    ):
        self.params = Params(**exp_config.params)
        self.model_params = Model_Params(**exp_config.model_params)
        self.config = make_config(
            self.params, self.model_params, env_config.get("n_envs")
        )
        # Built and checked before the results directory exists, so a rejected
        # configuration leaves nothing behind.
        self.env = make_env(env_config)
        validate(self.config, self.env)
        self._init_trial(
            device, batch_dir, trials_dir, trial_id, checkpoint,
            self.params.random_seeds,
        )

        config = self.config
        if config.n_steps != self.params.n_steps:
            print(
                f"  n_steps {self.params.n_steps} -> {config.n_steps} "
                f"(a multiple of data_chunk_length={config.data_chunk_length})"
            )
        policy = (
            f"rMAPPO (GRU, chunks of {config.data_chunk_length})"
            if config.use_recurrent_policy
            else "MAPPO (feedforward)"
        )
        print(
            f"{policy} (JAX) | env={env_config.get('environment')} | "
            f"n_agents={self.env.n_agents} | n_envs={config.n_envs} | "
            f"n_steps={config.n_steps} | hidden={config.hidden_size} | "
            f"epochs={config.n_epochs} x minibatches={config.n_minibatches} | "
            f"total={config.n_total_steps} | backend={jax.default_backend()}"
        )
        print(
            f"  valuenorm={config.use_valuenorm} huber={config.use_huber_loss} "
            f"clipped_value={config.use_clipped_value_loss} "
            f"proper_time_limits={config.use_proper_time_limits} "
            f"lr_decay={config.use_linear_lr_decay} | "
            f"centralized input: gs_dim={global_state_dim(self.env)} "
            f"(env hook: {hasattr(self.env, 'global_state')})"
        )

    def _make_train(self):
        return make_train(self.config, self.env)

    def _eval_every(self) -> int:
        return self.config.eval_interval

    def _view_policy(self, train_state):
        return RecurrentViewPolicy(train_state, self.env, self.config)

    # ------------------------------------------------------------------ io

    @staticmethod
    def _params_tree(train_state: RMAPPOTrainState) -> dict:
        return {
            "actor": train_state.actor_ts.params,
            "critic": train_state.critic_ts.params,
            "value_norm": train_state.value_norm,
        }

    def save_params(self, train_state, path):
        with open(path, "wb") as f:
            f.write(to_bytes(self._params_tree(train_state)))

    def _config_meta(self) -> np.ndarray:
        return np.array(
            [float(getattr(self.config, name)) for name in META_FIELDS], np.float64
        )

    def _checkpoint_tree(self, runner_state, eval_rng) -> dict:
        ts = runner_state.train_state
        return {
            "actor_ts": ts.actor_ts,
            "critic_ts": ts.critic_ts,
            "value_norm": ts.value_norm,
            "rng": runner_state.rng,
            "eval_rng": eval_rng,
            "config_meta": self._config_meta(),
        }

    def _check_config_meta(self, raw: dict, path):
        meta, stored = self._config_meta(), raw.get("config_meta")
        if stored is None:
            raise ValueError(f"{path} has no rmappo_jax config record")
        stored = np.asarray(stored, np.float64)
        if stored.shape != meta.shape or not np.array_equal(stored, meta):
            changed = {
                name: (old, new)
                for name, old, new in zip(META_FIELDS, stored.tolist(), meta.tolist())
                if old != new
            } if stored.shape == meta.shape else stored.tolist()
            raise ValueError(
                f"{path} was trained under other settings ((checkpoint, this "
                f"config): {changed}). Resume with the settings that trained it, "
                "or train into another model group."
            )

    def _save_train_checkpoint(self, runner_state, eval_rng, path):
        with open(path, "wb") as f:
            f.write(to_bytes(self._checkpoint_tree(runner_state, eval_rng)))

    def _load_train_checkpoint(self, runner_state, eval_rng, path):
        with open(path, "rb") as f:
            data = f.read()
        self._check_config_meta(msgpack_restore(data), path)
        loaded = from_bytes(self._checkpoint_tree(runner_state, eval_rng), data)
        print(f"Train state loaded from {path} (episodes restart from reset)")
        train_state = RMAPPOTrainState(
            actor_ts=loaded["actor_ts"],
            critic_ts=loaded["critic_ts"],
            value_norm=loaded["value_norm"],
        )
        return (
            runner_state._replace(train_state=train_state, rng=loaded["rng"]),
            loaded["eval_rng"],
        )

    def _load_train_state(self) -> RMAPPOTrainState:
        """A train state with the networks and ValueNorm of the latest save."""
        train_state = create_train_state(
            jax.random.PRNGKey(0),
            self.config,
            self.env.observation_dim,
            global_state_dim(self.env),
            self.env.action_dim,
            getattr(self.env, "discrete", False),
            num_updates=1,
        )
        path = self.dirs["models"] / "models_finished.msgpack"
        if not path.exists():
            path = self.dirs["models"] / "models_checkpoint.msgpack"
        with open(path, "rb") as f:
            params = from_bytes(self._params_tree(train_state), f.read())
        print(f"Params loaded from {path}")
        return RMAPPOTrainState(
            actor_ts=train_state.actor_ts.replace(params=params["actor"]),
            critic_ts=train_state.critic_ts.replace(params=params["critic"]),
            value_norm=params["value_norm"],
        )
