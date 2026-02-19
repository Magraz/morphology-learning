import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import random

# jaxmarl 0.1.0 uses jax.tree_* which were removed in JAX 0.9.x
for _name in ("map", "leaves", "structure", "unflatten", "flatten", "transpose"):
    if not hasattr(jax, f"tree_{_name}"):
        setattr(jax, f"tree_{_name}", getattr(jax.tree, _name))

import jaxmarl
from flax.serialization import to_bytes, from_bytes

from algorithms.mappo_jax.types import MAPPOConfig, Params, Experiment
from algorithms.mappo_jax.trainer import make_train
from environments.types import EnvironmentParams


def set_seeds(seed: int):
    """Set Python and NumPy seeds. JAX uses explicit PRNG keys."""
    random.seed(seed)
    np.random.seed(seed)


class MAPPO_JAX_Runner:
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        checkpoint: bool,
        exp_config: Experiment,
        env_config: EnvironmentParams,
        env_kwargs: dict = None,
    ):
        # Directory setup (no torch dependency)
        self.device = device
        self.trial_id = trial_id
        self.batch_dir = batch_dir
        self.trial_dir = trials_dir / trial_id
        self.logs_dir = self.trial_dir / "logs"
        self.models_dir = self.trial_dir / "models"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.dirs = {
            "batch": batch_dir,
            "logs": self.logs_dir,
            "models": self.models_dir,
        }

        self.checkpoint = checkpoint
        self.exp_config = exp_config
        self.env_config = env_config
        self.env_kwargs = env_kwargs or {}

        # Set params
        self.params = Params(**self.exp_config.params)

        # Set seeds
        random_seed = self.params.random_seeds[0]
        if self.trial_id.isdigit():
            random_seed = self.params.random_seeds[int(self.trial_id)]
        set_seeds(random_seed)
        self.rng_seed = random_seed

        # Create JaxMARL environment
        self.env = jaxmarl.make(self.env_config.environment, **self.env_kwargs)

        # Build config from Params
        self.config = MAPPOConfig(
            lr=self.params.lr,
            gamma=self.params.gamma,
            gae_lambda=self.params.lmbda,
            eps_clip=self.params.eps_clip,
            ent_coef=self.params.ent_coef,
            val_coef=self.params.val_coef,
            grad_clip=self.params.grad_clip,
            n_epochs=self.params.n_epochs,
            n_minibatches=self.params.n_minibatches,
            n_steps=self.params.batch_size // self.env_config.n_envs,
            n_envs=self.env_config.n_envs,
            n_total_steps=int(self.params.n_total_steps),
            parameter_sharing=self.params.parameter_sharing,
        )

        print(
            f"JAX MAPPO | env={self.env_config.environment} | n_envs={self.config.n_envs} | "
            f"n_steps={self.config.n_steps} | total={self.config.n_total_steps}"
        )

    def train(self):
        rng = jax.random.PRNGKey(self.rng_seed)

        init_fn, update_step_fn, num_updates = make_train(self.config, self.env)

        steps_per_update = self.config.n_steps * self.config.n_envs
        log_every = max(1, num_updates // 100)  # log ~100 times total

        print("JIT-compiling init and update functions (first call may be slow)...")
        runner_state = init_fn(rng)
        jax.block_until_ready(runner_state)

        print(f"Starting training: {num_updates} updates × {steps_per_update} steps")
        print(f"Logging every {log_every} updates\n")

        all_metrics = []
        train_start = time.time()

        for update in range(num_updates):
            runner_state, metrics = update_step_fn(runner_state)

            # Block JAX async execution so timing is accurate
            jax.block_until_ready(metrics)

            all_metrics.append(
                jax.tree.map(lambda x: float(x), metrics)
            )

            if update % log_every == 0 or update == num_updates - 1:
                elapsed = time.time() - train_start
                steps_done = (update + 1) * steps_per_update
                fps = steps_done / elapsed if elapsed > 0 else 0
                m = all_metrics[-1]
                print(
                    f"Update {update+1:>6}/{num_updates} | "
                    f"Steps {steps_done:>10,}/{self.config.n_total_steps:,} | "
                    f"Reward {m['mean_reward']:>8.3f} | "
                    f"FPS {fps:>7.0f} | "
                    f"Time {elapsed:>7.1f}s"
                )

        total_time = time.time() - train_start
        total_steps = num_updates * steps_per_update
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"{'='*60}")
        print(f"Total steps:  {total_steps:,}")
        print(f"Total time:   {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"FPS:          {total_steps/total_time:.0f}")
        print(f"{'='*60}\n")

        self.save_params(
            runner_state.train_state, self.dirs["models"] / "params_final.pkl"
        )
        self.save_metrics(all_metrics, self.dirs["logs"] / "training_stats_finished.pkl")

        return all_metrics

    def save_params(self, train_state, path):
        """Save actor + critic params."""
        params_dict = {
            "actor": train_state.actor_ts.params,
            "critic": train_state.critic_ts.params,
        }
        serialized = to_bytes(params_dict)
        with open(path, "wb") as f:
            f.write(serialized)
        print(f"Params saved to {path}")

    def load_params(self, path, train_state):
        """Load params from file, return updated train_state."""
        target = {
            "actor": train_state.actor_ts.params,
            "critic": train_state.critic_ts.params,
        }
        with open(path, "rb") as f:
            params_dict = from_bytes(target, f.read())

        actor_ts = train_state.actor_ts.replace(params=params_dict["actor"])
        critic_ts = train_state.critic_ts.replace(params=params_dict["critic"])
        from algorithms.mappo_jax.mappo import ActorCriticTrainState

        return ActorCriticTrainState(actor_ts=actor_ts, critic_ts=critic_ts)

    def save_metrics(self, metrics, path):
        """Save training metrics (list of per-update dicts)."""
        with open(path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Metrics saved to {path}")

    def view(self):
        pass

    def evaluate(self):
        pass
