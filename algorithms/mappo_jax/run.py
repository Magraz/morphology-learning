"""Runner for JAX MAPPO over the functional MJX envs.

Mirrors ``MAPPO_Vanilla_Runner`` + ``VecMAPPOTrainer.train``: same per-iteration
collect → update → deterministic-eval cadence, the same
``TrainingStatsTracker`` stats/pickle format (so the plotting notebooks read the
output unchanged), and the same results layout. Differences: params are saved
as flax msgpack (``models_*.msgpack``) instead of torch ``.pth``, and
checkpoint *resume* is not implemented (checkpoints are still written).

Deliberately does not subclass ``algorithms.runner.Runner``: that base imports
torch at module load and halves the torch thread pool, neither of which this
JAX path wants.
"""

import pickle
import random
import time
from pathlib import Path

import jax
import numpy as np
from flax.serialization import from_bytes, to_bytes

from algorithms.mappo_jax.mappo import create_train_state
from algorithms.mappo_jax.trainer import RunnerState, make_train
from algorithms.mappo_jax.types import Experiment, MAPPOConfig, Model_Params, Params
from environments.mjx_suite.multi_box_push_mjx import MultiBoxPushMJX
from environments.types import EnvironmentEnum


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
        debug: bool,
        exp_config: Experiment,
        env_config: dict,
    ):
        if debug:
            jax.config.update("jax_disable_jit", True)
        # Directory setup (same layout as Runner, without the torch import)
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

        # Set params
        self.params = Params(**exp_config.params)
        self.model_params = Model_Params(**exp_config.model_params)

        # Set seeds
        random_seed = self.params.random_seeds[0]
        if self.trial_id.isdigit():
            random_seed = self.params.random_seeds[int(self.trial_id)]
        set_seeds(random_seed)
        self.rng_seed = random_seed

        # Create the functional MJX environment. Two supported env groups:
        #   MULTI_BOX_MJX — continuous force control (the base env)
        #   MACRO_MJX     — the hierarchical macro layer (discrete skill choice,
        #                   one decision per macro_len low-level steps), which
        #                   wraps a base MultiBoxPushMJX.
        environment = env_config.get("environment")
        reward_mode = env_config.get("reward_mode", "dense")
        if environment == EnvironmentEnum.MULTI_BOX_MJX:
            self.env = MultiBoxPushMJX(
                n_agents=env_config.get("n_agents"),
                n_objects=env_config.get("n_objects", 3),
                reward_mode=reward_mode,
            )
        elif environment == EnvironmentEnum.MACRO_MJX:
            from environments.mjx_suite.macro_wrapper import (
                ALIGNED_WINDOWED_DIFFERENCE_REWARDS,
                WINDOWED_DIFFERENCE_REWARDS,
                SyncMacroMJX,
            )

            # The windowed difference rewards (global-window and decision-aligned)
            # are computed by the wrapper (it forks macro windows per agent), so the
            # base env must stay dense — it must not also emit its own per-step D.
            # The single-step "difference_rewards" mode instead lives on the base
            # env and passes through the wrapper's accumulation.
            base_reward_mode = (
                "dense"
                if reward_mode in (
                    WINDOWED_DIFFERENCE_REWARDS, ALIGNED_WINDOWED_DIFFERENCE_REWARDS
                )
                else reward_mode
            )
            base_env = MultiBoxPushMJX(
                n_agents=env_config.get("n_agents"),
                n_objects=env_config.get("n_objects", 3),
                reward_mode=base_reward_mode,
            )
            self.env = SyncMacroMJX(
                base_env,
                macro_len=env_config.get("macro_len", 10),
                reward_mode=reward_mode,
                # Staggered-starts async study: agents come online at random
                # low-level steps and decide on their own phase (max_start_delay in
                # low-level steps). Off by default -> ordinary lockstep options env.
                stagger_starts=env_config.get("stagger_starts", False),
                max_start_delay=env_config.get("max_start_delay", 0),
            )
        else:
            raise ValueError(
                f"mappo_jax supports only '{EnvironmentEnum.MULTI_BOX_MJX}' and "
                f"'{EnvironmentEnum.MACRO_MJX}' (functional JAX API); "
                f"got {environment!r}"
            )
        # A per-agent reward (single-step or windowed difference rewards) switches
        # the critic to a per-agent value head and runs GAE on the agent axis (see
        # MAPPOConfig.per_agent_rewards). The macro wrapper exposes the flag
        # directly; the base env is per-agent only under "difference_rewards".
        per_agent_rewards = getattr(
            self.env, "per_agent_rewards", self.env.reward_mode == "difference_rewards"
        )

        n_envs = env_config.get("n_envs")
        n_steps = self.params.n_steps

        # Build config from Params (the per-update batch is n_steps * n_envs
        # env-steps, scaling with parallelism — same derivation as vanilla)
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
            n_steps=n_steps,
            n_envs=n_envs,
            n_total_steps=int(self.params.n_total_steps),
            parameter_sharing=self.params.parameter_sharing,
            hidden_dim=self.model_params.hidden_dim,
            per_agent_rewards=per_agent_rewards,
        )

        print(
            f"JAX MAPPO | env={environment} | n_envs={self.config.n_envs} | "
            f"n_steps={self.config.n_steps} | total={self.config.n_total_steps} | "
            f"reward_mode={self.env.reward_mode} | "
            f"backend={jax.default_backend()}"
        )

    def train(self):
        from algorithms.mappo_vanilla.trainer_components import TrainingStatsTracker

        init_fn, collect_fn, update_fn, eval_fn, num_updates = make_train(
            self.config, self.env
        )
        steps_per_update = self.config.n_steps * self.config.n_envs
        total_steps = self.config.n_total_steps
        log_every = 10e3

        print("JIT-compiling init/collect/update/eval (first calls may be slow)...")
        runner_state = init_fn(jax.random.PRNGKey(self.rng_seed))
        jax.block_until_ready(runner_state)
        eval_rng = jax.random.PRNGKey(self.rng_seed + 1)

        # Resume from checkpoint if requested and checkpoint exists (vanilla
        # flow: stats checkpoint restores the progress counters, the train
        # checkpoint restores params + optimizer states + RNG keys)
        train_ckpt_path = self.dirs["models"] / "train_checkpoint.msgpack"
        stats_ckpt_path = self.dirs["logs"] / "training_stats_checkpoint.pkl"

        stats_tracker = TrainingStatsTracker()
        checkpoint_loaded = False
        if self.checkpoint and train_ckpt_path.exists() and stats_ckpt_path.exists():
            runner_state, eval_rng = self._load_train_checkpoint(
                runner_state, eval_rng, train_ckpt_path
            )
            with open(stats_ckpt_path, "rb") as f:
                stats_tracker.load_from_dict(pickle.load(f))
            checkpoint_loaded = True

        resume_steps, resume_episodes = stats_tracker.initialize_for_train(
            checkpoint=checkpoint_loaded
        )
        steps_completed = resume_steps
        episodes_completed = resume_episodes
        start_update = resume_steps // steps_per_update

        if resume_steps > 0:
            print(f"Resuming MAPPO training from step {resume_steps}/{total_steps}...")
        else:
            print(
                f"Starting MAPPO training for {total_steps} total environment steps "
                f"({num_updates} updates x {steps_per_update} steps)..."
            )

        # Eval costs a full env.max_steps sequential scan (episodes run to
        # truncation), so unlike vanilla it only runs every k updates; the
        # recorded `reward` stat carries the last eval forward in between to
        # keep the per-iteration series aligned.
        eval_every = 10
        eval_reward = (
            stats_tracker.training_stats["reward"][-1]
            if checkpoint_loaded and stats_tracker.training_stats["reward"]
            else 0.0
        )

        for update in range(start_update, num_updates):
            collection_start = time.time()
            runner_state, trajectory, last_value, rollout_stats = collect_fn(
                runner_state
            )
            jax.block_until_ready(last_value)
            collection_time = time.time() - collection_start

            update_start = time.time()
            runner_state, losses = update_fn(runner_state, trajectory, last_value)
            jax.block_until_ready(losses)
            update_time = time.time() - update_start

            eval_time = 0.0
            if update % eval_every == 0 or update == num_updates - 1:
                eval_start = time.time()
                eval_rng, eval_key = jax.random.split(eval_rng)
                eval_reward = float(eval_fn(runner_state.train_state, eval_key))
                eval_time = time.time() - eval_start

            steps_completed += steps_per_update
            episodes_completed += int(rollout_stats["episode_count"])

            stats_tracker.append_agent_stats(
                {key: float(value) for key, value in losses.items()}
            )
            elapsed_time = stats_tracker.record_iteration(
                steps_completed=steps_completed,
                episodes_completed=episodes_completed,
                reward=eval_reward,
                collection_time=collection_time,
                update_time=update_time,
                eval_time=eval_time,
            )

            steps_per_second = steps_completed / elapsed_time if elapsed_time > 0 else 0

            if steps_completed % log_every < steps_per_update:
                print(
                    f"Steps: {steps_completed}/{total_steps} "
                    f"({steps_completed/total_steps*100:.1f}%) | "
                    f"Episodes: {episodes_completed} | "
                    f"Reward: {eval_reward:.2f} | "
                    f"Time: {elapsed_time:.1f}s | "
                    f"FPS: {steps_per_second:.1f} | "
                    f"Collection: {collection_time:.2f}s | "
                    f"Update: {update_time:.2f}s | "
                    f"Eval: {eval_time:.2f}s"
                )
                self.save_training_stats(stats_tracker, stats_ckpt_path)
                self.save_params(
                    runner_state.train_state,
                    self.dirs["models"] / "models_checkpoint.msgpack",
                )
                self._save_train_checkpoint(runner_state, eval_rng, train_ckpt_path)

        summary = stats_tracker.summarize(steps_completed)
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")
        print(f"Total steps:          {steps_completed}")
        print(f"Total episodes:       {episodes_completed}")
        print(
            f"Total time:           {summary['total_time']:.2f}s "
            f"({summary['total_time']/60:.2f}m)"
        )
        print(f"Final FPS:            {summary['final_fps']:.1f} steps/second")
        print(f"{'='*60}\n")

        self.save_training_stats(
            stats_tracker, self.dirs["logs"] / "training_stats_finished.pkl"
        )
        self.save_params(
            runner_state.train_state, self.dirs["models"] / "models_finished.msgpack"
        )
        # Also refresh the checkpoint pair so a later run with a larger
        # n_total_steps and checkpoint=true extends this one seamlessly
        self.save_training_stats(stats_tracker, stats_ckpt_path)
        self._save_train_checkpoint(runner_state, eval_rng, train_ckpt_path)

    # ------------------------------------------------------------------ io

    def save_params(self, train_state, path):
        """Save actor + critic params as flax msgpack."""
        params_dict = {
            "actor": train_state.actor_ts.params,
            "critic": train_state.critic_ts.params,
        }
        with open(path, "wb") as f:
            f.write(to_bytes(params_dict))

    def _save_train_checkpoint(self, runner_state, eval_rng, path):
        """Full resumable training state: params + optimizer states + step
        counters (the TrainStates serialize as {step, params, opt_state}) and
        both RNG chains. Progress counters live in the stats checkpoint."""
        checkpoint = {
            "actor_ts": runner_state.train_state.actor_ts,
            "critic_ts": runner_state.train_state.critic_ts,
            "rng": runner_state.rng,
            "eval_rng": eval_rng,
        }
        with open(path, "wb") as f:
            f.write(to_bytes(checkpoint))

    def _load_train_checkpoint(self, runner_state, eval_rng, path):
        """Restore a _save_train_checkpoint file into a fresh RunnerState."""
        target = {
            "actor_ts": runner_state.train_state.actor_ts,
            "critic_ts": runner_state.train_state.critic_ts,
            "rng": runner_state.rng,
            "eval_rng": eval_rng,
        }
        with open(path, "rb") as f:
            loaded = from_bytes(target, f.read())
        print(f"Train state loaded from {path}")
        return (
            RunnerState(
                train_state=runner_state.train_state._replace(
                    actor_ts=loaded["actor_ts"], critic_ts=loaded["critic_ts"]
                ),
                rng=loaded["rng"],
            ),
            loaded["eval_rng"],
        )

    def _load_train_state(self):
        """Fresh train state with params restored from the latest save."""
        train_state = create_train_state(
            jax.random.PRNGKey(0),
            self.config,
            self.env.observation_dim,
            self.env.observation_dim * self.env.n_agents,
            self.env.action_dim,
            discrete=getattr(self.env, "discrete", False),
            # Must match training, or the restored critic params won't fit.
            n_critic_outputs=(
                self.env.n_agents if self.config.per_agent_rewards else 1
            ),
        )
        path = self.dirs["models"] / "models_finished.msgpack"
        if not path.exists():
            path = self.dirs["models"] / "models_checkpoint.msgpack"
        target = {
            "actor": train_state.actor_ts.params,
            "critic": train_state.critic_ts.params,
        }
        with open(path, "rb") as f:
            params_dict = from_bytes(target, f.read())
        print(f"Params loaded from {path}")
        return train_state._replace(
            actor_ts=train_state.actor_ts.replace(params=params_dict["actor"]),
            critic_ts=train_state.critic_ts.replace(params=params_dict["critic"]),
        )

    def save_training_stats(self, stats_tracker, path):
        with open(path, "wb") as f:
            pickle.dump(stats_tracker.to_dict(), f)

    # ------------------------------------------------------------------ view / eval

    def view(self):
        """Render deterministic episodes with the trained policy (vanilla view)."""
        import imageio
        import matplotlib.pyplot as plt

        from algorithms.mappo_jax.network import sample_action
        from environments.mjx_suite.renderer import MJXRenderer, MuJoCoNativeRenderer

        train_state = self._load_train_state()
        discrete = getattr(self.env, "discrete", False)
        # The macro env's state is the base EnvState, so the renderers (which
        # expect a MultiBoxPushMJX) run on the wrapped base env.
        is_macro = hasattr(self.env, "macro_len")
        render_env = getattr(self.env, "env", self.env)
        renderer = MJXRenderer(render_env)
        try:
            native_renderer = MuJoCoNativeRenderer(render_env)
        except Exception as e:  # no GL context (run with MUJOCO_GL=egl headless)
            print(f"Native MuJoCo renderer unavailable ({e}); skipping native videos")
            native_renderer = None
        reset_fn = jax.jit(self.env.reset)
        step_fn = jax.jit(self.env.step)

        @jax.jit
        def policy_fn(obs):
            actions, _ = sample_action(
                jax.random.PRNGKey(0),
                train_state.actor_ts.apply_fn,
                train_state.actor_ts.params,
                obs,
                discrete=discrete,
                deterministic=True,
            )
            return actions

        def _draw(state, obs):
            """Append one rendered frame (+ native) for the given base state."""
            frames.append(renderer.render(state, obs=np.asarray(obs)))
            if native_renderer is not None:
                native_frames.append(native_renderer.render(state))

        # For the macro env, render at *low-level* granularity: hold the
        # high-level skill choice fixed for macro_len steps but drive (and draw)
        # the base env one physics step at a time, so the video is smooth instead
        # of jumping macro_len steps per frame. The high-level policy re-decides
        # at each macro boundary off the base obs there, exactly as SyncMacroMJX
        # does internally.
        base_step_fn = jax.jit(render_env.step) if is_macro else None
        skill_actions_fn = (
            jax.jit(self.env._skill_actions) if is_macro else None
        )

        print("\nTesting trained agents...")
        for episode in range(10):
            key = jax.random.PRNGKey(int(np.random.randint(0, 2**31)))
            obs, state = reset_fn(key)
            # Under staggered starts the macro state wraps the base EnvState; the
            # low-level renderer drives the base env directly (stagger masking is
            # not reflected in the video).
            if is_macro:
                state = self.env.base_state(state)
            rewards, frames, native_frames = [], [], []

            if is_macro:
                done = False
                for _ in range(self.env.max_steps):  # macro decisions
                    skills = policy_fn(obs)
                    for _ in range(self.env.macro_len):  # low-level steps
                        _draw(state, obs)
                        actions = skill_actions_fn(state, skills)
                        obs, state, _, terminated, truncated, info = base_step_fn(
                            state, actions
                        )
                        rewards.append(float(info["task_reward"]))
                        if bool(terminated) or bool(truncated):
                            done = True
                            break
                    if done:
                        break
            else:
                for _ in range(self.env.max_steps):
                    _draw(state, obs)
                    obs, state, _, terminated, truncated, info = step_fn(
                        state, policy_fn(obs)
                    )
                    # Team reward: the env's `reward` is per-agent under
                    # difference_rewards, and the plot is of team performance.
                    rewards.append(float(info["task_reward"]))
                    if bool(terminated) or bool(truncated):
                        break
            rewards = np.asarray(rewards)

            print(f"REWARD: {rewards[-1]:.4f}")

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(np.arange(len(rewards)), rewards)
            ax.set_ylabel("Reward")
            ax.set_xlabel("Step")
            ax.set_title(f"Episode {episode} — Reward")
            plt.tight_layout()
            fig_path = self.dirs["logs"] / f"reward_episode_{episode}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved to {fig_path}")

            video_path = self.dirs["logs"] / f"episode_{episode}.mp4"
            imageio.mimwrite(video_path, frames, fps=30, macro_block_size=1)
            print(f"Video saved to {video_path}")

            if native_frames:
                native_path = self.dirs["logs"] / f"episode_{episode}_native.mp4"
                imageio.mimwrite(
                    native_path, native_frames, fps=30, macro_block_size=1
                )
                print(f"Native video saved to {native_path}")

    def evaluate(self):
        """Deterministic evaluation of the saved policy (PolicyEvaluator parity)."""
        train_state = self._load_train_state()
        _, _, _, eval_fn, _ = make_train(self.config, self.env)
        reward = float(eval_fn(train_state, jax.random.PRNGKey(self.rng_seed + 1)))
        print(f"Mean eval episode return: {reward:.2f}")
        return reward
