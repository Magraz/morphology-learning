"""Runner for the simplified waypoint hierarchy.

Subclasses `MAPPO_JAX_Runner`, so the train loop, stats tracking, checkpoint
cadence, resume and `evaluate()` are inherited unchanged. This class only
supplies the hierarchy's config, its jitted functions (`_make_train`), the
four-train-state checkpoint format, and a `view()` that draws the waypoints.
"""

import dataclasses
import math
from pathlib import Path

import jax
import numpy as np
from flax.serialization import from_bytes, to_bytes

from algorithms.mappo_jax.run import MAPPO_JAX_Runner, make_env
from algorithms.mappo_jax.trainer import RunnerState, global_state_dim
from algorithms.mappo_jax.types import MAPPOConfig
from algorithms.simplified_feudal_mappo_jax.trainer import (
    HierTrainState,
    create_hier_train_state,
    make_policy,
    make_train,
    validate_env,
)
from algorithms.simplified_feudal_mappo_jax.types import (
    Experiment,
    FeudalConfig,
    Model_Params,
    Params,
)

# Waypoint overlay colour in view(): violet, as in the feudal stack's videos.
_WAYPOINT_COLOR = (130, 60, 200)
_HALO_COLOR = (255, 255, 255)


class Simplified_Feudal_MAPPO_JAX_Runner(MAPPO_JAX_Runner):
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
        self._init_trial(
            device, batch_dir, trials_dir, trial_id, checkpoint,
            self.params.random_seeds,
        )

        self.env = make_env(env_config)
        validate_env(self.env)

        horizon = int(self.model_params.goal_horizon)
        # The rollout is a whole number of manager windows. Rounded UP so it
        # still covers the configured length (set >= max_steps in the env groups,
        # because every rollout starts from freshly reset envs).
        n_steps = math.ceil(self.params.n_steps / horizon) * horizon
        if n_steps != self.params.n_steps:
            print(
                f"  n_steps {self.params.n_steps} -> {n_steps} "
                f"(a multiple of goal_horizon={horizon})"
            )

        worker = MAPPOConfig(
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
            n_envs=env_config.get("n_envs"),
            n_total_steps=int(self.params.n_total_steps),
            parameter_sharing=self.params.parameter_sharing,
            hidden_dim=self.model_params.hidden_dim,
            n_eval_episodes=self.params.n_eval_episodes,
        )
        # One manager decision spans `horizon` env steps, so its per-decision
        # discount is gamma^c: both levels see the same horizon in env steps.
        manager = dataclasses.replace(
            worker,
            lr=self.params.manager_lr,
            gamma=self.params.gamma**horizon,
            n_epochs=self.params.manager_n_epochs,
            n_minibatches=self.params.manager_n_minibatches,
        )
        self.feudal_config = FeudalConfig(
            worker=worker,
            manager=manager,
            goal_horizon=horizon,
            waypoint_radius=float(self.model_params.waypoint_radius),
        )
        # The inherited train loop reads n_steps / n_envs / n_total_steps here.
        self.config = worker

        print(
            f"Simplified feudal MAPPO (JAX) | env={env_config.get('environment')} | "
            f"n_agents={self.env.n_agents} | n_envs={worker.n_envs} | "
            f"n_steps={n_steps} ({n_steps // horizon} windows of {horizon}) | "
            f"waypoint_radius={self.feudal_config.waypoint_radius} | "
            f"manager_gamma={manager.gamma:.4f} | total={worker.n_total_steps} | "
            f"backend={jax.default_backend()}"
        )
        print(
            f"  centralized input: gs_dim={global_state_dim(self.env)} "
            f"(env hook: {hasattr(self.env, 'global_state')})"
        )

    def _make_train(self):
        return make_train(self.feudal_config, self.env)

    # ------------------------------------------------------------------ io

    @staticmethod
    def _params_tree(train_state: HierTrainState) -> dict:
        return {
            "worker_actor": train_state.worker.actor_ts.params,
            "worker_critic": train_state.worker.critic_ts.params,
            "manager_actor": train_state.manager.actor_ts.params,
            "manager_critic": train_state.manager.critic_ts.params,
        }

    def save_params(self, train_state, path):
        with open(path, "wb") as f:
            f.write(to_bytes(self._params_tree(train_state)))

    @staticmethod
    def _checkpoint_tree(runner_state, eval_rng) -> dict:
        ts = runner_state.train_state
        return {
            "worker_actor_ts": ts.worker.actor_ts,
            "worker_critic_ts": ts.worker.critic_ts,
            "manager_actor_ts": ts.manager.actor_ts,
            "manager_critic_ts": ts.manager.critic_ts,
            "rng": runner_state.rng,
            "eval_rng": eval_rng,
        }

    def _save_train_checkpoint(self, runner_state, eval_rng, path):
        with open(path, "wb") as f:
            f.write(to_bytes(self._checkpoint_tree(runner_state, eval_rng)))

    def _load_train_checkpoint(self, runner_state, eval_rng, path):
        with open(path, "rb") as f:
            loaded = from_bytes(
                self._checkpoint_tree(runner_state, eval_rng), f.read()
            )
        print(f"Train state loaded from {path}")
        ts = runner_state.train_state
        train_state = HierTrainState(
            worker=ts.worker._replace(
                actor_ts=loaded["worker_actor_ts"],
                critic_ts=loaded["worker_critic_ts"],
            ),
            manager=ts.manager._replace(
                actor_ts=loaded["manager_actor_ts"],
                critic_ts=loaded["manager_critic_ts"],
            ),
        )
        return RunnerState(train_state=train_state, rng=loaded["rng"]), loaded[
            "eval_rng"
        ]

    def _load_train_state(self) -> HierTrainState:
        train_state = create_hier_train_state(
            jax.random.PRNGKey(0), self.feudal_config, self.env
        )
        path = self.dirs["models"] / "models_finished.msgpack"
        if not path.exists():
            path = self.dirs["models"] / "models_checkpoint.msgpack"
        with open(path, "rb") as f:
            params = from_bytes(self._params_tree(train_state), f.read())
        print(f"Params loaded from {path}")
        worker, manager = train_state.worker, train_state.manager
        return HierTrainState(
            worker=worker._replace(
                actor_ts=worker.actor_ts.replace(params=params["worker_actor"]),
                critic_ts=worker.critic_ts.replace(params=params["worker_critic"]),
            ),
            manager=manager._replace(
                actor_ts=manager.actor_ts.replace(params=params["manager_actor"]),
                critic_ts=manager.critic_ts.replace(params=params["manager_critic"]),
            ),
        )

    # ------------------------------------------------------------------ view

    def view(self, n_episodes: int = 10):
        """Render deterministic episodes with every agent's waypoint drawn.

        Each agent gets a line to its current waypoint and a cross on it. The
        waypoint is re-drawn every `goal_horizon` steps, when the manager
        decides again.
        """
        import imageio
        import matplotlib.pyplot as plt
        import pygame

        from environments.mjx_suite.renderer import MJXRenderer

        env = self.env
        horizon = self.feudal_config.goal_horizon
        train_state = self._load_train_state()
        policy = make_policy(self.feudal_config, env)
        no_rng = jax.random.PRNGKey(0)

        # Batch of one env, so view() runs exactly the batched functions that
        # trained and evaluated.
        reset_fn = jax.jit(jax.vmap(env.reset))
        step_fn = jax.jit(jax.vmap(env.step))

        @jax.jit
        def decide_fn(obs, env_state):
            gs, pos = policy.observe(obs, env_state)
            return policy.decide(train_state.manager, gs, pos, no_rng, True)[0]

        @jax.jit
        def act_fn(obs, env_state, waypoint, k):
            _, pos = policy.observe(obs, env_state)
            return policy.act(
                train_state.worker, obs, pos, waypoint, k, no_rng, True
            )[0]

        renderer = MJXRenderer(env)

        def draw_waypoints(agent_xy, waypoint_xy):
            def draw(surface, to_screen, scale):
                arm = max(4, int(0.8 * scale))
                # Halo first, then colour, so the marks read over coloured boxes.
                for color, extra in ((_HALO_COLOR, 3), (_WAYPOINT_COLOR, 0)):
                    for (ax, ay), (wx, wy) in zip(agent_xy, waypoint_xy):
                        a, w = to_screen(ax, ay), to_screen(wx, wy)
                        pygame.draw.line(surface, color, a, w, 2 + extra)
                        for sign in (1, -1):
                            pygame.draw.line(
                                surface, color,
                                (w[0] - arm, w[1] - sign * arm),
                                (w[0] + arm, w[1] + sign * arm),
                                3 + extra,
                            )

            return draw

        print("\nTesting trained agents...")
        for episode in range(n_episodes):
            key = jax.random.PRNGKey(int(np.random.randint(0, 2**31)))
            obs, env_state = reset_fn(key[None])
            frames, rewards = [], []
            for t in range(env.max_steps):
                if t % horizon == 0:
                    waypoint = decide_fn(obs, env_state)
                single = jax.tree.map(lambda x: x[0], env_state)
                # No sensor overlay: its arrows and sector wedges bury the marks.
                frame = renderer.render(single, focus_agent=None)
                frames.append(
                    renderer.annotate(
                        frame,
                        draw_waypoints(
                            renderer.agent_positions(single),
                            env.goal_state_to_world(np.asarray(waypoint[0])),
                        ),
                    )
                )
                obs, env_state, _, terminated, truncated, info = step_fn(
                    env_state, act_fn(obs, env_state, waypoint, t % horizon)
                )
                rewards.append(float(info["task_reward"][0]))
                if bool(terminated[0]) or bool(truncated[0]):
                    break
            rewards = np.asarray(rewards)
            print(f"RETURN: {rewards.sum():.4f}  ({len(rewards)} steps)")

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(np.arange(len(rewards)), rewards)
            for boundary in range(0, len(rewards), horizon):
                ax.axvline(boundary, color="0.85", linewidth=0.5, zorder=0)
            ax.set_ylabel("Team reward")
            ax.set_xlabel("Step (gray lines: manager decisions)")
            ax.set_title(f"Episode {episode} — return {rewards.sum():.2f}")
            plt.tight_layout()
            fig_path = self.dirs["logs"] / f"reward_episode_{episode}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            video_path = self.dirs["logs"] / f"episode_{episode}.mp4"
            imageio.mimwrite(video_path, frames, fps=30, macro_block_size=1)
            print(f"Video saved to {video_path}")
