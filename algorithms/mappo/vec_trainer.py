"""Vectorized MAPPO trainer.

This is the sole supported MAPPO trainer path; the legacy non-vector trainer
has been removed.
"""

from functools import partial
import time

import gymnasium as gym
import numpy as np
import psutil

from algorithms.create_env import make_vec_env
from algorithms.mappo.hypergraph import distance_based_hyperedges, object_contact_hyperedges
from algorithms.mappo.mappo import MAPPOAgent
from algorithms.mappo.trainer_components import (
    CheckpointIO,
    HypergraphRuntime,
    PolicyEvaluator,
    PolicyRenderer,
    RolloutCollector,
    TrainingStatsTracker,
)
from algorithms.mappo.types import Model_Params
from environments.types import EnvironmentEnum, EnvironmentParams


class VecMAPPOTrainer:
    def __init__(
        self,
        params=None,
        dirs=None,
        device: str = "cpu",
        env_params: EnvironmentParams = None,
        model_params: Model_Params = None,
    ):
        self.device = device
        self.dirs = dirs
        self.n_agents = env_params.n_agents
        self.params = params
        self.env_name = env_params.environment
        self.n_parallel_envs = env_params.n_envs
        self.env_variant = env_params.env_variant
        self.n_objects = env_params.n_objects
        self.reward_mode = env_params.reward_mode
        self.critic_type = model_params.critic_type
        self.entropy_pred_seq_len = model_params.entropy_pred_seq_len
        self.entropy_conditioning = model_params.entropy_conditioning

        self.n_eval_episodes = 5
        self.eval_env = make_vec_env(
            self.env_name,
            self.n_agents,
            self.n_eval_episodes,
            use_async=True,
            env_variant=self.env_variant,
            n_objects=self.n_objects,
            reward_mode=self.reward_mode,
        )
        self.vec_env = make_vec_env(
            self.env_name,
            self.n_agents,
            self.n_parallel_envs,
            use_async=True,
            env_variant=self.env_variant,
            n_objects=self.n_objects,
            reward_mode=self.reward_mode,
        )

        obs_space = self.vec_env.single_observation_space
        act_space = self.vec_env.single_action_space
        observation_dim = obs_space.shape[1]
        global_state_dim = observation_dim * self.n_agents

        if isinstance(act_space, gym.spaces.MultiDiscrete):
            action_dim = int(act_space.nvec[0])
        else:
            action_dim = act_space.shape[1]

        self.discrete = self.env_name in [
            EnvironmentEnum.MPE_SPREAD,
            EnvironmentEnum.MPE_SIMPLE,
            EnvironmentEnum.SMACV2,
            EnvironmentEnum.SMACLITE,
        ]

        hyperedge_fns = [
            (partial(distance_based_hyperedges, threshold=1.0), "obs"),
            (object_contact_hyperedges, "agents_2_objects"),
        ]

        self.agent = MAPPOAgent(
            observation_dim,
            global_state_dim,
            action_dim,
            self.n_agents,
            self.params,
            self.device,
            self.discrete,
            self.n_parallel_envs,
            model_params=model_params,
            hyperedge_fns=(hyperedge_fns if self.critic_type == "multi_hgnn" else None),
        )

        # Sync device — agent may have upgraded to CUDA in its __init__
        self.device = self.agent.device

        self._process = psutil.Process()

        self.hypergraph_runtime = HypergraphRuntime(
            agent=self.agent,
            device=self.device,
            n_agents=self.n_agents,
            n_parallel_envs=self.n_parallel_envs,
            critic_type=self.critic_type,
            model_params=model_params,
            batch_dir=self.dirs.get("batch"),
        )
        self.rollout_collector = RolloutCollector(
            vec_env=self.vec_env,
            agent=self.agent,
            device=self.device,
            n_agents=self.n_agents,
            n_parallel_envs=self.n_parallel_envs,
            discrete=self.discrete,
            entropy_conditioning=self.entropy_conditioning,
            hypergraph_runtime=self.hypergraph_runtime,
        )
        self.evaluator = PolicyEvaluator(
            eval_env=self.eval_env,
            agent=self.agent,
            n_eval_episodes=self.n_eval_episodes,
            discrete=self.discrete,
            entropy_conditioning=self.entropy_conditioning,
            hypergraph_runtime=self.hypergraph_runtime,
        )
        self.renderer = PolicyRenderer(
            agent=self.agent,
            device=self.device,
            env_name=self.env_name,
            env_variant=self.env_variant,
            n_agents=self.n_agents,
            n_objects=self.n_objects,
            reward_mode=self.reward_mode,
            discrete=self.discrete,
            entropy_conditioning=self.entropy_conditioning,
            hypergraph_runtime=self.hypergraph_runtime,
        )
        self.checkpoint_io = CheckpointIO(agent=self.agent, device=self.device)
        self.stats_tracker = TrainingStatsTracker()

    @property
    def training_stats(self):
        return self.stats_tracker.training_stats

    def _get_total_memory_mb(self):
        """Return RSS for the trainer process and all live child processes."""
        total_rss = self._process.memory_info().rss
        for child in self._process.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_rss / 1024 / 1024

    def collect_trajectory(self, max_steps):
        result = self.rollout_collector.collect(max_steps=max_steps)
        return result.step_count, result.episode_count, result.final_values

    def train(
        self,
        total_steps,
        batch_size,
        minibatches,
        epochs,
        checkpoint: bool = False,
        log_every: int = 10e3,
    ):
        """Train MAPPO agent."""
        resume_steps, resume_episodes = self.stats_tracker.initialize_for_train(checkpoint)

        steps_completed = resume_steps
        episodes_completed = resume_episodes

        if resume_steps > 0:
            print(f"Resuming MAPPO training from step {resume_steps}/{total_steps}...")
        else:
            print(f"Starting MAPPO training for {total_steps} total environment steps...")

        while steps_completed < total_steps:
            steps_to_collect = min(batch_size, total_steps - steps_completed)

            collection_start = time.time()
            rollout = self.rollout_collector.collect(max_steps=int(steps_to_collect))
            collection_time = time.time() - collection_start

            update_start = time.time()
            stats = self.agent.update(
                next_value=rollout.final_values,
                minibatch_size=batch_size // minibatches,
                epochs=epochs,
            )
            update_time = time.time() - update_start

            steps_completed += rollout.step_count
            episodes_completed += rollout.episode_count

            self.stats_tracker.append_agent_stats(stats)

            eval_start = time.time()
            eval_rewards = self.evaluator.evaluate()
            eval_time = time.time() - eval_start

            elapsed_time = self.stats_tracker.record_iteration(
                steps_completed=steps_completed,
                episodes_completed=episodes_completed,
                reward=eval_rewards,
                collection_time=collection_time,
                update_time=update_time,
                eval_time=eval_time,
            )

            steps_per_second = steps_completed / elapsed_time if elapsed_time > 0 else 0

            if steps_completed % log_every < rollout.step_count:
                mem = self._get_total_memory_mb()
                print(
                    f"Steps: {steps_completed}/{total_steps} ({steps_completed/total_steps*100:.1f}%) | "
                    f"Episodes: {episodes_completed} | "
                    f"Reward: {self.training_stats['reward'][-1]:.2f} | "
                    f"Time: {elapsed_time:.1f}s | "
                    f"FPS: {steps_per_second:.1f} | "
                    f"Mem: {mem:.0f}MB | "
                    f"Collection: {collection_time:.2f}s | "
                    f"Update: {update_time:.2f}s | "
                    f"Eval: {eval_time:.2f}s"
                )

                self.save_training_stats(self.dirs["logs"] / "training_stats_checkpoint.pkl")
                self.save_agent(self.dirs["models"] / "models_checkpoint.pth")

        self.close_environments()

        summary = self.stats_tracker.summarize(steps_completed)
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
        print("\nTime breakdown:")
        print(f"  Avg collection:     {summary['avg_collection_time']:.3f}s")
        print(f"  Avg update:         {summary['avg_update_time']:.3f}s")
        print(f"  Avg evaluation:     {summary['avg_eval_time']:.3f}s")
        print(f"{'='*60}\n")

    def evaluate(self):
        return self.evaluator.evaluate()

    def render(self, capture_video=False):
        return self.renderer.render(capture_video=capture_video)

    def build_snapshot_figure(self, frames, hypergraphs, n_snapshots=4):
        return self.renderer.build_snapshot_figure(
            frames, hypergraphs, n_snapshots=n_snapshots
        )

    def close_environments(self):
        """Properly close all vectorized environments."""
        if hasattr(self, "vec_env") and self.vec_env is not None:
            self.vec_env.close()
        if hasattr(self, "eval_env") and self.eval_env is not None:
            self.eval_env.close()

    def __del__(self):
        """Destructor to ensure environments are closed."""
        try:
            self.close_environments()
        except Exception:
            pass

    def save_agent(self, path):
        self.checkpoint_io.save_agent(path)

    def load_agent(self, filepath, restore_rng=False):
        self.checkpoint_io.load_agent(filepath, restore_rng=restore_rng)
        print(f"Agents loaded from {filepath}")

    def load_checkpoint_progress(self, path):
        """Load training progress counters from training stats checkpoint."""
        stats = self.checkpoint_io.load_training_stats(path)
        self.stats_tracker.load_from_dict(stats)

    def save_training_stats(self, path):
        self.checkpoint_io.save_training_stats(path, self.stats_tracker.to_dict())
