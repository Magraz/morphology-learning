import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
import time
import random
import psutil

from algorithms.mappo.mappo import MAPPOAgent
from environments.types import EnvironmentEnum
from algorithms.create_env import make_vec_env
from functools import partial
from algorithms.mappo.hypergraph import (
    batch_hypergraphs,
    build_hypergraph,
    canonicalize_edge_lists,
    compute_hyperedge_structural_entropy_batch,
    distance_based_hyperedges,
    object_contact_hyperedges,
)
import gymnasium as gym

import torch


class VecMAPPOTrainer:
    def __init__(
        self,
        env_name,
        n_agents,
        params=None,
        dirs=None,
        device: str = "cpu",
        n_parallel_envs: int = 1,
        env_variant: str = None,
        critic_type: str = "mlp",
        n_hyperedge_types: int = 0,
    ):
        self.device = device
        self.dirs = dirs
        self.n_agents = n_agents
        self.params = params
        self.env_name = env_name
        self.n_parallel_envs = n_parallel_envs
        self.env_variant = env_variant
        self.critic_type = critic_type
        self.n_hyperedge_types = n_hyperedge_types

        # Create environment for evaluation (parallel eval episodes)
        self.n_eval_episodes = 5
        self.eval_env = make_vec_env(
            self.env_name,
            self.n_agents,
            self.n_eval_episodes,
            use_async=True,
            env_variant=self.env_variant,
        )

        # Create vectorized environment using Gymnasium's API
        self.vec_env = make_vec_env(
            self.env_name,
            self.n_agents,
            self.n_parallel_envs,
            use_async=True,  # Use parallel processing
            env_variant=self.env_variant,
        )

        # Determine observation, action and and global state dimension
        obs_space = self.vec_env.single_observation_space  # Box(n_agents, obs_dim)
        act_space = self.vec_env.single_action_space
        observation_dim = obs_space.shape[1]
        global_state_dim = observation_dim * n_agents
        if isinstance(act_space, gym.spaces.MultiDiscrete):
            action_dim = int(act_space.nvec[0])
        else:
            action_dim = act_space.shape[1]

        # Set action bounds based on environment
        if self.env_name in [
            EnvironmentEnum.MPE_SPREAD,
            EnvironmentEnum.MPE_SIMPLE,
            EnvironmentEnum.SMACV2,
            EnvironmentEnum.SMACLITE,
        ]:
            self.discrete = True
        else:
            self.discrete = False

        # Hyperedge builder specs: (fn, source) pairs
        # "obs" means the fn receives the (n_agents, obs_dim) observation array;
        # any other string is looked up in the info dict.
        self.hyperedge_fns = [
            (partial(distance_based_hyperedges, threshold=1.0), "obs"),
            (object_contact_hyperedges, "agents_2_objects"),
        ]

        # Create MAPPO agent
        self.agent = MAPPOAgent(
            observation_dim,
            global_state_dim,
            action_dim,
            self.n_agents,
            self.params,
            self.device,
            self.discrete,
            self.n_parallel_envs,
            critic_type=self.critic_type,
            n_hyperedge_types=self.n_hyperedge_types,
            hyperedge_fns=(
                self.hyperedge_fns if self.critic_type == "multi_hgnn" else None
            ),
        )

        # Sync device — agent may have upgraded to CUDA in its __init__
        self.device = self.agent.device

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = defaultdict(list)

        # Timing statistics
        self.training_start_time = None
        self.total_training_time = 0.0

    def _build_inference_hypergraphs(self, obs, infos, n_envs):
        """Build batched block-diagonal hypergraphs for critic inference.

        Returns (None, None) when the critic doesn't need hypergraphs,
        otherwise (batched_hgs, per_env_sig_ids) where:
        - batched_hgs: list of dhg.Hypergraph (one per hyperedge type),
          each a block-diagonal graph batching all n_envs together.
        - per_env_sig_ids: list of int signature IDs, one per env,
          usable for buffer storage to avoid recomputation during update.
        """
        if self.critic_type != "multi_hgnn":
            return None, None

        # Compute per-env edge lists for each hyperedge type
        all_type_edge_lists = []  # [type_idx][env_idx] -> edge_list
        for fn, source in self.hyperedge_fns:
            if source == "obs":
                data = obs
            else:
                data = infos.get(source) if isinstance(infos, dict) else None
                if data is None:
                    continue
            all_type_edge_lists.append(
                [fn(data[e], self.n_agents) for e in range(n_envs)]
            )

        # Transpose to per-env: [env_idx] -> [type_edges_0, type_edges_1, ...]
        n_types = len(all_type_edge_lists)
        per_env_edge_lists = [
            [all_type_edge_lists[t][e] for t in range(n_types)] for e in range(n_envs)
        ]

        # Canonicalize and deduplicate via the shared agent-level cache
        per_env_sig_ids = []
        for edge_lists in per_env_edge_lists:
            sig = canonicalize_edge_lists(edge_lists)
            sig_id = self.agent.hg_signature_to_id.get(sig)
            if sig_id is None:
                sig_id = len(self.agent.hg_unique_edge_lists)
                self.agent.hg_signature_to_id[sig] = sig_id
                self.agent.hg_unique_edge_lists.append(edge_lists)
            per_env_sig_ids.append(sig_id)

        # Build batched hypergraphs, one per type, using the dhg cache
        batched_hgs = []
        for type_idx in range(n_types):
            type_edge_lists = [
                self.agent.hg_unique_edge_lists[sid][type_idx]
                for sid in per_env_sig_ids
            ]
            cache_key = tuple(per_env_sig_ids)
            cached = self.agent.hg_object_cache.get((type_idx, cache_key))
            if cached is not None:
                batched_hgs.append(cached)
            else:
                hg = batch_hypergraphs(
                    type_edge_lists, self.n_agents, device=self.device
                )
                self.agent.hg_object_cache[(type_idx, cache_key)] = hg
                batched_hgs.append(hg)

        return batched_hgs, per_env_sig_ids

    def collect_trajectory(self, max_steps):
        """
        Collect trajectory using Gymnasium's vectorized environments.

        Gymnasium VectorEnv provides:
        - obs: (n_envs, *obs_shape)
        - reward: (n_envs,)
        - terminated: (n_envs,)
        - truncated: (n_envs,)
        - info: dict with (n_envs,) arrays
        """

        # Reset environment with distinct seeds so parallel envs start differently
        train_seeds = [
            int(np.random.randint(0, 2**31)) for _ in range(self.n_parallel_envs)
        ]
        obs, infos = self.vec_env.reset(seed=train_seeds)
        batch_size = obs.shape[0]

        # Grab initial action masks if the env provides them (e.g. SMACv2)
        # Shape: (n_envs, n_agents, n_actions) or None
        current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None

        total_step_count = 0
        episode_count = 0
        current_episode_steps = np.zeros(self.n_parallel_envs, dtype=np.int32)

        while total_step_count <= max_steps:

            # Construct global states for each environment
            # Shape: (n_envs, n_agents * obs_dim)
            global_states = obs.reshape(batch_size, -1)

            # Build hypergraphs only when the critic requires them
            per_env_hgs, per_env_sig_ids = self._build_inference_hypergraphs(
                obs, infos, batch_size
            )

            # Single batched forward pass for all envs × agents
            actions_t, log_probs_t, values_t = self.agent.get_actions_batched(
                obs,
                global_states,
                deterministic=False,
                action_masks=current_masks,
                hypergraphs=per_env_hgs,
            )

            # Convert to numpy; shapes: (n_envs, n_agents, action_dim), (n_envs, n_agents), (n_envs, 1)
            actions_array = actions_t.cpu().numpy()
            log_probs_array = log_probs_t.cpu().numpy()
            values_array = values_t.cpu().numpy()

            # IMPORTANT: Reshape actions for discrete environments
            if self.discrete:
                # For discrete actions, squeeze the last dimension
                # Shape: (n_envs, n_agents, 1) -> (n_envs, n_agents)
                if actions_array.ndim == 3 and actions_array.shape[-1] == 1:
                    actions_array = actions_array.squeeze(-1)

                # Ensure integer type
                actions_array = actions_array.astype(np.int32)

            # Step all environments in parallel
            # Gymnasium VectorEnv step returns:
            # - next_obs: (n_envs, n_agents, obs_dim)
            # - rewards: (n_envs,)
            # - terminateds: (n_envs,)
            # - truncateds: (n_envs,)
            # - infos: dict with keys that have (n_envs,) shape
            next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(
                actions_array
            )

            # Compute dones for each environment
            dones = np.logical_or(terminateds, truncateds)

            # Store transitions for all environments in one vectorized call
            self.agent.store_transitions_batch(
                obs,
                global_states,
                actions_array,
                log_probs_array,
                values_array,
                rewards,
                dones,
                infos,
                action_masks=current_masks,
                hg_signature_ids=per_env_sig_ids,
            )

            # Update masks for next step.
            # For auto-reset envs (terminated), Gymnasium VecEnv merges the reset()
            # info into the regular infos dict, so avail_actions reflects the new episode.
            current_masks = (
                infos.get("avail_actions") if isinstance(infos, dict) else None
            )

            # Update for next iteration
            obs = next_obs
            total_step_count += batch_size
            current_episode_steps += 1

            # Track episode terminations with vectorized ops
            episode_count += int(dones.sum())
            current_episode_steps[dones] = 0

            # Gymnasium VectorEnv automatically resets terminated environments
            # The obs returned is already the reset observation for terminated envs

        # Compute final values for advantage computation in a single batched pass
        final_global_states = obs.reshape(batch_size, -1)
        with torch.no_grad():
            final_gs_tensor = torch.from_numpy(
                np.ascontiguousarray(final_global_states, dtype=np.float32)
            ).to(self.device)

            if self.agent.critic_type == "multi_hgnn":
                final_batched_hgs, _ = self._build_inference_hypergraphs(
                    obs, infos, batch_size
                )
                obs_tensor = torch.from_numpy(
                    np.ascontiguousarray(obs, dtype=np.float32)
                ).to(self.device)
                obs_flat = obs_tensor.reshape(batch_size * self.n_agents, -1)
                final_values = (
                    self.agent.network_old.get_value_batched(
                        obs_flat, final_batched_hgs, batch_size
                    )
                    .cpu()
                    .squeeze(-1)
                    .tolist()
                )
            else:
                final_values = (
                    self.agent.network_old.get_value(final_gs_tensor)
                    .cpu()
                    .squeeze(-1)
                    .tolist()
                )

        return total_step_count, episode_count, final_values

    def train(
        self,
        total_steps,
        batch_size,
        minibatches,
        epochs,
        checkpoint: bool = False,
        log_every: int = 10e3,
    ):
        """Train MAPPO agent"""

        if not checkpoint:
            self.training_stats["total_steps"] = []
            self.training_stats["reward"] = []
            self.training_stats["episodes"] = []
            self.training_stats["training_time"] = []
            self.training_stats["collection_time"] = []
            self.training_stats["update_time"] = []
            self.training_stats["eval_time"] = []

            self.total_training_time = 0
            resume_steps = 0
            resume_episodes = 0
        else:
            self.total_training_time = self.training_stats["training_time"][-1]
            resume_steps = self.training_stats["total_steps"][-1]
            resume_episodes = self.training_stats["episodes"][-1]

        # Track elapsed time across resumed runs so FPS remains correct with checkpoints.
        elapsed_time_offset = self.total_training_time if resume_steps > 0 else 0.0
        self.training_start_time = time.time() - elapsed_time_offset
        self.total_training_time = elapsed_time_offset

        steps_completed = resume_steps
        episodes_completed = resume_episodes

        if resume_steps > 0:
            print(f"Resuming MAPPO training from step {resume_steps}/{total_steps}...")
        else:
            print(
                f"Starting MAPPO training for {total_steps} total environment steps..."
            )

        while steps_completed < total_steps:
            steps_to_collect = min(batch_size, total_steps - steps_completed)

            # Time trajectory collection
            collection_start = time.time()
            step_count, episode_count, final_values = self.collect_trajectory(
                max_steps=int(steps_to_collect)
            )
            collection_time = time.time() - collection_start

            # Time agent update
            update_start = time.time()
            stats = self.agent.update(
                next_value=final_values,
                minibatch_size=batch_size // minibatches,
                epochs=epochs,
            )
            update_time = time.time() - update_start

            # Update tracking
            steps_completed += step_count
            episodes_completed += episode_count

            # Store statistics
            for key, value in stats.items():
                self.training_stats[key].append(value)

            # Time evaluation
            eval_start = time.time()
            eval_rewards = self.evaluate()
            eval_time = time.time() - eval_start

            # Calculate elapsed time
            elapsed_time = time.time() - self.training_start_time
            self.total_training_time = elapsed_time

            # Store timing statistics
            self.training_stats["total_steps"].append(steps_completed)
            self.training_stats["reward"].append(eval_rewards)
            self.training_stats["episodes"].append(episodes_completed)
            self.training_stats["training_time"].append(elapsed_time)
            self.training_stats["collection_time"].append(collection_time)
            self.training_stats["update_time"].append(update_time)
            self.training_stats["eval_time"].append(eval_time)

            # Calculate throughput
            steps_per_second = steps_completed / elapsed_time if elapsed_time > 0 else 0

            # Log progress
            if steps_completed % log_every < step_count:
                mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
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

                self.save_training_stats(
                    self.dirs["logs"] / "training_stats_checkpoint.pkl"
                )
                self.save_agent(self.dirs["models"] / "models_checkpoint.pth")

        # Close envs
        self.close_environments()

        # Final timing summary
        total_time = time.time() - self.training_start_time
        avg_collection_time = np.mean(self.training_stats["collection_time"])
        avg_update_time = np.mean(self.training_stats["update_time"])
        avg_eval_time = np.mean(self.training_stats["eval_time"])
        final_fps = steps_completed / total_time

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"{'='*60}")
        print(f"Total steps:          {steps_completed}")
        print(f"Total episodes:       {episodes_completed}")
        print(f"Total time:           {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"Final FPS:            {final_fps:.1f} steps/second")
        print(f"\nTime breakdown:")
        print(f"  Avg collection:     {avg_collection_time:.3f}s")
        print(f"  Avg update:         {avg_update_time:.3f}s")
        print(f"  Avg evaluation:     {avg_eval_time:.3f}s")
        print(f"{'='*60}\n")

    def evaluate(self):
        """Evaluate current policy using parallel episodes."""
        self.agent.network_old.eval()
        n_eps = self.n_eval_episodes

        with torch.no_grad():
            eval_seeds = [int(np.random.randint(0, 2**31)) for _ in range(n_eps)]
            obs, infos = self.eval_env.reset(seed=eval_seeds)
            current_masks = (
                infos.get("avail_actions") if isinstance(infos, dict) else None
            )
            episode_rewards = np.zeros(n_eps)
            finished = np.zeros(n_eps, dtype=bool)

            while not finished.all():
                global_states = obs.reshape(n_eps, -1)

                eval_hgs, _ = self._build_inference_hypergraphs(obs, infos, n_eps)

                actions_t, _, _ = self.agent.get_actions_batched(
                    obs,
                    global_states,
                    deterministic=True,
                    action_masks=current_masks,
                    hypergraphs=eval_hgs,
                )
                actions = actions_t.cpu().numpy()

                if self.discrete:
                    if actions.ndim == 3 and actions.shape[-1] == 1:
                        actions = actions.squeeze(-1)
                    actions = actions.astype(np.int32)

                obs, rewards, terminated, truncated, infos = self.eval_env.step(actions)
                current_masks = (
                    infos.get("avail_actions") if isinstance(infos, dict) else None
                )

                # Only accumulate rewards for episodes not yet finished
                dones = np.logical_or(terminated, truncated)
                episode_rewards[~finished] += rewards[~finished]
                finished |= dones

        self.agent.network_old.train()

        return episode_rewards.mean()

    def render(self):
        """Run one episode with the current policy in a render environment.

        Returns:
            rewards:      np.ndarray of shape (n_steps,), per-step rewards.
            entropy_logs: dict mapping hyperedge type ("proximity", "object")
                          to np.ndarray of shape (n_steps, 2) with
                          [S_e, S_normalized] per step.
        """
        render_env = self._make_render_env()

        self.agent.network_old.eval()
        episode_reward = []
        cum_sum = 0.0
        entropy_proximity_log = []
        entropy_object_log = []

        seed = int(np.random.randint(0, 2**31))
        obs, infos = render_env.reset(seed=seed)
        current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
        if current_masks is not None:
            current_masks = current_masks[np.newaxis]  # add env batch dim

        with torch.no_grad():
            while True:
                # Add batch dim expected by get_actions_batched: (1, n_agents, obs_dim)
                global_states = obs.reshape(1, -1)

                render_hgs, _ = self._build_inference_hypergraphs(obs, infos, 1)

                actions_t, _, _ = self.agent.get_actions_batched(
                    obs,
                    global_states,
                    deterministic=True,
                    action_masks=current_masks,
                    hypergraphs=render_hgs,
                )
                actions = actions_t.cpu().numpy()  # (n_agents, action_dim)

                if self.discrete:
                    if actions.ndim == 2 and actions.shape[-1] == 1:
                        actions = actions.squeeze(-1)
                    actions = actions.astype(np.int32)

                obs, reward, terminated, truncated, infos = render_env.step(actions)
                current_masks = (
                    infos.get("avail_actions") if isinstance(infos, dict) else None
                )
                if current_masks is not None:
                    current_masks = current_masks[np.newaxis]
                render_env.render()

                cum_sum += float(reward)
                episode_reward.append(cum_sum)

                # Hypergraph creation and entropy calculation
                entropies = compute_hyperedge_structural_entropy_batch(render_hgs)
                entropy_proximity_log.append(entropies[0])  # [S_e, S_normalized]
                entropy_object_log.append(entropies[1])

                if terminated or truncated:
                    break

        render_env.close()

        entropy_logs = {
            "proximity": np.array(entropy_proximity_log),
            "object": np.array(entropy_object_log),
        }
        return np.array(episode_reward), entropy_logs

    def _make_render_env(self):
        """Create a single env with render_mode='human' for the current env_name."""
        match self.env_name:
            case EnvironmentEnum.BOX2D_SALP:
                from environments.box2d_salp.domain import SalpChainEnv

                return SalpChainEnv(n_agents=self.n_agents, render_mode="human")

            case EnvironmentEnum.MULTI_BOX:
                from environments.multi_box_push.domain import MultiBoxPushEnv

                render_env = make_vec_env(
                    self.env_name,
                    self.n_agents,
                    1,
                    use_async=True,  # Use parallel processing
                    env_variant=self.env_variant,
                )

                render_env.envs[0].render_mode = "human"

                return render_env

            case EnvironmentEnum.MPE_SPREAD:
                from mpe2 import simple_spread_v3
                from algorithms.create_env import PettingZooToGymWrapper

                pz_env = simple_spread_v3.parallel_env(
                    N=self.n_agents,
                    local_ratio=0.5,
                    max_cycles=25,
                    continuous_actions=False,
                    dynamic_rescaling=True,
                    render_mode="human",
                )
                return PettingZooToGymWrapper(pz_env)
            case EnvironmentEnum.MPE_SIMPLE:
                from mpe2 import simple_v3
                from algorithms.create_env import PettingZooToGymWrapper

                pz_env = simple_v3.parallel_env(
                    max_cycles=25,
                    continuous_actions=False,
                    render_mode="human",
                )
                return PettingZooToGymWrapper(pz_env)
            case EnvironmentEnum.SMACLITE:
                from environments.smaclite.wrapper import SmacliteToGymWrapper

                return SmacliteToGymWrapper(map_name=self.env_variant)
            case _:
                return None

    def close_environments(self):
        """Properly close all vectorized environments"""
        self.vec_env.close()
        self.eval_env.close()

    def __del__(self):
        """Destructor to ensure environments are closed"""
        self.close_environments()

    def save_agent(self, path):
        """Save MAPPO agent weights, optimizer state, and RNG states."""
        torch.save(
            {
                "network": self.agent.network_old.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch_cpu": torch.random.get_rng_state(),
                "rng_torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            path,
        )

    def load_agent(self, filepath, restore_rng=False):
        checkpoint = torch.load(filepath, map_location=self.device)

        self.agent.network_old.load_state_dict(checkpoint["network"])
        self.agent.network.load_state_dict(checkpoint["network"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])

        if restore_rng and "rng_python" in checkpoint:
            random.setstate(checkpoint["rng_python"])
            np.random.set_state(checkpoint["rng_numpy"])
            torch.random.set_rng_state(checkpoint["rng_torch_cpu"].cpu())
            if torch.cuda.is_available() and checkpoint["rng_torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(
                    [s.cpu() for s in checkpoint["rng_torch_cuda"]]
                )

        print(f"Agents loaded from {filepath}")

    def load_checkpoint_progress(self, path):
        """Load training progress counters from training stats checkpoint."""

        with path.open("rb") as f:
            stats = pickle.load(f)

        self.training_stats["total_steps"] = stats.get("total_steps", [])
        self.training_stats["reward"] = stats.get("reward", [])
        self.training_stats["episodes"] = stats.get("episodes", [])
        self.training_stats["training_time"] = stats.get("training_time", [])
        self.training_stats["collection_time"] = stats.get("collection_time", [])
        self.training_stats["update_time"] = stats.get("update_time", [])
        self.training_stats["eval_time"] = stats.get("eval_time", [])

    def save_training_stats(self, path):
        """Save training statistics"""
        with open(path, "wb") as f:
            pickle.dump(dict(self.training_stats), f)
