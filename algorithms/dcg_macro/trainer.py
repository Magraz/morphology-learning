"""DCG trainer: drives the vendored PyMARL DCG stack against the framework's
Gymnasium vectorized envs.

DCG is off-policy episodic Q-learning, so this trainer keeps DCG's replay
buffer + learner and only replaces PyMARL's env harness: episodes are collected
from a Gymnasium ``AsyncVectorEnv`` (built by the shared ``make_vec_env``) and
packed into DCG's ``EpisodeBatch``. Everything downstream (message passing,
double-Q targets, target-network updates) is the unmodified vendored code.
"""

import random
import time

import gymnasium as gym
import numpy as np
import torch as th

from algorithms.create_env import make_vec_env
from algorithms.dcg import _vendor  # noqa: F401  (puts src/ on sys.path)
from algorithms.dcg.args_builder import build_args
from algorithms.dcg.logger_shim import DCGLogger
from algorithms.dcg.types import DCG_Model_Params, DCG_Params

from components.episode_buffer import EpisodeBatch, ReplayBuffer  # noqa: E402
from components.transforms import OneHot  # noqa: E402
from controllers.dcg_controller import DeepCoordinationGraphMAC  # noqa: E402
from learners.dcg_learner import DCGLearner  # noqa: E402


class DCGTrainer:
    def __init__(
        self,
        params: DCG_Params,
        dirs: dict,
        device: str = "cpu",
        env_params: dict = None,
        model_params: DCG_Model_Params = None,
    ):
        self.params = params
        self.model_params = model_params
        self.dirs = dirs
        self.device = device
        self.env_params = env_params

        self.episode_limit = params.episode_limit
        self.test_interval = params.test_interval
        self.test_nepisode = params.test_nepisode
        self.save_model_interval = params.save_model_interval
        self.log_interval = params.log_interval

        # --- Environments (shared factory) ---
        self.n_envs = env_params.get("n_envs")
        self.n_eval_envs = 5
        self.vec_env = make_vec_env(
            env_params.get("environment"),
            env_params.get("n_agents"),
            self.n_envs,
            use_async=True,
            env_params=env_params,
        )
        self.eval_env = make_vec_env(
            env_params.get("environment"),
            env_params.get("n_agents"),
            self.n_eval_envs,
            use_async=True,
            env_params=env_params,
        )

        # --- Env-derived dimensions ---
        obs_space = self.vec_env.single_observation_space
        act_space = self.vec_env.single_action_space

        if not isinstance(act_space, gym.spaces.MultiDiscrete):
            raise ValueError(
                "DCG requires a discrete (MultiDiscrete) action space; got "
                f"{type(act_space).__name__}. DCG supports discrete envs only "
                "(e.g. smaclite / smacv2)."
            )
        self.n_agents = int(obs_space.shape[0])
        self.obs_dim = int(obs_space.shape[1])
        self.n_actions = int(act_space.nvec[0])
        self.state_dim = self.obs_dim * self.n_agents

        # --- Flat args for the vendored modules ---
        self.args = build_args(
            params,
            model_params,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            device=self.device,
            batch_size_run=self.n_envs,
        )

        # --- Replay buffer scheme (mirrors PyMARL run.py) ---
        self.groups = {"agents": self.n_agents}
        self.preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=self.n_actions)])}
        scheme = {
            "state": {"vshape": self.state_dim},
            "obs": {"vshape": self.obs_dim, "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (self.n_actions,),
                "group": "agents",
                "dtype": th.int,
            },
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        # Keep the raw scheme for building fresh EpisodeBatches: EpisodeBatch
        # copies+augments its scheme in place (adding "actions_onehot" and the
        # reserved "filled" key), so reusing buffer.scheme would trip the
        # "filled is reserved" guard. The mac still gets buffer.scheme, which
        # carries the "actions_onehot" field its input builder needs.
        self.scheme = scheme
        self.buffer = ReplayBuffer(
            scheme,
            self.groups,
            params.buffer_size,
            self.episode_limit + 1,
            preprocess=self.preprocess,
            device=self.device,
        )

        # --- Controller + learner (unmodified vendored DCG) ---
        self.logger = DCGLogger()
        self.mac = DeepCoordinationGraphMAC(self.buffer.scheme, self.groups, self.args)
        self.learner = DCGLearner(self.mac, self.buffer.scheme, self.logger, self.args)
        if self.args.use_cuda:
            self.learner.cuda()

        # --- Training progress ---
        self.t_env = 0
        self.episode = 0
        self._stats = _StatsBook()

    # ================================================================= collect

    def _get_avail(self, infos, n_envs):
        """Batched avail-action mask from the vec-env info dict, or all-ones."""
        if isinstance(infos, dict) and infos.get("avail_actions") is not None:
            return np.asarray(infos["avail_actions"], dtype=np.float32)
        return np.ones((n_envs, self.n_agents, self.n_actions), dtype=np.float32)

    def _collect(self, test_mode: bool = False):
        """Run one lockstep batch of episodes into a fresh ``EpisodeBatch``.

        Mirrors PyMARL's ``parallel_runner``: all envs step together, each env
        is frozen the moment it terminates/truncates (so no post-reset data
        leaks into its episode), and the batch keeps ``episode_limit + 1``
        timesteps. Under Gymnasium's default ``NEXT_STEP`` autoreset the
        terminal observation is returned at the done step, so the stored next
        state is correct for bootstrapping on time-limit truncation.
        """
        env = self.eval_env if test_mode else self.vec_env
        n_envs = self.n_eval_envs if test_mode else self.n_envs

        batch = EpisodeBatch(
            self.scheme,
            self.groups,
            n_envs,
            self.episode_limit + 1,
            preprocess=self.preprocess,
            device=self.device,
        )

        seeds = [int(np.random.randint(0, 2**31 - 1)) for _ in range(n_envs)]
        obs, infos = env.reset(seed=seeds)
        # ``cur_avail`` tracks the *fresh* avail mask for every env each step
        # (including frozen ones, which auto-reset into new episodes). It backs
        # a valid stepping action for envs we no longer record.
        cur_avail = self._get_avail(infos, n_envs)
        self.mac.init_hidden(batch_size=n_envs)

        batch.update(
            {"state": obs.reshape(n_envs, -1), "obs": obs, "avail_actions": cur_avail},
            ts=0,
        )

        done = np.zeros(n_envs, dtype=bool)
        episode_returns = np.zeros(n_envs, dtype=np.float64)
        env_steps = 0
        t = 0

        while not done.all() and t < self.episode_limit:
            active = np.nonzero(~done)[0]
            active_list = active.tolist()

            actions = self.mac.select_actions(
                batch, t_ep=t, t_env=self.t_env, bs=active_list, test_mode=test_mode
            )
            batch.update(
                {"actions": actions.unsqueeze(1)},
                bs=active_list,
                ts=t,
                mark_filled=False,
            )

            # Step every env, but only override the *active* envs with the mac's
            # actions. Frozen envs (already done, now cycling through their own
            # auto-reset episodes) get the first available action from their
            # fresh mask — always valid, and their results are discarded.
            full_actions = cur_avail.argmax(axis=-1).astype(np.int64)
            full_actions[active] = actions.detach().cpu().numpy()
            next_obs, rewards, terms, truncs, infos = env.step(full_actions)
            cur_avail = self._get_avail(infos, n_envs)

            r = np.asarray(rewards, dtype=np.float32)[active]
            # Store true termination only: a time-limit truncation keeps
            # terminated=0 so its TD target still bootstraps (PyMARL semantics).
            term_real = np.asarray(terms, dtype=np.float32)[active]
            batch.update(
                {"reward": r.reshape(-1, 1), "terminated": term_real.reshape(-1, 1)},
                bs=active_list,
                ts=t,
                mark_filled=False,
            )
            episode_returns[active] += r
            env_steps += len(active)

            t += 1
            batch.update(
                {
                    "state": next_obs[active].reshape(len(active), -1),
                    "obs": next_obs[active],
                    "avail_actions": cur_avail[active],
                },
                bs=active_list,
                ts=t,
                mark_filled=True,
            )

            done[active] = np.asarray(terms)[active] | np.asarray(truncs)[active]

        if not test_mode:
            self.t_env += env_steps
        return batch, episode_returns, env_steps

    # =================================================================== train

    def train(self, total_steps, batch_size, checkpoint: bool = False):
        if not checkpoint:
            self.t_env = 0
            self.episode = 0

        last_test = -self.test_interval - 1
        last_save = self.t_env
        last_log = -self.log_interval - 1

        while self.t_env <= total_steps:
            t0 = time.time()
            batch, returns, _ = self._collect(test_mode=False)
            self.buffer.insert_episode_batch(batch)
            self.episode += self.n_envs
            collect_time = time.time() - t0

            update_time = 0.0
            if self.buffer.can_sample(batch_size):
                tu = time.time()
                sample = self.buffer.sample(batch_size)
                max_ep_t = sample.max_t_filled()
                sample = sample[:, :max_ep_t]
                if sample.device != self.device:
                    sample.to(self.device)
                self.learner.train(sample, self.t_env, self.episode)
                update_time = time.time() - tu

            train_return = float(np.mean(returns))

            eval_time = 0.0
            eval_return = None
            if (self.t_env - last_test) >= self.test_interval:
                te = time.time()
                eval_return = self._evaluate()
                eval_time = time.time() - te
                last_test = self.t_env

            self._stats.record(
                steps=self.t_env,
                episodes=self.episode,
                reward=train_return,
                eval_reward=eval_return,
                collection_time=collect_time,
                update_time=update_time,
                eval_time=eval_time,
            )

            if (self.t_env - last_log) >= self.log_interval:
                msg = (
                    f"[dcg] steps={self.t_env} episodes={self.episode} "
                    f"train_return={train_return:.3f} "
                    f"eps={self.mac.action_selector.epsilon:.3f}"
                )
                if eval_return is not None:
                    msg += f" eval_return={eval_return:.3f}"
                print(msg)
                last_log = self.t_env

            if (self.t_env - last_save) >= self.save_model_interval:
                self.save_agent(self.dirs["models"] / "models_checkpoint.pth")
                self.save_training_stats(
                    self.dirs["logs"] / "training_stats_checkpoint.pkl"
                )
                last_save = self.t_env

    def _evaluate(self) -> float:
        n_runs = max(1, self.test_nepisode // self.n_eval_envs)
        returns = []
        for _ in range(n_runs):
            _, ep_returns, _ = self._collect(test_mode=True)
            returns.extend(ep_returns.tolist())
        return float(np.mean(returns))

    # =============================================================== render/io

    def render(self, capture_video: bool = False):
        """Run one greedy eval episode; returns (per-step team rewards,)."""
        _, ep_returns, _ = self._collect(test_mode=True)
        return np.asarray(ep_returns, dtype=np.float32)

    def save_agent(self, path) -> None:
        ckpt = {
            "mac_agent": self.mac.agent.state_dict(),
            "utility_fun": self.mac.utility_fun.state_dict(),
            "payoff_fun": self.mac.payoff_fun.state_dict(),
            "optimiser": self.learner.optimiser.state_dict(),
            "t_env": self.t_env,
            "episode": self.episode,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch_cpu": th.random.get_rng_state(),
        }
        if self.mac.duelling:
            ckpt["state_value"] = self.mac.state_value.state_dict()
        th.save(ckpt, path)

    def load_agent(self, path, restore_rng: bool = False) -> None:
        ckpt = th.load(path, map_location=self.device, weights_only=False)
        self.mac.agent.load_state_dict(ckpt["mac_agent"])
        self.mac.utility_fun.load_state_dict(ckpt["utility_fun"])
        self.mac.payoff_fun.load_state_dict(ckpt["payoff_fun"])
        if self.mac.duelling and "state_value" in ckpt:
            self.mac.state_value.load_state_dict(ckpt["state_value"])
        self.learner.optimiser.load_state_dict(ckpt["optimiser"])
        # Keep the target network in sync with the loaded live network.
        self.learner.target_mac.load_state(self.mac)
        self.t_env = ckpt.get("t_env", 0)
        self.episode = ckpt.get("episode", 0)
        if restore_rng and "rng_python" in ckpt:
            random.setstate(ckpt["rng_python"])
            np.random.set_state(ckpt["rng_numpy"])
            th.random.set_rng_state(ckpt["rng_torch_cpu"].cpu())

    def save_training_stats(self, path) -> None:
        self._stats.save(path)

    def load_checkpoint_progress(self, path) -> None:
        self._stats.load(path)
        self.t_env = int(self._stats.last("total_steps", 0))
        self.episode = int(self._stats.last("episodes", 0))

    def close_environments(self) -> None:
        for env in (getattr(self, "vec_env", None), getattr(self, "eval_env", None)):
            if env is not None:
                env.close()


class _StatsBook:
    """Small training-stats recorder, pickle-compatible with the plotting code."""

    KEYS = (
        "total_steps",
        "reward",
        "eval_reward",
        "episodes",
        "training_time",
        "collection_time",
        "update_time",
        "eval_time",
    )

    def __init__(self):
        self.stats = {k: [] for k in self.KEYS}
        self._t0 = time.time()

    def record(
        self,
        *,
        steps,
        episodes,
        reward,
        eval_reward,
        collection_time,
        update_time,
        eval_time,
    ) -> None:
        self.stats["total_steps"].append(steps)
        self.stats["reward"].append(reward)
        self.stats["eval_reward"].append(
            eval_reward if eval_reward is not None else reward
        )
        self.stats["episodes"].append(episodes)
        self.stats["training_time"].append(time.time() - self._t0)
        self.stats["collection_time"].append(collection_time)
        self.stats["update_time"].append(update_time)
        self.stats["eval_time"].append(eval_time)

    def last(self, key, default=0):
        vals = self.stats.get(key, [])
        return vals[-1] if vals else default

    def save(self, path) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.stats, f)

    def load(self, path) -> None:
        import pickle

        with open(path, "rb") as f:
            loaded = pickle.load(f)
        for k in self.KEYS:
            self.stats[k] = loaded.get(k, [])
