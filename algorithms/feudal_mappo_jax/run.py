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
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes, to_bytes

from algorithms.feudal_mappo_jax.manager import (
    goal_concentration,
    mean_goal_direction,
    training_goal_variants,
)
from algorithms.feudal_mappo_jax.mappo import (
    create_train_state,
    resolve_goal_space,
    validate_worker_objective,
    validate_worker_encoder,
)
from algorithms.feudal_mappo_jax.trainer import (
    RunnerState,
    global_state_dim,
    make_train,
)
from algorithms.feudal_mappo_jax.types import (
    Experiment,
    MAPPOConfig,
    Model_Params,
    Params,
)
from environments.mjx_suite.multi_box_multi_goal_push_mjx import (
    MultiBoxMultiGoalPushMJX,
)
from environments.mjx_suite.multi_box_push_mjx import MultiBoxPushMJX
from environments.types import EnvironmentEnum


def set_seeds(seed: int):
    """Set Python and NumPy seeds. JAX uses explicit PRNG keys."""
    random.seed(seed)
    np.random.seed(seed)


def _paired_bootstrap_ci(diff, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI for the mean of a PAIRED per-episode difference.

    Paired because every variant block ran episode `j` from the same reset
    state, so `diff[j]` already differences out the episode's difficulty. A
    two-sample CI over independent means would be far wider and would need many
    more episodes to resolve the same gap.

    Resample the differences, not the two arms separately — that is what keeps
    the pairing.
    """
    diff = np.asarray(diff, dtype=np.float64)
    if diff.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def _direction_count(goal):
    """`mappo._agent_direction_count` on an arbitrary goal tensor, as a float."""
    from algorithms.feudal_mappo_jax.mappo import _agent_direction_count, _agent_gram

    return float(_agent_direction_count(_agent_gram(goal)))


class Feudal_MAPPO_JAX_Runner:
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

        # Create the functional MJX environment. Three supported env groups:
        #   MULTI_BOX_MJX            — continuous force control (the base env)
        #   MULTI_BOX_MULTI_GOAL_MJX — the same task on a circular arena with one
        #                   concentric goal ring per box (box j -> ring j from the
        #                   center out); same 40-dim obs, action space and reward
        #                   modes, so nothing downstream changes
        #   MACRO_MJX     — the hierarchical macro layer (discrete skill choice,
        #                   one decision per macro_len low-level steps), which
        #                   wraps a base MultiBoxPushMJX.
        # `coupling_def` is forwarded to every branch that builds a box-push env
        # (bare, multi-goal, and the macro wrapper's base). Default "even" ==
        # n_agents // n_objects for every box, i.e. exactly what each group got
        # before this was reachable — so no existing arm changes. "random" draws
        # per-box requirements in [2, n_agents//2] from a FIXED rng(42), which
        # also resizes the boxes (`box_half_extents = max(1.5, coupling*0.4)`)
        # and, in the multi-goal env, the goal rings derived from them.
        environment = env_config.get("environment")
        reward_mode = env_config.get("reward_mode", "dense")

        if environment == EnvironmentEnum.MULTI_BOX_MJX:
            self.env = MultiBoxPushMJX(
                n_agents=env_config.get("n_agents"),
                n_objects=env_config.get("n_objects"),
                reward_mode=reward_mode,
                variant=env_config.get("variant"),
                coupling_def=env_config.get("coupling_def", "even"),
                use_global_state=env_config.get("use_global_state", False),
            )
        elif environment == EnvironmentEnum.MULTI_BOX_MULTI_GOAL_MJX:
            self.env = MultiBoxMultiGoalPushMJX(
                n_agents=env_config.get("n_agents"),
                n_objects=env_config.get("n_objects"),
                reward_mode=reward_mode,
                boundary_ends_episode=env_config.get("boundary_ends_episode", False),
                coupling_def=env_config.get("coupling_def", "even"),
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
                if reward_mode
                in (WINDOWED_DIFFERENCE_REWARDS, ALIGNED_WINDOWED_DIFFERENCE_REWARDS)
                else reward_mode
            )
            base_env = MultiBoxPushMJX(
                n_agents=env_config.get("n_agents"),
                n_objects=env_config.get("n_objects"),
                reward_mode=base_reward_mode,
                variant=env_config.get("variant"),
                coupling_def=env_config.get("coupling_def", "even"),
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
        elif environment == EnvironmentEnum.SMAX:
            # JaxMARL StarCraft, behind an adapter that presents the same functional
            # array contract as the MJX envs. It brings three things the MJX envs do
            # not: legal-action masks, a real global state (which the manager reads as
            # its joint state), and units that die mid-episode (masked out of the loss
            # via info["active"]).
            # `max_steps` is the benchmark's own episode limit, NOT params.n_steps —
            # unlike the MJX branches above, the two are independent here.
            from environments.smax.smax_env import SMAXAdapter

            self.env = SMAXAdapter(
                map_name=env_config.get("env_variant", "3m"),
                smax_env_id=env_config.get("smax_env_id", "HeuristicEnemySMAX"),
                max_steps=env_config.get("max_steps"),
                walls_cause_death=env_config.get("walls_cause_death", True),
                use_self_play_reward=env_config.get("use_self_play_reward", False),
            )
        else:
            raise ValueError(
                f"feudal_mappo_jax supports only '{EnvironmentEnum.MULTI_BOX_MJX}', "
                f"'{EnvironmentEnum.MULTI_BOX_MULTI_GOAL_MJX}', "
                f"'{EnvironmentEnum.MACRO_MJX}' and '{EnvironmentEnum.SMAX}' "
                f"(functional JAX API); got {environment!r}"
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
            n_eval_episodes=self.params.n_eval_episodes,
            goal_permute_shift=self.params.goal_permute_shift,
            eval_goal_variants=self.params.eval_goal_variants,
            # ---- FeUdal ----
            goal_dim=self.model_params.goal_dim,
            goal_horizon=self.params.goal_horizon,
            intrinsic_coef=self.params.intrinsic_coef,
            intrinsic_anneal=self.params.intrinsic_anneal,
            manager_lr=self.params.manager_lr,
            manager_val_coef=self.params.manager_val_coef,
            manager_gamma=self.params.manager_gamma,
            n_manager_epochs=self.params.n_manager_epochs,
            n_manager_critic_epochs=self.params.n_manager_critic_epochs,
            manager_hidden_dim=self.model_params.manager_hidden_dim,
            manager_core=self.model_params.manager_core,
            manager_latent=self.model_params.manager_latent,
            manager_latent_dim=self.model_params.manager_latent_dim,
            goal_space=self.model_params.goal_space,
            waypoint_radius=self.model_params.waypoint_radius,
            goal_embed_dim=self.model_params.goal_embed_dim,
            normalize_pooled_goal=self.model_params.normalize_pooled_goal,
            zero_goal=self.model_params.zero_goal,
            worker_fusion=self.model_params.worker_fusion,
            worker_objective=self.model_params.worker_objective,
            worker_encoder=self.model_params.worker_encoder,
        )

        # Fail loudly rather than open. A zero-goal worker cannot see the goals,
        # so rewarding it for achieving them (r^I = d_cos(s_t - s_{t-i}, g_{t-i}))
        # is not an ablation of anything — it is an uninterpretable arm that
        # would still train, log healthy diagnostics, and look like a result.
        if self.config.worker_fusion not in ("concat", "film"):
            raise ValueError(
                f"model_params.worker_fusion={self.config.worker_fusion!r} is not "
                "'concat' or 'film'. Caught here rather than at the first forward "
                "pass so a typo fails before a run is launched."
            )

        if self.config.zero_goal and self.config.intrinsic_coef != 0.0:
            raise ValueError(
                "model_params.zero_goal=True is incompatible with "
                f"params.intrinsic_coef={self.config.intrinsic_coef} (!= 0): the "
                "worker cannot see the goals it would be rewarded for reaching. "
                "The zero-goal ablation is defined at alpha=0 — use "
                "model=feudal_zerogoal."
            )

        # Every worker_objective rule lives in ONE function, shared with
        # `trainer.make_train`, so the two cannot drift. It raises on the
        # incoherent combinations and warns on a non-local manager latent.
        validate_worker_objective(self.config)
        # Likewise for `worker_encoder`: raises when the arm cannot be built (no
        # `f_enc` on a centralized latent) or would apply a stale gradient
        # (n_manager_epochs > 1), warns on concat fusion and on intrinsic_only.
        validate_worker_encoder(self.config)
        # Likewise for `goal_space`, which also DERIVES `goal_dim` from the env
        # when the goal is grounded. Assigned back, so the banner below and the
        # runner both report the width that will actually be built — a stale yaml
        # `goal_dim` otherwise trains the worker on a wrong-width directive for a
        # full update before anything raises.
        self.config = resolve_goal_space(self.config, self.env)

        print(
            f"FeUdal MAPPO | env={environment} | n_envs={self.config.n_envs} | "
            f"n_steps={self.config.n_steps} | total={self.config.n_total_steps} | "
            f"reward_mode={self.env.reward_mode} | "
            f"backend={jax.default_backend()}"
        )
        # Nothing validates the `env:` block (CLAUDE.md), so a misspelled or
        # misindented `use_global_state` would train the plain baseline under a
        # name asserting otherwise. Print what actually resolved.
        print(
            f"  centralized input: gs_dim={global_state_dim(self.env)} "
            f"(env hook: {hasattr(self.env, 'global_state')})"
        )
        print(
            f"  manager: core={self.config.manager_core} "
            f"goal_dim={self.config.goal_dim} c={self.config.goal_horizon} "
            f"hidden={self.config.manager_hidden_dim} "
            f"latent={self.config.manager_latent} | "
            f"alpha(intrinsic)={self.config.intrinsic_coef}"
        )
        # What the goal MEANS. Printed unconditionally for the same reason the
        # centralized-input line is: nothing validates the `env:` block, and a
        # grounded arm whose flag did not land would train as plain latent feudal
        # under a name claiming otherwise.
        _grounded = self.config.goal_space != "latent"
        print(
            f"  goal space: {self.config.goal_space} "
            f"(s = {'env.goal_state, 0 params' if _grounded else 'learned f_Mspace'}) | "
            f"goal_dim={self.config.goal_dim} "
            f"latent_dim={self.config.manager_latent_dim or self.config.goal_dim}"
            + (
                f" | R={self.config.waypoint_radius:.4g} of world extent"
                if self.config.goal_space == "position_waypoint"
                else ""
            )
        )
        # Say what the WORKER is optimizing. Under 'intrinsic_only' alpha is
        # inert, so the line above would otherwise be the only intrinsic
        # information printed and would read as if alpha were weighting anything.
        print(
            f"  worker: fusion={self.config.worker_fusion} "
            f"objective={self.config.worker_objective}"
            + (
                "  (adv = adv_int only — extrinsic advantage NOT in the actor "
                "loss; all task pressure is the manager's)"
                if self.config.worker_objective == "intrinsic_only"
                else "  (adv = adv_ext + alpha*adv_int)"
            )
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

        # Goal-ablation variants ride the SAME scan as the real eval (one call,
        # V*n_eval_episodes vmapped width), so they cost far less than running
        # them as separate sequential scans. NOT free, though: measured on
        # `mjx_16a_4o_512/feudal`, 3 variants cost 1.91x one eval (~+8.5% wall
        # on a production run), so the 4th block here is worth budgeting for.
        # `eval_fn` returns the real block's mean unchanged, so the `reward`
        # series is untouched.
        eval_variants = training_goal_variants(
            self.env.n_agents, self.config.eval_goal_variants
        )
        # Carried forward between evals exactly as `eval_reward` is, and
        # appended EVERY iteration — appending only on eval iterations would
        # misalign these series against `total_steps` within a single run,
        # before resume even enters the picture.
        #
        # NaN, not 0.0, for the pre-first-eval fill: a 0.0 gap reads as a
        # measured "the goals make no difference", which is precisely the
        # finding this series exists to detect.
        ablation_series = {
            key: (
                stats_tracker.training_stats[key][-1]
                if checkpoint_loaded and stats_tracker.training_stats.get(key)
                else float("nan")
            )
            for key in (
                "eval_reward_permuted",
                "eval_reward_constant",
                "eval_reward_zeroed",
                "eval_gap_permuted",
                "eval_gap_constant",
                "eval_gap_zeroed",
                "eval_len_real",
                "eval_len_permuted",
                "eval_len_constant",
                "eval_len_zeroed",
            )
        }

        for update in range(start_update, num_updates):
            collection_start = time.time()
            runner_state, trajectory, last_value, rollout_stats = collect_fn(
                runner_state
            )
            jax.block_until_ready(last_value)
            collection_time = time.time() - collection_start

            update_start = time.time()
            # Fraction of training elapsed, for the alpha anneal. Passed as a
            # TRACED scalar — a python float would be baked in as a constant and
            # recompile the jitted update every iteration. Resume is correct for
            # free: `start_update` is derived from the restored step count.
            progress = jnp.float32(update / max(num_updates, 1))
            runner_state, losses = update_fn(
                runner_state, trajectory, last_value, progress
            )
            jax.block_until_ready(losses)
            update_time = time.time() - update_start

            eval_time = 0.0
            if update % eval_every == 0 or update == num_updates - 1:
                eval_start = time.time()
                # ONE split, ONE call: every variant block starts from the same
                # reset keys, so the gaps below are paired per episode and carry
                # no reset variance.
                eval_rng, eval_key = jax.random.split(eval_rng)
                # The `constant` block needs one vector to stand in for the whole
                # manager. Taken from the rollout that JUST ran, so it is an
                # in-distribution direction for the current policy and costs no
                # extra forward pass. It drifts slowly between evals (it is a
                # mean over n_steps*n_envs*n_agents goals), which is why the
                # offline probe re-derives it rather than reading this series
                # when an exactly reproducible number is wanted.
                ev_rewards, ev_lengths = eval_fn(
                    runner_state.train_state,
                    eval_key,
                    variants=eval_variants,
                    detail=True,
                    constant_goal=mean_goal_direction(trajectory.pooled_goal),
                )
                ev_rewards = np.asarray(ev_rewards)
                ev_lengths = np.asarray(ev_lengths)
                by_variant = dict(zip(eval_variants, ev_rewards))
                eval_reward = float(by_variant["real"].mean())
                ablation_series["eval_len_real"] = float(ev_lengths[0].mean())
                for variant in ("permuted", "constant", "zeroed"):
                    if variant not in by_variant:
                        continue
                    idx = eval_variants.index(variant)
                    ablation_series[f"eval_reward_{variant}"] = float(
                        by_variant[variant].mean()
                    )
                    ablation_series[f"eval_gap_{variant}"] = float(
                        (by_variant["real"] - by_variant[variant]).mean()
                    )
                    ablation_series[f"eval_len_{variant}"] = float(
                        ev_lengths[idx].mean()
                    )
                eval_time = time.time() - eval_start

            steps_completed += steps_per_update
            episodes_completed += int(rollout_stats["episode_count"])

            # Rollout-level scalars ride the same per-update stats path as the
            # losses (TrainingStatsTracker copies unknown keys through, so the
            # plotting notebooks are unaffected). `episode_count` is excluded —
            # it is already accumulated into `episodes_completed` above — and
            # `mean_reward` is renamed so it cannot collide with the `reward`
            # series, which is the deterministic EVAL return.
            rollout_series = {
                "train_reward": float(rollout_stats["mean_reward"]),
                "intrinsic_reward": float(rollout_stats["intrinsic_reward"]),
                "intrinsic_reward_abs": float(rollout_stats["intrinsic_reward_abs"]),
            }
            stats_tracker.append_agent_stats(
                {key: float(value) for key, value in losses.items()}
                | rollout_series
                # Appended every iteration, carrying the last eval forward — the
                # same staircase `reward` already forms, so these stay index-
                # aligned with `total_steps`.
                | dict(ablation_series)
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

    # NOTE: the four msgpack sites below must be kept in lockstep — `from_bytes`
    # needs an exactly-shaped target tree, so adding a component to one save site
    # without the matching restore site breaks resume loudly.
    #
    # V^I (the intrinsic critic) is included ONLY when alpha != 0, which is
    # exactly when `create_train_state` builds it. That keeps the alpha=0 file
    # format byte-for-byte what it was before the intrinsic stream existed, so
    # existing `feudal_a0` checkpoints still resume. All five sites route the
    # decision through `_intrinsic_tree` so they cannot disagree.

    @staticmethod
    def _intrinsic_tree(train_state, key, params_only):
        """`{key: <V^I subtree>}` when V^I exists, else `{}`."""
        ts = train_state.intrinsic_critic_ts
        if ts is None:
            return {}
        return {key: ts.params if params_only else ts}

    def save_params(self, train_state, path):
        """Save worker + critic + manager + manager-critic (+ V^I) params."""
        params_dict = {
            "actor": train_state.actor_ts.params,
            "critic": train_state.critic_ts.params,
            "manager": train_state.manager_ts.params,
            "manager_critic": train_state.manager_critic_ts.params,
            **self._intrinsic_tree(train_state, "intrinsic_critic", True),
        }
        with open(path, "wb") as f:
            f.write(to_bytes(params_dict))

    def _save_train_checkpoint(self, runner_state, eval_rng, path):
        """Full resumable training state: params + optimizer states + step
        counters (the TrainStates serialize as {step, params, opt_state}) and
        both RNG chains. Progress counters live in the stats checkpoint."""
        ts = runner_state.train_state
        checkpoint = {
            "actor_ts": ts.actor_ts,
            "critic_ts": ts.critic_ts,
            "manager_ts": ts.manager_ts,
            "manager_critic_ts": ts.manager_critic_ts,
            **self._intrinsic_tree(ts, "intrinsic_critic_ts", False),
            "rng": runner_state.rng,
            "eval_rng": eval_rng,
        }
        with open(path, "wb") as f:
            f.write(to_bytes(checkpoint))

    def _load_train_checkpoint(self, runner_state, eval_rng, path):
        """Restore a _save_train_checkpoint file into a fresh RunnerState."""
        ts = runner_state.train_state
        target = {
            "actor_ts": ts.actor_ts,
            "critic_ts": ts.critic_ts,
            "manager_ts": ts.manager_ts,
            "manager_critic_ts": ts.manager_critic_ts,
            **self._intrinsic_tree(ts, "intrinsic_critic_ts", False),
            "rng": runner_state.rng,
            "eval_rng": eval_rng,
        }
        with open(path, "rb") as f:
            loaded = from_bytes(target, f.read())
        print(f"Train state loaded from {path}")
        return (
            RunnerState(
                train_state=ts._replace(
                    actor_ts=loaded["actor_ts"],
                    critic_ts=loaded["critic_ts"],
                    manager_ts=loaded["manager_ts"],
                    manager_critic_ts=loaded["manager_critic_ts"],
                    intrinsic_critic_ts=loaded.get(
                        "intrinsic_critic_ts", ts.intrinsic_critic_ts
                    ),
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
            global_state_dim(self.env),
            self.env.action_dim,
            discrete=getattr(self.env, "discrete", False),
            n_agents=self.env.n_agents,
            # Must match training, or the restored params won't fit. The worker
            # critic is ALWAYS per-agent here (the intrinsic reward is per-agent);
            # only V^M's width follows the env's reward mode.
            n_critic_outputs=self.env.n_agents,
            n_manager_outputs=(
                self.env.n_agents if self.config.per_agent_rewards else 1
            ),
        )
        path = self.dirs["models"] / "models_finished.msgpack"
        if not path.exists():
            path = self.dirs["models"] / "models_checkpoint.msgpack"
        target = {
            "actor": train_state.actor_ts.params,
            "critic": train_state.critic_ts.params,
            "manager": train_state.manager_ts.params,
            "manager_critic": train_state.manager_critic_ts.params,
            **self._intrinsic_tree(train_state, "intrinsic_critic", True),
        }
        with open(path, "rb") as f:
            params_dict = from_bytes(target, f.read())
        print(f"Params loaded from {path}")
        restored = train_state._replace(
            actor_ts=train_state.actor_ts.replace(params=params_dict["actor"]),
            critic_ts=train_state.critic_ts.replace(params=params_dict["critic"]),
            manager_ts=train_state.manager_ts.replace(params=params_dict["manager"]),
            manager_critic_ts=train_state.manager_critic_ts.replace(
                params=params_dict["manager_critic"]
            ),
        )
        if "intrinsic_critic" in params_dict:
            restored = restored._replace(
                intrinsic_critic_ts=restored.intrinsic_critic_ts.replace(
                    params=params_dict["intrinsic_critic"]
                )
            )
        return restored

    def save_training_stats(self, stats_tracker, path):
        with open(path, "wb") as f:
            pickle.dump(stats_tracker.to_dict(), f)

    # ------------------------------------------------------------------ view / eval

    def view(self, *, detailed_goal_plots: bool = False):
        """Render episodes with task-return/alignment summaries.

        Per episode, besides the plain videos: `goal_following_episode_<i>.png`
        (reward + agents x time heatmaps of horizon / one-step goal alignment) and
        `episode_<i>_goals.mp4` (the same panel with a moving cursor, beside the
        frames with per-agent goal marks — see `goal_video.py`).
        Set detailed_goal_plots=True for the per-agent PCA and raw cosine plots.
        """
        # Envs that bring their own renderer (SMAX) take a separate path: the MJX
        # renderers below are not generic — they read world_width, sector_sensor_radius,
        # objects_push_coupling_list, _build_xml(), state.data.qpos and hardcoded MJX
        # observation-slice constants off the env.
        if hasattr(self.env, "render_episode"):
            return self._view_with_env_renderer(detailed_goal_plots=detailed_goal_plots)

        import imageio
        import matplotlib.pyplot as plt

        import jax.numpy as jnp

        from algorithms.feudal_mappo_jax.manager import GROUNDED_GOAL_SPACES
        from algorithms.feudal_mappo_jax.mappo import (
            build_goal_channel,
            build_manager,
        )
        from algorithms.feudal_mappo_jax.network import sample_action
        from algorithms.feudal_mappo_jax.worker import bind_goal, encode_obs
        from algorithms.feudal_mappo_jax.goal_video import save_goal_video
        from algorithms.feudal_mappo_jax.goal_visualization import (
            frame_alignment,
            goal_alignment_scores,
            save_goal_alignment_plot,
            save_goal_following_raster,
            save_goal_plot,
            save_task_alignment_episode,
            save_task_alignment_overview,
            summarize_task_alignment,
        )
        from environments.mjx_suite.multi_box_push_mjx import _AGENT_RADIUS
        from environments.mjx_suite.renderer import MJXRenderer, MuJoCoNativeRenderer

        train_state = self._load_train_state()
        # Static: does the worker read the manager's encoder instead of raw obs?
        share_encoder = self.config.worker_encoder != "none"
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

        # The full hierarchy has to run here too — the worker is goal-conditioned,
        # so rendering it without the manager would drive it with a goal it never
        # saw in training. Everything is UNBATCHED (obs is (n_agents, obs_dim)),
        # and the Python loop maintains the goal ring the rollout scan carries.
        manager = build_manager(self.config, self.env.n_agents)
        horizon, goal_dim = self.config.goal_horizon, self.config.goal_dim

        # The centralized input must be built the way the trainer built it, or a
        # hook-on arm feeds a concat-obs vector into a manager whose first layer
        # was sized for the compact state — a bare ScopeParamShapeError at render
        # time, and under manager_latent="centralized" that vector is the
        # manager's ONLY input. Mirrors `trainer.global_state_fn`, unbatched.
        if hasattr(self.env, "global_state"):
            _view_gs = jax.jit(self.env.global_state)
        else:
            def _view_gs(state):
                return None

        @jax.jit
        def manager_fn(m_carry, obs, gs):
            # `obs` is (n_agents, obs_dim) here (unbatched), and is passed
            # per-agent for manager_latent="local"; the centralized branch reads
            # `gs`. Without an env hook `gs` is the flattened obs, as before.
            return manager.apply(train_state.manager_ts.params, m_carry, gs, obs)

        def _global_state_for(state, obs):
            gs = _view_gs(state)
            return obs.reshape(-1) if gs is None else gs

        @jax.jit
        def policy_fn(obs, pooled_goal):
            # Under a shared encoder the worker eats f_enc(obs), not obs. Rendering
            # it on raw obs would show a different policy than the one that
            # trained — here it would also be a shape error, but the point is that
            # `view()` must run the whole hierarchy, for the same reason it runs
            # the manager rather than feeding the worker an invented goal.
            worker_obs = (
                encode_obs(manager.apply, train_state.manager_ts.params, obs)
                if share_encoder
                else obs
            )
            actions, _ = sample_action(
                jax.random.PRNGKey(0),
                bind_goal(train_state.actor_ts.apply_fn, pooled_goal),
                train_state.actor_ts.params,
                worker_obs,
                discrete=discrete,
                deterministic=True,
            )
            return actions

        # The SAME constructor the training and eval scans use. `view()` had its
        # own copy of the ring convention until that was consolidated, and a
        # drifted copy renders perfectly happily — it just shows a different
        # policy than the one that trained. Under `position_waypoint` the ring is
        # not even the right structure (the worker eats a live error vector, not
        # a sum of directions), so routing through the channel is what keeps the
        # rendered policy provably the trained one.
        channel = build_goal_channel(self.config, self.env.n_agents)
        grounded = self.config.goal_space in GROUNDED_GOAL_SPACES

        def _goal_state_for(state):
            """The grounded `s` for one UNBATCHED state, or None under `latent`."""
            if not grounded:
                return None
            base = type(self.env).base_state(state) if is_macro else state
            return self.env.goal_state(base)

        def _fresh_goal_state():
            """Per-episode manager memory: zeroed channel carry + core carry."""
            return (
                manager.initialize_carry(jax.random.PRNGKey(0), ()),
                channel.init(()),
            )

        def _advance_goal(m_carry, goal_hist, t, obs, state):
            """One manager decision + ring write; returns the pooled goal w_t.

            Uses the SAME `goal_ring_*` helpers as the training and eval scans —
            not a copy of them — so the rendered policy provably conditions on
            what the trained one did. (The helpers are rank-agnostic, so the
            unbatched `(c, N, D)` ring here needs no special case.) A private
            reimplementation would drift silently: a wrong slot index or a stale
            pool renders perfectly happily, just as a different policy.
            """
            m_carry, goal, latent = manager_fn(
                m_carry, obs, _global_state_for(state, obs)
            )
            # Under a grounded goal space the PLOT must show the space the goals
            # actually live in — the env readout — not the manager's internal
            # bottleneck. They are different widths, so feeding the latter would
            # raise in `save_goal_plot` rather than mislead; but the right fix is
            # to plot the real thing, which is also a literal 2-D map of the
            # arena rather than a PCA projection.
            s_now = _goal_state_for(state)
            if s_now is None:
                s_now = latent
            episode_goals.append(np.asarray(goal))
            episode_latents.append(np.asarray(s_now))
            goal_hist = channel.write(goal_hist, goal, t, s_now)
            pooled = channel.pool(goal_hist, s_now)
            episode_pooled_goals.append(np.asarray(pooled))
            return m_carry, goal_hist, pooled

        def _draw(state, obs, t):
            """Append one rendered frame (+ native) for the given base state.

            Also records a frame WITHOUT the sensor overlay, where every agent is
            in it, and which policy decision `t` it belongs to — the goal video
            draws its marks onto those frames after the episode (see
            `goal_video.py`). Clean frames because the focus agent's lidar,
            sectors and arrows would otherwise bury its goal marks (with one
            agent, that is the whole picture).
            """
            frames.append(renderer.render(state, obs=np.asarray(obs)))
            goal_frames.append(renderer.render(state))
            frame_agent_pos.append(renderer.agent_positions(state))
            frame_decision.append(t)
            if native_renderer is not None:
                native_frames.append(native_renderer.render(state))

        # For the macro env, render at *low-level* granularity: hold the
        # high-level skill choice fixed for macro_len steps but drive (and draw)
        # the base env one physics step at a time, so the video is smooth instead
        # of jumping macro_len steps per frame. The high-level policy re-decides
        # at each macro boundary off the base obs there, exactly as SyncMacroMJX
        # does internally.
        base_step_fn = jax.jit(render_env.step) if is_macro else None
        # Goal-influence arrows need a continuous action to difference; the
        # macro/skill action is a discrete index, so they are skipped there.
        record_influence = not discrete
        skill_actions_fn = jax.jit(self.env._skill_actions) if is_macro else None

        episode_summaries = []
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
            episode_goals, episode_latents = [], []
            episode_pooled_goals, episode_active = [], []
            episode_task_rewards = []
            frame_agent_pos, frame_decision, episode_influence = [], [], []
            goal_frames = []
            m_carry, goal_hist = _fresh_goal_state()

            if is_macro:
                done = False
                for t in range(self.env.max_steps):  # macro decisions
                    # The manager decides on the macro boundary, alongside the
                    # worker's skill choice.
                    m_carry, goal_hist, pooled = _advance_goal(
                        m_carry, goal_hist, t, obs, state
                    )
                    skills = policy_fn(obs, pooled)
                    macro_active = np.ones(self.env.n_agents, dtype=bool)
                    macro_reward = 0.0
                    for _ in range(self.env.macro_len):  # low-level steps
                        _draw(state, obs, t)
                        actions = skill_actions_fn(state, skills)
                        obs, state, _, terminated, truncated, info = base_step_fn(
                            state, actions
                        )
                        rewards.append(float(info["task_reward"]))
                        macro_reward += rewards[-1]
                        macro_active &= np.asarray(
                            info.get("active", np.ones(self.env.n_agents)), dtype=bool
                        )
                        if bool(terminated) or bool(truncated):
                            done = True
                            break
                    episode_active.append(macro_active)
                    episode_task_rewards.append(macro_reward)
                    if done:
                        break
            else:
                for t in range(self.env.max_steps):
                    _draw(state, obs, t)
                    m_carry, goal_hist, pooled = _advance_goal(
                        m_carry, goal_hist, t, obs, state
                    )
                    actions = policy_fn(obs, pooled)
                    if record_influence:
                        # What the goal makes each agent DO: the executed action
                        # (the env clips to +-1) under the real pooled goal minus
                        # under a zero goal. The per-step analogue of the probe's
                        # `zeroed` variant — exactly the unmodulated trunk for
                        # FiLM, off-distribution for concat, "arrived" for a
                        # waypoint. Exactly 0 on a `zero_goal` arm.
                        null_actions = policy_fn(obs, jnp.zeros_like(pooled))
                        episode_influence.append(
                            np.clip(np.asarray(actions), -1.0, 1.0)
                            - np.clip(np.asarray(null_actions), -1.0, 1.0)
                        )
                    obs, state, _, terminated, truncated, info = step_fn(
                        state, actions
                    )
                    # Team reward: the env's `reward` is per-agent under
                    # difference_rewards, and the plot is of team performance.
                    rewards.append(float(info["task_reward"]))
                    episode_task_rewards.append(rewards[-1])
                    episode_active.append(np.asarray(
                        info.get("active", np.ones(self.env.n_agents))
                    ))
                    if bool(terminated) or bool(truncated):
                        break
            # Include s_T from the final observation, without resetting the env
            # or advancing the policy. Macro horizons count policy decisions.
            _, _, final_latent = manager_fn(
                m_carry, obs, _global_state_for(state, obs)
            )
            final_s = _goal_state_for(state)
            episode_latents.append(
                np.asarray(final_latent if final_s is None else final_s)
            )
            summary = summarize_task_alignment(
                episode_goals, episode_latents, episode_pooled_goals, horizon,
                episode_task_rewards, episode=episode,
                active=episode_active, goal_space=self.config.goal_space,
                macro_steps=is_macro,
            )
            episode_summaries.append(summary)
            save_task_alignment_episode(
                summary, self.dirs["logs"] / f"task_vs_alignment_episode_{episode}.png",
            )
            if detailed_goal_plots:
                save_goal_plot(
                    episode_goals, episode_latents, horizon,
                    self.dirs["logs"] / f"goals_episode_{episode}.png", episode,
                )
                save_goal_alignment_plot(
                    episode_goals, episode_latents, episode_pooled_goals, horizon,
                    self.dirs["logs"] / f"goal_alignment_episode_{episode}.png", episode,
                    active=episode_active, goal_space=self.config.goal_space,
                    macro_steps=is_macro,
                )
            # Goal following on the frames themselves: one set of scores feeds
            # the raster, the panel and the per-agent marks.
            ring, dot = frame_alignment(
                *goal_alignment_scores(
                    episode_goals, episode_latents, episode_pooled_goals, horizon,
                    active=episode_active, goal_space=self.config.goal_space,
                ),
                np.asarray(frame_decision), horizon, goal_space=self.config.goal_space,
            )
            figure_kwargs = dict(
                horizon=horizon, goal_space=self.config.goal_space, episode=episode,
                macro_steps=is_macro,
            )
            save_goal_following_raster(
                ring, dot, rewards,
                self.dirs["logs"] / f"goal_following_episode_{episode}.png",
                **figure_kwargs,
            )
            influence = np.asarray(episode_influence) if record_influence else None
            if influence is not None:
                print(f"Goal influence: max |action change| {np.abs(influence).max():.4g}")
            save_goal_video(
                self.dirs["logs"] / f"episode_{episode}_goals.mp4", renderer, goal_frames,
                agent_pos=frame_agent_pos, frame_decision=frame_decision,
                ring=ring, dot=dot, rewards=rewards, agent_radius=_AGENT_RADIUS,
                influence=influence,
                goal_states=episode_latents if grounded else None,
                goals=episode_goals, pooled_goals=episode_pooled_goals,
                waypoint_radius=self.config.waypoint_radius,
                to_world=getattr(render_env, "goal_state_to_world", None),
                **figure_kwargs,
            )
            rewards = np.asarray(rewards)

            # Episode *return* (sum), not the final step's reward — the delivery
            # bonuses are paid on the steps the boxes land, so `rewards[-1]`
            # reported ~0 for any episode that did not happen to end on a
            # delivery. Matches what `evaluate()` and the `reward` stat report.
            print(
                f"RETURN: {rewards.sum():.4f}  "
                f"(final step {rewards[-1]:+.4f}, {len(rewards)} steps)"
            )

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
                imageio.mimwrite(native_path, native_frames, fps=30, macro_block_size=1)
                print(f"Native video saved to {native_path}")

        save_task_alignment_overview(
            episode_summaries, self.dirs["logs"] / "task_vs_alignment.png",
        )

    def _view_with_env_renderer(
        self, n_episodes: int = 3, *, detailed_goal_plots: bool = False,
    ):
        """Render episodes using the env's own renderer (SMAX -> SMAXVisualizer).

        Fewer episodes than the MJX path renders (10) because SMAXVisualizer animates
        through matplotlib, which costs ~2-4s per frame of episode: a battle that runs
        the full `max_steps` takes several minutes on its own. Raise `n_episodes` if you
        want more and are willing to wait.

        Runs the FULL hierarchy — manager read + goal ring + goal-conditioned worker —
        for the same reason the MJX path does: the worker is goal-conditioned, so
        rendering it without the manager drives it with a goal it never saw. Actions are
        a deterministic argmax under the legal-action mask, matching `eval_fn`.
        """
        import matplotlib.pyplot as plt

        import jax.numpy as jnp

        from algorithms.feudal_mappo_jax.manager import GROUNDED_GOAL_SPACES
        from algorithms.feudal_mappo_jax.mappo import (
            build_goal_channel,
            build_manager,
        )
        from algorithms.feudal_mappo_jax.network import sample_action
        from algorithms.feudal_mappo_jax.worker import bind_goal, encode_obs
        from algorithms.feudal_mappo_jax.goal_visualization import (
            frame_alignment,
            goal_alignment_scores,
            save_goal_alignment_plot,
            save_goal_following_raster,
            save_goal_plot,
            save_task_alignment_episode,
            save_task_alignment_overview,
            summarize_task_alignment,
        )

        train_state = self._load_train_state()
        # Static: does the worker read the manager's encoder instead of raw obs?
        share_encoder = self.config.worker_encoder != "none"
        n_agents = self.env.n_agents
        obs_dim = self.env.observation_dim
        manager = build_manager(self.config, n_agents)
        # Same constructor as the training and eval scans, so this renderer
        # cannot condition the worker differently than training did. SMAX
        # publishes no `goal_state`, so a grounded arm never reaches here
        # (`validate_goal_space` raises at construction) and this is the latent
        # ring — but it is built the one way regardless.
        channel = build_goal_channel(self.config, n_agents)
        horizon, goal_dim = self.config.goal_horizon, self.config.goal_dim

        reset_fn = jax.jit(self.env.reset)
        step_fn = jax.jit(self.env.step)
        avail_fn = jax.jit(self.env.avail_actions)
        gs_fn = jax.jit(self.env.global_state)

        @jax.jit
        def manager_fn(m_carry, gs, obs):
            return manager.apply(
                train_state.manager_ts.params, m_carry, gs, obs
            )

        @jax.jit
        def policy_fn(obs, pooled_goal, mask):
            # Flatten first, then encode — the same order as the training path, so
            # the rendered policy is bitwise the trained one.
            worker_obs = obs.reshape(n_agents, obs_dim)
            if share_encoder:
                worker_obs = encode_obs(
                    manager.apply, train_state.manager_ts.params, worker_obs
                )
            actions, _ = sample_action(
                jax.random.PRNGKey(0),
                bind_goal(train_state.actor_ts.apply_fn, pooled_goal),
                train_state.actor_ts.params,
                worker_obs,
                discrete=True,
                deterministic=True,
                action_mask=mask.reshape(n_agents, -1),
            )
            return actions

        episode_summaries = []
        for episode in range(n_episodes):
            obs, state = reset_fn(jax.random.PRNGKey(self.rng_seed + episode))
            # Per-episode manager memory: zeroed core carry + zeroed goal ring, the
            # same convention the training and eval scans use.
            m_carry = manager.initialize_carry(jax.random.PRNGKey(0), ())
            # Channel rather than a bare ring, for the same one-convention
            # reason as the MJX path above. SMAX publishes no `goal_state`, so a
            # grounded arm never reaches here — `validate_goal_space` raises at
            # construction — and this is the latent ring either way.
            goal_hist = channel.init(())

            state_seq, rewards = [], []
            episode_goals, episode_latents = [], []
            episode_pooled_goals, episode_active = [], []
            episode_task_rewards = []
            for t in range(self.env.max_steps):
                # The manager reads the env's real global state here, exactly as the
                # trainer does — not a reshape of the observations.
                m_carry, goal, latent = manager_fn(
                    m_carry, gs_fn(state), obs.reshape(n_agents, obs_dim)
                )
                episode_goals.append(np.asarray(goal))
                episode_latents.append(np.asarray(latent))
                goal_hist = channel.write(goal_hist, goal, t, latent)
                pooled = channel.pool(goal_hist, latent)
                episode_pooled_goals.append(np.asarray(pooled))
                actions = policy_fn(obs, pooled, avail_fn(state))
                state_seq.append(
                    (state.key, state.env_state, self.env.to_action_dict(actions))
                )
                obs, state, _, terminated, truncated, info = step_fn(state, actions)
                rewards.append(float(info["task_reward"]))
                episode_task_rewards.append(rewards[-1])
                episode_active.append(np.asarray(
                    info.get("active", np.ones(n_agents))
                ))
                if bool(terminated) or bool(truncated):
                    break

            _, _, final_latent = manager_fn(
                m_carry, gs_fn(state), obs.reshape(n_agents, obs_dim)
            )
            episode_latents.append(np.asarray(final_latent))
            summary = summarize_task_alignment(
                episode_goals, episode_latents, episode_pooled_goals, horizon,
                episode_task_rewards, episode=episode,
                active=episode_active, goal_space=self.config.goal_space,
                macro_steps=False,
            )
            episode_summaries.append(summary)
            save_task_alignment_episode(
                summary, self.dirs["logs"] / f"task_vs_alignment_episode_{episode}.png",
            )
            if detailed_goal_plots:
                save_goal_plot(
                    episode_goals, episode_latents, horizon,
                    self.dirs["logs"] / f"goals_episode_{episode}.png", episode,
                )
                save_goal_alignment_plot(
                    episode_goals, episode_latents, episode_pooled_goals, horizon,
                    self.dirs["logs"] / f"goal_alignment_episode_{episode}.png", episode,
                    active=episode_active, goal_space=self.config.goal_space,
                    macro_steps=False,
                )
            # Raster only: the GIF comes from jaxmarl's own visualizer, which
            # has no hook to draw per-agent marks on.
            ring, dot = frame_alignment(
                *goal_alignment_scores(
                    episode_goals, episode_latents, episode_pooled_goals, horizon,
                    active=episode_active, goal_space=self.config.goal_space,
                ),
                np.arange(len(rewards)), horizon, goal_space=self.config.goal_space,
            )
            save_goal_following_raster(
                ring, dot, rewards,
                self.dirs["logs"] / f"goal_following_episode_{episode}.png",
                horizon=horizon, goal_space=self.config.goal_space, episode=episode,
            )

            gif_path = self.dirs["logs"] / f"episode_{episode}.gif"
            self.env.render_episode(state_seq, gif_path)
            print(
                f"Episode {episode}: return={sum(rewards):.2f} "
                f"steps={len(rewards)} -> {gif_path}"
            )

            fig, ax = plt.subplots()
            ax.plot(rewards)
            ax.set_xlabel("step")
            ax.set_ylabel("team reward")
            ax.set_title(f"Episode {episode} (return {sum(rewards):.2f})")
            fig.savefig(self.dirs["logs"] / f"episode_{episode}_reward.png")
            plt.close(fig)

        save_task_alignment_overview(
            episode_summaries, self.dirs["logs"] / "task_vs_alignment.png",
        )


    def evaluate(self):
        """Deterministic evaluation of the saved policy (PolicyEvaluator parity)."""
        train_state = self._load_train_state()
        _, _, _, eval_fn, _ = make_train(self.config, self.env)
        # `("real",)` explicitly, not the config default: this path wants the
        # scalar return and nothing else, and the default now includes the
        # `constant` block, which needs a direction this call has no rollout to
        # derive. Also 4x cheaper.
        reward = float(
            eval_fn(
                train_state,
                jax.random.PRNGKey(self.rng_seed + 1),
                variants=("real",),
            )
        )
        print(f"Mean eval episode return: {reward:.2f}")
        return reward

    # ------------------------------------------------------------------
    # Goal-usefulness measurement (offline; see goal_dependence_probe.py)
    # ------------------------------------------------------------------

    def goal_dependence(self, n_eval_episodes=None, shifts=(1,), make_train_out=None):
        """Measure whether this checkpoint's manager goals are actually USEFUL.

        Every diagnostic logged during training tests whether the goals are
        well-FORMED (non-collapsed). This tests whether they DO anything, by
        two paired interventions that preserve the goal distribution exactly and
        destroy one property each:

        * **behavioural** — re-run deterministic eval with each worker handed a
          teammate's directive (`permuted`), a directive from an unrelated env
          (`env_permuted`), or none (`zeroed`). The return gaps against `real`
          are the ground-truth measurement.
        * **latent** — `d_cos` against the same two permutation nulls, straight
          off a rollout's stored goals and latents. No update, no extra network
          forward.

        READ ``d_cos_gap_env`` / the ``env_permuted`` return FIRST. If the goals
        turn out not to depend on the state, the agent-pairing numbers are
        uninterpretable however large they are.

        The eval gap's reading is ASYMMETRIC. A permuted rollout is off-policy
        twice (mispaired input, and it then visits different states), so the gap
        overstates the causal value of correct assignment: ``gap ~ 0`` is a
        STRONG negative, while ``gap > 0`` is weak evidence of dependence whose
        magnitude is not "the value of hierarchy".

        Args:
            n_eval_episodes: override the config's episode count for this
                measurement only (the offline probe wants far more than a
                training run does; it is width, so it is nearly free).
            shifts: permutation shifts to sweep. If the results agree across
                shifts, the choice of permutation is not load-bearing.
            make_train_out: reuse an already-built `make_train(...)` tuple, so a
                probe sweeping many trials of one arm compiles once.

        Returns a dict of plain floats/arrays, JSON/pickle-friendly.
        """
        import numpy as np

        from algorithms.feudal_mappo_jax.manager import offline_goal_variants
        from algorithms.feudal_mappo_jax.mappo import manager_cosine_metrics

        variants = offline_goal_variants(self.env.n_agents)

        config = self.config
        if n_eval_episodes is not None and n_eval_episodes != config.n_eval_episodes:
            config = replace(config, n_eval_episodes=int(n_eval_episodes))
            make_train_out = None  # episode count changes the traced shapes

        train_state = self._load_train_state()
        results = {
            "n_eval_episodes": int(config.n_eval_episodes),
            "checkpoint_mtime": self._checkpoint_mtime(),
            # `_load_train_state` rebuilds the networks from the CURRENT yaml.
            # `normalize_pooled_goal` has no parameters, so a checkpoint trained
            # under one setting loads cleanly against the other and silently
            # evaluates a different function. Record what was actually used.
            "normalize_pooled_goal": bool(config.normalize_pooled_goal),
            "zero_goal": bool(config.zero_goal),
            # Training-only (it never touches the forward pass), so unlike
            # `normalize_pooled_goal` it cannot make this eval measure a
            # different function. Recorded for provenance: it changes what the
            # gaps below MEAN — under 'intrinsic_only' the worker is defined to
            # depend on the goals, so every gap is large by construction and none
            # of them grades the arm.
            "worker_objective": str(config.worker_objective),
            # ⚠ UNPARAMETERIZED, and therefore recorded here or nowhere. Neither
            # appears in the checkpoint: `_dims_from_checkpoint` can tell a
            # grounded arm from a latent one (the goal head is narrower than
            # `f_Mspace`) but CANNOT tell `position_direction` from
            # `position_waypoint`, whose param trees are identical. They also
            # change what the gaps MEAN: under a waypoint goal the `zeroed`
            # variant is the in-distribution directive "you have arrived", not
            # the absence of one, so `gap_zeroed` does not grade that arm the way
            # it grades the others — read `gap_constant` there.
            "goal_space": str(config.goal_space),
            "waypoint_radius": float(config.waypoint_radius),
            "goal_dim": int(config.goal_dim),
            "manager_latent_dim": (
                None if config.manager_latent_dim is None
                else int(config.manager_latent_dim)
            ),
            "goal_horizon": int(config.goal_horizon),
            "n_agents": int(self.env.n_agents),
            "by_shift": {},
        }

        for shift in shifts:
            cfg = replace(config, goal_permute_shift=int(shift))
            init_fn, collect_fn, _, eval_fn, _ = (
                make_train(cfg, self.env) if make_train_out is None else make_train_out
            )

            # --- one rollout, reused by BOTH measurements below ---
            # It has to come first: the `constant` variant needs a direction to
            # stand in for the manager, and `mean_goal_direction` over this
            # rollout's pooled goals is it. Deriving it here rather than reading
            # the live `eval_*_constant` series is what makes the offline number
            # exactly reproducible from the checkpoint alone.
            runner_state = init_fn(jax.random.PRNGKey(self.rng_seed))
            runner_state = runner_state._replace(train_state=train_state)
            _, trajectory, _, _ = collect_fn(runner_state)
            constant_goal = mean_goal_direction(trajectory.pooled_goal)

            # --- behavioural: one scan, every variant, shared reset keys ---
            key = jax.random.PRNGKey(self.rng_seed + 1)
            rewards, lengths = eval_fn(
                train_state,
                key,
                variants=variants,
                detail=True,
                constant_goal=constant_goal,
            )
            rewards, lengths = np.asarray(rewards), np.asarray(lengths)
            per_variant = dict(zip(variants, rewards))
            real = per_variant["real"]

            entry = {
                "returns": {v: r.tolist() for v, r in zip(variants, rewards)},
                "lengths": {v: l.mean().item() for v, l in zip(variants, lengths)},
                "return_mean": {
                    v: r.mean().item() for v, r in zip(variants, rewards)
                },
            }
            for v in variants[1:]:
                # PAIRED: episode j of every block started from the same state,
                # so the per-episode difference removes reset variance.
                diff = real - per_variant[v]
                entry[f"gap_{v}"] = diff.mean().item()
                entry[f"gap_{v}_ci"] = _paired_bootstrap_ci(diff)

            # --- latent: d_cos against the same nulls, off that rollout ---
            done_a = jnp.broadcast_to(
                trajectory.done.astype(jnp.float32)[..., None],
                trajectory.goal.shape[:-1],
            )
            cos_metrics = manager_cosine_metrics(
                trajectory.state_latent,
                trajectory.goal,
                cfg.goal_horizon,
                done_a,
                trajectory.active_mask,
                int(shift),
            )
            entry.update({k: float(v) for k, v in cos_metrics.items()})

            # Collapse metrics on BOTH the raw goal (comparable to the training
            # series) and the POOLED w_t (what the worker eats, and what the
            # eval variants actually permute) — without the second there is a
            # gap between the collapse metric and the intervention.
            entry["goal_direction_count_raw"] = _direction_count(trajectory.goal)
            entry["goal_direction_count_pooled"] = _direction_count(
                trajectory.pooled_goal
            )
            # How close the manager already is to the `constant` variant: the
            # norm of the (row-normalized) mean pooled goal, 1.0 = one frozen
            # direction. Read it NEXT TO `gap_constant` — a small gap at a low
            # concentration would be the interesting case (varied goals that
            # nonetheless do not matter), while a small gap at ~1.0 just says the
            # manager had already collapsed to the constant it is compared with.
            entry["goal_concentration"] = float(
                goal_concentration(trajectory.pooled_goal)
            )
            results["by_shift"][int(shift)] = entry

        return results

    def _checkpoint_mtime(self):
        path = self.dirs["models"] / "models_finished.msgpack"
        if not path.exists():
            path = self.dirs["models"] / "models_checkpoint.msgpack"
        if not path.exists():
            return None
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
