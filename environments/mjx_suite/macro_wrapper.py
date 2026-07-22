"""Asynchronous macro-action layer over `MultiBoxPushMJX` (pure / jit / vmap).

Why this exists: the box2d `HierarchicalSkillEnv`
(`algorithms/hierarchical/hrl_env.py:125-148`) runs *every* agent's skill for a
fixed `macro_len` and has all agents re-decide in lockstep, so commitment
duration is a constant and every agent's decision point coincides. That is the
*synchronous* options setting. This layer decouples the decision points: each
agent commits to a skill for its own sampled duration, so at any instant the
other agents are mid-commitment with differing elapsed/remaining times.

That asymmetry is the whole point — it is what makes the difference reward
`D_i = G(z) - G(z_-i + c_i)` ill-defined ("others' actions held fixed" is not a
well-defined object when others are partially through overlapping commitments)
and what the bias study (`algorithms/difference_rewards/bias_study.py`) measures.

**Timescale.** One `step` is one *low-level* physics step. The policy is queried
every step but only agents whose commitment has expired actually adopt the
proposed skill — everyone else keeps flying their in-flight skill. This keeps all
shapes static (jit) while letting decision points fall wherever they like.

**Conditions** are set by two knobs, which together give the bias study its x-axis:
- `d_min == d_max` and `stagger=False` -> fully **synchronous** lockstep (the control).
- `d_min == d_max` and `stagger=True`  -> equal durations, offset phases.
- `d_min <  d_max`                     -> variable durations (the async setting).

Demo / smoke test:
    MUJOCO_GL=egl SDL_VIDEODRIVER=dummy uv run python -m environments.mjx_suite.macro_wrapper
"""

import dataclasses

import jax
import jax.numpy as jnp

from environments.mjx_suite.macro_skills import (
    N_SKILLS,
    SKILL_ORDER,
    all_skill_actions,
    null_action,
)
from environments.mjx_suite.multi_box_push_mjx import EnvState, MultiBoxPushMJX
from environments.mjx_suite.observation import OBS_DIM

# Commitment features appended to the base observation: one-hot skill + the two
# scalars that define the async setting.
COMMITMENT_FEAT_DIM = N_SKILLS + 2
MACRO_OBS_DIM = OBS_DIM + COMMITMENT_FEAT_DIM

# SyncMacroMJX reward modes (see the class docstring). The base env's own
# "dense"/"difference_rewards" pass straight through; this one is computed by the
# wrapper by forking the whole macro window per agent.
WINDOWED_DIFFERENCE_REWARDS = "windowed_difference_rewards"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MacroState:
    """`EnvState` plus the joint commitment. Forkable, like `EnvState`."""

    env_state: EnvState
    skill_idx: jax.Array  # (A,) int32 — in-flight skill per agent
    remaining: jax.Array  # (A,) int32 — low-level steps left on the commitment
    elapsed: jax.Array  # (A,) int32 — steps since this commitment started
    key: jax.Array  # duration-sampling chain


class AsyncMacroMJX:
    def __init__(
        self,
        env: MultiBoxPushMJX | None = None,
        *,
        d_min: int = 5,
        d_max: int = 15,
        stagger: bool = True,
        augment_obs: bool = True,
        **env_kwargs,
    ):
        if d_min < 1 or d_max < d_min:
            raise ValueError(f"need 1 <= d_min <= d_max, got {d_min}, {d_max}")

        self.env = env if env is not None else MultiBoxPushMJX(**env_kwargs)
        self.n_agents = self.env.n_agents
        self.n_skills = N_SKILLS
        self.d_min = int(d_min)
        self.d_max = int(d_max)
        self.stagger = bool(stagger)
        self.augment_obs = bool(augment_obs)
        self.obs_dim = MACRO_OBS_DIM if augment_obs else OBS_DIM

    @property
    def is_synchronous(self) -> bool:
        """True for the lockstep control condition."""
        return self.d_min == self.d_max and not self.stagger

    # ------------------------------------------------------------ internals

    def _obs(self, base_obs: jnp.ndarray, mstate: MacroState) -> jnp.ndarray:
        """Append the commitment features to the shared 40-dim observation."""
        if not self.augment_obs:
            return base_obs
        commitment = jnp.concatenate(
            [
                jax.nn.one_hot(mstate.skill_idx, self.n_skills),
                (mstate.remaining / self.d_max)[:, None],
                (mstate.elapsed / self.d_max)[:, None],
            ],
            axis=-1,
        )
        return jnp.concatenate([base_obs, commitment], axis=-1)

    def skill_actions(
        self, env_state: EnvState, skill_idx: jnp.ndarray
    ) -> jnp.ndarray:
        """(A,2) — each agent's action under the skill it is committed to."""
        per_skill = all_skill_actions(self.env, env_state)  # (K, A, 2)
        idx = skill_idx[None, :, None]  # (1, A, 1) -> broadcasts over the xy axis
        return jnp.take_along_axis(per_skill, idx, axis=0)[0]

    def actions_with_override(
        self,
        env_state: EnvState,
        skill_idx: jnp.ndarray,
        override_agent: jnp.ndarray,
        override_action: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Skill actions with one agent replaced by the counterfactual default.

        `override_agent` is a **traced** agent index (or -1 for "no override", so
        the factual rollout shares this code path). Used by the oracle to fork a
        rollout with agent i removed.
        """
        actions = self.skill_actions(env_state, skill_idx)
        default = (
            null_action(self.env, env_state)
            if override_action is None
            else override_action
        )
        mask = jnp.arange(self.n_agents) == override_agent  # (A,)
        return jnp.where(mask[:, None], default, actions)

    # ------------------------------------------------------------------ API

    def reset(self, key: jax.Array) -> tuple[jnp.ndarray, MacroState]:
        k_env, k_skill, k_phase, k_chain = jax.random.split(key, 4)
        base_obs, env_state = self.env.reset(k_env)

        skill_idx = jax.random.randint(k_skill, (self.n_agents,), 0, self.n_skills)
        if self.stagger:
            # Random initial phase -> decision points are decoupled from step 0.
            remaining = jax.random.randint(
                k_phase, (self.n_agents,), 1, self.d_max + 1
            )
        else:
            remaining = jnp.full((self.n_agents,), self.d_max, dtype=jnp.int32)

        mstate = MacroState(
            env_state=env_state,
            skill_idx=skill_idx.astype(jnp.int32),
            remaining=remaining.astype(jnp.int32),
            elapsed=jnp.zeros((self.n_agents,), dtype=jnp.int32),
            key=k_chain,
        )
        return self._obs(base_obs, mstate), mstate

    def commit(
        self, mstate: MacroState, proposed_skills: jnp.ndarray
    ) -> tuple[MacroState, jnp.ndarray]:
        """Adopt `proposed_skills` for agents whose commitment expired.

        Returns the post-commit state and the (A,) bool mask of who re-decided.
        Split out of `step` so the oracle can fork *after* commitment and hold the
        joint commitment fixed across its counterfactual rollouts.
        """
        decide = mstate.remaining <= 0
        key, k_dur = jax.random.split(mstate.key)
        duration = jax.random.randint(
            k_dur, (self.n_agents,), self.d_min, self.d_max + 1
        )
        return (
            MacroState(
                env_state=mstate.env_state,
                skill_idx=jnp.where(
                    decide, proposed_skills.astype(jnp.int32), mstate.skill_idx
                ),
                remaining=jnp.where(decide, duration, mstate.remaining),
                elapsed=jnp.where(decide, 0, mstate.elapsed),
                key=key,
            ),
            decide,
        )

    def step_committed(
        self, mstate: MacroState, override_agent: jnp.ndarray | None = None
    ):
        """Advance one low-level step under the *already committed* skills.

        `override_agent` (traced, -1 = none) nulls one agent for counterfactuals.
        """
        if override_agent is None:
            actions = self.skill_actions(mstate.env_state, mstate.skill_idx)
        else:
            actions = self.actions_with_override(
                mstate.env_state, mstate.skill_idx, override_agent
            )

        base_obs, env_state, reward, term, trunc, info = self.env.step(
            mstate.env_state, actions
        )
        new = MacroState(
            env_state=env_state,
            skill_idx=mstate.skill_idx,
            remaining=mstate.remaining - 1,
            elapsed=mstate.elapsed + 1,
            key=mstate.key,
        )
        info = {
            **info,
            "skill_idx": new.skill_idx,
            "remaining": new.remaining,
            "elapsed": new.elapsed,
            "low_level_actions": actions,
        }
        return self._obs(base_obs, new), new, reward, term, trunc, info

    def step(self, mstate: MacroState, proposed_skills: jnp.ndarray):
        """One low-level step: commit expired agents, then advance the physics.

        `proposed_skills` (A,) is what the policy *wants*; agents still mid-
        commitment ignore it. Returns
        (obs, state, reward, terminated, truncated, info); `info["decided"]` is
        the mask of agents that adopted a new skill this step.
        """
        committed, decide = self.commit(mstate, proposed_skills)
        obs, new, reward, term, trunc, info = self.step_committed(committed)
        return obs, new, reward, term, trunc, {**info, "decided": decide}


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StaggeredMacroState:
    """``EnvState`` plus the per-agent commitment, for the staggered-starts study.

    Forkable like ``EnvState`` (registered pytree), so it drops straight into the
    ``mappo_jax`` collector's ``v_reset`` / ``tree.map`` auto-reset. Only used when
    ``SyncMacroMJX(stagger_starts=True)``; the ordinary path keeps the plain
    ``EnvState`` so it stays byte-for-byte the original.

    Each agent comes online at ``onset`` (an absolute *low-level* step, sampled per
    episode) and re-decides every ``macro_len`` low-level steps counted from that
    onset — so decision phases stay decoupled all episode (``onset`` is not a
    multiple of ``macro_len``). ``skill_idx`` is the in-flight skill each agent
    flies between its own decision points; it must persist across the macro-window
    boundary because an agent's phase generally falls mid-window. The absolute
    low-level step is read from ``env_state.t`` (no separate counter needed).
    """

    env_state: EnvState
    skill_idx: jax.Array  # (A,) int32 — in-flight committed skill per agent
    onset: jax.Array  # (A,) int32 — absolute low-level step each agent comes online


class SyncMacroMJX:
    """Synchronous **options** layer over ``MultiBoxPushMJX`` (pure / jit / vmap).

    This is the JAX analogue of the box2d ``HierarchicalSkillEnv``
    (``algorithms/hierarchical/hrl_env.py``) and the env the JAX MAPPO stack
    trains a *hierarchical* (skill-selecting) policy on. Unlike its async sibling
    ``AsyncMacroMJX`` above — whose ``step`` is one *low-level* physics step, so
    the policy is queried every step and most proposals are discarded
    mid-commitment — here **one ``step`` is one macro decision**:

      1. Every agent adopts its proposed skill (lockstep — no in-flight
         commitments to respect).
      2. The chosen skills are rolled out for ``macro_len`` low-level physics
         steps, re-deriving the (reactive, closed-loop) skill actions each step.
      3. Reward is **accumulated** over the window and returned once; the episode
         ends the moment any low-level step terminates/truncates, and the state
         is frozen there.

    So the trainer's per-``step`` rollout scan stores exactly **one transition
    per genuine decision** (the SMDP / options view), which is what makes the
    PPO credit assignment correct — the policy gradient only ever touches skill
    choices that actually drove ``macro_len`` steps of physics. This mirrors the
    box2d wrapper's "pick a skill, run it for K steps, that's one env step"
    semantics (reward summed over the window, no intra-option discounting), so
    the two hierarchical stacks stay drop-in comparable.

    The macro state is just the base ``EnvState`` (no commitment bookkeeping is
    needed under lockstep), so it plugs straight into the ``mappo_jax`` collector,
    whose mid-rollout auto-reset already ``v_reset``/``tree.map``-s over it.

    The action is a per-agent **discrete** skill index; the observation is the
    shared 40-dim ``OBS_DIM`` (no commitment features — every agent re-decides
    every macro step, so ``remaining``/``elapsed`` carry no information).
    """

    def __init__(
        self,
        env: MultiBoxPushMJX | None = None,
        *,
        macro_len: int = 10,
        reward_mode: str | None = None,
        stagger_starts: bool = False,
        max_start_delay: int = 0,
        **env_kwargs,
    ):
        if macro_len < 1:
            raise ValueError(f"macro_len must be >= 1, got {macro_len}")

        self.env = env if env is not None else MultiBoxPushMJX(**env_kwargs)
        self.n_agents = self.env.n_agents
        self.n_skills = N_SKILLS
        self.macro_len = int(macro_len)

        # reward_mode=None -> inherit the base env's mode (its per-step reward is
        # accumulated over the window: "dense" -> team scalar, "difference_rewards"
        # -> sum of the base env's single-step D_i). WINDOWED_DIFFERENCE_REWARDS is
        # computed here instead, by forking the whole macro window per agent; it
        # needs a *dense* base env (it forks the team reward and must not trigger
        # the base env's own per-step D).
        self.reward_mode = reward_mode if reward_mode is not None else self.env.reward_mode
        if self.reward_mode == WINDOWED_DIFFERENCE_REWARDS and self.env.reward_mode != "dense":
            raise ValueError(
                f"{WINDOWED_DIFFERENCE_REWARDS} needs a dense base env (it forks the "
                f"team reward); got base reward_mode={self.env.reward_mode!r}"
            )
        # per-agent reward signal? (per-agent critic head + per-agent GAE)
        self.per_agent_rewards = self.reward_mode in (
            "difference_rewards", WINDOWED_DIFFERENCE_REWARDS
        )

        # Staggered-starts study: each episode every agent comes online at a random
        # absolute *low-level* step in [0, max_start_delay] and thereafter re-decides
        # every macro_len steps counted from its own onset. Because onsets are not
        # multiples of macro_len, the agents' decision phases stay decoupled the
        # whole episode (persistent asynchrony), unlike the lockstep options view.
        # Before its onset an agent is offline: null force + dropped from the
        # coupling count (via the base env's `active` mask), and its transition is
        # masked out of the PPO loss (see the trainer's active_mask). The world
        # still steps normally. This tests whether the hierarchical setup — which
        # records one transition per *global* macro window — copes with agents that
        # begin acting, and decide, out of phase.
        self.stagger_starts = bool(stagger_starts)
        self.max_start_delay = int(max_start_delay)
        if self.stagger_starts:
            if self.max_start_delay < 1:
                raise ValueError(
                    f"stagger_starts needs max_start_delay >= 1 (low-level steps), "
                    f"got {max_start_delay}"
                )
            if self.per_agent_rewards:
                # The per-agent difference-reward counterfactuals fork whole windows
                # with a *static* active mask; combining that with per-substep onset
                # masking is a follow-up. Dense (team scalar) is the clean first test.
                raise NotImplementedError(
                    "stagger_starts currently supports only the dense team reward; "
                    f"got reward_mode={self.reward_mode!r} (per-agent). The windowed "
                    "difference-reward-under-asynchrony arm is a follow-up."
                )

        # Interface the mappo_jax trainer/runner reads off the env (matching the
        # attribute names of MultiBoxPushMJX): discrete skill selection.
        self.observation_dim = OBS_DIM
        self.action_dim = N_SKILLS
        self.discrete = True
        # One macro step consumes macro_len low-level steps, so an episode of
        # base.max_steps low-level steps is ceil(base.max_steps / macro_len)
        # decisions — the horizon eval/view scan over.
        self.max_steps = -(-self.env.max_steps // self.macro_len)

    # ------------------------------------------------------------ internals

    def _skill_actions(
        self, env_state: EnvState, skill_idx: jnp.ndarray
    ) -> jnp.ndarray:
        """(A,2) — each agent's low-level action under the skill it selected."""
        per_skill = all_skill_actions(self.env, env_state)  # (K, A, 2)
        idx = skill_idx[None, :, None]  # (1, A, 1), broadcasts over the xy axis
        return jnp.take_along_axis(per_skill, idx, axis=0)[0]

    # ------------------------------------------------------------------ API

    def reset(self, key: jax.Array):
        """Reset. Non-staggered: the macro state is the base ``EnvState``.

        Staggered: wrap it in a ``StaggeredMacroState`` carrying each agent's
        random low-level onset and a placeholder in-flight skill (unused before
        onset, when the agent is offline/masked).
        """
        if not self.stagger_starts:
            return self.env.reset(key)
        k_env, k_onset = jax.random.split(key)
        obs, env_state = self.env.reset(k_env)
        onset = jax.random.randint(
            k_onset, (self.n_agents,), 0, self.max_start_delay + 1
        ).astype(jnp.int32)
        mstate = StaggeredMacroState(
            env_state=env_state,
            skill_idx=jnp.zeros((self.n_agents,), dtype=jnp.int32),
            onset=onset,
        )
        return obs, mstate

    def _rollout_window(self, state: EnvState, skills: jnp.ndarray, active):
        """Roll `skills` (fixed) forward `macro_len` low-level steps.

        `active` is either ``None`` (all agents participate, the ordinary step)
        or an (A,) bool mask; masked-out agents contribute zero force and are
        dropped from the coupling count for the whole window (threaded into
        ``env.step``), which is what makes an "agent i absent" counterfactual
        exact. The window freezes at the first low-level done. Returns
        ``(final_state, cum_reward, cum_team_reward, terminated, truncated)``
        where ``cum_reward`` accumulates the base env's per-step `reward` (team
        scalar, or per-agent single-step D when the base env is in
        difference_rewards mode) and ``cum_team_reward`` accumulates the team
        scalar `info["task_reward"]`.
        """
        def _low_step(carry, _):
            env_state, done, terminated, truncated, cum_r, cum_task = carry
            actions = self._skill_actions(env_state, skills)
            _, next_state, reward, term, trunc, info = self.env.step(
                env_state, actions, active
            )
            live = ~done  # accumulate/advance only while the episode is live

            # `live` broadcast to the reward rank (scalar team, or (A,) per-agent).
            r_live = jnp.reshape(live, (1,) * reward.ndim) if reward.ndim else live
            cum_r = cum_r + jnp.where(r_live, reward, 0.0)
            cum_task = cum_task + jnp.where(live, info["task_reward"], 0.0)

            # Freeze the state at the first done step; keep re-deciding otherwise.
            def _freeze(n, c):
                d = jnp.reshape(done, done.shape + (1,) * (c.ndim - done.ndim))
                return jnp.where(d, c, n)

            frozen_state = jax.tree.map(_freeze, next_state, env_state)
            return (
                frozen_state,
                done | term | trunc,
                terminated | (term & live),
                truncated | (trunc & live),
                cum_r,
                cum_task,
            ), None

        zero_r = jnp.zeros(
            (self.n_agents,) if self.reward_mode == "difference_rewards" else ()
        )
        init = (
            state,
            jnp.zeros((), dtype=bool),
            jnp.zeros((), dtype=bool),
            jnp.zeros((), dtype=bool),
            zero_r,
            jnp.zeros(()),
        )
        (final_state, _, terminated, truncated, cum_r, cum_task), _ = jax.lax.scan(
            _low_step, init, None, length=self.macro_len
        )
        return final_state, cum_r, cum_task, terminated, truncated

    def _step_windowed(self, state: EnvState, skills: jnp.ndarray):
        """One macro decision with the **windowed** difference reward.

        The per-agent reward is the exact windowed counterfactual
        ``D_i = G(window) - G_{-i}(window)``, where ``G_{-i}`` is the team reward
        of the *same* macro window re-rolled with agent i absent for the whole
        window (zero force + dropped from the coupling count). Unlike summing the
        base env's single-step D, holding agent i absent across the window lets
        the coupling mechanic act — the box stays heavy and stalls if i was a
        required member — so the credit reflects coalition necessity, not just an
        instantaneous force share. Exact because the scripted skills and the MJX
        step are deterministic, so the fork replays identically.

        NOTE: this reveals the coupling only once the window is long enough for
        the mass change to integrate into a displacement difference (empirically
        n >= ~30 low-level steps for this task); at the default macro_len=10 it is
        still mostly additive. It is a single-macro-window counterfactual — it
        does not span future decisions (that needs the policy, i.e. the trainer).
        """
        # Factual window (all agents active) — its termination is the real one.
        final_state, _, g_factual, terminated, truncated = self._rollout_window(
            state, skills, None
        )

        # Counterfactual windows: agent i absent for the whole window. vmap over
        # the agent axis runs all A forks in one compiled call from the same
        # start state (common random numbers), like the difference-reward oracle.
        def _counterfactual(i):
            active = jnp.arange(self.n_agents) != i
            _, _, g_cf, _, _ = self._rollout_window(state, skills, active)
            return g_cf

        g_cf = jax.vmap(_counterfactual)(jnp.arange(self.n_agents))  # (A,)
        reward = g_factual - g_cf  # (A,) per-agent windowed difference reward

        obs = self.env._get_obs(final_state.data)
        info = {"task_reward": g_factual, "delivered": final_state.delivered}
        return obs, final_state, reward, terminated, truncated, info

    def _step_staggered(self, mstate: "StaggeredMacroState", proposed: jnp.ndarray):
        """One global macro window under per-agent staggered onsets / phases.

        The policy is queried once per window (``proposed`` (A,), off the
        window-start obs). Within the window each agent re-decides at *its own*
        phase boundary — the low-level step ``t`` where ``t >= onset`` and
        ``(t - onset) % macro_len == 0`` — adopting ``proposed[i]`` there and
        flying its previous ``skill_idx`` until then. Because ``onset`` is not a
        multiple of ``macro_len``, those boundaries never coincide (persistent
        asynchrony), yet each online agent hits exactly one boundary per window, so
        the trainer still records exactly one transition per agent per window.

        An agent is *offline* (null force + dropped from the coupling count, via the
        base env's per-step ``active`` mask) until its onset. ``info["active"]`` is
        the per-agent mask of who made a decision this window (offline-all-window
        agents are 0), which the trainer uses to drop their transition from the
        loss. Reward is the team scalar accumulated over the window.
        """
        macro = self.macro_len
        onset = mstate.onset  # (A,) absolute low-level onset step

        def _low(carry, _):
            env_state, skill_idx, done, term, trunc, cum_task, decided_any = carry
            t = env_state.t  # absolute low-level step (pre-step), shared clock
            online = t >= onset  # (A,) bool — has this agent come online?
            decide = online & (((t - onset) % macro) == 0)  # (A,) own-phase boundary
            skill_idx = jnp.where(decide, proposed, skill_idx)

            actions = self._skill_actions(env_state, skill_idx)
            # `online` nulls offline agents' force and drops them from coupling.
            _, next_state, _, term_s, trunc_s, info = self.env.step(
                env_state, actions, online
            )
            live = ~done
            cum_task = cum_task + jnp.where(live, info["task_reward"], 0.0)
            decided_any = decided_any | (decide & live)

            # Freeze the env state at the first done step (matches _rollout_window).
            frozen = jax.tree.map(
                lambda n, c: jnp.where(done, c, n), next_state, env_state
            )
            return (
                frozen,
                skill_idx,
                done | term_s | trunc_s,
                term | (term_s & live),
                trunc | (trunc_s & live),
                cum_task,
                decided_any,
            ), None

        init = (
            mstate.env_state,
            mstate.skill_idx,
            jnp.zeros((), dtype=bool),
            jnp.zeros((), dtype=bool),
            jnp.zeros((), dtype=bool),
            jnp.zeros(()),
            jnp.zeros((self.n_agents,), dtype=bool),
        )
        (final_state, final_skill, _, terminated, truncated, cum_task, decided), _ = (
            jax.lax.scan(_low, init, None, length=macro)
        )

        new = StaggeredMacroState(
            env_state=final_state, skill_idx=final_skill, onset=onset
        )
        obs = self.env._get_obs(final_state.data)
        info = {
            "task_reward": cum_task,
            "delivered": final_state.delivered,
            "active": decided.astype(jnp.float32),  # who decided this window
        }
        return obs, new, cum_task, terminated, truncated, info

    def step(self, state, skills: jnp.ndarray):
        """One macro decision: run `skills` (A,) for `macro_len` low-level steps.

        Returns ``(obs, state, reward, terminated, truncated, info)`` with the
        base env's shapes: reward is the team scalar accumulated over the window
        (or per-agent under ``reward_mode="difference_rewards"`` / the windowed
        mode), and ``info["task_reward"]`` always carries the accumulated **team**
        scalar for logging/eval parity across reward modes.
        """
        skills = skills.astype(jnp.int32)
        if self.stagger_starts:
            return self._step_staggered(state, skills)
        if self.reward_mode == WINDOWED_DIFFERENCE_REWARDS:
            return self._step_windowed(state, skills)

        final_state, cum_r, cum_task, terminated, truncated = self._rollout_window(
            state, skills, None
        )
        obs = self.env._get_obs(final_state.data)
        info = {"task_reward": cum_task, "delivered": final_state.delivered}
        return obs, final_state, cum_r, terminated, truncated, info

    @staticmethod
    def base_state(state):
        """Underlying base ``EnvState`` from either macro-state representation."""
        return state.env_state if isinstance(state, StaggeredMacroState) else state


if __name__ == "__main__":
    import argparse
    import time

    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=9)
    parser.add_argument("--n-objects", type=int, default=3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--n-envs", type=int, default=16)
    args = parser.parse_args()

    base = MultiBoxPushMJX(n_agents=args.n_agents, n_objects=args.n_objects)
    async_env = AsyncMacroMJX(base, d_min=5, d_max=15, stagger=True)
    sync_env = AsyncMacroMJX(base, d_min=10, d_max=10, stagger=False)
    assert sync_env.is_synchronous and not async_env.is_synchronous

    key = jax.random.PRNGKey(0)
    step = jax.jit(async_env.step)
    obs, state = jax.jit(async_env.reset)(key)
    assert obs.shape == (args.n_agents, MACRO_OBS_DIM), obs.shape
    print(f"obs {obs.shape} (base {OBS_DIM} + commitment {COMMITMENT_FEAT_DIM}) OK")

    # --- async rollout: decision points must be staggered ---
    t0 = time.time()
    k = key
    decide_steps, total = [], 0.0
    for i in range(args.steps):
        k, sub = jax.random.split(k)
        proposed = jax.random.randint(sub, (args.n_agents,), 0, N_SKILLS)
        obs, state, r, term, trunc, info = step(state, proposed)
        total += float(r)
        decide_steps.append(np.asarray(info["decided"]))
        if bool(term) or bool(trunc):
            break
    decide_steps = np.stack(decide_steps)  # (T, A)
    n_decisions = decide_steps.sum(axis=0)
    # Steps where *some* but not *all* agents re-decided == genuine asynchrony.
    per_step = decide_steps.sum(axis=1)
    staggered = int(((per_step > 0) & (per_step < args.n_agents)).sum())
    print(
        f"async: return {total:.1f}, decisions/agent {n_decisions.tolist()}, "
        f"{staggered}/{len(per_step)} steps had a partial (staggered) decision set "
        f"({time.time() - t0:.1f}s incl. compile)"
    )
    assert staggered > 0, "async mode produced no staggered decisions"

    # --- sync control: every decision step must be unanimous ---
    sync_step = jax.jit(sync_env.step)
    obs, state = jax.jit(sync_env.reset)(key)
    k, per_step_sync = key, []
    for i in range(args.steps):
        k, sub = jax.random.split(k)
        proposed = jax.random.randint(sub, (args.n_agents,), 0, N_SKILLS)
        obs, state, r, term, trunc, info = sync_step(state, proposed)
        per_step_sync.append(int(np.asarray(info["decided"]).sum()))
        if bool(term) or bool(trunc):
            break
    assert set(per_step_sync) <= {0, args.n_agents}, (
        f"sync mode leaked a partial decision set: {sorted(set(per_step_sync))}"
    )
    fires = [i for i, c in enumerate(per_step_sync) if c == args.n_agents]
    print(f"sync control: unanimous decisions at steps {fires[:8]}... (lockstep OK)")

    # --- vmap ---
    v_reset = jax.jit(jax.vmap(async_env.reset))
    v_step = jax.jit(jax.vmap(async_env.step))
    keys = jax.random.split(key, args.n_envs)
    obs, vstate = v_reset(keys)
    proposed = jax.random.randint(key, (args.n_envs, args.n_agents), 0, N_SKILLS)
    obs, vstate, r, term, trunc, info = v_step(vstate, proposed)
    assert obs.shape == (args.n_envs, args.n_agents, MACRO_OBS_DIM), obs.shape
    print(f"vmap({args.n_envs}) OK: obs {obs.shape}, reward {r.shape}")

    # ---------------------------------------------------------------------- #
    # SyncMacroMJX: the hierarchical training env (one step == one decision). #
    # ---------------------------------------------------------------------- #
    print("\n=== SyncMacroMJX (options / one step == one macro decision) ===")
    macro_len = 10
    sync_macro = SyncMacroMJX(base, macro_len=macro_len)
    assert sync_macro.observation_dim == OBS_DIM
    assert sync_macro.action_dim == N_SKILLS and sync_macro.discrete
    print(
        f"obs_dim={sync_macro.observation_dim} action_dim={sync_macro.action_dim} "
        f"(discrete skills) macro_len={macro_len} "
        f"macro_horizon={sync_macro.max_steps} (base {base.max_steps} low-level)"
    )

    m_reset = jax.jit(sync_macro.reset)
    m_step = jax.jit(sync_macro.step)

    t0 = time.time()
    obs, state = m_reset(key)
    assert obs.shape == (args.n_agents, OBS_DIM), obs.shape

    # Each call advances the physics by macro_len low-level steps and returns one
    # transition. Reward is the summed team reward over the window.
    k, decisions, total = key, 0, 0.0
    for _ in range(sync_macro.max_steps):
        k, sub = jax.random.split(k)
        proposed = jax.random.randint(sub, (args.n_agents,), 0, N_SKILLS)
        obs, state, r, term, trunc, info = m_step(state, proposed)
        assert r.shape == (), r.shape  # scalar team reward per decision
        total += float(r)
        decisions += 1
        if bool(term) or bool(trunc):
            break
    # An all-random-skill policy typically truncates without delivering; the
    # point of the smoke test is the *interface*, not the return.
    print(
        f"sync-macro rollout: {decisions} decisions "
        f"(== {decisions * macro_len} low-level steps), summed team return "
        f"{total:.2f} ({time.time() - t0:.1f}s incl. compile)"
    )
    assert decisions <= sync_macro.max_steps

    # Reward really is accumulated over the window: one macro step must move the
    # physics as far as macro_len base steps. Everyone pushes; we warm up until
    # the box is moving so the compared window carries a *non-zero* reward.
    base_step = jax.jit(base.step)
    push = jnp.ones((args.n_agents,), jnp.int32)  # everyone -> "push"
    _, s0 = jax.jit(sync_macro.reset)(key)
    for _ in range(20):  # engage the box (shaping reward becomes non-zero)
        _, s0, r_warm, term, trunc, _ = m_step(s0, push)
        if float(r_warm) > 0.05 or bool(term) or bool(trunc):
            break

    _, _, r_macro, _, _, _ = m_step(s0, push)
    es, r_manual = s0, 0.0  # same start, same skills, summed by hand
    for _ in range(macro_len):
        acts = sync_macro._skill_actions(es, push)
        _, es, br, bterm, btrunc, _ = base_step(es, acts)
        r_manual += float(br)
        if bool(bterm) or bool(btrunc):
            break
    print(
        f"accumulation check: macro reward {float(r_macro):.4f} vs "
        f"hand-summed {r_manual:.4f} (should match to f32, and be non-zero)"
    )
    assert abs(float(r_macro) - r_manual) < 1e-3, (float(r_macro), r_manual)
    assert abs(float(r_macro)) > 1e-3, "warm-up failed to reach an engaged state"

    # vmap over envs (what the mappo_jax collector does).
    vm_reset = jax.jit(jax.vmap(sync_macro.reset))
    vm_step = jax.jit(jax.vmap(sync_macro.step))
    keys = jax.random.split(key, args.n_envs)
    obs, vstate = vm_reset(keys)
    proposed = jax.random.randint(key, (args.n_envs, args.n_agents), 0, N_SKILLS)
    obs, vstate, r, term, trunc, info = vm_step(vstate, proposed)
    assert obs.shape == (args.n_envs, args.n_agents, OBS_DIM), obs.shape
    assert r.shape == (args.n_envs,), r.shape
    print(f"vmap({args.n_envs}) OK: obs {obs.shape}, reward {r.shape}")
    print("SyncMacroMJX smoke test passed.")

    # ---------------------------------------------------------------------- #
    # Windowed macro difference reward (coalition-revealing counterfactual).  #
    # ---------------------------------------------------------------------- #
    print("\n=== windowed macro difference reward (D_i = G - G_-i over window) ===")
    w_env = SyncMacroMJX(base, macro_len=30, reward_mode=WINDOWED_DIFFERENCE_REWARDS)
    assert w_env.per_agent_rewards and w_env.reward_mode == WINDOWED_DIFFERENCE_REWARDS
    assert base.reward_mode == "dense", "windowed mode needs a dense base env"

    # s0 is the engaged state warmed up above (box moving, coalition formed).
    _, _, D, _, _, info_w = jax.jit(w_env.step)(s0, push)
    assert D.shape == (args.n_agents,), D.shape
    G = float(info_w["task_reward"])
    print(
        f"per-agent windowed D shape {D.shape}; team G(window)={G:.3f}, "
        f"sum_i D_i={float(D.sum()):.3f}, ratio sum_i D_i / G = {float(D.sum())/(G+1e-8):.2f}"
    )

    # Exactness: the wrapper's D[0] must equal a hand-rolled factual-minus-
    # counterfactual fork (agent 0 absent = zero force + dropped from coupling).
    base_step_active = jax.jit(base.step)

    def _manual_window(active):
        es, g = s0, 0.0
        for _ in range(w_env.macro_len):
            acts = w_env._skill_actions(es, push)
            _, es, _, bterm, btrunc, binfo = base_step_active(es, acts, active)
            g += float(binfo["task_reward"])
            if bool(bterm) or bool(btrunc):
                break
        return g

    g_fact = _manual_window(jnp.ones((args.n_agents,), bool))
    g_cf0 = _manual_window(jnp.arange(args.n_agents) != 0)
    d0_manual = g_fact - g_cf0
    print(
        f"exactness check: wrapper D[0]={float(D[0]):.4f} vs manual fork "
        f"{d0_manual:.4f} (deterministic, should match to f32)"
    )
    assert abs(float(D[0]) - d0_manual) < 1e-2, (float(D[0]), d0_manual)

    # Coalition structure emerges with window length: summing single-step D stays
    # ~additive (ratio ~1), but holding an agent absent for a *longer* window lets
    # the coupling stall the box, so the ratio climbs toward the coupling number.
    print("  ratio sum_i D_i / G by window length (coupling reveals as it grows):")
    for ml in (1, 5, 15, 30):
        we = SyncMacroMJX(base, macro_len=ml, reward_mode=WINDOWED_DIFFERENCE_REWARDS)
        _, _, Dml, _, _, iw = jax.jit(we.step)(s0, push)
        Gml = float(iw["task_reward"])
        print(f"    macro_len={ml:2d}: {float(Dml.sum())/(Gml+1e-8):+.2f}")
    print("Windowed difference-reward smoke test passed.")

    # ---------------------------------------------------------------------- #
    # Staggered starts (async onset / decoupled decision phases).            #
    # ---------------------------------------------------------------------- #
    print("\n=== SyncMacroMJX staggered starts (async onsets) ===")
    stag = SyncMacroMJX(base, macro_len=macro_len, stagger_starts=True,
                        max_start_delay=3 * macro_len)
    s_reset, s_step = jax.jit(stag.reset), jax.jit(stag.step)
    obs, st = s_reset(key)
    assert isinstance(st, StaggeredMacroState)
    assert obs.shape == (args.n_agents, OBS_DIM), obs.shape
    onset = np.asarray(st.onset)
    phases = (onset % macro_len).tolist()
    print(f"onset(low-level)={onset.tolist()} phases={phases}")
    assert len(set(phases)) > 1, "decision phases should be decoupled"

    k = key
    for w in range(stag.max_steps):
        k, sub = jax.random.split(k)
        proposed = jax.random.randint(sub, (args.n_agents,), 0, N_SKILLS)
        obs, st, r, term, trunc, info = s_step(st, proposed)
        assert r.shape == (), r.shape
        active = np.asarray(info["active"])
        done = bool(term) or bool(trunc)
        # active[i] => agent i decided => it was online (onset <= last window step)
        expected_online = onset <= (w + 1) * macro_len - 1
        assert np.all(active.astype(bool) <= expected_online), (w, active)
        if not done:  # full windows match the schedule exactly
            assert np.array_equal(active, expected_online.astype(np.float32)), w
        else:
            break
    v_sreset = jax.jit(jax.vmap(stag.reset))
    v_sstep = jax.jit(jax.vmap(stag.step))
    keys = jax.random.split(key, args.n_envs)
    obs, vst = v_sreset(keys)
    proposed = jax.random.randint(key, (args.n_envs, args.n_agents), 0, N_SKILLS)
    obs, vst, r, term, trunc, info = v_sstep(vst, proposed)
    assert obs.shape == (args.n_envs, args.n_agents, OBS_DIM), obs.shape
    assert r.shape == (args.n_envs,) and info["active"].shape == (
        args.n_envs, args.n_agents)
    print(f"vmap({args.n_envs}) OK: obs {obs.shape}, active {info['active'].shape}")
    print("Staggered-starts smoke test passed.")
