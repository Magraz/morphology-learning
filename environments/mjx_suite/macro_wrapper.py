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
        **env_kwargs,
    ):
        if macro_len < 1:
            raise ValueError(f"macro_len must be >= 1, got {macro_len}")

        self.env = env if env is not None else MultiBoxPushMJX(**env_kwargs)
        self.n_agents = self.env.n_agents
        self.n_skills = N_SKILLS
        self.macro_len = int(macro_len)

        # Interface the mappo_jax trainer/runner reads off the env (matching the
        # attribute names of MultiBoxPushMJX): discrete skill selection.
        self.observation_dim = OBS_DIM
        self.action_dim = N_SKILLS
        self.discrete = True
        self.reward_mode = self.env.reward_mode
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

    def reset(self, key: jax.Array) -> tuple[jnp.ndarray, EnvState]:
        """Delegates to the base env; the macro state is the base EnvState."""
        return self.env.reset(key)

    def step(self, state: EnvState, skills: jnp.ndarray):
        """One macro decision: run `skills` (A,) for `macro_len` low-level steps.

        Returns ``(obs, state, reward, terminated, truncated, info)`` with the
        base env's shapes: reward is the team scalar accumulated over the window
        (or per-agent under ``reward_mode="difference_rewards"``, summed the same
        way), and ``info["task_reward"]`` always carries the accumulated **team**
        scalar for logging/eval parity across reward modes.
        """
        skills = skills.astype(jnp.int32)

        def _low_step(carry, _):
            env_state, done, terminated, truncated, cum_r, cum_task = carry
            actions = self._skill_actions(env_state, skills)
            _, next_state, reward, term, trunc, info = self.env.step(
                env_state, actions
            )
            active = ~done  # accumulate/advance only while the episode is live

            # done broadcast to the reward rank (scalar team reward, or (A,)
            # per-agent under difference rewards)
            r_active = jnp.reshape(active, (1,) * reward.ndim) if reward.ndim else active
            cum_r = cum_r + jnp.where(r_active, reward, 0.0)
            cum_task = cum_task + jnp.where(active, info["task_reward"], 0.0)

            # Freeze the state at the first done step; keep re-deciding otherwise.
            def _freeze(n, c):
                d = jnp.reshape(done, done.shape + (1,) * (c.ndim - done.ndim))
                return jnp.where(d, c, n)

            frozen_state = jax.tree.map(_freeze, next_state, env_state)
            return (
                frozen_state,
                done | term | trunc,
                terminated | (term & active),
                truncated | (trunc & active),
                cum_r,
                cum_task,
            ), None

        zero_r = jnp.zeros(() if self.reward_mode != "difference_rewards"
                           else (self.n_agents,))
        init = (
            state,
            jnp.zeros((), dtype=bool),
            jnp.zeros((), dtype=bool),
            jnp.zeros((), dtype=bool),
            zero_r,
            jnp.zeros(()),
        )
        (final_state, done, terminated, truncated, cum_r, cum_task), _ = jax.lax.scan(
            _low_step, init, None, length=self.macro_len
        )

        obs = self.env._get_obs(final_state.data)
        info = {"task_reward": cum_task, "delivered": final_state.delivered}
        return obs, final_state, cum_r, terminated, truncated, info


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
