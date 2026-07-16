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
