"""Oracle difference rewards by forking the simulator.

`MultiBoxPushMJX` is a *pure* env: `EnvState` (and `MacroState` on top of it) is a
registered dataclass threaded through `step`. So we can fork a state and replay it
under a counterfactual — giving the **exact** difference reward

    D_i = G(z) - G(z_-i + c_i)

with no learned model. Most difference-reward work must approximate this (COMA
with a critic, Dr.Reinforce with a reward net) because their envs cannot rewind.
Here the oracle is ground truth, which is what lets `bias_study.py` *measure*
whether the synchronous estimator is wrong before we build anything to fix it.

Two things make the estimate low-variance and exact:

**Re-decisions must happen.** The counterfactual rollout lets agents adopt a new
skill when their commitment expires. This is essential, not incidental: with
skills frozen for the whole window, `remaining`/`elapsed` never touch the
dynamics, the sync and async estimators coincide *identically*, and the bias
study would report zero for a reason unrelated to the research question.
Durations only bite through *when agents re-decide*.

**Common random numbers.** The factual and all A counterfactual rollouts are
driven by the *same* key sequence, so the behaviour policy's skill proposals and
the sampled durations are identical across branches. The only difference is agent
i's force. Without CRN, D_i would be swamped by sampling noise.

`null_action` (zero force) is the default `c_i`, in the Wolpert-Tumer sense.
"""

import functools

import jax
import jax.numpy as jnp

from environments.mjx_suite.macro_wrapper import AsyncMacroMJX, MacroState

FACTUAL = -1  # override_agent sentinel: no agent is nulled


def uniform_skill_policy(env: AsyncMacroMJX):
    """State-independent behaviour policy.

    Deliberately not state-conditioned: under CRN its proposals are then bitwise
    identical across the factual and counterfactual branches, so D_i isolates the
    *physical* contribution of agent i rather than mixing in the policy's
    reaction to the perturbed state.
    """

    def policy(key, obs):
        return jax.random.randint(key, (env.n_agents,), 0, env.n_skills)

    return policy


@functools.partial(jax.jit, static_argnames=("env", "horizon", "policy"))
def rollout_return(
    env: AsyncMacroMJX,
    mstate: MacroState,
    key: jax.Array,
    override_agent: jax.Array,
    horizon: int,
    policy=None,
    gamma: float = 1.0,
) -> jax.Array:
    """Undiscounted-by-default return over `horizon` low-level steps from `mstate`.

    `override_agent` is traced: `FACTUAL` (-1) runs the factual branch, `i >= 0`
    nulls agent i. Reward accumulation stops at termination (`alive` mask) rather
    than breaking, so the shape stays static for `scan`.
    """
    policy = policy or uniform_skill_policy(env)

    def body(carry, step_key):
        state, ret, alive, disc = carry
        proposed = policy(step_key, None)
        committed, _ = env.commit(state, proposed)
        _, new, reward, term, trunc, _ = env.step_committed(committed, override_agent)
        ret = ret + disc * reward * alive
        alive = alive * (1.0 - (term | trunc).astype(jnp.float32))
        return (new, ret, alive, disc * gamma), None

    init = (mstate, jnp.float32(0.0), jnp.float32(1.0), jnp.float32(1.0))
    (_, ret, _, _), _ = jax.lax.scan(body, init, jax.random.split(key, horizon))
    return ret


@functools.partial(jax.jit, static_argnames=("env", "horizon", "policy"))
def difference_rewards(
    env: AsyncMacroMJX,
    mstate: MacroState,
    key: jax.Array,
    horizon: int,
    policy=None,
    gamma: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Exact `(D, G_factual)` from `mstate`: D is (A,), G_factual is scalar.

    One vmap over `[-1, 0, 1, ..., A-1]` runs the factual and every counterfactual
    in a single compiled call, all sharing `key` (CRN).
    """
    branches = jnp.arange(FACTUAL, env.n_agents)  # (A+1,), branches[0] == FACTUAL

    returns = jax.vmap(
        lambda a: rollout_return(env, mstate, key, a, horizon, policy, gamma)
    )(branches)

    g_factual = returns[0]
    return g_factual - returns[1:], g_factual


def horizon_own_commitment(mstate: MacroState) -> int:
    """`H = d_i` — credit only over the agent's own commitment window."""
    return int(jnp.max(mstate.remaining))


def horizon_commitment_closure(mstate: MacroState) -> int:
    """`H = max_j r_j` — roll until every *current* commitment has cleared.

    The natural scope under asynchrony: the window over which the joint
    commitment that exists *right now* is still partially in force.
    """
    return int(jnp.max(mstate.remaining))


def aligned_belief(mstate: MacroState) -> MacroState:
    """The synchronous estimator's *belief* about the commitment state.

    A practitioner applying a standard synchronous-options difference reward
    implicitly believes all agents share one decision epoch: same phase, same
    duration. This collapses each agent's phase to the joint mean, keeping the
    *physical* state (`env_state`) untouched — so `bias_study` rolls identical
    physics under the true vs. believed commitment structure and attributes the
    entire gap to asynchrony.

    Collapsing to the **mean** (rather than resetting to a nominal `L`) is what
    makes the control condition sound: under true synchrony every agent's phase is
    already identical, so the mean is that value, this function is the identity,
    and `D_sync == D_oracle` *exactly* — bias 0 by construction rather than by
    luck. Any non-zero bias in the control means the harness is broken.
    """
    n = mstate.remaining.shape[0]
    mean_remaining = jnp.round(jnp.mean(mstate.remaining)).astype(jnp.int32)
    mean_elapsed = jnp.round(jnp.mean(mstate.elapsed)).astype(jnp.int32)
    return MacroState(
        env_state=mstate.env_state,
        skill_idx=mstate.skill_idx,
        remaining=jnp.full((n,), mean_remaining, dtype=jnp.int32),
        elapsed=jnp.full((n,), mean_elapsed, dtype=jnp.int32),
        key=mstate.key,
    )
