"""SMAX (JaxMARL StarCraft) behind the functional array contract the JAX stacks use.

``mappo_jax`` / ``feudal_mappo_jax`` are written against the gymnax-style API of
``MultiBoxPushMJX``::

    reset(key)            -> (obs, state)
    step(state, actions)  -> (obs, state, reward, terminated, truncated, info)

with ``obs`` a dense ``(n_agents, obs_dim)`` array, a **scalar team reward**, separate
``terminated`` / ``truncated`` flags, and **no auto-reset** (the collector supplies the
resets itself).

SMAX is a different shape on every one of those axes: a PettingZoo-parallel API keyed by
agent name, a key-first ``step(key, state, actions)`` (the heuristic enemy is stochastic),
auto-reset inside ``MultiAgentEnv.step``, a single combined ``dones["__all__"]``, and
legal-action masks. :class:`SMAXAdapter` reconciles the two so that **no trainer ever
sees a dict**, and exposes three optional hooks the MJX envs do not have
(``avail_actions``, ``global_state``, ``info["active"]``) which the trainers treat as
opt-in — every existing MJX arm is unaffected.

Demo / smoke test::

    uv run python -m environments.smax.smax_env
"""

from __future__ import annotations

import dataclasses
from typing import Any

import environments.smax._compat  # noqa: F401  — MUST precede the jaxmarl import

import jax
import jax.numpy as jnp

import jaxmarl
from jaxmarl.environments.smax import map_name_to_scenario


# Default scenario. "3m" is SMAC's smallest symmetric map — the right size for a smoke
# test and fast enough that a full run is minutes, not hours.
DEFAULT_MAP = "3m"
DEFAULT_ENV_ID = "HeuristicEnemySMAX"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SMAXState:
    """Adapter state: the jaxmarl state plus the two things the contract needs.

    Every leaf must carry a leading ``n_envs`` axis after ``vmap`` and survive
    ``jnp.where`` — the collector's ``_restart_done`` does a ``jax.tree.map`` select of
    reset-vs-current state over the whole pytree. A Python ``int`` or ``None`` leaf here
    would break that, which is why nothing static lives on this dataclass.
    """

    env_state: Any  # jaxmarl SMAX (or EnemySMAX wrapper) State
    # SMAX.step needs a PRNG key; the MJX contract's `step(state, actions)` has nowhere
    # to pass one, so the key rides along in the state and is split every step.
    key: jax.Array
    # The world state that jaxmarl produced alongside *this* state's observation.
    # Carried rather than recomputed so `global_state` can never drift from the obs it
    # is paired with. (n_world,) float32.
    world_state: jax.Array


def _inner(env_state):
    """Unwrap the enemy-policy wrapper to the raw SMAX ``State``.

    ``HeuristicEnemySMAX``/``LearnedPolicyEnemySMAX`` wrap the SMAX state in one that
    also carries the enemy policy's own state; the bare ``SMAX`` env does not. The unit
    arrays (``unit_alive``, ``time``) live on the inner one either way. This is a
    structural (static) test, so it is jit-safe.
    """
    return getattr(env_state, "state", env_state)


class SMAXAdapter:
    """SMAX presented as a functional, array-API, non-auto-resetting env."""

    def __init__(
        self,
        map_name: str = DEFAULT_MAP,
        smax_env_id: str = DEFAULT_ENV_ID,
        max_steps: int | None = None,
        walls_cause_death: bool = True,
        use_self_play_reward: bool = False,
        see_enemy_actions: bool = False,
        **make_kwargs,
    ):
        if use_self_play_reward:
            raise ValueError(
                "SMAXAdapter assumes a shared team reward (it reports one scalar and "
                "sets per_agent_rewards=False); use_self_play_reward=True makes the "
                "reward zero-sum per team and is not supported"
            )

        scenario = map_name_to_scenario(map_name)
        kwargs = dict(
            scenario=scenario,
            walls_cause_death=walls_cause_death,
            use_self_play_reward=use_self_play_reward,
            see_enemy_actions=see_enemy_actions,
            **make_kwargs,
        )
        if max_steps is not None:
            kwargs["max_steps"] = int(max_steps)
        self._env = jaxmarl.make(smax_env_id, **kwargs)

        self.map_name = map_name
        self.smax_env_id = smax_env_id

        # `env.agents` is the set WE control. Under the EnemySMAX wrappers that is the
        # allies only (the enemy team is driven by the built-in heuristic), and the
        # wrapper already filters obs/rewards/dones down to it. Fixed once here and used
        # as the single ordering for every stack/un-stack below.
        self.agents = tuple(self._env.agents)
        self.n_agents = len(self.agents)
        self._num_allies = int(self._env.num_allies)

        a0 = self.agents[0]
        self.observation_dim = int(self._env.observation_space(a0).shape[0])
        self.action_dim = int(self._env.action_space(a0).n)
        self.discrete = True

        self.max_steps = int(self._env.max_steps)

        # `run.py` reads both of these unguarded at construction time.
        self.reward_mode = "dense"
        self.per_agent_rewards = False

        # The critic is sized from `state_size`, but the value actually fed to it is
        # `obs["world_state"]`. Verify the two agree once, here: a jaxmarl version that
        # changed one without the other would otherwise surface as an opaque shape error
        # deep inside the critic (or, worse, a checkpoint that silently fails to load).
        # One reset is negligible against a training run.
        self.global_state_dim = int(self._env.state_size)
        probe_obs, _ = self._env.reset(jax.random.PRNGKey(0))
        actual = int(probe_obs["world_state"].shape[-1])
        if actual != self.global_state_dim:
            raise RuntimeError(
                f"SMAX state_size ({self.global_state_dim}) disagrees with the actual "
                f"world_state width ({actual}) on map {map_name!r}; the centralized "
                "critic would be built at the wrong width"
            )
        probe_a0 = int(probe_obs[a0].shape[-1])
        if probe_a0 != self.observation_dim:
            raise RuntimeError(
                f"SMAX observation_space ({self.observation_dim}) disagrees with the "
                f"actual per-agent observation width ({probe_a0}) on map {map_name!r}"
            )

    # ------------------------------------------------------------------ helpers

    def _stack_obs(self, obs_dict) -> jnp.ndarray:
        """(n_agents, obs_dim), agent-major in `self.agents` order.

        That ordering must match everywhere — the trainers flatten obs as
        ``obs.reshape(b * n_agents, obs_dim)`` and pair row ``i`` with agent ``i``'s
        action, log-prob, mask and goal. One inconsistent stack silently trains every
        agent on a neighbour's data.
        """
        return jnp.stack([obs_dict[a] for a in self.agents], axis=0)

    def _alive(self, env_state) -> jnp.ndarray:
        """(n_agents,) float32 — which of OUR units are alive in this state."""
        return _inner(env_state).unit_alive[: self._num_allies].astype(jnp.float32)

    def _outcome(self, env_state):
        """(all_allies_dead, all_enemies_dead) for the given state."""
        alive = _inner(env_state).unit_alive
        return (
            ~jnp.any(alive[: self._num_allies]),
            ~jnp.any(alive[self._num_allies :]),
        )

    # ------------------------------------------------------------------ API

    def reset(self, key: jax.Array):
        reset_key, carry_key = jax.random.split(key)
        obs, env_state = self._env.reset(reset_key)
        state = SMAXState(
            env_state=env_state,
            key=carry_key,
            world_state=obs["world_state"],
        )
        return self._stack_obs(obs), state

    def step(self, state: SMAXState, actions: jnp.ndarray):
        """One env step. `actions` is (n_agents,) integer action indices."""
        step_key, carry_key = jax.random.split(state.key)

        # The alive mask must describe who was alive when they ACTED: the transition
        # stores this step's pre-step obs/action, so credit is masked against the
        # pre-step state, not the post-step one.
        active = self._alive(state.env_state)

        action_dict = {a: actions[i] for i, a in enumerate(self.agents)}

        # `step_env`, NOT `step`: MultiAgentEnv.step auto-resets on done. Running that
        # plus the collector's own restart is a double reset, and it would also make the
        # truncation bootstrap value the post-reset observation instead of the true
        # successor.
        obs, next_env_state, rewards, dones, _ = self._env.step_env(
            step_key, state.env_state, action_dict
        )

        # All allies share one team scalar (SMAX's compute_reward returns one value per
        # team), so any agent's entry IS the team reward.
        reward = rewards[self.agents[0]]

        # SMAX gives only the union. The trainers need the two apart: `truncated` keeps
        # the value bootstrap alive across a time limit, `terminated` cuts it. Splitting
        # `dones["__all__"]` (rather than recomputing the predicate) guarantees
        # `terminated | truncated` is exactly the env's own done flag.
        done = dones["__all__"]
        all_allies_dead, all_enemies_dead = self._outcome(next_env_state)
        battle_over = all_allies_dead | all_enemies_dead
        terminated = done & battle_over
        truncated = done & ~battle_over

        next_state = SMAXState(
            env_state=next_env_state,
            key=carry_key,
            world_state=obs["world_state"],
        )
        info = {
            # Read unconditionally by the collector and by eval; always the team scalar
            # so logged returns stay comparable across reward modes.
            "task_reward": reward,
            # Per-agent activity mask -> Transition.active_mask. Dead units are dropped
            # from the policy/entropy loss; every other env omits this and defaults to
            # all-ones, which leaves those reductions exact plain means.
            "active": active,
            # SMAX ships an empty info dict, so the win flag is synthesized here. A draw
            # (both teams wiped in the same step) counts as a loss, matching SMAX's own
            # won_battle_bonus condition.
            "won_episode": (all_enemies_dead & ~all_allies_dead).astype(jnp.float32),
        }
        return self._stack_obs(obs), next_state, reward, terminated, truncated, info

    # -------------------------------------------------------- optional hooks

    def avail_actions(self, state: SMAXState) -> jnp.ndarray:
        """(n_agents, action_dim) float32 legal-action mask for `state`.

        Presence of this method is what switches the trainers' masking path on.
        """
        avail = self._env.get_avail_actions(state.env_state)
        return jnp.stack([avail[a] for a in self.agents], axis=0).astype(jnp.float32)

    def global_state(self, state: SMAXState) -> jnp.ndarray:
        """(global_state_dim,) centralized critic input.

        SMAX's own world state, not the concatenation of the per-agent observations:
        it is the standard MAPPO-for-SMAX critic input, and it is far smaller
        (72 vs 195 dims at 3m) because it carries absolute unit features once rather
        than one egocentric view per agent. Presence of this method + `global_state_dim`
        is what switches the trainers off their `obs.reshape(n_envs, -1)` default.
        """
        return state.world_state

    # ------------------------------------------------------------- rendering

    def render_episode(self, state_seq, path):
        """Write a gif of one episode via jaxmarl's SMAXVisualizer.

        `state_seq` is a list of `(key, jaxmarl_state, action_dict)` tuples — the
        visualizer replays the env itself, so it needs the raw jaxmarl states, not
        `SMAXState`.
        """
        from jaxmarl.viz.visualizer import SMAXVisualizer

        SMAXVisualizer(self._env, state_seq).animate(view=False, save_fname=str(path))

    def to_action_dict(self, actions: jnp.ndarray) -> dict:
        """(n_agents,) array -> the action dict SMAXVisualizer expects in `state_seq`."""
        return {a: actions[i] for i, a in enumerate(self.agents)}


# ---------------------------------------------------------------------------
# Smoke test: interface conformance + jit/vmap + the contract's invariants.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = SMAXAdapter()
    print(
        f"SMAX {env.smax_env_id} / {env.map_name}: n_agents={env.n_agents} "
        f"obs_dim={env.observation_dim} action_dim={env.action_dim} "
        f"global_state_dim={env.global_state_dim} max_steps={env.max_steps}"
    )

    n_envs = 8
    keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
    obs, state = jax.jit(jax.vmap(env.reset))(keys)
    assert obs.shape == (n_envs, env.n_agents, env.observation_dim), obs.shape

    v_step = jax.jit(jax.vmap(env.step))
    v_avail = jax.jit(jax.vmap(env.avail_actions))
    v_gs = jax.jit(jax.vmap(env.global_state))

    assert v_gs(state).shape == (n_envs, env.global_state_dim)
    assert v_avail(state).shape == (n_envs, env.n_agents, env.action_dim)

    rng = jax.random.PRNGKey(1)
    total_done = 0
    for t in range(env.max_steps + 5):
        rng, sub = jax.random.split(rng)
        mask = v_avail(state)
        # Sample uniformly among LEGAL actions only.
        logits = jnp.where(mask > 0, 0.0, -1e9)
        actions = jax.random.categorical(sub, logits, axis=-1)
        obs, state, reward, terminated, truncated, info = v_step(state, actions)

        assert reward.shape == (n_envs,), reward.shape
        assert info["active"].shape == (n_envs, env.n_agents)
        # terminated and truncated must be mutually exclusive — they are two halves of
        # one done flag, and GAE would double-count if they overlapped.
        assert not bool(jnp.any(terminated & truncated))
        total_done += int(jnp.sum(terminated | truncated))

    print(f"stepped {env.max_steps + 5} steps x {n_envs} envs, {total_done} dones")

    # The state pytree must survive the collector's reset-select.
    done = jnp.zeros(n_envs, dtype=bool).at[0].set(True)
    reset_obs, reset_state = jax.jit(jax.vmap(env.reset))(keys)
    merged = jax.tree.map(
        lambda r, c: jnp.where(done.reshape((-1,) + (1,) * (c.ndim - 1)), r, c),
        reset_state,
        state,
    )
    assert jax.tree.structure(merged) == jax.tree.structure(state)
    print("tree.map reset-select OK — collector _restart_done is compatible")
    print("SMAXAdapter smoke test passed")
