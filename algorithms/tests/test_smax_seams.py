"""Seam tests for the three env hooks the SMAX port added to the JAX stacks.

These pin the joints where a mistake is SILENT — the run trains, the losses look
healthy, and the policy is wrong:

- a legal-action mask that is applied at sampling but not at update time (or vice
  versa) makes the PPO importance ratio compare two different distributions;
- a mask flattened the wrong way pairs each agent with a neighbour's legal actions;
- a global-state hook that is ignored builds the critic on the wrong input width;
- an env with no hooks must take byte-identical code paths to before they existed,
  or every MJX arm silently moves.

Everything runs on a tiny pure-JAX stub env, so unlike an MJX rollout these are fast and
deterministic. The real SMAX adapter is exercised separately by
``uv run python -m environments.smax.smax_env``.

Run: ``uv run pytest algorithms/tests/test_smax_seams.py -q``
"""

import jax
import jax.numpy as jnp
import pytest

from algorithms.mappo_jax.network import evaluate_action, sample_action
from algorithms.mappo_jax.trainer import global_state_dim, make_train
from algorithms.mappo_jax.types import MAPPOConfig

N_AGENTS, OBS_DIM, ACTION_DIM, MAX_STEPS, GS_DIM = 3, 5, 4, 10, 7


@pytest.fixture(autouse=True)
def _run_on_cpu():
    """Pin these tests to CPU.

    They are tiny, so the GPU buys nothing — and sharing it with a training or
    rendering job makes them fail on cuSolver/OOM errors that look exactly like
    assertion failures. `jax.default_device` is used rather than a module-level
    `JAX_PLATFORMS=cpu`, which only takes effect if this module happens to be imported
    before JAX is initialized (i.e. it depends on pytest collection order).
    """
    with jax.default_device(jax.devices("cpu")[0]):
        yield

# Agent i may not take action i (a fixed, per-agent-distinct pattern — a mask that is
# the same for every agent could not detect an agent-major flattening bug).
ILLEGAL = jnp.arange(N_AGENTS)


def _base_mask():
    m = jnp.ones((N_AGENTS, ACTION_DIM))
    return m.at[jnp.arange(N_AGENTS), ILLEGAL].set(0.0)


class StubEnv:
    """Discrete stub with no optional hooks — the MJX-shaped baseline."""

    n_agents = N_AGENTS
    observation_dim = OBS_DIM
    action_dim = ACTION_DIM
    discrete = True
    max_steps = MAX_STEPS
    reward_mode = "dense"

    def reset(self, key):
        return jax.random.normal(key, (N_AGENTS, OBS_DIM)), {
            "t": jnp.int32(0),
            "key": key,
        }

    def step(self, state, actions):
        t = state["t"] + 1
        obs = jax.random.normal(jax.random.fold_in(state["key"], t), (N_AGENTS, OBS_DIM))
        reward = jnp.float32(actions.sum())
        return (
            obs,
            {"t": t, "key": state["key"]},
            reward,
            jnp.bool_(False),
            t >= MAX_STEPS,
            {"task_reward": reward},
        )


class MaskedStubEnv(StubEnv):
    """Adds all three SMAX-style hooks: masks, a real global state, a dead agent."""

    def avail_actions(self, state):
        return _base_mask()

    def global_state(self, state):
        # Deliberately NOT obs.reshape(-1): a different width proves the hook is used.
        return jnp.arange(GS_DIM, dtype=jnp.float32) + state["t"]

    global_state_dim = GS_DIM

    def step(self, state, actions):
        obs, s, r, term, trunc, info = super().step(state, actions)
        # Agent 0 is "dead": present in the batch, excluded from the loss.
        info["active"] = jnp.array([0.0] + [1.0] * (N_AGENTS - 1))
        return obs, s, r, term, trunc, info


def _config(**kw):
    base = dict(
        n_steps=6, n_envs=4, n_total_steps=192, n_epochs=1, n_minibatches=2,
        hidden_dim=8, n_eval_episodes=2,
    )
    base.update(kw)
    return MAPPOConfig(**base)


def _collect(env):
    init_fn, collect_fn, update_fn, eval_fn, _ = make_train(_config(), env)
    rs = init_fn(jax.random.PRNGKey(0))
    rs, traj, last_value, _ = collect_fn(rs)
    return rs, traj, last_value, update_fn, eval_fn


# --------------------------------------------------------------------- masking


def test_masked_actions_are_never_sampled_or_stored():
    """No illegal action appears anywhere in a rollout, and the stored mask matches."""
    _, traj, _, _, _ = _collect(MaskedStubEnv())

    assert traj.action_mask.ndim == 4, "a real mask must be stored, not the placeholder"
    # (n_steps, n_envs, n_agents) actions vs each agent's own forbidden action.
    forbidden = ILLEGAL.reshape(1, 1, N_AGENTS)
    assert not bool(jnp.any(traj.action == forbidden)), (
        "an illegal action was sampled — the mask is not reaching sample_action, "
        "or it is flattened out of agent-major order"
    )
    # The stored mask must be the env's, not all-ones.
    assert bool(jnp.all(traj.action_mask == _base_mask()))


def test_ppo_ratio_is_exactly_one_before_any_update_under_masking():
    """The update must re-evaluate the SAME masked distribution that acted.

    If the mask is applied at sampling but dropped at update time (or flattened
    differently), `evaluate_action` scores the actions under an unmasked categorical and
    the ratio departs from 1 before a single gradient step — silently reweighting every
    sample.
    """
    rs, traj, _, _, _ = _collect(MaskedStubEnv())

    n_steps, n_envs = traj.action.shape[:2]
    flat = n_steps * n_envs * N_AGENTS
    log_probs, _ = evaluate_action(
        rs.train_state.actor_ts.apply_fn,
        rs.train_state.actor_ts.params,
        traj.obs.reshape(flat, OBS_DIM),
        traj.action.reshape(flat),
        discrete=True,
        action_mask=traj.action_mask.reshape(flat, ACTION_DIM),
    )
    ratio = jnp.exp(log_probs - traj.log_prob.reshape(flat))
    assert jnp.allclose(ratio, 1.0, atol=1e-5), float(jnp.abs(ratio - 1.0).max())


def test_mask_pairs_each_agent_with_its_own_legal_actions():
    """Agent-major flattening: row i*n_agents+a must carry agent a's mask.

    Reversing the per-agent mask changes which actions are reachable, so a stack that
    ignored the pairing would produce the same action set either way.
    """
    rng = jax.random.PRNGKey(0)
    env = MaskedStubEnv()
    obs, _ = env.reset(rng)

    def sample_with(mask, key):
        actions, _ = sample_action(
            key,
            lambda p, o: jnp.zeros((o.shape[0], ACTION_DIM)),  # uniform logits
            {},
            obs,
            discrete=True,
            action_mask=mask,
        )
        return actions

    mask = _base_mask()
    # The mask must not be symmetric, or reversing it proves nothing.
    assert not bool(jnp.all(mask == mask[::-1]))

    # With uniform logits the mask is the ONLY thing shaping the draw. Under the
    # correct per-agent pairing, agent a can never draw action a...
    correct = jnp.stack(
        [sample_with(mask, jax.random.fold_in(rng, i)) for i in range(200)]
    )
    assert not bool(jnp.any(correct == ILLEGAL)), "agent drew its own forbidden action"

    # ...whereas under a misaligned (reversed) pairing it demonstrably can. This is
    # what makes the test above evidence of pairing rather than of masking alone.
    misaligned = jnp.stack(
        [sample_with(mask[::-1], jax.random.fold_in(rng, i)) for i in range(200)]
    )
    assert bool(jnp.any(misaligned == ILLEGAL)), (
        "a reversed mask produced the same action set as the correct one — this test "
        "cannot distinguish agent-major pairing from plain masking"
    )


def test_deterministic_eval_respects_the_mask():
    """An unmasked argmax picks the top logit even when illegal."""
    logits = jnp.array([[10.0, 0.0, 1.0, 2.0]])  # action 0 is the argmax
    mask = jnp.array([[0.0, 1.0, 1.0, 1.0]])  # ...and it is illegal
    action, _ = sample_action(
        jax.random.PRNGKey(0),
        lambda p, o: logits,
        {},
        jnp.zeros((1, OBS_DIM)),
        discrete=True,
        deterministic=True,
        action_mask=mask,
    )
    assert int(action[0]) == 3, "argmax ignored the mask and chose an illegal action"


# ---------------------------------------------------------------- global state


def test_global_state_hook_is_used_and_sizes_the_critic():
    _, traj, _, _, _ = _collect(MaskedStubEnv())
    assert traj.global_state.shape[-1] == GS_DIM, traj.global_state.shape
    # ...and is genuinely the env's, not a reshape of the observations.
    assert GS_DIM != OBS_DIM * N_AGENTS
    assert global_state_dim(MaskedStubEnv()) == GS_DIM


def test_global_state_falls_back_to_concatenated_obs():
    _, traj, _, _, _ = _collect(StubEnv())
    assert traj.global_state.shape[-1] == OBS_DIM * N_AGENTS
    assert global_state_dim(StubEnv()) == OBS_DIM * N_AGENTS


# ------------------------------------------------------------- inert by default


def test_env_without_hooks_stores_a_placeholder_mask():
    """No `avail_actions` => no (T,E,N,A) buffer and no masking branch in the update.

    The per-step placeholder is a scalar; `lax.scan` stacks it to `(n_steps,)`. What
    matters is that it is not 4-D — that is the static flag `ppo_update` switches on —
    and that it costs `n_steps` floats rather than a full rollout-sized buffer.
    """
    cfg = _config()
    _, traj, _, _, _ = _collect(StubEnv())
    assert traj.action_mask.ndim != 4, "the masking branch must stay off"
    assert traj.action_mask.shape == (cfg.n_steps,), traj.action_mask.shape
    full_buffer = cfg.n_steps * cfg.n_envs * N_AGENTS * ACTION_DIM
    assert traj.action_mask.size < full_buffer


def test_active_mask_defaults_to_all_ones_without_the_hook():
    _, traj, _, _, _ = _collect(StubEnv())
    assert bool(jnp.all(traj.active_mask == 1.0))


def test_dead_agents_are_excluded_from_the_policy_loss():
    """A dead agent's transition must not move the actor.

    Compared against the same rollout with the agent marked alive: if `active_mask` were
    ignored, the two updates would produce identical params.
    """
    rs, traj, last_value, update_fn, _ = _collect(MaskedStubEnv())
    assert bool(jnp.all(traj.active_mask[..., 0] == 0.0)), "stub should mark agent 0 dead"

    masked_rs, _ = update_fn(rs, traj, last_value)
    all_alive = traj._replace(active_mask=jnp.ones_like(traj.active_mask))
    unmasked_rs, _ = update_fn(rs, all_alive, last_value)

    diffs = jax.tree.map(
        lambda a, b: jnp.abs(a - b).max(),
        masked_rs.train_state.actor_ts.params,
        unmasked_rs.train_state.actor_ts.params,
    )
    assert max(float(d) for d in jax.tree.leaves(diffs)) > 1e-8, (
        "masking a dead agent changed nothing — active_mask is not reaching the loss"
    )


@pytest.mark.parametrize("stack", ["mappo_jax", "feudal_mappo_jax"])
def test_both_stacks_expose_the_same_hook_contract(stack):
    """The two trainers are near-copies and CLAUDE.md requires they stay in sync."""
    mod = __import__(f"algorithms.{stack}.trainer", fromlist=["global_state_dim"])
    assert mod.global_state_dim(StubEnv()) == OBS_DIM * N_AGENTS
    assert mod.global_state_dim(MaskedStubEnv()) == GS_DIM
