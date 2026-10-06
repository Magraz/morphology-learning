"""Training-time simulator forks for the simplified waypoint hierarchy.

Under `interventions: true`, at every `intervention_interval`-th manager
decision, `trainer.make_train` forks each env once per agent i before the first
worker step of the window:

* N ~ U{1 .. n_agents - 1} teammates of agent i, drawn uniformly without
  replacement, are teleported within `intervention_radius` world units of it
  (`env.teleport_agents`, which owns the geometry);
* each recruit gets agent i's realized waypoint offset `w_i - s_i`, measured
  from its own new position; agent i and every other agent keep their
  positions and waypoints;
* the fork runs one `goal_horizon` window with the same frozen policies and is
  then discarded. It is its own episode: its worker steps end with `done` on the
  window's last step, and its single manager transition is a terminal whose
  return is the fork's discounted team reward. No time-limit bootstrap.

Fork transitions join both levels' PPO batches as extra env columns
(`types.Intervention`). Only agent i's manager action is a policy sample in its
fork (the recruits' goals were imposed), so it alone carries manager actor
weight there; every live agent's worker action is a real sample. The manager
critic reads a context block (`critic_context`) so that it can tell a
continuing main return from a one-window fork return. Execution and evaluation
are unchanged. Plan: `plans/simplified_feudal_interventions_2026-10-05.md`.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

# `fold_in` constant for every fork random draw, so enabling forks consumes no
# split of the main path's keys: the main rollout is unchanged by them.
FORK_KEY_NAMESPACE = 0x51F0


def validate_interventions(config, env, n_windows: int) -> None:
    """Raise on a fork configuration that cannot run as specified."""
    interval = config.intervention_interval
    if not config.interventions:
        if interval != 1:
            raise ValueError(
                f"intervention_interval={interval} has no effect without "
                "interventions: true (it would silently train the parent arm)"
            )
        return
    if config.manager_credit != "team":
        raise ValueError(
            "interventions support manager_credit='team' only, got "
            f"{config.manager_credit!r}"
        )
    if env.n_agents < 2:
        raise ValueError(
            "interventions need at least two agents (a fork recruits teammates), "
            f"got n_agents={env.n_agents}"
        )
    if not config.intervention_radius > 0:
        raise ValueError(
            f"intervention_radius must be > 0, got {config.intervention_radius}"
        )
    if not hasattr(env, "teleport_agents"):
        raise ValueError(
            f"interventions need an env with a `teleport_agents` hook; "
            f"{type(env).__name__} has none. Supported: multi_box_push_mjx."
        )
    if int(interval) != interval or interval < 1:
        raise ValueError(f"intervention_interval must be an integer >= 1, got {interval}")
    if math.ceil(n_windows / interval) < 2:
        # Each fork column is normalized over its own intervention windows, and an
        # unbiased std over one decision is undefined.
        raise ValueError(
            f"intervention_interval={interval} leaves "
            f"{math.ceil(n_windows / interval)} intervention window(s) in a "
            f"{n_windows}-window rollout; at least 2 are needed"
        )


def context_dim(config, n_agents: int) -> int:
    """Width of the manager critic's context block: 0 when forks are off."""
    return n_agents + 2 if config.interventions else 0


def critic_context(n_agents: int, batch: int, focal=None, n_recruits=None):
    """`(B, n_agents + 2)` manager critic context:
    `[is_intervention, one_hot(focal), N / (n_agents - 1)]`. All zeros for main
    rows (`focal=None`).

    N is drawn independently of the manager's action, so conditioning the
    critic on it does not make the baseline depend on the action it credits.
    The sampled goal and the post-teleport state are deliberately absent: both
    depend on that action.
    """
    if focal is None:
        return jnp.zeros((batch, n_agents + 2))
    return jnp.concatenate(
        [
            jnp.ones((batch, 1)),
            jax.nn.one_hot(focal, n_agents),
            (n_recruits / (n_agents - 1))[:, None],
        ],
        axis=-1,
    )


def focal_agents(n_envs: int, n_agents: int):
    """`(n_envs * n_agents,)` the focal agent of each fork lane: lane
    `e * n_agents + i` forks env e around agent i (env-major, as `to_lanes`)."""
    return jnp.tile(jnp.arange(n_agents), n_envs)


def to_lanes(tree, n_agents: int):
    """Repeat every leaf's leading env axis `n_agents` times, env-major."""
    return jax.tree.map(lambda x: jnp.repeat(x, n_agents, axis=0), tree)


def sample_recruits(key, focal, n_agents: int):
    """`(n (L,), mask (L, n_agents))` per fork lane: N ~ U{1..n_agents-1} and N
    distinct teammates of the focal agent, uniformly without replacement.

    A fixed-length mask, so no shape depends on N: rank the teammates by
    uniform noise (the focal agent ranked last) and keep ranks below N.
    """
    k_n, k_rank = jax.random.split(key)
    n = jax.random.randint(k_n, focal.shape, 1, n_agents)
    noise = jax.random.uniform(k_rank, focal.shape + (n_agents,))
    noise = jnp.where(jnp.arange(n_agents) == focal[:, None], jnp.inf, noise)
    rank = jnp.argsort(jnp.argsort(noise, axis=-1), axis=-1)
    return n, rank < n[:, None]


def applied_windows(n_windows: int, interval: int) -> np.ndarray:
    """`(n_windows,)` bool — the rollout windows that fork."""
    return np.arange(n_windows) % interval == 0


def simulator_steps_per_update(config, n_agents: int) -> int:
    """Env step calls one update makes, which is what `n_total_steps` caps and
    `total_steps` counts.

    The main rollout's `n_steps * n_envs`, plus under forks every fork lane on
    every intervention window for `goal_horizon` steps. Frozen and
    failed-placement lanes count: they are stepped, only masked. Windows skipped
    by the interval run no fork physics and do not count.
    """
    worker = config.worker
    main = worker.n_steps * worker.n_envs
    if not config.interventions:
        return main
    n_windows = worker.n_steps // config.goal_horizon
    n_forked = int(applied_windows(n_windows, config.intervention_interval).sum())
    return main + n_forked * worker.n_envs * n_agents * config.goal_horizon


def append_columns(main, fork):
    """Append the fork columns to a main `Transition` along the env axis.

    Leaves without an env axis (the scalar `action_mask` placeholder, stacked to
    `(T,)`) are taken from the main batch.
    """
    return jax.tree.map(
        lambda a, b: a if a.ndim < 2 else jnp.concatenate([a, b], axis=1), main, fork
    )


def fork_diagnostics(intervention, recruit_distance, interval: int, horizon: int):
    """Scalar per-rollout fork statistics, merged into the losses dict.

    Placement success is reported per N as well as overall: the geometric filter
    can change the realized N distribution, and with it which focal goals get
    extra training.
    """
    w, m = intervention.worker, intervention.manager
    n_windows, n_lanes = intervention.valid.shape
    n_agents = w.active_mask.shape[-1]
    applied = jnp.asarray(applied_windows(n_windows, interval))[:, None]
    attempted = applied.sum() * n_lanes
    valid = intervention.valid
    valid_f = valid.astype(jnp.float32)
    n_valid = jnp.maximum(valid_f.sum(), 1.0)
    live_agent_steps = w.active_mask.sum()
    stats = {
        "intervention_attempted": attempted.astype(jnp.float32),
        "intervention_valid_frac": valid_f.sum() / attempted,
        "intervention_recruit_distance": (recruit_distance * valid_f).sum() / n_valid,
        "intervention_fork_length": w.active_mask[..., 0].sum() / n_valid,
        "intervention_window_return": (m.reward * valid_f).sum() / n_valid,
        "intervention_worker_progress": (
            (w.reward * w.active_mask).sum() / jnp.maximum(live_agent_steps, 1.0)
        ),
        "intervention_sim_steps": w.active_mask[..., 0].sum(),
        "intervention_stepped_lanes": (attempted * horizon).astype(jnp.float32),
    }
    for n in range(1, n_agents):
        tried = (intervention.n_recruits == n) & applied
        stats[f"intervention_attempted_n{n}"] = tried.sum().astype(jnp.float32)
        stats[f"intervention_valid_frac_n{n}"] = (tried & valid).sum() / jnp.maximum(
            tried.sum(), 1
        )
    return stats
