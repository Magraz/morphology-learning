"""Training-time simulator forks for the simplified waypoint hierarchy.

Under `interventions: true`, at every `intervention_interval`-th manager
decision, `trainer.make_train` forks each env once, around one focal agent i
drawn uniformly per env and window, before the first worker step of the window:

* N ~ U{1 .. n_agents - 1} teammates of agent i, drawn uniformly without
  replacement, are teleported within `intervention_radius` world units of it
  (`env.teleport_agents`, which owns the geometry);
* each recruit gets agent i's realized waypoint offset `w_i - s_i`, measured
  from its own new position; agent i and every other agent keep their
  positions and waypoints;
* the fork runs one `goal_horizon` window with the same frozen policies and is
  then discarded. It is its own episode: its worker steps end with `done` on the
  window's last step, and its single manager transition is a terminal whose
  return is the fork's discounted team reward. No time-limit bootstrap (except
  under `intervention_redecide: all`, below).

Fork transitions join both levels' PPO batches as extra env columns
(`types.Intervention`). Only agent i's manager action is a policy sample in its
fork (the recruits' goals were imposed), so it alone carries manager actor
weight there; every live agent's worker action is a real sample. The manager
critic reads a context block (`critic_context`) so that it can tell a
continuing main return from a one-window fork return. Execution and evaluation
are unchanged. Plan: `plans/simplified_feudal_interventions_2026-10-05.md`.

`intervention_redecide` (`REDECIDE_MODES`) sets who decides the fork's goals.
In every mode the teleport is the same (same keys, and the placement filter uses
the main decision's offset), so fork start states match across modes at a seed.

* `none` (the default, the original fork): the fork's manager record is the main
  decision itself. Agent i's goal was chosen in the pre-teleport state, so its
  stored input shows no recruits: the same input as its main row, scored on a
  different outcome.
* `focal`: agent i samples a fresh goal from the post-teleport state. The
  recruits copy that new offset, clipped to the arena (the placement filter only
  guaranteed the old offset fits). Agent i's manager record and the critic input
  are the post-teleport ones, so the actor sees the help it is credited for.
  Every other agent keeps the main decision's waypoint.
* `all`: every agent samples its own goal from the post-teleport state; nothing
  is imposed, so every agent's manager row trains. The fork is then an ordinary
  on-policy window from a teleported start state, and it bootstraps like the
  main rollout: the time-limit bootstraps at both levels, plus the manager's
  continuing value at the window end (`gamma_M^c V^M(s_end)`, zero context).
  Its manager target is therefore the true return from the teleported state, not
  a one-window return.

The forks count TOWARD the per-update batch, not on top of it: the main rollout
runs `main_envs(...)` envs, fewer than the env group's `n_envs`, so that its
steps plus the fork steps equal the batch the parent collects
(`n_steps * n_envs`). Updates, gradient steps and simulator steps per update
then match the parent; the fork arm simply collects fewer fresh episodes.

`intervention_scheduler` (`intervention_bandit.SCHEDULERS`) sets who decides
which windows fork. `fixed` is everything above. `bandit` and `random` run the
adaptive collector (`adaptive_collection.py`): half of the batch's envs are
persistent main streams, a scheduler decides at each main decision whether to
fork it, and a skipped fork is replaced by an extra main window, so every update
still collects exactly `n_steps * batch_envs` simulator steps.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.simplified_feudal_mappo_jax import intervention_bandit as ib

# `fold_in` constant for every fork random draw, so enabling forks consumes no
# split of the main path's keys: the main rollout is unchanged by them.
FORK_KEY_NAMESPACE = 0x51F0

# `intervention_redecide`: who decides a fork's goals (see the module docstring).
REDECIDE_MODES = ("none", "focal", "all")


# Scheduler fields and the schedulers they act under. A field set away from its
# default under any other scheduler would silently do nothing, so it raises.
_SCHEDULER_FIELDS = {
    "intervention_bandit_hidden_dim": ("bandit",),
    "intervention_bandit_epsilon": ("bandit",),
    "intervention_bandit_lr": ("bandit",),
    # The target is also logged under `random`, as a diagnostic.
    "intervention_bandit_cost_coef": ("bandit", "random"),
    "intervention_bandit_zero_reward_penalty": ("bandit", "random"),
    "intervention_random_rate": ("random",),
}


def adaptive(config) -> bool:
    """True under the adaptive fork collector (`bandit` / `random` scheduler)."""
    return config.intervention_scheduler in ib.ADAPTIVE_SCHEDULERS


def adaptive_main_envs(batch_envs: int) -> int:
    """Persistent main streams of the adaptive collector: half the batch, so an
    update that forks every window still fits (one fork per main window)."""
    if int(batch_envs) != batch_envs or batch_envs < 2 or batch_envs % 2:
        raise ValueError(
            "an adaptive intervention_scheduler needs an even env-group n_envs >= 2 "
            f"(half of it are the main streams), got {batch_envs}"
        )
    return int(batch_envs) // 2


def _validate_scheduler(config) -> None:
    scheduler = config.intervention_scheduler
    if scheduler not in ib.SCHEDULERS:
        raise ValueError(
            f"unknown intervention_scheduler {scheduler!r}; expected one of "
            f"{ib.SCHEDULERS}"
        )
    defaults = {f.name: f.default for f in dataclasses.fields(type(config))}
    for name, schedulers in _SCHEDULER_FIELDS.items():
        if scheduler not in schedulers and getattr(config, name) != defaults[name]:
            raise ValueError(
                f"{name}={getattr(config, name)} has no effect under "
                f"intervention_scheduler={scheduler!r}"
            )
    if scheduler == "fixed":
        return
    if not config.interventions:
        raise ValueError(
            f"intervention_scheduler={scheduler!r} has no effect without "
            "interventions: true (it would silently train the parent arm)"
        )
    if config.intervention_interval != 1:
        raise ValueError(
            f"intervention_scheduler={scheduler!r} decides at every manager "
            f"decision; intervention_interval must be 1, got "
            f"{config.intervention_interval}"
        )
    batch = config.intervention_batch_envs
    if batch is None or adaptive_main_envs(batch) != config.worker.n_envs:
        raise ValueError(
            f"intervention_scheduler={scheduler!r} runs batch_envs / 2 main streams; "
            f"got intervention_batch_envs={batch} and worker.n_envs="
            f"{config.worker.n_envs} (build the config with make_feudal_config)"
        )
    if not 0.0 <= config.intervention_bandit_epsilon <= 1.0:
        raise ValueError(
            f"intervention_bandit_epsilon must be in [0, 1], got "
            f"{config.intervention_bandit_epsilon}"
        )
    if not 0.0 <= config.intervention_random_rate <= 1.0:
        raise ValueError(
            f"intervention_random_rate must be in [0, 1], got "
            f"{config.intervention_random_rate}"
        )
    if int(config.intervention_bandit_hidden_dim) < 1:
        raise ValueError(
            f"intervention_bandit_hidden_dim must be >= 1, got "
            f"{config.intervention_bandit_hidden_dim}"
        )
    if not config.intervention_bandit_lr > 0:
        raise ValueError(
            f"intervention_bandit_lr must be > 0, got {config.intervention_bandit_lr}"
        )
    for name in ("intervention_bandit_cost_coef", "intervention_bandit_zero_reward_penalty"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be >= 0, got {getattr(config, name)}")


def validate_interventions(config, env, n_windows: int) -> None:
    """Raise on a fork configuration that cannot run as specified."""
    _validate_scheduler(config)
    interval = config.intervention_interval
    redecide = config.intervention_redecide
    if redecide not in REDECIDE_MODES:
        raise ValueError(
            f"unknown intervention_redecide {redecide!r}; expected one of "
            f"{REDECIDE_MODES}"
        )
    if not config.interventions:
        if interval != 1:
            raise ValueError(
                f"intervention_interval={interval} has no effect without "
                "interventions: true (it would silently train the parent arm)"
            )
        if redecide != "none":
            raise ValueError(
                f"intervention_redecide={redecide!r} has no effect without "
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
    Under `intervention_redecide: none` the critic otherwise reads the
    pre-teleport state, because the placement filter depends on the credited
    (main) action. Under `focal` / `all` it reads the post-teleport state, which
    is fixed before the credited (re-decided) actions are sampled.
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


def sample_focal(key, n_envs: int, n_agents: int):
    """`(n_envs,)` the focal agent of each env's fork, uniform over agents. Fork
    lane e is env e's fork."""
    return jax.random.randint(key, (n_envs,), 0, n_agents)


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


def forked_windows(n_windows: int, interval: int) -> int:
    """Number of rollout windows that fork."""
    return int(applied_windows(n_windows, interval).sum())


def main_envs(n_envs: int, n_windows: int, interval: int) -> int:
    """Main-rollout envs under forks, so that main plus fork steps fill the
    parent's batch of `n_envs` envs.

    Each main env steps every one of the `n_windows` windows, and its fork lane
    steps every forked window, so E main envs cost `E * (n_windows + forked)`
    window-lengths against the parent's `n_envs * n_windows`. Rounded to the
    nearest whole env (exact at interval 1, where it is `n_envs / 2`).
    """
    if int(interval) != interval or interval < 1:
        raise ValueError(f"intervention_interval must be an integer >= 1, got {interval}")
    forked = forked_windows(n_windows, int(interval))
    return max(1, round(n_envs * n_windows / (n_windows + forked)))


def simulator_steps_per_update(config) -> int:
    """Env step calls one update makes, which is what `n_total_steps` caps and
    `total_steps` counts.

    The main rollout's `n_steps * n_envs`, plus under forks one fork lane per
    main env on every intervention window for `goal_horizon` steps. Frozen and
    failed-placement lanes count: they are stepped, only masked. Windows skipped
    by the interval run no fork physics and do not count. With the main env
    count from `main_envs`, this equals the parent's batch (to within one env's
    share when the interval does not divide evenly).
    """
    worker = config.worker
    if adaptive(config):
        # The adaptive collector fills exactly the batch's window slots, with
        # main windows or forks (`adaptive_collection.py`).
        return worker.n_steps * config.intervention_batch_envs
    main = worker.n_steps * worker.n_envs
    if not config.interventions:
        return main
    n_windows = worker.n_steps // config.goal_horizon
    n_forked = forked_windows(n_windows, config.intervention_interval)
    return main + n_forked * worker.n_envs * config.goal_horizon


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


def window_delivered(start_state, end_state):
    """`(B,)` True where at least one box was delivered between two env states.

    Reads the env state's latched `delivered` flags (`MultiBoxPushMJX`'s
    `EnvState`, the only env with the `teleport_agents` hook): a box delivered in
    the window is flagged at the end and not at the start. Exact, unlike a
    threshold on the window reward, whose dense shaping can push a delivery
    window below the +100 bonus.
    """
    return jnp.any(end_state.delivered & ~start_state.delivered, axis=-1)


def return_shares(prefix: str, team_reward, delivered, weight):
    """Shares of windows by outcome, over the rows where the 0/1 `weight` is set.

    `team_reward` is each window's summed team reward over its live steps, with
    no discount and no bootstrapped value (`Transition.team_reward`), so the
    shares mean the same under every `intervention_redecide` mode; the manager's
    `reward` adds a bootstrap under `all` and at a time limit. Keys:

    * `{prefix}_return_nonzero_frac`: team reward is not exactly 0. On dense
      groups that means some box moved (a box at rest pays exactly 0 shaping);
      on sparse groups it equals the delivery share.
    * `{prefix}_return_positive_frac`: net progress (team reward > 0).
    * `{prefix}_delivery_frac`: at least one box delivered (`window_delivered`).
    """
    w = weight.astype(jnp.float32)
    n = jnp.maximum(w.sum(), 1.0)

    def share(mask):
        return (mask.astype(jnp.float32) * w).sum() / n

    return {
        f"{prefix}_return_nonzero_frac": share(team_reward != 0),
        f"{prefix}_return_positive_frac": share(team_reward > 0),
        f"{prefix}_delivery_frac": share(delivered),
    }


def redecide_diagnostics(valid, per_fork: dict):
    """Per-rollout statistics of `intervention_redecide`: the mean of each
    `(n_windows, n_lanes)` per-fork series over valid forks. Keys by mode:

    * `intervention_redecide_mean_shift` (focal, all): how far the recruits moved
      the focal agent's mean goal, `||b(mu_post) - b(mu_pre)||` in units of R.
      Near 0 means the manager does not condition on the help, and the arm then
      trains almost the same signal as the default fork.
    * `intervention_redecide_recruit_clip_frac` (focal): share of recruits whose
      copy of the new offset left the arena and was clipped, i.e. got a goal
      that is not exactly the focal agent's.
    * `intervention_redecide_recruit_waypoint_gap` (all): mean distance from each
      recruit's own waypoint to the focal agent's, in units of R. About the
      placement distance when they pick the same offset, near 0 when they pick
      the same point, ~1 R and more when they choose independently.
    """
    valid_f = valid.astype(jnp.float32)
    n_valid = jnp.maximum(valid_f.sum(), 1.0)
    return {name: (x * valid_f).sum() / n_valid for name, x in per_fork.items()}
