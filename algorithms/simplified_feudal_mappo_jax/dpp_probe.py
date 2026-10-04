"""Offline check for `manager_credit: dpp`: does the joint-goal critic see coalitions?

The D++ term (`counterfactual.dpp_credit`) is only as good as the advantage model
`Â_φ`'s sensitivity to teammates being sent toward an agent. Before any full
`_dpp` run, this probe measures that with critics that already exist: every
finished `*_cf` or `*_dpp` checkpoint carries a trained `manager_adv`.

For each trial it rolls out the trained manager and worker (stochastic, as in
training) for full episodes, and at every manager decision computes each agent's
D++ term with that trial's own critic. Decisions are split by whether the agent
is waiting alone: touching an undelivered box whose coupling is at least 2 and is
not met. That is the stepping-stone state D++ exists to reward.

Pass: the term is clearly larger when the agent waits alone, on every seed.
Fail: no difference, i.e. the term would be the critic's own noise.

Control: the same recruitment move mirrored AWAY from the agent
(`direction=-1`). A critic that penalizes any departure from the policy's own
goals scores both moves negative; one that understands coalitions prefers the
move toward a waiting agent. So the gap `toward - away` when waiting alone,
against the same gap elsewhere, separates "the critic sees coalitions" from
"the critic penalizes deviation".

Only the critic is queried; no simulator state is forked.

    uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_probe \\
        --batch mjx_2a_4o_1122_1024_gs \\
        --model simplified_feudal_tanh_relative_input_cf --trials 0,1,2,3,4
"""

import argparse
import math
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes

from algorithms.mappo_jax.run import make_env
from algorithms.simplified_feudal_mappo_jax import counterfactual as cf
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.run import (
    Simplified_Feudal_MAPPO_JAX_Runner as Runner,
)
from algorithms.simplified_feudal_mappo_jax.run import make_feudal_config
from algorithms.simplified_feudal_mappo_jax.trainer import (
    HierTrainState,
    create_hier_train_state,
    make_policy,
)
from algorithms.simplified_feudal_mappo_jax.types import Model_Params, Params

REPO_ROOT = Path(__file__).resolve().parents[2]


def compose(batch: str, model: str, trial: str):
    """The arm's config exactly as `train.py` composes it (a hand-copied yaml
    value would silently measure a different network)."""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from train import _build_dispatch_args

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = hydra_compose(
            config_name="config",
            overrides=[
                "algorithm=simplified_feudal_mappo_jax",
                f"env={batch}",
                f"model={model}",
                f"trial_id={trial}",
            ],
        )
    return _build_dispatch_args(cfg, {"env": batch, "model": model})


def load_arm(batch: str, model: str, trial: str, n_envs: int):
    """`(env, feudal_config, train_state)` with the trial's saved params,
    including the joint-goal critic."""
    args = compose(batch, model, trial)
    env_config = dict(args["env_config"], n_envs=n_envs)
    env = make_env(env_config)
    exp = args["exp_dict"]
    config = make_feudal_config(
        Params(**exp["params"]), Model_Params(**exp["model_params"]), n_envs
    )
    if config.manager_credit not in cf.GOAL_MODEL_CREDITS:
        raise ValueError(
            f"{model} trains no joint-goal critic (manager_credit="
            f"{config.manager_credit!r}); use a `_cf` or `_dpp` arm"
        )
    ts = create_hier_train_state(jax.random.PRNGKey(0), config, env)
    models = REPO_ROOT / "experiments" / "results" / batch / model / str(trial) / "models"
    path = models / "models_finished.msgpack"
    if not path.exists():
        path = models / "models_checkpoint.msgpack"
    params = from_bytes(Runner._params_tree(ts), path.read_bytes())
    ts = HierTrainState(
        worker=ts.worker._replace(
            actor_ts=ts.worker.actor_ts.replace(params=params["worker_actor"]),
            critic_ts=ts.worker.critic_ts.replace(params=params["worker_critic"]),
        ),
        manager=ts.manager._replace(
            actor_ts=ts.manager.actor_ts.replace(params=params["manager_actor"]),
            critic_ts=ts.manager.critic_ts.replace(params=params["manager_critic"]),
        ),
        manager_adv=ts.manager_adv.replace(params=params["manager_adv"]),
    )
    return env, config, ts, path


def make_rollout(env, config, n_envs: int, deterministic: bool):
    """Jitted full-episode rollout recording every manager decision: the critic
    input, every agent's offset and position, and whether each agent waits alone
    at an undelivered box that needs a partner."""
    policy = make_policy(config, env)
    horizon, radius = config.goal_horizon, config.waypoint_radius
    n_windows = math.ceil(env.max_steps / horizon)
    v_reset, v_step = jax.vmap(env.reset), jax.vmap(env.step)
    coupling = jnp.asarray(env._coupling)

    def waiting_alone(state):
        agent_pos = env._agent_pos(state.data)
        box_pos, box_yaw = env._box_pose(state.data)
        touch = env._touch_matrix(agent_pos, box_pos, box_yaw)  # (A, O)
        short = ~state.delivered & (coupling >= 2) & (touch.sum(0) < coupling)
        return (touch & short[None]).any(-1)  # (A,)

    v_alone = jax.vmap(waiting_alone)

    def freeze(done, new, old):
        return jax.tree.map(
            lambda n, o: jnp.where(done.reshape((-1,) + (1,) * (n.ndim - 1)), o, n),
            new,
            old,
        )

    @jax.jit
    def run(ts, key):
        key, reset_key = jax.random.split(key)
        obs, state = v_reset(jax.random.split(reset_key, n_envs))

        def step(waypoint, carry, k):
            obs, state, finished, key = carry
            key, act_key = jax.random.split(key)
            _, pos = policy.observe(obs, state)
            action, _, _ = policy.act(
                ts.worker, obs, pos, waypoint, k, act_key, deterministic
            )
            next_obs, next_state, _, terminated, truncated, _ = v_step(state, action)
            next_state = freeze(finished, next_state, state)
            next_obs = freeze(finished, next_obs, obs)
            return (next_obs, next_state, finished | terminated | truncated, key), None

        def window(carry, _):
            obs, state, finished, key = carry
            key, decide_key = jax.random.split(key)
            gs, pos = policy.observe(obs, state)
            waypoint = policy.decide(
                ts.manager, obs, gs, pos, state, decide_key, deterministic
            )[0]
            record = {
                "critic_in": wp.manager_critic_input(gs, pos),
                "offset": wp.goal_error(waypoint, pos, radius),
                "pos": pos,
                "alone": v_alone(state),
                "live": ~finished,
            }
            carry, _ = jax.lax.scan(
                partial(step, waypoint), (obs, state, finished, key),
                jnp.arange(horizon),
            )
            return carry, record

        init = (obs, state, jnp.zeros(n_envs, dtype=bool), key)
        _, records = jax.lax.scan(window, init, None, length=n_windows)
        return records  # leaves lead with (n_windows, n_envs)

    return run


def episode_bootstrap(values, groups, mask, n_boot=2000, seed=0):
    """Mean of `values[mask & groups] - values[mask & ~groups]` and a 95%
    interval from resampling episodes (axis 1), since decisions within one
    episode are not independent."""
    rng = np.random.default_rng(seed)
    n_envs = values.shape[1]

    def diff(idx):
        v, g, m = values[:, idx], groups[:, idx], mask[:, idx]
        a, b = v[m & g], v[m & ~g]
        return (a.mean() if a.size else np.nan) - (b.mean() if b.size else np.nan)

    point = diff(np.arange(n_envs))
    boots = np.array([diff(rng.integers(0, n_envs, n_envs)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, lo, hi


def probe_trial(batch, model, trial, n_envs, deterministic, seed):
    env, config, ts, path = load_arm(batch, model, trial, n_envs)
    records = make_rollout(env, config, n_envs, deterministic)(
        ts, jax.random.PRNGKey(seed)
    )
    def credit(direction):
        fn = jax.jit(
            partial(
                cf.dpp_credit,
                radius=config.waypoint_radius,
                max_recruits=config.dpp_max_recruits,
                direction=direction,
            )
        )
        return fn(
            ts.manager_adv, records["critic_in"], records["offset"], records["pos"]
        )[0]

    dpp = np.asarray(credit(1.0))  # (T, E, N)
    away = np.asarray(credit(-1.0))
    alone = np.asarray(records["alone"])
    live = np.broadcast_to(np.asarray(records["live"])[..., None], dpp.shape)
    clipped = np.maximum(dpp, 0.0)

    def mean(x, m):
        return float(x[m].mean()) if m.any() else float("nan")

    out = {
        "checkpoint": str(path),
        "n_decisions": int(live.sum()),
        "n_alone": int((live & alone).sum()),
        "raw_alone": mean(dpp, live & alone),
        "raw_other": mean(dpp, live & ~alone),
        "clipped_alone": mean(clipped, live & alone),
        "clipped_other": mean(clipped, live & ~alone),
        "positive_alone": mean((dpp > 0).astype(float), live & alone),
        "positive_other": mean((dpp > 0).astype(float), live & ~alone),
        "toward_minus_away_alone": mean(dpp - away, live & alone),
        "toward_minus_away_other": mean(dpp - away, live & ~alone),
    }
    # Episodes are axis 1; a decision is (window, env, agent), so fold the agent
    # axis into the window axis before resampling episodes.
    flat = lambda x: np.moveaxis(x, 2, 1).reshape(-1, x.shape[1])  # noqa: E731
    point, lo, hi = episode_bootstrap(flat(clipped), flat(alone), flat(live))
    out.update(clipped_gap=point, clipped_gap_lo=lo, clipped_gap_hi=hi)
    point, lo, hi = episode_bootstrap(flat(dpp - away), flat(alone), flat(live))
    out.update(control_gap=point, control_gap_lo=lo, control_gap_hi=hi)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", default="mjx_2a_4o_1122_1024_gs")
    parser.add_argument("--model", default="simplified_feudal_tanh_relative_input_cf")
    parser.add_argument("--trials", default="0,1,2,3,4")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default=None, help="optional pickle of the results")
    args = parser.parse_args()

    results = {}
    for trial in args.trials.split(","):
        r = probe_trial(
            args.batch, args.model, trial, args.n_envs, args.deterministic,
            args.seed + int(trial) if trial.isdigit() else args.seed,
        )
        results[trial] = r
        print(
            f"{args.batch}/{args.model}/{trial}: decisions {r['n_decisions']}, "
            f"waiting alone {r['n_alone']}\n"
            f"  clipped term   alone {r['clipped_alone']:.4f}  other "
            f"{r['clipped_other']:.4f}  gap {r['clipped_gap']:+.4f} "
            f"[{r['clipped_gap_lo']:+.4f}, {r['clipped_gap_hi']:+.4f}]\n"
            f"  positive share alone {r['positive_alone']:.3f}  other "
            f"{r['positive_other']:.3f}\n"
            f"  raw term       alone {r['raw_alone']:+.4f}  other "
            f"{r['raw_other']:+.4f}\n"
            f"  toward - away  alone {r['toward_minus_away_alone']:+.4f}  other "
            f"{r['toward_minus_away_other']:+.4f}  gap {r['control_gap']:+.4f} "
            f"[{r['control_gap_lo']:+.4f}, {r['control_gap_hi']:+.4f}]",
            flush=True,
        )
    passed = [r["clipped_gap_lo"] > 0 for r in results.values()]
    control = [r["control_gap_lo"] > 0 for r in results.values()]
    print(
        f"\nseeds whose 95% interval for (alone - other) is above 0: "
        f"{sum(passed)}/{len(passed)}; for the toward-minus-away gap: "
        f"{sum(control)}/{len(control)}"
    )
    if args.out:
        with open(args.out, "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    main()
