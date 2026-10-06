"""Audit learned DPP recruitment gains against paired simulator continuations.

    uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_simulator_probe \\
        --batch mjx_2a_4o_1122_1024_gs \\
        --model simplified_feudal_tanh_relative_input_dpp --trials 0,1,2,3,4

Policies stay frozen. The intervention lasts one manager window; subsequent
goals come from the original manager. Each candidate and its factual reference
receive identical primitive-step/manager-window random keys. Physical returns
stop at termination or the original time limit, with no value bootstrap.
The training-style GAE target is recorded separately.
"""

import argparse
import dataclasses
import datetime
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.serialization import from_bytes, to_bytes

from algorithms.mappo_jax.mappo import compute_gae
from algorithms.mappo_jax.types import MAPPOConfig
from algorithms.simplified_feudal_mappo_jax import counterfactual as cf
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from algorithms.simplified_feudal_mappo_jax.trainer import (
    HierTrainState, _select, create_hier_train_state, make_policy,
)
from algorithms.simplified_feudal_mappo_jax.types import FeudalConfig

GROUPS = ("waiting_alone", "understaffed", "coalition", "other")
METRICS = ("discounted_return", "task_return", "deliveries", "coupled_box_steps",
           "focal_supported_steps", "box_progress", "live_steps", "episode_end_events")


def _batch(state):
    return jax.tree.map(lambda x: x[None], state)


def state_group(env, state):
    """Per-agent strata from live contacts, before the decision is executed."""
    boxes, yaw = env._box_pose(state.data)
    touch = env._touch_matrix(env._agent_pos(state.data), boxes, yaw)
    count = touch.sum(axis=0)
    live = ~state.delivered & (jnp.asarray(env._coupling) >= 2)
    alone = (touch & (live & (count == 1))[None]).any(axis=-1)
    short = (touch & (live & (count > 1) & (count < env._coupling))[None]).any(axis=-1)
    met = (touch & (live & (count >= env._coupling))[None]).any(axis=-1)
    return jnp.where(alone, 0, jnp.where(short, 1, jnp.where(met, 2, 3)))


def random_matched_offsets(pos, toward, key, radius):
    """Random feasible direction with exactly the toward offset's Euclidean norm.

    Choose among the eight rotations/reflections of the actual support step.
    At least the unchanged step is feasible. Near a corner it may be the only
    feasible one; that degeneracy is exposed in candidate displacement metrics.
    There is no post-sampling clip that could change the matched distance.
    """
    x, y = toward[..., 0], toward[..., 1]
    directions = jnp.stack([
        jnp.stack([x, y], -1), jnp.stack([-x, y], -1),
        jnp.stack([x, -y], -1), jnp.stack([-x, -y], -1),
        jnp.stack([y, x], -1), jnp.stack([-y, x], -1),
        jnp.stack([y, -x], -1), jnp.stack([-y, -x], -1),
    ], axis=-2)
    endpoints = pos[..., None, :] + radius * directions
    feasible = (jnp.abs(endpoints) <= wp.ARENA_HALF_EXTENT + 1e-7).all(axis=-1)
    scores = jnp.where(feasible, jax.random.uniform(key, feasible.shape), -1.0)
    index = scores.argmax(axis=-1)
    return jnp.take_along_axis(directions, index[..., None, None], axis=-2)[..., 0, :]


def make_candidates(pos, offsets, focal, radius, n_max, key):
    """Original joint plus toward/away/random joints for every recruit count."""
    joints, labels, counts = [offsets], ["original"], [0]
    toward = cf.support_offsets(pos, focal, radius)
    random_offsets = random_matched_offsets(pos, toward, key, radius)
    for n in range(1, n_max + 1):
        mask = cf.recruit_mask(pos, focal, n)[..., None]
        for label, joint in (
            ("toward", cf.dpp_joint(offsets, pos, focal, n, radius)),
            ("away", cf.dpp_joint(offsets, pos, focal, n, radius, direction=-1.0)),
            ("random", jnp.where(mask, random_offsets, offsets)),
        ):
            joints.append(joint)
            labels.append(label)
            counts.append(n)
    return jnp.stack(joints), labels, np.asarray(counts, dtype=int)


def make_evaluator(env, config, eval_steps, policy=None, waypoint_intervention=None,
                   extra_metrics=None):
    """Jitted vmapped paired continuation, preserving the complete source state.

    Random streams are keyed per source state and Monte Carlo replicate, so
    changing chunk size preserves random draws. Numerical trajectories can
    vary with vector width/backend. Keys follow a fixed schedule irrespective
    of early termination or intervention.
    """
    policy = make_policy(config, env) if policy is None else policy
    horizon = config.goal_horizon
    gamma = config.worker.gamma if config.manager_step_gamma is None else config.manager_step_gamma
    n_windows = math.ceil(eval_steps / horizon)
    stops = [min(horizon, eval_steps)]
    labels = ["window"]
    if eval_steps >= 4 * horizon:
        stops.append(4 * horizon)
        labels.append("four_windows")
    if stops[-1] != eval_steps:
        stops.append(eval_steps)
        labels.append("episode" if eval_steps >= env.max_steps else "cutoff")
    else:
        labels[-1] += "_episode" if eval_steps >= env.max_steps else "_cutoff"

    def value(ts, obs, state):
        gs, pos = policy.observe(obs[None], _batch(state))
        return ts.manager.critic_ts.apply_fn(
            ts.manager.critic_ts.params, wp.manager_critic_input(gs, pos))[0]

    def one(ts, state, obs, initial_waypoint, focal, key, context):
        initially_live = (state.t < env.max_steps) & ~state.delivered.all()

        def window(carry, w):
            state, obs, alive, _ = carry
            start_value = jnp.where(alive, value(ts, obs, state), 0.0)

            def decide(_):
                gs, pos = policy.observe(obs[None], _batch(state))
                manager_key = jax.random.fold_in(key, n_windows * horizon + w)
                return policy.decide(ts.manager, obs[None], gs, pos, _batch(state), manager_key)[0][0]

            waypoint = jax.lax.cond(w == 0, lambda _: initial_waypoint, decide, operand=None)
            if waypoint_intervention is not None:
                waypoint = waypoint_intervention(state, waypoint, focal, w, context)

            def step(carry, k):
                state, obs, alive, window_reward = carry
                t = w * horizon + k
                live = alive & (t < eval_steps)
                _, pos = policy.observe(obs[None], _batch(state))
                worker_key = jax.random.fold_in(key, t)
                action = policy.act(ts.worker, obs[None], pos, waypoint[None], k, worker_key)[0][0]
                next_obs, next_state, _, terminated, truncated, info = env.step(state, action)
                reward = jnp.where(live, info["task_reward"], 0.0)
                timeout = live & truncated & ~terminated
                window_reward += (gamma ** k) * reward
                window_reward += jnp.where(timeout, gamma ** (k + 1) * value(ts, next_obs, next_state), 0.0)
                boxes, yaw = env._box_pose(next_state.data)
                touch = env._touch_matrix(env._agent_pos(next_state.data), boxes, yaw)
                met = (touch.sum(0) >= env._coupling) & (env._coupling >= 2) & ~state.delivered
                metrics = jnp.array([
                    gamma ** t * reward, reward,
                    jnp.where(live, (next_state.delivered & ~state.delivered).sum(), 0),
                    jnp.where(live, met.sum(), 0),
                    jnp.where(live, (met & touch[focal]).any(), False),
                    jnp.where(live, ((state.prev_box_goal_dist - next_state.prev_box_goal_dist) * ~state.delivered).sum(), 0.0),
                    live.astype(jnp.float32),
                    (live & (terminated | truncated)).astype(jnp.float32),
                ], dtype=jnp.float32)
                extra = (extra_metrics(state, next_state, waypoint, focal, live, t, context)
                         if extra_metrics is not None else jnp.zeros((0,), jnp.float32))
                state = jax.tree.map(lambda new, old: jnp.where(live, new, old), next_state, state)
                obs = jnp.where(live, next_obs, obs)
                alive = alive & ~(live & (terminated | truncated))
                return (state, obs, alive, window_reward), (metrics, extra)

            (state, obs, alive, reward), (metrics, extra) = jax.lax.scan(
                step, (state, obs, alive, jnp.array(0.0)), jnp.arange(horizon))
            # No episode resets: all tails are masked and finite task return stops.
            return (state, obs, alive, reward), (metrics, extra, reward, start_value, ~alive)

        (state, obs, alive, _), (metrics, extra, rewards, values, done) = jax.lax.scan(
            window, (state, obs, initially_live, jnp.array(0.0)), jnp.arange(n_windows))
        last_value = jnp.where(alive, value(ts, obs, state), 0.0)
        gae, _ = compute_gae(rewards, values, done.astype(jnp.float32), last_value,
                             config.manager.gamma, config.manager.gae_lambda)
        cumulative = jnp.cumsum(metrics.reshape(-1, len(METRICS)), axis=0)
        result = {"metrics": cumulative[jnp.asarray(stops) - 1],
                  "training_gae": gae[0], "episode_complete": ~alive}
        if extra_metrics is not None:
            extra = extra.reshape(n_windows * horizon, -1)
            result["extra_metrics"] = jnp.cumsum(extra, axis=0)[jnp.asarray(stops) - 1]
            times = jnp.arange(n_windows * horizon)[:, None]
            first = jnp.stack([jnp.min(jnp.where((extra > 0) & (times < stop),
                                               times + 1, n_windows * horizon + 1), axis=0)
                               for stop in stops])
            result["extra_first_steps"] = jnp.where(first <= jnp.asarray(stops)[:, None], first, -1)
        return result

    compiled = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0, 0, 0)))

    def evaluate(ts, state, obs, initial_waypoint, focal, key, context=None):
        context = jnp.zeros(focal.shape, jnp.int32) if context is None else context
        return compiled(ts, state, obs, initial_waypoint, focal, key, context)

    return evaluate, labels, stops


class StratifiedReservoir:
    """Uniform priority reservoir per stratum; fill missing quotas from others."""

    def __init__(self, size, seed):
        self.size, self.rng = size, np.random.default_rng(seed)
        self.heaps = {g: [] for g in GROUPS}
        self.seen = {g: 0 for g in GROUPS}
        self.serial = 0

    def offer(self, group, factory):
        self.seen[group] += 1
        self.serial += 1
        priority = float(self.rng.random())
        heap = self.heaps[group]
        if len(heap) < self.size or priority < -heap[0][0]:
            item = (-priority, self.serial, factory())
            if len(heap) < self.size:
                heapq.heappush(heap, item)
            else:
                heapq.heapreplace(heap, item)

    def selected(self):
        selected, remaining = [], []
        for index, group in enumerate(GROUPS):
            quota = self.size // len(GROUPS) + (index < self.size % len(GROUPS))
            pool = sorted(self.heaps[group], key=lambda x: -x[0])
            selected.extend(pool[:quota])
            remaining.extend(pool[quota:])
        selected.extend(sorted(remaining, key=lambda x: -x[0])[:self.size - len(selected)])
        return [x[2] for x in sorted(selected, key=lambda x: x[1])]


def collect_sources(env, config, ts, n_envs, source_steps, size, seed):
    policy, reservoir = make_policy(config, env), StratifiedReservoir(size, seed)
    reset = jax.jit(jax.vmap(env.reset))

    @jax.jit
    def decision(obs, state, key):
        gs, pos = policy.observe(obs, state)
        waypoint = policy.decide(ts.manager, obs, gs, pos, state, key)[0]
        return waypoint, jax.vmap(lambda s: state_group(env, s))(state)

    @jax.jit
    def advance(obs, state, finished, waypoint, key):
        def step(carry, k):
            obs, state, finished = carry
            _, pos = policy.observe(obs, state)
            action = policy.act(ts.worker, obs, pos, waypoint, k, jax.random.fold_in(key, k))[0]
            no, ns, _, term, trunc, _ = jax.vmap(env.step)(state, action)
            return (_select(~finished, no, obs), _select(~finished, ns, state),
                    finished | term | trunc), None
        return jax.lax.scan(step, (obs, state, finished), jnp.arange(config.goal_horizon))[0]

    key = jax.random.PRNGKey(seed)
    obs, state = reset(jax.random.split(jax.random.fold_in(key, 0), n_envs))
    finished = jnp.zeros(n_envs, bool)
    for window in range(math.ceil(source_steps / config.goal_horizon)):
        if bool(finished.all()):
            break
        waypoint, group = decision(obs, state, jax.random.fold_in(key, 1 + 2 * window))
        host_group, host_finished = np.asarray(group), np.asarray(finished)
        for episode in range(n_envs):
            if host_finished[episode]:
                continue
            cache = {}

            def source(focal, label):
                if not cache:
                    cache.update(state=jax.device_get(jax.tree.map(lambda x: x[episode], state)),
                                 obs=np.asarray(obs[episode]), waypoint=np.asarray(waypoint[episode]))
                return {**cache, "focal": focal, "group": label,
                        "source_episode": episode, "source_window": window}

            for focal in range(env.n_agents):
                label = GROUPS[host_group[episode, focal]]
                reservoir.offer(label, lambda i=focal, g=label: source(i, g))
        obs, state, finished = advance(obs, state, finished, waypoint,
                                        jax.random.fold_in(key, 2 + 2 * window))
    sources = reservoir.selected()
    if not sources:
        raise ValueError("No live source decisions were collected")
    return sources, {"available": reservoir.seen,
                     "sampled": {g: sum(x["group"] == g for x in sources) for g in GROUPS}}


def state_schema(state, batched=False):
    """Stable dynamic-state schema; MJX tree repr contains object addresses."""
    paths, _ = jax.tree_util.tree_flatten_with_path(state)
    return [{"path": jax.tree_util.keystr(path), "shape": list(x.shape[1:] if batched else x.shape),
             "dtype": str(x.dtype)} for path, x in paths]


def save_sources(path, sources, ts, metadata):
    from algorithms.simplified_feudal_mappo_jax.run import Simplified_Feudal_MAPPO_JAX_Runner as Runner

    state = jax.tree.map(lambda *xs: np.stack(xs), *(s["state"] for s in sources))
    leaves, _ = jax.tree.flatten(state)
    metadata = {**metadata, "state_schema": state_schema(state, batched=True)}
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)),
        params=np.frombuffer(to_bytes(Runner._params_tree(ts)), dtype=np.uint8),
        **{f"state_{i}": np.asarray(x) for i, x in enumerate(leaves)},
        obs=np.stack([s["obs"] for s in sources]),
        waypoint=np.stack([s["waypoint"] for s in sources]),
        **{k: np.asarray([s[k] for s in sources]) for k in
           ("focal", "group", "source_episode", "source_window")})


def load_sources(path):
    from algorithms.mappo_jax.run import make_env
    from algorithms.simplified_feudal_mappo_jax.run import Simplified_Feudal_MAPPO_JAX_Runner as Runner

    with np.load(path, allow_pickle=False) as file:
        meta = json.loads(str(file["metadata"]))
        env = make_env(meta["env_config"])
        config_dict = dict(meta["feudal_config"])
        config_dict["worker"] = MAPPOConfig(**config_dict["worker"])
        config_dict["manager"] = MAPPOConfig(**config_dict["manager"])
        config = FeudalConfig(**config_dict)
        ts = create_hier_train_state(jax.random.PRNGKey(0), config, env)
        params = from_bytes(Runner._params_tree(ts), file["params"].tobytes())
        ts = HierTrainState(
            worker=ts.worker._replace(actor_ts=ts.worker.actor_ts.replace(params=params["worker_actor"]),
                                     critic_ts=ts.worker.critic_ts.replace(params=params["worker_critic"])),
            manager=ts.manager._replace(actor_ts=ts.manager.actor_ts.replace(params=params["manager_actor"]),
                                       critic_ts=ts.manager.critic_ts.replace(params=params["manager_critic"])),
            manager_adv=ts.manager_adv.replace(params=params["manager_adv"]))
        _, template = jax.jit(env.reset)(jax.random.PRNGKey(0))
        _, structure = jax.tree.flatten(template)
        if "state_schema" in meta and state_schema(template) != meta["state_schema"]:
            raise ValueError("Simulator state schema differs from the saved sources")
        leaves = [jnp.asarray(file[f"state_{i}"]) for i in range(structure.num_leaves)]
        template_leaves = jax.tree.leaves(template)
        for i, (saved, expected) in enumerate(zip(leaves, template_leaves)):
            if saved.shape[1:] != expected.shape or saved.dtype != expected.dtype:
                raise ValueError(f"Saved simulator state leaf {i} has an incompatible shape or dtype")
        states = jax.tree.unflatten(structure, leaves)
        sources = [{"state": jax.tree.map(lambda x: x[i], states),
                    **{k: file[k][i].copy() for k in ("obs", "waypoint", "focal", "group", "source_episode", "source_window")}}
                   for i in range(len(file["focal"]))]
    return env, config, ts, sources, meta


def _interval(values, rng, n_boot=1000):
    """Paired MC percentile bootstrap. Zero outcomes remain ties, not success."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None, None
    indices = rng.integers(0, values.size, (n_boot, values.size))
    return tuple(float(x) for x in np.quantile(values[indices].mean(axis=1), [.025, .975]))


def evaluate_sources(env, config, ts, sources, meta, out, mc_samples, chunk_size, eval_steps):
    evaluate, horizon_labels, stops = make_evaluator(env, config, eval_steps)
    policy = make_policy(config, env)
    n_max = min(config.dpp_max_recruits or env.n_agents - 1, env.n_agents - 1)
    if n_max < 1:
        raise ValueError("Recruitment audit requires at least two agents")
    rows = []
    rng = np.random.default_rng(meta["seed"])
    for start in range(0, len(sources), chunk_size):
        selected = sources[start:start + chunk_size]
        states = jax.tree.map(lambda *xs: jnp.stack(xs), *(s["state"] for s in selected))
        obs = jnp.asarray(np.stack([s["obs"] for s in selected]))
        pos = jax.vmap(env.goal_state)(states)
        offsets = wp.goal_error(jnp.asarray(np.stack([s["waypoint"] for s in selected])), pos, config.waypoint_radius)
        joints, predictions, candidate_metrics = [], [], []
        for local, source in enumerate(selected):
            sid = start + local
            key = jax.random.fold_in(jax.random.PRNGKey(meta["seed"]), 100000 + sid)
            joint, labels, counts = make_candidates(pos[local], offsets[local], int(source["focal"]),
                                                     config.waypoint_radius, n_max, key)
            gs, _ = policy.observe(obs[local:local+1], _batch(source["state"]))
            critic = wp.manager_critic_input(gs, pos[local:local+1])[0]
            inputs = cf.adv_model_input(jnp.broadcast_to(critic, (len(joint), critic.shape[-1])), joint)
            pred = ts.manager_adv.apply_fn(ts.manager_adv.params, inputs)
            predictions.append(np.asarray(pred - pred[0]))
            joints.append(joint)
            movement = np.linalg.norm(np.asarray(joint), axis=-1).sum(axis=-1)
            change = np.linalg.norm(np.asarray(joint - offsets[local]), axis=-1).sum(axis=-1)
            candidate_metrics.append((movement, change))
        if not np.isfinite(np.stack(predictions)).all():
            raise ValueError(f"Nonfinite critic predictions in source chunk {start}")
        joints = jnp.stack(joints)  # state, candidate, agent, xy
        keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(meta["seed"]), 200000 + (start + i) * mc_samples + r)
                          for i in range(len(selected)) for r in range(mc_samples)])
        repeated_states = jax.tree.map(lambda x: jnp.repeat(x, mc_samples, axis=0), states)
        repeated_obs = jnp.repeat(obs, mc_samples, axis=0)
        focals = jnp.repeat(jnp.asarray([s["focal"] for s in selected]), mc_samples)
        outcomes = []
        print(f"  Auditing states {start + 1}–{start + len(selected)} / {len(sources)}...", flush=True)
        for c in range(joints.shape[1]):
            first_goals = pos + config.waypoint_radius * joints[:, c]
            # Keep factual/unselected slots bit-identical to their original goals.
            actual_goals = jnp.asarray(np.stack([s["waypoint"] for s in selected]))
            if c == 0:
                first_goals = actual_goals
            else:
                masks = jnp.stack([cf.recruit_mask(pos[i], int(s["focal"]), int(counts[c]))
                                    for i, s in enumerate(selected)])
                first_goals = jnp.where(masks[..., None], first_goals, actual_goals)
            result = evaluate(ts, repeated_states, repeated_obs,
                              jnp.repeat(first_goals, mc_samples, axis=0), focals, keys)
            outcomes.append(jax.device_get(result))
        metrics = np.stack([x["metrics"].reshape(len(selected), mc_samples, len(stops), len(METRICS)) for x in outcomes], axis=1)
        gae = np.stack([x["training_gae"].reshape(len(selected), mc_samples) for x in outcomes], axis=1)
        complete = np.stack([x["episode_complete"].reshape(len(selected), mc_samples) for x in outcomes], axis=1)
        if not np.isfinite(metrics).all() or not np.isfinite(gae).all():
            raise ValueError(f"Nonfinite simulator outcomes or GAE targets in source chunk {start}")
        np.savez_compressed(out / f"trial_{meta['trial_index']}_chunk_{start}.npz",
            metrics=metrics, training_gae=gae, episode_complete=complete,
            predicted_delta=np.stack(predictions), labels=np.asarray(labels), counts=counts,
            horizon_labels=np.asarray(horizon_labels), horizon_steps=np.asarray(stops),
            source_ids=np.arange(start, start + len(selected)), continuation_keys=np.asarray(keys))
        for i, source in enumerate(selected):
            toward_indices = [c for c, label in enumerate(labels) if label == "toward"]
            selected_index = max(toward_indices, key=lambda c: predictions[i][c] / counts[c])
            for c in range(1, len(labels)):
                n = int(counts[c])
                gae_gain = (gae[i, c] - gae[i, 0]) / n
                for h, label in enumerate(horizon_labels):
                    actual = (metrics[i, c, :, h, 0] - metrics[i, 0, :, h, 0]) / n
                    lo, hi = _interval(actual, rng)
                    row = {
                        "trial": str(meta["trial"]), "source_id": start + i,
                        "source_episode": int(source["source_episode"]),
                        "source_window": int(source["source_window"]),
                        "source_time": int(source["state"].t), "focal": int(source["focal"]),
                        "group": str(source["group"]), "intervention": labels[c], "recruits": n,
                        "horizon": label, "horizon_steps": stops[h], "mc_samples": mc_samples,
                        "predicted_gain": float(predictions[i][c] / n),
                        "predicted_raw_delta": float(predictions[i][c]),
                        "selected_toward": c == selected_index,
                        "dpp_applied": bool(predictions[i][selected_index] > 0),
                        "actual_gain": float(actual.mean()),
                        "actual_gain_ci_lo": lo, "actual_gain_ci_hi": hi,
                        "actual_gain_mc_se": float(actual.std(ddof=1) / np.sqrt(mc_samples)) if mc_samples > 1 else None,
                        "paired_gain_nonzero_frac": float((actual != 0).mean()),
                        "training_gae_gain": float(gae_gain.mean()),
                        "branch_episode_complete_frac": float((metrics[i, c, :, h, METRICS.index("episode_end_events")] > 0).mean()),
                        "baseline_episode_complete_frac": float((metrics[i, 0, :, h, METRICS.index("episode_end_events")] > 0).mean()),
                        "waypoint_travel_norm_sum": float(candidate_metrics[i][0][c]),
                        "waypoint_change_norm_sum": float(candidate_metrics[i][1][c]),
                    }
                    for m, name in enumerate(METRICS):
                        row[f"baseline_{name}"] = float(metrics[i, 0, :, h, m].mean())
                        row[f"branch_{name}"] = float(metrics[i, c, :, h, m].mean())
                    rows.append(row)
    return rows


def _correlation(a, b, rank=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    if rank:
        a, b = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(a, b)[0, 1])


def summarize(rows, tolerance, seed):
    """Summaries per training seed; episode-clustered intervals on actual gains."""
    data = pd.DataFrame(rows)
    output = []
    rng = np.random.default_rng(seed)
    groups = [("all_stratified", data)] + [(str(g), f) for g, f in data.groupby("group")]
    for group, frame in groups:
        for (trial, intervention, horizon), f in frame.groupby(["trial", "intervention", "horizon"]):
            subsets = [("all_counts", f)] + [(str(n), x) for n, x in f.groupby("recruits")]
            if intervention == "toward":
                subsets.append(("dpp_selected", f[f.selected_toward & f.dpp_applied]))
            for count_label, f in subsets:
                if f.empty:
                    continue
                predicted, actual = f.predicted_gain.values, f.actual_gain.values
                # Production DPP applies every positive prediction, including tiny ones.
                positive = predicted > 0.0
                decisive = np.abs(actual) > tolerance
                clusters = [x.actual_gain.values for _, x in f.groupby("source_episode")]
                if len(clusters) > 1:
                    sums = np.array([x.sum() for x in clusters])
                    sizes = np.array([len(x) for x in clusters])
                    ids = rng.integers(0, len(clusters), (1000, len(clusters)))
                    lo, hi = np.quantile(sums[ids].sum(1) / sizes[ids].sum(1), [.025, .975])
                else:
                    lo = hi = None
                output.append({
                    "trial": str(trial), "group": group, "intervention": intervention,
                    "recruits": count_label, "horizon": horizon,
                    "n_candidates": len(f), "n_sources": f.source_id.nunique(),
                    "n_source_episodes": len(clusters),
                    "predicted_gain_mean": float(predicted.mean()),
                    "actual_gain_mean": float(actual.mean()),
                    "actual_gain_cluster_ci_lo": float(lo) if lo is not None else None,
                    "actual_gain_cluster_ci_hi": float(hi) if hi is not None else None,
                    "training_gae_gain_mean": float(f.training_gae_gain.mean()),
                    "predicted_positive_frac": float(positive.mean()),
                    "actual_positive_frac": float((actual > tolerance).mean()),
                    "actual_negative_frac": float((actual < -tolerance).mean()),
                    "actual_tied_frac": float((np.abs(actual) <= tolerance).mean()),
                    "positive_predictions_harmful_frac": float((actual[positive] < -tolerance).mean()) if positive.any() else None,
                    "positive_predictions_tied_frac": float((np.abs(actual[positive]) <= tolerance).mean()) if positive.any() else None,
                    "sign_accuracy_decisive": float((np.sign(predicted[decisive]) == np.sign(actual[decisive])).mean()) if decisive.any() else None,
                    "pearson": _correlation(predicted, actual),
                    "spearman": _correlation(predicted, actual, rank=True),
                })
    # Within-state rankings only; source-state value differences cannot inflate accuracy.
    rankings = []
    for candidate_set, pool in (("toward_counts", data[data.intervention == "toward"]),
                                ("all_interventions", data)):
        for (trial, sid, horizon), f in pool.groupby(["trial", "source_id", "horizon"]):
            actual, pred = f.actual_gain.values, f.predicted_gain.values
            correct, decisive = 0, 0
            for i, j in itertools.combinations(range(len(f)), 2):
                if abs(actual[i] - actual[j]) > tolerance:
                    decisive += 1
                    correct += (pred[i] - pred[j]) * (actual[i] - actual[j]) > 0
            index = int(np.argmax(pred))
            chosen = float(actual[index]) if pred[index] > 0 else 0.0
            rankings.append({"trial": str(trial), "source_id": int(sid), "group": str(f.group.iloc[0]),
                             "source_episode": int(f.source_episode.iloc[0]), "horizon": horizon,
                             "candidate_set": candidate_set,
                             "selected_intervention": str(f.intervention.iloc[index]) if pred[index] > 0 else "original",
                             "selected_recruits": int(f.recruits.iloc[index]) if pred[index] > 0 else 0,
                             "decisive_count_pairs": decisive,
                             "correct_count_pairs": int(correct),
                             "selected_actual_gain": chosen,
                             "mc_estimated_selection_regret": max(0.0, float(actual.max())) - chosen})
    return output, rankings


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_reports(out, rows, metadata, tolerance, seed):
    summary, ranking = summarize(rows, tolerance, seed)
    pd.DataFrame(rows).to_csv(out / "candidates.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(ranking).to_csv(out / "rankings.csv", index=False)
    report = {"snapshots": metadata, "gain_tolerance": tolerance,
        "target": "Discounted environmental team return until original termination/time limit. No simulator reset or return bootstrap.",
        "intervention": "One manager window. Subsequent goals from the frozen original manager.",
        "pairing": "Identical state, focal goal, untouched teammate goals, and per-step/per-window random streams for each candidate and reference.",
        "prediction": "Raw learned advantage-model difference, divided by recruit count. Scores are not clipped for evaluation.",
        "training_target": "A separate GAE comparison uses the training discounts, lambda and timeout value bootstrap; it is not physical task return.",
        "sampling": "Stratified source decisions. Missing strata are filled from observed strata. all_stratified is not an on-policy population estimate.",
        "uncertainty": "Candidate intervals resample paired MC continuations; summary intervals cluster by source episode within each training seed. All-zero outcomes are ties, not evidence of accurate credit.",
        "selection_regret": "Estimated using the same finite MC samples; selecting the empirical best can inflate apparent regret.",
        "summary": summary}
    (out / "report.json").write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", default="mjx_2a_4o_1122_1024_gs")
    parser.add_argument("--model", default="simplified_feudal_tanh_relative_input_dpp")
    parser.add_argument("--trials", default="0,1,2,3,4")
    parser.add_argument("--n-envs", type=int, default=32, help="Source episodes collected in parallel")
    parser.add_argument("--states", type=int, default=64, help="Source (decision, focal agent) samples per trial")
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4, help="Simulation width is chunk-size times mc-samples")
    parser.add_argument("--source-steps", type=int, help="Limit source collection; default is original episode length")
    parser.add_argument("--eval-steps", type=int, help="Limit continuation; default is original episode length")
    parser.add_argument("--gain-tolerance", type=float, default=1e-4, help="Per-recruit realized return units; smaller gains are ties")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--snapshot", type=Path, help="Replay complete saved source states and checkpoint parameters")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    for name in ("n_envs", "states", "mc_samples", "chunk_size", "source_steps", "eval_steps"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.gain_tolerance) or args.gain_tolerance < 0:
        parser.error("gain-tolerance must be finite and nonnegative")
    trials = [x.strip() for x in args.trials.split(",") if x.strip()]
    if not trials:
        parser.error("trials must contain at least one trial")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out or Path("plotting/feudal_goal_analysis") / f"dpp_simulator_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    all_rows, metadata = [], []
    for trial_index, trial in enumerate([None] if args.snapshot else trials):
        if args.snapshot:
            print(f"Loading saved sources {args.snapshot}...", flush=True)
            env, config, ts, sources, meta = load_sources(args.snapshot)
            meta = {**meta, "trial_index": 0, "replayed_from": str(args.snapshot.resolve())}
        else:
            from algorithms.simplified_feudal_mappo_jax.dpp_probe import compose, load_arm

            print(f"Loading {args.batch}/{args.model}/{trial}...", flush=True)
            env, config, ts, path = load_arm(args.batch, args.model, trial, args.n_envs)
            if env.n_agents < 2:
                parser.error("Recruitment audit requires at least two agents")
            seed = args.seed + (int(trial) if trial.isdigit() else trial_index)
            source_steps = min(args.source_steps or env.max_steps, env.max_steps)
            print(f"Collecting source states ({args.n_envs} episodes, {source_steps} steps)...", flush=True)
            sources, coverage = collect_sources(env, config, ts, args.n_envs, source_steps, args.states, seed)
            meta = {"batch": args.batch, "model": args.model, "trial": trial,
                "trial_index": trial_index, "seed": seed,
                "checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "env_config": dict(compose(args.batch, args.model, trial)["env_config"], n_envs=args.n_envs),
                "feudal_config": dataclasses.asdict(config), "coverage": coverage,
                "source_steps": source_steps, "source_episodes": args.n_envs,
                "jax_version": jax.__version__, "collection_backend": jax.default_backend()}
        eval_steps = min(args.eval_steps or env.max_steps, env.max_steps)
        if eval_steps < env.max_steps and eval_steps % config.goal_horizon:
            parser.error("An artificial eval-steps cutoff must contain whole manager windows")
        meta = {**meta, "eval_steps": eval_steps, "mc_samples": args.mc_samples,
                "chunk_size": args.chunk_size, "evaluation_backend": jax.default_backend()}
        print(f"Source coverage: {meta['coverage']}", flush=True)
        save_sources(out / f"trial_{trial_index}_sources.npz", sources, ts, meta)
        rows = evaluate_sources(env, config, ts, sources, meta, out,
                                args.mc_samples, args.chunk_size, eval_steps)
        all_rows.extend(rows)
        metadata.append(meta)
        write_reports(out, all_rows, metadata, args.gain_tolerance, args.seed)
        final = pd.DataFrame(rows)
        selected = final[final.selected_toward & final.dpp_applied]
        if not selected.empty:
            last_horizon = selected.horizon.iloc[-1]
            selected = selected[selected.horizon == last_horizon]
            print(f"  DPP selected {len(selected)} positive predictions: "
                  f"realized better {(selected.actual_gain > args.gain_tolerance).sum()}, "
                  f"worse {(selected.actual_gain < -args.gain_tolerance).sum()}, "
                  f"tied {(selected.actual_gain.abs() <= args.gain_tolerance).sum()} ({last_horizon}).", flush=True)
    print(f"Reports and complete source snapshots: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
