"""Wall-clock shootout: Box2D multi_box_push vs the MJX port.

Workload per engine: ``--episodes`` episodes of ``--steps`` steps (default
100 x 1024 = 102,400 env-steps) of uniform random actions, each engine using
its natural parallelism:

- Box2D: ``AsyncVectorEnv`` across all usable cores (fork context — the same
  setup the training collector uses via ``make_vec_env``), episodes spread
  over the workers, gymnasium autoreset on termination.
- MJX: one jit + vmap batch of ``--episodes`` envs on the default JAX device
  (GPU if available), one reset + ``--steps`` vmapped step calls. No
  auto-reset (terminated envs keep stepping — same per-step cost).

Both engines pay for full observation building every step (density sensors,
lidar raycasts, contact forces). The protocols are step-count-matched: the
metric is wall-clock time / env-steps-per-second for the same total number of
env-steps, not identical trajectories.

Run:
    uv run python -m environments.mjx_suite.profile_multi_box_push
"""

import argparse
import math
import os
import time

import numpy as np


def bench_box2d(n_agents, n_objects, total_steps, n_envs, seed=0):
    from gymnasium.vector import AsyncVectorEnv

    from algorithms.create_env import make_single_env
    from environments.types import EnvironmentEnum

    env_params = {"n_objects": n_objects, "reward_mode": "dense", "comm_radius": None}

    def make():
        return make_single_env(EnvironmentEnum.MULTI_BOX, n_agents, env_params)

    vec_steps = math.ceil(total_steps / n_envs)
    print(f"[box2d] {n_envs} async workers x {vec_steps} vector steps "
          f"= {vec_steps * n_envs:,} env-steps")

    vec = AsyncVectorEnv([make for _ in range(n_envs)], context="fork")
    rng = np.random.default_rng(seed)
    try:
        t0 = time.time()
        vec.reset(seed=seed)
        for i in range(vec_steps):
            actions = rng.uniform(-1, 1, (n_envs, n_agents, 2)).astype(np.float32)
            vec.step(actions)
            if (i + 1) % max(1, vec_steps // 10) == 0:
                print(f"[box2d] {i + 1}/{vec_steps} vector steps "
                      f"({time.time() - t0:.1f}s)", flush=True)
        wall = time.time() - t0
    finally:
        vec.close()
    return wall, vec_steps * n_envs


def bench_mjx(n_agents, n_objects, n_episodes, steps, seed=0):
    import jax
    import jax.numpy as jnp

    from environments.mjx_suite.multi_box_push_mjx import MultiBoxPushMJX

    print(f"[mjx] backend: {jax.default_backend()} ({jax.devices()[0]}), "
          f"batch {n_episodes} x {steps} steps = {n_episodes * steps:,} env-steps")

    env = MultiBoxPushMJX(n_agents=n_agents, n_objects=n_objects, max_steps=steps)
    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))

    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    actions = jax.random.uniform(
        jax.random.PRNGKey(seed + 1),
        (steps, n_episodes, n_agents, 2),
        minval=-1.0,
        maxval=1.0,
    )
    jax.block_until_ready(actions)

    # compile both paths on a throwaway state, timed separately
    t0 = time.time()
    obs, state = v_reset(keys)
    out = v_step(state, actions[0])
    jax.block_until_ready(out[0])
    compile_time = time.time() - t0

    t0 = time.time()
    obs, state = v_reset(keys)
    for i in range(steps):
        obs, state, reward, terminated, truncated, info = v_step(state, actions[i])
        if (i + 1) % max(1, steps // 10) == 0:
            jax.block_until_ready(obs)
            print(f"[mjx] {i + 1}/{steps} vmapped steps "
                  f"({time.time() - t0:.1f}s)", flush=True)
    jax.block_until_ready(obs)
    wall = time.time() - t0
    assert bool(jnp.isfinite(state.data.qpos).all()), "mjx state went non-finite"
    return wall, n_episodes * steps, compile_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--n-agents", type=int, default=30)
    parser.add_argument("--n-objects", type=int, default=6)
    parser.add_argument("--box2d-envs", type=int, default=0,
                        help="async workers (0 = all usable cores)")
    parser.add_argument("--skip-box2d", action="store_true")
    parser.add_argument("--skip-mjx", action="store_true")
    args = parser.parse_args()

    total_steps = args.episodes * args.steps
    n_workers = args.box2d_envs or len(os.sched_getaffinity(0))
    results = {}

    if not args.skip_mjx:
        wall, n, compile_time = bench_mjx(
            args.n_agents, args.n_objects, args.episodes, args.steps
        )
        results["mjx"] = (wall, n)
        print(f"[mjx] compile {compile_time:.1f}s (one-time, excluded)\n")

    if not args.skip_box2d:
        wall, n = bench_box2d(args.n_agents, args.n_objects, total_steps, n_workers)
        results["box2d"] = (wall, n)
        print()

    print(f"=== {args.n_agents} agents, {args.n_objects} objects, "
          f"{args.episodes} episodes x {args.steps} steps ===")
    for name, (wall, n) in results.items():
        print(f"{name:6s} {wall:8.1f}s wall   {n / wall:10,.0f} env-steps/s   "
              f"({n:,} env-steps)")
    if len(results) == 2:
        ratio = results["box2d"][0] / results["mjx"][0]
        faster = "mjx" if ratio > 1 else "box2d"
        print(f"--> {faster} is {max(ratio, 1 / ratio):.1f}x faster")


if __name__ == "__main__":
    main()
