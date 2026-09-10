"""Is the manager's latent row `s[i]` actually ABOUT agent i?

The worker's intrinsic reward is

    r^I_t[i] = 1/c * sum_k d_cos( s_t[i] - s_{t-k}[i] , g_{t-k}[i] )

so whatever `s[i]` responds to is what agent i gets paid for. Under
``manager_latent="centralized"`` (the original), `s` is one
``Dense(N*goal_dim)`` over a 2-layer MLP of the FULL joint state, reshaped to
``(N, goal_dim)`` — i.e. ``s_i = W_i z + b_i``. The agent axis is a SLICE INDEX,
not a factorization, and nothing forces row i to be about agent i.

If it is not, `r^I` is per-agent in INDEXING but not in CAUSATION: agent i
collects reward for motion its teammates caused, which is exactly the credit-
assignment confound a worker-level reward exists to remove. Every metric already
logged by ``manager_update`` is blind to this — ``state_latent_erank`` and
``state_pairwise_cos`` are computed on the rows ALONE and correctly report
"healthy" for rows that are distinct from each other but non-local. Distinct and
local are different properties.

THE MEASUREMENT — a block Jacobian:

    B[i,j] = || d s[i,:] / d obs_block_j ||_F        (--wrt obs, exact)
    B[i,j] ~ || s_i(pos_j + delta) - s_i(pos_j) || / delta   (--wrt positions, finite diff)

evaluated at on-policy states under the trained hierarchy, then

    diag_share[i] = B[i,i] / sum_j B[i,j]

Uniform (no localization at all) is 1/N. Full localization is 1.0. `best-perm`
is the Hungarian assignment over the row-normalized B, which catches a manager
that localizes on agent i but STORES it in row pi(i); if that is also ~1/N there
is no localization under any relabeling.

⚠ A FLAT READING IS MEANINGLESS WITHOUT THE POSITIVE CONTROL. Run ``--control``:
it builds a manager whose params are surgically block-diagonal and one that is
half-local, and asserts the metric separates them (1.0 / ~0.128 / ~1/N). Without
that, "no localization" is indistinguishable from a dead metric.

TWO VARIANTS, and they answer different questions:

* ``--wrt obs`` (default) — exact, and the one that produced the finding below.
  For ``manager_latent="local"`` it is 1.0 BY CONSTRUCTION (the encoder is
  applied per agent), so it is a structural regression test there, not a
  measurement.
* ``--wrt positions`` — the PHYSICAL residual. ``obs_i`` is egocentric but not
  proprioceptive: density sensors, ``neighbor_fraction`` and lidar all respond to
  teammates inside ``sector_sensor_radius``. So even a perfectly obs-local
  manager still has `s_i` perturbed by teammates' motion, through the channels
  agent i can perceive. This variant measures how much. Finite-difference rather
  than autodiff because ``mjx.ray`` (the lidar) is not usefully differentiable.

MEASURED 2026-09-09 — 12 arms (4 env groups x {feudal, feudal_n01, feudal_n05},
trial 0, 256 on-policy states each), ``--wrt obs``, N=16 so uniform = 0.0625:

    s (the space r^I is measured in)   trained 0.0631 (0.0621-0.0645)  init 0.0623
    g (the assigned goal)              trained 0.0628 (0.0621-0.0636)  init 0.0624

Paired trained-minus-init: +0.00086 for `s` (positive in 10/12), +0.00041 for
`g`. Against the control's calibration (half-local = 0.128, i.e. +0.0655 over
uniform) the trained arms moved ~1.3% of the way to HALF localized. `best-perm`
is 0.0655 trained vs 0.0650 at init.

So: no localization, in any arm, under any relabeling. See
``conf/model/feudal_film_local.yaml`` for the fix and
``plans/feudal_goal_reward_diagnosis_2026-09-09.md`` for what it explains.

Run:
    MUJOCO_GL=egl uv run python -m algorithms.feudal_mappo_jax.latent_locality_probe \\
        --batches mjx_16a_4o_trunc_1024 --models feudal,feudal_n01 --trial 0
    uv run python -m algorithms.feudal_mappo_jax.latent_locality_probe --control
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


def summarize(B: np.ndarray) -> dict:
    """diag share, best-permutation share, and mean row peak of a block matrix."""
    from scipy.optimize import linear_sum_assignment

    row = B / np.maximum(B.sum(axis=1, keepdims=True), 1e-30)
    r, c = linear_sum_assignment(-row)
    return {
        "diag": float(np.mean(np.diag(row))),
        "best_perm": float(row[r, c].mean()),
        "peak": float(row.max(axis=1).mean()),
    }


def block_jacobian_wrt_obs(manager, params, states, n_agents, obs_dim, chunk=16):
    """Mean ``B`` over `states`, for both `s` and `g`, w.r.t. observation blocks.

    Input columns are scaled by their on-policy std, so a block norm reads "how
    much does s[i] move per TYPICAL variation of agent j's features" rather than
    weighting an inert constant channel the same as a live one.
    """
    std = jnp.asarray(states.std(axis=0)) + 1e-8

    def _out(u, which):
        gs = (u * std)[None]
        obs = gs.reshape(1, n_agents, obs_dim)
        _, goal, s = manager.apply(params, None, gs, obs)
        return (s if which == "s" else goal)[0]

    def blocks(jac):
        j = jac.reshape(jac.shape[0], jac.shape[1], n_agents, obs_dim)
        return jnp.sqrt((j**2).sum(axis=(1, 3)))

    @jax.jit
    def one(u_batch):
        js = jax.vmap(jax.jacrev(lambda u: _out(u, "s")))(u_batch)
        jg = jax.vmap(jax.jacrev(lambda u: _out(u, "g")))(u_batch)
        return jax.vmap(blocks)(js), jax.vmap(blocks)(jg)

    acc_s = acc_g = 0.0
    n = 0
    for k in range(0, len(states), chunk):
        u = jnp.asarray(states[k : k + chunk]) / std
        bs, bg = one(u)
        acc_s = acc_s + np.asarray(bs).sum(0)
        acc_g = acc_g + np.asarray(bg).sum(0)
        n += u.shape[0]
    return acc_s / n, acc_g / n


def block_response_wrt_positions(env, manager, params, env_states, delta=0.25):
    """Finite-difference ``B[i,j] = ||s_i(pos_j + d) - s_i(pos_j)|| / d``.

    The PHYSICAL residual: how much agent j's world position moves agent i's
    latent, through whatever sensor channels carry it. Finite difference because
    the lidar goes through ``mjx.ray``, which autodiff does not handle usefully.
    Averaged over +x and +y perturbations of each agent in turn.
    """
    from mujoco import mjx

    n_agents = env.n_agents
    qadr = np.asarray(env._agent_qadr)  # (A, 2) qpos indices

    def latent(data):
        obs = env._get_obs(data)
        _, _, s = manager.apply(params, None, obs.reshape(1, -1), obs[None])
        return s[0]

    @jax.jit
    def one_state(state):
        base = latent(state.data)

        def perturb(j, axis):
            qpos = state.data.qpos.at[qadr[j, axis]].add(delta)
            d = mjx.forward(env.model, state.data.replace(qpos=qpos))
            return jnp.linalg.norm(latent(d) - base, axis=-1) / delta  # (A,)

        cols = [
            0.5 * (perturb(j, 0) + perturb(j, 1)) for j in range(n_agents)
        ]
        return jnp.stack(cols, axis=1)  # (A_row=i, A_col=j)

    acc, n = 0.0, 0
    for k in range(env_states[0].data.qpos.shape[0] if False else len(env_states)):
        acc = acc + np.asarray(one_state(env_states[k]))
        n += 1
    return acc / n


# ---------------------------------------------------------------------------
# On-policy state collection (the FULL hierarchy — the worker is goal-conditioned)
# ---------------------------------------------------------------------------


def collect_states(runner, manager, m_params, worker, w_params, key,
                   n_envs, n_samples, stride, keep_env_states=False):
    """Roll the trained hierarchy, snapshotting every `stride`-th step.

    Returns ``(global_states, env_states_or_None)``. Done envs are restarted the
    way ``trainer._env_step`` does (MJX has no auto-reset), so the state
    distribution stays on-policy instead of freezing on whatever ended the
    episode.
    """
    from algorithms.feudal_mappo_jax.manager import (
        goal_ring_pool,
        goal_ring_reset,
        goal_ring_write,
    )
    from algorithms.feudal_mappo_jax.network import sample_action
    from algorithms.feudal_mappo_jax.worker import bind_goal

    env = runner.env
    N, obs_dim, act_dim = env.n_agents, env.observation_dim, env.action_dim
    horizon, goal_dim = runner.config.goal_horizon, runner.config.goal_dim
    v_reset, v_step = jax.vmap(env.reset), jax.vmap(env.step)

    def _advance(carry, _):
        obs, state, ring, rng, t = carry
        rng, a_rng, r_rng = jax.random.split(rng, 3)
        gs = obs.reshape(obs.shape[0], -1)
        _, goal, _ = manager.apply(m_params, None, gs, obs)
        ring = goal_ring_write(ring, goal, t)
        pooled = goal_ring_pool(ring)
        b = obs.shape[0]
        action, _ = sample_action(
            a_rng,
            bind_goal(worker.apply, pooled.reshape(b * N, goal_dim)),
            w_params,
            obs.reshape(b * N, obs_dim),
            discrete=False,
            deterministic=True,
        )
        nobs, nstate, _, term, trunc, _ = v_step(state, action.reshape(b, N, act_dim))
        done = term | trunc
        robs, rstate = v_reset(jax.random.split(r_rng, b))
        nstate = jax.tree.map(
            lambda r, n: jnp.where(done.reshape((-1,) + (1,) * (n.ndim - 1)), r, n),
            rstate,
            nstate,
        )
        nobs = jnp.where(done[:, None, None], robs, nobs)
        ring = goal_ring_reset(ring, done)
        return (nobs, nstate, ring, rng, t + 1), None

    def _sample(carry, _):
        carry, _u = jax.lax.scan(_advance, carry, None, length=stride)
        return carry, (carry[0].reshape(carry[0].shape[0], -1), carry[1])

    key, rk = jax.random.split(key)
    obs, state = v_reset(jax.random.split(rk, n_envs))
    ring = jnp.zeros((horizon, n_envs, N, goal_dim))
    _, (gs, env_states) = jax.lax.scan(
        _sample, (obs, state, ring, key, jnp.int32(0)), None, length=n_samples
    )
    flat_gs = np.asarray(gs).reshape(-1, gs.shape[-1])
    if not keep_env_states:
        return flat_gs, None
    # Flatten (n_samples, n_envs) -> a list of single-env states.
    per = [
        jax.tree.map(lambda x, a=a, b=b: x[a, b], env_states)
        for a in range(n_samples)
        for b in range(n_envs)
    ]
    return flat_gs, per


# ---------------------------------------------------------------------------
# Positive control — REQUIRED before believing any flat reading
# ---------------------------------------------------------------------------


def run_control(n_agents=16, obs_dim=40, goal_dim=32, hidden=256):
    """Does the metric detect localization when it EXISTS? Calibrates the scale."""
    import flax

    from algorithms.feudal_mappo_jax.manager import FeudalManager

    mgr = FeudalManager(
        n_agents=n_agents, goal_dim=goal_dim, hidden_dim=hidden, core="mlp", horizon=10
    )
    p = mgr.init(jax.random.PRNGKey(0), None, jnp.zeros(n_agents * obs_dim))
    states = np.asarray(
        jax.random.normal(jax.random.PRNGKey(1), (64, n_agents * obs_dim))
    )

    def report(tag, params):
        Bs, Bg = block_jacobian_wrt_obs(mgr, params, states, n_agents, obs_dim)
        for name, B in (("s", Bs), ("g", Bg)):
            m = summarize(B)
            print(
                f"  {tag:16s} {name}: diag={m['diag']:.4f} "
                f"best-perm={m['best_perm']:.4f} peak={m['peak']:.4f}"
            )
        return summarize(Bs)["diag"]

    print(f"uniform baseline 1/N = {1 / n_agents:.4f}")
    d_init = report("as-init", p)

    per = hidden // n_agents
    k0 = np.zeros((n_agents * obs_dim, hidden), np.float32)
    k1 = np.zeros((hidden, hidden), np.float32)
    ks = np.zeros((hidden, n_agents * goal_dim), np.float32)
    kc = np.zeros((n_agents * goal_dim, hidden), np.float32)
    kg = np.zeros((hidden, n_agents * goal_dim), np.float32)
    for i in range(n_agents):
        rn = lambda seed, shp: np.asarray(
            jax.random.normal(jax.random.PRNGKey(seed), shp)
        )
        k0[i * obs_dim : (i + 1) * obs_dim, i * per : (i + 1) * per] = (
            rn(10 + i, (obs_dim, per)) * 0.3
        )
        k1[i * per : (i + 1) * per, i * per : (i + 1) * per] = (
            rn(100 + i, (per, per)) * 0.5
        )
        ks[i * per : (i + 1) * per, i * goal_dim : (i + 1) * goal_dim] = (
            rn(200 + i, (per, goal_dim)) * 0.5
        )
        kc[i * goal_dim : (i + 1) * goal_dim, i * per : (i + 1) * per] = (
            rn(300 + i, (goal_dim, per)) * 0.5
        )
        kg[i * per : (i + 1) * per, i * goal_dim : (i + 1) * goal_dim] = (
            rn(400 + i, (per, goal_dim)) * 0.5
        )

    q = flax.core.unfreeze(jax.tree.map(np.asarray, p))
    for name, kern in (
        ("f_percept_0", k0), ("f_percept_1", k1), ("f_Mspace", ks),
        ("core", kc), ("goal_head", kg),
    ):
        q["params"][name]["kernel"] = kern
        q["params"][name]["bias"] = np.zeros_like(q["params"][name]["bias"])
    d_block = report("block-diagonal", jax.tree.map(jnp.asarray, q))

    q2 = flax.core.unfreeze(jax.tree.map(np.asarray, q))
    q2["params"]["f_Mspace"]["kernel"] = ks + 0.5 * np.asarray(
        jax.random.normal(jax.random.PRNGKey(7), (hidden, n_agents * goal_dim))
    ) * 0.5
    d_half = report("half-mixed", jax.tree.map(jnp.asarray, q2))

    assert d_block > 0.99, f"metric is dead: block-diagonal reads {d_block}"
    assert d_half > 1.5 / n_agents, f"metric cannot see partial locality: {d_half}"
    assert abs(d_init - 1 / n_agents) < 0.005, f"init is not uniform: {d_init}"
    print("\ncontrol OK — the metric separates local from non-local")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true", help="run the positive control only")
    ap.add_argument("--batches", default="mjx_16a_4o_trunc_1024")
    ap.add_argument("--models", default="feudal")
    ap.add_argument("--trial", default="0")
    ap.add_argument("--wrt", choices=["obs", "positions"], default="obs")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--n-position-states", type=int, default=16)
    args = ap.parse_args()

    if args.control:
        run_control()
        return

    from flax.serialization import msgpack_restore

    from algorithms.feudal_mappo_jax.goal_dependence_probe import (
        _checkpoint_path,
        _runner,
    )
    from algorithms.feudal_mappo_jax.mappo import build_manager
    from algorithms.feudal_mappo_jax.worker import FeudalWorker

    for batch in args.batches.split(","):
        for model in args.models.split(","):
            path = _checkpoint_path(batch, model, args.trial)
            if path is None:
                print(f"{batch}/{model}: no checkpoint")
                continue
            runner = _runner(batch, model, args.trial, quiet=True)
            env = runner.env
            N, obs_dim = env.n_agents, env.observation_dim
            cfg = runner.config
            tree = msgpack_restore(path.read_bytes())
            m_params = jax.tree.map(jnp.asarray, {"params": tree["manager"]["params"]})
            w_params = jax.tree.map(jnp.asarray, {"params": tree["actor"]["params"]})

            manager = build_manager(cfg, N)
            worker = FeudalWorker(
                action_dim=env.action_dim,
                goal_dim=cfg.goal_dim,
                hidden_dim=cfg.hidden_dim,
                discrete=getattr(env, "discrete", False),
                goal_embed_dim=cfg.goal_embed_dim,
                normalize_pooled_goal=cfg.normalize_pooled_goal,
                zero_goal=cfg.zero_goal,
                worker_fusion=cfg.worker_fusion,
            )

            states, env_states = collect_states(
                runner, manager, m_params, worker, w_params,
                jax.random.PRNGKey(0), args.n_envs, args.n_samples, args.stride,
                keep_env_states=(args.wrt == "positions"),
            )

            head = (
                f"\n=== {batch}/{model}/{args.trial}  N={N} "
                f"goal_dim={cfg.goal_dim} latent={cfg.manager_latent} "
                f"wrt={args.wrt}  uniform=1/N={1 / N:.4f}"
            )
            print(head)

            if args.wrt == "obs":
                Bs, Bg = block_jacobian_wrt_obs(manager, m_params, states, N, obs_dim)
                init_p = manager.init(
                    jax.random.PRNGKey(0),
                    None,
                    jnp.zeros(N * obs_dim),
                    jnp.zeros((N, obs_dim))
                    if cfg.manager_latent in ("local", "local_global")
                    else None,
                )
                Bs0, Bg0 = block_jacobian_wrt_obs(
                    manager, init_p, states, N, obs_dim
                )
                for name, B, B0 in (("s (r^I space)", Bs, Bs0), ("g (goal)", Bg, Bg0)):
                    m, m0 = summarize(B), summarize(B0)
                    print(
                        f"  {name:16s} trained: diag={m['diag']:.4f} "
                        f"best-perm={m['best_perm']:.4f} peak={m['peak']:.4f}"
                    )
                    print(
                        f"  {'':16s} init   : diag={m0['diag']:.4f} "
                        f"best-perm={m0['best_perm']:.4f} peak={m0['peak']:.4f}"
                    )
            else:
                sub = env_states[: args.n_position_states]
                B = block_response_wrt_positions(env, manager, m_params, sub)
                m = summarize(B)
                print(
                    f"  s vs agent POSITIONS  diag={m['diag']:.4f} "
                    f"best-perm={m['best_perm']:.4f} peak={m['peak']:.4f} "
                    f"({len(sub)} states)"
                )
                print(
                    "  NOTE: even a perfectly obs-local manager scores < 1.0 here — "
                    "obs_i itself responds to teammates inside sector_sensor_radius."
                )


if __name__ == "__main__":
    main()
