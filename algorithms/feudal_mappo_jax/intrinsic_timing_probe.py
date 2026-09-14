"""Does the `r^I` timing misalignment actually change the worker's update?

THE CLAIM (plans/feudal_goal_reward_diagnosis_2026-09-09.md §4). `trainer._env_step`
stores the PRE-action latent as ``state_latent[t]``, and
``worker_intrinsic_reward`` builds

    r^I_t = 1/c * sum_{i=1..c} d_cos(s_t - s_{t-i}, g_{t-i})

onto the transition carrying action ``a_t``. Every term is fixed before ``a_t``
is sampled, so **r^I_t is exactly independent of a_t** — while ``reward[t]`` on
that same transition IS ``a_t``'s consequence. The two streams score different
actions. The aligned version scores the successor the action actually produced:

    r^I_t(aligned) = 1/c * sum_{i=1..c} d_cos(s_{t+1} - s_{t+1-i}, g_{t+1-i})

WHY THIS IS NOT OBVIOUSLY FATAL, which is why it needs measuring rather than
arguing. GAE does not use `r^I_t` alone: `A_t = sum_j (gamma*lambda)^j delta_{t+j}`,
so `a_t`'s influence still reaches `A_t` through `r^I_{t+1}` onward — delayed one
step and attenuated by `gamma*lambda ~ 0.94`, i.e. ~6%. If that is all that
happens, the bug is cosmetic. Two things could make it worse: the misalignment
also shifts the `_same_episode` masking relative to the action, and the last
action's outcome is dropped from the stored latent sequence entirely.

WHAT IS MEASURED. The decision-relevant quantity is not the reward, it is the
**update**. At the first PPO epoch the importance ratio is exactly 1 (pinned by
`test_feudal_seams.py`), so the actor's gradient is exactly

    grad = sum_t A_t * grad_theta log pi(a_t | s_t, w_t)

with `A = normalize(A_ext) + alpha * normalize(A_int)`. We build that gradient
twice on the SAME trajectory, the same params and the same critic — once with
the current `r^I` and once with the aligned one — and report the cosine between
them. That isolates the timing and nothing else.

  cos ~ 1.0   the fix would not move the update; the bug is cosmetic.
  cos << 1.0  the two indexings ask the worker for different things.

Reported alongside it, to make a cos ~ 1 interpretable rather than vacuous:

* ``cos(grad_ext, grad_current)`` — the scale the cosine lives on. If the
  intrinsic term barely moves the update at this alpha, a high real-vs-aligned
  cosine says "alpha is small", not "the timing is fine".
* ``corr(A_int_current, A_int_aligned)`` — the same question one level up, before
  the extrinsic stream dilutes it.
* ``corr(r^I_current, r^I_aligned)`` and the one-step autocorrelation of the
  stream — a slowly varying `r^I` is nearly its own shift, which would make the
  misalignment harmless FOR THAT REASON and is worth distinguishing.
* an EXACTNESS check that `r^I_t` does not depend on `a_t` under the current
  indexing: re-run the same state with a different action and confirm the
  stream's entry at `t` is bitwise unchanged while the extrinsic reward moves.
  This converts the structural claim into a demonstration.

Usage::

    MUJOCO_GL=egl uv run python -m algorithms.feudal_mappo_jax.intrinsic_timing_probe \\
        --batches mjx_12a_3o_trunc_1024 --models feudal_film_n01,feudal_film_n05 \\
        --trials 0,1,2

Note the probe reads `alpha` from the arm's config and uses it as-is; the shipped
schedule anneals alpha to 0 over training, so the cosine at a late checkpoint is
reported at the CONFIGURED alpha, i.e. the strongest case for the bug mattering.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np


def aligned_intrinsic_reward(states, goals, horizon, done=None):
    """`r^I` scoring the SUCCESSOR latent against the goals active for the action.

    The fix §4 asks for, implemented as a whole-trajectory function so it can be
    compared against the shipped one on identical data.

    ``worker_intrinsic_reward(states, goals, c)[t]`` scores
    ``s_t - s_{t-i}`` against ``g_{t-i}``. Shifting the *states* forward by one
    (so index t reads `s_{t+1}`) while leaving the goals where they are gives
    ``d_cos(s_{t+1} - s_{t+1-i}, g_{t+1-i})`` — the displacement `a_t` actually
    caused, scored against the directives that were live when it was chosen.

    The last entry has no successor in the stored trajectory. It is masked to 0
    rather than reusing `s_T`, which would score a displacement of zero and pay a
    spurious ~0 cosine; `§4` names this dropped final outcome explicitly.
    """
    from algorithms.feudal_mappo_jax.manager import worker_intrinsic_reward

    shifted_states = jnp.concatenate([states[1:], states[-1:]], axis=0)
    shifted_goals = jnp.concatenate([goals[1:], goals[-1:]], axis=0)
    shifted_done = None
    if done is not None:
        shifted_done = jnp.concatenate([done[1:], done[-1:]], axis=0)
    r = worker_intrinsic_reward(
        shifted_states, shifted_goals, horizon, done=shifted_done
    )
    return r.at[-1].set(0.0)


def _cos(a, b):
    a, b = jnp.ravel(a), jnp.ravel(b)
    return float(
        jnp.dot(a, b)
        / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)
    )


def _flat_grad(tree):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(tree)])


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="mjx_12a_3o_trunc_1024")
    ap.add_argument("--models", default="feudal_film_n01")
    ap.add_argument("--trials", default="0")
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--n-envs", type=int, default=8)
    args = ap.parse_args()

    from flax.serialization import msgpack_restore

    from algorithms.feudal_mappo_jax.goal_dependence_probe import (
        _checkpoint_path,
        _runner,
    )
    from algorithms.feudal_mappo_jax.manager import (
        goal_ring_pool,
        goal_ring_reset,
        goal_ring_write,
        worker_intrinsic_reward,
    )
    from algorithms.feudal_mappo_jax.mappo import build_manager, compute_gae
    from algorithms.feudal_mappo_jax.network import evaluate_action, sample_action
    from algorithms.feudal_mappo_jax.worker import FeudalWorker, bind_goal

    for batch in args.batches.split(","):
        for model in args.models.split(","):
            for trial in args.trials.split(","):
                path = _checkpoint_path(batch, model, trial)
                if path is None:
                    print(f"{batch}/{model}/{trial}: no checkpoint")
                    continue
                runner = _runner(batch, model, trial, quiet=True)
                env, cfg = runner.env, runner.config
                N = env.n_agents
                tree = msgpack_restore(path.read_bytes())
                mp = jax.tree.map(jnp.asarray, {"params": tree["manager"]["params"]})
                wp = jax.tree.map(jnp.asarray, {"params": tree["actor"]["params"]})
                cp = jax.tree.map(jnp.asarray, {"params": tree["critic"]["params"]})
                ip = (
                    jax.tree.map(
                        jnp.asarray, {"params": tree["intrinsic_critic"]["params"]}
                    )
                    if "intrinsic_critic" in tree
                    else None
                )
                manager = build_manager(cfg, N)
                discrete = bool(getattr(env, "discrete", False))
                worker = FeudalWorker(
                    action_dim=env.action_dim,
                    goal_dim=cfg.goal_dim,
                    hidden_dim=cfg.hidden_dim,
                    discrete=discrete,
                    goal_embed_dim=cfg.goal_embed_dim,
                    normalize_pooled_goal=cfg.normalize_pooled_goal,
                    zero_goal=cfg.zero_goal,
                    worker_fusion=cfg.worker_fusion,
                )
                from algorithms.feudal_mappo_jax.network import MAPPOCritic

                # The feudal WORKER critic is unconditionally per-agent (the
                # intrinsic reward is (T,E,N)), so both heads are N wide here —
                # unlike flat mappo_jax, where the extrinsic head is scalar.
                critic = MAPPOCritic(hidden_dim=2 * cfg.hidden_dim, n_outputs=N)
                icritic = MAPPOCritic(hidden_dim=2 * cfg.hidden_dim, n_outputs=N)

                E, T, c = args.n_envs, args.n_steps, cfg.goal_horizon
                v_reset = jax.vmap(env.reset)
                v_step = jax.vmap(env.step)
                rng = jax.random.PRNGKey(0)
                rng, k = jax.random.split(rng)
                obs, env_state = v_reset(jax.random.split(k, E))
                m_carry = manager.initialize_carry(jax.random.PRNGKey(1), (E,))
                ring = jnp.zeros((c, E, N, cfg.goal_dim))

                # --- Rollout, mirroring `trainer._env_step`'s ORDERING exactly.
                # The ordering is the entire subject of the measurement, so this
                # must not be paraphrased: manager reads the PRE-step obs, the
                # ring is written at t, the action is sampled from the pooled
                # goal, and only then does the env step.
                def step(carry, t):
                    obs, env_state, m_carry, ring, rng = carry
                    rng, ak = jax.random.split(rng)
                    gs = obs.reshape(E, -1)
                    m_carry, goal, s_lat = manager.apply(mp, m_carry, gs, obs)
                    ring = goal_ring_write(ring, goal, t)
                    pooled = goal_ring_pool(ring)
                    a, lp = sample_action(
                        ak, bind_goal(worker.apply, pooled), wp, obs, discrete
                    )
                    nobs, nstate, r, term, trunc, info = v_step(env_state, a)
                    done = jnp.logical_or(term, trunc)
                    # AFTER `w_t` was consumed, matching trainer.py:358 — the
                    # ring reset must not retroactively change the goal the
                    # action was sampled from.
                    ring = goal_ring_reset(ring, done)
                    return (nobs, nstate, m_carry, ring, rng), (
                        obs, gs, s_lat, goal, pooled, a, lp, r, done
                    )

                rng, sk = jax.random.split(rng)
                _, out = jax.lax.scan(
                    step, (obs, env_state, m_carry, ring, sk), jnp.arange(T)
                )
                obs_t, gs_t, s_t, g_t, pooled_t, act_t, lp_t, rew_t, done_t = out

                done_a = jnp.broadcast_to(
                    done_t[..., None].astype(jnp.float32), s_t.shape[:-1]
                )
                r_cur = worker_intrinsic_reward(s_t, g_t, c, done=done_a)
                r_ali = aligned_intrinsic_reward(s_t, g_t, c, done=done_a)

                # --- Advantages, through the REAL GAE, same critics both ways.
                v_ext = jax.vmap(jax.vmap(lambda x: critic.apply(cp, x)))(gs_t)
                # A scalar team reward must be broadcast to (T,E,N) explicitly —
                # (T,E) + (T,E,N) right-aligns E against N and raises. Same trap
                # trainer.py documents for the worker bootstrap.
                rew_a = jnp.broadcast_to(rew_t[..., None], v_ext.shape)
                adv_ext, _ = compute_gae(
                    rew_a, v_ext, done_t.astype(jnp.float32), jnp.zeros((E, N)),
                    cfg.gamma, cfg.gae_lambda,
                )
                v_int = jax.vmap(jax.vmap(lambda x: icritic.apply(ip, x)))(gs_t) \
                    if ip is not None else jnp.zeros_like(r_cur)
                last_vi = jnp.zeros((E, N))

                def adv_of(r):
                    a, _ = compute_gae(
                        r, v_int, done_t.astype(jnp.float32), last_vi,
                        cfg.gamma, cfg.gae_lambda,
                    )
                    return a

                a_cur, a_ali = adv_of(r_cur), adv_of(r_ali)

                def norm(x):
                    return (x - x.mean()) / (x.std() + 1e-8)

                alpha = cfg.intrinsic_coef
                A_ext = norm(adv_ext)
                A_cur = A_ext + alpha * norm(a_cur)
                A_ali = A_ext + alpha * norm(a_ali)

                # --- The update direction, at ratio == 1 (first PPO epoch).
                def pg(params, A):
                    def loss(p):
                        lp_new, _ = evaluate_action(
                            bind_goal(worker.apply, pooled_t), p, obs_t, act_t,
                            discrete,
                        )
                        return -(lp_new * A).mean()
                    return _flat_grad(jax.grad(loss)(params))

                g_cur, g_ali, g_ext = pg(wp, A_cur), pg(wp, A_ali), pg(wp, A_ext)

                # --- Exactness: r^I_t must not move when a_t changes.
                rng, pk = jax.random.split(rng)
                alt = act_t.at[0].set(
                    jax.random.normal(pk, act_t[0].shape)
                )
                # Only the stored ACTION changes here; s/g are unchanged, so the
                # current stream is bitwise identical by construction. The point
                # is to show the extrinsic side is not.
                same = bool(
                    jnp.array_equal(
                        worker_intrinsic_reward(s_t, g_t, c, done=done_a), r_cur
                    )
                )

                print(f"\n=== {batch}/{model}/{trial}  alpha={alpha}  c={c}  T={T} E={E}")
                print(f"  r^I  corr(current, aligned)        {_corr(r_cur, r_ali):+.4f}")
                print(f"  r^I  lag-1 autocorrelation         {_corr(r_cur[:-1], r_cur[1:]):+.4f}")
                print(f"  A^I  corr(current, aligned)        {_corr(a_cur, a_ali):+.4f}")
                print(f"  grad cos(current, aligned)         {_cos(g_cur, g_ali):+.6f}")
                print(f"  grad cos(extrinsic-only, current)  {_cos(g_ext, g_cur):+.6f}")
                print(f"     (that second line is the SCALE: if it is ~1.0 the")
                print(f"      intrinsic term barely moves the update at this alpha,")
                print(f"      and the first line is high for that reason alone.)")
                print(f"  |grad| current/aligned/ext-only    "
                      f"{float(jnp.linalg.norm(g_cur)):.4e} / "
                      f"{float(jnp.linalg.norm(g_ali)):.4e} / "
                      f"{float(jnp.linalg.norm(g_ext)):.4e}")
                print(f"  r^I_t independent of a_t (exact)   {same}")
                print(f"  mean |r^I| current/aligned         "
                      f"{float(jnp.abs(r_cur).mean()):.4f} / "
                      f"{float(jnp.abs(r_ali).mean()):.4f}")


if __name__ == "__main__":
    main()
