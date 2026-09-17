"""How much did fixing the `r^I` timing actually change the worker's update?

THE DEFECT, now FIXED (`manager.worker_intrinsic_reward_aligned`, wired in
`trainer._apply_intrinsic_reward`). `trainer._env_step` stored the PRE-action
latent as ``state_latent[t]``, and ``worker_intrinsic_reward`` built

    r^I_t = 1/c * sum_{i=1..c} d_cos(s_t - s_{t-i}, g_{t-i})

onto the transition carrying ``a_t``. Every term is fixed before ``a_t`` is
sampled, so r^I_t was exactly independent of ``a_t`` — while ``reward[t]`` on
that same transition IS ``a_t``'s consequence. The two streams scored different
actions. Production now scores the successor the action actually produced:

    r^I_t = mean_{k=0..c-1} d_cos(s_plus_t - s_{t-k}, g_{t-k})

with ``s_plus_t`` the latent of the true successor, captured before the reset.

WHY THIS WAS NEVER OBVIOUSLY FATAL, and why the probe still exists. GAE does not
use `r^I_t` alone: `A_t = sum_j (gamma*lambda)^j delta_{t+j}`, so `a_t`'s
influence still reached `A_t` through `r^I_{t+1}` onward — delayed one step and
attenuated by `gamma*lambda ~ 0.94`. Measured, that is essentially all that
happened: 0.08 / 1.2 / 3.6 degrees of gradient rotation at alpha 0.01 / 0.1 /
0.5, against 0.68 / 6.7 / 26.3 for the intrinsic term's own effect. So the fix
is ~13% of what `r^I` is already doing, and it is NOT why positive-alpha arms
lose to alpha=0. It was made for the boundary, not the magnitude: any endpoint
derived by shifting the stored latents must pay 0 on the action that ENDS an
episode, which is a standing bonus for terminating.

WHAT IS MEASURED. Not the reward — the **update**. At the first PPO epoch the
importance ratio is exactly 1 (pinned by `test_feudal_seams.py`), so the actor's
gradient is exactly

    grad = sum_t A_t * grad_theta log pi(a_t | s_t, w_t)

with `A = normalize(A_ext) + alpha * normalize(A_int)`. That gradient is built
THREE ways on the SAME trajectory, same params, same critics, and the cosines
between them reported:

  LEGACY      the pre-fix reward (endpoint `s_t`).
  SHIFT       `legacy_shift_approximation` — the estimate this probe used before
              the fix existed. Exact in the interior (`r_corrected[t] ==
              r_legacy[t+1]`), 0 at every boundary transition.
  CORRECTED   production: the real pre-reset successor latent.

legacy-vs-CORRECTED is what the fix did. shift-vs-CORRECTED is the part a shift
could never have delivered, i.e. terminal-action credit alone. A legacy-vs-
CORRECTED angle far above the recorded 0.08/1.2/3.6 deg is a signal that the
implementation is wrong — most likely the successor encoded below
`_restart_done` — rather than a discovery.

Reported alongside, to keep a high cosine interpretable rather than vacuous:

* ``cos(grad_ext, grad_legacy)`` — the scale the cosines live on. If the
  intrinsic term barely moves the update at this alpha, everything else is high
  for that reason alone.
* ``corr(A_int_legacy, A_int_corrected)`` — the same question one level up,
  before the extrinsic stream dilutes it.
* ``corr(r^I)`` between arms, and the stream's one-step autocorrelation — a
  slowly varying `r^I` is nearly its own shift, which makes the misalignment
  harmless FOR THAT REASON and is worth distinguishing.
* a CAUSAL check that the corrected `r^I_t` depends on `a_t`: the env is stepped
  from one state under two different actions and the resulting rewards compared.
  (The old version of this check edited a stored action while holding the stored
  states fixed, which cannot show causal dependence — the legacy reward is a
  function of (s, g) alone, so it was bitwise unchanged by construction.)

Usage::

    MUJOCO_GL=egl uv run python -m algorithms.feudal_mappo_jax.intrinsic_timing_probe \\
        --batches mjx_12a_3o_trunc_1024 --models feudal_film_n01,feudal_film_n05 \\
        --trials 0,1,2

Note the probe reads `alpha` from the arm's config and uses it as-is; the shipped
schedule anneals alpha to 0 over training, so the cosine at a late checkpoint is
reported at the CONFIGURED alpha, i.e. the strongest case for the bug mattering.
Run it at an EARLY checkpoint too: the reason the misalignment is nearly harmless
is `r^I`'s lag-1 autocorrelation (0.53-0.73 on trained nets), and a manager still
moving fast may not have it.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np


def legacy_shift_approximation(states, goals, horizon, done=None):
    """`r^I` realigned by SHIFTING the stored latents — the pre-fix approximation.

    Kept, and deliberately renamed away from "aligned", because it is no longer
    the corrected reward: it is the **third arm** of the comparison, and the gap
    between it and the real fix is the one quantity that isolates terminal-action
    credit.

    Shifting the states forward by one (so index `t` reads `s_{t+1}`) while
    leaving the goals in place gives ``d_cos(s_{t+1} - s_{t+1-i}, g_{t+1-i})``,
    which is **exactly** the corrected reward in the interior: working the
    indices through, ``r_corrected[t] == r_legacy[t+1]``. That identity is why
    the gradient rotations this probe measured before the fix existed are a
    sound estimate of what the fix does.

    What it cannot reproduce is the boundary, in two places:

    * at a `done` step the shifted mask covers ``done[t]`` itself, so **every**
      term is masked and the episode-ENDING action is paid 0 — a standing bonus
      for terminating, the same shape as the `boundary_truncates` failure the MJX
      env removed. (`states[t+1]` would be the freshly RESET latent there, so
      paying 0 is the conservative reading, not a fixable indexing detail.)
    * the last stored step has no successor at all and is masked to 0 rather than
      reusing `s_T`, which would score a zero displacement.

    On a 1024-step `trunc` episode at `n_steps=1048` that is ~2 transitions of
    1048; on the ~43-step boundary-terminating baseline arm it is ~2.4% of them.
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
        worker_intrinsic_reward_aligned,
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
                    # The successor latent, read BEFORE the restart below. This
                    # is the endpoint of the corrected reward and the only thing
                    # a shift cannot reconstruct; encoding it after the restart
                    # would measure the reset instead.
                    s_plus = manager.apply(
                        mp, None, nobs.reshape(E, -1), nobs, latent_only=True
                    )
                    # Restart finished envs, as trainer._env_step does. The probe
                    # used to run on past terminal states, so the boundary — the
                    # whole subject of the corrected reward — was never
                    # exercised. (This consumes an extra rng split per step, so
                    # action sequences differ from the pre-fix probe.)
                    rng, rk = jax.random.split(rng)
                    robs, rstate = v_reset(jax.random.split(rk, E))

                    def _sel(fresh, cur):
                        d = done.reshape((-1,) + (1,) * (cur.ndim - 1))
                        return jnp.where(d, fresh, cur)

                    nobs = _sel(robs, nobs)
                    nstate = jax.tree.map(_sel, rstate, nstate)
                    # AFTER `w_t` was consumed, matching trainer.py:358 — the
                    # ring reset must not retroactively change the goal the
                    # action was sampled from.
                    ring = goal_ring_reset(ring, done)
                    return (nobs, nstate, m_carry, ring, rng), (
                        obs, gs, s_lat, s_plus, goal, pooled, a, lp, r, done
                    )

                rng, sk = jax.random.split(rng)
                _, out = jax.lax.scan(
                    step, (obs, env_state, m_carry, ring, sk), jnp.arange(T)
                )
                (obs_t, gs_t, s_t, sp_t, g_t, pooled_t, act_t, lp_t, rew_t,
                 done_t) = out

                done_a = jnp.broadcast_to(
                    done_t[..., None].astype(jnp.float32), s_t.shape[:-1]
                )
                # LEGACY = the pre-fix production reward (endpoint `s_t`).
                # SHIFT   = the old approximation to the fix (endpoint `s_{t+1}`
                #           from the stored array, boundaries dropped).
                # CORRECTED = production today: the real pre-reset successor.
                r_cur = worker_intrinsic_reward(s_t, g_t, c, done=done_a)
                r_shift = legacy_shift_approximation(s_t, g_t, c, done=done_a)
                r_ali = worker_intrinsic_reward_aligned(
                    s_t, sp_t, g_t, c, done=done_a
                )

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

                a_cur, a_ali, a_shift = adv_of(r_cur), adv_of(r_ali), adv_of(r_shift)

                def norm(x):
                    return (x - x.mean()) / (x.std() + 1e-8)

                alpha = cfg.intrinsic_coef
                A_ext = norm(adv_ext)
                A_cur = A_ext + alpha * norm(a_cur)
                A_ali = A_ext + alpha * norm(a_ali)
                A_shift = A_ext + alpha * norm(a_shift)

                # --- The update direction, at ratio == 1 (first PPO epoch).
                def pg(params, A):
                    def loss(p):
                        lp_new, _ = evaluate_action(
                            bind_goal(worker.apply, pooled_t), p, obs_t, act_t,
                            discrete,
                        )
                        return -(lp_new * A).mean()
                    return _flat_grad(jax.grad(loss)(params))

                g_cur, g_ali = pg(wp, A_cur), pg(wp, A_ali)
                g_ext, g_shift = pg(wp, A_ext), pg(wp, A_shift)

                # --- Action dependence, measured by STEPPING THE ENV.
                # Editing a stored action while holding the stored states fixed
                # cannot establish causal dependence: the legacy reward is a
                # function of (s, g) alone, so it is bitwise unchanged by
                # construction and that comparison is vacuous. Instead: roll a
                # short prefix, then branch the SAME state on two different
                # actions and read what each earns on its own transition.
                def _prefix(n):
                    rngp = jax.random.PRNGKey(7)
                    obs_p, st_p = v_reset(jax.random.split(rngp, E))
                    mc = manager.initialize_carry(jax.random.PRNGKey(8), (E,))
                    rg = jnp.zeros((c, E, N, cfg.goal_dim))
                    S, G, branch = [], [], None
                    for t in range(n):
                        mc, goal, s_lat = manager.apply(
                            mp, mc, obs_p.reshape(E, -1), obs_p
                        )
                        rg = goal_ring_write(rg, goal, t)
                        a, _ = sample_action(
                            jax.random.fold_in(rngp, t),
                            bind_goal(worker.apply, goal_ring_pool(rg)),
                            wp, obs_p, discrete,
                        )
                        S.append(s_lat)
                        G.append(goal)
                        if t == n - 1:
                            branch = (st_p, a)
                            break
                        obs_p, st_p = v_step(st_p, a)[:2]
                    return jnp.stack(S), jnp.stack(G), branch

                S_p, G_p, (st_b, a_b) = _prefix(c)

                def _last_reward_under(action):
                    nobs = v_step(st_b, action)[0]
                    sp = manager.apply(
                        mp, None, nobs.reshape(E, -1), nobs, latent_only=True
                    )
                    # Only index -1 is read, and it depends on endpoints[-1]
                    # alone, so the earlier endpoints are irrelevant filler.
                    sp_seq = jnp.concatenate([jnp.zeros_like(S_p[:-1]), sp[None]])
                    return worker_intrinsic_reward_aligned(S_p, sp_seq, G_p, c)[-1]

                alt = (a_b + 1) % env.action_dim if discrete else -a_b
                moved = float(
                    jnp.abs(_last_reward_under(a_b) - _last_reward_under(alt)).max()
                )
                # The legacy form never reads a successor, so its value at that
                # index is the same number for both branches — structurally, not
                # empirically.
                legacy_same = True

                boundary = int(done_t.sum()) + 1  # done steps + the final step

                print(f"\n=== {batch}/{model}/{trial}  alpha={alpha}  c={c}  T={T} E={E}")
                print(f"  dones in rollout / boundary transitions   "
                      f"{int(done_t.sum())} / {boundary} of {T * E}")
                print("  -- reward streams")
                print(f"  r^I  corr(legacy, corrected)        {_corr(r_cur, r_ali):+.4f}")
                print(f"  r^I  corr(shift-approx, corrected)  {_corr(r_shift, r_ali):+.4f}")
                print(f"  r^I  lag-1 autocorrelation          {_corr(r_cur[:-1], r_cur[1:]):+.4f}")
                print(f"  A^I  corr(legacy, corrected)        {_corr(a_cur, a_ali):+.4f}")
                print("  -- the update direction (ratio == 1, first PPO epoch)")
                print(f"  grad cos(legacy, CORRECTED)         {_cos(g_cur, g_ali):+.6f}")
                print(f"  grad cos(legacy, shift-approx)      {_cos(g_cur, g_shift):+.6f}")
                print(f"  grad cos(shift-approx, CORRECTED)   {_cos(g_shift, g_ali):+.6f}")
                print(f"  grad cos(extrinsic-only, legacy)    {_cos(g_ext, g_cur):+.6f}")
                print("     (that last line is the SCALE: if it is ~1.0 the")
                print("      intrinsic term barely moves the update at this alpha,")
                print("      and the others are high for that reason alone.)")
                print("     EXPECTED: legacy-vs-corrected ~= legacy-vs-shift, since")
                print("      the two agree everywhere but the boundary. A much")
                print("      LARGER gap than the recorded 0.08/1.2/3.6 deg at")
                print("      alpha 0.01/0.1/0.5 means the implementation is wrong")
                print("      (most likely the successor encoded after the reset),")
                print("      not that the bug was bigger than measured.")
                print(f"  |grad| legacy/corrected/ext-only    "
                      f"{float(jnp.linalg.norm(g_cur)):.4e} / "
                      f"{float(jnp.linalg.norm(g_ali)):.4e} / "
                      f"{float(jnp.linalg.norm(g_ext)):.4e}")
                print("  -- causal dependence on a_t (env stepped, not restored)")
                print(f"  corrected r^I_t moves by            {moved:.4e}")
                print(f"  legacy r^I_t reads no successor     {legacy_same}")
                print(f"  mean |r^I| legacy/corrected         "
                      f"{float(jnp.abs(r_cur).mean()):.4f} / "
                      f"{float(jnp.abs(r_ali).mean()):.4f}")


if __name__ == "__main__":
    main()
