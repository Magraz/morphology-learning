"""Where along the manager's local path does per-agent diversity disappear?

Runs the trained manager on on-policy states and, at each stage, measures how
distinct the N per-agent rows are, using the SAME statistic the live diagnostic
uses (participation ratio of the Gram of the row-normalized vectors).
  PR = N^2 / ||G||_F^2 ,  1.0 = all rows identical, N = mutually orthogonal.
"""
import argparse, numpy as np, jax, jax.numpy as jnp
from flax.serialization import msgpack_restore

from algorithms.feudal_mappo_jax.goal_dependence_probe import _checkpoint_path, _runner
from algorithms.feudal_mappo_jax.latent_locality_probe import collect_states
from algorithms.feudal_mappo_jax.manager import (
    GLOBAL_LATENTS,
    LOCAL_LATENTS,
    PRIVATE_LATENTS,
)
from algorithms.feudal_mappo_jax.mappo import build_manager
from algorithms.feudal_mappo_jax.worker import FeudalWorker

def pr(X):
    """X: (..., N, D) -> participation ratio of the N row directions."""
    U = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
    G = U @ np.swapaxes(U, -1, -2)
    N = X.shape[-2]
    return float(np.mean(N * N / (G ** 2).sum((-1, -2))))

def meanabscos(X):
    U = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
    G = np.abs(U @ np.swapaxes(U, -1, -2))
    N = X.shape[-2]; iu = np.triu_indices(N, 1)
    return float(np.mean(G[..., iu[0], iu[1]]))

ap = argparse.ArgumentParser()
ap.add_argument("--batches", default="mjx_12a_3o_trunc_1024")
ap.add_argument("--models", default="feudal_film_local,feudal_film")
ap.add_argument("--trial", default="0")
ap.add_argument("--n-envs", type=int, default=16)
ap.add_argument("--n-samples", type=int, default=16)
a = ap.parse_args()

for batch in a.batches.split(","):
  for model in a.models.split(","):
    path = _checkpoint_path(batch, model, a.trial)
    if path is None: print(f"{batch}/{model}: no ckpt"); continue
    runner = _runner(batch, model, a.trial, quiet=True)
    env = runner.env; N, obs_dim = env.n_agents, env.observation_dim
    cfg = runner.config
    tree = msgpack_restore(path.read_bytes())
    mp = jax.tree.map(jnp.asarray, {"params": tree["manager"]["params"]})
    wp = jax.tree.map(jnp.asarray, {"params": tree["actor"]["params"]})
    manager = build_manager(cfg, N)
    worker = FeudalWorker(action_dim=env.action_dim, goal_dim=cfg.goal_dim,
        hidden_dim=cfg.hidden_dim, discrete=getattr(env,"discrete",False),
        goal_embed_dim=cfg.goal_embed_dim, normalize_pooled_goal=cfg.normalize_pooled_goal,
        zero_goal=cfg.zero_goal, worker_fusion=cfg.worker_fusion)
    states, _ = collect_states(runner, manager, mp, worker, wp,
        jax.random.PRNGKey(0), a.n_envs, a.n_samples, 32, keep_env_states=False)
    # states: (S, N*obs_dim) flattened joint obs.
    #
    # NOTE this un-flattening assumes `global_state == obs.reshape(-1)`, i.e. an
    # env with NO `global_state` hook (every MJX env). On an env that has one
    # (SMAX: a 72-dim world state at 3m, against 3*obs_dim of concatenated
    # observations) the reshape is meaningless, so refuse rather than print
    # confident numbers about the wrong array.
    S = np.asarray(states)
    if S.shape[1] != N * obs_dim:
        print(
            f"{batch}/{model}: global_state is {S.shape[1]}-dim but "
            f"N*obs_dim is {N * obs_dim} — this env has its own global_state "
            "hook, which `collect_states` does not store observations alongside. "
            "Skipping (the probe would reshape the world state into fake obs)."
        )
        continue
    obs = S.reshape(S.shape[0], N, obs_dim)
    P = mp["params"]
    def dense(x, name): return x @ np.asarray(P[name]["kernel"]) + np.asarray(P[name]["bias"])
    print(f"\n=== {batch}/{model}/{a.trial}  latent={cfg.manager_latent} N={N} goal_dim={cfg.goal_dim}")
    print(f"    statistic: PR over the {N} agent rows (1.0 = identical, {N} = orthogonal) | mean|cos|")
    stages = []
    stages.append(("obs_i  (raw input)", obs))

    def with_global(core_in):
        """Append `z = f_percept(global_state)` for the `global` axis.

        The core reads `concat(s_flat, z)` under GLOBAL_LATENTS, so omitting it
        does not merely drop a stage — it makes `y`, and therefore every goal
        number below, a forward pass the checkpoint never computed. (This was
        missing for `local_global` before `local_global_private` was added.)
        """
        if cfg.manager_latent not in GLOBAL_LATENTS:
            return core_in
        zg = np.tanh(dense(np.tanh(dense(S, "f_percept_0")), "f_percept_1"))
        return np.concatenate([core_in, zg], axis=-1)

    if cfg.manager_latent in PRIVATE_LATENTS:
        # Shared encoder, PER-AGENT projections on both `s` and `g`. The point of
        # the variant is that the two einsums below are where row diversity is
        # MANUFACTURED rather than merely transmitted, so `s`/`g` should read near
        # the centralized branch's numbers despite the same ~4-of-N input.
        h0 = np.tanh(dense(obs, "f_enc_0")); stages.append(("f_enc_0 (tanh)", h0))
        h1 = np.tanh(dense(h0, "f_enc_1")); stages.append(("f_enc_1 (tanh)", h1))
        Ws = np.asarray(P["f_Mspace_agent_kernel"]); bs = np.asarray(P["f_Mspace_agent_bias"])
        s = np.einsum("...nh,nhg->...ng", h1, Ws) + bs
        stages.append(("s = W_i h_i", s))
        core_in = with_global(s.reshape(s.shape[0], N*cfg.goal_dim))
        y = np.tanh(dense(core_in, "core"))
        Wg = np.asarray(P["goal_head_agent_kernel"]); bg = np.asarray(P["goal_head_agent_bias"])
        g = np.einsum("...h,nhg->...ng", y, Wg) + bg
        stages.append(("g = W^g_i y", g))
    elif cfg.manager_latent in LOCAL_LATENTS:
        h0 = np.tanh(dense(obs, "f_enc_0")); stages.append(("f_enc_0 (tanh)", h0))
        h1 = np.tanh(dense(h0, "f_enc_1")); stages.append(("f_enc_1 (tanh)", h1))
        s  = dense(h1, "f_Mspace");         stages.append(("s = f_Mspace(h)", s))
        core_in = with_global(s.reshape(s.shape[0], N*cfg.goal_dim))
        y = np.tanh(dense(core_in, "core"))
        yi = dense(y, "f_gpre").reshape(-1, N, cfg.goal_dim); stages.append(("y_i = f_gpre(y)", yi))
        tyi = np.tanh(yi);                  stages.append(("tanh(y_i)", tyi))
        g = dense(tyi, "f_goalhead");       stages.append(("g = f_goalhead(.)", g))
    else:
        z0 = np.tanh(dense(S, "f_percept_0")); z1 = np.tanh(dense(z0, "f_percept_1"))
        s = dense(z1, "f_Mspace").reshape(-1, N, cfg.goal_dim); stages.append(("s = f_Mspace(z)", s))
        core_in = s.reshape(s.shape[0], N*cfg.goal_dim)
        y = np.tanh(dense(core_in, "core"))
        g = dense(y, "goal_head").reshape(-1, N, cfg.goal_dim); stages.append(("g = goal_head(y)", g))
    for name, X in stages:
        print(f"      {name:22s} PR = {pr(X):6.2f} / {N}    mean|cos| = {meanabscos(X):.3f}")
    # the quantity r^I actually scores: temporal DIFFERENCES of s
    if S.shape[0] > 1:
        ds = s[1:] - s[:-1]
        print(f"      {'s_t - s_{t-1} (r^I arg)':22s} PR = {pr(ds):6.2f} / {N}    mean|cos| = {meanabscos(ds):.3f}")
