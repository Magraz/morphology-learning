"""Is the joint state recoverable from N egocentric observations?

CLAUDE.md asserts, but has never measured, that the feudal manager's input is
crippled: it reads ``obs.reshape(n_envs, -1)`` — N egocentric views with **no
shared frame** — so before it can assign a division of labour it must first
learn to *localize* every agent from `goal_distance` and the lidar wall returns.
That claim is the entire case for giving the MJX envs a real ``global_state``
hook (the trainers already switch onto one when it exists; see
``trainer.global_state_dim`` and the SMAX adapter). It is also cheap to falsify,
and it should be falsified before anyone spends two full-length runs on it.

This module is that falsification test. It fits a **decoding probe** from each
candidate centralized input to the true world coordinates of every agent and
box, and reports the held-out error in world units:

    concat   (A * OBS_DIM = 640 dims)  the manager's input TODAY
    proposed (~93 dims)                the candidate compact global state
    mean     (0 dims)                  predict the training mean — the no-information floor

Read it as follows.

* **concat error near `mean` error** ⇒ positions are genuinely not recoverable.
  The frame problem is real, the manager is spending capacity on localization it
  cannot even win, and the ``global_state`` hook is justified.
* **concat error near `proposed` error** ⇒ the information is right there and
  linearly/MLP-decodable. The premise is falsified, the hook buys nothing but
  width, and the case for changing the manager rests entirely on weight sharing
  (the permutation-equivariant encoder), not on the frame.

Two design choices make the answer meaningful rather than rhetorical:

* **The probe IS the critic.** It reuses ``network.MAPPOCritic`` at the exact
  width the stack builds (``2 * hidden_dim`` = 336), so a failure to decode is a
  statement about the function class that actually has to do this job, not about
  some weaker model chosen to make a point. A linear probe runs alongside it to
  separate "present but entangled" from "absent".
* **States come from a TRAINED policy.** The state distribution a converged
  policy visits is what the manager is actually trained on; a random policy
  wanders into configurations neither network ever sees.

``candidate_global_state`` below is deliberately *not* in the env yet. It is the
implementation that would move into ``MultiBoxPushMJX.global_state`` if this
probe justifies the change — kept here so the measurement never modifies the
thing being measured.

Run:
    MUJOCO_GL=egl uv run python -m algorithms.feudal_mappo_jax.global_state_probe \
        --results mjx_16a_4o_trunc_512 --model mlp --trial 0
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import from_bytes

from algorithms.mappo_jax.network import MAPPOActor, MAPPOCritic, sample_action
from environments.mjx_suite.multi_box_push_mjx import MultiBoxPushMJX

_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "results"


# ---------------------------------------------------------------------------
# The candidate global state (the thing option A would add to the env)
# ---------------------------------------------------------------------------


def candidate_global_state(env: MultiBoxPushMJX, state) -> jnp.ndarray:
    """``(global_state_dim,)`` compact world state for one (unbatched) EnvState.

    Everything here is exact simulator state in a **shared world frame**, which
    is precisely what N egocentric observations do not carry. Normalized with
    the env's own constants (``world_width``/``world_height``/``velocity_norm``)
    so no block enters a Tanh layer at a different scale — the defect measured
    for ``normalize_pooled_goal``, where a 5x-scaled goal block took 91.6% of the
    first layer's preactivation variance.

    Positions are **centered** before normalizing (roughly zero-mean input), and
    box yaw is emitted as ``(cos, sin)`` — the hinge DOF accumulates without
    wrapping, so the raw angle is unbounded and jumps by 2*pi for no physical
    reason.
    """
    d = state.data
    agent_pos = env._agent_pos(d)  # (A, 2)
    agent_vel = env._agent_vel(d)  # (A, 2)
    box_pos, box_yaw = env._box_pose(d)  # (O, 2), (O,)
    touch = env._touch_matrix(agent_pos, box_pos, box_yaw)  # (A, O) bool

    center = jnp.array([env.world_center_x, env.world_center_y], dtype=jnp.float32)
    extent = jnp.array([env.world_width, env.world_height], dtype=jnp.float32)

    agents = jnp.concatenate(
        [(agent_pos - center) / extent, agent_vel / env.velocity_norm], axis=-1
    )  # (A, 4)

    boxes = jnp.concatenate(
        [
            (box_pos - center) / extent,
            jnp.cos(box_yaw)[:, None],
            jnp.sin(box_yaw)[:, None],
            state.delivered[:, None].astype(jnp.float32),
            # How crewed each box currently is, and how crewed it needs to be.
            # The second is constant under `coupling_def: even` and informative
            # only under an explicit unequal list (e.g. the partition arm).
            (touch.sum(axis=0) / env.n_agents)[:, None],
            (jnp.asarray(env._coupling, jnp.float32) / env.n_agents)[:, None],
        ],
        axis=-1,
    )  # (O, 7)

    return jnp.concatenate([agents.ravel(), boxes.ravel()])


def candidate_global_state_dim(env: MultiBoxPushMJX) -> int:
    return env.n_agents * 4 + env.n_objects * 7


# ---------------------------------------------------------------------------
# Rollout collection under a trained policy
# ---------------------------------------------------------------------------


def load_actor_params(
    path: Path, obs_dim: int, action_dim: int, hidden_dim: int, n_agents: int
):
    """Read {"actor", "critic"} out of a `models_finished.msgpack`.

    `from_bytes` needs an exactly-shaped target tree, so both halves are rebuilt
    even though only the actor is used — and the critic's input is the global
    state (``n_agents * obs_dim``), not one agent's obs.
    """
    actor = MAPPOActor(action_dim=action_dim, hidden_dim=hidden_dim, discrete=False)
    critic = MAPPOCritic(hidden_dim=2 * hidden_dim, n_outputs=1)
    rng = jax.random.PRNGKey(0)
    target = {
        "actor": actor.init(rng, jnp.zeros(obs_dim)),
        "critic": critic.init(rng, jnp.zeros(n_agents * obs_dim)),
    }
    with open(path, "rb") as f:
        loaded = from_bytes(target, f.read())
    return actor, loaded["actor"]


def collect_states(
    env: MultiBoxPushMJX,
    actor,
    actor_params,
    key: jax.Array,
    n_envs: int,
    n_samples: int,
    stride: int,
):
    """Roll the trained policy and snapshot every `stride`-th step.

    Returns numpy arrays of shape ``(n_samples * n_envs, ...)``:
    ``obs``, ``global_state``, ``agent_pos``, ``box_pos``.

    The inner scan advances `stride` steps and discards them; only the outer
    scan emits, so memory is set by `n_samples`, not by total steps. Snapshots
    are taken **before** the step, so obs and state describe the same instant.
    Done envs are restarted the way ``trainer._env_step`` does (MJX has no
    auto-reset), which keeps the state distribution on-policy rather than
    letting finished episodes freeze.
    """
    n_agents, obs_dim = env.n_agents, env.observation_dim
    action_dim = env.action_dim
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)
    v_gs = jax.vmap(lambda s: candidate_global_state(env, s))
    v_apos = jax.vmap(lambda s: env._agent_pos(s.data))
    v_bpos = jax.vmap(lambda s: env._box_pose(s.data)[0])

    def _act(obs, rng):
        flat = obs.reshape(-1, obs_dim)
        action, _ = sample_action(
            rng, actor.apply, actor_params, flat, discrete=False, deterministic=True
        )
        return action.reshape(obs.shape[0], n_agents, action_dim)

    def _advance(carry, _):
        obs, state, rng = carry
        rng, a_rng, r_rng = jax.random.split(rng, 3)
        next_obs, next_state, _, term, trunc, _ = v_step(state, _act(obs, a_rng))
        # MJX has no auto-reset; restart finished envs exactly as
        # `trainer._env_step` does, so the state distribution stays on-policy
        # instead of freezing on whatever ended the episode.
        done = term | trunc
        reset_obs, reset_state = v_reset(jax.random.split(r_rng, obs.shape[0]))
        next_state = jax.tree.map(
            lambda r, n: jnp.where(done.reshape((-1,) + (1,) * (n.ndim - 1)), r, n),
            reset_state,
            next_state,
        )
        next_obs = jnp.where(done[:, None, None], reset_obs, next_obs)
        return (next_obs, next_state, rng), None

    def _sample(carry, _):
        (obs, state, rng), _ = jax.lax.scan(_advance, carry, None, length=stride)
        rec = (obs, v_gs(state), v_apos(state), v_bpos(state))
        return (obs, state, rng), rec

    key, reset_key = jax.random.split(key)
    obs, state = v_reset(jax.random.split(reset_key, n_envs))
    _, recs = jax.lax.scan(_sample, (obs, state, key), None, length=n_samples)

    obs, gs, apos, bpos = jax.tree.map(
        lambda x: np.asarray(x).reshape((-1,) + x.shape[2:]), recs
    )
    return obs, gs, apos, bpos


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _standardize(x_train, x_test):
    mu = x_train.mean(0, keepdims=True)
    sd = x_train.std(0, keepdims=True) + 1e-6
    return (x_train - mu) / sd, (x_test - mu) / sd


def linear_probe(x_train, y_train, x_test, ridge: float = 1e-3):
    """Closed-form ridge regression with a bias column."""
    xa = jnp.concatenate([x_train, jnp.ones((x_train.shape[0], 1))], axis=1)
    xb = jnp.concatenate([x_test, jnp.ones((x_test.shape[0], 1))], axis=1)
    d = xa.shape[1]
    reg = ridge * jnp.eye(d).at[d - 1, d - 1].set(0.0)  # never penalize the bias
    w = jnp.linalg.solve(xa.T @ xa + reg, xa.T @ y_train)
    return np.asarray(xb @ w)


def mlp_probe(
    x_train,
    y_train,
    x_test,
    hidden_dim: int,
    steps: int = 8000,
    batch: int = 512,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    eval_every: int = 200,
    seed: int = 0,
):
    """Fit `MAPPOCritic` — literally the stack's own critic architecture — as a
    regressor from the candidate state to the target coordinates.

    Using the real critic module is the point: a decoding failure then says
    something about the network that actually has to do this job in training,
    not about a weaker model picked to make the answer come out a certain way.

    **Early-stopped on a held-out validation split**, which is what keeps the
    comparison honest rather than merely favourable. The 640-dim concat feeds
    ~215k probe parameters against a few thousand rows, so a fixed step budget
    would let it overfit and report a test error that understates what is
    actually decodable — i.e. the run would "confirm" the frame problem by
    construction. Selecting the best-validation params instead gives the concat
    the strongest reading its own information supports.
    """
    batch = min(batch, x_train.shape[0])
    n_val = max(1, int(val_frac * x_train.shape[0]))
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_fit, y_fit = x_train[n_val:], y_train[n_val:]

    net = MAPPOCritic(hidden_dim=hidden_dim, n_outputs=y_train.shape[1])
    rng = jax.random.PRNGKey(seed)
    params = net.init(rng, jnp.zeros(x_train.shape[1]))
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    def loss_fn(p, xb, yb):
        return jnp.mean((net.apply(p, xb) - yb) ** 2)

    @jax.jit
    def _update(params, opt_state, rng):
        rng, sub = jax.random.split(rng)
        idx = jax.random.randint(sub, (batch,), 0, x_fit.shape[0])
        loss, grads = jax.value_and_grad(loss_fn)(params, x_fit[idx], y_fit[idx])
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, rng, loss

    val_loss = jax.jit(lambda p: loss_fn(p, x_val, y_val))

    best_params, best_val = params, float("inf")
    for step in range(steps):
        params, opt_state, rng, _ = _update(params, opt_state, rng)
        if (step + 1) % eval_every == 0:
            v = float(val_loss(params))
            if v < best_val:
                best_val, best_params = v, params

    return np.asarray(net.apply(best_params, x_test))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def position_error(pred, true, world: float, n_entities: int):
    """Decoding error per entity, in **world units**.

    Returns ``(mean_dist, median_dist, r2, rmse_x, rmse_y)``.

    `pred`/`true` are flattened ``(n, n_entities * 2)`` normalized coordinates;
    the normalizer is the world extent, so scaling by it recovers real distance —
    the only form in which the number is interpretable (against the 47-wide arena
    and the 15.67 sensor radius at 16a/4o).

    **The per-axis split is the load-bearing part of this measurement.** The
    observation's `goal_distance` channel is a signed distance to the goal band
    along the goal axis (y here), normalized by `world_height` — i.e. an exact
    affine function of the agent's own y coordinate. So y is expected to be
    decodable essentially perfectly from the concat, by construction, and a
    Euclidean-only report would blend that freebie with the x coordinate, which
    is the axis nothing in the egocentric vector anchors. Read the two columns
    separately or the headline number understates the frame problem.
    """
    p3 = pred.reshape(-1, n_entities, 2) * world
    t3 = true.reshape(-1, n_entities, 2) * world
    err = p3 - t3
    dist = np.linalg.norm(err, axis=-1)  # (n, n_entities)
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean(0, keepdims=True)) ** 2).sum())
    rmse_x = float(np.sqrt((err[..., 0] ** 2).mean()))
    rmse_y = float(np.sqrt((err[..., 1] ** 2).mean()))
    return float(dist.mean()), float(np.median(dist)), 1.0 - ss_res / ss_tot, rmse_x, rmse_y


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="mjx_16a_4o_trunc_512")
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--trial", default="0")
    ap.add_argument("--n-agents", type=int, default=16)
    ap.add_argument("--n-objects", type=int, default=4)
    ap.add_argument("--variant", default="trunc")
    ap.add_argument("--coupling-def", default="even")
    ap.add_argument("--hidden-dim", type=int, default=168, help="conf/model/mlp.yaml")
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=192, help="snapshots per env")
    ap.add_argument("--stride", type=int, default=8, help="steps between snapshots")
    ap.add_argument("--probe-steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="algorithms/feudal_mappo_jax/global_state_probe")
    ap.add_argument("--from-cache", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    cache = out.with_suffix(".data.pkl")

    env = MultiBoxPushMJX(
        n_agents=args.n_agents,
        n_objects=args.n_objects,
        reward_mode="dense",
        variant=None if args.variant in ("", "none") else args.variant,
        coupling_def=args.coupling_def,
    )
    concat_dim = env.n_agents * env.observation_dim
    gs_dim = candidate_global_state_dim(env)
    print(
        f"env {args.n_agents}a/{args.n_objects}o variant={args.variant} | "
        f"world {env.world_width}x{env.world_height} | "
        f"concat {concat_dim} vs proposed {gs_dim} dims"
    )

    if args.from_cache and cache.exists():
        with open(cache, "rb") as f:
            obs, gs, apos, bpos = pickle.load(f)
        print(f"loaded {obs.shape[0]} states from {cache}")
    else:
        ckpt = (
            _RESULTS_ROOT
            / args.results
            / args.model
            / str(args.trial)
            / "models"
            / "models_finished.msgpack"
        )
        if not ckpt.exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        actor, actor_params = load_actor_params(
            ckpt,
            env.observation_dim,
            env.action_dim,
            args.hidden_dim,
            env.n_agents,
        )
        print(f"policy: {ckpt}")
        t0 = time.time()
        obs, gs, apos, bpos = collect_states(
            env,
            actor,
            actor_params,
            jax.random.PRNGKey(args.seed),
            args.n_envs,
            args.n_samples,
            args.stride,
        )
        print(
            f"collected {obs.shape[0]} states "
            f"({args.n_samples * args.stride} steps x {args.n_envs} envs) "
            f"in {time.time() - t0:.1f}s"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump((obs, gs, apos, bpos), f)

    # --- features / targets ------------------------------------------------
    n = obs.shape[0]
    center = np.array([env.world_center_x, env.world_center_y], np.float32)
    extent = np.array([env.world_width, env.world_height], np.float32)

    x_concat = jnp.asarray(obs.reshape(n, -1))
    x_gs = jnp.asarray(gs)
    y_agent = jnp.asarray(((apos - center) / extent).reshape(n, -1))
    y_box = jnp.asarray(((bpos - center) / extent).reshape(n, -1))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, n // 5)
    te, tr = perm[:n_test], perm[n_test:]
    print(f"split: {len(tr)} train / {len(te)} test")

    features = {"concat (640)": x_concat, "proposed (92)": x_gs}
    targets = {
        "agent xy": (y_agent, env.n_agents),
        "box xy": (y_box, env.n_objects),
    }

    results = {}
    world = float(env.world_width)
    for tname, (y, n_ent) in targets.items():
        y_tr, y_te = y[tr], y[te]
        base = np.tile(np.asarray(y_tr.mean(0)), (len(te), 1))
        results[(tname, "mean baseline", "-")] = position_error(
            base, np.asarray(y_te), world, n_ent
        )
        for fname, x in features.items():
            x_tr, x_te = _standardize(x[tr], x[te])
            results[(tname, fname, "linear")] = position_error(
                linear_probe(x_tr, y_tr, x_te), np.asarray(y_te), world, n_ent
            )
            results[(tname, fname, "mlp")] = position_error(
                mlp_probe(
                    x_tr,
                    y_tr,
                    x_te,
                    hidden_dim=2 * args.hidden_dim,
                    steps=args.probe_steps,
                    seed=args.seed,
                ),
                np.asarray(y_te),
                world,
                n_ent,
            )

    # --- report ------------------------------------------------------------
    print(
        f"\nHeld-out decoding error, world units "
        f"(arena {world:.0f} wide, sensor radius {env.sector_sensor_radius:.2f})"
    )
    print(f"{'target':<10} {'input':<15} {'probe':<7} {'mean err':>9} "
          f"{'median':>8} {'R^2':>7} {'rmse x':>8} {'rmse y':>8}")
    print("-" * 72)
    for (tname, fname, pname), (mean_e, med_e, r2, ex, ey) in results.items():
        print(
            f"{tname:<10} {fname:<15} {pname:<7} {mean_e:>9.2f} {med_e:>8.2f} "
            f"{r2:>7.3f} {ex:>8.2f} {ey:>8.2f}"
        )

    a_mean = results[("agent xy", "mean baseline", "-")][0]
    a_concat = min(
        results[("agent xy", "concat (640)", "mlp")][0],
        results[("agent xy", "concat (640)", "linear")][0],
    )
    a_gs = min(
        results[("agent xy", "proposed (92)", "mlp")][0],
        results[("agent xy", "proposed (92)", "linear")][0],
    )
    recovered = (a_mean - a_concat) / max(a_mean - a_gs, 1e-9)
    ax_mean = results[("agent xy", "mean baseline", "-")][3]
    ax_concat = min(
        results[("agent xy", "concat (640)", "mlp")][3],
        results[("agent xy", "concat (640)", "linear")][3],
    )
    ay_concat = min(
        results[("agent xy", "concat (640)", "mlp")][4],
        results[("agent xy", "concat (640)", "linear")][4],
    )
    print(
        f"\nper axis from the concat: x {ax_concat:.2f} (vs {ax_mean:.2f} knowing "
        f"nothing), y {ay_concat:.2f}. `goal_distance` hands every agent its own y "
        f"exactly, so y is the freebie and x is the real question."
    )
    print(
        f"\nagent positions: {a_mean:.2f} (no info) -> {a_concat:.2f} (concat) "
        f"-> {a_gs:.2f} (proposed).\n"
        f"the concat closes {100 * recovered:.1f}% of the gap between knowing "
        f"nothing and knowing the true state."
    )
    print(
        "read: near 100% => positions ARE recoverable, the frame problem is not "
        "real and option A buys only width.\n"
        "      near 0%   => they are not, and the global_state hook is justified."
    )

    with open(out.with_suffix(".results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"\nwrote {out.with_suffix('.results.pkl')}")


if __name__ == "__main__":
    main()
