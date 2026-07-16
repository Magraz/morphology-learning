"""THE FALSIFICATION: is the synchronous difference reward biased under asynchrony?

The whole contribution rests on one empirical claim:

    Under asynchronous commitments, the difference reward computed with the
    synchronous assumption is biased relative to the true (oracle) difference
    reward, and the bias grows with how asynchronous the setting is.

This module measures that and nothing else — **no training**. If the gap is small,
the contribution does not exist and we stop, cheaply.

Method. At engaged mid-episode states we compute two exact quantities from the
*same* physical state, with the *same* rollout key (CRN), over the same horizon:

- `D_oracle` — forked counterfactuals under the **true** staggered commitments.
- `D_sync`   — forked counterfactuals under the synchronous estimator's **belief**
  (`oracle.aligned_belief`: every agent shares one decision epoch), rolled in a
  wrapper whose durations are fixed at the nominal `L` so re-decisions stay aligned.

Both branches share the base env, so the physics is identical and the entire gap is
attributable to the commitment structure.

Conditions sweep the duration spread at fixed mean `L`, which is the x-axis:

    sync (control) : d=L, phases aligned -> `aligned_belief` is the identity,
                     so D_sync == D_oracle EXACTLY. Non-zero bias here means the
                     harness is broken, not that the theory is right.
    stagger-only   : d=L, phases offset
    spread 8-12 / 5-15 / 2-18 : increasingly variable durations

Run:
    MUJOCO_GL=egl uv run python -m algorithms.difference_rewards.bias_study
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.difference_rewards.oracle import aligned_belief, difference_rewards
from environments.mjx_suite.macro_wrapper import AsyncMacroMJX
from environments.mjx_suite.multi_box_push_mjx import MultiBoxPushMJX

CONTACT = 0  # skill index used to warm up into an engaged state


_JIT_CACHE: dict = {}


def _jitted(env: AsyncMacroMJX):
    """Cache jitted reset/step per env.

    `jax.jit(env.step)` on a bound method builds a fresh wrapper each call, so
    jit's cache misses and the whole rollout recompiles for every sampled state.
    """
    if id(env) not in _JIT_CACHE:
        _JIT_CACHE[id(env)] = (jax.jit(env.reset), jax.jit(env.step))
    return _JIT_CACHE[id(env)]


def engaged_state(env: AsyncMacroMJX, key: jax.Array, warmup: int):
    """A mid-episode state where agents are on the boxes and reward is live.

    Measuring at reset is useless: nothing has touched a box, every D_i is 0, and
    an attribution test there passes vacuously. We warm up under `contact` (the
    skill that reliably engages the coupling) to reach states with real credit.
    """
    reset, step = _jitted(env)
    _, state = reset(key)
    proposal = jnp.full((env.n_agents,), CONTACT)
    for _ in range(warmup):
        _, state, _, term, trunc, _ = step(state, proposal)
        if bool(term) or bool(trunc):
            return None
    return state


def compare_at(env_true, env_belief, state, rkey, horizon):
    """(D_oracle, D_sync) from one physical state under both commitment structures."""
    d_oracle, _ = difference_rewards(env_true, state, rkey, horizon)
    d_sync, _ = difference_rewards(env_belief, aligned_belief(state), rkey, horizon)
    return np.asarray(d_oracle), np.asarray(d_sync)


def summarize(per_state: list, noise_floor: float) -> dict:
    """Bias metrics over a list of per-state (D_oracle, D_sync) pairs.

    Metrics are computed **per state and then aggregated**, not pooled. Credit
    scale varies enormously across states (a state where a +100 delivery is in
    play has |D| ~ 100; a state without one has |D| ~ 0.3), so a pooled ratio is
    dominated by whichever states happened to have small denominators — that is
    what produced a meaningless `norm_bias = 29.9` in the first run. The
    per-state median is scale-free and robust.

    Only agents whose oracle credit clears the numerical noise floor are scored:
    at |D| ~ 1e-3 the value is f32 chaos, and its sign is a coin flip.
    """
    biases, corrs, signs, scales, n_used = [], [], [], [], 0
    for d_o, d_s in per_state:
        signal = np.abs(d_o) > noise_floor
        o, s = d_o[signal], d_s[signal]
        if o.size < 2:
            continue
        n_used += o.size
        biases.append(np.abs(o - s).mean() / (np.abs(o).mean() + 1e-12))
        signs.append((np.sign(o) == np.sign(s)).mean())
        scales.append(np.abs(o).mean())
        if o.std() > 1e-9 and s.std() > 1e-9:
            corrs.append(np.corrcoef(o, s)[0, 1])
    if not biases:
        return {"n": 0, "pearson": np.nan, "sign_agree": np.nan,
                "norm_bias": np.nan, "mean_abs_oracle": np.nan}
    return {
        "n": n_used,
        "pearson": float(np.mean(corrs)) if corrs else np.nan,
        "sign_agree": float(np.mean(signs)),
        "norm_bias": float(np.median(biases)),
        "mean_abs_oracle": float(np.median(scales)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-agents", type=int, default=9)
    p.add_argument("--n-objects", type=int, default=3)
    p.add_argument("--macro-len", type=int, default=10, help="nominal duration L")
    p.add_argument("--horizon", type=int, default=60, help="counterfactual horizon H")
    p.add_argument("--n-states", type=int, default=12)
    p.add_argument("--warmup", type=int, default=160)
    p.add_argument("--noise-floor", type=float, default=5e-3)
    args = p.parse_args()

    L = args.macro_len
    base = MultiBoxPushMJX(n_agents=args.n_agents, n_objects=args.n_objects)

    # The estimator's belief always rolls with fixed-L, phase-aligned commitments.
    env_belief = AsyncMacroMJX(base, d_min=L, d_max=L, stagger=False)

    conditions = [
        ("sync (control)", AsyncMacroMJX(base, d_min=L, d_max=L, stagger=False)),
        ("stagger only", AsyncMacroMJX(base, d_min=L, d_max=L, stagger=True)),
        ("spread 8-12", AsyncMacroMJX(base, d_min=8, d_max=12, stagger=True)),
        ("spread 5-15", AsyncMacroMJX(base, d_min=5, d_max=15, stagger=True)),
        ("spread 2-18", AsyncMacroMJX(base, d_min=2, d_max=18, stagger=True)),
    ]

    print(f"agents={args.n_agents} objects={args.n_objects} L={L} H={args.horizon} "
          f"states={args.n_states} noise_floor={args.noise_floor}")
    print(f"{'condition':<16} {'n':>4} {'pearson':>8} {'sign%':>7} {'norm_bias':>10} "
          f"{'mean|D_o|':>10}")
    print("-" * 60)

    results = {}
    for name, env_true in conditions:
        per_state = []
        for i in range(args.n_states):
            state = engaged_state(env_true, jax.random.PRNGKey(1000 + i), args.warmup)
            if state is None:
                continue
            per_state.append(
                compare_at(
                    env_true, env_belief, state,
                    jax.random.PRNGKey(7000 + i), args.horizon,
                )
            )
        m = summarize(per_state, args.noise_floor)
        results[name] = m
        print(f"{name:<16} {m['n']:>4} {m['pearson']:>8.4f} "
              f"{100 * m['sign_agree']:>6.1f}% {m['norm_bias']:>10.4f} "
              f"{m['mean_abs_oracle']:>10.3f}")

    print("-" * 60)
    control = results["sync (control)"]
    ok = control["norm_bias"] < 1e-6
    print(f"HARNESS CHECK: control norm_bias = {control['norm_bias']:.2e} "
          f"-> {'VALID (D_sync == D_oracle under true synchrony)' if ok else 'BROKEN'}")
    if not ok:
        print("  Control must be exactly 0: aligned_belief is the identity when every")
        print("  agent already shares a phase. A non-zero value means the harness is")
        print("  wrong, and no async result from this run can be trusted.")
        return

    worst = results["spread 2-18"]
    print(f"\nDECISION: async(2-18) norm_bias = {worst['norm_bias']:.3f}, "
          f"pearson = {worst['pearson']:.3f}, sign agreement = "
          f"{100 * worst['sign_agree']:.1f}%")


if __name__ == "__main__":
    main()
