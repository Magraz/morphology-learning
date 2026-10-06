This probe tests whether a small team-advantage standard deviation amplifies
DPP into a large manager policy gradient. It collects stochastic rollouts from
a saved checkpoint using the training collector, with its normal resets,
window rewards, and timeout bootstraps. The checkpoint parameters stay frozen.

Run the sparse six-agent case first:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_normalization_probe \
  --batch mjx_6a_4o_1024_gs_sparse \
  --model simplified_feudal_tanh_relative_input_dpp \
  --trials 0,1,2,3,4 --rollouts 3
```

Then compare with the dense two-agent case:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_normalization_probe \
  --batch mjx_2a_4o_1122_1024_gs \
  --model simplified_feudal_tanh_relative_input_dpp \
  --trials 0,1,2,3,4 --rollouts 3
```

Use `--model simplified_feudal_tanh_local_input_dpp` for the local-input arm.
Defaults are 32 environments, the configured training rollout length,
coefficients `0,0.1,1`, and std floors `0,0.0001,0.001,0.01`. The zero coefficient
and zero floor are always included as controls. Each trial and rollout gets a
reproducible PRNG key derived from `--seed` (default 1000).

Outputs go into a new timestamped directory under
`plotting/feudal_goal_analysis/`. `--out PATH` chooses a new directory explicitly.

- `summary.csv`: one row per trial, rollout, coefficient, and floor. Includes
  denominator quantiles, raw DPP and advantage magnitudes, normalized bonus RMS,
  sign flips relative to team credit, gradient norms, and gradient cosines.
- `streams.csv`: the advantage metrics for each individual environment and
  agent, so a few small-denominator streams cannot disappear in a batch average.
- `report.json`: scenario summaries and provenance, including resolved manager
  configuration, checkpoint path/hash, backend, and collection length.
- `trial_*_rollout_*.npz`: frozen inputs, GAE advantages, raw DPP predictions, and
  actor parameters. Loading requires no simulator or original checkpoint.

Replay exactly the same data with different settings:

```sh
uv run python -m algorithms.simplified_feudal_mappo_jax.dpp_normalization_probe \
  --snapshot PATH/trial_0_rollout_0.npz \
  --coefs 0,0.05,0.1,0.25,1 --std-floors 0,0.001,0.01
```

Read the results in this order:

1. Compare raw bonus size with `team_std_*` and the per-stream denominator.
2. Check `bonus_normalized_rms`, `bonus_to_team_rms`, and `sign_flip_frac`.
   The sign comparison is after centering over time, as in PPO. A nonnegative
   raw bonus can therefore reduce some decisions' normalized advantages.
3. Check `bonus_to_team_grad_norm`: values above 1 mean the DPP-induced gradient
   is larger than the team policy gradient at the same floor. The corresponding
   RMS ratio measures advantage size; it does not establish gradient dominance.
4. Check `policy_grad_cos_team` and `total_grad_cos_team`. Negative values mean
   the corrected gradient points against the team-only gradient. The total
   includes the configured entropy coefficient; `weighted_entropy_grad_norm`
   shows its contribution separately. Metrics ending in `unfloored_team` keep
   the original team normalization as the reference when a floor also changes
   the team gradient.
5. Compare floor=0 with nonzero floors at the same coefficient. A small raw
   bonus, large normalized/gradient effect, and substantial reduction after
   flooring support the amplification hypothesis.

The floor scales both team credit and DPP within each env. With one env it
cannot change their relative gradient direction or ratio; it changes their
absolute size and strength relative to entropy. With multiple envs it can also
change their relative contributions to the batch gradient. A floor reducing
gradient magnitude does not establish that the DPP signal is useful.

Floors are absolute values in raw advantage/discounted-return units, so they
are not invariant to reward scaling. The probe mirrors
`max(team_std_ddof1, floor) + 1e-8`; floor=0 reproduces current PPO normalization.
Ratios/cosines with zero reference norms are null, and nonfinite values are
reported explicitly rather than silently included in quantiles.

Gradients are measured on the full-batch PPO surrogate at the frozen checkpoint,
before clipping or Adam. `full_batch_clip_scale` describes how global norm
clipping would scale that full-batch gradient; training clips separate minibatch
gradients. This experiment establishes effects on the immediate objective, not
whether a floor improves multi-epoch PPO updates or long-run learning. It leaves
the training implementation unchanged.

For a quick collection smoke test, use `--trials 0 --n-envs 1 --rollout-steps 64`.
Short rollouts are useful for testing the command but can miss reward events and
are not a substitute for the configured full rollout. A GPU is recommended for
the full MJX collection; snapshots can be analyzed on CPU.
