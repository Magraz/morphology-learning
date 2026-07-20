---
name: run-morphology-learning
description: Build, smoke-test, and screenshot the morphology-learning multi-agent RL experiment manager. Use when asked to run, launch, train, smoke-test, render, or screenshot this project, its box2d/MJX environments, or MAPPO/MAPPO-JAX/DCG training.
---

# Run morphology-learning

This repo is a **multi-agent RL experiment manager**, not a GUI or web app. The
"app" is the Hydra training entry point [train.py](train.py) (`uv run python
train.py <group>=<choice> ...`), which composes a run from `algorithm × env ×
model × seeds` and dispatches to a trainer. The one **visual** surface is the
MJX environment renderer, which writes a `.png`/`.mp4` headlessly.

The driver — [.claude/skills/run-morphology-learning/driver.sh](.claude/skills/run-morphology-learning/driver.sh)
— wraps the three paths worth smoke-testing, each with the exact overrides that
make it finish in seconds instead of hours. **Paths below are relative to the
repo root.** Everything runs through `uv run` (deps are locked in `uv.lock`);
there is no separate install step on an already-provisioned machine.

## Run (agent path) — the driver

```bash
# from repo root
.claude/skills/run-morphology-learning/driver.sh render      # MJX env -> screenshot .png + rollout .mp4
.claude/skills/run-morphology-learning/driver.sh smoke       # short box2d MAPPO (torch) training run
.claude/skills/run-morphology-learning/driver.sh smoke-jax   # short MJX MAPPO-JAX (jitted, GPU) training run
.claude/skills/run-morphology-learning/driver.sh all         # render + both smokes
```

- **`render`** is the visual proof. It writes `.driver-out/mjx_frame.png` (a
  mid-rollout frame with the agent-0 sensor overlay: 8+8 density sectors, lidar,
  `nearest_box_vec` arrow, green `goal_distance`/DROP ZONE) and
  `.driver-out/mjx_rollout.mp4`. **Open the PNG and look at it** — a real scene
  shows red agents, colored boxes with `n/N` touch counters, and the green goal
  band. Override the output dir with `OUT=/some/dir`.
- **`smoke`** exercises the box2d + torch MAPPO loop (AsyncVectorEnv over a
  `fork` pool, CPU). Finishes in ~4s; prints `Training completed!` and a time
  breakdown.
- **`smoke-jax`** exercises the fully-jitted MJX MAPPO-JAX stack. It JIT-compiles
  first (slow first call), then trains; ~23s total on this box. Uses the GPU if
  one is visible (`Using JAX default device: cuda:0`), else CPU — `device=cpu`
  in the config only affects the torch stacks, not JAX device selection.

Both smokes write to `experiments/results/<env>/<model>/smoke_test/` (a
`training_stats_finished.pkl` + `models_finished.pth`). Delete that dir when done.

## Run (real training)

Drop the tiny overrides and pick a real env/model/algorithm. The migrated,
runnable combos live in `conf/{env,model,algorithm}/`:

```bash
# box2d MAPPO (the default)
uv run python train.py env=multi_box_push_9a_3o model=mlp_shared algorithm=mappo trial_id=0
# MJX MAPPO-JAX (jitted; pin n_envs — it is vmapped on one device, not core-bound)
MUJOCO_GL=egl uv run python train.py algorithm=mappo_jax env=multi_box_push_mjx_20a_5o model=mlp trial_id=0 env.n_envs=32
# a local parallel sweep (joblib launcher; n_jobs concurrent jobs, n_envs auto-shrinks)
uv run python train.py -m model=mlp_shared,mlp trial_id=0 n_jobs=2 hydra/launcher=joblib_auto
```

`train.sh BATCH ALGORITHM ENVIRONMENT TRIAL_ID EXP_NAME` is a thin positional
wrapper over the same command (`$ENVIRONMENT` is vestigial). See
[CLAUDE.md](CLAUDE.md) for the full config model and every migrated batch.

## Prerequisites

On this already-provisioned machine, nothing beyond `uv` was needed — `uv run`
resolves the locked env on first call. The suite needs box2d (`gymnasium[box2d]`,
via `uv`) and, for the MJX/JAX paths, a MuJoCo GL context: the driver sets
`MUJOCO_GL=egl` (headless EGL) and `SDL_VIDEODRIVER=dummy` (pygame overlay). A
CUDA GPU is optional but auto-used by JAX when present (`jax[cuda13]`).

## Gotchas

- **`smoke-jax` ignores `device=cpu`.** That flag routes the *torch* stacks;
  JAX picks its own default device and will print `Using JAX default device:
  cuda:0` on a GPU box. There is no CPU override for the JAX path here.
- **MJX `n_envs` is not core-bound.** The central autoscale in
  [conf/config.yaml](conf/config.yaml) (`env.n_envs = usable_cores // n_jobs`)
  targets box2d *subprocess* envs. MJX is vmapped on one device, so pin
  `env.n_envs=<N>` on the CLI (the driver uses 8 for the smoke, 32 is reasonable
  for real runs).
- **The box2d MAPPO `view=true` render path is broken in this tree.** It first
  demands `models_checkpoint.pth` (only `models_finished.pth` exists after a run
  that finished with `checkpoint=false`), and even with a checkpoint present it
  throws `TypeError: only 0-dimensional arrays can be converted to Python
  scalars` at `algorithms/mappo/trainer_components/renderer.py:163`
  (`cum_sum += float(reward)`). Use the MJX `render` path for a screenshot
  instead; don't rely on `view=true` for box2d until that bug is fixed.
- **All `environments/box2d_suite/*.py` `__main__` blocks are interactive
  keyboard debuggers** (arrow keys under a pygame window), not headless renders —
  useless in a container. The MJX renderer is the only headless visual path.
- **Harmless startup noise:** `Failed to import warp` / `mujoco_warp`,
  `overflow encountered in cast`, and an `os.fork()` + JAX multithread warning
  from the ffmpeg subprocess all print but do not affect output.
- **`render` writes to `.driver-out/` at the repo root.** `.png`/`.mp4` are
  gitignored, so it won't dirty the tree.

## Troubleshooting

- `FileNotFoundError: models_checkpoint.pth` on `view=true` — see the box2d
  view gotcha above; the file is only written mid-run, not at finish.
- MJX render/train crashes with a GL/EGL error — ensure `MUJOCO_GL=egl` is set
  (the driver does this; a bare `uv run python -m environments.mjx_suite.renderer`
  does not).
- First `smoke-jax` call seems to hang ~10-15s with no output — that is JAX
  JIT compilation (`JIT-compiling init/collect/update/eval`), not a hang.
