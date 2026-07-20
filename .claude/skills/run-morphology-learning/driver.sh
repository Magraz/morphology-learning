#!/usr/bin/env bash
# Driver for the morphology-learning experiment manager.
#
# This is a research RL codebase, not a GUI/web app. The "app" is the Hydra
# training entry point (train.py) plus the MJX environment renderer. This driver
# encodes the exact working overrides (tiny step budgets, headless GL/SDL flags,
# pinned env counts) so a future agent can smoke-test each path in seconds
# instead of hours.
#
# All commands run from the repo root via `uv run`. Usage:
#   ./driver.sh smoke      # short box2d MAPPO (torch) training run
#   ./driver.sh smoke-jax  # short MJX MAPPO-JAX (jitted, GPU) training run
#   ./driver.sh render     # render the MJX env -> screenshot .png + rollout .mp4
#   ./driver.sh all        # render + both smokes
set -euo pipefail

# Repo root = two levels up from .claude/skills/run-morphology-learning/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-$ROOT/.driver-out}"
mkdir -p "$OUT"

smoke() {
  echo ">>> box2d MAPPO smoke (torch, CPU, AsyncVectorEnv fork)"
  uv run python train.py \
      algorithm=mappo env=multi_box_push_9a_3o model=mlp_shared \
      trial_id=smoke_test device=cpu \
      env.n_envs=2 params.n_steps=128 params.n_total_steps=512
  echo ">>> results: experiments/results/multi_box_push_9a_3o/mlp_shared/smoke_test/"
}

smoke_jax() {
  echo ">>> MJX MAPPO-JAX smoke (jitted; uses GPU if visible, else CPU)"
  # MUJOCO_GL=egl lets MJX init a GL context headless. n_envs is vmapped on one
  # device, so pin it small on the CLI (the central autoscale targets box2d subprocs).
  MUJOCO_GL=egl uv run python train.py \
      algorithm=mappo_jax env=multi_box_push_mjx_20a_5o model=mlp \
      trial_id=smoke_test device=cpu \
      env.n_envs=8 params.n_steps=128 params.n_total_steps=2048
  echo ">>> results: experiments/results/multi_box_push_mjx_20a_5o/mlp/smoke_test/"
}

render() {
  local png="$OUT/mjx_frame.png" mp4="$OUT/mjx_rollout.mp4"
  echo ">>> MJX renderer -> $png (+ $mp4)"
  # egl: headless GL for MJX raycasts/native render. SDL dummy: pygame overlay
  # needs a video driver even in rgb_array mode.
  MUJOCO_GL=egl SDL_VIDEODRIVER=dummy uv run python -m environments.mjx_suite.renderer \
      --steps 120 --every 4 --out "$mp4" --frame-png "$png"
  echo ">>> screenshot written: $png"
}

case "${1:-help}" in
  smoke)     smoke ;;
  smoke-jax) smoke_jax ;;
  render)    render ;;
  all)       render; smoke; smoke_jax ;;
  *)
    echo "usage: $0 {smoke|smoke-jax|render|all}"
    echo "  smoke      short box2d MAPPO (torch) training run"
    echo "  smoke-jax  short MJX MAPPO-JAX (jitted, GPU) training run"
    echo "  render     MJX env -> screenshot png + rollout mp4 (in \$OUT, default .driver-out/)"
    echo "  all        render + both smokes"
    ;;
esac
