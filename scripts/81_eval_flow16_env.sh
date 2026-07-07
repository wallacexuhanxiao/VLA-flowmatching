#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/flow16.yaml}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/outputs/vla_project/u_flow16/checkpoints/best.pt}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/outputs/vla_project/rollouts/u_flow16}"

mkdir -p "$OUT_DIR"

python unified_env_run.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --suite libero_spatial \
  --tasks ${TASKS:-0} \
  --episodes-per-task "${EPISODES_PER_TASK:-5}" \
  --execute-horizon "${EXECUTE_HORIZON:-8}" \
  --ode-steps "${ODE_STEPS:-10}" \
  --sampler "${SAMPLER:-heun}" \
  --out "$OUT_DIR/summary.json" \
  --jsonl "$OUT_DIR/episodes.jsonl"
