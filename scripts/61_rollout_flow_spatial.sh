#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/flow_matching.yaml}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/outputs/vla_project/custom_flow_fixed_10k/checkpoints/best.pt}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/outputs/vla_project/rollouts/flow_ode10}"

mkdir -p "$OUT_DIR"

python -u rollout_libero.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --suite libero_spatial \
  --tasks 0 1 2 3 4 5 6 7 8 9 \
  --episodes-per-task "${EPISODES_PER_TASK:-10}" \
  --seed "${SEED:-7}" \
  --max-steps "${MAX_STEPS:-500}" \
  --execute-horizon "${EXECUTE_HORIZON:-10}" \
  --ode-steps "${ODE_STEPS:-10}" \
  --sampler "${SAMPLER:-euler}" \
  --out "$OUT_DIR/summary.json" \
  --jsonl "$OUT_DIR/episodes.jsonl"

