#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/l1_regression.yaml}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/outputs/vla_project/custom_l1_fixed_10k/checkpoints/best.pt}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/outputs/vla_project/rollouts/l1}"
LOCK_DIR="${LOCK_DIR:-/root/autodl-tmp/outputs/vla_project/locks/l1_rollout.lock}"

mkdir -p "$(dirname "$LOCK_DIR")" "$OUT_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "L1 rollout is already running; waiting for existing run to finish: $LOCK_DIR"
  while [ -d "$LOCK_DIR" ]; do sleep 30; done
  echo "Existing L1 rollout finished; skipping duplicate launch."
  exit 0
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

python -u rollout_libero.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --suite libero_spatial \
  --tasks 0 1 2 3 4 5 6 7 8 9 \
  --episodes-per-task "${EPISODES_PER_TASK:-10}" \
  --seed "${SEED:-7}" \
  --max-steps "${MAX_STEPS:-500}" \
  --execute-horizon "${EXECUTE_HORIZON:-10}" \
  --out "$OUT_DIR/summary.json" \
  --jsonl "$OUT_DIR/episodes.jsonl"
