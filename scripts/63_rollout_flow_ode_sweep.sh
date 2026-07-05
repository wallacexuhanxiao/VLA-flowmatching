#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/flow_matching.yaml}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/outputs/vla_project/custom_flow_fixed_10k/checkpoints/best.pt}"
ROOT="${OUT_ROOT:-/root/autodl-tmp/outputs/vla_project/rollouts/flow_ode_sweep}"

for steps in ${ODE_SWEEP:-5 10 20}; do
  OUT_DIR="$ROOT/ode_${steps}"
  mkdir -p "$OUT_DIR"
  ODE_STEPS="$steps" OUT_DIR="$OUT_DIR" CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" \
    bash scripts/61_rollout_flow_spatial.sh
done

