#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/project.env

POLICY_PATH="${1:-$CHECKPOINT_DIR/checkpoints/last/pretrained_model}"
OUT="${RESULTS_DIR}/finetuned_${SUITE}_k${ODE_STEPS}"
mkdir -p "$OUT"

cd "$LEROBOT_ROOT"
lerobot-eval \
  --output_dir="$OUT" \
  --policy.path="$POLICY_PATH" \
  --policy.num_steps="$ODE_STEPS" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.control_mode="$CONTROL_MODE" \
  --env.max_parallel_tasks=1 \
  --eval.batch_size="$EVAL_BATCH_SIZE" \
  --eval.n_episodes="$EVAL_EPISODES"
