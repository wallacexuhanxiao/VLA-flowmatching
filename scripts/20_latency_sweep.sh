#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/project.env

POLICY_PATH="${1:-$MODEL_ROOT}"
mkdir -p "$RESULTS_DIR/latency_sweep"

for K in 5 10 20; do
  echo "Running ODE step sweep K=$K"
  ODE_STEPS="$K" RESULTS_DIR="$RESULTS_DIR/latency_sweep" POLICY_PATH="$POLICY_PATH" bash -c '
    cd "$LEROBOT_ROOT"
    lerobot-eval \
      --output_dir="${RESULTS_DIR}/k${ODE_STEPS}" \
      --policy.path="$POLICY_PATH" \
      --policy.num_steps="$ODE_STEPS" \
      --policy.n_action_steps="$N_ACTION_STEPS" \
      --env.type=libero \
      --env.task="$SUITE" \
      --env.control_mode="$CONTROL_MODE" \
      --env.max_parallel_tasks=1 \
      --eval.batch_size=1 \
      --eval.n_episodes="${LATENCY_EPISODES:-3}"
  '
done

python scripts/30_summarize_results.py --root "$RESULTS_DIR/latency_sweep" --out "$RESULTS_DIR/latency_sweep/summary.csv"
