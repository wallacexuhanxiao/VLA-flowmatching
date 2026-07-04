#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/project.env

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR"

TRAIN_EPISODES_JSON="${SPLIT_DIR}/libero_spatial_train_episodes.json"
if [[ ! -f "$TRAIN_EPISODES_JSON" ]]; then
  echo "Missing $TRAIN_EPISODES_JSON. Run scripts/01_prepare_spatial_split.py first." >&2
  exit 1
fi
TRAIN_EPISODES="$(python - <<PY
import json
print(json.dumps(json.load(open("$TRAIN_EPISODES_JSON"))))
PY
)"

cd "$LEROBOT_ROOT"
lerobot-train \
  --policy.path="$MODEL_ROOT" \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --policy.num_steps="$ODE_STEPS" \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root="$DATA_ROOT" \
  --dataset.episodes="$TRAIN_EPISODES" \
  --dataset.eval_split=0.0 \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.control_mode="$CONTROL_MODE" \
  --output_dir="$CHECKPOINT_DIR" \
  --job_name="$PROJECT_NAME" \
  --steps="$TRAIN_STEPS" \
  --batch_size="$BATCH_SIZE" \
  --save_freq="$SAVE_FREQ" \
  --env_eval_freq="$ENV_EVAL_FREQ" \
  --eval.batch_size="$EVAL_BATCH_SIZE" \
  --eval.n_episodes=1 \
  --policy.device=cuda \
  --wandb.enable=false
