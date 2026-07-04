#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/project.env

mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$VLM_ROOT"

hf download HuggingFaceVLA/libero \
  --repo-type dataset \
  --local-dir "$DATA_ROOT" \
  --include '*' \
  --max-workers "${LIBERO_DOWNLOAD_WORKERS:-8}"

hf download lerobot/smolvla_base \
  --repo-type model \
  --local-dir "$MODEL_ROOT" \
  --include '*' \
  --max-workers "${MODEL_DOWNLOAD_WORKERS:-1}"

hf download "$VLM_MODEL_ID" \
  --repo-type model \
  --local-dir "$VLM_ROOT" \
  --include '*' \
  --max-workers "${VLM_DOWNLOAD_WORKERS:-1}"

python scripts/00_check_assets.py \
  --data-root "$DATA_ROOT" \
  --model-root "$MODEL_ROOT" \
  --vlm-root "$VLM_ROOT"
