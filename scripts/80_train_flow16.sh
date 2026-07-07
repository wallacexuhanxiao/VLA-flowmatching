#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/project.env

python scripts/01_prepare_spatial_split.py

python unified_entry.py \
  --config configs/flow16.yaml \
  --output-dir "${OUTPUT_DIR:-/root/autodl-tmp/outputs/vla_project/u_flow16}" \
  "$@"
