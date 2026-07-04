#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/project.env

python train.py \
  --config configs/flow_matching.yaml \
  "$@"
