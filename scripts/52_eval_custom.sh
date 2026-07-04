#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/project.env

CONFIG="${1:-configs/flow_matching.yaml}"
CHECKPOINT="${2:-}"
shift || true
shift || true

if [[ -n "$CHECKPOINT" ]]; then
  python evaluate.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --latency "$@"
else
  python evaluate.py --config "$CONFIG" --latency "$@"
fi
