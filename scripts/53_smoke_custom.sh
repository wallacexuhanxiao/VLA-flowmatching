#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/project.env

python scripts/01_prepare_spatial_split.py
python train.py --config configs/flow_matching.yaml --smoke --batch-size 1 --output-dir /root/autodl-tmp/outputs/vla_project/smoke_flow
python train.py --config configs/l1_regression.yaml --smoke --batch-size 1 --output-dir /root/autodl-tmp/outputs/vla_project/smoke_l1
python evaluate.py --config configs/flow_matching.yaml --checkpoint /root/autodl-tmp/outputs/vla_project/smoke_flow/checkpoints/last.pt --max-episodes 1 --max-batches 1 --latency --ode-steps 5 10
