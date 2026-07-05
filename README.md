# Flow-Matching VLA for LIBERO Manipulation

This project builds a lightweight Vision-Language-Action policy on top of a
frozen SmolVLM2 backbone. LeRobot/LIBERO are used for data and simulator
infrastructure, while the policy, action chunking, Flow Matching objective,
ODE sampler, offline metrics, and latency benchmark are implemented in this
repo.

## Core Idea

Inputs:

- main camera image
- wrist camera image
- natural-language task instruction
- 8D robot state

Output:

- continuous action chunk `[H, 7]`

Implemented modules:

- `SmolVLMFeatureExtractor`: frozen SmolVLM2 visual-language hidden states
- `ConditionResampler`: cross-attention compression to 16 condition tokens
- `StateEncoder`: robot state MLP
- `FlowMatchingActionHead`: Transformer decoder velocity-field predictor
- `L1ActionHead`: direct action-regression baseline
- explicit Flow Matching loss and Euler/Heun ODE sampling
- explicit action chunking with episode-end padding masks

## Assets

Stored on the server data disk:

- LIBERO dataset: `/root/autodl-tmp/datasets/HuggingFaceVLA_libero`
- SmolVLM2 backbone: `/root/autodl-tmp/models/SmolVLM2-500M-Video-Instruct`
- SmolVLA base reference: `/root/autodl-tmp/models/lerobot_smolvla_base`
- outputs: `/root/autodl-tmp/outputs/vla_project`

## Workflow

```bash
source /root/vla_project/activate_vla.sh
cd /root/vla_project
source configs/project.env

python scripts/00_check_assets.py --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" --vlm-root "$VLM_ROOT"
python scripts/01_prepare_spatial_split.py

# Smoke test both custom policies.
bash scripts/53_smoke_custom.sh

# Main Flow Matching model.
bash scripts/50_train_custom_flow.sh

# L1 regression baseline.
bash scripts/51_train_l1_baseline.sh

# Offline metrics and latency sweep.
bash scripts/52_eval_custom.sh configs/flow_matching.yaml /root/autodl-tmp/outputs/vla_project/custom_flow/checkpoints/last.pt
```

## Closed-Loop LIBERO Rollout

Rollout requires the simulator stack, not just the offline dataset:

- `libero`
- `robosuite`
- MuJoCo with EGL rendering, usually `MUJOCO_GL=egl`
- a visible CUDA GPU for practical SmolVLM2 inference
- trained checkpoints that include normalization statistics

Check the runtime first:

```bash
source /root/vla_project/activate_vla.sh
cd /root/vla_project
bash scripts/60_check_libero_rollout_env.sh
```

Run the main Flow Matching policy on LIBERO-Spatial:

```bash
CHECKPOINT=/root/autodl-tmp/outputs/vla_project/custom_flow_fixed_10k/checkpoints/best.pt \
OUT_DIR=/root/autodl-tmp/outputs/vla_project/rollouts/flow_ode10 \
bash scripts/61_rollout_flow_spatial.sh
```

Run the L1 baseline:

```bash
CHECKPOINT=/root/autodl-tmp/outputs/vla_project/custom_l1_fixed_10k/checkpoints/best.pt \
OUT_DIR=/root/autodl-tmp/outputs/vla_project/rollouts/l1 \
bash scripts/62_rollout_l1_spatial.sh
```

Run the Flow Matching ODE-step sweep:

```bash
CHECKPOINT=/root/autodl-tmp/outputs/vla_project/custom_flow_fixed_10k/checkpoints/best.pt \
OUT_ROOT=/root/autodl-tmp/outputs/vla_project/rollouts/flow_ode_sweep \
bash scripts/63_rollout_flow_ode_sweep.sh
```

Each rollout writes:

- `summary.json`: aggregate success rate, per-task success rate, average steps, average decision latency
- `episodes.jsonl`: one row per closed-loop episode

The controller uses receding horizon execution: sample a 50-step action chunk,
execute the first 10 actions, observe again, and repeat until success or timeout.

## Main Metrics

- offline Flow Matching validation loss
- first-action L2 error
- action-chunk L2 error
- predicted action smoothness
- action chunk inference latency for 5/10/20 ODE steps
- closed-loop LIBERO success rate
