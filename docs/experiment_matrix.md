# Experiment Matrix

## Assets

- Dataset: `HuggingFaceVLA/libero`, stored at `/root/autodl-tmp/datasets/HuggingFaceVLA_libero`
- Base model: `lerobot/smolvla_base`, stored at `/root/autodl-tmp/models/lerobot_smolvla_base`
- Suite for v1: `libero_spatial`

## Models

| Name | Policy | Training |
| --- | --- | --- |
| Zero-shot | `lerobot/smolvla_base` | none |
| Expert-only FT | SmolVLA | freeze vision/VLM, train state projector and action expert |
| Full FT | SmolVLA | optional later |

## Tests

| Test | Script | Episodes |
| --- | --- | --- |
| Standard closed-loop | `scripts/11_eval_zero_shot.sh`, `scripts/12_eval_finetuned.sh` | 10 tasks x 10 init states |
| Language robustness | `configs/paraphrases_libero_spatial.json` plus custom env language override | 3 tasks x 3 paraphrases x 5 seeds |
| ODE latency sweep | `scripts/20_latency_sweep.sh` | K in 5, 10, 20 |

## Metrics

- Closed-loop success rate
- Per-task success rate
- Validation Flow Matching loss
- First-action error
- Action chunk error
- Peak VRAM
- Train steps/s
- Action chunk inference latency
- Control frequency
