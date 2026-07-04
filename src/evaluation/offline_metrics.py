from __future__ import annotations

import torch


def action_errors(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor | None = None) -> dict[str, float]:
    dist = torch.linalg.vector_norm(pred - target, dim=-1)
    first = dist[:, 0].mean()
    if action_is_pad is not None:
        valid = (~action_is_pad).to(dist.dtype)
        chunk = (dist * valid).sum() / valid.sum().clamp_min(1)
    else:
        valid = None
        chunk = dist.mean()
    pred_delta = torch.linalg.vector_norm(pred[:, 1:] - pred[:, :-1], dim=-1)
    target_delta = torch.linalg.vector_norm(target[:, 1:] - target[:, :-1], dim=-1)
    if valid is not None:
        valid_pairs = valid[:, 1:] * valid[:, :-1]
        pred_smoothness = (pred_delta * valid_pairs).sum() / valid_pairs.sum().clamp_min(1)
        target_smoothness = (target_delta * valid_pairs).sum() / valid_pairs.sum().clamp_min(1)
    else:
        pred_smoothness = pred_delta.mean()
        target_smoothness = target_delta.mean()
    smoothness_ratio = pred_smoothness / target_smoothness.clamp_min(1e-6)
    return {
        "first_action_l2": float(first.detach().cpu()),
        "chunk_l2": float(chunk.detach().cpu()),
        "pred_smoothness": float(pred_smoothness.detach().cpu()),
        "target_smoothness": float(target_smoothness.detach().cpu()),
        "smoothness_ratio": float(smoothness_ratio.detach().cpu()),
    }
