from __future__ import annotations

import torch


def action_errors(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor | None = None) -> dict[str, float]:
    dist = torch.linalg.vector_norm(pred - target, dim=-1)
    first = dist[:, 0].mean()
    if action_is_pad is not None:
        valid = (~action_is_pad).to(dist.dtype)
        chunk = (dist * valid).sum() / valid.sum().clamp_min(1)
    else:
        chunk = dist.mean()
    smoothness = torch.linalg.vector_norm(pred[:, 1:] - pred[:, :-1], dim=-1).mean()
    return {
        "first_action_l2": float(first.detach().cpu()),
        "chunk_l2": float(chunk.detach().cpu()),
        "pred_smoothness": float(smoothness.detach().cpu()),
    }
