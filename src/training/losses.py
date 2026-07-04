from __future__ import annotations

import torch

from src.models.flow_matching import compute_flow_matching_loss


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    loss = (pred - target).abs()
    if action_is_pad is not None:
        valid = (~action_is_pad).unsqueeze(-1).to(loss.dtype)
        value = (loss * valid).sum() / (valid.sum().clamp_min(1) * target.shape[-1])
    else:
        value = loss.mean()
    return value, {"l1": float(value.detach().cpu())}


def compute_policy_loss(policy, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    condition = policy.encode_condition(batch, device)
    actions = batch["actions"].to(device)
    action_is_pad = batch["action_is_pad"].to(device)
    if policy.mode == "flow":
        return compute_flow_matching_loss(policy.action_head, actions, condition, action_is_pad)
    pred = policy.action_head(condition)
    return masked_l1_loss(pred, actions, action_is_pad)
