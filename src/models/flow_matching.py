from __future__ import annotations

import torch


def compute_flow_matching_loss(
    action_head,
    actions: torch.Tensor,
    condition: torch.Tensor,
    action_is_pad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = actions.shape[0]
    noise = torch.randn_like(actions)
    time = torch.rand(batch, device=actions.device, dtype=actions.dtype)
    time_expanded = time[:, None, None]
    noisy_actions = (1.0 - time_expanded) * actions + time_expanded * noise
    target_velocity = noise - actions
    predicted_velocity = action_head(noisy_actions, time, condition)
    loss = (predicted_velocity - target_velocity).square()
    if action_is_pad is not None:
        valid = (~action_is_pad).unsqueeze(-1).to(loss.dtype)
        loss_value = (loss * valid).sum() / (valid.sum().clamp_min(1) * actions.shape[-1])
    else:
        loss_value = loss.mean()
    return loss_value, {"fm_mse": float(loss_value.detach().cpu())}


@torch.no_grad()
def sample_action_chunk(
    action_head,
    condition: torch.Tensor,
    chunk_size: int,
    action_dim: int,
    num_steps: int = 10,
    method: str = "euler",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x_t = torch.randn(
        condition.shape[0],
        chunk_size,
        action_dim,
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    dt = -1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((condition.shape[0],), 1.0 + step * dt, device=condition.device, dtype=condition.dtype)
        velocity = action_head(x_t, t, condition)
        if method == "heun":
            proposal = x_t + dt * velocity
            t_next = torch.full((condition.shape[0],), max(0.0, 1.0 + (step + 1) * dt), device=condition.device, dtype=condition.dtype)
            velocity_next = action_head(proposal, t_next, condition)
            velocity = 0.5 * (velocity + velocity_next)
        x_t = x_t + dt * velocity
    return x_t
