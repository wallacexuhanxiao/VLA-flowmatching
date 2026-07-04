from __future__ import annotations

import time

import torch


@torch.no_grad()
def measure_latency(policy, batch: dict, device: torch.device, ode_steps: int, repeats: int = 20, warmup: int = 5) -> dict[str, float]:
    policy.eval()
    for _ in range(warmup):
        _ = policy.act(batch, device, ode_steps=ode_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = policy.act(batch, device, ode_steps=ode_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms = elapsed * 1000.0 / repeats
    return {"ode_steps": ode_steps, "latency_ms": ms, "control_hz": 1000.0 / ms}


@torch.no_grad()
def measure_head_latency(policy, batch: dict, device: torch.device, ode_steps: int, repeats: int = 50, warmup: int = 10) -> dict[str, float]:
    policy.eval()
    condition = policy.encode_condition(batch, device)
    for _ in range(warmup):
        if policy.mode == "flow":
            from src.models.flow_matching import sample_action_chunk

            _ = sample_action_chunk(policy.action_head, condition, policy.chunk_size, policy.action_dim, ode_steps)
        else:
            _ = policy.action_head(condition)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        if policy.mode == "flow":
            from src.models.flow_matching import sample_action_chunk

            _ = sample_action_chunk(policy.action_head, condition, policy.chunk_size, policy.action_dim, ode_steps)
        else:
            _ = policy.action_head(condition)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms = elapsed * 1000.0 / repeats
    return {"ode_steps": ode_steps, "head_latency_ms": ms, "head_control_hz": 1000.0 / ms}
