from __future__ import annotations

import torch


def build_action_chunk(actions: torch.Tensor, start: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed-length action chunk and a boolean padding mask."""
    end = min(start + chunk_size, actions.shape[0])
    chunk = actions[start:end]
    valid = chunk.shape[0]
    if valid == 0:
        raise ValueError("Cannot build an action chunk from an empty episode.")
    if valid < chunk_size:
        pad = chunk[-1:].repeat(chunk_size - valid, 1)
        chunk = torch.cat([chunk, pad], dim=0)
    is_pad = torch.zeros(chunk_size, dtype=torch.bool)
    is_pad[valid:] = True
    return chunk, is_pad
