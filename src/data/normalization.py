from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RunningStats:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


def compute_stats(tensors: list[torch.Tensor], eps: float = 1e-6) -> RunningStats:
    values = torch.cat([t.reshape(-1, t.shape[-1]).float() for t in tensors], dim=0)
    return RunningStats(values.mean(dim=0), values.std(dim=0).clamp_min(eps))
