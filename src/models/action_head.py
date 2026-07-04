from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class FlowMatchingActionHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.velocity_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_tokens = self.action_proj(noisy_actions)
        action_tokens = action_tokens + self.time_mlp(time).unsqueeze(1)
        hidden = self.decoder(
            tgt=action_tokens,
            memory=condition,
            memory_key_padding_mask=condition_mask,
        )
        return self.velocity_proj(hidden)


class L1ActionHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 512,
        chunk_size: int = 50,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, condition: torch.Tensor, condition_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = condition.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        hidden = self.decoder(queries, condition, memory_key_padding_mask=condition_mask)
        return self.out(hidden)
