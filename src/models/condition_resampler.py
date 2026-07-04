from __future__ import annotations

import torch
from torch import nn


class ConditionResampler(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_tokens: int = 16, num_heads: int = 8) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.input_proj(tokens)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attn, _ = self.attn(queries, memory, memory, key_padding_mask=key_padding_mask, need_weights=False)
        hidden = self.norm(queries + attn)
        return self.out_norm(hidden + self.ffn(hidden))
