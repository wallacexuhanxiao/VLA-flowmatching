from __future__ import annotations

import torch
from torch import nn

from .action_head import FlowMatchingActionHead, L1ActionHead
from .condition_resampler import ConditionResampler
from .flow_matching import sample_action_chunk
from .state_encoder import StateEncoder
from .vlm_encoder import SmolVLMFeatureExtractor


class CustomVLAPolicy(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        dtype = torch.bfloat16 if cfg.get("use_bf16", True) else None
        self.vlm = SmolVLMFeatureExtractor(
            cfg["vlm_root"],
            freeze=cfg.get("freeze_vlm", True),
            torch_dtype=dtype,
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        )
        hidden = int(cfg.get("condition_dim", 512))
        self.resampler = ConditionResampler(
            input_dim=self.vlm.hidden_size,
            hidden_dim=hidden,
            num_tokens=int(cfg.get("condition_tokens", 16)),
            num_heads=int(cfg.get("head_heads", 8)),
        )
        self.state_encoder = StateEncoder(int(cfg.get("state_dim", 8)), hidden)
        self.mode = cfg.get("mode", "flow")
        self.chunk_size = int(cfg.get("chunk_size", 50))
        self.action_dim = int(cfg.get("action_dim", 7))
        if self.mode == "flow":
            self.action_head = FlowMatchingActionHead(
                action_dim=self.action_dim,
                hidden_dim=hidden,
                chunk_size=self.chunk_size,
                num_layers=int(cfg.get("head_layers", 4)),
                num_heads=int(cfg.get("head_heads", 8)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
        elif self.mode == "l1":
            self.action_head = L1ActionHead(
                action_dim=self.action_dim,
                hidden_dim=hidden,
                chunk_size=self.chunk_size,
                num_layers=int(cfg.get("head_layers", 4)),
                num_heads=int(cfg.get("head_heads", 8)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
        else:
            raise ValueError(f"Unknown policy mode: {self.mode}")

    def encode_condition(self, batch: dict, device: torch.device) -> torch.Tensor:
        vlm_tokens, vlm_mask = self.vlm(
            batch["main_images"],
            batch["wrist_images"],
            batch["texts"],
            device=device,
        )
        target_dtype = next(self.resampler.parameters()).dtype
        vlm_tokens = vlm_tokens.to(dtype=target_dtype)
        condition = self.resampler(vlm_tokens, vlm_mask)
        state_token = self.state_encoder(batch["states"].to(device)).unsqueeze(1)
        return torch.cat([condition, state_token], dim=1)

    @torch.no_grad()
    def act(
        self,
        batch: dict,
        device: torch.device,
        ode_steps: int = 10,
        sampler: str = "euler",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(batch, device)
        if self.mode == "flow":
            return sample_action_chunk(
                self.action_head,
                condition,
                chunk_size=self.chunk_size,
                action_dim=self.action_dim,
                num_steps=ode_steps,
                method=sampler,
                generator=generator,
            )
        return self.action_head(condition)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        modules = {
            "resampler": self.resampler.state_dict(),
            "state_encoder": self.state_encoder.state_dict(),
            "action_head": self.action_head.state_dict(),
        }
        return modules

    def load_trainable_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.resampler.load_state_dict(state["resampler"])
        self.state_encoder.load_state_dict(state["state_encoder"])
        self.action_head.load_state_dict(state["action_head"], strict=False)
