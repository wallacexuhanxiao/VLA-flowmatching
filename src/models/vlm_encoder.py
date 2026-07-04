from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor


def _load_vlm_model(path: str, torch_dtype: torch.dtype | None = None):
    kwargs = {"local_files_only": True}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(path, **kwargs)
    except Exception:
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(path, **kwargs)
        except Exception:
            from transformers import AutoModelForCausalLM

            kwargs["trust_remote_code"] = True
            return AutoModelForCausalLM.from_pretrained(path, **kwargs)


class SmolVLMFeatureExtractor(nn.Module):
    """Frozen SmolVLM2 wrapper returning visual-language token features."""

    def __init__(
        self,
        model_path: str,
        freeze: bool = True,
        torch_dtype: torch.dtype | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = _load_vlm_model(model_path, torch_dtype=torch_dtype)
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()
        self.freeze = freeze
        self.hidden_size = self._infer_hidden_size()

    def _infer_hidden_size(self) -> int:
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        raise AttributeError("Could not infer VLM hidden size from config.")

    def _prompts(self, texts: list[str]) -> list[str]:
        prompts = []
        for text in texts:
            prompts.append(
                "<image><image>\n"
                "You are observing a robot manipulation scene from the main camera and wrist camera.\n"
                f"Instruction: {text}\n"
                "Return compact visual-language features for control."
            )
        return prompts

    def _processor_call(self, main_images: list[Image.Image], wrist_images: list[Image.Image], texts: list[str]) -> dict[str, Any]:
        prompts = self._prompts(texts)
        paired_images = [[m, w] for m, w in zip(main_images, wrist_images)]
        try:
            return self.processor(text=prompts, images=paired_images, return_tensors="pt", padding=True)
        except Exception:
            flat_images = [img for pair in paired_images for img in pair]
            return self.processor(text=prompts, images=flat_images, return_tensors="pt", padding=True)

    def forward(
        self,
        main_images: list[Image.Image],
        wrist_images: list[Image.Image],
        texts: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        inputs = self._processor_call(main_images, wrist_images, texts)
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            tokens = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        else:
            raise RuntimeError("VLM output did not include hidden states.")
        attention_mask = inputs.get("attention_mask")
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        return tokens, key_padding_mask
