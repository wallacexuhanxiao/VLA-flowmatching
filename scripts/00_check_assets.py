#!/usr/bin/env python3
"""Check downloaded LIBERO data, SmolVLA baseline, and VLM backbone assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def size_gib(path: Path) -> float:
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    return total / 1024**3


def count_suffix(path: Path, suffix: str) -> int:
    return sum(1 for p in path.rglob(f"*{suffix}") if p.is_file())


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    parser.add_argument("--model-root", default=os.environ.get("MODEL_ROOT", ""))
    parser.add_argument("--vlm-root", default=os.environ.get("VLM_ROOT", ""))
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    model_root = Path(args.model_root).expanduser()
    vlm_root = Path(args.vlm_root).expanduser()

    print(f"DATA_ROOT={data_root}")
    print(f"exists={data_root.exists()} size_gib={size_gib(data_root):.2f}")
    print(f"parquet_files={count_suffix(data_root, '.parquet')}")
    print(f"json_files={count_suffix(data_root, '.json')}")
    print(f"jsonl_files={count_suffix(data_root, '.jsonl')}")

    meta = data_root / "meta"
    if meta.exists():
        print(f"meta_files={len([p for p in meta.rglob('*') if p.is_file()])}")
        for name in ["info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"]:
            p = meta / name
            if p.exists():
                print(f"meta/{name}: {p.stat().st_size} bytes")

    model_file = model_root / "model.safetensors"
    config_file = model_root / "config.json"
    print(f"MODEL_ROOT={model_root}")
    print(f"exists={model_root.exists()} size_gib={size_gib(model_root):.2f}")
    print(f"model.safetensors_exists={model_file.exists()}")
    if model_file.exists():
        print(f"model.safetensors_mib={model_file.stat().st_size / 1024**2:.1f}")
    print(f"config.json_exists={config_file.exists()}")
    if config_file.exists():
        cfg = read_json(config_file) or {}
        keys = ["type", "chunk_size", "n_action_steps", "num_steps", "freeze_vision_encoder", "train_expert_only"]
        print("config_subset=" + json.dumps({k: cfg.get(k) for k in keys}, ensure_ascii=False))

    if args.vlm_root:
        vlm_config = vlm_root / "config.json"
        print(f"VLM_ROOT={vlm_root}")
        print(f"exists={vlm_root.exists()} size_gib={size_gib(vlm_root):.2f}")
        print(f"config.json_exists={vlm_config.exists()}")
        safetensors = sorted(vlm_root.glob("*.safetensors"))
        index_file = vlm_root / "model.safetensors.index.json"
        print(f"safetensors_files={len(safetensors)}")
        print(f"model.safetensors.index.json_exists={index_file.exists()}")
        if vlm_config.exists():
            cfg = read_json(vlm_config) or {}
            keys = ["model_type", "architectures", "vision_config", "text_config"]
            compact = {k: cfg.get(k) for k in keys if k in cfg}
            if isinstance(compact.get("vision_config"), dict):
                compact["vision_config"] = {
                    k: compact["vision_config"].get(k)
                    for k in ["model_type", "hidden_size", "image_size", "patch_size"]
                }
            if isinstance(compact.get("text_config"), dict):
                compact["text_config"] = {
                    k: compact["text_config"].get(k)
                    for k in ["model_type", "hidden_size", "num_hidden_layers"]
                }
            print("vlm_config_subset=" + json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
