#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config, save_json
from src.data.libero_dataset import LiberoChunkDataset, collate_libero, load_episode_list
from src.evaluation.offline_metrics import action_errors
from src.evaluation.latency import measure_head_latency, measure_latency
from src.models.vla_policy import CustomVLAPolicy


def load_policy(checkpoint: str | None, cfg: dict, device: torch.device) -> CustomVLAPolicy:
    policy = CustomVLAPolicy(cfg).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        policy.load_trainable_state_dict(state["trainable"])
    policy.eval()
    return policy


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flow_matching.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--ode-steps", type=int, nargs="*", default=[5, 10, 20])
    parser.add_argument("--out")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = load_episode_list(cfg.get(f"{args.split}_episodes"))
    dataset = LiberoChunkDataset(cfg["data_root"], episode_indices=episodes, chunk_size=int(cfg["chunk_size"]), max_episodes=args.max_episodes)
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0, collate_fn=collate_libero)
    policy = load_policy(args.checkpoint, cfg, device)

    rows = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        pred = policy.act(batch, device, ode_steps=int(cfg.get("ode_steps", 10)), sampler=cfg.get("sampler", "euler"))
        metrics = action_errors(pred.float(), batch["actions"].to(device).float(), batch["action_is_pad"].to(device))
        rows.append(metrics)
    avg = {k: sum(r[k] for r in rows) / max(len(rows), 1) for k in rows[0]} if rows else {}
    result = {"offline": avg, "num_batches": len(rows), "checkpoint": args.checkpoint}

    if args.latency:
        first_batch = next(iter(loader))
        result["latency"] = [measure_latency(policy, first_batch, device, k, repeats=5, warmup=2) for k in args.ode_steps]
        result["head_latency"] = [measure_head_latency(policy, first_batch, device, k, repeats=20, warmup=5) for k in args.ode_steps]

    print(result)
    if args.out:
        save_json(result, Path(args.out))


if __name__ == "__main__":
    main()
