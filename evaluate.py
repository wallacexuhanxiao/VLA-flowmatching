#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import load_config, save_json
from src.data.libero_dataset import LiberoChunkDataset, collate_libero, load_episode_list
from src.data.normalization import stats_from_dict
from src.evaluation.offline_metrics import action_errors
from src.evaluation.latency import measure_head_latency, measure_latency
from src.models.flow_matching import flow_validation_metrics, sample_action_chunk
from src.models.vla_policy import CustomVLAPolicy


def load_checkpoint_payload(checkpoint: str | None) -> dict | None:
    if not checkpoint:
        return None
    return torch.load(checkpoint, map_location="cpu")


def load_policy(payload: dict | None, cfg: dict, device: torch.device) -> CustomVLAPolicy:
    policy = CustomVLAPolicy(cfg).to(device)
    if payload:
        policy.load_trainable_state_dict(payload["trainable"])
    policy.eval()
    return policy


def balanced_validation_indices(dataset: LiberoChunkDataset, frames_per_task: int) -> list[int]:
    by_task: dict[int, list[int]] = {}
    for sample_idx, (episode_pos, _frame_idx) in enumerate(dataset.index):
        task = int(dataset.episodes[episode_pos]["task_index"])
        by_task.setdefault(task, []).append(sample_idx)
    selected: list[int] = []
    for task in sorted(by_task):
        indices = by_task[task]
        if len(indices) <= frames_per_task:
            selected.extend(indices)
            continue
        step = (len(indices) - 1) / max(frames_per_task - 1, 1)
        selected.extend(indices[round(i * step)] for i in range(frames_per_task))
    return selected


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flow_matching.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--ode-steps", type=int, nargs="*", default=[5, 10, 20])
    parser.add_argument("--out")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_checkpoint_payload(args.checkpoint)
    episodes = load_episode_list(cfg.get(f"{args.split}_episodes"))
    state_stats = action_stats = None
    if payload and "normalization" in payload:
        state_stats, action_stats = stats_from_dict(payload["normalization"])
    dataset = LiberoChunkDataset(
        cfg["data_root"],
        episode_indices=episodes,
        chunk_size=int(cfg["chunk_size"]),
        main_image_key=cfg["main_image_key"],
        wrist_image_key=cfg["wrist_image_key"],
        state_key=cfg["state_key"],
        action_key=cfg["action_key"],
        task_key=cfg["task_key"],
        episode_key=cfg["episode_key"],
        state_stats=state_stats,
        action_stats=action_stats,
        max_episodes=args.max_episodes,
    )
    eval_dataset = dataset
    if cfg.get("val_frames_per_task"):
        eval_dataset = Subset(dataset, balanced_validation_indices(dataset, int(cfg["val_frames_per_task"])))
    loader = DataLoader(eval_dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0, collate_fn=collate_libero)
    policy = load_policy(payload, cfg, device)

    rows = []
    eval_seed = int(cfg.get("eval_seed", 12345))
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        actions = batch["actions"].to(device)
        action_is_pad = batch["action_is_pad"].to(device)
        condition = policy.encode_condition(batch, device)
        if policy.mode == "flow":
            pred = sample_action_chunk(
                policy.action_head,
                condition,
                int(cfg["chunk_size"]),
                int(cfg["action_dim"]),
                num_steps=int(cfg.get("ode_steps", 10)),
                method=cfg.get("sampler", "euler"),
                generator=make_generator(device, eval_seed + batch_idx),
            )
        else:
            pred = policy.action_head(condition)
        metrics = action_errors(pred.float(), actions.float(), action_is_pad)
        if policy.mode == "flow":
            metrics.update(
                flow_validation_metrics(
                    policy.action_head,
                    actions,
                    condition,
                    action_is_pad,
                    generator=make_generator(device, eval_seed + 100000 + batch_idx),
                )
            )
        rows.append(metrics)
    avg = {k: sum(r[k] for r in rows) / max(len(rows), 1) for k in rows[0]} if rows else {}
    result = {"offline": avg, "num_batches": len(rows), "checkpoint": args.checkpoint, "uses_checkpoint_normalization": bool(payload and "normalization" in payload)}

    if args.latency:
        first_batch = next(iter(loader))
        result["latency"] = [measure_latency(policy, first_batch, device, k, repeats=5, warmup=2) for k in args.ode_steps]
        result["head_latency"] = [measure_head_latency(policy, first_batch, device, k, repeats=20, warmup=5) for k in args.ode_steps]

    print(result)
    if args.out:
        save_json(result, Path(args.out))


if __name__ == "__main__":
    main()
