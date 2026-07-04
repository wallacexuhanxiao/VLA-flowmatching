#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config, save_json
from src.data.libero_dataset import LiberoChunkDataset, collate_libero, load_episode_list
from src.evaluation.offline_metrics import action_errors
from src.models.vla_policy import CustomVLAPolicy
from src.training.losses import compute_policy_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(cfg: dict, split: str, state_stats=None, action_stats=None, max_episodes=None, shuffle=True):
    episodes = load_episode_list(cfg.get(f"{split}_episodes"))
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
        max_episodes=max_episodes,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_libero,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )
    return dataset, loader


def save_checkpoint(policy, optimizer, cfg: dict, step: int, metrics: dict, out_dir: Path) -> None:
    checkpoint_root = out_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": cfg,
        "trainable": policy.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, checkpoint_root / "last.pt")

    score = metrics.get("val/chunk_l2")
    best_meta_path = checkpoint_root / "best_meta.json"
    best_score = None
    if best_meta_path.exists():
        import json

        best_score = json.loads(best_meta_path.read_text()).get("val/chunk_l2")
    if score is not None and (best_score is None or score < best_score):
        torch.save(payload, checkpoint_root / "best.pt")
        save_json({"step": step, "val/chunk_l2": score, "metrics": metrics}, best_meta_path)


def load_checkpoint(path: str | Path, policy, optimizer=None) -> tuple[int, dict[str, float]]:
    checkpoint = torch.load(path, map_location="cpu")
    policy.load_trainable_state_dict(checkpoint["trainable"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0)), dict(checkpoint.get("metrics", {}))


@torch.no_grad()
def evaluate(policy, loader, device: torch.device, cfg: dict, max_batches: int) -> dict[str, float]:
    policy.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        actions = batch["actions"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and cfg.get("use_bf16", True)):
            pred = policy.act(batch, device, ode_steps=int(cfg.get("ode_steps", 10)), sampler=cfg.get("sampler", "euler"))
        metrics = action_errors(pred.float(), actions.float(), batch["action_is_pad"].to(device))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    policy.train()
    return {f"val/{k}": v / max(count, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flow_matching.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-train-episodes", type=int)
    parser.add_argument("--max-val-episodes", type=int)
    parser.add_argument("--resume", help="Path to a checkpoint saved by train.py.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.smoke:
        cfg["max_steps"] = min(int(cfg["max_steps"]), 3)
        cfg["eval_every"] = 1
        cfg["save_every"] = 3
        cfg["max_val_batches"] = 1
        args.max_train_episodes = args.max_train_episodes or 1
        args.max_val_episodes = args.max_val_episodes or 1

    set_seed(int(cfg.get("seed", 7)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "resolved_config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, train_loader = make_loader(cfg, "train", max_episodes=args.max_train_episodes, shuffle=True)
    val_dataset, val_loader = make_loader(
        cfg,
        "val",
        state_stats=train_dataset.state_stats,
        action_stats=train_dataset.action_stats,
        max_episodes=args.max_val_episodes,
        shuffle=False,
    )
    policy = CustomVLAPolicy(cfg).to(device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    print(f"device={device} train_samples={len(train_dataset)} val_samples={len(val_dataset)} trainable_params={sum(p.numel() for p in trainable):,}")
    step = 0
    last_metrics: dict[str, float] = {}
    if args.resume:
        step, last_metrics = load_checkpoint(args.resume, policy, optimizer)
        print(f"resumed checkpoint={args.resume} step={step}")
    start = time.perf_counter()
    while step < int(cfg["max_steps"]):
        for batch in train_loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and cfg.get("use_bf16", True)):
                loss, metrics = compute_policy_loss(policy, batch, device)
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg["grad_clip"]))
            optimizer.step()

            last_metrics = {"train/loss": float(loss.detach().cpu()), **{f"train/{k}": v for k, v in metrics.items()}}
            if step % int(cfg["log_every"]) == 0 or step == 1:
                elapsed = time.perf_counter() - start
                print(f"step={step} loss={last_metrics['train/loss']:.5f} steps_per_sec={step / max(elapsed, 1e-6):.3f}")
            if step % int(cfg["eval_every"]) == 0 or (args.smoke and step == 1):
                val_metrics = evaluate(policy, val_loader, device, cfg, int(cfg.get("max_val_batches", 64)))
                last_metrics.update(val_metrics)
                print("eval " + " ".join(f"{k}={v:.5f}" for k, v in val_metrics.items()))
                save_json({"step": step, **last_metrics}, out_dir / "metrics_latest.json")
            if step % int(cfg["save_every"]) == 0 or step == int(cfg["max_steps"]):
                save_checkpoint(policy, optimizer, cfg, step, last_metrics, out_dir)
            if step >= int(cfg["max_steps"]):
                break
    print(f"done step={step} output_dir={out_dir}")


if __name__ == "__main__":
    main()
