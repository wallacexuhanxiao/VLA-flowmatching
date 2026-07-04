#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import load_config, save_json
from src.data.libero_dataset import LiberoChunkDataset, collate_libero, load_episode_list
from src.data.normalization import stats_to_dict
from src.evaluation.offline_metrics import action_errors
from src.models.flow_matching import flow_validation_metrics, sample_action_chunk
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
        if frames_per_task == 1:
            selected.append(indices[len(indices) // 2])
            continue
        step = (len(indices) - 1) / (frames_per_task - 1)
        selected.extend(indices[round(i * step)] for i in range(frames_per_task))
    return selected


def make_eval_loader(dataset: LiberoChunkDataset, cfg: dict) -> DataLoader:
    frames_per_task = cfg.get("val_frames_per_task")
    eval_dataset = dataset
    if frames_per_task:
        eval_dataset = Subset(dataset, balanced_validation_indices(dataset, int(frames_per_task)))
    return DataLoader(
        eval_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_libero,
        pin_memory=torch.cuda.is_available(),
    )


def save_checkpoint(
    policy,
    optimizer,
    cfg: dict,
    step: int,
    metrics: dict,
    out_dir: Path,
    normalization: dict[str, torch.Tensor],
) -> None:
    checkpoint_root = out_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": cfg,
        "trainable": policy.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "normalization": normalization,
    }
    torch.save(payload, checkpoint_root / "last.pt")

    score = metrics.get("val/macro_chunk_l2", metrics.get("val/chunk_l2"))
    best_meta_path = checkpoint_root / "best_meta.json"
    best_score = None
    if best_meta_path.exists():
        import json

        meta = json.loads(best_meta_path.read_text())
        best_score = meta.get("val/macro_chunk_l2", meta.get("val/chunk_l2"))
    if score is not None and (best_score is None or score < best_score):
        torch.save(payload, checkpoint_root / "best.pt")
        save_json({"step": step, "val/macro_chunk_l2": score, "metrics": metrics}, best_meta_path)


def load_checkpoint(path: str | Path, policy, optimizer=None) -> tuple[int, dict[str, float]]:
    checkpoint = torch.load(path, map_location="cpu")
    policy.load_trainable_state_dict(checkpoint["trainable"])
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            print(f"warning: optimizer state was not loaded: {exc}")
    return int(checkpoint.get("step", 0)), dict(checkpoint.get("metrics", {}))


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def evaluate(policy, loader, device: torch.device, cfg: dict, max_batches: int) -> dict[str, float]:
    policy.eval()
    totals: dict[str, float] = {}
    per_task: dict[int, dict[str, float]] = {}
    per_task_counts: dict[int, int] = {}
    count = 0
    eval_seed = int(cfg.get("eval_seed", 12345))
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        actions = batch["actions"].to(device)
        action_is_pad = batch["action_is_pad"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and cfg.get("use_bf16", True)):
            condition = policy.encode_condition(batch, device)
            if policy.mode == "flow":
                pred = sample_action_chunk(
                    policy.action_head,
                    condition,
                    chunk_size=policy.chunk_size,
                    action_dim=policy.action_dim,
                    num_steps=int(cfg.get("ode_steps", 10)),
                    method=cfg.get("sampler", "euler"),
                    generator=make_generator(device, eval_seed + batch_idx),
                )
            else:
                pred = policy.action_head(condition)
        metrics = action_errors(pred.float(), actions.float(), action_is_pad)
        if policy.mode == "flow":
            diag = flow_validation_metrics(
                policy.action_head,
                actions,
                condition,
                action_is_pad,
                generator=make_generator(device, eval_seed + 100000 + batch_idx),
            )
            metrics.update(diag)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        for task in batch["task_index"].unique().tolist():
            mask = batch["task_index"] == task
            task_metrics = action_errors(pred[mask].float(), actions[mask].float(), action_is_pad[mask])
            bucket = per_task.setdefault(int(task), {})
            per_task_counts[int(task)] = per_task_counts.get(int(task), 0) + 1
            for key, value in task_metrics.items():
                bucket[key] = bucket.get(key, 0.0) + value
        count += 1
    policy.train()
    result = {f"val/{k}": v / max(count, 1) for k, v in totals.items()}
    task_first = []
    task_chunk = []
    for task, values in sorted(per_task.items()):
        denom = max(per_task_counts.get(task, 1), 1)
        first = values["first_action_l2"] / denom
        chunk = values["chunk_l2"] / denom
        result[f"val/task_{task}/first_action_l2"] = first
        result[f"val/task_{task}/chunk_l2"] = chunk
        task_first.append(first)
        task_chunk.append(chunk)
    if task_chunk:
        result["val/macro_first_action_l2"] = sum(task_first) / len(task_first)
        result["val/macro_chunk_l2"] = sum(task_chunk) / len(task_chunk)
    return result


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
    val_dataset, _ = make_loader(
        cfg,
        "val",
        state_stats=train_dataset.state_stats,
        action_stats=train_dataset.action_stats,
        max_episodes=args.max_val_episodes,
        shuffle=False,
    )
    val_loader = make_eval_loader(val_dataset, cfg)
    policy = CustomVLAPolicy(cfg).to(device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    normalization = stats_to_dict(train_dataset.state_stats, train_dataset.action_stats)

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
                save_checkpoint(policy, optimizer, cfg, step, last_metrics, out_dir, normalization)
            if step >= int(cfg["max_steps"]):
                break
    print(f"done step={step} output_dir={out_dir}")


if __name__ == "__main__":
    main()
