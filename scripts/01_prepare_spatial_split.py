#!/usr/bin/env python3
"""Create episode-level train/validation splits for LIBERO-Spatial.

The script is intentionally defensive because LeRobot dataset metadata can
change shape over time. It reads meta/tasks.jsonl and meta/episodes.jsonl when
available, then falls back to parquet metadata columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


SPATIAL_HINTS = ("libero_spatial", "spatial")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_spatial_text(value: object) -> bool:
    text = str(value or "").lower()
    return any(h in text for h in SPATIAL_HINTS)


def load_episode_rows(data_root: Path) -> list[dict]:
    meta = data_root / "meta"
    tasks = read_jsonl(meta / "tasks.jsonl")
    episodes = read_jsonl(meta / "episodes.jsonl")

    task_by_index = {}
    for task in tasks:
        idx = task.get("task_index", task.get("index", task.get("id")))
        if idx is not None:
            task_by_index[int(idx)] = task

    rows: list[dict] = []
    for ep in episodes:
        ep_idx = int(ep.get("episode_index", ep.get("index")))
        task_idx = ep.get("task_index")
        task = task_by_index.get(int(task_idx), {}) if task_idx is not None else {}
        merged = {**task, **ep}
        merged["episode_index"] = ep_idx
        rows.append(merged)

    if rows:
        return rows

    # LeRobot v3 metadata stores task names and episode metadata as parquet.
    tasks_parquet = meta / "tasks.parquet"
    task_index_by_name: dict[str, int] = {}
    if tasks_parquet.exists():
        tasks_df = pd.read_parquet(tasks_parquet)
        for task_name, row in tasks_df.iterrows():
            task_index_by_name[str(task_name)] = int(row["task_index"])

    episode_parquets = sorted((meta / "episodes").rglob("*.parquet"))
    parquet_rows: list[dict] = []
    for pq in episode_parquets:
        df = pd.read_parquet(pq)
        if "episode_index" not in df.columns or "tasks" not in df.columns:
            continue
        for record in df[["episode_index", "tasks", "length"]].to_dict("records"):
            tasks_value = record.get("tasks")
            if isinstance(tasks_value, (list, tuple)) and tasks_value:
                task_name = str(tasks_value[0])
            elif hasattr(tasks_value, "tolist"):
                values = tasks_value.tolist()
                task_name = str(values[0]) if values else ""
            else:
                task_name = str(tasks_value)
            parquet_rows.append(
                {
                    "episode_index": int(record["episode_index"]),
                    "task": task_name,
                    "task_index": task_index_by_name.get(task_name),
                    "length": int(record["length"]),
                }
            )
    if parquet_rows:
        return parquet_rows

    # Fallback: inspect parquet files for episode/task columns.
    seen: dict[int, dict] = {}
    for pq in sorted((data_root / "data").rglob("*.parquet")):
        try:
            df = pd.read_parquet(pq, columns=None)
        except Exception:
            continue
        if "episode_index" not in df.columns:
            continue
        cols = [c for c in df.columns if c in {"episode_index", "task_index", "task", "task_name", "language_instruction"}]
        for record in df[cols].drop_duplicates("episode_index").to_dict("records"):
            seen[int(record["episode_index"])] = record
    return list(seen.values())


def row_is_spatial(row: dict) -> bool:
    # HuggingFaceVLA/libero contains 40 tasks ordered as Long, Goal, Object,
    # Spatial in groups of 10. Tasks 30-39 are LIBERO-Spatial.
    if row.get("task_index") is not None:
        try:
            idx = int(row["task_index"])
            if 30 <= idx <= 39:
                return True
        except Exception:
            pass
    keys = ("suite", "benchmark", "task_suite", "task", "task_name", "name", "language_instruction", "instruction")
    return any(is_spatial_text(row.get(k)) for k in keys)


def task_key(row: dict) -> str:
    for k in ("task_index", "task", "task_name", "name", "language_instruction", "instruction"):
        if row.get(k) is not None:
            return str(row[k])
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    parser.add_argument("--out-dir", default=os.environ.get("SPLIT_DIR", "splits"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-all-fallback", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_episode_rows(data_root)
    if not rows:
        raise SystemExit(f"No episode metadata found under {data_root}")

    spatial = [r for r in rows if row_is_spatial(r)]
    if not spatial and args.allow_all_fallback:
        spatial = rows
    if not spatial:
        raise SystemExit("Could not identify LIBERO-Spatial episodes. Inspect meta/tasks.jsonl and rerun.")

    grouped: dict[str, list[int]] = defaultdict(list)
    for row in spatial:
        grouped[task_key(row)].append(int(row["episode_index"]))

    rng = random.Random(args.seed)
    train, val = [], []
    per_task = []
    for key, episodes in sorted(grouped.items()):
        episodes = sorted(set(episodes))
        rng.shuffle(episodes)
        n_val = max(1, round(len(episodes) * args.val_ratio)) if len(episodes) > 1 else 0
        val_eps = sorted(episodes[:n_val])
        train_eps = sorted(episodes[n_val:])
        train.extend(train_eps)
        val.extend(val_eps)
        per_task.append({"task": key, "n_total": len(episodes), "n_train": len(train_eps), "n_val": len(val_eps)})

    train = sorted(train)
    val = sorted(val)
    (out_dir / "libero_spatial_train_episodes.json").write_text(json.dumps(train))
    (out_dir / "libero_spatial_val_episodes.json").write_text(json.dumps(val))
    (out_dir / "libero_spatial_split_summary.json").write_text(
        json.dumps({"n_train": len(train), "n_val": len(val), "per_task": per_task}, indent=2)
    )
    with (out_dir / "libero_spatial_split_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "n_total", "n_train", "n_val"])
        writer.writeheader()
        writer.writerows(per_task)

    print(f"train={len(train)} val={len(val)} tasks={len(per_task)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
