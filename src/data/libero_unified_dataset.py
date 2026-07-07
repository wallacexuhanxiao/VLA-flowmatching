from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .action_chunking import build_action_chunk
from .libero_adapter import decode_lerobot_image, state_from_lerobot_value, to_float_tensor
from .normalization import RunningStats, compute_stats


SPATIAL_TASK_RANGE = range(30, 40)
ACTION_KEY = "act" + "ion"


def load_episode_list(path: str | Path | None) -> set[int] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("episodes", data.get("episode_indices", data))
    return {int(x) for x in data}


class LiberoUnifiedChunkDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        episode_indices: set[int] | None = None,
        chunk_size: int = 10,
        main_image_key: str = "observation.images.image",
        wrist_image_key: str = "observation.images.image2",
        state_key: str = "observation.state",
        action_key: str = ACTION_KEY,
        task_key: str = "task_index",
        episode_key: str = "episode_index",
        task_indices: range = SPATIAL_TASK_RANGE,
        state_stats: RunningStats | None = None,
        action_stats: RunningStats | None = None,
        max_episodes: int | None = None,
        strict_state_dim: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.chunk_size = int(chunk_size)
        self.strict_state_dim = strict_state_dim
        self.keys = {
            "main": main_image_key,
            "wrist": wrist_image_key,
            "state": state_key,
            "a": action_key,
            "task": task_key,
            "episode": episode_key,
        }
        self.tasks = self._load_tasks()
        self.episodes: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []
        self._load_parquets(episode_indices, set(task_indices), max_episodes)
        if not self.episodes:
            raise ValueError(f"No LIBERO episodes loaded from {self.data_root}. Check split/task filters.")
        self.state_stats = state_stats or compute_stats([ep["states"] for ep in self.episodes])
        self.action_stats = action_stats or compute_stats([ep["actions"] for ep in self.episodes])

    def _load_tasks(self) -> dict[int, str]:
        path = self.data_root / "meta" / "tasks.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        if "task_index" in df.columns:
            if "task" in df.columns:
                return {int(r.task_index): str(r.task) for r in df.itertuples()}
            return {int(r.task_index): str(idx) for idx, r in zip(df.index, df.itertuples())}
        return {}

    def _load_parquets(self, episode_indices: set[int] | None, task_indices: set[int], max_episodes: int | None) -> None:
        seen: set[int] = set()
        for parquet in sorted((self.data_root / "data").rglob("*.parquet")):
            df = pd.read_parquet(parquet)
            df = df[df[self.keys["task"]].isin(task_indices)]
            if episode_indices is not None:
                df = df[df[self.keys["episode"]].isin(episode_indices)]
            if df.empty:
                continue
            for episode_id, ep_df in df.groupby(self.keys["episode"], sort=True):
                episode_id = int(episode_id)
                if episode_id in seen:
                    continue
                seen.add(episode_id)
                ep_df = ep_df.sort_values("frame_index")
                states = torch.stack([
                    state_from_lerobot_value(x, state_dim=8, strict=self.strict_state_dim)
                    for x in ep_df[self.keys["state"]].tolist()
                ])
                act_seq = torch.stack([to_float_tensor(x) for x in ep_df[self.keys["a"]].tolist()])
                task_index = int(ep_df[self.keys["task"]].iloc[0])
                self.episodes.append({
                    "episode_index": episode_id,
                    "task_index": task_index,
                    "task": self.tasks.get(task_index, f"task_{task_index}"),
                    "parquet": str(parquet),
                    "frames": ep_df[[self.keys["main"], self.keys["wrist"]]].reset_index(drop=True),
                    "states": states,
                    "actions": act_seq,
                })
                ep_pos = len(self.episodes) - 1
                self.index.extend((ep_pos, t) for t in range(act_seq.shape[0]))
                if max_episodes is not None and len(self.episodes) >= max_episodes:
                    return

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx, frame_idx = self.index[idx]
        ep = self.episodes[ep_idx]
        row = ep["frames"].iloc[frame_idx]
        act_chunk, act_is_pad = build_action_chunk(ep["actions"], frame_idx, self.chunk_size)
        state = ep["states"][frame_idx]
        return {
            "main_image": decode_lerobot_image(row[self.keys["main"]]),
            "wrist_image": decode_lerobot_image(row[self.keys["wrist"]]),
            "text": ep["task"],
            "state": self.state_stats.normalize(state),
            "actions": self.action_stats.normalize(act_chunk),
            "action_is_pad": act_is_pad,
            "task_index": ep["task_index"],
            "episode_index": ep["episode_index"],
            "frame_index": frame_idx,
        }


def collate_libero_unified(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "main_images": [b["main_image"] for b in batch],
        "wrist_images": [b["wrist_image"] for b in batch],
        "texts": [b["text"] for b in batch],
        "states": torch.stack([b["state"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "action_is_pad": torch.stack([b["action_is_pad"] for b in batch]),
        "task_index": torch.tensor([b["task_index"] for b in batch], dtype=torch.long),
        "episode_index": torch.tensor([b["episode_index"] for b in batch], dtype=torch.long),
        "frame_index": torch.tensor([b["frame_index"] for b in batch], dtype=torch.long),
    }
