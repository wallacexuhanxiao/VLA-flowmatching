from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from src.config import load_config, save_json
from src.data.normalization import RunningStats, stats_from_dict
from src.models.vla_policy import CustomVLAPolicy


DEFAULT_STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


@dataclass
class RolloutResult:
    suite: str
    task_id: int
    task_name: str
    instruction: str
    episode_id: int
    seed: int
    success: bool
    steps: int
    decisions: int
    elapsed_sec: float
    avg_decision_latency_ms: float


def configure_headless_rendering() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def load_policy_for_rollout(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[CustomVLAPolicy, dict[str, Any], RunningStats, RunningStats]:
    cfg = load_config(config_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_cfg = payload.get("config", {})
    cfg.update({k: v for k, v in checkpoint_cfg.items() if k not in {"output_dir"}})
    policy = CustomVLAPolicy(cfg).to(device)
    policy.load_trainable_state_dict(payload["trainable"])
    policy.eval()
    if "normalization" not in payload:
        raise KeyError("Checkpoint is missing normalization statistics; rollout would use the wrong action scale.")
    state_stats, action_stats = stats_from_dict(payload["normalization"])
    return policy, cfg, state_stats, action_stats


def get_benchmark_suite(suite_name: str):
    configure_headless_rendering()
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise KeyError(f"Unknown LIBERO suite {suite_name!r}. Available: {sorted(benchmark_dict)}")
    return benchmark_dict[suite_name]()


def make_libero_env(
    suite,
    task_id: int,
    image_size: int = 256,
    camera_names: tuple[str, str] = ("agentview", "robot0_eye_in_hand"),
):
    configure_headless_rendering()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(bddl_file),
        "camera_heights": image_size,
        "camera_widths": image_size,
        "camera_names": list(camera_names),
    }
    return OffScreenRenderEnv(**env_args), task


def seed_env(env, seed: int) -> None:
    if hasattr(env, "seed"):
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reset_env(env, init_state: np.ndarray | None, seed: int) -> dict[str, Any]:
    seed_env(env, seed)
    obs = env.reset()
    if init_state is not None and hasattr(env, "set_init_state"):
        obs = env.set_init_state(init_state)
    return obs


def get_task_init_state(suite, task_id: int, episode_id: int) -> np.ndarray | None:
    if not hasattr(suite, "get_task_init_states"):
        return None
    init_states = suite.get_task_init_states(task_id)
    if init_states is None or len(init_states) == 0:
        return None
    return np.asarray(init_states[episode_id % len(init_states)])


def _to_rgb_image(value: Any, flip: bool = True) -> Image.Image:
    array = np.asarray(value)
    if flip:
        array = array[::-1]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return Image.fromarray(array[..., :3]).convert("RGB")


def image_from_obs(obs: dict[str, Any], key: str, flip: bool = True) -> Image.Image:
    if key in obs:
        return _to_rgb_image(obs[key], flip=flip)
    image_key = f"{key}_image"
    if image_key in obs:
        return _to_rgb_image(obs[image_key], flip=flip)
    available = sorted(k for k in obs if "image" in k)
    raise KeyError(f"Observation does not contain image key {key!r}. Image-like keys: {available}")


def state_from_obs(
    obs: dict[str, Any],
    state_keys: Iterable[str] = DEFAULT_STATE_KEYS,
    state_dim: int = 8,
) -> torch.Tensor:
    if "observation.state" in obs:
        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
    elif "state" in obs:
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    else:
        parts: list[np.ndarray] = []
        for key in state_keys:
            if key not in obs:
                raise KeyError(f"Observation is missing state key {key!r}. Available keys: {sorted(obs)}")
            value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            remaining = state_dim - sum(part.shape[0] for part in parts)
            if remaining == 1 and value.shape[0] > 1 and "gripper" in key:
                value = np.asarray([value.mean()], dtype=np.float32)
            parts.append(value)
        state = np.concatenate(parts, axis=0)
    if state.shape[0] > state_dim:
        state = state[:state_dim]
    if state.shape[0] < state_dim:
        state = np.pad(state, (0, state_dim - state.shape[0]))
    return torch.from_numpy(state.astype(np.float32, copy=False))


def build_policy_batch(
    obs: dict[str, Any],
    instruction: str,
    state_stats: RunningStats,
    state_dim: int,
    main_image_key: str,
    wrist_image_key: str,
    state_keys: Iterable[str],
    flip_images: bool,
) -> dict[str, Any]:
    raw_state = state_from_obs(obs, state_keys=state_keys, state_dim=state_dim)
    return {
        "main_images": [image_from_obs(obs, main_image_key, flip=flip_images)],
        "wrist_images": [image_from_obs(obs, wrist_image_key, flip=flip_images)],
        "texts": [instruction],
        "states": state_stats.normalize(raw_state).unsqueeze(0),
    }


def env_success(env, reward: float, done: bool, info: dict[str, Any]) -> bool:
    if isinstance(info, dict):
        for key in ("success", "is_success", "task_success"):
            if key in info:
                return bool(info[key])
    if hasattr(env, "check_success"):
        try:
            return bool(env.check_success())
        except Exception:
            pass
    return bool(done and reward > 0)


def step_env(env, action: np.ndarray):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, float(reward), bool(done), info


@torch.no_grad()
def predict_action_chunk(
    policy: CustomVLAPolicy,
    batch: dict[str, Any],
    device: torch.device,
    action_stats: RunningStats,
    ode_steps: int,
    sampler: str,
    seed: int,
    action_clip: float | None,
) -> tuple[np.ndarray, float]:
    generator = torch.Generator(device=device).manual_seed(seed)
    start = time.perf_counter()
    pred = policy.act(batch, device, ode_steps=ode_steps, sampler=sampler, generator=generator)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    raw = action_stats.denormalize(pred.float().cpu()).squeeze(0).numpy()
    if action_clip is not None:
        raw = np.clip(raw, -action_clip, action_clip)
    return raw.astype(np.float32), latency_ms


def maybe_write_video(frames: list[np.ndarray], path: str | Path, fps: int = 20) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def rollout_episode(
    *,
    env,
    task,
    suite_name: str,
    task_id: int,
    episode_id: int,
    seed: int,
    policy: CustomVLAPolicy,
    device: torch.device,
    state_stats: RunningStats,
    action_stats: RunningStats,
    state_dim: int,
    execute_horizon: int,
    max_steps: int,
    ode_steps: int,
    sampler: str,
    main_image_key: str,
    wrist_image_key: str,
    state_keys: Iterable[str],
    flip_images: bool,
    action_clip: float | None,
    init_state: np.ndarray | None,
    video_path: str | Path | None = None,
) -> RolloutResult:
    obs = reset_env(env, init_state=init_state, seed=seed)
    instruction = str(getattr(task, "language", getattr(task, "name", f"task_{task_id}")))
    task_name = str(getattr(task, "name", f"task_{task_id}"))
    frames: list[np.ndarray] = []
    step_count = 0
    decisions = 0
    latencies: list[float] = []
    success = False
    start = time.perf_counter()

    while step_count < max_steps and not success:
        batch = build_policy_batch(
            obs,
            instruction,
            state_stats,
            state_dim,
            main_image_key,
            wrist_image_key,
            state_keys,
            flip_images,
        )
        chunk, latency_ms = predict_action_chunk(
            policy,
            batch,
            device,
            action_stats,
            ode_steps,
            sampler,
            seed=seed * 100000 + decisions,
            action_clip=action_clip,
        )
        decisions += 1
        latencies.append(latency_ms)
        for action in chunk[:execute_horizon]:
            obs, reward, done, info = step_env(env, action)
            step_count += 1
            if video_path:
                frame = np.asarray(image_from_obs(obs, main_image_key, flip=flip_images))
                frames.append(frame)
            success = env_success(env, reward, done, info)
            if done or success or step_count >= max_steps:
                break

    elapsed = time.perf_counter() - start
    if video_path:
        maybe_write_video(frames, video_path)
    return RolloutResult(
        suite=suite_name,
        task_id=task_id,
        task_name=task_name,
        instruction=instruction,
        episode_id=episode_id,
        seed=seed,
        success=success,
        steps=step_count,
        decisions=decisions,
        elapsed_sec=elapsed,
        avg_decision_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
    )


def summarize_rollouts(results: list[RolloutResult]) -> dict[str, Any]:
    rows = [r.__dict__ for r in results]
    by_task: dict[str, list[RolloutResult]] = {}
    for result in results:
        by_task.setdefault(str(result.task_id), []).append(result)
    task_summary = {
        task: {
            "episodes": len(items),
            "success_rate": float(np.mean([x.success for x in items])) if items else 0.0,
            "avg_steps": float(np.mean([x.steps for x in items])) if items else 0.0,
            "avg_decision_latency_ms": float(np.mean([x.avg_decision_latency_ms for x in items])) if items else 0.0,
        }
        for task, items in sorted(by_task.items(), key=lambda kv: int(kv[0]))
    }
    return {
        "episodes": len(results),
        "success_rate": float(np.mean([r.success for r in results])) if results else 0.0,
        "task_summary": task_summary,
        "episodes_detail": rows,
    }


def save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
