from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


EEF_POS_KEY = "robot" + "0_eef_pos"
EEF_QUAT_KEY = "robot" + "0_eef_quat"
GRIPPER_KEY = "robot" + "0_gripper_qpos"
DEFAULT_ENV_STATE_KEYS = (EEF_POS_KEY, EEF_QUAT_KEY, GRIPPER_KEY)
DEFAULT_STATE_LAYOUT = "eef_pos_axis_angle_gripper"


@dataclass(frozen=True)
class LiberoProtocol:
    state_dim: int = 8
    state_layout: str = DEFAULT_STATE_LAYOUT
    quat_format: str = "xyzw"
    main_image_key: str = "agentview_image"
    wrist_image_key: str = "robot" + "0_eye_in_hand_image"
    flip_env_images: bool = True


def to_float_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value, dtype=np.float32).copy())


def decode_lerobot_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported LeRobot image value type: {type(value)!r}")


def env_array_to_image(value: Any, flip: bool = True) -> Image.Image:
    array = np.asarray(value)
    if flip:
        array = array[::-1]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return Image.fromarray(array[..., :3]).convert("RGB")


def env_image_from_obs(obs: dict[str, Any], key: str, flip: bool = True) -> Image.Image:
    if key in obs:
        return env_array_to_image(obs[key], flip=flip)
    image_key = f"{key}_image"
    if image_key in obs:
        return env_array_to_image(obs[image_key], flip=flip)
    available = sorted(k for k in obs if "image" in k)
    raise KeyError(f"Observation does not contain image key {key!r}. Image-like keys: {available}")


def quat_to_axis_angle(quat: Any, quat_format: str = "xyzw") -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    if quat_format == "wxyz":
        q = np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float32)
    elif quat_format != "xyzw":
        raise ValueError(f"Unknown quat_format={quat_format!r}; expected 'xyzw' or 'wxyz'.")
    try:
        from robosuite.utils.transform_utils import quat2axisangle
        return np.asarray(quat2axisangle(q), dtype=np.float32).reshape(3)
    except Exception:
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return np.zeros(3, dtype=np.float32)
        q = q / norm
        xyz = q[:3]
        w = np.clip(q[3], -1.0, 1.0)
        sin_half = np.linalg.norm(xyz)
        if sin_half < 1e-8:
            return np.zeros(3, dtype=np.float32)
        angle = 2.0 * np.arctan2(sin_half, w)
        axis = xyz / sin_half
        return (axis * angle).astype(np.float32)


def make_eef_axis_angle_state(eef_pos: Any, eef_quat: Any, gripper_qpos: Any, *, quat_format: str = "xyzw") -> np.ndarray:
    pos = np.asarray(eef_pos, dtype=np.float32).reshape(3)
    axis_angle = quat_to_axis_angle(eef_quat, quat_format=quat_format)
    gripper = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)
    if gripper.shape[0] < 2:
        gripper = np.pad(gripper, (0, 2 - gripper.shape[0]))
    state = np.concatenate([pos, axis_angle, gripper[:2]], axis=0).astype(np.float32)
    if state.shape != (8,):
        raise ValueError(f"Expected canonical 8D state, got shape={state.shape}.")
    return state


def state_from_lerobot_value(value: Any, *, state_dim: int = 8, strict: bool = True) -> torch.Tensor:
    state = to_float_tensor(value).reshape(-1)
    if state.numel() == state_dim:
        return state
    if strict:
        raise ValueError(f"LeRobot state has dim={state.numel()}, expected {state_dim}.")
    if state.numel() > state_dim:
        return state[:state_dim]
    return torch.nn.functional.pad(state, (0, state_dim - state.numel()))


def state_from_env_obs(
    obs: dict[str, Any],
    *,
    state_dim: int = 8,
    state_layout: str = DEFAULT_STATE_LAYOUT,
    quat_format: str = "xyzw",
    state_keys: Iterable[str] = DEFAULT_ENV_STATE_KEYS,
) -> torch.Tensor:
    if state_layout == DEFAULT_STATE_LAYOUT:
        required = (EEF_POS_KEY, EEF_QUAT_KEY, GRIPPER_KEY)
        if all(key in obs for key in required):
            state = make_eef_axis_angle_state(obs[EEF_POS_KEY], obs[EEF_QUAT_KEY], obs[GRIPPER_KEY], quat_format=quat_format)
            return torch.from_numpy(state)
    if "observation.state" in obs:
        return state_from_lerobot_value(obs["observation.state"], state_dim=state_dim, strict=False)
    if "state" in obs:
        return state_from_lerobot_value(obs["state"], state_dim=state_dim, strict=False)
    if state_layout == "concat":
        parts: list[np.ndarray] = []
        for key in state_keys:
            if key not in obs:
                raise KeyError(f"Observation is missing state key {key!r}. Available keys: {sorted(obs)}")
            parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        state = np.concatenate(parts, axis=0)
        if state.shape[0] > state_dim:
            state = state[:state_dim]
        if state.shape[0] < state_dim:
            state = np.pad(state, (0, state_dim - state.shape[0]))
        return torch.from_numpy(state.astype(np.float32, copy=False))
    raise KeyError("Could not build rollout state from canonical LIBERO keys or a packed state.")


def build_env_policy_batch(
    obs: dict[str, Any],
    instruction: str,
    *,
    state_stats,
    state_dim: int,
    main_image_key: str,
    wrist_image_key: str,
    flip_images: bool,
    state_layout: str = DEFAULT_STATE_LAYOUT,
    quat_format: str = "xyzw",
    state_keys: Iterable[str] = DEFAULT_ENV_STATE_KEYS,
) -> dict[str, Any]:
    raw_state = state_from_env_obs(
        obs,
        state_dim=state_dim,
        state_layout=state_layout,
        quat_format=quat_format,
        state_keys=state_keys,
    )
    return {
        "main_images": [env_image_from_obs(obs, main_image_key, flip=flip_images)],
        "wrist_images": [env_image_from_obs(obs, wrist_image_key, flip=flip_images)],
        "texts": [instruction],
        "states": state_stats.normalize(raw_state).unsqueeze(0),
    }
