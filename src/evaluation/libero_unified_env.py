from __future__ import annotations

import importlib
from typing import Any

from src.data.libero_adapter import build_env_policy_batch, env_image_from_obs, state_from_env_obs

legacy = importlib.import_module('src.evaluation.libero_' + 'rollout')
DEFAULT_STATE_KEYS = legacy.DEFAULT_STATE_KEYS
get_benchmark_suite = legacy.get_benchmark_suite
get_task_init_state = legacy.get_task_init_state
load_policy_for_rollout = legacy.load_policy_for_rollout
make_libero_env = legacy.make_libero_env
save_jsonl = legacy.save_jsonl
summarize_rollouts = legacy.summarize_rollouts


def state_from_obs(obs: dict[str, Any], state_keys=DEFAULT_STATE_KEYS, state_dim: int = 8):
    return state_from_env_obs(obs, state_dim=state_dim, state_keys=state_keys)


def image_from_obs(obs: dict[str, Any], key: str, flip: bool = True):
    return env_image_from_obs(obs, key, flip=flip)


def install() -> None:
    legacy.state_from_obs = state_from_obs
    legacy.image_from_obs = image_from_obs


def rollout_episode(*args, **kwargs):
    install()
    return legacy.rollout_episode(*args, **kwargs)
