#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import save_json
from src.evaluation.libero_rollout import (
    DEFAULT_STATE_KEYS,
    get_benchmark_suite,
    get_task_init_state,
    load_policy_for_rollout,
    make_libero_env,
    rollout_episode,
    save_jsonl,
    summarize_rollouts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop LIBERO rollout for the custom VLA policy.")
    parser.add_argument("--config", default="configs/flow_matching.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(10)), help="LIBERO suite-local task ids.")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--execute-horizon", type=int)
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--sampler", choices=["euler", "heun"], default="euler")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--main-camera", default="agentview")
    parser.add_argument("--wrist-camera", default="robot0_eye_in_hand")
    parser.add_argument("--main-image-key", default="agentview_image")
    parser.add_argument("--wrist-image-key", default="robot0_eye_in_hand_image")
    parser.add_argument("--state-keys", nargs="*", default=list(DEFAULT_STATE_KEYS))
    parser.add_argument("--no-flip-images", action="store_true")
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cfg, state_stats, action_stats = load_policy_for_rollout(args.config, args.checkpoint, device)
    execute_horizon = args.execute_horizon or int(cfg.get("execute_horizon", 10))
    state_dim = int(cfg.get("state_dim", 8))

    suite = get_benchmark_suite(args.suite)
    results = []
    for task_id in args.tasks:
        env, task = make_libero_env(
            suite,
            task_id,
            image_size=args.image_size,
            camera_names=(args.main_camera, args.wrist_camera),
        )
        try:
            for episode_id in range(args.episodes_per_task):
                seed = args.seed + task_id * 1000 + episode_id
                init_state = get_task_init_state(suite, task_id, episode_id)
                video_path = None
                if args.save_videos:
                    video_root = Path(args.video_dir or Path(args.out or "rollout_results.json").with_suffix("").as_posix() + "_videos")
                    video_path = video_root / f"task{task_id:02d}_ep{episode_id:02d}.mp4"
                result = rollout_episode(
                    env=env,
                    task=task,
                    suite_name=args.suite,
                    task_id=task_id,
                    episode_id=episode_id,
                    seed=seed,
                    policy=policy,
                    device=device,
                    state_stats=state_stats,
                    action_stats=action_stats,
                    state_dim=state_dim,
                    execute_horizon=execute_horizon,
                    max_steps=args.max_steps,
                    ode_steps=args.ode_steps,
                    sampler=args.sampler,
                    main_image_key=args.main_image_key,
                    wrist_image_key=args.wrist_image_key,
                    state_keys=args.state_keys,
                    flip_images=not args.no_flip_images,
                    action_clip=args.action_clip,
                    init_state=init_state,
                    video_path=video_path,
                )
                results.append(result)
                print(
                    "rollout",
                    f"task={task_id}",
                    f"episode={episode_id}",
                    f"success={int(result.success)}",
                    f"steps={result.steps}",
                    f"latency_ms={result.avg_decision_latency_ms:.2f}",
                    flush=True,
                )
        finally:
            if hasattr(env, "close"):
                env.close()

    summary = summarize_rollouts(results)
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "suite": args.suite,
            "tasks": args.tasks,
            "episodes_per_task": args.episodes_per_task,
            "ode_steps": args.ode_steps,
            "sampler": args.sampler,
            "execute_horizon": execute_horizon,
            "max_steps": args.max_steps,
        }
    )
    print(summary)
    if args.out:
        save_json(summary, args.out)
    if args.jsonl:
        save_jsonl(summary["episodes_detail"], args.jsonl)


if __name__ == "__main__":
    main()

