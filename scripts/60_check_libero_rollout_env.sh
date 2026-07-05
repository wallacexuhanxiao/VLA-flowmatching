#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import importlib.util
import os

mods = ("libero", "robosuite", "mujoco", "torch", "transformers", "imageio")
for name in mods:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'OK' if spec else 'MISSING'}")

try:
    import torch

    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"torch_probe_error: {type(exc).__name__}: {exc}")

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL')}")
print(f"PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM')}")

try:
    from libero.libero import benchmark

    keys = sorted(benchmark.get_benchmark_dict().keys())
    print(f"libero_benchmarks: {keys}")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    print(f"libero_spatial_tasks: {suite.n_tasks}")
    task = suite.get_task(0)
    print(f"task0_name: {getattr(task, 'name', None)}")
    print(f"task0_language: {getattr(task, 'language', None)}")
except Exception as exc:
    print(f"libero_probe_error: {type(exc).__name__}: {exc}")
PY

