#!/usr/bin/env python3
"""Summarize LeRobot eval outputs into a flat CSV.

The parser accepts several likely JSON schemas so it remains useful as LeRobot
logs evolve. Unknown JSON files are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def extract_record(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    flat: dict = {}
    flatten("", data, flat)
    interesting = {}
    for key, value in flat.items():
        lk = key.lower()
        if any(token in lk for token in ["success", "reward", "episode", "latency", "fps", "time", "duration"]):
            interesting[key] = value
    if not interesting:
        return None
    m = re.search(r"k(\d+)", str(path))
    return {"path": str(path), "ode_steps": m.group(1) if m else "", **interesting}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for path in sorted(root.rglob("*.json")):
        rec = extract_record(path)
        if rec:
            records.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for rec in records for k in rec})
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {out}")


if __name__ == "__main__":
    main()
