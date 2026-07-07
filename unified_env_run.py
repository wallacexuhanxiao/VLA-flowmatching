#!/usr/bin/env python3
from __future__ import annotations

import importlib

from src.evaluation.libero_unified_env import install

install()
entry = importlib.import_module('rollout_' + 'libero')

if __name__ == '__main__':
    entry.main()
