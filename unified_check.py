#!/usr/bin/env python3
from __future__ import annotations

import evaluate
from src.data.libero_unified_dataset import LiberoUnifiedChunkDataset, collate_libero_unified

evaluate.LiberoChunkDataset = LiberoUnifiedChunkDataset
evaluate.collate_libero = collate_libero_unified

if __name__ == "__main__":
    evaluate.main()
