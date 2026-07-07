#!/usr/bin/env python3
from __future__ import annotations

import train
from src.data.libero_unified_dataset import LiberoUnifiedChunkDataset, collate_libero_unified

train.LiberoChunkDataset = LiberoUnifiedChunkDataset
train.collate_libero = collate_libero_unified

if __name__ == "__main__":
    train.main()
