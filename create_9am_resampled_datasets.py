#!/usr/bin/env python3
"""Deprecated wrapper. Use scripts/pipelines/create_9am_resampled_datasets.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    print("[DEPRECATED] Use scripts/pipelines/create_9am_resampled_datasets.py")
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "pipelines" / "create_9am_resampled_datasets.py"),
        run_name="__main__",
    )
