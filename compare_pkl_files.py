#!/usr/bin/env python3
"""Deprecated wrapper. Use scripts/diagnostics/compare_pkl_files.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    print("[DEPRECATED] Use scripts/diagnostics/compare_pkl_files.py")
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "diagnostics" / "compare_pkl_files.py"),
        run_name="__main__",
    )
