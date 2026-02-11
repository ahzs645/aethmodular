#!/usr/bin/env python3
"""Deprecated wrapper. Use scripts/diagnostics/check_etad_data.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    print("[DEPRECATED] Use scripts/diagnostics/check_etad_data.py")
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "diagnostics" / "check_etad_data.py"),
        run_name="__main__",
    )
