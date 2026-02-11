#!/usr/bin/env python3
"""Run smoke checks for active notebooks.

Checks:
1. No legacy pre-reorg notebook path references in sources.
2. No machine-specific absolute `/Users/...` paths in sources.
3. Execute selected notebooks end-to-end with `nbclient`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOKS = [
    "notebooks/AAARpos.ipynb",
    "notebooks/ETAD_Aug13.ipynb",
    "notebooks/ETAD_comprehensive_absorption_analysis.ipynb",
    "notebooks/filter_data_availability_strip_chart.ipynb",
    "notebooks/warren_ratio_diagnostics.ipynb",
]

LEGACY_PATTERNS = [
    re.compile(r"\.\./FTIR_HIPS_Chem"),
    re.compile(r"/FTIR_HIPS_Chem(?:/|$)"),
    re.compile(r"aethmodular-clean/FTIR_HIPS_Chem"),
]
MACHINE_PATH_PATTERN = re.compile(r"/Users/")


def iter_cell_sources(nb: nbformat.NotebookNode) -> Iterable[str]:
    for cell in nb.cells:
        if "source" in cell:
            yield str(cell["source"])


def run_notebook(path: Path, timeout: int) -> None:
    with path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    combined_source = "\n".join(iter_cell_sources(nb))

    for pattern in LEGACY_PATTERNS:
        if pattern.search(combined_source):
            raise RuntimeError(
                f"legacy path reference found in source ({pattern.pattern})"
            )

    if MACHINE_PATH_PATTERN.search(combined_source):
        raise RuntimeError("machine-specific absolute path found in source (/Users/...)")

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def resolve_notebooks(all_root: bool, notebooks: List[str]) -> List[Path]:
    if notebooks:
        return [REPO_ROOT / n for n in notebooks]
    if all_root:
        return sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    return [REPO_ROOT / n for n in DEFAULT_NOTEBOOKS]


def summarize_error(exc: BaseException, limit: int = 320) -> str:
    text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
    return text[: limit - 3] + "..." if len(text) > limit else text


def main() -> int:
    parser = argparse.ArgumentParser(description="Run notebook smoke checks.")
    parser.add_argument(
        "--all-root",
        action="store_true",
        help="Run all notebooks/*.ipynb (non-archive).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-cell timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Optional notebook paths relative to repo root.",
    )
    args = parser.parse_args()

    notebook_paths = resolve_notebooks(args.all_root, args.notebooks)
    failures = []

    for path in notebook_paths:
        rel = path.relative_to(REPO_ROOT)
        if not path.exists():
            failures.append(f"{rel}: missing file")
            print(f"FAIL {rel} (missing file)")
            continue

        try:
            print(f"RUN  {rel}")
            run_notebook(path, timeout=args.timeout)
            print(f"PASS {rel}")
        except (RuntimeError, CellExecutionError, OSError, ValueError) as exc:
            message = summarize_error(exc)
            failures.append(f"{rel}: {message}")
            print(f"FAIL {rel} ({message})")

    if failures:
        print("\nNotebook smoke check failures:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nNotebook smoke check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
