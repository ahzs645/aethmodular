#!/usr/bin/env python3
"""Export the Machine learning section: the notebooks' model-fitting results, as they ran.

Everything on these pages needs a model fitted on spectra (PLS calibrations, component selection,
cross-validation, residual learners, model forms, bootstraps of refits), so it is precomputed and
static: the browser draws the notebooks' numbers and refits nothing. Each page is one module in
gallery/data/ml_pages/ that defines KEY and build() -> export_research_spectral.Page, and reads, in
order of preference, the CSV the notebook wrote (when its gitignored output/ exists), the table the
executed notebook displayed (committed with it), or a tracked CSV. A per-filter table only a re-run
writes is listed under the page's `missing` with the command that makes it.

Writes one JSON per page to gallery/app/public/data/ml/ (private: the section is left out of public
builds, gallery/app/src/sections/sections.json).

Run:  uv run python gallery/data/export_ml.py            every page
      uv run python gallery/data/export_ml.py residual   pages whose KEY starts with these
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def pages():
    """Every ml_pages module, in file-name order (modules starting with _ are helpers)."""
    for path in sorted((HERE / "ml_pages").glob("[!_]*.py")):
        module = importlib.import_module(f"ml_pages.{path.stem}")
        yield module.KEY, module.build


def main(only: list[str]) -> None:
    built = 0
    for key, build in pages():
        if only and not any(key.startswith(o) for o in only):
            continue
        build().write()
        built += 1
    if not built:
        raise SystemExit(f"no ML page matches {only}")


if __name__ == "__main__":
    main(sys.argv[1:])
