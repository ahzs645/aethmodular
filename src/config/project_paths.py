"""Project path helpers for repo-relative and environment-driven paths."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "research" / "ftir_hips_chem"


def get_project_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT


def get_data_root() -> Path:
    """
    Resolve data root using environment override.

    - `AETHMODULAR_DATA_ROOT` if set
    - fallback: `<repo>/research/ftir_hips_chem`
    """
    override = os.environ.get("AETHMODULAR_DATA_ROOT")
    return Path(override).expanduser().resolve() if override else DEFAULT_DATA_ROOT


def repo_path(*parts: str) -> Path:
    """Build a path under the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    """Build a path under the resolved data root."""
    return get_data_root().joinpath(*parts)
