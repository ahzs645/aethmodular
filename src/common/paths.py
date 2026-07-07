"""Repo-root discovery, consolidated from the copy-pasted path bootstrap.

Nearly every research subdir defined its own `find_repo_root` / `find_root`
to walk up from a notebook or script to the project root. This is the single
implementation.
"""

from __future__ import annotations

from pathlib import Path

# Markers that identify the repository root.
_ROOT_MARKERS = ("pyproject.toml", ".git", "environment.yml")


def find_repo_root(start=None, markers=_ROOT_MARKERS) -> Path:
    """Walk upward from `start` until a directory containing a root marker.

    `start` defaults to this file's location, so it works from inside src/ and
    from scripts that import it. Raises FileNotFoundError if no marker is found
    up to the filesystem root.
    """
    start = Path(start) if start is not None else Path(__file__).resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        if any((directory / m).exists() for m in markers):
            return directory
    raise FileNotFoundError(
        f"No repo-root marker {markers} found above {start}"
    )
