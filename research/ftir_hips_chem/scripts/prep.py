"""Small data-prep / bootstrap helpers shared across notebooks.

Consolidated from copies previously redefined inline in research subdirs:
- to_ugm3      (from research/catch_up)
- find_repo_root (the copy-pasted path bootstrap)
"""

from pathlib import Path

import pandas as pd

_ROOT_MARKERS = ("pyproject.toml", ".git", "environment.yml")


def to_ugm3(series, ng_threshold=100.0):
    """Coerce a mass-concentration series to ug/m3, auto-detecting ng/m3.

    Some source columns arrive in ng/m3. If the median absolute value exceeds
    ng_threshold the series is assumed ng/m3 and divided by 1000; otherwise it
    is returned unchanged. Non-numeric entries become NaN.
    """
    s = pd.to_numeric(series, errors="coerce")
    med = s.dropna().abs().median()
    if pd.notna(med) and med > ng_threshold:
        return s / 1000.0
    return s


def find_repo_root(start=None, markers=_ROOT_MARKERS):
    """Walk upward from `start` until a directory containing a root marker.

    `start` defaults to this file's location. Raises FileNotFoundError if no
    marker is found up to the filesystem root.
    """
    start = Path(start) if start is not None else Path(__file__).resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        if any((directory / m).exists() for m in markers):
            return directory
    raise FileNotFoundError(f"No repo-root marker {markers} found above {start}")
