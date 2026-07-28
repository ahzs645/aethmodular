"""Small data-prep / bootstrap helpers shared across notebooks.

Consolidated from copies previously redefined inline in research subdirs:
- to_ugm3      (from research/archive/catch_up)
- find_repo_root (the copy-pasted path bootstrap)
"""

from pathlib import Path

import pandas as pd

try:
    from config import DATA_ROOT, ETHIOPIA_SEASONS, season_for_month
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.*
    from .config import DATA_ROOT, ETHIOPIA_SEASONS, season_for_month

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


def output_dirs(slug, subdirs=("plots", "tables"), data_root=None) -> dict[str, Path]:
    """Create and return absolute output directories for a notebook or workflow.

    Directories follow the research workspace convention
    ``<data_root>/output/<subdir>/<slug>``.
    """
    root = Path(DATA_ROOT if data_root is None else data_root).expanduser().resolve()
    output_root = root / "output"
    directories = {
        subdir: (output_root / subdir / slug).resolve()
        for subdir in subdirs
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def add_calendar_columns(df, date_col=None, seasons=None, inplace=False) -> pd.DataFrame:
    """Add standard calendar columns from a date column or DatetimeIndex.

    The canonical Ethiopian seasons from :mod:`config` are used unless an
    explicit season mapping is supplied. Season mappings may use either the
    canonical ``{"months": [...]}`` value shape or a direct iterable of months.
    """
    result = df if inplace else df.copy()

    if date_col is None:
        if not isinstance(result.index, pd.DatetimeIndex):
            raise TypeError("date_col is required when df does not have a DatetimeIndex")
        dates = result.index
        result["Month"] = dates.month
        result["Hour"] = dates.hour
        result["DayOfWeek"] = dates.dayofweek
        result["DayOfYear"] = dates.dayofyear
    else:
        dates = pd.to_datetime(result[date_col])
        result["Month"] = dates.dt.month
        result["Hour"] = dates.dt.hour
        result["DayOfWeek"] = dates.dt.dayofweek
        result["DayOfYear"] = dates.dt.dayofyear

    season_definitions = ETHIOPIA_SEASONS if seasons is None else seasons
    if season_definitions is ETHIOPIA_SEASONS:
        result["season"] = result["Month"].map(season_for_month)
    else:
        month_to_season = {}
        for name, specification in season_definitions.items():
            months = (
                specification["months"]
                if isinstance(specification, dict)
                else specification
            )
            month_to_season.update({month: name for month in months})
        result["season"] = result["Month"].map(month_to_season)

    return result
