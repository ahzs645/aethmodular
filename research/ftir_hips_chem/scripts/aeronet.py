"""Shared loading and column helpers for AERONET portal exports."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

try:
    from config import AERONET_DATA_DIR
    from data_paths import AERONET_SUBDIR, maia_data_root
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.*
    from .config import AERONET_DATA_DIR
    from .data_paths import AERONET_SUBDIR, maia_data_root


AERONET_MISSING = -999.0

COLS = {
    "aod500": "AOD_500nm",
    "pw": "Precipitable_Water(cm)",
    "ae": "440-870_Angstrom_Exponent",
    "tau_f": "Fine_Mode_AOD_500nm[tau_f]",
    "tau_c": "Coarse_Mode_AOD_500nm[tau_c]",
    "fmf": "FineModeFraction_500nm[eta]",
}

_DATE_COLUMNS = ("Date(dd:mm:yyyy)", "Date_(dd:mm:yyyy)")


def find_header_line(path: str | Path, max_scan: int = 10) -> int:
    """Return the zero-based line containing an AERONET CSV header."""
    with open(path, "r", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                return 0
            stripped = line.strip()
            if not stripped:
                continue
            if "," in stripped and not stripped.lstrip().startswith("#"):
                low = stripped.lower()
                if any(date_col.lower() in low for date_col in _DATE_COLUMNS):
                    return i
    return 0


def aeronet_dir() -> Path:
    """Resolve the directory containing locally available AERONET exports."""
    env = os.environ.get("AETHMODULAR_AERONET_DIR")
    if env:
        return Path(env).expanduser()

    configured = Path(AERONET_DATA_DIR)
    if configured.is_dir() and any(configured.iterdir()):
        return configured

    return maia_data_root() / AERONET_SUBDIR


def load_aeronet(
    path: str | Path,
    kind: str = "aod",
    tz: str | None = None,
) -> pd.DataFrame:
    """Load an AOD or SDA export with a sorted date index and missing values."""
    if kind not in {"aod", "sda"}:
        raise ValueError("kind must be 'aod' or 'sda'")

    path = Path(path)
    df = pd.read_csv(
        path,
        skiprows=find_header_line(path),
        na_values=[AERONET_MISSING],
        low_memory=False,
    )
    date_col = next((column for column in _DATE_COLUMNS if column in df.columns), None)
    if date_col is None:
        raise KeyError(
            f"No AERONET date column found. Available columns: {list(df.columns)!r}"
        )

    dates = pd.to_datetime(df.pop(date_col), format="%d:%m:%Y")
    if tz is not None:
        dates = dates.dt.tz_localize(tz)
    df.index = pd.DatetimeIndex(dates, name="Date")
    df.replace(AERONET_MISSING, float("nan"), inplace=True)
    return df.sort_index()


def merge_aeronet(
    aod: pd.DataFrame | None = None,
    sda: pd.DataFrame | None = None,
    ssa: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Outer-merge available AERONET frames, preferring earlier arguments."""
    frames = [frame for frame in (aod, sda, ssa) if frame is not None]
    if not frames:
        return pd.DataFrame()

    merged = frames[0].copy()
    for frame in frames[1:]:
        merged = merged.combine_first(frame)
    return merged.sort_index()


def resolve_column(df: pd.DataFrame, alias: str) -> str:
    """Resolve a short AERONET alias to the actual column name in ``df``."""
    column = COLS.get(alias, alias)
    if column in df.columns:
        return column
    raise KeyError(
        f"AERONET column {column!r} (alias {alias!r}) is absent. "
        f"Available columns: {list(df.columns)!r}"
    )
