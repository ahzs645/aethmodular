"""Shared I/O helpers and constants for the SPARTAN pipelines.

Every ``scripts/pipelines/spartan_*.py`` script used to carry its own copy of
the CSV header sniffer, the ``pd.read_csv`` wrapper, the site-code parser, the
case-insensitive column lookup, the raw/output directory constants and the
HIPS blank-filter rule. They now all live here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.paths import REPO_ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Cached mirror of http://data.spartan-network.org/GroupedByProduct/, laid out
# as RAW_DIR/<Product>/<SubProduct>/<Product>_<SubProduct>_<SITE>.csv
RAW_DIR = REPO_ROOT / "data" / "spartan" / "raw"

OUT_DIR = REPO_ROOT / "research" / "spartan" / "inventory"
FIG_DIR = OUT_DIR / "figures"

# Drive-shared HIPS bundle (folder 1YVmkYP_0pzs5TQwwbcTQi7gEJ-rTl9LZ).
HIPS_PATH = REPO_ROOT / "data" / "drive_bridge" / "Spartan" / "SPARTAN_HIPS_Batch1-51.v2.csv"
LOOKUP_PATH = REPO_ROOT / "data" / "drive_bridge" / "Spartan" / "SPARTAN_Site_quick_lookup.xlsx"


# ---------------------------------------------------------------------------
# Product / subproduct vocabulary
# ---------------------------------------------------------------------------

# Expected sample step in hours for each subproduct (None = filter-based, irregular).
EXPECTED_STEP_H: dict[str, float | None] = {
    "ChemSpecPM10": None,
    "ChemSpecPM25": None,
    "ReconstrPM25": None,
    "DailyScaPM10": 24.0,
    "DailyScaPM25": 24.0,
    "HourlyScaPM10": 1.0,
    "HourlyScaPM25": 1.0,
    "DailyEstPM25": 24.0,
    "HourlyEstPM25": 1.0,
}

# Subproducts that only exist at a nephelometer-equipped site.
NEPHEL_SUBS = {"DailyScaPM10", "DailyScaPM25", "HourlyScaPM10",
               "HourlyScaPM25", "DailyEstPM25", "HourlyEstPM25"}


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def find_header_line(path: Path, max_scan: int = 5) -> int:
    """Some SPARTAN CSVs start with 1-2 comment lines before the header row."""
    with open(path, "r", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                return 0
            stripped = line.strip()
            if not stripped:
                continue
            # header rows include a comma and start with a typical column token
            if "," in stripped and not stripped.lstrip().startswith("#"):
                # Heuristic: real CSV header rather than a free-text first line
                low = stripped.lower()
                if any(tok in low for tok in ("site_code", "year", "year_local")):
                    return i
    return 0


def read_spartan_csv(path: Path) -> pd.DataFrame:
    """Read a SPARTAN CSV, skipping any leading free-text/comment lines."""
    return pd.read_csv(path, skiprows=find_header_line(path), low_memory=False)


def site_from_path(path: Path) -> str:
    """Extract the 4-letter SPARTAN site code from ``<Product>_<SubProduct>_<SITE>.csv``."""
    return path.stem.rsplit("_", 1)[-1]


# ---------------------------------------------------------------------------
# Case-insensitive column lookup
# ---------------------------------------------------------------------------

def lower_col_map(df: pd.DataFrame) -> dict[str, str]:
    """Map each lowercased column name to the column as it actually appears."""
    return {c.lower(): c for c in df.columns}


def find_col(df: pd.DataFrame, *names: str) -> str | None:
    """Return the real column matching the first of ``names`` present (case-insensitive)."""
    cols = lower_col_map(df)
    for n in names:
        if n in cols:
            return cols[n]
    return None


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def build_datetime(df: pd.DataFrame) -> pd.Series:
    """Construct a datetime series from whatever year/month/day fields exist.

    Rows outside 2010-2030 are dropped: the ILNZ source files carry year
    21xx rows that otherwise distort every date-range and interval statistic.
    """
    y = find_col(df, "year_local", "start_year_local", "year")
    m = find_col(df, "month_local", "start_month_local", "month")
    d = find_col(df, "day_local", "start_day_local", "day")
    h = find_col(df, "hour_local", "start_hour_local", "hour")
    if not (y and m and d):
        return pd.Series([], dtype="datetime64[ns]")

    parts = {
        "year": pd.to_numeric(df[y], errors="coerce"),
        "month": pd.to_numeric(df[m], errors="coerce"),
        "day": pd.to_numeric(df[d], errors="coerce"),
    }
    if h:
        parts["hour"] = pd.to_numeric(df[h], errors="coerce").fillna(0).astype(int)
    frame = pd.DataFrame(parts).dropna(subset=["year", "month", "day"])
    frame = frame[(frame["year"] >= 2010) & (frame["year"] <= 2030)]
    return pd.to_datetime(frame, errors="coerce").dropna()


# ---------------------------------------------------------------------------
# HIPS (optical absorption) bundle
# ---------------------------------------------------------------------------

def normalize_fid(s: pd.Series) -> pd.Series:
    """Some HIPS rows carry the per-replicate suffix ('-1', '-2', '-3') while
    the public files key only to the base filter ('SITE-NNNN'). Strip the
    trailing replicate so we can join both forms.

    Same rule as research/ftir_hips_chem/scripts/data_matching.base_filter_id
    (kept as a local copy rather than an import so this operational pipeline
    does not depend on the research package -- see AGENTS.md on not mixing the
    two areas). Anchoring on the SITE-NNNN prefix matters twice over: a bare
    r"-(\\d)$" misses two-digit replicates ('ZAJB-0041-12'), while a bare
    r"-\\d+$" would strip the sample number from ids already in base form.
    """
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"^([A-Za-z]+-\d{4})-\d+$", r"\1", regex=True)
    )


def hips_blank_mask(df: pd.DataFrame) -> pd.Series:
    """Blank-filter rule for the HIPS table.

    Replicate "-7" is the SPARTAN field blank; "*-LB*" / numeric L-codes are lab blanks.
    """
    return (
        df["FilterId"].astype(str).str.endswith("-7")
        | df["FilterId"].astype(str).str.contains("-LB", regex=False)
        | df["SampleDate"].isna()
    )


def load_hips() -> pd.DataFrame:
    """Load the HIPS table with parsed dates, base filter ids and the blank flag."""
    df = pd.read_csv(HIPS_PATH)
    df["SampleDate"] = pd.to_datetime(df["SampleDate"], errors="coerce")
    df["FilterId_base"] = normalize_fid(df["FilterId"])
    df["is_blank"] = hips_blank_mask(df)
    return df
