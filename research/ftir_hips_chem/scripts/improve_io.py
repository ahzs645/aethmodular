"""Load, clean, and cache FED Query Wizard IMPROVE exports."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import (
        IMPROVE_DATA_DIR,
        IMPROVE_DEPOSIT_AREA_CM2,
        IMPROVE_HIGH_FABS_AREAS_CM2,
        SPARTAN_DEPOSIT_AREA_CM2,
    )
    from data_paths import IMPROVE_SUBDIR, maia_data_root
    from prep import find_repo_root
except ImportError:  # Support importing as research.ftir_hips_chem.scripts.*
    from .config import (
        IMPROVE_DATA_DIR,
        IMPROVE_DEPOSIT_AREA_CM2,
        IMPROVE_HIGH_FABS_AREAS_CM2,
        SPARTAN_DEPOSIT_AREA_CM2,
    )
    from .data_paths import IMPROVE_SUBDIR, maia_data_root
    from .prep import find_repo_root


NA_VALUES = ["-999", -999, "-999.0", -999.0]
CHEMISTRY_COLS = [
    "Dataset",
    "SiteCode",
    "POC",
    "Date",
    "AuxID",
    "ECf_Val",
    "OCf_Val",
    "fAbs_Val",
    "FlowRate_Val",
    "FEf_Val",
    "MF_Val",
    "SampDur_Val",
    "SOILf_Val",
]
RT_REQUIRED = {"SiteCode", "Date", "RefF_635_Val", "TransF_635_Val"}
METADATA_SHEETS = [
    "Datasets",
    "Overview",
    "Sites",
    "Parameters",
    "Site History",
    "Dataset History",
    "Status Flags",
    "Provider Flags",
]

_NUMERIC_COLS = [
    "POC",
    "AuxID",
    "ECf_Val",
    "OCf_Val",
    "fAbs_Val",
    "FlowRate_Val",
    "FEf_Val",
    "MF_Val",
    "SampDur_Val",
    "SOILf_Val",
]


def improve_dir() -> Path:
    """Resolve the directory containing locally available IMPROVE exports."""
    env = os.environ.get("AETHMODULAR_IMPROVE_DIR")
    if env:
        return Path(env).expanduser()

    configured = Path(IMPROVE_DATA_DIR)
    if configured.is_dir() and any(configured.iterdir()):
        return configured

    return maia_data_root() / IMPROVE_SUBDIR


def improve_clean_path() -> Path:
    """Return the canonical cached cleaned IMPROVE CSV path."""
    return (
        find_repo_root()
        / "research"
        / "ftir_hips_chem"
        / "output"
        / "improve_high_fabs_comparison"
        / "improve_valid_cleaned.csv"
    )


def safe_area(area):
    """Convert a numeric deposit area to its derived-column name fragment."""
    return str(area).replace(".", "p")


def _read_text_file(path: Path) -> str:
    return path.read_text(errors="replace")


def _iter_text_sources(directory: Path):
    patterns = ["*.txt", "*.TXT", "*.csv", "*.CSV", "*.psv", "*.PSV", "*.zip", "*.ZIP"]
    seen = set()
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    for member in zf.namelist():
                        if member.endswith("/"):
                            continue
                        if not member.lower().endswith((".txt", ".csv", ".psv")):
                            continue
                        data = zf.read(member).decode("utf-8", errors="replace")
                        yield f"{path}::{member}", data
            else:
                yield str(path), _read_text_file(path)


def _find_fed_data_header(lines):
    """Return the header line index for a FED multi-section export."""
    for i, line in enumerate(lines):
        if line.strip() == "Data":
            for j in range(i + 1, min(i + 20, len(lines))):
                if lines[j].lstrip().startswith("Dataset") and "SiteCode" in lines[j]:
                    return j
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Dataset") and "SiteCode" in line:
            return i
    return None


def _parse_table_from_text(source_name: str, text: str):
    lines = text.splitlines()
    header_idx = _find_fed_data_header(lines)

    if header_idx is not None:
        header = lines[header_idx].strip()
        cols = header.split("|") if "|" in header else header.split()
        data_text = "\n".join(lines[header_idx + 1 :])
        sep = "|" if "|" in header else r"\s+"
        df = pd.read_csv(
            io.StringIO(data_text),
            sep=sep,
            names=cols,
            engine="python",
            na_values=NA_VALUES,
            keep_default_na=True,
        )
        return df, {
            "source": source_name,
            "format": "fed_multisection",
            "header_line": header_idx + 1,
            "columns": cols,
        }

    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return pd.DataFrame(), {
            "source": source_name,
            "format": "empty",
            "header_line": None,
            "columns": [],
        }
    first = nonempty[0]
    sep = "|" if "|" in first else r"\s+"
    df = pd.read_csv(
        io.StringIO("\n".join(nonempty)),
        sep=sep,
        engine="python",
        na_values=NA_VALUES,
        keep_default_na=True,
    )
    return df, {
        "source": source_name,
        "format": "simple_delimited",
        "header_line": 1,
        "columns": list(df.columns),
    }


def _normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace(NA_VALUES, np.nan)
    return df


def _classify_table(df):
    cols = set(df.columns)
    if {"SiteCode", "Date", "ECf_Val", "fAbs_Val"}.issubset(cols):
        return "chemistry"
    if RT_REQUIRED.issubset(cols):
        return "reflectance_transmittance"
    return "ignored"


def _append_role_frame(
    df,
    source_name,
    meta,
    chemistry_frames,
    rt_frames,
):
    df = _normalise_columns(df)
    role = _classify_table(df)
    meta.update({"rows": len(df), "status": "parsed", "role": role})

    if role == "chemistry":
        keep = [column for column in CHEMISTRY_COLS if column in df.columns]
        tmp = df[keep].copy()
        tmp["source_file"] = source_name
        chemistry_frames.append(tmp)
    elif role == "reflectance_transmittance":
        tmp = df.copy()
        tmp["source_file"] = source_name
        rt_frames.append(tmp)

    return meta


def read_improve_exports(
    directory,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read chemistry and optical exports plus a parse manifest.

    Parameters
    ----------
    directory
        Directory containing Query Wizard workbooks or delimited FED exports.

    Returns
    -------
    tuple of pandas.DataFrame
        Chemistry rows, reflectance/transmittance rows, and a parse manifest.
    """
    directory = Path(directory)
    chemistry_frames = []
    rt_frames = []
    manifest = []

    for path in sorted(list(directory.glob("*.xlsx")) + list(directory.glob("*.XLSX"))):
        try:
            xl = pd.ExcelFile(path)
            if "Data" not in xl.sheet_names:
                manifest.append(
                    {
                        "source": str(path),
                        "format": "excel_workbook",
                        "status": "ignored_no_data_sheet",
                        "role": "ignored",
                        "rows": 0,
                    }
                )
                continue
            df = pd.read_excel(xl, sheet_name="Data", na_values=NA_VALUES)
            meta = {
                "source": str(path),
                "format": "excel_sheet",
                "sheet": "Data",
                "header_line": 1,
                "columns": list(df.columns),
                "workbook_sheets": "; ".join(xl.sheet_names),
            }
            manifest.append(
                _append_role_frame(
                    df,
                    str(path),
                    meta,
                    chemistry_frames,
                    rt_frames,
                )
            )
        except Exception as exc:
            manifest.append(
                {
                    "source": str(path),
                    "format": "excel_sheet",
                    "status": "parse_failed",
                    "role": "unknown",
                    "error": repr(exc),
                }
            )

    for source_name, text in _iter_text_sources(directory):
        try:
            df, meta = _parse_table_from_text(source_name, text)
            manifest.append(
                _append_role_frame(
                    df,
                    source_name,
                    meta,
                    chemistry_frames,
                    rt_frames,
                )
            )
        except Exception as exc:
            manifest.append(
                {
                    "source": source_name,
                    "format": "text_or_zip",
                    "status": "parse_failed",
                    "role": "unknown",
                    "error": repr(exc),
                }
            )

    chemistry = (
        pd.concat(chemistry_frames, ignore_index=True)
        if chemistry_frames
        else pd.DataFrame()
    )
    rt = pd.concat(rt_frames, ignore_index=True) if rt_frames else pd.DataFrame()
    manifest_df = pd.DataFrame(manifest)
    return chemistry, rt, manifest_df


def read_improve_metadata(directory) -> dict[str, pd.DataFrame]:
    """Read standard metadata sheets from all Query Wizard workbooks."""
    directory = Path(directory)
    tables = {sheet: [] for sheet in METADATA_SHEETS}
    for path in sorted(list(directory.glob("*.xlsx")) + list(directory.glob("*.XLSX"))):
        try:
            xl = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in METADATA_SHEETS:
            if sheet not in xl.sheet_names:
                continue
            try:
                df = pd.read_excel(xl, sheet_name=sheet, na_values=NA_VALUES)
            except Exception:
                continue
            df = _normalise_columns(df)
            df["source_file"] = str(path)
            tables[sheet].append(df)
    return {
        sheet: pd.concat(frames, ignore_index=True)
        for sheet, frames in tables.items()
        if frames
    }


def numeric_clean(df, cols):
    """Coerce selected columns to numeric values in place."""
    for column in cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _default_areas():
    """Return the deposit-area sweep used by improve_high_fabs_comparison.

    Taken from a dedicated constant rather than reshaped from
    ``IMPROVE_AREA_SENSITIVITY_CM2``: that tuple belongs to a different notebook
    and differs in its middle value (3.53 vs 3.5), so deriving one from the other
    would break silently if either were edited.
    """
    return list(IMPROVE_HIGH_FABS_AREAS_CM2)


def clean_improve(
    chemistry,
    rt=None,
    metadata=None,
    *,
    areas=None,
    primary_area=None,
) -> pd.DataFrame:
    """Clean IMPROVE chemistry rows and derive loading and ratio columns.

    Parameters
    ----------
    chemistry
        Chemistry rows returned by :func:`read_improve_exports`.
    rt
        Optional reflectance/transmittance rows to join by FED sample key.
    metadata
        Optional metadata mapping returned by :func:`read_improve_metadata`.
    areas
        IMPROVE deposit areas used for loading sensitivity columns.
    primary_area
        Deposit area used for ``EC_loading_ug_cm2_primary``.

    Returns
    -------
    pandas.DataFrame
        Positive EC and fAbs rows with notebook-equivalent derived columns.
    """
    if areas is None:
        areas = _default_areas()
    if primary_area is None:
        primary_area = IMPROVE_DEPOSIT_AREA_CM2
    if metadata is None:
        metadata = {}
    if rt is None:
        rt = pd.DataFrame()

    improve = chemistry.copy()
    improve = numeric_clean(improve, _NUMERIC_COLS)
    improve["Date"] = pd.to_datetime(improve["Date"], errors="coerce")

    key_cols = [
        column
        for column in ["Dataset", "SiteCode", "POC", "Date", "AuxID"]
        if column in improve.columns
    ]
    improve["_source_priority"] = (
        improve["source_file"].astype(str).str.lower().str.endswith(".xlsx").astype(int)
    )
    improve["_required_nonmissing"] = improve[
        [column for column in _NUMERIC_COLS if column in improve.columns]
    ].notna().sum(axis=1)
    if key_cols:
        improve = improve.sort_values(
            ["_source_priority", "_required_nonmissing"]
        ).drop_duplicates(subset=key_cols, keep="last")
    improve = improve.drop(
        columns=["_source_priority", "_required_nonmissing"],
        errors="ignore",
    )

    valid = improve.dropna(subset=["SiteCode", "Date", "ECf_Val", "fAbs_Val"]).copy()
    valid = valid[(valid["ECf_Val"] > 0) & (valid["fAbs_Val"] > 0)].copy()

    valid["volume_m3"] = np.nan
    flow_ok = (
        valid["FlowRate_Val"].notna()
        & valid["SampDur_Val"].notna()
        & (valid["FlowRate_Val"] > 0)
        & (valid["SampDur_Val"] > 0)
    )
    valid.loc[flow_ok, "volume_m3"] = (
        valid.loc[flow_ok, "FlowRate_Val"]
        * valid.loc[flow_ok, "SampDur_Val"]
        / 1000.0
    )
    valid["EC_loading_ug"] = valid["ECf_Val"] * valid["volume_m3"]
    valid["MF_loading_ug"] = valid["MF_Val"] * valid["volume_m3"]

    for area in list(areas) + [SPARTAN_DEPOSIT_AREA_CM2]:
        column = f"EC_loading_ug_cm2_area_{safe_area(area)}"
        valid[column] = valid["EC_loading_ug"] / area
    valid["EC_loading_ug_cm2_primary"] = valid[
        f"EC_loading_ug_cm2_area_{safe_area(primary_area)}"
    ]

    valid["fAbs_per_EC"] = valid["fAbs_Val"] / valid["ECf_Val"]
    valid["OC_EC"] = valid["OCf_Val"] / valid["ECf_Val"]
    valid["FE_EC"] = valid["FEf_Val"] / valid["ECf_Val"]
    valid["SOIL_EC"] = valid["SOILf_Val"] / valid["ECf_Val"]
    valid["MF_EC"] = valid["MF_Val"] / valid["ECf_Val"]
    valid["year"] = valid["Date"].dt.year
    valid["month"] = valid["Date"].dt.month
    valid["year_month"] = valid["Date"].dt.to_period("M").astype(str)
    valid["post_2017_hips_processing"] = valid["Date"] >= pd.Timestamp("2017-01-01")

    if "Sites" in metadata:
        sites = metadata["Sites"].copy()
        site_keep = [
            column
            for column in [
                "Code",
                "Site",
                "Country",
                "State",
                "County",
                "Latitude",
                "Longitude",
                "LandUseCode",
                "DemographicCode",
                "Sponsor",
            ]
            if column in sites.columns
        ]
        sites = (
            sites[site_keep]
            .drop_duplicates(subset=["Code"])
            .rename(columns={"Code": "SiteCode", "Site": "SiteName"})
        )
        valid = valid.merge(sites, on="SiteCode", how="left")

    if not rt.empty:
        rt = rt.copy()
        rt_num_cols = [
            column
            for column in rt.columns
            if column.endswith("_Val") or column in ["POC", "AuxID"]
        ]
        rt = numeric_clean(rt, rt_num_cols)
        rt["Date"] = pd.to_datetime(rt["Date"], errors="coerce")
        rt_key_cols = [
            column
            for column in ["Dataset", "SiteCode", "POC", "Date", "AuxID"]
            if column in rt.columns and column in valid.columns
        ]
        rt["_source_priority"] = (
            rt["source_file"].astype(str).str.lower().str.endswith(".xlsx").astype(int)
        )
        rt["_nonmissing"] = rt.notna().sum(axis=1)
        if rt_key_cols:
            rt = rt.sort_values(
                ["_source_priority", "_nonmissing"]
            ).drop_duplicates(subset=rt_key_cols, keep="last")
            rt_keep = (
                rt_key_cols
                + [column for column in rt.columns if column.endswith("_635_Val")]
                + ["source_file"]
            )
            rt = rt[rt_keep].rename(columns={"source_file": "rt_source_file"})
            valid = valid.merge(rt, on=rt_key_cols, how="left")
    valid["rt_available"] = (
        valid[
            [column for column in valid.columns if column.endswith("_635_Val")]
        ].notna().any(axis=1)
        if any(column.endswith("_635_Val") for column in valid.columns)
        else False
    )
    return valid


def load_improve_clean(path=None, *, rebuild=False, usecols=None) -> pd.DataFrame:
    """Load the cleaned IMPROVE cache, rebuilding it from exports if needed.

    Parameters
    ----------
    path
        Optional cache CSV path. The canonical research output path is the default.
    rebuild
        Rebuild the cache even if it already exists.
    usecols
        Column selection passed directly to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Cached cleaned IMPROVE rows.

    Raises
    ------
    FileNotFoundError
        If neither the cache nor the resolved IMPROVE source directory exists.
    RuntimeError
        If the source directory contains no recognized chemistry exports.
    """
    cache_path = improve_clean_path() if path is None else Path(path)
    if cache_path.is_file() and not rebuild:
        return pd.read_csv(cache_path, usecols=usecols, low_memory=False)

    source_dir = improve_dir()
    if not source_dir.is_dir():
        raise FileNotFoundError(
            "Cleaned IMPROVE cache was not found at "
            f"{cache_path}, and the source directory is unavailable at {source_dir}. "
            "Set AETHMODULAR_IMPROVE_DIR to the directory containing FED Query "
            "Wizard exports."
        )

    chemistry, rt, _manifest = read_improve_exports(source_dir)
    if chemistry.empty:
        raise RuntimeError(
            f"No IMPROVE chemistry exports with ECf_Val and fAbs_Val were found in "
            f"{source_dir}."
        )
    metadata = read_improve_metadata(source_dir)
    valid = clean_improve(chemistry, rt=rt, metadata=metadata)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(cache_path, index=False)
    return pd.read_csv(cache_path, usecols=usecols, low_memory=False)
