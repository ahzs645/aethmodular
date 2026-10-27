"""The aethalometer's quality-control columns per daily bin, for the Instruments section's "Data quality" page.

Run with:  uv run python gallery/data/export_aeth_quality.py   (after gallery/data/export_aethalometer.py)

Inputs (in the repo):
  research/ftir_hips_chem/processed_sites/df_<Site>_9am_resampled.pkl   the same daily aggregates as export_aethalometer.py
  research/ftir_hips_chem/scripts/config.py                           FLOW_FIX_PERIODS, SMOOTH_RAW_THRESHOLDS
  src/data/qc/pkl_cleaning.py                                         the QC bounds the minute cleaning applied

The daily pickles carry the MA350's diagnostic columns that export_aethalometer.py leaves out: flows, attenuation
on both spots, the attenuation rate, the dual-spot ratios, the roughness metric and the smoothed BCc (the cleaning pipeline's DEMA, src/data/qc/pkl_cleaning.py);
at Jacros also the raw spot concentrations and the loading factor K. They are daily means of minute values (the
same legacy bins, labelled at their start; docs/matched-sample-audit-2026-09-10.md), so a ratio here is the daily
mean of a minute ratio as the cleaning computed it, not a ratio of daily means. Bins are matched to
aethalometer/daily/<CODE>.json by their start (the same parsing, order and de-duplication).

Writes to gallery/app/public/data/aethalometer/quality/ (private with the rest of aethalometer/):
  index.json      flow-fix periods per site, the smooth/raw thresholds, the cleaning bounds, which series each site has
  <CODE>.json     per site: bin starts and one array per series (null where the pickle has no value)
"""

from __future__ import annotations

import inspect
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_aethalometer as aeth  # noqa: E402  (also puts research/ftir_hips_chem/scripts on the path)
from config import FLOW_FIX_PERIODS, SMOOTH_RAW_THRESHOLDS  # noqa: E402

ROOT = aeth.ROOT
OUT = aeth.OUT / "quality"
CHANNELS = aeth.CHANNELS

# series key -> (pickle column(s), first found wins; scale; how to round)
#   ("d", n): n decimals   ("s", n): n significant figures (rates and roughness span decades)
BASE_SERIES: dict[str, tuple[tuple[str, ...], float, tuple[str, int]]] = {
    "flow1": (("Flow1 (mL/min)",), 1.0, ("d", 2)),
    "flow2": (("Flow2 (mL/min)",), 1.0, ("d", 2)),
    "flowTotal": (("Flow total (mL/min)",), 1.0, ("d", 2)),
    "flowSetpoint": (("Flow setpoint (mL/min)",), 1.0, ("d", 2)),
    "tInternal": (("Internal temp (C)",), 1.0, ("d", 2)),
    "tSample": (("Sample temp (C)",), 1.0, ("d", 2)),
    "rhSample": (("Sample RH (%)",), 1.0, ("d", 2)),
}


def channel_series(ch: str) -> dict[str, tuple[tuple[str, ...], float, tuple[str, int]]]:
    """The per-channel QC columns. BCc and the spot BCs arrive in ng/m³ and leave in µg/m³."""
    return {
        f"{ch}.ATN1": ((f"{ch} ATN1",), 1.0, ("d", 2)),
        f"{ch}.ATN2": ((f"{ch} ATN2",), 1.0, ("d", 2)),
        f"{ch}.dATN1": ((f"delta {ch} ATN1 rolling mean",), 1.0, ("s", 4)),
        f"{ch}.dATN2": ((f"delta {ch} ATN2 rolling mean",), 1.0, ("s", 4)),
        f"{ch}.leak": ((f"{ch}_ratio_dATN_flow",), 1.0, ("d", 4)),
        f"{ch}.BCc/BC1": ((f"{ch} BCc/BC1 smoothed",), 1.0, ("d", 4)),
        f"{ch}.BC1/BC2": ((f"{ch} BC1/BC2 smoothed",), 1.0, ("d", 4)),
        f"{ch}.rough1": ((f"{ch} ATN1_roughness",), 1.0, ("s", 4)),
        f"{ch}.rough2": ((f"{ch} ATN2_roughness",), 1.0, ("s", 4)),
        f"{ch}.smoothed": ((f"{ch} BCc smoothed", f"{ch} BCc smoothed  (ng/m^3)"), 1e-3, ("d", 4)),
        # the raw dual-spot chain (Jacros only): instrument loading factor and the two spots' concentrations
        f"{ch}.K": ((f"{ch} K",), 1.0, ("s", 4)),
        f"{ch}.BC1": ((f"{ch} BC1",), 1e-3, ("d", 4)),
        f"{ch}.BC2": ((f"{ch} BC2",), 1e-3, ("d", 4)),
    }


SERIES = {**BASE_SERIES, **{k: v for ch in CHANNELS for k, v in channel_series(ch).items()}}

# what each series is, for the page's tooltips and axis titles
SERIES_NOTES = {
    "flow1": "Flow through spot 1 (mL/min), daily mean",
    "flow2": "Flow through spot 2 (mL/min), daily mean",
    "flowTotal": "Total flow (mL/min), daily mean",
    "flowSetpoint": "Flow setpoint (mL/min), daily mean",
    "tInternal": "Internal temperature (°C), daily mean",
    "tSample": "Sample temperature (°C), daily mean",
    "rhSample": "Sample relative humidity (%), daily mean",
    "ATN1": "Attenuation on spot 1, daily mean (rises as the spot loads, resets at a tape advance)",
    "ATN2": "Attenuation on spot 2, daily mean",
    "dATN1": "Per-minute change in spot-1 attenuation, rolling mean, daily mean",
    "dATN2": "Per-minute change in spot-2 attenuation, rolling mean, daily mean",
    "leak": "Leak ratio (dATN1/Flow1)/(dATN2/Flow2), daily mean of the minute ratio (src/data/qc/pkl_cleaning.py clean_leak_ratio)",
    "BCc/BC1": "Smoothed BCc over smoothed BC1: the loading compensation factor",
    "BC1/BC2": "Smoothed BC1 over smoothed BC2: the two spots' uncompensated concentrations",
    "rough1": "Roughness of spot-1 attenuation: the 60-minute mean of |dATN − its rolling mean| (pkl_cleaning.add_roughness_columns)",
    "rough2": "Roughness of spot-2 attenuation",
    "smoothed": "Smoothed BCc (µg/m³; the cleaning pipeline's 10-min DEMA per tape spot), daily mean",
    "K": "The instrument's loading-compensation parameter K, daily mean (Jacros raw MA350 export)",
    "BC1": "Spot-1 BC before loading compensation (µg/m³), daily mean",
    "BC2": "Spot-2 BC before loading compensation (µg/m³), daily mean",
}


def rounded(values: pd.Series, how: tuple[str, int]) -> list[float | None]:
    """JSON-ready values: None for missing or non-finite, else rounded to decimals or significant figures."""
    kind, n = how
    out: list[float | None] = []
    for v in pd.to_numeric(values, errors="coerce").to_numpy(dtype=float):
        if not np.isfinite(v):
            out.append(None)
        elif kind == "d":
            out.append(round(float(v), n))
        else:
            out.append(0.0 if v == 0 else float(f"{v:.{n}g}"))
    return out


def bin_starts(raw: pd.DataFrame) -> pd.Series:
    """Each bin's local start, parsed exactly as export_aethalometer.daily_frame does (so the two files line up)."""
    start = aeth.column(raw, "datetime_local")
    if start is None:   # a pickle with only the 9 AM day
        start = pd.to_datetime(aeth.column(raw, "day_9am")) + pd.Timedelta(hours=9)
    start = pd.to_datetime(start)
    return start.dt.tz_localize(None) if getattr(start.dt, "tz", None) else start


def quality_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per bin: its start and every QC series the pickle has (columns it lacks are left out)."""
    out = pd.DataFrame({"start": bin_starts(raw).to_numpy()})
    for key, (names, scale, _) in SERIES.items():
        col = aeth.column(raw, *names)
        if col is not None and pd.to_numeric(col, errors="coerce").notna().any():
            out[key] = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float) * scale
    return out.dropna(subset=["start"]).sort_values("start", kind="stable").drop_duplicates("start").reset_index(drop=True)


def flow_periods() -> dict[str, dict]:
    """config.FLOW_FIX_PERIODS by gallery site code (scripts/flow_periods.classify_flow_period reads the same dict)."""
    out = {}
    for code, (site, _) in aeth.SITE_FILES.items():
        cfg = FLOW_FIX_PERIODS.get(site)
        if cfg is None:
            continue
        out[code] = {k: cfg.get(k) for k in ("before_end", "after_start", "has_before_data", "description", "notes")}
    return out


def cleaning_bounds() -> dict[str, dict]:
    """The defaults of the minute cleaning's QC steps, read from their signatures (src/data/qc/pkl_cleaning.py)."""
    sys.path.insert(0, str(ROOT))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # src/external/calibration.py has an invalid escape sequence
        from src.data.qc.pkl_cleaning import PKLDataCleaner
    defaults = lambda fn: {k: p.default for k, p in inspect.signature(fn).parameters.items() if p.default is not inspect.Parameter.empty}  # noqa: E731
    return {
        "flow_ratio": defaults(PKLDataCleaner.clean_flow_ratio),        # Flow1/Flow2 kept within [lower, upper]
        "flow_range": defaults(PKLDataCleaner.clean_flow_range),        # total flow within setpoint × (1 ± threshold)
        "leak_ratio": defaults(PKLDataCleaner.clean_leak_ratio),        # (dATN1/F1)/(dATN2/F2) within (lower, upper)
        "bcc_ratio": defaults(PKLDataCleaner.clean_bcc_ratio),          # BCc/BC1 or BCc/BC2 within [lower, upper]
        "bcc_denominator": defaults(PKLDataCleaner.clean_bcc_denominator),  # |1 − K·ATN1| above threshold (IR: threshold_IR)
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sites = {}
    for code, (site_name, fname) in aeth.SITE_FILES.items():
        path = aeth.PROCESSED / fname
        if not path.exists():
            continue
        q = quality_frame(pd.read_pickle(path))
        keys = [k for k in SERIES if k in q.columns]
        payload = {"start": q["start"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
                   "series": {k: rounded(q[k], SERIES[k][2]) for k in keys}}
        (OUT / f"{code}.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        sites[code] = dict(site=site_name, file=fname, bins=len(q), series=keys,
                           channels=[ch for ch in CHANNELS if f"{ch}.ATN1" in keys])
        print(f"{code} ({site_name}): {len(q)} bins, {len(keys)} series")
    index = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        source="MA350 diagnostic columns in research/ftir_hips_chem/processed_sites (legacy daily aggregates of minute values; bins labelled at their start, coverage unverified)",
        flow_periods=flow_periods(),
        smooth_raw_thresholds=list(SMOOTH_RAW_THRESHOLDS),
        cleaning_bounds=cleaning_bounds(),
        notes=SERIES_NOTES,
        sites=sites,
    )
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":"), allow_nan=False))
    size = sum(p.stat().st_size for p in OUT.glob("*.json"))
    print(f"{size / 1e6:.2f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
