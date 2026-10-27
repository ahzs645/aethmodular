"""Raw HIPS optics, lot and loading per filter at the four focus sites, for the Research diagnostics tab.

Run with:  uv run python gallery/data/export_research_hips_optics.py

Input (in the repo): research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl (config.FILTER_DATA_PATH).

filters.json carries HIPS Fabs, MDL and BC but not what the operating-envelope notebooks read underneath
them: the raw HIPS coordinates (t, r, tau, T1, R1), the lot and calibration coefficients a filter was
read against, the field blanks, the deposit area and sampled volume, and FTIR's direct mass on the filter.
This export adds those, one row per physical filter (HIPS and FTIR pieces share the base id SITE-NNNN,
the id the gallery's rows use), so a page can join them to the rows by id. Blanks (FilterType FB) have no
sample date and no gallery row; they are kept with type "FB".

Pages that read it (gallery/app/src/sections/research/pages):
  HipsEnvelope    research/ftir_hips_chem/addis_06_hips_ptfe_operating_envelope.ipynb,
                  research/ftir_ec_phase3/ftir_37_hips_loading_and_curvature.ipynb (SPARTAN blank domain),
                  research/improve_hips_offset/improve_fed_rt_proxy_figure2.ipynb (SPARTAN R1/T1 analog)
  ImproveEnvelope the Addis p05-p95 bounds the IMPROVE notebooks screen with

Writes gallery/app/public/data/research/hips_optics/filters.json (private: the research section's data).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from config import FILTER_DATA_PATH  # noqa: E402

OUT = ROOT / "gallery/app/public/data/research/hips_optics"

# output column -> (Parameter in the long table, decimals); values are the Concentration column
HIPS_PARAMS = {
    "t": ("HIPS_t", 4), "r": ("HIPS_r", 4), "tau": ("HIPS_tau", 4),
    "T1": ("HIPS_T1", 1), "R1": ("HIPS_R1", 1),
    "cal_intercept": ("HIPS_Intercept", 1), "cal_slope": ("HIPS_Slope", 4),
    "fabs": ("HIPS_Fabs", 3), "mdl": ("HIPS_MDL", 3),
}
# FTIR mass on the filter (MassLoading_ug of these parameters), µg
MASS_PARAMS = {"ec_ug": "EC_ftir", "oc_ug": "OC_ftir", "om_ug": "OM"}
COLUMNS = ["id", "filter_id", "code", "type", "date", "lot", "analysis_date",
           *HIPS_PARAMS, "area_cm2", "volume_m3", *MASS_PARAMS]


def base_id(filter_id: str) -> str:
    """SITE-NNNN-k (a HIPS/FTIR piece) -> SITE-NNNN, the gallery's row id; other ids unchanged."""
    return pd.Series([str(filter_id)]).str.replace(r"^([A-Z]+-\d{4})-\d+$", r"\1", regex=True).iloc[0]


def first(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def build(long: pd.DataFrame) -> pd.DataFrame:
    """One row per HIPS-measured filter (samples and field blanks) with its optics, lot and FTIR mass."""
    hips = long[long["Parameter"].isin([p for p, _ in HIPS_PARAMS.values()])].copy()
    hips["base"] = hips["FilterId"].map(base_id)
    wide = hips.pivot_table(index="FilterId", columns="Parameter", values="Concentration", aggfunc="first")
    meta = hips.sort_values(["FilterId", "AnalysisDate"]).groupby("FilterId").agg(
        base=("base", "first"), code=("Site", "first"), type=("FilterType", "first"), date=("SampleDate", first),
        lot=("LotId", first), analysis_date=("AnalysisDate", first), area_cm2=("DepositArea_cm2", first),
        volume_m3=("Volume_m3", first))
    out = meta.join(wide)
    ftir = long[long["Parameter"].isin(MASS_PARAMS.values())].copy()
    ftir["base"] = ftir["FilterId"].map(base_id)
    mass = ftir.pivot_table(index="base", columns="Parameter", values="MassLoading_ug", aggfunc="first")
    out = out.reset_index().merge(mass, left_on="base", right_index=True, how="left")
    frame = pd.DataFrame({
        "id": out["base"], "filter_id": out["FilterId"], "code": out["code"], "type": out["type"],
        "date": pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "lot": out["lot"].map(lambda v: None if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(v)),
        "analysis_date": pd.to_datetime(out["analysis_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
    })
    for col, (param, _) in HIPS_PARAMS.items():
        frame[col] = pd.to_numeric(out.get(param), errors="coerce")
    frame["area_cm2"] = pd.to_numeric(out["area_cm2"], errors="coerce")
    frame["volume_m3"] = pd.to_numeric(out["volume_m3"], errors="coerce")
    for col, param in MASS_PARAMS.items():
        frame[col] = pd.to_numeric(out.get(param), errors="coerce")
    # a sample has one HIPS piece per base id; keep the first if a re-read ever duplicates one
    samples = frame[frame["type"].ne("FB")].drop_duplicates("id")
    blanks = frame[frame["type"].eq("FB")]
    return pd.concat([samples, blanks], ignore_index=True).sort_values(["code", "type", "id"]).reset_index(drop=True)


def cell(v, digits: int | None):
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or (not isinstance(v, str) and pd.isna(v)):
        return None
    if digits is None:
        return v
    return round(float(v), digits)


def to_json(frame: pd.DataFrame) -> dict:
    digits = {**{c: d for c, (_, d) in HIPS_PARAMS.items()}, "area_cm2": 2, "volume_m3": 3, "ec_ug": 3, "oc_ug": 3, "om_ug": 3}
    rows = [[cell(r[c], digits.get(c)) for c in COLUMNS] for _, r in frame.iterrows()]
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl",
        "note": ("One row per HIPS-read filter at Beijing, Delhi, JPL and Addis Ababa; id is the gallery row id (SITE-NNNN). "
                 "type PM2.5 = sample, FB = field blank (no sample date, no gallery row). t, r, tau, T1, R1 are the raw HIPS "
                 "readings; cal_intercept / cal_slope the calibration line the filter was read against; fabs and mdl in Mm-1; "
                 "ec_ug / oc_ug / om_ug are FTIR's mass on the filter (MassLoading_ug)."),
        "units": {"fabs": "Mm-1", "mdl": "Mm-1", "area_cm2": "cm2", "volume_m3": "m3", "ec_ug": "ug", "oc_ug": "ug", "om_ug": "ug"},
        "columns": COLUMNS,
        "rows": rows,
    }


def main() -> None:
    long = pd.read_pickle(FILTER_DATA_PATH)
    frame = build(long)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "filters.json").write_text(json.dumps(to_json(frame), separators=(",", ":"), ensure_ascii=False))
    counts = frame.groupby(["code", "type"]).size().unstack(fill_value=0)
    print(f"wrote {OUT / 'filters.json'}: {len(frame)} filters")
    print(counts.to_string())


if __name__ == "__main__":
    main()
