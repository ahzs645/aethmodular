"""The Addis station's daily weather on each filter's sample day, as the meteorology notebooks matched it.

Run with:  uv run python gallery/data/export_research_addis_met.py

Inputs (tracked in the repo):
  research/ftir_hips_chem/Weather Data/Meteostat/Addis Ababa daily Average met Data.csv          daily tavg, tmin, tmax, prcp
  research/ftir_hips_chem/Weather Data/Meteostat/master_meteostat_AddisAbaba_63450_2022-12-01_2024-10-01.csv   hourly rhum, dwpt, wspd
  gallery/app/public/data/filters.json                                                           the Addis rows (id, sample date)

The gallery's own weather fields are means over each filter's sampling window. rh_sweep_seasonal,
hips_ftir_intercept_rh_recursive_diagnostics, temperature_gradient_analysis and meteorology_source_interaction
instead join the station's day to the filter's sample date: the daily file's tavg / tmin / tmax / prcp, and the
mean of the hourly RH, dew point and wind over that calendar day. This export gives the research pages those
same values, so a sweep or a T_min test can be run the notebooks' way as well as the gallery's. Absolute
humidity follows the recursive notebook: 6.112 exp(17.67 T / (T + 243.5)) × RH × 2.1674 / (273.15 + T), g/m³.

Writes gallery/app/public/data/research/addis_met/filters.json (private: the research section's data).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MET = ROOT / "research/ftir_hips_chem/Weather Data/Meteostat"
DAILY = MET / "Addis Ababa daily Average met Data.csv"
HOURLY = MET / "master_meteostat_AddisAbaba_63450_2022-12-01_2024-10-01.csv"
FILTERS = ROOT / "gallery/app/public/data/filters.json"
OUT = ROOT / "gallery/app/public/data/research/addis_met"
SITE = "ETAD"
FIELDS = {  # gallery field -> (column, unit, decimals)
    "RH (station, sample day)": ("rhum_mean", "%", 1),
    "Dew point (station, sample day)": ("dwpt_mean", "°C", 1),
    "Absolute humidity (station, sample day)": ("abs_humidity", "g/m³", 2),
    "Mean temperature (station, sample day)": ("tavg", "°C", 1),
    "Minimum temperature (station, sample day)": ("tmin", "°C", 1),
    "Maximum temperature (station, sample day)": ("tmax", "°C", 1),
    "Temperature range (station, sample day)": ("trange", "°C", 1),
    "Rain (station, sample day)": ("prcp", "mm", 1),
    "Wind speed (station, sample day)": ("wspd_mean", "km/h", 1),
}


def daily_met(daily: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: the daily file's values and the hourly means (the notebooks' hourly_daily_agg)."""
    d = daily.copy()
    d["day"] = pd.to_datetime(d["date"]).dt.normalize()
    h = hourly.copy()   # "NA" strings become NaN through to_numeric below
    h["day"] = pd.to_datetime(h["time"]).dt.normalize()
    for col in ["temp", "dwpt", "rhum", "wspd"]:
        h[col] = pd.to_numeric(h[col], errors="coerce")
    agg = h.groupby("day").agg(rhum_mean=("rhum", "mean"), dwpt_mean=("dwpt", "mean"), wspd_mean=("wspd", "mean")).reset_index()
    out = d[["day", "tavg", "tmin", "tmax", "prcp"]].merge(agg, on="day", how="outer")
    out["trange"] = out["tmax"] - out["tmin"]
    t = out["tavg"]
    out["abs_humidity"] = 6.112 * np.exp(17.67 * t / (t + 243.5)) * out["rhum_mean"] * 2.1674 / (273.15 + t)
    return out.drop_duplicates("day").set_index("day").sort_index()


def per_filter(rows: list[dict], met: pd.DataFrame) -> dict[str, list]:
    out = {}
    for r in rows:
        if r.get("code") != SITE or not r.get("date"):
            continue
        day = pd.Timestamp(r["date"]).normalize()
        if day not in met.index:
            continue
        m = met.loc[day]
        vals = [None if pd.isna(m[c]) else round(float(m[c]), d) for c, _, d in FIELDS.values()]
        if any(v is not None for v in vals):
            out[r["id"]] = vals
    return out


def main() -> None:
    met = daily_met(pd.read_csv(DAILY, encoding="utf-8-sig"), pd.read_csv(HOURLY))
    rows = json.loads(FILTERS.read_text())["rows"]
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Meteostat WMO 63450 (Bole airport): research/ftir_hips_chem/Weather Data/Meteostat (daily file and hourly master)",
        "note": "The station's calendar day matching each filter's sample date, as the meteorology notebooks join it; RH, dew point and wind are means of the hourly records that day.",
        "fields": list(FIELDS),
        "field_units": {f: u for f, (_, u, _) in FIELDS.items()},
        "rows": per_filter(rows, met),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "filters.json").write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {OUT / 'filters.json'}: {len(payload['rows'])} Addis filters")


if __name__ == "__main__":
    main()
