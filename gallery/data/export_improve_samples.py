"""Per-site IMPROVE sample files for the gallery's network picker.

Run with:  uv run python gallery/data/export_improve_samples.py

The Filters sidebar can switch the chart tabs from the four focus sites to a whole
network. IMPROVE has ~388,000 samples, far more than SVG charts can draw at once,
so each site is its own file, fetched only when that site is picked:

    gallery/app/public/data/improve_samples/index.json      sites, fields, units
    gallery/app/public/data/improve_samples/<SITE>.json     columnar samples for one site

Source: the validated IMPROVE cleaned table (improve_io.improve_clean_path(); rebuilt from
the FED yearly archives by workflows/refresh_improve_gallery.py). Every retained sample
has valid, positive EC and absorption. Field names follow the focus-site export where the
quantity is the same: HIPS BC = HIPS Fabs / MAC 10, as in filters.json. Sulfate and the
other ions are not in the cleaned table yet.

TOR thermal fractions (OP, EC1-EC3, OC1-OC4) come from the same cleaned table, so they share
the existing SiteCode + POC + Date rows with no extra join. OP is the reflectance pyrolysis
correction (IMPROVE OPf, "OPTR"), so EC (TOR) = EC1 + EC2 + EC3 - OP and OC (TOR) = OC1..OC4 + OP.
OP = 0 is a reported value (often laser saturation on heavily loaded filters) and is kept.
The cleaned table is validated IMPAER only: preliminary IMPAERPRE records are never read, and
values whose status is not valid ("V*") are already blank. Sampled volume is the carbon
(quartz, module C) sampler's FlowRate (L/min) / 1000 x SampDur (min), as tor_volume_m3 in
analog_bias.py; the cleaned table has flow and duration from mid-2015 on. Earlier samples take
the same quantity from the FTIR local database's TOR results (AverageFlowRate x ElapsedTime,
2005 on, only filters in the FTIR lots: about 31,000 more), which agrees with the cleaned table
to a ratio of 1.0000 on 27,189 shared samples. The rest have no volume (docs/mass-per-filter.md).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from config import MAC_VALUE  # noqa: E402
from improve_io import improve_clean_path  # noqa: E402

OUT = ROOT / "gallery/app/public/data/improve_samples"
FIELDS = {  # gallery field -> (cleaned column, unit, scale)
    "EC (TOR)": ("ECf_Val", "ug/m3", 1.0),
    "OC (TOR)": ("OCf_Val", "ug/m3", 1.0),
    "HIPS Fabs": ("fAbs_Val", "Mm-1", 1.0),
    "HIPS BC": ("fAbs_Val", "ug/m3", 1.0 / MAC_VALUE),
    "PM2.5 mass": ("MF_Val", "ug/m3", 1.0),
    "Iron": ("FEf_Val", "ug/m3", 1.0),
    "Silicon": ("SIf_Val", "ug/m3", 1.0),
    "Soil": ("SOILf_Val", "ug/m3", 1.0),
    "OP": ("OPf_Val", "ug/m3", 1.0),
    "EC1": ("EC1f_Val", "ug/m3", 1.0),
    "EC2": ("EC2f_Val", "ug/m3", 1.0),
    "EC3": ("EC3f_Val", "ug/m3", 1.0),
    "OC1": ("OC1f_Val", "ug/m3", 1.0),
    "OC2": ("OC2f_Val", "ug/m3", 1.0),
    "OC3": ("OC3f_Val", "ug/m3", 1.0),
    "OC4": ("OC4f_Val", "ug/m3", 1.0),
    # derived in main(): carbon-module FlowRate_C_Val (L/min) / 1000 x SampDur_C_Val (min)
    "Sampled volume": ("volume_m3", "m³", 1.0),
}
VOLUME_COLS = ["FlowRate_C_Val", "SampDur_C_Val"]
GROUPS = [{"label": "Carbon (TOR)", "fields": ["EC (TOR)", "OC (TOR)"]},
          {"label": "Optical (HIPS)", "fields": ["HIPS Fabs", "HIPS BC"]},
          {"label": "Mass", "fields": ["PM2.5 mass"]},
          {"label": "Crustal elements", "fields": ["Silicon", "Iron", "Soil"]},
          {"label": "TOR thermal fractions", "fields": ["OP", "EC1", "EC2", "EC3", "OC1", "OC2", "OC3", "OC4"]},
          {"label": "Sampling", "fields": ["Sampled volume"]}]


def sig(v, digits=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    if v == 0:
        return 0.0
    return round(float(v), max(0, digits - 1 - int(math.floor(math.log10(abs(v))))))


def tor_volumes() -> dict | None:
    """(SiteCode, YYYY-MM-DD) -> m³ from the FTIR local database's TOR results, or None when unavailable."""
    from data_paths import ftir_local_db  # research scripts are on sys.path (above)
    path = ftir_local_db() / "tables" / "results_tor.csv"
    if not path.exists():
        print(f"! {path.name} not found; sampled volume stays mid-2015 on")
        return None
    r = pd.read_csv(path, usecols=["Site", "SampleDate", "AverageFlowRate", "ElapsedTime"], dtype={"Site": str, "SampleDate": str})
    r = r.dropna(subset=["AverageFlowRate", "ElapsedTime"])
    r["date"] = pd.to_datetime(r.SampleDate.str[:10], format="%m/%d/%Y", errors="coerce").dt.strftime("%Y-%m-%d")
    r["vol"] = r.AverageFlowRate / 1000.0 * r.ElapsedTime
    # one value per site and date (every parameter row of a filter repeats it)
    return r.groupby(["Site", "date"]).vol.median().to_dict()


def main():
    cols = ["SiteCode", "POC", "Date", "SiteName", "State", "Country", "Latitude", "Longitude"] + sorted({c for c, _, _ in FIELDS.values()} - {"volume_m3"}) + VOLUME_COLS
    df = pd.read_csv(improve_clean_path(), usecols=cols, low_memory=False)
    df["volume_m3"] = df.FlowRate_C_Val / 1000.0 * df.SampDur_C_Val
    df["Date"] = pd.to_datetime(df.Date).dt.strftime("%Y-%m-%d")
    tor = tor_volumes()
    if tor is not None:
        fill = df.set_index(["SiteCode", "Date"]).index.map(tor.get)
        before = df.volume_m3.notna().sum()
        df["volume_m3"] = df.volume_m3.fillna(pd.Series(fill, index=df.index, dtype=float))
        print(f"sampled volume: {before:,} from the cleaned table, {df.volume_m3.notna().sum() - before:,} more from the TOR results")
    df = df.sort_values(["SiteCode", "Date", "POC"])
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    sites = []
    for code, g in df.groupby("SiteCode"):
        rec = {"code": code, "n": len(g), "date": g.Date.tolist(), "poc": g.POC.astype(int).tolist()}
        for name, (col, _, scale) in FIELDS.items():
            rec[name] = [sig(v * scale) if pd.notna(v) else None for v in g[col]]
        (OUT / f"{code}.json").write_text(json.dumps(rec, separators=(",", ":")))
        first = g.iloc[0]
        lat, lon = first.Latitude, first.Longitude
        sites.append({"code": code, "name": str(first.SiteName), "state": None if pd.isna(first.State) else str(first.State),
                      "country": None if pd.isna(first.Country) else str(first.Country),
                      "lat": None if pd.isna(lat) else float(lat), "lon": None if pd.isna(lon) else float(lon),
                      "n_filters": len(g), "date_min": g.Date.min(), "date_max": g.Date.max(),
                      "coverage": {f: int(g[c].notna().sum()) for f, (c, _, _) in FIELDS.items()}})
    index = {"generated": datetime.now(timezone.utc).isoformat(), "source": "validated IMPROVE cleaned table",
             "notes": ["TOR thermal fractions: validated IMPAER only (no preliminary IMPAERPRE); OP is OPf (reflectance), "
                       "EC (TOR) = EC1+EC2+EC3-OP, OC (TOR) = OC1+OC2+OC3+OC4+OP; OP = 0 is reported, not missing.",
                       "Sampled volume: carbon (quartz, module C) FlowRate L/min / 1000 x SampDur min (cleaned table, mid-2015 on), else AverageFlowRate x ElapsedTime from the FTIR local database's TOR results (FTIR-lot filters, 2005 on)."],
             "mac_value": MAC_VALUE, "fields": list(FIELDS), "field_units": {f: u for f, (_, u, _) in FIELDS.items()},
             "field_groups": GROUPS, "n": int(len(df)), "sites": sites}
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":"), ensure_ascii=False))
    size = sum(p.stat().st_size for p in OUT.glob("*.json"))
    print(f"wrote {len(sites)} site files + index ({len(df):,} samples, {size / 1e6:.1f} MB) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
