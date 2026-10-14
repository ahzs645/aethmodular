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
}
GROUPS = [{"label": "Carbon (TOR)", "fields": ["EC (TOR)", "OC (TOR)"]},
          {"label": "Optical (HIPS)", "fields": ["HIPS Fabs", "HIPS BC"]},
          {"label": "Mass", "fields": ["PM2.5 mass"]},
          {"label": "Metals & crustal", "fields": ["Silicon", "Iron", "Soil"]}]


def sig(v, digits=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    if v == 0:
        return 0.0
    return round(float(v), max(0, digits - 1 - int(math.floor(math.log10(abs(v))))))


def main():
    cols = ["SiteCode", "POC", "Date", "SiteName", "State", "Country", "Latitude", "Longitude"] + sorted({c for c, _, _ in FIELDS.values()})
    df = pd.read_csv(improve_clean_path(), usecols=cols, low_memory=False)
    df["Date"] = pd.to_datetime(df.Date).dt.strftime("%Y-%m-%d")
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
             "mac_value": MAC_VALUE, "fields": list(FIELDS), "field_units": {f: u for f, (_, u, _) in FIELDS.items()},
             "field_groups": GROUPS, "n": int(len(df)), "sites": sites}
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":"), ensure_ascii=False))
    size = sum(p.stat().st_size for p in OUT.glob("*.json"))
    print(f"wrote {len(sites)} site files + index ({len(df):,} samples, {size / 1e6:.1f} MB) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
