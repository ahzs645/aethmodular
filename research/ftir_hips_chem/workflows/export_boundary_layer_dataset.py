"""Bundle the cached ERA5 boundary-layer height into a tracked dataset.

Reads the gitignored fetch cache (output/tables/boundary_layer_era5/<network>/, from
fetch_boundary_layer_era5.py plus the colab_arco_blh_gapfill.py fill) and writes
research/ftir_hips_chem/Weather Data/ERA5 boundary layer/:

    blh_hourly_<network>.parquet  time (UTC), code, blh_m, source ("open-meteo" | "arco-era5")
    blh_daily_<network>.csv       per local solar day (UTC + round(lon/15) h): mean, max, 00-06 mean, hours
    sites_<network>.csv           coordinates, ERA5 grid cell, span, share of hours from the ARCO fill
    README.md                     source, citation, the 2024 gap, how to rebuild

Run from the repo root:
    uv run python research/ftir_hips_chem/workflows/export_boundary_layer_dataset.py spartan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
from boundary_layer import FILL_FILE, daily_summary  # noqa: E402

CACHE = AREA / "output/tables/boundary_layer_era5"
OUT = AREA / "Weather Data/ERA5 boundary layer"


def site_hours(folder: Path, code: str, fill: pd.DataFrame) -> pd.DataFrame:
    raw = pd.concat(pd.read_parquet(f).astype({"blh_m": "float64"}) for f in sorted((folder / code).glob("*.parquet")))
    raw = raw.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    raw["source"] = "open-meteo"
    gap = raw.blh_m.isna()
    if gap.any():
        f = fill.loc[fill.code.eq(code)].set_index("time").blh_m
        raw.loc[gap, "blh_m"] = raw.loc[gap, "time"].map(f).to_numpy()
        raw.loc[gap & raw.blh_m.notna(), "source"] = "arco-era5"
    raw["code"] = code
    return raw[["time", "code", "blh_m", "source"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("network", choices=["spartan", "improve"])
    args = ap.parse_args()
    folder = CACHE / args.network
    sites = pd.read_csv(folder / "sites.csv")
    fill = pd.read_parquet(folder / FILL_FILE) if (folder / FILL_FILE).exists() else pd.DataFrame(columns=["time", "code", "blh_m"])

    hourly, daily, meta = [], [], []
    for s in sites.itertuples():
        h = site_hours(folder, s.code, fill)
        offset = round(s.lon / 15)
        d = daily_summary(h.set_index("time").blh_m, offset).reset_index()
        d.insert(0, "code", s.code)
        grid = json.loads((folder / s.code / "site.json").read_text())
        hourly.append(h)
        daily.append(d)
        meta.append({"code": s.code, "name": s.name, "lat": s.lat, "lon": s.lon,
                     "era5_grid_lat": grid.get("grid_lat"), "era5_grid_lon": grid.get("grid_lon"),
                     "solar_utc_offset_h": offset, "start_utc": h.time.min().strftime("%Y-%m-%d %H:%M"),
                     "end_utc": h.time.max().strftime("%Y-%m-%d %H:%M"),
                     "hours": len(h), "missing_hours": int(h.blh_m.isna().sum()),
                     "pct_from_arco_fill": round(100 * h.source.eq("arco-era5").mean(), 2),
                     "median_blh_m": round(float(h.blh_m.median()), 1)})

    OUT.mkdir(parents=True, exist_ok=True)
    hourly = pd.concat(hourly, ignore_index=True)
    hourly["blh_m"] = hourly.blh_m.astype("float32")
    hourly["code"] = hourly.code.astype("category")
    hourly["source"] = hourly.source.astype("category")
    hourly.to_parquet(OUT / f"blh_hourly_{args.network}.parquet", index=False, compression="zstd")
    pd.concat(daily, ignore_index=True).round(1).to_csv(OUT / f"blh_daily_{args.network}.csv", index=False)
    pd.DataFrame(meta).to_csv(OUT / f"sites_{args.network}.csv", index=False)
    print(f"{args.network}: {len(meta)} sites, {len(hourly):,} hours, "
          f"{int(hourly.blh_m.isna().sum())} missing, {int(hourly.source.eq('arco-era5').sum()):,} from the ARCO fill -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
