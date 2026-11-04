"""Pull ERA5 total column water vapour for every SPARTAN site in the weather cache.

AERONET precipitable water needs a working sun photometer: it has none in a filter window with
cloud or an outage, and Bishoftu's only station is 34 km away in Addis (and was silent from
14 Nov 2025 to 12 Jan 2026). ERA5's column water fills those windows, at the site's own grid cell.

Each site is pulled over the span of its hours in weather_era5/spartan_hourly.parquet, so it
covers every window export_weather.py averages. Resumable: cached site-years are skipped.

Run from the repo root:
    uv run python research/ftir_hips_chem/workflows/fetch_column_water_era5.py
    uv run python research/ftir_hips_chem/workflows/fetch_column_water_era5.py --sites ETBI ETAD

Writes research/ftir_hips_chem/output/tables/weather_era5/column_water/<CODE>/<year>.parquet
(time in UTC, tcwv_kg_m2), read by gallery/data/export_weather.py. 1 kg/m² is 0.1 cm of water.

ERA5 data: Copernicus Climate Change Service information, via the Open-Meteo archive API (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
from boundary_layer import CallBudget, _request  # noqa: E402

VARIABLE = "total_column_integrated_water_vapour"
HOURLY = AREA / "output/tables/weather_era5/spartan_hourly.parquet"
CACHE = AREA / "output/tables/weather_era5/column_water"
SITES = ROOT / "gallery/app/public/data/spartan_map.json"


def fetch_site(code: str, lat: float, lon: float, start: date, end: date, budget: CallBudget) -> int:
    """Download every missing site-year in [start, end]; returns the number of years fetched."""
    fetched = 0
    for year in range(start.year, end.year + 1):
        path = CACHE / code / f"{year}.parquet"
        a, b = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        if path.exists():
            have = pd.read_parquet(path)
            if have.time.min().date() <= a and have.time.max().date() >= b:
                continue
        budget.spend(max(1.0, ((b - a).days + 1) / 14))
        d = _request(lat, lon, a.isoformat(), b.isoformat(), VARIABLE)
        frame = pd.DataFrame({"time": pd.to_datetime(d["hourly"]["time"]), "tcwv_kg_m2": d["hourly"][VARIABLE]})
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        fetched += 1
    return fetched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*", help="only these site codes")
    args = ap.parse_args()

    hourly = pd.read_parquet(HOURLY, columns=["site", "time_utc"])
    hourly["site"] = hourly["site"].astype(str)
    spans = hourly.groupby("site")["time_utc"].agg(["min", "max"])
    coords = {s["code"]: s for s in json.loads(SITES.read_text())["sites"]}
    budget, t0 = CallBudget(), time.time()
    for code, span in spans.iterrows():
        if args.sites and code not in args.sites:
            continue
        s = coords.get(code)
        if s is None or s.get("lat") is None:
            print(f"{code}: no coordinates, skipped", flush=True)
            continue
        n = fetch_site(code, s["lat"], s["lon"], span["min"].date(), span["max"].date(), budget)
        print(f"[{time.time() - t0:5.0f}s] {code} {span['min'].date()}..{span['max'].date()} ({n} new site-years)", flush=True)


if __name__ == "__main__":
    main()
