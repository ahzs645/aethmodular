"""ERA5 climate normals for every SPARTAN site: monthly temperature and precipitation over 1991-2020.

    python research/ftir_hips_chem/workflows/fetch_climate_normals_era5.py            # every site
    python research/ftir_hips_chem/workflows/fetch_climate_normals_era5.py --site ETAD INDH

The gallery's own weather (gallery/data/export_weather.py) covers each site's sampling period, from a few months
to twelve years. A season calendar should be judged against the climate, so this fetches the WMO standard
normal period for every site, the same way, from the archive that export already uses. Daily mean temperature
and precipitation come from ERA5 through the Open-Meteo archive API; only the site's public coordinates are
sent. Writes gallery/app/public/data/weather/normals_1991_2020.json: per site and calendar month, the mean
temperature (°C), mean precipitation (mm/day), the share of days with 1 mm or more, and the year-to-year
standard deviation of the month's precipitation.

ERA5: Copernicus Climate Change Service information, via Open-Meteo (CC BY 4.0). ERA5 is a reanalysis on a
grid of about 25 km (9 km over land in ERA5-Land where Open-Meteo uses it); a station can differ, most of
all for rain in mountains and on coasts.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SITES = ROOT / "gallery/app/public/data/spartan_map.json"
OUT = ROOT / "gallery/app/public/data/weather/normals_1991_2020.json"
API = "https://archive-api.open-meteo.com/v1/archive"
PERIOD = ("1991-01-01", "2020-12-31")
WET_DAY_MM = 1.0


def fetch(lat: float, lon: float) -> pd.DataFrame:
    query = urllib.parse.urlencode({"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": PERIOD[0], "end_date": PERIOD[1],
                                    "daily": "temperature_2m_mean,precipitation_sum", "timezone": "auto"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=120) as response:
                body = json.load(response)
            break
        except Exception:  # noqa: BLE001 (a busy archive answers after a pause)
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    daily = pd.DataFrame(body["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    return daily.rename(columns={"temperature_2m_mean": "t", "precipitation_sum": "p"}).assign(grid_lat=body["latitude"], grid_lon=body["longitude"],
                                                                                       elevation=body.get("elevation"))


def normals(daily: pd.DataFrame) -> dict:
    month, year = daily["time"].dt.month, daily["time"].dt.year
    by_month = daily.groupby(month)
    yearly = daily.groupby([year, month])["p"].mean().groupby(level=1)
    r = lambda series, digits: [None if not np.isfinite(v) else round(float(v), digits) for v in series.reindex(range(1, 13))]  # noqa: E731
    return {"temperature": r(by_month["t"].mean(), 2), "precipitation": r(by_month["p"].mean(), 3),
            "wet_days": r(by_month["p"].apply(lambda v: (v >= WET_DAY_MM).mean()), 3), "precipitation_sd_between_years": r(yearly.std(), 3),
            "days": int(daily["t"].notna().sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", nargs="+", default=[], help="site codes (default: every site with coordinates)")
    a = ap.parse_args(argv)
    sites = [s for s in json.loads(SITES.read_text())["sites"] if s.get("lat") is not None and (not a.site or s["code"] in a.site)]
    saved = json.loads(OUT.read_text()) if OUT.exists() else {"sites": {}}
    for s in sites:
        daily = fetch(s["lat"], s["lon"])
        saved["sites"][s["code"]] = {"site": s.get("name") or s["code"], "lat": s["lat"], "lon": s["lon"],
                                     "grid_lat": float(daily["grid_lat"].iloc[0]), "grid_lon": float(daily["grid_lon"].iloc[0]),
                                     "grid_elevation_m": None if pd.isna(daily["elevation"].iloc[0]) else float(daily["elevation"].iloc[0]), **normals(daily)}
        print(f"{s['code']}: {saved['sites'][s['code']]['days']} days", flush=True)
        time.sleep(0.4)
    saved |= {"schema_version": 1, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "period": list(PERIOD),
              "source": "ERA5 (Copernicus Climate Change Service) via the Open-Meteo archive API, CC BY 4.0",
              "fields": {"temperature": "mean daily temperature at 2 m, °C", "precipitation": "mean precipitation, mm/day",
                         "wet_days": f"share of days with {WET_DAY_MM:g} mm or more", "precipitation_sd_between_years": "standard deviation of the month's mean precipitation between years, mm/day"},
              "months": list(range(1, 13))}
    saved["sites"] = dict(sorted(saved["sites"].items()))
    OUT.write_text(json.dumps(saved, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(saved['sites'])} sites)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
