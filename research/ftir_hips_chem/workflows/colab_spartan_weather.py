"""Hourly and daily ERA5 surface weather for every SPARTAN site over its sampling span, on a Colab VM.

Pulls, from the Open-Meteo ERA5 archive (the same source and model as the boundary-layer
pull, scripts/boundary_layer.py), the variables the Addis meteorology notebooks used from
the Meteostat station (temperature, dew point, relative humidity, precipitation, wind speed
and direction, pressure) for every SPARTAN site with filter dates, from its first to its
last filter. Daily values are made from the hourly ones over each site's local standard
day (midnight to midnight, no daylight saving), the way the filters' dates are kept.

Writes to /content/weather/ and prints a size and coverage report:
    hourly.parquet   site, time_utc, 7 variables (float32)
    daily.parquet    site, date (local standard), 11 variables (float32)
    plus the same tables as zstd/snappy parquet and gzip CSV for the size comparison, and sizes.json.

Open-Meteo weighs a request by its length (two weeks = one call) and by variables in tens,
so seven hourly variables cost the same as one: all SPARTAN is ~4,300 weighted calls,
inside the free tier's 5,000 / h and 10,000 / day. Requests are paced to stay under them.

From the repo root:
    uv run python research/ftir_hips_chem/workflows/colab_spartan_weather.py --sites-arg > /tmp/sites.txt
    colab run --keep -s spwx --timeout 5400 research/ftir_hips_chem/workflows/colab_spartan_weather.py "$(cat /tmp/sites.txt)"
    colab download -s spwx /content/weather/hourly.parquet research/ftir_hips_chem/output/tables/weather_era5/spartan_hourly.parquet
    colab download -s spwx /content/weather/daily.parquet research/ftir_hips_chem/output/tables/weather_era5/spartan_daily.parquet
    colab download -s spwx /content/weather/sizes.json research/ftir_hips_chem/output/tables/weather_era5/sizes.json
    colab stop -s spwx
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import date, datetime
from pathlib import Path

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5"
HOURLY = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "precipitation",
          "wind_speed_10m", "wind_direction_10m", "surface_pressure"]
LIMITS = ((60, 500), (3600, 4500), (86400, 9500))   # (window s, weighted calls), with margin
OUT = Path(os.environ.get("WEATHER_OUT", "/content/weather"))


def sites_arg() -> str:
    """CODE:lat:lon:first:last;... for every SPARTAN site with filter dates (run locally)."""
    root = Path(__file__).resolve().parents[3]
    # the focus sites' HIPS filters (gallery filters.json) can start before or run past the public chemistry
    focus: dict[str, list[str]] = {}
    for f in json.loads((root / "gallery/app/public/data/filters.json").read_text())["rows"]:
        focus.setdefault(f["code"], []).append(f["date"])
    parts = []
    for s in json.loads((root / "gallery/app/public/data/spartan_map.json").read_text())["sites"]:
        if s.get("lat") is None or str(s.get("date_min", "—")).startswith("—"):
            continue
        dates = [s["date_min"][:10], s["date_max"][:10], *focus.get(s["code"], [])]
        parts.append(f"{s['code']}:{s['lat']:.4f}:{s['lon']:.4f}:{min(dates)}:{max(dates)}")
    return ";".join(parts)


class Budget:
    def __init__(self):
        self.log: deque[tuple[float, float]] = deque()

    def spend(self, weight: float):
        while True:
            now = time.time()
            while self.log and now - self.log[0][0] > 86400:
                self.log.popleft()
            waits = [next(t for t, _ in self.log if now - t < w) + w - now + 1
                     for w, cap in LIMITS if sum(c for t, c in self.log if now - t < w) + weight > cap]
            if not waits:
                self.log.append((now, weight))
                return
            time.sleep(max(waits))


def weight(start: date, end: date) -> float:
    days = (end - start).days + 1
    return max(1.0, days / 14) * max(1.0, len(HOURLY) / 10)


def fetch(lat: float, lon: float, start: date, end: date) -> dict:
    q = urllib.parse.urlencode({"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": start.isoformat(),
                                "end_date": end.isoformat(), "hourly": ",".join(HOURLY), "models": MODEL,
                                "timezone": "UTC", "wind_speed_unit": "ms"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(f"{ARCHIVE}?{q}", timeout=240) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # a shared Colab address can hit the limit through other users' calls too: back off and retry
            wait = 65 if e.code == 429 else 10 * (attempt + 1)
            print(f"  HTTP {e.code}; retry in {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:  # network hiccup
            print(f"  {type(e).__name__}: {e}; retry", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Open-Meteo kept failing")


def std_offset_hours(lat: float, lon: float) -> float:
    """The site's standard (non-DST) UTC offset, from its time zone."""
    from timezonefinder import TimezoneFinder
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC")
    offs = [datetime(2023, m, 15, 12, tzinfo=tz).utcoffset().total_seconds() / 3600 for m in (1, 7)]
    return min(offs)   # DST only ever adds an hour


def main(spec: str):
    import numpy as np
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    sites = []
    for part in spec.split(";"):
        code, lat, lon, d0, d1 = part.split(":")
        sites.append((code, float(lat), float(lon), date.fromisoformat(d0), date.fromisoformat(d1)))
    total_w = sum(weight(date(y, 1, 1) if y > s[3].year else s[3], date(y, 12, 31) if y < s[4].year else s[4])
                  for s in sites for y in range(s[3].year, s[4].year + 1))
    print(f"{len(sites)} sites, {sum((s[4] - s[3]).days + 1 for s in sites)} site-days, ~{total_w:.0f} weighted calls", flush=True)

    budget, frames, t0 = Budget(), [], time.time()
    for i, (code, lat, lon, d0, d1) in enumerate(sites):
        offset = std_offset_hours(lat, lon)
        for y in range(d0.year, d1.year + 1):
            # one calendar year per request, padded a day each side so local days at the ends are whole
            a = max(d0, date(y, 1, 1)) - pd.Timedelta(days=1 if y == d0.year else 0)
            b = min(d1, date(y, 12, 31)) + pd.Timedelta(days=1 if y == d1.year else 0)
            a, b = pd.Timestamp(a).date(), pd.Timestamp(b).date()
            budget.spend(weight(a, b))
            h = fetch(lat, lon, a, b)["hourly"]
            f = pd.DataFrame({"time_utc": pd.to_datetime(h["time"]), **{v: np.asarray(h[v], dtype="float64") for v in HOURLY}})
            f.insert(0, "site", code)
            f["utc_offset_h"] = offset
            frames.append(f)
        print(f"[{i + 1}/{len(sites)}] {code} {d0}..{d1} (UTC{offset:+g}) {time.time() - t0:.0f}s", flush=True)

    hourly = pd.concat(frames, ignore_index=True).drop_duplicates(["site", "time_utc"])
    # keep each site's span (plus the padding hours that complete its first and last local day)
    local = hourly["time_utc"] + pd.to_timedelta(hourly["utc_offset_h"], unit="h")
    span = {c: (pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1)) for c, _, _, a, b in sites}
    lo = hourly["site"].map(lambda c: span[c][0]); hi = hourly["site"].map(lambda c: span[c][1])
    hourly = hourly[(local >= lo) & (local < hi)].copy()
    hourly["local_date"] = (hourly["time_utc"] + pd.to_timedelta(hourly["utc_offset_h"], unit="h")).dt.normalize()

    # daily over the local standard day; wind direction as a vector mean
    rad = np.deg2rad(hourly["wind_direction_10m"])
    hourly["_u"] = -hourly["wind_speed_10m"] * np.sin(rad)
    hourly["_v"] = -hourly["wind_speed_10m"] * np.cos(rad)
    g = hourly.groupby(["site", "local_date"])
    daily = pd.DataFrame({
        "t_mean": g["temperature_2m"].mean(), "t_min": g["temperature_2m"].min(), "t_max": g["temperature_2m"].max(),
        "precip_sum": g["precipitation"].sum(min_count=1), "rh_mean": g["relative_humidity_2m"].mean(),
        "rh_min": g["relative_humidity_2m"].min(), "rh_max": g["relative_humidity_2m"].max(),
        "dew_point_mean": g["dew_point_2m"].mean(), "wind_speed_mean": g["wind_speed_10m"].mean(),
        "pressure_mean": g["surface_pressure"].mean(),
    })
    u, v = g["_u"].mean(), g["_v"].mean()
    daily["wind_dir_vector_mean"] = (np.rad2deg(np.arctan2(-u, -v)) + 360) % 360
    daily["n_hours"] = g["temperature_2m"].count()
    daily = daily.reset_index().rename(columns={"local_date": "date"})

    hourly = hourly[["site", "time_utc", *HOURLY]].copy()
    for t in (hourly, daily):
        t["site"] = t["site"].astype("category")
        for c in t.columns:
            if t[c].dtype == "float64":
                t[c] = t[c].astype("float32")
    daily["n_hours"] = daily["n_hours"].astype("int8")

    sizes = {"sites": len(sites), "hourly_rows": len(hourly), "daily_rows": len(daily),
             "hourly_vars": len(HOURLY), "daily_vars": daily.shape[1] - 3, "files": {}}
    for name, t in (("hourly", hourly), ("daily", daily)):
        for comp in ("zstd", "snappy"):
            p = OUT / (f"{name}.parquet" if comp == "zstd" else f"{name}.{comp}.parquet")
            t.to_parquet(p, compression=comp, index=False)
            sizes["files"][p.name] = p.stat().st_size
        p = OUT / f"{name}.csv.gz"
        t.to_csv(p, index=False, compression="gzip")
        sizes["files"][p.name] = p.stat().st_size
        csv_bytes = len(t.to_csv(index=False).encode())
        sizes["files"][f"{name}.csv (uncompressed)"] = csv_bytes
        sizes[f"{name}_in_memory_bytes"] = int(t.memory_usage(deep=True).sum())

    # coverage: Open-Meteo's ERA5 boundary-layer height is empty Jan-Jun 2024; check these are not
    nulls = hourly.assign(period=np.where((hourly["time_utc"] >= "2024-01-01") & (hourly["time_utc"] < "2024-07-01"), "2024H1", "other"))
    sizes["null_fraction"] = {p: {v: round(float(f[v].isna().mean()), 4) for v in HOURLY} for p, f in nulls.groupby("period")}
    sizes["elapsed_s"] = round(time.time() - t0)
    (OUT / "sizes.json").write_text(json.dumps(sizes, indent=1))
    print(json.dumps(sizes, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["--sites-arg"]:
        print(sites_arg())
    else:
        try:
            import timezonefinder  # noqa: F401
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timezonefinder"], check=True)
        main(sys.argv[1])
