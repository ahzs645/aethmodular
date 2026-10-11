"""ERA5 boundary-layer height (BLH) at filter sites, via the Open-Meteo archive API.

Why: Ann (9 Oct 2026) wants concentrations normalised by boundary-layer height, so
seasonal differences in emissions are not confused with a shallow Kiremt boundary
layer. ERA5 ``boundary_layer_height`` is hourly, global, 0.25°, 1940 to ~5 days ago;
Open-Meteo serves it per point with no account (non-commercial use; cite Open-Meteo
and "Copernicus Climate Change Service information"). The same variable comes from
the Copernicus CDS directly (``reanalysis-era5-single-levels``, ``boundary_layer_height``)
when a pull is too big for the free API.

Requests are paced under the free-tier limits (600 / min, 5,000 / h, 10,000 / day,
where each 2 weeks of one variable at one site counts as one call) and cached per
site-year as parquet, so an interrupted pull resumes where it stopped.

    from boundary_layer import fetch_site, load_site, window_mean
    fetch_site('ETAD', 9.0, 38.76, '2022-01-01', '2024-12-31', cache)   # downloads missing years
    blh = load_site('ETAD', cache)                                       # hourly UTC series, metres
    window_mean(blh, '2023-07-01 00:00', '2023-07-02 00:00')             # mean over a sampling window
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import date
from pathlib import Path

import pandas as pd

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5"   # pinned: Open-Meteo's default "best match" may blend in other models
LIMITS = ((60, 500), (3600, 4500), (86400, 9500))   # (window s, weighted calls), with margin


class CallBudget:
    """Weighted-call pacing for the free tier: waits until a request fits every window."""

    def __init__(self, limits=LIMITS):
        self.limits = limits
        self.log: deque[tuple[float, float]] = deque()

    def spend(self, weight: float):
        while True:
            now = time.time()
            while self.log and now - self.log[0][0] > max(w for w, _ in self.limits):
                self.log.popleft()
            waits = []
            for window, cap in self.limits:
                used = sum(c for t, c in self.log if now - t < window)
                if used + weight > cap:
                    oldest = next(t for t, _ in self.log if now - t < window)
                    waits.append(oldest + window - now + 1)
            if not waits:
                self.log.append((now, weight))
                return
            wait = max(waits)
            if wait > 3 * 3600:
                raise RuntimeError(f"daily Open-Meteo budget used up; resume in {wait / 3600:.1f} h")
            time.sleep(wait)


def _year_path(cache: Path, code: str, year: int) -> Path:
    return Path(cache) / code / f"{year}.parquet"


def _request(lat: float, lon: float, start: str, end: str) -> dict:
    query = urllib.parse.urlencode({"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": start,
                                    "end_date": end, "hourly": "boundary_layer_height", "models": MODEL,
                                    "timezone": "UTC"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{ARCHIVE}?{query}", timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(60 * (attempt + 1))
                continue
            raise RuntimeError(f"Open-Meteo {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError):
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo kept failing for {lat},{lon} {start}..{end}")


def fetch_site(code: str, lat: float, lon: float, start, end, cache, budget: CallBudget | None = None) -> int:
    """Download every missing site-year in [start, end]; returns the number of years fetched.

    The current year is refetched each run until ERA5 has caught up with its end.
    """
    budget = budget or CallBudget()
    start, end = pd.Timestamp(start).date(), min(pd.Timestamp(end).date(), date.today())
    fetched = 0
    for year in range(start.year, end.year + 1):
        path = _year_path(cache, code, year)
        a, b = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        if path.exists():
            have = pd.read_parquet(path)
            if have.time.min().date() <= a and have.time.max().date() >= b:
                continue
        budget.spend(max(1.0, ((b - a).days + 1) / 14))
        d = _request(lat, lon, a.isoformat(), b.isoformat())
        frame = pd.DataFrame({"time": pd.to_datetime(d["hourly"]["time"]), "blh_m": d["hourly"]["boundary_layer_height"]})
        frame.attrs = {"grid_lat": d.get("latitude"), "grid_lon": d.get("longitude"), "elevation": d.get("elevation")}
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        meta = path.parent / "site.json"
        meta.write_text(json.dumps({"code": code, "lat": lat, "lon": lon, "grid_lat": d.get("latitude"),
                                    "grid_lon": d.get("longitude"), "grid_elevation_m": d.get("elevation"),
                                    "model": MODEL, "source": ARCHIVE}, indent=1))
        fetched += 1
    return fetched


def load_site(code: str, cache) -> pd.Series:
    """Hourly BLH (m) for one site, UTC index, from every cached year."""
    files = sorted((Path(cache) / code).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cached boundary-layer height for {code} in {cache}")
    frame = pd.concat(pd.read_parquet(f).astype({"blh_m": "float64"}) for f in files).drop_duplicates("time").sort_values("time")
    return frame.set_index("time").blh_m


def window_mean(blh: pd.Series, start, end) -> float:
    """Mean BLH over a sampling window [start, end), both UTC (e.g. a filter's start and end)."""
    part = blh.loc[pd.Timestamp(start):pd.Timestamp(end) - pd.Timedelta(seconds=1)]
    return float(part.mean()) if len(part) else float("nan")


def daily_summary(blh: pd.Series, utc_offset_hours: float = 0.0) -> pd.DataFrame:
    """Per local day: mean, daytime maximum and night-time (00-06 local) mean BLH."""
    local = blh.copy()
    local.index = local.index + pd.Timedelta(hours=utc_offset_hours)
    day = local.groupby(local.index.date)
    night = local[local.index.hour < 6].groupby(local[local.index.hour < 6].index.date).mean()
    out = pd.DataFrame({"blh_mean_m": day.mean(), "blh_max_m": day.max(), "blh_night_m": night, "hours": day.size()})
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out
