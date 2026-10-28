"""Weather over each filter's sampling window, for the gallery's Weather tab and weather fields.

Run with:  uv run --with timezonefinder python gallery/data/export_weather.py

Inputs (all local; see research/ftir_hips_chem/workflows/colab_spartan_weather.py and
fetch_boundary_layer_era5.py for how they were made):
  research/ftir_hips_chem/output/tables/weather_era5/spartan_hourly.parquet   ERA5 hourly, 37 SPARTAN sites
  research/ftir_hips_chem/output/tables/boundary_layer_era5/spartan/<CODE>/   ERA5 boundary-layer height
  research/ftir_hips_chem/Weather Data/Meteostat/master_meteostat_AddisAbaba_...csv   Addis airport station (WMO 63450), hourly
  data/spartan/raw/FilterBased/ChemSpecPM25/*.csv   each filter's local start, end and hours sampled

Each filter's weather is the mean over its sampling window (start to end, local time). Most
filters run 24 h (Addis 09:00 to 09:00); older intermittent ones spread their hours over
about 8 days, and the window is then the whole elapsed period. A window that is missing,
negative or longer than 40 days falls back to 24 h from the start; filters not in the
public files (focus-site HIPS filters) get 24 h from their gallery date, flagged "assumed".

Writes to gallery/app/public/data/weather/:
  filters.json        per filter: the weather fields below, ERA5 and (Addis) station (the whole set, for scripts)
  index.json + filters/<CODE>.json   the same split by site: the gallery loads only the sites in view
                      (``--split-only`` rewrites these from an existing filters.json)
  roses.json          per filter: hours by wind sector (8) and speed class (3), plus calm hours
  wind.json           per filter: calm hours, then hours by wind sector (16) and 0.5 m/s speed bin, sparse
                      (the Weather tab's pollution rose and polar plot)
  daily/<CODE>.json   per site: daily ERA5 (local standard day) for the timeline and climatology
  station_ETAD.json   daily station values beside ERA5 at Addis, for the comparison chart

ERA5 data: Copernicus Climate Change Service information, via the Open-Meteo archive API (CC BY 4.0).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "research/ftir_hips_chem"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(AREA / "scripts"))
from common.spartan_io import RAW_DIR, normalize_fid, read_spartan_csv  # noqa: E402
from boundary_layer import load_site  # noqa: E402

HOURLY = AREA / "output/tables/weather_era5/spartan_hourly.parquet"
BLH_CACHE = AREA / "output/tables/boundary_layer_era5/spartan"
STATION = AREA / "Weather Data/Meteostat/master_meteostat_AddisAbaba_63450_2022-12-01_2024-10-01.csv"
OUT = ROOT / "gallery/app/public/data/weather"

# gallery field name -> (unit, how it is made over a window)
ERA5_FIELDS = {
    "Temperature (ERA5)": "°C",
    "Temperature range (ERA5)": "°C",
    "Precipitation (ERA5)": "mm/day",
    "Relative humidity (ERA5)": "%",
    "Dew point (ERA5)": "°C",
    "Wind speed (ERA5)": "m/s",
    "Surface pressure (ERA5)": "hPa",
    "Boundary-layer height (ERA5)": "m",
}
# the station reports sea-level pressure (Meteostat "pres"), not surface pressure
STATION_FIELDS = {("Sea-level pressure (station)" if k.startswith("Surface pressure") else k.replace("(ERA5)", "(station)")): u
                  for k, u in ERA5_FIELDS.items() if "Boundary" not in k and "Dew" not in k}
SECTORS = 8
SPEED_EDGES = (1.0, 3.0)   # m/s: light < 1, moderate 1-3, fresh >= 3 (ERA5 10 m winds run low in cities)
CALM = 0.5                 # m/s
# the finer grid of wind.json: 16 sectors, and speed bins from calm up (the last bin is open)
WIND_SECTORS = 16
WIND_EDGES = (CALM, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
WIND_NAMES = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def std_offset_hours(lat: float, lon: float) -> float:
    from timezonefinder import TimezoneFinder
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC")
    return min(datetime(2023, m, 15, 12, tzinfo=tz).utcoffset().total_seconds() / 3600 for m in (1, 7))


def windows() -> pd.DataFrame:
    """Local start / end and hours sampled for every public SPARTAN filter."""
    frames = []
    for p in sorted((RAW_DIR / "FilterBased/ChemSpecPM25").glob("*.csv")):
        r = read_spartan_csv(p)
        r["id"] = normalize_fid(r.Filter_ID)
        frames.append(r.groupby("id").agg(code=("Site_Code", "first"), sy=("Start_Year_local", "first"), sm=("Start_Month_local", "first"),
            sd=("Start_Day_local", "first"), sh=("Start_hour_local", "first"), ey=("End_Year_local", "first"), em=("End_Month_local", "first"),
            ed=("End_Day_local", "first"), eh=("End_hour_local", "first"), hours=("Hours_sampled", "first")))
    w = pd.concat(frames)
    mk = lambda y, m, d, h: pd.to_datetime(dict(year=y, month=m, day=d, hour=pd.to_numeric(h, errors="coerce").fillna(0)), errors="coerce")
    w["start"], w["end"] = mk(w.sy, w.sm, w.sd, w.sh), mk(w.ey, w.em, w.ed, w.eh)
    w["assumed"] = False
    bad = w.end.isna() | (w.end <= w.start) | ((w.end - w.start) > pd.Timedelta(days=40))
    w.loc[bad, "end"] = w.loc[bad, "start"] + pd.Timedelta(hours=24)
    w.loc[bad, "assumed"] = True
    return w[["code", "start", "end", "hours", "assumed"]]


def window_stats(h: pd.DataFrame, temp, rh, prcp, wspd, wdir, pres, dew=None, blh=None) -> dict:
    """Means over one window's hours (the columns are names in h)."""
    if len(h) == 0:
        return {}
    hours = len(h)
    t = h[temp]
    # temperature range: max - min in each successive 24 h block of the window, averaged
    blocks = np.arange(hours) // 24
    rng = t.groupby(blocks).agg(lambda s: s.max() - s.min() if s.count() >= 12 else np.nan)
    out = {
        "temp": t.mean(), "range": rng.mean(),
        "prcp": h[prcp].sum(min_count=1) / (hours / 24) if h[prcp].notna().any() else np.nan,
        "rh": h[rh].mean(), "wspd": h[wspd].mean(), "pres": h[pres].mean(),
    }
    if dew:
        out["dew"] = h[dew].mean()
    if blh is not None:
        out["blh"] = blh
    return out


def rose(h: pd.DataFrame, wspd: str, wdir: str) -> list[int]:
    """Hours by sector (N, NE, … NW) x speed class, then calm hours: 25 counts."""
    s, d = h[wspd].to_numpy(), h[wdir].to_numpy()
    ok = np.isfinite(s) & np.isfinite(d)
    s, d = s[ok], d[ok]
    calm = s < CALM
    sector = ((d[~calm] + 360 / SECTORS / 2) % 360 // (360 / SECTORS)).astype(int)
    speed = np.digitize(s[~calm], SPEED_EDGES)
    counts = np.zeros((SECTORS, len(SPEED_EDGES) + 1), dtype=int)
    np.add.at(counts, (sector, speed), 1)
    return [*counts.ravel().tolist(), int(calm.sum())]


def wind_cells(h: pd.DataFrame, wspd: str, wdir: str) -> list[int]:
    """Calm hours, then (cell, hours) pairs for every cell with hours: cell = sector * bins + speed bin."""
    s, d = h[wspd].to_numpy(), h[wdir].to_numpy()
    ok = np.isfinite(s) & np.isfinite(d)
    s, d = s[ok], d[ok]
    calm = s < CALM
    sector = ((d[~calm] + 360 / WIND_SECTORS / 2) % 360 // (360 / WIND_SECTORS)).astype(int)
    speed = np.digitize(s[~calm], WIND_EDGES) - 1
    counts = np.zeros((WIND_SECTORS, len(WIND_EDGES)), dtype=int)
    np.add.at(counts, (sector, speed), 1)
    flat = counts.ravel()
    return [int(calm.sum()), *[int(v) for i in np.flatnonzero(flat) for v in (i, flat[i])]]


def r(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def write_split(payload: dict) -> None:
    """filters.json split by site (a filter id starts with its site code): a small index with the
    fields, units and sources, and one file per site with its filters' values and windows."""
    rows, wins = {}, {}
    for fid, v in payload["rows"].items():
        rows.setdefault(fid.split("-")[0], {})[fid] = v
    for fid, v in payload.get("windows", {}).items():
        wins.setdefault(fid.split("-")[0], {})[fid] = v
    folder = OUT / "filters"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.json"):
        old.unlink()
    for code in sorted(rows):
        (folder / f"{code}.json").write_text(json.dumps(dict(rows=rows[code], windows=wins.get(code, {})),
                                                        separators=(",", ":"), allow_nan=False))
    index = {k: v for k, v in payload.items() if k not in ("rows", "windows")}
    index["sites"] = {code: len(v) for code, v in sorted(rows.items())}
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":"), allow_nan=False))
    print(f"weather split: {len(rows)} sites, {sum(index['sites'].values())} filters")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "daily").mkdir(exist_ok=True)
    smap = {s["code"]: s for s in json.loads((ROOT / "gallery/app/public/data/spartan_map.json").read_text())["sites"]}
    hourly = pd.read_parquet(HOURLY)
    hourly["site"] = hourly["site"].astype(str)
    w = windows()

    # the focus filters' gallery ids, some of which the public files do not carry
    focus = json.loads((ROOT / "gallery/app/public/data/filters.json").read_text())["rows"]
    extra = [f for f in focus if f["id"] not in w.index and f.get("code")]
    for f in extra:
        start = pd.Timestamp(f["date"])
        w.loc[f["id"]] = [f["code"], start, start + pd.Timedelta(hours=24), 24.0, True]

    offsets = {c: std_offset_hours(s["lat"], s["lon"]) for c, s in smap.items() if s.get("lat") is not None}
    era5_keys = ["temp", "range", "prcp", "rh", "dew", "wspd", "pres", "blh"]
    rows, roses, winds, wins = {}, {}, {}, {}
    for code, part in w.groupby("code"):
        site_h = hourly[hourly.site == code].set_index("time_utc").sort_index()
        if site_h.empty or code not in offsets:
            continue
        try:
            blh = load_site(code, BLH_CACHE)
        except FileNotFoundError:
            blh = None
        off = pd.Timedelta(hours=offsets[code])
        for fid, x in part.iterrows():
            a, b = x.start - off, x.end - off       # local standard time -> UTC
            h = site_h.loc[a:b - pd.Timedelta(minutes=1)]
            if h.empty:
                continue
            bl = blh.loc[a:b - pd.Timedelta(minutes=1)].mean() if blh is not None else np.nan
            st = window_stats(h.reset_index(), "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m",
                              "wind_direction_10m", "surface_pressure", dew="dew_point_2m", blh=bl)
            rows[fid] = [r(st.get(k), 1 if k in ("blh", "pres") else 2) for k in era5_keys]
            roses[fid] = rose(h, "wind_speed_10m", "wind_direction_10m")
            winds[fid] = wind_cells(h, "wind_speed_10m", "wind_direction_10m")
            wins[fid] = [x.start.strftime("%Y-%m-%d %H:%M"), x.end.strftime("%Y-%m-%d %H:%M"), r(x.hours, 1), bool(x.assumed)]
        print(f"{code}: {sum(1 for i in part.index if i in rows)} filters")

    # Addis airport station (Meteostat hourly is in UTC; wind in km/h)
    st = pd.read_csv(STATION, parse_dates=["time"], date_format="%m/%d/%Y %H:%M").set_index("time").sort_index()
    st["wspd"] = st["wspd"] / 3.6
    station_keys = ["temp", "range", "prcp", "rh", "wspd", "pres"]
    srows, sroses, swinds = {}, {}, {}
    off = pd.Timedelta(hours=offsets["ETAD"])
    for fid, x in w[w.code == "ETAD"].iterrows():
        a, b = x.start - off, x.end - off
        h = st.loc[a:b - pd.Timedelta(minutes=1)]
        if len(h) < 12:       # the station file has gaps; skip windows it barely covers
            continue
        s = window_stats(h.reset_index(), "temp", "rhum", "prcp", "wspd", "wdir", "pres")
        srows[fid] = [r(s.get(k), 1 if k == "pres" else 2) for k in station_keys]
        sroses[fid] = rose(h, "wspd", "wdir")
        swinds[fid] = wind_cells(h, "wspd", "wdir")

    era5_names, station_names = list(ERA5_FIELDS), list(STATION_FIELDS)
    meta = dict(generated=datetime.now(timezone.utc).isoformat(),
                source="ERA5 (Copernicus Climate Change Service) via the Open-Meteo archive API, CC BY 4.0; Addis station: Meteostat, WMO 63450 (Bole airport)",
                note="Means over each filter's sampling window (local start to end). Precipitation is the window mean in mm/day; temperature range is the mean max-min over 24 h blocks.")
    payload = dict(**meta, fields=era5_names + station_names, field_units={**ERA5_FIELDS, **STATION_FIELDS},
                   era5=era5_names, station=station_names,
                   rows={fid: v + srows.get(fid, [None] * len(station_names)) for fid, v in rows.items()},
                   windows=wins)
    (OUT / "filters.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    write_split(payload)
    (OUT / "roses.json").write_text(json.dumps(dict(**meta, sectors=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        speed_classes=[f"< {SPEED_EDGES[0]:g} m/s", f"{SPEED_EDGES[0]:g}–{SPEED_EDGES[1]:g} m/s", f"≥ {SPEED_EDGES[1]:g} m/s"], calm=f"< {CALM:g} m/s",
        era5=roses, station=sroses), separators=(",", ":")))
    (OUT / "wind.json").write_text(json.dumps(dict(**meta, sectors=WIND_NAMES, speed_edges=list(WIND_EDGES), calm=f"< {CALM:g} m/s",
        note_cells="per filter: calm hours, then (cell, hours) pairs; cell = sector * len(speed_edges) + speed bin, bin k from speed_edges[k] (the last open)",
        era5=winds, station=swinds), separators=(",", ":")))

    # daily per site, for the timeline and the monthly climatology
    daily = pd.read_parquet(HOURLY.with_name("spartan_daily.parquet"))
    daily["site"] = daily["site"].astype(str)
    cols = {"t_mean": "t", "t_min": "tmin", "t_max": "tmax", "precip_sum": "p", "rh_mean": "rh", "wind_speed_mean": "ws"}
    for code, d in daily.groupby("site"):
        d = d.sort_values("date")
        try:
            blh = load_site(code, BLH_CACHE)
            off = pd.Timedelta(hours=offsets[code])
            bd = blh.groupby((blh.index + off).normalize()).mean()
            d["blh"] = d["date"].map(bd)
        except FileNotFoundError:
            d["blh"] = np.nan
        out = {"date": d["date"].dt.strftime("%Y-%m-%d").tolist()}
        for c, k in {**cols, "blh": "blh"}.items():
            out[k] = [r(v, 0 if k == "blh" else 1) for v in d[c]]
        (OUT / "daily" / f"{code}.json").write_text(json.dumps(out, separators=(",", ":")))

    # Addis: station beside ERA5, by local day, for the comparison chart
    sd = st.copy()
    sd["local_date"] = (sd.index + pd.Timedelta(hours=offsets["ETAD"])).normalize()
    g = sd.groupby("local_date")
    sday = pd.DataFrame({"t": g.temp.mean(), "tmin": g.temp.min(), "tmax": g.temp.max(), "p": g.prcp.sum(min_count=1),
                         "rh": g.rhum.mean(), "ws": g.wspd.mean(), "n": g.temp.count()})
    sday = sday[sday.n >= 18]
    e = daily[daily.site == "ETAD"].set_index("date")
    both = sday.join(e[["t_mean", "t_min", "t_max", "precip_sum", "rh_mean", "wind_speed_mean"]], how="inner")
    comp = {"date": both.index.strftime("%Y-%m-%d").tolist(),
            "station": {k: [r(v, 1) for v in both[k]] for k in ["t", "tmin", "tmax", "p", "rh", "ws"]},
            "era5": {k: [r(v, 1) for v in both[c]] for k, c in zip(["t", "tmin", "tmax", "p", "rh", "ws"],
                     ["t_mean", "t_min", "t_max", "precip_sum", "rh_mean", "wind_speed_mean"])}}
    (OUT / "station_ETAD.json").write_text(json.dumps(dict(**meta, **comp), separators=(",", ":")))

    sizes = {p.name: p.stat().st_size for p in OUT.glob("*.json")}
    print(f"{len(rows)} filters with ERA5 weather ({sum(1 for v in wins.values() if v[3])} assumed windows), "
          f"{len(srows)} with station weather; daily files for {daily.site.nunique()} sites; sizes {sizes}")


if __name__ == "__main__":
    if "--split-only" in sys.argv:
        write_split(json.loads((OUT / "filters.json").read_text()))
    else:
        main()
