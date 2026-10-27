"""Pull AERONET for every SPARTAN site we hold filters for: raw downloads and every individual retrieval.

For each site, the AERONET stations within 50 km are ranked by how many filter days have a daily
AOD, and the best one is pulled over the site's filter period (padded by its longest sampling
window). Six products, each by calendar year so a rerun only fetches what is missing:

    aod_points   direct-sun AOD, every retrieval (AVG=10)     aod_daily   AERONET's daily means (AVG=20)
    sda_points   spectral deconvolution: fine/coarse AOD      sda_daily
    inv_points   almucantar inversion: AAOD, SSA, AAE         inv_daily

The raw text is kept as downloaded (gzipped) and parsed into one parquet per product, one row per
retrieval or day, with the columns the gallery export reads (gallery/data/export_aeronet.py). Every
other column stays in the raw files.

Run from the repo root:
    uv run python research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py
    uv run python research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py --sites ETAD INDH
    uv run python research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py --level 20
    uv run python research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py --parse-only   # rebuild the parquet

Writes research/ftir_hips_chem/output/tables/aeronet_spartan/ (gitignored):
    raw/<CODE>/<station>/lev<L>/<product>_<year>.txt.gz   the downloads, unchanged
    raw/aeronet_locations_v3.txt                         AERONET's site list
    stations.csv          each site's candidate stations, distance, and calendar days its filter windows touch that have a daily AOD; `chosen` marks the pull
    <product>_lev<L>.parquet                             parsed retrievals / days (site, station, time_utc, …)
    manifest.json         every request: URL, bytes, rows, when

Level 1.5 (default) is cloud-screened and quality-controlled but not final-calibrated; Level 2.0
is final but lags by months and drops low-AOD inversions. A Level 2.0 pull uses the stations the
Level 1.5 pull chose (stations_lev15.csv, copied to stations_lev20.csv), so the two levels describe
the same photometer; without a Level 1.5 table it ranks the stations on Level 2.0 itself. Data: NASA AERONET (aeronet.gsfc.nasa.gov);
cite the station PIs as AERONET asks.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
CACHE = AREA / "output/tables/aeronet_spartan"
GALLERY = ROOT / "gallery/app/public/data"
API = "https://aeronet.gsfc.nasa.gov/cgi-bin"
LOCATIONS = "https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt"
MAX_KM = 50
MAX_CANDIDATES = 3

# product -> (endpoint and query, the columns kept in the parquet: AERONET name -> short name)
PRODUCTS = {
    "aod": ("print_web_data_v3?AOD{L}=1", {
        "AOD_340nm": "aod340", "AOD_380nm": "aod380", "AOD_440nm": "aod440", "AOD_500nm": "aod500",
        "AOD_675nm": "aod675", "AOD_870nm": "aod870", "AOD_1020nm": "aod1020",
        "440-870_Angstrom_Exponent": "ae440_870", "Precipitable_Water(cm)": "pw_cm",
        "Solar_Zenith_Angle(Degrees)": "sza",
    }),
    "sda": ("print_web_data_v3?SDA{L}=1", {
        "Total_AOD_500nm[tau_a]": "sda_aod500", "Fine_Mode_AOD_500nm[tau_f]": "fine_aod500",
        "Coarse_Mode_AOD_500nm[tau_c]": "coarse_aod500", "FineModeFraction_500nm[eta]": "fmf500",
    }),
    "inv": ("print_web_data_inv_v3?product=ALL&ALM{L}=1", {
        "AOD_Coincident_Input[440nm]": "inv_aod440", "AOD_Coincident_Input[675nm]": "inv_aod675",
        "AOD_Coincident_Input[870nm]": "inv_aod870",
        "Absorption_AOD[440nm]": "aaod440", "Absorption_AOD[675nm]": "aaod675", "Absorption_AOD[870nm]": "aaod870",
        "Absorption_AOD[1020nm]": "aaod1020", "Absorption_Angstrom_Exponent_440-870nm": "aae440_870",
        "Single_Scattering_Albedo[440nm]": "ssa440", "Single_Scattering_Albedo[675nm]": "ssa675",
        "Single_Scattering_Albedo[870nm]": "ssa870", "Single_Scattering_Albedo[1020nm]": "ssa1020",
        "Extinction_Angstrom_Exponent_440-870nm-Total": "eae440_870", "Sphericity_Factor(%)": "sphericity",
        "If_Retrieval_is_L2(without_L2_0.4_AOD_440_threshold)": "l2_checks",
    }),
}
AVERAGING = {"points": 10, "daily": 20}


def km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def get(url: str) -> bytes:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=600) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def filter_windows() -> pd.DataFrame:
    """Every SPARTAN filter's sampling window (local), from the weather export; else 24 h from its date."""
    weather = json.loads((GALLERY / "weather/filters.json").read_text())["windows"]
    rows = [{"id": fid, "start": pd.Timestamp(w[0]), "end": pd.Timestamp(w[1])} for fid, w in weather.items()]
    have = set(weather)
    for name in ("spartan_reference.json", "filters.json"):
        for r in json.loads((GALLERY / name).read_text())["rows"]:
            if r.get("date") and r["id"] not in have:
                start = pd.Timestamp(r["date"])
                rows.append({"id": r["id"], "start": start, "end": start + pd.Timedelta(hours=24)})
                have.add(r["id"])
    frame = pd.DataFrame(rows)
    frame["code"] = frame["id"].str.split("-").str[0]
    return frame


def window_days(windows: pd.DataFrame) -> set[str]:
    """Calendar days touched by any of these windows."""
    days = set()
    for start, end in zip(windows["start"], windows["end"]):
        d = start.normalize()
        while d < end:
            days.add(d.strftime("%Y-%m-%d"))
            d += pd.Timedelta(days=1)
    return days


def query(product: str, kind: str, station: str, start: date, end: date, level: str) -> str:
    base = PRODUCTS[product][0].replace("{L}", level)
    return (f"{API}/{base}&site={station}&year={start.year}&month={start.month}&day={start.day}"
            f"&year2={end.year}&month2={end.month}&day2={end.day}&AVG={AVERAGING[kind]}&if_no_html=1")


def parse(text: str, keep: dict[str, str]) -> pd.DataFrame:
    """One AERONET download -> station, UTC time and the kept columns (-999 -> missing)."""
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith("AERONET_Site") or line.startswith("AERONET_AERONET_Site")), None)
    if header is None or header + 1 >= len(lines):
        return pd.DataFrame(columns=["station", "time_utc", *keep.values()])
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:])), na_values=["-999", "-999.", "-999.0", "-999.000000"],
                        low_memory=False)
    date_col = next(c for c in frame.columns if c.startswith("Date"))
    time_col = next(c for c in frame.columns if c.startswith("Time"))
    out = pd.DataFrame({
        "station": frame.iloc[:, 0].astype(str),
        "time_utc": pd.to_datetime(frame[date_col] + " " + frame[time_col], format="%d:%m:%Y %H:%M:%S", errors="coerce"),
    })
    for name, short in keep.items():
        out[short] = pd.to_numeric(frame[name], errors="coerce") if name in frame.columns else float("nan")
    return out.dropna(subset=["time_utc"])


def rank_stations(sites: pd.DataFrame, windows: pd.DataFrame, level: str, locations: list[tuple]) -> pd.DataFrame:
    """Each site's candidate stations, by filter days that have a daily AOD (one cheap request each)."""
    jobs = []
    for site in sites.itertuples():
        w = windows[windows.code == site.code]
        if w.empty:
            continue
        near = sorted((km(site.lat, site.lon, lat, lon), name) for name, lon, lat in locations)
        for dist, name in [n for n in near if n[0] <= MAX_KM][:MAX_CANDIDATES]:
            jobs.append((site.code, name, round(dist, 1), w.start.min().date() - timedelta(days=1), w.end.max().date() + timedelta(days=1)))

    def run(job):
        code, name, dist, a, b = job
        days = parse(get(query("aod", "daily", name, a, b, level)).decode("latin-1"), {})
        return job, set(pd.to_datetime(days["time_utc"]).dt.strftime("%Y-%m-%d"))

    rows = []
    for (code, name, dist, a, b), days in ThreadPoolExecutor(6).map(run, jobs):
        fdays = window_days(windows[windows.code == code])
        rows.append({"code": code, "station": name, "km": dist, "first": a.isoformat(), "last": b.isoformat(),
                     "window_days": len(fdays), "aod_days": len(days), "window_days_with_aod": len(fdays & days)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # the station with the most filter days covered; distance breaks ties
    frame = frame.sort_values(["code", "window_days_with_aod", "km"], ascending=[True, False, True])
    frame["chosen"] = ~frame.duplicated("code") & (frame["window_days_with_aod"] > 0)
    return frame.reset_index(drop=True)


def pull(stations: pd.DataFrame, level: str, refresh: bool) -> list[dict]:
    jobs = []
    for s in stations[stations.chosen].itertuples():
        a, b = date.fromisoformat(s.first), date.fromisoformat(s.last)
        for year in range(a.year, b.year + 1):
            ya, yb = max(a, date(year, 1, 1)), min(b, date(year, 12, 31))
            for product in PRODUCTS:
                for kind in AVERAGING:
                    path = CACHE / "raw" / s.code / s.station / f"lev{level}" / f"{product}_{kind}_{year}.txt.gz"
                    if refresh or not path.exists():
                        jobs.append((path, query(product, kind, s.station, ya, yb, level)))

    def run(job):
        path, url = job
        body = get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(gzip.compress(body))
        tmp.replace(path)
        return {"path": str(path.relative_to(CACHE)), "url": url, "bytes": len(body),
                "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    print(f"{len(jobs)} downloads to make")
    log = []
    with ThreadPoolExecutor(6) as pool:
        for i, entry in enumerate(pool.map(run, jobs), 1):
            log.append(entry)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({sum(e['bytes'] for e in log) / 1e6:.0f} MB)")
    return log


def build_parquet(stations: pd.DataFrame, level: str) -> dict[str, int]:
    counts = {}
    chosen = stations[stations.chosen]
    for product, (_, keep) in PRODUCTS.items():
        for kind in AVERAGING:
            frames = []
            for s in chosen.itertuples():
                for path in sorted((CACHE / "raw" / s.code / s.station / f"lev{level}").glob(f"{product}_{kind}_*.txt.gz")):
                    part = parse(gzip.decompress(path.read_bytes()).decode("latin-1"), keep)
                    if len(part):
                        part.insert(0, "site", s.code)
                        frames.append(part)
            table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["site", "station", "time_utc", *keep.values()])
            table = table.drop_duplicates(["site", "time_utc"]).sort_values(["site", "time_utc"])
            table.to_parquet(CACHE / f"{product}_{kind}_lev{level}.parquet", index=False)
            counts[f"{product}_{kind}"] = len(table)
            print(f"  {product}_{kind}_lev{level}.parquet: {len(table):,} rows")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*", help="only these SPARTAN site codes")
    ap.add_argument("--level", choices=["15", "20"], default="15", help="AERONET level: 1.5 (default) or 2.0")
    ap.add_argument("--refresh", action="store_true", help="download again even where a year is cached (to pick up reprocessing)")
    ap.add_argument("--parse-only", action="store_true", help="rebuild the parquet from the cached raw files; no network")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    sites = pd.DataFrame(json.loads((GALLERY / "spartan_map.json").read_text())["sites"])
    if args.sites:
        sites = sites[sites.code.isin(args.sites)]
    windows = filter_windows()
    station_file = CACHE / f"stations_lev{args.level}.csv"

    if args.parse_only:
        stations = pd.read_csv(station_file)
        if args.sites:
            stations = stations[stations.code.isin(args.sites)]
    elif args.level == "20" and (CACHE / "stations_lev15.csv").exists():
        # the same photometer at both levels: Level 2.0 only re-reads the stations Level 1.5 chose
        stations = pd.read_csv(CACHE / "stations_lev15.csv")
        stations.to_csv(station_file, index=False)
        log = pull(stations if not args.sites else stations[stations.code.isin(args.sites)], args.level, args.refresh)
        manifest_path = CACHE / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        manifest_path.write_text(json.dumps(manifest + log, indent=1))
    else:
        loc_path = CACHE / "raw/aeronet_locations_v3.txt"
        loc_path.parent.mkdir(parents=True, exist_ok=True)
        loc_path.write_bytes(get(LOCATIONS))
        locations = [(r[0], float(r[1]), float(r[2])) for r in pd.read_csv(loc_path, skiprows=1).itertuples(index=False)]
        stations = rank_stations(sites, windows, args.level, locations)
        if args.sites and station_file.exists():
            old = pd.read_csv(station_file)
            stations = pd.concat([old[~old.code.isin(args.sites)], stations], ignore_index=True)
        stations.to_csv(station_file, index=False)
        print(stations[stations.chosen][["code", "station", "km", "window_days", "window_days_with_aod"]].to_string(index=False))
        log = pull(stations if not args.sites else stations[stations.code.isin(args.sites)], args.level, args.refresh)
        manifest_path = CACHE / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        manifest_path.write_text(json.dumps(manifest + log, indent=1))

    counts = build_parquet(pd.read_csv(station_file), args.level)
    raw_mb = sum(p.stat().st_size for p in (CACHE / "raw").rglob("*.gz")) / 1e6
    print(f"done: {int(stations.chosen.sum())} stations, raw cache {raw_mb:.0f} MB gzipped, rows {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
