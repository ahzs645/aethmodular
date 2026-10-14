"""ERA5 boundary-layer height for every IMPROVE sample day, read from ARCO-ERA5 on Colab VMs.

IMPROVE samples 24 h, midnight to midnight local standard time, on a shared 1-in-3-day
schedule. Instead of every hour of 2003-2026 (~200k hours), only the UTC hours that cover
some site's sample day are read (~88k hours, ~280 GB on the VM). Each chunk runs on its
own Colab VM (Colab runtime tokens expire after an hour), returns a few MB of site values,
and is downloaded to the gitignored cache. The Open-Meteo free tier could not do this
(~96,000 calls), and the Jan-Jun 2024 hole Open-Meteo has does not exist in ARCO.

Run from the repo root:
    uv run --with timezonefinder python research/ftir_hips_chem/workflows/improve_blh_colab.py timezones
    uv run python research/ftir_hips_chem/workflows/improve_blh_colab.py plan
    uv run python research/ftir_hips_chem/workflows/improve_blh_colab.py run 2003-2010     # one VM per chunk
    uv run python research/ftir_hips_chem/workflows/improve_blh_colab.py run 2011-2018
    uv run python research/ftir_hips_chem/workflows/improve_blh_colab.py run 2019-2026
    uv run python research/ftir_hips_chem/workflows/improve_blh_colab.py assemble          # tracked dataset

Sites and coordinates: gallery/app/public/data/improve_map.json (217 sites; FOCO2 has none).
Sample dates: the validated IMPROVE archive's cleaned table (improve_io.improve_clean_path()).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
from improve_io import improve_clean_path  # noqa: E402

CACHE = AREA / "output/tables/boundary_layer_era5/improve"
OUT = AREA / "Weather Data/ERA5 boundary layer"
JOB = AREA / "workflows/colab_arco_blh_gapfill.py"
SITES_MAP = ROOT / "gallery/app/public/data/improve_map.json"
TZ_TABLE = CACHE / "site_timezones.csv"
EPOCH = pd.Timestamp("1900-01-01")
CHUNKS = {"2003-2010": (2003, 2010), "2011-2018": (2011, 2018), "2019-2026": (2019, 2026)}


def sites() -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(SITES_MAP.read_text())["sites"])
    return frame.dropna(subset=["lat", "lon"])[["code", "name", "lat", "lon", "date_min", "date_max"]]


def timezones():
    """IANA zone and standard (no-DST) UTC offset per site; IMPROVE days are local standard time."""
    from zoneinfo import ZoneInfo
    from timezonefinder import TimezoneFinder
    tf, rows = TimezoneFinder(), []
    for s in sites().itertuples():
        tz = tf.timezone_at(lat=s.lat, lng=s.lon) or tf.closest_timezone_at(lat=s.lat, lng=s.lon)
        t = datetime(2023, 1, 15, 12, tzinfo=ZoneInfo(tz))
        rows.append({"code": s.code, "tz": tz, "std_utc_offset_h": (t.utcoffset() - t.dst()).total_seconds() / 3600})
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TZ_TABLE, index=False)
    print(f"{len(rows)} sites -> {TZ_TABLE.relative_to(ROOT)}")


def sample_days() -> pd.DataFrame:
    """Every (site, schedule date) inside each site's active range, with its UTC offset."""
    dates = pd.to_datetime(pd.read_csv(improve_clean_path(), usecols=["Date"]).Date).drop_duplicates().sort_values()
    s = sites().merge(pd.read_csv(TZ_TABLE), on="code")
    rows = []
    for x in s.itertuples():
        d = dates[(dates >= pd.Timestamp(x.date_min)) & (dates <= pd.Timestamp(x.date_max))]
        rows.append(pd.DataFrame({"code": x.code, "date": d.to_numpy(), "offset": x.std_utc_offset_h}))
    return pd.concat(rows, ignore_index=True)


def chunk_hours(days: pd.DataFrame, first: int, last: int) -> list[int]:
    part = days.loc[days.date.dt.year.between(first, last), ["date", "offset"]].drop_duplicates()
    start = ((part.date - EPOCH) / pd.Timedelta(hours=1)).astype(int) - part.offset.astype(int)
    return sorted({h for s in start for h in range(s, s + 24)})


def runs(hours: list[int]) -> str:
    out, a = [], hours[0]
    for prev, h in zip(hours, hours[1:] + [None]):
        if h != prev + 1:
            out.append(f"{a}-{prev}")
            a = h
    return "r:" + ",".join(out)


def plan():
    days = sample_days()
    for name, (a, b) in CHUNKS.items():
        h = chunk_hours(days, a, b)
        (CACHE / f"plan_{name}.txt").write_text(runs(h))
        print(f"{name}: {len(h):,} hours, ~{len(h) * 3.19 / 1000:.0f} GB on the VM")


def run(name: str):
    spec = (CACHE / f"plan_{name}.txt").read_text()
    site_arg = subprocess.run([sys.executable, str(JOB), "--sites-arg", "improve"], capture_output=True, text=True, check=True).stdout.strip()
    session = f"blh-improve-{name}"
    local = CACHE / f"arco_hours_{name}.npz"
    try:
        subprocess.run(["colab", "run", "--keep", "-s", session, "--timeout", "3500", str(JOB), spec, site_arg], check=True)
        subprocess.run(["colab", "download", "-s", session, "/content/arco_blh_fill.npz", str(local)], check=True)
    finally:
        subprocess.run(["colab", "stop", "-s", session])
    print(f"-> {local.relative_to(ROOT)} ({local.stat().st_size / 1e6:.1f} MB)")


def assemble():
    tz = pd.read_csv(TZ_TABLE).set_index("code")
    parts = [np.load(f) for f in sorted(CACHE.glob("arco_hours_*.npz"))]
    if len(parts) != len(CHUNKS):
        raise SystemExit(f"{len(parts)} of {len(CHUNKS)} chunks downloaded; run the rest first")
    codes = [c.split("/", 1)[1] for c in parts[0]["code"]]
    times = np.concatenate([p["time"] for p in parts]).astype("datetime64[h]")
    values = np.concatenate([p["blh_m"] for p in parts])
    hourly = pd.DataFrame(values, index=pd.to_datetime(times), columns=codes).sort_index()
    hourly = hourly[~hourly.index.duplicated()]
    days = sample_days()
    rows = []
    for code, g in days.groupby("code"):
        series = hourly[code]
        local = series.copy()
        local.index = local.index + pd.Timedelta(hours=float(tz.loc[code, "std_utc_offset_h"]))
        for d in g.date:
            day = local.loc[d:d + pd.Timedelta(hours=23)]
            rows.append({"code": code, "date": d.date().isoformat(), "blh_mean_m": day.mean(), "blh_max_m": day.max(),
                         "blh_night_m": day.loc[d:d + pd.Timedelta(hours=5)].mean(), "hours": int(day.notna().sum())})
    out = pd.DataFrame(rows).round(1)
    meta = sites().merge(tz.reset_index(), on="code")
    stats = out.groupby("code").agg(sample_days=("date", "size"), first_day=("date", "min"), last_day=("date", "max"),
                                    median_blh_mean_m=("blh_mean_m", "median"), incomplete_days=("hours", lambda h: int((h < 24).sum())))
    meta = meta.drop(columns=["date_min", "date_max"]).merge(stats.reset_index(), on="code", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "blh_sampleday_improve.csv", index=False)
    meta.to_csv(OUT / "sites_improve.csv", index=False)
    print(f"improve: {meta.code.size} sites, {len(out):,} site sample days, {int((out.hours < 24).sum())} incomplete -> {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["timezones", "plan", "run", "assemble"])
    ap.add_argument("chunk", nargs="?", choices=sorted(CHUNKS))
    args = ap.parse_args()
    if args.stage == "run" and not args.chunk:
        raise SystemExit("run needs a chunk: " + ", ".join(CHUNKS))
    {"timezones": timezones, "plan": plan, "assemble": assemble}.get(args.stage, lambda: run(args.chunk))()


if __name__ == "__main__":
    main()
