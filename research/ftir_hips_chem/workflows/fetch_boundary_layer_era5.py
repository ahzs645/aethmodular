"""Pull ERA5 boundary-layer height for every SPARTAN and IMPROVE site we hold filters for.

Site coordinates and filter date ranges come from the gallery's committed site
exports (gallery/app/public/data/{spartan,improve}_map.json), which already
reconcile SPARTAN lookups and the cleaned IMPROVE cache. Each site is fetched over
its filter date range, optionally clipped with --start/--end. Resumable: cached
site-years are skipped, so rerun the same command after an interruption or after
the free tier's daily budget resets.

Run from the repo root:
    uv run python research/ftir_hips_chem/workflows/fetch_boundary_layer_era5.py spartan
    uv run python research/ftir_hips_chem/workflows/fetch_boundary_layer_era5.py improve --start 2020-12-01 --end 2023-05-31
    uv run python research/ftir_hips_chem/workflows/fetch_boundary_layer_era5.py spartan --sites ETAD CLST CLTA

Budget (Open-Meteo free tier, 10,000 weighted calls/day): all SPARTAN ~4,300; IMPROVE over the
FTIR calibration-pool window ~10,200; IMPROVE 2003-2025 ~96,000 (use the Copernicus CDS for that).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
from boundary_layer import CallBudget, fetch_site  # noqa: E402

CACHE = AREA / "output/tables/boundary_layer_era5"
MAPS = {"spartan": ROOT / "gallery/app/public/data/spartan_map.json",
        "improve": ROOT / "gallery/app/public/data/improve_map.json"}


def sites(network: str) -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(MAPS[network].read_text())["sites"])
    frame["date_min"] = pd.to_datetime(frame.date_min, errors="coerce")
    frame["date_max"] = pd.to_datetime(frame.date_max, errors="coerce")
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("network", choices=sorted(MAPS))
    ap.add_argument("--start", help="clip every site's range to start on or after this date")
    ap.add_argument("--end", help="clip every site's range to end on or before this date")
    ap.add_argument("--sites", nargs="*", help="only these site codes")
    args = ap.parse_args()

    table = sites(args.network)
    if args.sites:
        table = table.loc[table.code.isin(args.sites)]
    cache = CACHE / args.network
    budget, t0, skipped = CallBudget(), time.time(), []
    rows = []
    for s in table.itertuples():
        start = max(s.date_min, pd.Timestamp(args.start)) if args.start and pd.notna(s.date_min) else s.date_min
        end = min(s.date_max, pd.Timestamp(args.end)) if args.end and pd.notna(s.date_max) else s.date_max
        if pd.isna(s.lat) or pd.isna(s.lon) or pd.isna(start) or pd.isna(end) or end < start:
            skipped.append(s.code)
            continue
        n = fetch_site(s.code, s.lat, s.lon, start, end, cache, budget)
        rows.append({"code": s.code, "name": s.name, "lat": s.lat, "lon": s.lon,
                     "start": start.date().isoformat(), "end": end.date().isoformat()})
        print(f"[{time.time() - t0:5.0f}s] {s.code:6s} {start.date()}..{end.date()} ({n} new site-years)", flush=True)
    # merge, so a run limited with --sites does not drop the sites fetched before it
    index = cache / "sites.csv"
    done = pd.DataFrame(rows)
    if index.exists():
        done = pd.concat([pd.read_csv(index).loc[lambda d: ~d.code.isin(done.code)], done])
    done.sort_values("code").to_csv(index, index=False)
    print(f"done: {len(rows)} sites in {cache.relative_to(ROOT)}; skipped (no coordinates or dates): {skipped}")


if __name__ == "__main__":
    main()
