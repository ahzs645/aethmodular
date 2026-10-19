"""Build a read-only copy of the gallery with only the views a reader needs.

Run from the repo root:
    python3 gallery/share/build_share.py                 # Meeting + Map (Ann and Satoshi, 9 Oct 2026)
    python3 gallery/share/build_share.py --tabs meeting,map,blanks

The full gallery ships ~126 MB of data because every tab's exports sit in
public/data. A share build sets VITE_SHARE_TABS so the app offers only the
listed tabs, uses relative paths (base ./) so it works from any host or folder,
and keeps only the data files those tabs fetch. The output in gallery/share/dist/
is static: open it from any web server (`python3 -m http.server -d gallery/share/dist`),
not from file://, because the app fetches its data.

Nothing here publishes anything; the bundle stays local until you put it somewhere.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "gallery/app"
OUT = ROOT / "gallery/share/dist"

# Loaded on every page: filters/meta/census/pmf (calibration*.json are optional and
# only feed the Calibration tab), the sample drawer's SPARTAN reference, and the
# per-filter spectra the drawer and Meeting spectra panels draw.
COMMON = ["filters.json", "meta.json", "census.json", "pmf.json", "spartan_network/index.json",
          "similarity/similarity.json", "similarity/etad_ids.json",
          "similarity/traces_AIRSpec.bin", "similarity/traces_VIBES-full-r70.bin",
          "similarity/traces_VIBES-full-r70.json"]
BY_TAB = {
    "meeting": ["meeting-r70/followup_20260917.json", "meeting-r70/spec_curve_20260917.json",
                "satoshi/followup_20261002.json", "satoshi/oct1_remaining_20261007.json",
                "vibes_fullrange_results.json"],
    "map": ["world.geojson", "spartan_map.json", "improve_map.json"],
    "blanks": ["blanks/blanks_20261007.json"],
    "calibration": ["calibration.json", "calibration_runs.json", "spec_curve.json"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabs", default="meeting,map", help="comma-separated tab keys; the first is the landing tab")
    args = ap.parse_args()
    tabs = [t.strip() for t in args.tabs.split(",") if t.strip()]
    unknown = [t for t in tabs if t not in BY_TAB]
    if unknown:
        raise SystemExit(f"no data list for tab(s) {unknown}; add them to BY_TAB first")

    keep = COMMON + [f for t in tabs for f in BY_TAB[t]]
    # Site picks and drawer spectra fetch content-addressed shards lazily. Keep
    # them in shared copies too; they add no initial page transfer.
    keep += [p.relative_to(APP / 'public/data').as_posix()
             for p in (APP / 'public/data/spartan_network').rglob('*') if p.is_file()]
    missing = [f for f in keep if not (APP / "public/data" / f).is_file()]
    if missing:
        raise SystemExit(f"missing exports (run the gallery/data exporters first): {missing}")

    env = {**os.environ, "VITE_SHARE_TABS": ",".join(tabs), "VITE_HIDE_EXPLAIN": "1"}   # no Explain toggle for outside readers
    subprocess.run(["npx", "vite", "build", "--base", "./", "--outDir", str(OUT), "--emptyOutDir"],
                   cwd=APP, env=env, check=True)

    # vite copies all of public/; drop every data file the shared tabs never fetch
    data = OUT / "data"
    kept = {data / f for f in keep}
    for path in sorted(data.rglob("*"), reverse=True):
        if path.is_file() and path not in kept:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"share build for tabs {tabs}: {len(kept)} data files, {size / 1e6:.1f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
