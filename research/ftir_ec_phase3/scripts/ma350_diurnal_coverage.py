"""Close the AERONET diurnal-coverage confound with raw 1-min MA350 data.

AERONET almucantar inversions need solar zenith angle >= 50 deg, so they sample
only part of the day — and *which* part depends on latitude. A 24-h filter is
therefore compared against a column measured during a biased subset of hours.

This script quantifies the bias directly: for each site it computes

    ratio = mean(Red BCc over all 24 h) / mean(Red BCc over AERONET hours)

from the raw minute-resolution MA350 files (the `processed_sites/*_9am_resampled`
pickles are daily aggregates and cannot answer this). Red BCc is the 625 nm
channel — effectively the HIPS wavelength (632.8 nm). If the column tracks the
surface, H should be multiplied by this ratio.

Result (2026-08-23): Addis 0.92-1.05, Pasadena 1.07-1.24. The correction is
negligible at Addis and *larger* at Pasadena, so applying it widens the H gap
rather than closing it. See `AERONET_SPARTAN_2026-08-22.md`.

Usage:
    python ma350_diurnal_coverage.py                 # both sites
    python ma350_diurnal_coverage.py --site Addis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
from data_paths import aethalometry_dir  # noqa: E402

OUT = REPO / "research/ftir_ec_phase3/output/tables/aeronet"

# AERONET hours: Addis is the *observed* Jackros retrieval hours; Pasadena is
# the SZA 50-80 deg window from solar geometry (no per-retrieval file on Drive).
SITES = {
    "Addis": dict(file="Jacros_MA350_1-min_2022-2024_Cleaned.csv",
                  hours=[7, 8, 9, 10, 14, 15, 16, 17], H=190.0),
    "Pasadena": dict(file="Pasadena_MA350_1-min_2023-2024_Cleaned.csv",
                     hours=list(range(6, 19)), H=544.0),
}
COLS = ["Date local (yyyy/MM/dd)", "Time local (hh:mm:ss)", "Red BCc"]


def load(path: Path) -> pd.DataFrame:
    """Minute records with a local hour. Time local is 12-hour with AM/PM."""
    parts = []
    for ch in pd.read_csv(path, usecols=COLS, chunksize=500_000):
        ch.columns = ["date", "time", "bc"]
        ch["hr"] = pd.to_datetime(ch["time"].str.strip(), format="%I:%M:%S %p",
                                  errors="coerce").dt.hour
        parts.append(ch.loc[ch["hr"].notna() & ch["bc"].notna(), ["date", "hr", "bc"]])
    x = pd.concat(parts, ignore_index=True)
    x["hr"] = x["hr"].astype(int)
    return x


def ratios(x: pd.DataFrame, hours: list[int]) -> dict:
    """Per-day 24h/AERONET-hours ratio, on near-complete days only."""
    out = {}
    sub = x[x["hr"].isin(hours)]
    n_full, n_el = x.groupby("date")["bc"].size(), sub.groupby("date")["bc"].size()
    ok = (n_full > 1000) & (n_el > 300)
    for stat in ("mean", "median"):
        full = getattr(x.groupby("date")["bc"], stat)()
        el = getattr(sub.groupby("date")["bc"], stat)()
        r = (full[ok] / el[ok]).replace([np.inf, -np.inf], np.nan).dropna()
        out[stat] = (r.median(), r.quantile(.25), r.quantile(.75))
    out["days"] = int(ok.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", choices=sorted(SITES), default=None)
    args = ap.parse_args()
    raw = aethalometry_dir() / "Raw"

    rows = []
    for name in ([args.site] if args.site else list(SITES)):
        cfg = SITES[name]
        x = load(raw / cfg["file"])
        print(f"\n=== {name} === {len(x):,} minutes, {x['date'].nunique()} days, "
              f"mean Red BCc {x['bc'].mean():.0f} ng/m3 "
              f"({(x['bc'] < 0).mean() * 100:.1f}% negative)")
        hm = x.groupby("hr")["bc"].mean()
        print("  hourly mean: " + " ".join(f"{h:02d}={hm.get(h, np.nan):.0f}"
                                           for h in range(24)))
        r = ratios(x, cfg["hours"])
        print(f"  AERONET hours {cfg['hours']}  ({r['days']} near-complete days)")
        for stat in ("mean", "median"):
            m, lo, hi = r[stat]
            print(f"    {stat:6s} ratio 24h / AERONET-hours = {m:.3f} (IQR {lo:.3f}-{hi:.3f})"
                  f"   -> H {cfg['H']:.0f} -> {cfg['H'] * m:.0f} m")
        rows.append({"site": name, "days": r["days"],
                     "hours": ",".join(map(str, cfg["hours"])),
                     "ratio_mean": round(r["mean"][0], 3),
                     "ratio_median": round(r["median"][0], 3),
                     "H_uncorrected": cfg["H"],
                     "H_corrected_mean": round(cfg["H"] * r["mean"][0]),
                     "H_corrected_median": round(cfg["H"] * r["median"][0])})

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "diurnal_coverage_correction.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"\nwrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
