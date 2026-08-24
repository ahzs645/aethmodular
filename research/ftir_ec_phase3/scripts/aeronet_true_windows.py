"""Match AERONET retrievals to SPARTAN filters on their TRUE sampling windows.

Companion to `run_aeronet_spartan.py`, which pairs AERONET's *daily* inversion
product with a filter's start date. This script tests whether that shortcut
matters, by using per-retrieval AERONET timestamps and each filter's actual
`[SamplingStartDate, SamplingEndDate)` interval from the site metadata.

Answer (Addis, 2026-08-23): it does not. True-window and same-UTC-day matching
agree to 0.0%, because every almucantar retrieval is daytime and therefore falls
on the same UTC calendar day as the local filter day. See
`AERONET_SPARTAN_2026-08-22.md` for the write-up.

What the script *does* establish is the diurnal-coverage limit: the fraction of
the 24-hour filter period the photometer can actually see.

Usage:
    python aeronet_true_windows.py                       # Addis / ETAD
    python aeronet_true_windows.py --tz 3 --min-hours 20
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
from aeronet import aeronet_dir            # noqa: E402
from pls_transfer import FTIRTransferPaths  # noqa: E402

PATHS = FTIRTransferPaths.defaults()
OUT = REPO / "research/ftir_ec_phase3/output/tables/aeronet"

AAOD = "Absorption_AOD[675nm]"           # nearest inversion channel to HIPS 632.8 nm
AAE = "Absorption_Angstrom_Exponent_440-870nm"


def load_retrievals(lev15: Path, tz_hours: float) -> pd.DataFrame:
    """Per-retrieval AERONET inversions, timestamped in site-local time."""
    inv = pd.read_csv(lev15, skiprows=6).replace(-999.0, np.nan)
    dc = next(c for c in inv.columns if "Date" in c)
    tc = next(c for c in inv.columns if "Time" in c and "Day" not in c)
    inv["ts_utc"] = pd.to_datetime(inv[dc] + " " + inv[tc],
                                   format="%d:%m:%Y %H:%M:%S", errors="coerce")
    inv = inv.dropna(subset=["ts_utc"])[["ts_utc", AAOD, AAE]].dropna(subset=[AAOD])
    inv["ts_loc"] = inv["ts_utc"] + pd.Timedelta(hours=tz_hours)
    return inv


def load_windows(meta: Path, site: str, lo: float, hi: float) -> pd.DataFrame:
    """Filters whose metadata gives a clean ~24 h window and that have a HIPS Fabs."""
    m = pd.read_csv(meta)
    m["s"] = pd.to_datetime(m["SamplingStartDate"], errors="coerce")
    m["e"] = pd.to_datetime(m["SamplingEndDate"], errors="coerce")
    m = m.dropna(subset=["s", "e"])
    m["dur_h"] = (m["e"] - m["s"]).dt.total_seconds() / 3600
    clean = m[m["dur_h"].between(lo, hi)].copy()
    print(f"filter windows: {len(m)} total, {len(clean)} clean "
          f"({lo}-{hi} h) | duration median {m['dur_h'].median():.1f} h")

    h = pd.read_csv(PATHS.spartan_hips_primary, encoding="cp1252",
                    usecols=["Site", "FilterId", "Fabs"])
    h = (h[h["Site"].eq(site)].dropna(subset=["Fabs"])
         .drop_duplicates("FilterId").rename(columns={"FilterId": "ExternalFilterId"}))
    out = clean.merge(h, on="ExternalFilterId", how="inner")
    print(f"  ... with a HIPS Fabs: {len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="ETAD")
    ap.add_argument("--tz", type=float, default=3.0, help="site UTC offset, hours")
    ap.add_argument("--min-hours", type=float, default=20.0)
    ap.add_argument("--max-hours", type=float, default=28.0)
    args = ap.parse_args()

    d = aeronet_dir() / "Jacros/20220101_20251231_AAU_Jackros_ET Daily"
    inv = load_retrievals(d / "20220101_20251231_AAU_Jackros_ET.ALL_ALM_lev15",
                          args.tz)
    print(f"AERONET per-retrieval with AAOD675: {len(inv)}")

    m = load_windows(PATHS.etad_dir / "ETAD_metadata.csv", args.site,
                     args.min_hours, args.max_hours)

    # --- TRUE WINDOW: retrieval local time inside [start, end) ---
    rows = []
    for _, f in m.iterrows():
        w = inv[(inv["ts_loc"] >= f["s"]) & (inv["ts_loc"] < f["e"])]
        if len(w):
            rows.append({"FilterId": f["ExternalFilterId"], "Fabs": f["Fabs"],
                         "n_ret": len(w), "AAOD": w[AAOD].mean(),
                         "AAE": w[AAE].mean(),
                         "hours": ",".join(map(str, sorted(w["ts_loc"].dt.hour.unique())))})
    tw = pd.DataFrame(rows)
    tw["H"] = tw["AAOD"] / (tw["Fabs"] * 1e-6)
    print(f"\nTRUE-WINDOW matches: {len(tw)} filters | retrievals each: median "
          f"{tw['n_ret'].median():.0f}, range {tw['n_ret'].min()}-{tw['n_ret'].max()}")
    print(f"  H median {tw['H'].median():.0f} m "
          f"(IQR {tw['H'].quantile(.25):.0f}-{tw['H'].quantile(.75):.0f})"
          f" | r(AAOD,Fabs) {np.corrcoef(tw['AAOD'], tw['Fabs'])[0, 1]:+.3f}"
          f" | AAE median {tw['AAE'].median():.2f}")

    # --- NAIVE same-UTC-day match, as used in the published table ---
    inv["date_utc"] = inv["ts_utc"].dt.normalize()
    daily = inv.groupby("date_utc")[[AAOD, AAE]].mean().reset_index()
    m["date"] = m["s"].dt.normalize()
    nv = m.merge(daily, left_on="date", right_on="date_utc", how="inner")
    nv["H_naive"] = nv[AAOD] / (nv["Fabs"] * 1e-6)
    print(f"\nNAIVE same-UTC-day match: n={len(nv)}, H median {nv['H_naive'].median():.0f} m")

    both = tw.merge(nv[["ExternalFilterId", "H_naive"]]
                    .rename(columns={"ExternalFilterId": "FilterId"}), on="FilterId")
    dif = (both["H"] - both["H_naive"]).abs()
    print(f"  filters in both: {len(both)} | median |H_true - H_naive| = {dif.median():.0f} m"
          f" | median rel diff {(dif / both['H_naive']).median() * 100:.1f}%")

    # --- diurnal coverage: how much of the filter day AERONET can see ---
    allh = sorted({int(x) for s in tw["hours"] for x in s.split(",")})
    print(f"\nDiurnal coverage: retrievals fall in {len(allh)}/24 local hours: {allh}")
    print(f"  => each 24 h filter is optically sampled during ~{len(allh)} distinct "
          "hours; the rest (night, midday) is unobserved.")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{args.site}_true_window_match.csv"
    both.to_csv(p, index=False)
    print(f"\nwrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
