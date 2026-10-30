"""Per-site quantiles of IMPROVE measurements and of their per-filter ratios, for ranking any site against the network.

Run where the Davis Data Drive folder is available (it is not in the repo):
    uv run python gallery/data/export_improve_site_quantiles.py
    uv run python gallery/data/export_improve_site_quantiles.py --xlsx <path to IMPROVE_chem-mass-fabs_2026-04-22.xlsx>

Reads "Davis Data"/Improve/IMPROVE_chem-mass-fabs_2026-04-22.xlsx (the FED IMPAER export renamed from
ahzs645_20260422_122542_N0100.xlsx; resolved through data_paths.maia_data_root(), so AETHMODULAR_MAIA_DATA_ROOT
works), parsed exactly as research/spartan/follow_up_mass_and_ratio_plots.ipynb builds the
improve_chemistry_clean_cache.pkl that research/ftir_ec_phase3/scripts/run_ftir_51.py §11 reads: FED's -999
sentinel to missing, duplicate Dataset/SiteCode/POC/Date/AuxID records dropped, collocated samplers kept.

The gallery's Ranking tab (charts/ranking/NetworkRankPlacement.tsx) ranks the IMPROVE sites by the median of a
field or of a per-filter ratio and places the sites in view at the rank their own median takes. ~626,000 filters
cannot be shipped, but every statistic it needs is a per-site quantile, so this writes those:

  populations  paired: filters with OC, EC and fAbs all > 0 (run_ftir_51 §11, 11a/11b: 378,322 filters, 214 sites)
               own:    each quantity's own filters: a field where it is reported, a ratio where both parts are > 0
  quantities   each field, and every ordered ratio of two fields, as per-filter ratios (never a ratio of quantiles)
  per site     n, p25, median, p75 (pandas' linear interpolation, the same as d3.quantileSorted)

HIPS BC = fAbs / MAC is not written: its quantiles, and those of any ratio with it, are fAbs's scaled by the
reader's MAC in the browser (a positive constant keeps every quantile in place).

The IMPROVE data carry a data-use condition (for journal publications, inform the NPS contact named on the
export's Overview sheet; <Davis>/Improve/MANIFEST.md); the file says so in `data_use`.

Writes gallery/app/public/data/improve_site_quantiles.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))

OUT = ROOT / "gallery/app/public/data/improve_site_quantiles.json"
DRIVE_REL = "Improve/IMPROVE_chem-mass-fabs_2026-04-22.xlsx"
# the gallery's field names (improve_samples/, export_improve_samples.py) for the export's columns
FIELDS = {
    "EC (TOR)": ("ECf_Val", "ug/m3"),
    "OC (TOR)": ("OCf_Val", "ug/m3"),
    "HIPS Fabs": ("fAbs_Val", "Mm-1"),
    "PM2.5 mass": ("MF_Val", "ug/m3"),
    "Iron": ("FEf_Val", "ug/m3"),
    "Soil": ("SOILf_Val", "ug/m3"),
}
NUMERIC = ["POC", "AuxID", "ECf_Val", "OCf_Val", "fAbs_Val", "FlowRate_Val", "FEf_Val", "MF_Val", "SampDur_Val", "SOILf_Val"]
MIN_N = 10            # sites with fewer filters for a quantity are left out (the chart's default floor is 50)
DIGITS = 5            # significant figures: enough that no rank changes against the unrounded medians


def r(v: float) -> float | None:
    v = float(v)
    return float(f"{v:.{DIGITS}g}") if np.isfinite(v) else None


def resolve_xlsx(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    from data_paths import maia_data_root
    return maia_data_root() / DRIVE_REL


def load(xlsx: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """The FED export as follow_up_mass_and_ratio_plots.ipynb caches it (improve_chemistry_clean_cache.pkl)."""
    raw = pd.read_excel(xlsx, sheet_name="Data")
    for c in NUMERIC:
        if c in raw:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
            raw.loc[raw[c] <= -999, c] = np.nan            # FED missing-value sentinel
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.drop_duplicates(subset=["Dataset", "SiteCode", "POC", "Date", "AuxID"])
    sites = pd.read_excel(xlsx, sheet_name="Sites")
    names = (sites.drop_duplicates("Code").set_index("Code")["Site"].astype(str).to_dict()
             if {"Code", "Site"} <= set(sites.columns) else {})
    return raw, names


def site_quantiles(site: pd.Series, values: pd.Series, codes: list[str]) -> dict | None:
    """n, p25, median, p75 of `values` per site (sites with at least MIN_N), as parallel arrays over `codes` indices."""
    d = pd.DataFrame({"site": site, "v": values}).replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return None
    g = d.groupby("site")["v"]
    n = g.size()
    q = g.quantile([0.25, 0.5, 0.75]).unstack()
    keep = n[n >= MIN_N].index
    if not len(keep):
        return None
    index = {c: i for i, c in enumerate(codes)}
    return {"site": [index[s] for s in keep], "n": [int(n[s]) for s in keep],
            "p25": [r(q.loc[s, 0.25]) for s in keep], "p50": [r(q.loc[s, 0.5]) for s in keep],
            "p75": [r(q.loc[s, 0.75]) for s in keep]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xlsx", help="the FED export, when the Davis Data folder does not resolve")
    args = ap.parse_args()
    xlsx = resolve_xlsx(args.xlsx)
    if not xlsx.exists():
        raise SystemExit(f"IMPROVE export not found: Davis Data/{DRIVE_REL} (set AETHMODULAR_MAIA_DATA_ROOT or pass --xlsx)")
    with xlsx.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    imp, names = load(xlsx)
    cols = {f: imp[c] for f, (c, _) in FIELDS.items()}
    paired = (cols["OC (TOR)"] > 0) & (cols["EC (TOR)"] > 0) & (cols["HIPS Fabs"] > 0)
    codes = sorted(imp["SiteCode"].dropna().astype(str).unique())

    quantities: dict[str, dict] = {"paired": {}, "own": {}}
    for pop, mask in (("paired", paired), ("own", pd.Series(True, index=imp.index))):
        site = imp.loc[mask, "SiteCode"].astype(str)
        for f, v in cols.items():
            if (q := site_quantiles(site, v[mask], codes)):
                quantities[pop][f] = q
        for a, b in permutations(FIELDS, 2):
            num, den = cols[a][mask], cols[b][mask]
            ratio = (num / den).where((num > 0) & (den > 0))   # per filter, both parts positive
            if (q := site_quantiles(site, ratio, codes)):
                quantities[pop][f"{a} / {b}"] = q

    used = sorted({codes[i] for pop in quantities.values() for q in pop.values() for i in q["site"]})
    # re-index onto the sites that appear, so the file carries no site it never ranks
    remap = {codes.index(c): i for i, c in enumerate(used)}
    for pop in quantities.values():
        for q in pop.values():
            q["site"] = [remap[i] for i in q["site"]]

    dates = imp["Date"].dropna()
    out = {
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"Davis Data/{DRIVE_REL} (FED IMPROVE Aerosol, IMPAER; sha256 {digest[:16]}); parsed as "
                  "research/spartan/follow_up_mass_and_ratio_plots.ipynb builds improve_chemistry_clean_cache.pkl",
        "note": (f"{len(imp):,} IMPROVE filter records at {len(codes)} sites, {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}, "
                 "collocated samplers included. EC and OC are thermal-optical (TOR); HIPS Fabs is IMPROVE's fAbs (hybrid "
                 "integrating plate and sphere). Ratios are per filter, then summarised per site; sites with fewer than "
                 f"{MIN_N} filters for a quantity are left out of it."),
        "data_use": "IMPROVE data-use condition: for journal publications, inform the National Park Service contact named "
                    "on the export's Overview sheet (Davis Data/Improve/MANIFEST.md) of the planned publication, and cite "
                    "Malm et al. (1994) with the IMPROVE acknowledgement.",
        "min_n": MIN_N,
        "populations": {
            "paired": {"label": "Filters with OC, EC and fAbs > 0",
                       "rule": "OC (TOR), EC (TOR) and HIPS Fabs all > 0 on the filter (run_ftir_51 §11, figures 11a/11b)",
                       "n_filters": int(paired.sum()), "n_sites": int(imp.loc[paired, "SiteCode"].nunique())},
            "own": {"label": "Every filter with the quantity",
                    "rule": "a field wherever it is reported; a ratio wherever both parts are > 0",
                    "n_filters": len(imp), "n_sites": len(codes)},
        },
        "fields": {f: {"column": c.removesuffix("_Val"), "unit": u} for f, (c, u) in FIELDS.items()},
        "sites": used,
        "site_names": {c: names.get(c, c) for c in used},
        "quantities": quantities,
    }
    OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False) + "\n")
    p = out["populations"]["paired"]
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB): paired {p['n_filters']:,} filters at "
          f"{p['n_sites']} sites; {sum(len(v) for v in quantities.values())} quantities")


if __name__ == "__main__":
    main()
