"""Recover Fabs for SPARTAN filters the shipped HIPS CSV does not carry.

`get_hips_internals.ps1` pulls `hips.Results` straight from Networks_1_0. Two
facts established 2026-08-23 make this reconstruction sound:

  * the raw `Transmittance`/`Reflectance` are **bit-identical** to the shipped
    `T1`/`R1` (3,859 filters, max |diff| = 0), and
  * `tau = ln((Intercept + Slope*R1)/T1)` reproduces the shipped `tau` for
    100% of filters to within 1e-4.

So for a filter present in `hips.Results` but absent from
`SPARTAN_HIPS_Batch1-51.v2.csv`, we can recompute tau with its lot's blank line
and then `Fabs = 100 * tau * DepositArea / Volume`.

**These are reconstructed, not official.** The production pipeline may apply QC
this does not replicate (MDL, uncertainty, comment-based rejection), and
DepositArea is taken as the site median rather than per-filter. Treat them as
provisional until they appear in a shipped batch.

Usage:
    python reconstruct_hips_fabs.py ~/Downloads/hips/spartan_hips_raw_all.csv
    python reconstruct_hips_fabs.py <csv> --site ETAD
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
from pls_transfer import FTIRTransferPaths  # noqa: E402

PATHS = FTIRTransferPaths.defaults()
OUT = REPO / "research/ftir_ec_phase3/output/tables/hips"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", help="spartan_hips_raw_all.csv from get_hips_internals.ps1")
    ap.add_argument("--site", default=None, help="restrict to one SPARTAN site")
    args = ap.parse_args()

    r = pd.read_csv(args.raw, encoding="utf-8-sig")
    r.columns = [c.strip('"') for c in r.columns]
    sample = r[r["ResultTypeId"].eq(0)].drop_duplicates("ExternalFilterId")

    h = (pd.read_csv(PATHS.spartan_hips_primary, encoding="cp1252", low_memory=False)
         .drop_duplicates("FilterId"))
    shipped = set(h["FilterId"])

    # per-lot blank lines, as actually applied in the shipped file
    lines = {str(k): (g["Intercept"].median(), g["Slope"].median())
             for k, g in h.groupby(h["LotId"].astype(str)) if g["Intercept"].notna().any()}
    # DepositArea is constant per site in practice; fall back to the network median
    da_site = h.groupby("Site")["DepositArea"].median().to_dict()
    da_all = h["DepositArea"].median()

    new = sample[~sample["ExternalFilterId"].isin(shipped)].copy()
    if args.site:
        new = new[new["SiteCode"].eq(args.site)]
    print(f"filters in hips.Results but not in the shipped CSV: {len(new):,} "
          f"across {new['SiteCode'].nunique()} sites")

    # Volumes, keyed on ExternalFilterId, from every source we have:
    #   - ETAD_metadata.csv (Addis)
    #   - the staged SPARTAN pulls, DAVIS/SPARTAN FTIR pulls/<SITE>/<SITE>_filters.csv
    vol: dict[str, float] = {}
    meta_path = PATHS.etad_dir / "ETAD_metadata.csv"
    if meta_path.exists():
        m = pd.read_csv(meta_path)
        vol.update(m.set_index("ExternalFilterId")["SampleVolume_m3"].to_dict())
    staged = PATHS.maia_root / "DAVIS/SPARTAN FTIR pulls" if hasattr(PATHS, "maia_root") else None
    if staged is None or not staged.exists():
        from data_paths import maia_data_root
        staged = maia_data_root() / "DAVIS/SPARTAN FTIR pulls"
    n_staged = 0
    if staged.exists():
        for f in sorted(staged.glob("*/*_filters.csv")):
            try:
                d = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
                d.columns = [c.strip('\ufeff"') for c in d.columns]
                if {"ExternalFilterId", "SampleVolume_m3"} <= set(d.columns):
                    vol.update(d.set_index("ExternalFilterId")["SampleVolume_m3"].to_dict())
                    n_staged += 1
            except Exception:
                continue
    print(f"volume sources: ETAD_metadata + {n_staged} staged site pulls "
          f"({len(vol):,} filters with a volume)")

    rows = []
    for _, x in new.iterrows():
        try:
            lot = str(int(float(x["ExternalLotId"])))
        except (TypeError, ValueError):
            lot = str(x["ExternalLotId"])
        if lot not in lines:
            continue
        I, S = lines[lot]
        top = I + S * x["Reflectance"]
        if top <= 0 or x["Transmittance"] <= 0:
            continue
        tau = float(np.log(top / x["Transmittance"]))
        da = da_site.get(x["SiteCode"], da_all)
        V = vol.get(x["ExternalFilterId"], np.nan)
        fabs = 100.0 * tau * da / V if (V == V and V > 0) else np.nan
        rows.append({"Site": x["SiteCode"], "FilterId": x["ExternalFilterId"], "Lot": lot,
                     "AnalysisTimestamp": x["Timestamp"], "T1": x["Transmittance"],
                     "R1": x["Reflectance"], "tau": round(tau, 5),
                     "DepositArea": da, "Volume_m3": V,
                     "Fabs_reconstructed": round(fabs, 2) if fabs == fabs else np.nan,
                     # tau ~ 0 with no volume is the signature of a field/lab blank
                     "looks_like_blank": bool(tau < 0.05 and not (V == V and V > 0))})
    out = pd.DataFrame(rows)
    if out.empty:
        print("nothing reconstructable")
        return

    ok = out["Fabs_reconstructed"].notna()
    print(f"  reconstructed Fabs for {ok.sum()} filters; "
          f"{out['looks_like_blank'].sum()} look like blanks (tau ~ 0, no volume)")
    for site, g in out.groupby("Site"):
        got = g["Fabs_reconstructed"].notna().sum()
        if got:
            v = g.loc[g["Fabs_reconstructed"].notna(), "Fabs_reconstructed"]
            print(f"    {site}: {got:3d} with Fabs, median {v.median():6.2f} Mm-1 "
                  f"({v.min():.1f}-{v.max():.1f}), lots "
                  f"{sorted(g['Lot'].unique())}")
        else:
            print(f"    {site}: {len(g):3d} filters, no volume available -> tau only")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / ("reconstructed_fabs" + (f"_{args.site}" if args.site else "") + ".csv")
    out.to_csv(p, index=False)
    print(f"\nwrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
