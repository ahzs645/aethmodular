"""AERONET vs SPARTAN filter absorption — the column/surface consistency test.

Question this answers
---------------------
Phase 3 finds an Addis-specific additive offset on the HIPS axis (~21.5 Mm^-1
absorption-equivalent) that Bishoftu does NOT share (ETBI_FIRST_LOOK_2026-08-22).
Two live readings: (a) the Addis HIPS Fabs is biased high (measurement artifact),
or (b) Addis really carries extra non-EC absorption.

AERONET is an INDEPENDENT optical measurement of the same air column, so it can
separate these:

  * Effective absorption scale height  H = AAOD / (Fabs * 1e-6)  [m]
    A surface Fabs that is too high makes H implausibly small. H is not a
    physical BLH (filters run 24 h, AERONET only daytime clear-sky, and cities
    have near-surface sources), so it is only meaningful COMPARED ACROSS SITES
    measured the same way. Addis being an outlier vs Beijing/Delhi/Pasadena
    points at (a); Addis looking normal points at (b).
  * Absorption Angstrom Exponent (440-870 nm) attributes the absorber:
    ~1 = BC-dominated, >1.5 suggests BrC/dust, >2 dust-like.

Wavelength note: HIPS is 632.8 nm (He-Ne). The nearest AERONET inversion channel
is 675 nm -- used throughout here. An earlier repo analysis assumed 405 nm, from
before the HIPS wavelength question was resolved; those numbers are superseded.

Data
----
Addis reads the committed Drive export (AAU_Jackros_ET, 2022-2025, Level 1.5
almucantar inversion). Other sites pull from the AERONET web service via monetio
and cache to output/tables/aeronet/. SPARTAN Fabs comes from the HIPS batch file.

Run:  python run_aeronet_spartan.py            # Addis only (no network)
      python run_aeronet_spartan.py --pull     # add Delhi/Beijing/Pasadena
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
from aeronet import aeronet_dir  # noqa: E402
from pls_transfer import FTIRTransferPaths  # noqa: E402

PATHS = FTIRTransferPaths.defaults()
OUT = REPO / "research/ftir_ec_phase3/output/tables/aeronet"

# SPARTAN site -> nearest AERONET site with inversion products
SITES = {
    "ETAD": ("AAU_Jackros_ET", "Addis Ababa"),
    "INDH": ("Gual_Pahari", "Delhi"),   # New_Delhi has no ALM15 inversion series
    "CHTS": ("Beijing", "Beijing"),
    "USPA": ("CalTech", "Pasadena"),
}
# fallback AERONET sites per SPARTAN city (tried in order if the first has
# no almucantar inversion series)
ALT_SITES = {
    "INDH": ["New_Delhi_IMD", "Amity_Univ_Gurgaon", "New_Delhi"],
    "CHTS": ["Beijing-CAMS", "XiangHe", "PKU_PEK"],
    "USPA": ["MISR-JPL", "Mount_Wilson", "El_Segundo"],
}
AAOD = "Absorption_AOD[675nm]"          # nearest channel to HIPS 632.8 nm
AAE = "Absorption_Angstrom_Exponent_440-870nm"
SSA = "Single_Scattering_Albedo[675nm]"


def addis_inversion() -> pd.DataFrame:
    """Level 1.5 almucantar daily inversion from the committed Drive export."""
    d = aeronet_dir() / "Jacros/20220101_20251231_AAU_Jackros_ET Daily"
    f = d / "20220101_20251231_AAU_Jackros_ET.ALL_ALM_lev15_daily"
    df = pd.read_csv(f, skiprows=6).replace(-999.0, np.nan)
    dcol = next(c for c in df.columns if "Date" in c)
    df["date"] = pd.to_datetime(df[dcol], format="%d:%m:%Y", errors="coerce")
    return df.dropna(subset=["date"])


def pull_inversion(siteid: str, start="2015-01-01", end="2026-06-30") -> pd.DataFrame:
    """AERONET almucantar inversion via monetio, cached to CSV."""
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"{siteid}_inv_daily.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        df["date"] = pd.to_datetime(df["date"])
        return df
    from monetio import aeronet as an

    # inversion products: product is the family ('ALL'), the level/geometry
    # lives in inv_type ('ALM15' = almucantar, Level 1.5 — matches the
    # committed Addis export)
    dates = pd.date_range(start, end, freq="D")
    df = an.add_data(dates, product="ALL", inv_type="ALM15",
                     siteid=siteid, daily=True)
    df = df.reset_index().rename(columns={"time": "date"}).replace(-999.0, np.nan)
    df.to_csv(cache, index=False)
    return df


def spartan_fabs(site: str) -> pd.DataFrame:
    h = pd.read_csv(PATHS.spartan_hips_primary, encoding="cp1252",
                    usecols=["Site", "FilterId", "SampleDate", "Fabs", "LotId"])
    e = h[h["Site"].eq(site)].dropna(subset=["Fabs"]).copy()
    e["date"] = pd.to_datetime(e["SampleDate"], errors="coerce").dt.normalize()
    return e.dropna(subset=["date"])


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """monetio lowercases/renames columns; map back to the canonical names.

    Skips the per-day sample-count columns (``N[...]``), which otherwise match
    the same patterns and collide into duplicate names.
    """
    ren = {}
    for c in df.columns:
        # monetio lowercases everything, so the count columns arrive as 'n[...]'
        if c.lower().startswith("n[") or c in (AAOD, AAE, SSA):
            continue
        lc = c.lower().replace(" ", "").replace("_", "")
        if "absorptionaod" in lc and "675" in lc:
            ren[c] = AAOD
        elif "absorptionangstrom" in lc and "440" in lc:
            ren[c] = AAE
        elif "singlescatteringalbedo" in lc and "675" in lc:
            ren[c] = SSA
    out = df.rename(columns=ren)
    return out.loc[:, ~out.columns.duplicated()]


def analyze(site: str, city: str, inv: pd.DataFrame) -> dict | None:
    inv = normalize(inv)
    if AAOD not in inv.columns:
        print(f"  {site}: no {AAOD} column in inversion data — skipped")
        return None
    # monetio stamps daily rows at 12:00; SPARTAN dates are midnight. Normalize
    # both to calendar days or every merge silently returns zero rows.
    inv = inv.copy()
    inv["date"] = pd.to_datetime(inv["date"]).dt.normalize()
    fab = spartan_fabs(site)
    m = fab.merge(inv[["date"] + [c for c in (AAOD, AAE, SSA) if c in inv.columns]],
                  on="date", how="inner")
    m = m[m[AAOD].notna() & (m[AAOD] > 0)]
    if len(m) < 5:
        print(f"  {site}: only {len(m)} same-day matches — skipped")
        return None
    # effective absorption scale height, metres
    m["H_m"] = m[AAOD] / (m["Fabs"] * 1e-6)
    r = float(np.corrcoef(m[AAOD], m["Fabs"])[0, 1])
    row = {
        "site": site, "city": city, "n_matched": len(m),
        "Fabs_median_Mm": round(float(m["Fabs"].median()), 2),
        "AAOD675_median": round(float(m[AAOD].median()), 4),
        "H_median_m": round(float(m["H_m"].median())),
        "H_IQR_m": f"{m['H_m'].quantile(.25):.0f}-{m['H_m'].quantile(.75):.0f}",
        "r_AAOD_Fabs": round(r, 3),
    }
    if AAE in m.columns and m[AAE].notna().any():
        row["AAE_median"] = round(float(m[AAE].median()), 2)
        row["AAE_frac_gt1.5"] = round(float((m[AAE] > 1.5).mean()), 3)
    if SSA in m.columns and m[SSA].notna().any():
        row["SSA675_median"] = round(float(m[SSA].median()), 3)
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / f"{site}_matched_daily.csv", index=False)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pull", action="store_true",
                    help="fetch Delhi/Beijing/Pasadena from the AERONET service")
    args = ap.parse_args()

    rows = []
    r = analyze("ETAD", "Addis Ababa", addis_inversion())
    if r:
        rows.append(r)
    if args.pull:
        for site, (aid, city) in SITES.items():
            if site == "ETAD":
                continue
            print(f"pulling {aid} ({city}) ...")
            # try each candidate AERONET site until one actually overlaps the
            # filter record — nearby sites often cover different years
            r = None
            for cand in [aid] + ALT_SITES.get(site, []):
                try:
                    inv = pull_inversion(cand)
                except Exception as exc:                 # noqa: BLE001
                    print(f"  {cand}: pull failed — {type(exc).__name__}: {exc}")
                    continue
                r = analyze(site, city, inv)
                if r:
                    r["aeronet_site"] = cand
                    rows.append(r)
                    if cand != aid:
                        print(f"  (used fallback AERONET site {cand})")
                    break

    if not rows:
        raise SystemExit("no sites analyzed")
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "site_comparison.csv", index=False)
    print("\n=== AERONET vs SPARTAN filter absorption (HIPS 633 nm vs AAOD 675 nm) ===")
    print(tab.to_string(index=False))
    print(f"\nwrote {OUT}/site_comparison.csv + per-site matched dailies")
    if len(tab) > 1:
        h = tab.set_index("site")["H_median_m"]
        print(f"\nEffective scale height H: Addis {h.get('ETAD')} m vs others "
              f"{{{', '.join(f'{s} {v}' for s, v in h.drop('ETAD', errors='ignore').items())}}}")
        print("Addis much SMALLER than the others => surface Fabs anomalously high "
              "(measurement-artifact reading). Comparable => Addis Fabs is normal "
              "and the offset is aerosol, not instrument.")


if __name__ == "__main__":
    main()
