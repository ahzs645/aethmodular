"""Build the biomass-only lot-251 EC calibration ENTIRELY LOCALLY.

X = the 916 biomass spectra downloaded from the Shiny tool (data/spectra_lot251_biomass_2021-2025.csv).
Y = measured EC loading (µg) from the repo's IMPROVE reference, joined on Site + SampleDate
    (96.5% coverage — the tool itself only matched 906/916).

This removes the dependency on the flaky Shiny server: it reproduces the tool's calibration
(component sweep + Mona's outlier rules) as scripted, reproducible code, and exports coefficients
in the tool's shape (wavenumber, coefficient + intercept) for application to the Addis spectra.

Run:  python _build_local_ec_calibration.py
"""
import time
from pathlib import Path

import pandas as pd

import ftir_pls_calibration as f

HERE = Path(__file__).resolve().parent
SPECTRA = HERE / "data/spectra_lot251_biomass_2021-2025.csv"
IMPROVE_REF = HERE / "../../research/ftir_hips_chem/output/improve_high_fabs_comparison/improve_valid_cleaned.csv"
OUT_COEFFS = HERE / "data/calib_EC_biomass_lot251_coeffs.csv"

t0 = time.time()

# --- X: spectra ---
spec, wn = f.load_spectra_csv(SPECTRA)
spec["date"] = pd.to_datetime(spec["SampleDate"]).dt.tz_localize(None).dt.normalize()

# --- Y: measured EC loading (µg), joined on Site + date ---
ref = pd.read_csv(IMPROVE_REF, low_memory=False)
ref["date"] = pd.to_datetime(ref["Date"]).dt.normalize()
refd = (ref.sort_values("POC")
           .drop_duplicates(["SiteCode", "date"])[["SiteCode", "date", "EC_loading_ug", "ECf_Val", "volume_m3"]])
m = spec.merge(refd, left_on=["Site", "date"], right_on=["SiteCode", "date"], how="left")
n_matched = int(m["EC_loading_ug"].notna().sum())
print(f"spectra: {len(spec)} | EC_loading matched: {n_matched} ({n_matched/len(spec):.1%})")

df = m.dropna(subset=["EC_loading_ug"]).copy()

# --- build calibration (coarse component grid for speed; 2 CV sweeps total) ---
res = f.build_calibration(
    df, wn, "EC_loading_ug", species="EC",
    ncomp_range=range(6, 41, 2),      # coarse grid: 6,8,...,40
    residual_thresholds=(400, 300, 200),
    cv=5,
)

print("\n=== LOCAL EC biomass calibration (lot 251, IMPROVE biomass samples) ===")
print(f"  n kept            : {res.stats['n']}")
print(f"  dropped negatives : {len(res.dropped['measured_negative'])}")
print(f"  dropped residual  : {len(res.dropped['residual'])}")
print(f"  n_components      : {res.n_components}      (tool picked 34)")
print(f"  R^2               : {res.stats['r2']:.4f}")
print(f"  RMSE (in-sample)  : {res.stats['rmse']:.2f} µg")
print(f"  bias %            : {res.stats['bias_pct']:+.2f}")
rb = res.stats["rmsep_by_ncomp"]
near = {k: round(v, 2) for k, v in rb.items() if 28 <= k <= 40}
print(f"  CV-RMSEP (k=28..40): {near}        (tool min RMSEP ~18.97 @ 34)")
print(f"  trimming history  : {res.history}")

res.to_csv(OUT_COEFFS)
print(f"\nwrote {OUT_COEFFS.relative_to(HERE)}  ({len(res.coefficients)} wavenumbers + intercept)")
print(f"elapsed: {time.time()-t0:.1f}s")
