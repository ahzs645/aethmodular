# Phase 3 results summary (July 2026)

Follow-ups from the July 2026 meeting with Ann and Satoshi. All Addis metrics below are on the
same fixed 190-filter cohort as `ftir_10`, HIPS EC-equivalent (Fabs/MAC) on the x-axis,
MAC = 10 headline. Nothing here uses Addis data for cohort selection or fitting unless
explicitly stated.

## ftir_11 — Ann's lowest-OC/EC cohort: best locked result so far

Cohort: the 800 lowest TOR-OC/EC IMPROVE filters (OC/EC ≤ 2.27; pool median 5.54), locked
disjoint-site TOR test before fitting, first-major-minimum k = 6.

| Model (MAC = 10, n = 190) | slope | intercept | R² | RMSE |
|---|---:|---:|---:|---:|
| Deployed SPARTAN FTIR EC | 1.90 | −4.17 | 0.764 | 1.49 |
| Smoke IMPROVE (906) | 2.65 | −6.91 | 0.685 | 2.85 |
| Ethiopia-shaped smoke (300, ftir_10) | 1.75 | −3.69 | 0.742 | 1.36 |
| **Lowest-OC/EC (800, k=6)** | **1.59** | **−3.22** | **0.774** | **1.16** |

- Held-out TOR (disjoint sites): RMSE **3.41 µg/filter** vs **4.39–7.31** for ten size-matched
  random cohorts at the same k — the advantage is compositional, not statistical.
- **VIP convergence (key finding):** the low-OC/EC model's VIP profile correlates
  **ρ = 0.74** with the Addis-only HIPS model from `ftir_08`, while the current 906-smoke EC
  model correlates only **ρ = 0.12** with it. Two independent routes (Addis optical training
  vs IMPROVE composition selection) point to the same spectral features.
- Caveats: intercept still ≈ −3 µg m⁻³; slope MAC-dependent; most random nulls reach
  less-negative intercepts — but only with degraded slopes and worse held-out TOR — so
  intercept alone is not an acceptance criterion.

## ftir_12 — the ~1600 cm⁻¹ band: not amine; carboxylate and/or aromatic C=C

- Addis peak center: median **1618–1619 cm⁻¹** (IQR 1616–1620), and unchanged on
  AIRSpec-corrected spectra (ftir_13). Every IMPROVE cohort sits higher — medians
  **≥ 1633 cm⁻¹** under the canonical edge-corrected window (ftir_12), ~1650 under the
  replication's convention — with essentially no distributional overlap.
- Within Addis the band co-varies with CH (ρ = 0.88) and carbonyl (ρ = 0.93) but **not** with
  the 3100–3400 cm⁻¹ N–H/O–H window (ρ = 0.17; IMPROVE groups: 0.62–0.67). An N–H bend
  without its stretch is implausible → **amine assignment rejected**. The apparent 1520–1560
  companion on corrected spectra (raw-height r = 0.87) vanishes under CH normalization
  (Spearman −0.29) — a loading artifact, not a ring-mode pair detection.
- Remaining candidates — carboxylate COO⁻ asym and aromatic C=C ring stretch (both
  charcoal-consistent) — cannot be separated on Teflon spectra: the ~1400 cm⁻¹ symmetric
  partner sits below the trusted range (AIRSpec segment 2 ends at 1425 cm⁻¹). Needs the
  sub-1500 model Satoshi's group is developing, or lab charcoal spectra.

## AIRSpec port + ftir_13 — EDF 6–8 on real baselines

R is gone from this machine, but the exact APRLssb sources (`reference/APRLssb/`, verified
byte-identical to the Drive backup) and the R run's ETAD output survive, so the Python port in
`scripts/airspec_baseline.py` is validated against real R ground truth (DF1 = 6, DF2 = 4)
before being applied to the 13,634-spectrum IMPROVE pool at DF1 = 6 and 8.

Port validation: all 319 R-corrected ETAD scans reproduced to ≤6×10⁻⁷ absorbance (worst
relative error 5.6×10⁻⁴ of signal RMS) — this is a validated stand-in for AIRSpec, not an
approximation. All 13,634 pool spectra + Addis corrected at DF1 = 6 and 8 (DF2 = 4).

Results on the fixed 190-filter Addis cohort (MAC = 10):

| Model | slope | intercept | R² | RMSE |
|---|---:|---:|---:|---:|
| Lowest-OC/EC raw (ftir_11) | 1.59 | −3.22 | 0.774 | 1.16 |
| **Lowest-OC/EC corrected EDF6** | **0.86** | **−1.61** | 0.657 | 2.41 |
| Smoke 906 raw (ftir_10) | 2.65 | −6.91 | 0.685 | 2.85 |
| Smoke 906 corrected EDF6 | 0.37 | −0.68 | 0.458 | 3.83 |

- **Correction halves the low-OC/EC intercept** (−3.22 → −1.61) and yields a held-out TOR
  slope of 1.01 (R² 0.90) — the best locked intercept-with-defensible-slope so far, at the
  cost of Addis precision (RMSE 1.16 → 2.41).
- **The smoke calibration collapses on corrected spectra** (slope 0.37): its raw-spectrum
  response was substantially carried by the broad baseline — the component the correction
  removes and the component EC's sloping absorption lives in.
- **EDF 6 vs 8 is indistinguishable** (Δslope < 0.01): the EDF choice inside Satoshi's range
  is not a sensitive parameter.
- **The MAC = 6 vs 10 fork is now the deciding unknown**: raw low-OC/EC is self-consistent at
  MAC = 6 (slope 0.95), corrected low-OC/EC at MAC = 10 (slope 0.86). An independent Addis
  TOR/EC reference or a HIPS MAC/protocol bridge resolves it; further cohort engineering
  will not.

## ftir_14 — Delhi/Beijing score-space comparison: blocked on data

The local DB pull covers IMPROVE only (169,566 analyses, no SPARTAN sites); ETAD spectra came
from a separate site-specific pull. Doing the same for INDH/CHTS (adapt
`etad_lots_query.sql` / `pull_scans.ps1`) unblocks a one-notebook analysis using the
already-fitted models. Second-order per the meeting: Delhi/Beijing are test sets, not
calibration inputs.
