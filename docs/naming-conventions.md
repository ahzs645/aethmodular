# Naming conventions for the FTIR EC calibration work

Agreed with Ann on 23 Sep 2026 for the AAAR poster and the paper. The gallery
reads the same names from `gallery/app/src/lib/labels.ts`; use them in notebooks,
decks and figures too. Internal codes (k, B, AIRSpec, "held-out TOR") should not
appear on reader-facing figures without the name below.

## Data roles

| Term | Meaning |
|---|---|
| **Calibration set** | The IMPROVE filters a PLS model is built on (a *cohort*). |
| **Cross-validation** | Holding IMPROVE sites (or every 10th filter) out of the calibration set and refitting, to choose the number of factors and to check calibration quality. It is not a test set. |
| **Test set** | Data the model never saw. For this work: **the Addis filters**. Every Addis crossplot is a test set result. |
| **Cohort** | Which IMPROVE filters were selected for the calibration set. |

## Settings

| Code | Reader-facing name |
|---|---|
| `airspec` / AIRSpec | **Spline baseline** (smoothing-spline baseline correction; AIRSpec, Kuzmiakova, Dillner and Takahama 2016). Confirm the exact term in the AIRSpec paper before the poster. |
| `vibes` / VIBES | **VIBES baseline** (probabilistic background removal learned from field blanks; pyvibes) |
| `raw` | **Raw spectra** (never "baseline raw") |
| `deriv2` | **Second derivative** |
| `ocec` | **Lowest OC/EC** cohort |
| `analogs` | **Spectral analogs (VIP-weighted distance)** |
| `corr_*` | **Spectral analogs (correlation)**, with the Addis season they were matched to |
| `eth_shaped` | **Ethiopia-shaped smoke** |
| `pool` | **All IMPROVE filters** |
| `smoke` | **Biomass-smoke filters (906)** |
| selection space `airspec` | **selected on spline-baselined spectra** (the cohort choice), as opposed to the spectra the model is calibrated on |
| `mode` / protocol | **Cross-validation protocol**: `site_heldout` = site-grouped CV (5-fold, whole sites held out); `app` = interleaved CV (10-fold, every 10th filter) |
| `k` | **Number of PLS factors** |
| cutoff | **Calibration cohort size (IMPROVE filters)**, e.g. "cohort < 600 filters" |

## Metrics

| Code | Reader-facing name |
|---|---|
| held-out TOR R² | **IMPROVE cross-validation R² (calibration set)**. Report it with RMSE, in a panel separate from Addis results. |
| Addis Deming slope / intercept / R² | **Test set Deming slope / intercept / R² (Addis)** |
| Fabs/10 | **HIPS Fabs / MAC 10** |

Regression on comparison plots is **Deming only** (both axes uncertain). Show R²
and the Deming equation; do not show ordinary least squares.

## Words to avoid

- "Polluted": say **high aerosol concentration**, **high particle concentration**
  or **high elemental carbon concentration**.
- "Held out" for Addis: say **test set**.
