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
| `vibes` / `VIBES` (frozen runs) | **Archived VIBES (fit 4000–1425)**: the samples and blanks were cropped before background fitting. Historical comparison, hidden by default in the meeting page. |
| `VIBES-full` | **VIBES baseline (4000–500, rank cap 30)**: background fitted on the full available grid; preliminary capped calculation. |
| `VIBES-full-cut` | **Sensitivity: VIBES fit 4000–500, predict 4000–1425 (rank cap 30)**: same full-range correction, narrower predictor range. This is separate from fitting a cropped spectrum. |
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

The interactive gallery defaults to Deming and offers OLS/York for exploration.
For publication exports select Deming only. Python axes-level comparison
helpers now display one Deming line and equation, preserving OLS coefficients
in returned metrics for calculations. Pass `sigma_x` and `sigma_y` in the
plotted units when available, or supply saved Deming coefficients and their
variance ratio. Previously saved notebook outputs need rerunning to change.

The archived 4000–1425 VIBES fit is a different experiment, not evidence that
pyvibes cannot fit a restricted grid. The local adapter requires matching
sample/blank grids and imposes no AIRSpec-style lower-wavenumber cutoff.
Current spectral comparison panels may display the full-range correction
only down to 1425; that display crop does not refit the baseline. The narrower
prediction variant is a secondary sensitivity analysis of the information
below 1425, not a preferred calibration established by matching AIRSpec's range.

## Words to avoid

- "Polluted": say **high aerosol concentration**, **high particle concentration**
  or **high elemental carbon concentration**.
- "Held out" for Addis: say **test set**.
