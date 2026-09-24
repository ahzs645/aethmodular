# How VIBES (pyvibes) differs from the AIRSpec spline baseline

Written 2026-09-23 from the code: the upstream pyvibes source
(`~/Downloads/pyvibes-main`, byte-identical to `research/ftir_hips_chem/vendor/pyvibes/src`
in this repo, released 2026-09-18 by Kamper, Krymova and Takahama) and the validated AIRSpec
port `research/ftir_ec_phase3/scripts/airspec_baseline.py` (APRLssb `FitSplineKDT.R`).
Ahmad should check this against his own reading before it goes to Ann.

## In one paragraph each

**AIRSpec (spline baseline).** Each spectrum is corrected on its own. The spectrum is split
into two segments, 4000–1820 cm⁻¹ and 2000–1425 cm⁻¹. In each segment a cubic smoothing
spline is fitted only to the points assumed to contain no analyte: the analyte windows get zero
weight (their edges found per spectrum, e.g. `FindBound` searches down from 3710 cm⁻¹ and
`FindMinPos` looks for the minimum in 1600–1520 cm⁻¹). The spline's stiffness is fixed by its
effective degrees of freedom (DF1 = 6, DF2 = 4 in our runs). The baseline is that spline; the
corrected spectrum is the measurement minus it. Outside 4000–1425 cm⁻¹ the output is empty
(NaN) **by design**: the method has no model below 1425.

**VIBES.** The background is learned, not assumed smooth. A set of field-blank spectra is
decomposed by PCA (`interference_models/pca.py`), giving a mean blank μ and components W; the
number of components is chosen by leave-one-out on the blanks (one-standard-error rule). Every
spectrum is modelled as `y = μ + W x + a`: blank-like background (μ + W x) plus analyte signal
a. The analyte is given an asymmetric penalty (pinball loss with τ ≈ 0.1, `utils/loss_functions.py`)
that makes negative analyte expensive, so the fit pushes the background up to the spectrum's
floor but not above it. Per spectrum, τ and a noise/temperature term are calibrated by
maximizing a variational lower bound (ELBO, `absorbance_estimators/vibes.py`); then the
background is the maximum-a-posteriori solution of a convex problem (cvxpy,
`absorbance_estimators/map.py`). It works on whatever wavenumber grid it is given.

## The differences that matter for our results

| | AIRSpec (spline baseline) | VIBES |
|---|---|---|
| What defines "background" | a smooth curve through the analyte-free points of *this* spectrum | shapes seen in field blanks (PTFE, substrate, instrument), fitted to this spectrum |
| Needs blanks | no | yes: our runs use 87 blanks (IMPROVE training-site blanks + Addis blanks) |
| Assumptions about the analyte | analyte windows are known; outside them the signal is baseline | analyte is (mostly) non-negative; no window list |
| Wavenumber range | 4000–1425 cm⁻¹ only | any; the instrument grid here is 4000–500 cm⁻¹ |
| Per-spectrum tuning | none (fixed DF) | τ and noise level fitted per spectrum (ELBO) |
| Cost | milliseconds per spectrum | about 1–6 s per spectrum on the full grid |
| Failure mode | a stiff spline can leave curvature or cut into broad bands | a blank library that lacks a background shape leaves it in the "analyte" |

## What we have measured so far

- **Blank residuals:** on nine held-out Addis field blanks, VIBES leaves far less signal than
  AIRSpec (median RMS 6.7e-6 vs 5.8e-4 absorbance; `docs/vibes-subgroup-audit-2026-09-21.md`).
- **EC prediction on IMPROVE** (4000–1425 run): no overall advantage (full-pool RMSE 3.224
  AIRSpec vs 3.296 µg/filter VIBES; paired interval includes zero).
- **Addis:** VIBES calibrations give steeper Addis slopes and more negative intercepts than
  AIRSpec on the same filters (meeting follow-up, `output/tables/meeting_followup_20260917/`).

## Why the earlier VIBES run stopped at 1425 cm⁻¹, and the rerun

Nothing in pyvibes stops at 1425 (its own demo data run 590–3998 cm⁻¹). Our Colab bundle
restricted the grid to AIRSpec's window so both methods were compared on identical channels
(`build_vibes_colab_bundle.py`, `1425 < wn < 4000`). Ann's point on 23 Sep: the new method's
advantage includes the region below 1500 (nitrate, ammonium and other bands), so it should be
run on the whole spectrum. Our raw IMPROVE and Addis spectra end at **500 cm⁻¹** (2722
channels, 3998–500), so the full-range rerun uses 4000–500, not 4000–400.

On the full grid the blank PCA wants more components: the one-SE rule picks 67, against 28 on
the 1425–4000 window. Two runs are therefore made: rank capped at 30 (same setting as before,
so the range is the only change) and uncapped (upstream behaviour). Outputs:
`research/ftir_hips_chem/output/tables/vibes_fullrange/`.

For comparisons with AIRSpec, plots are cut at 1425 cm⁻¹ (the spline method has no data
below that), with a separate full-range VIBES plot.
