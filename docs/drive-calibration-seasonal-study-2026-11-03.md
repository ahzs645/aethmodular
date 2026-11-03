# FTIR calibration with Dry/Bega, Belg and Kiremt retained

**Completed 3 November 2026 — existing Davis data, matched calibration experiments.**

## Main result

Seasonality materially changes the interpretation of the FTIR–optical comparison.
On the same Addis filters, the raw shared-window calibration is below the HIPS
optical comparator in Dry/Bega and Belg and above it in Kiremt. The spline and
arPLS models remain substantially below that comparator in all three seasons.
This pattern survives a lot-251-only comparison and a comparison restricted to
2023–2024, the years with observations in all twelve months.

The independent source-reference experiment gives a different part of the
picture. Raw spectra have somewhat smaller cross-validation error against
IMPROVE thermal-optical EC, while spline models give field-blank predictions
closer to nominal zero. The raw-versus-spline source-error interval includes
zero. These results support retaining multiple diagnostics; they do not establish
one improved Addis EC calibration.

This study verifies the completed eleven-arm calibration against fresh downloads
of the supplied Drive data, adds a newly executed arPLS iteration-budget arm,
and runs a new season-aware analysis of all twelve arms. The published FTIR
product is included as separate context. It is not a thirteenth recalibrated model.

## 1. Data actually used and verified

| Population | Contents and use |
|---|---|
| Historical IMPROVE calibration | 906 physical filters listed in the official EC export; 905 in the consistently flagged primary analysis; 140 sites; sampling 17 May 2022–30 April 2023. Independent thermal-optical EC supplies the response. |
| Source spectral archive | 935 spectra, 2,722 channels. The 29 records outside the official 906-filter list are retained in the selection audit. |
| Native Addis scans | 319 original scans of 296 physical filters: 253 PM2.5, 37 field blanks and six PM10. Repeated scans retain their physical-filter identity. |
| Primary optical comparison | 232 PM2.5 physical filters with direct HIPS Fabs, valid native volume, matching collection date and eligibility in the frozen current-product comparison cohort. |
| Missing seasonal information | Six PM2.5 filters have missing native dates and invalid native volume. All 37 field blanks lack collection dates. They remain in the audit, without invented seasonal labels. |

Fresh downloads of **all ten calibration/native inputs** reproduce the exact
SHA-256 hashes in the completed calibration manifest. These are `apps_data.npz`,
`EC_samples.csv`, `EC_stats_chosen.csv`, `EC_fitted_long.csv`, `OC_samples.csv`,
`ftir_catalog.csv`, `ETAD_FTIR_spectra.csv`, `ETAD_metadata.csv`,
`etad_spectra_lotmap.csv`, and `spectra_baselined_AIRSPEC.csv`.

Direct optical lineage was independently checked against the freshly downloaded
`SPARTAN_HIPS_Batch1-51.v2.csv`. Every one of the 232 eligible physical filters
has exactly one PM2.5 HIPS record; all collection dates and sampling volumes
match. The export rounds Fabs to six decimal places before conversion. Repeating
that operation reproduces the stored optical values to 8.88 × 10⁻¹⁶ µg/m³.
The difference from unrounded raw Fabs is at most 4.99 × 10⁻⁸ µg/m³.

The current repository optical export was also hashed, and all 3,564 frozen
target prediction/join rows across twelve arms reproduce its dates, values,
optical-source labels, missingness and eligibility. The older portal ChemSpec
snapshot in the supplied folder is dated 17 April 2025. Its EC is an FTIR-derived
product and is not used as independent thermal-optical truth.

The optical cohort is deliberately fixed for comparability: it requires the
existing public FTIR and optical pair to be available. It should not be described
as every possible Addis filter with an optical measurement.

## 2. Calibration and validation protocol

Every preprocessing representation receives its own PLS calibration against
the same source EC loadings. Corrected spectra are never substituted into
coefficients trained on raw spectra. The full raw representation uses all
2,722 channels; most matched comparisons use the same 2,002 channels strictly
between 1425 and 4000 cm⁻¹.

Five outer folds hold out entire IMPROVE sites. Each source filter has one
out-of-fold prediction, and every arm uses identical site-fold membership.
Within each outer training fold, five site-grouped inner folds select among
1–40 PLS components with the repository's existing first-major-minimum rule.
Only source TOR labels enter that selection. A separate time check trains on
807 source filters before 1 November 2022 and evaluates 98 later filters;
that check separates time but does not separately hold out every site.

The primary 905-filter cohort consistently flags a previously documented
TONT1 scan with an approximately +3 absorbance offset. This flag predates the
method comparison. The original 906-filter run and all original measurements
remain in a separate sensitivity ledger. The offset matters: raw-full RMSE
is 63.423 µg/filter in the original 906 experiment and 16.904 after all arms
are refit under the common documented-QC policy. That contrast cannot be
silently attributed to a baseline algorithm.

Final models are fit using the source data, then frozen before applying them
to Addis. Source and native grids agree to float32 precision; the maximum
alignment displacement is 0.000122 cm⁻¹. Native scans are corrected before
averaging within physical filters. All original scans remain primary, and
the previously documented ETAD-0139 scan-1668-only alternative is a separately
labelled sensitivity. Negative predictions remain signed. Concentrations use
the native sampling volume only when it is finite and positive.

These are exploratory comparisons of a historical smoke-selected calibration
population. They are not an untouched external test of the selected method,
and they do not reproduce the current operational SPARTAN calibration.

## 3. Independent source EC results and blank behavior

The source errors below pool the five outer held-site folds, **n = 905**.
RMSE and MAE are µg/filter. Predictive R² is 1 − SSE/SST, rather than squared
correlation. Blank medians are predictions for 37 independent Addis field
blanks, in µg/filter; nominal zero is a diagnostic reference, not a measured
blank TOR label.

| Preprocessing with matched recalibration | Source RMSE | Source MAE | Predictive R² | Median field-blank loading |
|---|---:|---:|---:|---:|
| Raw, full spectral grid | 16.904 | 6.690 | 0.574 | +6.306 |
| Raw, shared window | 17.652 | 6.575 | 0.535 | +7.041 |
| Spline baseline, df 6/4 | 18.787 | 7.105 | 0.474 | −0.367 |
| Spline baseline, df 6/3 | 18.733 | 7.083 | 0.477 | −0.338 |
| Spline baseline, df 6/6 | 19.777 | 7.466 | 0.417 | +0.858 |
| Removed spline background only | 17.134 | 6.341 | 0.562 | +3.856 |
| arPLS, default iteration cap 50 | 19.738 | 7.483 | 0.419 | −1.431 |
| arPLS, iteration cap 500 | 19.745 | 7.484 | 0.418 | −1.357 |
| Raw, excluding 1580–1650 cm⁻¹ | 17.710 | 6.615 | 0.532 | +7.511 |
| Spline, excluding 1580–1650 cm⁻¹ | 18.900 | 7.128 | 0.467 | +4.346 |
| Spline, using only 1580–1650 cm⁻¹ | 20.729 | 7.859 | 0.359 | +0.110 |
| Second derivative | 18.726 | 6.877 | 0.477 | +3.719 |

Relative to the spline df 6/4 reference, the raw-full RMSE difference is
**−1.883 µg/filter**, with a paired site-bootstrap 95% interval of
**−5.891 to +1.732**. The raw shared-window difference is −1.135, with an
interval of −3.886 to +1.370. These intervals resample the 140 sites using
fixed held-out predictions; they do not refit the whole model or represent
uncertainty over every possible fold partition.

The continuum removed by the spline retains predictive covariance with source
EC. It cannot automatically be treated as disposable noise. Conversely, this
does not establish that the removed component is pure soot absorption:
loading, substrate, scattering and composition can covary. Removing only the
1580–1650 cm⁻¹ window changes the spline source RMSE little, while using that
window alone performs worse. The available predictive information is not
confined to that narrow region.

The field blanks show why source RMSE alone is insufficient for transfer.
Raw-full predicts a median +6.306 µg/filter on the Addis blanks, versus
−0.367 for spline df 6/4. Neither a good nominal-zero blank result nor a
smaller source RMSE establishes the target-domain EC accuracy by itself.

## 4. New numerical test: increase the arPLS iteration budget

The existing neutral correction uses the penalized-spline arPLS implementation
in pybaselines 1.2.1, with lambda 10⁶. Its baseline fit uses native channels
at or above 700 cm⁻¹, excluding the 1100–1300 cm⁻¹ region, before selecting
the shared prediction window. This is distinct from the independent Whittaker
arPLS implementation used in the earlier synthetic recovery challenge.

The new test changes only the iteration cap from 50 to 500. The default
correction was replayed first and reproduced the previous cache **exactly**.
The same source rows, site folds, component rule, spectral masks, target scans
and optical cohort are used afterward.

| Numerical or validation result | Cap 50 | Cap 500 |
|---|---:|---:|
| Source spectra meeting tolerance | 779/905 | 899/905 |
| Native target scans meeting tolerance | 268/319 | 319/319 |
| Source site-held-out RMSE, µg/filter | 19.7377 | 19.7448 |
| Later-date source RMSE, µg/filter; n=98 | 12.2319 | 12.2350 |

More iterations resolve most numerical stopping failures, but do not improve
the source error estimate. Six source spectra still miss the tolerance at cap
500, including early numerical-safeguard stops; their finite outputs remain
explicitly flagged in the matched analysis. The new arm is therefore a
**500-iteration sensitivity**, not a completely converged source calibration.
Relative to spline df 6/4, its source RMSE difference is +0.958 µg/filter,
with a site-bootstrap interval of +0.273 to +1.705.

## 5. Primary Ethiopian seasonal comparison

For new Addis reporting, the project recommends the calendar used by the
[Ethiopian Meteorological Institute](https://www.ethiomet.gov.et/documents/108/Annual_bull_for_2023.pdf):
**Bega/dry October–January; Belg February–May; Kiremt June–September**.
The registered `belg_feb` convention is primary. The older `dry_feb`
convention, with February in dry, is evaluated separately.

Seasons are assigned from native sampling start dates. End-date crossings
are flagged. These are calendar bins; no assertion about observed rainfall
on an individual sampling day is added.

The table reports the median of each filter's **FTIR prediction minus
HIPS Fabs/10**, in µg/m³. Negative values mean FTIR is below the optical
comparator. MAC 10 m²/g is an explicit optical-conversion assumption.

| Model or contextual product | Bega/dry, n=81 | Belg, n=81 | Kiremt, n=70 |
|---|---:|---:|---:|
| Raw, full grid | −2.176 | −2.011 | −1.482 |
| Raw, shared window | −0.896 | −0.268 | +1.420 |
| Spline baseline, df 6/4 | −3.428 | −3.728 | −3.823 |
| Spline baseline, df 6/3 | −3.424 | −3.706 | −3.778 |
| Spline baseline, df 6/6 | −3.477 | −3.779 | −3.926 |
| arPLS, cap 50 | −3.718 | −3.681 | −3.856 |
| arPLS, cap 500 | −3.710 | −3.689 | −3.850 |
| Removed spline background only | −0.889 | −0.182 | +2.215 |
| Raw, excluding 1580–1650 cm⁻¹ | −0.839 | −0.196 | +1.473 |
| Spline, excluding 1580–1650 cm⁻¹ | −3.251 | −3.336 | −3.494 |
| Spline, using only 1580–1650 cm⁻¹ | −3.870 | −4.123 | −5.041 |
| Second derivative | −2.425 | −2.685 | −2.816 |
| Existing published FTIR, context | −1.046 | −0.510 | +1.547 |

The annual median of −0.310 µg/m³ for raw shared-window optical agreement
conceals its sign reversal across seasons. A model can also alter the seasonal
pattern substantially while changing source RMSE much less. That is a transfer
diagnostic that an annual source-error ranking alone does not reveal.

These differences are not measured Ethiopian EC errors. These SPARTAN filters
lack an independent thermal-optical EC reference, and HIPS-derived optical BC
depends on MAC. The existing published FTIR value is a contextual product,
not an additional independent carbon measurement.

## 6. Seasonality checks: February, lot and year

### February ownership

Moving February changes membership for **21 direct-HIPS pairs**:

| Calendar | Dry/Bega | Belg | Kiremt |
|---|---:|---:|---:|
| Primary: dry Oct–Jan, Belg Feb–May | 81 | 81 | 70 |
| Historical: dry Oct–Feb, Belg Mar–May | 102 | 60 | 70 |

The existing published product's Belg median optical difference changes from
−0.236 to −0.510 µg/m³ when February belongs to Belg. The raw shared-window
result changes from −0.175 to −0.268; the spline df 6/4 result changes from
−3.764 to −3.728. Kiremt is unchanged. Calendar choice therefore belongs in
every reported seasonal number; it is not chosen to improve agreement.

### Filter-lot overlap

Among all native PM2.5 physical filters, lot 248 has 18 Bega, 16 Belg, zero
Kiremt and one undated filter. Lot 251 has 37 Bega, 53 Belg, 65 Kiremt and
five undated filters. The unknown-lot group has 32 Bega, 20 Belg and six
Kiremt filters. A lot-248-versus-lot-251 Kiremt comparison is not identifiable.
The two known lots share only March 2023 in observed calendar-month support.

The seasonal pattern nevertheless persists **within lot 251 alone**. On its
37/53/64 direct pairs, raw shared-window median optical differences are
−0.777/−0.217/+1.367 µg/m³; existing published FTIR gives
−0.695/−0.080/+1.519. Thus pooling different known lots is insufficient to
explain the whole pattern. The remaining comparisons are observational and
do not separate every time, composition or environmental influence.

### Partial-year coverage

The primary pairs include dry-season-only tails in 2022 and 2026, and incomplete
2025 coverage. A reproducible sensitivity selects years with direct measurements
in all twelve months, using coverage rather than prediction values. This retains
**188 filters in 2023–2024**, with 55 Bega, 69 Belg and 64 Kiremt pairs.

Raw shared-window median differences remain **−0.853/−0.226/+1.367 µg/m³**;
published FTIR context remains **−0.847/−0.450/+1.519**. The sign reversal
is therefore not created by those partial-year dry-season tails. Every month
is represented in this sensitivity, but sample counts are still unequal.

### Source-calendar support

The source calibration is heavily concentrated in June–October. There are
only five February, eleven March, eleven April and two May source filters.
The source errors are therefore reported by calendar month and quarter,
without calling those American observations Ethiopian rainy seasons. Those
summaries reuse site-held-out predictions; they are not newly fit seasonal
models or leave-season-out validation.

## 7. Optical conversion sensitivity

Four predeclared MAC scenarios are retained: 7.5, 10, 12.5 and 15 m²/g at
the HIPS comparison wavelength of 633 nm. They are illustrative assumptions,
not fitted or independently measured local MAC values.

For existing published FTIR context, median optical differences are:

| MAC scenario, m²/g | Bega/dry | Belg | Kiremt |
|---|---:|---:|---:|
| 7.5 | −2.427 | −1.947 | −0.279 |
| 10 | −1.046 | −0.510 | +1.547 |
| 12.5 | −0.124 | +0.363 | +2.640 |
| 15 | +0.463 | +0.904 | +3.388 |

The sign of disagreement depends on the conversion assumption. The commonly
cited 7.5 m²/g value from the Bond–Bergstrom review is at 550 nm and is not
adopted here as a measured 633 nm MAC. No MAC is selected by making FTIR
and HIPS agree. Separate source training, target optical comparison and
optical-conversion assumptions must remain distinguishable.

## 8. Decisions supported by this test

1. **Keep seasonal reporting in the calibration comparison.** Dry/Bega, Belg,
   Kiremt, calendar name, physical-filter count and year support belong together.
2. **Retain broad-spectrum and removed-background controls.** They retain
   source predictive information and reveal a different target seasonal pattern.
   This motivates investigation without declaring that information pure EC.
3. **Keep spline correction as a serious blank-transfer comparator.** Its
   nominal-zero field-blank behavior is much better than raw-full here, even
   though its source RMSE point estimate is somewhat higher.
4. **Treat the arPLS iteration issue as numerical, with its measured limits.**
   Raising the cap repairs most stopping failures but does not resolve the
   calibration or seasonal optical disagreement in this experiment.
5. **Do not select a season-specific EC calibration by optical agreement.**
   The current data support a season-aware diagnostic, not three independently
   validated Ethiopian EC models. A claim of improved target EC accuracy still
   needs an independent target EC reference or a separately justified validation
   design; that does not invalidate the existing-data comparisons completed here.

## 9. Verification, code and reproduction

Fifteen focused tests pass for the imported calibration contracts, numerical
status handling and seasonality extension. Ruff passes for all nine relevant
Python files. The independent review checks physical identities, identical
source-fold membership, all fresh input hashes, direct optical lineage and
all 1,040 original seasonal summary rows, with maximum arithmetic difference
1.78 × 10⁻¹⁵. The complete-year sensitivity also has a coverage-selection test.

The scientific workflows and interpretation are in draft PR 14:
[research calibration and seasonal experiments](https://github.com/ahzs645/aethmodular/pull/14).
Detailed companion documents are the matched-calibration review, seasonal
results and independent validation review in `docs/`.

Reproduction uses the source input tree and native input folder supplied as
explicit paths. The completed run was reproduced with Python 3.12.14,
NumPy 2.4.2, pandas 2.3.3, SciPy 1.17.0, scikit-learn 1.8.0 and
pybaselines 1.2.1; the repository recommends Python 3.13 generally. The
iteration extension pins pybaselines 1.2.1 to isolate the numerical change.

```bash
uv run python research/ftir_hips_chem/workflows/run_existing_calibration_controls.py \
  --ftir-dir "$FTIR_CALIBRATION_INPUT_DIR" \
  --native-dir "$ETAD_NATIVE_INPUT_DIR" \
  --source-policy documented_qc

uv run python research/ftir_hips_chem/workflows/run_calibration_convergence_sensitivity.py \
  --parent-run "$CALIBRATION_RESULTS_DIR" \
  --ftir-dir "$FTIR_CALIBRATION_INPUT_DIR" \
  --native-dir "$ETAD_NATIVE_INPUT_DIR"

uv run python research/ftir_hips_chem/workflows/run_calibration_seasonality.py \
  --calibration-dir "$CALIBRATION_RESULTS_DIR" \
  --additional-calibration-dir "$ARPLS500_RESULTS_DIR"

uv run pytest -q tests/test_existing_calibration_controls.py \
  tests/test_neutral_convergence.py tests/test_calibration_seasonality.py
```

The optional pybaselines package must be available to the chosen environment.
To regenerate the original-906 sensitivity, run the first workflow with
`--source-policy all_original` and its separate output directory before
running `summarize_existing_calibration_controls.py`.

The evidence bundle includes the complete derived source predictions,
component-selection curves, temporal checks, target predictions, field-blank
outputs, both seasonal calendars, MAC scenarios, February movers, lot/year/month
coverage, complete-year sensitivity, numerical status ledgers and provenance
receipts. Original external data and reusable numerical caches are not duplicated
in that bundle; the recorded hashes identify the inputs needed to regenerate them.

## Research and method references

- Ethiopian Meteorological Institute, *Annual Climate Bulletin 2023*, page 2:
  [season definitions](https://www.ethiomet.gov.et/documents/108/Annual_bull_for_2023.pdf).
- Kuzmiakova et al. (2016), baseline correction for atmospheric aerosol spectra:
  [doi:10.5194/amt-9-2615-2016](https://doi.org/10.5194/amt-9-2615-2016).
- Reggente et al. (2016), FTIR calibration transfer across aerosol populations:
  [doi:10.5194/amt-9-441-2016](https://doi.org/10.5194/amt-9-441-2016).
- Baek et al. (2015), asymmetrically reweighted penalized least squares:
  [doi:10.1039/C4AN01061B](https://doi.org/10.1039/C4AN01061B).
- Official pybaselines documentation, penalized-spline arPLS and stopping history:
  [pspline_arpls API](https://pybaselines.readthedocs.io/en/latest/generated/api/pybaselines.Baseline.pspline_arpls.html).
- Bond and Bergstrom (2006), light absorption review:
  [doi:10.1080/02786820500421521](https://doi.org/10.1080/02786820500421521).

The methods and references extend the earlier annotated research synthesis,
`docs/ftir-research-experiments-2026-11-02.md`. Numerical results above come
from the supplied data and recorded workflows, rather than from those papers.
