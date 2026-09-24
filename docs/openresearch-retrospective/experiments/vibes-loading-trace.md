# VIBES loading range: saved-model trace

## Question
Which corrected-spectrum, fitted-coefficient, intercept and numerical-precision changes account for the saved VIBES versus AIRSpec prediction differences in the full-pool Q3 loading range?

## Fixed inputs
[vibes-inputs.json](vibes-inputs.json) freezes nine saved data/model files against the earlier subgroup audit, plus current diagnostic code and environment hashes. Source signature: `83dcf32e86bc0f092f49f7f59671ac8a89f8efcc28d990287f8b1f895d2b2ae5`. Keep the full-pool and locked800 cohorts separate. All 2,327 full-pool and 137 locked800 held-out filters remain in the evaluation.

Q3 uses training-derived cutoffs (3.739, 6.994] µg/filter, not new cutoffs optimized on test errors. BRIS1 and CACR1 are post-hoc diagnostic priorities from the earlier subgroup audit. They are not a fresh test set.

## Comparison method
Reconstruct each saved prediction with its saved coefficient, centering mean and intercept. Match the original float32 in-place centering; export its difference from float64 algebra explicitly. Decompose the prediction difference symmetrically into `(XV-XA)*(bV+bA)/2`, `(XV+XA)*(bV-bA)/2`, intercept/centering constant and numerical-precision contributions. Aggregate across four fixed, exhaustive wavenumber intervals (1425–1799, 1800–2499, 2500–2999, 3000–4000 cm⁻¹). These broad intervals are not chemical-species assignments.

Export every held-out case and loading-group means. For visual follow-up select the three largest positive and three largest negative squared-error changes at each priority site, breaking ties by sample ID. This inspection subset never removes cases from evaluation. Do not refit, tune or change exclusions.

## Success criteria
All frozen hashes match; sample identity, cohort and train/test invariants pass; all saved predictions reconstruct within 1e-8 µg/filter using the original numerical centering; decomposition closes within 1e-8; report coverage and rounding separately. A useful trace explains the saved prediction arithmetic even if no chemical mechanism is identified. It does not establish better out-of-sample performance.

## Reproduction
From the repository root, verify only:

```sh
uv run --locked --no-sync python research/ftir_hips_chem/workflows/trace_vibes_loading_predictions.py --freeze docs/openresearch-retrospective/experiments/vibes-inputs.json --check
```

Run the diagnostic with the same command without `--check`, supplying `--output` with a new directory under `research/ftir_hips_chem/output/tables/`. The workflow refuses to overwrite existing results. [Input/prediction preflight](vibes-preflight.json).

## Decision after the trace
Inspect paired spectra and solver diagnostics for the declared cases. A physical mechanism needs blank/standard evidence. Any method change must be selected using training-only grouped cross-validation and evaluated on new held-out data; this already-inspected test set cannot validate that change. The original pooled RMSE interval includes zero; the Q3 subgroup contrast is exploratory and conditional on the saved fitted models.

## Local diagnostic completed
All 2,464 held-out rows reconciled; the maximum saved-prediction discrepancy was 5.68e-14 µg/filter. [Results and limitations](../../../research/ftir_hips_chem/output/tables/vibes_loading_trace/report.md). This completes the arithmetic trace, not the physical-mechanism investigation.

## Case investigation completed

All 12 predeclared worse/better cases were inspected and their corrections repeated with the saved background and original settings. The two leading cases have severely negative predictions under both methods, with VIBES more negative; numerical repeatability does not support a gross solver failure as the explanation. No filters were removed and no parameters tuned. [Case report, controls and limitations](../../../research/ftir_hips_chem/output/tables/vibes_case_investigation/report.md).
