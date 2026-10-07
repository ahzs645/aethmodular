# Completed VIBES gallery migration — 2 October 2026

The active VIBES views use the completed rank-70 correction. Original truncated
and preliminary rank-30 results remain separately labelled archives. This
migration refits dependent models; it does not substitute a new trace underneath
an old model score.

## Inputs and claim limits

`research/ftir_hips_chem/scripts/vibes_current.py` validates the recovered array,
original source hashes, case identities, shared channel grid and frozen eligibility
ledger before use. The recovered array is at
`research/ftir_hips_chem/output/tables/vibes_fullrange/interpretation_20261002/corrected_VIBES_full_recovered.npy`.
The original source signature is
`9345e818c61929fe4510f70b12c5ca4fa5390d47cfa5b774c5a677804bf27bf7`.

There are 12,808 finite correction cases: 12,393 IMPROVE calibration filters,
253 Addis target filters, 135 blank cases and 27 injection cases. Two held-out
blanks were recovered with a higher optimizer budget against the same frozen
rank-70 background. The measured grid is 3998–500 cm⁻¹, 2,722 channels; the
shared spline/model window contains 2,002 channels at or above 1425 cm⁻¹.
There are no measured channels down to 400 cm⁻¹ in this export.

IMPROVE TOR is the source-domain thermal EC reference. Addis HIPS/MAC is an
optical comparison, not independent thermal EC validation. The previously
inspected IMPROVE test set makes new comparisons exploratory. No negative
predictions were clipped, no new exclusions were introduced, and site splits
remain fixed. Existing SPARTAN FTIR/ChemSpec chemistry is not replaced by these
experimental predictions.

## Migrated views

| Gallery view | Current data and verification |
| --- | --- |
| Baseline benchmark, spectral median/IQR overlays, blank residuals, synthetic recovery | Completed recovered array; full and shared model inputs; original benchmark cohorts |
| Per-filter spectra, sample and record drawers | Rank 70 by default; native endpoints retained; preliminary rank 30 remains an explicit choice |
| Baseline residuals, paired error histogram, loading bands, site/loading heatmap | 2,327 full-pool and 137 restricted test filters; saved scores reproduce from paired rows |
| Spectral region decomposition | All 2,464 paired predictions reconstructed and algebraically closed within 1e-8 µg/filter; original 12 inspection identities retained |
| Blank distributions | Current shared-window rank-70 residuals; all 135 blank cases retained |
| Common-test calibration selection and routing | Six paired calibration selections, 2,327 common test filters; nested training-site routing refits; current predictions and score tables reconcile |
| Training spectral space | Training-only eight-component PCA for each correction; 5,160 method/population rows, including all 253 Addis target spectra |
| External Ethiopia pilot | All 31 external spectra converged at rank 70; full-range correction followed by shared-window EC prediction; original 5 Adama, 26 Bishoftu and 5 date-matched pairs retained |
| Follow up meeting | Seasonal, full-pool, stitched and combined models refitted for both rank-70 windows; 1,200 repeat fits; original calendars and paired-measurement coverage preserved |
| Calibration and meeting specification curves/category heatmap | Complete 9,010-row grid: 3,138 unchanged spline fits and 2,936 fits per rank-70 VIBES window; incomplete/duplicated exports are rejected |
| Spectral similarity rankings, matrices and trace overlays | Separate current full-range and shared-window matrices, masks and binaries for 12,646 physical filters and 696 site/season groups |

Calibration plots below the baseline specification curve use original
spline/raw/derivative inputs and do not contain VIBES models. Seasonal chemistry
stacks describe existing SPARTAN products; those source measurements retain their
original provenance.

## Reproduction and review

Derived outputs and input receipts are under
`research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/`.
The workflow commands are in [gallery/README.md](../gallery/README.md).
Generated browser assets are git-ignored and must be rebuilt when distributing
this local gallery. No deployment or commit was performed.

Current analysis is the default in the Calibration and Follow up meeting tabs.
Their version selectors persist as `cversion` and `mversion`. Baselines and
Similarity retain separate archive toggles. Archive switches wait for the
matching export before rendering, preventing mixed-version charts.

Validation: 39 gallery tests pass, TypeScript/production build passes, and the
new Python migration/paired workflows pass Ruff. Data checks cover complete
configuration keys, paired benchmark score reproduction, decomposition closure,
common-test prediction identities, routing arithmetic, grid-specific binary
sizes and external convergence/matching. Browser checks cover rank-70 seasonal
crossplots, current and archived specification selectors, archive switching,
and both similarity channel windows.
