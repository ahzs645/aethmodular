# Proportionality and limits of temporal transfer — specification v2

This separate extension preserves the completed 545/480-population stability
analysis. Its specification was frozen before the new proportional and ID-11
restricted-training fits, after the earlier results were inspected.

Addis benefits consistently from an intercept: primary MAE falls from 11.576 for
proportional prediction to 3.938 Mm⁻¹, with all eight quarters improved. Later-period
MAE falls from 13.110 to 4.274 Mm⁻¹, with all six supported quarters improved.
This establishes predictive usefulness within the evaluated record, not the
physical origin of an offset, a universal conversion or independent FTIR accuracy.

The pattern is not universal. Delhi's primary proportional MAE is lower overall;
JPL's differences are small. Beijing's later-period conclusion changes slightly
with the weighting estimand. Both equal-filter and equal-quarter scores are shown.
Delhi's primary intercept bias is −1.779 versus +2.151 Mm⁻¹ under those two
weightings. Its later-period error remains strongly negative, and 26/38 test
filters are in 2024Q2.

The bounded Addis ID-11 sensitivity compares 127 common later ID-11 filters.
Restricting training lowers OLS MAE from 4.009 to 3.884 Mm⁻¹ and mean error from
+1.621 to +0.619 Mm⁻¹, with three quarter-level wins and two losses. Identifier/date
confounding prevents causal attribution. No residual correction was applied.

## Artifacts

- [Frozen v2 specification](filter-proportionality-spec-v2-2026-09-10.md).
- [Scientific report](../research/ftir_hips_chem/output/tables/filter_proportionality/proportionality_temporal_transfer_report.md).
- [Executed notebook: eight figures and notes](../research/ftir_hips_chem/notebooks/archive/executed/filter_proportionality.ipynb).
- [Active notebook](../research/ftir_hips_chem/filter_proportionality.ipynb).
- [Predictions](../research/ftir_hips_chem/output/tables/filter_proportionality/proportionality_predictions.parquet).
- [Paired block comparisons](../research/ftir_hips_chem/output/tables/filter_proportionality/proportionality_blocks.csv).
- [Both weighting schemes](../research/ftir_hips_chem/output/tables/filter_proportionality/proportionality_summary.csv).
- [Extended upstream questions — not sent](../research/ftir_hips_chem/output/tables/filter_proportionality/upstream_questions_v2_draft.md).
- [USPA-0257 candidate evidence package](../research/ftir_hips_chem/output/tables/filter_proportionality/USPA-0257_evidence_package.md).

## Reproduce

```sh
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/analyze_filter_proportionality.py
```

Add `--notebook` to regenerate the active source and execute its archived copy.
The workflow verifies the input/specification hashes and the previous stability
outputs. It does not rerun the SQLite query or send the evidence questions.

Validation: 30 targeted tests passed. All 4,668 previous baseline prediction rows
(including unavailable rows) retain their median/OLS predictions and original
split fingerprints. All 257 proportional fits were independently recomputed with
least squares. Equal-filter and equal-quarter errors reconcile with the block
ledger; common ID-11 support and the five nonpositive diagnostic EC values are
preserved.
