# Filter relationship stability — 10 September 2026

The frozen 545 diagnostic and 480 ratio pairs now have a reported-date stability
analysis. The specification was written and hashed before fitting; quarter
boundaries, memberships and model rules were not changed after seeing errors.

Addis improves on a training-median HIPS baseline in all eight withheld quarters
(MAE 3.938 versus 8.888 Mm⁻¹), with persistent block bias. Its 2× MDL sensitivity
retains 189 filters and 55.9% improvement. ETAD-0037 omission changes MAE on the
other 189 held-out filters by only −0.0049 Mm⁻¹. Calibration-set IDs and reported
date are confounded, so the source of temporal structure remains unresolved.

JPL's improvement falls from 23.9% in the diagnostic population to 9.6% for ratio
eligibility and 0.5% at 1.5× MDL. The seven-point 2× subset remains visible but
cannot meet the frozen minimum of ten training filters. These are different
populations, not competing estimates from which a threshold was selected.

## Deliverables

- [Analysis specification](filter-relationship-stability-spec-2026-09-10.md).
- [Scientific report](../research/ftir_hips_chem/output/tables/filter_relationship_stability/filter_relationship_stability_report.md).
- [Executed notebook with nine figures and notes](../research/ftir_hips_chem/notebooks/archive/executed/filter_relationship_stability.ipynb).
- [Active notebook source](../research/ftir_hips_chem/filter_relationship_stability.ipynb).
- [Held-out predictions and original measurement links](../research/ftir_hips_chem/output/tables/filter_relationship_stability/heldout_filter_predictions.parquet).
- [Block performance and individual/block influence](../research/ftir_hips_chem/output/tables/filter_relationship_stability/block_performance_and_influence.csv).
- [Draft upstream ChemSpec packet — not sent](../research/ftir_hips_chem/output/tables/filter_relationship_stability/upstream_ChemSpec_question_packet.md).

## Reproduce

From the repository root:

```sh
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/analyze_filter_relationship_stability.py
```

Use `--notebook` to regenerate the active notebook and execute its archived copy.
Use `--retrieve-candidate` only to refresh the separate bounded, read-only SQLite
query for JPL:USPA-0257. Default analysis uses the saved retrieval evidence.

The candidate query recovered 1,440 session-46 records with exact timestamp/datum
and IR BCc value agreement against staged inputs. Filter-linked active operation,
scoped correction/quality history and source-backed eBC units remain unresolved;
no verified interval eBC or absorption mean is reported.

The original 37 descriptive artifacts remain byte-identical. All 545 physical
filter links were checked against original EC and HIPS rows; 341 supported block
fits were independently recomputed with a separate least-squares calculation.
The targeted regression/split and earlier interval tests passed (26 tests).
Neither the downstream holdouts nor these checks establish independence from the
original FTIR training process.
