# Completed VIBES comparison: error audit and next research step

> **Follow-through update, 2026-09-22:** the spectral inspection proposed in step 1 below is complete for 12 selected cases; see the [case investigation](../research/ftir_hips_chem/output/tables/vibes_case_investigation/report.md). New method improvements and independent Addis validation remain open. The [current synthesis](current-research-summary.md) separates these outcomes from the original audit and its prospective recommendations.

The 12,808-case Colab run is complete. The saved predictions support keeping
AIRSpec as the current EC default. VIBES improves blank correction, but its
pooled EC accuracy advantage is unestablished. The full-pool and restricted
cohorts have different test populations and must remain separate comparisons.

The new [audit report](../research/ftir_hips_chem/output/tables/vibes_subgroup_audit/report.md)
contains site, filter-lot and training-defined loading comparisons. All saved
RMSE, MAE, bias and predictive R² values reconcile within `1e-10`; both methods
cover exactly the same frozen test identifiers in each cohort. No filters were
newly excluded and no model was refitted.

## What the audit adds

| Finding | Interpretation |
|---|---|
| Full-pool RMSE: AIRSpec 3.224, VIBES 3.296 µg/filter; paired delta interval −0.083 to 0.264 | No established pooled improvement from VIBES. This is not proof of equivalence. |
| Full-pool third training-defined loading quartile, (3.739, 6.994] µg/filter: 545 filters, 30 sites | AIRSpec RMSE 2.300 versus VIBES 2.752; delta 0.453, pointwise interval 0.187 to 0.743. |
| That band's MAE also increases by 0.209 µg/filter, with almost unchanged mean bias | The discrepancy is not explained by a common prediction offset alone. |
| VIBES MSE is higher at 23/30 sites within that band; omitting any one site leaves delta RMSE 0.313–0.489 | The pattern is not entirely attributable to a single site. BRIS1 and CACR1 are priorities for inspecting the stored spectra. |
| Both full-pool lot intervals include zero | The lot audit does not establish a lot-based method choice. Lots, sites and loading are associated. |
| Restricted-cohort RMSE improves, but MAE worsens and the pooled RMSE interval includes zero | Avoid choosing VIBES from its headline restricted-cohort RMSE alone. |

Intervals use 10,000 paired site-cluster draws, conditional on the fitted models.
These are exploratory, pointwise intervals after inspecting the test results;
they do not account for multiple subgroup comparisons or refitting uncertainty.
Single-site groups have descriptive metrics and no bootstrap interval. The new
intervals differ slightly from the original 2,000-draw report because the seed
and number of draws differ; the underlying predictions are unchanged.

## Next work, with separate objectives

1. **Inspect the stored spectra and prediction contributions.** Trace the largest
   AIRSpec/VIBES discrepancies in BRIS1 and CACR1 to frozen corrected spectra,
   raw input spectra and PLS coefficients. For each selected case, include
   a comparable case where VIBES improves. Record spectral changes and solver
   diagnostics without declaring bad samples or modifying the evaluated models.
   This can establish how predictions changed, not whether the reference or
   either correction is physically correct.
2. **If that inspection identifies a plausible preprocessing change**, evaluate
   it using only the original training sites with grouped or nested validation.
   Freeze the primary metric and a fresh external evaluation before tuning.
   Use a separate VIBES/AIRSpec experiment family with a fixed command and
   comparison contract in OpenResearch. Link this audit as motivation.
3. **For chemistry**, expand independent blank and standard recovery across
   filter lots and weak-signal amplitudes. The current nine parent blanks
   support a limited synthetic recovery comparison; strong feature association
   between methods does not establish chemical accuracy.
4. **For Addis EC accuracy**, obtain independently measured thermal EC for
   matched physical filters, with method, units, uncertainty, blanks/MDLs and
   sample identities recorded. HIPS agreement is supporting optical evidence,
   not the missing independent thermal reference.

## Evidence and reproduction

- [Audit manifest and input/output hashes](../research/ftir_hips_chem/output/tables/vibes_subgroup_audit/audit_manifest.json).
- [All subgroup metrics](../research/ftir_hips_chem/output/tables/vibes_subgroup_audit/subgroup_metrics.csv).
- [Completed cloud-run record](../research/ftir_hips_chem/output/tables/vibes_colab_cloud/COMPLETION.json).
- [OpenResearch retrospective](openresearch-retrospective/README.md).

```sh
uv run --locked --no-sync aeth doctor
uv run --locked --no-sync python research/ftir_hips_chem/workflows/audit_vibes_saved_predictions.py
```

The workflow verifies the input bundle's content manifest and every listed
member, the archived notebook's error-free saved state, test membership,
training/test site separation, group coverage and source hashes. The cloud
completion is retained as external evidence; this local audit is not presented
as a newly executed OpenResearch cloud run.
