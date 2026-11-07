# Closeout figure populations — reviewed 7 November 2026

The six-figure preset combines several evidence populations. **The corrected
source comparison is pending in this reviewed ledger.** The other counts below
are reconstructed from the frozen 4 November export and restored decision
metadata; they are not new model fits or a claim of UI/CI acceptance.

Frozen `continuum_baseline.json` SHA-256:
`90b8e16bfcb002db7e6bc7553d190c94f462753182fcb80d5bacc1c65a2111c9`.
Pinned `filters.json` SHA-256:
`a043679b673e82cdc6b539c6aeab38f97855f142ca7f1901a61bc66ff416089d`.
Both belong under `research/ftir_hips_chem/output/tables/closeout_20261107/frozen_public/`.

## Reviewed figure ledger

| Preset figure | Population actually represented | Metric, reference and limits |
|---|---|---|
| Coverage and identity | **233 unique Addis physical filters**: Bega/Belg/Kiremt **81/81/71**; lots 248/251/253 **34/185/14**. All have HIPS; 172 have aethalometer values. | Physical filters, not scans. Dates 2022-12-07–2026-01-04. This is different from the separate 232-filter matched seasonal study. |
| Source-reference performance | **Pending verified corrected-score receipt.** The legacy 12,387 fit-row count is not a verified common-score denominator. | IMPROVE TOR EC, µg/filter, fixed 18 components, site-grouped cross-validation. Legacy `improve_error` is median absolute error; MAE, RMSE, signed bias, tail error, common coverage and recorded QC require the new receipt. Historical EPO scores remain withheld. |
| Baseline behavior | **102 displayed bin medians** across six preprocessing panels. Per panel: 10,701 IMPROVE lot-251, 1,271 lot-248 and **231 Addis** bin members. Two further Addis filters occupy an omitted n=2 bin. | Per-filter mean absorbance at 1850–1910 cm⁻¹, then bin median; display in thousandths of absorbance. HIPS loading at MAC 10; bins require n≥5. Populations repeat across panels. |
| Seasonal aethalometer contrasts | **170 complete cases** for each of four outcomes. Seasons **51/63/56**; lots 248/251 **33/137**. | Saved log(outcome) on log(MA350 IR BCc) plus season intercepts, relative to Belg. The historical eligibility also required the inferred comparator. Approximate 90% intervals transform saved log-scale SEs; no refit. |
| Optical-response sensitivity | Different selected, overlapping curve populations; see below. Parametric fits: **11,311 filters/152 sites**, or **794/121** for low OC/EC. Addis response population **232**, with **171** aeth matches. | IMPROVE TOR-derived response transferred to Addis. Curve x is geometric-mean HIPS/reference depth; site-table depth is raw HIPS. Curve bands are approximately 90%; parameter bootstrap ranges are **80%** from 300 site resamples. |
| Candidate target comparison | **172 identical physical-filter pairs** in both panels, out of 233 export filters. Seasons **51/63/58**; lots 248/251 **33/139**. | Historical production VIBES and PTFE-anchored VIBES, both k=18, against MA350 IR BCc; pooled Deming λ=1. Filter-window aeth means require ≥12 h coverage. These are not implied Newton products or independent Addis TOR validation. |

Calendar: **`belg_feb`** — Bega October–January, Belg February–May,
Kiremt June–September. Seasonal complete cases end on 2024-09-12; target pairs
end on 2024-09-21. Both begin on 2022-12-07.

## Why 233, 172, 170 and 232 differ

The target plot has 61 filters without aethalometer values. Its 172 pairs include
ETAD-0241 and ETAD-0246. The saved seasonal model omits both because its complete-case
gate requires `exp_tor_hips`. Older network metadata excludes ETAD-0241;
ETAD-0246 lacks network FTIR OC/EC and therefore the inferred comparator. The
historical response population excludes ETAD-0241 but can retain ETAD-0246.

The current canonical exclusion is **ETAD-0248**, a PM10 filter collected on
2024-08-07. **ETAD-0241 is retained.**
[`scripts/outliers.py`](../research/ftir_hips_chem/scripts/outliers.py) matches IDs
when available, so the shared date does not exclude both filters. The pinned
focus export and Addis metadata agree with this current rule. Preserve the
historical 170/232 summaries with their eligibility qualification; do not invent
a new exclusion in the 172-pair graph.

Coverage and target plots consult the focus `filters.json` exclusion map.
Of the 233 exported IDs, 189 are known to it and 44 newer IDs count under its
unknown-ID rule. Date/site/season selection is not applied by this
`useCounted` path. Aggregate continuum, seasonal, response and source summaries
are fixed; live inclusion changes do not refit them.

## Aggregate and synthetic limits

| Optical curve | Exported bin members | Members in displayed n≥5 bins |
|---|---:|---:|
| IMPROVE HIPS/TOR | 8,895 | 8,895 |
| IMPROVE HIPS/TOR, OC/EC <2.5 | 392 | 384 |
| IMPROVE HIPS/out-of-site FTIR | 9,364 | 9,364 |
| SPARTAN HIPS/anchored FTIR, Ethiopia omitted | 2,434 | 2,430 |
| SPARTAN HIPS/published FTIR, Ethiopia omitted | 2,119 | 2,115 |

Do not add these overlapping populations. The site table displays the heaviest
12 of 27 eligible sites, representing 1,323 eligible filter records, under a different
selection. The export does not record modelwise denominators for the six Addis
response-summary rows. Aggregate curves omit member IDs, and continuum `n`
is bin membership rather than an explicit finite-absorbance denominator.

The baseline figure reports the **2–3 November light ambient-filter injection
study**. Its reviewed workflow defaults are 16 parent filters, three backgrounds,
and three additions plus unspiked spectra: **192 fitting jobs**. The 87 blanks
train the background. Parent IDs, fit statuses and failed-fit counts are absent
from the export. This is separate from the later independent-blank continuum
control. The generic 32,286-fit/zero-failure provenance describes a different
baseline run and cannot certify injection convergence. The noise summary measures
scatter across light ambient filters, not repeated blank measurements.

## Reproduce the bounded audit

From the repository root, with the frozen inputs restored:

```bash
uv run python research/ftir_hips_chem/workflows/audit_closeout_figures_20261107.py \
  --output research/ftir_hips_chem/output/tables/closeout_20261107/figure_audit_pending_review
```

For the eventual corrected source attachment:

```bash
uv run python research/ftir_hips_chem/workflows/audit_closeout_figures_20261107.py \
  --source-scores research/ftir_hips_chem/output/tables/closeout_20261107/source_scores_v2 \
  --output research/ftir_hips_chem/output/tables/closeout_20261107/figure_audit
```

`--frozen-export`, `--decision-bundle` and `--filters-export` override the
repo-relative defaults. `--output` must be a new directory; existing receipts
are never overwritten. The outputs are `figure-ledger.json` and
`figure-ledger.md`.

Without `--source-scores`, the source row remains explicitly pending. With it,
the workflow uses `candidate_metrics` to verify the corrected receipt, binds it
to the supplied metadata/manifest/grid, and reconciles recorded QC, common IDs,
availability and saved population counts. It reads saved error values without
rescoring or fitting. The separate matched-resolution control remains outside
the six primary score rows.

The JSON records input hashes, physical-ID hashes, interval reconstructions and
reviewed source-file hashes. Source hashes do not certify execution or test
success. Fit/projection lineage and held-out-label invariance belong to
`verify_closeout_source_20261107.py`; browser and release checks remain separate.
