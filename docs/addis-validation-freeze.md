# Addis independent EC validation: frozen prediction package

Status: **predictions frozen; independent thermal EC and authoritative pairs unavailable**. This package prepares a future evaluation of AIRSpec and VIBES on the existing ETAD filters. It does not claim Addis accuracy or change the completed Colab comparison.

## Fixed prediction set

The [frozen prediction table](../research/ftir_hips_chem/output/tables/addis_validation_freeze/frozen_full_pool_predictions.csv) has one row per physical PTFE filter for all 253 ETAD sample spectra. It uses the completed Colab run signature `83dcf32e86bc0f092f49f7f59671ac8a89f8efcc28d990287f8b1f895d2b2ae5`, its saved AIRSpec/VIBES corrected arrays and its **full-pool** PLS coefficients. No thermal, HIPS or ChemSpec target was used to reconstruct or select the predictions. The [freeze manifest](../research/ftir_hips_chem/output/tables/addis_validation_freeze/freeze_manifest.json) records source and output SHA-256 hashes.

Mass predictions are in µg/filter for all 253 filters. Saved concentration predictions in µg/m³ reconcile to within 5×10⁻⁶ for the 247 filters with positive recorded sample volume. The other six—`etad:855`, `etad:1953` through `etad:1957`—have a mass prediction but lack volume and date in the source export. Their concentration and original sampling season remain unknown until records are confirmed. The `dry_feb` season labels on other rows are calendar categories, not measured weather.

Run from the repository root:

```bash
uv run --locked --no-sync aeth doctor
uv run --locked --no-sync python research/ftir_hips_chem/workflows/freeze_addis_validation.py freeze
uv run --locked --no-sync python research/ftir_hips_chem/workflows/freeze_addis_validation.py package
uv run --locked --no-sync pytest -q research/ftir_hips_chem/workflows/test_freeze_addis_validation.py
```

The freeze refuses to overwrite an altered prediction file or changed source set. The source correction arrays and models remain in the Colab result bundle; the small release archive under `deliverables/addis_validation_freeze_2026-09-22/` preserves the prediction table, empty pairing template and manifest for review. It is a prediction snapshot, not a backup of the complete 12,808-case source run.

On a clean checkout without those large Colab arrays, restore the verified prediction snapshot before pairing work:

```bash
uv run --locked --no-sync python research/ftir_hips_chem/workflows/freeze_addis_validation.py restore
```

## Pairing and evaluation gate

The [pairing template](../research/ftir_hips_chem/output/tables/addis_validation_freeze/independent_tor_pairing_template.csv) starts with the 253 frozen PTFE identities and empty fields for eligibility, independent quartz IDs, laboratory TOR mass, uncertainty, MDL, QA, sampling start/end, time zone, volumes and identity/sampling provenance. Do not fill it by matching date alone or by inferring IDs from EC, FTIR or HIPS values. The five Adama candidate pairs belong to a separate population and are not valid ETAD references.

**Before opening TOR values**, complete the metadata for all 253 rows: mark each `eligible` or `not_eligible`, give an ineligibility reason for every excluded row, and fill the identity and sampling fields for eligible pairs. Leave all TOR result, QA and laboratory-source fields blank. Lock that complete pairing table:

```bash
uv run --locked --no-sync python research/ftir_hips_chem/workflows/freeze_addis_validation.py lock-pairs --pairs /path/to/pairing_before_TOR.csv
```

Only after the lock exists, add thermal results and laboratory-source references to an otherwise unchanged copy. The scoring command rejects changes to the locked eligibility or pairing metadata. It validates matching frozen PTFE IDs, unique quartz IDs, explicit sampling intervals and time zones, positive volumes, documented identity and sampling equivalence, TOR protocol, QA, and finite nonnegative EC, uncertainty and MDL. A missing or materially changed original PTFE volume needs a documented correction reason before locking. It refuses below-MDL results until a handling rule is declared. Submitted evidence flags still require human review against the original records; the command cannot certify that a laboratory document is authentic.

```bash
uv run --locked --no-sync python research/ftir_hips_chem/workflows/freeze_addis_validation.py check-pairs --pairs /path/to/pairing_with_TOR.csv
```

If valid labels arrive, this writes a pair-level file and AIRSpec/VIBES RMSE, MAE, bias, predictive R² and a paired 95% date-block bootstrap interval for VIBES minus AIRSpec. The primary comparison is **concentration in µg/m³**: TOR EC mass divided by the quartz sample volume versus frozen FTIR EC mass divided by the paired PTFE volume. Raw masses and both volumes remain in the output. Comparing the two raw filter masses directly would be invalid if the volumes differ. The fixed bootstrap uses Monday–Sunday local-date blocks, 10,000 draws and seed 20260922. Relative improvement requires the entire paired interval to exclude zero and at least the proposed 36-pair seasonal coverage (13 Dry, 12 Belg, 11 Kiremt); otherwise it is exploratory or inconclusive. No absolute-accuracy pass/fail is issued because that limit still needs scientific agreement before unblinding. The lock has **not** been created for real samples, because authoritative identities and independent TOR results are unavailable.

This executable scorer covers the two frozen full-pool AIRSpec/VIBES methods. The broader study contract also asks for comparison with the historical deployed product; that comparison is not implemented here because the staged deployed column covers only 189 of the 253 filters and its provenance/eligible overlap need a separate audit. It must not be silently treated as an independent thermal reference.

This freeze covers the **existing 253 filters only**. New campaign filters require processing, identity confirmation and their own prediction hash before their thermal outcomes are inspected. For the current evidence gap and unsent request, see the [readiness report](../research/ftir_hips_chem/output/tables/addis_validation_readiness/report.md).
