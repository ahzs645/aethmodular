# Open cleanup items

This is the current action list. Historical investigations, measurements, and
resolved or rejected proposals are retained in
[`cleanup-history.md`](cleanup-history.md); the plotting census and measurement
details are in [`plot-taxonomy.md`](plot-taxonomy.md).

## Decisions needed

### Choose the canonical IMPROVE deposit area

Six notebooks use 3.5 cm² and three use 3.53 cm². The current default is 3.5
cm² because that is what produced the published envelope figures, but the
difference creates a systematic 0.86% offset in ETAD-vs-IMPROVE µg/cm²
comparisons. A domain owner must decide whether the canonical
`IMPROVE_DEPOSIT_AREA_CM2` is 3.5 or 3.53 cm², and whether any affected outputs
must be regenerated. This unblocks consistent mass-per-area comparisons. See
[the original constants audit](cleanup-history.md#other-fixes) and the separate
3.5 cm² FED Module A sweep documented in
[the resolved IMPROVE loader work](cleanup-history.md#resolved-2026-07-27).

### Establish the provenance of `EC_ftir`

`EC_ftir` and `ChemSpec_EC_PM2.5` have r² = 0.99992, which is implausibly high
for independent FTIR and thermal-optical measurements. The checkout establishes
that the columns are neither copies nor simple arithmetic restatements, but
`EC_ftir` arrives already computed in `Four_Sites_FTIR_data.v2.csv`. Ask whoever
produced that CSV whether these filters were in the calibration training set.
The answer determines whether 18 crossplotting files show calibration fit or
independent validation; until then, label those panels as calibration
diagnostics. See [the full provenance investigation](plot-taxonomy.md#findings).

### Obtain HIPS Fabs uncertainties

RESOLVED 2026-08-23: `HIPS_Uncertainty` and `HIPS_MDL` are populated in
`unified_filter_dataset.pkl` at ETAD (190), CHTS (163), INDH (63) and USPA
(130), and per-filter weighted York/EIV fits using them now exist —
`research/ftir_ec_phase3/scripts/york_cross_site.py`, results in
`research/ftir_ec_phase3/OFFSET_ADJUDICATION_2026-08-23.md`. Remaining:
port the York estimator into `calculate_regression_stats` / the explorer
readout so the app's Deming rows stop assuming a pooled lambda.

## Unfinished migrations

### Migrate stored-output notebooks to the canonical site palette

Red means Beijing in `config.SITES`; Delhi is blue, JPL green, and Addis orange.
Seven stored-output notebooks still carry a rival palette:
`Task_Analysis_Notebook`, `primary_tasks_notebook`, `hips_offset_narrative`,
`improve_hips_offset_narrative`, `anne_spartan_improve_ec_mass_fabs`,
`improve_spartan_may_full_analysis`, and generated
`spartan_ec_2026_06_16/04_new_plots`. Change each source only when it is re-run,
so code does not contradict stored figures. This completes palette consistency
without desynchronizing notebook outputs. See
[the migration record](cleanup-history.md#historical-follow-ups).

### Finish the remaining helper-migration families

The shared helpers exist, but inline copies remain in three families:

- `regression_stats` / `calculate_regression_stats` across
  `ftir_hips_chem`, `improve_hips_offset`, and `spartan`;
- season helpers in absorption and meteorology notebooks; and
- `find_repo_root` variants across the notebook estate.

Migrate generated families through their `_build_*.py` sources and hand-authored
notebooks by editing and re-running one family at a time. Confirm outputs are
unchanged before removing inline definitions. This unblocks one canonical path
for regression, season, and root-discovery behavior. See the
[import replacements and migration recipe](library-usage.md#scripts--what-to-import-instead-of-redefining)
and [historical inventory](cleanup-history.md#current-automated-cleanup-inventory).

## Explicitly declined—not open

### Do not merge `src/analysis/quality` and `src/data/qc`

The structural merge was investigated and **declined, not deferred**. Across
5,660 LOC, the packages share only two public-name collisions out of 23 and 77
public names; most functionality differs, neither stack has an active consumer,
and merging would churn unused code for negligible consolidation. The real
quality bugs found during the audit were fixed. Revisit only if active consumers
or a materially different API goal emerge. See
[the measured final verdict](cleanup-history.md#structural-cleanup-2026-07-28).
