# Open cleanup items

This is the current action list. Historical investigations, measurements, and
resolved or rejected proposals are retained in
[`cleanup-history.md`](cleanup-history.md); the plotting census and measurement
details are in [`plot-taxonomy.md`](plot-taxonomy.md).
Research questions (as opposed to code decisions) are in
[`research-questions-2026-10.md`](research-questions-2026-10.md); sampled volumes and the
mass-per-filter view in [`mass-per-filter.md`](mass-per-filter.md).

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

### Establish the provenance of `EC_ftir` — substantially resolved

**Resolved (2026-08-12): `ChemSpec_EC_PM2.5` is not an independent measurement —
it is `EC_ftir`.** The r² = 0.99992 recorded here for the four-sites data
reproduces at ETAD, where the two columns match at r² = 0.999693 over 175
base-joined filters, with ratio median 1.0000 (IQR 0.999–1.001) and median
absolute difference 0.0030 µg/m³ — exactly the half-width of 2-decimal rounding.
They are the same product, routed through SPARTAN's speciation table, and the
"neither copies nor simple arithmetic restatements" finding above is explained
by that rounding rather than by measurement independence. See
`research/ftir_ec_phase3/ftir_25_intercept_invariant.md`.

**Consequence, which is what this item was really asking for:** the 18
crossplotting files are **calibration diagnostics, not independent validation** —
adopt that conservative reading permanently, not "until then". A second
consequence for phase 3: neither ChemSpec column can serve as an EC reference —
`ChemSpec_BC_PM2.5` is Fabs / 10 rounded (x-circular) and `ChemSpec_EC_PM2.5` is
FTIR-derived (y-circular).

**Still open, and now a smaller question:** whether these filters were in the
calibration training set. The high r² is no longer evidence either way — it is
column identity, not agreement between two analyses — so it neither implicates
nor exonerates the training set. Answering it still requires asking whoever
produced `Four_Sites_FTIR_data.v2.csv`, but the labelling decision no longer
waits on the answer. See
[the full provenance investigation](plot-taxonomy.md#findings).

### Confirm the semantics of the HIPS uncertainties (they exist)

**Corrected (2026-08-12): the premise that HIPS has no uncertainties was wrong.**
`HIPS_Uncertainty` and `HIPS_MDL` are their own **parameter rows**, not columns —
both are populated **190/190 at ETAD** (median **2.9075 Mm⁻¹** and 1.5534). What
made them look absent is that the `Uncertainty` and `MDL` *columns* on
`HIPS_Fabs` rows are empty; the values sit in sibling rows and are missed by any
probe that reads the columns.

So the data-driven Deming lambda is available now: sigma_x = 0.308 µg/m³,
sigma_y ≈ 0.531, giving **lambda\* ≈ 2.96, not 1.0**. Assuming lambda = 1
overstates the AIRSpec errors-in-variables intercept correction by ~55%
(−2.66 against −2.09), so existing lambda = 1 results are not just imprecise but
biased in a known direction.

What remains is a question for SPARTAN, not a data pull: confirm what
`HIPS_Uncertainty` and `HIPS_MDL` represent (repeat-measurement precision,
propagated calibration error, or a reporting floor) before passing them as
`sigma_x`/`sigma_y` to `calculate_regression_stats`, and then re-run the Deming
sweep at lambda\* — including the slope-crosses-1 case that lambda = 1 produces
for the AIRSpec branch at MAC 10. See
[the estimator census and MDL analysis](plot-taxonomy.md#findings) and
`research/ftir_ec_phase3/INTERCEPT_ATTACK_PLAN.md` item 3.

**Update (2026-08-23):** the uncertainties are populated at all four sites
(ETAD 190, CHTS 163, INDH 63, USPA 130) and per-filter weighted York/EIV
fits using them now exist — `research/ftir_ec_phase3/scripts/york_cross_site.py`,
results in `research/ftir_ec_phase3/OFFSET_ADJUDICATION_2026-08-23.md`.
Remaining here: the semantics confirmation from SPARTAN above, and porting
the York estimator into `calculate_regression_stats` / the explorer readout
so the app's Deming rows stop assuming a pooled lambda.

### Confirm CODC's station identity before using its calendar

`spartan_map.json` names CODC "Palmira, Colombia" but places it at 4.6369 N, 74.0808 W,
which is Bogotá. The two cities have different season calendars (the gallery offers both,
Bogotá by default). Someone with access to the SPARTAN Site Details Sheet must confirm the
code, coordinates, sampling dates and any relocation, and then keep only the right calendar.
CODC has no filters in the current export, so nothing is affected yet. See
[site-seasonality-network.md](site-seasonality-network.md#sites-with-two-calendars).

### Decide whether season-specific Addis calibrations move to `belg_feb`

The gallery and new reporting use the Ethiopian Meteorological Institute calendar
(`belg_feb`), while every frozen calibration result is on `dry_feb`. Moving February
changes Figure 9's Belg calibration from slope 0.32 to 0.67, and it removes the within-Belg
organic-signal correlation. Pooled calibrations are unaffected. Decide which calendar the
AAAR poster and paper quote for season-specific results, and say so in Methods. See
[site-seasonality.md](site-seasonality.md#addis-february-sensitivity-28-oct-2026).

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

### Export ftir_36's per-filter di-PLS predictions, for its bootstrap intervals at any MAC

The gallery's MAC setting (Filters → units) moves RMSE and mean residual against Fabs/MAC as well
as slopes. Each evaluation set's HIPS BC mean and variance are solved in the browser from the
saved fits (`gallery/app/src/lib/macMath.ts`, checked against the explorer's MAC 6 and 17 readouts
in `lib/mac.test.mjs`). The blind-half splits use their RMSEs saved at three MACs. ftir_39's
locked holdouts use each filter's Fabs, rebuilt from the raw HIPS on Drive by
`gallery/data/export_holdout_fabs.py` with `reconstruct_hips_fabs.py`'s line schedule; the export
refuses to write unless residual + Fabs/10 refits ftir_36's six ordinary-PLS lines. The one thing
still at MAC 10 is ftir_36's paired-bootstrap interval on the di-PLS RMSE change: it resamples each
filter's ordinary-PLS and di-PLS errors, and no export carries the di-PLS predictions per filter.
Have `ftir_36_domain_invariant_pls` write them (`locked_target_predictions.csv`: config, target,
variant, ExternalFilterId, prediction) and the gallery can bootstrap at any MAC.

### Join the last per-filter exports to Filters → excluded

Every gallery chart with per-filter data follows the one exclusion setting (docs/gallery-chart-standards.md,
"Excluded filters"). Three exports cannot yet be joined, so their charts keep a fixed-set note:
- The baseline research pages draw IMPROVE `sample_id`s, which no gallery filter row carries; the IMPROVE
  network has no exclusion list in the gallery to follow.
- `BaselineResearchDiagnostics` lists the 253 frozen Addis spectra as saved summaries only.
- `export_improve_envelope`'s exclusion list keeps the PM10 filter ETAD-0248 (which `filters.json` excludes)
  and drops INDH-0172, so its saved Addis windows match neither Hide nor Include
  (`sections/research/common/envelope.test.mjs` documents it). Re-export with `filters.json`'s exclusions,
  or have the export write the list it used.

## Explicitly declined—not open

### Do not merge `src/analysis/quality` and `src/data/qc`

The structural merge was investigated and **declined, not deferred**. Across
5,660 LOC, the packages share only two public-name collisions out of 23 and 77
public names; most functionality differs, neither stack has an active consumer,
and merging would churn unused code for negligible consolidation. The real
quality bugs found during the audit were fixed. Revisit only if active consumers
or a materially different API goal emerge. See
[the measured final verdict](cleanup-history.md#structural-cleanup-2026-07-28).
