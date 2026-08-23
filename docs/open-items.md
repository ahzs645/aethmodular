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
