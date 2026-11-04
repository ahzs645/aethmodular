# Graph comparison and remaining gaps — 3 November 2026

> Implementation follow-through: [graph-gap-implementation-2026-11-03.md](graph-gap-implementation-2026-11-03.md) records the new shared views, export/save steps, verification and remaining source-data requirements. The comparison below is the earlier review snapshot.

The gallery covers most of the recurring graph families. The largest remaining gaps are **specific diagnostics and the saved numerical inputs behind them**, plus the older repository's **ONA/CMA/DEMA processing comparison**. A second issue is discoverability: a graph can exist locally but be absent from a public build or its navigation.

This review assumes the question is: **which notebook analyses are missing or incomplete in the gallery?** It also flags differences that prevent apparently equivalent graphs from agreeing.

## Scope and limits

Reviewed the local `aethmodular` checkout at `7958498`, the separate `aethdata-analysis` repository, and the two notebook exports in `aethmodular-gallery`. The source revision agrees with the gallery README's latest successful publication, but the currently deployed website was not fetched or rendered.

Every notebook was scanned for plotting code, saved image outputs, errors and gallery name references. Existing detailed coverage audits were reconciled with current chart components and exported tables. The priority findings below were checked against current source; this is **not a new visual, numerical, cell-by-cell verification of every rendered figure**. No notebooks were executed, fitted, changed or republished.

| Repository | Notebook files | Outside archive/scratch/attic folders | Detected plotting-code cells | Embedded PNG outputs |
|---|---:|---:|---:|---:|
| `aethmodular` | 224 | 193 | 884 | 1,116 |
| `aethdata-analysis` | 26 | 12 | 181 | 408 |
| `aethmodular-gallery` | 2 | 2 | 4 | 2 |
| **Total** | **252** | **207** | **1,069** | **1,526** |

These are inventory counts, **not counts of unique graphs**. A code cell may define a plotting function without calling it, call a helper, or render many figures in a loop. Copies and executed exports remain in the counts; the inventory identifies notebooks with identical code. Detection uses explicit Matplotlib/Seaborn primitives plus the project's standard plotting helpers; indirect calls and other visualization libraries can be missed. All 252 files parsed successfully.

The [30 October audit](notebook-gallery-audit-2026-10-30.md) remains useful for its detailed research-notebook figure tables. Its headline **196 missing / 169 partial** is a historical snapshot, not a current remaining-work count: several large groups were implemented later that day. The [27 October coverage document](notebook-gallery-coverage.md) is older still.

## What is already covered

“Covered” here means an implementation exists for the analytical view. It does not assert that its saved data, cohort, estimator or numbers exactly reproduce every notebook variant.

| Notebook graph family | Current gallery equivalent | Remaining qualification |
|---|---|---|
| Four-site HIPS/FTIR/aethalometer crossplots, iron colouring and method comparisons | Correlation; Aethalometer → Against the filters | Public builds do not expose all aethalometer fields in other tabs. |
| Iron/EC cutoff splits and residuals | Correlation → split fits | Do not count the older audit's generic split-panel gap as still open. |
| Histograms, boxplots, ECDFs, seasonal and weekday views | Distribution; Aethalometer → Overview | The same chart type can use different date spans and eligible pairs. |
| Flow fixes, raw/DEMA smoothing, rolling medians, channel/attenuation QC | Aethalometer → Data quality | This does not include the complete ONA/CMA/DEMA experiment below. |
| Delta-C, wavelength CV, representative days, normalized hour × wavelength heatmap, cross-site overlay and summary | Aethalometer → Diurnal | Present in `SpectralCharts.tsx` and mounted by `pages/Diurnal.tsx`; these were originally missing in the October audit. |
| AAE, source classes, MAC cross-section sensitivity | Aethalometer → Sources | Absorption-based AAE differs from legacy BCc-ratio AAE. |
| Four MAC estimators: mean ratios, ratio of means, OLS with intercept, through-origin OLS | Aethalometer → Against the filters → Four MAC methods | Implemented in `filters/ConstraintCheck.tsx` and `filters/mac.ts`; these are deliberately distinct estimators. |
| PMF coverage, dominance histogram and seasonal residual-versus-biomass panels | PMF sources | Present in `PmfCoverage`, `DominanceHistogram` and `ResidualVsBiomass`. |
| Meteorology, AERONET, surface-versus-column comparisons | Weather; AERONET; Research diagnostics | Same-day, sampling-window and hourly analyses are not interchangeable. |
| Filter-only proportionality and temporal-transfer limits | Research → Filter-only transfer limits | The previously missing 18-figure group now has a live implementation. |
| HIPS operating-envelope residuals and deployed blank-line views | Research → HIPS operating envelope; blank-line diagnostics | The full five-target York ladder is still absent. |
| Decision workbook, specialist tests, SPARTAN flag and k-sweep-versus-IMPROVE | Machine learning → Decision workbook; k sweep vs IMPROVE | A separate mean-over-folds curve remains unsaved. |
| HIPS-target PLS: CV curves, Addis predictions and VIP | Machine learning → Model form and VIP | The export contains 239 Addis prediction rows and a 2,722-point HIPS VIP profile. EC-versus-OC VIP profiles are a separate gap. |
| Charcoal similarity by season, lot and loading | Research → Charcoal similarity; Machine learning → Charcoal archive | Broad coverage exists; several per-filter and unsupervised diagnostic outputs remain absent. |
| Adama TOR comparisons, lot-251 smoke calibration and full-fit residual diagnostics | Research → Ethiopia calibrations | Does not fill the missing repeated-CV curves or cleaned-refit prediction exports. |

Current implementation evidence: [Diurnal](../gallery/app/src/sections/instruments/pages/Diurnal.tsx), [PMF diagnostics](../gallery/app/src/charts/pmf/PmfDiagnostics.tsx), [split fits](../gallery/app/src/charts/correlation/SplitFit.tsx), [four MAC methods](../gallery/app/src/sections/instruments/filters/ConstraintCheck.tsx), [filter-only views](../gallery/app/src/sections/research/pages/filterOnly), [HIPS-target plots](../gallery/app/src/sections/ml/pages/ModelForm.tsx), [Ethiopia comparisons](../gallery/app/src/sections/research/pages/EthiopiaCalibrations.tsx).

## Remaining gaps, in recommended order

Priority reflects usefulness for explaining the existing results, not a measured implementation effort.

| Priority | Missing or incomplete comparison | Source notebooks | What completes it |
|---|---|---|---|
| 1 | **Full ONA/CMA/DEMA comparison**: adaptive window-size ECDF, signal-preservation lag/cross-correlation, daily differences, processing-method Bland–Altman and source-fraction changes | Older `Post Processing/CMA DEMA ONA Comparison Polars.ipynb`, especially cells 35, 38, 57, 63 and 65; `AETH-HIPS-FTIR Comparison/smoothening test.ipynb` | Export aligned raw/ONA/CMA/DEMA time series and processing settings. Existing DEMA/rolling-median charts are only partial coverage. No ONA/CMA experiment implementation was found in the checked gallery source. |
| 2 | **Full cross-validation curves by scheme and fold count** | `ftir_20`, `ftir_23`, `ftir_32`; June variants 06/09; SPARTAN EC 05/07 | Save RMSECV/RMSEP by k, scheme, fold count, variant and repeat/seed. The gallery has picked k and verdict tables, but they do not replace these curves. Some arrays were only plotted, so adding a save step must precede a rerun. |
| 3 | **Per-filter transfer and applicability diagnostics**: residual versus D²/Q, protocol-specific prediction clouds, training-choice/model-form/target comparisons | `ftir_11`, `ftir_15`, `ftir_18`, `ftir_21`, `ftir_22`, `ftir_42`, `ftir_48`, `ftir_50`, `ftir_54`; June notebook 12 | Export FilterId-keyed predictions, D²/Q, target and protocol labels, observed Fabs, dates and membership. Current aggregate scores and correlations do not reconstruct these scatterplots or support new exclusions. |
| 4 | **Exact legacy exclusion/reproduction path** | `Multi_Site_Analysis_Fixed`, `Multi_Site_Analysis_FollowUp`, `primary_tasks_notebook` | Export `is_outlier` and `outlier_reason` from the canonical threshold-flag API, and retain the legacy matched values if reproducing old numbers is the goal. `export_data.py` currently exports `is_excluded` as `excluded`, not the separate threshold flags. The gallery already has some minute-level alternative-window comparisons; this gap is faithful per-filter legacy reproduction. |
| 5 | **Charcoal unsupervised and reference-envelope views** | `char_02`, `char_04`, `char_11`, `char_12`; per-feedstock/class panels in `char_05`, `char_07`–`char_09` | Save PCA scores with labels, per-filter Mahalanobis distances, Ward linkage, AIRSpec fit traces, recovery pairs, per-feedstock/class scores, and raw/derivative mean/p05/p95 envelopes plus escape scores. Existing similarity scores and group summaries cover only part of these views. |
| 6 | **Distributions behind uncertainty summaries** | Bootstrap draws in `ftir_15`/`ftir_22`; null draws in `ftir_43`/`ftir_46`; seasonal analog selection in `ann_weekly_20260910_diagnostics` | Save draws and analog membership/selection frequencies. A confidence interval already drawn is not a missing chart; the missing item is the underlying distribution or stability view. |
| 7 | **EC-versus-OC VIP spectral profiles** | `ftir_07_current_pls_vip_diagnostics` | Supply `current_ec_oc_vip_profiles.csv`. Current export has overlap/band summaries and the HIPS VIP profile, but no `vip_ec_oc` table. Do not classify all VIP plots as absent. |
| 8 | **Full York ladder under four blank constructions** | `ftir_47` [14], `ftir_53` | Export the five-target ladder, or supply the per-site uncertainty models, whole-batch lot-pooled blank fits and Bishoftu raw T1/R1. `BlankLines.tsx` explicitly explains why this view is not redrawn. The narrower Calibration → HIPS York fits already exist. |
| 9 | **Observed missingness stratified by hour/year/season and filter quality** | Older `Raw Data Charateristic/Data Completness.ipynb`, cells 14, 16, 22, 24–29 and 35–36 | Export the observed minute-coverage summaries and filter-day quality memberships. The gallery has daily gap calendars and simulated missingness bias benchmarks; those answer different questions from the exact observed partial-day outage patterns. |
| 10 | **Decision workbook §2 mean-over-folds RMSE curve** | `decision_workbook_20261022` | Save the fold-mean curve alongside the pooled-RMSE curve. The gallery's current map says this was PNG-only; pooled RMSE is already covered. |

The precise export/table names and the existing rerun guidance are in [export_requirements.csv](../research/ftir_hips_chem/output/tables/graph_review_20261103/export_requirements.csv). It contains **41 exporter-declared requirements** from the current ML and Research data files. This is **not 41 missing figures**, not a deduplicated task estimate, and not an exhaustive count of every gap above. One export can unlock many figures; an alternate export may supply an equivalent field. None of those declared table keys is present in its own export's `tables` object.

The gallery's ML map separately has 32 curated result groups: 9 in ML, 16 elsewhere, 4 partial and 3 missing. Its group-level classifications and 30 October timestamp are preserved in [ml_coverage_map.csv](../research/ftir_hips_chem/output/tables/graph_review_20261103/ml_coverage_map.csv). “Elsewhere” establishes a location, not current public visibility or complete data access.

## Older repository: all 12 non-archive notebooks

The older notebooks contain many predecessors of newer analyses. Their absence by filename is not proof that their analysis is missing.

| Older notebook | Comparison with current gallery |
|---|---|
| `Post Processing/CMA DEMA ONA Comparison Polars` | The clearest distinct gap: full processing-algorithm experiment, adaptive windows and lag preservation. |
| `AETH-HIPS-FTIR Comparison/smoothening test` | Partial: raw/smoothed comparisons exist, but not all processing methods and their method-specific seasonal/filter comparisons. |
| `Raw Data Charateristic/Data Completness` | Partial: daily coverage and missingness sensitivity exist; exact observed stratified partial-day and filter-overlap quality views are not reproduced. |
| `Datasahre/Untitled-1` | Overlaps coverage, diurnal and raw/cleaned views. No saved PNG output in this notebook, so only source intent was reviewable. |
| `EC HIPS/Refactored analysis with Finalized R Review` | Four-MAC methods, season comparisons and residual diagnostics broadly overlap existing views. Its particular combined-score/physical-constraint rankings are not equivalent to independent validation. |
| `EC HIPS/Elemental Cmparison ECFTIR` | Generic chemistry/MAC colouring and residual screens overlap. Its chemistry-based random-forest/model-consensus panels are distinct from the gallery's spectra-model VIP/robustness studies; exact parity is not established. Review the target and validation before porting. |
| `EC HIPS/Refined Elemental Analysis` | Similar chemistry/MAC family, largely reorganized as individual figures. Do not treat every reformatted panel as new coverage. |
| `Sectioned/Sectioned Analysis` | Broad overlap in temporal, diurnal, seasonal, source and meteorology views. Data-derived seasonal transitions and exact threshold sweeps are distinct from a site-calendar toggle; these need a separate parity check if wanted. |
| `Meeting Based/jun 10/Jun 10` | Iron colouring, MAC methods and season plots overlap. Manganese literature context and placeholder aethalometer-integration panels are not evidence of an unimplemented measured result. |
| `Meeting Based/jun 10/Untitled-1` | Broad overlap in HIPS/aethalometer comparisons, correlations, wavelengths, seasons and agreement diagnostics. Exact cohort, units and legacy timing still need reconciliation for numerical matching. |
| `Meeting Based/jun 10/additional` | Additional cross-method/source-coloured plots overlap current families. This file has two saved errors and no PNG outputs; its figures were not verified as successfully rendered. |
| `Misc/wavelength overlap` | Explanatory instrument-range diagram, not a new quantitative analysis. Optional documentation addition rather than a scientific coverage priority. |

Source files are in the separate [aethdata-analysis repository](../../aethdata-analysis). The 14 notebooks in its archive folders are retained in the inventory, but are not counted as 14 additional active analyses.

## Why apparently matching plots can disagree

1. **Sampling and aggregation:** legacy 9 AM/day matching and the gallery's overlap-weighted filter sampling windows can use different measurements. Minute, daily, filter-level and site-season aggregates also have different denominators. Match FilterIds and coverage before comparing fit statistics.
2. **AAE basis and wavelength:** absorption-ratio AAE and BCc-ratio AAE are different; verify the standard sign and MA350 wavelengths. A visually matching source-class plot can otherwise tell a different story.
3. **Seasons:** Addis `dry_feb` and `belg_feb` place February differently. Shared Ethiopian month bins are not local Beijing/Delhi/JPL seasons. The calendar changes fitted subsets as well as labels.
4. **Units and MAC:** Fabs in Mm⁻¹, Fabs/MAC in µg/m³ and filter mass in µg are different quantities. Compare at the same MAC, and distinguish static fitted results at MAC 10 from browser-recomputed results. Published BC-at-MAC-10 product checks should retain their defined MAC.
5. **Fits and exclusions:** OLS, Deming and York slopes are not interchangeable. Preserve each graph's intended estimator and uncertainty assumptions. Manual exclusions, threshold flags and pairwise missing-data masks are separate selections.

These conventions are documented in [AGENTS.md](../AGENTS.md), [site-seasonality.md](site-seasonality.md) and the [current research summary](current-research-summary.md).

## Scientific gaps a new chart would not resolve

- **Independent EC validation at the four focus SPARTAN sites remains absent.** Public ChemSpec EC/OC are FTIR-derived products; near-perfect agreement with in-house FTIR is a duplicate-product check. HIPS supplies a separate optical readout, not independent thermal-optical EC truth. IMPROVE thermal EC and the separate five-filter Adama campaign are different populations.
- **Do not copy intercept-as-source-fraction graphics without new evidence.** The older elemental notebook computes a “brown carbon fraction” from the fitted intercept divided by mean Fabs. That construction does not identify an absorber or establish a physical source fraction. Likewise, post hoc feature importance does not establish chemical causality.
- **Do not label a chart missing simply because its PNG is absent.** In the current scan, 84 `aethmodular`, 3 older-repository and 1 gallery-export notebooks contain detected plotting code but no embedded PNGs. Some may intentionally omit outputs or use other formats. Conversely, saved outputs do not establish that a notebook still runs: 13 files contain saved error outputs.

## Recommended next work

First, decide the intended comparison surface: full local gallery or public build. **Do not publish private sections just to close a coverage checkbox.** The [section manifest](../gallery/app/src/sections/sections.json) marks Research and ML private and Instruments unlisted; therefore some apparent public gaps are intentional build behaviour.

Then complete one consistent processing-comparison page from the older ONA/CMA/DEMA notebook, with all methods evaluated on the same timestamps and filter pairs. In parallel as a workstream, prioritize saving CV curves and FilterId-keyed predictions from the existing model workflows before designing more figures. Preserve preprocessing, folds, membership, uncertainty assumptions, units and provenance with each export.

After those inputs exist, fill the residual/domain, charcoal and uncertainty-distribution views. For any exact legacy reproduction, make its matching/calendar/estimator explicit rather than silently replacing the gallery's current conventions.

## Review files

- [Notebook inventory](../research/ftir_hips_chem/output/tables/graph_review_20261103/notebook_inventory.csv): all 252 files, saved-output/error counts, name references and exact-code duplicates.
- [Plot-code-cell inventory](../research/ftir_hips_chem/output/tables/graph_review_20261103/plot_cell_inventory.csv): all 1,069 detected cells, headings, titles, plotting primitives/helpers and zero-based cell indices. Coverage is deliberately not inferred from a name match.
- [Export requirements](../research/ftir_hips_chem/output/tables/graph_review_20261103/export_requirements.csv): current declared missing tables, expected paths, prerequisites and suggested commands, **not executed**.
- [Existing ML coverage map](../research/ftir_hips_chem/output/tables/graph_review_20261103/ml_coverage_map.csv): curated group-level statuses and locations, with its timestamp in the [inventory summary](../research/ftir_hips_chem/output/tables/graph_review_20261103/inventory_summary.json).

The inventory was generated read-only from notebook JSON and gallery sources; no dependencies, source datasets or model state were changed. Execution, rendered layout and numerical fidelity remain unverified by this review.
