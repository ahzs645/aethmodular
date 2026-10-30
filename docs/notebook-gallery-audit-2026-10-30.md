# Notebook audit, 30 Oct 2026: every figure, where the gallery has it, and whether it can be computed live

Six read-only agents went through the 164 active notebooks under `research/` (archive and scratch left out),
one slice each, and checked every figure against the gallery's code and data as of 30 Oct 2026. This updates
[notebook-gallery-coverage.md](notebook-gallery-coverage.md) (27 Oct), which several slices found out of date.

## Classes

- **D** — dynamic now: the gallery computes it in the browser from data it already loads (it follows the
  filters, hand exclusions and the MAC).
- **D+** — would be dynamic with a small export: the slice names the per-filter columns and where they exist.
- **S-ML** — static because it needs a model fitted on spectra (PLS, component selection, cross-validation,
  di-PLS, PCA or clustering, residual learners, bootstraps of refits): precomputed. These belong in the
  Machine learning section (`#tab=ml`), whose Overview page maps every one of them to where it lives.
- **S** — static for another reason (IMPROVE network data too large to ship, external archives, minute data).

## Totals

Rows are figures, or figure groups where a slice grouped near-duplicate notebooks (A), so totals are approximate.

| Slice | Notebooks | Rows | In gallery | Partial | Missing | n/a | D | D+ | S-ML | S |
|---|---|---|---|---|---|---|---|---|---|---|
| A · ftir_hips_chem: aethalometer, HIPS, sources | 35 | 103 | 75 | 20 | 5 | 3 | 99 | 1 | 0 | 0 |
| B · ftir_hips_chem: FTIR, VIBES, meetings, decisions | 26 | 135 | 55 | 37 | 41 | 2 | 33 | 38 | 46 | 18 |
| C · ftir_ec_phase3 ftir_11–34 | 21 | 72 | 28 | 27 | 16 | 1 | 18 | 4 | 32 | 17 |
| D · ftir_ec_phase3 ftir_35–55 | 21 | 76 | 25 | 19 | 31 | 1 | 22 | 8 | 24 | 21 |
| E · ftir_ec_calibration_2026_06_25, improve_hips_offset | 29 | 98 | 32 | 41 | 22 | 3 | 22 | 8 | 14 | 51 |
| F · ftir_etad_char, spartan_ec, spartan, July07, Deming | 32 | 149 | 43 | 25 | 81 | 0 | 58 | 23 | 28 | 40 |
| **Total** | **164** | **633** | **258** | **169** | **196** | **10** | **252** | **82** | **144** | **147** |

So about 40 % of the figures can be computed live today and another 13 % with a small export; about 23 % need
model fitting and stay static; the rest are static for size or source reasons.

## Built since the audit (30 Oct, later)

- **Machine learning section** (`#tab=ml`): Overview, Components and CV, Protocols and transfer, June 2026
  variants, Residual learners, What counts as EC, Model form and VIP.
- **Aethalometer → Diurnal**: addis_05 / multisite's Delta-C and wavelength CV, representative days, the normalised
  hour × wavelength heatmap, the cross-site overlay and summary table; ftir_28 [15]'s channel health.
- **Research → HIPS operating envelope**: addis_06 c14 / c22 residual views, hips_offset_narrative [10] / [23],
  warren [30]. **Blank lines per deployed line** (ftir_47 [8], [11]; the York ladder [14] still needs exports).
- **Research → The mass lead** (ftir_46 [13], [19a], [19b], live); ftir_16 [7] on MAC 6 vs 10; ftir_52 [19] on the
  network map; ftir_26 [29B] on IMPROVE HIPS at EC = 0.
- **Research → Filter-only transfer limits**: all 18 figures of filter_proportionality, filter_relationship_stability
  and filter_only_diagnostics 05, refitted live (export_research_filter_only.py, tracked release data).
- **Research → Ethiopia calibrations**: the lot-251 smoke tool at Addis (July07, spartan_ec 02), the Adama TOR set
  (June 02/03/08/15, spartan_ec 03/05/07), spartan_ec 08's residual diagnostics.
- **Chart tabs**: Correlation → split at a cutoff with the residual against a variable (the iron/EC analyses);
  Distribution → ECDF; groupings → iron split at each site's own median; PMF → coverage, dominance and residual vs
  biomass by season.

From the user's shared copy of their local outputs and the decision workbook's Drive folder (30 Oct, later still):

- **Machine learning → k sweep vs IMPROVE** (ksweep_20261014 [3], [5], [7], [9], [11]; all 590 displayed numbers
  match). The notebook's dashed TOR-vs-HIPS references were divided by mean TOR while its curves were divided by
  mean HIPS; the page draws them on the curves' scale by default.
- **Machine learning → Decision workbook** (§2/§7 RMSE over k, §3 failure groups, §4 specialists, §6 the flag on
  SPARTAN live at the reader's MAC, §7 option A, §8).
- **Machine learning → Model form and VIP**: ftir_08's VIP spectra, CV curves and Addis transfer crossplot.
- **Machine learning → Charcoal archive** and **Research → Charcoal similarity** (char_02/03/05–09/11; the
  per-filter scores live by season, lot and loading).

Still not exported anywhere: ftir_07's EC/OC VIP profile CSV, the charcoal PCA scores, per-filter Mahalanobis,
Ward linkage and AIRSpec traces (never saved), char_12's per-filter escape and envelopes, the decision workbook's
§2 mean-over-folds curve (PNG only), and the per-filter outputs each ML page lists under `missing`.

## Fixed while auditing

- Research → Addis vs IMPROVE loading: the fAbs / EC panel's reference line now follows the MAC set under
  Filters → units (it used the export's).
- Research → Ethiopia calibrations: the variants' HIPS fits table is notebook 08's (`_build_08_adama_hips_crossplots.py`
  writes `etad_newec_vs_hips_fits.csv`), not 11's.
- Research → 1617 band, dust panel: its partial r does not depend on the MAC (another MAC adds a multiple of HIPS
  BC to the residual, which the adjustment removes); only the unadjusted ρ is MAC 10. Slice D's flag of
  Generalise's protocol swing does not hold: it is a spread of Deming intercepts, which the MAC does not move
  (λ ∝ MAC²).

The slice reports follow as the agents wrote them (their own wording and conventions).


---

## Audit A: research/ftir_hips_chem notebooks (aethalometer, HIPS, sources, temporal/met) against the gallery

Checked on 30 Oct 2026 against `gallery/app/src` and `gallery/data` as they are now. Nothing in the repo was edited and no notebook was run.

## How to read this

- **Execution state.** Only nine of the 35 notebooks have saved image outputs: FlowFix (13), HIPS_vs_Aeth (10), Multi_Site_Analysis_Fixed (104), Multi_Site_Analysis_FollowUp (69), Raw_ATN (5), Task_Analysis (38), dominant_source_threshold (169), follow_up_analysis (13) and primary_tasks (25). The figures of the other 26 come from their code (titles, `savefig` names). Their PNGs and CSVs go to `output/`, which is git-ignored and missing from this checkout. **None of the tables these notebooks write is tracked.**
- **Where things live.** `Aethalometer →` is the Instruments section, `gallery/app/src/sections/instruments`. It is *unlisted*: a public build publishes it but leaves it out of the menu, and it opens from `#tab=aethalometer`. Its per-filter fields ("Aethalometer (sampling window)") join the pickers on every other tab only in a full or local build; a public build has them on the Aethalometer tab alone (`App.tsx:655`). `Research →` is `sections/research`. It is *private*: a public build leaves it out.
- **Class D for Partial and Missing rows** means the data the figure needs is already loaded in the browser, and only the chart component is missing. **D+** means a small export is needed.
- **MAC.** `HIPS BC` is a derived field, Fabs ÷ the reader's MAC (`lib/derived.ts`), and every Aethalometer and Research page here reads it or `meta.mac_value`. So every gallery version of a "HIPS BC" or "Fabs/MAC" figure **follows the MAC**. The "MAC sensitivity" in addis_01 means the MA350 cross-section σ, not the HIPS MAC; the gallery handles it in `CrossSectionSweep`.
- **Matching differs from the notebooks.** The notebooks join aethalometer days by a legacy 9 AM date match. The gallery uses the overlap-weighted mean over each filter's sampling window (`export_aethalometer.py`). It also takes AAE from absorption, which reads about 1 higher than the notebooks' BCc-ratio AAE. The Sources page explains both. So the gallery's numbers will not match the notebook PNGs exactly; see cheap win (a) in §2.

## 1. Figure table

Cell indices are notebook cell numbers. "MSF" is Multi_Site_Analysis_Fixed, "FU" is Multi_Site_Analysis_FollowUp, "a05" is addis_05 and "ms" is multisite_diurnal.

| Notebook | Figure (cell: what) | Status + where | Class | What it would take | ML | Notes |
|---|---|---|---|---|---|---|
| Analysis_Tasks_Jan2025 | c10: iron vs FTIR EC per site, mean Fe/EC | In: Correlation (Iron × EC (FTIR), per-site layout); Fe/EC via Ratio dialog | D | | N | Not executed |
| Analysis_Tasks_Jan2025 | c13: Delhi axis scaling before/after | n/a | n/a | | N | A plot-format fix |
| Analysis_Tasks_Jan2025 | c16: HIPS vs aeth, filters with all three measurements | In: Aethalometer → Against the filters (MethodAgreement; AgreementTimeline gives "all three" means) | D | | N | Follows MAC |
| Analysis_Tasks_Jan2025 | c18: all pairs × sites crossplots | In: Correlation with aethalometer fields (full build); MethodAgreement | D | | N | |
| Analysis_Tasks_Jan2025 | c20: slope / R² summary table (CSV) | Partial: each chart shows its own fit; no table across all pairs | D | A small table over METHOD_PAIRS × sites | N | |
| Analysis_Tasks_Jan2025 | c22: BC wavelength correlation matrix | In: Aethalometer → Data quality → attenuation (AtnCorrelation, BCc triangle) | D | | N | |
| ETAD_Factor_Analysis | c12: ETAD filters, all vs PMF-analysed | Partial: the "No PMF day" group exists; no coverage chart | D | Counts from filters.json + pmf.json | N | Not executed |
| ETAD_Factor_Analysis | c13: parameter coverage timeline, all vs PMF | Partial: Data availability has no PMF lane | D | A PMF lane in the availability heatmap | N | |
| ETAD_Factor_Analysis | c14: FTIR vs HIPS with PMF samples highlighted | In: Correlation, colour by PMF source ("No PMF day" grey) | D | | N | Follows MAC |
| ETAD_Factor_Analysis | c20–21: daily source fractions and µg/m³; normalised stack | In: PMF sources tab (`charts/pmf/SourceStack`) | D | | N | The PMF solution is external (`Filter Data/ETAD Factor Contributions .csv` → pmf.json) |
| ETAD_Factor_Analysis | c23: correlation matrix, PMF sources × filter parameters | In: Correlation → Correlogram with the PMF share fields (`withPmf`) | D | | N | |
| ETAD_Factor_Analysis | c24: each source vs each parameter | In: Correlation (PMF shares as axes) | D | | N | |
| ETAD_Factor_Analysis | c31–32: method pairs coloured by dominant source (K_F, GF) | In: Aethalometer → Sources (MethodBySource, GF / K_F split) | D | | N | |
| ETAD_Factor_Analysis | c36: progressive removal by biomass share; residual vs each biomass factor | In: Sources → ThresholdSweep, ResidualVsDominance | D | | N | |
| ETAD_Factor_Analysis | c37: absolute and % residual vs biomass % | In: ResidualVsDominance, AaeVsBurning; the % form via Research → What tracks the offset ("relative") | D | | N | |
| ETAD_Factor_Analysis | c38: residual vs biomass by Ethiopian season; FTIR vs aeth per season | Partial: per-season scatter layout is there; the residual chart has no season facet | D | A season facet on ResidualVsDominance | N | |
| Example_Modular_Analysis | c3, c13–36: framework demo (before/after exclusion, timeseries, crossplots, distributions, flow periods, all vs common samples) | In: the generic equivalents (include-excluded toggle, Evolution, Correlation, Distribution, Data quality → Flow) | D | | N | A demo with nothing notebook-specific |
| FlowFix_BeforeAfter_Analysis | c12: all aeth BC by flow period, per site (3) | In: Data quality → Flow (FlowTimeline, CoverageStrip) | D | | N | |
| FlowFix_BeforeAfter_Analysis | c14, c16: FTIR EC and HIPS vs aeth, before/after | In: Data quality → FlowFixCompare | D | | N | HIPS follows MAC |
| FlowFix_BeforeAfter_Analysis | c18–19: R² and slope before vs after (bars) | In: FlowFixCompare summary | D | | N | |
| FlowFix_BeforeAfter_Analysis | c26: wavelength correlation matrix, distribution and boxplot per period | Partial: FlowSummary gives UV/IR by period only | D | Daily bins carry `flow`; a matrix and box per period | N | |
| FlowFix_BeforeAfter_Analysis | c27: UV vs IR by period, UV/IR over time | In: FlowSummary, FlowTimeline | D | | N | |
| flow_fix_explorer | c9, c12, c15: stacked timeseries, rolling correlation, ratio timeseries | In: Data quality → AgreementOverTime | D | | N | Not executed |
| flow_fix_explorer | c17–25: date-range explorer, availability, recommended dates | In: FlowFixCompare (test dates), CoverageStrip | D | | N | |
| HIPS_Aeth_SmoothRaw_Analysis | c10: smooth/raw difference histogram | In: Data quality → smoothing (SmoothRaw) | D | | N | Not executed (repaired 2026-09-21) |
| HIPS_Aeth_SmoothRaw_Analysis | c12, c14: baseline and per-threshold crossplots | In: SmoothRaw cross (set-aside filters as grey crosses) | D | | N | Follows MAC |
| HIPS_Aeth_SmoothRaw_Analysis | c16–17: R² and slope vs threshold | In: SmoothRaw sweep | D | | N | |
| HIPS_vs_Aethalometer_Optical | c13: HIPS Fabs vs raw ATN (4 sites) | In: Against the filters → ChannelAgreement (ATN1 from `filters_extra`) | D | | N | |
| HIPS_vs_Aethalometer_Optical | c15: HIPS vs BCc (4) | In: MethodAgreement, ChannelAgreement | D | | N | |
| HIPS_vs_Aethalometer_Optical | c18, c21: ATN vs BCc R²; R² by wavelength | In: ChannelAgreement | D | | N | |
| Multi_Site_Analysis | c7–11: IR and multi-wavelength BC timeseries | In: Overview (SitesTimeline, DailyTimeline) | D | | N | Not executed |
| Multi_Site_Analysis | c13: BC boxplots by site and wavelength | In: SeasonalSpread / SeasonalSpreadBySite | D | | N | |
| Multi_Site_Analysis | c14–15: completeness | In: Overview → Completeness | D | | N | |
| Multi_Site_Analysis | c17, c19: aeth vs filter EC (timeseries; scatter EC ≥ 0.5) | In: DailyTimeline (filters marked), MethodAgreement (EC range in the Filters panel) | D | | N | |
| Multi_Site_Analysis | c21–24: summary table, multi-parameter correlation | In: SiteTable; Correlogram (full build) | D | | N | |
| Multi_Site_Analysis | c26: temperature vs BC | In: Correlation (weather × IR BCc), Weather → correlations | D | | N | |
| MSF / FU | MSF c8–13: IR BC timeseries; aeth vs FTIR EC | In: Overview; MethodAgreement | D | | N | Executed |
| MSF / FU | MSF c17, FU c8: outliers flagged by MANUAL_OUTLIERS thresholds, before/after | Partial: EXCLUDED_SAMPLES (Excluded chart, include-excluded toggle) are there; the threshold flags (`is_outlier`) are not exported | D+ | Export `is_outlier` / `outlier_reason` from `outliers.apply_threshold_flags` (inputs tracked: processed_sites pickles + `unified_filter_dataset.pkl`) | N | The rules use the 9 AM-matched `aeth_bc` in ng/m³ |
| MSF / FU | MSF c20–21, c39; FU c16: smooth/raw distribution, CDF, threshold crossplots | In: SmoothRaw | D | | N | |
| MSF / FU | MSF c25, c33; FU c8, c10: cross-comparisons coloured by iron and by Fe/EC | In: MethodAgreement (colour Iron / Fe/EC / Fe/BC) | D | | N | Follows MAC |
| MSF / FU | MSF c35, FU c12: multi-site comparison | In: MethodAgreement (sites compared) | D | | N | |
| MSF / FU | MSF c43, FU c20: flow-fix split | In: FlowFixCompare | D | | N | |
| MSF / FU | MSF c47, c49; FU c24, c26: UV vs IR; multi-wavelength timeseries | In: SpectralShape, Correlation scatter, DailyTimeline | D | | N | |
| MSF / FU | FU c29, c32: wavelength correlation and distribution; excluded dates marked | In: AtnCorrelation, SeasonalSpread, Overview → Excluded | D | | N | |
| Raw_ATN_Correlation_Analysis | c13, c15, c17: ATN and BCc matrices; cross-site panel; BCc − ATN difference | In: Data quality → AtnCorrelation | D | | N | |
| Task_Analysis_Notebook | c9: Beijing cross-method verification grid | In: Correlation / MethodAgreement for CHTS | D | | N | Executed |
| Task_Analysis_Notebook | c12–18: absolute iron vs Fe/EC colouring (HIPS–EC, matched, HIPS–aeth) | In: MethodAgreement colour toggle; Correlation colour-by | D | | N | Follows MAC |
| Task_Analysis_Notebook | c21–31: high-concentration (Addis, Delhi) vs low (Beijing, JPL) site groups | Partial: one site pick per view; no pooled site group | D | A categorical grouping on site | N | |
| Task_Analysis_Notebook | c34–38: Fe/EC threshold split (low / high / residual vs Fe/EC) by group | Partial: Ratio + cutoff grouping gives split fits; residual vs ratio only in Research → What tracks the offset (private, one site) | D | A generic "split at a cutoff + residual" panel | N | |
| Task_Analysis_Notebook | c44, c47: impact of the exclusions; JPL flow fix | In: include-excluded toggle, Excluded; FlowFixCompare | D | | N | |
| primary_tasks_notebook | c11: Fe/EC histogram and CDF, all sites | Partial: Distribution has a histogram and ridgeline, no ECDF | D | An ECDF mode in Distribution | N | Executed |
| primary_tasks_notebook | c14–17: Fe/EC threshold (low / high / residual) | Partial (as Task c34) | D | Same as above | N | |
| primary_tasks_notebook | c21–23: Fe/EC > 200/300 combined with aeth BC > 7.5 exclusion; continuous Fe/EC colour | Partial: colouring is there; the compound rule can only be applied by hand (brush / drop) | D | A rule-based exclusion (two conditions) | N | |
| primary_tasks_notebook | c24, c26–27, c40: EC vs aeth by site coloured Fe/EC; HIPS-based iron; all comparisons | In: MethodAgreement, IronTest, Correlation | D | | N | |
| primary_tasks_notebook | c32–34: JPL flow fix | In: FlowFixCompare | D | | N | |
| follow_up_analysis | c10–12: Delhi Green outlier (timeseries, distribution, before/after) | In: Overview DailyTimeline + Screen (mean + 3 SD), Excluded | D | | N | |
| follow_up_analysis | c16: Delhi axis fix | n/a | n/a | | N | |
| follow_up_analysis | c22–28: Fe/EC by site; iron vs EC, HIPS, BCc | In: Distribution (ratio field), Correlation, IronTest | D | | N | |
| follow_up_analysis | c24: Fe/EC median split + residual | Partial (as Task c34) | D | Same | N | |
| follow_up_analysis | c39: 4-panel summary (mean Fe/EC, R², slope, n by site) | Partial: fit tables per site in MethodAgreement; mean by site in Ranking; not one panel | D | | N | |
| addis_01 (+_daily) | c12: AAE distribution with source regions | In: Aethalometer → Sources → AaeDistribution | D | | N | AAE ≈ +1 vs the notebook (absorption basis), explained on the page |
| addis_01 (+_daily) | c14: "MAC sensitivity" (σ pairs) | In: CrossSectionSweep | D | | N | This is the MA350 σ, not the HIPS MAC |
| addis_01 (+_daily) | c16, c18: seasonal source shares; AAE timeseries | In: ClassShares, AaeTimeline | D | | N | The 60-s Jacros AAE: the hourly AAE is in `diurnal/aeth/ETAD.json` |
| addis_02_temporal_patterns | c9, c11: diurnal BC (and by season); week × hour heatmap | In: Overview hourly (PeakTiming, WeekHour); Diurnal ChannelCycle | D | | N | Not executed |
| addis_02_temporal_patterns | c13, c15, c17: seasonal box and timeseries; extreme events by hour and season; rate of change | In: SeasonalSpread, DailyTimeline, HighEpisodes, EpisodesBySite | D | | N | The episode rule is the gallery's own |
| addis_02_temporal_patterns_daily | c9–17: monthly, weekday, seasonal, extreme days, day-to-day change | In: MonthlyPattern, WeeklyPattern, SeasonalSpread, Episodes | D | | N | Every site, not only Addis |
| addis_02.5_temporal_patterns_filter | c9, c11, c13: monthly, weekday and seasonal for the 3 methods | In: Overview → FilterMethods | D | | N | Follows MAC |
| addis_02.5_temporal_patterns_filter | c15, c17: timeseries + inter-method ratios; inter-method scatter | In: DailyTimeline, AgreementTimeline, Evolution (ratio field), MethodAgreement | D | | N | |
| addis_03_meteorology (+_daily) | c9: monthly and daily precipitation | In: Weather → WeatherTimeline, MonthlyClimate | D | | N | |
| addis_03_meteorology (+_daily) | c11, c17: BC by rain intensity; rain-threshold sensitivity | In: Weather → RainWashout (field = IR BCc (aethalometer), full build); rain groupings with editable cut points | D | | N | Per filter window rather than hourly or daily |
| addis_03_meteorology (+_daily) | c13: static vs dynamic seasons | In: MonthlyClimate + the dynamic season editor (re-seasons every chart) | D | | N | |
| addis_03_meteorology (+_daily) | c15: BC vs temperature by season | In: Weather correlations; Correlation per-season layout | D | | N | |
| addis_04_aeronet (+_daily) | c9: AOD timeseries; AOD by season | In: AERONET → AeronetTimeline, AeronetClimatology | D | | N | |
| addis_04_aeronet (+_daily) | c11, c15, c17: BC vs AOD, FMF and fine AOD, PW, by season | In: AERONET scatter, saved views (`aeronet-bc-fmf` …), Weather → PwMoisture | D | | N | |
| addis_04_aeronet (+_daily) | c13: surface AAE vs column AE | In: Diurnal → SurfaceVsColumnAae | D | | N | |
| a05 / ms | a05 c5, c7; ms c6, c8, c16: diurnal BC by wavelength, ÷ IR, by season | In: Aethalometer → Diurnal → ChannelCycle (BCc / ÷ IR, seasons split) | D | | N | Both not executed |
| a05 / ms | a05 c9; ms c10, c17: AAE diurnal (and seasonal) | In: Diurnal → AaeCycle | D | | N | |
| a05 / ms | a05 c9 (right): AAE by time block (1–5 vs 5–9 AM) | Partial | D | A box per time block from the hourly table | N | |
| a05 / ms | a05 c11, ms c12: Delta-C (UV − IR) diurnal and by season | **Missing** | D | UV − IR on `diurnal/aeth/<CODE>.json` hourly (dd, h, UV…IR) | N | |
| a05 / ms | a05 c13, ms c14: wavelength CV by hour (and season) | **Missing** | D | CV across 5 channels per hour; IntradayCv is a different metric | N | |
| a05 / ms | a05 c17, ms c22: representative days (5-λ day traces) | **Missing** | D | The day-hour table is already in the browser | N | |
| a05 / ms | a05 c19, ms c24: wavelength × hour heatmap normalised to the daily mean | **Missing** | D | Same data | N | |
| a05 / ms | ms c19: cross-site overlay (UV/IR, Blue/IR, AAE, normalised ΔC) | Partial: one site at a time | D | Load several `diurnal/aeth` files at once | N | |
| a05 / ms | ms c20: Green/IR ratio by site (Addis offset green) | In: Overview → SpectralShape by site (daily) | D | | N | |
| a05 / ms | a05 c21, ms c25: summary statistics CSV | **Missing** | D | A table | N | |
| addis_06_hips_ptfe_operating_envelope | c8, c10, c12: optical regime; mass per area; residual screen | In: Research → HIPS operating envelope (he-optics, he-loading, he-screen) | D | | N | Private section |
| addis_06_hips_ptfe_operating_envelope | c14: residual crossplots vs loading, composition, tau | Partial: the screen ranks ρ² only | D | Crossplots from `research/hips_optics` + rows | N | |
| addis_06_hips_ptfe_operating_envelope | c16, c18, §5: slope/intercept fits; EC-loading bins; lot screen | In: he-fits, he-bins, he-lots | D | | N | Follows MAC |
| addis_06_hips_ptfe_operating_envelope | c22: Addis residual vs AERONET SSA | Partial: not on the envelope page; reachable in Research → What tracks the offset with covariate "SSA 440 nm (AERONET)" | D | One panel | N | |
| addis_06_hips_ptfe_operating_envelope | c24: IMPROVE handoff windows | In: he-windows | D | | N | |
| addis_ababa_source_analysis | c12, c14, c16: baseline, by dominant source, ≥ threshold, for 3 pairs | In: Sources → MethodBySource, ThresholdSweep | D | | N | Follows MAC; OLS + Deming |
| addis_ababa_source_analysis | c18: stacked daily shares, overall, by season, dominant-source series | In: PMF sources (SourceStack daily bars, groupings) | D | | N | |
| source_apportionment_regression | c8–12: HIPS vs EC baseline, by source, thresholds | In: MethodBySource, ThresholdSweep | D | | N | |
| source_apportionment_regression | c14: aeth IR / UV vs EC and aeth vs HIPS, by source | Partial: the IR pairs are there; UV vs EC is not a METHOD_PAIR (scatter coloured by PMF gives it) | D | Add a UV pair | N | |
| source_apportionment_regression | c16: source contribution bars | In: SourceStack | D | | N | |
| dominant_source_comparison | c12, c16–22: GF vs K_F rules and agreement matrix; by-source regressions per rule; side by side | In: Sources → DominanceRules, MethodBySource (GF / K_F split) | D | | N | |
| dominant_source_threshold_analysis | c12: dominance-fraction distribution (GF, K_F) | Partial: DominanceRules is the agreement matrix, not the distribution | D | A histogram of the dominant share | N | Executed (169 images) |
| dominant_source_threshold_analysis | c14–22: low/high/residual per threshold, overall and per source | In: ThresholdSweep (every threshold), ResidualVsDominance | D | | N | |
| dominant_source_threshold_analysis | c26: all sources at the 50 % threshold, side by side | Partial | D | Facet ThresholdSweep by source | N | |
| threshold_by_source_analysis | c13–17, c21: per-source threshold grids GF vs K_F; R² vs threshold | In: ThresholdSweep | D | | N | |
| figure7_source_contributions | c4, c7, c9: Figure 7 OM fractions, normalised, K_F | In: PMF sources (daily bars in the PMF paper's colours) | D | | N | |
| cross_plots_and_distributions | c5: 3 pairs × all sites | In: Correlation (full build), MethodAgreement | D | | N | Follows MAC |
| cross_plots_and_distributions | c7: histograms and violins by site × method | In: Distribution (Histogram, Ridgeline) | D | | N | |
| cross_plots_and_distributions | c10: AERONET sequential exclusion (drop lowest 0/20/40/60 % AOD) | Partial: AOD quintile grouping; switching bands off reproduces each step; no 4-step panel | D | A small-multiples panel | N | Per-window AOD, not daily |
| cross_plots_and_distributions | c12: crossplot statistics CSV | Partial | D | Table | N | |
| results_summary | c3–12: re-displays PNGs from cross_plots and multisite | n/a | n/a | | N | Covered by those rows |
| smoothing_comparison | c5, c9: raw vs instrument-smoothed vs rolling median; best-window scatter | In: Data quality → MinuteSmoothing (Jacros and Pasadena minute files), SmoothTimeline (daily) | D | | N | Beijing and Delhi have only hourly files (S) |
| smoothing_comparison | c11, c13: diurnal by smoothing method; IQR width | In: aq-min-smooth-diurnal | D | | N | |

## 2. Slice summary

**Counts** (table rows; near-duplicate notebooks and repeated cells are grouped, so rows ≠ figures):

| Status | Rows |
|---|---|
| In gallery | 75 |
| Partial | 20 |
| Missing | 5 |
| n/a | 3 |
| **Total** | **103** (covering 446 saved images in the 9 executed notebooks and the code-only figures of the other 26) |

| Class | Rows |
|---|---|
| D (in browser now, or data in browser and only a chart missing) | 99 (75 drawn today, 24 data-ready but not built) |
| D+ | 1 |
| S-ML | 0 |
| S | 0 rows; one caveat: minute data for Beijing and Delhi does not exist (hourly files only) |
| n/a | 3 |

**Compared with the 27 Oct coverage doc.** Section B ("not in gallery") is out of date for this slice. All of it is now in Instruments → Aethalometer, and every one of these notebooks except Analysis_Tasks_Jan2025, Example_Modular, Task_Analysis, primary_tasks, follow_up, cross_plots, figure7 and results_summary is cited by name in a component's `provenance`. ETAD_Factor_Analysis and figure7 stay "In gallery". addis_01 moves from Partial to In. addis_06 moves from "Not in gallery" to In (Research → HIPS operating envelope). addis_03's "BC side missing" is closed in a full build (aethalometer fields in RainWashout and the Weather correlations).

**The 5 highest-value missing or partial figures**

1. **The spectral diagnostics in addis_05 and multisite_diurnal: Delta-C, wavelength CV, the hour × wavelength heatmap normalised to the daily mean, representative days, the cross-site overlay and the summary table.** These are the green-channel and brown-carbon story that results_summary §3 presents. The data is already in the browser: `aethalometer/diurnal/aeth/<CODE>.json` holds a day × hour table of all five channels and AAE for CHTS, ETAD, INDH and USPA. It needs 3–4 components on Aethalometer → Diurnal, and the cross-site overlay needs those files loaded for several sites at once.
2. **The iron/EC threshold analysis:** split at a cutoff, fit each side, and plot the residual vs Fe/EC (primary_tasks, Task_Analysis, follow_up, also run by concentration group and with the aeth BC > 7.5 compound exclusion). These notebooks ran it repeatedly. Today it takes Ratio dialog + GroupingDialog + Research → What tracks the offset, and that page is private and shows one site. It needs a generic "split at a cutoff + residual vs a variable" panel on the Correlation tab, and optionally an ECDF mode in Distribution for the Fe/EC CDF.
3. **The MANUAL_OUTLIERS threshold flags before and after** (MSF §5, FU, Example_Modular). This is the only D+ row. It needs `is_outlier` and `outlier_reason` per filter from `outliers.apply_threshold_flags`, and its inputs are tracked.
4. **addis_06's residual against the mechanism variables:** crossplots of the residual vs tau, loading and composition (c14), and the residual vs AERONET SSA (c22). Both are D in the private Research section, from `research/hips_optics/filters.json` and the AERONET fields. They need two panels on the HIPS envelope page.
5. **PMF coverage and dominance:** the ETAD all-vs-PMF counts and timeline (ETAD_Factor c12–13), the dominance-fraction histogram (dominant_source_threshold c12), and a season facet on the residual-vs-biomass chart (ETAD_Factor c38). All are D from filters.json + pmf.json.

The others are small: FlowFix §11's per-period wavelength correlation matrix (daily bins already carry `flow`), the AAE time-block box (1–5 vs 5–9 AM), a UV pair in MethodBySource, the facet by source at 50 % in ThresholdSweep, and the high/low concentration site group.

**Cheap wins (D+ with the data already in the repo)**

- (a) **The legacy 9 AM-matched aethalometer value per filter** (`IR BCc 9AM-matched`, and AAE from BCc), from `research/ftir_hips_chem/processed_sites/df_*_9am_resampled.pkl` via `data_matching.match_aeth_filter_data`, both tracked. With it the reader could reproduce every source-regression notebook's numbers exactly (the Sources page now explains why R² is 0.78 and not their 0.60). It is also the input the MANUAL_OUTLIERS rules need. Add it to `gallery/data/export_aethalometer.py` as a second field set.
- (b) **`is_outlier` and `outlier_reason` from `outliers.apply_threshold_flags`**, using the same tracked inputs plus `Filter Data/unified_filter_dataset.pkl`. Export it with `export_data.py`, next to `excluded`.

Everything else Partial or Missing in this slice is class D: the data is loaded and only a component is missing.

**Public-build caveat.** In a public build the aethalometer fields appear only on the Aethalometer tab, and the Research pages (addis_06) are absent. So the "In: Correlation with aethalometer fields" rows hold only in a full build.

## 3. ML figures

None. No notebook in this slice fits a model of its own: the fits are OLS, Deming, Theil–Sen, threshold splits and Spearman screens, and all of them run in the browser. The one model-derived input is the **ETAD PMF solution** (Naveed's EPA-PMF factors, `Filter Data/ETAD Factor Contributions .csv` → `pmf.json`). It is an external, static product (class S as an input), and every figure built on it is D. No page for this slice is proposed for an ML section. If an ML section wants a PMF page, it would be a "model provenance" card for the PMF solution (factor profiles, Q values), and that data is not in the repo.

---

## Slice B: research/ftir_hips_chem FTIR calibration, VIBES, meeting and decision notebooks

Audit date 30 Oct 2026, read-only. I started from `docs/notebook-gallery-coverage.md` (27 Oct, §G/§H) and then checked each row against today's `gallery/app/src` and `gallery/data`. This covers 26 notebooks and 135 figures. A figure is one figure cell or one displayed PNG. Rows that hold several near-identical figures give the count in Notes.

**Conventions.** Status: In / Partial / Missing / n/a. Class: D (computed in the browser now) · D+ (would be D with a small per-filter export) · S-ML (precomputed model fitting) · S (static for another reason: spectra summaries, IMPROVE volume, metadata). ML = candidate for an ML section.

**Checkout caveats that matter here**
- These export folders are git-ignored and absent: `meeting/`, `meeting-r70/` (17 Sep page), `satoshi/`, `experimental/` (every Experimental study), `baseline_*` (Baselines), `spec_curve.json` and `similarity/`. Pages in those areas exist in code but render an "export missing" state here.
- `research/ftir_hips_chem/output/` (`tables/`, `plots/`) is git-ignored and empty. A table described as "local" exists only on the author's machine or in Drive (`<Drive>/Aethmodular VIBES/...`).
- The tracked exceptions are `deliverables/filter_only_scientific_release_2026-09-11/data/**` (parquet/CSV) and `Filter Data/unified_filter_dataset.pkl`. The pkl is long-format and has LotId, AnalysisDate, CalibrationSetId, MDL, Volume_m3 and DepositArea_cm2.
- Several notebooks are unexecuted copies with no stored images. Their figures come from `scripts/plotting/*` builders or from PNGs in `output/` and Drive:
  - `ann_weekly_*`
  - `filter_*`
  - `matched_sample_audit_figures`
  - `meeting_followup_20260917`
  - `ftir_vibes_vs_airspec`
  - `vibes_followup_experiments`
  - the Colab notebook
- MAC handling:
  - Meeting-spec pages (Ann 9 Oct, 28 Oct; `components/Figure.tsx`) are frozen at MAC 10.
  - These follow the reader's MAC: MeetingFollowupPage, the Experimental Analogs/TOR studies, the Calibration run crossplots, HipsEnvelope, MethodAgreement and the Research MAC/Applicability pages.
  - FixesStudy rescales its saved slopes by MAC/10, so its intercepts stay fixed.

## 1. Figure table

| Notebook | Figure | Status · where | Class | What it would take | ML | Notes |
|---|---|---|---|---|---|---|
| `FTIR_Group_Talk_Apr2026_Draft` | [8] HIPS vs aeth IR BCc, per site | In · Aethalometer › Filter comparison › MethodAgreement | D |  | N | HIPS BC follows MAC |
| 〃 | [10] HIPS vs FTIR EC, per site | In · MethodAgreement; Correlation tab | D |  | N | follows MAC |
| 〃 | [12] aeth IR BCc vs FTIR EC | In · MethodAgreement | D |  | N | no HIPS MAC |
| 〃 | [17] iron median split, 4 sites (HIPS vs FTIR EC) | Partial · iron-split grouping (lib/groupings.ts), pooled median | D | grouping option: split at each site's own median | N | follows MAC |
| 〃 | [19] same, one figure per site (4 PNG) | Partial · same grouping + site facet | D | as [17] | N |  |
| `addis_feb_season_sensitivity_20261028` | [18] Deming slope/intercept vs PLS k per season, both calendars (look-alike refits) | Missing | S-ML | export lookalike_k_sweep.csv / lookalike_set_vs_k.csv (output/tables, local only) | Y | refits PLS per k |
| 〃 | [24] season coefficients, Feb in Dry vs Belg, 95% bootstrap | Partial · Experimental › Quick fixes “February: Dry or Belg?” (22 Oct run: published, full spline, raw) | S-ML | 28 Oct look-alike rows: deming_by_season.csv (local). Published-EC row is D already: calendar switch dry_feb/belg_feb + CI | Y | FixesStudy rescales slope by MAC/10, intercept fixed |
| 〃 | [25] Figure 9 (season look-alike predictions vs HIPS) under each calendar | Partial · Meeting 17 Sep › Seasons stitched (Feb in Dry only) | S-ML | per-filter predictions under belg_feb re-selection | Y |  |
| `addis_public246_20261020` | [18] 3 crossplots on 246 public filters (current SPARTAN, spline all-IMPROVE, season-specific), 14 added ringed | Partial · filters.json has 190 in-house Addis rows; Calibration run crossplots | D+ | export public v3.1.4 HIPS/EC for 246 ETAD + addis_predictions.csv (output/tables/addis_public246_20261020, local) | N | Fabs/10 fixed in nb; gallery would follow MAC |
| `ann_followup_20261009` | [11] concentration vs mass-per-filter histograms, Addis vs IMPROVE | In · Meeting › Ann 9 Oct §3 (a1009-conc/mass) | S | Addis side could be D: Fabs + volume_m3 in research/hips_optics/filters.json; IMPROVE side needs volumes | N | bundle fixed at MAC 10; does NOT follow MAC |
| 〃 | [14] Addis vs IMPROVE <1900 cm⁻¹, CH-normalised median spectra | In · Ann 9 Oct §4 (a1009-hump, + band_chem heatmap) | S | VIBES-corrected spectra summarised | N |  |
| `ann_followup_20261021` | [5] B: Fig 4b/5b in µg per filter, Addis + 97 look-alikes, raw vs baselined cal | Partial · global mass basis on Addis crossplots; 28 Oct §6 loading sweep | S-ML | look-alike predictions under raw (k18) and baselined cals; nb refits PLS (cell 3) | Y |  |
| 〃 | [7] C: Addis vs 97 look-alike spectra, raw/baselined, absorbance and per µg EC | Partial · Meeting 17 Sep › Spectra (analog median+IQR) | D+ | per-spectrum EC (HIPS EC Addis, TOR EC IMPROVE) next to the exported spectra, so “per µg EC” is a toggle | N | per-µg view would follow MAC |
| 〃 | [11] D: by season vs season look-alike set, raw and per µg EC | Partial · Meeting 17 Sep spectra (season picker) | D+ | as [7] | N |  |
| 〃 | [13] E: seasons pooled vs separate (season means, stitched pooled fit) | Partial · Meeting 17 Sep › Seasons stitched | S-ML | season-means within/between panel is D from filters.json; stitched preds are in meeting/ export | Y |  |
| `ann_weekly_20260910_figures` | [4] slide 2: spectral exclusion regions | In · Baselines › mask notes / spectra shading | S |  | N |  |
| 〃 | [6] slide 3: analog membership change, seasonal overlap | In · Baselines › “Earlier spectral cuts changed analog membership” | S-ML | saved (baseline_* export, git-ignored) | Y |  |
| 〃 | [8] slide 4: seasonal regressions | Partial · Meeting 17 Sep seasonal crossplots (later runs) | S-ML | Sept-10 model predictions (addis_evaluation.csv, local) | N | Meeting page follows MAC |
| 〃 | [10] slide 5: per-PMF-source calibrations, Addis R² vs TOR R² | Missing | S-ML | summary table only (summary.json pmf_counts) | Y |  |
| 〃 | [12] slide 6: training membership, Bishoftu provenance | In · Baselines › Sample eligibility and provenance | S |  | N |  |
| 〃 | [14] slide 7: proposed Addis selection/validation split | Missing | D+ | split membership (split_counts); low value | N |  |
| 〃 | [16] slide 8: Deming 95% CIs by model | Partial · CI toggle on crossplots (lib/fitUncertainty) | S-ML | model rows need predictions; published row D | N |  |
| 〃 | [18] slide 9: calibration screening, TOR R² | Partial · Calibration tab (calibration.json held-out R²) | S-ML |  | Y |  |
| 〃 | [20][22][24] slides 10–12: every training + Addis spectrum, Dry/Belg/Kiremt | In · Meeting 17 Sep › Spectra, individual mode | S |  | N | 3 figures |
| `ann_weekly_20260910_diagnostics` | [4] paired season-specific vs all-Addis, month-block bootstrap | Missing | D+ | paired_addis_predictions.csv (proxy + model cols, output/tables/ann_weekly_20260910, local/Drive) | N | proxy HIPS/MAC: would follow MAC |
| 〃 | [6] cross-season transfer heatmap | Missing | D+ | same table | N | follows MAC if dynamic |
| 〃 | [8] monthly discrepancy FTIR − HIPS/MAC | Missing | D+ | same table | N | follows MAC if dynamic |
| 〃 | [10] analog stability, 200 month resamples | Missing | S-ML | analog_bootstrap_draws / selection_frequencies (local) | Y |  |
| 〃 | [12] spectral-shape PCA map, Q above source q95 | Partial · Baselines › Research diagnostics; Research › Applicability (T²/Q) | S-ML | pca_*_scores.csv (local) | Y |  |
| 〃 | [14] source-site concentration of analogs | Partial · Experimental › TOR study (neighbour share by IMPROVE site) | D+ | analog_membership (FilterId, Site per group) | Y |  |
| `colab/VIBES_AIRSpec_Large_Run_Colab` | 01 training-only CV | Partial · Calibration › CV curve (explorer runs, not this run) | S-ML | cv curves from RESULTS (Drive) | Y |  |
| 〃 | 02 held-out TOR performance | In · Baselines › Paired EC performance, residual plot | S-ML |  | Y |  |
| 〃 | 03 paired model comparison (site bootstrap) | In · Baselines › Which method is closer; loading-band differences | S-ML |  | Y |  |
| 〃 | 04 Addis transfer | In · FTIR-baseline switch (export_baseline_ftir_ec) + External pilot | D |  | N | follows MAC |
| 〃 | 05 Addis band features | Missing | D+ | per-filter CH/carbonyl/1600 heights from RESULTS | Y |  |
| 〃 | 06 blanks and known-addition recovery | In · Baselines › Blank removal and recovery | S-ML |  | Y |  |
| 〃 | 07 runtime and coverage | n/a | S |  | N |  |
| 〃 | 08 matched spectral overlays | In · Baselines › spectra plot | S |  | N |  |
| `decision_workbook_20261022` | [8] §1 sites vs IMPROVE: loading density, site percentile loading × OC/BC | Partial · Experimental › TOR study (“every population on the same axes”, “sites compared”) | D+ | IMPROVE per-filter HIPS µg + OC/BC (improve_samples export, git-ignored) | N | HIPS µg at MAC 10 |
| 〃 | [11] §2 out-of-site RMSE vs k, 3 preprocessings | Partial · Experimental › Forward (by k, new-site split) | S-ML | CURVE per prep (Drive results) | Y |  |
| 〃 | [13] §3 where the reference fails in IMPROVE (bias/RMSE % by hard group) | Missing | S-ML | section3_groups.csv (Drive); with OOF preds per IMPROVE filter the grouping is D+ | Y |  |
| 〃 | [17] §4 specialists vs full (RMSE change) | Missing | S-ML | section4_splits/summary.csv (Drive) | Y |  |
| 〃 | [20] §5 flag permutation importance | Partial · Experimental › Flag (ablation ladder) | S-ML | section5_flag_scores.csv | Y |  |
| 〃 | [23] §6 flag on every SPARTAN site-season vs Deming intercept/mean | Missing | S-ML | section6_site_seasons.csv + network_filters (flag, ec_*) — the Deming side would then be D | Y | Deming side follows MAC once per-filter |
| 〃 | [28] §6b ablation ladder (AUC, site bootstrap) | In · Experimental › Flag | S-ML |  | Y |  |
| 〃 | [31] §7 hard vs other RMSE over k; Addis Deming vs k | Partial · Experimental › Attribution (trade-off) | S-ML | section7_tradeoff / addis_by_k (Drive) | Y |  |
| 〃 | [34] §7b local calibration on Addis; heavier than neighbours | In · Experimental › Analogs (shape rule = 7b local cal) | S-ML |  | Y | follows MAC |
| `experiments_20261022` | [6] out-of-site RMSE vs k (copy of workbook §2) | Partial · Experimental › Forward by k | S-ML |  | Y |  |
| 〃 | [9] §3 loading-matched analogs | In · Experimental › Analogs | S-ML |  | Y | follows MAC |
| `filter_only_diagnostics` | 01 within-site HIPS vs FTIR EC | In · Correlation tab; MethodAgreement | D |  | N | Fabs in Mm⁻¹, MAC-free |
| 〃 | 02 HIPS/FTIR ratio distributions | In · Distribution tab with ratio fields; ImpliedMac | D |  | N |  |
| 〃 | 03 ratio vs denominator | Partial · Correlation tab (ratio vs EC) | D | MDL-multiple strata need FTIR EC MDL | N |  |
| 〃 | 04 ratio by reported date | In · MacTimeline / Evolution tab | D |  | N |  |
| 〃 | 05 stricter-denominator sensitivity | Missing | D+ | ftir_ec_mdl_ugm3 per filter: deliverables/filter_only_scientific_release_2026-09-11/data/diagnostic/analysis_points.parquet (tracked) | N |  |
| 〃 | 06 exclusion-registry context | Partial · Overview › Excluded | D |  | N |  |
| `filter_proportionality` | 8 figs: paired blocks, bias blocks, forward errors, denominator sensitivity, weighting, ID-11 errors, ID-11 support, test weights | Missing | D+ | leave-quarter-out OLS / median-ratio on HIPS vs FTIR EC: browser-cheap. Needs MDL + ID-11 calibration id: release data/diagnostic/analysis_points.parquet, data/proportionality/FTIR_calibration_identifier_filter_links.csv (tracked) | N | MAC-free (Fabs in Mm⁻¹) |
| `filter_relationship_stability` | 9 figs: withheld-block MAE, block bias, frozen-denominator sensitivity, coefficients (one filter vs blocks), held-out scatter, later-period MAE, train/test EC ranges, metadata residuals, ETAD-0037 paired influence | Missing | D+ | same inputs; processing metadata from data/stability/processing_metadata_residuals (tracked) | N | MAC-free |
| `ftir_07_current_pls_vip_diagnostics` | [5] EC vs OC VIP profiles and overlap | Partial · VIP charts: Meeting 28 Oct §8, Satoshi VIP (other models) | S-ML | current_ec_oc_vip_profiles.csv (local); overlap tables in nb HTML | Y |  |
| `ftir_08_hips_transfer_and_vip` | [7] IMPROVE HIPS-target PLS, site-held-out CV curve | Missing | S-ML | improve_hips_group_cv_curve.csv (local) | Y |  |
| 〃 | [10] IMPROVE→Addis HIPS transfer, observed vs predicted Fabs | Missing | S-ML | addis_hips_transfer_predictions.csv (local); then D | Y | Fabs target, MAC-free |
| 〃 | [12] IMPROVE vs Addis HIPS VIP | Partial · VIP spectrum component exists | S-ML | improve_addis_hips_vip_profiles.csv (local) | Y |  |
| `ftir_09_improve_analog_selection` | [16] 3 panels: Addis D² in HIPS score space, analog spectra, analog EC projection | Partial · Calibration › Analog lab; Research › Applicability | S-ML | score_and_q_domain_summary, selected analogs, tor comparison (local) | Y | EC panel at MAC 10 |
| `ftir_10_ec_calibration_cohort_comparison` | [17] Addis seasonal spectra >1500 cm⁻¹ | In · Satoshi / Meeting 17 Sep seasonal spectra | S |  | N |  |
| 〃 | [19] CH vs carbonyl peak heights (smoke vs Addis); CV curves, first-major minimum | Partial · Calibration › CV curve (curves); peaks missing | S-ML | peak heights per spectrum (smoke_cohort_spectral_selection.csv, local) | Y |  |
| 〃 | [21] each calibration FTIR vs HIPS on one fixed Addis cohort | In · Calibration › run crossplots (calibration_runs.json eval) | D |  | N | follows MAC |
| `ftir_vibes_vs_airspec` | §1 blank background, VIBES rank selection | Partial · Experimental › Lot-aware VIBES (rank curve, 26 Oct run) | S-ML | vibes_fit_diagnostics / rank curve (output/tables/vibes_vs_airspec, local) | Y |  |
| 〃 | §2 same-filter baseline overlays (p10/50/90) | In · Meeting 17 Sep “Baselines compared — one filter”; Baselines spectra | S |  | N |  |
| 〃 | §3 population spectra, channel-wise differences | Partial · median spectra by baseline (Satoshi/Meeting) | S | paired channel differences need per-filter pairs | Y |  |
| 〃 | §4 band features both methods (CH, carbonyl, 1600) | Missing | D+ | paired_sample_metrics.csv band cols (local) | Y |  |
| 〃 | §5 agreement distribution, lot dependence | Missing | D+ | same table (rms, spectral R², LotId) | Y |  |
| 〃 | §6 held-out blanks, known-addition recovery | In · Baselines › Blank removal and recovery; Lot study | S-ML |  | Y |  |
| 〃 | §7 runtime and convergence | n/a | S |  | N |  |
| `hips_offset_narrative` | [4] cross-site HIPS vs FTIR baselines rebuilt | In · MethodAgreement; Research › HIPS envelope fits | D |  | N | follows MAC |
| 〃 | [7] raw HIPS blank audit (t+r, blank tau) | In · HIPS envelope › raw optical coordinates | D |  | N |  |
| 〃 | [10] Addis tau vs FTIR EC; tau over time by coefficient family | Partial · HIPS envelope › lot/calibration screen | D | tau-vs-time view on research/hips_optics (has tau, cal line, date) | N |  |
| 〃 | [13] raw T1 vs R1 geometry | In · HIPS envelope › R1/T1 analog | D |  | N |  |
| 〃 | [19] sample vs blank tau by FTIR batch | Partial · HIPS envelope lot screen | D+ | FTIR batch id per filter (unified_filter_dataset.pkl, tracked) | N |  |
| 〃 | [21] extreme loading: T rel. blank, tau dist, t+r | In · HIPS envelope | D |  | N |  |
| 〃 | [23] 6-panel ChemSpec loading (tau vs PM, BC fraction, composition…) | Partial · HIPS envelope mass/area; Composition | D | join hips_optics tau to filters rows (both in repo) | N | BC fraction follows MAC |
| 〃 | [35] White-style calibration space (2 PNG) | In · HIPS envelope › R1/T1 + blank domain | D |  | N |  |
| `improve_high_fabs_comparison` | [20] IMPROVE fAbs vs EC: ETAD-comparable, scatter, high-fAbs audit (3 PNG) | In · Research › IMPROVE envelope; high-tail/decile groupings | S | 380k IMPROVE rows summarised by exporter | N | fAbs vs EC: MAC-free |
| 〃 | [21] EC surface loading comparison | Partial · HIPS envelope › mass per area (SPARTAN only) | S | IMPROVE loading quantiles in envelope.json | N |  |
| `ksweep_20261014` | [3] Addis intercept vs k, standard model, 4 baselines × 4 seasons | Partial · Ann 9 Oct §2 a1009-k; Calibration › k sweep | S-ML | curves.csv (output/tables/ksweep_20261014, local) | Y | Fabs/10 fixed |
| 〃 | [5] IMPROVE R² vs k; Addis intercept vs IMPROVE R² | Partial · Calibration › k sweep R² panel | S-ML | curves.csv | Y |  |
| 〃 | [7] smoke-like IMPROVE subsets vs Addis, relative intercept by k | Missing | S-ML | parts/pooled.csv (local) | Y |  |
| 〃 | [11] seasonal analog cohorts, intercept vs k | In · Ann 9 Oct §2 (a1009-k) | S-ML |  | Y | static MAC 10 |
| `matched_sample_audit_figures` | 01 same-filter pair availability | Partial · Data availability / Census | D |  | N |  |
| 〃 | 02–08: recovered windows, 15:00 timestamps, completeness, timing status, multi-day collections, sampled hours, ChemSpec EC conflicts | Missing | D+ | per-filter collection window/timing columns: matched_sample_audit/*.parquet (Drive, DATA_ROOT) | N | 7 figures; metadata QA |
| `meeting_followup_20260917` | [5] starting point + seasons stitched | In · Meeting › Follow up 17 Sep | S-ML | meeting/, meeting-r70/ git-ignored, absent | Y | Addis axes follow MAC |
| 〃 | [7] slope vs intercept map | In · same | S-ML |  | Y |  |
| 〃 | [9] four-panel analog check (TOR/FTIR/HIPS, held-out IMPROVE + Addis) | In · same › ThreePlots | S-ML |  | Y |  |
| 〃 | [12] 100 CV splits per season | In · same › RepeatStrip | S-ML |  | Y |  |
| 〃 | [14] MAC envelope (analogs in TOR vs HIPS) | In · same › MacEnvelope | S | IMPROVE TOR/HIPS cohort | N |  |
| 〃 | [16] dry analogs coloured by OP | In · same › OpStep | S |  | N |  |
| 〃 | [20] specification curve | In · same (spec_curve_20260917.json) | S-ML |  | Y |  |
| 〃 | [22] category grid heat map | In · same › GridStep | S-ML |  | Y |  |
| `vibes_followup_experiments` | [6] common-test selection RMSE + membership change | In · Baselines › Notebook follow-up | S-ML | baseline_* git-ignored | Y |  |
| 〃 | [10] spectrum-only routing | In · same | S-ML |  | Y |  |
| `warren_meeting_slides` | [8] Fabs vs FTIR EC conc, per site | In · Correlation tab | D |  | N | MAC-free |
| 〃 | [10] Fabs vs FTIR EC mass on filter | In · Correlation tab, mass basis | D |  | N |  |
| 〃 | [12] Fabs vs aeth IR BCc | In · MethodAgreement | D |  | N |  |
| 〃 | [14][16] FTIR EC vs aeth BCc; coloured by Fe/EC (no stored image) | In · MethodAgreement; Correlation colour + ratio | D |  | N |  |
| 〃 | [18] Fe vs Fabs; [19] Fe/EC by site | In · Correlation / Distribution with ratios | D |  | N | 2 figures |
| 〃 | [21] seasonality of HIPS Fabs by site | In · Distribution/Evolution by season | D |  | N |  |
| 〃 | [23] 4 SPARTAN sites over IMPROVE, EC mass on filter | In · Research › IMPROVE envelope (binned cloud) | S |  | N |  |
| 〃 | [25] per site, 5–95 % shading on both axes | Partial · Correlation tab (no percentile box) | D | percentile-box overlay | N |  |
| 〃 | [28][29][31] White-style space, scaled t+r, all sites shared T axis | In · HIPS envelope | D |  | N | 3 figures |
| 〃 | [30] per site vs its row/lot calibration line | Partial · HIPS envelope lot/cal screen | D | cal_intercept/slope already in hips_optics | N |  |
| `workflows/data_space_recipes/hosted_lab_check` | (no figures: parquet/DuckDB check) | n/a | S |  | N |  |

## 2. Slice summary

**By status (135 figures):** In 55 · Partial 37 · Missing 41 · n/a 2.
**By class:** D 33 · D+ 38 · S-ML 46 · S 18. **ML-section candidates:** 49.

| | D | D+ | S-ML | S |
|---|---|---|---|---|
| In | 24 | 0 | 17 | 14 |
| Partial | 9 | 6 | 20 | 2 |
| Missing | 0 | 32 | 9 | 0 |

How to read this:
- Every missing figure is either D+ (32: mostly the filter-only release and metadata audits) or S-ML (9).
- Of the D figures, 24 are already live; the other 9 are partial and need code only, no new data.
- 16 of the 17 In-gallery S-ML figures sit on git-ignored exports (`meeting/`, `experimental/`, `baseline_*`), so they render only where those exports have been generated. The exception is `ksweep` [11], which lives in the tracked `meetings/ann-20261009.json`.

**Corrections to the 27 Oct coverage doc**
- `ftir_07` and `ftir_08` are Partial or Missing, not "In gallery". The gallery's VIP charts (28 Oct §8, Satoshi) use other models. The EC-vs-OC VIP overlap and all of ftir_08's HIPS-target PLS (CV curve, transfer, VIP) are absent.
- `ftir_09` and `ftir_10` are not referenced anywhere in gallery code. Their cohorts reach the gallery only through calibration presets (smoke, eth300, analog500). ftir_10's common-cohort crossplot is effectively In (D); ftir_09's summary figure is Partial.
- `decision_workbook_20261022` is mostly not in the gallery. Only §6b (Flag) and §7b (via Analogs) are. §3, §4 and §6 have no page.
- `ann_followup_20261021` is not a 28 Oct page. The 28 Oct page is `run_meeting_20261028.py` (to-do items 1–11). The 21 Oct figures B–E are only partly covered by the 17 Sep page.
- `ann_weekly_20260910_diagnostics` (paired, cross-season, monthly) is not in the gallery. Only the membership and provenance slides of `ann_weekly_figures` are.

**Five highest-value missing or partial figures**
1. **Decision workbook options B and C, plus the flag on SPARTAN** (`decision_workbook` [13] [17] [23]). These carry the workbook's argument: where the reference model fails inside IMPROVE, whether specialists beat the full calibration set, and which SPARTAN site-seasons the flag marks.
   - Needed: extend `gallery/data/export_experimental.py` to also read `section3_groups`, `section4_splits/summary`, `section6_site_seasons` and `section6_network_filters` from the same Drive run (cell 39 already writes them).
   - The per-filter `section6_network_filters` (flag, ec_this_model, ec_local, hips_bc) also makes the Deming side D and MAC-aware.
2. **Temporal transfer limits** (`filter_proportionality` ×8, `filter_relationship_stability` ×9, `filter_only_diagnostics` 05). These are 17 figures with no gallery page, and the models are one-predictor OLS and median ratio.
   - Needed: join the FTIR EC MDL and the calibration identifier (ID-11) per filter onto `filters.json`, then compute leave-quarter-out and later-period fits in the browser.
   - The inputs are tracked: `deliverables/filter_only_scientific_release_2026-09-11/data/{diagnostic,proportionality,stability}`, or the pkl's MDL and CalibrationSetId.
3. **"IMPROVE validation is blind to k"** (`ksweep` [7] smoke subsets vs Addis, plus [3] and [5] on the standard model). The gallery shows intercept vs k for the explorer configurations and for the seasonal analogs, but not the deciding check.
   - Needed: export `output/tables/ksweep_20261014/{curves,pooled}.csv` and `parts/*.csv` (local only).
4. **Ann's C7/C9 answers** (`ann_followup_20261021` [5] [7] [11] [13]). These show that the opposite-sign intercepts come from comparing two different calibrations.
   - Needed: look-alike predictions under the raw (k 18) and baselined calibrations, in µg per filter.
   - Needed: one EC value per exported spectrum (HIPS EC for Addis, TOR EC for IMPROVE), so "per µg EC" is a toggle on the existing 17 Sep spectra.
   - The season-means panel (E) can be built from `filters.json` today.
5. **HIPS as the PLS target** (`ftir_08` [7] [10] [12]). No gallery page fits FTIR to HIPS.
   - Needed: `improve_hips_group_cv_curve.csv`, `addis_hips_transfer_predictions.csv` and `improve_addis_hips_vip_profiles.csv` (local `output/tables`). With the predictions per filter, the transfer crossplot becomes D, and it does not depend on the MAC because the target is Fabs.

Close behind: `addis_feb_season_sensitivity` [18]/[25] (look-alikes re-selected under the Belg calendar) and `ann_weekly_diagnostics` [4]–[8] (D+ from `paired_addis_predictions.csv`).

**Cheap wins (data already in the repo)**
- **Filter-only release** (17 + 1 figures; item 2 above). The tracked parquet/CSV in `deliverables/filter_only_scientific_release_2026-09-11/data/` already holds every block, prediction and membership table. The page could even read the frozen tables directly as S and switch to D later.
- **HIPS envelope additions, code only** (`research/hips_optics/filters.json` already carries tau, t, r, T1, R1, lot, cal line, date, volume):
  - `hips_offset_narrative` [10]: tau through time, coloured by calibration line.
  - `hips_offset_narrative` [23]: tau vs PM2.5 mass.
  - `warren` [30]: per-site T vs R with each row's own calibration line.
- **`hips_offset_narrative` [19], FTIR batch:** add the FTIR AnalysisDate and CalibrationSetId from `unified_filter_dataset.pkl` in `export_research_hips_optics.py`.
- **Iron split per site's own median** (`FTIR_Group_Talk` [17] [19]): a grouping option in `lib/groupings.ts`; no data needed.
- **Feb-in-Belg seasonal Deming for published EC** (`addis_feb` [24]): already D through the dry_feb/belg_feb calendar plus the CI toggle. A saved view in `lib/views.ts` is enough.
- **Mass-per-filter histogram that follows the MAC** (`ann_followup_20261009` [11], Addis side): Fabs/MAC × `volume_m3` from hips_optics. The IMPROVE side stays static.
- **Percentile box on crossplots** (`warren` [25]), and the season-means within/between panel (`ann_followup_20261021` [13]): D, code only.

## 3. ML figures as proposed ML-section pages

Pages are listed most-ready first. "Saved" means the table the page needs already exists on disk; "PNG only" means only the image survives.

**ML-1 · Calibration training and k selection** (13 figures)
- Figures:
  - `decision_workbook` [11] and `experiments` [6]: out-of-site RMSE vs k, three preprocessings.
  - `ksweep` [3] [5] [7] [11].
  - `addis_feb` [18].
  - `ftir_08` [7] (HIPS target).
  - `ftir_10` [19] (first-major-minimum CV curves).
  - Colab 01 (training-only CV).
  - `ann_weekly_figures` [18] (TOR R² screening).
  - `meeting_followup` [12] (100 splits per season).
- Load:
  - Per (prep or baseline, cohort, k): RMSE ± SE by fold, held-out IMPROVE R² and bias by subset, Addis slope and intercept quantiles over splits.
  - The Addis readout should be stored as per-k predictions, not slopes, so it follows the MAC.
- Saved: ksweep `curves/pooled/parts` (local); `lookalike_k_sweep.csv` (local); ftir_08/10 CV curves (local); `calibration_runs.json` `curve` (tracked, explorer presets only).
- PNG only: the decision workbook's §2 `CURVE`. It is not in cell 39's save list, so only the PNG and the printed k/RMSE lines exist. Re-running needs one more `to_csv`.

**ML-2 · VIP and feature importance** (6 figures plus existing charts)
- Figures: `ftir_07` [5] (EC vs OC VIP), `ftir_08` [12] (IMPROVE vs Addis HIPS VIP), `decision_workbook` [20] (permutation importance), [28] (ablation ladder, In). Existing: 28 Oct §8 and Satoshi VIP.
- Load: VIP vectors per model on a wavenumber grid, VIP² share by region, overlap and Spearman between profiles, and permutation importances with site-bootstrap intervals.
- Saved: `current_ec_oc_vip_profiles.csv`, `improve_addis_hips_vip_profiles.csv`, `*_vip_bands.csv` (local); `section5_flag_scores` (Drive). The overlap summaries are also HTML tables in the executed ftir_07/08 notebooks.

**ML-3 · Baseline correction (VIBES vs AIRSpec)** (12 figures)
- Figures:
  - `ftir_vibes_vs_airspec` §1–§6.
  - Colab 01–06 and 08.
  - `vibes_followup` [6] [10].
  - `ann_followup_20261009` [14] (static spectra; listed for context).
  - Related: the Experimental Lot study.
- Load:
  - Paired per-filter metrics: RMS difference, spectral R², CH / carbonyl / 1600 band heights, LotId.
  - Held-out blank RMS and known-addition increments.
  - Blank-PCA rank curves.
  - Calibration scores with the paired site bootstrap.
  - Per-filter Addis predictions by baseline.
- Saved: `output/tables/vibes_vs_airspec/{paired_sample_metrics,heldout_blank_metrics,injection_metrics,vibes_fit_diagnostics}.csv` (local); Colab `RESULTS/*.csv` (Drive); `baseline_*` exports (git-ignored).
- New pieces: band features (§4, Colab 05) and lot dependence (§5). Both are D+ once `paired_sample_metrics` ships.

**ML-4 · Analog and cohort selection** (18 figures)
- Figures:
  - `ftir_09` [16].
  - `ann_weekly_figures` [6].
  - `ann_weekly_diagnostics` [10] [12] [14].
  - `ann_followup_20261021` [5] [13].
  - `addis_feb` [25].
  - `decision_workbook` [34] and `experiments` [9] (both In).
  - `meeting_followup` [5] [7] [9] [20] [22] (all In).
  - `ann_weekly_figures` [8] [16].
- Load:
  - Analog membership per target group and mask (FilterId, Site, rank), plus bootstrap selection frequencies.
  - PCA/T²/Q scores for source and target spectra.
  - Look-alike predictions per filter (Addis and the held-out IMPROVE look-alikes).
  - The specification-curve and category-grid tables.
- Saved: `analog_membership`, `analog_bootstrap_draws`, `pca_*_scores` (local, `ann_weekly_20260910`); `selected_improve_addis_analogs.csv`, `score_and_q_domain_summary.csv` (local, ftir_09); `experimental/analogs.json`, `meeting/*.json` (git-ignored exports).

**ML-5 · Error predictors, flags and the decision options** (8 figures)
- Figures:
  - `decision_workbook` [13] (failure groups), [17] (specialists, option B), [20] [28] (flag), [23] (flag on SPARTAN), [31] (option A, one model over k).
  - `ann_weekly_figures` [10] (per-PMF-source calibrations).
  - Existing: the Selector study (experiments §5) and the Attribution grid.
- Load:
  - Out-of-fold IMPROVE predictions with their hard-group labels.
  - Per-split RMSE for full / same-kind / local calibrations.
  - Flag scores per IMPROVE and SPARTAN filter.
  - Site-season summaries, the trade-off over k, and the decision board.
- Saved: all on Drive in `decision_workbook_20261022/results/<run>/section{3,4,5,6,7,7b,8}_*.csv` plus `settings.json` (cell 39), and `experiments/<run>/e3–e8_*.csv`. Exporters exist for only part of these (`export_experimental.py`, `export_experiments.py`). This page is mostly one exporter extension away.

---

## C: research/ftir_ec_phase3 ftir_11 to ftir_34, figures against the gallery

Audit date: 30 Oct 2026. Read-only. Checked against today's `gallery/app/src` and `gallery/data`, not only `docs/notebook-gallery-coverage.md` (27 Oct).

**Checkout facts that matter**
- `research/ftir_ec_phase3/output/` is absent and git-ignored, and `git ls-files` lists nothing under it. Every `output/tables/ftirNN/*.csv` named below exists only after a re-run on a machine with the Drive (`scripts/build_notebooks.py <n>`; see `PREREQUISITES` in `gallery/data/export_research_spectral.py`). The executed notebooks commit their displayed HTML tables, which `notebook_outputs.ExecutedNotebook` can read. Their curves, draws and per-filter rows are not displayed.
- The phase-2 tables (`research/ftir_hips_chem/output/tables/pls_calibration_phase2`, `pls_transfer`) are git-ignored too.
- The tracked per-filter file is `docs/openresearch-execution/airspec/expected/addis_predictions_corrected.csv`: ftir_13's AIRSpec predictions for OCEC-800 and Smoke-906, df1 = 6 and 8, on 239 Addis filters. `hips_transfer_corrected.csv` in the same folder holds metrics only.
- Present and tracked: `research/spectral/{mac_choice,provisional_series,band_1617,committee,applicability}.json`, `research/improve_origin/origin.json`, `aethalometer/diurnal/brc_test.json`, `calibration.json`, `calibration_runs.json`, `filters.json`, `improve_reference.json`, `meetings/*.json`. Absent (git-ignored): `meeting/`, `meeting-r70/`, `baseline_*`, `experimental/`, `spec_curve.json`. The Baselines pages and the 17 Sep and Satoshi meeting pages render empty in this checkout.
- `calibration.json` has more in it than the coverage doc credits:
  - `grid`: 14,356 rows at the rule k, by mode (`site_heldout`, `app` = interleaved within-tolerance, `app_fmm` = interleaved first minimum), cohort, OC/EC cutoffs 100–2000 (400/800/1600 included), and spectra (raw/airspec/deriv2). Each row has `k`/`ak`, OLS and Deming slope and intercept at MAC 10, `ob6`/`db6`, and R², RMSE, `ho`, `rmsecv`.
  - `sweeps`: 14,092 rows giving site-grouped RMSECV and the Addis readouts at every k for every cohort, cutoff and spectra.
  - The Calibration tab re-expresses both at the reader's MAC (`atMac` in `CalibrationPage.tsx`).
- `calibration_runs.json` is `site_heldout` only. Each of its 9 presets × 5 sites carries per-filter `eval`, the CV `curve` with SE, `heldout` metrics, `extrap_pct` and `q_residual_pct`, but no per-filter D², Q or held-out IMPROVE points.

**Keys.** Status: In / Partial / Missing / n/a. Class: D, D+, S-ML, S (defined in the brief). MAC: "MAC✓" means the figure depends on the MAC and the gallery version follows `useMac`/`lib/mac`; "MAC✗" means it depends on the MAC but the gallery version, or the needed version, does not follow it; "no MAC" means it does not depend on the MAC. ML? = candidate for an ML section.

## 1. Figure table

| NB | Figure (cell: what) | Status · where | Class | What it would take | ML? | Notes |
|---|---|---|---|---|---|---|
| 11 | [17a] IMPROVE OC/EC histogram with the 400/800/1600 cohort cuts | In · Calibration → CohortComposition + SelectionRanking (ocec ranking, cutoff line) | S | precomputed pool hist (`calibration_runs.rankings`/`cohorts`); one cut at a time | N | no MAC |
| 11 | [17b] site-held-out RMSECV vs k for N = 400/800/1600, first-major-minimum ringed | Partial · CVCurve (one preset: 800) | S-ML | already in `calibration.json` sweeps (`rmsecv` per k, every cutoff); KSweep plots no rmsecv panel → add one, overlaying cutoffs | Y | no MAC (µg/filter) |
| 11 | [18] Addis crossplots: deployed + OCEC 400/800/1600 on the fixed cohort | Partial · RunCrossplot (ocec800_raw; deployed via eval) | S-ML | 400/1600 per-filter preds = `output/tables/ftir11/addis_predictions.csv` (ignored); calibration.json has their fit lines only | Y | MAC✓ |
| 12 | [7] 1600-band peak centre by group (Addis vs IMPROVE cohorts) | In · Research → 1617 band, `band-1617-centres` | S | per-filter peak-finding on spectra the browser doesn't hold; IQR from tracked `band_1617.centers`; lib/bandCenter runs on site medians | N | no MAC |
| 12 | [15] median diagnostic-window spectra (offset-corrected raw, C–H normalised) | Partial · `band-1617-spectra` (spline-baseline stand-in) | S | `calibration_runs.site_spectra.raw` exists → offer raw space + C–H normalisation in browser (median-of-normalised ≠ normalised median: note) | N | no MAC |
| 13 | [17] Addis crossplots: OCEC-800 and Smoke-906, raw vs AIRSpec | Partial · RunCrossplot presets ocec800_raw/airspec, smoke_raw | D (3 panels) | 4th panel: tracked `docs/openresearch-execution/airspec/expected/addis_predictions_corrected.csv` column "Smoke 906, AIRSpec df1=6" (join on ExternalFilterId) | N | MAC✓ |
| 15 | [5] site-cluster bootstrap of Addis intercept/slope, raw vs AIRSpec (B=200) | Partial · ProvisionalSeries `provisional-offset` (x-intercept c CI only) | S-ML | CI table shown in exec nb cell 5 (extractable); draws `ftir15/addis_bootstrap_draws.csv` ignored | Y | intercept MAC-free; slope draws ×MAC/10 |
| 15 | [7] residual vs Mahalanobis D² by season, two models | Missing | S-ML | per-filter D² per preset (add to `export_calibration_runs` eval; explorer already computes it for `extrap_pct`) or `ftir22/addis_residuals_by_mode.csv` → then residual axis is D | Y | MAC✗ (residual = pred − Fabs/10) |
| 15 | [10] hybrid OC/EC + VIP-weighted-similarity cohort vs OCEC-800 crossplots | Missing | S-ML | hybrid predictions not saved (only `hybrid_vs_ocec_metrics.csv`, `hybrid_cohort.csv`) → re-run + save preds | Y | MAC✗ |
| 16 | [7] Addis implied-MAC distribution under Adama OC/EC | Partial · MacChoice `mac-adama` (p25/med/p75, static) | D+ (no export) | per filter MAC = Fabs·r/OC(FTIR): `filters.json` has HIPS Fabs + OC (FTIR); r = 3 constants in `mac_choice.adama_scenarios` | N | output *is* a MAC; draw reader MAC as a line |
| 16 | [11] IMPROVE implied MAC (Fabs/TOR EC) by OC/EC decile | In · MacChoice `mac-bridge` | S | 151k-filter Drive bridge; approximate D from `improve_reference.json` sample (24/site: EC/OC TOR, Fabs) | N | same |
| 17 | [5] deployed SPARTAN FTIR EC vs Fabs/MAC at MAC 10 and 6 | In · Instruments → Filters method agreement; MacChoice `mac-ftir19` (deployed row) | D | — | N | MAC✓ |
| 17 | [7] no-cleaning full-pool TOR calibration, raw and AIRSpec, at Addis | Partial · pool_raw preset (RunCrossplot); pool×airspec only as grid summary; Baselines full_pool (data absent) | S-ML | add a `pool_airspec` preset to `export_calibration_runs.py` | N | MAC✓ |
| 17 | [9] IMPROVE vs Addis median spectra, full range, raw and AIRSpec | In · Calibration → SpectraOverlay (site/cohort spectra, 3 spaces) | S | — | N | no MAC |
| 17 | [11] Addis median spectra by Ethiopian season | Missing | S | small export: per-season Addis medians/IQR into `calibration_runs.site_spectra` (explorer target spectra + dates) | N | no MAC; also ftir_31 §5 |
| 18 | [9] roundup: 6 training choices on one Addis axis (HIPS-τ transfer, smoke-906, full-pool raw, deployed, OCEC-800 AIRSpec, Addis-only nested CV) | Partial · 4/6 via presets (RunCrossplot; MacChoice `mac-sweep` lines on one plane) | S-ML | `output/tables/ftir18/addis_roundup_predictions.csv` (ignored) for HIPS-τ + nested-CV columns | Y | MAC✓ (HIPS-native panels are Fabs-space → ÷MAC) |
| 19 | [7] each setup at Fabs/10 and Fabs/6, fits pivot on intercept | In · MacChoice `mac-pivot` + `mac-sweep` | D | — | N | MAC✓ |
| 19 | [9] decision view: slopes ×0.6, intercepts fixed | In · MacChoice `mac-ftir19` (refit in browser from tracked `setup_filters`) | D | — | N | MAC✓ |
| 20 | [13] RMSECV curves per setup: {interleaved 10-fold, site-grouped 5-fold} × {raw, 2nd deriv}, both rules marked | Partial · CVCurve (site-grouped, one preset) | S-ML | site-grouped curves in `calibration.json` sweeps; interleaved curves need `ftir20/rmsecv_curves.csv` (ignored) or an `app`-mode `curve` in calibration_runs | Y | no MAC |
| 20 | [15] deck strip: k under each rule, per setup | Missing | S-ML | **no export**: `calibration.json` grid `ak` by `mo` (site_heldout/app/app_fmm) × co × sp; exec nb cells 7/11 cross-check | Y | no MAC |
| 21 | [11] both protocols on the same Addis axes, 6 setups | Partial · site_heldout per filter (RunCrossplot); app = fit lines in grid only | S-ML | `ftir21/addis_predictions_by_mode.csv` (ignored) for app points | Y | MAC✓ (site), MAC✗ (app not drawn) |
| 21 | [13] deck strip: Addis intercept and slope, protocol by protocol | Missing | S-ML | **no export**: grid rows tg=addis at rule k, mo app vs site_heldout; slope follows `atMac` | Y | MAC✓ once built |
| 22 | [10] bootstrap intercept CI by protocol | Missing (Generalise `generalise-swing` is ftir_51's swing, a different measure) | S-ML | CI table in exec nb cell 9 (extractable); draws `ftir22/bootstrap_draws_by_mode.csv` ignored | Y | intercept MAC-free |
| 22 | [13] residual vs D² by protocol (raw/AIRSpec) | Missing | S-ML | `ftir22/addis_residuals_by_mode.csv` (D², season, residual) → residual then D | Y | MAC✗ |
| 22 | [16] cohort-size sweep by protocol | In · Calibration → CutoffSweep (mode select) | S-ML | precomputed in calibration.json | Y | MAC✓ (`atMac`) |
| 22 | [18] MAC pivot by protocol | Partial · MacChoice `mac-pivot` (site_heldout) | D | app lines from grid `om/ob/ob6` (no export) | N | MAC✓ |
| 23 | [9] how each protocol picks k: curve, threshold, chosen k, 6 cohorts (×2 figs) | Partial · CVCurve (site-grouped ±SE, rule ringed) | S-ML | app curves + thresholds `ftir23/selection_curves.csv` (ignored); decisions table in exec nb cell 7 | Y | no MAC |
| 23 | [11] selection next to consequence (curve + Addis crossplot, both protocols) for OCEC-800 raw/AIRSpec (×2) | Partial · Calibration run view stacks CVCurve over RunCrossplot (site only) | S-ML | as above + app per-filter preds | Y | MAC✓ (site) |
| 26 | [27A] near EC=0: binned median Fabs vs TOR EC, power law, OLS | Missing | S | 160k Drive bridge; approximate D from `improve_reference.json` sample (EC TOR, Fabs) | N | no MAC (Fabs space) |
| 26 | [27B] intercept vs fit range, site-cluster bootstrap | In · ImproveOrigin `io-intercepts` | S | OLS bootstrap, not ML | N | no MAC |
| 26 | [27C] Deming intercept vs λ | In · ImproveOrigin `io-lambda` | S | — | N | no MAC |
| 26 | [27D] per-site intercepts vs Addis C | Partial · `io-curvature` per-site (ftir_37 halves, not ftir_26 full-range) | S | add `per_site_intercepts` to `export_research_improve_origin.py` | N | no MAC |
| 26 | [29A] site intercept ÷ site mean Fabs vs noiseless power-law null | Missing | S | same per-site table + simulated null | N | no MAC |
| 26 | [29B] share of IMPROVE filters with Fabs ≥ Addis C (survival) | Partial · `io-intercepts` note (`scale` counts) | D+ (no export) | `improve_reference.json` has 1001 quantiles of HIPS Fabs over 388k filters → survival curve | N | no MAC |
| 27 | [23] ChemSpec_BC vs Fabs/10 (rounding lattice) · ChemSpec_EC vs FTIR EC | Missing (App.tsx deliberately keeps ChemSpec carbon out of pickers; MacChoice `mac-chemspec` table only) | D+ | panel B: `filters.json` rows already carry `EC (ChemSpec FTIR)` and `EC (FTIR)`; panel A needs ChemSpec_BC_PM2.5 (not exported) | N | panel A is MAC 10 by construction; must **not** follow the reader's MAC |
| 28 | [11] AAE(625,880) histogram vs anchor; implied BrC vs target | In · Instruments → Diurnal, BrcTest `diurnal-brc` | D | per-day arrays in `brc_test.json` | N | no MAC (target \|b\|·10/slope₁₀ is MAC-invariant) |
| 28 | [15] channel health: UV/Green/Red/Blue vs IR scatter | Partial · `diurnal-channel-health` (table of checks) | D (no export) | `brc_test.json.b` already has per-day b for all 5 channels | N | no MAC |
| 29 | [9] provisional Addis EC time series | In · Calibration run view RunSeries (ocec800_airspec, trailing 45-d median) | D | — | N | no MAC |
| 29 | [11] provisional vs deployed | In · ProvisionalSeries `provisional-vs` | D | — | N | no MAC |
| 29 | [13] seasonal structure | In · `provisional-seasons` | D | — | N | HIPS series MAC✓ |
| 29 | [15] model envelope (corrected vs raw) + vs Fabs/10 with x-intercept c | In · `provisional-vs` (against raw) + RunCrossplot | D | c line not drawn (cheap) | N | MAC✓ |
| 29 | [17] offset c with bootstrap CI | In · `provisional-offset` | S-ML | from tracked `c_bootstrap_ci` | Y (uncertainty) | c in Mm⁻¹ invariant; µg/m³ at reader MAC ✓ |
| 30 | [5] cohort = bottom tail of IMPROVE OC/EC, Addis + Adama markers | In · CohortComposition (no Adama band) | S | — | N | no MAC |
| 30 | [9] residual vs D², raw vs corrected | Missing (Applicability has `extrap_pct` share only) | S-ML | per-filter D² (as ftir_15 [7]) | Y | MAC✗ |
| 30 | [11] held-out TOR test scatter + test-filter OC/EC vs Addis | Partial · `provisional-readout` table (heldout slope/R²) | S-ML | `ftir30/heldout_test_points.csv` (ignored), or held-out points per preset in calibration_runs | Y | no MAC (TOR) |
| 30 | [13] where the proof stops: D² by group, share > training p95 | Partial · Applicability "every calibration at every site" (`extrap_pct`, `q_residual_pct`) | S-ML | `ftir30/proof_boundary.csv` or per-filter D² | Y | no MAC |
| 31 | §1.1 setup matrix (per protocol) | Partial · no matrix; numbers in MacChoice/grid | S-ML | no export (grid) | Y | intercept MAC-free |
| 31 | §1.2 six-setup crossplots (per protocol) | Partial · = ftir_21 [11] | S-ML | app per-filter preds | Y | MAC✓ site |
| 31 | §1.3 residual vs D² | Missing · = ftir_22 [13] | S-ML | per-filter D² | Y | MAC✗ |
| 31 | §1.4 MAC slope pivot | In · MacChoice | D | — | N | MAC✓ |
| 31 | §1.5 component-selection curves | Partial · = ftir_23 [9] | S-ML | app curves | Y | no MAC |
| 31 | §1.6 / §2.6 intercept–slope ladder | Partial · SlopeInterceptTrap (all configs, mode select) | S-ML | per-setup ladder from grid (no export) | Y | MAC✓ (`atMac`) |
| 31 | §1.7 bootstrap intercept CIs | Missing · = ftir_22 [10] | S-ML | exec-nb CI tables | Y | MAC-free |
| 31 | §1.8 cohort-size sweep | In · CutoffSweep | S-ML | — | Y | MAC✓ |
| 31 | §1.9 MAC effect, all setups | In · MacChoice `mac-sweep`/`mac-ftir19` | D | — | N | MAC✓ |
| 31 | §1.10 six setups × four estimators (OLS, Deming, weighted, Huber-robust Deming) | Partial · OLS/Deming/York in browser (`lib/regressionMethods`) | D | add weighted + robust Deming to lib/stats; app per-filter preds | N | MAC✓ (λ ∝ MAC² handled) |
| 31 | §1.11 MAC pivot with Deming | In · `mac-pivot` | D | — | N | MAC✓ |
| 31 | §2.1 OC/EC filtering strip | In · CohortComposition/SelectionRanking | S | — | N | no MAC |
| 31 | §2.2 combined setup matrix (both intercept columns) | Missing | S-ML | grid (no export) | Y | MAC-free |
| 31 | §2.3–2.5 AIRSpec explainers (one filter + baseline; subtracted; background gap) | In · Baselines `baseline-spectrum` (data `baseline_*` absent here) | S-ML (baseline fit) | — | Y (baselines) | no MAC |
| 31 | §3 implied-MAC bridge | In · `mac-bridge` | S | — | N | — |
| 31 | §4 deployed crossplot MAC 10/6 | In · = ftir_17 [5] | D | — | N | MAC✓ |
| 31 | §5 seasonal corrected spectra | Missing · = ftir_17 [11] | S | per-season medians export | N | no MAC |
| 31 | §6 1600 peak centres | In · `band-1617-centres` | S | — | N | no MAC |
| 31 | §7 ETAD vs ETBI Fabs histograms; Adama OC/EC vs IMPROVE median | Partial · Fabs by site D anywhere (filters.json); Adama table in `ethiopia_calibrations.adama` | D | compose one figure | N | Fabs: no MAC |
| 32 | [11] CV-scheme 2×2 curves (grouping colour × fold count style), per cohort | Missing | S-ML | only the verdict table in the exec nb (cell 9); `ftir32/cv_scheme_2x2_summary.csv` ignored → re-run to export curves | Y | no MAC |
| 33 | [3] Ethiopia-shaped selection: distance histogram + carbonyl/CH vs 1600/CH scatter | Partial · SelectionRanking (eth_shaped metric/distribution, cutoff) | S | ratio scatter: per-filter band ratios from phase-2 `smoke_cohort_spectral_selection.csv` (ignored; `export_calibration.py` already reads it) | Y (cohort sel.) | no MAC |
| 33 | [5] analog selection: two ranked metrics; locked top 500 | In · AnalogLab + SelectionRanking (analogs) | S-ML (PCA score space) | — | Y | no MAC |
| 33 | [7] three cohorts on one OC/EC ruler | Partial · CohortComposition one cohort at a time; OverlapMatrix | S | overlay the 3 `calibration_runs.cohorts[*].composition` (no export) | N | no MAC |
| 33 | [9] eth300 and analog500 at Addis (×2) | In · RunCrossplot presets eth300_raw, analog500_raw | D | — | N | MAC✓ |
| 34 | no figures (tables: five-site bands, per-filter residual chemistry, AERONET n=125, York × blank-line) | n/a · echoes: HipsYork (ftir_47), Generalise five-site, Aeronet | — | tables could go in via ExecutedNotebook (stdout only, no HTML tables) | N | — |

## 2. Slice summary

**Rows: 72.** That is 51 for the 20 notebooks with figures, plus 20 for ftir_31 (most of them re-draws of ftir_19–23) and 1 for ftir_34, which has no figures.

| | In | Partial | Missing | n/a | total |
|---|---|---|---|---|---|
| Status | 28 | 27 | 16 | 1 | 72 |

| Class | D | D+ | S | S-ML | n/a |
|---|---|---|---|---|---|
| Count | 18 | 4 | 17 | 32 | 1 |

Without the ftir_31 duplicates (51 rows): In 18, Partial 20, Missing 13. Classes: D 13, D+ 4, S 14, S-ML 20.

**Changes since the 27 Oct coverage doc.**
- ftir_16 and ftir_19 are now largely covered. MacChoice draws every MAC from `calibration_runs.json` plus ftir_19's tracked per-filter table, so it is no longer limited to "28 Oct §4 only".
- ftir_29 and ftir_30 have their own page (ProvisionalSeries).
- ftir_26 has ImproveOrigin.
- ftir_12 has the Band1617 page.
- ftir_28 is in Instruments → Diurnal.
- Still unchanged: ftir_20–23 are not "in gallery" as the doc's section G implies. Only the site-grouped half is, and the protocol comparison itself is not drawn anywhere.

**The five highest-value missing or partial figures.**
1. **Protocol strips: k by rule, and Addis intercept and slope by protocol.** These are ftir_20 [15], ftir_21 [13], and ftir_31 §1.1, §2.2 and §1.6. They are the deck's "the protocol moves the answer" slides. The data needs no export: the `calibration.json` grid has `ak`, `ob`/`db` and `om`/`dm` for `mo ∈ {site_heldout, app, app_fmm}` at Addis for every cohort, cutoff and spectra. One new chart is needed, and its slopes follow `atMac`.
2. **Residual vs D².** These are ftir_15 [7], ftir_22 [13], ftir_30 [9] and ftir_31 §1.3. They carry the "extrapolation disease is gone" argument behind the provisional series. It needs per-filter D² (and Q) for each preset's eval rows, added in `export_calibration_runs.py`; the explorer already computes these for `extrap_pct`. After that the figure is D: the residual follows the MAC and exclusions.
3. **The held-out TOR test scatter**, ftir_30 [11], which is the validation behind ProvisionalSeries. It needs held-out IMPROVE points per preset (TOR, prediction, site, OC/EC) in calibration_runs, or `ftir30/heldout_test_points.csv`.
4. **CV curves under both schemes.** These are ftir_20 [13], ftir_23 [9]/[11] and ftir_32 [11]. The site-grouped curves already exist in `calibration.json` sweeps; the interleaved and 2×2 curves need `ftir20/rmsecv_curves.csv`, `ftir23/selection_curves.csv` and a ftir_32 curve export (a re-run on the Drive).
5. **Transfer roundup and both-protocol crossplots**, ftir_18 [9] and ftir_21 [11]. They need per-filter predictions for the app mode (`ftir21/addis_predictions_by_mode.csv`) and the HIPS-τ and Addis nested-CV columns (`ftir18/addis_roundup_predictions.csv`). Smoke-906×AIRSpec (ftir_13) is already in a tracked CSV.

**Cheap wins (no export, or a tracked file).**
- ftir_16 [7], implied MAC per Addis filter: computed in the browser from `filters.json` OC (FTIR) and Fabs times the three Adama ratios. It then follows exclusions, and the reader's MAC can be drawn as a line.
- ftir_28 [15], channel-health scatter: `brc_test.json.b` already holds per-day absorption for all five channels.
- ftir_26 [29B], Fabs survival against Addis C: `improve_reference.json` quantiles.
- The two protocol strips in item 1, from the `calibration.json` grid.
- A RMSECV-vs-k panel in KSweep, overlaying OC/EC cutoffs 400/800/1600: `calibration.json` sweeps (ftir_11 [17b], and the site-grouped half of ftir_20).
- App-mode pivot lines in MacChoice `mac-pivot`: grid `om`/`ob`/`ob6` (ftir_22 [18]).
- ftir_13's fourth panel, from the tracked `docs/openresearch-execution/.../addis_predictions_corrected.csv`.
- ftir_33 [7]: overlay the three cohort compositions already in `calibration_runs.cohorts`.
- ftir_29 [15]: draw the x-intercept c line on the RunCrossplot.
- Read the displayed summary tables with `ExecutedNotebook` into a new `research/spectral/protocols.json` (pattern of `export_research_spectral.py`): ftir_15 cell 5 (bootstrap CI), ftir_20 cells 7 and 11 (k by rule), ftir_22 cells 9 and 12 (CI and D² correlations by protocol), ftir_23 cell 7 (selection decisions) and ftir_32 cell 9 (2×2 verdict).

## 3. ML figures, as proposed pages for an ML section

The slice has 32 S-ML rows. Grouped by question, they form six pages. The slice has no VIP or feature-importance figure: VIP is ftir_07/08, shown in 28 Oct §8 and on the Satoshi page. The only VIP use here is ftir_15's VIP-weighted similarity, which serves the hybrid cohort.

**A. Component selection and CV schemes.** Figures: ftir_11 [17b], ftir_20 [13] [15], ftir_23 [9] [11], ftir_32 [11], ftir_31 §1.5.
- Exists today:
  - site-grouped RMSECV by k for every cohort, cutoff and spectra (`calibration.json` sweeps);
  - rule k by protocol (grid `ak` × `mo`);
  - per-preset curve with SE and auto k (`calibration_runs.runs[].curve`).
- Missing (re-run on the Drive):
  - interleaved 10-fold curves and their within-tolerance thresholds (`output/tables/ftir20/rmsecv_curves.csv`, `ftir23/selection_curves.csv`);
  - the 2×2 grouping × fold-count curves (`ftir32/cv_scheme_2x2_summary.csv`).
- Extractable now: the decision and verdict tables from the executed notebooks.
- MAC: none.

**B. Protocols end to end (interleaved app vs site-held-out).** Figures: ftir_21 [11] [13], ftir_22 [16] [18], ftir_31 §1.1, §1.2, §1.6, §2.2, §2.6.
- Exists: the `calibration.json` grid at Addis for all three modes (matrix, ladder, strips, cutoff sweep), and the site-held-out per-filter predictions (`calibration_runs`).
- Missing: per-filter app-mode predictions (`ftir21/addis_predictions_by_mode.csv`), or app-mode runs in `export_calibration_runs.py`, which today exports `mode: site_heldout` only.
- MAC: slopes follow `atMac`; intercepts are MAC-free.
- Link to Generalise `generalise-swing`, ftir_51's A/B/B2 swing.

**C. Transfer and training-set roundup.** Figures: ftir_11 [18], ftir_13 [17], ftir_17 [7], ftir_18 [9], ftir_15 [10].
- Exists: per-filter predictions for the 9 presets plus deployed EC (`calibration_runs.eval`), ftir_19's 6 setups (`mac_choice.setup_filters`), and the tracked AIRSpec predictions CSV.
- Missing:
  - OCEC 400/1600 predictions (`ftir11/addis_predictions.csv`);
  - the HIPS-τ transfer and Addis nested-CV columns (`ftir18/addis_roundup_predictions.csv`);
  - a pool×AIRSpec preset;
  - hybrid predictions, which were never saved.
- MAC: every panel MAC✓ once the rows are there; the HIPS-native models are divided by the MAC.

**D. Extrapolation and the validation boundary.** Figures: ftir_15 [7], ftir_22 [13], ftir_30 [9] [11] [13], ftir_31 §1.3. This could extend the existing Applicability page rather than be a new page.
- Exists: `extrap_pct` and `q_residual_pct` per run and site, `heldout` metrics, and ftir_39's per-filter T²/Q for the 40 locked holdouts (`applicability.domain_filters`).
- Missing: per-filter D² and Q on the Addis eval set per preset, held-out IMPROVE test points, and `proof_boundary`.
- MAC: residuals follow the reader's MAC once D² is exported.

**E. Uncertainty (bootstrap of refits).** Figures: ftir_15 [5], ftir_22 [10], ftir_29 [17] (already shown), ftir_31 §1.7.
- Exists: `provisional_series.offset_ci`.
- Extractable: the CI tables of ftir_15 and ftir_22.
- Missing: the draws (`ftir15/addis_bootstrap_draws.csv`, `ftir22/bootstrap_draws_by_mode.csv`), needed to draw distributions rather than intervals.
- MAC: intercepts invariant; slope draws re-expressed (×MAC/10).

**F. Cohort selection.** Figures: ftir_33 [3] [5] [7], ftir_11 [17a], ftir_30 [5], ftir_31 §2.1. This is mostly the Calibration tab's cohort section already: rankings, compositions, overlap, AnalogLab.
- Exists: everything except the band-ratio scatter.
- Missing: per-filter carbonyl/CH and 1600/CH for the 906 smoke filters (phase-2 `smoke_cohort_spectral_selection.csv`).
- Only the analog ranking (PCA score space) is ML. The eth-shaped and OC/EC rankings are sorts.

**Baseline fits (outside the six pages).** ftir_31 §2.3–2.5 (the AIRSpec explainers) live on the Baselines page, whose `baseline_*` data is absent from this checkout. They belong with any "preprocessing" ML page, not with the six above.

---

## Slice D — research/ftir_ec_phase3 ftir_35 … ftir_55: notebook figures vs the gallery

Audit of 30 Oct 2026, read-only. Starting point: `docs/notebook-gallery-coverage.md` (27 Oct). Every row
was checked against today's `gallery/app/src` and `gallery/data`, and against the JSON in
`gallery/app/public/data`.

**What changed since 27 Oct.**
- `research/spectral/applicability.json` now has ftir_39's per-filter `domain_filters` (120 rows).
- `holdout_fabs.json` (generated 30 Oct) gives the 40 locked holdouts their Fabs, so ftir_39's per-filter
  figure and summary now follow the MAC exactly.
- `band_1617.json` has `dust_tests` and `network_map.json` has `site_correlation` (156 × 156). None of the
  three exporters' `missing` lists is non-empty now.
- The doc's "ftir_47 → In gallery (HipsYork)" is only partial. HipsYork draws deployed, lot-linear and
  lot-quadratic, but not ftir_47's per-deployed-line variants.

**Absent from this checkout** (git-ignored):
- `spec_curve.json`, used by the Calibration specification curve (ftir_55).
- `experimental/`, behind Experimental → Adama TOR study (ftir_41, ftir_44, ftir_48 context).
- `baseline_*`, behind Baselines → external pilot, which has the Adama AIRSpec/VIBES EC against ECTR/ECTT.
- `aethalometer/quality/minute/*.json`, the ftir_49 minute panels. The daily fallbacks are present.

All of `research/ftir_ec_phase3/output/` (tables and PNGs) is git-ignored (`research/**/output/`) and is not in
the checkout. Only the PNGs exist, on a machine that ran the notebooks. The **tables that each executed
notebook displayed** are committed in the `.ipynb`, and `gallery/data/notebook_outputs.ExecutedNotebook`
already reads them (export_research_spectral.py's fallback 2). The class notes below say which data are
available that way ("displayed") and which only through a rerun ("CSV only").

Abbreviations:
- `cr` = calibration_runs.json: 9 presets × 5 sites, per-filter `eval.{id,ref=Fabs,pred,group,date}`, MAC 10.
- `cal` = calibration.json: 14,356-row grid with cohort, cutoff, selection, spectra, mode, lot and k.
- `ho` = research/hips_optics/filters.json: the four focus sites, samples and field blanks, with
  T1, R1, t, r, tau, the deployed `cal_intercept`/`cal_slope`, lot, `area_cm2`, `volume_m3` and FTIR µg.
- `fj` = filters.json.
- MAC column: "MAC: no" means the figure has no MAC dependence; "follows" means the gallery version uses
  `useMac` / `lib/mac`; "fixed 10" means it is drawn at MAC 10 only.

## 1. Figure table

| Notebook | Figure (cell: what it shows) | Status: where | Class | What it would take | ML | Notes / MAC |
|---|---|---|---|---|---|---|
| ftir_35 | No figure. Tables only: locked HIPS confirmation, HIPS-side variants, season interaction, AAE axes, chemistry partial r | Missing (n/a for figures) | S | 9 displayed tables; an export page could read them as is | N | Its residual tables are against HIPS/10 (MAC-dependent); nothing ported |
| ftir_36 | No figure. Tables: frozen protocol, λ curves, outer IMPROVE validation, locked-target summary, di-PLS vs PLS | In gallery as gallery-made figures: Research → Applicability (`applicability-dipls`, `applicability-dipls-rmse`) | S-ML | — (5 tables, displayed + CSV) | Y | Follows: RMSE re-solved from the 18 fits per holdout, or exact via holdout_fabs; the paired-bootstrap CI is dropped off MAC 10 (needs per-filter di-PLS predictions, never saved) |
| ftir_37 | [6] 2×2: (a) IMPROVE Fabs–TOR hexbin + linear/quad/spline curves, (b) held-out-site RMSE bars, (c) SPARTAN % below blank R1 by Fabs bin, (d) histogram of per-site high−low intercept shift | In gallery: Research → IMPROVE HIPS at EC = 0 (`io-curvature`: curves, CV table, shift as IQR strip; `io-blank`) + HIPS envelope `he-domain` (focus sites, live) | S (a, b, d: held-out-site regression CV on the Drive IMPROVE mirror); D for (c) on the focus sites | (a) hexbin cloud missing: improve_reference.json's per-site sample could draw it (D); (d) as a histogram from `origin.json` `curvature.sites` (D) | N | MAC: no (Fabs units). Exporter export_research_improve_origin.py |
| ftir_38 | [5] 1×3: (a) site median peak vs band/CH, (b) PMF partial r with 1617/CH, (c) dust/K partial r with prediction − HIPS/10 | In gallery: Research → 1617 cm⁻¹ band (site_bands, pmf_tests, dust_tests) | S (permutation partial r on saved tables) | (c) could be D for the gallery presets: cr pred + fj Al/Si/Ca/Fe/Ti/K; exact for ftir_35's locked configs needs their per-filter predictions (D+, CSV only) | N | (c) is MAC-dependent and drawn fixed 10 (Band1617 has no useMac) |
| ftir_39 | [6] (a) % of locked targets outside source 95 % by config, (b) domain percentile vs \|error\| per holdout filter | In gallery: Applicability (`applicability-ftir39`, `applicability-domain-filters`) | S-ML (scaler + PCA T²/Q on the source) | — | Y | Follows exactly (holdout_fabs.json; summary rebuilt in `common/holdout.ts`) |
| ftir_40 | [5] (a) simulated power (>5σ) vs paired days per season, (b) allocation stacked bars | Missing. Related: MacChoice `power` (ftir_16), Experimental "How many pairs?" (data absent) | (a) D+: Monte Carlo on Addis Fabs by season (fj), median HIPS σ (fj uncertainty) and one scalar, Adama ECTR−ECTT sd (DAVIS/Adama TOR/OC_EC_concs_Batch54.csv); (b) S, a plan table (displayed) | Export one scalar; the simulation is about 20 lines of JS | N | Built to test MAC 6 vs 10, so it does not follow the reader's MAC; could take the two MACs as inputs |
| ftir_41 | [6] Adama ECTR vs ECTT, and OC+EC TOR vs TOT (5 dates) | Missing (5-row ECTR/ECTT sit in Baselines external pilot, data absent) | D+ (5 rows; displayed table [5]) | Read the displayed ledger and reference tables | N | MAC: no |
| ftir_41 | [8] thermal-fraction shares (OC1..EC3, OP) + OP/EC | Missing | D+ (Carbon_concs_Batch54.csv fractions; CSV only, Drive DAVIS/Adama TOR) | Export the 5 × ~10 fraction table | N | MAC: no |
| ftir_41 | [11] total-carbon budget: thermal TC vs FTIR OC+EC, and ratio strip | Missing | D+ (displayed [10]) | ExecutedNotebook read | N | MAC: no |
| ftir_41 | [13] three EC definitions vs HIPS EC-equivalent, and implied MAC by definition | Partial: EthiopiaCalibrations `adama` table (TOR EC vs June calibrations); Experimental TorStudy (data absent) | D+ (displayed [13]) | ExecutedNotebook read; HIPS Fabs for 5 filters | N | The EC-equivalent panel is MAC-dependent (could follow); implied MAC is MAC-free |
| ftir_42 | [9] held-out R²/slope for four targets (TOR EC, TOT EC, OC, TC) × two cohorts | Missing | S-ML (PLS per target; displayed [6]) | Export target_definition_summary | Y | MAC: no (TOR units) |
| ftir_42 | [12] held-out TC direct vs summed, and EC/TC share crossplots | Missing | S-ML (summary displayed [11]; per-filter predictions not saved) | Rerun writing per-filter held-out predictions | Y | MAC: no |
| ftir_42 | [14] Addis crossplot vs HIPS/10 by EC convention + TOT/TOR prediction ratio histogram | Missing | S-ML; per-filter `addis_predictions_all_targets.csv` (CSV only) | That CSV joined to cr ids; then D for MAC | Y | MAC-dependent; with per-filter predictions it can follow |
| ftir_43 | [8a] f(X) vs residual with the proportional-disagreement null line | Missing | **D**: cr `ocec800_airspec` × addis pred and ref; OLS of b on f(X) | ~40 lines; also tracked `docs/openresearch-execution/airspec/expected/addis_predictions_corrected.csv` | N (its own panel) | Follows if built (b = Fabs/MAC). Cheap win |
| ftir_43 | [8b,c] residual R² for 5 learners with null violin; HIPS-prediction RMSE ladder | Missing | S-ML (time-blocked PLS residual learners, 200 null refits; displayed [5],[7] give medians and q95, not the draws) | Export the two tables; the violin needs the null draws (CSV only) | Y | RMSE ladder is MAC-dependent, fixed 10 |
| ftir_44 | [9] Adama: locked phase-3 models vs TOR/TOT EC per date + model/reference ratios | Partial: Baselines external pilot (AIRSpec/VIBES vs ECTR/ECTT; data absent) | S-ML (locked PLS refits on the pool; predictions displayed [7]) | ExecutedNotebook read of the [7] and [9] tables | Y | MAC: no (thermal reference), HIPS band drawn at 10 |
| ftir_44 | [11] OC and TC: deployed, OC/TC-target models vs quartz | Missing | S-ML (displayed [11]) | Same | Y | MAC: no |
| ftir_45 | [9a] stepwise residual R² (season, volume, lot, char class, bands) | Missing | S-ML (displayed [5]) | Export | Y | MAC 10 |
| ftir_45 | [9b] residual histograms by char_06 class | Missing | D+: cr pred/ref + per-filter char_06 class (`research/ftir_etad_char/output/tables/char06/per_filter_classification.csv`, git-ignored) | Export the class per FilterId | N | Follows if built |
| ftir_45 | [9c] residual learner's PLS coefficient spectrum | Missing | S-ML (`residual_learner_coefficients.csv`, CSV only) | Rerun output | Y | |
| ftir_46 | [11] paired ladder: pooled RMSE with block-bootstrap CI, paired ΔRMSE (blocked vs forward) | Missing | S-ML (refit bootstraps; displayed [5],[7]) | Export ladder and paired_differences | Y | MAC 10 |
| ftir_46 | [13] Addis residual vs mass, per volume and per deposit, raw and added-variable | Missing | **D**: cr pred/ref/group + fj `PM2.5 mass` + ho `volume_m3`, `area_cm2` | OLS partial slopes + per-filter bootstrap in JS | N | Residual = f(X) − Fabs/MAC, so it can follow the MAC. Cheap, high value (the "mass lead") |
| ftir_46 | [19a] per-site mass coefficient in Fabs ~ f(X) + mass (full and Addis range) | Partial: Generalise `generalise-mass` (Addis-range coefficient only, static via ftir_53) | **D** for 5 sites (cr locked preset at all targets + fj mass; per-deposit framing needs ho, focus sites only) | Compute live; Bishoftu per-deposit needs volume/area (D+) | N | Coefficient in Mm⁻¹ per µg/m³ is MAC-free |
| ftir_46 | [19b] cross-site residual vs mass grid | Missing | **D** (per-volume framing) | As above | N | Follows if built |
| ftir_47 | [8] blank T1 vs R1 per deployed line: shipped line, per-line quadratic | Partial: HIPS envelope `he-rt` (per site, blanks' OLS; not per line) | **D** for the focus sites: ho FB rows have T1, R1, lot and deployed line (`cal_intercept`, `cal_slope`); quadratic fit in JS | Group by (cal_intercept, cal_slope); network-wide is D+ (SPARTAN_HIPS_Batch1-51.v2.csv) | N | MAC: no |
| ftir_47 | [11] network exposure: median \|ΔFabs\| per site under per-line / lot-pooled quadratics; % outside own line's blank R1 | Partial: `io-blank` (whole batch, by quintile); `he-domain` (focus sites) | D for the focus sites (recompute Fabs = 100·ln((a+bR1)/T1)·A/V from ho); D+ for the network | — | N | MAC: no |
| ftir_47 | [14] five-target York intercept/slope under 4 blank-line constructions | Partial: Calibration → HipsYork (deployed, lot-linear, lot-quadratic; no per-line variants) | D+ (4 focus sites are D from ho + cr; Bishoftu needs raw T1/R1) | Add `line_lin`/`line_quad` to `/api/hips_york` or compute York in JS | N | HipsYork follows MAC (slope rescales) |
| ftir_48 | [9] six split targets (EC_TOR, EC_TOT, Δ, OP…) held-out crossplots | Missing | S-ML (summary displayed [8]; per-filter not saved) | Rerun with per-filter output | Y | MAC: no |
| ftir_48 | [14] four routes to EC_TOT, held-out crossplots | Missing | S-ML (route tables displayed [11],[13],[15]) | Tables now; points need a rerun | Y | MAC: no |
| ftir_48 | [19] Adama Δ measured vs Δ models; EC_TOT by route | Missing | S-ML predictions + 5 measured rows (displayed [18]) | ExecutedNotebook read | Y | MAC: no |
| ftir_48 | [22] Addis EC_TOT by route vs HIPS/10 | Missing | S-ML (slopes displayed [21]; per-filter not saved) | Rerun | Y | MAC-dependent, fixed 10 |
| ftir_49 | [10] A1 K and BC1/BC2/BCc closure | In gallery: Instruments → Aethalometer → Data quality (`DualSpotTimeline`, `JacrosChain`, `MinuteKstar`) | S (Drive minute CSVs → export_aeth_quality_minute.py; minute JSON absent here, daily fallback present) | — | N | MAC: no |
| ftir_49 | [13] A2 tape-advance step | In gallery: `MinuteTape` | S | — | N | |
| ftir_49 | [16] A3 clean mask, AAE by environment | In gallery: `MinuteClean` | S | — | N | |
| ftir_49 | [18] B4 missingness benchmark | In gallery: `MinuteMissing` (+ CoverageStrip) | S | — | N | |
| ftir_49 | [20] B5 AAE aggregation order | In gallery: `MinuteAaeOrder` | S | — | N | |
| ftir_49 | [22] B6 filter-day windows | In gallery: `WindowEffect` (all instruments, daily) + `MinuteWindows` | S (WindowEffect is D on daily pickles) | — | N | |
| ftir_50 | [6] per-filter T²/limit vs Q/limit, 5 targets over the library cloud | Partial: Applicability `applicability-t2q` (percent flagged only) | S-ML (library PCA) | Per-filter T²/Q ratios + a library sample (CSV only; `domain_diagnostics.csv` is the summary) | Y | MAC: no |
| ftir_50 | [9] hubness Lorenz curve | Partial: `applicability-hubness` (unique ratio, % no mutual) | S-ML (kNN counts, not saved) | Rerun writing neighbour counts | Y | MAC: no |
| ftir_50 | [12] band-resolved median r vs library median | In gallery: Band1617 (band_r) | S | — | N | MAC: no |
| ftir_51 | 01 Addis crossplots before/after (7 variants) | In gallery for 5 of 7 (Calibration → RunCrossplot per preset). network_corr, analogs_selcorr and ocec450_corr_k9 are not presets | D | Add 3 presets to export_calibration_runs.py | N | Follows (cr ref = Fabs) |
| ftir_51 | 02a/02b component sweep: Addis Deming intercept, held-out R² vs k | In gallery: KSweep (cal sweeps) | D | — | N | Follows |
| ftir_51 | 03 seasonal robustness (monthly medians, season boxes) | In gallery: RunSeries; Provisional Addis EC page | D | — | N | Prediction-only (no MAC) |
| ftir_51 | 04 analog rank distributions | In gallery: AnalogLab | S (export) | — | N | MAC: no |
| ftir_51 | 05 CV protocol A / B / B2 comparison | Partial: protocol is a choice in CutoffSweep/KSweep/SlopeInterceptTrap; no side-by-side; Generalise `generalise-swing` (static) | D (cal grid has all three modes) | One small aggregate chart | N | Follows |
| ftir_51 | 06 lot 248 vs 251 | Partial: lot is a cal dimension (fits); Experimental LotStudy (data absent) | D for fits; per-filter crossplot needs explorer runs (S) | — | N | Follows (fits) |
| ftir_51 | 07 plausibility markers per variant | In gallery: cr `plausibility` (RunSeries card, ProvisionalSeries) | D | — | N | Prediction-only |
| ftir_51 | 08 explorer screenshot; 09 backup image copies | n/a | — | — | N | |
| ftir_51 | 11a/11b OC/EC and OC/fAbs by site, SPARTAN in the IMPROVE ranking | Partial: SpartanImprove `si-ratios`, `si-analogs` | D (improve_reference.json per-site sample + fj) | Ranked-site chart | N | OC/fAbs MAC-free |
| ftir_51 | 11c/11d cohort 800 vs Addis per filter, OC/EC and OC/fAbs | Partial: CohortComposition (OC/EC histogram), 28 Oct `m1028-ocec` | D+ (cohort members' OC, fAbs from the IMPROVE FED export, Drive) | — | N | MAC: no |
| ftir_52 | [10] median (IQR) spectrum per Ward class | Missing | S-ML (Ward on PCA) | Export per-class median/IQR spectra (~4 × 3 × 1,300 values) | Y | MAC: no |
| ftir_52 | [11] PCA map coloured by class, targets overlaid | Missing | S-ML | Export PC1/PC2 for a library subsample + targets, with class labels | Y | |
| ftir_52 | [14] site dendrogram | In gallery: `SiteSimilarityMatrix` (clustered in the browser, lib/cluster) | D from the exported site × site r | — | N | |
| ftir_52 | [16] validated threshold curve (precision vs r cutoff) | Partial: `network-threshold`, `network-coverage` tables | S (LOO nearest neighbour over 13.6k spectra) | Export the threshold grid (written as threshold_coverage.csv, CSV only) | N | |
| ftir_52 | [19] IMPROVE site-pair r histogram | Missing (tables of top/least pairs only) | **D** from `site_correlation` already shipped | Histogram of the upper triangle | N | Cheap win |
| ftir_52 | [22] raw vs baselined site-pair dissimilarity | Partial: `network-raw` (one-row summary) | S | Export the raw-space site matrix | N | |
| ftir_52 | [26],[27] Addis analogs in raw vs baselined space (spectra) | Missing | S (library spectra) | Export ~10 analog spectra | N | |
| ftir_52 | [30] Addis season medians and season − Dry | Partial: Spectral similarity (season targets overlay); Band1617 medians by site | D (similarity export spectra) | Add the difference panel | N | |
| ftir_52 | [31] season analog character bars | In gallery: `network-seasons`, `network-season-loading` (tables) | S | — | N | |
| ftir_52 | [36] each season's five nearest IMPROVE spectra | Partial: Similarity "Most similar filters" (ranked list) | S | Export 15 analog spectra | N | |
| ftir_53 | 01 five-site York intercepts and slopes | In gallery: Generalise `generalise-five` | S (static York table); D+ from ho + cr | — | N | Follows (slopes and SE re-expressed; intercept fixed) |
| ftir_53 | 02 Bishoftu replication | In gallery: `generalise-bishoftu` | S | — | N | Follows (OLS slope; mean residual via meanResidualAtMac) |
| ftir_53 | 03 protocol intercept swing by variant | In gallery: `generalise-swing` (static) | S now, D possible from cal (range of intercept over modes per config) | — | N | Static at MAC 10. Intercepts are MAC-invariant only for York/OLS-on-x scaling: fine |
| ftir_53 | 04 Kiremt/Dry across 7 variants | In gallery: `generalise-kiremt` + ProvisionalSeries (9 presets, D) | S / D | — | N | Prediction-only |
| ftir_53 | 05 mass term by site | In gallery: `generalise-mass` | S (from ftir_46); D possible (see ftir_46) | — | N | MAC-free |
| ftir_54 | [9] Addis crossplot by model form (PLS, PCR-LW, RF, HGB, PLS+HGB) | Missing (PLS panel ≈ cr `ocec800_airspec`) | S-ML; per-filter predictions not saved | Rerun writing per-model Addis predictions (+190 fixed ids) | Y | MAC-dependent; would follow with per-filter predictions |
| ftir_54 | [10] model-form metrics (held-out R², RMSE, Addis slope/intercept) | Missing | S-ML (displayed [8]/[16]) | ExecutedNotebook read | Y | Addis slope at MAC 10: re-expressable |
| ftir_54 | [13a] labelled extrapolation RMSE inside IMPROVE | Missing | S-ML (displayed [12]) | ExecutedNotebook read | Y | MAC: no |
| ftir_54 | [13b] Test A crossplots (800 lowest-OC/EC predicted by models that never saw them) | Missing | S-ML (not saved) | Rerun | Y | MAC: no |
| ftir_54 | [15] Addis prediction and residual vs PCA D² by model | Missing | S-ML (Spearman summary displayed [15]) | Rerun for points | Y | Residual vs Fabs/10 |
| ftir_55 | [5] specification plane: every combination, objective contours, guardrails | In gallery: Calibration → SlopeInterceptTrap (cal, rule-k) and SpecCurve (every k; spec_curve.json absent here) | D | — | N | Follows (calibAtMac; specRmseAtMac) |
| ftir_55 | [7] folded plane \|slope−1\| vs \|intercept\|, held-out floor colour | Partial: the score is used in SlopeInterceptTrap ranking, no folded view | D (cal) | Axis option on SlopeInterceptTrap | N | Follows |
| ftir_55 | [9] plane, one panel per cohort family | Partial: SlopeInterceptTrap colours by cohort (no facets) | D | Facet toggle | N | Follows |
| ftir_55 | [11] best score by cohort × spectra | Missing | **D** (cal aggregate) | Heatmap | N | Follows. Cheap win |
| ftir_55 | [13] score vs cutoff for the winning family | In gallery: CutoffSweep | D | — | N | Follows |
| ftir_55 | [15] crossplot grid, cohort × spectra | Partial: RunCrossplot for the 9 cr presets | D for presets; others need explorer runs (S) | — | N | Follows |

## 2. Slice summary

**Counts.** 76 rows over 21 notebooks:
- 73 figure rows. Multi-panel figures are split where the panels differ in class.
- 2 rows for the tables-only notebooks ftir_35 and ftir_36.
- 1 row for ftir_51 08/09.

| Status | Rows |
|---|---|
| In gallery | 25 (incl. ftir_36 as gallery-made figures) |
| Partial | 19 |
| Missing | 31 (incl. ftir_35) |
| n/a | 1 (ftir_51 08 + 09) |

| Class | Rows |
|---|---|
| D (now, or only needs a JS port of data already shipped) | 22 |
| D+ | 8 |
| S-ML | 24 |
| S | 21 |

24 rows are ML-section candidates (Y).

By status within class:
- In gallery: mostly D (Calibration-tab ftir_51/55 panels) and S (ftir_49, ftir_52 and ftir_53 tables).
- Missing: dominated by S-ML (ftir_42, 43, 45, 46, 48, 54).
- 15 Missing or Partial rows are **D today** and need no export at all: ftir_43a, ftir_46 ×3, ftir_47 [8]
  and [11], ftir_51 05/06/11a-b, ftir_52 [19] and [30], ftir_55 [7], [9], [11] and [15].

**MAC.**
- Following the MAC: the Calibration-tab panels (ftir_51 and 55), Applicability (ftir_36 and 39, now
  exact), Generalise (ftir_53 01–02) and HipsYork.
- Fixed at MAC 10 although MAC-dependent: Band1617 dust partial r (ftir_38c) and Generalise
  `generalise-swing`. Every missing residual figure (ftir_42[14], 43, 45, 46, 48[22], 54) would also be
  fixed at MAC 10 if ported as static tables. ftir_43a and the ftir_46 mass panels can follow the MAC if
  built from cr.

**Five highest-value missing or partial figures, and what each needs.**
1. **ftir_46 mass lead** ([13] Addis residual vs mass, [19a] per-site mass coefficient, [19b] grid).
   - Data: cr `ocec800_airspec` pred/ref at 5 sites, fj `PM2.5 mass`, ho `volume_m3`/`area_cm2` for the
     per-deposit framing.
   - Work: OLS partial slopes and per-filter bootstrap in JS. Fully dynamic and follows the MAC.
   - Replaces Generalise's static `mass_term`.
2. **ftir_47 blank lines per deployed line** ([8], [11], [14]).
   - Data: ho already holds T1/R1 for samples and field blanks, plus the deployed line (cal_intercept,
     cal_slope), lot, area and volume, for the four focus sites. Per-line linear/quadratic fits and
     recomputed Fabs are D.
   - The York ladder needs cr preds (D) plus Bishoftu's raw T1/R1 (D+, `spartan_hips_raw_all.csv`).
   - Would also make ftir_53 01 dynamic.
3. **ftir_42 + ftir_48 target definition** (TOR vs TOT, Δ, composed routes).
   - S-ML. All summary tables are displayed in the executed notebooks, so the existing ExecutedNotebook
     route works.
   - Point clouds need a rerun that writes per-filter held-out and Addis predictions.
   - ftir_42's `addis_predictions_all_targets.csv` exists only as output.
4. **ftir_54 model-form robustness.**
   - S-ML. Metrics and extrapolation tables are displayed.
   - Per-filter Addis predictions by model form are never written. One `to_csv` would let the crossplot
     and D² panels follow the MAC.
5. **ftir_43/45 residual learner and null control.**
   - 43a is D now.
   - Bars are S-ML from displayed tables; the null violin needs the draws (CSV only).
   - 45b needs the char_06 per-filter class (git-ignored, ftir_etad_char output).
   - 45c needs the coefficient CSV.
   - (Runner-up: ftir_41 Adama three-method, four figures on 5 rows, all displayed. Trivial to export, but
     n = 5.)

**Cheap wins** (no new export, or one tiny one):
- ftir_43[8a]: f(X) vs residual with the null line, from cr.
- ftir_46[13] and [19b]: residual vs mass, from cr + fj + ho.
- ftir_52[19]: site-pair r histogram, from `network_map.json.site_correlation`.
- ftir_55[11]: best score by cohort × spectra heatmap, from cal.
- ftir_55[7]: folded-plane axis option, from cal.
- ftir_51 05 and ftir_53 03: protocol comparison and intercept swing from cal's three modes, made to
  follow the MAC and every preset.
- ftir_47[8]: blank T1–R1 per deployed line, from ho.
- ftir_37[d]: histogram from `origin.json.curvature.sites`.
- ftir_40[a]: power simulation in JS. Needs one scalar (Adama ECTR−ECTT sd).
- ftir_41: four 5-row figures from displayed tables.

## 3. ML figures, grouped into proposed ML-section pages

Shared rule: nothing is refitted in the browser. Exporters read the CSV when `output/` exists, else the
executed notebook's displayed table (the export_research_spectral.py pattern). Where a figure is
MAC-dependent, the export should carry per-filter predictions plus Fabs, so `lib/macMath` can move
residuals and RMSE exactly (the holdout_fabs pattern), rather than shipping a fixed-MAC table.

### A. Domain adaptation and applicability. Exists: Research → Applicability
- Notebooks: ftir_36 (di-PLS), ftir_39 (T²/Q audit), ftir_50 §2–3, ftir_54 §3.
- Loads today: `research/spectral/applicability.json`, `holdout_fabs.json` and `calibration_runs.json`
  (extrap_pct, q_residual_pct).
- To add:
  - ftir_50[6]: per-filter T²/Q ratios and a library sample (rerun; `domain_diagnostics.csv` is a
    summary).
  - ftir_50[9]: hubness Lorenz counts (rerun).
  - ftir_54[15]: prediction and residual vs D² per model form (rerun with per-filter output).
  - ftir_36's paired-bootstrap CI off MAC 10: per-filter di-PLS predictions (rerun).

### B. Residual learners and null controls. Does not exist
- Notebooks: ftir_43, ftir_45, ftir_46, plus the ftir_53 05 backup.
- Proposed `research/spectral/residual_learner.json`. From displayed tables:
  - model_comparison and null_distribution_summary (43);
  - stepwise_attribution and residual_by_char06_class (45);
  - ladder_summary, paired_differences, null_increment_summary and addis_residual_vs_mass_partial_slopes
    (46);
  - cross_site_mass_slopes (46 [17]).
- CSV only (rerun):
  - null draws (43 has no draws file; 46 writes `null_increment_draws.csv`);
  - `residual_learner_coefficients.csv` (45);
  - char_06 per-filter class.
- Live panels (D): 43a and all of 46's mass panels, from cr + fj + ho.

### C. What counts as EC: target definition and Adama. Does not exist
- Notebooks: ftir_42, ftir_48, ftir_41, ftir_44. ftir_40 is the non-ML design companion.
- Proposed `research/spectral/target_definition.json`.
  - Displayed: target_definition_summary, tc_direct_vs_summed, addis_by_ec_convention,
    split_target_summary, ec_tot_routes (+ per site), composed_error_decomposition,
    addis_ec_tot_routes.
  - Adama 5-row tables: 41 [3],[5],[10],[13]; 44 [7],[9],[11]; 48 [18].
- CSV only: `addis_predictions_all_targets.csv` (42). The held-out per-filter predictions for 42[12],
  48[9] and 48[14] are not written at all.
- Today the Adama numbers are only partly in Experimental → Adama TOR study and Baselines → external
  pilot, and both data folders are absent from this checkout.

### D. Model-form robustness and every combination tried. Partly exists (Calibration tab)
- Notebooks: ftir_54, ftir_55, ftir_51 05/06.
- ftir_55 already lives on Calibration (cal + spec_curve.json, absent here). Add the folded plane, the
  facets and the best-score heatmap from cal (D).
- ftir_54, proposed `research/spectral/model_form.json`.
  - Displayed: model_form_comparison, labelled_extrapolation_tests, addis_domain_diagnostics_spearman,
    hgb tuning.
  - Missing: per-filter Addis and Test-A predictions per model form (never written; a rerun with one
    `to_csv` each).

### E. Spectral comparison and network maps. Exists: Research → Network spectral map, Applicability, 1617 band
- Notebooks: ftir_50, ftir_52.
- Loads today: `network_map.json` (incl. site_correlation), `band_1617.json`, `applicability.json`.
- To add:
  - Ward-class median/IQR spectra (52[10]) and a PCA score sample with class labels (52[11]): S-ML,
    rerun export.
  - Threshold grid (52[16], CSV only).
  - Raw-space site matrix (52[22]).
  - Analog spectra for 52[26], [27] and [36].
  - Site-pair histogram (52[19]): D now.

Source paths cited: exporters `gallery/data/export_research_spectral.py`, `export_holdout_fabs.py`,
`export_research_improve_origin.py`, `export_aeth_quality_minute.py`, `export_spec_curve.py`,
`export_calibration_runs.py`. Pages: `gallery/app/src/sections/research/pages/{Applicability,Band1617,
NetworkMap,SiteSimilarityMatrix,Generalise,ImproveOrigin,HipsEnvelope,ProvisionalSeries}.tsx`,
`charts/calibration/{SlopeInterceptTrap,SpecCurve,CutoffSweep,KSweep,HipsYork,AnalogLab,RunCrossplot}.tsx`,
`sections/instruments/quality/{DualSpot,MinuteCharts}.tsx`.

---

## Slice E: `ftir_ec_calibration_2026_06_25` (17) and `improve_hips_offset` (12)

Audit date: 30 Oct 2026. Read-only. I checked against today's `gallery/app/src` and `gallery/data`, not against
the 27 Oct coverage doc alone.

**Abbreviations**
- **EJ**: Research → Ethiopia calibrations (Jun 2026) (`sections/research/pages/EthiopiaCalibrations.tsx`, data
  `research/spectral/ethiopia_calibrations.json` from `gallery/data/export_research_spectral.py`).
- **IE**: Research → Addis vs IMPROVE loading (`ImproveEnvelope.tsx`, `research/improve_envelope/envelope.json`
  from `export_research_improve_envelope.py`, generated 27 Oct 2026).
- **HE**: Research → HIPS operating envelope (`HipsEnvelope.tsx`, `research/hips_optics/filters.json`).
- **SI**: Research → SPARTAN vs IMPROVE (`SpartanImprove.tsx`).
- **Corr**: the Correlation chart tab (`filters.json`, plus the IMPROVE network through the per-site `improve_samples/`).
- **Cal**: the Calibration tab (`pages/CalibrationPage.tsx`, `charts/calibration/*`).
- The Research section is private, so it is left out of public builds.

**Data folders that are git-ignored and missing from this checkout.** Only the local build has them.
- `public/data/baseline_*`: the Adama/Bishoftu external pilot.
- `public/data/experimental/`: the Experimental → Adama TOR study.
- `public/data/meeting/`
- `public/data/improve_samples/`: the IMPROVE per-site files behind the network picker.
- `research/spartan_ec_2026_06_16/data/` (Adama CSVs and `rds_EC_X.csv`): the 906-filter training set.
- `research/improve_hips_offset/**/outputs`

**How the notebooks were stored**
- The 17 June notebooks were committed **without executed image outputs**. Their figures exist only as tracked
  PNGs in `research/ftir_ec_calibration_2026_06_25/figures/`, and their tables as tracked CSVs in `tables/`
  (26 files).
- The `improve_hips_offset` notebooks were committed with outputs in 7 of 12. Their CSV/PNG output folders are
  not tracked.

**Class key**
- **D**: drawn in the browser now.
- **D+**: a small export would make it dynamic.
- **S-ML**: needs model fitting.
- **S**: static for another reason; here, IMPROVE FED data that is too large to ship or not in the repo.

**MAC**
- "MAC: yes/follows" means the figure depends on the MAC and the gallery version follows the reader's MAC through
  `useMac`/`macMath`.
- Every IMPROVE figure plots fAbs in Mm⁻¹ against EC, so none of them depends on the MAC. Their slopes *are* MAC
  estimates.

## 1. Figure table

### 1a. `research/ftir_ec_calibration_2026_06_25`

| Notebook | Figure (cell: what) | Status + where | Class | What it would take | ML | Notes |
|---|---|---|---|---|---|---|
| 01_adama_spectra | [7] fig01: 5 raw Adama PTFE spectra, Teflon artifacts marked | Missing | D+ | Export the 5 spectra (`adama_ptfe_spectra_batch54.csv` in `spartan_ec_2026_06_16/data/adama`, git-ignored, missing here) | N | No MAC. EJ's Adama table uses notebook 15's interpolation. |
| 01_adama_spectra | [9] fig01b: CH-peak-area rank vs EC rank | Missing | D+ | **Tracked** `tables/adama_spectra_peak_vs_ec.csv` (5 rows): draw it | N | Cheap win |
| 02_adama_tor_vs_ftir | [9] fig02: TOR EC vs general FTIR-EC, n=5, 1:1 | Partial: `BaselineExternalPilot` "Adama against quartz thermal EC" (frozen AIRSpec/VIBES models, not the general calibration; `baseline_*` data missing here); Experimental → Adama TOR study (`experimental/` missing) | D+ | **Tracked** `tables/adama_tor_vs_ftir.csv` | N | Cheap win. No MAC. |
| 03_adama_char_soot | [7] fig03: char-EC vs soot-EC per sample, OPR and OPT | Missing | D+ | **Tracked** `tables/adama_char_soot_opr_opt.csv` | N | Cheap win. The IMPROVE equivalent could be D (EC1–EC3 and OP are in `improve_samples`). |
| 03_adama_char_soot | [9] fig03b: char/soot ratio, OPR vs OPT | Missing | D+ | Same CSV | N | Cheap win |
| 04_ethiopia_etad_spectra | [5] fig04: all 319 ETAD spectra overlaid | Partial: Similarity page (Addis target, SPARTAN Addis spectra as 8-channel display means); the per-filter spectrum in the record drawer | D | Full-resolution ETAD export (Drive `DAVIS/ETAD FTIR/ETAD_FTIR_spectra.csv`) only if exact parity is wanted | N | |
| 04_ethiopia_etad_spectra | [9] fig04: spectra by "biomass vs diesel" months (placeholder split) | Partial: Similarity → Addis site-seasons (calendar seasons, not Navid's split) | D | **Tracked** `tables/etad_sample_month_summary.csv` holds the notebook's season per MediaId; it could become a grouping | N | The split was a placeholder in the notebook itself |
| 05_calibration_variants | [6] fig05: RMSECV vs k, 906-filter EC set | Partial: Cal → CV curve (explorer runs, not this set) | S-ML | **Tracked** `tables/ec_rmsecv_curve.csv` (k 1–30): drawable now as a static curve | Y | No MAC |
| 06_…_components | [9] fig06: six variants (CAL-0…5), predicted vs measured in sample, k=20 | Partial: EJ "All 906 calibration filters…" crossplot = the CAL-0 panel with the removed filters marked (the `calibration_all` table is in the shipped JSON) | S-ML | Re-export per-filter in-sample predictions for CAL-1…5 (needs `rds_EC_X.csv`, local). Summary in **tracked** `calibration_variants_results.csv`. | Y | |
| 06_…_components | [12] CAL-3 "below 1:1" training set | Partial (same crossplot) | D | Colour `calibration_all` by predicted < measured: config only | Y | |
| 06_…_components | [13] CAL-2: the 3σ-removed training set | In gallery: EJ "removed filters" crossplot + table | D | – | Y | EJ uses spartan_ec 08's 9 rounds (61 removed); 07's CAL-1 stops after 3 rounds (862 kept). EJ says so. |
| 06_…_components | [15] RMSECV curve per variant | Missing | S-ML | Only the k picks are tracked (`component_selection_by_variant.csv`); the curves are PNG only. Needs a CV re-export. | Y | Superseded by 09_component_selection_fixed |
| 06_…_components | [19] RMSECV raw vs 2nd derivative (CAL-0) | Partial: Cal specification curve compares raw vs 2nd-derivative explorer runs | S-ML | PNG only; re-export | Y | |
| 06_…_components | [21] mean raw vs mean 2nd-derivative spectrum | Missing | D+ | Mean raw spectrum of the 906 set (local `rds_EC_X.csv`); Savitzky–Golay is linear, so the browser can derive it | N | |
| 07_ethiopia_variant_ec | [9] ETAD EC by variant (box) | **In gallery**: EJ "How far the calibration choice moves…" (07 family) | D | – | N | Per-spectrum predictions come from tracked `etad_ec_by_variant_conc.csv`; static ML output, drawn live. Does not follow gallery exclusions (own MediaId export). |
| 07_ethiopia_variant_ec | [11] each variant vs CAL-0, median ratio | Partial (data in browser, no paired plot) | D | Crossplot config on EJ `per_filter`; **tracked** `etad_variant_ratio_vs_cal0.csv` | N | Cheap win |
| 08_adama_hips_crossplots | [6] Adama new EC vs TOR, per variant | Missing (EJ only shows notebook 15's old/new, as a table) | D+ | **Tracked** `adama_new_ec_by_variant.csv` | N | Cheap win |
| 08_adama_hips_crossplots | [8] ETAD new EC vs HIPS, per variant | **In gallery**: EJ crossplot (choose "07 · CAL-x") + fits table | D | – | N | MAC: follows (x = Fabs/MAC). Gallery line is Deming; the notebook used OLS. **Mislabel:** EJ titles `etad_newec_vs_hips_fits.csv` "as notebook 11 saved them", but `_build_08…py:190` writes it (exporter line also says 11). |
| 09_component_selection_fixed | [7] %RMSECV vs k per variant, 10-fold interleaved | Missing | S-ML | Only k* is tracked (`component_selection_corrected.csv`); curves PNG only | Y | The corrected version of 06[15] |
| 09_component_selection_fixed | [9] raw vs 2nd-derivative RMSECV, CAL-0 | Missing | S-ML | PNG only | Y | |
| 09_ethiopia_component_ec | [12] panel A: RMSECV with recommended k | Missing | S-ML | Probably the same CAL-0 curve as **tracked** `ec_rmsecv_curve.csv` (verify k_min 21) | Y | |
| 09_ethiopia_component_ec | [12] panel B: ETAD EC by component count | **In gallery**: EJ intervals (09 family) | D | – | N | |
| 09_ethiopia_component_ec | [12] panel C: median ETAD EC vs k | Partial: `component_summary` is in the JSON but **not drawn** | D | Sweep figure on existing data | N | Cheap win |
| 10_calibration_variants_effect | [7] variants 1:1 at each corrected k* (CAL-0…7) | Missing | S-ML | PNG only; per-filter CV predictions need a re-export | Y | |
| 10_calibration_variants_effect | [9] one set of 906 filters, predicted EC per calibration | Missing | S-ML | Summary **tracked** (`calibration_variants_effect.csv`: median, ratio, % negative); distributions need a re-export | Y | |
| 10_ethiopia_ec_both_lots | [12] panel A: median EC vs k by training set | **In gallery**: EJ "Spanning both filter lots…" sweep | D | – | N | Tracked `etad_ec_both_lots_median_by_trainset.csv` |
| 10_ethiopia_ec_both_lots | [12] panel B: per filter, lot-251-only vs both lots at k=30 | Partial (both columns in EJ `per_filter`) | D | Crossplot config | N | Cheap win |
| 11_hips_intercept_calibrations | [7] ETAD vs HIPS: CAL-0 k20 / EC≤50 k10 / EC≤30 k5 | Partial (CAL-0 k20 only, EJ crossplot) | S-ML | Per-filter ETAD predictions for the other two (Drive: ETAD spectra, `SPARTAN_HIPS_Batch1-51.v2.csv`, local `rds_EC_X.csv`). **Tracked** `hips_intercept_sweep.csv` (14 fits) is drawable now as a slope/intercept sweep. | Y | MAC: yes (slope × MAC/10 via macMath; intercept unchanged) |
| 12_dedicated_ethiopia_calibration | [13] progression crossplot: current → lot-248+251 → +Weakley | Partial: EJ "Notebook 12" slope/intercept intervals (from printed output) + crossplot for current and notebook 15's NEW | S-ML | PCA-neighbourhood and Weakley per-filter predictions are not saved; re-run with `ftir_local_db` mirror | Y | MAC: follows (slopes) |
| 13_chase_origin_and_biomass | [5] slope–intercept frontier for 7 spectral corrections, target (1,0) | Missing | S-ML | **Tracked** `hips_origin_frontier.csv` (7 rows): drawable now | Y | MAC: yes; the slope and the (1,0) target move with MAC/10 |
| 13_chase_origin_and_biomass | [7] origin crossplot: drop-CF (slope≈1) vs SNV (intercept≈0) | Missing | S-ML | PNG only; per-filter predictions need a re-run | Y | MAC: yes |
| 13_chase_origin_and_biomass | [9] organic-band profile (CH/C=O/OH/aromatic), ETAD vs IMPROVE | Missing | D+ | Band areas, no model fitting: per-spectrum fractions from ETAD spectra (Drive `DAVIS/ETAD FTIR`) and the lot-248+251 mirror (`ftir_local_db`) | N | The 27 Oct doc's "home-grown biomass classifier" |
| 14_choosing_calibration_data | [5] PCA leverage, ETAD vs training | Partial: Research → Applicability (T²/Q, ftir_39); Cal cross-site transfer "score OOD" | S-ML | PNG only. LOSO and Kennard–Stone results: **tracked** `calibration_data_selection_summary.csv` (no figure in the notebook) | Y | |
| 15_downstream_recommended | [5] ETAD vs HIPS, old vs validated calibration | **In gallery**: EJ crossplot (15 family) + intervals | D | – | N | MAC: follows |
| 15_downstream_recommended | [9] Adama new EC vs TOR, old vs new | Partial: EJ table "Notebook 15 at Adama" | D | Crossplot of the in-browser `adama` table | N | Cheap win. No MAC. |

### 1b. `research/improve_hips_offset`

All IMPROVE network data comes from the FED export, which is not in the repo. `envelope.json` ships it pre-binned,
so the IMPROVE side is **S** throughout. Where the SPARTAN overlay is drawn live, it follows exclusions and site
filters.

| Notebook | Figure (cell: what) | Status + where | Class | What it would take | ML | Notes |
|---|---|---|---|---|---|---|
| anne_spartan_improve_ec_mass_fabs | [8] fig01: all IMPROVE (grey) + 4 SPARTAN sites, EC mass vs fAbs | **In gallery**: IE "The IMPROVE cloud and the four SPARTAN sites" (mass) | S + D | – | N | IMPROVE is a 2-D density; SPARTAN dots are live (`hips_optics` + exclusions) |
| anne… | [10] fig02: per-site p05–p95 box panels, IMPROVE overlap | Partial: IE draws the Addis box only | D | The other sites' boxes from live SPARTAN rows; IMPROVE overlap counts approximated from the shipped density cells | N | |
| anne… | [12] fig03: origin-fit slopes per site and IMPROVE | Partial: HE "Slope and intercept separated — OLS, through the origin…" (SPARTAN, D); SI fits (IMPROVE OLS, not origin) | D / S | Add the IMPROVE origin slope to `envelope.json` (one number) | N | |
| anne… | [15] fig04: SPARTAN R/T, samples vs field blanks by site | **In gallery**: HE "SPARTAN R1 / T1 — the Warren Figure 2 analog" | D | – | N | |
| etad_vs_improve_ec_matched | [9] fig01: distributions, ETAD vs EC-matched IMPROVE subsets | **In gallery**: IE "Matched on EC…" intervals | S | – | N | ETAD quantiles are exported too, so they do not follow gallery exclusions live |
| etad_vs_improve… | [11] fig02: fAbs vs EC scatter of the matched subsets | Partial (quantiles only) | S | Export a per-subset 2-D density or subsample | N | |
| etad_vs_improve… | [13] fig03: MAC-style origin slopes | Partial: IE matched table gives OLS slope and intercept, not the origin slope | S | Add `origin_slope` to `matched[].fit` | N | IE's fAbs/EC reference line uses `meta.mac_value`, not `useMac()`, so it does not follow the reader's MAC |
| etad_vs_improve… | [15] fig04: top IMPROVE sites contributing matched rows | Missing | S | Small export: site counts per subset | N | |
| etad_vs_improve… | [17] fig05: year distribution of matched rows | Missing | S | Small export: year counts | N | |
| improve_addis_analog_audit | [5] SPARTAN reference: HIPS BC vs FTIR EC, and τ vs EC | **In gallery**: Corr (HIPS BC vs FTIR EC; MAC follows); τ is in `hips_optics` but has no τ-vs-EC crossplot | D | – | N | The 27 Oct doc pointed this notebook at Similarity/analog lab (spectral analogs); IE is the right home |
| improve_addis_analog_audit | [15] overview: group sizes, fAbs vs EC with Addis overlay, slope by subset | **In gallery**: IE screens + density + matched | S | – | N | |
| improve_addis_analog_audit | [19] smoke candidates: by context, top sites, site-month | Partial: IE "Are the Addis-like IMPROVE filters smoke events?" (counts only) | S | Export site and site-month counts of smoke-like rows | N | |
| improve_addis_analog_audit | [23] RT saturation: geometry, fAbs vs T ratio, smoke by saturation | Partial: IE "FED 635 nm…" (binned R/T coloured by median fAbs) | S | Add a saturation flag share to the RT export | N | |
| improve_addis_analog_audit | [26] regression sensitivity across groups | Partial: IE fAbs sweep + intercept table | S | Per-group fits table in the export | N | |
| improve_fed_rt_proxy_figure2 | [8] FED R/T proxy, Figure 2 style ×3 (initial/final/minimum) | **In gallery**: IE R/T chart with initial/final/minimum toggle | S | – | N | The 27 Oct doc said "Not in gallery"; now covered |
| improve_fed_rt_proxy_figure2 | [10] zoomed views around highlighted rows | Partial (IE has no zoom on the comparable rows) | S | Ship comparable/high-tail points at full precision | N | Shown with plt.show, never saved |
| improve_fed_rt_proxy_figure2 | [12] SPARTAN raw R/T analog, all sites | **In gallery**: HE R1/T1 facets | D | – | N | |
| improve_first_order_loading_range | [8] IMPROVE availability by year | Partial: Data availability → IMPROVE (per site; `improve_samples` missing here) | S | Year counts in `envelope.json` | N | |
| improve_first_order… | [9] post-2003 baseline distributions | Partial: Distribution tab with IMPROVE network (`improve_reference.json` quantiles) | S/D | – | N | |
| improve_first_order… | [12] sequential screening waterfall p05–p95 | **In gallery**: IE "How many IMPROVE filters look like Addis?" | S | – | N | |
| improve_first_order… | [15] fAbs vs EC mass and surface loading, comparable regions | **In gallery**: IE density, mass/surface toggle | S + D | – | N | |
| improve_first_order… | [18] full vs all-bounds scatter with fits | Partial | S | Density of the bounded subset + its fit | N | |
| improve_first_order… | [19] bounded-group regression comparison | Partial: IE matched fits and intercepts | S | Per-group fit table | N | |
| improve_first_order… | [22] RT proxy space of comparable filters | **In gallery**: IE R/T (comparable points overlaid) | S | – | N | |
| improve_hips_offset_narrative | [12] characteristic smoke-like candidates (zoomed scatter) | Partial: IE smoke table | S | Export candidate points (a few hundred) | N | |
| improve_hips_offset_narrative | [15] fAbs threshold sweep with moving average | **In gallery**: IE "Selecting on fAbs manufactures an intercept" | S | – | N | Gallery has no moving average; 12 thresholds vs 0.25 Mm⁻¹ steps |
| improve_hips_offset_narrative | [16] ×2: fAbs-vs-EC regression panels across thresholds, per screen | Partial (sweep summarises; no panels) | S | Per-threshold densities; or keep as sweep | N | |
| improve_hips_offset_narrative | [17] ipywidgets threshold slider | n/a (interactive, no saved image); its intent is IE sweep | S | – | N | |
| improve_hips_offset_narrative | [19] IMPROVE matched to ETAD by EC surface loading, fAbs vs EC | **In gallery**: IE density (surface) | S | – | N | |
| improve_hips_offset_narrative | [21] RT geometry proxy for loading-matched samples (2 panels) | Partial: IE R/T | S | – | N | |
| improve_hips_offset_narrative | [23] T-vs-R initial-line diagnostic (2 panels) | **In gallery**: IE R/T initial line + "Trans_M − initial line" by fAbs | S | – | N | |
| improve_smoke_event_qc | [5] row coverage by year | Partial: Data availability → IMPROVE | S | Year counts | N | |
| improve_smoke_event_qc | [9] QC flag counts | Missing | S | Small export of flag counts | N | |
| improve_smoke_event_qc | [12] fAbs vs EC with ETAD-comparable overlays | **In gallery**: IE density | S | – | N | |
| improve_smoke_event_qc | [15] top candidate sites + site-month clustering | Missing | S | Export site and site-month counts | N | The OC/EC ≥ 5 cut is a gallery grouping (`lib/groupings.ts` `ocec-smoke`) |
| improve_smoke_event_qc | [15] regional/seasonal context | Missing | S | Same export + region field | N | |
| improve_smoke_event_qc | [17] RT geometry: min R vs T; fAbs vs T ratio | Partial: IE R/T | S | – | N | |
| improve_smoke_event_qc | [19] regression sensitivity across QC groups | Partial: IE sweep/intercepts | S | Per-group fit table | N | |
| improve_spartan_may_full | [5] fig01: raw post-2003 fAbs vs ECf (µg/m³) | Partial: Corr with IMPROVE network (24/site sample in `improve_reference.json`, or per-site files) | D (sampled) | – | N | |
| improve_spartan_may_full | [6] fig02: binned density, fAbs vs ECf concentration | Partial: IE density is in mass/surface, not concentration | S | Add a concentration density to `envelope.json` | N | |
| improve_spartan_may_full | [8] fig03: IMPROVE EC surface loading vs SPARTAN ranges | **In gallery**: IE density (surface) | S + D | – | N | |
| improve_spartan_may_full | [9] fig04: loading range summary | **In gallery**: IE "The networks in mass on the filter" | S + D | – | N | |
| improve_spartan_may_full | [11] fig05: axis-by-axis screening waterfall | **In gallery**: IE screens | S | – | N | |
| improve_spartan_may_full | [12] fig06: axis overlap counts | **In gallery**: IE screens ("alone" bars) | S | – | N | |
| improve_spartan_may_full | [14] fig07: rows in the ETAD EC range highlighted | Partial: IE density box | S | – | N | |
| improve_spartan_may_full | [16] fig07b: top-40 high-fAbs rows, EC loading ranked | Missing | S | Export the top-40 table (small) | N | |
| improve_spartan_may_full | [17] fig08: fAbs vs loading, comparable rows highlighted | **In gallery**: IE density + box | S | – | N | |
| improve_spartan_may_full | [19] fig09: bounded-group origin slopes | Partial: IE matched (OLS) | S | Origin slopes per group | N | |
| improve_spartan_may_full | [21] fig10: FED laser R/T diagnostic space | **In gallery**: IE R/T | S | – | N | |
| improve_spartan_may_full | [23] fig11: SPARTAN raw R/T by MTL lot | Partial: HE lot screen (table), HE R1/T1 (not coloured by lot) | D | Colour HE R1/T1 by lot: config | N | |
| improve_spartan_may_full | [25] ×2: re-displays warren_cena PNGs | n/a (duplicates of warren [6] and [12]) | – | – | N | |
| improve_spartan_may_prep_task_map | no figures | n/a | – | – | N | Task map |
| improve_white_style_rt_calibration_space | [11] A: min readings vs initial line; B: displacement vs fAbs; C: fAbs vs ECf highlighted | **In gallery** (A, B): IE R/T + deficit; C Partial | S | – | N | The 27 Oct doc said "Not in gallery"; now covered |
| improve_white_style… | [13] site panels for high-fAbs IMPROVE sites in RT space | Missing | S | Export RT points for the ~10 named sites (small) | N | |
| improve_white_style… | [15] fAbs vs ECf by optical/loading group, with MAC0 | Partial: IE matched/intercepts | S | Per-group fits incl. origin MAC | N | |
| warren_cena_improve_prep | [6] SPARTAN four sites, HIPS fAbs vs FTIR EC with fits | **In gallery**: Corr (HIPS Fabs or BC vs EC (FTIR)); HE fits | D | – | N | Follows exclusions; with HIPS BC it follows the MAC |
| warren_cena… | [12] post-2003 baseline + high-fAbs tail (2 panels) | Partial: Corr/Distribution with IMPROVE network; IE density | S/D | – | N | |
| warren_cena… | [15] which ETAD axis is rare in IMPROVE | **In gallery**: IE screens | S | – | N | |
| warren_cena… | [17] IMPROVE in ETAD loading space | **In gallery**: IE density | S | – | N | |
| warren_cena… | [20] full vs comparable fits | Partial: IE matched | S | – | N | |
| warren_cena… | [23] SPARTAN T1/R1 by filter lot, JPL and Addis | Partial: HE R1/T1 (not by lot) | D | Lot colouring | N | |
| warren_cena… | [25] IMPROVE RT proxy caveat | **In gallery**: IE R/T | S | – | N | |
| next_tests_2026_04/00_fed_addis_loading_envelope | [9] fAbs vs EC surface loading, origin at zero | **In gallery**: IE density (surface; linear axes from 0) | S | – | N | |

## 2. Slice summary

**Figures:** 98 rows.
- **Calibration:** 35 rows from 32 saved figures. Panels that land in different places are split into their own
  rows.
- **IMPROVE:** 63 rows. 3 of them are n/a: the slider, the 2 re-displays (one row) and the figure-free task map.

### Counts by status

| | In gallery | Partial | Missing | n/a |
|---|---|---|---|---|
| ftir_ec_calibration_2026_06_25 (35) | 6 | 14 | 15 | 0 |
| improve_hips_offset (63) | 26 | 27 | 7 | 3 |
| **Total (98)** | **32** | **41** | **22** | **3** |

### Counts by class
Mixed rows are counted under their first class: "D / S" as D, "S + D" and "S/D" as S.

| | D | D+ | S-ML | S |
|---|---|---|---|---|
| Calibration (35) | 13 | 8 | 14 | 0 |
| IMPROVE (60 excl. n/a) | 9 | 0 | 0 | 51 |

### No IMPROVE figure needs model fitting
- Everything there is OLS, thresholds or binning.
- The obstacle is that the FED data is not in the repo and is too large to ship (~388k rows), so the work is all
  in `export_research_improve_envelope.py`.
- All 14 S-ML figures are in the calibration notebooks.

### Changes since the 27 Oct coverage doc (its rows for this slice need updating)
- **Now covered:** `improve_fed_rt_proxy_figure2` and `improve_white_style_rt_calibration_space` were "Not in
  gallery". They are now In gallery/Partial (IE R/T chart, HE R1/T1).
- **Wrong home:** `improve_addis_analog_audit` was "In gallery: Similarity, analog lab". Those pages hold spectral
  analogs; the notebook's figures are on IE and are Partial.
- **Overstated:** June `05`, `06`, `09_component_selection_fixed`, `10_calibration_variants_effect`, `11` and `14`
  were listed "In gallery" under "Components, protocols and the search".
  - Cal shows the *same kind* of chart (CV curve, k sweep, slope–intercept trap) for the calibration-explorer runs,
    not for the CAL-0…CAL-7 variants.
  - Most of these June figures are Missing or Partial.
- **Overstated:** `01` and `04` were "Spectra in gallery". `01` is Missing and `04` is Partial.
- **Data missing here:** `02` and `08` point to BaselineExternalPilot and TorStudy, whose data is git-ignored and
  missing from this checkout.
- **Now in EJ:** `12` and `15` had blank cells and are now in EJ (Partial and In gallery). `13` is still Missing.

### Two code issues found
1. EJ labels the table built from `etad_newec_vs_hips_fits.csv` as notebook 11's ("…as notebook 11 saved them";
   `export_research_spectral.py` `notebook="11_hips_intercept_calibrations"`). Notebook 08 writes it
   (`_build_08_adama_hips_crossplots.py:190`). Notebook 11's own table, `hips_intercept_sweep.csv`, is not exported.
2. IE's fAbs/EC reference line uses `meta.mac_value` (the export's MAC), not `useMac()`, so it ignores the
   reader's MAC.

### Five highest-value missing or partial figures
1. **13 [5] and [7]: the (1,0) frontier and the origin crossplot.** This is the June series' conclusion that you
   get slope≈1 or intercept≈0, not both.
   - The frontier can be drawn now from tracked `hips_origin_frontier.csv`, with slopes re-expressed by macMath.
   - The crossplot needs per-filter predictions for drop-CF and SNV from a re-run (Drive: ETAD spectra, HIPS batch
     CSV, the lot-248+251 mirror).
2. **11 [7]: intercept vs component count and training-EC range.** Tracked `hips_intercept_sweep.csv` gives a
   static slope/intercept sweep now. The two extra crossplot panels need re-exported per-filter predictions. MAC
   applies.
3. **06/09/10 component selection and variant 1:1 grids** (06[15], 09fix[7], 10eff[7], 10eff[9]). These are the
   core CAL-variant method figures, and only PNGs and k* summaries exist. They need one exporter run against the
   local `spartan_ec_2026_06_16/data/rds_EC_X.csv` that writes:
   - per-variant 10-fold RMSECV curves;
   - per-filter CV and in-sample predictions (906 × 8 calibrations).
4. **The Adama TOR set (02, 03 ×2, 08[6], 15[9]).** These are the only thermal-EC references in Ethiopia. All five
   tables are tracked or already in the browser; nothing new needs computing.
5. **IMPROVE EC-matched scatter, top sites and years, and smoke site-month clustering** (etad_vs_improve
   fig02/04/05, smoke_event_qc [15], analog_audit [19]).
   - These need small additions to `export_research_improve_envelope.py`: subset densities, site/year/site-month
     counts, and origin slopes.
   - The exporter only runs where the IMPROVE FED cache exists.

### Cheap wins: data already tracked in the repo or already in the browser

| Figure | Source | Work |
|---|---|---|
| 01[9] | `adama_spectra_peak_vs_ec.csv` | Draw it |
| 02[9] | `adama_tor_vs_ftir.csv` | Crossplot with 1:1 |
| 03[7], 03[9] | `adama_char_soot_opr_opt.csv` | Bars and ratio |
| 08[6] | `adama_new_ec_by_variant.csv` | Per-variant crossplot vs TOR |
| 15[9] | EJ `adama` table, already in the browser | Crossplot |
| 07[11] | EJ `per_filter`; `etad_variant_ratio_vs_cal0.csv` | Crossplot config |
| 10both[12]B | EJ `per_filter` | Crossplot config |
| 09eth[12]C | EJ `component_summary`, exported but undrawn | Draw it |
| 06[12] | `calibration_all`, already exported | Colour by predicted < measured |
| 04[9] | `etad_sample_month_summary.csv` | Season per MediaId as a grouping |
| Static S-ML summaries | `ec_rmsecv_curve.csv`, `hips_intercept_sweep.csv`, `hips_origin_frontier.csv`, `calibration_variants_effect.csv`, `component_selection_corrected.csv`, `calibration_data_selection_summary.csv` | Draw now, no refit |
| HE R1/T1 | Lot is already in `hips_optics` | Colour by lot (warren [23], may_full [23]) |

All tracked files above are in `research/ftir_ec_calibration_2026_06_25/tables/`. Adding them to
`export_research_spectral.py` only takes `pd.read_csv` calls. That exporter already reads that folder as source 3.

## 3. ML figures grouped into proposed ML-section pages

### Page ML-1: "PLS calibration variants and component selection" (CAL-0…CAL-7, 906 IMPROVE filters)

**Figures:**
- 05[6] RMSECV curve
- 06[9] variants 1:1 grid
- 06[12], 06[13] inverse sets
- 06[15] RMSECV per variant
- 06[19] raw vs 2nd-derivative RMSECV
- 09fix[7] %RMSECV, 10-fold
- 09fix[9] raw vs d2
- 09eth[12]A RMSECV with k_min
- 10eff[7] 1:1 at k*
- 10eff[9] effect on the same 906 filters

**Exists today:**
- Tracked: `ec_rmsecv_curve.csv`, `calibration_variants_results.csv`, `component_selection_by_variant.csv`,
  `component_selection_corrected.csv`, `calibration_variants_effect.csv`, `calibration_variants_plan.csv`,
  `ec_threshold_retention.csv`
- Already shipped: EJ `calibration_all`, `removed`, `cleaning_rounds` (906 in-sample at k=20, CAL-0)

**Missing:**
- Per-variant RMSECV curves (raw and 2nd derivative)
- Per-filter predictions for CAL-1…CAL-7

**To make it:** one new exporter reading `spartan_ec_2026_06_16/data/rds_EC_X.csv` and `rds_EC_Ymeasured.csv`
(local, git-ignored), about 906 × 10 values plus curves of k ≤ 40.

**MAC:** none; this is all TOR EC.

### Page ML-2: "Which training set and components set the Ethiopia EC"
This page extends the existing EJ.

**Figures:**
- 07[9], 07[11]
- 08[8]
- 09eth[12]B/C
- 10both[12]
- 12[13]
- 15[5]
- Plus the Adama views: 08[6], 15[9]

**Exists today:** almost all of it, in `ethiopia_calibrations.json` (`per_filter`: 319 spectra × 21 calibrations,
`trainset_medians`, `component_summary`, `dedicated_progression`, `adama`) and tracked
`adama_new_ec_by_variant.csv`.

**Missing:** the PCA-neighbourhood and Weakley per-filter predictions from notebook 12.

**MAC:** follows. The HIPS axis is Fabs/MAC, and slopes are scaled by macMath.

### Page ML-3: "Chasing the origin: corrections that move the ETAD–HIPS line"

**Figures:**
- 11[7]: three crossplots plus the sweep
- 13[5]: frontier
- 13[7]: drop-CF vs SNV crossplot
- 12[13]: progression, shared with ML-2

**Exists today:** tracked `hips_intercept_sweep.csv` (14 fits) and `hips_origin_frontier.csv` (7 fits); EJ
`dedicated_progression`; per-filter predictions only for CAL-0 k20 and 15's NEW.

**Missing:** per-filter ETAD predictions for EC≤50 k10, EC≤30 k5, drop-CF and SNV.

**To make it:** a re-run on a machine with the Drive (`maia_data_root()/DAVIS/ETAD FTIR`,
`Spartan/SPARTAN_HIPS_Batch1-51.v2.csv`, `ftir_local_db` lot-248+251 mirror) that writes the 4 columns to
`tables/`.

**MAC:** yes for every figure on this page. The slope and the (1,0) target both scale with MAC/10.

### Page ML-4: "Choosing and validating the calibration data"

**Figures:**
- 14[5]: PCA leverage, ETAD inside the training domain
- 14's LOSO and Kennard–Stone/EC-stratified results (printed and tabled, no figure)

This page overlaps Research → Applicability (T²/Q) and Cal cross-site transfer, so it could be a section of
Applicability rather than a new page.

**Exists today:** tracked `calibration_data_selection_summary.csv`.

**Missing:** per-spectrum leverage values (ETAD 319, training ~13k; a histogram's bins would do) and the LOSO
per-site table.

**To make it:** a re-run with the Drive mirror.

**Not ML, but belongs next to these pages:** 13[9] organic-band profile and 06[21] mean raw vs 2nd-derivative
spectrum. Both are deterministic transforms of spectra (D+) and fit a spectra page. ML-1 could link to them.

---

## Slice F: ftir_etad_char, spartan_ec_2026_06_16, spartan, July07, addis_fabs_ec_deming

Audit date 2026-10-30. Read-only, checked against the code as it is today, not the 27 Oct
`docs/notebook-gallery-coverage.md`. 32 notebooks, 149 figures. One notebook
(`spartan_ec_2026_06_16/01_carbon_methods_audit`) has no figures, so it is n/a.

**Abbreviations for where a figure lives**

| Abbrev. | Gallery location (file) |
|---|---|
| **OS** | Research › EC shift and the offset (`sections/research/pages/OffsetShift.tsx`): dynamic, from `filters.json` |
| **SI** | Research › SPARTAN vs IMPROVE (`SpartanImprove.tsx`): static tables in `research/spartan_improve/tables.json`, which come from tracked CSVs in `research/spartan/{improve_comparison,follow_up_plots}/tables` |
| **IE** | Research › Addis vs IMPROVE loading (`ImproveEnvelope.tsx`): static summaries in `research/improve_envelope/envelope.json` (needs the IMPROVE FED cache) |
| **AP** | Research › Applicability domain (`Applicability.tsx`): `research/spectral/applicability.json` |
| **EC6** | Research › Ethiopia calibrations (Jun 2026) (`EthiopiaCalibrations.tsx`): `research/spectral/ethiopia_calibrations.json` |
| **CT** | The generic chart tabs (`App.tsx`: Scatterplot, SiteRank, Distribution/Histogram/Ridgeline, Timeseries, Map). SPARTAN all-site rows come from `spartan_reference.json` (38 sites, with HIPS Fabs, FTIR EC/OC, Fe, PM2.5 and Sampled volume). IMPROVE per-filter rows come from `improve_samples/`, which is **git-ignored and absent from this checkout**; `improve_reference.json` and `improve_map.json` give quantiles, a sample of 24 filters per site, and site medians. The View options in `lib/viewFields.ts` add reader-defined ratios (A/B), a mass-per-filter basis (× Sampled volume) and the MAC. |
| **DA / MAP / CAL** | The Data availability page; the Map tab; the Calibration page (`charts/calibration/*`, `calibration_runs.json`) |

**Data that is absent here:** `ftir_etad_char` writes everything to `output/` (git-ignored by
`research/**/output/`). Its inputs are also local-only: the charcoal archive sits in
`research/ftir_hips_chem/charcoal_ftir_sources/downloads/` (~225 MB, ignored), and the ETAD
AIRSpec spectra in `research/ftir_ec_phase3/output/corrected/etad_corrected_df6.npz` (ignored).
For the char notebooks, only the embedded PNGs and the displayed HTML summary tables (3–9 per
notebook) are tracked. The `similarity/`, `experimental/`, `meeting/` and `improve_samples/`
gallery data folders are git-ignored and absent. `addis_fabs_ec_deming/*`, July07 and spartan_ec
02/03/04 are committed **without outputs**, so they have 0 embedded images; the spartan_ec and
July07 PNGs are tracked under their `figures/` folders.

**MAC column (in notes):** "MAC–" means the figure does not depend on the MAC. fAbs in Mm⁻¹,
fAbs/EC, TOR and spectra are all MAC-free. "MAC✓" means it depends on the MAC and the gallery
version follows the reader's MAC. "MAC✗" means it depends on the MAC and the notebook fixes
MAC = 10.

**Class codes:** D = dynamic now. D+ = needs a small export. S-ML = model fitting, so it must be
precomputed. S = static for another reason.

## 1. Figure table

| Notebook | Fig (cell: what) | Status · where | Class | What it would take | ML | Notes |
|---|---|---|---|---|---|---|
| char_01 | [5] archive spectra, native grid/scale per collection | Missing | S | Export a subsample of the reference spectra per collection; the source is `downloads/` (ignored) | N | MAC–. Archive-only, no Addis |
| char_01 | [10] mean spectrum by pyrolysis T, IQR, per collection | Missing | S | Export group-mean/IQR spectra (collection × T) | N | |
| char_01 | [12] the two unlabeled sets (Minatre species, Ayauchi depth) | Missing | S | Same export, by species and depth | N | |
| char_01 | [14] band trajectories vs T | Missing | S | `band_summary_by_collection_temperature.csv` (in output/, ignored) | N | |
| char_01 | [16] aromatization index vs T | Missing | S | `aromatization_index_by_temperature.csv` (ignored; shown in the notebook) | N | |
| char_01 | [20] pretreatment (Gosling) | Missing | S | Group means by pretreatment × T | N | |
| char_02 | [8] Addis over charcoal series, SNV | Missing | S | Group means of the charcoal series plus Addis median/IQR on the shared grid | N | MAC– |
| char_02 | [10] same overlay, unit-area | Missing | S | As [8] | N | |
| char_02 | [12] band space: Addis filters on the charring path | Missing | D+ | Per-filter Addis band intensities (FilterId, aliphatic, O–H, aromatic) plus charcoal group means; this would then follow seasons and exclusions | N | `band_comparison.csv` is a summary only |
| char_02 | [15] nearest-analogue match, recovery check, implied T by season | Missing | S-ML | `addis_analogue_matches.csv` per filter (ignored) | Y | The recovery check is a validation |
| char_02 | [18] charcoal PCA with Addis projected | Missing | S-ML | PCA scores (PC1–2) per spectrum | Y | PNG only |
| char_02 | [20] Mahalanobis distance of Addis | Missing | S-ML | Per-filter distance | Y | PNG only |
| char_03 | [5] McCall raw vs char, full range | Missing | S | Class-mean spectra | N | |
| char_03 | [7] same, unit-area | Missing | S | As [5] | N | |
| char_03 | [8] burning difference spectra | Missing | S | As [5] | N | |
| char_03 | [10] Addis vs burned/unburned classes | Missing | S | As [5] plus Addis median | N | |
| char_03 | [12] same, unit-area | Missing | S | As [10] | N | |
| char_03 | [16] PCA+logistic burned/unburned classifier, LOO-CV | Missing | S-ML | `addis_burned_scores.csv`, `loo_feedstock_cv.csv` (ignored) | Y | |
| char_03 | [18] Addis position on each burned/unburned axis | Missing | D+ | Per-filter band positions | N | `band_positions.csv` is a summary |
| char_04 | [5] silhouette vs k | Missing | S-ML | Silhouette by k (table) | Y | KMeans on PCA |
| char_04 | [7] Pass A types: archetypes, scores, by T | Missing | S-ML | Cluster labels plus PCs per spectrum; `passA_*` tables (ignored) | Y | |
| char_04 | [9] Pass A, unit-area | Missing | S-ML | As [7] | Y | |
| char_04 | [11] Pass B with Addis | Missing | S-ML | Labels plus PCs incl. Addis; `passB_type_by_source.csv` | Y | |
| char_04 | [13] Pass B, unit-area | Missing | S-ML | As [11] | Y | |
| char_04 | [15] Ward dendrogram of group means | Missing | S-ML | Linkage matrix | Y | |
| char_05 | [5] six unburned feedstocks, full range | Missing | S | Feedstock mean spectra | N | |
| char_05 | [7] same, unit-area | Missing | S | As [5] | N | |
| char_05 | [9] Addis (10–90 %) vs each feedstock | Missing | S | As [5] plus Addis band | N | |
| char_05 | [11] same, unit-area | Missing | S | As [9] | N | |
| char_05 | [14] raw biomass vs char as analogue, per feedstock | Missing | D+ | Per-filter r to each feedstock and r_char | N | No fitting: Pearson r |
| char_05 | [16] residual Addis − raw biomass | Missing | S | Residual mean spectrum | N | |
| char_05 | [18] every Addis filter scored vs both classes, by season | Missing | D+ | Per-filter `r_raw`, `r_char` + FilterId/date. The gallery season convention and exclusions would then apply | N | `per_filter_classification.csv` (char_06) has exactly these columns |
| char_06 | [8] similarity collapse every dry season; monthly rate | Missing | D+ | Per-filter `r_char`, `component` by date | N | `monthly_anomaly_rate.csv` is a summary |
| char_06 | [11] two populations (Gaussian mixture) | Missing | S-ML | `component` label per filter (in `per_filter_classification.csv`) | Y | Shipping the labels makes the plot D |
| char_06 | [13] anomalous vs other mean spectra + difference | Missing | S | Two group-mean spectra | N | |
| char_06 | [15] same, unit-area | Missing | S | As [13] | N | |
| char_06 | [20] char similarity vs EC loading and vs Fabs | Missing | D+ | Join `r_char` per FilterId to `filters.json` EC (FTIR), HIPS Fabs | N | MAC– (uses Fabs). High value: loading control |
| char_07 | [7] char_05 scatter with dry season greyed; advantage | Missing | D+ | Per-filter `r_raw`, `r_char`, `component`; exclusion rule as a toggle | N | |
| char_07 | [9] feedstock reordering without dry | Missing | D+ | Per-filter r per feedstock | N | |
| char_07 | [11] Addis (non-dry) vs each feedstock | Missing | S | Spectra | N | |
| char_07 | [13] same, unit-area | Missing | S | Spectra | N | |
| char_07 | [15] mean spectra, residual vs best feedstock | Missing | S | Spectra | N | |
| char_07 | [17] same, unit-area | Missing | S | Spectra | N | |
| char_08 | [6] reference classes ranked; furnace vs fire per filter | Missing | D+ | Per-filter r per reference class (7 classes) | N | |
| char_08 | [10] furnace vs fire vs Addis spectra | Missing | S | Class means | N | |
| char_08 | [12] same, unit-area | Missing | S | As [10] | N | |
| char_08 | [16] every class worse in dry season | Missing | D+ | Per-filter r per class + season | N | |
| char_09 | [8] ranking by subset; furnace vs fire coloured by season | Missing | D+ | As char_08 [6] | N | |
| char_09 | [10] relative loss by class | Missing | D+ | As char_08 [6] | N | |
| char_09 | [13] spectra dry vs non-dry vs refs | Missing | S | Group means | N | |
| char_09 | [15] same, unit-area | Missing | S | As [13] | N | |
| char_09 | [18] residual vs best ref, dry vs non-dry | Missing | S | `residuals.csv` (ignored) | N | |
| char_10 | [5] raw ETAD spectra by season | Missing | S | Raw ETAD spectra (Drive; not in any gallery export) | N | |
| char_10 | [6] AIRSpec + area-normalised by season | Partial · Similarity page overlays site-season groups; CAL `site_spectra` has the Addis median | D+ | Per-filter AIRSpec traces with season are already in the `similarity/` export (ignored, absent); area-normalise in the browser | N | The baseline itself is precomputed |
| char_10 | [8] seasonal means, 10–90 % | Partial · as [6] | D+ | As [6] | N | |
| char_10 | [10] every filter + seasonal means | Partial · as [6] | D+ | As [6] | N | |
| char_10 | [12] one panel per season | Partial · as [6] | D+ | As [6] | N | |
| char_11 | [11] one spectrum under each transform | Missing | S | Two spectra (didactic) | N | |
| char_11 | [17] AIRSpec fit on charcoal vs Addis side | Missing | S-ML | Baseline fits on spectra | Y | |
| char_11 | [21] benchmark (PCA silhouette, cross-lab recovery) | Missing | S-ML | `benchmark.csv` (ignored) | Y | |
| char_11 | [24] implied temperature by transform | Missing | S-ML | `addis_conclusion_by_transform.csv` | Y | |
| char_12 | [5] raw IMPROVE vs ETAD mean + 5–95 % envelope | Partial · AP `applicability-scale` (table only) | S | Envelope spectra (mean/p05/p95 per wn, two populations) | N | |
| char_12 | [7] 2nd-derivative envelopes, two regions | Missing | S | As [5], in derivative space | N | |
| char_12 | [9] envelope-escape per filter, by group | Partial · AP `applicability-escape` (group medians, static) | D+ | Per-filter escape score + FilterId/date/lot. Seasons and exclusions would then follow | N | Not ML: a percentile test |
| char_12 | [11] lot effect: dry filters by lot; escape lot × season | Partial · AP `applicability-lots` (table) | S | Per-filter escape (as [9]) plus lot-mean spectra | N | |
| char_12 | [13] lot comparison in the AIRSpec view | Missing | S | Lot-mean AIRSpec spectra | N | |
| char_12 | [16] furnace-vs-fire recomputed in derivative space | Missing | D+ | Per-filter derivative-space r per class | N | |
| spartan_ec 01 | none | n/a | — | — | N | Tables only (carbon method codes) |
| spartan_ec 02 | [9] fig02 general vs biomass(lot-251 tool) EC vs HIPS, ETAD | Missing | D+ | **Tracked** `tables/biomass_vs_general_ec.csv` (FilterId ETAD-…, EC_general, EC_biomass, HIPS_Fabs): join on FilterId and plot against HIPS BC | N | MAC✗ (fabs/10). Coverage doc said "In gallery": not so (see naming note in §2) |
| spartan_ec 03 | [8] fig03 Adama char-EC vs soot-EC (Han) | Missing | S | **Tracked** `adama_char_soot_classification.csv` (5 rows); a static table or bars | N | MAC–. TOR thermograms are not in any gallery export |
| spartan_ec 04 | [7] (A) τ vs EC mass, corrected ref lines | In gallery · CT Scatterplot, mass basis (HIPS BC × V stands in for τ) | D | — | N | MAC✓ (as BC mass). Ref lines not drawn |
| spartan_ec 04 | [10] (B) fAbs/EC by site vs meeting values & IMPROVE | In gallery · CT SiteRank, ratio HIPS Fabs / EC (FTIR) | D | Meeting reference values are not drawn | N | MAC– |
| spartan_ec 04 | [12] (C) EC/OC by site, FTIR | In gallery · CT SiteRank ratio | D | — | N | |
| spartan_ec 04 | [14] (D) fractional EC, focus vs IMPROVE | In gallery · CT ratio, both networks | D | IMPROVE needs `improve_samples/` | N | |
| spartan_ec 05 | [4] training set: EC loading hist + 8 spectra | Missing | S | Biomass lot-251 tool training set (`spartan_ec_2026_06_16/data/`, not tracked) | N | |
| spartan_ec 05 | [7] baseline RMSEP vs k + pred vs meas | Missing (analogue: CAL › CV curve, for gallery presets) | S-ML | Curve and predictions not saved (PNG only) | Y | |
| spartan_ec 05 | [10] residual hist; residual vs fitted ±2/3σ | Missing | S-ML | As [7] | Y | |
| spartan_ec 05 | [15] sweep before/after cleaning + optimized fit | Missing | S-ML | As [7]; only the coefficients CSV is written | Y | |
| spartan_ec 05 | [18] Adama EC general / biomass base / optimized vs TOR | Missing | S | **Tracked** `adama_ec_optimized_comparison.csv` (5 rows) | N | MAC– |
| spartan_ec 07 | [3] fig07_1 jagged RMSEP across CV seeds | Missing | S-ML | RMSEP by k × seed (not saved; PNG tracked) | Y | |
| spartan_ec 07 | [6] fig07_2 Adama EC vs k | Missing | S-ML | Adama EC by k (not saved) | Y | |
| spartan_ec 07 | [11] fig07_3 repeated-CV valley | Missing | S-ML | Mean ± sd RMSEP by k (not saved) | Y | |
| spartan_ec 07 | [15] fig07_4 Adama stability by approach | Missing | S-ML | Not saved | Y | |
| spartan_ec 07 | [17] fig07_5 Adama general vs stabilized vs TOR | Missing | S | **Tracked** `adama_ec_stabilized_comparison.csv` | N | MAC– |
| spartan_ec 08 | [3] fig08_1 all 906 cal filters, removed ringed | In gallery · EC6 `ethiopia-removed` | S-ML | `calibration_all` (906 rows, recomputed from Drive) is present in this export; the fallback shows the 61 tracked rows | Y | MAC– (TOR µg) |
| spartan_ec 08 | [5] fig08_2 residual diagnostics | Partial · EC6 (rounds table) | S-ML | Residual = measured − predicted is already in `calibration_all`; this only needs a chart | Y | |
| spartan_ec 08 | [7] fig08_3 before → after refit | Partial · EC6 `cleaning_fit` (n, R², RMSE) | S-ML | Cleaned-refit predictions for the 845 (Drive) | Y | |
| spartan_ec 08 | [11] fig08_4 removed vs typical spectra | Missing | S | 4 spectra from the Drive 906 set | N | |
| spartan follow_up | [6] fig00 IMPROVE site funnel | Missing | D+ | Per-step site counts (or compute from `improve_samples/` volume/flow fields) | N | |
| spartan follow_up | [10] fig01 mass cross-plots (fAbs~EC, OC~EC, fAbs~Fe, fAbs~PM) | In gallery · CT Scatterplot, mass basis, both networks | D | — | N | MAC–. IMPROVE cloud: `improve_reference` sample, or full `improve_samples/` |
| spartan follow_up | [12] fig01b τ vs EC mass | Partial · CT (HIPS BC mass) | D | A τ = Fabs·V/A field (area constant per network) | N | MAC✓ via BC |
| spartan follow_up | [14] fig02 fAbs/EC by site, SPARTAN on IMPROVE ranking | In gallery · CT SiteRank + ratio, both networks; SI `si-ratios` (static) | D | — | N | MAC– |
| spartan follow_up | [16] fig03 OC/EC by site | In gallery · as [14] | D | — | N | |
| spartan follow_up | [18] fig04 Addis-analog map (site medians OC/EC vs fAbs/EC) | Partial · SI `si-analogs` (table) | D | A site-median scatter (from rows) | N | |
| spartan follow_up | [20] fig05 fAbs vs Fe; fAbs vs Fe fraction | In gallery · CT Scatterplot + ratio | D | — | N | |
| spartan follow_up | [22] fig06 Fabs across SPARTAN sites, ETBI vs ETAD | In gallery · CT SiteRank / Distribution | D | — | N | |
| spartan follow_up | [23] fig07 ETBI time series; fAbs vs τ | Partial · CT Timeseries; τ as [12] | D | τ field | N | |
| spartan follow_up | [27] fig08 OC/Fabs by site | In gallery · CT ratio | D | — | N | |
| spartan follow_up | [31] fig09 Fe/HIPS, Fe/EC by site | In gallery · CT ratio | D | — | N | |
| spartan follow_up | [33] fig10 fAbs vs EC coloured by Fe/EC | In gallery · CT Scatterplot colour by value (ratio) | D | — | N | |
| spartan follow_up | [36] fig11 mass on filter by site | In gallery · CT SiteRank, mass basis | D | — | N | |
| spartan follow_up | [38] fig12 deposit mass vs composition | In gallery · CT Scatterplot mass + ratio | D | — | N | |
| spartan follow_up | [41] fig13 fAbs vs OC mass coloured by EC share | In gallery · CT (colour by EC/OC) | D | An EC/(EC+OC) field would be exact | N | |
| spartan follow_up | [43] fig14 within-IMPROVE residual vs EC share deciles | Missing | D | A residual-by-decile chart over `improve_samples/` | N | |
| spartan presentation | [6] P1 fAbs vs FTIR EC by SPARTAN site, fits | In gallery · CT Scatterplot per-site, SPARTAN network | D | — | N | MAC– |
| spartan presentation | [8] P2 two additive fixes | Partial · OS `os-shift` (EC+Δ drawn; fAbs − intercept only stated in a Note) | D | A second panel | N | MAC– |
| spartan presentation | [10] P3 implied missing EC by month; season fits | In gallery · OS `os-missing` | D | — | N | MAC– (slope-based) |
| spartan presentation | [12] P4 history reconciliation (TOR 13.7 vs FTIR vs fAbs/10 vs fixes) | Missing | D | Bars from `filters.json` means + OS fit + one constant | N | MAC✗ in notebook; would be MAC✓ |
| spartan presentation | [14] P5 IMPROVE site slope–intercept plane; R² | Missing | D+ | Per-site OLS (≈217 rows; `improve_site_fabs_ec_fits.csv` not tracked), or compute over `improve_samples/` | N | MAC– |
| spartan site_ratios | [6] R1 per-site ratios in mass space | In gallery · CT SiteRank ratios (basis-free) | D | — | N | |
| spartan site_ratios | [8] R2 mass vs conc equivalence | Missing (stated in SI note) | D | A trivial scatter | N | Low value |
| spartan all_sites | [8] fig01 map, public vs HIPS subset | In gallery · MAP (`spartan_map.json`) | D | — | N | |
| spartan all_sites | [10] fig02 site × product coverage matrix | In gallery · DA filter inventory | D | — | N | |
| spartan all_sites | [11] fig03 HIPS filters by site | In gallery · DA / SiteRank n | D | — | N | |
| spartan all_sites | [14] fig04 Fabs distribution by site | In gallery · CT SiteRank / Distribution (SPARTAN) | D | — | N | MAC– |
| spartan all_sites | [15] fig05 focus four in all-site context | In gallery · as [14] | D | — | N | |
| spartan all_sites | [17] fig06 monthly HIPS counts heatmap, all sites | Partial · DA sampling history (one site at a time) | D | A network-wide site × month count heatmap | N | |
| spartan all_sites | [19] fig07 HIPS Fabs vs public BC (= Fabs/10) | In gallery · CT Scatterplot `BC (public HIPS)` | D | — | N | MAC✓: the gallery rescales public BC with the reader's MAC, so "slope 1/10" only reads at MAC 10 |
| spartan orientation | [5] SPARTAN map | In gallery · MAP | D | — | N | |
| spartan orientation | [9] IMPROVE map | In gallery · MAP (`improve_map.json`) | D | — | N | |
| spartan orientation | [15] shared-parameter distributions (log) | In gallery · CT Distribution, both networks | D | — | N | |
| spartan orientation | [17] cross-plots log–log | In gallery · CT Scatterplot | D | — | N | MAC– |
| spartan orientation | [19] derived ratios | In gallery · CT ratios | D | — | N | |
| spartan vs_improve | [11] fig01 distribution boxplots | In gallery · SI `si-dist` (static, focus four) + CT | D | — | N | |
| spartan vs_improve | [12] fig02 ECDFs | Partial · CT has histogram/ridgeline, no ECDF | D | An ECDF chart (no export) | N | |
| spartan vs_improve | [13] fig03 SPARTAN by site vs IMPROVE median | In gallery · CT SiteRank | D | — | N | |
| spartan vs_improve | [16] fig04 relationship bridge (fits) | In gallery · SI `si-fits` table + CT Scatterplot | D | — | N | MAC– |
| spartan vs_improve | [18] fig05 ratio boxplots | In gallery · SI `si-ratios` + CT | D | — | N | |
| spartan mass_comparison | [6] M1 mass boxplots | In gallery · IE `ie-mass` (static) + CT mass basis | D | — | N | |
| spartan mass_comparison | [8] M2 mass ECDFs | Partial · no ECDF | D | ECDF chart | N | |
| spartan mass_comparison | [10] M3 core four vs IMPROVE | In gallery · IE `ie-mass` | D | — | N | |
| spartan mass_comparison | [12] M4 relationship bridge in mass | In gallery · IE `ie-density` (static hexbin) + CT mass scatter | D | — | N | MAC– |
| spartan mass_comparison | [14] M5 volume-free ratio boxplots | In gallery · CT ratios | D | — | N | |
| July07 regular_before_cal0 | [7] reported EC and smoke(lot-251 tool) EC vs HIPS, Addis | Partial · CT EC (FTIR) vs HIPS BC is the "regular" panel | D+ | **Tracked** `tables/addis_regular_before_cal0_vs_smoke.csv` (date, ec_cal0, ec_smoke, hips_fabs): join by date to Addis rows | N | MAC✓ for the regular panel; the smoke panel needs the join |
| July07 regular_vs_smoke | [9] CAL-0 (906 refit) and smoke vs HIPS, n=259 | Partial · CAL preset "Biomass-smoke (906) × raw" vs HIPS (CAL-0's training set, gallery k rule) | D+ | **Tracked** `tables/etad_regular_vs_smoke_ec.csv` (ExternalFilterId, EC_regular, EC_smoke, Fabs) | N | MAC✓ in CAL via `macMath` |
| July07 952 | [11] IMPROVE 952 cal set vs ETAD, raw + ALS-baselined | Partial · CAL SpectraOverlay (Addis vs eth300/analog500/ocec800 cohorts, raw/AIRSpec) | S-ML | Add the 906/952 sets as series in `cohort_spectra` (median/IQR; Drive spectra) | Y | Baseline fits on spectra |
| July07 952 | [13] normalized mean shapes; ETAD homogeneity | Partial · as above | S-ML | As above, area-normalised | Y | |
| July07 906 | [9] 906 cal set vs ETAD, raw + baselined | Partial · as above | S-ML | As above | Y | |
| July07 906 | [11] normalized shapes | Partial · as above | S-ML | As above | Y | |
| deming | [10] Deming slope/intercept vs error ratio | In gallery · OS `os-lambda` | D | — | N | MAC–. Notebook not executed in repo |
| deming | [13] OLS vs Deming + EC shift on one scatter | Partial · OS `os-shift` (table + refit panels, not one overlay) | D | An overlay panel | N | |
| deming | [23] raw / additive / multiplicative re-fits | In gallery · OS `os-shift` | D | — | N | |
| deming | [25] geometry: additive slides, multiplier pivots | Missing | D | A didactic panel from the same points | N | |
| offset_correction | [17] three regressions side by side | In gallery · OS `os-shift` | D | — | N | MAC– |
| offset_correction | [19] geometry | Missing | D | As deming [25] | N | |
| offset_correction | [24] all sites from origin, intercept ◆ | Partial · CT Scatterplot per-site fits | D | Origin-anchored axes + intercept marker | N | |
| offset_correction | [25] per-site panels with equation | In gallery · CT Scatterplot per-site layout | D | — | N | |
| offset_correction | [28] Addis raw vs additive-corrected | In gallery · OS `os-shift` (Raw / Additive) | D | — | N | |
| offset_correction | [30] four sites, Addis corrected, others raw | Partial · OS is one site at a time | D | A multi-site mode | N | |
| offset_correction | [32] combined k and Δ (Δ = k·offset/slope) | Missing | D | A closed-form curve from the OS fit | N | |

## 2. Slice summary

**Counts (149 figures; 1 notebook n/a)**

| | In gallery | Partial | Missing | Total |
|---|---:|---:|---:|---:|
| ftir_etad_char (67) | 0 | 7 | 60 | 67 |
| spartan_ec_2026_06_16 (20) | 5 | 2 | 13 | 20 |
| spartan (45) | 33 | 7 | 5 | 45 |
| July07 (6) | 0 | 6 | 0 | 6 |
| addis_fabs_ec_deming (11) | 5 | 3 | 3 | 11 |
| **All** | **43** | **25** | **81** | **149** |

By class: **D 58**, **D+ 23**, **S 40**, **S-ML 28**. The charcoal archive accounts for 35 of
the 40 S figures (most are mean-spectrum overlays) and 14 of the 28 S-ML figures.

**What changed since the 27 Oct doc**

- New Research pages now cover these items:
  - OffsetShift covers the deming and offset-correction notebooks and presentation P2/P3. The doc said "the EC shift that zeros the intercept is not there"; it now is, dynamically.
  - SpartanImprove covers vs_improve, follow_up and site_ratios.
  - ImproveEnvelope covers mass_comparison M1/M3/M4.
  - Applicability covers char_12's escape, lot and raw-scale tables.
  - EthiopiaCalibrations covers spartan_ec 08.
- I **disagree** with the doc on three entries:
  - spartan_ec 05/07 is listed "In gallery". Only the generic CV curve and K sweep exist, for the gallery's own presets. None of the K-instability, repeated-CV, ensemble or Adama figures are there.
  - spartan_ec 02 is listed "In gallery". The lot-251 tool biomass calibration is in no export.
  - July07 is listed "In gallery". It is Partial.
- **Naming hazard:** the gallery's "Biomass-smoke (906)" preset is the 906-filter IMPROVE set. July07 calls that set "regular CAL-0", per the lineage note in `research/ftir_ec_phase3/scripts/build_notebooks.py`. July07's and 02's "smoke/biomass" is a different model: SPARTAN's tool coefficients for lot 251 (`tool_EC_coeffs_lot251_biomass.csv`).

**Five highest-value Missing/Partial figures**

1. **Per-filter charcoal similarity scores** (char_05 [14][18], char_06 [8][20], char_07 [7][9], char_08 [6][16], char_09 [8][10]; 10 figures).
   - A single per-filter export unlocks them all: FilterId, `r_char`, `r_raw`, `delta`, `component`, LotId, plus r per reference class / feedstock. char_06 already writes the core of it as `output/tables/char06/per_filter_classification.csv`, but that file is ignored and needs a Drive re-run.
   - Joined to `filters.json`, these become D: they follow the gallery seasons, exclusions and the EC/Fabs loading check (MAC-free).
2. **spartan_ec 05/07 K-instability story** (7 S-ML figures, PNG only).
   - This is the evidence that "optimal K" is unstable and that the Adama EC swings with K.
   - Needs a re-run that saves: RMSEP by k × CV seed, mean ± sd RMSEP by k under repeated CV, Adama EC by k, and the per-approach spread.
3. **July07 + spartan_ec 02: lot-251 tool biomass calibration vs HIPS at Addis** (3 figures).
   - The data is tracked; this answers "does smoke calibration help" and would follow the MAC.
4. **presentation P5 + P4** (the IMPROVE slope–intercept plane, and the history reconciliation).
   - P4 is D now, with no export.
   - P5 needs a ~217-row per-site fit table, from the FED cache or from `improve_samples/`.
5. **char_12 per-filter escape + envelope spectra** (figs [5][7][9][11][13]).
   - Upgrades the Applicability page's static group tables. Per-filter escape with date/lot makes the season story dynamic, and mean/p05/p95 envelopes per wavenumber give the Weakley Figure 5 itself.

**Cheap wins (D+ with data already tracked, or D needing no export)**

- `research/July07/tables/addis_regular_before_cal0_vs_smoke.csv`: join on date for the Addis smoke and CAL-0 EC.
- `research/July07/tables/etad_regular_vs_smoke_ec.csv`: join on ExternalFilterId.
- `research/spartan_ec_2026_06_16/tables/biomass_vs_general_ec.csv`: FilterId, EC_general, EC_biomass.
- Tracked tables of 5 rows each: `adama_char_soot_classification.csv`, `adama_ec_optimized_comparison.csv`, `adama_ec_stabilized_comparison.csv`. These are static, but trivial to add next to EC6's Adama table.
- spartan_ec 08 residual diagnostics: `calibration_all` is already exported, so this needs only a chart.
- D with no export at all:
  - presentation P4.
  - The deming/offset geometry and combined-k-and-Δ panels.
  - An ECDF chart (vs_improve fig02, mass M2).
  - A network-wide site × month count heatmap.
  - A τ field (Fabs·V/A).
  - The fig14 within-IMPROVE residual-by-decile chart. This one needs `improve_samples/`.

## 3. ML figures, as proposed ML-section pages

| Proposed page | Figures | Data it would load | Exists today? |
|---|---|---|---|
| **A. Calibration stability: components, CV noise, cleaning** | spartan_ec 05 [7][10][15]; 07 [3][6][11][15]; 08 [3][5][7] | (1) RMSEP by k × seed and by repeated CV; (2) Adama EC by k and by approach; (3) 906 measured/predicted/round/status; (4) cleaned-refit predictions | Partly. (3) is in `research/spectral/ethiopia_calibrations.json` (`calibration_all`, 906 rows, recomputed from Drive; the fallback is the 61 tracked rows), with `cleaning_rounds`/`cleaning_fit`. (1), (2) and (4) are PNG-only and need a re-run of the `_build_05/_build_07` scripts on the tool's training data. Could extend EC6 rather than be a new page |
| **B. Charcoal archive: unsupervised structure** | char_04 [5][7][9][11][13][15]; char_02 [18][20]; char_11 [21] | PCA scores (PC1–3) per spectrum (~5,889 archive + ~239 Addis) with collection/T/source; KMeans labels for passes A/B; silhouette by k; Ward linkage; per-filter Mahalanobis; transform benchmark table | No. All in ignored `research/ftir_etad_char/output/`; inputs are `charcoal_ftir_sources/downloads/` (ignored) + `ftir_ec_phase3/output/corrected/etad_corrected_df6.npz` (ignored). Only embedded PNGs and some displayed summary tables are tracked. Regenerate with `scripts/run_char_02/04/11.py` |
| **C. Charcoal analog matching & classifiers** | char_02 [15]; char_03 [16]; char_06 [11]; char_11 [17][24] | `addis_analogue_matches.csv` (per filter: best analogue, implied T); recovery check; `addis_burned_scores.csv` + `loo_feedstock_cv.csv`; Gaussian-mixture `component` per filter (`per_filter_classification.csv`); AIRSpec fit traces for one charcoal and one Addis spectrum; `addis_conclusion_by_transform.csv` | No (ignored output/). Pairs naturally with the D+ per-filter scores in §2 item 1: one export (FilterId-keyed) would also unlock 10 dynamic figures |
| **D. Calibration-set spectra vs Addis (baseline-corrected)** | July07 952 [11][13], 906 [9][11] (+ the S envelope figures char_12 [5][7]) | Median/IQR (or mean/p05/p95) spectra of the 906 and 952 IMPROVE calibration sets and ETAD, raw and baselined; area-normalised shapes | Partly. `calibration_runs.json` `cohort_spectra` already carries Addis vs three cohorts (raw/AIRSpec/VIBES-r70) in this exact shape; the 906 ("Biomass-smoke") and 952 sets would be two more series. The char_12 envelopes are not exported |

Not ML-section material:
- The 35 S spectral-overlay figures in char_01/03/05/07/08/09. These are group means; a
  non-ML "Charcoal reference spectra" page loading one small export of group-mean spectra would
  cover them.
- The D/D+ figures, which belong in the existing Research pages and chart tabs.
