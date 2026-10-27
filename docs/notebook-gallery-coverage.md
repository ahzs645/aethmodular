# Notebooks vs the gallery: what has been ported and what has not

Snapshot of 27 Oct 2026. It compares the 192 active notebooks (223 in the repo,
less `archive/`, `scratch/` and `attic/`) with what `gallery/app` draws today:
the chart tabs, Sites (Map, Data availability, Weather, PMF), Calibration,
Baselines, Spectral similarity, Blanks, the four Meeting views and the hidden
Experimental pages.

Status key:

- **In gallery**: the gallery answers the notebook's question with its own data.
- **Partial**: the data is in the gallery and a generic chart can draw it, but the
  notebook's specific analysis (its fit, threshold, decomposition) is not there.
- **Not in gallery**: the gallery has none of the data the notebook uses.
- **n/a**: launchers and infrastructure, with nothing to port.

## The main finding

The chart tabs (Correlation, Distribution, Evolution, Ranking, Part of a whole)
read `filters.json`, which holds 28 fields: FTIR EC/OC/OM and functional groups,
HIPS Fabs/BC, PM2.5 mass, ions and metals. Four kinds of data that many
notebooks are built on are missing from the gallery:

1. **Aethalometer (MA350) data**: BCc at five wavelengths, AAE, MAC, the
   smoothing and flow-fix variants, and hourly or diurnal data. The gallery has
   only the coverage counts on Data availability. This is the largest gap,
   about 30 notebooks.
2. **AERONET and columnar data**: AOD, AAOD, SSA and the surface–columnar
   linkage. About 9 notebooks.
3. **The charcoal and biochar reference spectra** (`research/ftir_etad_char`).
   All 12 notebooks.
4. **TOR sub-fractions beyond EC/OC** (char-EC/soot-EC, TOT, pyrolysis): only
   the Adama TOR study and the 28 Oct item 7 touch them.

The FTIR EC calibration-transfer work is the best covered part of the repo.
Calibration, Baselines, Similarity, Blanks, Meetings and Experimental cover
most of `ftir_ec_phase3`, `ftir_ec_calibration_2026_06_25` and the dated meeting
notebooks. The gaps there are individual side studies, listed under G below.

## A. Four-site filter chemistry: crossplots, distributions, seasons

Gallery: Correlation / Distribution / Evolution / Ranking / Part of a whole,
over the four focus sites plus SPARTAN and IMPROVE. OLS/Deming fits, colour by
a third variable, groupings (iron median split, Fabs deciles, etc.).

| Notebook | Status | What is missing |
|---|---|---|
| `research/ftir_hips_chem/Analysis_Tasks_Jan2025` | Partial | The aethalometer panels |
| `research/ftir_hips_chem/Task_Analysis_Notebook` | Partial | The aethalometer panels |
| `research/ftir_hips_chem/primary_tasks_notebook` | Partial | The aethalometer panels |
| `research/ftir_hips_chem/follow_up_analysis` | Partial | The aethalometer panels |
| `research/ftir_hips_chem/Example_Modular_Analysis` | Partial | Framework demo |
| `research/ftir_hips_chem/warren_meeting_slides` | Partial | The BCc and MAC slides |
| `research/ftir_hips_chem/FTIR_Group_Talk_Apr2026_Draft` | In gallery | Iron split is a grouping (pooled rather than per site) |
| `research/ftir_hips_chem/results_summary` | Partial | |
| `research/ftir_hips_chem/cross_plots_and_distributions` | Partial | The AERONET filtering |
| `research/ftir_hips_chem/hips_offset_narrative` | In gallery | The Addis-vs-Delhi grouping is cited |
| `research/spartan_ec_2026_06_16/04_new_plots` | In gallery | FABS/EC, EC/OC and fractional EC by site are all drawable |
| `notebooks/analysis/absorption/warren_ratio_diagnostics` | Partial | The Warren ratio as a derived field |
| `research/addis_fabs_ec_deming/addis_fabs_ec_deming` | Partial | Deming is there; the "EC shift that zeros the intercept" is not |
| `research/addis_fabs_ec_deming/addis_fabs_ec_offset_correction` | Partial | The offset-correction comparison of three regressions |

## B. Aethalometer, the MA350 chain, MAC and source classification: not in gallery

None of these can be drawn: `filters.json` has no BCc, no AAE and no hourly data.

- Multi-site BCc and filter matching: `Multi_Site_Analysis`, `Multi_Site_Analysis_Fixed`,
  `Multi_Site_Analysis_FollowUp` (Multi_Site_Analysis's coverage part is on Data availability)
- Flow fix: `FlowFix_BeforeAfter_Analysis`, `flow_fix_explorer`, `notebooks/qc/flow_qc_analysis`
- Smoothing and raw data: `HIPS_Aeth_SmoothRaw_Analysis`, `smoothing_comparison`, `Raw_ATN_Correlation_Analysis`
- HIPS vs aethalometer: `HIPS_vs_Aethalometer_Optical_Comparison`, `notebooks/analysis/absorption/AAARpos`
- Diurnal and multi-wavelength: `addis_05_diurnal_wavelength_analysis`, `multisite_diurnal_wavelength_analysis`,
  `notebooks/analysis/diurnal_pub_figures`
- Addis temporal patterns: `addis_02_temporal_patterns`, `addis_02_temporal_patterns_daily`,
  `addis_02.5_temporal_patterns_filter` (the filter part is partial, through Evolution)
- MAC and iron at ETAD: `notebooks/analysis/absorption/ETAD_Aug13`, `ETAD_comprehensive_absorption_analysis`,
  `notebooks/analysis/iron_correction_test`
- AAE source classification and BC/EC by source: `addis_ababa_source_analysis`, `source_apportionment_regression`,
  `threshold_by_source_analysis`, `dominant_source_comparison`, `dominant_source_threshold_analysis`
- MA350 in phase 3: `ftir_28_ma350_brc_falsification`, `ftir_49_ma350_raw_chain_diagnostics` (Jacros)
- `notebooks/plotting_gaps_scenarios`, a gap-handling scenario gallery

## C. PMF source apportionment (ETAD factors)

Gallery: the **PMF sources** tab (stacked area / streamgraph, seasonality).

| Notebook | Status | What is missing |
|---|---|---|
| `ETAD_Factor_Analysis` | In gallery | It is the source of `pmf.json` |
| `figure7_source_contributions` | In gallery | |
| `addis_01_source_apportionment`, `addis_01_source_apportionment_daily` | Partial | Factor × aethalometer/HIPS/FTIR regressions |
| `notebooks/analysis/meteorology/meteorology_source_interaction` | Partial | The meteorology bins are groupings; factor × meteorology is not |

## D. Meteorology

Gallery: the **Weather** tab (ERA5 at every SPARTAN site, the Addis airport
station, wind roses, rain washout, humidity × method agreement, weather ×
chemistry correlations). `lib/groupings.ts` cites several of these notebooks
as the source of its bins.

| Notebook | Status | What is missing |
|---|---|---|
| `addis_03_meteorology`, `addis_03_meteorology_daily` | In gallery | The BC (aethalometer) side |
| `notebooks/analysis/meteorology/meteorology_revisions` | In gallery | |
| `.../Seasonalclass`, `.../map` | In gallery | Season calendar and site map |
| `.../temperature_gradient_analysis` | Partial | ΔT is a grouping only, and uses the window mean rather than the daily minimum |
| `.../hips_ftir_intercept_rh_recursive_diagnostics` | Partial | The RH split is there; the recursive threshold search is not |
| `.../rh_sweep_seasonal` | Not in gallery | The season-stratified RH threshold sweep |
| `.../hips_intercept_investigation` | Partial | The drivers-of-offset regressions |
| `.../friday_summary_consolidated` | Partial | |
| `.../pw_mac_calibration` | Not in gallery | Precipitable water against MAC |

## E. AERONET and columnar: not in gallery

`addis_04_aeronet`, `addis_04_aeronet_daily`, `notebooks/analysis/meteorology/aeronet_addis_deep_dive`,
`aeronet_addis_diagnostics`, `aeronet_method_agreement`, `notebooks/analysis/surface_columnar_linkage`,
`surface_columnar_figures`. Boundary-layer height does appear, as a grouping and
as 28 Oct item 11. AOD, AAOD and SSA do not.

## F. HIPS offset, and SPARTAN against IMPROVE

Gallery: the Map (SPARTAN and IMPROVE), the IMPROVE network in the chart tabs
(TOR EC/OC, HIPS, mass, iron, soil), Data availability, the HipsYork chart on
the Calibration tab, Ann 9 Oct §3 (mass per filter), and 28 Oct §6 (loading)
and §7 (TOR on dark filters).

| Notebook | Status | What is missing |
|---|---|---|
| `research/spartan/spartan_all_sites_hips_overview` | In gallery | Data availability |
| `research/spartan/spartan_improve_orientation` | In gallery | Data availability, prior comparison |
| `research/spartan/spartan_vs_improve_comparison`, `spartan_vs_improve_mass_comparison` | Partial | Both networks can be drawn together; the shared-parameter panels as a set cannot |
| `research/spartan/site_ratios_mass_space` | Partial | Ann 9 Oct §3 does this for Addis only |
| `research/spartan/follow_up_mass_and_ratio_plots` | Partial | The ETBI and iron panels |
| `research/spartan/presentation_offset_story` | Partial | |
| `research/improve_hips_offset/anne_spartan_improve_ec_mass_fabs` | Partial | |
| `improve_high_fabs_comparison` | In gallery | The Fabs high-tail and decile groupings |
| `improve_hips_offset/improve_smoke_event_qc` | Partial | The OC/EC smoke threshold is a grouping; the QC itself is not |
| `improve_hips_offset/etad_vs_improve_ec_matched_comparisons` | Partial | EC-matched pairing |
| `improve_hips_offset/improve_addis_analog_audit` | In gallery | Similarity, Calibration analog lab |
| `improve_hips_offset/improve_first_order_loading_range_analysis`, `next_tests_2026_04/00_fed_addis_loading_envelope` | Partial | 28 Oct §6 |
| `improve_hips_offset/improve_hips_offset_narrative` | Partial | |
| `improve_hips_offset/improve_fed_rt_proxy_figure2`, `improve_white_style_rt_calibration_space` | Not in gallery | The RT proxy and the White-style calibration space |
| `improve_hips_offset/improve_spartan_may_full_analysis`, `warren_cena_improve_prep_analysis` | Partial | |
| `improve_hips_offset/improve_spartan_may_prep_task_map` | n/a | A task map with no figures |
| `notebooks/analysis/meteorology/addis_hips_two_problem_decomposition`, `hips_two_problem_decomposition` | Not in gallery | The baseline + sensitivity decomposition |
| `addis_06_hips_ptfe_operating_envelope` | Not in gallery | |
| `ftir_26_improve_hips_origin` | Partial | IMPROVE TOR EC vs HIPS can be drawn; the origin fit is not there |
| `ftir_37_hips_loading_and_curvature` | Partial | 28 Oct §6–7 |
| `ftir_47_blank_lines_per_deployed_line` | In gallery | HipsYork ("does the offset move with the blank line?") |

## G. FTIR EC calibration transfer, baselines and spectra

Gallery: Calibration (cohorts, cutoff sweep, k sweep, CV curve, specification
curve, slope/intercept trap, cross-site transfer, analog lab, cohort overlap,
blind-half check), Baselines, Spectral similarity, Blanks, Meeting (17 Sep,
Satoshi 2 Oct, Ann 9 Oct, 28 Oct) and Experimental (22 Oct tests 1–6, quick
fixes, lot-aware VIBES, Adama TOR study).

**In gallery**

- The dated meeting notebooks: `meeting_followup_20260917` (Follow up meeting view),
  the Satoshi 2 Oct workflow (Satoshi view), `ann_followup_20261009` (Ann 9 Oct),
  `ann_followup_20261021` (28 Oct meeting), `ksweep_20261014` (Ann 9 Oct §2, KSweep),
  `decision_workbook_20261022` and `experiments_20261022` (Experimental),
  `ann_weekly_20260910_figures` and `ann_weekly_20260910_diagnostics` (baseline comparison)
- Baselines: `ftir_vibes_vs_airspec`, `vibes_followup_experiments`, `ftir_13_airspec_corrected_calibrations`
- Cohorts and selection: `ftir_09_improve_analog_selection`, `ftir_10_ec_calibration_cohort_comparison`,
  `ftir_11_ocec_ratio_cohort`, `ftir_33_shape_cohort_explainers`,
  `July07/regular_vs_smoke_ec_vs_hips`, `July07/regular_before_cal0_vs_smoke_hips`,
  `spartan_ec_2026_06_16/02_biomass_calibration_comparison`
- Components, protocols and the search: `ftir_ec_calibration_2026_06_25/05`, `06`, `09_component_selection_fixed`,
  `10_calibration_variants_effect`, `11_hips_intercept_calibrations`, `14_choosing_calibration_data`,
  `ftir_18_transfer_roundup`, `ftir_20`, `ftir_21`, `ftir_22`, `ftir_23`, `ftir_55_every_combination_tried` (specification curve),
  `spartan_ec_2026_06_16/05_calibration_optimization`, `07_calibration_optimization_story`
- Spectra: `July07/spectra_side_by_side_*`, `ftir_ec_calibration_2026_06_25/01_adama_spectra`,
  `04_ethiopia_etad_spectra`, `ftir_17_deck_figures_and_seasonal_spectra`
- Adama: `ftir_41_adama_three_method_reconciliation`, `ftir_44_adama_through_locked_calibrations`,
  `ftir_ec_calibration_2026_06_25/02_adama_tor_vs_ftir`, `08_adama_hips_crossplots`
  (Baselines external pilot, Experimental TOR study)
- VIP: `ftir_07_current_pls_vip_diagnostics`, `ftir_08_hips_transfer_and_vip` (28 Oct §8, Satoshi VIP)

**Partial**

| Notebook | What the gallery has | What is missing |
|---|---|---|
| `ftir_12_band_1600_identity`, `ftir_38_1617_and_dust_attribution` | 28 Oct §3 (dust), Ann 9 Oct §4 (Si-O band) | The 1600/1617 cm⁻¹ band attribution |
| `ftir_16_mac_decision_prep`, `ftir_19_mac_effect_on_calibrations` | 28 Oct §4 (MAC by season) | MAC 6 vs 10 applied across every calibration |
| `ftir_29_provisional_addis_ec_series` | RunSeries (dated series) | The provisional series as a product |
| `ftir_30_composition_case_and_its_limits` | 28 Oct §10 (composition) | |
| `ftir_36_domain_invariant_pls`, `ftir_39_pls_applicability_domain` | Baselines research diagnostics (distance from training) | Domain-invariant PLS |
| `ftir_50_spectral_comparison_methods` | Similarity (Pearson r) | T²/Q, hubness, band-resolved analogs |
| `ftir_52_network_spectral_map` | Similarity, site-level | Ward clusters over the IMPROVE network |
| `char_12_weakley_transferability` | Cross-site transfer | The Weakley-style audit |
| `ftir_ec_calibration_2026_06_25/07_ethiopia_variant_ec`, `09_ethiopia_component_ec` | Calibration run views | |
| `ftir_ec_calibration_2026_06_25/10_ethiopia_ec_both_lots` | Experimental, lot-aware VIBES | |
| `ftir_ec_calibration_2026_06_25/12_dedicated_ethiopia_calibration` | | A local-only calibration |
| `ftir_ec_calibration_2026_06_25/15_downstream_recommended_calibration` | | |
| `spartan_ec_2026_06_16/08_outlier_removal_explainer` | Exclusion toggle and reasons | The explainer itself |
| `ftir_31`, `ftir_51`, `ftir_53` (deck, priority and committee figures) | Most panels individually | The committee "does the diagnosis generalise" figures |

**Not in gallery**

- Charcoal reference archive: `ftir_etad_char/char_01` … `char_11` (survey, Addis vs charcoal,
  burned vs unburned, typology, unburned feedstock, dry-season anomaly, fire vs furnace,
  seasonal spectra, normalisation alternatives)
- Char-EC / soot-EC (Han et al.): `ftir_ec_calibration_2026_06_25/03_adama_char_soot`,
  `spartan_ec_2026_06_16/03_adama_han_char_soot`, `01_carbon_methods_audit`
- Target definition and pyrolysis: `ftir_42_target_definition_experiment`, `ftir_48_pyrolysis_split_target`
- Residual learner: `ftir_43_residual_learner_null_control`, `ftir_45_residual_increment_attribution`,
  `ftir_46_paired_increment_and_mass`
- Robustness: `ftir_15_uncertainty_and_hybrid`, `ftir_32_cv_scheme_ablation`, `ftir_54_model_form_robustness`
- `ftir_27_chemspec_circularity`
- `ftir_34_offset_adjudication_checks`, `ftir_35_variation_closure`
- `ftir_40_quartz_tor_campaign_design`
- `ftir_ec_calibration_2026_06_25/13_chase_origin_and_biomass` (home-grown biomass classifier)
- `addis_public246_20261020` (FTIR EC on all 246 public SPARTAN filters)
- `filter_only_diagnostics`, `filter_proportionality`, `filter_relationship_stability`,
  `deliverables/filter_only_scientific_release_2026-09-11/notebooks/filter_only_results` (site-dependent
  proportionality and temporal transfer)
- `matched_sample_audit_figures`

## n/a

`calibration_explorer/colab/Calibration_Explorer_Colab`, `research/ftir_hips_chem/colab/VIBES_AIRSpec_Large_Run_Colab`,
`research/ftir_hips_chem/workflows/data_space_recipes/hosted_lab_check`, and the
`.executed` copies of notebooks listed above.

## The other direction: in the gallery but not in a notebook

The Census page lists chart types the gallery introduced: the Sankey (Flow),
treemap, circle pack and donut (Part of a whole), the seasonal clock, ridgeline,
parallel coordinates, radar, the species network and connected scatter.

## How this was made

The figures come from `gallery/census/chart_census.json` (211 notebooks;
the census predates about 12 notebooks, so the 0-figure counts for newer ones
mean "not scanned" as well as "draws through a workflow"). They were matched
against `gallery/app/src/chartIndex.json`, the meeting specs, the experiment
index, the `.ipynb` citations in `gallery/app/src` and `gallery/data`, and the
field list in `public/data/meta.json`. The statuses are judgements from the
notebooks' titles, figure titles and cited sources, not a cell-by-cell
comparison. Rebuild the census (`python gallery/census/build_census.py`)
before relying on the counts.
