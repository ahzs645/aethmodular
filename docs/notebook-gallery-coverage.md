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
   linkage. About 9 notebooks. **Now in the gallery** (Sites → AERONET, and
   AERONET fields in every chart tab) for the 35 SPARTAN sites with a station:
   see section E.
3. **The FTIR spectra learning studies** (section H): the charcoal and biochar
   reference spectra (`research/ftir_etad_char`), and the phase-3 side studies
   on target definition, the residual learner and robustness.

Section I covers what the MAIA portal holds, and section J recommends where
each new data type should go in the gallery.

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

## E. AERONET and columnar: in the gallery since 27 Oct 2026

`addis_04_aeronet`, `addis_04_aeronet_daily`, `notebooks/analysis/meteorology/aeronet_addis_deep_dive`,
`aeronet_addis_diagnostics`, `aeronet_method_agreement`, `notebooks/analysis/surface_columnar_linkage`,
`surface_columnar_figures`.

**Sites → AERONET** now carries these, laid out like Weather. It has two views:

- **One site**: the daily AOD, fine-mode fraction, AAOD and SSA above the
  measurement, then the measurement against any AERONET quantity, then the
  scale height by season.
- **Several sites**: a station table, the absorption scale height
  `H = AAOD 675 / Fabs` by site, and the same scatter.

Sixteen AERONET fields, averaged over each filter's sampling window, join every
chart tab's pickers as the group "AERONET (sampling window)".

Partial still: the notebooks' boundary-layer decoupling and the
surface–columnar regressions, and Level 2.0 data (`--level 20` pulls it).

The pipeline:

1. `research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py` ranks the
   stations within 50 km of each site by filter days covered. It downloads the
   best one by year: direct-sun AOD, SDA and almucantar inversions, each as
   every retrieval and as daily means. It keeps the raw text and parses it to
   parquet. On 27 Oct 2026 that was 35 stations: 2.8 GB downloaded, 397 MB
   gzipped on disk. The rows were 1.51 M AOD retrievals, 1.47 M SDA retrievals
   and 101 k inversions, plus the daily means.
2. `gallery/data/export_aeronet.py` averages the individual retrievals inside
   each filter's sampling window. It uses the windows from the weather export,
   in local standard time. That gave 3,094 filters with a value, against 2,636
   from matching on the start date alone. Plus daily series per site: 3.3 MB in
   `public/data/aeronet/`.

Over all filters in the export, the median `H` is 256 m at Addis, 515 m at
Delhi, 591 m at Pasadena, 623 m at Bishoftu and 1,081 m at Beijing. That
matches the August finding (`AERONET_SPARTAN_2026-08-22.md`: Addis is the
low outlier).

### What AERONET data the repo had before

- **Addis**: `AAU_Jackros_ET`, Level 1.5 almucantar inversion, daily, 2022–2025, on
  the Drive (`aeronet.aeronet_dir()`). Every Addis notebook reads it from there.
- **Beijing, Delhi, Pasadena**: `research/ftir_ec_phase3/scripts/run_aeronet_spartan.py --pull`
  downloads them from the AERONET service into `output/tables/aeronet/`, which is
  gitignored. They were used once, in `AERONET_SPARTAN_2026-08-22.md`, and are not
  in this checkout.
- The committed `notebooks/analysis/data/aeronet_aod_{beijing,delhi,jpl}.csv` each
  hold **one day** (2026-05-13), so they cannot be matched to filters. The archived
  `research/archive/catch_up/04_aeronet_multisite_availability_overlap.ipynb`
  found 0 matched filter days with them for that reason.
- No AERONET data is in the repo for any other SPARTAN site.

### Does AERONET overlap the SPARTAN sites?

Yes, at nearly all of them. SPARTAN sites were placed beside AERONET
photometers, so the nearest station is usually 0–5 km away. On 27 Oct 2026
each SPARTAN site was matched to the AERONET V3 site list and the AERONET web
service was asked for daily Level 1.5 values over the site's filter period. The
first column counts direct-sun AOD; the second counts the almucantar inversion,
which gives absorption AOD and SSA. A filter day counts as matched when AERONET
has a daily value on that calendar day (the filter's start date). Bold rows are
the four focus sites.

| Site | Nearest useful AERONET station | km | Filter days | Days with daily AOD | Days with AAOD (inversion) |
|---|---|---:|---:|---:|---:|
| CHTS **Beijing** | Beijing | 5.6 | 402 | 212 (53 %) | 152 (38 %) |
| ETAD **Addis Ababa** | AAU_Jackros_ET | 0.1 | 247 | 192 (78 %) | 126 (51 %) |
| USPA **Pasadena** | CalTech (MISR-JPL is 0.1 km but has fewer days) | 8.2 | 266 | 184 (69 %) | 144 (54 %) |
| INDH **Delhi** | IIT_Delhi | 0.0 | 247 | 131 (53 %) | 107 (43 %) |
| ILNZ Rehovot | Weizmann_Institute | 0.0 | 296 | 220 (74 %) | 202 (68 %) |
| ZAPR Pretoria | Pretoria_CSIR-EC | 0.7 | 309 | 139 (45 %) | 118 (38 %) |
| CLST Santiago | Santiago_Beauchef | 0.3 | 211 | 123 (58 %) | 68 (32 %) |
| ILHA Haifa | Technion_Haifa_IL | 0.0 | 191 | 123 (64 %) | 108 (57 %) |
| KRUL Ulsan | KORUS_UNIST_Ulsan | 0.0 | 170 | 122 (72 %) | 96 (56 %) |
| CAHA Halifax | Halifax | 0.0 | 173 | 99 (57 %) | 76 (44 %) |
| TWKA Kaohsiung | Kaohsiung | 3.6 | 175 | 99 (57 %) | 76 (43 %) |
| INJA Jaipur | Jaipur | 0.0 | 106 | 97 (92 %) | 87 (82 %) |
| CASH Sherbrooke | CARTEL | 0.0 | 155 | 91 (59 %) | 48 (31 %) |
| KRSE Seoul | Yonsei_University | 0.0 | 99 | 73 (74 %) | 58 (59 %) |
| IDBD Bandung | Bandung | 0.0 | 161 | 68 (42 %) | 34 (21 %) |
| MXMC Mexico City | Mexico_City | 0.1 | 101 | 68 (67 %) | 38 (38 %) |
| PRFJ Fajardo | Cape_San_Juan | 0.0 | 103 | 61 (59 %) | 43 (42 %) |
| BDDU Dhaka | Dhaka_University | 0.0 | 108 | 49 (45 %) | 31 (29 %) |
| NGIL Ilorin | Ilorin | 0.0 | 81 | 49 (60 %) | 35 (43 %) |
| AEAZ Abu Dhabi | Masdar_Institute | 0.0 | 171 | 45 (26 %) | 37 (22 %) |
| CADO Downsview | Toronto | 0.0 | 70 | 43 (61 %) | 27 (39 %) |
| TWTA Taipei | Taipei_CWB | 4.6 | 295 | 41 (14 %) | 25 (8 %) |
| AUMN Melbourne | Aspendale_Mel_AU | 28.1 | 110 | 37 (34 %) | 26 (24 %) |
| ETBI Bishoftu | AAU_Jackros_ET (the Addis station) | 34.3 | 60 | 36 (60 %) | 24 (40 %) |
| USNO Norman | OU_NWC | 0.0 | 51 | 34 (67 %) | 25 (49 %) |
| SGSU Singapore | Singapore | 0.0 | 50 | 33 (66 %) | 14 (28 %) |
| INKA Kanpur | Kanpur | 0.0 | 43 | 28 (65 %) | 21 (49 %) |
| ARCB Buenos Aires | CEILAP-BA | 0.0 | 43 | 23 (53 %) | 17 (40 %) |
| USMC Mammoth Cave | IMPROVE-MammothCave | 0.0 | 19 | 17 (89 %) | 11 (58 %) |
| CALE Lethbridge | Univ_of_Lethbridge | 0.0 | 31 | 11 (35 %) | 8 (26 %) |
| PHMO Manila | Manila_Observatory | 0.0 | 12 | 9 (75 %) | 6 (50 %) |
| USBO Bondville | BONDVILLE | 0.0 | 11 | 7 (64 %) | 4 (36 %) |
| BIBU Bujumbura | Bujumbura | 0.0 | 57 | 4 (7 %) | 0 |
| VNHN Hanoi | NGHIA_DO | 0.0 | 27 | 4 (15 %) | 0 |
| CAKE Kelowna | Kelowna_UAS | 0.0 | 18 | 3 (17 %) | 0 |
| USSL St Louis | St_Louis_WashU: no data in the window | 0.4 | 42 | 0 | 0 |
| ZAJB Johannesburg | Joberg, Wits_University: no data in the window | 3.0 | 214 | 0 | 0 |

There is no station within 50 km of CLTA Talca. CODC Palmira and USBA Baltimore
have no filters in the gallery export.

Notes:

- Level 1.5 is cloud-screened but not final-calibrated. The repo's own audit
  (`DATA_OPPORTUNITY_AUDIT_2026-08-24.md`) notes that no Level 2 set is in hand.
  Level 1.5 AAOD at low AOD is uncertain.
- A filter is matched only on its start date. Recent ETAD and ETBI filters are
  single 24-h days, so that works for them. Older SPARTAN filters integrated
  over several days, so a one-day match is only approximate there.
- **Size as a gallery dataset** (the first estimate; the built export is above). A prototype was built like the Weather export,
  with daily values per site plus one row per filter on matching days. Fields:
  AOD 440/500/675/870, AE 440–870, precipitable water, fine-mode AOD and
  fraction at 500, AAOD 440/675/870, AAE 440–870, and SSA 440/675. It came to
  **3.9 MB** (1.2 MB gzipped): 3.4 MB of daily series (35 sites, 32,093 days)
  and 0.6 MB for 2,636 matched filters. Weather is 4.6 MB. The raw daily
  downloads it is built from are 53 MB (direct-sun 14 MB, SDA 8 MB,
  inversion 32 MB). That cache belongs in a gitignored `output/` folder, as the
  ERA5 parquet does. The all-points (per-retrieval) files are 1.8 GB and are
  not needed for daily or filter-day values.
- Made with `research/ftir_ec_phase3/scripts/aeronet_spartan_overlap.py`, using
  `https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt` and
  `cgi-bin/print_web_data_v3` (`AOD15=1`) and `print_web_data_inv_v3`
  (`product=TAB`, `ALM15=1`), with `AVG=20` (daily).

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

**Not in gallery**: the notebooks in section H, plus
`ftir_34_offset_adjudication_checks`, `ftir_35_variation_closure`,
`ftir_ec_calibration_2026_06_25/13_chase_origin_and_biomass` (home-grown biomass classifier)
and `matched_sample_audit_figures`.

## H. FTIR spectra learning: not in gallery

These studies ask what the spectra and the PLS model learn, rather than which
calibration to use. None of them is in the gallery.

### The charcoal and biochar reference archive (`research/ftir_etad_char`)

The gallery has no reference spectra. Its spectra are SPARTAN and IMPROVE filters only.

| Notebook | Question |
|---|---|
| `char_01_reference_spectra_survey` | What is in the charcoal/biochar archive, collection by collection |
| `char_02_addis_vs_charcoal` | Addis filter spectra against the archive |
| `char_03_burned_vs_unburned` | Burned versus unburned, and where Addis falls between them |
| `char_04_spectral_typology` | What types of spectra there are |
| `char_05_addis_vs_unburned_feedstock` | Addis against unburned feedstock |
| `char_06_dry_season_anomaly` | Isolating the dry-season anomaly |
| `char_07_char05_without_dry_season` | char_05 with the dry season excluded |
| `char_08_fire_vs_furnace_char` | Fire vs furnace char, woody vs herbaceous |
| `char_09_char08_dry_vs_nondry` | char_08 with the dry season split out |
| `char_10_seasonal_spectra` | The three Ethiopian seasons, spectrum by spectrum |
| `char_11_normalization_alternatives` | Alternatives to SNV for overlaying spectra |

(`char_12_weakley_transferability` is partly covered by Cross-site transfer; see G.)

### What counts as EC: the reference and target definition

| Notebook | Question |
|---|---|
| `ftir_ec_calibration_2026_06_25/03_adama_char_soot`, `spartan_ec_2026_06_16/03_adama_han_char_soot` | Char-EC vs soot-EC (Han et al.) at Adama, OPR and OPT |
| `spartan_ec_2026_06_16/01_carbon_methods_audit` | What "thermal" EC is in SPARTAN |
| `ftir_42_target_definition_experiment` | Calibrating to TOR EC, TOT EC, OC and total carbon |
| `ftir_48_pyrolysis_split_target` | Predicting the pyrolysis split itself |
| `ftir_27_chemspec_circularity` | The ChemSpec carbon columns are circular with both axes |
| `ftir_40_quartz_tor_campaign_design` | Design of a seasonal paired quartz/TOR campaign |

### What the model learns beyond PLS

| Notebook | Question |
|---|---|
| `ftir_43_residual_learner_null_control` | What a residual learner learns (the f(X)-preserving null) |
| `ftir_45_residual_increment_attribution` | Where its increment lives: season, char anomaly or bands |
| `ftir_46_paired_increment_and_mass` | The paired increment test, and the mass lead |

### Robustness

| Notebook | Question |
|---|---|
| `ftir_15_uncertainty_and_hybrid` | How solid the −1.6 intercept is; the hybrid model |
| `ftir_32_cv_scheme_ablation` | Site grouping or fold count: the CV-scheme 2 × 2 |
| `ftir_54_model_form_robustness` | Whether the Addis result depends on PLS |

### Applying the calibration, and the filter-only release

| Notebook | Question |
|---|---|
| `addis_public246_20261020` | FTIR EC on all 246 public SPARTAN Addis filters |
| `filter_only_diagnostics`, `filter_proportionality`, `filter_relationship_stability`, `deliverables/filter_only_scientific_release_2026-09-11/notebooks/filter_only_results` | Site-dependent proportionality and the limits of temporal transfer |

The partial items in G that are also about the spectra could join this
section: the 1600/1617 cm⁻¹ band (`ftir_12`, `ftir_38`), T²/Q and hubness
(`ftir_50`), and the Ward clusters over the network (`ftir_52`).

## I. MAIA (maia-mmgis.jpl.nasa.gov)

**Now catalogued**: `uv run aeth maia catalog` writes
[`research/maia/inventory/CATALOG.md`](../research/maia/inventory/CATALOG.md)
and CSVs: target areas, portal layers, monitors, the ASDC collections, and
day-by-day granule coverage per target. `aeth maia pull` downloads granules
(see [commands.md](commands.md#maia-catalogue)). Before this, the repo's only
use of "MAIA" was the name of the Drive data folder
(`AETHMODULAR_MAIA_DATA_ROOT`).

The archived products are at NASA's ASDC (found through CMR). On 27 Oct 2026 there were three:

- `MAIA_ANC_SURFACEMONITOR_PM_TOTAL` and `MAIA_ANC_SURFACEMONITOR_PM_2.5_SPECIES`:
  daily surface-monitor PM per target area, 2021 onward, still updating.
  At ETH-AddisAbaba, total-PM granules exist on all 247 ETAD filter days and
  speciated ones on 227. At IND-Delhi, the counts are 232 and 147 of 247 INDH days.
  At USA-LosAngeles, they are 264 and 258 of 266 USPA days.
- `MAIA_L4_GFPM` (VSIM001): a simulated Level 4 PM product, 2018 only, Boston only.

The catalogue finds no satellite retrieval products yet. It will list them
when they appear, because it discovers every `MAIA*` collection.
CHN-Beijing, a secondary target, has no surface-monitor granules.

On 27 Oct 2026 the portal's configuration (`/api/configure/get?mission=MAIA`)
held these layers:

- **Target-area outlines**: 12 primary target areas (PTA), 26 secondary (STA)
  and 8 calibration/validation areas (CVTA), plus target centres.
- **Static context grids for 40 targets**, as map tiles: surface elevation, land
  cover, land/water, population density, road density, built-up extent and
  impervious surface (all 1 km), NDVI at peak greenness (250 m, 2021 or 2023),
  and the 2019 annual mean ground-level PM2.5 (1 km).
- **Surface monitor inventories**: 7,938 PM monitors from EEA, AirNow, AirKorea,
  EPA AQS, SPARTAN and others. 470 of them are speciated, including 26 SPARTAN
  sites. There is also an AERONET site list dated 2023-01-29.
- No time-dependent layers (the time slider is off) and no MAIA retrieval
  products. The portal is a map of where MAIA looks and what is on the ground.
  Its PM and speciation retrievals are distributed elsewhere; that was not
  checked here.

The grids are PNG map tiles rather than values, so values at a site would have
to come from the underlying datasets.

21 of the 40 SPARTAN sites fall inside a MAIA target outline, including all
four focus sites:

| Target (type) | SPARTAN sites inside |
|---|---|
| ETH-AddisAbaba (PTA) | ETAD Addis Ababa, ETBI Bishoftu |
| IND-Delhi (PTA) | INDH Delhi, INJA Jaipur |
| USA-LosAngeles (PTA) | USPA Pasadena |
| CHN-Beijing (STA) | CHTS Beijing |
| CHL-Santiago (PTA) | CLST Santiago, CLTA Talca |
| ISR-TelAviv (PTA) | ILNZ Rehovot, ILHA Haifa |
| KOR-Seoul (PTA) | KRSE Seoul, KRUL Ulsan |
| TWN-Taipei (PTA) | TWTA Taipei, TWKA Kaohsiung |
| ZAF-Johannesburg (PTA) | ZAJB Johannesburg, ZAPR Pretoria |
| BGD-Dhaka, MEX-MexicoCity, NGA-Lagos, PRI-SanJuan, VNM-Hanoi (STA) | BDDU, MXMC, NGIL Ilorin, PRFJ Fajardo, VNHN |

The primary target outlines are large: Jaipur, Ulsan, Kaohsiung and Talca are
150–300 km from the target city. Four primary target areas have no SPARTAN site:
Barcelona, Rome, Atlanta and Boston.

## J. Where new data types should go in the gallery

The coverage of each data type decides where it goes:

| Data | Sites | Time base | Where it fits |
|---|---|---|---|
| Filter chemistry, HIPS, FTIR | 40 SPARTAN + IMPROVE | per filter | The existing chart tabs |
| AERONET AOD/AAOD/AAE/SSA | 30 SPARTAN sites with >10 matched days | daily, matched to filter days | New fields in the existing tabs, plus a column-vs-surface page |
| Aethalometer (MA350) | 4 sites + Jacros (`processed_sites/`) | minute → hourly → 9am-day | Filter-matched values as fields; a dedicated page for the rest |
| Charcoal reference spectra | no site; a library | none | A spectra page next to Similarity |
| MAIA target areas and context grids | 21 SPARTAN sites in targets | static | Map layers |

Recommendation: a new menu group with subpages for the **instrument** data,
and new fields for the **filter-matched** values. Either alone falls short.

- **A new group, "Co-located instruments"**, beside Charts, Sites and
  Calibration, with subpages:
  - Aethalometer: BCc time series, diurnal cycles, AAE and source class,
    flow-fix and smoothing QC, HIPS vs aethalometer.
  - Column vs surface (AERONET): AOD/AAOD over time, the scale height
    `H = AAOD / Fabs`, AAE by site.

  These pages need hourly data and their own controls. They fail on most of the
  network, so they should show only the sites that have data, as Weather and
  PMF already do.
- **Filter-matched fields in the existing tabs.** Put the 9am-day aethalometer
  BCc, AAE and MAC, and the filter-day AERONET AOD/AAOD/AAE, into
  `filters.json` / `spartan_reference.json` as new field groups. Then
  "HIPS Fabs vs aethalometer BCc" or "AAOD vs Fabs" is a dropdown in Correlation,
  like every other pair. The sidebar already handles fields that exist at only
  some sites (hollow markers, coverage counts). AERONET would fill most of the
  network there. The aethalometer would fill four sites.
- **The FTIR spectra learning group** (section H) belongs under Calibration,
  for example as a "Spectra" page with subpages: reference library (charcoal),
  band attribution, and model diagnostics (residual learner, robustness). Each
  would need an exporter like the existing `export_similarity.py`.
- **MAIA** is most useful as optional layers on the existing Map: target
  outlines, plus a marker for each site saying which target it falls in.

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
