# Notebook Inventory

Active notebooks stay at `research/ftir_hips_chem/`. Archived executed and
scratch notebooks live under `notebooks/archive/`.

Use this file as the cleanup queue before editing notebooks. The notes identify
organization issues only; they do not judge the scientific status of a notebook.

| Notebook | Purpose | Cells | Organization Notes |
|---|---|---:|---|
| `Analysis_Tasks_Jan2025.ipynb` | Analysis Tasks - January 2025 | 24 | Standardized output paths |
| `ETAD_Factor_Analysis.ipynb` | ETAD Factor Contributions Analysis - Ethiopia (Addis Ababa) | 39 | Darkgrid style |
| `Example_Modular_Analysis.ipynb` | Example: Modular Analysis Framework | 41 | Template |
| `FTIR_Group_Talk_Apr2026_Draft.ipynb` | FTIR Group Talk Draft: Multi-Site BC Storyboard | 25 | OK |
| `FlowFix_BeforeAfter_Analysis.ipynb` | Before/After Flow Fix Analysis | 28 | Darkgrid style |
| `HIPS_Aeth_SmoothRaw_Analysis.ipynb` | HIPS vs Aethalometer: Smooth/Raw Threshold Analysis | 25 | Darkgrid style |
| `HIPS_vs_Aethalometer_Optical_Comparison.ipynb` | HIPS vs Aethalometer Optical Comparison | 25 | OK |
| `Multi_Site_Analysis.ipynb` | Multi-Site Aethalometer and Filter Data Analysis | 29 | Darkgrid style |
| `Multi_Site_Analysis_Fixed.ipynb` | Multi-Site Aethalometer Analysis (Modular Version) | 53 | Uses `plotting_legacy`; darkgrid style |
| `Multi_Site_Analysis_FollowUp.ipynb` | Multi-Site Aethalometer Analysis: Follow-Up Analyses | 34 | Uses `plotting_legacy`; darkgrid style |
| `Multi_Site_Analysis_Modular.ipynb` | Notes on Follow-Up Analyses | 53 | Uses `plotting_legacy`; darkgrid style |
| `Raw_ATN_Correlation_Analysis.ipynb` | Raw Attenuation vs BCc Correlation Analysis | 25 | OK |
| `Task_Analysis_Notebook.ipynb` | Task Analysis Notebook - January 2026 | 51 | Darkgrid style |
| `addis_01_source_apportionment.ipynb` | Addis Ababa: Source Apportionment Analysis | 21 | Darkgrid style; external Google Drive minute-data path |
| `addis_01_source_apportionment_daily.ipynb` | Addis Ababa: Source Apportionment Analysis (Daily 9am-Resampled Data) | 21 | Darkgrid style |
| `addis_02.5_temporal_patterns_filter.ipynb` | Addis Ababa: Temporal Patterns - Aethalometer BC, FTIR EC & HIPS Fabs | 20 | Darkgrid style |
| `addis_02_temporal_patterns.ipynb` | Addis Ababa: Temporal Patterns Analysis | 20 | Darkgrid style; external Google Drive minute-data path |
| `addis_02_temporal_patterns_daily.ipynb` | Addis Ababa: Temporal Patterns Analysis (Daily 9am-Resampled Data) | 20 | Darkgrid style |
| `addis_03_meteorology.ipynb` | Addis Ababa: Meteorological Analysis | 20 | Darkgrid style; external Google Drive minute-data/weather paths |
| `addis_03_meteorology_daily.ipynb` | Addis Ababa: Meteorological Analysis (Daily 9am-Resampled Data) | 20 | Darkgrid style |
| `addis_04_aeronet.ipynb` | Addis Ababa: AERONET Columnar Aerosol Analysis | 20 | Darkgrid style; external Google Drive minute-data/AERONET paths |
| `addis_04_aeronet_daily.ipynb` | Addis Ababa: AERONET Columnar Aerosol Analysis (Daily 9am-Resampled Data) | 20 | Darkgrid style |
| `addis_05_diurnal_wavelength_analysis.ipynb` | Addis Ababa: Diurnal Multi-Wavelength Analysis | 23 | Darkgrid style; external Google Drive minute-data path |
| `addis_06_hips_ptfe_operating_envelope.ipynb` | Addis Ababa: HIPS/PTFE Operating Envelope | 26 | OK |
| `addis_ababa_source_analysis.ipynb` | Addis Ababa: BC/EC Method Comparison by Source Apportionment | 24 | Darkgrid style |
| `cross_plots_and_distributions.ipynb` | Cross-Plots, Distributions, and AERONET Filtering | 13 | Uses config paths; AERONET files must be supplied under `AERONET/daily/` for the AERONET section |
| `dominant_source_comparison.ipynb` | Addis Ababa: Dominant Source Determination Comparison | 25 | Darkgrid style |
| `dominant_source_threshold_analysis.ipynb` | Dominant Source Threshold Analysis | 31 | Darkgrid style |
| `figure7_source_contributions.ipynb` | Figure 7: Daily Contribution of Sources to OM in Addis Ababa in 2023 | 10 | OK |
| `flow_fix_explorer.ipynb` | Flow Fix Period Explorer | 27 | OK |
| `follow_up_analysis.ipynb` | Follow-Up Analysis Notebook | 41 | Uses `plotting_legacy`; darkgrid style |
| `hips_offset_narrative.ipynb` | Addis HIPS anomaly audit | 40 | OK |
| `improve_high_fabs_comparison.ipynb` | IMPROVE High-Fabs Comparison | 25 | OK |
| `multisite_diurnal_wavelength_analysis.ipynb` | Multi-Site Diurnal Wavelength Analysis | 26 | Uses `processed_sites/` via config |
| `primary_tasks_notebook.ipynb` | Primary Tasks - Pre-Meeting Analysis | 44 | Darkgrid style |
| `results_summary.ipynb` | Results Summary - Presentation Figures | 15 | OK |
| `smoothing_comparison.ipynb` | Smoothing Comparison: Raw vs Instrument-Smoothed vs Our Methods | 15 | Uses `processed_sites/` via config |
| `source_apportionment_regression.ipynb` | Addis Ababa: BC/EC Method Comparison by Source Apportionment | 22 | Darkgrid style |
| `threshold_by_source_analysis.ipynb` | Threshold-Filtered Regressions by Source Type | 24 | Darkgrid style |
| `warren_meeting_slides.ipynb` | Warren Meeting Slides (May 2026) | 34 | OK; currently has unrelated local edits |

## Priority Cleanup Queue

1. Migrate `plotting_legacy` users to the modular plotting package or archive
   them if superseded.
2. Remove `plt.style.use('seaborn-v0_8-darkgrid')` from active notebooks unless
   the different style is intentional and documented in that notebook.
3. Decide whether AERONET source files should be copied into a local ignored
   `AERONET/daily/` folder or kept as an external data dependency.
