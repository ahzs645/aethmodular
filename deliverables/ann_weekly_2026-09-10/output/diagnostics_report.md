# Additional Addis FTIR calculations — 10 September 2026

The paired comparison reinforces the concern about the Dry and Belg calibrations. New diagnostics also show that analog lists depend on sampled months and that rainy-season cohorts concentrate heavily in a few source sites. The six new notebook figures are appended as slides 13–18, leaving the main weekly update intact.

All Addis results are exploratory and use HIPS Fabs/MAC as a comparison proxy. They do not establish chemical-EC accuracy. No PLS calibration is refit and the proposed validation split remains unscored.

## Dry and Belg increase disagreement with HIPS

I compared the models on exactly the same Addis filters, with the historical calibration as the reference. The Dry and Belg seasonal models increase RMS disagreement with HIPS by about 1.11 and 1.00 micrograms per cubic metre. Kiremt improves by only 0.09, and its interval includes no improvement. The pooled analog model increases disagreement in all three seasons. This gives a more direct comparison than ranking models by correlation alone.

HIPS Fabs/MAC is a proxy, not chemical EC truth. Differences are candidate minus historical RMS discrepancy. Four thousand whole-calendar-month bootstrap draws keep all models and HIPS paired. Predictions, model choices and MAC remain fixed; intervals exclude fitting/selection uncertainty and ignore dependence between months. Existing Addis data remain exploratory.
Dry (Oct-Feb), Season-specific: RMS 2.038 → 3.145; change +1.107 [+1.024, +1.194] µg/m³.
Dry (Oct-Feb), All-Addis: RMS 2.038 → 2.887; change +0.848 [+0.746, +0.954] µg/m³.
Belg (Mar-May), Season-specific: RMS 1.729 → 2.724; change +0.995 [+0.809, +1.167] µg/m³.
Belg (Mar-May), All-Addis: RMS 1.729 → 3.005; change +1.276 [+1.081, +1.454] µg/m³.
Kiremt (Jun-Sep), Season-specific: RMS 1.745 → 1.657; change -0.088 [-0.302, +0.132] µg/m³.
Kiremt (Jun-Sep), All-Addis: RMS 1.745 → 3.001; change +1.256 [+1.072, +1.462] µg/m³.

![Dry and Belg increase disagreement with HIPS](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/plots/ann_weekly_20260910_diagnostics/slide_13.png)

Sources: [paired_model_comparison.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/paired_model_comparison.csv), [paired_addis_predictions.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/paired_addis_predictions.csv).

## Seasonal matching does not guarantee agreement

Each column holds the Addis evaluation filters constant; each row changes the calibration. The left heatmap shows RMS disagreement and the right shows correlation. The Dry-selected calibration has large disagreement in every season. The Kiremt calibration gives the smallest RMS discrepancy in Dry and Kiremt, while the historical calibration is slightly lower in Belg. This is a useful clue about transfer, but it does not make the Kiremt calibration acceptable: its source-site TOR screening was weak.

All five models were already fitted before this diagnostic. The seasonal selection masks exclude 1800–2500 cm⁻¹; PLS uses the full grid. R² means squared Pearson correlation. This heatmap provides no paired uncertainty for the cross-season Kiremt-versus-historical comparison on Dry, so the apparent ranking is descriptive. No model is selected or promoted based on this display. Fixed HIPS MAC, previously explored target data, differing source cohorts and their weak TOR validation limit inference.

![Seasonal matching does not guarantee agreement](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/plots/ann_weekly_20260910_diagnostics/slide_14.png)

Sources: [cross_season_transfer.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/cross_season_transfer.csv), [paired_addis_predictions.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/paired_addis_predictions.csv).

## Monthly differences remain below zero

The monthly plot shows where the disagreement occurs in time. All three calibration approaches remain below the HIPS proxy in every observed monthly mean. The historical calibration is generally closer, while the seasonal approach changes sharply as its calibration switches between seasons. These jumps could arise from the modelling rule as well as changes in the aerosol, so this plot does not establish a physical cause.

There are 36 observed calendar months, with 1–11 filters per month. Means are calculated over the available filters; months with one filter are retained and do not have a meaningful within-month correlation. Missing calendar months break the lines. Month counts and per-model RMS, MAE, bias and R² are retained in the table. This is not a uniformly sampled monthly climatology. No meteorological adjustment is applied.

![Monthly differences remain below zero](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/plots/ann_weekly_20260910_diagnostics/slide_15.png)

Sources: [monthly_discrepancy.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/monthly_discrepancy.csv), [paired_addis_predictions.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/paired_addis_predictions.csv).

## Changing sampled months changes the analog lists

I resampled the observed months and repeated the entire analog selection, including the median spectrum. Median overlap with the original list ranges from 76 to 87 percent. The pooled and Belg lists are less stable than Dry and Kiremt in this experiment. The curves show a stable core and a less stable edge. That suggests we should examine sensitivity to calibration-list membership before treating an exact top-500 list as definitive.

Every draw samples as many whole year-month blocks as observed, with replacement, then keeps all filters in each sampled block. The All-Addis draw is not season-stratified, so seasonal balance can vary. Correlation is signed, centred Pearson r over retained channels; duplicate scans collapse to physical filters using the same deterministic ranking helper. The baseline exactly reproduces every saved primary list. These resample distributions are conditional sampling sensitivity, not uncertainty intervals for calibration performance. No PLS model is refit.
All Addis: median retention 76.3%; 5th–95th resample percentiles 55.2–94.2%; 227/500 original filters selected in at least 80% of draws; union 1625.
Dry (Oct-Feb): median retention 86.0%; 5th–95th resample percentiles 72.4–95.0%; 341/500 original filters selected in at least 80% of draws; union 1139.
Belg (Mar-May): median retention 78.7%; 5th–95th resample percentiles 59.0–91.2%; 259/500 original filters selected in at least 80% of draws; union 1573.
Kiremt (Jun-Sep): median retention 86.8%; 5th–95th resample percentiles 68.5–97.8%; 344/500 original filters selected in at least 80% of draws; union 1152.

![Changing sampled months changes the analog lists](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/plots/ann_weekly_20260910_diagnostics/slide_16.png)

Sources: [analog_stability_summary.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/analog_stability_summary.csv), [analog_bootstrap_draws.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/analog_bootstrap_draws.csv), [analog_selection_frequencies.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/analog_selection_frequencies.csv).

## Belg more often exceeds the source residual threshold

The map summarizes spectral shape after the CO₂ region is removed. Grey points are the eligible IMPROVE library; colours show Addis seasons. The two displayed components explain about 75 percent of source shape variation. A separate ten-component reconstruction check finds 10 of 61 Belg spectra above the source residual threshold, compared with 5 of 105 Dry and 3 of 73 Kiremt spectra. Belg therefore deserves a closer inspection of the individual spectra, without automatically excluding these filters.

Each masked spectrum is centred across channels and normalized to unit length, matching Pearson geometry. PCA is fitted only to 13,010 unique eligible source filters, choosing the lowest AnalysisId for duplicate source filters. Ten fixed PCs explain 98.98% of source variance. Addis is projected using the source basis. The threshold is the in-sample source 95th percentile of squared reconstruction residual (0.00718935); it is descriptive, not a calibrated acceptance rule. The 5% dashed line is the source reference fraction. No chemical meaning is assigned to PCs, and no uncertainty or significance test is claimed for seasonal percentages.
Dry (Oct-Feb): 5/105 (4.8%) above source 95th-percentile reconstruction residual.
Belg (Mar-May): 10/61 (16.4%) above source 95th-percentile reconstruction residual.
Kiremt (Jun-Sep): 3/73 (4.1%) above source 95th-percentile reconstruction residual.

![Belg more often exceeds the source residual threshold](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/plots/ann_weekly_20260910_diagnostics/slide_17.png)

Sources: [pca_target_scores.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/pca_target_scores.csv), [pca_source_scores.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/pca_source_scores.csv), [pca_season_summary.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/pca_season_summary.csv), [methods.json](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/methods.json).

## Rainy-season cohorts are more concentrated by site

All cohorts contain 500 filters, but those filters are distributed differently across source sites. Dry has 121 distinct sites and about 70 effective sites. Belg and Kiremt have only about 32 and 29 effective sites. In each rainy-season list, five sites supply 32 percent of filters. This suggests a concrete sensitivity check: repeat calibration with source-site contributions capped, then compare models on a common source test set.

The effective site count is the inverse of the sum of squared site shares, also called inverse-Simpson concentration. Equal site contributions recover the distinct count; concentration reduces it. This is not an estimate of independent filters or a replacement for grouped validation. It is computed over the full selected 500-filter cohort, including its source training and test roles. Site identities and complete ranked contribution curves are supplied. Site capping and calibration refitting are a next experiment, not a result of this diagnostic.
All Addis: 109 distinct sites; 56.5 effective sites; top five supply 18.6% of filters.
Dry (Oct-Feb): 121 distinct sites; 69.7 effective sites; top five supply 15.2% of filters.
Belg (Mar-May): 103 distinct sites; 32.3 effective sites; top five supply 32.0% of filters.
Kiremt (Jun-Sep): 92 distinct sites; 28.7 effective sites; top five supply 32.0% of filters.

![Rainy-season cohorts are more concentrated by site](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/plots/ann_weekly_20260910_diagnostics/slide_18.png)

Sources: [site_concentration.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/site_concentration.csv), [site_concentration_curves.csv](/Users/ahmadjalil/github/aethmodular/research/ftir_hips_chem/output/tables/ann_weekly_20260910_diagnostics/site_concentration_curves.csv).

## Reproduce and interpret

Run `uv run aeth doctor`, then `uv run python research/ftir_hips_chem/workflows/create_ann_weekly_diagnostics_notebook.py` from the repository root. The source notebook is `research/ftir_hips_chem/ann_weekly_20260910_diagnostics.ipynb`; run it from that active folder. The archive copy retains six displayed figures and all calculation outputs. Tables include exact paired predictions, resample-level overlap, per-filter selection frequencies, named site contributions and PCA scores. `methods.json` records seeds and input hashes; `pca_basis.npz` retains the source mean and component basis.

The paired-resampling principle is described in [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); this analysis applies it to whole observed year-month clusters. Source-fitted PCA transforms follow [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

Useful next experiments are a common source-site test set for all candidates, source-site contribution caps, a fixed grid of analog counts, and direct inspection of the flagged Belg spectra. Set these choices before evaluating any newly reserved target filters. The present diagnostics do not supply results for those experiments.
