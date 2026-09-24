# AIRSpec and VIBES research update

## 1. AIRSpec remains our EC default while VIBES merits spectroscopy follow-up

SAY: We will compare what the methods do, then separate blank removal from elemental-carbon prediction. I will show the completed Colab comparison, the cases we traced afterwards, and the earlier experiments that changed our interpretation. The main result is that VIBES removes blank background much more strongly, while the current EC comparison does not establish an overall accuracy improvement.

NOTES: Research group update, evidence through 21 September 2026. AIRSpec refers here to its segmented baseline correction implementation, not every feature of the broader AIRSpec platform. VIBES is the method implemented by the supplied pyvibes 1.0.0. The presentation preserves mass units because the paired benchmark uses µg/filter. It does not apply historical explorer concentration anchors to this different benchmark.

Sources:
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/report.md
- docs/openresearch-retrospective/followthrough-2026-09-21.md

## 2. AIRSpec fits a smooth baseline while VIBES learns background patterns

SAY: Read each row across. AIRSpec estimates a smooth curve using selected regions of the individual spectrum. VIBES first learns how independent blank spectra vary, then estimates how much of those patterns appears in the sample. Both subtract their estimated background. The assumptions differ, so neither corrected spectrum becomes chemical ground truth simply because it looks cleaner.

NOTES: AIRSpec port: weighted natural cubic smoothing splines with effective degrees of freedom 6 and 4, zero weights in masked analyte intervals, adaptive boundary rules, and averaging in the segment overlap. Segments in this implementation are 4000–1820 and 2000–1425 cm⁻¹. VIBES uses PCA background mean/components, variational inference for sample-level parameters and a MAP background estimate. PB means pinball loss, with fixed tau=0.1 in this run. The 87 independent physical training blanks yield PCA rank 28 under blank-only leave-one-out selection. No TOR labels select the background model.

Sources:
- research/ftir_ec_phase3/scripts/airspec_baseline.py
- research/ftir_hips_chem/scripts/vibes_baseline.py
- research/ftir_hips_chem/vendor/pyvibes/README.md
- https://amt.copernicus.org/articles/12/2313/2019/
- https://aprl.gitlab.io/spec_vignette/R-baseline.html

## 3. The same raw spectrum leads to different background estimates

SAY: This is one real BRIS1 filter, not an average. Grey shows its measured spectrum. Blue and purple show the two estimated baselines. This filter was selected because it contributes the largest squared-error penalty in the loading-band investigation, so it is a diagnostic example rather than a typical spectrum.

NOTES: Filter improve:1970697, BRIS1, 13 June 2022, lot 251, outer-test sample outside locked800. Selection: largest VIBES-minus-AIRSpec squared-error increase in the inspected full-pool Q3 cases. Baselines reconstructed exactly as saved raw minus saved corrected arrays. All 2002 spectral channels retained. Plot x increases left to right and explicitly labels wavenumber. Neither estimated baseline is an independently measured physical truth.

Sources:
- research/ftir_hips_chem/output/tables/vibes_case_investigation/inspection_spectra.npz
- research/ftir_hips_chem/output/tables/vibes_case_investigation/case_evidence.csv

## 4. Background subtraction changes the spectrum given to the EC model

SAY: Here is the same filter after each correction, on a much smaller vertical scale. The methods leave different broad structure as well as different local features. That affects the input to the EC calibration. It does not identify which curve is physically correct or which chemical species explains the difference.

NOTES: Same selected single physical filter as the preceding slide. No normalization, clipping, smoothing or averaging for display. Corrected spectra are the saved results. Broad 3000–4000 cm⁻¹ differences motivate independent background and standard checks. They do not establish an O–H, fuel-source or other causal chemical explanation.

Sources:
- research/ftir_hips_chem/output/tables/vibes_case_investigation/inspection_spectra.npz
- research/ftir_hips_chem/output/tables/vibes_case_investigation/report.md

## 5. The EC benchmark refits a separate calibration after each correction

SAY: We do not feed a VIBES spectrum into an AIRSpec-trained model. Each method gets a separately fitted partial least squares model, using the same training and test identities within its cohort. PLS learns spectral combinations associated with measured thermal EC. That means the final prediction difference includes both the change in spectrum and the change in fitted coefficients.

NOTES: Protocol: site-disjoint outer split, training-only five-fold site-grouped component CV, first major minimum, components 1–30, PLSRegression scale=False. This shares the grouped-CV idea of historical Option A, but the frozen Colab split and eligible rows are different. Do not relabel it as an identical historical explorer run. Evaluation reference is saved IMPROVE TOR EC mass. PLS is an empirical prediction model and does not independently isolate EC absorption.

Sources:
- research/ftir_hips_chem/scripts/vibes_large_run.py
- research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09/calibration_scores.csv

## 6. The completed run supports two distinct paired comparisons

SAY: The larger cohort has 2327 test filters. The restricted cohort has 137. Both compare AIRSpec and VIBES on identical test identities within that row. The rows themselves use different populations, so their errors cannot rank the training-cohort strategies. The restricted label also does not mean 800 eligible test or training rows survived this particular frozen split.

NOTES: All 12,808 correction cases completed with zero final failures and 451 logged retries. That total includes calibration, target, blanks and injection cases, not 12,808 independent held-out EC observations. Full pool has 125 training sites and 32 test sites. Restricted has 101 training sites and 24 test sites. Background training uses 87 separate physical blanks, PCA rank 28, below cap 30. Actual eligible restricted counts are 625 train and 137 test. Training/test sites are disjoint.

Sources:
- research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09/RUN_MANIFEST.json
- research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09/calibration_scores.csv
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/report.md

## 7. Full-pool EC errors are slightly higher with VIBES

SAY: Lower bars mean smaller prediction error. VIBES has a slightly larger RMSE and MAE in the full-pool benchmark. The RMSE difference is about 0.071 micrograms per filter, and the paired uncertainty interval includes zero. So these point estimates support retaining our current AIRSpec default, but they do not prove a universal superiority claim.

NOTES: Full pool n=2327, 32 test sites. AIRSpec k=6, predictive R²=0.679. VIBES k=7, predictive R²=0.664. Both biases are about +0.294 µg/filter. Protocol: five-fold training-site CV, first major minimum, frozen outer test. Paired site-cluster 95% interval for VIBES minus AIRSpec RMSE is −0.082942 to +0.263527 µg/filter. Do not interpret overlap with zero as equivalence.

Sources:
- research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09/calibration_scores.csv
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/subgroup_metrics.csv

## 8. Restricted-cohort VIBES improves RMSE while MAE gets worse

SAY: The restricted result goes in different directions depending on the metric. VIBES reduces the error measure that emphasizes large mistakes, but increases the average absolute error. Its RMSE interval also crosses zero. Choosing VIBES using only the restricted RMSE would leave out an important part of the evidence.

NOTES: Restricted cohort n=137, 24 test sites. AIRSpec k=5, predictive R²=0.868. VIBES k=7, predictive R²=0.881. Bias changes from 0.294 to -0.040 µg/filter. RMSE is the square root of mean squared prediction error, while MAE is mean absolute error. This cohort is not the separate historical reproduction with 194 held-out rows.

Sources:
- research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09/calibration_scores.csv
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/subgroup_metrics.csv

## 9. The pooled intervals establish neither a gain nor equivalence

SAY: This table makes the uncertainty explicit. Negative differences favor VIBES and positive differences favor AIRSpec. Both intervals cross zero. The intervals describe uncertainty over held-out sites conditional on the fitted models. They do not cover uncertainty from retraining the entire pipeline.

NOTES: Audit intervals use 10,000 paired site-cluster resamples and sample-weighted metrics. The same resampled sites feed both methods in each draw. These exploratory, pointwise intervals follow inspection of the test results and have no multiplicity adjustment. The initial Colab report used 2000 draws with another seed, so interval endpoints differ slightly while point estimates agree. No equivalence margin was prespecified.

Sources:
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/report.md
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/subgroup_metrics.csv

## 10. VIBES leaves substantially less residual signal on held-out blanks

SAY: This result is much clearer than the EC prediction comparison. VIBES leaves far less RMS signal on both sets of held-out blanks. That supports background-removal performance on these blanks. Field blanks may also carry handling contamination, so proximity to zero does not by itself prove that all remaining signal should disappear.

NOTES: Median corrected-spectrum RMS in absorbance. ETAD n=9: AIRSpec 0.00057725183, VIBES 0.000006726343, about 86-fold smaller. IMPROVE n=126: AIRSpec about 0.0005411, VIBES about 0.000006784, about 80-fold smaller. Each evaluation set is separate from background training. This is the completed combined-blank-library run, not the earlier ETAD-only pilot. Spectroscopy endpoint, not an EC accuracy measure.

Sources:
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/blank_summary.csv
- research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09/background_training_blanks.csv
- research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09/RUN_MANIFEST.json

## 11. Synthetic recovery favors VIBES at larger additions, not the weakest

SAY: We added the same known synthetic peak pattern at three amplitudes. VIBES has a smaller recovery error at the two larger amplitudes, while AIRSpec does better at the smallest. The nine parent blanks recur at every amplitude. This is useful evidence about that injected shape, but we still need more independent blanks and real standards.

NOTES: Recovery compares correct(blank + injected peaks) minus correct(blank) with the known injected signal. This avoids assuming the original field blank contains no chemistry. Each bar is the median error for nine independent parent blanks, not 27 independent blanks across the figure. Amplitudes and error are absorbance. No claim of recovery of ambient EC concentration or every molecular feature follows from this test.

Sources:
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/injection_summary.csv
- research/ftir_hips_chem/VIBES_COMPARISON.md

## 12. An exploratory mid-loading band shows a larger VIBES error penalty

SAY: Quartiles use thresholds derived from training data. The third band is the clearest discrepancy: 545 test filters between about 3.74 and 6.99 micrograms of EC per filter. VIBES increases RMSE by about 0.453 there. We inspected these subgroups after seeing the test results, so this is a lead for investigation rather than a validated method-switching rule.

NOTES: Q3 is (3.73908411142889, 6.993548115901122] µg/filter, n=545 at 30 sites. AIRSpec RMSE 2.2996, VIBES 2.7522. Paired pointwise 95% interval for ΔRMSE is 0.1869 to 0.7425. MAE increases about 0.209 while mean bias is nearly unchanged. 23/30 sites have higher VIBES MSE, and omission of each site leaves ΔRMSE 0.313–0.489. Other bands have intervals spanning zero. Loading uses measured TOR reference and is not a directly available inference-time selector. Full pool only.

Sources:
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/subgroup_metrics.csv
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/training_loading_bands.csv
- research/ftir_hips_chem/output/tables/vibes_subgroup_audit/loading_site_sensitivity.csv

## 13. Two leading cases already fail under AIRSpec and worsen under VIBES

SAY: Grey bars show positive measured TOR EC. Both models predict negative mass for these two filters, with VIBES more negative. Together they contribute about half of the third-band increase in squared error. We retain both in every score. They reveal a shared prediction failure that VIBES intensifies, rather than two filters we can remove to improve the result.

NOTES: BRIS1 improve:1970697 and CACR1 improve:1973864, both lot 251 and full-pool outer-test samples outside locked800. Values are saved predictions, with no clipping. Combined contribution is about 52.2% of the full-pool Q3 squared-error increase, not 52.2% of all pooled error. Selection followed the trace protocol of three worse and three better cases at each of BRIS1/CACR1. Negative estimated aerosol mass is physically inadmissible, but retaining it in error evaluation exposes the model failure.

Sources:
- research/ftir_hips_chem/output/tables/vibes_case_investigation/case_evidence.csv
- research/ftir_hips_chem/output/tables/vibes_case_investigation/report.md

## 14. Repeated corrections argue against a gross convergence failure

SAY: We reran the correction for all twelve predeclared inspection cases using the saved blank model and original settings. Every fit converged again, and the largest EC prediction change was about two thousandths of a microgram per filter. That is far smaller than the errors on the prior slide. It narrows the numerical explanation without establishing a chemical cause.

NOTES: All twelve original inspection fits converged without retries. All twelve repeats converged. Largest corrected-spectrum absolute difference was 1.1947e-5 absorbance, and largest repeated prediction difference was 0.001993968 µg/filter. Frozen coefficients and original float32 centering are used for prediction reconciliation. Symmetric decomposition shows opposing spectral and coefficient contributions, with a larger negative coefficient term in the two leading cases. That decomposition is conditional on the two models, not causal attribution. No parameters were tuned on test outcomes.

Sources:
- research/ftir_hips_chem/output/tables/vibes_case_investigation/report.md
- research/ftir_hips_chem/output/tables/vibes_case_investigation/repeat_prediction_comparison.csv

## 15. The historical AIRSpec calibration reproduces on its own frozen inputs

SAY: This is a separate reproducibility check. The unchanged historical calibration runner reproduced all five result tables, with every one of the 43 comparisons passing. Its 194 test rows belong to a different split from the 137 restricted test rows in the VIBES benchmark. Reproducing this calculation strengthens our baseline record, while independent Addis accuracy remains unanswered.

NOTES: Detached checkout at original HEAD e78fa862806a06d300d93a91210ecbe4a828b1c0. Reuses frozen corrected spectral caches, rather than rerunning full-pool baseline correction. DF1=6 and DF1=8 both recover k=5. The DF1=6 held-out TOR RMSE is 3.85865444 µg/filter and predictive R²=0.90423786. An independent historical split-membership artifact was not available: this reconstructs the documented seed and input order, with matching saved tables. The separate 319-scan R-port comparison supports correction implementation and maximum absolute difference about 6.01e-7. Neither exercise establishes an independent Addis thermal validation.

Sources:
- research/ftir_hips_chem/output/tables/airspec_locked_reproduction/report.md
- research/ftir_hips_chem/output/tables/airspec_locked_reproduction/table_comparison.csv
- research/ftir_hips_chem/output/tables/airspec_reproduction_preflight/

## 16. Earlier experiments changed how we interpret transfer and chemistry

SAY: These are documented historical findings, rather than newly rerun comparisons in this deck. Selection of low-OC/EC and other calibration subsets changed model behavior, but IMPROVE holdouts cannot validate Addis transfer. The adjusted PMF work supports a regional spectral marker, without identifying a particular fuel. Later target decomposition also shows why we need cohort-specific explanations.

NOTES: Historical evidence status: preserved notebook outputs and audited claim corrections, except the explicitly reproduced AIRSpec runner on the prior slide. Do not generalize numerical rankings across different test populations. The 1617–1620 cm⁻¹ marker does not establish charcoal or eucalyptus. EC_TOT vs EC_TOR mechanism depends on target scaling and cohort; the selected-cohort oracle/composed analysis and full-pool decomposition differ. Detailed sources and limits are in the claim ledger.

Sources:
- docs/openresearch-retrospective/claim_updates.json
- research/ftir_ec_phase3/ftir_38_1617_and_dust_attribution.ipynb
- research/ftir_ec_phase3/ftir_48_pyrolysis_split_target.ipynb

## 17. Blank and metadata audits qualified earlier explanations

SAY: The later audits corrected particular claims, without erasing the earlier work. For blanks, a pooled lot curve can mix separate deployed calibration lines. For metadata, the paired comparison does not establish the incremental benefit of spectra. For ChemSpec, the apparent reference turns out to mirror quantities we already used, so it cannot supply an independent thermal test.

NOTES: Claim lineage: ftir_37 to ftir_47 revises pooled-lot blank interpretation and the Pasadena-dissolves claim. ftir_43 to ftir_46 qualifies the spectra-over-metadata increment because its paired RMSE interval includes zero; the mass-conditioned contrast is a different comparison. ftir_16 to ftir_27 supersedes the independence assumption: ChemSpec EC mirrors FTIR EC, and ChemSpec BC mirrors HIPS/MAC. These are historical audited findings, not newly executed tests in the presentation.

Sources:
- docs/openresearch-retrospective/claim_updates.json

## 18. Independent Addis EC accuracy still requires new reference evidence

SAY: The local search found useful files but did not establish an independent Addis thermal EC reference. The five Adama date pairings are a separate, provisional comparison. We need laboratory-confirmed identities and measurements before claiming Addis validation. In particular, date proximity alone does not prove that two physical filters sampled equivalent air.

NOTES: Readiness inventory covered 753 local files and inspected seven relevant exports. No independent ETAD thermal reference or authoritative mapping for Adama spectral IDs 4744–4748 was established in that search. July 9 pair has a 39.73-minute start offset, July 30 has a PTFE/quartz volume ratio of 0.456. Five Adama pairings do not resolve missing Addis thermal truth. The data request exists as a draft and has not been sent. No result is inferred from absence in the bounded local search.

Sources:
- research/ftir_hips_chem/output/tables/addis_validation_readiness/report.md
- research/ftir_hips_chem/output/tables/addis_validation_readiness/data_request_draft.md

## 19. The next experiments should test applicability and weak-signal recovery

SAY: The next defensible step is to learn an applicability check using training data only, then see what it says about unusual spectral shapes. Separately, extend weak-signal tests using independent blanks and standards. Changes to the method need fresh evaluation because we have already inspected these test cases. For Addis, obtaining the independent reference remains the essential experimental task.

NOTES: Proposed work, not completed results: training-only applicability/shape diagnostic, no test-driven thresholds or removal of the two failing filters, independent blanks/standards across lots and amplitudes, then new evaluation of any preprocessing change. Historical backlog remains 25 sources without an executed companion. Three P1 items are ETAD factors, HIPS/PTFE operating envelope and filter temporal patterns. Those are triaged, not verified by prioritization. AIRSpec remains the current EC default under the available paired evidence; VIBES remains worth pursuing for spectroscopy.

Sources:
- research/ftir_hips_chem/output/tables/vibes_case_investigation/report.md
- research/ftir_hips_chem/output/tables/historical_gap_followthrough/priority_queue.json
- research/ftir_hips_chem/output/tables/addis_validation_readiness/report.md
