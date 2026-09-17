# FTIR EC calibration — weekly meeting

Results through 17 September 2026. Main discussion: slides 1–18; backup: slides 19–26.

## Slide 1 — FTIR EC calibration: what transfers, and what does not?

This update brings together the controlled 47-configuration batch, the later frozen-model transfer checks, and the new Colab experiment workspace. Start with the spectral inputs and the cohorts, then show prediction and transfer results.

**Interpretation / limits:** This is an internal retrospective research update. No replacement calibration is accepted. Values are reproduced from the packaged results; the deck build recalculated the 15 five-site OLS result rows from saved predictions. It did not refit all calibrations.

**Discussion:** Which scientific criterion should determine the next candidate: TOR prediction, HIPS agreement, or both?

**Sources:** September 10 meeting transcript; aethmodular_tests_2026-09-16/results/frozen_experiment.json; tor_spartan_stability_2026-09-17/audit.json; Aethmodular_Experiment_Lab/validation/validation_receipt.json

## Slide 2 — The September 10 questions now have controlled tests

The full-grid control and the masked fit now use identical physical-filter memberships within each pair. The small-cohort test implements a stated magnitude rule, rather than assuming that top correlation automatically matches amplitude. Combined PMF groups remain secondary.

**Interpretation / limits:** The Colab workspace is an engineering deliverable, not evidence that every available preset has been scientifically evaluated. The transcript withdrew normalization as a calibration solution because it removes concentration information.

**Discussion:** Do these experiments address the intended September 10 questions before we expand the search?

**Sources:** audio-notes-export-2026-09-16.md, September 10: 62:31–63:59; 67:11–69:02; 71:19–73:58. aethmodular_tests_2026-09-16/results/frozen_experiment.json.

## Slide 3 — The magnitude-matching step narrows the spectral envelope

Show the two example seasons before the results. Blue is the Addis selection population; orange is the 500 shape analogs; green is the 50 magnitude-matched analogs. Both plots share the same absorbance range. The source shortlist contains eligible IMPROVE physical filters with TOR labels.

**Interpretation / limits:** These are distributions from existing corrected spectra, not new AIRSpec baseline fits. Quantile envelopes are not confidence intervals. The tested matching rule uses absolute log RMS-amplitude distance to the median development-target amplitude within the top-500 shape shortlist. It does not normalize spectra during regression.

**Discussion:** Does this magnitude-matching rule reflect the intended physical comparison, or should another loading/peak-height definition be prespecified?

**Sources:** Derived chart_data/spectra_Dry__Oct_Feb.csv and spectra_Kiremt__Jun_Sep.csv. aethmodular_tests_2026-09-16/results/all_model_membership.csv. Aethmodular_Experiment_Lab/data/source.csv, source_airspec.npy, targets.csv, target_airspec.npy, wn.npy.

## Slide 4 — A common test population makes the model comparisons interpretable

There are two distinct split systems. The source test sites are excluded before new analog selection. Within source development data, components use grouped cross-validation. For Addis, only the 166 selection-side spectra determine target-analog selection. The 73 reserve filters are grouped by month and split as 29 Dry, 21 Belg and 23 Kiremt.

**Interpretation / limits:** The similar-source subset is fixed from spectra, not picked by prediction quality. Test statistics on the full and restricted populations should be reported together. The existing 73-filter reserve is retrospective; repeated inspection cannot become prospective validation. There are 296 eligible source records absent from the supplied catalog; they were flagged and retained under the frozen rule.

**Discussion:** Should the next evaluation emphasize the full source population, a declared target-like domain, or both?

**Sources:** aethmodular_tests_2026-09-16/results/frozen_experiment.json; data_audit.json; addis_split.csv; Prepared source.csv records catalog support.

## Slide 5 — Keep chemical-reference prediction separate from optical agreement

The source tests assess the spectrum-to-TOR calibration. Target tests assess agreement with the shipped/cached optical product divided by a fixed conversion coefficient. The optical benchmark has not been established as chemical EC truth. OLS is used consistently for the five-site comparison; earlier Addis Deming screens use a different estimator.

**Interpretation / limits:** AIRSpec baseline removal is not the regression intercept in this plot. The source loading calculation was preserved from the repository and reproduced numerically; the raw TOR export lacks an independent unit column. Do not reinterpret the exported Value field as verified ambient concentration. Negative predictions are retained in the evaluation.

**Discussion:** Can we agree on the primary reference, regression direction and tolerance definitions before optimizing?

**Sources:** aethmodular_tests_2026-09-16/results/frozen_experiment.json, reference_units and metrics. tor_spartan_stability_2026-09-17/audit.json, five_site_target_definition and cross_site_primary_regression.

## Slide 6 — Masking changes the predictions, but not consistently for the better

The upper-cut mask excludes 1800–2500 cm⁻¹ and channels above 3500 cm⁻¹. Negative changes favour masking. Pooled RMS discrepancy falls by 0.183; Dry improves by 0.071; Belg is essentially unchanged; Kiremt increases by 0.261.

**Interpretation / limits:** Intervals are paired month-block bootstraps conditional on fixed fitted models, not full model uncertainty. There are only 5, 3 and 4 month clusters for the seasonal comparisons. Do not combine separate group confidence intervals; the seasonal-family row is the proper paired aggregate. CO₂-only seasonal aggregation is also essentially neutral: 2.929 to 2.930.

**Discussion:** Is further masking refinement still the best use of the next run, or should we hold this mask fixed while testing cohort/domain choices?

**Sources:** aethmodular_tests_2026-09-16/results/headline_mask_results.csv; seasonal_mask_family_contrasts.csv; frozen_experiment.json.

## Slide 7 — Smaller or magnitude-matched cohorts did not resolve the discrepancy

The 500, 50 and 100 shape rows use ranked shape similarity. The two magnitude rows use closer RMS-absorbance magnitude within the fixed shape shortlist. The low-OC/EC 440 reference is separately refitted on the common source-development split, with a fixed eight components and full-grid features.

**Interpretation / limits:** The comparison supports a conclusion about this matching rule, not all possible loading-aware calibration strategies. Lower HIPS discrepancy is not enough to select a new EC calibration. This is not a direct comparison to the historical original-split 440 model.

**Discussion:** Should a next cohort rule be based on a prespecified physical loading metric rather than another target-fit search?

**Sources:** aethmodular_tests_2026-09-16/results/addis_family_metrics.csv; frozen_experiment.json, amplitude_experiment and reference.

## Slide 8 — TOR correlation is stronger in the target-like subset than in the full test

Read each model across the two populations. For the pooled pair, masking has relatively little effect on TOR correlation. Belg is the strongest example of domain dependence: 0.162 on the full set versus 0.831 on the fixed similar subset.

**Interpretation / limits:** The 0.85 line is the earlier project screening convention, not a universal standard. The restricted subset changes the evaluation domain and loading distribution, so a higher R² there is not proof of broad transferability. All 47 models were already screened; avoid presenting a post hoc best value as untouched validation.

**Discussion:** What source domain should a defensible deployment claim cover?

**Sources:** aethmodular_tests_2026-09-16/results/model_metrics.csv; frozen_experiment.json. Chart source: chart_data/tor_common_test.csv.

## Slide 9 — The historical R² = 0.924 is sensitive to two high-loading observations

The two orange diamonds are PHOE1, the highest-loading observations in the historical test. The original result reproduces. Holding coefficients fixed but removing that site from evaluation lowers both correlation and slope. Removing either observation alone gives R² about 0.864 or 0.893.

**Interpretation / limits:** Do not recommend deleting PHOE1. The correct interpretation is that overall correlation depends on the available reference range. The whole-site conditional R² interval is approximately 0.447–0.983. This is not a refit and does not measure TOR laboratory repeatability.

**Discussion:** Should historical performance always be accompanied by within-range and leave-one-evaluation-site sensitivity?

**Sources:** tor_spartan_stability_2026-09-17/historical_test_predictions.csv; historical_leave_one_evaluation_site_out.csv; historical_leave_one_evaluation_filter_out.csv; historical_TOR_conditional_stability.csv.

## Slide 10 — The pooled source result conceals substantial between-site variation

Each site contributes one within-site R² to this histogram. Some site-specific prediction relationships are much weaker than the pooled relationship. Q² is positive at 13 of 32 sites.

**Interpretation / limits:** Within-site R² depends on the concentration/loading range within that site, so interpret it together with RMSE, bias and sample size. The bootstrap resamples evaluation sites with frozen coefficients. It does not assess variability caused by new training sets or parameter selection. Repeated refitting exists in the new workspace but was not the basis for these scientific results.

**Discussion:** Can we specify a domain-of-applicability rule and a stability criterion before a much larger parameter sweep?

**Sources:** tor_spartan_stability_2026-09-17/tor_per_site.csv; tor_site_bootstrap.csv; audit.json.

## Slide 11 — The same frozen calibration does not give the same relationship everywhere

Compare both slope and intercept. Pooled masked slopes are approximately 0.60 Addis, 0.72 Bishoftu, 0.97 Beijing, 0.96 Delhi and 0.69 Pasadena. The low-OC/EC reference changes the site pattern rather than uniformly fixing it.

**Interpretation / limits:** Reference coefficients are from the new common-split fit, not the historical original-split fit. HIPS is an assumed EC-equivalent, not TOR. Bishoftu contains only three sampled calendar months, so its intervals are particularly fragile. Negative predictions are not clipped. Other quality comments remain unadjudicated.

**Discussion:** Would we accept a candidate that improves Addis but substantially worsens another existing site?

**Sources:** tor_spartan_stability_2026-09-17/five_site_fixed_models.csv; five_site_predictions.csv; five_site_explicit_exclusions.csv; audit.json.

## Slide 12 — Near-one slope or near-zero intercept can still hide poor agreement

Use these plots to prevent the coefficient summary becoming abstract. The same two model families are applied to each site. Addis remains displaced below the HIPS-equivalent. Beijing has a weak sample-by-sample relationship despite a near-one pooled-model slope. Pasadena illustrates a near-zero intercept with a slope far above one.

**Interpretation / limits:** Axis limits differ because the sites have very different observed ranges; do not compare cloud sizes visually. These are retrospective target diagnostics with HIPS as the reference. No outcome-based point removal was used to make these three plots.

**Discussion:** Which joint criterion captures acceptable prediction rather than an attractive regression line?

**Sources:** tor_spartan_stability_2026-09-17/five_site_predictions.csv and five_site_fixed_models.csv. Recomputed OLS rows agree with stored values to 1e-10.

## Slide 13 — Delhi’s near-one overall slope does not persist across the lot subset

The 145-filter primary cohort includes 84 observations on the source-supported lots and 61 on lot 253. The lower-lot subset has a substantially different slope and correlation.

**Interpretation / limits:** Do not conclude that lot 253 caused the difference. The subset also changes time and concentration distribution. This motivates checking domain and batch transfer explicitly, not selectively excluding points to improve fit. The seven excluded filters are independently flagged as invalid by the laboratory; all-152 sensitivities are retained in the source tables.

**Discussion:** Do we treat lot 253 as a separate transfer evaluation until preprocessing and calibration support are confirmed?

**Sources:** tor_spartan_stability_2026-09-17/five_site_fixed_models.csv; five_site_predictions.csv; five_site_explicit_exclusions.csv; audit.json.

## Slide 14 — No saved candidate meets both of the proposed Addis targets

This plot is a screen of existing predictions, not a successful optimization. Each point is a saved model evaluated on the same Addis reserve. A new target-driven search could differ, but must be selected on development data and tested without re-adjustment.

**Interpretation / limits:** The intercept is not the AIRSpec spectral baseline. Do not shift or scale displayed predictions to force the line into the target rectangle. A post-calibration correction would be a separate model and require its own development/evaluation separation. None of the saved candidates passes both the stated OLS constraints. Deming sensitivity is documented separately.

**Discussion:** Are these absolute tolerances scientifically appropriate, and what minimum TOR and sample-level agreement should be required?

**Sources:** aethmodular_tests_2026-09-16/results/model_metrics.csv. /mnt/data/spartan_coverage_check/saved_models_intercept_slope_screen.csv and coverage_audit.json.

## Slide 15 — We can examine 22 sites—but only five have the verified spectral inputs here

The inventory found 1,631 candidate pairs across 22 site codes, of which 1,617 agree by sampling date. Twelve dates are missing and two disagree. Raw HIPS has 28 site codes; processed HIPS has 27. Those are archive snapshots, not a live network census.

**Interpretation / limits:** The named v2 HIPS snapshot was used for the inventory; 134 Fabs values differ from v1, and authoritative version status has not been resolved. Older FTIR uncertainty notes are not adjudicated. Do not add these pairs to earlier cohorts as independent observations. The appendix preserves the complete existing-product site table.

**Discussion:** Should the 22-site archive be used first for a quality/provenance review and descriptive benchmark, before requesting more spectra?

**Sources:** spartan_coverage_check/coverage_audit.json; reported_product_candidate_pairs.csv. tor_spartan_stability_2026-09-17/archived_22_site_OLS.csv. Aethmodular_Experiment_Lab/data/manifest.json.

## Slide 16 — The remaining reference questions cannot be answered by tuning alone

The corrected source-export comparison uses each sampler’s own volume. It does not confuse µg/filter with µg/m³. Three unflagged pairs have reported FTIR/TOR concentration ratios from about 0.639 to 0.813.

**Interpretation / limits:** Five paired filters are a small check, and two have pairing flags. The spectrum row identifiers were not definitively linked to physical filter IDs. Existing deployed product values can be compared by the documented filter metadata, but no new spectrum-calibration score is claimed here. Availability of additional paired Addis or Delhi quartz remains a request, not data already secured.

**Discussion:** Who can resolve the Adama crosswalk, and which independent quartz measurements are realistically available next?

**Sources:** aethmodular_tests_2026-09-16/results/adama_source_checked_pairs.csv; adama_audit.json. Raw batch-54 exports in Aethmodular_Experiment_Lab/data/adama/.

## Slide 17 — The Colab workspace supports a larger—but auditable—next experiment

The Colab notebook and complete ZIP work together. It supports changing preprocessing, model family and candidate cohorts while persisting model configurations, memberships, metrics and failures. Resume was checked, as were two repeated-source seeds and all five target-site-out code paths.

**Interpretation / limits:** The small neural-network check did not converge within its iteration budget and was recorded; nonconverged candidates are blocked from automatic selection by default. Raw AIRSpec was exercised on two rows only; the full source-tree rebuild was not executed. These are software checks, not evidence of scientific calibration acceptance.

**Discussion:** Choose one bounded preset and one declared selection objective, rather than launching every variation at once.

**Sources:** Aethmodular_Experiment_Lab/validation/validation_receipt.json; feature_checks.json; presets/intercept_slope.json; METHODS_AND_LIMITATIONS.md.

## Slide 18 — Three decisions will make the next run more informative

The main message is not that no solution is possible. The experiments show that improving one metric at Addis does not automatically improve TOR prediction or transfer. Use the next batch to test an explicit hypothesis under a declared domain and acceptance rule.

**Interpretation / limits:** The 73 Addis reserve and all existing site snapshots have been examined. Future evaluations of them remain retrospective. New observations would strengthen an eventual confirmation. No evidence here establishes a causal explanation for the Addis discrepancy.

**Discussion:** Which single next experiment and which upstream data request should be prioritized before the next weekly meeting?

**Sources:** All source packages listed in backup slide 26 and the accompanying source ledger.

## Slide 19 — Backup: exact spectral and calibration choices

This detail matters because the meeting referred approximately to a 1500 cm⁻¹ lower bound, whereas the actual corrected array begins at 1425.804 cm⁻¹. The completed batch kept the existing array’s lower edge rather than silently introducing another cutoff.

**Interpretation / limits:** These specific spectra and candidate selection rules differ from any later mean-correlation implementation in the repository. This deck reports the frozen experiment’s actual rule, not an assumed replacement.

**Discussion:** Confirm whether the next run should preserve these exact choices or explicitly change one of them.

**Sources:** aethmodular_tests_2026-09-16/results/frozen_experiment.json; data_audit.json.

## Slide 20 — Backup: combined PMF groups remain a secondary result

Combined groups preserve the original dominant-source assignments and take the union of the two corresponding groups. They do not sum factor contributions and redefine dominance.

**Interpretation / limits:** With two month clusters, a numerically narrow bootstrap interval is not strong evidence of generalization. PMF matching covers only 102 of 239 Addis filters. Keep this secondary to the seasonal experiments as discussed.

**Discussion:** Should combined-source groups remain exploratory until there is a stronger reference and a larger target development set?

**Sources:** aethmodular_tests_2026-09-16/results/combined_pmf_contrasts.csv; data_audit.json; frozen_experiment.json.

## Slide 21 — Backup: five-site frozen-model coefficients

These are exact main-transfer populations: Addis 73, Bishoftu 26, Beijing 192, Delhi 145, Pasadena 158. The complete package also retains full-grid control, all packaged observations, lot restrictions and questionable-note sensitivities.

**Interpretation / limits:** Do not compare the two model families with the older 22-site products as a controlled AIRSpec effect. Distinct cohorts and versions prevent that inference.

**Discussion:** Use this table for exact coefficient questions.

**Sources:** tor_spartan_stability_2026-09-17/five_site_fixed_models.csv; audit.json.

## Slide 22 — Backup: older reported products across 22 sites — 1/2

This is a descriptive benchmark of existing lab-reported products. The two FTIR workbooks span older batches; finite values were matched on site plus complete filter ID including suffix, using the named HIPS v2 snapshot.

**Interpretation / limits:** There were 1,631 candidate pairs; 12 lack dates and 2 have conflicting dates, leaving 1,617 date-agreeing pairs. Uncertainty notes remain unadjudicated. HIPS v1 and v2 differ numerically at 134 Fabs values. IDBD n=3 and USNO n=6 estimates are particularly fragile.

**Discussion:** Which archive version and quality flags should govern any future all-site descriptive analysis?

**Sources:** tor_spartan_stability_2026-09-17/archived_22_site_OLS.csv; archived_22_site_pairs_used.csv; spartan_coverage_check/coverage_audit.json.

## Slide 23 — Backup: older reported products across 22 sites — 2/2

This is a descriptive benchmark of existing lab-reported products. The two FTIR workbooks span older batches; finite values were matched on site plus complete filter ID including suffix, using the named HIPS v2 snapshot.

**Interpretation / limits:** There were 1,631 candidate pairs; 12 lack dates and 2 have conflicting dates, leaving 1,617 date-agreeing pairs. Uncertainty notes remain unadjudicated. HIPS v1 and v2 differ numerically at 134 Fabs values. IDBD n=3 and USNO n=6 estimates are particularly fragile.

**Discussion:** Which archive version and quality flags should govern any future all-site descriptive analysis?

**Sources:** tor_spartan_stability_2026-09-17/archived_22_site_OLS.csv; archived_22_site_pairs_used.csv; spartan_coverage_check/coverage_audit.json.

## Slide 24 — Backup: TOR correlation and actual predictive skill

The common-split low-OC/EC reference has R² near 0.50 on the full test but Q² −3.702 because absolute prediction errors are large. This is the clearest reason not to optimize correlation alone.

**Interpretation / limits:** The constant benchmark is the evaluation-set mean, not a separately trained mean predictor. Use the named definition consistently and report errors, scale and bias alongside it. These source results are on a single fixed outer split; the conditional site bootstrap does not include training-set variation.

**Discussion:** Should the acceptance rule require both R² and Q², and what absolute error tolerance is meaningful on the intended loading domain?

**Sources:** aethmodular_tests_2026-09-16/results/model_metrics.csv; analysis_helpers.py; frozen_experiment.json. tor_spartan_stability_2026-09-17/tor_metric_recalculation.csv.

## Slide 25 — Backup: full-grid control and the cross-site population windows

This supports the conclusion that masking is not uniformly beneficial. Beijing R² decreases from 0.365 to 0.282; Delhi decreases from 0.771 to 0.704; the changes elsewhere differ.

**Interpretation / limits:** The site periods and loading distributions differ. This is not a matched cross-site causal experiment. The original AIRSpec cache was used without a full raw rebuild. Coefficients were frozen and applied without site refits.

**Discussion:** Do the available site windows give enough coverage for the next transfer claim?

**Sources:** tor_spartan_stability_2026-09-17/five_site_fixed_models.csv; five_site_reference_audit.csv; audit.json.

## Slide 26 — Backup: result packages, provenance and interpretation boundaries

Use the package sources to trace every number. The deck includes underlying chart tables, vector/raster figures, editable slide text and tables, and this speaker-note transcript.

**Interpretation / limits:** No MA350 active-interval qualification was performed in this testing batch. No new inference that HIPS or FTIR is the correct Addis EC reference is supported. Unit semantics, raw-baseline verification and complete laboratory QC remain relevant upstream checks. The default notebook’s local success does not establish live Colab authentication or the full large search.

**Discussion:** Record which next experiment and which upstream checks the team agrees to prioritize.

**Sources:** Source_Ledger.csv gives the exact relative files. Commit reviewed before the batch: d54602e6e7f6b8c01ef664b829e66287cb75903e. Prewarm manifest commit: b7f258746fb749bd77d9fc4a9e9d9de16cac3302, git_dirty=true.
