# FTIR EC calibration — slide text

Exact text extracted from the delivered PowerPoint; charts are not recreated in this text view. See the verified bundle for the 26-slide visual presentation.

Main discussion: slides 1–18. Backup: slides 19–26. Results through 17 September 2026.

## Slide 1

WEEKLY RESEARCH UPDATE

FTIR EC calibration

What transfers—and what does not?

AIRSpec masks • calibration cohorts • TOR stability • SPARTAN transfer

Can better HIPS agreement coexist with reliable TOR prediction?

Question for today

Ahmad Jalil  |  UNBC / UC Davis collaboration

Results through 17 September 2026

## Slide 2

1 / FOLLOW-UP

The September 10 questions now have controlled tests

The sequence is spectra → selected filters → prediction results.

Meeting request / later follow-up

What was done

Status

Use the same regions for selection and fitting

Paired full-grid versus masked PLS fits

Completed

Try 50–100 closer-magnitude analogs

50 / 100 / 500, shape versus magnitude

Completed

Keep seasons central; try combined PMF groups

Seasonal families and two PMF unions

Completed

Does apparent improvement transfer?

Common TOR test + five-site frozen fits

Diagnostic

Explore more variations without losing track

Colab presets, checkpoints and constraints

Workspace ready

Completed experiments are separated from the larger searches that the workspace can run.

02

Source: September 10 transcript, 62:31–73:58; frozen_experiment.json; validation receipts.

## Slide 3

2 / SPECTRA FIRST

The magnitude-matching step narrows the spectral envelope

Existing AIRSpec-corrected spectra; source membership is held to the saved experiments.

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

Lines show medians; bands show the 10th–90th percentiles. Hatched regions are omitted by the upper-cut mask.

03

Source: all_model_membership.csv + prepared source/target AIRSpec arrays. Quantile display, not a new fit.

## Slide 4

3 / COHORTS

A common test population makes the model comparisons interpretable

A source-site split precedes analog selection; target months stay together.

IMPROVE / TOR

ADDIS / HIPS COMPARISON

13,010

239

Eligible physical filters

Physical filters with spectra and HIPS

10,443 development filters
2,567 test filters at 32 held-out sites

166 selection filters
73 retrospective reserve filters

A fixed similar-source subset contains 278 test filters. The 73 Addis reserve filters have already been examined.

04

Source: frozen_experiment.json; data_audit.json; addis_split.csv.

## Slide 5

4 / HOW TO READ THE RESULTS

Keep chemical-reference prediction separate from optical agreement

The regression direction and units stay fixed within each comparison.

TOR prediction

SPARTAN agreement

x = measured TOR EC loading
y = predicted FTIR EC loading
Units: repository µg/filter scale

x = HIPS Fabs / 10
y = predicted FTIR EC concentration
Units: µg/m³ on both axes

Report the slope, intercept, correlation R² and absolute prediction error—not only a good-looking line.

y = a + b x

HIPS / 10 is an EC-equivalent under an assumed MAC of 10 m²/g. It is not independent TOR EC.

05

Source: frozen_experiment.json and tor_spartan_stability audit.json; definitions used in saved metrics.

## Slide 6

5 / MASK EXPERIMENT

Masking changes the predictions, but not consistently for the better

500 analogs per calibration; identical memberships within each full/masked pair.

Paired 95% month-bootstrap intervals • fixed predictions

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

Seasonal family
All 73 reserve filters

2.740 → 2.756

RMS discrepancy, µg/m³
Δ = +0.016
95% interval: −0.062 to +0.123

Pooled agreement improves; Kiremt worsens. The combined seasonal result does not demonstrate an overall gain.

06

Source: headline_mask_results.csv; seasonal_mask_family_contrasts.csv.

## Slide 7

6 / COHORT-SIZE EXPERIMENT

Smaller or magnitude-matched cohorts did not resolve the discrepancy

Season-specific calibrations, each applied to its season; all summaries use the same 73 Addis filters.

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

What changed

Magnitude matching narrows the absorbance range.

What did not

It did not improve the aggregate result for this tested selection rule.

The low-OC/EC reference is closer to HIPS—but its independent TOR behaviour must be checked next.

07

Source: addis_family_metrics.csv, group = All Addis. The low-OC/EC model is a comparison, not validated truth.

## Slide 8

7 / COMMON TOR BENCHMARK

TOR correlation is stronger in the target-like subset than in the full test

Same held-out observations for every candidate; no training-fit R² is shown here.

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

Pooled masked model

R² = 0.505  /  0.801
Q² = 0.475  /  0.789

Full test / similar subset

No configuration in the 47-model batch reaches R² ≥ 0.85 on either common benchmark.

Low-OC/EC reference: Q² = −3.702 on the full TOR test.

08

Source: model_metrics.csv, populations source_full and source_spectral.

## Slide 9

8 / HISTORICAL TEST STABILITY

The historical R² = 0.924 is sensitive to two high-loading observations

Original 440-filter model: 387 fitting filters and 53 test filters from 21 sites.

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

Evaluation

R²

Q²

Slope

All 53 filters

0.924

0.905

1.055

Without PHOE1*

0.653

0.639

0.606

*Two test observations omitted only for sensitivity; the model is not refitted.

This is evidence of evaluation-range influence—not evidence that the filters are invalid.

09

Source: historical_test_predictions.csv; historical_leave_one_evaluation_site_out.csv.

## Slide 10

9 / WITHIN-SOURCE STABILITY

The pooled source result conceals substantial between-site variation

Pooled 500-analog masked model; the same 32 held-out IMPROVE sites.

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

0.312

Median within-site R²

13 / 32

Sites with positive within-site Q²

Overall R² = 0.505; whole-site conditional 95% interval = 0.191–0.642. Broad stability is not established.

10

Source: tor_per_site.csv and tor_site_bootstrap.csv; frozen model, no outer refitting in this analysis.

## Slide 11

10 / FIVE-SITE TRANSFER

The same frozen calibration does not give the same relationship everywhere

OLS of predicted FTIR EC on HIPS / 10; no site-specific refit or offset adjustment.

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

Addis uses 73 retrospective reserve filters; Bishoftu uses 26 released HIPS values; Delhi excludes seven explicitly invalid filters.

11

Source: five_site_fixed_models.csv, population = primary. Intervals: fixed-model month-block bootstrap.

## Slide 12

11 / SEE THE INDIVIDUAL OBSERVATIONS

Near-one slope or near-zero intercept can still hide poor agreement

Blue circles: pooled 500 / masked. Orange squares: low-OC/EC 440. Dashed line: 1:1.

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

Addis / 73

Beijing / 192

Pasadena / 158

Low-OC/EC R² = 0.777
Intercept = −1.349

Pooled slope = 0.967
R² = 0.282

Low-OC/EC intercept = +0.017
Slope = 2.431

12

Source: five_site_predictions.csv, primary populations. Axis ranges differ by site; both axes are µg/m³.

## Slide 13

12 / TRANSFER SENSITIVITY

Delhi’s near-one overall slope does not persist across the lot subset

Pooled masked model, after the seven documented invalid filters are excluded.

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

Evaluation

n

Slope

R²

All retained lots

145

0.963

0.704

Lots 248/251

84

0.577

0.298

The remaining 61 observations are lot 253.

Lot, sampling period and concentration change together. This is not a causal lot-effect estimate.

13

Source: five_site_fixed_models.csv, primary and primary_lots248_251; five_site_predictions.csv.

## Slide 14

13 / INTERCEPT–SLOPE SEARCH

No saved candidate meets both of the proposed Addis targets

Descriptive re-screen of all 47 configurations on the same 73 reserve filters; OLS shown.

[Figure in the PowerPoint; see the package’s `presentation_source/charts/` directory.]

Proposed search constraints

−0.02 ≤ intercept ≤ +0.10
0.90 ≤ slope ≤ 1.10

Intercept units: µg/m³
Slope range is an editable example.

Keep TOR skill and prediction error as independent requirements.

14

Source: model_metrics.csv, addis_reserved_all; saved_models_intercept_slope_screen.csv. No new optimized calibration.

## Slide 15

14 / NETWORK CONTEXT

We can examine 22 sites—but only five have the verified spectral inputs here

Availability is not the same as a quality-approved calibration dataset.

22 sites

5 sites

1,617 date-agreeing pairs of
older reported FTIR EC and HIPS

767 packaged spectral rows
for new-model applications

What the older archive can answer

What the five-site spectra can answer

How widespread are the existing product relationships?

Does a new frozen calibration transfer without a site adjustment?

The 22-site values are not new AIRSpec outputs. Differing models, periods and populations prevent a controlled old-versus-new claim.

15

Source: coverage_audit.json; archived_22_site_OLS.csv; Experiment Lab prepared-data manifest.

## Slide 16

15 / DATA NEEDED TO RESOLVE INTERPRETATION

The remaining reference questions cannot be answered by tuning alone

Adama provides a small thermal-optical check, but not a general Addis validation set.

Adama source-export check

Still unresolved

Five July 2024 PTFE/quartz pairs.
TOR EC: 1.862–3.379 µg/m³.
Median reported FTIR/TOR ratio: 0.694.

A definitive spectrum-to-filter crosswalk.
July 9 timing and July 30 volume flags.
Useful independent Addis quartz EC.

No new-calibration score against Adama quartz was produced from inferred row order or rank agreement.

The current evidence does not identify whether the remaining Addis difference is FTIR transfer, HIPS/MAC behaviour, or both.

16

Source: adama_source_checked_pairs.csv; adama_audit.json; frozen_experiment.json; five-site audit.

## Slide 17

16 / TOOLING READY

The Colab workspace supports a larger—but auditable—next experiment

Available options are not the same as completed scientific experiments.

Available controls

Completed local checks

Still to run / resolve

8 presets; configurable trial budget

6-configuration default batch

Full 75-configuration target search

PLS, ridge, PCR, SVR, trees, small NN

12 unit tests; 22 feature trials

Live Colab authentication and install

Masks, cohorts, scaling, derivatives

All six model families exercised

Full raw IMPROVE baseline rebuild

Site splits, target-site-out, checkpoints

3-trial persistent optimizer check

Prespecified scientific acceptance rule

Default: TOR-first selection. HIPS-agreement mode adds the proposed constraints and can return “no qualifying candidate.”

17

Source: Aethmodular_Experiment_Lab presets, METHODS_AND_LIMITATIONS.md and validation/validation_receipt.json.

## Slide 18

17 / DISCUSSION

Three decisions will make the next run more informative

Proposed next step: a bounded search with TOR safeguards and frozen cross-site evaluation.

Agree on success

1

Keep intercept and slope targets, but specify TOR R²/Q², absolute error and convergence requirements.

Agree on the evaluation domain

2

Report full and target-like TOR tests; hold out sites/months for model selection and show each site separately.

Resolve reference evidence

3

Prioritize Adama identities, source-unit/calibration provenance, HIPS version review and independent quartz EC.

Working conclusion: the tested changes are informative, but no replacement EC calibration is ready to adopt.

18

Synthesis of completed result packages and September 10 meeting questions. Proposed decisions, not new findings.

## Slide 19

METHODS APPENDIX

Backup: exact spectral and calibration choices

These settings describe the completed 47-configuration batch—not all options in the Colab workspace.

Item

Completed-batch definition

Available grid

2,002 channels; 1425.804–3998.423 cm⁻¹

CO₂ mask

Exclude the inclusive 1800–2500 cm⁻¹ interval

Upper-cut mask

CO₂ mask plus exclude channels above 3500 cm⁻¹

Selection

Signed centred Pearson score to selection-target median; eligible physical filters first

Magnitude rule

Closest absolute log RMS-amplitude distance within a fixed top-500 shape shortlist

PLS components

5-fold source-site GroupKFold; ≤30; first local minimum within the 1-SE rule

Feature scale

PLS scale=False; amplitude retained during regression

BACKUP · 19

Source: frozen_experiment.json; data_audit.json.

## Slide 20

PMF APPENDIX

Backup: combined PMF groups remain a secondary result

Same held-out observations within each combined-versus-separate comparison.

Source-associated group

n

Months

Separate

Combined

Δ RMS

Wood + charcoal

17

4

2.379

2.489

+0.109

Sea salt + polluted marine

12

2

2.919

2.801

−0.119

Wood + charcoal

Sea salt + polluted marine

No observed improvement from combining; the interval for the change includes zero.

Small observed reduction, but only 12 filters across two months.

These are PMF-associated subsets, not pure-source reference materials. The wood-only selection side contains three spectra.

BACKUP · 20

Source: combined_pmf_contrasts.csv. RMS discrepancy is against HIPS / 10, in µg/m³.

## Slide 21

TRANSFER APPENDIX

Backup: five-site frozen-model coefficients

OLS; y = predicted FTIR EC, x = HIPS / 10; intercept and RMSE in µg/m³.

Site

Frozen model

n

Slope

Intercept

R²

RMS

Addis

Pooled masked

73

0.599

−1.006

0.692

3.068

Addis

Low-OC/EC

73

0.861

−1.349

0.777

2.118

Bishoftu

Pooled masked

26

0.722

−1.043

0.407

1.893

Bishoftu

Low-OC/EC

26

0.821

−0.301

0.564

0.915

Beijing

Pooled masked

192

0.967

−0.186

0.282

1.203

Beijing

Low-OC/EC

192

0.398

+1.004

0.070

1.205

Delhi

Pooled masked

145

0.963

−1.549

0.704

3.284

Delhi

Low-OC/EC

145

1.767

−1.677

0.778

6.238

Pasadena

Pooled masked

158

0.687

+0.209

0.212

0.281

Pasadena

Low-OC/EC

158

2.431

+0.017

0.548

0.860

BACKUP · 21

Source: five_site_fixed_models.csv, population = primary. Not independent TOR validation.

## Slide 22

ARCHIVE APPENDIX

Backup: older reported products across 22 sites — 1/2

OLS; y = older reported FTIR EC, x = HIPS / 10. Not the new AIRSpec model applications.

Site code

n

Slope

Intercept (µg/m³)

R²

AEAZ

48

0.373

+0.991

0.197

AUMN

39

0.431

+0.191

0.449

BDDU

42

0.839

−0.466

0.601

BIBU

24

1.400

−1.446

0.865

CAHA

44

0.993

+0.028

0.596

CASH

30

1.062

−0.004

0.113

CHTS

126

0.906

+0.069

0.455

ETAD

113

1.996

−4.813

0.795

IDBD

3

1.267

+1.088

0.487

ILHA

105

0.695

+0.043

0.815

ILNZ

91

0.796

−0.011

0.823

Do not interpret a good slope/intercept alone as strong agreement. CASH: slope 1.062, intercept −0.004, R² 0.113.

BACKUP · 22

Source: archived_22_site_OLS.csv; 1,617 date-agreeing candidate pairs. QC/provenance review incomplete.

## Slide 23

ARCHIVE APPENDIX

Backup: older reported products across 22 sites — 2/2

OLS; y = older reported FTIR EC, x = HIPS / 10. Not the new AIRSpec model applications.

Site code

n

Slope

Intercept (µg/m³)

R²

INDH

29

1.204

−0.193

0.634

KRSE

32

0.861

+0.054

0.534

KRUL

63

0.779

+0.147

0.623

MXMC

32

1.177

−0.662

0.706

PRFJ

24

0.128

+0.146

0.024

TWKA

132

1.026

−0.172

0.815

TWTA

159

0.802

+0.031

0.824

USNO

6

0.315

+0.213

0.088

USPA

158

0.693

+0.146

0.517

ZAJB

160

0.891

−0.150

0.740

ZAPR

157

0.912

−0.053

0.781

Do not interpret a good slope/intercept alone as strong agreement. CASH: slope 1.062, intercept −0.004, R² 0.113.

BACKUP · 23

Source: archived_22_site_OLS.csv; 1,617 date-agreeing candidate pairs. QC/provenance review incomplete.

## Slide 24

METRIC APPENDIX

Backup: TOR correlation and actual predictive skill

Correlation R² = squared Pearson r. Q² = 1 − SSE / SST, using the test-set mean for SST.

Calibration

R² full

R² similar

Q² full

Q² similar

Pooled / full grid

0.483

0.822

0.443

0.800

Pooled / masked

0.505

0.801

0.475

0.789

Dry / masked

0.565

0.796

0.487

0.789

Belg / masked

0.162

0.831

−0.534

0.828

Kiremt / masked

0.278

0.761

−0.424

0.691

Low-OC/EC reference

0.504

0.640

−3.702

−0.135

Negative Q² means larger squared error than the constant test-set mean benchmark. High correlation can coexist with bias.

BACKUP · 24

Source: model_metrics.csv; exact populations source_full (2,567) and source_spectral (278).

## Slide 25

TRANSFER APPENDIX

Backup: full-grid control and the cross-site population windows

The paired pooled models use the same calibration filters; the target observations are also held fixed.

Site

n

Sampling dates in primary cohort

R² full

R² masked

Addis

73

2022-12-07 to 2025-08-02

0.637

0.692

Bishoftu

26

2025-10-20 to 2025-12-29

0.264

0.407

Beijing

192

2022-07-05 to 2024-12-08

0.365

0.282

Delhi

145

2022-07-17 to 2026-02-11

0.771

0.704

Pasadena

158

2022-07-22 to 2023-11-14

0.219

0.212

Bishoftu has only three calendar months. Different site windows limit any claim of all-season or network-wide stability.

BACKUP · 25

Source: five_site_fixed_models.csv, primary full-grid and masked rows.

## Slide 26

SOURCES & LIMITATIONS

Backup: result packages, provenance and interpretation boundaries

Every numerical figure is backed by the packaged tables or saved predictions.

Evidence package

Used for

September 10 meeting transcript

Masking, 50/100-filter cohorts, seasonal focus, spectra-first presentation

47-configuration batch / 16–17 September

Fixed splits, masks, cohorts, PMF unions, predictions and rerun receipt

TOR/SPARTAN stability / 17 September

Historical influence, source-site stability, 3 frozen models across 5 sites

Coverage audit / 17 September

22-site older-product inventory, date and version flags

Experiment Lab / 17 September

Available presets and local software-validation status

Interpretation boundaries

Retrospective observations • conditional uncertainty • no causal explanation established • no replacement calibration adopted

BACKUP · 26

See the accompanying Source_Ledger.csv, Presenter_Notes.md and Numerical_Checks.json.

