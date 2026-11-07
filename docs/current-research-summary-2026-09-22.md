# FTIR calibration transfer to Addis: current scientific position

Evidence reviewed through **22 September 2026**. This is the current synthesis
for the Addis FTIR/HIPS work, not a replacement for the dated experiment records
or a review of every subproject in this repository.

## Main question and claim

**Question:** How reliably can an IMPROVE-trained FTIR EC calibration transfer
to Addis, and under what conditions?

**Claim supported now:** The locked calibration is numerically reproducible,
and its held-out IMPROVE EC performance can be quantified. At Addis, changing
the calibration and spectral background correction changes the relationship
between predicted EC and optical absorption. That relationship is a transfer
diagnostic; independent Addis EC accuracy and a validated operating range have
not yet been established.

The scientific contribution is presently **a reproducible assessment of transfer
limitations**, not a validated Addis EC measurement method or an identified
chemical cause of the discrepancy. Retain AIRSpec as the current operational
EC preprocessing default while investigating applicability; the paired VIBES
benchmark does not establish an overall EC improvement or method equivalence.

The next confirmatory claim would be: *A frozen calibration meets predeclared
error and coverage requirements against independently measured EC on
identity-confirmed Addis samples across a stated loading/seasonal domain.*
Those requirements and the independent evaluation set still need to be fixed.
Do not choose them retrospectively from the inspected failures.

## Established findings within their tested scope

Here, “established” means supported by the specified saved results or completed
reproduction. It does not imply independent validation of all upstream inputs.

| Finding | Population and result | Limit and consequence |
|---|---|---|
| **The locked AIRSpec calculation reproduces.** | Native OpenResearch run `5e953103-e1b4-4218-8ddf-f9c67188bfd4`, source `fb0abea`: all 84 column checks across 11 historical/prior-local tables pass at absolute tolerance 1e-6; maximum difference 2.47e-13. | Reuses corrected spectral caches. It does not rerun full-pool baseline correction or introduce independent Addis truth. [Run and recipe](openresearch-retrospective/native-execution-2026-09-22.md). |
| **Source-domain prediction has a measured error.** | Historical lowest-OC/EC 800: 606 training and 194 test rows, disjoint sites; corrected DF1=6 selects k=5. Recomputed test RMSE **3.859 µg/filter**, MAE **2.523**, bias **+1.085**. | A single source-domain split, without a new refitting interval in this review. This is not Addis accuracy. [Predictions](../research/ftir_hips_chem/output/tables/airspec_locked_reproduction/ocec_df6_predictions.csv). |
| **Background correction changes the Addis optical comparison.** | Historical fixed 190-filter subset, x=HIPS Fabs/10: raw low-OC/EC to AIRSpec DF1=6 changes the saved OLS intercept **−3.22 → −1.62 µg/m³**, while optical-disagreement RMSE increases **1.16 → 2.41 µg/m³**. | These are reproduced historical OLS diagnostics, not a new errors-in-variables fit or independent thermal errors. A smaller intercept alone is not an accuracy improvement. [Saved table](../research/ftir_hips_chem/output/tables/airspec_locked_reproduction/reproduced_tables/addis_metrics_corrected.csv). |
| **The completed paired EC benchmark does not establish an overall VIBES advantage.** | Full pool: same **2,327** test filters, AIRSpec/VIBES RMSE **3.224/3.296 µg/filter**. Restricted comparison: same **137** test filters, RMSE **2.548/2.420**, but MAE **1.581/1.824**. | Full-pool ΔRMSE (VIBES−AIRSpec) **+0.071**, existing 95% interval **[−0.083, +0.264]**. Both cohort-level RMSE intervals include zero; this is not equivalence. Methods are paired within each cohort; cohorts are not interchangeable. [Scores](../research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09/calibration_scores.csv), [audit and interval scope](vibes-subgroup-audit-2026-09-21.md). |
| **Better blank removal does not imply better EC prediction.** | Saved held-out ETAD blank residual RMS medians: AIRSpec **5.77e-4**, VIBES **6.73e-6 absorbance**, nine blanks. Synthetic recovery is better with VIBES at amplitudes 0.05 and 0.15, worse at 0.01. | Nine independent parent blanks, reused across amplitudes. Field blanks are not guaranteed chemically empty; synthetic additions do not establish recovery of ambient EC or identify a species. [Blank summary](../research/ftir_hips_chem/output/tables/vibes_subgroup_audit/blank_summary.csv), [recovery summary](../research/ftir_hips_chem/output/tables/vibes_subgroup_audit/injection_summary.csv). |

**Metric discipline:** calibration errors above are µg/filter; Addis optical
comparisons are µg/m³; blank residuals are absorbance. The historical AIRSpec
table's `R2=0.90424` is **squared correlation**. From the same 194 predictions,
predictive R² = 1−SSE/SST is **0.88234**. The VIBES score table uses predictive R².
Do not compare those two R² definitions as if they were the same metric.

The original AIRSpec reproduction has 194 test rows; the Colab restricted
comparison has 137. Their similar cohort names do not establish identical
membership or evaluation conditions. The 12,808 completed Colab cases are
processing cases, not 12,808 independent EC validation observations.
These source-domain scores describe the recorded models and splits; they do
not account for the entire historical model search or repeated inspection of
Addis optical diagnostics when choosing which candidate to pursue.

## Exploratory findings and hypotheses

| Finding | Evidence and uncertainty | What it supports next |
|---|---|---|
| **A loading-associated failure pattern warrants investigation.** | Full-pool Q3, defined from training loadings: 545 test filters at 30 sites; VIBES increases RMSE by **0.453 µg/filter**, existing pointwise 95% interval **[0.187, 0.743]**. Intervals use 10,000 paired site-cluster draws conditional on saved fits; they omit refitting uncertainty and multiple-comparison adjustment. | Investigate prospective applicability indicators. Q3 uses reference EC; its boundary is not an operational flag available for Addis without independent EC. [Subgroup evidence](../research/ftir_hips_chem/output/tables/vibes_subgroup_audit/subgroup_metrics.csv). |
| **The largest two failures are shared by both methods.** | BRIS1 `1970697` and CACR1 `1973864` account for **52.2% of the net Q3 squared-error increase**. Both methods predict negative EC; VIBES is more negative. All 12 selected correction repeats converge, maximum prediction change **0.001994 µg/filter**. | Gross solver nonconvergence is not supported as the explanation for those two cases. Chemical identity, reference error and physically correct background remain unresolved. Keep cases in evaluation. [Case investigation](../research/ftir_hips_chem/output/tables/vibes_case_investigation/report.md). |
| **Spectral residual information depends on the comparator.** | Saved ftir_46 blocked ΔRMSE for spectra beyond metadata is **−0.060 [−0.134, +0.020] µg/m³**; beyond metadata plus mass it is **−0.126 [−0.199, −0.032]**. Existing block-bootstrap intervals; no new fit here. | Preserve both comparisons. Predicting an FTIR–HIPS residual is not independent thermal validation or chemical attribution. [Paired results](../research/ftir_ec_phase3/output/tables/ftir46/paired_differences.csv). |
| **The 1617 cm⁻¹ feature is a candidate regional marker, not an established cause.** | Later neutral-baseline analysis reports the feature at Addis and Bishoftu; Bishoftu's band does not share the same offset. Earlier site patterns changed with the baseline definition. | Require independent chemical standards and controlled attribution before naming charcoal/eucalyptus or claiming the band causes the offset. [Corrected historical account](../research/ftir_ec_phase3/BAND1617_LEAD_2026-08-23.md). |

Training-only threshold fitting does not make a hypothesis confirmatory when it
was designed after inspecting the outer-test failures. Fresh evaluation is
needed for improvement claims. New work should test whether spectral/loading
indicators available at prediction time identify unreliable predictions, and
report retained coverage together with error.

## Superseded or overstrong claims

| Earlier claim | Current replacement | Correcting evidence |
|---|---|---|
| Low-OC/EC selection beats all random cohorts on held-out RMSE. | Lowest-OC/EC 800 RMSE is 3.415 on 194 filters; random #3 is 3.097 on 154. Different test populations prevent a common-test superiority claim. Higher R² across different populations does not resolve that comparison. | [ftir_11 saved scores](../research/ftir_ec_phase3/output/tables/ftir11/site_held_out_tor_metrics.csv). |
| A smaller Addis intercept validates better EC, or identifies a non-EC absorber. | The intercept is relative to an optical proxy and depends on model, population and estimator. Reduced offset does not identify the correct EC or its chemical cause. Changing a constant MAC rescales x; it cannot remove the fitted intercept by itself. | [Corrected ftir_13 table](../research/ftir_hips_chem/output/tables/airspec_locked_reproduction/reproduced_tables/addis_metrics_corrected.csv), [intercept invariant](../research/ftir_ec_phase3/ftir_25_intercept_invariant.md). |
| ChemSpec EC or BC supplies an independent reference. | EC tracks the FTIR product; BC is derived from Fabs/MAC. Neither arbitrates FTIR versus HIPS independently. | [Provenance findings](open-items.md#establish-the-provenance-of-ec_ftir--substantially-resolved). |
| The early spectral residual learner proves an increment beyond metadata. | The paired comparison without mass includes zero; the mass-adjusted comparison is different and must be named. | [ftir_46 paired contrasts](../research/ftir_ec_phase3/output/tables/ftir46/paired_differences.csv), qualifying ftir_43. |
| A pooled-lot blank correction resolves Pasadena's slope or a large part of the Addis offset. | Blank response must be keyed to the deployed calibration line, not lot alone. ftir_47 withdraws the pooled-lot interpretation; the historical correction reports only a small Addis change. | [Dated correction](../research/ftir_ec_phase3/OFFSET_ADJUDICATION_2026-08-23.md), [per-line ledger](../research/ftir_ec_phase3/output/tables/ftir47/blank_ledger_per_line.csv). |
| The 1617 band identifies the offset's chemical carrier. | The band/offset link was withdrawn. Regional association and a specific source or absorbing mechanism are separate hypotheses. | [Two explicit corrections](../research/ftir_ec_phase3/BAND1617_LEAD_2026-08-23.md). |
| Historical aethalometer/filter crossplots establish co-sampled independent validation. | Legacy timing, ±1-day matching and processing provenance need evidence recovery. A repaired notebook does not repair sampling identity or stored data provenance. | [Active-interval audit](active-interval-matching-2026-09-10.md). |

Older narratives retain value as a record of hypothesis development. Their
stronger mechanistic conclusions are not carried forward by this synthesis.

## Missing evidence and decisions

1. **Independent Addis reference:** thermal EC, method and target definition
   (including TOR versus TOT), units, uncertainty and blanks/MDLs on confirmed
   physical samples. The local search checked 753 filenames and seven exports;
   absence there is not proof the data do not exist in laboratory systems.
2. **Identity and sampling:** confirm the five candidate Adama pairs, the
   authoritative mapping of spectral IDs 4744–4748, and the July 9 start offset
   and July 30 volume mismatch. Adama is not an Addis reference population.
3. **A prospective operating domain:** predeclare input-based applicability
   indicators, minimum retained coverage, acceptable errors and evaluation
   design. Freeze predictions before opening independent outcomes. Seasonal
   statements must name the calendar convention; cross-lot tests must document
   whether sites and time periods are genuinely new.
4. **Measurement conventions:** confirm HIPS uncertainty semantics before
   treating them as measurement-error variances, and resolve the IMPROVE deposit
   area where mass-per-area comparisons depend on it. Existing sensitivity fits
   do not supply those confirmations.
5. **Independent mechanism evidence:** blank/standard recovery across lots and
   weak signals, and chemical evidence to distinguish composition, background
   correction and loading effects. Neither optimizer convergence nor optical
   agreement identifies the correct chemical mechanism.

The [validation readiness report](../research/ftir_hips_chem/output/tables/addis_validation_readiness/report.md)
links the confirmation queue and unsent request. [Open decisions](open-items.md)
retain the detailed measurement questions. No request was sent in this review.
The [253-filter Addis prediction freeze](addis-validation-freeze.md) now preserves
the full-pool AIRSpec/VIBES outputs and a pre-outcome pairing lock procedure;
it does not supply thermal values or an eligible paired cohort.

## Focus for the next research cycle

Use three main figures for the current story: **the frozen source-domain
prediction benchmark**, **the Addis optical comparison explicitly labelled as
a diagnostic**, and **the paired AIRSpec/VIBES result with uncertainty**.
Keep spectra, solver traces and mechanism leads as supporting material.

Prioritize claim-bearing reproductions in this order:

1. Retain the completed ftir_13/native baseline; do not rerun it simply to fill
   another dashboard node. A fresh raw-spectra correction would be a separate scope.
2. Reproduce ftir_15's uncertainty and ftir_46's paired contrasts if they enter
   the final claim; retain their exact comparators, sampling blocks and estimators.
3. Reproduce ftir_47's per-line blank analysis before making a quantitative
   blank-artifact claim; preserve the refuted pooled-lot comparator.
4. Use ftir_11/32 as protocol evidence. Any common-test cohort ranking is a new
   experiment, not a conclusion from unlike historical splits.
5. Gate ftir_41/44 external-reference interpretations on confirmed identities.
   Leave the 25 unexecuted historical sources as a prioritized context backlog.

This selection is based on what supports the main claim. It is not a declaration
that those historical analyses have all been rerun or independently validated.

## Current operational status and verification

The live retrospective contains **101 historical/release records + 3 prospective
contracts = 104 research records**, plus **6 family headings = 110 nodes**. Its
**82 typed links** comprise 45 dependencies, 29 related-work links, five
qualifications and three revisions. Counts describe documentation, not validated
experiments. The separate AIRSpec execution project contains one experiment and
one completed run; the retrospective retains its original completed release run.

The [retrospective overview](openresearch-retrospective/README.md) tracks the
catalog, and [project next steps](project-next-steps.md) tracks execution priorities.

This review independently recomputed both methods' headline RMSE, MAE, bias and
predictive R² from the saved Colab predictions; checked paired test identities,
truth and sites; recomputed Q3 errors and the two-case share; checked all 12
repeat results; recomputed historical AIRSpec test errors and both R² definitions;
and checked the 43 historical and 84 native comparison rows. Results and input
hashes are in the [verification receipt](current-research-summary-checks.json).
Existing bootstrap intervals were read, not recalculated. The review did not
refit models, validate raw laboratory records or regenerate historical figures.
Local-only output links require the preserved evidence bundle; this summary is
not yet a portable release. **Ready within this reviewed scope, with these limits.**
