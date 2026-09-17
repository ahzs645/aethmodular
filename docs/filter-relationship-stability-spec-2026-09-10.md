# Frozen specification: reported-date relationship stability

Written before fitting the models below, on 10 September 2026. The questions and
population choices were motivated by already inspected descriptive results. This
is a frozen analysis specification, not a claim of prospective preregistration.

## Inputs and estimand

Freeze byte-for-byte copies of the descriptive `filter_diagnostics` manifest,
analysis points, original measurements, sensitivity memberships and ChemSpec
source trace before fitting. Verify their hashes against the descriptive manifest.
Retain the original 545 diagnostic and 480 ratio pairs, registry decisions,
physical-filter identities and reported dates. No new permanent exclusions.

The question is whether a within-site linear relationship between reported HIPS
(Mm⁻¹) and reported FTIR-predicted EC (µg/m³) transfers across reported-date
blocks. Its slope is a relationship coefficient, not a physical MAC. Neither
FTIR accuracy nor aethalometer calibration is evaluated here.

## Fixed populations, blocks and models

Process Addis Ababa first, then Beijing, Delhi and JPL with the same specification.
Analyze the diagnostic population, baseline ratio population (1× MDL), and saved
1.5×, 2×, 3× and 5× MDL memberships. Do not recompute eligibility or optimize a
threshold. Display membership counts and predictor ranges even when fitting fails.

Assign fixed calendar quarters (January–March, April–June, July–September,
October–December) using the frozen reported date. Each site's block calendar spans
its full diagnostic date range, including empty quarters. Keep these assignments
for every population and analysis. Reported dates are not verified active periods.
Quarters were chosen for consistent calendar duration and interpretable sample
sizes, without inspecting held-out errors or searching block boundaries.

Two models, fitted only on training filters:

1. Constant prediction = median training HIPS.
2. Unweighted OLS with intercept: HIPS = intercept + slope × FTIR-predicted EC.

No preprocessing, uncertainty weights, pooled fit, forced zero, flexible model,
hyperparameter search or uncertainty substitution. Require at least 10 training
filters and two distinct finite EC values for the paired model comparison.
Otherwise retain the fold with an explicit unavailable status and no predictions.
This support rule is an analysis convention, not a statistical adequacy guarantee.
Report every nonempty test block; flag n < 5 as small without dropping it.

## Two distinct evaluations

Primary: leave one calendar quarter out; train on all other quarters, including
earlier and later dates. This tests stability across the observed record, not
prospective prediction. Secondary: expanding training on strictly earlier quarters,
testing the next calendar quarter. This evaluates later-period prediction.

Physical filter IDs cannot occur in both training and test. Use explicit calendar
blocks because irregular filter dates do not meet the equal-spacing assumption of
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
Training-only model fitting follows the separation principles in the
[cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
and [leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

## Fixed reporting

For each block and population: calendar bounds, actual training/test reported-date
ranges, filter counts and IDs, EC ranges, model coefficients, mean signed error
(prediction minus reported HIPS), MAE, RMSE, baseline MAE minus OLS MAE (positive
means improvement), relative MAE improvement, and number/fraction of test EC values
outside the training EC range. No extrapolation-based exclusion.

Save per-filter predictions and residuals, population/scenario membership, split
identifiers, original HIPS/EC row links and source hashes. Summaries include
filter-weighted held-out errors and equal-block MAE improvement, with unavailable
folds counted. Do not use full-cohort fitted errors as held-out performance.

## Influence and processing metadata

Retain ETAD-0037 in all baseline populations where originally eligible. Repeat the
Addis analyses without it as a named sensitivity, with no registry edit. Compare
coefficient changes to each whole-quarter omission from the corresponding full
cohort. Compare changed predictions/errors on common evaluated filters only, so
removing the difficult evaluation point cannot itself create an apparent gain.
Report its own held-out error separately in the baseline results.

Join actual EC_ftir source metadata by original measurement row: CalibrationSetId,
AnalysisDate/Time, LotId and MDL. Report groups and residual summaries descriptively.
IDs 11/17 are reported calibration-set identifiers until documented otherwise;
LotId is not established as analytical batch. ChemSpec 217/218 are not assumed to
be FTIR versions or batches. Do not fit metadata-selected models or attribute
temporal structure to atmosphere/source without evidence. Explicitly record the
unknown original FTIR target, training population, version mapping and sample
independence. Downstream held-out regression does not validate upstream training
independence. Do not assign a general model RMSE as per-filter uncertainty.

## Separate evidence actions and completion

Draft, but do not send, a narrowly specified upstream question packet for the two
CHTS-0658 EC28203 values (0.93/0.06), including original rows, metadata, source hash
and unified row links. Preserve both; no parser correction or result-role guess.

Attempt one candidate-specific instrument lookup for JPL:USPA-0257, reported UTC
envelope 23 June 2023 16:00 to 24 June 2023 16:00, using its staged session IDs.
Inspect SQLite indexes/query plan and make a bounded read-only date/session query;
do not repeat full-range aggregation. Report exactly what evidence was recovered
and what still prevents a supported interval eBC or absorption comparison.

Completion requires reproducible predictions, block/influence tables and a report
characterizing stability or instability. A positive result is not required. Freeze
this file's hash in the output manifest; any later deviation must be disclosed.
