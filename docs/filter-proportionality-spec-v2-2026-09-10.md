# Specification v2: proportionality and limits of temporal transfer

This is a separately frozen extension to the completed reported-date stability
analysis. It was chosen after inspecting that analysis, including its eight-quarter
Addis improvement, residual date structure and source-linked IDs 11/17. The original
specification and results remain unchanged. This file is saved and hashed before
fitting the new proportional model or the restricted-training sensitivity.

## Primary extension: identical holdouts, three predictions

Use the previous baseline-variant populations, physical filter IDs, training/test
ID lists, calendar quarters, reported dates, and support decisions exactly as saved.
Analyze Addis first, then Beijing, Delhi and JPL. Keep diagnostic, ratio/1× MDL,
1.5×, 2×, 3× and 5× memberships; do not recompute eligibility. ETAD-0037 stays in
its baseline populations. Its completed omission analysis is retained by reference,
not repeated or expanded into an exclusion search.

The three equal-filter-weight predictions are:

1. Training-median reported HIPS, copied from the completed analysis.
2. Proportional least squares, H = kE, with k = sum(EH) / sum(E²), fitted only on
   that fold's training IDs.
3. OLS with intercept, H = a + bE, copied from the completed analysis.

H is reported HIPS in Mm⁻¹ and E is reported FTIR-predicted EC in µg/m³. k and b
are statistical relationship coefficients, not established physical absorption
efficiencies. No positivity filtering, prediction clipping, weights, denominator
optimization, preprocessing or flexible model. Nonpositive reported EC values
remain in the diagnostic cohort and can yield nonpositive proportional predictions.

Preserve the common support rule: at least ten training filters and two distinct
finite EC values. Retain empty and unavailable folds and their population counts.
All three models use exactly the same supported test filters. The proportional
denominator must be finite and positive; otherwise fail explicitly.

Evaluate leave-quarter-out stability and earlier-only later-period prediction
separately. The former includes future training dates and is not prospective.
Dates are reported dates, not verified active sampling periods.

## Fixed reporting and interpretation

For each quarter and overall, report every model's signed error (prediction minus
HIPS), MAE and RMSE. The primary paired contrast is proportional MAE minus
intercept-model MAE; positive values favor the intercept. Retain training-median
errors as context. Report common counts, training/test EC and date ranges, IDs,
nonpositive predictor/prediction counts, split fingerprints and original source links.

Present both equal-filter and equal-quarter summaries together. Equal-quarter MAE
and signed error average supported quarterly MAEs and biases; equal-quarter RMSE
is sqrt(mean(quarterly MSE)), not the average of quarterly RMSEs. Report the largest
quarter's fraction of evaluated filters. Neither weighting is selected after seeing
its sign. Surface the retained Beijing 2024Q3 failure and Delhi's directional
later-period error, including their counts and range-extrapolation diagnostics.

No binary practical-adequacy margin is invented. Describe the size, consistency
and concentration of the paired improvements, including losses and unavailable
folds. A full-cohort intercept alone does not establish predictive necessity. No
physical origin, universal offset or upstream FTIR validation follows from a win.

## Labeled secondary extension: Addis ID-11 training sensitivity

For Addis only, repeat the existing earlier-only calendar folds in each fixed
population. Test only ID-11 filters in the held-out quarter. Compare training on
all eligible earlier filters against training on earlier ID-11 filters only.
Use the three models above, with the same training support rule independently
applied to each training choice. Compare only folds supported by both choices,
and identical ID-11 test filters within those folds. Save unsupported folds, their
training/test counts and reasons; no alternative boundaries or minimum are tried.

Report restricted-training MAE minus all-earlier-training MAE, signed errors,
paired predictions and both weighting schemes on this common evaluation set.
Reconcile the all-earlier median/OLS predictions to the previous results on these
same filters. Do not compare its aggregate to an earlier aggregate with different
test membership as if it were an improvement.

Use actual source-linked CalibrationSetId == '11'; do not infer IDs from date.
Dates and identifiers are confounded. This tests whether including the earlier
group helps prediction of later ID-11 filters, not whether a model change caused
the difference. Do not subtract observed residual means or label groups corrected.
LotId remains an unconfirmed batch field; ChemSpec 217/218 are not FTIR versions.

## Bounded evidence work and completion

Extend a copy of the unsent upstream packet with definitions and applicability of
IDs 11/17, model and reference-target mappings, and original training/evaluation
membership for the linked filters. Preserve the exact ChemSpec rows and hashes.
No external messages are sent and no local parser is changed.

Assemble one evidence package for JPL:USPA-0257 / MA350-0229 session 46 from the
existing matched records, source links, candidate dates and relevant local
documentation. Seek filter-linked collection mode/active periods, scoped correction
history, an applicable quality rule and source-backed channel units. Limit any
additional search to this candidate and its relevant documentation; do not repeat
the completed SQLite query or conduct a broad inventory. Document both recovered
facts and unresolved requirements. Corrected observations can be valid when their
history and applicability are supported; require documented provenance rather than
an absence of corrections. Absorption still requires additional optical evidence.

Deliver a separately reproducible workflow, prediction and paired block tables,
scientific report and graph-focused notebook with notes. Preserve both completed
baselines. Evidence gaps do not prevent completion of the filter-only extension.
