# FTIR calibration transfer: current scientific position

Evidence reconciled **7 November 2026** against source `cb2e1469ec222719c347b9b5f672765ede3ed7c7` and the public data release `data-82d701248232`. This update distinguishes saved findings from calculations newly verified in the closeout implementation. The [September synthesis](current-research-summary-2026-09-22.md) and its [numerical verification receipt](current-research-summary-checks.json) remain historical evidence; that receipt does not validate the November analyses.

## Question and present conclusion

**How reliably can an IMPROVE-trained FTIR EC calibration transfer to Addis, and under what conditions?**

The existing work supports a reproducible investigation of spectral preprocessing, calibration transfer, numerical convergence, Ethiopian seasonality and optical-response assumptions. It does **not** yet establish independent absolute EC accuracy for the Addis target population or select one universally validated target EC product.

PTFE anchoring remains a research candidate. Converged VIBES is a numerical improvement with a separately defined downstream scope. EPO's historical source score is not an end-to-end held-out evaluation: its projection used held-out source chemical labels. The closeout implementation repairs that pipeline and withholds the old comparative score until corrected, identity-verified predictions are available. A code regression test is not a numerical rerun on the scientific data.

## November evidence and the decision each supports

All numerical findings in this table are **reported saved results**, unless the implementation receipt explicitly records a new reproduction. Different populations, model versions and metrics must not be pooled into one ranking.

| Question | Saved evidence | Interpretation and limits |
|---|---|---|
| Does broad-signal recovery follow from successful optimization? | The documented 6 November continuum rerun has 89/90 successful fits. The failed unspiked blank leaves eight independent parent blanks for VIBES recovery and nine for spline. At 0.05 AU, reported VIBES recovery is approximately 98.27% for four bands, 18.80% for the broad hump and 0.138% for the power-law addition. | Successful fitting does not establish preservation of these broad synthetic shapes. The additions are repeated tests on parent blanks; paired method differences need common available parents. This does not identify the real continuum's chemical origin. [Run and limits](vibes-gallery-continuum-check.md). |
| Does anchoring improve transfer diagnostics? | The fixed-18-component saved source MedAE is 1.15 µg/filter for production VIBES and 1.10 for anchored VIBES. Their Addis aethalometer intercepts are approximately −0.955 and +0.098 µg/m³. The three unflagged provisional Adama pairs give median FTIR/TOR of 0.96 and 1.31. | A promising tradeoff, not a demonstrated absolute accuracy improvement or equivalence. Adama is a different site, lot and sampling arrangement with provisional mapping. Full-source target predictions and held-out source errors are different products. [Comparison implementation](../research/ftir_hips_chem/workflows/run_hips_response_20261103.py). |
| Can the old EPO score select a method? | Historical EPO projection jointly regressed source spectra on EC, OC, other PM, Fe and K before site splitting. Its 1.24 µg/filter saved MedAE and heavy-subset ratio therefore use a label-dependent transformation informed by test rows. | Withhold the source comparison until fold-specific training-only EPO is rerun and all candidates have reconciled physical-filter identities, folds and coverage. No direction or magnitude of leakage bias is assumed. Target optical comparisons are preserved as exploratory. [Corrected workflow](../research/ftir_hips_chem/workflows/run_intercept_fixes_20261103.py). |
| What does Newton change? | The adoption study reports 4,452/4,452 network analyses fitted and certified, and rebuilt source test RMSE 3.8501 → 3.8499 µg/filter. Functional-group, lot, blank-bank and continuum checks are documented. | Supports numerical control and reproducibility, not improved absolute EC accuracy. Season-composition and fingerprint-grouping impact checks remain necessary before numerical v2 is promoted into those conclusion-bearing analyses. Preserve both runs meanwhile. [Adoption study](vibes-newton-adoption-tests-2026-11-03.md). |
| Is seasonality a central result? | The matched study has common calibration populations/folds and 232 direct-HIPS Addis physical-filter pairs. Raw shared-window median FTIR-minus-HIPS differences at MAC 10 are approximately −0.896, −0.268 and +1.420 µg/m³ for Bega, Belg and Kiremt. The pattern persists within lot 251 and the 188-filter complete-month 2023–24 subset. | Preserve the calendar and cohort. New reporting uses `belg_feb`: Bega Oct–Jan, Belg Feb–May, Kiremt Jun–Sep; `dry_feb` is a labelled sensitivity. These are optical disagreements, not thermal EC errors or validated seasonal correction factors. [Matched study](drive-calibration-seasonal-study-2026-11-03.md). |
| What does spectral ML add? | The completed seasonal ML work evaluates physical filters with blocked later-cycle tests; task and test cycle change the raw/corrected ranking. | A predictive diagnostic, not chemical identification or a season-specific EC correction. Do not restart it as uncompleted work. [Study and evaluation](https://github.com/ahzs645/aethmodular/pull/15). |
| How should HIPS-response changes be interpreted? | An IMPROVE TOR-derived response model changes the Addis optical comparison. Its fitted bend is around optical-depth coordinate 0.3, using a geometric-mean HIPS/reference coordinate. | An empirical fitted feature, not an instrument-certified raw-HIPS cutoff. Transferring the response does not create independently measured Addis TOR. Keep raw HIPS, MAC conversion, response scenarios and inferred comparators distinct. [Response model](../research/ftir_hips_chem/workflows/run_hips_response_20261103.py). |

## Reference and metric discipline

| Quantity | Meaning |
|---|---|
| HIPS Fabs | Optical absorption measurement, normally Mm⁻¹. |
| Fabs / assumed MAC | HIPS-derived optical BC/eBC; report the MAC, normally 10 m²/g here. |
| HIPS under an IMPROVE-derived response | Model-based transferred-response sensitivity. |
| HIPS × IMPROVE TOR/HIPS relationship | Inferred comparator, not TOR measured on Addis filters. |
| Matched thermal-optical TOR | Direct reference within its actual paired population and method limits. The Adama pilot is not representative Addis validation. |
| ChemSpec EC/OC and in-house FTIR EC/OC | Two feeds of FTIR-derived products, not independent validation observations. See the provenance rules in [AGENTS.md](../AGENTS.md). |
| Median absolute error (MedAE) | `median(abs(prediction − TOR))`. The historical `improve_error` and misleading `improve_mae` fields used this statistic. |
| Mean absolute error (MAE) / RMSE | `mean(abs(error))` / `sqrt(mean(error²))`; computed as distinct fields in the corrected scorer. |
| Mean signed bias / tail errors | Mean of `prediction − TOR`; explicitly named 90th/95th percentiles of absolute error. |

Source scores use µg/filter, while target optical comparisons use µg/m³. The fixed-k primary source population has finite TOR below 150 µg/filter, an outcome-restricted domain. Report high-loading exclusions/stress cases and candidate availability alongside the common-filter scores. Repeated scans and synthetic injections do not increase the number of independent physical filters or parent blanks. Historical squared correlation and predictive R² also remain different statistics.

The October 28 organic-signal figure's y value is **FTIR residual from seasonal HIPS fit (µg/m³)**: FTIR EC minus its seasonal Deming fitted line. It is neither a simple FTIR−HIPS difference nor an independently measured EC error. The gallery now retains that definition with explanations off and in exports.

## Finite closeout scope

The next endpoint is a **versioned existing-data research release**, not a universally corrected Addis EC product. The implementation batch provides training-only EPO code, source-score provenance checks, corrected terminology, a six-figure review section and strict verification profiles. The original decision bundle and production background have now been recovered and hash-verified. The [reproduction protocol](gallery-closeout-reproduction-2026-11-07.md) records the new continuum run, restored all-candidate raw-spectrum QC rule, matched-resolution EPO control and source/target version boundary. Execution results belong in the [implementation receipt](gallery-closeout-implementation-2026-11-07.md).

The review sequence covers physical-filter coverage, source-reference performance, baseline behavior, Bega/Belg/Kiremt, optical-response sensitivity and candidate tradeoffs. A shared section refers to the active exports; it is not itself a frozen data archive. Freeze and retain the source/data receipt with exported figures.

PR #19 remains a separate Research-only timeline review. Its catalog records source history and saved evidence, not scientific validation. The closeout fix replaces the indexer's checkout-based `missingData` flags with checks of declared primary exports on the active review host. Exact-head CI at `316a53ec423c7344253aef409027b0f607fff23f` passed 20 focused tests, four browser journeys, the research build and public exclusion ([run 42396251333](https://github.com/ahzs645/aethmodular/actions/runs/42396251333)). A loaded export does not establish scientific reproduction; public builds intentionally omit this area.

New Ethiopian sampling/TOR, full PMF scientific qualification, further GPU/WebGPU work and exhaustive historical notebook migration are follow-on work. None is required to state the supported existing-data findings and the unresolved absolute Addis level. The proposed future TOR rule still requires agreement on eligibility, uncertainty, acceptance limits and target coverage.

## Claims not carried forward

Do not claim that a near-zero aethalometer intercept proves correct EC, that the fitted HIPS bend certifies an instrument operating range, that a seasonal residual identifies its chemical cause, or that three provisional Adama pairs select a universally accurate method. Do not transfer the September reproduction receipt to a different November experiment. Historical mechanistic corrections and unavailable independent evidence are preserved in the [September synthesis](current-research-summary-2026-09-22.md) and [open items](open-items.md).
