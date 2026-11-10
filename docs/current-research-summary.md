# FTIR calibration transfer: current scientific position

Updated **7 November 2026 after the completed existing-data research closeout**. Numerical acceptance, independent score arithmetic, the figure-population audit, exact-source public/browser checks and the seven-page visual review are complete. The actual corrected overlay passed at source `2341cbbbab41b9dfde143d36aac5ea84585aea31`; the preserved release remains a review candidate, with no deployment or absolute Addis EC validation implied. See the [final implementation receipt](gallery-closeout-implementation-2026-11-07.md#corrected-overlay-release-verification).

The new source evaluation follows the [reproduction protocol](gallery-closeout-reproduction-2026-11-07.md) at scientific checkpoint `ee1c6adac64b773944ee3bbaea3aac07d60f2f13`. The review's saved target/optical figures retain the export generated **4 November 2026 at 03:44:59 UTC**, from public data release `data-82d701248232` and its earlier deployment source `cb2e1469ec222719c347b9b5f672765ede3ed7c7`. The [review manifest](../gallery/review/closeout_20261107/manifest.json) binds the base and source overlay separately. The [September synthesis](current-research-summary-2026-09-22.md) and its [numerical verification receipt](current-research-summary-checks.json) remain historical evidence; that receipt does not validate the November analyses.

## Question and present conclusion

**How reliably can an IMPROVE-trained FTIR EC calibration transfer to Addis, and under what conditions?**

The existing work supports a reproducible investigation of spectral preprocessing, calibration transfer, numerical convergence, Ethiopian seasonality and optical-response assumptions. It does **not** yet establish independent absolute EC accuracy for the Addis target population or select one universally validated target EC product.

The corrected source evaluation now quantifies candidate tradeoffs on the same physical filters. PTFE-anchored VIBES has the lowest observed source MAE and P95 absolute error, raw spectra the lowest RMSE, and spline the lowest MedAE. EPO moves a heavy, low-OC/EC subset's median FTIR/TOR ratio toward one while increasing overall error relative to its matched-resolution anchored control. These descriptive results do not establish statistical superiority or a universal target winner. Converged VIBES remains a numerical improvement with a separately defined downstream scope.

## Completed 7 November source evaluation

All six primary candidates use **18 PLS components, the same five site-grouped outer folds, and 12,386 common eligible physical filters across 157 IMPROVE sites**. Every candidate has 12,386 available predictions: **100% eligible coverage and zero prediction loss**. The full ledger retains 12,393 filters. Six finite TOR values at or above 150 µg/filter lie outside the declared source domain; the original all-candidate raw-spectrum QC rule excludes one additional filter from training and primary scoring. This QC rule was restored before examining the new scores. [Population and acceptance](../gallery/review/closeout_20261107/source_acceptance.json).

The table reports pooled out-of-fold errors against IMPROVE TOR EC. All error columns are **µg/filter**; FTIR/TOR is dimensionless. The heavy subset contains **163 filters**, with 15 ≤ TOR EC <150 µg/filter and TOR OC/EC <2.5. Values are rounded here; the saved tables and [source-score export](../gallery/review/closeout_20261107/continuum_baseline.json) retain their declared precision.

| Candidate | MedAE | MAE | RMSE | Mean signed bias | P95 absolute error | Heavy median FTIR/TOR |
|---|---:|---:|---:|---:|---:|---:|
| Raw spectra | 1.1426 | 1.7617 | 3.7456 | −0.0169 | 5.0782 | 0.7784 |
| Spline baseline | 1.0828 | 1.8153 | 4.4310 | −0.0531 | 5.2372 | 0.7393 |
| Production VIBES | 1.1301 | 1.8066 | 4.1923 | −0.0153 | 5.2646 | 0.7390 |
| Deflated VIBES | 1.3298 | 2.1074 | 4.1333 | −0.0116 | 6.2898 | 0.6308 |
| PTFE-anchored VIBES | 1.0905 | 1.6986 | 3.7613 | −0.0120 | 4.8428 | 0.7845 |
| Anchored + EPO | 1.2401 | 1.8410 | 4.0899 | −0.0296 | 4.9693 | 0.9008 |

These rows compare complete pipelines, including different spectral representations. Raw/VIBES candidates use 2,722 channels, spline 2,002 and EPO 1,361 after adjacent-channel averaging. A separate control holds that last resolution fixed and uses the same 12,386 common filters and folds:

| Matched 1,361-channel control | MedAE, µg/filter | RMSE, µg/filter | Heavy median FTIR/TOR |
|---|---:|---:|---:|
| Anchored PLS without EPO | 1.089408 | 3.761128 | 0.784547 |
| Anchored PLS with training-only EPO | 1.240100 | 4.089865 | 0.900773 |

The control supports a composition-dependent tradeoff: EPO's heavy-subset ratio is nearer one, while its overall MedAE, MAE, RMSE and P95 error increase. It does not establish EPO as the preferred general EC product. Fixed settings include prior exploratory choices, so this is retrospective source evaluation rather than independent model-selection confirmation. Differences from the old November scores cannot be attributed solely to leakage repair: the numerical runtime, regenerated continuum spectra and restored QC rule also have explicit versions. [Results, control and limits](gallery-closeout-implementation-2026-11-07.md#numerical-results-and-acceptance).

### What numerical acceptance established

The two continuum variants were collected for **32,286/32,286 fits with zero failed fits**, covering 12,393 IMPROVE, 233 Addis and 3,517 network spectra per variant. This run uses the original CPU `vibes_baseline_matrix` adapter and recorded PB/tau/maxiter/retry procedure. Its completed diagnostics are not a Newton convergence certificate. The [reproduction record](gallery-closeout-reproduction-2026-11-07.md#numerical-procedure) preserves the numerical environment and execution history.

All five EPO held-out prediction blocks were reconstructed from their saved projection and PLS parameters with **exact equality and maximum absolute difference 0**. Training/test sites and filter IDs do not overlap. Only fold 0 was refitted for the integrity test: changing its **2,481 held-out TOR labels**, including values crossing the domain boundary and missing values, and changing held-out nuisance chemistry left the projection, signatures, centering, coefficients and predictions **exactly unchanged**. Spectra, splits, QC and training inputs stayed fixed; the artificial evaluation scores changed. Those artificial scores are not scientific performance estimates. Baseline OOF identities, folds, QC and score inputs were checked, but baseline fold coefficients were not saved for independent reconstruction. [Actual acceptance receipt](../gallery/review/closeout_20261107/source_acceptance.json).

A separate CSV/NumPy arithmetic review, importing no project scoring or fitting helper, agreed with the primary and control tables, population, folds and hashes; its maximum saved-table difference was approximately `2.22e-16`. It checks the same saved predictions and references and supplies no new validation observations. The [implementation receipt](gallery-closeout-implementation-2026-11-07.md#numerical-results-and-acceptance) records that review alongside the scientific acceptance.

## Saved November evidence and the decision each supports

The numerical findings below are **reported saved results with their original populations and versions**. The source-only refresh does not replace historical target predictions, optical-response models, seasonal summaries or synthetic controls with the new all-source prediction archives.

| Question | Saved evidence | Interpretation and limits |
|---|---|---|
| Does broad-signal recovery follow from successful optimization? | The documented 6 November continuum rerun has 89/90 successful fits. The failed unspiked blank leaves eight independent parent blanks for VIBES recovery and nine for spline. At 0.05 AU, reported VIBES recovery is approximately 98.27% for four bands, 18.80% for the broad hump and 0.138% for the power-law addition. | Successful fitting does not establish preservation of these broad synthetic shapes. The additions are repeated tests on parent blanks; paired method differences need common available parents. This does not identify the real continuum's chemical origin. [Run and limits](vibes-gallery-continuum-check.md). |
| Does anchoring improve transfer diagnostics? | The saved production/anchored Addis aethalometer intercepts are approximately −0.955 and +0.098 µg/m³. The three unflagged provisional Adama pairs give median FTIR/TOR of 0.96 and 1.31. | A promising tradeoff, not a demonstrated absolute accuracy improvement or equivalence. Adama is a different site, lot and sampling arrangement with provisional mapping. These saved target comparisons are separate from the new held-out source table above. [Comparison implementation](../research/ftir_hips_chem/workflows/run_hips_response_20261103.py). |
| Can the old EPO score select a method? | Historical EPO projection jointly regressed source spectra on EC, OC, other PM, Fe and K before site splitting. Its saved 1.24 µg/filter MedAE and heavy-subset ratio used a label-dependent transformation informed by test rows. | The completed training-only fixed-k evaluation above supersedes that primary source score. Numerical proximity does not validate the old pipeline or isolate the size of leakage bias. Historical EPO sweeps/pass-rule simulations are outside the new receipt and remain withheld; target optical comparisons remain exploratory. [Corrected workflow](../research/ftir_hips_chem/workflows/run_intercept_fixes_20261103.py). |
| What does Newton change? | The adoption study reports 4,452/4,452 network analyses fitted and certified, and rebuilt source test RMSE 3.8501 → 3.8499 µg/filter. Functional-group, lot, blank-bank and continuum checks are documented. | Supports numerical control and reproducibility, not improved absolute EC accuracy. Season-composition and fingerprint-grouping impact checks remain necessary before numerical v2 is promoted into those conclusion-bearing analyses. Preserve both runs meanwhile. [Adoption study](vibes-newton-adoption-tests-2026-11-03.md). |
| Does the rank-70 background absorb aerosol? | Every network scan solved to convergence (4,452 of 4,452). The 108 sample scans with a noise level as low as a field blank's all have raw spectra without aerosol bands; 10 are flat scans. An injected band up to 150 cm⁻¹ wide is recovered in full on loaded filters at ranks 20, 40 and 70; a 1,500 cm⁻¹ wide feature keeps 0 to 14 % at every rank. | The background removes breadth, not narrow bands, and lowering the rank does not bring the continuum back. Gaussian bands at two positions only; ranks were truncated, not refitted. [Rank check](vibes-rank-and-wasm-2026-11-10.md). |
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

Source scores use µg/filter, while target optical comparisons use µg/m³. The fixed-k primary source population retains the recorded raw-spectrum QC flag and finite TOR below 150 µg/filter, an outcome-restricted domain. Report high-loading exclusions/stress cases and candidate availability alongside the common-filter scores. Repeated scans and synthetic injections do not increase the number of independent physical filters or parent blanks. Historical squared correlation and predictive R² also remain different statistics.

The October 28 organic-signal figure's y value is **FTIR residual from seasonal HIPS fit (µg/m³)**: FTIR EC minus its seasonal Deming fitted line. It is neither a simple FTIR−HIPS difference nor an independently measured EC error. The gallery now retains that definition with explanations off and in exports.

## Completed figure audit and release acceptance

The completed [figure-population audit](gallery-closeout-figure-populations-2026-11-07.md) reconciles six distinct views. Its source panel attaches the verified scores above; its other panels preserve the saved November export.

| Review view | Audited population or summary |
|---|---|
| Coverage | 233 counted physical filters; 172 have aethalometer values. |
| Source-reference performance | 12,386 common eligible IMPROVE filters / 157 sites. |
| Baseline behavior | 102 displayed bin medians across preprocessing panels; repeated views of the same filters. |
| Seasonal contrasts | 170 historical complete cases per outcome, under `belg_feb`, with an additional inferred-comparator eligibility requirement. |
| Optical response | 232 historical response-eligible Addis filters; 171 aethalometer matches. |
| Candidate target comparison | The same 172 physical aethalometer pairs in the saved production/anchored panels. |

The 170 seasonal cases, 172 target pairs and 232 response cases are not interchangeable. The matched seasonal study's 232 direct-HIPS pairs and its 188-filter sensitivity subset retain their own definitions. The saved ambient-parent synthetic display also has its own population: 16 parents, three backgrounds and 192 fitting jobs, with parent IDs/fit diagnostics absent from that export. It is separate from the later independent-blank control described above. Historical flag and inclusion differences remain documented in the ledger rather than silently harmonized.

The endpoint is met as a **versioned existing-data research review release**. The real-data rerun, source acceptance, independent arithmetic review, source export, figure audit, corrected-overlay browser journeys, phone/export review and final source/base/overlay receipt are complete. [Run 42407030717](https://github.com/ahzs645/aethmodular/actions/runs/42407030717) passed 114 Python, 372 source Node, 91 public-data Node and three browser tests, plus type/build checks, on the exact review source. All seven exported pages were visually inspected. The full legacy research gate remains unavailable because its original input tables and daily-instrument pickles have not been restored; the selected public scope does not certify that full profile. [Release identities and limits](gallery-closeout-implementation-2026-11-07.md#corrected-overlay-release-verification).

The review sequence covers physical-filter coverage, source-reference performance, baseline behavior, Bega/Belg/Kiremt, optical-response sensitivity and candidate tradeoffs. A shared section refers to the active exports; it is not itself a frozen data archive. Preserve the separately hashed historical base, corrected source overlay, acceptance/figure evidence and final export receipt together. The [reproduction protocol](gallery-closeout-reproduction-2026-11-07.md) records the restored inputs and exact source/target version boundary.

PR #19 remains a separate Research-only timeline review. Its catalog records source history and saved evidence, not scientific validation. The closeout fix replaces the indexer's checkout-based `missingData` flags with checks of declared primary exports on the active review host. Exact-head CI at `316a53ec423c7344253aef409027b0f607fff23f` passed 20 focused tests, four browser journeys, the research build and public exclusion ([run 42396251333](https://github.com/ahzs645/aethmodular/actions/runs/42396251333)). A loaded export does not establish scientific reproduction; public builds intentionally omit this area.

New Ethiopian sampling/TOR, full PMF scientific qualification, further GPU/WebGPU work and exhaustive historical notebook migration are follow-on work. None is required to state the supported existing-data findings and the unresolved absolute Addis level. The proposed future TOR rule still requires agreement on eligibility, uncertainty, acceptance limits and target coverage.

## Claims not carried forward

Do not claim that a near-zero aethalometer intercept proves correct EC, that the fitted HIPS bend certifies an instrument operating range, that a seasonal residual identifies its chemical cause, or that three provisional Adama pairs select a universally accurate method. Do not transfer the September reproduction receipt to a different November experiment. Historical mechanistic corrections and unavailable independent evidence are preserved in the [September synthesis](current-research-summary-2026-09-22.md) and [open items](open-items.md).
