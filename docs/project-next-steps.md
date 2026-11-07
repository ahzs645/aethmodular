# Project next steps: existing-data closeout

Updated **7 November 2026 after the completed real-data source evaluation**. Numerical acceptance, independent arithmetic review and the figure-population audit are complete. Final browser/publication acceptance of the actual corrected overlay and the release/UI receipt remain pending. This plan supersedes the [September plan](project-next-steps-2026-09-22.md). The [current synthesis](current-research-summary.md) gives the candidate results and scientific limits; the [implementation receipt](gallery-closeout-implementation-2026-11-07.md) records each verification scope.

## Endpoint

Conclude the existing-data investigation with one recoverable source/data release and a short figure sequence. Absolute Addis EC accuracy may remain unresolved. A method does not have to win for this stage to finish.

## Completed work to reuse

- Matched calibration/seasonal study and sensitivities (PR #14).
- Baseline lab and seasonal spectral ML (PR #15).
- Initial GPU investigation and later Newton/JAX/WebGPU research (PR #16 and subsequent dated studies).
- Notebook diagnostics and narrow/broad continuum controls (PR #17).
- Existing graph focus, custom sections, shared links and figure/section export.
- Recovery and hash verification of the decision bundle and original production background; collection of both regenerated continuum variants, **32,286/32,286 fits with zero failed fits**, using the original adapter.
- The six-candidate source evaluation and separate matched-resolution EPO control: fixed **18 PLS components**, the same **five site folds**, **12,386 common eligible physical filters / 157 sites**, and **100% eligible prediction coverage** for every candidate.
- Actual artifact acceptance: exact reconstruction of all five EPO held-out blocks; one fold-0 refit after changing **2,481 held-out TOR labels and nuisance chemistry**, with exact zero model/prediction differences and unchanged training inputs, spectra, splits and QC.
- Independent CSV/NumPy score arithmetic and the final figure-population audit. The arithmetic check uses the same saved predictions and references; the figure audit reconstructs saved populations and is not a browser test.

These are inputs to the synthesis, not tasks to restart. Preserve their distinct populations and versions. The [real-data acceptance receipt](../gallery/review/closeout_20261107/source_acceptance.json) and [numerical results](gallery-closeout-implementation-2026-11-07.md#numerical-results-and-acceptance) record the new work separately from historical studies and source/synthetic tests.

## Closeout status and remaining acceptance

| Work | Current status | Remaining acceptance |
|---|---|---|
| Current synthesis | Updated with the new source results and saved target findings. | Keep the source/target version boundary and historical receipt scopes visible. |
| EPO and source metrics | **Complete on real data.** Fold-specific training-only fitting, model reconstruction and the held-out-label integrity test passed. | Preserve accepted arrays, ledgers, code/input hashes and distinct metric definitions. No new fitting is required for release acceptance. |
| Comparison population | **Complete.** Identical frozen folds, zero train/test site or filter overlap, one contribution per physical filter and zero eligible prediction loss. | Retain all 12,393 saved rows and the declared TOR/QC exclusions alongside the 12,386-filter score. |
| Six-figure review | **Numerical/population audit complete; actual corrected-overlay UI acceptance pending.** | Review the corrected source panel, retained target dates, notes-off evidence, phone layout, fresh-context shared link and completed export. The old-snapshot EPO guard must still work. |
| PR #19 | Active-host availability repair and its bounded research-browser review are verified at the recorded checkpoint. | Review/merge remains separate from this public snapshot; retain Research visibility and distinguish availability from scientific validation. |
| Release preservation | Inputs, completed continuum run and new scientific outputs have recorded identities. | Bind the exact tested source, pinned historical base, corrected overlay, acceptance/figure evidence and final PDF/section export. Record selected checks and unavailable scopes explicitly. |
| Closeout declaration | **Pending the final release/UI receipt.** | State which existing-data scope is complete and keep new sampling, PMF and further numerical work separate. |

The current implementation adds a **Build six-figure review section** button at Experimental → Continuum. It uses existing section editing, Copy link, PDF and Deck. A section is a selection of live exports; the release must preserve the data separately.

## Completed scientific evaluation and reproduction

The October decision bundle and original production background are restored and hash-verified. The old November corrected-array bundle was not located; both variants have now been regenerated and collected under the [versioned reproduction protocol](gallery-closeout-reproduction-2026-11-07.md). This uses the original PB/tau/maxiter/retry adapter, not Newton certification or numerical-v2 promotion. New baseline/EPO archives carry ordered physical-filter IDs, sites, fold assignments, the recorded raw-spectrum QC mask and metadata hashes. Equal array lengths alone do not prove alignment.

The primary domain applies the existing all-candidate raw-QC exclusion before fitting and scoring: 12,393 saved filters, 12,387 finite TOR values below 150 µg/filter, and 12,386 eligible after the one recorded QC exclusion. All six candidates predict every eligible filter. Preserve the full row ledger, the six complete-pipeline comparison and the separate matched-resolution control. Reproduction commands require new output destinations; do not overwrite accepted or historical predictions. [Population and verification](../gallery/review/closeout_20261107/source_acceptance.json).

The [current source table](current-research-summary.md#completed-7-november-source-evaluation) supports a tradeoff: anchored VIBES has the lowest observed MAE/P95, raw spectra the lowest RMSE, and spline the lowest MedAE. In the matched 1,361-channel control, anchored PLS without EPO has MedAE **1.089408**, RMSE **3.761128** µg/filter and heavy median FTIR/TOR **0.784547**; EPO gives **1.240100**, **4.089865** and **0.900773**. No significance claim or universally validated target model follows. Differences from older scores cannot be assigned solely to leakage repair because QC, regenerated spectra and numerical runtime also changed. Prior exploratory settings do not become independent model-selection confirmation when evaluated on this scoreboard.

The prepared source-only review refresh attaches the verified fixed-k scores to the frozen display export without recreating unavailable historical target tables. Its [manifest](../gallery/review/closeout_20261107/manifest.json) hashes the overlay and base separately. Historical target comparisons retain their values and **4 November generation date**. The new all-source target predictions remain separate; the corrected fixed-k receipt does not certify old EPO sweep/pass-rule derivatives.

The [figure audit](gallery-closeout-figure-populations-2026-11-07.md) preserves **233** counted Addis filters, **172** target/aethalometer pairs, **170** historical seasonal complete cases and **232** historical response cases. These cohorts have different eligibility rules; the matched seasonal study, its 188-filter sensitivity subset and the independent-blank/ambient-parent synthetic studies also retain their own definitions. Reopening and exporting those figures on the actual corrected overlay is still part of final UI acceptance.

## Verification profiles

- `uv run aeth gallery check --profile source`: source/synthetic behavior and compiled visibility, without claiming scientific input availability.
- `uv run aeth gallery check --profile public`: source checks plus the public snapshot's declared integration checks.
- `uv run aeth gallery check --profile research`: the full set, including private/legacy research evidence. This is the default for `check`.

Unexpected failures are fatal. Missing declared inputs are unavailable and block their selected profile. Preserve the generated receipt, data hash, commands and tested source revision. A deployment success alone is not the whole acceptance record.

The **full legacy research profile remains unavailable** because required original tables and daily-instrument pickles are missing. Downstream display exports are not substituted for those inputs to make the checks pass. The earlier exact-checkpoint public/browser receipt covered the historical base before the corrected source overlay existed; retain that narrower scope until the new acceptance run supplies the final receipt. See the [release status](gallery-closeout-implementation-2026-11-07.md#corrected-overlay-release-verification) and [legacy prerequisites](gallery-closeout-software-batch-2026-11-07.md#remaining-research-gate-prerequisites).

## Follow-on work outside this closeout

Independent Ethiopian thermal EC and sampling confirmation; agreed future TOR acceptance criteria; full PMF qualification (PR #18); numerical-v2 season-composition/fingerprint promotion checks if those analyses adopt v2; further acceleration; historical notebook coverage without a specific conclusion to support.

Do not send the existing data request or contact collaborators without explicit authorization. Keep the proposed TOR rule labelled proposed; no new sampling requirement is introduced as a condition for finishing the existing-data account.
