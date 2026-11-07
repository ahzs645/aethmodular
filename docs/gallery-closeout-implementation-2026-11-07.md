# Gallery closeout research receipt — 7 November 2026

## Current status

The existing-data research closeout is complete as a **verified review release**. The recovered original decision bundle and production background produced **32,286/32,286 completed continuum fits with zero recorded fit failures**. All six source candidates have predictions for the same **12,386 eligible physical filters across 157 sites**. Saved-model acceptance, the real-data held-out-label check, independent score arithmetic, the figure-population audit, exact-source public CI and all seven exported PDF pages passed their stated checks. Absolute Addis EC accuracy remains unresolved; this completion does not certify the unavailable full legacy research profile or publish a deployment.

The [first software-batch receipt](gallery-closeout-software-batch-2026-11-07.md) is preserved separately. Its missing-spectrum blocker, earlier test counts and earlier PDF describe that batch; they are not verification of this numerical rerun.

Source changes are in [aethmodular PR #20](https://github.com/ahzs645/aethmodular/pull/20). The companion strict publication workflow is in [aethmodular-gallery PR #1](https://github.com/ahzs645/aethmodular-gallery/pull/1). These review branches have not been merged or deployed by this work.

The [preservation manifest](gallery-closeout-preservation-2026-11-07.json) records the five review payloads, the original gallery base needed by review CI, ordered transport parts, and the separately preserved completion-documentation CI receipt. Its preservation status tracks Google Drive upload verification separately from scientific acceptance. Provider file identities and share links remain in the private archive index. The scientific implementation, accepted PDF source and completion-documentation head retain their distinct identities; archival work does not change the published gallery.

## Frozen inputs and procedure

| Item | Verified identity and scope |
|---|---|
| Scientific implementation checkpoint | `ee1c6adac64b773944ee3bbaea3aac07d60f2f13`; subsequent UI/data/document changes require their own verification identity. |
| Original decision bundle | 18 files plus the original manifest; 12,393 IMPROVE, 233 Addis and 3,517 network rows. All declared hashes match. Manifest SHA-256 `121557f8bf470816e02ddeafed8b7acf91b1c2cd50a59eab08088b7af169db48`. |
| Original production background | Rank 70, 87 blank IDs, 2,722 channels; SHA-256 `8b717b6b988465083e0e102f887da4f61c63aed86cce43be640e52dece2165e3`. |
| Prepared continuum bundle | 16,143 spectra × two variants, or 32,286 fitting jobs; content hash `3583bdb74db1ac2d111d9cdfb6365c1e2a692c545fb8459e507fe9af4633362d`. |
| Frozen runner | SHA-256 `cb7e689475b31d567558782ad046e7cc1e18b1d117baa6f9277ad584fb1f7ec0`. Original pinball-loss adapter, tau 0.1, initial maximum 10,000 iterations and existing retry policy. No Newton substitution. |
| Public base archive | `data-82d701248232`; SHA-256 `82d701248232b1975ea2e452bc1d682877e811103a2aa1040aff9d4adc989051`. |
| Frozen historical display | Generated 4 November; continuum export SHA-256 `90b8e16bfcb002db7e6bc7553d190c94f462753182fcb80d5bacc1c65a2111c9`. Its paired filter map has SHA-256 `a043679b673e82cdc6b539c6aeab38f97855f142ca7f1901a61bc66ff416089d`. |
| Preserved input archive | 22 scientific inputs, including both frozen display inputs; SHA-256 `d8a9fac2455aecd4d0b7b2b96686c62c4043a91370ba8eaaeb0f109eb97a56a0`, 443,084,226 bytes. |
| Completed continuum archive | 310 regular members: immutable bundle, 270 checkpoints, six corrected arrays, diagnostics and receipts; SHA-256 `cd44e62160e3717afcfc13f793eea294852e139983c25139ed31cf32ac4d4853`, 980,487,622 bytes. |
| Completed collection receipt | 32,286 expected/collected fits and zero failures; SHA-256 `5a7f59a79a9fa8e3b028ca0d08a73a99ea057b4927db65f0c30a7153a7b6be72`. This is original-adapter fit-status evidence, not Newton convergence certification. |
| Corrected source-score receipt | SHA-256 `ff10bd831f6c11d824b5f244b9696103910c7709c276a06766ac39a265b08c79`. Full-precision CSVs and physical-filter ledgers accompany the research package. |
| Scientific acceptance | [Compact receipt](../gallery/review/closeout_20261107/source_acceptance.json), SHA-256 `2b3836fda7783815fccb41462b7256651558a5bac7c3598b94f6e0ce07273ce6`. |
| Corrected display overlay | [Review manifest](../gallery/review/closeout_20261107/manifest.json); overlay SHA-256 `f38baa0c6756fadf92583ef3b23c018b4c6f1773efd4bf52b7ce6f56db1649a2`. The base archive retains its original identity. |
| Complete gallery data candidate | 5,137 files, 167,794,774 bytes; SHA-256 `accc09aa8eea12d32c9c58c2daa54f3d01658edd54dd12c82a67fc363db2f49c`. Only `experimental/continuum_baseline.json` differs from the frozen base; all 5,136 other member bytes match. Suggested tag `data-accc09aa8eea` is not published. |

The [reproduction protocol](gallery-closeout-reproduction-2026-11-07.md) contains exact commands, numerical settings, package versions and restoration boundaries. Completed scientific destinations are never overwritten. Collection requires every expected job exactly once, preserves failures explicitly, and publishes its receipt last.

The first execution attempt ended when a worker was killed (`SIGKILL`) after 22,800 jobs were checkpointed. Those completed jobs reported no failed fits. Execution resumed with four workers, preserving all scientific identities and reusing only verified checkpoints. The interruption is recorded separately from fit failures; no incomplete batch is accepted as a result.

## Evaluation fixed before reading the new scores

The six primary pipelines are raw spectra, spline, production VIBES, deflated VIBES, PTFE-anchored VIBES, and anchored EPO. They use 18 PLS components and the same five site-grouped outer folds, defined on all 12,393 physical source filters before label/QC eligibility. The full source population spans 157 sites.

The original decision workbook's all-candidate raw-spectrum QC rule is restored. There are 12,387 finite TOR values below 150 µg/filter before QC, and **12,386 eligible filters** after the one recorded exclusion. The mask hash is `d697b108246566491a2c5222abc0e656e4b42c136338d02ccffbf7e8bc7c8389`. All rows remain in predictions and ledgers. Complete chemistry is required for projection fitting, not for prediction after the projection is frozen.

Each EPO fold estimates chemical signatures, nuisance projection and PLS parameters from its training rows only. It saves actual projection/PLS arrays, training and projection-training IDs, test IDs, membership, ranks and hashes. A separate all-source model predicts Addis and the network. It is not used for source out-of-fold predictions.

Raw and VIBES variants use 2,722 channels; the saved spline representation uses 2,002. EPO averages adjacent full-grid channels to 1,361. A separate `anchorhp_bin2_base` versus `epo_bin2` control holds this resolution fixed. It does not replace the full-grid anchored row in the primary comparison.

The scorer reports MedAE, true MAE, RMSE, mean/median signed bias, P90/P95 absolute error, composition subsets and eligible/available/common coverage. The primary comparison uses the same finite-prediction intersection; missing predictions stay visible. Heavy low-OC/EC means 15 ≤ TOR <150 µg/filter and TOR OC/EC <2.5. The older sweep uses a different heavy definition and retains its historical status.

This is retrospective evaluation under settings informed by prior exploration. Differences from the historical table cannot all be attributed to leakage repair: QC, numerical runtime and regenerated spectra also have explicit versions. Source-domain performance does not establish absolute target EC accuracy.

## Numerical results and acceptance

The scientific pipeline completed on 7 November at 19:22:17 UTC. Collection, five primary baseline fits, the binned anchored control/EPO fits, scoring, scientific acceptance, display export and the figure audit all exited successfully. Every primary candidate and both control rows have **100% prediction coverage on all 12,386 eligible filters**. The common population includes 163 heavy low-OC/EC filters and 2,065 filters in the separately declared light subset. Six finite references at or above 150 µg/filter and the one recorded raw-QC exclusion remain in the full ledgers, outside the primary score.

All error quantities below are **µg/filter**; FTIR/TOR is dimensionless. Values are rounded only for this table.

| Complete candidate pipeline | MedAE | MAE | RMSE | Mean signed bias | P95 absolute error | Heavy FTIR/TOR |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 1.1426 | 1.7617 | 3.7456 | −0.0169 | 5.0782 | 0.7784 |
| Spline | 1.0828 | 1.8153 | 4.4310 | −0.0531 | 5.2372 | 0.7393 |
| Production VIBES | 1.1301 | 1.8066 | 4.1923 | −0.0153 | 5.2646 | 0.7390 |
| Deflated VIBES | 1.3298 | 2.1074 | 4.1333 | −0.0116 | 6.2898 | 0.6308 |
| PTFE-anchored VIBES | 1.0905 | 1.6986 | 3.7613 | −0.0120 | 4.8428 | 0.7845 |
| Anchored + fold-local EPO | 1.2401 | 1.8410 | 4.0899 | −0.0296 | 4.9693 | 0.9008 |

Anchored VIBES has the lowest MAE and P95 absolute error among these six pipelines. Raw spectra have the lowest RMSE; spline has the lowest median error but the highest RMSE. These are descriptive rankings under the fixed retrospective evaluation, not significance tests or a universally validated target-method selection. Small pooled signed biases coexist with substantial underprediction in the heavy low-OC/EC subset.

The separate **1,361-channel matched-resolution control** isolates the EPO comparison from adjacent-channel averaging:

| Matched-resolution pipeline | MedAE | MAE | RMSE | Mean signed bias | P95 absolute error | Heavy FTIR/TOR |
|---|---:|---:|---:|---:|---:|---:|
| Anchored, pair-averaged, without EPO | 1.0894 | 1.7000 | 3.7611 | −0.0125 | 4.8497 | 0.7845 |
| Same resolution, with fold-local EPO | 1.2401 | 1.8410 | 4.0899 | −0.0296 | 4.9693 | 0.9008 |

Both control rows use the same 12,386 filters, folds and 18-component setting as the primary comparison. EPO moves the heavy-subset ratio closer to one while increasing the overall MedAE, MAE, RMSE and P95 error. The corrected result supports a composition-dependent tradeoff; it does not establish EPO as the preferred general EC product.

Acceptance reconstructed all five EPO held-out blocks from saved models and the separate EPO/baseline all-source target predictions under predeclared `atol=rtol=1e-10`. It refitted only EPO fold 0 after replacing the **2,481 held-out TOR labels**, including values crossing 150 and missing values, and changing held-out nuisance chemistry. Training records, spectra, QC and splits remained fixed. The projection basis, signatures, centering, PLS coefficients and all 18-component held-out predictions were **exactly unchanged, with maximum absolute difference 0**. The artificial evaluation scores changed as required; they are not scientific performance estimates. Baseline OOF identities, folds, QC and score inputs were checked, but baseline fold coefficients were not saved for independent reconstruction.

A separate CSV/NumPy arithmetic review imported no project scoring or fitting helper. It found no discrepancies in the six reported primary metrics, matched-resolution control, QC, common population, site folds or input hashes; the maximum difference from saved table values was `2.220446049250313e-16`. This checks arithmetic on the same saved predictions and references, not new validation data. The source score receipt and complete ledgers retain the full precision.

## Figure populations and interpretation

The [reviewed figure ledger](gallery-closeout-figure-populations-2026-11-07.md) and its executable audit distinguish the following populations:

| Figure | Population and reference |
|---|---|
| Coverage | 233 unique Addis physical filters; Bega/Belg/Kiremt 81/81/71. Repeated scans do not increase this count. |
| Source performance | 12,386 common eligible physical filters/157 sites, fixed k=18, five held-out site folds, 100% candidate coverage; IMPROVE TOR EC in µg/filter. |
| Baseline behavior | 102 displayed aggregate bins; per preprocessing, 231/233 Addis bin members appear. The saved 2–3 November injection study uses 16 ambient parent filters, distinct from the later blank experiment. Its export lacks parent IDs and fit-failure counts. |
| Seasonal contrast | 170 historical complete cases, including an inferred-comparator eligibility requirement. Calendar `belg_feb`: Bega Oct–Jan, Belg Feb–May, Kiremt Jun–Sep. Approximate 90% intervals are correctly transformed from the saved rounded log-scale summaries. |
| Optical response | Selected overlapping curve populations; main parametric fit 11,311 filters/152 sites. The fitted bend uses a geometric-mean optical-depth coordinate. It is not a certified raw-HIPS reliability boundary. Historical Addis response population: 232. |
| Target comparison | 172 identical aethalometer pairs for the saved production/anchored panels. MA350 IR BCc is an optical comparator, not independently measured Addis TOR. |

The source-only refresh preserves historical target values, populations and generation date. New all-source target predictions are archived separately. The corrected fixed-k source receipt does not certify the old EPO sweep or pass-rule simulation; those source quantities remain withheld.

The current exclusion is ETAD-0248, not ETAD-0241. Historical 170/232-case summaries retain the qualification that their eligibility used an older network exclusion state. Their count is not silently imposed on the current 172-pair graph. The October 28 organic-signal axis now names its actual residual: FTIR EC minus the seasonal Deming fitted line.

The six-figure section uses the existing section/share/PDF tools. A shared link loads active exports; the archive and receipt preserve the research version separately.

The completed executable figure ledger has SHA-256 `d0efa3623570c07892d5a8410068aaa36d5a2ae44219f887442d9b9bfe7e39bd`. Attachment checks matched 150 displayed score fields to the saved CSV at the export's rounding precision and confirmed all 35 unchanged historical top-level sections. This source-only export does not refit those historical figures.

## Corrected-overlay release verification

[Run 42407030717](https://github.com/ahzs645/aethmodular/actions/runs/42407030717) passed on exact source **`2341cbbbab41b9dfde143d36aac5ea84585aea31`**, using the original base archive plus the separately verified corrected source overlay. The downloaded final receipt records `verified_base_and_overlay`, `fold_local_source_verified` and `public_review_status: success`. Its source/base/overlay/acceptance hashes match the exact checkout.

| Check | Completed result |
|---|---|
| Python scientific/gate regressions | 114 passed; zero failures, errors or skips. |
| Strict Node source suite | 372 passed across 69 test files; zero failures, skips, TODOs or cancellations. |
| Public-data reconciliation | 91 passed across 19 test files and 283 declared inputs; zero failures, skips, TODOs or cancellations. Neither selected Node profile reported unavailable files or changed inputs. |
| TypeScript and production build | Passed. |
| Browser acceptance | Three passed; zero unexpected, skipped or flaky results, with no retries. Actual corrected source metrics/counts, the legacy-EPO guard, preserved target date, phone readability, fresh-context copied link, saved-section reload and completed PDF download. |
| Visual review | All seven PDF pages inspected at 110 dpi; titles, axes, legends, counts and evidence text are readable with no observed clipping. Phone plots fit the viewport, and the Evidence column is reachable by normal vertical and horizontal scrolling. |

The final public-review artifact is **617,156 bytes**, SHA-256 `025e05a686006c65c762b8a751b5723e23c685d0fa3cb8a932f01b3abefb5c5a`; the scientific-regression artifact is **3,308 bytes**, SHA-256 `ad58bbe40692b9e44d9969dc07e7b918453e1a788ebd55accfd8ebe60825d0df`. Both were checked against the GitHub API digests and safely extracted. The original seven-page CI PDF is **166,336 bytes**, SHA-256 `9c6cd156e6b45fc18ee243b01df1174defd077b7c56a5963083838edf7e9726c`; that exact export is included in the research-results ZIP. The compact final visual/CI receipt has SHA-256 `b228b3911acf990776eaa328cc5010919ac166cfaae43b3d41b3b20b0f931f66`.

The preceding actual-overlay run at `22373ef5` correctly failed one phone assertion: three stacked metric panels placed the table below the initial viewport, but the test scrolled only horizontally before requiring its Evidence header on screen. The correction adds a vertical scroll to the table header and retains the viewport assertion. No scientific, application UI or data file changed. The failed receipt is preserved; it is not counted as a passing release run.

This document records the artifact source above. Later receipt-only documentation updates do not relabel that PDF or imply a different numerical run. The earlier implementation-checkpoint evidence below retains its narrower base-only scope.

The compact research release, `Aethmodular_Closeout_Research_Release_20261107.zip`, is **22,580,271 bytes**, SHA-256 `df6fcd117041ce5163ed75996c4db881ce9031662656972bbdb4ca944dc651a7`. All **42 members** were read back and verified. It contains new fitted models/predictions, full-precision metrics and filter ledgers, acceptance and figure-audit outputs, independent arithmetic review, execution history, structured raw CI reports, the original PDF, and restoration/member manifests. Companion inputs, continuum parts and the complete gallery-data candidate have their separate recorded identities. Repository-backed code and documentation stay at the source revision; browser screenshots are not release artifacts.

## Verification recorded for the implementation checkpoint

[Run 42400410346](https://github.com/ahzs645/aethmodular/actions/runs/42400410346) passed at exact source `ee1c6adac64b773944ee3bbaea3aac07d60f2f13` against the pinned historical public archive, **before the corrected source overlay was available**.

| Check | Result |
|---|---|
| Python scientific/gate regressions | 114 passed; zero failures, errors or skips. These use source/synthetic fixtures. |
| Strict Node source suite | 372 passed; zero failures, skips or TODO assertions. |
| Public-data reconciliation | 91 passed; zero failures, skips or TODO assertions. |
| TypeScript and public production build | Passed. |
| Browser acceptance | Three tests passed; zero skipped, unexpected or flaky results. Phone evidence/plot/table readability, preserved target date, fresh-context shared section, saved-section reload and completed seven-page PDF download. |
| Visual inspection | All seven PDF pages and the active phone view inspected; no observed clipping or unreadable headers. |

The verified public-review artifact is 518,496 bytes, SHA-256 `d42e288a77ab97fa7ef33a33e3aeb435ba3960de06d766003486573d2cb34f31`. Its seven-page PDF is 164,264 bytes, SHA-256 `875a00eaba3ce8510e920e0dc8e831645844b14546a2ec571f76db541b0c8732`. This checkpoint PDF correctly withholds historical EPO scores. It is not the final corrected-source figure package.

Local browser execution remains unavailable because Chromium cannot create its required socket. The hosted results supply the completed browser/build evidence; no local success is inferred. The optional broad Ruff invocation reported existing style issues in the frozen continuum runner. No global Ruff pass is asserted for this continuation.

## Separate timeline review

PR #19's active-host availability repair is verified at `316a53ec423c7344253aef409027b0f607fff23f`: 20 focused Node tests, the research TypeScript/Vite build, four browser journeys and canonical public exclusion passed in [run 42396251333](https://github.com/ahzs645/aethmodular/actions/runs/42396251333). Loaded exports, saved execution evidence and scientific validation remain distinct. Research visibility is preserved, and the PR remains separate from the public closeout snapshot.

## Limits and completion boundary

The strict source/public/research profiles no longer tolerate arbitrary Node failures. Missing declared inputs block their selected profile. The full legacy research gate remains unavailable: three Node test files need seven original paths/file groups, and the full Python research profile needs five daily-instrument pickles. The [historical receipt](gallery-closeout-software-batch-2026-11-07.md#remaining-research-gate-prerequisites) preserves the exact input list.

A bounded recovery search did not restore those inputs. Complete inspection of the 90,894,200-byte October 19 GPU bundle verified its internal hashes and found no matching original gate files. Membership of the 289,297,763-byte resume ZIP remains unverified because it exceeds the connector's download limit. Downstream display exports and differing ChemSpec versions are not substituted for original inputs to make the tests pass.

The existing-data endpoint is met for the declared research-review scope: new numerical artifacts and acceptance receipt, reconciled figure populations, an exact tested base/overlay/source combination, a completed figure package and preserved restoration materials. Public CI verifies saved accepted inputs; it does not itself rerun the private scientific fits. The completed continuum archive is supplied in three byte-verified parts because its single-file save failed; ordered concatenation restores the exact archive identity above. Absolute Addis EC accuracy remains unresolved.

New Ethiopian TOR/sampling, agreement on the proposed future TOR rule, full PMF qualification, numerical-v2 season-composition/fingerprint promotion checks, further GPU optimization and exhaustive historical notebook migration remain follow-on work.
