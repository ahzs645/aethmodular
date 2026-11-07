# Gallery closeout implementation receipt — 7 November 2026

## Continuation after the first software batch

The original October decision bundle and production background have now been recovered and hash-verified. A new versioned continuum regeneration and source-evaluation run are described in the [reproduction protocol](gallery-closeout-reproduction-2026-11-07.md). The input-blocker section below records the earlier batch's state; it is not the current recovery instruction. The existing all-candidate raw-QC rule is restored before fitting, and a matched-resolution control separates EPO from channel averaging.

PR #19's availability fix is also verified at `316a53ec423c7344253aef409027b0f607fff23f`: 20 focused Node tests, the research TypeScript/Vite build, four browser journeys and canonical public exclusion passed in [run 42396251333](https://github.com/ahzs645/aethmodular/actions/runs/42396251333). Availability now comes from the active host; it remains distinct from scientific validation.

The original software-batch receipt follows. Later source results and figure corrections must retain their own run and verification identities; the earlier PDF and CI result below are not relabelled as evidence for new calculations.

## Status and scope

This is a **review candidate for the software and interpretation batch**. It is not a completed scientific release. Public software checks, the production build and browser acceptance passed in CI for the exact review revision recorded below. The corrected real-data EPO comparison and full research reconciliation remain unresolved. No method is selected by this implementation, and no new absolute Addis EC accuracy claim is made.

Starting analysis source: `cb2e1469ec222719c347b9b5f672765ede3ed7c7`. Starting publication source: `ef92395d1cfd5b7fcacc3b78574f6fb8af53abc6`. Changes are prepared on `codex/gallery-closeout-20261107` in [aethmodular PR #20](https://github.com/ahzs645/aethmodular/pull/20); the companion publication workflow is in [aethmodular-gallery PR #1](https://github.com/ahzs645/aethmodular-gallery/pull/1). The review PR identifies the exact resulting source commit. Neither branch has been merged or deployed as part of this receipt.

The November public archive was restored and its full SHA-256 verified:

| Field | Value |
|---|---|
| Release | `data-82d701248232` |
| Asset | `gallery-data.tar.gz` |
| SHA-256 | `82d701248232b1975ea2e452bc1d682877e811103a2aa1040aff9d4adc989051` |
| Extracted members | 5,137 |
| Scientific outputs changed | None; saved chart summaries were not used to fabricate a rerun. |

## What changed

### EPO evaluation and provenance

`epo_validation.py` treats the EPO projection as part of the fitted pipeline. Each frozen outer site fold estimates chemical spectral signatures, the nuisance projection and PLS coefficients from its training rows only. The source domain is finite training TOR below 150 µg/filter. All bundled source rows determine the site splits before that outcome restriction is applied.

The workflow saves each actual projection and PLS model, projection-fit IDs, PLS-training IDs, held-out IDs, site/fold membership, ranks, settings and hashes. A separate model uses all eligible source data for target prediction. That final model is not used to generate source out-of-fold predictions.

Original spectral files must match the decision-bundle manifest. The corrected spectra additionally require the continuum bundle's manifest-bound `rows.csv` and `wn.npy`, matching physical-filter order and spectral grid. Equal array lengths are insufficient. New archives carry ordered source IDs, sites, folds and metadata/prediction hashes.

The primary regression changes held-out TOR across the 150 µg threshold, introduces missing held-out labels and changes nuisance chemistry while keeping spectra and folds fixed. Its own-fold projection, model and predictions must be unchanged. Other folds and the separate all-source fit may change because those records are training data there.

The implementation reports insufficient chemistry, rank-deficient chemistry and impossible component counts instead of manufacturing a projection. The numerical nuisance basis uses an SVD with a recorded rank tolerance, preventing zero nuisance signal from producing arbitrary QR directions.

### Source metrics

The fixed-18-component scorer checks candidate identities and common site folds, rejects repeated physical-filter IDs, and scores the same finite-prediction intersection. It reports MedAE, true MAE, RMSE, mean and median signed bias, P90/P95 absolute error, eligible/available/common counts and coverage. Candidate failures remain visible in the availability ledger.

New scores live in `intercept_fixes_20261103/source_scores_v2/`; historical tables remain intact. The gallery exporter binds a corrected status to the score and filter-ledger byte hashes. A corrected score receipt cannot promote a stale sweep or pass-rule simulation. Those products need their own matching provenance, and otherwise retain an explicit withheld status for EPO source quantities.

For the fixed-k scoreboard, the heavy subset is TOR ≥15 and <150 µg/filter with TOR OC/EC <2.5. The older sweep uses HIPS-derived loading >20 µg with OC/EC <2.5. These are different subsets.

### Gallery and synthesis

The September synthesis and next-steps documents are preserved as dated files. Their replacements incorporate the November findings, distinguish numerical convergence from accuracy, and define a finite existing-data endpoint.

Experimental → Continuum has a **Build six-figure review section** action. It uses existing section editing, sharing and PDF/Deck export. The figures cover coverage, source-reference performance, baseline behavior, seasonality, optical-response sensitivity and candidate tradeoffs. Saved section links preserve the selected views and references; they continue to load the active data export, so a link is not a frozen research bundle.

Old EPO source scores remain withheld even when an old schema-1 snapshot is opened with the new UI. Essential evidence labels remain visible with explanations off. The October 28 organic-signal axis is now “FTIR residual from seasonal HIPS fit (µg/m³)”, and the distinct carbon-ratio references are explicit. Export composition preserves wrapped evidence text; the source table's population and coverage fields are included in exported content.

### Verification and publication

The blanket nonfatal Node-test exception is removed. Source, public-data and research checks have explicit scopes. Missing prerequisites, unexpected failures and skipped/TODO assertions fail the selected gate. Input hashes are checked for changes during verification.

The canonical Pages workflow now requires type checking, source/public-data tests and the normal production build, verifies the data archive hash against its tag, and writes a source/data verification receipt. The companion publication PR contains the same template. The new source CI workflow checks the pinned November snapshot and includes a phone-width, fresh-context section and completed-PDF-download acceptance journey. It publishes its receipts and browser outputs as a CI artifact; it does not deploy the gallery.

## Verification recorded locally

| Check | Result and boundary |
|---|---|
| Strict Node source suite | **368 passed**, no failures, skips or TODO assertions. Includes legacy-score guards, section serialization and wrapped-evidence export tests. |
| Public-data suite, restored archive | **91 passed**, no failures/skips/TODO assertions. |
| Available research Node assertions | **104 passed**, but the overall research gate is **unavailable, exit 1** because three test files lack declared prerequisites. |
| Existing selected Python source suite | **104 passed** before adding the focused EPO and metric suites. |
| EPO/metrics/gate/publication regressions | **40 passed**: 9 EPO, 10 metric/provenance, 12 rebuild-gate, 4 section-policy and 5 publication tests. Only synthetic/source inputs are used for EPO and metric tests. Ruff passed all nine touched science/export/test files. |
| Workflow syntax | Canonical publication and source-CI YAML parsed; their shell blocks passed `bash -n`. This does not execute a deployment. |
| Final local TypeScript / production build | **Blocked by execution resources.** TypeScript was killed with exit 137; reduced heap attempts exhausted the configured heap. A separate Vite attempt also exhausted its heap. A prior intermediate typecheck does not substitute for a final pass. |
| Browser acceptance | **Not completed locally.** Chromium could not create its required socket; the environment rejected an escalation request. The connected Browser service was unavailable, and shared-browser navigation to the local server was refused. No phone-width, fresh-browser or completed-download pass is asserted from those attempts. |

The local execution blocks above are historical context. The production build and browser journeys subsequently passed in CI, as recorded below. The missing research and scientific inputs remain separate from that public verification.

## Verification recorded in CI

[Gallery closeout checks, run 42359566329](https://github.com/ahzs645/aethmodular/actions/runs/42359566329) completed successfully on 7 November 2026 for source revision `c366f5095c166acf0e11d72ba44dafeeaeefe6f4`, using the pinned public archive identified above. Both jobs checked out that PR head directly. The downloaded artifact receipt confirms the same `source_sha`, `public_review_status: success`, data tag and full data SHA-256. Both `scientific-regressions` and `public-review` passed.

| Check | Verified result |
|---|---|
| Synthetic scientific and release-gate Python regressions | **40 passed**: 9 EPO, 10 metric/provenance, 12 rebuild-gate, 4 section-policy and 5 publication tests. |
| Node source suite | **368 passed**, zero failures, skips or TODO assertions. |
| Public-data reconciliation | **91 passed**, zero failures, skips or TODO assertions. |
| Type checks and public production build | **Passed**. This supplies the successful hosted verification that the local resource-limited attempts could not complete. |
| Phone-width browser journey | **Passed**: historical EPO source scores remain withheld and reference labels remain visible with notes off. The strengthened checks confirm source-plot text stays within its SVG bounds, table headers retain readable height, and the Evidence column is reachable by scrolling within the table. |
| Sharing and export browser journey | **Passed**: the six-figure review reopens in a fresh browser context and completes a **seven-page PDF download**. |
| Browser total and evidence preservation | **2 browser tests passed**; scope recording and CI artifact upload succeeded. |

The retained [gallery-closeout-public-review artifact](https://github.com/ahzs645/aethmodular/actions/runs/42359566329/artifacts/13568394319) is 374,399 bytes, with SHA-256 `eda78aaf8d9b0cad01929c1043860401061e756ded603f7083eefa8d1596ffe4`. The downloaded archive was checked against that hash.

Visual review of the downloaded artifact confirmed that both phone views are readable, source-plot labels and axes fit, and the table scrolls to Evidence. All seven PDF pages were reviewed; the revised source-performance page is visually clean, and the other six pages are pixel-identical to the previously reviewed package. The delivered `closeout-review.pdf` is **163,562 bytes**, with SHA-256 `19e7c84b75a527e2df1d85beaeca3b1802c55f10db9a954b03ab3c0077e057a2`, independently checked against the downloaded file. This PDF comes from tested source `c366f5095c166acf0e11d72ba44dafeeaeefe6f4`.

An earlier [successful run, 42358778130](https://github.com/ahzs645/aethmodular/actions/runs/42358778130), was associated with PR head `b3d2219368e8f43d9209d2f94990b71915981de6` and tested the default PR merge revision `b885897b7ab20b796cb750711953fa4c827e5281`. Subsequent visual inspection found phone-width label clipping and excessive table-header wrapping. The current revision fixes those issues and verifies the additional readability conditions above.

This record applies to tested source revision `c366f5095c166acf0e11d72ba44dafeeaeefe6f4` and the pinned public snapshot. A later documentation-only commit recording this evidence does not change the artifact's tested source identity. Subsequent implementation or figure-wording changes require their own CI result. These successful public checks do not constitute a real-data EPO rerun, full research reconciliation, or independent validation of absolute Addis EC accuracy. The unresolved inputs below still block those claims.

## Earlier input blocker and original restoration plan

At the end of the first software batch, the published archive contained display products but the checkout lacked the spectral matrices and row provenance needed to refit EPO. The original checkout did not contain the November transfer, continuum or intercept-fix tables. The following was that batch's restoration plan. It is superseded by the recovered inputs and explicit versioned commands in the [reproduction protocol](gallery-closeout-reproduction-2026-11-07.md).

Restore:

- The frozen October decision bundle, including its manifest, all three metadata tables, `improve_mass.csv`, original raw/spline/VIBES arrays and grids.
- `vibes_continuum_20261102/corrected_{anchorhp,deflate}_{improve,addis,network}.npy`, plus `bundle/MANIFEST.json`, `bundle/rows.csv` and `bundle/wn.npy`.
- The recorded VIBES background model used by the baseline workflow.
- Historical November tables needed by the full continuum exporter. The corrected source stage itself does not need target optical labels.

Set `AETHMODULAR_CLOSEOUT_BUNDLE` to the restored decision-bundle directory, then run from the repository root:

```bash
uv run python research/ftir_hips_chem/workflows/run_transfer_diagnosis_20261102.py \
  --bundle "${AETHMODULAR_CLOSEOUT_BUNDLE:?Set the restored decision bundle}" \
  --stages fit \
  --output research/ftir_hips_chem/output/tables/transfer_diagnosis_20261102/source_cv_v2

uv run python research/ftir_hips_chem/workflows/run_intercept_fixes_20261103.py \
  --bundle "$AETHMODULAR_CLOSEOUT_BUNDLE" --stages fit --models base epo

uv run python research/ftir_hips_chem/workflows/run_hips_response_20261103.py \
  --bundle "$AETHMODULAR_CLOSEOUT_BUNDLE" --stages scores \
  --baseline-output research/ftir_hips_chem/output/tables/transfer_diagnosis_20261102/source_cv_v2 \
  --epo-output research/ftir_hips_chem/output/tables/intercept_fixes_20261103/epo_site_cv_v2

uv run python gallery/data/export_continuum_baseline.py
```

The last command also requires the historical display tables listed above. It consumes the corrected source-score child directory while preserving separate saved target diagnostics. Do not label the old EPO sweep corrected merely because the fixed-k scorer succeeded. The new fixed settings were informed by earlier exploration; this rerun is retrospective evaluation, not independent confirmation of model selection.

## Remaining research-gate prerequisites

Three Node test files require these seven missing paths or file groups:

```text
research/ftir_hips_chem/output/tables/vibes_fullrange/interpretation_20261002/benchmark/heldout_predictions.csv
research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/meeting/category_grid.jsonl
research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/followup/common_test_predictions.csv
research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/followup/routing_nested_training_predictions.csv
research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/followup/routing_prior_test_predictions.csv
research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/external/external_fit_diagnostics.csv
data/spartan/raw/FilterBased/ChemSpecPM25/*.csv
```

The full Python research profile also requires all five saved daily-instrument pickle inputs. The new runner declares these inputs; it cannot silently skip a missing site.

## PR #19 review boundary and completion criteria

The original PR #19 history indexer derived `missingData` from its local checkout. The continuation replaces those stale flags with declared primary-export checks on the active host and verifies failure/restoration/recheck behavior in exact-head CI. Source history, execution evidence, data availability and scientific validation remain separate facts. Neither that interface check nor this receipt merges PR #19 or certifies missing historical scientific inputs.

Close the existing-data stage only after the corrected numerical artifacts are restored and scored, figure populations/calendars/versions are reconciled, the intended build passes its declared checks, a copied link opens in a fresh context, a complete figure package downloads, and the exact source/data pair plus scientific ledgers are preserved. An explicit conclusion that absolute Addis EC accuracy remains unresolved is compatible with that endpoint.

Independent Ethiopian TOR, full PMF qualification, further GPU optimization and exhaustive notebook migration remain follow-on work. They are not substituted for the bounded tasks above.
