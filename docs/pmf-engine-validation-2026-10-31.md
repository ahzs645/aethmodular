# Executed PMF validation — 31 October 2026

Historical execution record. For recovery onto current `main`, subsequent fixes,
and newly executed checks, see [the November integration record](pmf-integration-2026-11-05.md).

This is an execution record, not a claim of complete scientific validation or of
publication of the extension. The detailed private `EXECUTION_RECORD.json`, logs,
source snapshots and result archives accompany the source package separately.

## Recovered numerical core

The archived weighted/Huber solver and uncertainty module were actually recovered.
The solver source hash matches the archive. Independent numerical scripts were
rerun against SciPy NNLS/optimization and analytical fixtures: weighted blocks,
Huber expression, signed observations, normalization/product invariance, gradients,
fixed-profile projection, deterministic seeds, constrained projections and
resampling/checkpoint behavior. These checks passed; they do not establish ME-2
algorithmic parity. The focused regression suite passed **71 tests** under Python
3.13.5. This is not the old 177-test claim and not the full repository suite.

## Actual raw preparation

| Cohort | Physical filters | Eligible PM2.5 | Held | Excluded | Field blanks | Model channels |
|---|---:|---:|---:|---:|---:|---:|
| ETAD | 296 | 246 | 6 | 7 | 37 | 1,771 |
| ETBI | 72 | 53 | 10 | 0 | 9 | 1,600 |

The original 102-filter ETAD cohort is a subset, not 102 additional filters. The
ETAD grid is strictly restricted to native overlap and drops one old extrapolated
upper endpoint. Every disposition is recorded; the implementation does not hardcode
these counts as expectations for future batches.

## Independent real-data fitting

Ordinary and Huber five-factor fits were executed on full ETAD, its historical
102-filter subset, and ETBI, with seeds 1 and 10. All **12 selected starts** passed
the recorded objective/gradient convergence criteria (two starts in each of six
cases). Reports, source archives, every start and residual tables were generated.

Frozen-profile projection was executed as an operational self-check on the two
full cohorts, with H unchanged. This is not held-out predictive validation. The
broad historical rank search, 20-resample scientific pilots and total-OM calibration
study were not silently counted as rerun in this integration iteration.

## Actual ESAT backend

The original source-pinned Python SA backend was executed, not mocked:

- Stock WS on signed ETAD/ETBI fails on the controller/update `hold_h` keyword
  mismatch. The failure receipts are preserved.
- The explicitly local repaired WS path produced finite signed-G/nonnegative-H
  results, with decreasing objectives. Both 1,200-iteration and extended
  6,000-iteration attempts remain **capped**, because not all declared convergence
  criteria passed. No unmodified-ESAT or physical-source validity is inferred.
- Stock LS ran on the public St. Louis 418×13 data at rank six with two seeds.
  The original controller stopped at 6,302 and 9,507 iterations, but the additional
  stationarity criterion failed. These results remain unqualified/capped even
  though the upstream controller declared convergence. That distinction is in
  each model's metrics; it is not hidden by a green execution receipt.

## Actual native EPA ME-2 evidence

The PR already acquired an actual Windows native workflow during this work.
The following remote execution was inspected and its artifact retrieved:

- Commit: `60ee83833571d6adad0a0c1f8e872cdbd63490de`.
- GitHub Actions run: `41187638735` (completed successfully).
- Artifact: `13037273109`, `native-pmf-public-validation`.
- ZIP digest: `d2c103af7636f8491e2ab8f3082687f372265eaa8aee8647f6dc37e359454894`.

There are **12 native fits**: ordinary and robust, seeds 1 and 10, on each of a
30×9 signed synthetic mixture, a 246×1,772 full-width synthetic matrix, and the
public 418×13 St. Louis dataset. The synthetic 246-row case is **not Addis data**.
The original official runtime is unmodified and pinned by file hashes.

Every downloaded case was independently rechecked: input numeric stream,
artifact hashes, native completion/convergence logs, factor constraints and
reconstructed Q within native text-rounding bounds. They were ingested into the
common reporting format. This is genuine native execution evidence, not an EPA
export smoke test or Python solver labeled as ME-2.

The local environment is Linux without Windows/Wine. **No native EPA fit on the
private ETAD/ETBI measurements was executed here.** The source extension provides
that Windows execution path, but its private-cohort run remains an explicit gate.
The success of this Windows CI environment does not imply EPA officially supports
all newer Windows versions; EPA's public page still lists versions 7–10.

## Publication and release gate

The remote PR remains open/draft. Read-only GitHub actions in this session could
inspect native CI but could not publish the new source extension. The source ZIP
includes a guarded installer for the reviewed PR head, tests and proposed PR body.
Do not state that the extension is on GitHub until the normal push succeeds.

Native full-cohort execution, stock ESAT signed-path qualification, laboratory
uncertainty/batch review, baseline adequacy and source-mass calibration remain
separate from code integration and numerical convergence. A usable pipeline must
retain these statuses, not force every case into a successful-source result.
