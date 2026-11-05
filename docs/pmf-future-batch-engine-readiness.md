# PMF future-batch readiness and remaining gates

This extends the initial implementation merged through PR #13 with recovered
implementation pointers. See [the current integration record](pmf-integration-2026-11-05.md).
The implementation extension is not automatically considered published: inspect
the PR head and the supplied release receipt to distinguish a local package from
commits actually present in GitHub.

## Acceptance matrix

| Requirement | Implementation | Qualification |
|---|---|---|
| Reuse archived robust solver | `src/pmf/solver.py` | Recovered byte-for-byte; new numerical tests are separately recorded |
| Raw sample/background processing | `src/pmf/prepare.py` | Supported native export schema, explicit grid and AIRSpec baseline |
| Future cohorts without fixed counts | `src/pmf/intake.py`, `prepare.py` | Every physical filter receives a disposition |
| Preserve historical membership | Explicit allowlist | Unknown IDs are false, never missing-value Boolean truth |
| Missing metadata/blank support | Per-filter held/excluded reasons | No guessed values or invented single-blank SD |
| Immutable inputs/results | `src/pmf/io.py`, `runner.py` | Strict source/settings/input checks on resume |
| Actual ESAT execution | `esat_backend.py` | Stock LS and WS, source pinned; WS failure retained; local repair explicit |
| Actual EPA ME-2 execution | `native_backend.py` + existing pinned `me2.py` | Original verified runtime, appropriate Windows host, no proprietary files bundled |
| Native result verification | `epa-ingest` | Exact X/U, artifacts, logs, factors and reconstructed Q checked |
| Projection | `project_case` | Compatible features/protocol, frozen H; no silent conversion of signed-G/native robust conventions |
| Cross-version matching | `compare_runs` | Common physical IDs; no automatic physical source labels |
| Uncertainty diagnostics | Recovered module and generic block bootstrap | Conditional independent diagnostics, not native EPA DISP/FPEAK/BS-DISP parity |
| Automatically generated reports | `report.py`, pipeline reports | Dynamic counts/dates/statuses, verified numerical artifacts |
| Regression and numerical checks | `tests/test_pmf*.py`, `scripts/pmf/validate_*.py` | Observe actual executed records; do not reuse a historical test-count claim |

## Scientific release gates, not code placeholders

1. Validate the baseline/uncertainty model using actual lab batch/instrument
   provenance and adequate matching blanks. Same-lot blank SD alone is not a full
   measurement-error budget. Large signed residuals require investigation.
2. Resolve missing date/volume and insufficient-blank holds before including those
   filters. Do not inflate eligibility counts by treating field blanks or PM10 as
   PM2.5 observations.
3. Assess factor count, local minima, preparation sensitivity, and stable matching.
   A completed run or lower Q alone is insufficient for physical source claims.
4. For stock ESAT signed data, verify a repaired/replacement version before treating
   it as a production backend. The explicitly modified WS path remains a distinct
   model with signed contributions; capped results remain capped.
5. Native EPA Windows public/synthetic CI is concrete execution evidence. It does
   not replace actual native runs on the approved private ETAD/ETBI input snapshots.
6. Recover/validate profile-to-functional-group/OM mass calibration and independent
   source evidence before naming new factors or publishing source mass fractions.
   Do not transfer the historical KF coefficients by factor index.
7. The new strict ETAD overlap grid has 1,771 channels; old 1,772-channel inputs and
   the manuscript's unrecovered 1,656-channel mask are different versions. Do not
   label their results an exact historical reproduction.

See [implemented workflow](pmf-implemented-workflow.md),
[data contract](pmf-data-contract.md), and
[source provenance](pmf-recovery-provenance.md). The delivered execution record
is the authority for which code and models were actually tested in this iteration.
