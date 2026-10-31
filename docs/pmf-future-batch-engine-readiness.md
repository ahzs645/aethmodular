# PMF future-batch workflow: implementation and validation contract

Status: **design and handoff documentation**, not evidence that native EPA PMF5 or ESAT has run successfully. See the separate saved Python PMF analysis package in Davis Data / Python PMF Analysis - 2026-10-30. Do not equate the independent weighted solver with EPA's proprietary ME-2.

## Scope and scientific boundaries

The recovered exploratory cohorts are ETAD 246 PM2.5 filters (1,772 spectral channels), the historical ETAD 102-filter subset, and ETBI 53 PM2.5 filters (1,600 channels). ETAD and ETBI have different grids and must remain separate unless harmonization is explicitly validated. The previous analysis used same-lot blank variability as a **provisional** uncertainty proxy. Baseline-corrected spectra include substantial negative values. Model convergence is not source identification, and spectral contributions are not source-specific OM mass concentrations.

## Proposed run contract

Every run must be tied to a versioned manifest:

- `dataset_id`, `dataset_version`, site, PM fraction, acquisition/analysis date range, source checksums.
- Exact ordered physical filter IDs and selected scan IDs; replicate-selection rule; date, sampled volume, filter lot, instrument and batch.
- Exact ordered wavenumber vector, unit conventions, spectral windows, baseline method and settings.
- Concentration/absorbance matrix `X`, absolute uncertainty matrix `U`, and SHA-256 of both, with identical row and column identities.
- Blank IDs and uncertainty calculation, effective blank count, flags for inadequate matching, and whether blank subtraction was performed.
- Engine (`python-independent`, `esat`, or `epa-pmf5`), pinned version/build, factor count, random seeds, optimization settings, and source hashes.
- Execution status: `passed`, `capped`, `failed`, `not_run`, or `blocked`. Never turn an unsuccessful external-engine run into a successful result.
- Output paths and hashes for profiles, contributions, residuals, diagnostics, log, and report.

## Intake and QC

1. Read new raw scans and sample/background spectra; preserve original files.
2. Select the **retained valid analysis** for each physical filter. Do not average invalidated/reanalysis records indiscriminately.
3. Apply the repository's exclusion registry and preserve reasons, including known damaged filters. Distinguish PM10 and field blanks from PM2.5 samples.
4. Validate dates, sampled volumes, lot, instrument, physical-filter uniqueness, and wavenumber grid.
5. For each site/lot/batch, assess blank coverage and whether a defensible uncertainty estimate is available. Missing metadata and single-blank groups must be **held** rather than assigned fabricated uncertainties.
6. Recompute raw absorbance and AIRSpec baseline from new scans. Never require new IDs to appear in a frozen historical export.
7. Freeze the 102 historical ETAD membership by an **explicit allowlist**, never by a missing-value-to-boolean conversion.
8. Generate immutable per-cohort prepared matrices and an inclusion/exclusion inventory. Validate finite X, strictly positive finite U, and exact row/column identity.
9. Treat the current snapshot counts as regression fixtures, **not** permanent requirements: future ETBI intake must not assert exactly 72 physical filters.
10. Save an audit comparing previous and new versions, including new, changed, excluded, and held filters.

## Engine integration requirements

**Independent Python PMF:** use the saved weighted nonnegative-factorization implementation, preserve signed observations, report ordinary and robust objectives separately, and retain every starting seed, including capped runs. Verify numerical equivalence of the objective and reconstruct `Xhat = G @ H`.

**ESAT:** pin a specific release/commit and backend; check its imported matrices *after* preprocessing; run a known synthetic example and the real cohorts; reject nonfinite profiles, nonfinite contributions, and objective regressions on exact synthetic solutions. The previously tested signed WS-NMF path failed numerical checks; do not mark ESAT operational until a corrected version passes. Document whether the method permits signed contributions.

**EPA PMF5:** export ordered concentration and uncertainty tables plus metadata, and verify readback. Record variable classifications, uncertainty settings, seeds, and the native engine version. EPA PMF5 is a Windows application; a validated export is **not** a completed native run. Import native results only after a real execution and independently verify reconstruction and diagnostics. Do not claim ME-2 parity for independent Python, bootstrap, constrained-band sensitivity, or rotational diagnostics.

## Routine operations

- **Refit:** combine compatible, approved old and new samples; learn new profiles and contributions; match factors to prior models before making comparisons.
- **Project:** apply a frozen reference profile to compatible new spectra with verified feature ordering, units, preprocessing and uncertainty conventions; do not change the reference profiles; flag high residuals and out-of-distribution observations.
- **Report:** derive counts, dates, inclusion reasons, convergence, Q metrics, residuals, bootstrap outcomes, plots, and factor comparisons from the actual run manifest. Never hardcode historical counts into new reports.
- **Release gate:** report `ready_for_interpretation` only after residual and uncertainty review. Physical source names and mass fractions require corroborating chemistry/meteorology and validated profile-to-OM calibration.

## Required tests before production claims

- A new ETBI filter changes the inventory without tripping a fixed 72-filter assertion.
- An unseen ETAD ID is **not** placed in the historical 102-filter cohort.
- Missing volume/date, damaged filter, PM10, and inadequate blanks produce auditable held/excluded statuses without stopping unrelated eligible filters.
- Duplicate scan records are selected by explicit QC precedence.
- Swapped rows/columns, duplicate IDs, zero/negative U, stale hashes, and incompatible grids are rejected.
- Repeated identical runs reuse immutable results; changed inputs/settings invalidate caches.
- Native ESAT and EPA outcomes are recorded as `blocked` or `not_run` until actually executed and verified.
- Fixed-profile projection leaves H byte-identical and flags large residuals.
- Fresh reports reflect newly added filters and distinguish spectral contributions from calibrated mass.

## Outstanding work

Implement and test the intake changes and run-manifest/report generator in the repository, then integrate a pinned ESAT backend and an EPA native-run handoff. Recover batch-level uncertainty provenance, investigate signed baseline residuals and ETAD-0037-5, and verify source-specific mass calibration. This document is the acceptance checklist; it is **not** a claim that these tasks have been implemented on this branch.
