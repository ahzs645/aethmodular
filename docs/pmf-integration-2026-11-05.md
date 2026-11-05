# PMF recovery and integration, 5 November 2026

## Recovered work and base

PR #13 was already merged. The unpublished implementation was recovered from
`PMF_PR13_Implementation_20261031.zip`, verified against its release manifest,
and integrated onto `main` commit `33d0eb52ba49814f8971c86037fc25eb6e3d7879`.
No open aethmodular PR remained to combine at the start of this recovery.
The old branch-specific installer was not run against a newer checkout.

Source archive SHA-256:
`46ff0d998747c88e28648a583907653162039b34c09c70166bd6170dbe26f51e`.
The recovered solver, uncertainty implementation and vendored numerical sources
retain their original bytes and hashes. The existing native ME-2 driver is
unchanged. All other repository files remain based on current main, not the old
PR head.

## Included implementation

- Native FTIR sample/background decoding, retained-analysis QC, AIRSpec spline
  baseline, traceable blank pools and versioned prepared inputs.
- Recovered weighted quadratic/Huber solver, independent multi-start fitting,
  convergence-aware selection, fixed-profile projection and factor matching.
- Source-pinned ESAT execution, with original failures retained and the opt-in
  signed-contribution repair explicitly distinguished from conventional PMF.
- EPA export, operator-attested import, pinned native ME-2 adapter and verified
  native result ingestion. Export and process launch never imply a completed fit.
- Data-derived reports, immutable snapshots, source archives, input contracts,
  generic block bootstrap and documented scientific acceptance gates.
- Lazy `aeth pmf` routing, focused Python 3.12/3.13 CI, and an expanded Windows
  public/synthetic native check covering the adapter, result import and reports.

## Additional review fixes

Pipeline names now reject `.` and `..`, the reserved `prepared` directory,
duplicate names, empty job sets and unknown operations before creating output.
This closes an output-directory escape and prevents empty pipelines from being
reported as successful execution.

Operator-attested EPA robust imports now preserve the native loss convention.
They cannot silently enter quadratic frozen-profile projection; native robust
projection remains explicitly unsupported. Conflicting or malformed native loss
settings are rejected. Q(true) stays a quadratic diagnostic, not native Q_main.

Reports respect the caller's configured figure resolution instead of hardcoding
a savefig DPI. None of these changes modifies the recovered numerical core.

## Newly executed checks

Local platform: Linux, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3, Matplotlib 3.10.8, pytest 9.2.0 and Ruff 0.17.4.

- **105 focused tests passed**, comprising all PMF regression files plus
  `tests/test_cli.py` and `tests/test_repo_env.py`.
- Scoped Ruff passed for the implementation, CLI integration, PMF tests,
  validation scripts and native smoke script.
- Compileall and both PMF help entrypoints passed.
- Recovered solver numerical validation passed against independent convex and
  analytical references; recovered uncertainty numerical validation passed.
- Current native driver blob pin verified locally. Native execution was not
  attempted on this Linux host.
- Real native raw preparation was rerun successfully: ETAD has 296 inventoried
  filters, 246 eligible observations and 1,771 channels; ETBI has 72 inventoried
  filters, 53 eligible observations and 1,600 channels. Counts and holds match
  the archived record. All underlying data and outputs remain private.
- ETAD and ETBI quadratic and Huber five-factor fits, seed 1, converged again:
  **four converged starts**, with reports generated from the saved runs. This is
  a bounded reproduction, not the original 12-start campaign. Scaled RMSE was
  5.9124/6.3431 for ETAD and 4.5659/4.9147 for ETBI (quadratic/Huber). These high
  residuals continue to require baseline and uncertainty review.

The full repository suite and aggregate `aeth check` were not run in this bounded
source checkout. The PR CI results, rather than this local record, establish
whether the expanded Windows and Python matrix checks pass on the published head.
The October execution record is preserved as history, not silently counted as a
new run. No private input archive, prepared matrix, fit, plot, or proprietary EPA
runtime file is part of the source changes.

## Remaining scientific and execution gates

1. A suitable Windows runtime is still needed to execute native EPA against the
   approved private ETAD/ETBI snapshots. Public/synthetic native CI is not that test.
2. Stock signed-data ESAT remains unqualified. The repaired WS path allows signed
   contributions and is not a conventional nonnegative-source model. Capped
   results remain capped.
3. Laboratory batch/instrument uncertainty provenance, baseline residuals,
   factor count/stability and source-specific mass calibration require scientific
   review. No source labels or calibrated mass fractions are inferred here.

See [workflow commands](pmf-implemented-workflow.md),
[input contracts](pmf-data-contract.md), and
[acceptance gates](pmf-future-batch-engine-readiness.md).
