# Recovery and source provenance

The robust code was recovered from the user's saved
`Addis_PMF_01_Code_Raw_Data_Reports.zip` and prepared inputs from
`Addis_PMF_02_Prepared_Inputs.zip`. No earlier claim of 177 passing tests is adopted
as current validation. The delivered validation record lists tests actually rerun.
No private raw measurements, share links, account-specific paths or credentials
belong in the public source commit.

## Preserved components

* `src/pmf/solver.py` is byte-identical to the archived `src/python_pmf/solver.py`.
  SHA-256: `1b8225871bcf976f873371f525c6ea7bee61a6a962f262fe541777188db0f328`.
* `src/pmf/uncertainty.py` is recovered from the same archive. The generic workflow
  wrapper explicitly uses converged-candidate selection and retains capped starts.
* `src/pmf/vendor/airspec_baseline.py` preserves the archived AIRSpec implementation.
  Its source/license context and GPL-3.0 license are retained in `vendor/`.
* `src/pmf/vendor/esat_source/` preserves the original selected ESAT files and
  Apache-2.0 license. `PINNED_SOURCES.json` records the upstream commit and checksums.
  Local repairs live in `esat_backend.py`, not disguised edits of upstream files.
* The existing native ME-2 driver on PR #13 is preserved at its original path.
  The bridge pins its Git blob and requires the original official runtime hashes.
  Proprietary executables, sealed scripts and key material are never republished.

The wrapper, intake, preparation orchestration, reports, engine adapters and tests
are new code. Scientific computation and software-engine execution are validated
separately. Public native CI artifacts are evidence of native public/synthetic
execution, not private dataset results or proof of algorithmic equivalence.

## Source and outputs stay separate

Public source: implementation, documentation, source/license records, configs,
synthetic regression tests, and numerical validation scripts.

Private validation bundle: native raw exports, prepared matrices, actual private
fit outputs, source snapshots, figures, detailed receipts and inventories. Keep
this separate from a public repository commit. The self-contained source archives
inside results allow recovery even when the working checkout changes.
