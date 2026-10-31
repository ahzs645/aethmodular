# PMF workflow: implemented commands and limits

This is the **first implementation slice** of the future-batch plan in [pmf-future-batch-engine-readiness.md](pmf-future-batch-engine-readiness.md). It is not the missing historical implementation branch. The independent solver here is an alternating weighted-NNLS reference implementation, **not** the previously archived robust solver and **not** EPA PMF5/ME-2. This slice has no Huber option, native ESAT execution, or automated raw FTIR scan preparation.

## 1. Audit new filter metadata

Provide a CSV with columns `filter_id,site,particle_fraction,analysis_id,analysis_status,date,volume_m3,lot,blank_count`. Optionally include `exclusion_reason`. An explicit `retained` analysis is required per physical filter. A separate historical CSV with `filter_id` defines historical membership; unknown IDs are never historical.

```bash
python -m aethmodular_cli.pmf_intake \
  --metadata new_filters.csv \
  --historical-ids etad_2023_historical_ids.csv \
  --output runs/intake-20261101.csv
```

Eligible means **metadata/blank-count screening passed**, not that the spectrum has been prepared or that analytical uncertainties have been validated. Two matching blanks is only the mathematical minimum for an SD, not a scientific adequacy standard. Inspect and approve the audit before preparing X/U.

## 2. Validate and fit prepared matrices

Both CSVs have the same first-column physical sample IDs and the same ordered spectral channel headers. X may be signed; U must be strictly positive, finite, and in the same units as X. Missing measurements must be resolved explicitly, not replaced by zeros.

```bash
python -m aethmodular_cli.pmf validate --x concentrations.csv --u uncertainties.csv \
  --dataset ETAD-2026-11 --output runs/etad-validate-01

python -m aethmodular_cli.pmf fit --x concentrations.csv --u uncertainties.csv \
  --dataset ETAD-2026-11 --output runs/etad-fit-01 \
  --factors 5 --seeds 1 10 100 --iterations 1000
```

The fit exports normalized residuals, nonnegative spectral profiles and contributions, and a JSON manifest with source hashes and per-seed objective/termination status. A plateau stopping rule is **not** proof of global optimality. Spectral contributions are not calibrated source OM mass. The implementation uses the full squared uncertainty-weighted objective; it is not robust fitting.

## 3. Project new samples onto frozen profiles

```bash
python -m aethmodular_cli.pmf project --x new_concentrations.csv \
  --u new_uncertainties.csv --dataset ETAD-new-2026-11 \
  --profiles runs/etad-fit-01/profiles.csv \
  --output runs/etad-project-01 --threshold 3
```

The exact wavenumber labels/order must match. The profiles file is read-only and its SHA-256 is recorded. RMS standardized residual above the chosen threshold triggers review. The operator must separately confirm baseline, unit, instrument, lot and uncertainty-method compatibility.

## 4. EPA PMF5 handoff and ESAT status

```bash
python -m aethmodular_cli.pmf epa-export --x concentrations.csv \
  --u uncertainties.csv --dataset ETAD-2026-11 --output runs/epa-handoff-01

python -m aethmodular_cli.pmf esat-check --x concentrations.csv \
  --u uncertainties.csv --dataset ETAD-2026-11 --output runs/esat-check-01
```

EPA handoff verifies a full-precision CSV round trip and emits a **not_run** receipt. It does not claim EPA's GUI has imported the file, nor does it provide native EPA results. ESAT emits a **blocked** receipt because the signed-data backend remains unvalidated. No ESAT result is silently substituted from the independent solver.

## 5. Outstanding implementation work

- Wire the commands into `aeth pmf` once CLI compatibility is reviewed.
- Connect raw scan/background selection and AIRSpec baseline recomputation to the intake audit.
- Replace provisional blank-count screening with instrument/batch-matched uncertainty provenance.
- Integrate the archived Python robust solver and its verified diagnostics, or independently validate a replacement.
- Verify a pinned ESAT backend against synthetic and real signed-data examples.
- Validate native EPA PMF5 import and execution on a supported Windows host, then import its actual outputs.
- Generate reports and versioned cohort comparisons from manifests, including factor matching, full QC provenance and mass calibration.
- Resolve ETAD-0037-5 and the source-specific OM calibration before assigning physical source percentages.

The regression tests in `tests/test_pmf_runner.py` and `tests/test_pmf_intake.py` exercise the basic interfaces; the dedicated PR workflow runs them in GitHub Actions. Passing these tests does not certify a scientifically validated source-apportionment analysis.
