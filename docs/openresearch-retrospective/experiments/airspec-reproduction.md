# AIRSpec: reproduce the locked calibration

## Question
Can the saved R baseline comparison and the ftir_13 locked calibration be reproduced from frozen inputs, without selecting on Addis HIPS agreement?

## Fixed inputs and readiness
[airspec-inputs.json](airspec-inputs.json) records current hashes and availability for the corrected DF1=6/8, DF2=4 caches, raw ETAD spectra and metadata, R-corrected truth, TOR and HIPS tables, deployed model exports, fixed cohorts, prior results and current code/environment. Missing or unhydrated entries block reproduction. An isolated checkout with staged input paths and complete transitive code/environment capture is also required. Existing runners write fixed output paths; never run them against the historical output directories.

The current uv.lock and source hashes describe a new-environment reproduction, not recovered historical software versions. Keep originals and full-precision expected tables immutable.

## Comparison method
Stage A: run `research/ftir_ec_phase3/scripts/validate_airspec_port.py` with explicit `--raw`, `--truth`, `--output` and `--jobs 2`. Match 319 valid scan IDs/order, the strict AIRSpec region and 2,002 corrected wavenumbers exactly.

Stage B: in the isolated checkout, from `research/ftir_ec_phase3`, run:

```sh
uv run --locked --no-sync python scripts/run_ftir_13.py
```

Use the saved lowest-OCEC 800 cohort, site-held-out seed 20260717, 20% test fraction, five-fold grouped training CV with seed 42, first-major-minimum component selection across 1–40, and scale=False. Evaluate DF1=6 and 8 with DF2=4. No selection on Addis HIPS. Verify physical filter/scan aggregation, units, IDs and split membership before comparing results. Smoke-cohort and HIPS-transfer branches must retain their own fixed inputs.

## Success criteria
Require no train/test site overlap and exact cohort/row/grid alignment. For the primary DF1=6 lowest-OCEC fit, recover the saved selected k=5 and compare unrounded saved tables/predictions at absolute tolerance 1e-6, with discrepancies reported rather than silently rounded away. Where only a rounded narrative survives, use its stated precision and label that weaker check.

For Stage A report BOTH acceptance definitions: historical worst per-spectrum absolute deviation ≤2e-4 and median of per-spectrum maxima ≤2e-5 absorbance; current validator median(max_abs_error / per-spectrum p95 absolute corrected signal) ≤0.02. The historical report used a different threshold from today's validator. Preserve both verdicts and the distributions; do not silently substitute one definition for the other.

The historical report's approximately 6e-7 worst deviation is a target to reproduce, not a tolerance invented from a new run. Passing is numerical reproduction. Addis HIPS/MAC agreement remains optical agreement, not independent thermal EC validation. New 1:1 panels must report errors-in-variables alongside OLS using available measurement uncertainties.

## Deliverables and blocking gate
An isolated execution manifest (source, environment, all inputs and output hashes), held-out membership comparison, component-choice comparison, metric/prediction difference tables, executed notebook/log and both R-port acceptance verdicts. Resolve missing inputs and isolated staging before starting. The completed execution is documented below.

## Readiness check completed in this audit
All 34 identified files are now present and hashed. The Drive HIPS placeholder was hydrated and checked. Stage A ran locally into a fresh directory: 319/319 scans, 2,002 analyzed wavenumbers, worst absolute deviation 6.014259418e-7 absorbance, median per-spectrum maximum 1.928894357e-7, median normalized error 1.390204962e-5. Both acceptance definitions pass. See [execution manifest](../../../research/ftir_hips_chem/output/tables/airspec_reproduction_preflight/manifest.json) and [log](../../../research/ftir_hips_chem/output/tables/airspec_reproduction_preflight/stdout.txt).

The repeatable bounded check is:

```sh
uv run --locked --no-sync python research/ftir_hips_chem/workflows/check_airspec_reproduction_inputs.py --manifest docs/openresearch-retrospective/experiments/airspec-inputs.json --output research/ftir_hips_chem/output/tables/airspec_reproduction_preflight_repeat
```

The output directory must be new. Stage B has now completed separately; see the full reproduction evidence below.

## Full calibration reproduction completed

The unchanged runner completed in an isolated Git worktree. All five historical result tables match across 43 column checks at absolute tolerance 1e-6; maximum difference is 2.469e-13. Both OCEC DF1 variants and both smoke fits recover k=5. The OCEC split has 606 training and 194 held-out rows with disjoint sites. The archived corrected caches were reused; full-pool correction was not repeated.

[Full reproduction report](../../../research/ftir_hips_chem/output/tables/airspec_locked_reproduction/report.md) · [Execution/input manifest](../../../research/ftir_hips_chem/output/tables/airspec_locked_reproduction/manifest.json). The independent Addis validation requirement remains unchanged.


## Native OpenResearch continuation — 2026-09-22

The executable baseline now passes in [Aethmodular — AIRSpec execution](http://127.0.0.1:4791/projects/203b7e92-c6ad-4d8a-9936-411a08bce8c6/tasks/new?pane=%7B%22kind%22%3A%22experiment%22%2C%22experimentId%22%3A%22d2b46613-884e-4eef-b9d8-d06d62459b63%22%2C%22view%22%3A%22overview%22%7D).
Run `5e953103-e1b4-4218-8ddf-f9c67188bfd4` at commit `fb0abeaf5d272b76c64bf6616743e56f00800362` passed all 84 comparisons.
Use this [native execution recipe and evidence](http://127.0.0.1:4791/projects/203b7e92-c6ad-4d8a-9936-411a08bce8c6/tasks/new?pane=%7B%22kind%22%3A%22file%22%2C%22path%22%3A%22airspec-baseline/report.md%22%2C%22source%22%3A%22artifacts%22%7D) for future runs;
the earlier manual commands remain as historical provenance for the prior local execution.
