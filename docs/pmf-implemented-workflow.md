# Reproducible FTIR PMF workflow

This extension recovers the archived weighted/Huber solver and connects raw FTIR
intake, native conversion, preparation, model execution, fixed-profile projection,
comparison and reports. It replaces the provisional NNLS-only command from the
first PR #13 slice. The full implementation is in `src/pmf/`; command wrappers
remain in `aethmodular_cli/`.

## What is implemented, and what is not an execution claim

| Path | Implementation | Evidence boundary |
|---|---|---|
| Native FTIR records → X/U | Raw sample/background decoding, QC selection, AIRSpec baseline, blank pools, per-filter dispositions, immutable snapshots | Same-lot blank SD remains a provisional uncertainty model; batch provenance is not fabricated |
| Independent PMF | Recovered ordinary and Huber solver, multi-rank/multi-seed fitting, convergence checks, strict resume, diagnostics | Not a source-code conversion of ME-2 |
| ESAT | Pinned original SA Python controller and LS/WS paths; explicit opt-in local WS update repair | Stock signed WS failure is retained. The repair is not stock ESAT and allows signed contributions |
| Native EPA PMF5 | Verified bridge to the official-runtime ME-2 driver already on PR #13, native ordinary/robust modes, native result ingestion | Requires an appropriate Windows runtime. Public/synthetic Windows CI is not a run on private FTIR data |
| New filters | Refit compatible cohorts or project onto frozen profiles | Projection checks protocols and features, but does not independently validate new source mixtures |
| Reports | Dataset/run-derived HTML and Markdown, figures, residual tables, seed comparisons, status counts, provenance | Reports do not add unexecuted uncertainty analyses or claim calibrated source mass |

The execution record supplied with this extension distinguishes completed local
fits, unsuccessful/capped ESAT trials, and independently verified Windows CI
artifacts. Never use a green export or source-check status as evidence of a fit.

## Environment and entrypoints

Use the repository environment (`uv sync --python 3.13`) or install the focused
requirements in a separate Python 3.12/3.13 environment:

```bash
python -m pip install -r requirements-pmf.txt
python -m aethmodular_cli.pmf --help
```

The root command router registers `aeth pmf` without importing
scientific dependencies into unrelated `aeth doctor` or help commands. The module
form above works directly and has the same command vocabulary.

## 1. Prepare a new raw-data snapshot

Native inputs are three CSV exports: filters, analyses, and scan payloads. Their
contracts are documented in [pmf-data-contract.md](pmf-data-contract.md). Copy and
edit the example JSON configuration; set `AETHMODULAR_PMF_RAW` to the local input
folder rather than committing account-specific paths or Drive IDs.

```bash
python -m aethmodular_cli.pmf prepare \
  --config examples/pmf/etad.prepare.example.json \
  --output runs/ETAD-v2026-11-prepared
```

The reader selects exactly one retained analysis per physical filter. Invalidated
and superseded scans are retained in `analysis_selection.csv`, not indiscriminately
averaged. Sample and background beams are decoded and checked before computing
`-log10(sample/background)`. The configured model grid must lie within native
coverage. There is no implicit extrapolation, negative clipping, offset, or
fabricated uncertainty floor.

Preparation produces `dataset.npz` and a checksummed JSON sidecar, `inventory.csv`,
`analysis_selection.csv`, `blank_pools.json`, `processed_scans.npz`, and
`preparation.json`. Missing metadata, ambiguous retained analyses, unusable scans,
insufficient blanks, or invalid blank variability produce auditable held statuses
without blocking unrelated eligible filters. Blank pools can be grouped by lot,
lot/instrument, or explicit batch as configured; calendar month is not invented
as a laboratory batch. Other-site records are listed separately.

ETAD historical membership is an explicit ID allowlist. New IDs default to false.
No ETBI count is hardcoded. PM10 and field blanks are not ambient PM2.5 rows.
Known integrity exclusions must have recorded reasons in the preparation config.

## 2. Fit the independent solver

```bash
python -m aethmodular_cli.pmf fit \
  --input runs/ETAD-v2026-11-prepared/dataset.npz \
  --dataset ETAD-v2026-11 --output runs/ETAD-huber-v1 \
  --engine python-independent --loss huber \
  --factors 4 5 6 --seeds 1 10 100 --iterations 3000
```

Use `--loss quadratic` for ordinary uncertainty-weighted fitting. Huber fits may
use a quadratic warmup, whose status is recorded separately. The recovered solver
requires both objective and conditioned projected-gradient checks for convergence.
All starts remain available, including failed and capped starts. The selected
model prefers the lowest-objective **converged** candidate; the lowest attained
capped alternative is reported separately rather than silently replacing it.

Every run saves original prepared inputs, a source archive, package/version and
source hashes, settings, per-start factors and histories, selected tables, and a
report. `--resume` only reuses results when input, code, versions and settings match.
It is not a way to reuse stale results after editing source files. An existing
output directory is never overwritten by a new run.

A paired CSV route is also supported:

```bash
python -m aethmodular_cli.pmf validate --x X.csv --u U.csv \
  --dataset imported-example --output runs/import-validation-v1
```

Use `--protocol protocol.json` to describe units/preprocessing for CSV imports.
Unknown protocol remains explicit, and frozen projection does not silently accept
it. CSV IDs preserve values such as `NA` and `0001`; row and feature order must
match exactly. Ragged/empty data rows are rejected, not silently assigned values.

## 3. Execute pinned ESAT deliberately

```bash
python -m aethmodular_cli.pmf fit --input prepared.npz \
  --dataset example --output runs/esat-ls-v1 \
  --engine esat --esat-method ls-nmf --loss quadratic \
  --factors 5 --seeds 1 10 --iterations 10000
```

The upstream files are pinned to Leah Gibson's `fix_resample_return` commit
`0cc8e8d58b06a4484c8255ee77d7f5d5dc96181c` and hash-verified before execution.
The backend is explicitly Python, not Rust. The adapter checks the arrays after
upstream float32 conversion, performs independent residual and stationarity
checks, and records its backend and conditioning.

`ls-nmf` rejects signed X. `ws-nmf` executes the stock signed path and preserves its
observed failure; it does not silently return independent-PMF factors. The opt-in
`ws-nmf-repaired` fixes the controller keyword mismatch and replaces the defective
signed update with signed weighted least squares for G and nonnegative coordinate
NNLS for H under the original SA controller. Upstream source files remain unchanged.
The repair is conspicuously marked in provenance and reports. It permits signed G
and therefore is not conventional PMF with nonnegative source contributions.

Robust requests to the pinned ESAT Python path are rejected: setting its documented
flag is not proof that robust weights participate in optimization. Use the
independent Huber path or native EPA robust path, without conflating their objectives.

## 4. Run the actual EPA ME-2 engine

PR #13's independently reviewed `research/ftir_hips_chem/scripts/python_pmf/me2.py`
is preserved, not replaced. The bridge verifies Git blob
`da2ae8291ed0d592767f51ac37e85c72516334ab` before loading it. It verifies the original
runtime executable, sealed script and key by SHA-256, uses the original parameter
contract, and does not bundle or modify those runtime files.

On a suitable Windows execution host with `EPA_PMF_RUNTIME` pointing to the
extracted official distribution:

```powershell
python -m aethmodular_cli.pmf fit --input prepared.npz `
  --dataset ETAD-v2026-11 --output runs/ETAD-native-v1 `
  --engine epa-pmf5 --loss native-robust `
  --factors 5 --seeds 1 10 100 --native-timeout 900
```

Use `--loss quadratic` for native ordinary fitting. `native-robust` is a separate
name from independent `huber`. Native selection uses reported `Q_main`; the common
recomputed quadratic diagnostic remains separate. Native convergence must appear
in the native log, and returned factors must reconstruct the reported Q within a
rounding-derived bound. An unsupported host produces failed/blocked evidence,
never a silent fallback to another solver. Python iteration/tolerance/warmup
options do not tune the sealed native solver; its verified parameter contract and
wall-clock timeout control the native execution.

For a manual GUI path, `epa-export` writes exact paired tables and a `not_run`
receipt. `epa-launch` records an explicit executable's launch; process exit is not
convergence. `epa-import` accepts normalized native outputs with original evidence
and explicitly reported effective inputs if variable handling changed them.

`epa-ingest` verifies the complete result bundle produced by the native driver:

```bash
python -m aethmodular_cli.pmf epa-ingest --input prepared.npz \
  --case native-driver-output --loss native-robust \
  --dataset ETAD-v2026-11 --output runs/ETAD-native-import-v1
```

Input values, per-file hashes, native completion/convergence logs, factors and Q
are independently checked before generating a common report. This is an import
of a recorded native execution, not a new execution on the importing machine.

## 5. Frozen projection, comparison and uncertainty pilots

```bash
python -m aethmodular_cli.pmf project --input new-prepared.npz \
  --dataset ETAD-new --reference-run runs/ETAD-huber-v1 --rank 5 \
  --output runs/new-projection-v1

python -m aethmodular_cli.pmf compare --reference runs/ETAD-huber-v1 \
  --candidate runs/ETAD-quadratic-v1 --rank 5 --output runs/loss-comparison-v1

python -m aethmodular_cli.pmf bootstrap --reference runs/ETAD-huber-v1 \
  --rank 5 --repeats 20 --starts 2 --block-length 6 \
  --output runs/bootstrap-v1
```

Projection verifies exact features and preparation compatibility and keeps H
unchanged. Residual and date-range flags trigger review; they are not calibrated
out-of-distribution probabilities. The present projector rejects signed-G ESAT
and native robust references rather than silently changing their model convention.
Ordinary/native quadratic profiles can be projected using the declared independent
quadratic projector. A cohort refit can alter old contributions and requires
factor matching; projection and refitting are distinct operations.

Bootstrap is segmented by chronology, lot and long gaps. Each resample retains
all starts and uses converged-candidate selection. Only accepted mapped/converged
factors enter central 90% empirical spreads. This is conditional on fixed rank,
preprocessing and U, not native EPA BS/DISP/BS-DISP or validated confidence coverage.
Recovered constrained-band and rotation functions remain importable in
`src.pmf.uncertainty`; their numerical tests run separately. No new scientific
DISP/rotation campaign is inferred from those unit tests.

## 6. Reports and pipeline operation

```bash
python -m aethmodular_cli.pmf pipeline --config examples/pmf/pipeline.example.json
python -m aethmodular_cli.pmf report --run runs/ETAD-huber-v1
```

The pipeline creates new versioned preparation/run directories and an overview.
Each fit/projection generates HTML, Markdown and figures. Keep HTML and its sibling
image files together. Reports verify saved artifacts before reading results.
Counts, dates, rank, engine, seeds, failure/cap status, uncertainties, residuals and
provenance derive from the actual run; no 102/246/53 narrative is hardcoded.
An exported-but-not-run EPA case makes a pipeline `review_required`, not fully run.

## Release limits

The delivered implementation is a reproducible research workflow, not a certification
of source identities. Native public/synthetic CI establishes runtime feasibility,
not private-filter scientific validity. Same-lot blank uncertainty, baseline
sensitivity, missing batch data, historical spectral masks, ETAD-0037-5 and verified
factor-to-source-mass calibration remain scientific review gates. The historical
mass-calibration pilot is not rerun or silently transferred to the new 1,771-channel
ETAD grid. Source names and µg/m³ allocations require their own evidence.
