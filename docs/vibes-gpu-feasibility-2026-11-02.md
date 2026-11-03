# VIBES GPU feasibility and existing implementations

Research date: 2 November 2026 (America/Vancouver).

## Decision

The current implementation runs on CPU, but neither VIBES nor CVXPY is inherently
CPU-only. Current CVXPY exposes GPU solver interfaces. For this workload, batching
many related spectra is more promising than launching a separate GPU solve for
every spectrum. GPU speed on the reported Colab G4 runtime has **not** been measured
in this investigation.

Start with a measured comparison of a cached CPU solver and an existing GPU QP
solver, then consider a specialized batched implementation. Profile both stages:
accelerating the MAP fit cannot remove time spent estimating its hyperparameters.

The accompanying prototype is an isolated numerical experiment. It is not wired
into `vibes_baseline_matrix`, does not change scientific defaults, and does not
replace previously generated corrections. No production or rental decision should
use a speed claim from this prototype until the actual GPU and real inputs pass
the acceptance checks below.

## Code inspected

Repository main: `bdb47d78fff6dd6b2468d806bb5741ed46d8bc33`.

- [`vibes_baseline.py`](../research/ftir_hips_chem/scripts/vibes_baseline.py)
  validates inputs, runs SciPy L-BFGS-B on the ELBO, estimates `sigma`/`tau`, then
  calls the MAP solver. Its elapsed-time diagnostic combines the stages.
- [Vendored MAP solver](../research/ftir_hips_chem/vendor/pyvibes/src/vibes/absorbance_estimators/map.py)
  builds a fresh CVXPY problem per spectrum and explicitly selects CPU CLARABEL.
- [Vendored PCA](../research/ftir_hips_chem/vendor/pyvibes/src/vibes/interference_models/pca.py)
  supplies a scaled orthogonal blank basis.
- [Rank-70 migration record](vibes-gallery-rank70-migration.md) documents a prior
  2,722-channel full-range run. This is a useful representative shape; it does not
  establish the shape, status or input manifest of the newly quoted 32,286 fits.

The authors' public repository, [FKamper/pyvibes](https://github.com/FKamper/pyvibes),
was also inspected at `52192970c02b94ee2ea476e503ccdd16f5c37de1`. Its MAP path still
selects CLARABEL and its ELBO path uses SciPy L-BFGS-B. No GPU implementation,
upstream GPU issue or pull request was found there. This is a bounded search
finding, not a claim that no unpublished implementation exists.

## What others have implemented

| Work | Verified capability | Relevance and limitation |
| --- | --- | --- |
| [CVXPY GPU interfaces](https://www.cvxpy.org/updates/index.html) | GPU interfaces introduced in 1.7: MPAX, cuOpt, CuClarabel, SCS/cuDSS; newer releases also expose Moreau | A model rewrite is not required merely to try a GPU backend. The existing explicit CLARABEL call still uses CPU. |
| [Moreau batching](https://moreau.so/guide/batching.html) | GPU batches with a fixed sparsity pattern, shared or per-problem matrix values, and individual statuses | Especially relevant when spectra share a blank basis. Native `CompiledSolver` is the throughput interface; a Python loop over CVXPY solves is a different workload. |
| [Moreau CVXPY interface](https://moreau.so/guide/cvxpy-integration.html) | Quadratic objectives and conic constraints, selectable CUDA device | A low-disruption comparison before native batching. Version requirements and CUDA installation must be checked on the VM. Documentation is not a VIBES speed benchmark. |
| [cuPIQP, EPFL](https://predict-epfl.github.io/cupiqp/) | GPU-resident proximal interior-point QPs, explicitly including large batches of small/medium problems | Strong alternative for the exact pinball epigraph QP. Same dimensions/sparsity are required within a batch. Current installation docs provide PyPI and source routes with CUDA-specific extras. |
| [MPAX](https://arxiv.org/html/2412.09734v3) | JAX LP/QP methods, compilation and same-shape batching | Useful existing QP engine and architecture precedent. Its LP algorithm is not its QP algorithm; use the appropriate implementation and matched accuracy. |
| [SCS/cuDSS](https://www.cvxgrp.org/scs/install/python.html) | GPU direct sparse linear algebra through SCS | A straightforward CVXPY comparison, but special installation and setup overhead matter for small fits. Its official docs link a Colab installation example. |
| [CuClarabel](https://arxiv.org/html/2412.19027v1) | GPU implementation of a quadratic conic interior-point solver | Closely related to the existing solver. Current CVXPY installation docs route it through Julia. Published large-QP results are not a VIBES throughput forecast. |
| [OSQP CUDA](https://osqp.org/docs/backends/index.html) | GPU indirect ADMM QP backend | Algorithmic precedent and another QP candidate; many-small-fit throughput still needs measurement. |
| [OptNet](https://proceedings.mlr.press/v70/amos17a.html) | GPU batched primal-dual QP solving demonstrated in 2017 | Established evidence that batching convex solves is practical. Do not add a ridge penalty to slack variables merely to meet a solver's positive-definite-Hessian requirement. |
| [BoTorch batched L-BFGS-B](https://botorch.readthedocs.io/en/latest/_modules/botorch/optim/batched_lbfgs_b.html) | Independent optimizer states, batched objective and gradient evaluations | Particularly relevant to the ELBO stage. Optimizer control remains CPU-side; GPU algebra would be a hybrid implementation. It relies on SciPy internals. |
| [POGS](https://pogs.readthedocs.io/en/latest/README/) | CUDA/cuBLAS ADMM graph-form convex solver with separable proximal operations | A direct algorithmic precedent for pinball loss plus a latent ridge penalty. Exploiting the VIBES PCA basis is our derived formulation, not an existing POGS VIBES benchmark. |

Other useful precedents include [ExaTron](https://epubs.siam.org/doi/10.1137/21M1450112)
for batches of small bound-constrained nonlinear problems and
[Gpufit](https://pmc.ncbi.nlm.nih.gov/articles/PMC5691161/) for many GPU-resident
nonlinear fits. Gpufit's ordinary least-squares/Poisson estimators do not implement
the VIBES variational and pinball objectives. Replacing VIBES with a different
spectral estimator is a scientific method change, not simply acceleration.

[JAXopt](https://github.com/google/jaxopt) provides useful GPU/vmap/LBFGSB examples,
but its official repository now says it is no longer maintained or developed.
It should not be selected as a maintained long-term dependency by assumption.

### A published example of the batch crossover

MPAX's cargo-planning LP experiment reports the following, on an H100 versus
Gurobi on a 16-thread CPU configuration:

| Problems in batch | MPAX GPU | Gurobi CPU |
| --- | ---: | ---: |
| 1 | 0.05 s | 0.01 s |
| 100 | 0.19 s | 0.57 s |
| 10,000 | 11.21 s | 57.57 s |

Gurobi solved the instances sequentially, allowing 16 threads for each. This is
not a comparison with 48 independent VIBES CPU workers. The result establishes a
possible crossover with batching, not a transferable speed multiplier.

The [SCS benchmark methodology](https://www.cvxgrp.org/scs/benchmarks/index.html)
is also useful: it recomputes residuals and duality gaps independently, includes
GPU transfers/setup, and records failed solves. Solver status alone is not enough
to compare speed fairly.

## Exact MAP formulation

For a spectrum `y`, blank mean `mu`, shared basis `W`, and positive fitted `sigma`,
the current PB MAP problem is

\[
\min_x \sum_j \rho_\tau(y_j-\mu_j-(Wx)_j)
  + \frac{\sigma}{2}\lVert x\rVert^2,
\quad \rho_\tau(r)=\max(\tau r,(\tau-1)r).
\]

This is convex. With nonnegative residual parts `p,n`, an exact QP is

\[
\min_{x,p,n}\frac{\sigma}{2}\lVert x\rVert^2
  +\tau\mathbf1^\top p+(1-\tau)\mathbf1^\top n,
\quad Wx+p-n=y-\mu,\quad p,n\ge0.
\]

There are no constraints requiring the latent coefficients or corrected spectrum
to be nonnegative. Preserve the original objective and all residual signs.

For an ADMM residual split `Wx+r=y-mu`, the latent update involves
`sigma*I + rho*W.T@W`, while the residual update is asymmetric soft thresholding.
The scaled PCA basis has orthogonal columns. Its Gram matrix is diagonal, so the
latent update becomes elementwise divisions after the shared matrix multiply.
For a nonorthogonal supplied basis, a small orthogonal rotation diagonalizes the
Gram without changing the ridge norm or fitted spectrum.

This yields a same-objective GPU-compatible sequence of matrix multiplies and
elementwise operations. It is an inference from the repository's equations and
established ADMM, not a claim that an existing paper has tested this exact port.

The feasible dual is

\[
\max_{\tau-1\le a\le\tau}
 a^\top(y-\mu)-\frac{\lVert W^\top a\rVert^2}{2\sigma}.
\]

The prototype uses this dual to check an objective gap, alongside primal and dual
residuals. Positive sigma is required; zero-sigma EBS needs a separate treatment.

## Numerical experiment provided here

- [`vibes_map_batch.py`](../research/ftir_hips_chem/scripts/vibes_map_batch.py):
  NumPy and explicitly requested optional CuPy execution, float64, per-row
  certificates and NaN outputs for failed/unconverged cases.
- [`test_vibes_map_batch.py`](../tests/test_vibes_map_batch.py): seven tests,
  including an independent scalar closed form, CLARABEL parity, weak/broad
  injections, nonorthogonal/rank-deficient bases, batch independence, reversed
  channels and failure handling.
- [`benchmark_vibes_gpu.py`](../research/ftir_hips_chem/workflows/benchmark_vibes_gpu.py):
  stage profiling of the unchanged adapter, a cached CPU QP comparator, cold/warm
  batched MAP runs, source hashes, versions, device metadata, and original row
  mappings. It writes diagnostics only, not raw spectra.

Seven tests passed on CPU using Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and
CVXPY 1.9.3. CUDA execution remains untested in this environment.

### Observed convergence limitation

A four-row, 140-channel small-sigma stress fixture used sigma values `1e-5`,
`3e-5`, `1e-4`, `1e-3` with `atol=rtol=2e-8`. At 2,000 iterations none of the
four rows converged; at 10,000, two converged. Unconverged rows remained masked.
Primal and dual values bracketed the independent CLARABEL optimum. Passing this
test means failure was detected correctly, not that the difficult fits succeeded.

At tiny sigma, an apparently small objective gap can still allow a different
baseline: coefficient error is bounded by `sqrt(2*gap/sigma)`. Therefore directly
compare baseline and corrected spectra, including broad continuum windows.

The prototype freezes each row after convergence, but its matrix multiplications
still include all rows until the slowest row stops. Stopping iterations are not
counts of arithmetic avoided. Production work should use bounded chunks and a
separate queue for difficult cases. The current experiment does not implement
that orchestration or replace the full-run checkpoint system.

### Stage profile and bounded full-shape pilot (synthetic)

Six synthetic spectra at 2,722 channels and rank 70, one BLAS thread, required
37.64 seconds in the unchanged adapter: 26.42 seconds in ELBO optimization and
11.16 seconds in the MAP call, including canonicalization. All six upstream
fits reported optimal MAP status. The synthetic signals and basis are generated
by the benchmark script; these times must not replace the user's measured
real-data rate. The complete receipt is
[`vibes_gpu_synthetic_20261102.json`](benchmarks/vibes_gpu_synthetic_20261102.json).

The cached CPU QP took 5.60 seconds including construction and all six solves,
passed independent dual-gap checks, and differed from the original baselines by
at most `3.58e-10` absorbance. This improves the MAP portion on this fixture;
the ELBO portion is unchanged.

The experimental NumPy ADMM batch took 1.15 seconds at a 2,000-iteration limit,
but **0/6 rows passed convergence**. Its warm run similarly passed 0/6. These are
failed attempts, not an acceleration result; no baseline arrays were accepted.
The benchmark exited with status 2 as designed. The tested small-sigma case is
reason to prefer evaluating existing GPU interior-point solvers first.

In this fixture, approximately 70% of adapter time belonged to ELBO. Even an
instantaneous MAP stage would reduce the measured total by at most about 1.42x.
This motivates batching the variational objective/gradient too, following the
BoTorch pattern while retaining spectrum-specific optimizer states and bounds.

## Running a bounded benchmark

With the repository environment and VIBES extra installed:

```bash
uv run --extra vibes python -m unittest discover -s tests -p test_vibes_map_batch.py -v
uv run --extra vibes python research/ftir_hips_chem/workflows/benchmark_vibes_gpu.py \
  --samples 12 --channels 2722 --rank 70 --backend numpy \
  --output research/ftir_hips_chem/output/tables/vibes_gpu_feasibility/cpu.json
```

For real inputs, export an NPZ containing `wavenumbers`, `spectra`, `mean` and
`components` from one already frozen blank model and supply `--input`. The
benchmark runs the original ELBO stage itself. Keep that NPZ private and retain
its input hash in the result. Use separate files/batches for different blank
models, lots, grids or ranks; do not refit a shared model to force shapes to match.

On the intended CUDA runtime, install the CuPy build matching its CUDA version,
then explicitly set `--backend cupy`. Missing CuPy/CUDA raises an error instead
of silently switching to CPU. Both cold and warm batch measurements include
setup, transfer and completion; cached CPU output also includes total wall time
and separately labelled solve-call timings. The synthetic candidate may exit
with status 2 because the stringent convergence or parity gate was not met.

The default absolute baseline threshold of `2e-5` is only a numerical pilot
threshold. A real-data acceptance criterion must be specified from measurement
scale, weak-signal requirements and downstream sensitivity before benchmarking.

## Acceptance before a full GPU run

1. Freeze the same source/input hashes, blanks, rank, grid, tau policy, precision
   and sample order on CPU and GPU. Fit and compare in float64 before float32
   checkpoint conversion.
2. Profile ELBO, canonicalization/setup, numerical MAP solve, transfers and output
   separately. Record actual CPU affinity/quota, GPU, driver, CUDA and solver
   versions. Advertised CPU count is not a throughput measurement.
3. Compare original CPU, cached CPU and GPU on the same stratified input subset.
   Include blanks, weak/strong loading, lots, Addis seasons, Bishoftu and other
   relevant sites. Existing exclusion flags remain intact.
4. Check optimizer status, finite outputs, residuals, duality gap, per-channel
   differences, continuum/injection recovery and fixed downstream predictions.
   The old adapter currently discards the MAP solver status and accepts finite
   arrays; a stricter benchmark must not count inaccurate results as fast fits.
5. Sweep useful batch sizes (for example 32, 128, 512), measuring cold/warm time
   and peak memory. Compare against an actual CPU process pool with one BLAS
   thread per worker, not only a serial CPU baseline.
6. Count successful validated fits per minute, failures/retries and end-to-end
   batch wall time. Extrapolate only after a larger representative pilot.
7. Integrate results with the existing signature/checkpoint/shard assembly rules
   before allowing GPU corrections into production outputs.

Matching CPU and GPU establishes numerical implementation agreement. It does not
establish independent EC accuracy or physical separation of broad aerosol signal
from a similar blank-PCA component. Hardware acceleration does not resolve that
identifiability issue.

## Interpreting the quoted timings

The user's `120 fits / 15 seconds` is aggregate throughput: 480 fits/minute.
At that same validated throughput, 32,286 fits take 67.26 minutes before other
overhead. The separate `0.75 seconds/fit` implies 6.73 serial fit-hours. These are
compatible if about six fits were active concurrently. Neither establishes
scaling to 48 advertised CPUs or the CPU time/billing of a specific rented VM.

For one batch of this duration, a cached CPU implementation may finish before a
large GPU rewrite is justified. Repeated blank/rank/season sensitivity studies
increase the value of reusable batched GPU execution.
