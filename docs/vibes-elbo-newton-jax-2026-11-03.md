# VIBES ELBO on the GPU: why it was slow, and two fixes tested on Colab

> **Correction:** this note calls the JAX loop the "same optimiser" as SciPy and its plain runs "faithful". Both overstate it: it shares SciPy's settings, line search and stopping rules, but not L-BFGS-B's Cauchy-point handling of the bound on d, and it leaves SciPy's path at iteration 1. See [vibes-reference-webgpu-decision-2026-11-03.md](vibes-reference-webgpu-decision-2026-11-03.md).

> **Follow-up:** the head-to-head comparison (job size, single spectra, other tau/rank, multiple
> starts, warm starts, settings) is in [vibes-elbo-newton-vs-jax-2026-11-03.md](vibes-elbo-newton-vs-jax-2026-11-03.md).

Date: 3 November 2026. Follows [vibes-batch-gpu-tests-2026-11-03.md](vibes-batch-gpu-tests-2026-11-03.md),
which found the batched GPU ELBO "not worth it in its current form". This note explains why
(measurements plus published work), then tests the two fixes it pointed to on Colab G4 and
A100 runtimes, on the full 19 Oct network (4,452 SPARTAN spectra, 2,722 channels, rank-70
background, fixed tau 0.1).

## Summary

| Question | Answer (measured) |
|---|---|
| Why was the batched GPU L-BFGS slow? | Every L-BFGS iteration is ~250 small kernel launches plus ~8 device→host syncs driven from Python (the line search runs on the host): ~16 ms per iteration on a G4, of which ~6 ms is GPU work. The slowest spectra need ~60,000 *sequential* iterations, so the tail sets the wall time no matter how many rows run in parallel. |
| **Fix 1: batched damped Newton** (new optimiser, same objective) | **Whole network ELBO in 1.5 s (G4, float32 Hessian), 2.9 s (A100), 82 s on the Mac CPU.** Median 12, max 52 Newton iterations. With the GPU MAP: **~55,000–59,000 fits/min, vs 641 for the best previous path and 112 for the adapter.** |
| Is the Newton answer right? | All 4,452 converge and all MAPs certify. ELBO ≥ the 5-restart polished L-BFGS on **every** spectrum (min +1e-5, median +0.13, max +1,377). SciPy restarted at Newton's end point gains ≤ 7e-10. Identical across G4/A100/CPU and float32/float64 Hessians (sigma to 1e-7, baselines to 2e-6 of corrected RMS). |
| **Fix 2: compile the L-BFGS loop with JAX** (same optimiser) | Per-iteration latency drops 4–6× on the A100 (8.4 vs 33.5 ms at 2,048 rows; 2.8 vs 17.5 ms at 64), but on the G4 only small batches gain (2.0 vs 5.3 ms at 64; at 2,048 it is slower, 21.7 vs 16.3 ms). Full network: **144 s on the A100** (5.9× the CuPy batch's 845 s), 419 s on the G4 (≈ that VM's 48 CPU cores). Still ~50× slower than Newton: the slowest spectrum's ~62,000 sequential iterations set the wall time. |
| What does this mean for the saved corrections? | Newton is the optimum the polish restarts were approaching. vs the saved 19 Oct run it moves **273** baselines > 5 % of corrected RMS (110 > 20 %); polish moved 217/107. The extra are mostly field blanks, where even polished L-BFGS was far from converged (median ELBO gap 418). Addis PM2.5: 12 of 274 > 5 %, 6 > 20 %. |

**Recommendation.** If the polish decision (previous note) is taken, use Newton instead of
polish restarts: it is ~100× faster than any L-BFGS path, needs no GPU to be fast (82 s for the
network on the Mac), and gives the same answer on every machine. The compiled JAX L-BFGS is
only worth keeping if the *upstream SciPy stopping behaviour* must be reproduced, and even then
it cannot reproduce SciPy's end points (they are path dependent); it is documented here as the
measured answer to "would compiling the loop help?".

## Background: where the time went and what the literature offers

### The measured cost of one batched L-BFGS iteration

From the cProfile of `fit_elbo_batch` on the G4 at 2,048 rows (previous note):

- ~16 ms per iteration: ~6 ms waiting on the GPU, ~9 ms of host Python.
- ~250 kernel launches per iteration (two-loop recursion over m = 10 pairs, objective, gradient,
  bookkeeping); launch overhead alone is ~3–7 µs each, i.e. 1–2 ms.
- ~8 device→host syncs per iteration, because More–Thuente decides on the host whether each
  row needs another evaluation (`.get` was 1.85 s of a 4.7 s profile).
- The arithmetic floor is ~4.5 GFLOP per iteration (four (B×70)·(70×2,722) products per
  evaluation): ~3 ms on the G4 (FP64 1.5 TFLOP/s) but only ~0.26 ms on the A100 (17.6 TFLOP/s).
  So the A100's GPU sat ~95 % idle while its slower CPU did the bookkeeping, which is why it lost.
- The tail: the slowest field blanks take ~60,000 iterations. At 10–14 ms each that is
  10–14 minutes, matching the A100's 845 s full run. More parallel rows cannot shorten it.

### What published work and tools say

- **BoTorch's batched L-BFGS-B (D-BE).** BoTorch 0.15 switched multi-start acquisition
  optimisation to "decoupled batched evaluation": one optimiser state per start, evaluations
  batched, exactly our design. The paper comparing it to coupled batching reports only
  1.1–1.8× gains, because their per-point evaluations are expensive and the per-row bookkeeping
  is a small fraction. Ours are cheap (~0.15 ms on one CPU core), so the bookkeeping dominates.
  ([BoTorch source](https://botorch.readthedocs.io/en/latest/_modules/botorch/optim/batched_lbfgs_b.html),
  [arXiv 2511.13625](https://arxiv.org/html/2511.13625v3))
- **Compiling the loop (JAX `jit` + `while_loop`).** Removes Python and launch overhead, but a
  batched compiled loop runs in lock-step: `vmap`/batched `while_loop` keeps iterating until the
  slowest lane is done and batched `cond` becomes a `select` that computes both branches
  ([JAX `lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html),
  [jax#8409](https://github.com/google/jax/issues/8409)). With a 60,000-iteration tail this
  forces K-iteration chunks with host-side refill.
  Library options: `jaxopt.LBFGSB` has bounds but is no longer maintained and has an open
  report of its line search leaving the bounds
  ([docs](https://jaxopt.github.io/stable/_autosummary/jaxopt.LBFGSB.html),
  [jaxopt#590](https://github.com/google/jaxopt/issues/590)); `optax.lbfgs` has no bounds.
  We therefore ported our own L-BFGS (with SciPy's More–Thuente) rather than use either.
- **CUDA graphs.** Record a fixed sequence of kernels once and replay it, saving launch cost.
  Capture cannot contain device→host copies, so the line search would first have to move to the
  device with a fixed number of trials ([CuPy graph capture](https://docs.cupy.dev/en/latest/reference/generated/cupy.cuda.Stream.html),
  [NVIDIA PyTorch CUDA graphs](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/torch-integration.html),
  [`torch.compile`](https://PyTorch.org/docs/stable/generated/torch.compile.html)). NVIDIA's
  nvalchemi ships a batched L-BFGS that replays each step as a CUDA graph, the closest precedent,
  but without a line search or per-system convergence test
  ([JAX docs](https://nvidia.github.io/nvalchemi-toolkit-ops/main/modules/jax/dynamics.html),
  [PyTorch source](https://nvidia.github.io/nvalchemi-toolkit-ops/v0.5.0/_modules/nvalchemiops/torch/lbfgs.html)).
  Even so, it does not touch the 60,000-step chain.
- **Batched second-order fitting.** Gpufit runs one Levenberg–Marquardt fit per thread block
  entirely on the GPU and reports up to 42× over the same algorithm on CPU
  ([Sci. Rep. 2017](https://dx.doi.org/10.1038/s41598-017-15313-9)). The lesson is to trade many
  cheap sequential steps for a few heavy, parallel ones, which is what GPUs are good at. For us
  the heavy step is a per-spectrum 140×140 Hessian, built by GEMM exactly like the MAP solver's
  70×70 system.
- Launch overhead and many-small-kernels: ([GPUOS paper](https://arxiv.org/pdf/2604.17861),
  [NVIDIA forum](https://forums.developer.nvidia.com/t/launch-of-many-small-kernels-10x-slower-compared-to-one-kernel/350194)).

So there were two candidate fixes: keep the optimiser and compile it (JAX), or change the
optimiser so that there are few steps (Newton). Both were built and tested.

## Fix 1: batched damped Newton (`fit_elbo_newton`)

### Method

Same objective, parameters (nu, d; 140 per spectrum) and start (nu = 0, d = 1) as the adapter.
With theta_j = v_j − W_j·nu, D_j = Σ_k W_jk² d_k², g_j the expected pinball loss and
s = mean_j g_j:

    −ELBO = p log s + |nu|²/2 + |d|²/2 − Σ log d + const
    Hess  = Hs/s + diag(1, 1 + 1/d²) − u uᵀ,      u = ∇S / (s √p),  S = Σ_j g_j

- `Hs = Σ_j ∇²g_j` is positive semidefinite: g is convex in (theta, delta) and increasing in
  delta = √D, which is convex in d. So `A = Hs/s + diag ⪰ I` always, and the only
  indefinite part is the rank-one `−u uᵀ` from the log.
- `Hs` is three GEMMs of per-channel weights against fixed Khatri–Rao products of W and W²
  (W⊙W, W⊙W², W²⊙W², upper triangles where symmetric): (B×2,722)·(2,722×~9,900).
- Each step solves `A p = [−g, u]` (one batched LU) and applies Sherman–Morrison for `−u uᵀ`
  when `κ = uᵀA⁻¹u < 0.99` (the true Hessian is positive definite); otherwise it takes the `A`
  step, which is still a descent direction.
- Armijo backtracking with a fraction-to-boundary of 0.95 on d. The `−Σ log d` barrier keeps d
  away from 0, so the reference's `d ≥ 1e-10` bound never binds.
- A row converges when max|∇| ≤ 1e-5 (SciPy's pgtol), or when the exact Newton decrement is
  ≤ 1e-13·|f| (no representable progress left). The decrement test is only allowed on an exact
  (Sherman–Morrison) step, because `A ⪰ Hess` means a small decrement on `A` alone does not
  bound the remaining gain. Every network fit ended on an exact step.
- `hess_dtype='float32'` builds the Hessian GEMMs in single precision (an inexact Newton step;
  objective and gradient stay float64). It reaches the same optimum.

Checks: the Hessian matches central finite differences of the gradient to 5e-10 relative;
objective and gradient equal the existing batched (and so the upstream) ones; on the unit-test
fixture Newton converges in 9–10 iterations and reaches at least the polished L-BFGS ELBO.

### Speed (full network, 4,452 spectra)

| Where | Hessian | ELBO | MAP | Total | Fits/min |
|---|---|---|---|---|---|
| Colab G4 GPU | float64 | 3.9 s | 3.4 s | 7.3 s | 36,353 |
| Colab G4 GPU | float32 | **1.5 s** | 3.4 s | 4.9 s | **54,576** |
| Colab A100 GPU | float64 | **2.9 s** | 1.6 s | 4.5 s | **58,929** |
| Colab A100 GPU | float32 | 3.0 s | 1.6 s | 4.6 s | 57,530 |
| Colab G4 VM, 48 CPUs (numpy, MAP numpy) | float64 | 114 s | 41 s | 154 s | 1,730 |
| Colab A100 VM, 12 CPUs (numpy, MAP numpy) | float64 | 158 s | 142 s | 300 s | 891 |
| Mac, 8 BLAS threads (numpy) | float64 | 82 s | — | — | 3,263 (ELBO only) |

For comparison (previous note, same spectra): the vendored adapter, 44 workers on the G4 VM,
112 fits/min; fused SciPy ELBO on 44 cores + GPU MAP, 641 fits/min; the same with 5 polish
restarts, 557 fits/min; batched CuPy L-BFGS on the A100, 315 fits/min.

Newton iterations per spectrum: median 12, p90 24, p99 35, max 46–52 (vs a median of
1,444–1,730 L-BFGS iterations and a 60,000 tail).

Latency per Newton iteration (all rows of the batch, until all converge):

| Batch | G4 float64 | G4 float32 | A100 float64 | A100 float32 |
|---|---|---|---|---|
| 64 | 12.9 ms | — | 29.2 ms | — |
| 512 | 20.8 ms | 14.1 ms | 22.3 ms | 29.7 ms |
| 2,048 | 65.8 ms | 19.7 ms | 29.6 ms | 29.4 ms |

On the G4 the float64 Hessian is FP64-GEMM bound (~110 GFLOP per iteration at 2,048 rows at
1.5 TFLOP/s ≈ 73 ms, as measured), so float32 helps 3.3×. On the A100 the FP64 GEMM is fast and
Newton is host bound (~29 ms per iteration regardless of batch or precision), so float32 does
not help. Either way the whole fit is ~50 iterations, so overhead stops mattering.

### Accuracy

*Against the polished L-BFGS* (G4 fused SciPy with 5 polish restarts, the best previous estimate
of the optimum; all 4,449 rows that succeeded in both):

| | value |
|---|---|
| ELBO(Newton) − ELBO(polished) | min +1.0e-5, p1 +1.0e-3, median +0.13, p99 +715, max +1,377 |
| rows where Newton's ELBO is lower by > 1e-6 relative | **0** |
| rows where Newton is higher by > 1 / > 100 | 1,192 / 631 |
| sigma differs > 10 %: PM2.5 / PM10 / FB | 2.8 % / 2.3 % / 90.8 % |
| sigma differs > 2× | 0 |
| FB median sigma ratio Newton/polished | 0.83 |

The 137 spectra where polishing used all 5 restarts (previous note: "still improving slightly")
are short of the optimum by a median 326 in ELBO. Upstream's `vibes_pb_elbo` evaluated at
Newton's parameters agrees with Newton's ELBO to < 3e-10, so these are values of the same
objective, not an artefact of the new code.

*Is it a true optimum?* On 60 spectra (the 30 with the largest final gradient and 30 random),
SciPy L-BFGS-B started at Newton's end point improves the ELBO by at most 7.2e-10 (median 4e-12).

*Implementation independence* (vs the G4 float64 GPU run):

| Run | max sigma rel. diff | max \|ΔELBO\| | max baseline diff / corrected RMS |
|---|---|---|---|
| G4 GPU, float32 Hessian | 1.1e-7 | 1.5e-9 | 1.2e-6 |
| G4 CPU (numpy) | 5.3e-8 | 1.3e-9 | 1.8e-6 |
| A100 GPU, float64 | 4.4e-8 | 1.6e-9 | 1.5e-6 |
| A100 GPU, float32 | 6.6e-8 | 1.6e-9 | 1.5e-6 |

Compare: the fused-objective SciPy run differs from the saved run in sigma by > 10 % on 5.6 % of
spectra (36 % of field blanks), and the vendored objective on the Mac differs on 15 % of a 440-row sample.

### Effect on the baselines

Baseline change as a fraction of the saved corrected-signal RMS (4,447 rows that succeeded in
the saved run, the polished run and Newton):

| Comparison | median | > 5 % (PM2.5 / PM10 / FB) | > 20 % (PM2.5 / PM10 / FB) |
|---|---|---|---|
| Newton vs saved | 0.011 % | **273** (119 / 18 / 136) | **110** (46 / 7 / 57) |
| polished vs saved (previous note) | 0.011 % | 217 (102 / 16 / 99) | 107 (46 / 7 / 54) |
| Newton vs polished | 0.011 % | 62 (17 / 1 / 44) | 5 (1 / 0 / 4) |

Addis PM2.5: 12 of 274 move > 5 % vs saved and 6 > 20 % (1 > 5 % vs polished).
Per-spectrum list: `output/tables/vibes_gpu_bench_20261103/newton_vs_saved_all.csv`.

So the conclusion of the polish analysis stands and is sharpened: the premature-stop spectra
(~110 with large changes) are the material issue; the remaining Newton-vs-polish differences are
mostly field blanks, where L-BFGS converges very slowly.

## Fix 2: compiled L-BFGS with JAX (`vibes_jax.fit_elbo_jax`)

### Method

A port of `fit_elbo_batch` to JAX with identical logic: per-row memory (m = 10), SciPy's
More–Thuente line search (the same `_dcstep` arithmetic, now backend-generic), projected bound
handling, SciPy's stopping tests, the adapter's retry and optional polish. `chunk_iters` = 100
L-BFGS iterations are one `jax.jit`-compiled `lax.while_loop` with the line search as a nested
`while_loop`; no Python and no host sync inside a chunk. Between chunks the host reads the stop
codes, applies retry/polish/finish and refills free slots. Once the queue is empty the pool
shrinks to a smaller compiled size (2,048 → … → 8) so the tail is cheap. All shapes are compiled
up front (`precompile`, 8–16 s per shape set, reported separately).

Check: with the same inputs it follows the numpy batch exactly at first (iterates equal to
4e-16 after one iteration, identical evaluation counts after 1/5/20 iterations) and then diverges
chaotically at the roundoff level, the same way two SciPy runs do.

### Latency per L-BFGS iteration (slow rows, all slots busy; after compile)

| Batch | G4 CuPy batch | G4 JAX | A100 CuPy batch | A100 JAX |
|---|---|---|---|---|
| 64 | 5.3 ms | **2.0 ms** | 17.5 ms | **2.8 ms** |
| 512 | 8.4 ms | **6.8 ms** | 23.1 ms | **4.2 ms** |
| 2,048 | **16.3 ms** | 21.7 ms | 33.5 ms | **8.4 ms** |

JAX removes the host overhead (the A100 gains 4–6× at every size), but in lock-step every row is
evaluated on every line-search trial, including rows that already accepted a step, and XLA's
float64 `ndtr` is slower than CuPy's fused `normcdf` kernel. On the G4, whose FP64 rate is low,
that extra device work outweighs the saved overhead at 2,048 rows.

### Full network

| VM | Run | ELBO | Compile (one-off) | MAP | Fits/min | Same run before |
|---|---|---|---|---|---|---|
| A100 | faithful (polish 0) | **143.5 s** | 75.8 s | 1.7 s | 1,840 | CuPy batch 845 s (315 fits/min) |
| A100 | 5 polish restarts | **163.2 s** | 75.7 s | 1.6 s | 1,621 | CuPy batch 925 s (288 fits/min) |
| G4 | faithful | 418.7 s | 41.5 s | 3.5 s | 633 | fused SciPy on 44 cores 422 s |
| G4 | 5 polish restarts | 478.7 s | 41.9 s | 3.5 s | 554 | fused SciPy on 44 cores 476 s |

Where the time goes: 619–625 compiled chunks, i.e. ~62,000 lock-step iterations, set by the
single slowest spectrum (it runs 10,000 iterations, hits maxiter, and the retry continues to
~50,000 more). Host time between chunks is only 7–17 s; the rest is device time, averaging
2.1 ms per lock-step iteration on the A100 and 6.6 ms on the G4. The pool shrank
2,048 → 512 → 128 → 32 → 8 as the queue emptied.

Agreement: the faithful JAX run differs from the saved run in sigma by > 2× on 9.7 % of spectra
(the CuPy batch: 9.6 %), the same path dependence as before. With 5 polish restarts it agrees
with the polished CPU run as well as the CuPy batch did (sigma > 10 % on 2.8 %, none > 2×), and
1,650 spectra still sit below the polished ELBO by > 1e-6 relative (min −786): both are short
of the optimum. Newton's ELBO is ≥ every JAX run's on every spectrum (min +1.6e-6).

So compiling the loop removes the host overhead, as predicted, and is worth ~6× on an A100, but
it cannot fix the sequential tail, which only a different optimiser does.

## What to use

| Goal | Path |
|---|---|
| Reach the ELBO optimum (fixes the premature stops) | `fit_elbo_newton` + `solve_map_batch`; GPU if available (seconds), otherwise CPU (~1.5–2.5 min for the network) |
| Reproduce the upstream SciPy procedure as closely as possible | fused SciPy on all CPU cores + GPU MAP (previous note). No batched or compiled re-implementation reproduces SciPy's end points. |
| Repeated sensitivity runs (blank banks, lots, 30k+ fits) | Newton on a GPU: the 32,286-fit job would take ~35 s of fitting instead of ~55 min |

Adopting Newton is the same scientific decision as adopting polish (it changes ~270 baselines
materially vs the saved run and needs downstream EC recomputed for those filters); it is not
made here.

## Code

All opt-in; `vibes_baseline.vibes_baseline_matrix` and saved corrections are unchanged.

- [`scripts/vibes_batch.py`](../research/ftir_hips_chem/scripts/vibes_batch.py):
  `fit_elbo_newton`, `_elbo_newton_terms` (objective, gradient, Hessian pieces), `_khatri_rao`;
  `_dcstep` now takes `xp` so JAX can reuse it.
- [`scripts/vibes_jax.py`](../research/ftir_hips_chem/scripts/vibes_jax.py): `fit_elbo_jax`,
  `precompile`. Importing it enables `jax_enable_x64`.
- [`workflows/benchmark_vibes_batch.py`](../research/ftir_hips_chem/workflows/benchmark_vibes_batch.py):
  `run --elbo newton|jax` (`--hess-dtype`, `--newton-chunk`, `--chunk-iters`, `--min-pool`),
  a `vs_polished` block when the bundle carries `polished_reference.csv` (`prepare --polished`),
  and a `latency` stage.
- [`tests/test_vibes_batch.py`](../tests/test_vibes_batch.py): +5 tests (Hessian vs finite
  differences; Newton ≥ polished L-BFGS, float32 and float64, chunk < rows; tau check; JAX follows
  the numpy batch; JAX full fit with polish and refill, skipped without JAX).

## Reproduce

```bash
python -m pytest -q tests/test_vibes_batch.py tests/test_vibes_map_batch.py
python research/ftir_hips_chem/workflows/benchmark_vibes_batch.py prepare \
    --out research/ftir_hips_chem/output/tables/vibes_gpu_bench_20261103/bundle_v2 \
    --polished research/ftir_hips_chem/output/tables/vibes_gpu_bench_20261103/colab_results/g4/cpu_all_polish5.csv
# on a GPU VM (Colab has JAX and CuPy preinstalled)
python bundle_v2/code/benchmark_vibes_batch.py latency --bundle bundle_v2 --out latency_v2.json --backend cupy
python bundle_v2/code/benchmark_vibes_batch.py run --bundle bundle_v2 --out run_newton32 --backend cupy --rows all --elbo newton --newton-chunk 2048 --hess-dtype float32
python bundle_v2/code/benchmark_vibes_batch.py run --bundle bundle_v2 --out run_jax --backend cupy --rows all --elbo jax --pool 2048
```

The notebook [`vibes_gpu_colab_20261103.ipynb`](../research/ftir_hips_chem/vibes_gpu_colab_20261103.ipynb)
has the same steps. Bundle: `bundle_v2.tar.gz` (sha256 `05294234…ebce5`), in Drive
`Aethmodular VIBES/gpu_bench/`. Results: `output/tables/vibes_gpu_bench_20261103/colab_results_v2/{g4,a100}/`.
Copies of the results are in Drive at `gpu_bench/results_v2_20261103/`, along with the per-spectrum table
`newton_vs_saved_all.csv` and `baseline_effects.json`. Both VMs were stopped after the runs.

Software on both VMs: JAX 0.11.1 (CUDA 13 plugin), CuPy 14.0.1, NumPy 2.4.6, SciPy 1.16.3.

## Limits

- Fixed tau only. Newton relies on the log-barrier keeping d interior, which holds for this
  objective; a different prior on d would need the bound handled explicitly.
- "Optimum" means a local maximum reached from the adapter's start; Newton and polished L-BFGS
  start from the same point and Newton is higher everywhere, but global optimality is not shown.
- Numerical agreement only; nothing here is about EC accuracy.
- Single timing runs on shared Colab VMs; JAX timings exclude compile time (reported separately).
