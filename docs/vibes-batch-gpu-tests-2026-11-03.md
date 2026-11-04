# Batched VIBES on CPU and Colab GPUs: tests on real spectra

> **Update (later on 3 Nov):** the GPU ELBO question is answered in
> [vibes-elbo-newton-jax-2026-11-03.md](vibes-elbo-newton-jax-2026-11-03.md). A batched damped Newton
> optimiser fits the whole network's ELBO in 1.5–3 s on a G4/A100 (82 s on the Mac CPU) and reaches
> a higher ELBO than polished L-BFGS on every spectrum; compiling the L-BFGS loop with JAX helps the
> A100 4–6× per iteration but not the G4 at large batches.

Date: 3 November 2026. Follows [vibes-gpu-feasibility-2026-11-02.md](vibes-gpu-feasibility-2026-11-02.md)
(PR #16), which profiled synthetic spectra and left real-data and GPU behaviour unmeasured.

## Summary

| Question | Answer (measured) |
|---|---|
| Where does VIBES spend its time on real spectra? | **ELBO ≈ 85 %** (L-BFGS-B, median 1,444–1,730 iterations, tail 60,000). MAP ≈ 0.6 s/fit/core. |
| Exact CPU speed-up, same algorithm? | Fused objective + gradient: **3× per evaluation**, identical to 1e-15. 4,452 ELBO fits in 422 s on 44 G4 cores (**633 fits/min**). |
| Best same-machine end-to-end rate? | **Fused ELBO on 44 CPU cores + GPU MAP: 641 fits/min vs 112 for the current adapter (5.7×)** on the same G4 VM and spectra. The whole network takes ~8 min. |
| Does a GPU MAP solver work? | **Yes.** Batched interior-point MAP: 4,449/4,449 certified, ≤ 7.4e-7 from the saved baselines, **~157,000 fits/min on an A100** (1.7 s for the network), ~40,000/min on a G4. Works at sigma = 7.5e-7 where the PR #16 ADMM failed. |
| Does a GPU ELBO help? | **Not yet.** The batched L-BFGS is host/Python bound (~16 ms per iteration at 2,048 rows). On a G4 it roughly ties that VM's 48 CPU cores; on an A100 (12 slower CPUs) it is slower per row than the G4. A compiled loop (JAX `jit`+`vmap` or CUDA graphs) is the next step if more ELBO speed is wanted. *(Tested since: see the update at the top.)* |
| Can any re-implementation "match" the saved run? | **Corrected, see [vibes-reference-webgpu-decision-2026-11-03.md](vibes-reference-webgpu-decision-2026-11-03.md).** The vendored adapter on a Colab G4 is bit-identical to the saved run (440 of 440). The fused objective, which changes only the last bits of each evaluation, moves sigma > 10 % on 5.6 % of spectra (36 % of field blanks). |
| Is the saved VIBES output at the ELBO optimum? | **Not for ~3.5 % of spectra.** 155 fits (100 PM2.5, 14 PM10, 41 FB; 16 Addis) stop after < 30 iterations with a gradient of ~1e4. The optimum has a far higher ELBO (e.g. 8,757 → 16,560), sigma 10–1,000× smaller, and the baseline moves by a median **27 %** (max 86 %) of the corrected-signal RMS. |

**Recommended use today:** fused SciPy ELBO on every CPU core + batched MAP on the GPU
(or on CPU if no GPU). Decide separately whether to add polish restarts (below) to fix the
premature stops; over the network it moves 217 baselines by > 5 % of their corrected-signal RMS
(107 by > 20 %), leaves the rest within noise, and makes the result independent of the optimiser
implementation. It needs your sign-off.

## Code

All new code is opt-in; `vibes_baseline.vibes_baseline_matrix` and every saved correction are unchanged.

- [`scripts/vibes_batch.py`](../research/ftir_hips_chem/scripts/vibes_batch.py)
  - `fit_elbo_scipy_fused`: the adapter's ELBO procedure (L-BFGS-B, maxiter 10,000, same one retry)
    with fun and gradient computed in one pass and W² cached. Same objective to 1e-15.
  - `fit_elbo_batch`: per-row L-BFGS for many spectra at once (numpy or cupy). It mirrors
    L-BFGS-B's memory (m = 10), first step, skip rule, memory reset and stopping tests, and runs
    SciPy's More–Thuente line search vectorised across rows (bit-identical to `DCSRCH` in tests).
    It does **not** reproduce L-BFGS-B's Cauchy-point handling of the `d >= 1e-10` bound, so its
    path differs; see "Optimiser path" below. A rolling pool refills finished slots.
  - `solve_map_batch`: Mehrotra primal-dual interior point on the pinball MAP, with the latent x
    carried as its own variable (so small sigma does not amplify dual roundoff) and each Newton
    step one 70×70 Cholesky per row, built as a single GEMM against W⊗W. A row is accepted only if
    the primal–dual gap of the original pinball objective is ≤ 1e-9 + 1e-8·|P| (CLARABEL's default
    gap tolerances are 1e-8); otherwise NaN.
  - `polish_restarts=N` (both ELBO paths): after convergence, restart from the end point with fresh
    memory while the restart still lowers −ELBO beyond the factr threshold. Off by default.
- [`workflows/benchmark_vibes_batch.py`](../research/ftir_hips_chem/workflows/benchmark_vibes_batch.py):
  `prepare` builds a self-contained bundle (code + 19 Oct inputs + saved reference, 91 MB tar);
  `run` reports MAP-given-saved-sigma, end-to-end and same-machine CPU numbers to `summary.json`.
- [`vibes_gpu_colab_20261103.ipynb`](../research/ftir_hips_chem/vibes_gpu_colab_20261103.ipynb): the
  same on a Colab GPU from the Drive bundle (`My Drive/Aethmodular VIBES/gpu_bench/bundle.tar.gz`).
- [`tests/test_vibes_batch.py`](../tests/test_vibes_batch.py): 10 tests (objective vs upstream,
  line search vs `DCSRCH`, batch reaches SciPy's optimum, polish monotone, MAP vs closed form and
  CLARABEL down to sigma 1e-6, failure masking). PR #16's 7 tests still pass.

## Inputs

The frozen 19 Oct network run (`output/tables/network_baselines_20261019/`): 4,452 SPARTAN analyses
(3,685 PM2.5, 222 PM10, 545 FB; 28 sites) on the 2,722-channel model grid, the 87-blank rank-70
background, fixed tau 0.1, and that run's saved sigma, ELBO, iterations and float32 baselines.

## Hardware

| Runtime | GPU | FP64 GEMM | CPUs |
|---|---|---|---|
| Colab G4 | RTX PRO 6000 Blackwell, 96 GB | 1.5 TFLOP/s (FP32 75) | 48 |
| Colab A100 | A100-SXM4 40 GB | 17.6 TFLOP/s | 12 |
| Mac | — | — | 14 |

H100 was refused for this account. CuPy 14.0.1, NumPy 2.4.6, SciPy 1.16.3 on both VMs.

## Results

### MAP stage

| Test | Rows | Certified | max abs. diff vs saved | Throughput |
|---|---|---|---|---|
| Mac, numpy | 32 | 32 | 3.2e-6 (CLARABEL itself differs by the same at sigma 0.029) | — |
| G4, cupy, saved sigma | 96 | 96 | 7.4e-7 (median 1.6e-8) | ~40,000 fits/min |
| A100, cupy, saved sigma | 4,449 | 4,449 | see `run_all` (median 1.8e-8) | ~157,000 fits/min |
| CLARABEL (vendored), one core | — | — | — | ~100 fits/min/core |

The saved baselines are float32, so ~1e-8 is the comparison floor. IPM iterations: median 17, max 34.

### ELBO stage: throughput

| Path | Where | Rate |
|---|---|---|
| Vendored adapter (ELBO + CLARABEL), 44 workers, 440 random spectra | G4 VM CPUs | **112 fits/min** |
| Fused ELBO, 44 workers (all 4,452) | G4 VM CPUs | 633 fits/min (ELBO only) |
| Fused ELBO + GPU MAP, 44 workers, same 440 spectra | G4 VM | **641 fits/min (5.7×)**, 440/440 certified |
| Fused ELBO, 12 workers | Mac | ~105 fits/min (ELBO only) |
| Batched ELBO, cupy, pool 2,048 | G4 | ~5 µs per row evaluation; ~0.6× the 48 cores |
| Batched ELBO, cupy, pool 2,048 | A100 | ~10.6 µs per row evaluation |
| Full network (4,452), batched ELBO + MAP, faithful | A100 | 845 s + 1.7 s = 315 fits/min; 4,451 succeeded |
| Full network, fused ELBO with 5 polish restarts (44 workers) + GPU MAP | G4 VM | 476 s + 3.4 s = 557 fits/min; 4,449/4,449 certified |

Profile of the batched ELBO on the G4 at 2,048 rows: ~16 ms per L-BFGS iteration, of which ~6 ms
waiting on the GPU and ~9 ms of Python (≈ 250 small kernel launches, More–Thuente bookkeeping,
device→host syncs). The cost is per operation, not per row, which is why the A100 (faster GPU,
slower CPU) loses. Fusing the ELBO's elementwise math into one CUDA kernel and storing the L-BFGS
history newest-first halved the per-batch latency (2.6 → 1.25 ms). Further gains need the whole
loop compiled on the device.

A pool smaller than the job (e.g. 96 rows) is latency bound: the slowest spectrum's ~60,000
sequential iterations set the wall time (259 s for 96 rows on the G4).

### Optimiser path and reproducibility

The reference is SciPy L-BFGS-B with `factr = 1e7`: it stops when one iteration lowers −ELBO by
less than 2.2e-9 relative (≈ 2e-5 absolute at ELBO ~ 1e4). Results:

*Fused-objective SciPy, Colab G4 CPU, vs the saved 19 Oct run (all 4,447 fits that succeeded in both). Correction: this table was first labelled "same SciPy code". It is not: the objective's arithmetic differs in the last bits. The vendored code itself reproduces the saved run exactly on the G4.*

| Type | n | sigma differs > 1 % | > 10 % | median iterations |
|---|---|---|---|---|
| PM2.5 | 3,685 | 8.1 % | 1.4 % | 1,444 |
| PM10 | 222 | 17.1 % | 1.8 % | 3,778 |
| FB | 540 | 82.8 % | 35.6 % | 32,486 |
| all | 4,447 | 17.6 % | 5.6 % (> 2×: 1.1 %) | — |

*Premature stops.* 155 saved fits stopped in < 30 iterations; 154 of them stop at the same point
when re-run (deterministic, not noise). At that point the gradient's infinity norm is ~1e4 (pgtol is
1e-5). Example row 466: SciPy stops at iteration 12, ELBO 8,756.6, sigma 1.3e-3. Restarting SciPy
from there, or running the batch, reaches ELBO 16,560 (checked with the vendored `vibes_pb_elbo`;
SciPy started at that point stops after 1–5 iterations), sigma 6.4e-5.
`output/tables/vibes_gpu_bench_20261103/short_stop_spectra.csv` lists them.
By site, Addis (ETAD) has 16 of 319 (10 PM2.5).

*Batched vs SciPy (stratified 96 rows on the G4: 62 random, 14 short-stop, 20 slow-tail).*
Without polish the two optimisers stop prematurely on *different* spectra, so sigma differs > 2×
on 19 % of this (deliberately hard) sample. With 5 polish restarts on both:

| Stratum | GPU-polished vs CPU-polished sigma > 10 % | > 2× | CPU-polished vs saved > 2× |
|---|---|---|---|
| random (62) | 0 % | 0 % | 0 % |
| short stop (14) | 0 % | 0 % | 93 % |
| slow tail, mostly FB (20) | 35 % | 0 % | 0 % |

Baseline change, polished vs saved, as a fraction of the saved corrected-signal RMS: random rows
median 0.0 % (max 6 %); slow tail median 0.7 % (max 3 %); short stops median **27 %** (max 86 %).

*Full network, two independent implementations, both with 5 polish restarts* (A100 batched GPU
ELBO vs G4 fused-SciPy CPU ELBO, both followed by the GPU MAP; 4,442 fits that succeeded in all runs):

| Type | n | sigma > 10 % | sigma > 2× | baseline diff > 5 % of corrected RMS | > 20 % |
|---|---|---|---|---|---|
| PM2.5 | 3,685 | 0.5 % | 0 | 5 | 0 |
| PM10 | 222 | 0.0 % | 0 | 0 | 0 |
| FB | 535 | 16.4 % | 0 | 8 | 0 |
| all | 4,442 | 2.4 % | 0 | 13 | 0 |

Median baseline difference 1e-4 of the corrected RMS. Without polish, the same batched GPU run
differs from the saved run by > 2× in sigma on 9.6 % of spectra (PM2.5 7.5 %, FB 23 %).

So with polishing the result no longer depends on the optimiser implementation, except for small
differences on very flat field-blank fits; the main effect of polishing is to fix the premature stops.

## Decision needed

1. **Polish restarts.** Over the full network (polished run on the G4, all 4,447 fits that
   succeeded in both runs), the baseline moves by > 5 % of the saved corrected-signal RMS for
   **217 spectra** (102 PM2.5, 16 PM10, 99 FB) and by > 20 % for **107** (46 PM2.5, 7 PM10, 54 FB);
   median change 0.01 %. Addis PM2.5: 11 of 274 > 5 %, 6 > 20 %. The list is
   `output/tables/vibes_gpu_bench_20261103/polish_changes_gt5pct.csv`. 137 field-blank fits used all
   5 restarts and were still improving slightly. It is a departure from the published upstream procedure (which uses
   SciPy's defaults) but it reaches a higher value of the *same* objective. Downstream EC for the
   affected filters would need recomputing.
2. **ELBO on GPU.** Worth pursuing only with a compiled loop; otherwise CPU cores are as fast.
   *Superseded:* batched Newton makes the ELBO take seconds on a GPU and ~1.5 min on CPU; see
   [vibes-elbo-newton-jax-2026-11-03.md](vibes-elbo-newton-jax-2026-11-03.md). If polish is adopted, use Newton instead.

## Reproduce

```bash
# Mac: unit tests and bundle
python -m pytest -q tests/test_vibes_batch.py tests/test_vibes_map_batch.py
python research/ftir_hips_chem/workflows/benchmark_vibes_batch.py prepare
# any machine (GPU: --backend cupy)
python bundle/code/benchmark_vibes_batch.py run --bundle bundle --out run_all --backend cupy --rows all --pool 2048
python bundle/code/benchmark_vibes_batch.py run --bundle bundle --out run_s96 --backend cupy --rows sample:96 --pool 96 --cpu-ref 96
```

Results from these runs: `output/tables/vibes_gpu_bench_20261103/colab_results/{g4,a100}/`
(gitignored): `cpu_all_faithful.csv`, `cpu_all_polish5.csv` + `cpu_all_polish5_baselines.npy`
(polished candidate for the whole network), `adapter440.csv`/`fused440.csv`, `run_*` summaries,
`latency.json`. The bundle and notebook are also in Drive `Aethmodular VIBES/gpu_bench/`.
Both Colab VMs were stopped after the runs.

## Limits

- Fixed tau only (the production setting); tau estimation raises.
- Agreement with the saved run is numerical. It says nothing about EC accuracy, continuum
  recovery or the blank-PCA identifiability issue noted in the 2 Nov report.
- Timings are single runs on shared Colab VMs.
