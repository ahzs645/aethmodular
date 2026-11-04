# VIBES ELBO: Newton vs compiled L-BFGS (JAX), head to head

> **Correction:** the "Reproduce the upstream SciPy procedure" row and the JAX conclusion below were too generous. Measured against the saved run, plain JAX is further away than Newton (baselines moved > 20 %: 270 vs 110) and it stops early on different spectra. To reproduce the saved run, use the vendored adapter on the same stack. See [vibes-reference-webgpu-decision-2026-11-03.md](vibes-reference-webgpu-decision-2026-11-03.md).

Date: 3 November 2026. Third round after
[vibes-batch-gpu-tests-2026-11-03.md](vibes-batch-gpu-tests-2026-11-03.md) and
[vibes-elbo-newton-jax-2026-11-03.md](vibes-elbo-newton-jax-2026-11-03.md). The same tests ran on a
Colab G4 (RTX PRO 6000, 48 CPUs) and an A100 (12 CPUs), with the SciPy reference on the G4's CPUs.
Inputs are the 19 Oct network bundle (4,452 spectra, rank-70 background, tau 0.1) unless stated.

The two methods compared:

- **Newton**: `vibes_batch.fit_elbo_newton`. Batched damped Newton with the exact Hessian. A
  different optimiser for the same objective, aimed at the optimum.
- **JAX**: `vibes_jax.fit_elbo_jax`. The adapter's L-BFGS-B procedure (same memory, line search,
  stopping rules, retry), compiled into 100-iteration GPU chunks. "faithful" means the upstream
  procedure; "polish 5" adds up to 5 restarts from the converged point.
- **SciPy**: the adapter's own procedure (`fit_elbo_scipy_fused`), one spectrum per CPU core.

## What each one is good for

| Use | Best choice | Why (measured below) |
|---|---|---|
| Correct the network, or any batch, at the ELBO optimum | **Newton on a GPU** | 1.5–3 s for 4,452 spectra, 4.6–8.7 s for 17,808; every fit converges; same answer on every machine |
| No GPU available | **Newton on CPU** | 26 s for 1,024 spectra on the G4 VM, 114 s for the network; ~4× faster than SciPy on 46 cores, and it reaches the optimum |
| One spectrum at a time (app, interactive) | **Newton**, CPU or GPU | 0.03–0.2 s per spectrum on GPU, 0.1–0.7 s on CPU. SciPy takes 0.4 s on a typical spectrum and 10–47 s on the slowest |
| Sensitivity runs (change tau, the model, or the data slightly) | **Newton, warm-started** for parameter changes | tau 0.10 → 0.11: 3 iterations instead of 12, identical answer. Not for noisy data changes (see "Warm starts") |
| Reproduce the *upstream SciPy procedure* (same algorithm, same stopping rule) | **JAX on an A100**, else SciPy on many CPU cores | It is the only fast implementation of that algorithm: 138 s for the network on the A100, vs 845 s for the CuPy batch. It still cannot reproduce SciPy's end points, because they depend on rounding |
| Anything on a G4 | Newton | JAX on the G4 only ties the 46-core SciPy pool (101 vs 103 s for 1,024 spectra) |

JAX's only advantage is that it runs the published algorithm. Newton wins on every other
measure: speed, convergence, independence from the start point and hardware, and warm starts.

## 1. Job size (wall time, seconds; spectra drawn at random; > 4,452 = network tiled with 1e-5 noise)

| Spectra (slow-tail) | Newton G4 f32 | Newton G4 f64 | Newton A100 | Newton CPU (G4 VM) | JAX G4 | JAX A100 | SciPy 46 cores (G4 VM) |
|---|---|---|---|---|---|---|---|
| 1 (0) | 0.11 | 0.29 | 0.22 | 0.37 | 7.7 | 11.4 | 6.9 |
| 16 (2) | 0.16 | 0.20 | 0.37 | 0.64 | 8.6 | 11.4 | 6.7 |
| 128 (15) | 0.25 | 0.29 | 0.58 | 3.4 | 22.9 | 19.6 | 16.3 |
| 1,024 (120) | 0.40 | 1.0 | 0.88 | 26.4 | 101.5 | 49.9 | 103.2 |
| 4,452 (546) | 1.5 | 3.9 | 2.9 | 113.6 | 415.7 | 138.5 | (422, previous note) |
| 17,808 (2,184) | **4.6** | 13.8 | **8.7** | — | — | **304.2** | — |

- Newton grows roughly linearly once the GPU is full: ~3,800 spectra/s on the G4 (float32
  Hessian) and ~2,000/s on the A100. On the A100, float32 makes no difference.
- JAX has a floor of ~8–11 s even for one spectrum. Any spectrum that needs thousands of
  sequential L-BFGS iterations sets the wall time, and the field-blank tail needs ~60,000. It
  also needs a one-off compile: 6–28 s per batch shape set.
- The SciPy pool costs ~6 s to start up (spawned workers), then is tail bound like JAX.
- Newton's ELBO was never below JAX's or SciPy's on any spectrum, at any size.
- JAX left 2–3 of 4,452 fits (and 6 of 17,808 on the A100) unconverged: they hit maxiter on the
  retry. Newton converged on all of them (max 50 iterations).

## 2. One spectrum (best of up to 3 runs, after warm-up)

| Spectrum | | SciPy 1 core | SciPy + polish 5 | Newton CPU | Newton GPU | JAX GPU |
|---|---|---|---|---|---|---|
| typical PM2.5 (row 1278; saved 1,459 its) | G4 s | 0.37 | 0.32 | 0.12 | **0.03** | 0.38 |
| | A100 s | 1.6 | 1.6 | 0.28 | 0.08 | 0.42 |
| | ELBO | 10,428.78 | 10,428.78 | **10,429.14** | **10,429.14** | 10,428.98–10,429.02 |
| slowest (row 3802; saved 58,569 its) | G4 s | 9.8 | 9.8 | 0.15 | 0.08 | 0.04 |
| | A100 s | 47.1 | 48.9 | 0.39 | 0.22 | 13.7 |
| | ELBO | 25,142.99 | 25,142.99 | **25,371.64** | **25,371.64** | **14,048 (G4)** / 25,235 (A100) |
| premature stop (row 466; saved 12 its) | G4 s | 0.007 | 2.4 | 0.12 | 0.04 | 2.8 |
| | ELBO | 8,756.6 | 16,564.1 | **16,573.5** | **16,573.5** | 16,568–16,571 |

The slowest spectrum shows the core problem with L-BFGS. The same JAX code stopped after 104
iterations at ELBO 14,048 on the G4, but ran 56,770 iterations to 25,235 on the A100. Rounding
differences between the two GPUs sent it down different paths. Newton takes 23 iterations and
gives 25,371.64 on every machine. The A100 VM's CPUs are about 4× slower than the G4's, which is
why its CPU timings are worse.

## 3. Robustness: other tau values and background ranks (stratified 256 spectra)

The sample deliberately includes the hard cases: 32 slow-tail and 32 premature-stop spectra.
JAX and SciPy both use 5 polish restarts. Newton ran with default settings every time.

| tau, rank | Newton converged, median/max its | Newton s (G4/A100) | JAX polish s (G4/A100) | SciPy polish s (46 cores) | spectra where JAX's ELBO is below Newton's (> 1e-6 rel.) | sigma differs > 10 % (JAX vs Newton) |
|---|---|---|---|---|---|---|
| 0.05, 70 | 256/256, 14/40 | 0.46 / 0.83 | 62 / 32 | 50 | 245 | 26 % |
| 0.10, 70 | 256/256, 13/39 | 0.42 / 0.69 | 60 / 33 | 43 | 226–229 | 23–26 % |
| 0.20, 70 | 256/256, 12/29 | 0.32 / 0.48 | 60 / 28 | 41 | 198–201 | 22–23 % |
| 0.30, 70 | 256/256, 12/26 | 0.30 / 0.44 | 54 / 30 | 39 | 180–185 | 22 % |
| 0.10, 40 | 256/256, 12/47 | 0.33 / 0.86 | 43 / 23 | 22 | 223–229 | 3 % |
| 0.10, 20 | 256/256, 13/32 | 0.17 / 0.52 | 12 / 7 | 8 | 224–229 | 0 % |

- Newton converged on every spectrum in every configuration, and its ELBO was never lower than
  JAX's or SciPy's (0 spectra, every configuration).
- Most of the gaps are tiny (median 0.07–0.44 in ELBO), but the top 1 % are 300–1,400 at rank
  70. Those are the spectra where L-BFGS stops early or crawls.
- The lower the rank, the closer L-BFGS gets: at rank 20, sigma agrees within 10 % everywhere.
  The rank-70 production setting is the hard case.
- Lower tau (0.05) is slightly harder for every method.
- JAX polish and SciPy polish agree with each other about as well as either agrees with Newton
  (the same ~23 % sigma differences), so both are short of the optimum.

## 4. Multiple starts: is there more than one optimum? (128 spectra × 8 starts)

Start 0 is the adapter's start (nu = 0, d = 1). Starts 1–7 draw nu from N(0, 1) and d
log-uniformly from 0.001 to 3.

| | Newton (G4 / A100) | JAX + polish 5 (G4 / A100) |
|---|---|---|
| fits converged | 1,024 / 1,024 | 1,021–1,022 of 1,024 |
| ELBO spread across the 8 starts, max − min: median / max | **2e-10 / 1.3e-9** | 0.38–0.44 / 685–700 |
| spectra where all 8 starts agree (within 1e-6 rel.) | **128 / 128** | 8–9 / 128 |
| the default start reaches that spectrum's best ELBO | 128 / 128 | 38–46 / 128 |
| distinct end points per spectrum (median) | 1 | 8 |
| time | 1.3–1.5 s | 74–215 s |

From every start, Newton lands on the same point to within 1e-9 in ELBO. For these spectra and
these starts the ELBO has one optimum, and Newton finds it. L-BFGS ends somewhere different from
each start, because it stops before converging, not because there are several optima. The best of
JAX's 8 starts is still below Newton by a median of 0.1 (max 570).

## 5. Warm starts: refitting after a small change

The previous fit is the starting point (network on the A100; a stratified 1,024 sample on the G4).

| Change | Newton cold | Newton warm | JAX cold | JAX warm | Warm vs cold answer |
|---|---|---|---|---|---|
| tau 0.10 → 0.11 (A100, 4,452) | 2.9 s, 12/55 its | **0.48 s, 3/10 its** | 128 s | 41 s | Newton identical (1e-9); JAX sigma > 10 % different on 8 % |
| tau 0.10 → 0.11 (G4, 1,024) | 1.0 s | **0.25 s** | 121 s | 45 s | Newton identical; JAX on 10 % |
| spectra + 1e-4 noise (A100, 4,452) | 1.7 s, 11/31 its | 3.3 s, 3/61 its | 48 s | 31 s | Newton identical; JAX on 6 % |
| spectra + 1e-4 noise (G4, 1,024) | 0.78 s | 0.95 s | 50 s | 36 s | Newton identical; JAX on 8 % |

- For a parameter change, warm-started Newton is 4–6× faster and gives exactly the cold answer.
- For changed data, warm starts don't help Newton. The median drops to 3 iterations, but a few
  spectra need up to 61 iterations from the old point, against 31 from the default start, and the
  batch waits for them. Start from scratch in that case.
- For JAX, a warm start is faster but changes the answer, because L-BFGS's end point depends on
  where it starts.

## 6. Settings

*Newton chunk size* (full network, seconds; GPU memory = CuPy pool):

| Chunk | G4 f64 | G4 f32 | A100 f64 | A100 f32 | memory |
|---|---|---|---|---|---|
| 256 | 6.2 | 4.2 | 9.8 | 10.5 | 0.3–0.4 GB |
| 1,024 | 4.2 | 1.9 | 3.9 | 4.3 | 1.0–1.1 GB |
| 2,048 (default) | 3.8 | 1.4 | 2.8 | 2.9 | 1.9–2.1 GB |
| 4,096 | 3.6 | **1.2** | **2.2** | 2.3 | 3.3–3.5 GB |

Use chunk 4,096 with float32 on a G4, and 4,096 with float64 on an A100. Memory is small on both
(the A100 has 40 GB, the G4 96 GB).

*JAX* (A100, network, faithful): chunk length makes little difference (25 iterations: 137 s, 400
iterations: 130 s; host time 4–6 s). Pool 1,024 was fastest (107 s); pool 4,096 took 120 s. All
settings needed 60,000–67,000 sequential iterations for the tail; peak memory was 0.5–1 GB. No
setting fixes the tail.

## What this means

- **For the polish / optimum decision:** Newton is the tool. It reaches the optimum (the same one
  from every start), is the fastest at every job size, and gives the same answer on every machine.
- **JAX:** keep it as a fast, faithful re-implementation of the upstream SciPy algorithm, mainly
  for an A100 when the published procedure has to be run at scale. Don't use it to judge whether
  a fit has converged, and don't expect it to match SciPy spectrum by spectrum.
- **SciPy:** it remains the reference for the upstream procedure. On many CPU cores it is as fast
  as JAX on a G4.

## Reproduce

```bash
# bundle_v3 = bundle_v2 + compare_elbo_paths.py (+ vibes_jax x0); Drive gpu_bench/bundle_v3.tar.gz, sha256 e570618f…e422
python bundle_v3/code/compare_elbo_paths.py --bundle bundle_v3 --out cmp --no-cpu                      # A100
python bundle_v3/code/compare_elbo_paths.py --bundle bundle_v3 --out cmp --jax-max-n 4452 \
    --warm-rows sample:1024 --tune-jax '' --cpu-workers 46                                            # G4
```

Results: `output/tables/vibes_gpu_bench_20261103/compare_20261103/{g4,a100}/*.json` and Drive
`gpu_bench/compare_20261103/`. Script:
[`workflows/compare_elbo_paths.py`](../research/ftir_hips_chem/workflows/compare_elbo_paths.py).

## Notes and limits

- The first A100 session was dropped by Colab partway through; the A100 numbers are from a full
  rerun on a fresh session. The first G4 attempt deadlocked: the SciPy pool forked after JAX had
  started threads. The script now uses `spawn` workers.
- One timing run per configuration on shared VMs. Treat differences under ~20 % as noise.
- Spectra beyond 4,452 are copies of the network with 1e-5 noise, so they are similarly hard
  rather than new spectra.
- "One optimum" is shown for 128 spectra and 8 random starts each, not proven in general.
- These are numerical comparisons only; nothing here tests EC accuracy.
