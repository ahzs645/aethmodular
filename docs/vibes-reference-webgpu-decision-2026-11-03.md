# VIBES ELBO: distance from the reference, WebGPU, and what to adopt

> **Update:** the full L-BFGS-B port was built after all, and it reaches the noise floor described below (144 of 155
> early stops reproduced). The HIPS reading in section 2 is superseded by a fairer within-site test. See
> [vibes-newton-adoption-tests-2026-11-03.md](vibes-newton-adoption-tests-2026-11-03.md).

Date: 3 November 2026. Follows [vibes-batch-gpu-tests-2026-11-03.md](vibes-batch-gpu-tests-2026-11-03.md),
[vibes-elbo-newton-jax-2026-11-03.md](vibes-elbo-newton-jax-2026-11-03.md) and
[vibes-elbo-newton-vs-jax-2026-11-03.md](vibes-elbo-newton-vs-jax-2026-11-03.md).
The reference is the frozen 19 Oct network VIBES run (4,452 spectra; 4,449 succeeded).

## Summary

| Question | Answer (measured) |
|---|---|
| How far is each method from the saved reference? | The vendored adapter on the G4: **0, bit-identical** (440 of 440). The vendored objective on the Mac: sigma > 10 % on 15 %, > 2× on 1.6 %. Plain JAX: > 10 % on 13 %, **> 2× on 9.7 %**, 270 baselines moved > 20 %. Newton: > 10 % on 16 %, > 2× on 4.6 %, 110 baselines moved > 20 %. |
| Is JAX closer to the reference than Newton? | **No, it is further away**, even though it has the same stopping rule. It stops early on 276 spectra, against 155 in the saved run, and only 17 are the same spectra. |
| Does the difference reach EC? | **Barely.** Newton moves 10 of 3,406 PM2.5 scans by > 0.1 µg/m³, and 1 by > 0.5 µg/m³. Repeat scans and field blanks do not improve materially, and agreement with HIPS does not improve. |
| Can it run on WebGPU? | **Yes, as a hybrid.** WGSL has no float64, so the GPU builds the matrices in float32 and JavaScript does the rest in float64. In Chrome on the M4 Pro: 0.09 s per spectrum (0.51 s without WebGPU), sigma within 2e-8 of the float64 result, every fit certified. |
| What should we do? | **Adopt Newton as the method for new work, in a new versioned run. Keep the 19 Oct run frozen, and don't change reported EC yet.** The reason is reproducibility, not better EC. Refit the IMPROVE calibration with Newton first. |

## Corrections to the earlier notes

Three earlier statements were wrong. The notes now carry corrections.

1. **"Re-running the same SciPy code changes sigma > 10 % on 5.6 % of spectra."** That run used my
   fused objective, which is not the same code. Its evaluations differ from the vendored ones only
   in the last bits. The vendored code on the G4 reproduces the saved run exactly.
2. **"JAX is the same optimiser, a faithful re-implementation of the upstream procedure."** It shares
   SciPy's memory size, line search and stopping rules, but it has no Cauchy-point step for the
   bound on d. It leaves SciPy's path at iteration 1 (see below).
3. **"JAX can't reproduce SciPy's end points because they depend on rounding."** Rounding is part of
   it, but the larger cause is that structural difference.

## 1. Distance from the reference

Same 440 random spectra for every row; "iterations" counts spectra with the same iteration count
as the saved run. Baseline change is RMS change as a share of the saved corrected-signal RMS.

| Run, vs saved | Same iterations | sigma > 1 % | > 10 % | > 2× | Baseline > 5 % | > 20 % | PM2.5 scans with EC moved > 0.1 µg/m³ |
|---|---|---|---|---|---|---|---|
| Vendored adapter, Colab G4 | **440 / 440** | 0 | 0 | 0 | 0 | 0 | 0 / 332 |
| Vendored objective + SciPy, Mac (M4 Pro) | 17 / 440 | 21.1 % | 15.2 % | 1.6 % | 5.9 % | 0.7 % | 0 |
| Fused objective + SciPy, G4 | 18 / 440 | 19.5 % | 6.8 % | 1.6 % | 2.5 % | 0.5 % | 0 |
| JAX, G4 | — | 27.7 % | 15.2 % | 11.8 % | 11.8 % | 8.2 % | 2 |
| JAX, A100 | — | 27.7 % | 14.5 % | 11.6 % | 11.8 % | 8.4 % | 3 |
| Newton | — | 27.5 % | 17.3 % | 5.2 % | 5.7 % | 2.7 % | 0 |

The whole network, for the runs that cover it:

| Run, vs saved | n | sigma > 10 % | > 2× | Baseline > 5 % | > 20 % | EC moved > 0.1 µg/m³ | > 0.5 µg/m³ (of 3,406 PM2.5 scans) |
|---|---|---|---|---|---|---|---|
| Fused SciPy, G4 | 4,447 | 5.6 % | 1.1 % | 2.2 % | 0.5 % | 0 | 0 |
| JAX, G4 | 4,448 | 13.1 % | 9.7 % | 10.3 % | 6.1 % | 28 | 4 |
| JAX, A100 | 4,447 | 13.3 % | 9.8 % | 10.5 % | 6.2 % | 28 | 4 |
| Newton | 4,449 | 16.4 % | 4.6 % | 6.2 % | 2.5 % | 10 | 1 |

By type, sigma differing > 10 % from saved:

| Run | PM2.5 | PM10 | Field blanks |
|---|---|---|---|
| Fused SciPy | 1.4 % | 1.8 % | 35.6 % |
| JAX (G4) | 8.5 % | 15.3 % | 43.4 % |
| Newton | 5.8 % | 9.5 % | 91.7 % |

Reading these:

- **The saved run is exact on its own stack and nowhere else.** Identical code on the Mac follows a
  different path on 96 % of spectra. So nobody can reproduce it on another machine.
- **The noise floor** of true L-BFGS-B without the same bits (fused SciPy and the Mac) is 2–6 % of
  baselines moved > 5 %, and 0.5–0.7 % moved > 20 %. No re-implementation can beat that.
- **Newton's differences are mostly real improvements.** About two-thirds of its sigma differences
  (497 of 730) are field blanks, where L-BFGS was still far from converged. Its ELBO is at or above
  every other run on every spectrum.
- **JAX's differences are new errors.** It adds early stops, so its baselines sit further from the
  saved run than Newton's do, and further from the optimum as well.

### Why JAX leaves SciPy's path

| Check | Saved | Fused SciPy | JAX G4 | JAX A100 | CuPy batch |
|---|---|---|---|---|---|
| Spectra stopping in < 30 iterations | 155 | 156 | 276 | 281 | 281 |
| ...of which the same spectra as saved | — | 154 | 17 | 15 | 19 |

SciPy's early stops survive last-bit changes (154 of 155), so they are a property of the algorithm.
The batched code's early stops fall on different spectra and are almost twice as common, so it is a
different algorithm in practice. After a single iteration, its iterate is already 0.9–2.9 away from
SciPy's in parameter value (rows 1278, 466, 3802).

The cause is confirmed. SciPy's first step lies exactly along the projected direction P(x − g) − x
(cosine 1.000000 on 5 of 5 spectra), which limits every d to its bound; the batched code uses the
plain −g direction (cosine 0.14–0.92). Adding only that projection in a scratch copy matched SciPy's
first iterate to 1e-16, but the paths separated again at iteration 2, and early-stop overlap on the
155 spectra did not recover (10, against 18 without it). L-BFGS-B's Cauchy-point and subspace steps
matter at every iteration.

So matching SciPy would mean a full port of those steps. It wouldn't be worth it: the best possible
outcome is the noise floor above, and the vendored code already gives the exact result on its own stack.

Spectra with an ELBO more than 0.1 % below the optimum: saved 862 (19 %), fused SciPy 857, JAX 924–928.

## 2. Does it reach EC?

I recomputed EC by applying the frozen all-IMPROVE VIBES PLS model to each set of corrected spectra.
The saved baselines reproduce `network_ec.csv` to 1e-14. Filters with a recorded volume of 0 are
excluded from µg/m³ figures.

**EC change, Newton vs saved.** On the 94 PM2.5 scans with a sampled volume whose baseline moved
> 5 %: median |ΔEC| 0.024 µg/m³, 90th percentile 0.10, max 0.54; 66 move down and 28 up.

| Filter | Saved iterations | EC saved → Newton (µg on filter) | ΔEC (µg/m³) | HIPS BC (µg/m³) |
|---|---|---|---|---|
| INDH-0302 | 10 | 1.90 → 1.49 | −0.54 | 4.00 |
| ETAD-0053 | 13 | 10.64 → 8.19 | −0.33 | 4.72 |
| ETAD-0166 | 15 | 12.02 → 9.89 | −0.29 | 5.44 |
| ETAD-0205 | 12 | 10.18 → 8.45 | −0.24 | 4.77 |
| ETAD-0194 | 22 | 9.16 → 7.94 | −0.17 | 4.41 |
| AUMN-0012 | 16 | 2.83 → 2.00 | −0.12 | — |
| ETAD-0162 | 12 | 8.62 → 7.77 | −0.12 | 4.49 |
| ETAD-0004 | 12 | 6.51 → 5.72 | −0.11 | 4.08 |
| AUMN-0011 | 13 | 2.42 → 1.69 | −0.10 | — |
| ETAD-0056 | 13 | 10.22 → 9.50 | −0.10 | 4.13 |

These ten largest changes are all premature-stop spectra and all move down. Seven of the ten are Addis, up to 23 % lower
on that filter. Addis has 7 of its 268 PM2.5 scans above 0.1 µg/m³, none above 0.5.

**Independent checks.** None of these depends on the ELBO.

| Check | Saved | Newton | Reading |
|---|---|---|---|
| Repeat scans, same filter: median \|EC difference\|, 196 pairs (µg on filter) | 0.059 | 0.057 | Negligible |
| Repeat PM pairs with a changed scan (17) | | better on 16, worse on 1 | Direction favours Newton; median gain only 0.001 µg |
| Field blanks: median EC (true ≈ 0) | 1.269 µg | 1.269 µg | No change; the 1.27 µg is the model's offset, not the baseline |
| 135 changed field blanks: closer to zero | 42 | 93 | Favours Newton by 0.0004 µg |
| 75 changed filters vs HIPS: median \|error\|, unpaired | 0.488 | 0.457 | Mixed |
| Same 75, RMSE (µg/m³) | 1.61 | 1.67 | Slightly worse |
| Same 75, paired: which is closer to HIPS | 50 | 25 | Favours saved, by a median 0.010 µg/m³ |

So **Newton does not improve EC**, and it's slightly further from HIPS on those filters. That doesn't
show the saved fits are better: HIPS is already above FTIR EC on 65 of the 75, and 52 of the 75 move
down, so almost any downward shift counts against Newton here. It does mean EC offers no argument for
switching.

Baselines moved a lot but EC did not, probably because the PLS model largely ignores the smooth
shape that differs between fits. I didn't test that directly. It also means analyses that use the
corrected spectra themselves may be more affected than EC. **I have not tested those**: functional
groups, the blank-bank and lot studies, continuum and transfer diagnostics.

## 3. WebGPU

**What constrains it.** WGSL defines [`f32` and an optional `f16`, and no 64-bit float](https://www.w3.org/TR/WGSL/);
`f64` was [discussed in 2022 as a possible optional extension](https://lists.w3.org/Archives/Public/public-gpu/2022Apr/0006.html)
that Metal-based implementations wouldn't offer. Single precision is therefore the design limit.
The [default limits](https://www.w3.org/TR/webgpu/) are enough: the largest matrix is 53 MB against
128 MiB per binding, and the shader uses a workgroup of 32 against a default of 128. My test
requested the adapter's larger limits, but this workload doesn't need them.

**How each stage copes with float32:**

| Stage in float32 | Result |
|---|---|
| Newton: Hessian only (the design used) | Identical optimum: sigma within 8e-8, ELBO within 1.3e-9, over the whole network |
| Newton: everything, simulated in numpy (316 spectra, including the extreme sigmas) | Sigma within 2.6e-4, baselines within 2.8e-5 of corrected RMS. Workable, but the objective can't be resolved near the optimum, some spectra ran to the 300-iteration limit, and the linear solve was still float64 in this simulation |
| MAP: system matrix used directly | **Fails.** Its weights span too many orders of magnitude |
| MAP: float32 matrix as preconditioner for float64 conjugate gradients | Works; every result keeps the float64 certificate |

**Design.** The GPU does the one heavy product, weights × Khatri–Rao matrix, in WGSL. JavaScript,
which uses float64 numbers, does the objective, gradient, Cholesky solves, conjugate gradients and
certificate. The code is in [`webgpu_demo/`](../research/ftir_hips_chem/webgpu_demo/).

**Measured on an Apple M4 Pro:**

| Run | Spectra | ELBO | MAP | Per spectrum | Agreement with float64 Newton + MAP |
|---|---|---|---|---|---|
| Chrome 156 (headless), JavaScript + WGSL | 256 | 7.4 s | 16.3 s | **0.09 s** | sigma 2.2e-8, ELBO 1.0e-9, baselines median 4e-7, max 2.1e-4; 256/256 converged and certified |
| Chrome 156, JavaScript only, no GPU | 128 | 45.8 s | 19.4 s | 0.51 s | sigma 1.3e-8; 128/128 |
| Python + WGSL through wgpu-py, whole network | 4,452 | 42 s | 226 s | 0.06 s | ELBO 4,452/4,452, sigma 8.3e-8; MAP **4,443/4,452 certified**, baselines max 2.1e-4 |

- The sample includes the typical, slowest, premature-stop and smallest-sigma spectra.
- JavaScript's `ndtr` port matches SciPy to 2.2e-16 in absolute terms.
- The browser runs shared the machine with the network run above, so treat them as upper limits.
- The 2.1e-4 maximum matches between the WebGPU and JavaScript-only runs, so it comes from the MAP
  stopping tolerance, not from float32.
- The 9 uncertified MAP fits are correctly rejected, not wrong. They would need a float64 retry,
  which the demo does not implement.

**What it is good for.** Running the correction in the gallery or an app on someone's own machine,
with no server and no Python. It is not a faster way to run the network: the WGSL product reaches
75–125 billion multiply-adds per second here, against ~960 billion for native BLAS on the same Mac.
On CuPy, the network takes seconds. In the browser the GPU only accounts for 1.5 of the 24 s, so the
next gain would come from Web Workers or WebAssembly on the JavaScript side.

**Not tested.** Only one GPU was tried. Firefox 153 exposed `navigator.gpu` but gave no adapter when
headless, and Safari wasn't tested. Reports on [browser support](https://webo360solutions.com/blog/webgpu-browser-support)
disagree about Firefox and Linux, so check them against real target machines. The worst case is a
low-end integrated GPU, where the JavaScript path at ~0.5 s per spectrum is the fallback.

## 4. Recommendation

**Adopt Newton as the method for new work, as a new versioned run. Keep the 19 Oct run frozen, and
leave reported EC unchanged for now.**

The case for moving is reproducibility and correctness, not EC:

- The saved run can only be reproduced on one software stack. On the Mac, the same code changes
  sigma by > 10 % on 15 % of spectra. Newton gives one answer on every machine tested.
- 19 % of saved fits are more than 0.1 % below the optimum of the objective we say we're maximising.
  155 stopped within 30 iterations. Newton also fits the 3 spectra the saved run failed on.
- It makes the sensitivity work practical: blank-bank and lot studies take seconds, not hours.

The case for not rushing:

- EC barely moves, so nothing reported is wrong in a way that matters today, and there's no reason
  to disturb numbers in use.
- Newton is a departure from the published procedure, even though it maximises the same objective.
  That needs saying openly, and it's a call for you, Ann and Satoshi.
- The calibration is the real gap. The frozen PLS model was trained on IMPROVE spectra corrected with
  SciPy, so applying it to Newton baselines mixes two optimisers.

**Suggested order:**

1. Refit the IMPROVE training spectra with Newton, rebuild the PLS model, and compare. This is the
   step most likely to change conclusions, and it is cheap now.
2. Rerun the baseline-sensitive analyses on Newton baselines: functional groups, blank banks, lots
   and continuum work. This is where the 273 changed baselines may matter.
3. If those hold up, publish `network_baselines` v2 next to the 19 Oct run, with a flag on the
   premature-stop filters. The 7 Addis scans deserve a look now, since they are the largest cluster.
4. Describe the change in the methods as "same objective, solved to convergence".
5. Tell the VIBES authors about the premature stops. `factr` stopping at iteration 12 with a
   gradient of ~1e4 affects anyone running the default settings.

Keep the vendored adapter available, on the same Colab stack, for exact reproduction of the 19 Oct
results. Use JAX only as a measurement of what compiling buys. It isn't a route to the reference.

## Reproduce

```bash
python -m pytest -q tests/test_vibes_batch.py tests/test_vibes_map_batch.py
python research/ftir_hips_chem/workflows/webgpu_demo.py prepare --bundle bundle_v2 --newton run_newton64/results.npz --out web/
python -m http.server 8765 --bind 127.0.0.1 --directory web/
python research/ftir_hips_chem/workflows/webgpu_demo.py run --mode webgpu --rows 256
```

`scripts/vibes_webgpu.py` is the wgpu-py version: pass `WebGpuGemm(W)` as `gemm` to `fit_elbo_newton`
and `solve_map_batch`. Results are in `output/tables/vibes_gpu_bench_20261103/reference_webgpu_20261103/`:
`refdiff.csv`, `mac_adapter.csv`, `newton_ec_by_scan.csv`, `webgpu_py.npz`, and the browser logs.

## Limits

- The Mac rerun used the vendored objective and SciPy directly, on 440 random spectra, not the full
  adapter. The G4 rerun used the adapter, also on 440.
- EC uses the frozen model, trained on SciPy-corrected IMPROVE spectra.
- HIPS is known to compress at high loading, so it's a weak yardstick for the Addis filters.
- Only 23 repeat-scan pairs include a changed spectrum.
- Numerical and EC comparisons only; no corrected-spectrum science was evaluated.
