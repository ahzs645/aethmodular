# VIBES: what the rank-70 background absorbs, and the solver in WebAssembly

Date: 10 November 2026. Follows [vibes-newton-adoption-tests-2026-11-03.md](vibes-newton-adoption-tests-2026-11-03.md)
and [vibes-reference-webgpu-decision-2026-11-03.md](vibes-reference-webgpu-decision-2026-11-03.md). The numbers are in
[`vibes-rank-check-2026-11-10.json`](vibes-rank-check-2026-11-10.json). Input: the 19 Oct network bundle (4,452 scans,
the 87-blank rank-70 background, tau 0.1), every scan solved to convergence.

## Summary

| Question | Answer (measured) |
|---|---|
| Are the 155 flagged fits a sign that rank 70 is too high? | **No.** They are early stops of the original optimiser (3 Nov notes): 100 PM2.5, 14 PM10 and 41 field blanks. Their noise level is too high and the baseline is under-fitted. Solving to convergence removes them. |
| Does the background absorb the aerosol on light filters? | **No case found.** 108 sample scans have a noise level as low as a field blank's. All 108 have raw spectra with no aerosol bands, measured without any baseline model. 10 of them are flat scans. |
| Does the rank matter? | **Not on loaded filters.** An injected band is recovered the same at ranks 20, 40 and 70 on typical and Addis filters. It matters only for a faint, wide band on a blank. |
| What does the background absorb? | **Breadth, at every rank.** A 1,500 cm⁻¹ wide feature keeps 0 to 14 % of its height. This is the continuum loss again, and lowering the rank does not fix it. |
| Were the SPARTAN field blanks used to build the background? | **Only Addis ones.** The bank is 59 IMPROVE blanks and 28 Addis field blanks (30 of the 545 field-blank scans). The other 515 fit almost as tightly: median noise level 1.6e-6 against 0.9e-6. |
| Can the converged solver run in a browser as WebAssembly? | **Yes.** A Rust port, 80 KB with no dependencies, matches the Python solver to 9e-10 in noise level and 8e-14 in baseline. In Chromium it takes 0.33 s per spectrum on one worker, 3.8 times faster than the existing JavaScript path on the same machine. |
| Is it in the gallery? | **Yes**, as Experimental → VIBES on one spectrum. It solves any network scan or a file you choose, at a chosen rank and tau, and draws it against the saved 19 Oct trace. It needs the background export in the data release. |

## 1. A correction

A reply earlier on 10 Nov read one light Pasadena scan (USPA-0229-5) as a filter whose spectrum the rank-70 background
had absorbed, aerosol included, and linked that to the 155 flagged fits. Both parts were wrong.

- The 155 are the early stops documented on 3 Nov. They err the other way: the fit stops with the noise level 10 to
  1,000 times too high.
- USPA-0229-5 has no aerosol bands in its raw spectrum. Its C-H, ammonium and carbonyl heights sit inside the
  field-blank range. The low noise level is the correct fit to a scan that carries no deposit signal.

## 2. Scans with a noise level as low as a blank

The fitted noise level is the scale of what the baseline leaves behind. On a field blank that is instrument noise: the
median is 1.6e-6, against 6.1e-4 for PM2.5 scans.

| Filter type | Scans | Median noise level | Median corrected RMS | Share of corrected RMS carried by components 41 to 70 |
|---|---|---|---|---|
| Field blank | 545 | 1.6e-6 | 2.9e-5 | 39 % |
| PM10 | 222 | 1.9e-4 | 3.1e-3 | 0.08 % |
| PM2.5 | 3,685 | 6.1e-4 | 9.6e-3 | 0.008 % |

Being in the bank is not what makes a blank fit tightly. The bank holds 59 IMPROVE blanks and 28 Addis field blanks, which
are 30 of the 545 field-blank scans here. Their median noise level is 0.9e-6; the 515 scans outside the bank have 1.6e-6.
An earlier version of this note said all 87 were IMPROVE blanks. That check compared the wrong identifiers.

Components 41 to 70 are tiny (2e-6 to 5e-7 in absorbance at unit weight, against 0.027 for the first). On sample scans
they carry under a hundredth of a percent of the corrected signal. They only do work on blanks.

108 sample scans (103 PM2.5, 5 PM10) have a noise level below the field blanks' 95th percentile. If the baseline had
absorbed their aerosol, the raw spectra would still show bands. They do not. Heights are measured above a straight line
between each band's edges, so no baseline model is involved.

| Group | Scans | C-H 2920 | Ammonium 1420 | Carbonyl 1720 |
|---|---|---|---|---|
| Field blank | 545 | 0.00004 | 0.0015 | 0.0002 |
| Sample scan, noise as low as a blank | 108 | 0.00007 | 0.0015 | 0.0002 |
| Other sample scan | 3,799 | 0.0018 | 0.017 | 0.0034 |

All 108 sit inside the field blanks' 99th percentile on every band. They come in runs of consecutive cartridge positions
(ZAJB-0075 to 0080, ILHA-0195 to 0200, CASH-0321 to 0326), which looks like cartridges that were not sampled. The bundle
has no volumes, so that is a reading, not a finding.

On the 1,287 sample scans with a clear raw ammonium band, the corrected spectrum keeps a median 96 % of its height
(5th percentile 92 %). On the 1,244 with a clear C-H band it keeps 87 % (5th percentile 71 %). The straight-line height
includes any curved background under the band, so these are lower bounds on what the baseline keeps.

**Flat scans.** 11 scans span less than 0.05 in absorbance; the next-smallest span in the network is 0.17 and a normal
scan spans about 0.4. Ten span under 0.006 and are among the 108 above. The eleventh, KRSE-0136-8 (PM10), spans 0.05 and
its noise level is above the blank range. Three of the 11 carry the QC code that normal scans carry (1): ETAD-0037-5
(analysis 406), PRFJ-0073-1 (530) and KRSE-0136-8 (470). The bundle does not define its QC codes, so this is a list to
check. ETAD-0037-5 has no second scan; PRFJ-0073-1 has one. ETAD-0037 and ETAD-0139 are already open items.

## 3. Injection by rank and width

A Gaussian band was added to 160 real scans (40 per group) at 1720 and 2920 cm⁻¹, five widths and two heights, and each
was refitted at ranks 70, 40 and 20. The table is the median share of the band left in the corrected spectrum. Every one
of the 10,080 fits converged and was certified.

Band height 0.001 (faint):

| Group | Rank | 15 cm⁻¹ | 50 | 150 | 500 | 1,500 |
|---|---|---|---|---|---|---|
| Addis PM2.5 | 70 | 1.00 | 1.00 | 1.00 | 0.88 | 0.07 |
| | 20 | 1.00 | 1.00 | 1.00 | 0.89 | 0.08 |
| Typical PM2.5 | 70 | 1.00 | 1.00 | 1.00 | 0.63 | 0.03 |
| | 20 | 1.00 | 1.00 | 1.00 | 0.65 | 0.02 |
| Field blank | 70 | 0.99 | 0.96 | 0.71 | 0.13 | 0.00 |
| | 40 | 1.00 | 0.97 | 0.92 | 0.25 | 0.00 |
| | 20 | 1.00 | 0.99 | 0.98 | 0.60 | 0.01 |

Band height 0.01:

| Group | Rank | 15 cm⁻¹ | 50 | 150 | 500 | 1,500 |
|---|---|---|---|---|---|---|
| Addis PM2.5 | 70 | 1.00 | 1.00 | 1.01 | 0.95 | 0.14 |
| Typical PM2.5 | 70 | 1.00 | 1.00 | 1.00 | 0.89 | 0.12 |
| Field blank | 70 | 1.00 | 0.99 | 0.99 | 0.72 | 0.01 |
| | 20 | 1.00 | 1.00 | 1.00 | 0.80 | 0.04 |

- Bands up to 150 cm⁻¹ wide survive on loaded filters at every rank.
- Rank changes nothing on loaded filters. It costs up to 29 % of a faint 150 cm⁻¹ band on a blank.
- A 1,500 cm⁻¹ wide feature is removed at every rank. The first 20 components already span it. The EC continuum is that
  kind of feature, which is why the fix was a different background (PTFE-anchored), not a lower rank.

## 4. The solver in WebAssembly

[`wasm_demo/`](../research/ftir_hips_chem/wasm_demo/) is a Rust port of `fit_elbo_newton` and `solve_map_batch`: Newton on
the ELBO with the exact Hessian, then the interior-point MAP with its duality-gap certificate. Everything is float64, so
it needs none of the float32 workarounds of the WebGPU version. The module is 80 KB and has no dependencies.

Agreement with the Python float64 solver, on 64 random network scans:

| Check | Largest difference |
|---|---|
| Noise level, relative | 8.8e-10 |
| ELBO, relative | 1.8e-14 |
| Baseline, absorbance | 8.1e-14 |
| Converged and certified | 64 of 64 |

Speed on the same 64 scans, on a 4-vCPU cloud machine (not the M4 Pro of the 3 Nov tests):

| Run | Per spectrum |
|---|---|
| Existing JavaScript solver, no GPU (Node 22) | 1.06 s |
| WebAssembly, one thread (Node 22) | 0.28 s |
| WebAssembly in Chromium, 1 worker | 0.33 s |
| WebAssembly in Chromium, 2 workers | 0.20 s of wall time |
| WebAssembly in Chromium, 4 workers | 0.16 s of wall time |

- Each worker holds its own copy of the solver and shares no memory, so the page needs no cross-origin isolation headers
  and works from a static host.
- The build uses WebAssembly SIMD, which gives 1.5 times over the plain build.
- Scaling stops at 2 times here because the machine has two physical cores.
- The 3 Nov WebGPU version took 0.09 s per spectrum on the M4 Pro, where the JavaScript path took 0.51 s. WebAssembly
  was not run on that machine. If the 3.8 times ratio holds there, one worker takes about 0.13 s, and several workers
  would pass the WebGPU figure. That is an estimate.

**First fit in a page.** A browser first runs WebAssembly in a quick-to-compile tier and swaps in optimised code for a
function only between calls to it. With the whole fit inlined into one call, the first fit in a page took 1.0 s. The
loops that run once per iteration are now separate functions, so they are optimised after the first iteration: the
first fit takes 0.35 s, and results are unchanged to the last digit shown above.

**What it is good for.** Correcting one spectrum, or changing the rank or tau, with no server. It is not the way to run
the network: Python does all 4,452 scans in about three minutes on four cores.

## 5. In the gallery

Experimental → VIBES on one spectrum (`#tab=experimental&xp=vibes-tool`).

- **Spectrum.** Any scan in the gallery's native SPARTAN archive (site, filter, scan), or a text file with two columns,
  wavenumber then absorbance. A file is read in the browser and is not uploaded.
- **Preparation.** The spectrum is averaged onto the background's 2,722 channels by the rule the network run used
  (`resample` in `network_baselines_20261019.py`, ported and tested against it).
- **Settings.** Rank 10, 20, 40 or 70 (the first components of the 87-blank background), tau 0.05, 0.1 or 0.2, and the
  noise level: solved to convergence here, or the value the saved run stopped at for that scan.
- **Presets.** *Converged* is rank 70, tau 0.1 and the noise level solved here (the 3 Nov method). *19 Oct run* is the
  same rank and tau with the saved run's noise level, which repeats the saved trace. Changing a setting shows *Custom*.
  The 19 Oct preset is offered for network scans saved under the 87-blank background (4,373 of 4,452); the 76 lot 241
  and 241a scans used their own blank banks and the 3 failed fits have no noise level.
- **Result.** The corrected spectrum beside the saved 19 Oct trace, or the raw spectrum with its baseline; the noise
  level, steps, certificate and solver time; a CSV download.
- **What each line was made with.** A table under the plot gives the background, rank, tau and noise level of both
  lines, how each noise level was set, and puts what differs in bold.

**Why a line solved here can differ from the saved one.** The background, rank and tau are the same. The noise level is
not: it is fitted per scan, and the saved run's optimiser stopped early on 155 scans. A noise level that is too high
holds the baseline close to the average blank, so the filter's own broad slope is left in the corrected spectrum.

| Check | ETAD-0053 (early stop) | ETAD-0001 (typical) |
|---|---|---|
| Saved noise level, optimiser steps | 4.9e-3, 13 | 7.9e-4, 1,142 |
| Noise level solved here | 4.2e-4 | 7.9e-4 |
| Converged preset against the saved trace | 0.015 (216 % of corrected RMS) | 2.6e-6 (under 0.1 %) |
| 19 Oct run preset against the saved trace | 4.2e-8 | — |
| Original pyvibes rerun here against the saved trace | 0.0 | 0.0 |

Solving the baseline at the saved noise level reproduces the gallery's saved trace on 361 scans checked in Python (300
random, 61 early stops): median difference 2e-8 in absorbance, largest 5e-6. The noise level solved here equals the
3 Nov converged run's on every 87-blank scan, to 4e-8 relative.

Where the pieces are:

| Piece | Path |
|---|---|
| Solver source (Rust) | `research/ftir_hips_chem/wasm_demo/src/lib.rs` |
| Built module and its build record | `gallery/app/src/lib/vibes/vibes_wasm.wasm`, `vibes_wasm.build.json` |
| Wrapper, worker, resampler, file parser, tests | `gallery/app/src/lib/vibes/` |
| Page | `gallery/app/src/pages/experimental/VibesTool.tsx` |
| Background and saved-run export | `gallery/data/export_vibes_background.py` → `public/data/experimental/vibes_background.json`, `.bin`, `vibes_saved_fits.json` |

The built module is committed, so the gallery builds without Rust. `tests/test_vibes_wasm.py` fails if the Rust source
changes without a rebuild (`wasm_demo.py build --gallery`).

The background (1.5 MB at full precision) and the saved-run records (0.2 MB) are committed, as an exception to the
Experimental ignore rule, so the page works on the site before a data release carries them. Rerun
`export_vibes_background.py` if the background or the saved runs change. Without `vibes_saved_fits.json` the page still
solves, but offers no 19 Oct preset and no settings table.

## Limits

- The bundle has no sampled volumes, so "not sampled" for the 98 intact blank-like scans is inferred from their spectra.
- About 2 of the 40 field blanks in the injection test are expected to be bank members; they were not separated.
- Injection used Gaussian bands at two positions. Real aerosol bands overlap and sit on PTFE structure.
- The rank was lowered by truncating the rank-70 background, not by refitting a bank at each rank.
- Browser timing is from one machine and one browser, headless. Firefox and Safari were not tested.
- The WebAssembly module fits one spectrum per call with a fixed tau; it does not estimate tau.
- The gallery page uses the 87-blank background only. Lot 241 and 241a filters were saved with their own blank banks, and the
  page says so instead of comparing like with like.
- The page was tested in headless Chromium at desktop and phone widths, not in Firefox or Safari.

## Reproduce

```bash
W=research/ftir_hips_chem/workflows
python $W/run_vibes_rank_check_20261110.py --bundle VIBES_BUNDLE --out OUT --summary docs/vibes-rank-check-2026-11-10.json

rustup target add wasm32-unknown-unknown
python $W/wasm_demo.py build --gallery        # --gallery also installs the module in gallery/app/src/lib/vibes
python $W/wasm_demo.py prepare --bundle VIBES_BUNDLE --out web/ --rows 64
python $W/wasm_demo.py check --dir web/
python -m http.server 8766 --bind 127.0.0.1 --directory web/
python $W/wasm_demo.py run --rows 64 --workers 4
python -m pytest -q tests/test_vibes_wasm.py

python gallery/data/export_vibes_background.py      # the background for the gallery page
(cd gallery/app && node --test src/lib/vibes/vibes.test.mjs)
```

`VIBES_BUNDLE` is `results/network_baselines_20261019/outputs/vibes_bundle` in the VIBES Drive folder.
