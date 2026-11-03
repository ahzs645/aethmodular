# VIBES: gallery comparison and continuum check

The Baselines tab now connects a bounded signal-preservation experiment to the gallery's shared figures. The existing four-band injection check did not establish preservation of a broad continuum. The new controls test that question directly, without changing the background or selecting a model against HIPS.

## What was compared

The frozen full-range background has rank 70 and was trained on 87 blanks. Its original case ledger identifies nine Addis field blanks held out from that training. Their physical IDs reconcile with the native FTIR metadata; none overlaps the training blank IDs. These blanks have been inspected in earlier diagnostics, so this is an exploratory control experiment, not a new confirmatory test population.

For each parent blank, the original VIBES adapter and the validated spline baseline were run on the unspiked spectrum and on nine additions: three shapes at scales 0.01, 0.05 and 0.15 AU. This gives **90 fits per method**. Repeated additions do not increase the number of independent parent blanks beyond nine.

- **Four diagnostic bands:** the existing Gaussian additions at 1610, 1720, 2920 and 3400 cm⁻¹.
- **Broad hump:** a Gaussian centred at 2100 cm⁻¹ with width σ = 850 cm⁻¹.
- **Power-law continuum:** `scale × wavenumber / 2000`, equivalent to a wavelength⁻¹ shape, with the specified scale at 2000 cm⁻¹.

Recovery is `corrected(blank + addition) − corrected(blank)`. Signed integrated area recovery is 1 for full retention, 0 for complete removal, and negative for inversion. Numerical failures make a whole window unavailable; failed channels are not silently omitted. The spline has no full-grid result below 1425 cm⁻¹. Comparisons between both methods use the shared window, approximately 1426–3998 cm⁻¹.

## Observed recovery

At addition scale **0.05 AU**, median area recovery across the **nine independent blanks** was:

| Known addition | VIBES | Spline baseline |
|---|---:|---:|
| Four diagnostic bands | 98.26% | 84.32% |
| Broad hump | 18.92% | 0.95% |
| Power-law continuum | 0.139% | approximately 0% |

The VIBES adapter's ELBO-convergence and finite-output checks passed **90/90**, with **30 retries**. Convergence and signal preservation are different outcomes: this configuration largely removed the broad power-law addition even though fitting succeeded. Broad-hump recovery also depended strongly on scale, increasing from about 0.38% at 0.01 AU to 40.61% at 0.15 AU.

These synthetic additions probe what the correction can retain; they do not identify real EC absorption or establish target-site EC accuracy. No acceptance threshold was provided, and no model or background was retuned to improve these results.

An additional MAP audit replayed the saved pinball objectives with their saved ELBO parameters: **90/90 reported optimal** in CLARABEL. Every VIBES recovery metric reproduced, with maximum difference **7.64 × 10⁻¹³**, below the 10⁻⁸ numerical tolerance. This did not rerun ELBO or change the correction settings.

## Gallery changes and connections

Open **Calibration → Baselines**, or `/#tab=baseline&net=focus` in the local app.

- One set of URL-backed **signal and window controls** drives recovery curves, individual-filter error points, blank-residual distributions and spectral overlays.
- `Figure`, `SavedFigure`, `SpectrumPlot`, `ScatterPlot`, `SweepPlot`, `Legend`, `lib/data` and the shared exclusion/identity helpers provide the rendering and behaviour.
- Band definitions and essential scope notes stay visible with Explain off. Recovery curves show medians and interquartile ranges across independent parents at each scale.
- The displayed spectrum is explicitly the first parent blank in the frozen ledger. Adjacent groups of three channels are averaged for display; recovery metrics use every native channel.
- The original solver log used row numbers as its `sample_id`. A checked `case_row` join now resolves those rows to physical filter IDs and rejects conflicting labels.
- The original log's two failed blanks are labelled **before recovery**. That log, the existing recovered benchmark, the new local controls and the separately reported G4 job remain distinct.
- The independent check renders even when an older benchmark bundle is missing. Search now discovers declarative `Figure`/`SavedFigure` call sites, including all seven new Baselines panels.

## Timing and the quoted Colab job

The local control experiment used **four workers** on an environment reporting **five logical CPUs**. Its 90 VIBES fits took **307.37 seconds** including worker startup, or **17.57 successful fits/minute**. The spline's supported-window fits took **4.75 seconds**. These measurements describe this particular full-grid, rank-70, retry-enabled control experiment.

The separately quoted batch of 120 fits in 15 seconds implies **480 fits/minute** and approximately **67.3 elapsed minutes** for 32,286 fits at the same sustained rate. A per-fit duration of 0.75 seconds instead implies **6.73 single-worker hours**; it is not measured CPU time unless that was timed explicitly. Grid size, background rank, optimizer settings, retries, worker count and timing boundaries must match before comparing throughput. Reported core counts alone do not establish scaling to the G4 machine.

The repository's pinball MAP implementation uses cvxpy's **CLARABEL CPU solver**. No G4 result file or session identity has been supplied for the new job. Its benchmark and 32,286-fit completion are therefore not represented as measured gallery results. No VM was rented or modified during this work.

## Reproduction and evidence

The numerical helpers are in `research/ftir_hips_chem/scripts/vibes_validation.py`. Run the bounded check with the repository environment and existing native FTIR path settings:

```bash
uv run --extra vibes python research/ftir_hips_chem/workflows/check_vibes_continuum.py --reference-dir DIR --workers 4
uv run --extra vibes python research/ftir_hips_chem/workflows/audit_vibes_map.py --reference-dir DIR
uv run --extra vibes aeth gallery rebuild --only vibes-validation
```

`DIR` contains the original `background_model.npz`, `wn.npy`, `cases.csv`, `fit_diagnostics.csv` and `RUN_MANIFEST.json`. Source files remain unchanged. Output CSVs, fit diagnostics, runtime, MAP audit and checksums are under `research/ftir_hips_chem/output/tables/vibes_continuum_check/`. The exporter verifies its result checksums before updating `gallery/app/public/data/baseline_validation.json`.

The generated gallery bundle is local and gitignored, following the existing baseline-export convention. No publication or visibility settings changed. Older rank-70 model, seasonal and similarity bundles absent from this checkout remain separate missing assets; this change does not manufacture their model scores.

Verification covers numerical preservation/erasure/inversion controls, missing channels, CSV boolean parsing, row-to-filter identity joins, upstream solver parity, held-out isolation and rebuild registration. The production build, 29 targeted Python checks, 9 targeted Node checks and 9 browser checks pass. The two VIBES browser checks also pass after the final solver-audit display update. Browser checks cover native plots, linked controls and copied URLs, visible interval/scope notes, mobile layout, and rendering when the old benchmark is absent. The private-data public-build/archive check passes. The separately known broad Node failures caused by absent legacy bundles remain outside this targeted pass.
