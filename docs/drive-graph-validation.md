# Drive data and graph validation

The five main graph gaps now have results computed from source inputs. The local gallery uses the existing shared charts, controls, filter exclusions and provenance. Nothing was published.

## Inputs and execution

Downloaded a selected **24-file, 1.83 GB** input set from Davis Data and the connected research archive: the complete Jacros minute CSV, FTIR model arrays and model exports, Addis spectra and metadata, the 13,634-spectrum IMPROVE pool, TOR/HIPS tables, and the six charcoal reference files. The supporting older HIPS backup came through the Drive readable-text connector and is stored as normalized UTF-8/LF text; its receipt identifies that transformation. Other inputs were downloaded as raw files and checked against their listed sizes.

Machine paths are configured in the gitignored `.env`. The Jacros file is linked into the canonical data tree, so the normal `aeth-processing` rebuild step can find it. Input sizes, local SHA-256 checksums and numerical reconciliation results are saved under `research/ftir_hips_chem/output/tables/drive_graph_validation/`.

Executed the existing repository notebooks top to bottom:

- FTIR phase 2: **07, 08, 09, 10**; executed copies are in `research/ftir_hips_chem/notebooks/archive/executed/`.
- FTIR phase 3: **11, 13, 15, 17, 18, 20, 21, 22, 23, 32, 43**.
- Charcoal: **02, 04, 06, 12**.

The validated shared spline-baseline implementation rebuilt the Addis and IMPROVE caches. Downloaded scripts were not executed.

## Results now available

| Area | Source-backed coverage |
|---|---|
| ONA / CMA / DEMA | **1,095,086 unique minute records**, April 2022–August 2024; trace, exact adaptive-window distributions, lag curves, input ranges, matched-minute summaries, daily means and daily agreement. Each channel has 1,094,754 common finite BCc observations. |
| Cross-validation | **720** component-curve rows for the six setups and preprocessing comparisons, **360** protocol-selection curve rows, and **360** rows for the grouped/interleaved × 5/10-fold comparison. |
| Per-filter diagnostics | **239 Addis filters**; 478 locked-model and 956 protocol-specific residual/domain records, retaining filter identities, Fabs, predictions, D² and Q. |
| Charcoal | 4,402 archive/projection PCA rows; 12,017 typology-score rows; 4,163 distance records; 6,988 raw/derivative envelope rows; 239 per-filter envelope-escape scores. The temperature-labelled PCA reference has 3,924 spectra; the broader typology archive is a different population. |
| VIP and uncertainty | EC/OC profiles at **2,722 wavenumbers**, **1,200 site-cluster bootstrap records**, and **400 null-refit records** (200 per pipeline). |

The reconstructed EC and OC models pass the existing checks against their saved R predictions: maximum differences are **0.0000508** and **0.0031970 µg/filter**, respectively, below the 0.02 tolerance. The rebuilt Addis predictions match the repository's tracked spline-baseline reference within **3.05 × 10⁻⁸ µg/m³** over all 239 filter IDs.

## Issues found and fixed

- The Jacros “cleaned” file still contains extreme Blue-channel values: approximately **−234,330 to +216,329 µg/m³**, while its 1st–99th percentiles are about −0.50 to 48.78. Input ranges and negative counts are now visible. No additional exclusions were invented; full-record RMSE is a processing-difference diagnostic and can be dominated by these values.
- Exact window-frequency counts now feed the shared ECDF renderer, preserving every window's contribution without a large repeated-value payload.
- PCA exports initially carried native spectral columns along with metadata. A shared metadata helper now keeps those spectra out of score/distance tables.
- Four-decimal export rounding flattened many small derivative values. The shared exporter now supports per-table full precision, including retention across rebuilds. Envelope values were reconciled against their CSVs.
- The shared legend's interval definition now stays visible with Explain off. Envelope controls expose the full range and the two diagnostic spectral windows.
- The phase-3 builder now supports later scripts with finalized prose, including notebook 43; its saved null draws are consumed by the shared exporter.

HIPS/MAC is an optical comparison, not an independent thermal EC reference at Addis. These reruns validate reproduction and graph fidelity, not EC accuracy at the target site.

## Remaining archive coverage

All main areas above are populated. The older June CAL-variant/repeated-shuffle CV arrays and cleaned-refit views still need their original `rds_EC_*` and Adama exports; no substitute cohort or row order was inferred from the newer model arrays. Other optional gaps remain explicitly listed in each page's export: hybrid-cohort per-filter predictions; charcoal baseline-fit traces and per-reference recovery; additional residual-attribution draws/coefficient spectra; and model-form per-filter arrays.

## Reproduce

With the data roots configured:

```bash
uv run aeth doctor
uv run aeth gallery rebuild --only aeth-processing
uv run python gallery/data/export_ml.py components protocols charcoal residual model-form
```

For scientific reruns, use each research folder's existing notebook builder. Phase-2 notebooks run with `nbclient` from `research/ftir_hips_chem`; spline caches are built by `research/ftir_ec_phase3/scripts/apply_airspec_baseline.py`.

The affected checks pass: **47 targeted Python tests**, **17 targeted Node tests**, **7 browser checks**, TypeScript/production build, and the private-data public-build/archive check. Browser coverage includes native data, mobile layout, shared MAC/URL settings, negative PCA scores, spectral window controls and visible interval definitions. Source PCA/CV/envelope plots and gallery screenshots were visually inspected. Separate synthetic fixtures exercise unavailable-data states and do not enter research results.

The broader Node suite retains the previously identified missing baseline/rank-70/similarity bundle failures; this work does not claim a clean full suite.
