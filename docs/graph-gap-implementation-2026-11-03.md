# Notebook graph gaps: implementation and data status

The five requested areas now use the gallery's shared rendering, controls, loading, exclusion and provenance paths. This is a local implementation; nothing has been published. The five main areas are now populated from source data; see [Drive validation](drive-graph-validation.md) for execution evidence and remaining archive coverage.

## What works now, and what still needs data

| Area | Implemented | Available result / remaining input |
|---|---|---|
| ONA/CMA/DEMA | Shared smoothing kernels; aligned minute comparisons; ONA window ECDF; lag curves; common-minute summaries; daily means and agreement when coverage permits. | Populated from the complete **1,095,086-minute Jacros record**, including daily agreement. Input extremes and negative counts are visible. |
| Full CV curves | Added the missing 2×2 grouping/fold-count save step and view; saved June variant curves and SPARTAN repeated-fold curves; connected those CSVs to the existing ML exporter and shared figures. | Full phase-3 curves are populated: 720 setup/preprocessing, 360 protocol-selection and 360 fold/grouping comparison rows. Older June/repeated-CV arrays still need their original exports. |
| Per-filter predictions/residuals | Added optical-agreement and residual-versus-loading views from the existing calibration-run predictions, selectable by calibration and test site. Added reusable D²/Q diagnostics, per-filter IDs and source save steps. | Native reruns populated `domain_locked` and `residuals_by_mode` for all 239 Addis filters. MAC changes move the optical reference and residual, while model predictions remain unchanged. |
| Charcoal PCA/distances/envelopes | Shared scatter with negative PCA axes; distance ECDF; raw/derivative mean and 5–95 % envelopes; per-filter envelope escape. Added save steps for PCA scores, distances, linkage and envelopes. | The native archive and corrected Addis spectra were downloaded and notebooks 02/04/06/12 executed. PCA, distance, raw/derivative envelopes and per-filter escape now render real results. |
| EC/OC VIP and uncertainty distributions | Kept the existing shared VIP renderer and made the EC/OC pair selectable even when unavailable. Added bootstrap-intercept and null-refit ECDF views; saved the null draws previously kept only in memory. | EC/OC profiles were reconstructed and checked against the R fits. The exports contain 1,200 bootstrap and 400 null-refit records, alongside the existing HIPS profiles. |

## Shared components and invariants

- [Figure](../gallery/app/src/components/Figure.tsx) now supports **scatter** and **ECDF**, delegating to `ScatterPlot` and `SweepPlot`. PCA uses general scatter ranges so negative scores are retained; it does not inherit a comparison plot's origin or 1:1 line.
- [SavedFigure](../gallery/app/src/components/SavedFigure.tsx) centralizes saved-table availability, physical filter IDs, MediaId joins, exclusions and source notes. Known filters follow the existing exclusion rules before statistics; point marks continue through the shared scatter layer.
- [PredictionDiagnostics](../gallery/app/src/components/PredictionDiagnostics.tsx) composes agreement, loading-residual and D²/Q views. Its [calculation helper](../gallery/app/src/lib/predictionDiagnostics.ts) re-expresses residuals from each filter's Fabs rather than scaling predictions.
- ECDF calculations live in [lib/stats](../gallery/app/src/lib/stats.ts), including tied-value jumps and finite-value handling. Controls use the existing URL-backed option machinery.
- [Legend](../gallery/app/src/components/Legend.tsx) accepts an always-visible measurement note. Explicit mean / 5–95 % envelope labels remain visible with Explain off; existing median/IQR defaults remain unchanged.
- [SweepPlot](../gallery/app/src/components/SweepPlot.tsx) uses a [shared time formatter](../gallery/app/src/lib/timeTicks.ts) for hour/minute labels on short records and month/year labels on long histories.
- [Smoothing kernels](../src/data/qc/smoothing.py) are shared by the comparison exporter and existing DEMA consumers. ONA and all other comparison methods restart at gaps/tape/session boundaries; missing observations do not become zero-valued measurements.
- [Page.optional / Page.retained](../gallery/data/export_research_spectral.py) preserve previously exported optional results when their gitignored source CSV is unavailable, retaining the original source date. An available source CSV takes precedence. Available native CSVs now take precedence; previously exported optional results remain available when their source CSVs are absent.

## Where to open the views locally

With `npm run dev` in `gallery/app`, append these paths to the local server address:

- `/?aq.topic=smoothing#tab=aethalometer&aep=quality&net=focus`
- `/#tab=ml&aep=components&net=focus`
- `/#tab=ml&aep=protocols&net=focus`
- `/#tab=ml&aep=june&net=focus`
- `/#tab=ml&aep=charcoal&net=focus`
- `/?ml-model-form-vip-profiles.set=ec-oc#tab=ml&aep=model-form&net=focus`

Research and ML remain private in public builds; Instruments remains unlisted. No visibility settings changed.

## Populate the missing results

The supplied Davis Data folder and connected research archive provided the native inputs for the main reruns. The direct-download route transferred large raw files beyond the connector limit. Input receipts, scientific reconciliation and remaining inputs are documented in [Drive validation](drive-graph-validation.md).

Full Jacros comparison:

```bash
uv run aeth doctor
uv run python gallery/data/export_processing_comparison.py --minute PATH_TO_JACROS_MINUTE_CSV
```

The optional short sample is reproducible, when the sibling older repository is present, with:

```bash
uv run python gallery/data/export_processing_comparison.py --sample
```

After the native data are available, regenerate the affected research notebooks with their existing builders. The added save steps are in:

- `research/ftir_ec_phase3/scripts/run_ftir_15.py`, `run_ftir_22.py`, `run_ftir_32.py`, `run_ftir_43.py`.
- `research/ftir_etad_char/scripts/run_char_02.py`, `run_char_04.py`, `run_char_12.py`, using `diagnostic_exports.py` for aligned array tables.
- `research/ftir_ec_calibration_2026_06_25/_build_09_component_selection_fixed.py`.
- `research/spartan_ec_2026_06_16/_build_07_calibration_optimization_story.py`, `_build_08_removal_explainer.py`.

For phase-3 / charcoal scripts, their `scripts/build_notebooks.py NUMBER` regenerates **and executes** the notebook. For June/SPARTAN `_build_*.py` scripts, regenerate the notebook and then execute it. EC/OC VIP already has a save step in `ftir_07_current_pls_vip_diagnostics.ipynb`. Source-input requirements and exact commands remain in each generated ML export's `missing` list and the page's Explain panel.

Then refresh the relevant gallery bundles:

```bash
uv run python gallery/data/export_ml.py components protocols charcoal residual model-form june
```

The processing exporter is registered in the shared rebuild plan as `aeth-processing`; its normal rebuild requires the canonical full minute file and leaves the existing export alone when that input is unavailable.

## Verification

See [Drive validation](drive-graph-validation.md) for source sizes, executed notebook coverage, numerical reconciliation, visual checks and remaining archive inputs. The primary views now use real native results. Separate synthetic browser fixtures exercise edge and unavailable-data states without entering the scientific exports.
