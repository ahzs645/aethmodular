# Ann weekly follow-up, 10 September 2026

The expanded presentation is **`output/ann_weekly_2026-09-10_additional_analyses.pptx`**. It adds six diagnostic figures as slides 13–18: paired model comparisons with month-bootstrap intervals, cross-season transfer, monthly discrepancies, analog-selection stability, spectral-shape PCA and reconstruction residuals, and source-site concentration. Each added slide has a spoken talk track, supporting detail and source paths. The first 12 slides retain the shorter weekly presentation.

The additional calculations are explained in **`output/diagnostics_report.md`**. Their active notebook is `research/ftir_hips_chem/ann_weekly_20260910_diagnostics.ipynb`; its executed copy is `research/ftir_hips_chem/notebooks/archive/executed/ann_weekly_20260910_diagnostics.ipynb`. Tables and figures use the `ann_weekly_20260910_diagnostics` output directories. The new helpers are `scripts/analog_diagnostics.py` and `scripts/plotting/ann_weekly_diagnostics.py` within the active area. Builders are `create_ann_weekly_diagnostics_notebook.py`, `prepare_ann_weekly_diagnostics.py` and `build_ann_weekly_diagnostics_deck.mjs`. The 12 targeted diagnostic and analog-selection tests passed. All comparisons remain exploratory; the proposed Addis validation split remains unscored.

The shorter presentation is **`output/ann_weekly_2026-09-10_notebook_plots.pptx`**, following the request to generate graphs in Jupyter and place their PNG images in the slides. It has 12 slides, 11 figures, and a spoken talk track, supporting detail, and sources in every slide's notes. Slides 10–12 show every individual training and Addis spectrum for each season. No calibrations were refit for these presentation revisions.

The active notebook is `research/ftir_hips_chem/ann_weekly_20260910_figures.ipynb`; the executed copy with all 11 displayed plots is `research/ftir_hips_chem/notebooks/archive/executed/ann_weekly_20260910_figures.ipynb`. Run the active notebook from its directory using the repository's uv environment. PNGs are in `research/ftir_hips_chem/output/plots/ann_weekly_20260910_notebook/`. Reusable figure functions are in `scripts/plotting/ann_weekly_figures.py` within the active analysis area.

To regenerate the notebook and images, run `uv run python research/ftir_hips_chem/workflows/create_ann_weekly_figure_notebook.py` from the repository root. `build_ann_weekly_notebook_deck.mjs` embeds those images and uses the speaker-note content prepared by `prepare_ann_weekly_visual_revision.py`. The earlier editable-chart revision, `output/ann_weekly_2026-09-10_visual.pptx`, is retained as an intermediate version.

The finished PowerPoint is in `output/ann_weekly_2026-09-10.pptx`: nine main slides and three spectral appendix slides, with speaker notes. `output/analysis_report.md` contains the task-by-task results, complete spectral plots, assumptions and links to the exact tables.

Analysis outputs are under `research/ftir_hips_chem/output/{tables,plots}/ann_weekly_20260910/`. The parallel `_mean_sensitivity` directories retain the alternative aggregation run. The Addis split is a proposed retrospective design and has not been scored.

Builders live in `research/ftir_hips_chem/workflows/`:

- `run_ann_weekly_20260910.py`: masked analog selection, 22 fits, bootstrap intervals and figures.
- `summarize_ann_weekly_20260910.py`: exact meeting-figure membership, raw Bishoftu lot audit, lambda sensitivity and report.
- `prepare_ann_weekly_slides_20260910.py`: data-derived presentation content.
- `build_ann_weekly_slides_20260910.mjs`: native editable PowerPoint, package validation and rendering.

Run the analysis commands shown in the report from the repository root. To build the deck using the bundled presentation runtime:

```sh
uv run python research/ftir_hips_chem/workflows/prepare_ann_weekly_slides_20260910.py
RUNTIME_NODE_MODULES=/Users/ahmadjalil/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules AETHMODULAR_WEEKLY_DECK_NAME=ann_weekly_2026-09-10_v2.pptx /Users/ahmadjalil/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node research/ftir_hips_chem/workflows/build_ann_weekly_slides_20260910.mjs
```

Use a fresh output filename for revisions; finalization intentionally refuses to overwrite a finished presentation. `.build/` is private build material and is ignored by Git.

Validation on the expanded notebook-image version: the diagnostic notebook executed with six PNG outputs and no cell errors; all six new final slides were visually reviewed and the original 12 renders remained byte-identical to the previously reviewed deck. Package integrity, slide geometry, fonts and first-party import passed. The PPTX contains 17 figure images, no native charts, and complete notes on all 18 slides. The original analysis passed 32 targeted analysis/regression/transfer tests; the additional diagnostics and analog-selection tests passed 12 checks. This does not claim native Microsoft PowerPoint rendering was tested.
