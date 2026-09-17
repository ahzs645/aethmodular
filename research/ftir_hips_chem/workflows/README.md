# Workflows

One-off report, deck, and board-asset builders live here. These scripts may
call reusable logic from `../scripts/`, but they are not part of the importable
analysis API.

`audit_matched_samples.py` reproduces the optical/filter readiness audit:

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/audit_matched_samples.py
```

Outputs live under `../output/tables/matched_sample_audit/`: a catalog keyed by
site/base FilterId, preserved measurement rows, legacy date-window candidate
links, a selection report, and a manifest with input/code/output hashes and
environment versions. Available current SPARTAN portal exports additionally
restore reported collection windows, sampled hours and source conditions. They
resolve via `--portal-dir`, `AETHMODULAR_PORTAL_DIR`, a local copy, or the
existing Drive resolver; `--skip-portal` runs from local pickles alone.
Candidates are explicitly unvalidated. See
[`docs/matched-sample-audit-2026-09-10.md`](../../../docs/matched-sample-audit-2026-09-10.md)
for findings and the remaining requirements for interval matching.

`create_matched_sample_audit_notebook.py` builds and executes eight audit figures:

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/create_matched_sample_audit_notebook.py
```

The active source is `../matched_sample_audit_figures.ipynb`; its executed copy
is under `../notebooks/archive/executed/`. Reusable plotting code lives in
`../scripts/plotting/matched_sample_audit.py`. Figures (PNG and SVG) are saved
under `../output/plots/matched_sample_audit/`, with the plotted rows, summaries,
and figure/input hashes under `../output/tables/matched_sample_audit_figures/`.
The notebook checks every loaded input against the original audit manifest.
It preserves registry and timing flags and fits no calibration or regression.

`build_matched_sample_audit_notebook_deck.mjs` places these notebook figures in
slides 10–17 of a copy of `ann_weekly_with_audit.pptx`, retaining its original
results and spectral appendix. Run it with the bundled presentation Node
runtime and `RUNTIME_NODE_MODULES`, `AETH_PRESENTATIONS_SKILL_DIR`, and
`AETH_RUNTIME_PYTHON` set to the installed runtime/skill locations.
`AETH_AUDIT_DECK_NAME` selects a fresh output filename for subsequent revisions.
Detailed methods and caveats come from the executed notebook into speaker notes.

Keep reusable analysis functions in `../scripts/`. Move code from workflows
into `../scripts/` only when more than one notebook or workflow needs it.

`build_active_interval_matches.py` extends the audit into evidence-gated active
sampling periods, EC provenance and timestamped instrument inputs:

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/build_active_interval_matches.py
```

Outputs live under `../output/tables/active_interval_matches/`. The current
build preserves unresolved schedules and observation histories; its typed
active-interval table is empty until reviewed source evidence is supplied.
`--sampling-intervals` accepts documented periods and `--processing-evidence`
accepts reviewed stream contracts. No calibration is fitted. See
[`docs/active-interval-matching-2026-09-10.md`](../../../docs/active-interval-matching-2026-09-10.md)
for findings, schemas, coverage policy and the remaining evidence requirements.

`analyze_filter_diagnostics.py` produces the scientific filter-only report from
the frozen 545 diagnostic / 480 ratio cohorts. It writes six figure families,
site and denominator-sensitivity tables, a recovery queue, and the earlier
ChemSpec export-to-unified trace. `create_filter_diagnostics_notebook.py`
regenerates these outputs in an executed notebook with figure explanations.

```bash
uv run python research/ftir_hips_chem/workflows/analyze_filter_diagnostics.py
uv run python research/ftir_hips_chem/workflows/create_filter_diagnostics_notebook.py
```

See [`docs/filter-only-results-2026-09-10.md`](../../../docs/filter-only-results-2026-09-10.md).
Filter-only outputs use the frozen interval-build decisions without changing
their sampling evidence. Reviewed stream assertions now require explicit
time/session scope, and the interval builder exports descriptive quarter
coverage without adding an exclusion rule.
