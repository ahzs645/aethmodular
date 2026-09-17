# Filter-only scientific results — 10 September 2026

The substantive deliverables are a six-figure results report, an executed
Jupyter notebook, a ranked evidence-recovery queue and a source-to-output
ChemSpec trace. The frozen populations remain 545 diagnostic pairs and 480
ratio pairs. No instrument calibration or independent EC validation is claimed.

## Read the results

- [Full report with figures](../research/ftir_hips_chem/output/tables/filter_diagnostics/results_report.md)
- [Active notebook](../research/ftir_hips_chem/filter_only_diagnostics.ipynb)
- [Executed notebook](../research/ftir_hips_chem/notebooks/archive/executed/filter_only_diagnostics.ipynb)
- [Recovery queue](../research/ftir_hips_chem/output/tables/filter_diagnostics/evidence_recovery_queue.csv)
- [Source-to-output EC trace](../research/ftir_hips_chem/output/tables/filter_diagnostics/chemspec_source_to_unified.parquet)

| Site | Diagnostic pairs | Ratio pairs | Descriptive R² | Median ratio | Ratio IQR |
|---|---:|---:|---:|---:|---|
| Beijing | 163 | 150 | 0.543 | 10.15 | 7.84–11.99 |
| Delhi | 62 | 56 | 0.688 | 8.61 | 7.60–11.66 |
| JPL | 130 | 84 | 0.522 | 9.51 | 7.70–11.57 |
| Addis Ababa | 190 | 190 | 0.764 | 10.16 | 8.14–12.69 |

These ratios are HIPS in Mm⁻¹ divided by FTIR-predicted EC in µg m⁻³. Their
dimensions do not establish independently validated BC absorption efficiency.
The site distributions overlap substantially. OLS fits are unweighted,
within-site descriptions; measurement error and differing concentration ranges
limit comparisons of their slopes and R² values.

All 480 ratio points are among the 545 diagnostic points. The difference is
60 positive predictions below MDL and 5 nonpositive predictions also below MDL.
No value is substituted into a denominator. The registered Delhi exclusion is
shown separately and remains excluded from the diagnostic statistics.

Denominator sensitivity differs sharply by site. JPL has 77 of 84 ratio points
between 1× and <2× their own MDL; applying a 2× rule leaves seven. Addis retains
189 of 190. Its largest ratio, 55.41 for ETAD-0037, is the one near-MDL point.
Beijing's median declines as stricter thresholds change the retained population.
The baseline threshold is unchanged. Inverse ratio-versus-EC patterns can also
arise because EC appears in the denominator; no causal inference is made.

Reported-date plots show individual filters and substantial date structure,
particularly at Addis. They contain no temporal interpolation, active-interval
claim or seasonal/trend fit. Full site/population date ranges and medians/IQRs
are in the distribution table.

## Retrieval priorities and remaining real-data gate

The current exports overlap reported bounds for 233 filters with usable HIPS:
184 Addis filters and 49 JPL filters. Of these, 230 have hour-consistent
envelopes and three require active-period reconciliation. The queue ranks them
for retrieval only. It does not convert bounds or regular input timestamps into
active sampling or observed coverage.

ETAD-0243 is deliberately lower priority for these exports: its portal start
on 24 August 2024 follows the Addis export's final timestamp on 20 August.
Later instrument records would also be needed. Earlier 2025 ChemSpec exports
provide additional, separately labeled retrieval bounds where the frozen
catalog had none, including Beijing/Delhi; diagnostic dates remain frozen.

No filter-linked sampler on/off record has been recovered. Instrument
observation, correction, unit and optical-conversion histories remain
unverified. A first reviewed real interval subset therefore remains pending.
The Drive instrument SQLite schema was readable, but a grouped range query
did not complete within roughly four minutes and was canceled; its coverage
remains an access gap, not evidence of absent later data.

## ChemSpec finding: upstream of the recovered importer

An earlier inventory missed the local files under `research/filter_combine/`.
Those exports are labeled updated 2025-07-29, data version 3.0. All 1,043
current ChemSpec EC rows match their earlier values and MDLs by physical filter
and method. EC is code 28203, analysis description FTIR, units µg m⁻³,
conditions Ambient local, with method codes 217 or 218.

For CHTS-0658, earlier CSV rows 5883 and 5884 already contain 0.93 and 0.06
in the **Value** field, with the same code, method and MDL. They map to unified
rows 5859 and 5860. All row numbers are zero-based data rows after headers.

The deleted historical `FilterDataIntegrator` was recovered from Git history.
Its `load_chem_spec_data` maps `Value` to `Concentration` and `MDL` separately
to `MDL`, appending one output row per original row. It does not create the
additional concentration row by melting MDL. Thus the ambiguity is already
present in the earlier export. No active parser correction is justified by
this finding; no value is selected by magnitude or MDL equality. Upstream
export-generation code and method/role definitions remain needed for the 500
conflicting groups. The trace restores source metadata discarded by that importer.

This finding supersedes the previous inventory gap about EC parameter codes.
It does not resolve competing value roles or turn ChemSpec into an independent
EC reference. Historical code, commit and source hashes are retained alongside
the original rows and group-level adjudication status.

## Two narrow implementation changes

Reviewed processing assertions now require `evidence_scope` (`full_export` or
`bounded_period`) and `scope_justification`. Bounded evidence additionally needs
timezone-aware `scope_start_utc`/`scope_end_utc`. Optional
`scope_session_ids_json` restricts which sessions the assertion supports.
The matcher refuses to certify observed coverage or eBC units outside those
bounds/sessions. Explicitly claiming a full export requires evidence covering
that whole export; a source hash alone does not establish it.

For each active interval, coverage is also summarized in four successive
quarters. These are **descriptive diagnostics**, not new exclusion thresholds.
A continuous interval with 75% coverage concentrated in its first three
quarters still passes the original 75% total / 50% segment policy, while its
quarters show 100%, 100%, 100%, 0%. Real quarter summaries await verified schedules.

## Reproduce

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/analyze_filter_diagnostics.py
uv run python research/ftir_hips_chem/workflows/create_filter_diagnostics_notebook.py
uv run pytest -o addopts='' -q tests/test_active_interval_matching.py tests/test_resample_intervals.py
```

Tables, original measurement links, sensitivity membership, queue, importer
trace and manifest are under `research/ftir_hips_chem/output/tables/filter_diagnostics/`.
Six PNG/SVG figure families are under `output/plots/filter_diagnostics/`.
The notebook follows the canonical setup and regenerates the same report/figures.
Frozen source audit tables remain unchanged.

Validation: 23 targeted interval/resampling tests pass, including evidence scope
and concentrated-quarter gaps. All 37 report, table and figure outputs reproduced
byte-for-byte between notebook and command-line execution. Every diagnostic
HIPS/FTIR source-row link and all 1,043 earlier EC links were checked.
