# OpenResearch retrospective

The local project **Aethmodular — research retrospective** now holds a new,
successful reproduction of the September filter-only release and a historical
evidence catalog. The VIBES Colab run is now recorded as completed external
evidence, with a separate local audit of its saved predictions.

## Current status — 2026-09-22

Read the [current scientific summary](../current-research-summary.md) for the
main claim, evidence limits and superseded interpretations.

The retrospective has **104 research records**: 101 historical/release records
and three prospective contracts. Six family headings bring it to **110 nodes**.
The Relationships view displays **82 typed connections**: 45 data dependencies,
29 related-work links, five qualifications and three revisions. These are
catalog counts, not counts of independently validated results.

The separate **AIRSpec execution** project holds one experiment with a completed
native run: 84 column checks across 11 tables pass. Its fixed baseline uses
606 training and 194 held-out rows. [Execution evidence and scope](native-execution-2026-09-22.md).
The retrospective retains the original completed September release run.

Historical records show evidence badges, with actual run status kept separate.
The family map has a searchable catalog and direct-neighbor relationships, plus
a linked native-experiment panel. The local extension now runs on OpenResearch
**0.2.8**, using `orx-research up`; see [local UI details](local-ui.md).

The AIRSpec reproduction, 12-case VIBES investigation and smooth/raw notebook
repair are complete within their stated scope. Independent Addis validation is
still blocked; 25 sources on the historical gap queue remain unexecuted.
[Follow-through](followthrough-2026-09-21.md) records the local investigations;
[native AIRSpec execution](native-execution-2026-09-22.md) records the subsequent run.

- [Open the local project](http://127.0.0.1:4791/projects/c8f10563-d27a-4b5a-a8fd-ad79f3d07568).
- [Evidence inventory and reviewed findings](../../research/ftir_hips_chem/output/tables/openresearch_retrospective/report.md).
- [Machine-readable inventory](../../research/ftir_hips_chem/output/tables/openresearch_retrospective/inventory.json).
- [New reproduction result](../../research/ftir_hips_chem/output/tables/openresearch_retrospective/reproduction/verification.json).
- [Recorded run and source snapshot](../../research/ftir_hips_chem/output/tables/openresearch_retrospective/reproduction/run_record.json).
- [Curated research connections and caveats](curation.json).
- [Completed VIBES subgroup audit and next steps](../vibes-subgroup-audit-2026-09-21.md).

## Initial inventory history (September 20–21)

The initial catalog covers 92 top-level notebooks in `ftir_hips_chem` and
`ftir_ec_phase3`, plus eight selected release, audit, reference and ongoing-work
records: 100 research records, 593 indexed files and 29 conceptual links.
The September 21 refresh adds the VIBES subgroup audit as record 101 and links
the completed Colab evidence to it. Current file and link counts are recorded
in the machine-readable inventory.
These are not 100 independently validated experiments. Each record separates
source history, saved notebook execution state, existing artifacts and new
reproduction evidence. Archive-only analyses and other repository subprojects
are outside this first inventory.

All 125 entries in the release's immutable-file manifest and all 123 entries in
its prior tested-numerical-files manifest match the current files. Those sets
overlap. Existing output files elsewhere are hashed as found; the inventory
does not infer that current source code generated them.

The OpenResearch artifacts folder contains `research-retrospective/report.md`,
the catalog, copied source/evidence files and reproduction outputs. Report
links there target the copied evidence, so later edits to the original source
do not silently alter this snapshot.

## Recorded reproduction

| Field | Value |
|---|---|
| Project | `c8f10563-d27a-4b5a-a8fd-ad79f3d07568` |
| Experiment | `398e3311-13fb-42e4-8919-55349e409a56` |
| Run | `bcbfcace-6a45-4c50-9013-7ef8086ee34e` |
| Source commit | `d54602e6e7f6b8c01ef664b829e66287cb75903e` |
| Branch | `orx/reproduce-september-filter-only-release` |
| Backend | Local; separate extracted source and Python environment |
| Python | 3.13.9, release-pinned dependencies |
| Result | Passed in approximately 20 seconds |
| Scope | Numerical downstream reproduction; figures not regenerated |

The run verified 545 diagnostic and 480 ratio filters, 6,924 stability prediction
rows, 4,668 proportionality prediction rows and 257 supported proportional fits.
The entry point's checked numerical fields agree within `1e-10`; checked split
identifiers and common ID-11 memberships agree exactly. This does not recreate
upstream FTIR predictions from spectra or establish independent EC accuracy.

The fixed command is:

```sh
cd deliverables/filter_only_scientific_release_2026-09-11 &&
uv venv --python 3.13 .venv &&
uv pip sync --python .venv/bin/python requirements.lock &&
.venv/bin/python reproduce.py --no-figures
```

OpenResearch ran the committed snapshot, excluding uncommitted workspace files.
The source checkout remains on its original branch. The answered experiment
branch is frozen; future scientific changes belong on a child or separate
baseline with an appropriate comparison contract. No remote compute or GitHub
publication was enabled.

## What needs reconciliation before further reruns

| Priority | Evidence | Next concrete step |
|---|---|---|
| 1 | `ftir_11` saved scores and later summary qualify older cohort-ranking language | Freeze input identities and each cohort's split; preserve the old claim and its correction. A shared-test ranking would be a new experiment. |
| 2 | AIRSpec `ftir_13` reproduction is complete, including its native run; `ftir_15` uncertainty remains separate | Preserve the frozen baseline. Reproduce the bootstrap only if it supports the final claim; a fresh full-pool correction would be a separate experiment. |
| 3 | `ftir_43` → `ftir_46` changes the residual-signal interpretation | Reproduce the paired contrasts on identical blocks, retaining each comparator and uncertainty interval. |
| 4 | `ftir_47` revises pooled-lot blank results | Preserve the per-line and pooled-lot ledgers and their distinct definitions before replaying fits. |
| 5 | Historical optical/filter plots predate sampling/provenance findings | Recover validated sampling and processing evidence before describing them as co-sampled instrument comparisons. |

The imported `HIPS_Aeth_SmoothRaw_Analysis` snapshot retains its historical
import error. A repaired copy has since executed all 13 code cells without
errors. Its legacy ±1-day matching still does not establish sampling overlap.
See the [repair record](../../research/ftir_hips_chem/output/tables/historical_gap_followthrough/report.md).

The current CLI supports creating new experiment nodes and runs, but the
inspected interface does not expose a historical-run import operation. Historical
work is therefore stored as linked evidence artifacts and explicitly labelled
historical experiment nodes. Only genuinely executed OpenResearch runs are marked Done. Historical
completion uses its own evidence badge; no old run timestamps or commits were fabricated.

## Research-family migration

The current six families cover calibration selection (14 records), AIRSpec
correction (8), blanks and optical loading (7), validation and provenance (26),
filter relationships and site context (45), and VIBES (4). Family membership is
for navigation. The current 82 typed links supersede the initial migration count
of 31. The Relationships view distinguishes dependencies, related work and
claim corrections; Git nesting is not used as proof of a research dependency.
The existing release node retains its original branch, command and completed run.

Each record has a question or purpose, saved findings or an explicit extraction
gap, reviewed corrections, evidence links and reproduction prerequisites.
Standalone commands were located for 48 historical records, alongside the
verified release recipe and verified local VIBES audit recipe. Other entries
point to notebooks or reference-validation sources. Commands have not all been
executed in a reconstructed environment. Archive-only notebooks and individual
sweep configurations remain outside the node count.

The repeatable importer checks complete family coverage and source evidence,
then uses stable import keys to reuse existing nodes. It preserves pre-migration
descriptions and writes a receipt under `output/tables/openresearch_families`.
It creates registration branches from the current source snapshot; these are
not assertions about historical run commits. It launches no analyses or agents.

```sh
# Build and validate the plan only:
uv run --locked --no-sync python research/ftir_hips_chem/workflows/migrate_openresearch_families.py
# Reconcile the native tree and local family guides:
uv run --locked --no-sync python research/ftir_hips_chem/workflows/migrate_openresearch_families.py --apply
```

[Family assignments and documented dependencies](families.json) are curated.
Read the family guide before using an imported node to start new work; stage a
separate executable baseline with the correct inputs and comparison contract.

## Refresh the inventory

```sh
uv run --locked --no-sync python research/ftir_hips_chem/workflows/build_openresearch_retrospective.py
```

This reads source and artifacts and writes the catalog under `output/tables`.
It neither executes historical notebooks nor changes their results. Its optional
`--publish-artifacts PATH` writes a local, hash-checked copy of the indexed
evidence and rewrites report links to that copy. The curated links describe
research reasoning; they are not assertions of Git ancestry or equal test sets.
