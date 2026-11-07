# Project next steps: existing-data closeout

Updated **7 November 2026**. This plan supersedes the [September plan](project-next-steps-2026-09-22.md). The [current synthesis](current-research-summary.md) names the scientific evidence and limits; the [implementation receipt](gallery-closeout-implementation-2026-11-07.md) records what this batch actually verified.

## Endpoint

Conclude the existing-data investigation with one recoverable source/data release and a short figure sequence. Absolute Addis EC accuracy may remain unresolved. A method does not have to win for this stage to finish.

## Completed work to reuse

- Matched calibration/seasonal study and sensitivities (PR #14).
- Baseline lab and seasonal spectral ML (PR #15).
- Initial GPU investigation and later Newton/JAX/WebGPU research (PR #16 and subsequent dated studies).
- Notebook diagnostics and narrow/broad continuum controls (PR #17).
- Existing graph focus, custom sections, shared links and figure/section export.

These are inputs to the synthesis, not tasks to restart. Preserve their distinct populations and versions.

## Remaining sequence and acceptance

| Order | Work | Done when |
|---|---|---|
| 1 | Reconcile the current synthesis | November findings, superseded labels and exact interpretation limits are recorded; historical receipts retain their original scope. |
| 2 | Repair EPO and source metrics | Each outer site fold estimates EPO on training data only; perturbing fixed-test labels cannot change the projection or predictions; identities/folds/coverage and distinct error definitions are saved. Corrected scientific numbers require a real-data rerun. |
| 3 | Reconcile the comparison population | One contribution per physical filter; no train/test site or filter overlap; common scored filters plus each candidate's availability; explicit outcome-restricted source domain. |
| 4 | Finish the review sequence | Six figures reopen with their references, cohort/calendar, evidence status and interpretation. Old EPO scores are withheld unless corrected provenance is attached. |
| 5 | Review PR #19 against the research snapshot | Availability reflects the build people open; source, execution evidence, graph availability and validation remain separate. Research visibility is preserved. |
| 6 | Verify and preserve the release | Declared source/public/research checks pass or are explicitly unavailable; a fresh-context shared section and completed export work; the source/data pair and selected scientific tables can be restored. |
| 7 | Record completion | The receipt states exactly which scope is complete and lists separate follow-on work. |

The current implementation adds a **Build six-figure review section** button at Experimental → Continuum. It uses existing section editing, Copy link, PDF and Deck. A section is a selection of live exports; the release must preserve the data separately.

## Scientific rerun requirements

Restore the October decision bundle with its manifest and metadata, `improve_mass.csv`, the November `corrected_anchorhp_*` spectra, and the recorded source baseline artifacts. New baseline/EPO archives must carry ordered physical-filter IDs, sites, fold assignments and metadata hashes. Equal array lengths alone do not prove alignment.

Use the corrected fixed-k scoring command recorded in the implementation receipt. Do not overwrite historical predictions or infer corrected values from the saved rounded chart summaries. Keep source evaluation separate from all-source target prediction. No tuning on the outer scoreboard is called independent confirmation.

If those inputs are unavailable, the software and interpretation batch can be reviewed with EPO source claims withdrawn. The research release remains pending; unavailable numerical checks are not passed checks.

## Verification profiles

- `uv run aeth gallery check --profile source`: source/synthetic behavior and compiled visibility, without claiming scientific input availability.
- `uv run aeth gallery check --profile public`: source checks plus the public snapshot's declared integration checks.
- `uv run aeth gallery check --profile research`: the full set, including private/legacy research evidence. This is the default for `check`.

Unexpected failures are fatal. Missing declared inputs are unavailable and block their selected profile. Preserve the generated receipt, data hash, commands and tested source revision. A deployment success alone is not the whole acceptance record.

## Follow-on work outside this closeout

Independent Ethiopian thermal EC and sampling confirmation; agreed future TOR acceptance criteria; full PMF qualification (PR #18); numerical-v2 season-composition/fingerprint promotion checks if those analyses adopt v2; further acceleration; historical notebook coverage without a specific conclusion to support.

Do not send the existing data request or contact collaborators without explicit authorization. Keep the proposed TOR rule labelled proposed; no new sampling requirement is introduced as a condition for finishing the existing-data account.
