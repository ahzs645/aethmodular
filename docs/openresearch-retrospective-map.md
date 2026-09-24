# Retrospective research map for OpenResearch

> Historical planning map (September 20). Current [scientific synthesis](current-research-summary.md) and [catalog status](openresearch-retrospective/README.md) supersede pending-task language and counts below. Both the September release and locked AIRSpec baseline now have completed native runs.

Prepared 2026-09-20. This initial map has now been followed by a
[research inventory and a successful OpenResearch reproduction](openresearch-retrospective/README.md).
The family map below remains conceptual; the linked execution record identifies
the one newly reproduced release separately from historical evidence.

## Approach

Start with a completed deliverable and trace its claims back to tables,
notebooks, scripts, inputs, and Git history. Preserve negative results,
corrections, and unresolved evidence. A conceptual research dependency is not
necessarily a Git parent or a directly comparable experiment.

Record historical work as historical evidence. Only label a result reproduced
after executing it and comparing its outputs. Do not assign an old execution
to today's commit merely because its notebook exists there. Git commit dates
record source changes, not necessarily experiment dates.

## Candidate families, working backwards

| Family | Existing evidence | Retrospective treatment |
|---|---|---|
| September filter-only scientific release | [Release README](../deliverables/filter_only_scientific_release_2026-09-11/README.md), claim and figure ledgers, frozen data, `reproduce.py` | Best first reproduction candidate: the documented entry point needs no original Drive mount after dependency installation. It starts from frozen analysis inputs, not raw spectra. |
| Matched-sample and active-interval audit | [Workflow index](../research/ftir_hips_chem/workflows/README.md), [filter-only findings](filter-only-results-2026-09-10.md) | Preserve sampling and processing evidence gaps as unresolved. A reported date overlap is not a validated active interval. |
| Model-form robustness | `research/ftir_ec_phase3/ftir_54_model_form_robustness.ipynb` and `scripts/run_ftir_54.py` | Recover common split, preprocessing, model variants, and extrapolation tests before comparing scores. |
| Spectral analogs and network map | Phase-3 `ftir_50` and `ftir_52`, summarized in the [phase-3 index](../research/ftir_ec_phase3/README.md) | Link to earlier analog-cohort questions; preserve the distinction between spectral similarity and predictive validity. |
| Residual and loading investigations | Phase-3 `ftir_43`, `ftir_45`, `ftir_46`, `ftir_47` | Explicitly link the later paired test and per-calibration-line blank analysis to the claims they revise. |
| Carbon target definitions and external transfer | Phase-3 `ftir_41`, `ftir_42`, `ftir_44`, `ftir_48` | Keep TOR, TOT, OC, TC, and external Adama comparisons separate; they do not share a single interchangeable target or score. |
| Validation protocol | `research/ftir_ec_phase3/ftir_32_cv_scheme_ablation.ipynb` and `scripts/run_ftir_32.py` | Record grouped versus interleaved folds as a protocol experiment, including its implications for older comparisons. |
| AIRSpec correction and uncertainty | Phase-3 `ftir_13`, `ftir_15`, their `run_ftir_*.py` scripts, and `scripts/airspec_baseline.py` | Trace from raw low-OC/EC calibration through corrected variants and bootstrap analysis. Keep R-reference validation as separate supporting evidence. |
| Calibration cohorts and deployed-model diagnosis | Phase-2 `ftir_07`–`ftir_10`, followed by phase-3 `ftir_11` | Recover the deployed baseline, candidate cohorts, exclusions, component selection, and actual evaluation populations. |
| Earlier multisite comparisons and mechanisms | [April research summary](../research/ftir_hips_chem/COMPLETE_RESEARCH_SUMMARY.md) | Use as an index to historical work, not as current scientific authority. Reconcile its language, methods, and conclusions with later source records. |
| Current VIBES comparison | [VIBES record](../research/ftir_hips_chem/VIBES_COMPARISON.md) and `scripts/vibes_large_run.py` | A separate continuation of preprocessing research. These files are currently untracked; do not represent them as part of a historical committed run. |

Paths abbreviated to phase-2 refer to `research/ftir_hips_chem/`; phase-3
refers to `research/ftir_ec_phase3/`. This is a family-level inventory, not
an exhaustive catalog of every notebook or hypothesis.

## Corrections the history must retain

- The phase-3 index qualifies the early residual-learner finding using the
  later paired comparison in `ftir_46`. Keep both the original question and
  its revised interpretation.
- `ftir_47` corrects pooled-lot blank interpretations by using deployed
  calibration lines. Preserve the correction instead of treating both as
  independent accepted findings.
- The [phase-3 summary](../research/ftir_ec_phase3/PHASE3_SUMMARY.md) notes
  that `ftir_11` cohort-specific test sets differ, so their held-out RMSEs
  are not directly comparable. It also flags an uncorroborated VIP statistic
  whose replication scripts and committed tables are missing.
- The September release documents its scope: filter-only analysis does not
  recreate upstream FTIR predictions, certify sampling intervals, or establish
  independent EC validation.

## Record needed for each historical experiment

1. Research question and historical rationale.
2. Related prior experiment(s), distinguishing conceptual links from Git ancestry.
3. Source paths and verified source revision; unknown when not established.
4. Input identity/hashes, exclusions, units, cohort, and evaluation split.
5. Original command/environment if recoverable; proposed reproduction command separately.
6. Result tables, figure paths, metrics, uncertainty, and interpretation.
7. Evidence status: documented, artifacts inspected, reproduced, superseded,
   or missing evidence. These are inventory labels, not asserted OpenResearch statuses.
8. What changed the interpretation and what remains unresolved.

## Initial Git anchors

These are inspected source-history anchors, not verified run commits:

- `d54602e` (2026-09-16): adds filter-only analyses, weekly decks, and the
  September 11 scientific release.
- `1161d79` (2026-09-06): saves phase-3 analyses, figure deliverables, and
  data-path fixes; latest inspected change for `ftir_54`.
- `7a833da` (2026-07-19): latest inspected change for the `ftir_13` notebook.
- `998cf43` (2026-07-17): latest inspected change for the `ftir_07` notebook.

## Proposed first migration

Complete the evidence record for the September release, then reproduce it in
isolation and compare with its frozen expectations. Follow its original
environment contract. Trace the upstream audit separately before moving into
the calibration families. Investigate OpenResearch's supported representation
for historical evidence before creating nodes; never fabricate completed run
records. Use separate experiment families when targets, populations, or run
contracts differ, with cross-references describing the research connections.
