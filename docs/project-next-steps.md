# Project next steps

Updated 2026-09-22. This is the project-level work plan; the detailed technical
backlog remains in [open-items.md](open-items.md).

## Scientific direction

The proposed central question is: **How reliably can the FTIR calibration
transfer to Addis, and under what conditions?** The next research summary should
state the intended scientific claim, the evidence that supports it, and the
measurements needed to test it independently.

OpenResearch now supports navigating the historical catalog and executing a
frozen AIRSpec baseline. Prioritize scientific evidence and reproducibility;
extend the dashboard when a concrete research task needs it.

## Current position

- The September filter-only release and a locked AIRSpec calibration have real
  completed OpenResearch runs. AIRSpec reproduced 11 result tables with all
  84 column checks passing at absolute tolerance 1e-6. This is numerical
  reproduction from corrected caches, not independent Addis accuracy validation.
- The VIBES investigation repeated corrections for 12 selected cases. Both
  methods severely underpredict the two leading problematic cases, with VIBES
  making predictions more negative. The repeats converged; a chemical cause
  has not been established.
- Independent Addis validation remains blocked on reference measurements,
  authoritative filter identities and sampling confirmation. A confirmation
  queue and data-request draft exist locally; the request has not been sent.
- The retrospective contains 104 research records and six family headings.
  Twenty-five inventoried sources without executed companions remain
  unexecuted. Evidence labels do not turn them into verified results.
- Substantial analysis, environment, gallery and documentation work remains
  uncommitted. This plan does not constitute a checkpoint of those changes.

These status statements summarize the local evidence reviewed on 2026-09-22.
The consolidation below must preserve its receipts and manifests in a durable,
reviewable release. The frozen AIRSpec source commit is
`fb0abeaf5d272b76c64bf6616743e56f00800362`; its local OpenResearch run is
`5e953103-e1b4-4218-8ddf-f9c67188bfd4`.

## 1. Consolidate the existing work

- [ ] Review the current changes and separate analysis/environment, evidence
  catalog, gallery and presentation work into coherent commits.
- [ ] Verify the relevant tests and reproduction commands for each group;
  inspect dependency-lock and vendored-source changes before committing.
- [ ] Inventory large inputs, generated results and local-only artifacts with
  hashes, provenance and a documented backup location. Verify recovery of a
  representative bundle; a manifest alone is not a backup.
- [ ] Preserve the completed AIRSpec baseline and September release snapshots.
  New scientific changes must have their own recorded source and inputs.

Done when: maintained source and tests are committed, necessary local-only
inputs/results are recoverable, and another checkout can locate its requirements.

## 2. Write one current research summary

- [ ] Create a claim-to-evidence table with the question, result, population,
  split, uncertainty, supporting files/run, limitations and next decision.
- [ ] Distinguish established, exploratory, superseded and blocked claims;
  link corrections to the specific earlier statements they revise.
- [ ] Reconcile older overview documents and pending-task lists with the
  completed reproductions and current catalog counts.
- [ ] Select the historical analyses needed for the intended final claims.
  Prioritize those for reproduction rather than rerunning every notebook.

Done when: each proposed conclusion has traceable evidence and an explicit
scope; completed work is no longer presented as an outstanding task.

## 3. Obtain the missing independent validation evidence

This can proceed alongside consolidation and the research summary.

- [ ] Review the existing data-request draft and identify its intended recipient.
  Obtain explicit authorization before sending it.
- [ ] Obtain independent thermal EC measurements, units, uncertainties and
  method/provenance information for the relevant physical filters.
- [ ] Resolve the authoritative Adama spectral-ID crosswalk and sampling flags,
  including the July 9 start-time offset and July 30 PTFE/quartz volume mismatch.
- [ ] Finalize identity and sampling acceptance rules before evaluation; freeze
  predictions and retain an auditable inclusion/exclusion ledger.

Done when: a defensible independently measured paired evaluation set exists,
or the scientific summary clearly limits its claims to the available evidence.
ChemSpec EC derived from FTIR cannot serve as an independent EC reference.

## 4. Specify and run the next VIBES experiment

Proposed question: **Can unreliable predictions be identified using information
available before observing reference EC?** This is a new experiment, not a
retroactive correction of the completed comparison.

- [ ] Freeze the comparison population, physical-filter identities, site split,
  preprocessing settings and input/source hashes.
- [ ] Predeclare the reliability indicators, training-only selection procedure,
  evaluation metrics and success criteria, including retained coverage and error.
- [ ] Compare AIRSpec and VIBES on the same population and evaluation protocol.
  The reproduced 800-filter AIRSpec calibration is not an interchangeable
  comparator for the full-pool VIBES study.
- [ ] Keep the original problematic cases and predictions in the reported
  evaluation. Do not improve reported scores by clipping or removing failures
  after inspecting reference values.
- [ ] Use fresh evaluation data for improvement claims. Rules motivated by
  already-inspected held-out failures remain exploratory without fresh evidence,
  even when their thresholds are subsequently fitted on training data.
- [ ] Register the executable experiment with one fixed command, its contract,
  logs, metrics and an explicit decision after completion.

Done when: the result answers the declared question, reports the coverage/error
tradeoff and supports a clear accept, reject or inconclusive decision. If fresh
evaluation is unavailable, report an exploratory result rather than a validated
improvement.

## 5. Assemble a reproducible scientific release

- [ ] Resolve the deposit-area convention and uncertainty semantics where they
  affect the selected claims; see [open scientific decisions](open-items.md).
- [ ] Connect inputs, processing, predictions, evaluation, figures and claims in
  a release manifest with fixed commands and environment versions.
- [ ] Reproduce the selected supporting analyses in a clean environment;
  preserve historical evidence separately from the new execution results.
- [ ] Produce the final figures and narrative from the verified outputs, with
  calibration diagnostics distinguished from independent validation.

Done when: a clean checkout plus the documented input bundle reproduces the
reported results within declared tolerances, and the release states which
claims remain limited by missing measurements.

## Execution order

Start with consolidation and the research summary. Review the missing-data
request in parallel. Use the summary to finalize the VIBES contract and select
the necessary historical reruns. Assemble the release after those results and
limitations have been adjudicated. Broader notebook cleanup follows scientific
priority; it is not a prerequisite for every new result.
