# Plan: lot-aware VIBES correction across the SPARTAN network

Written 2026-10-25. Status: proposed work; no new models have been fitted or deployed.

## Objective

Extend the existing Addis blank-training and validation workflow to the native
SPARTAN FTIR network. Select the blank library using verified filter-lot and
measurement provenance, evaluate whether separate models improve correction,
and expose the resulting coverage in the gallery.

Lot is an explicit modelling factor. A separate model for every lot or city is
a candidate to evaluate, rather than an assumed improvement. Sites can share a
lot, and one site can span several lots and measurement conditions.

The primary deliverable is validated baseline correction with traceable model
assignment. Any dependent experimental EC predictions require a matched
calibration evaluation and migration of their own.

## Starting evidence

- The current native export has 534 distinct media labelled field blank,
  represented by 545 analyses at 28 sites. These are inventory counts before
  quality screening. The wider gallery has 40 sites; the other sites are not
  thereby established to have native spectra or suitable blanks.
- Field-blank media counts include Addis 37, Beijing 29, Delhi 32 and JPL 24.
  Counts by site do not establish counts by lot or instrument period.
- The network workflow explicitly applies the existing 87-blank, rank-70
  IMPROVE/Addis model without retraining its background basis. It still fits
  that basis to each measured spectrum separately.
- The current per-analysis gallery descriptor contains media, analysis and scan
  identities, QC and spectral-grid information, but does not carry lot identity.
  Lot and instrument provenance must be recovered from authoritative source
  metadata before assigning new models.
- The original Addis workflow already averages repeat scans by physical media,
  splits training and test blanks within lot, selects PCA rank from training
  blanks, and checks held-out blanks and synthetic peak recovery.

A blank library's site labels alone cannot establish whether a sample's lot is
represented. Recorded instrument background scans and the learned VIBES blank
basis are separate parts of the processing chain and need separate provenance.

## 1. Recover a verified media, lot and measurement ledger

Join the archived filter, FTIR analysis and scan metadata by physical MediaId
and analysis/scan IDs. Retain the original source identifiers alongside any
normalized display IDs. Use the existing ID helpers.

Record for each physical filter and each contributing analysis:

- Source network, site, MediaId, external filter ID and filter type.
- Source lot ID, external/manufacturing lot identifier where available, and the
  evidence linking them. Equal-looking lot numbers in different databases are
  not sufficient evidence of the same manufacturing lot.
- Sample and background scan IDs, instrument identity where recorded, analysis
  date, and documented acquisition/session or processing batch where available.
- Native grid, resolution metadata where available, QC status, finite/invalid
  channel counts and archived source hashes.
- Missing, conflicting or unresolved metadata, with a reason.

Distinguish field blanks, laboratory blanks and instrument background scans.
Use verified field blanks for the primary extension of the Addis procedure;
other blank types can be separately labelled sensitivity analyses.

Build site-by-lot-by-measurement-group coverage tables showing unique physical
blanks, analyses, ambient filters and dates. Flag exclusions using the shared
exclusion system. Preserve every original record in the ledger. Average only
compatible repeat scans when constructing one training row per physical blank;
keep incompatible sessions separate in the ledger and in the same evaluation
group so the physical blank cannot cross a split.

**Output:** a lot/measurement ledger, blank eligibility ledger, source receipt
and coverage report. Unresolved lots remain explicitly unknown. Missing source
fields become a concrete metadata-recovery task, rather than an inferred lot.

## 2. Freeze the comparison and evaluation protocol

Use the inventory to define feasible strata and training/test allocations
before inspecting candidate performance. Save seeds, eligibility rules, splits,
spectral windows, metrics, model-selection rules, minimum coverage and numerical
acceptance limits in a versioned protocol.

Evaluate these questions separately:

| Question | Evaluation design |
| --- | --- |
| Can a model correct another blank from a represented lot and measurement group? | Hold out complete physical blanks within the group; select rank using only training blanks. |
| Does a model transfer between sites using a verified shared lot? | Hold out sites where feasible; state explicitly whether target-site blanks are available for adaptation. |
| Does it transfer to a new acquisition period or instrument group? | Hold out documented batches/periods or instruments; retain all analyses of a physical blank together. |
| What happens when a sample's lot is absent from training? | Hold out whole lots and evaluate the pooled fallback; a lot-specific model cannot be trained on the unseen lot in this test. |

All repeated scans and synthetic versions of a physical blank stay in its fold.
If a source-domain EC benchmark is also run, preserve its calibration/test site
separation and explicitly record whether target-domain blank adaptation is
permitted. Never describe adaptation using target-site blanks as a completely
unseen-site test.

The adapter's requirement of four independent training blanks is only a
computational minimum. After reserving validation and final test blanks, a
stratum must also meet the protocol's evidence and rank requirements. Sparse
groups remain unsupported for separate-model selection. Numerical deployment
tolerances must be fixed before results are inspected; they are not established
by the current inventory.

Previously inspected test results can support exploratory comparisons. A new
confirmatory improvement claim needs an evaluation set not used to develop or
choose the method. If none exists, label the result exploratory.

**Output:** frozen protocol and physical-blank membership tables, including
which questions each split can answer.

## 3. Prepare compatible spectra and candidate blank models

Reuse `fit_vibes_background` and `vibes_baseline_matrix` in
`research/ftir_hips_chem/scripts/vibes_baseline.py`. Extend reusable selection
and provenance helpers in `scripts/`; keep orchestration in `workflows/` and
use a notebook to document the experiment and results.

Compare a limited set of blank-bank choices:

| Candidate | Purpose |
| --- | --- |
| Frozen current 87-blank model | Reproduce the current gallery correction as the historical reference. |
| Expanded pooled model | Incorporate eligible network blanks across verified lots and compatible measurement conditions. |
| Lot-matched model | Pool sites sharing a verified lot, within compatible measurement conditions. |
| Lot and measurement-group model | Test additional instrument/batch separation where independent blank coverage supports it. |

Hold the solver, loss, asymmetry setting, spectral window and other fitting
choices constant when testing the effect of blank-bank choice. Select each new
model's PCA rank from training blanks only, using the existing blank-only
cross-validation procedure. Record the selected rank and whether it reaches a
cap. Do not impose rank 70 on smaller banks.

Define comparison grids from measured spectral support. Quantify resampling
effects and distinguish the full-range VIBES assessment from the shared
AIRSpec window. Do not silently turn unsupported endpoints into measurements.
Document the current model's flat endpoint extension when reproducing it; any
change to that policy is a separately identified processing change.

**Output:** immutable model bundles with blank IDs, verified lot coverage,
measurement coverage, grid, rank-selection results, code/configuration hashes
and solver settings.

## 4. Pilot, evaluate and freeze the model-assignment rule

Start with well-covered lots represented among the four focus sites, then add
a different grid/instrument group and a sparse or unknown-lot case. Select
these cases from metadata coverage, rather than their eventual correction
results. Expand to the other native-FTIR sites after the pilot reconciles.

For each feasible candidate, evaluate the same independent test blanks and
the same synthetic additions:

- Blank residual RMS, signed residuals, residual structure by spectral region
  and negative corrected signal. A field blank need not be chemically empty;
  a smaller RMS alone is not sufficient evidence of correct removal.
- Recovery of predefined narrow and broad additions at weak and stronger
  amplitudes, including tests that detect removal or distortion of analyte
  peaks. Reuse the Addis design as a reference and freeze any extensions.
- Fit success, numerical warnings, convergence, repeatability and sensitivity
  to removal of influential training blanks.
- Coverage and performance by lot, instrument/batch and site, including sparse
  groups. Count independent parent blanks, not injected versions, as the
  experimental units. Use uncertainty estimates appropriate to the grouping
  and report when there are too few independent groups for useful intervals.

Use training/validation results to select a simple assignment rule; evaluate
that frozen rule once on reserved final test blanks. Do not choose a model
for each ambient filter according to its HIPS agreement or how attractive its
corrected spectrum looks.

The assignment outcomes are:

| Evidence for the sample's lot and measurement conditions | Assignment |
| --- | --- |
| A specific model meets the frozen criteria | Use that model and record the matching provenance. |
| Specific models lack support, but a pooled model meets transfer criteria | Use the validated pooled model with an explicit coverage description. |
| Lot or measurement identity is unknown | Mark the missing identity; any pooled result remains explicitly experimental unless the protocol supports that missing-metadata case. |
| No candidate meets the criteria, or fitting fails | Report unsupported/failed VIBES correction; retain available raw and spline traces. |

Test results may justify retaining the current model for some groups. Every
decision must include the evaluated population, uncertainty and coverage.

**Output:** paired comparison tables, a short interpretation report, a frozen
assignment manifest and explicit unsupported cases.

## 5. Produce versioned network corrections

Apply the frozen assignment rule to each eligible native analysis, preserving
the source raw spectrum and original analysis identity. Keep repeat analyses
inspectable, while training/evaluation counts remain based on physical blanks.

Save the model ID, lot-match status, measurement-match status, reason for any
pooled assignment, fitted grid, resampling details, solver status and output
hash with every correction. Keep the existing correction as a named archive.
For a retrospective gallery release, record the dated blank snapshot used;
any claimed prospective workflow must restrict inputs to those available at
the prediction date.

Run `uv run --no-sync aeth doctor` before scientific execution. Use resolved
data paths and source manifests, checkpoint longer runs and keep failed fits
visible. Begin with a small pilot to measure runtime and storage before
committing to the full network run.

**Output:** complete correction and provenance exports with coverage totals
reconciled against the original media/analysis ledger.

## 6. Handle dependent experimental models consistently

If the new corrections feed EC predictions, calibration scores, spectral
similarity or other derived analyses, rebuild the affected outputs against
the matching preprocessing version. Do not substitute a new correction beneath
an old calibration score. Compare calibrations on paired, identity-confirmed
evaluation filters using the correct spectral window.

IMPROVE TOR can evaluate source-domain EC prediction. HIPS comparisons at
SPARTAN remain optical agreement diagnostics; public ChemSpec EC is an
FTIR-derived product. Neither supplies the missing independent thermal EC
validation at the focus SPARTAN sites. Keep source-domain and target-domain
claims separate. Published SPARTAN chemistry retains its source provenance.

The first gallery release may expose validated baseline traces while retaining
clearly versioned existing experimental predictions. A later switch of those
predictions requires completion of their matched refitting and evaluation.

**Output:** a dependency/migration checklist and, where applicable, matched
calibration bundles, score tables and archived previous results.

## 7. Expose provenance and coverage in the gallery

Replace the city-based warning with information derived from the assignment
manifest. Show, where verified:

- Filter lot and the VIBES model used.
- Whether that lot and measurement group are represented in training.
- Number of independent training blanks and provenance of the blank bank.
- Available held-out evidence, with links to the relevant comparison.
- Clear pooled, unknown-provenance, unsupported and failed-fit states.

Add a coverage table by lot and measurement group, with site as a useful
filter. Use plain-language labels; model IDs, hashes and detailed numerical
diagnostics can live in expandable provenance details. Build display strings
from exported metadata so the warning cannot disagree with the actual model.

Verify assignment logic, missing/conflicting lot handling, physical-blank
separation, grid support, failed-fit handling, archive/version consistency and
export round trips. Recheck gallery controls, drawers, legends and provenance
tables across the existing 320-2560 px responsive sweep, including touch,
keyboard and local horizontal scrolling. Publish the completed gallery export
with its exact source/model version and preserve the existing Site audience.

**Output:** a reviewable gallery showing each correction's applicability and
its evidence, with all supported, uncertain and unavailable cases accounted for.

## Completion criteria

1. Every native filter has a verified lot identity or an explicit unresolved
   status, and source identities reconcile without duplicate physical counts.
2. Training, validation and final evaluation memberships are auditable and
   independent at the physical-blank level.
3. Each model assignment follows the frozen rule and records the evidence for
   its lot/measurement coverage; unsupported groups remain visible.
4. Paired blank and recovery results support the stated correction claims;
   uncertainty and sparse-group limitations accompany the results.
5. Every displayed trace and dependent experimental result identifies the
   matching processing/model version, and the gallery passes responsive checks.

## Existing code and evidence to build on

- [Addis comparison and lot-aware blank split](../research/ftir_hips_chem/scripts/vibes_comparison.py).
- [Reusable VIBES adapter and blank-only rank selection](../research/ftir_hips_chem/scripts/vibes_baseline.py).
- [Pooled experiment, blank-bank membership and leakage checks](../research/ftir_hips_chem/scripts/vibes_large_run.py).
- [Current frozen-model network correction](../research/ftir_hips_chem/workflows/network_baselines_20261019.py).
- [Network exporter](../gallery/data/export_spartan_network.py).
- [Native network export inventory and source procedure](spartan-network-data-pulls.md).
- [Rank-70 migration and dependent-model handling](vibes-gallery-rank70-migration.md).
- [Current research claims and validation limits](current-research-summary.md).
