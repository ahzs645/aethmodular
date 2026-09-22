# AIRSpec execution baseline

This is the executable counterpart to the historical AIRSpec catalog record.
OpenResearch project: `203b7e92-c6ad-4d8a-9936-411a08bce8c6`.
Baseline experiment: `d2b46613-884e-4eef-b9d8-d06d62459b63`.

The baseline tests whether the unchanged `run_ftir_13.py` reproduces five
historical tables and six tables from the independently executed local
reproduction. It fits the OCEC and smoke calibrations and HIPS transfer models
using frozen, previously corrected spectral caches. It does not repeat full-pool
baseline correction, establish independent Addis accuracy, or select a winner
between AIRSpec and VIBES.

## Fixed run contract

```sh
orx exp run d2b46613-884e-4eef-b9d8-d06d62459b63 --backend local
```

OpenResearch executes this fixed command from its immutable source snapshot:

```sh
bash research/ftir_hips_chem/workflows/openresearch_airspec/run.sh
```

The shell recipe creates its own environment with Python 3.13.9 and `uv.lock`.
The baseline uses one computational thread and a noninteractive plotting backend.
It does not reuse the user's active virtual environment. Logs include effective
configuration, input checks, model progress, and a final JSON evidence summary.

## Inputs and portability

`inputs.json` pins 17 files (291,137,277 bytes). The complete manifest's SHA-256
is the bundle directory name:

`~/.local/share/aethmodular/input-bundles/5b651a3e20880cc1d7284823b0422b833d2d00e5101bfff9f74bd249bf3ba665/`

The large data files are local and outside Git. To use another computer, transfer
that complete directory to the same home-relative location; preserve its
`manifest.json` and relative filenames. The runner checks every file and the
manifest before copying data into the run's own directory. Missing or changed
data fail before fitting. Access to the original Google Drive mount is not
required during a run. Dependency installation needs the uv package cache or
package-index access. This recipe has been prepared for local CPU execution;
remote execution needs the same input bundle staged on the compute host first.

`bundle-provenance.json` records original source locations and the prior
execution's manifest hash. One cloud placeholder was retrieved directly from
Google Drive and accepted only after its SHA-256 matched the earlier evidence.
Original data files and historical snapshots are untouched.

## Evidence gate

- Six scientific source files must match their pinned hashes.
- Historical and previous-local reference files must match their pinned hashes.
- Scientific CSV/NPZ/NPY/pickle reads are restricted to declared inputs,
  comparison references, and outputs within this run.
- Compare all five historical tables, including structure, sample identities,
  missing values, and numeric values at absolute tolerance 1e-6 (no relative tolerance).
- Compare six previous-local tables: split membership, held-out metrics, component
  curves, smoke component choices, and DF1=6/8 predictions.
- Recover 606 training and 194 test OCEC rows, disjoint sites, and k=5 for both
  DF1=6 and DF1=8. Verify staged inputs and source hashes again after execution.
- Exit nonzero on any mismatch. A successful process exit alone is insufficient:
  inspect `ORX_FINAL_SUMMARY` in the persisted OpenResearch log.

Results are written beneath
`research/ftir_hips_chem/output/tables/openresearch_airspec_run/` in the run
snapshot. The unchanged historical runner also generates its legacy figure;
its original OLS-only annotations are preserved as reproduction evidence and
are not offered as a new uncertainty-aware scientific figure.

## Continuing research

Once this baseline passes, freeze its branch. Create a child for a new question;
inherit the same command and change committed code/config only. For a deliberate
scientific modification, declare the child question and evaluation criteria in
its config before running; historical equality is the reproduction baseline's
criterion, not a claim that every future model must produce identical predictions.
Retain the same input identities and evaluation split where a comparison is intended.

The next suggested question is whether training-only spectral applicability
diagnostics identify shared AIRSpec/VIBES failure cases. The already inspected
test cases are exploratory evidence, so they cannot provide fresh confirmation
for rules designed after inspecting them. Preserve full-cohort scores and use
new evaluation data for subsequent improvement claims.

The retrospective's family/dependency/correction graph remains contextual
research evidence. Parent-child branches in this project represent actual code
inheritance and tested decisions. No catalog record is relabelled as a native run.
