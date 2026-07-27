# research/archive/

Retired research workspaces. The work is finished or superseded; nothing in the
active tree imports from here.

Kept rather than deleted because each directory is the **only surviving record**
of the analysis it contains — in several cases the input data is no longer on
disk, so these notebooks cannot be re-run.

Named `archive/` to match the existing convention in `notebooks/archive/` —
`aeth notebook check` skips any path with an `archive` component, so notebooks
in here are excluded from portability checks automatically.

## Contents

| Directory | Retired | Why | Re-runnable? |
|---|---|---|---|
| `catch_up/` | 2026-07-26 | Six diagnostics, last touched 2026-05-14. No inbound code references. All six notebooks hardcode a machine-specific repo root and a Google Drive data path. | Only after repointing paths; source generator `_build_catch_up_notebooks.py` is included. |
| `charcoal_ftir/` | 2026-07-26 | Superseded by `research/ftir_hips_chem/charcoal_ftir_sources/`. Its reference-spectra fetcher, `sources.json`, and `leads.md` were **merged into** that archive; only the two KBr-pellet lab notebooks remain here. | No — `charcoal_ftir/data/` is gone from disk. |

## Rules

- Don't import from `archive/` in active code. If you need something here,
  promote it back out in the same change.
- Don't "fix" notebooks in here — portability checks are expected to fail.
  `aeth notebook check` skips archive paths.
- Deleting a directory from here is a real decision: `git log --diff-filter=D`
  is the only way back, and for these the outputs are the results.
