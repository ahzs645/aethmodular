# Local OpenResearch checkouts

The two active development checkouts are kept under this repository's
`.local-checkouts/` directory. That directory is Git-ignored because its entries
are complete Git worktrees, not files to copy into the Aethmodular history. Do
not run `git clean -xfd` against the main checkout: that command can remove
ignored local checkouts.

| Local path | Git history | Purpose |
|---|---|---|
| `.local-checkouts/airspec-openresearch/` | Aethmodular branch `orx/airspec-locked-baseline-native-reproduction` at `fb0abea` | Frozen source checkout for the completed native AIRSpec baseline |
| `.local-checkouts/openresearch-ui/` | OpenResearch CLI branch `local/research-ui-v0.2.8` | Local dashboard extension for historical evidence and relationships |

The AIRSpec runner and its small reference tables are also available in the
main checkout at [the AIRSpec recipe](airspec/README.md) and
`research/ftir_hips_chem/workflows/openresearch_airspec/`. The large input bundle,
completed run artifacts, and OpenResearch database remain in their existing
locations under `~/.local/share/`; they are not stored in Git. The original
run receipt records the source checkout's location *at run time*. That
historical path is intentionally unchanged after the move.

OpenResearch's registered path for the AIRSpec project now points to
`.local-checkouts/airspec-openresearch/`. The database was backed up to
`~/.local/share/openresearch/path-move-backup-2026-09-22/orx.db` before the
path change. The installed `orx-research` development executable was rebuilt
for the new UI checkout path; the official `orx` executable is separate.
The UI checkout's `LOCAL_HISTORY_UI.md` has rebuild and restart commands. The
prior 4.1 MB UI assets are retained under
`.local-checkouts/.previous-ui-runtime-assets/` as a local rollback copy.

A fresh clone of Aethmodular will contain the tracked AIRSpec recipe, but not
these local checkouts or the input bundle. The OpenResearch UI remains a separate
upstream-derived Git history so updates can be merged and tested there.
