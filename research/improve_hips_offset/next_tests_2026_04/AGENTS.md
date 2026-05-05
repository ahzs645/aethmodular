# AGENTS.md

This folder is a clean workspace for the next IMPROVE/FED vs Addis/SPARTAN
tests. It intentionally ignores the older exploratory notebooks in
`research/improve_hips_offset/`.

Follow the root guidance in `/Users/ahmadjalil/github/aethmodular/AGENTS.md`.
Use `/opt/anaconda3/bin/python3.13` when executing notebooks or scripts.

## Conventions

- Do not import code from older notebooks.
- Prefer reusable loaders/helpers from `research/ftir_hips_chem/scripts/`.
- Keep generated figures and tables under `output/` inside this folder.
- Label FED `Ref*` / `Trans*` fields as IMPROVE TOR/carbon-analyzer laser
  R/T ratio fields, not raw HIPS sphere/plate R/T.
- For concentration, mass-loading, fAbs-vs-EC, and optical-loading scatter
  plots, pin axes to origin `(0, 0)` unless a markdown note in the notebook
  explains why a zoomed/nonzero origin is being used.
- Do not use smoke score screening as a quality criterion.
- Keep MTL filter-lot work low priority unless new lot metadata are provided.

