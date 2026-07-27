# Repo Layout

## Top-level Zones

- `src/`: package code and reusable library logic
- `attic/`: working, tested code with no current consumer (see
  [`attic/README.md`](../attic/README.md)) — do not import from `src/`,
  `scripts/`, `research/`, or notebooks
- `aethmodular_cli/`: dependency-light `aeth` command orchestration
- `tests/`: automated validation
- `research/`: notebook/data-heavy research assets
- `notebooks/`: active and archived analysis notebooks
- `scripts/`: operational scripts — `diagnostics/`, `pipelines/`, and `common/`
  (helpers shared between them; not research logic, and not shipped in the wheel)
- `manuscript/`: citation tooling and bibliography sources (needs `pandoc`)
- `docs/`: maintenance and usage docs

## Source Code

`src/` follows canonical imports:
- `src.analysis.*`
- `src.data.*`
- `src.core.*`
- `src.config.*`

`src.utils.*` no longer exists — it moved to `attic/utils/` on 2026-07-26 after
an audit found no live consumer. See [`attic/README.md`](../attic/README.md).

## Research

- `research/ftir_hips_chem/`: ETAD/FTIR/HIPS working assets and datasets
- `research/filter_combine/`: filter-combine source datasets (the rival loader
  implementations were retired 2026-07-26; the CSV/PKL data is still read by
  `research/spartan_ec_2026_06_16`)
- `research/archive/`: retired workspaces — see
  [`research/archive/README.md`](../research/archive/README.md)

Generated outputs under `research/**/output/` are ignored by default. Two
historical presentation decks remain tracked pending retention review.

## Notebook Organization

- `notebooks/analysis/absorption/`: ETAD absorption and ratio diagnostics
- `notebooks/analysis/meteorology/`: meteorology, seasonality, and map context notebooks
- `notebooks/analysis/data_availability/`: filter availability diagnostics
- `notebooks/qc/`: QC-focused notebooks (e.g., flow quality control)
- `notebooks/archive/`: legacy or superseded notebooks

## Reference Docs

- [`commands.md`](commands.md): stable diagnostics, notebook, pipeline,
  SPARTAN, and research-builder commands.

- [`filter-optics-reference.md`](filter-optics-reference.md): SPARTAN PTFE
  filter spec (PT25DMCAN-PF03A, 3 μm pore, FEP ring), SPARTAN SSR + HIPS
  optical chain (MAC = 10 m² g⁻¹, public BC = Fabs / 10), IMPROVE filter +
  HIPS calibration history, every Warren White paper parameter we have a
  value for, and the parameters that are still open (SPARTAN `H`, `α`,
  HIPS wavelength).
