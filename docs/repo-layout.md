# Repo Layout

## Top-level Zones

- `src/`: package code and reusable library logic
- `tests/`: automated validation
- `research/`: notebook/data-heavy research assets
- `scripts/`: operational scripts
- `docs/`: maintenance and usage docs
- `artifacts/`: local/generated outputs (ignored)

## Source Code

`src/` follows canonical imports:
- `src.analysis.*`
- `src.data.*`
- `src.core.*`
- `src.config.*`
- `src.utils.*`
- `src.visualization.*`

## Research

- `research/ftir_hips_chem/`: ETAD/FTIR/HIPS working assets and datasets
- `research/filter_combine/`: filter-combine logic and inputs

Generated outputs under `research/**/output/` are ignored.
