# AETH Modular

Research-first toolkit for ETAD aethalometer and FTIR workflows.

This repository now uses a `pyproject.toml`-based setup (uv-ready), canonical `src.*` imports, and a clearer separation between package code, research assets, and operational scripts.

## Quick Start

### 1. Environment (uv)
```bash
# create and sync environment (if uv is installed)
uv sync
```

If you do not have `uv`, install dependencies with pip:
```bash
pip install -r requirements.txt
```

### 2. Quality Gates
```bash
uv run pytest -q
uv run ruff check src tests
uv run python scripts/diagnostics/run_notebook_smoke.py
```

`run_notebook_smoke.py` executes the current portable notebook set:
- `notebooks/AAARpos.ipynb`
- `notebooks/ETAD_Aug13.ipynb`
- `notebooks/ETAD_comprehensive_absorption_analysis.ipynb`
- `notebooks/filter_data_availability_strip_chart.ipynb`
- `notebooks/warren_ratio_diagnostics.ipynb`

## Data Root Configuration

Data-heavy scripts resolve paths from:
- `AETHMODULAR_DATA_ROOT` (if set)
- otherwise: `research/ftir_hips_chem`

Example:
```bash
export AETHMODULAR_DATA_ROOT=/path/to/your/data/root
```

## Repository Layout

- `src/`: importable package code (`src.*`)
- `tests/`: pytest suite
- `research/ftir_hips_chem/`: FTIR/HIPS research assets and source datasets
- `research/filter_combine/`: filter-combine research assets
- `scripts/diagnostics/`: diagnostics and inspection scripts
- `scripts/pipelines/`: data processing pipelines
- `docs/`: repository and workflow docs
- `artifacts/`: generated outputs (ignored)

## Script Entrypoints

Preferred entrypoints:
- `scripts/diagnostics/check_etad_data.py`
- `scripts/diagnostics/check_matching_statistics.py`
- `scripts/diagnostics/compare_pkl_files.py`
- `scripts/diagnostics/get_etad_stats.py`
- `scripts/diagnostics/inspect_flow_columns.py`
- `scripts/diagnostics/test_system.py`
- `scripts/pipelines/create_9am_resampled_datasets.py`

Legacy root-level script names are kept as temporary wrappers and print deprecation messages.

## Notes

- Generated output under `research/**/output/` is now ignored and no longer tracked.
- Core import paths have been normalized to `src.*` for reproducibility.
- See `docs/migration-paths.md` for old-to-new paths.
