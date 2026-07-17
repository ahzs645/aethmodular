# AETH Modular

Research-first toolkit for ETAD aethalometer and FTIR workflows.

This repository now uses a `pyproject.toml`-based setup (uv-ready), canonical `src.*` imports, and a clearer separation between package code, research assets, and operational scripts.

Detailed usage guides: `docs/library-usage.md` and `docs/commands.md`.

## Quick Start

### 1. Environment
```bash
# Python 3.13 is recommended
uv sync --python 3.13
uv run aeth doctor
```

`pyproject.toml` and `uv.lock` are authoritative. `environment.yml` and
`requirements.txt` are compatibility fallbacks for Conda and pip users.

If `uv` is unavailable, use a Python 3.9+ environment and install with pip:
```bash
python -m pip install -e .
```

### 2. Repository Commands

The `aeth` command is the supported front door for routine work:

```bash
uv run aeth --help
uv run aeth doctor
uv run aeth notebook list
uv run aeth build list
```

### 3. Quality Gates
```bash
uv run aeth check
uv run aeth check --notebooks
uv run aeth notebook run
```

`aeth check --notebooks` checks notebooks changed in the worktree. Use
`aeth notebook check` for the full portability audit; that audit currently
tracks a documented migration backlog.

`run_notebook_smoke.py` executes the current portable notebook set:
- `notebooks/analysis/absorption/AAARpos.ipynb`
- `notebooks/analysis/absorption/ETAD_Aug13.ipynb`
- `notebooks/analysis/absorption/ETAD_comprehensive_absorption_analysis.ipynb`
- `notebooks/analysis/data_availability/filter_data_availability_strip_chart.ipynb`
- `notebooks/analysis/absorption/warren_ratio_diagnostics.ipynb`

## Data Root Configuration

Data-heavy scripts resolve paths from:
- `AETHMODULAR_DATA_ROOT` (if set)
- otherwise: `research/ftir_hips_chem`

Example:
```bash
export AETHMODULAR_DATA_ROOT=/path/to/your/data/root
```

Meteorology resources now tracked in:
- `research/ftir_hips_chem/Weather Data/Meteostat/Addis Ababa daily Average met Data.csv`
- `research/ftir_hips_chem/Weather Data/Meteostat/master_meteostat_AddisAbaba_63450_2022-12-01_2024-10-01.csv`

## Library Usage

Canonical Python imports use `src.*`:

```bash
uv run python - <<'PY'
import src
import src.core.monitoring
import src.analysis.aethalometer.smoothening
from src.config.project_paths import get_project_root, get_data_root, data_path

print("project_root =", get_project_root())
print("data_root =", get_data_root())
print("processed_sites =", data_path("processed_sites"))
PY
```

Notebook execution (including the new meteorology notebook):

```bash
uv run aeth notebook run \
  notebooks/analysis/meteorology/friday_summary_consolidated.ipynb \
  notebooks/analysis/meteorology/meteorology_source_interaction.ipynb
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

## Common Commands

- `aeth diagnose etad|matching|compare-pkl|etad-stats|flow|system`
- `aeth data resample [--site SITE]`
- `aeth spartan pull|coverage|coverage-plots|bridge|connections|extras`
- `aeth notebook list|check|run`
- `aeth build list` and `aeth build run GROUP TARGET`

The underlying files remain in `scripts/` for direct debugging, but documentation
and routine usage should use `aeth` so paths and workflow names remain stable.

## Notes

- New generated output under `research/**/output/` is ignored. A small set of
  historical presentation/summary artifacts remains tracked pending content review.
- Core import paths have been normalized to `src.*` for reproducibility.
- See `docs/migration-paths.md` for old-to-new paths.
