# Library Usage

This document describes the current, supported way to use `aethmodular` after the cleanup.

## 1. Environment

Install/sync with `uv`:

```bash
uv sync
```

## 2. Core Import Pattern

Use canonical `src.*` imports:

```python
import src
import src.core.monitoring
import src.analysis.aethalometer.smoothening

from src.config.project_paths import get_project_root, get_data_root, data_path
```

Path helpers:
- `get_project_root()`: repo root
- `get_data_root()`: `AETHMODULAR_DATA_ROOT` override or `research/ftir_hips_chem`
- `data_path(...)`: build a path under resolved data root

## 3. Data Root

Set a machine-specific data root if needed:

```bash
export AETHMODULAR_DATA_ROOT=/path/to/data/root
```

Default (when env var is unset):
- `research/ftir_hips_chem`

## 4. Quality Gates

```bash
uv run pytest -q
uv run ruff check src tests
```

## 5. Notebook Workflow

Run selected notebooks through smoke execution:

```bash
uv run python scripts/diagnostics/run_notebook_smoke.py \
  notebooks/friday_summary_consolidated.ipynb \
  notebooks/Seasonalclass.ipynb \
  notebooks/flow_qc_analysis.ipynb \
  notebooks/map.ipynb \
  notebooks/meteorology_source_interaction.ipynb
```

Meteorology notebook input files now tracked in:
- `research/ftir_hips_chem/Weather Data/Meteostat/Addis Ababa daily Average met Data.csv`
- `research/ftir_hips_chem/Weather Data/Meteostat/master_meteostat_AddisAbaba_63450_2022-12-01_2024-10-01.csv`

`notebooks/meteorology_source_interaction.ipynb` resolves these files from:
1. `<AETHMODULAR_DATA_ROOT>/Weather Data/Meteostat/`
2. `notebooks/` (fallback)

## 6. Script Entrypoints

Use:
- `scripts/diagnostics/*`
- `scripts/pipelines/*`

Root wrappers are transitional and print deprecation messages.
