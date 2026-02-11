# Research Workflow

## 1. Set Data Root

Use repo data by default, or override with:

```bash
export AETHMODULAR_DATA_ROOT=/path/to/data/root
```

## 2. Run Diagnostics

```bash
python scripts/diagnostics/check_etad_data.py
python scripts/diagnostics/get_etad_stats.py
python scripts/diagnostics/check_matching_statistics.py
```

## 3. Run Pipeline

```bash
python scripts/pipelines/create_9am_resampled_datasets.py
```

Optional external site inputs for the pipeline:
- `AETH_BEIJING_PKL`
- `AETH_DELHI_PKL`
- `AETH_JPL_PKL`

## 4. Validate

```bash
pytest -q
ruff check src tests
```

## 5. Notebook Smoke Checks

```bash
uv run python scripts/diagnostics/run_notebook_smoke.py \
  notebooks/analysis/meteorology/friday_summary_consolidated.ipynb \
  notebooks/analysis/meteorology/meteorology_source_interaction.ipynb
```

Meteorology notebook data files:
- `research/ftir_hips_chem/Weather Data/Meteostat/Addis Ababa daily Average met Data.csv`
- `research/ftir_hips_chem/Weather Data/Meteostat/master_meteostat_AddisAbaba_63450_2022-12-01_2024-10-01.csv`
