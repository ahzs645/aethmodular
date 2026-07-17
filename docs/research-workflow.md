# Research Workflow

## 1. Verify the environment

```bash
uv sync --python 3.13
uv run aeth doctor
```

## 2. Set Data Root

Use repo data by default, or override with:

```bash
export AETHMODULAR_DATA_ROOT=/path/to/data/root
```

## 3. Run Diagnostics

```bash
uv run aeth diagnose etad
uv run aeth diagnose etad-stats
uv run aeth diagnose matching
```

## 4. Run Pipeline

```bash
uv run aeth data resample
# Or select one or more sites:
uv run aeth data resample --site ETAD --site Delhi
```

Optional external site inputs for the pipeline:
- `AETH_BEIJING_PKL`
- `AETH_DELHI_PKL`
- `AETH_JPL_PKL`

## 5. Validate

```bash
uv run aeth check
uv run aeth check --notebooks
```

Run `uv run aeth notebook check` separately to inventory the full historical
portability backlog.

## 6. Notebook Smoke Checks

```bash
uv run aeth notebook run \
  notebooks/analysis/meteorology/friday_summary_consolidated.ipynb \
  notebooks/analysis/meteorology/meteorology_source_interaction.ipynb
```

Meteorology notebook data files:
- `research/ftir_hips_chem/Weather Data/Meteostat/Addis Ababa daily Average met Data.csv`
- `research/ftir_hips_chem/Weather Data/Meteostat/master_meteostat_AddisAbaba_63450_2022-12-01_2024-10-01.csv`
