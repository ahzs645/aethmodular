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
