# Migration Paths

## Directory Moves

- `FTIR_HIPS_Chem` -> `research/ftir_hips_chem`
- `Filter Combine` -> `research/filter_combine`

## Script Moves

- `check_etad_data.py` -> `scripts/diagnostics/check_etad_data.py`
- `check_matching_statistics.py` -> `scripts/diagnostics/check_matching_statistics.py`
- `compare_pkl_files.py` -> `scripts/diagnostics/compare_pkl_files.py`
- `get_etad_stats.py` -> `scripts/diagnostics/get_etad_stats.py`
- `inspect_flow_columns.py` -> `scripts/diagnostics/inspect_flow_columns.py`
- `test_system.py` -> `scripts/diagnostics/test_system.py`
- `create_9am_resampled_datasets.py` -> `scripts/pipelines/create_9am_resampled_datasets.py`

Root-level files with the same names remain as temporary deprecation wrappers.

## Import Migration

Canonical imports are now:
- `from src...`

Legacy direct package-root imports (`from analysis...`, `from data...`, etc.) should be migrated.
