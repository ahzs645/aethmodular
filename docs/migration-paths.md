# Migration Paths

## Directory Moves

- `FTIR_HIPS_Chem` -> `research/ftir_hips_chem`
- `Filter Combine` -> `research/filter_combine`

## Notebook Moves

- `notebooks/AAARpos.ipynb` -> `notebooks/analysis/absorption/AAARpos.ipynb`
- `notebooks/ETAD_Aug13.ipynb` -> `notebooks/analysis/absorption/ETAD_Aug13.ipynb`
- `notebooks/ETAD_comprehensive_absorption_analysis.ipynb` -> `notebooks/analysis/absorption/ETAD_comprehensive_absorption_analysis.ipynb`
- `notebooks/warren_ratio_diagnostics.ipynb` -> `notebooks/analysis/absorption/warren_ratio_diagnostics.ipynb`
- `notebooks/filter_data_availability_strip_chart.ipynb` -> `notebooks/analysis/data_availability/filter_data_availability_strip_chart.ipynb`
- `notebooks/friday_summary_consolidated.ipynb` -> `notebooks/analysis/meteorology/friday_summary_consolidated.ipynb`
- `notebooks/meteorology_source_interaction.ipynb` -> `notebooks/analysis/meteorology/meteorology_source_interaction.ipynb`
- `notebooks/Seasonalclass.ipynb` -> `notebooks/analysis/meteorology/Seasonalclass.ipynb`
- `notebooks/map.ipynb` -> `notebooks/analysis/meteorology/map.ipynb`
- `notebooks/flow_qc_analysis.ipynb` -> `notebooks/qc/flow_qc_analysis.ipynb`

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
