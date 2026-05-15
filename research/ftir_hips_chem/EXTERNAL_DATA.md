# External Data Dependencies

Most active analyses now load from repo-local paths defined in
`scripts/config.py`:

- `PROCESSED_SITES_DIR`
- `FILTER_DATA_PATH`
- `ETAD_FACTOR_CONTRIBUTIONS_PATH`
- `ETAD_FILTER_ID_PATH`
- `WEATHER_DATA_DIR`
- `AERONET_DATA_DIR`

The repo does not currently include the raw minute-resolution Addis MA350 files
or AERONET exports referenced by some older notebooks. Keep those large/source
files outside git, or place local working copies in ignored folders under
`research/ftir_hips_chem/`.

## Local External Data Layout

Use this layout for local-only files:

```text
research/ftir_hips_chem/
├── AERONET/
│   └── daily/
│       ├── 20220101_20251231_AAU_Jackros_ET.lev15
│       └── 20220101_20251231_AAU_Jackros_ET.ONEILL_lev15
└── Weather Data/
    └── Meteostat/
```

`AERONET/` is ignored by git. `Weather Data/` currently contains tracked
Meteostat CSV inputs.

## Notebooks Still Using External Absolute Paths

These notebooks still refer to files in the user's Google Drive. Do not replace
them with 9am-resampled files unless the analysis is intentionally being changed
from minute-resolution to daily/resampled data.

- `addis_01_source_apportionment.ipynb`
- `addis_02_temporal_patterns.ipynb`
- `addis_03_meteorology.ipynb`
- `addis_04_aeronet.ipynb`
- `addis_05_diurnal_wavelength_analysis.ipynb`

