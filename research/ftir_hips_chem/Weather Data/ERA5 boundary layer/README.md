# ERA5 boundary-layer height at SPARTAN and IMPROVE sites

Hourly ERA5 `boundary_layer_height` (metres above ground) at the 37 SPARTAN sites that have
filter dates, each over its filter date range. Pulled for Ann's request (9 Oct 2026) to
normalise concentrations by boundary-layer height, so a shallow Kiremt boundary layer is
not mistaken for higher emissions.

| File | Contents |
|---|---|
| `blh_hourly_spartan.parquet` | `time` (UTC), `code`, `blh_m`, `source`; 1,445,904 rows, none missing |
| `blh_daily_spartan.csv` | per site and local solar day (UTC + round(lon/15) h): `blh_mean_m`, `blh_max_m`, `blh_night_m` (00–06 local mean), `hours` (fewer than 24 at the ends of a range) |
| `sites_spartan.csv` | coordinates, the ERA5 grid cell used, span, share of hours from the gap fill, median BLH |

## Source

- **ERA5 reanalysis** (ECMWF / Copernicus), 0.25° grid, hourly, nearest grid cell to each site.
- Pulled through the **Open-Meteo archive API** (`models=era5`) on 2026-10-09.
- **Gap:** Open-Meteo has no BLH for **2024-01-01 to 2024-06-30** at any site. Those hours
  (`source = arco-era5`, 90,192 rows) come from the public **ARCO-ERA5** store on Google Cloud
  (`gcp-public-data-arco-era5`, `ar/full_37-1h-0p25deg-chunk-1.zarr-v3`), read on a Colab VM on
  2026-10-14. Both are the same ERA5 field: at Addis they agree to within Open-Meteo's 5 m rounding,
  and the joins at 1 Jan and 1 Jul 2024 step by a median of 17 m (typical hourly change 40 m).
- Accuracy: against 300+ radiosonde sites, ERA5 daytime BLH is ~130 m low on average, worse over
  complex terrain (Guo et al. 2021, ACP 21, 17079). Addis Ababa sits at ~2,300 m.

Cite as: ERA5 (Hersbach et al. 2020), "Generated using Copernicus Climate Change Service
information"; accessed via Open-Meteo (open-meteo.com) and ARCO-ERA5 (Carver & Merose 2023).

## Using it

SPARTAN filters integrate over about 9 days, so normalise with the mean over each filter's
sampling window rather than a single day:

```python
import sys; sys.path.insert(0, 'research/ftir_hips_chem/scripts')
import pandas as pd
from boundary_layer import window_mean
h = pd.read_parquet('research/ftir_hips_chem/Weather Data/ERA5 boundary layer/blh_hourly_spartan.parquet')
addis = h.loc[h.code.eq('ETAD')].set_index('time').blh_m
window_mean(addis, '2023-07-01 00:00', '2023-07-10 00:00')
```

## Rebuilding

```bash
uv run python research/ftir_hips_chem/workflows/fetch_boundary_layer_era5.py spartan          # Open-Meteo, resumable
# Jan–Jun 2024 fill on Colab: see research/ftir_hips_chem/workflows/colab_arco_blh_gapfill.py
uv run python research/ftir_hips_chem/workflows/export_boundary_layer_dataset.py spartan     # this folder
```

## IMPROVE (added 2026-10-14)

| File | Contents |
|---|---|
| `blh_sampleday_improve.parquet` | one row per IMPROVE site and sample day (2003-01-03 to 2026-02-28): `blh_mean_m` over the 24 h sample day, `blh_max_m`, `blh_night_m` (00–06 local mean), `hours` (24 for every row); 462,215 rows, 216 sites |
| `sites_improve.csv` | coordinates, IANA time zone and **standard** UTC offset (IMPROVE days run midnight to midnight local standard time, no daylight saving), number of sample days, span, median BLH |

Read entirely from ARCO-ERA5 on Colab VMs (`workflows/improve_blh_colab.py`, three chunks,
~93,000 hourly global fields, ~300 GB read on the VMs, ~65 MB returned): only the UTC hours that
cover some site's sample day, on IMPROVE's shared 1-in-3-day schedule (dates from the validated
IMPROVE archive). So no Open-Meteo gap applies here. FOCO2 has no coordinates and is absent.
Sites within one 0.25° grid cell get identical values: Antumapu and Las Condes (Santiago, 12 km
apart) are an example.

```python
h = pd.read_parquet('research/ftir_hips_chem/Weather Data/ERA5 boundary layer/blh_sampleday_improve.parquet')
h.loc[h.code.eq('ACAD1')].set_index('date').blh_mean_m   # Acadia, one value per sample day
```
