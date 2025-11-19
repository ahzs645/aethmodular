# Processed Aethalometer Datasets - 9 AM Resampled

This directory contains daily-averaged aethalometer data from four global sites, resampled to 9 AM-to-9 AM local time periods to match filter sampling schedules.

## Overview

All datasets have been processed to:
- **24-hour averaging period**: 9 AM (Day N) to 9 AM (Day N+1) local time
- **Matched to filter availability**: Only dates within ±1 day of filter samples are included
- **Multi-wavelength BC measurements**: UV, Blue, Green, Red, and IR wavelengths
- **Both raw and smoothed data**: Where available

## Dataset Summary

| Site | File | Records | Date Range | Device | BC Level (ng/m³) |
|------|------|---------|------------|--------|------------------|
| **Beijing, China** | `df_Beijing_9am_resampled.pkl` | 590 | Feb 2022 - Oct 2024 | WF0010 | ~1,318 (IR) |
| **Delhi, India** | `df_Delhi_9am_resampled.pkl` | 289 | Jun 2022 - Jul 2024 | MA350-0216 | ~11,026 (IR) |
| **JPL, California** | `df_JPL_9am_resampled.pkl` | 765 | Nov 2021 - Aug 2024 | MA350-0229 | ~641 (IR) |
| **Addis Ababa, Ethiopia** | `df_Addis_Ababa_9am_resampled.pkl` | 515 | Dec 2022 - Sep 2024 | MA350-0238 | ~7,918 (IR) |

## Data Structure

Each dataset contains ~150-165 columns organized into:

### Core Temporal Information
- `datetime_local`: Timestamp (9 AM local time, end of averaging period)
- `day_9am`: Date of the 9 AM measurement
- `Site_Code`: Site code (CHTS, INDH, USPA, ETAD)
- `Site_Name`: Site name (Beijing, Delhi, JPL, Addis_Ababa)
- `Device_ID`: Aethalometer serial number

### Black Carbon Measurements

#### Raw BCc (All Sites)
- `Blue BCc`: Blue wavelength (~470 nm) black carbon concentration
- `Green BCc`: Green wavelength (~520 nm)
- `Red BCc`: Red wavelength (~660 nm)
- `IR BCc`: Infrared wavelength (~880 nm)
- `UV BCc`: Ultraviolet wavelength (~370 nm)

#### Smoothed BCc (Site-Dependent Availability)
- `Blue BCc smoothed (ng/m^3)`: Smoothed blue BC
- `IR BCc smoothed (ng/m^3)`: Smoothed IR BC
- `UV BCc smoothed (ng/m^3)`: Smoothed UV BC
- Additional smoothed versions for BC1 and BC2

**Note**:
- **Addis Ababa (ETAD)**: Has both raw and smoothed data for all wavelengths
- **JPL (USPA)**: Has both raw and smoothed data
- **Delhi (INDH)**: Partial smoothed data availability
- **Beijing (CHTS)**: Raw data only; smoothed columns are NULL

#### Source Apportionment (Where Available)
- `AAE`: Absorption Angstrom Exponent
- `AAE biomass`: Biomass burning AAE
- `AAE fossil fuel`: Fossil fuel AAE
- `BB (%)`: Biomass burning percentage
- `Biomass BCc (ng/m^3)`: Biomass-derived black carbon
- `Fossil fuel BCc (ng/m^3)`: Fossil fuel-derived black carbon
- `Delta-C (ng/m^3)`: Delta-C marker

### Attenuation Measurements
- `[Wavelength] ATN1`: Attenuation at spot 1
- `[Wavelength] ATN2`: Attenuation at spot 2
- Available for: Blue, Green, Red, IR, UV

### Environmental Sensors
- `Internal temp (C)`: Internal instrument temperature
- `Sample temp (C)`: Sample air temperature
- `Sample RH (%)`: Sample relative humidity
- `Internal pressure (Pa)`: Internal pressure
- `Sample dewpoint (C)`: Dewpoint temperature

### Flow Measurements
- `Flow total (mL/min)`: Total air flow rate
- `Flow1 (mL/min)`: Flow channel 1
- `Flow2 (mL/min)`: Flow channel 2
- `Flow setpoint (mL/min)`: Target flow rate

### Particulate Matter (OPC)
- `opc.bins.i0` through `opc.bins.i23`: Optical particle counter bins
- `opc.pms.i0`, `opc.pms.i1`, `opc.pms.i2`: PM mass concentrations
- `particulate.counts.i0` through `particulate.counts.i23`: Particle counts by size
- `particulate.masses.i0`, `i1`, `i2`: Particle mass concentrations

### Additional Sensors
- `Accel X`, `Accel Y`, `Accel Z`: Accelerometer readings
- `co2.co2`: CO2 concentration (where available)
- `co2.temperature`: CO2 sensor temperature
- `co2.relativeHumidity`: CO2 sensor RH

### Quality Control Flags
- `high_rough_period`: Boolean flag for high data roughness periods
- `[Wavelength] ATN1_roughness`: Roughness metric for ATN1
- `[Wavelength] ATN2_roughness`: Roughness metric for ATN2
- `Status`: Instrument status code
- `test`: Test mode flag

### Metadata
- `Serial number`: Device serial number
- `Firmware version`: Firmware version
- `device_type`: Type of device
- `Optical config`: Optical configuration

## Data Quality Notes

### Beijing (CHTS)
- ⚠️ **Limited data availability**: Only 222/590 days have BC measurements
- ⚠️ **No smoothed data**: All smoothed BC columns are NULL
- ✅ **Raw data available**: IR, Blue, UV, Red, Green BCc present where data exists

### Delhi (INDH)
- ⚠️ **Partial data coverage**: Only 113/289 days have raw BC data
- ⚠️ **Limited smoothed data**: Only 37/289 days have smoothed BC
- ⚠️ **Anomalies**: Some negative Green BCc values indicate potential calibration issues
- ✅ **Source apportionment available** for subset of data

### JPL (USPA)
- ✅ **Good data coverage**: 489/765 days have raw BC data
- ✅ **Smoothed data available**: 206/765 days
- ✅ **Source apportionment**: Biomass vs. fossil fuel BC available
- ✅ **Clean baseline**: Low BC concentrations ideal for validation

### Addis Ababa (ETAD)
- ✅ **Complete data**: 515/515 days have all measurements
- ✅ **Both raw and smoothed**: All wavelengths fully available
- ✅ **High quality**: No missing values in BC measurements
- ⚠️ **Some extreme values**: UV BCc shows large fluctuations

## Usage Example

```python
import pandas as pd
import pickle

# Load a dataset
with open('df_JPL_9am_resampled.pkl', 'rb') as f:
    df = pd.read_pickle(f)

# Filter to valid BC measurements
df_valid = df[df['IR BCc'].notna()]

# Calculate daily mean BC
mean_bc = df_valid['IR BCc'].mean()
print(f"Mean IR BC: {mean_bc:.2f} ng/m³")

# Get smoothed data where available
df_smoothed = df[df['IR BCc smoothed (ng/m^3)'].notna()]
```

## Matching with Filter Data

These datasets are designed to be matched with the filter chemical speciation data in:
```
../Filter Data/unified_filter_dataset.pkl
```

Matching procedure:
1. Each row represents 9 AM-to-9 AM averaged aethalometer data
2. Match `day_9am` to filter `SampleDate` (±1 day tolerance)
3. For multi-day filters (e.g., 3-day), sum consecutive daily averages
4. Compare aethalometer BC with filter-based EC/OC measurements

## Why 9 AM to 9 AM?

Filter samples are typically collected and changed at 9 AM local time, integrating all particles over the previous 24 hours (or multi-day period). By averaging aethalometer data from 9 AM to 9 AM, we ensure:

1. **Temporal alignment** with filter integration periods
2. **Direct comparison** between optical BC and filter-based EC
3. **Consistency** across all measurement methods
4. **Calibration compatibility** for source apportionment models

## Site Information

### Beijing (CHTS)
- **Location**: Beijing, China
- **Coordinates**: 40.004°N, 116.326°E
- **Environment**: Urban, high pollution
- **Dominant sources**: Coal combustion, traffic, industrial

### Delhi (INDH)
- **Location**: Delhi, India
- **Coordinates**: 28.6°N, 77.2°E
- **Environment**: Megacity, extremely high pollution
- **Dominant sources**: Biomass burning, traffic, diesel emissions

### JPL (USPA)
- **Location**: Pasadena, California, USA
- **Coordinates**: 34.2°N, 118.2°W
- **Environment**: Suburban, clean baseline
- **Dominant sources**: Light traffic, minimal local pollution

### Addis Ababa (ETAD)
- **Location**: Addis Ababa, Ethiopia
- **Coordinates**: 9.0°N, 38.7°E
- **Environment**: Urban, moderate pollution
- **Dominant sources**: Traffic, biomass burning (cookstoves)

## Processing Details

**Script**: `create_9am_resampled_datasets.py`

**Processing steps**:
1. Load high-resolution aethalometer data (1-minute resolution)
2. Shift timestamps by 15 hours to align 9 AM with midnight
3. Resample to daily means using pandas `resample('D')`
4. Shift back to represent 9 AM as the end of averaging period
5. Filter to dates within ±1 day of filter samples
6. Select key measurement columns
7. Add site metadata
8. Save as compressed pickle files

**Timezone handling**:
- Beijing: Asia/Shanghai (UTC+8)
- Delhi: Asia/Kolkata (UTC+5:30)
- JPL: America/Los_Angeles (UTC-8/-7)
- Addis Ababa: Africa/Addis_Ababa (UTC+3)

## File Sizes

- `df_Beijing_9am_resampled.pkl`: 0.79 MB
- `df_Delhi_9am_resampled.pkl`: 0.40 MB
- `df_JPL_9am_resampled.pkl`: 1.04 MB
- `df_Addis_Ababa_9am_resampled.pkl`: 1.26 MB

**Total**: ~3.5 MB (compared to ~11 GB for original high-resolution data)

## Contact & Citation

For questions about these datasets or the processing methodology, please contact:
[Your contact information]

When using this data, please cite:
[Appropriate citation]

---

**Generated**: 2025-11-19
**Script version**: 1.0
**Source data**: High-resolution aethalometer measurements (1-minute resolution)
**Filter data**: unified_filter_dataset.pkl
