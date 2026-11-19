# Applying Shlipak et al. (2026) in Aethmodular

The [Addis study](https://doi.org/10.1021/acsestair.6c00140) supplies a useful
comparison protocol for MA350 data. It does not supply an independent thermal
EC reference for the current Addis SPARTAN filter cohort. The named workflow
below compares explicitly identified, previously quality-controlled hourly
records with the paper's reported BC values. It does not claim exact paper
reproduction while supporting-information settings remain unresolved.

The [9 November verification receipt](aethalometer-paper-integration-verification-2026-11-09.md)
records the passed local checks and the pending browser and real-data acceptance.

The [Addis study gallery page](aethalometer-addis-study-gallery.md) now applies
the study's calendar and event context to the saved Jacros hourly IR record,
with coverage, seasonal and festival views and explicit missing-input states.

## Implemented components

| Component | Implementation and interpretation |
|---|---|
| Two-component optical model | `src/analysis/bc/source_model.py` solves the two power laws at explicit wavelengths. It retains invalid observations, unbounded fractions and clipping counts. |
| Legacy analyzer | Uses a named Blue/IR pair or a documented UV/IR fallback; converts MA350 BCc to an attenuation proxy before estimating AAE. AE33 BC inputs require their own declared cross-sections. |
| Instrument-source coverage | Counts distinct absolute-hour intervals, with independent AAE, BB/FF and IR support; reports conflicting duplicates and actual UTC boundaries. Existing bins remain 24 elapsed hours. |
| Named study profile | `research/ftir_hips_chem/workflows/profiles/shlipak_2026.json` records the paper, period, sites, seasons, reference values and unresolved methods. |
| Reusable comparison command | Reports per-deployment and common-hour BC means, coverage, seasonal summaries and optional optical fractions, with hashes of the exact inputs parsed. |
| Gallery interpretation | Distinguishes AAE class frequencies from model-derived source fractions and the Jacros record from the paper's separate Central deployment. |
| Accepted source results | The publication path verifies the immutable closeout and accepted scientific source hashes, applies the corrected continuum export, and verifies the actual built/served bytes. |

## Run a comparison

From the repository environment:

```bash
uv run --no-sync aeth study shlipak-2026 --describe
uv run --no-sync aeth study shlipak-2026 \
  --hourly path/to/quality_controlled_hourly.csv \
  --metadata path/to/deployments.json \
  --output research/ftir_hips_chem/output/tables/paper_comparison/run-001.json
```

The destination must be new. The command never overwrites an earlier report.
It reads the profile without loading observations with `--describe`. The
comparison itself requires the repository's scientific dependencies.

### Hourly input contract

| Column | Required meaning |
|---|---|
| `deployment_id` | `central` or `jacros`, explicitly mapped in the deployment metadata. A generic `ETAD` label is insufficient. |
| `timestamp` | Hour-start timestamp with an explicit UTC offset, e.g. `2022-04-01T00:00:00+03:00`. Converted to UTC before matching. |
| `bc_ug_m3` | Hourly optical BC in µg/m³ under the declared upstream processing. Missing, nonfinite and negative values are unavailable; zero remains a valid measurement. |
| `qc_pass` | `true`/`false` or `1`/`0`. Upstream QC and its evidence must be described in the metadata. |
| `absorption_short_mm1`, `absorption_long_mm1` | Optional pair of corrected absorption coefficients in inverse megametres (Mm⁻¹), at the explicitly declared wavelengths. Do not put BCc into these columns. |

Duplicate deployment/hour rows fail rather than increasing coverage. Reconcile
duplicates upstream with a recorded policy. Minute observations must first go
through an explicit QC and minute-to-hour aggregation workflow; this command
does not invent that workflow or its completeness threshold.

Minimal metadata structure (replace the null serials and periods with actual
deployment evidence; the nulls deliberately prevent this example from running):

```json
{
  "schema_version": 1,
  "timezone": "Africa/Addis_Ababa",
  "timestamp_basis": "hour_start_with_offset",
  "bc_unit": "ug/m3",
  "instrument_model": "MA350",
  "inlet": "no_size_selective_inlet",
  "qc": {
    "description": "Describe the actual upstream QC and hourly aggregation",
    "reference": "Path, version or receipt identifying that processing"
  },
  "deployments": {
    "central": {
      "serial": null,
      "start": "2022-04-01T00:00:00+03:00",
      "end_exclusive": "2025-04-01T00:00:00+03:00"
    },
    "jacros": {
      "serial": null,
      "start": "2022-04-01T00:00:00+03:00",
      "end_exclusive": "2025-04-01T00:00:00+03:00"
    }
  }
}
```

One deployment is sufficient for a single-site report. The same normalized
serial cannot be assigned to overlapping Central and Jacros deployments.
The repository's Addis and Jacros daily files are the same MA350-0238 over
overlapping periods, as recorded in the [data pipeline audit](gallery-data-pipeline.md).
They must not be supplied as two independent sites. Establish the Central
deployment from its own metadata before making the two-site comparison.

For the optional optical calculation, add this object with the **actual** input
wavelengths and correction evidence. This Blue/IR example is not a claim that
the paper's final pair has been confirmed:

```json
{
  "source_model": {
    "signal_basis": "absorption",
    "short_wavelength_nm": 470,
    "long_wavelength_nm": 880,
    "correction_reference": "Receipt identifying the applied optical corrections"
  }
}
```

The named workflow fixes fossil/biomass exponents to 1/2, clips the displayed
fraction to [0, 1], and separately reports unbounded estimates and clipping.
It reports the arithmetic mean of hourly fractions and the fraction weighted
by long-wavelength absorption. Neither is automatically the paper's overall
BC-mass source fraction. That comparison remains marked unavailable.

## What the report establishes

The study window is 1 April 2022 through 31 March 2025, represented by the
end-exclusive interval `[2022-04-01, 2025-04-01)` in Addis local time:
26,304 calendar hours. Each deployment's expected-hour denominator is its
declared active interval intersected with that study window. The report also
shows full-study calendar coverage so a short deployment cannot appear to
cover the entire study merely because its own interval is complete.

BC summaries use finite nonnegative, QC-eligible observations. Source-model
coverage is separate and requires finite positive signals at both wavelengths.
Two-site common-hour means use only hours supported at both sites. The paper's
reported means are shown for descriptive comparison: Central 3.73 and Jacros
7.67 µg/m³. Its 93.9% completeness is retained as a published reference; the
precise paper denominator still needs confirmation from the supporting methods.
No discrepancy is converted into a statistical acceptance/failure claim.

The profile uses Bega October–January, Belg February–May and Kiremt June–September.
Seasonal summaries currently include event dates. The profile records the
paper's Meskel and Hidar Sitaten dates, but exact event exclusions must be
resolved before reproducing its event-excluded climatologies.

All report files carry the exact CSV, metadata and profile SHA-256 hashes.
They identify input protocol differences, QC provenance and unresolved methods.
Reports always state that exact paper reproduction is incomplete and that
overlapping observational data is not independent validation.

## Optical model and limitations

With `r = λ_long / λ_short`, the biomass-end-member fraction of absorption at
the long wavelength is:

```text
f_bb = (b_short / b_long - r**alpha_fossil)
       / (r**alpha_biomass - r**alpha_fossil)
```

This is nonlinear in observed AAE. Linear interpolation of AAE between 1 and 2
does not implement the same physical mixture. Fractions below zero or above
one indicate an observation outside the chosen end members; clipping does not
erase that diagnostic. Numerical eligibility is separate from instrument QC.

BCc is already divided by an instrument-specific cross-section. The legacy
analyzer now uses the MA350 attenuation cross-section to reconstruct an
attenuation proxy, and discloses the assumed common multiple-scattering factor.
A wavelength-dependent correction can change the ratio. The low-level helper
also accepts explicitly corrected absorption. Optical component fractions
become equivalent-BC fractions only under an additional shared reference-MAC
assumption; they are not chemical PMF source fractions or unique source labels.
The paper itself notes that high AAE can have sources other than biomass.

The paper's MA350 had no size-selective inlet. Its optical BC should not be
silently relabelled PM2.5 EC, and its BC/PM2.5 comparisons do not create an
independent thermal EC reference for the current FTIR transfer study.

## Remaining evidence and next implementation batch

1. Obtain the [supporting information](https://pubs.acs.org/doi/suppl/10.1021/acsestair.6c00140/suppl_file/ea6c00140_si_001.pdf)
   and confirm exact status, flow, temperature, leak, loading and noise rules;
   the selected wavelength pair; uncertainty/clipping; and temporal completeness.
2. Identify the raw Central and Jacros deployments, their active periods,
   instrument serials and upstream processing receipts. Rebuild the hourly
   source export on those actual inputs and inspect retained/rejected populations.
3. Run the named workflow and explain differences using common hours, source
   settings and population definitions. Preserve each result as a new report.
4. Add event-aligned and event-excluded diurnal/weekly comparisons after confirming
   exact exclusion windows. Keep this separate from filter-window averaging.
5. Consider Jacros wind-sector and transport analyses after establishing the
   meteorological joins. The paper's trajectories/PSCF require additional inputs
   and should be a separate analysis, not a numerical preset applied blindly.

PurpleAir calibration is also a separate work item: the paper's local calibration
needs its own data and validation before being applied to other sites or periods.
The FTIR source-score acceptance, historical target fits and proposed future TOR
validation retain their existing boundaries throughout this integration.
