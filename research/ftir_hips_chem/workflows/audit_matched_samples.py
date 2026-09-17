"""Reproduce the first optical/filter comparison milestone from existing inputs.

This is an audit and candidate-match export, not a validated interval matcher.
The current unified input has sample dates but no actual collection intervals.
Every filter is retained, conflicting values are unresolved, and legacy ±1-day
matches are labelled candidates. No calibration is fitted to these candidates.

Run from the repository root with:
    uv run python research/ftir_hips_chem/workflows/audit_matched_samples.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))
from config import DATA_ROOT, FILTER_DATA_PATH, PROCESSED_SITES_DIR, SITES, WAVELENGTHS_NM
from data_matching import add_base_filter_id
from data_paths import maia_data_root
from outliers import apply_exclusion_flags, get_clean_data

# Explicit names retain measurement meaning and units. Do not treat ChemSpec
# carbon columns as independent reference measurements; see docs/open-items.md.
PARAMETERS = {
    'EC_ftir': ('ftir_ec_ugm3', 'Concentration', 'ug/m3'),
    'OC_ftir': ('ftir_oc_ugm3', 'Concentration', 'ug/m3'),
    'HIPS_Fabs': ('hips_fabs_Mm1', 'Concentration', 'Mm-1'),
    'HIPS_Uncertainty': ('hips_uncertainty_Mm1', 'Uncertainty', 'Mm-1'),
    'HIPS_MDL': ('hips_mdl_Mm1', 'MDL', 'Mm-1'),
    'ChemSpec_EC_PM2.5': ('chemspec_ec_ugm3', 'Concentration', 'ug/m3'),
    'ChemSpec_BC_PM2.5': ('chemspec_bc_ugm3', 'Concentration', 'ug/m3'),
    'ChemSpec_Iron_PM2.5': ('iron_ugm3', 'Concentration', 'ug/m3'),
}


def _json(value):
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False)


def _values(series):
    return sorted(series.dropna().unique().tolist(), key=str)


def _numeric(series):
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


def build_filter_catalog(filters, sites=SITES):
    """One row per site/base ID; retain every input row and flag ambiguities.

    Exact repeated numeric values can supply a consensus value but their row
    count remains visible. Conflicting values never use first/mean aggregation.
    Source row numbers are zero-based positional offsets into the input pickle.
    """
    required = {'Site', 'FilterId', 'SampleDate', 'Parameter', 'Concentration',
                'Concentration_Units', 'MDL', 'Uncertainty', 'FilterType', 'Volume_m3'}
    if not required.issubset(filters):
        raise ValueError(f'Missing filter columns: {sorted(required - set(filters))}')
    if not filters.columns.is_unique:
        raise ValueError('Filter input has duplicate columns')
    evidence = add_base_filter_id(filters).reset_index(drop=True)
    evidence.insert(0, 'source_row', np.arange(len(evidence)))
    if evidence[['Site', 'base_filter_id']].isna().any().any():
        raise ValueError('Missing site/filter identity: resolve before grouping samples')
    evidence['parsed_date'] = pd.to_datetime(evidence.SampleDate, errors='coerce')
    site_names = {spec['code']: name for name, spec in sites.items()}
    if not set(evidence.Site).issubset(site_names):
        raise ValueError('Filter input contains sites absent from the site configuration')
    records = []
    for (site_code, filter_id), group in evidence.groupby(['Site', 'base_filter_id'], sort=True):
        dates = _values(group.parsed_date)
        types = _values(group.FilterType)
        volumes = _values(_numeric(group.Volume_m3))
        record = {
            'site': site_names[site_code], 'site_code': site_code, 'base_filter_id': filter_id,
            'filter_ids': _json(_values(group.FilterId)),
            'date': dates[0] if len(dates) == 1 else pd.NaT,
            'date_values': _json(dates), 'date_conflict': len(dates) > 1,
            'source_rows': _json(group.source_row.tolist()), 'n_source_rows': len(group),
            'n_source_rows_missing_date': int(group.parsed_date.isna().sum()),
            'filter_type': types[0] if len(types) == 1 else None,
            'filter_type_conflict': len(types) > 1,
            'volume_m3': volumes[0] if len(volumes) == 1 else np.nan,
            'volume_conflict': len(volumes) > 1,
            'timezone_from_site_config': sites[site_names[site_code]]['timezone'],
            'sampling_schedule_verified': False,
            'sampling_schedule_reason': 'unified_input_has_no_collection_or_active_intervals',
            'ec_method': 'FTIR prediction; calibration training membership unresolved',
            'independent_ec_reference_available': False,
        }
        for parameter, (column, value_field, units) in PARAMETERS.items():
            rows = group.loc[group.Parameter.eq(parameter)]
            values = _values(_numeric(rows[value_field]))
            unit_values = _values(rows.Concentration_Units)
            units_ok = bool(len(rows)) and unit_values == [units] and rows.Concentration_Units.notna().all()
            record[column] = values[0] if len(values) == 1 and units_ok else np.nan
            record[f'{column}_conflict'] = len(values) > 1
            record[f'{column}_units_valid'] = units_ok
            record[f'{column}_source_rows'] = _json(rows.source_row.tolist())
            record[f'{column}_n_rows'] = len(rows)
        ec_rows = group.loc[group.Parameter.eq('EC_ftir')]
        ec_mdl = _values(_numeric(ec_rows.MDL))
        record['ftir_ec_mdl_ugm3'] = ec_mdl[0] if len(ec_mdl) == 1 else np.nan
        record['ftir_ec_mdl_conflict'] = len(ec_mdl) > 1
        record['ftir_ec_nonpositive'] = bool(record['ftir_ec_ugm3'] <= 0)
        record['ftir_ec_below_mdl'] = bool(record['ftir_ec_ugm3'] < record['ftir_ec_mdl_ugm3'])
        record['has_hips_ftir_pair'] = bool(np.isfinite(record['hips_fabs_Mm1']) and
                                            np.isfinite(record['ftir_ec_ugm3']))
        # Apply the existing registry to ALL original IDs, not an arbitrarily
        # chosen replicate. This retains the registry's date tolerance semantics.
        probes = group[['FilterId', 'parsed_date']].drop_duplicates().rename(
            columns={'FilterId': 'filter_id', 'parsed_date': 'date'})
        flags = apply_exclusion_flags(probes, record['site'])
        record['is_excluded'] = bool(flags.is_excluded.any())
        record['exclusion_reason'] = '; '.join(sorted(set(flags.exclusion_reason) - {''}))
        records.append(record)
    catalog = pd.DataFrame(records)
    if catalog.empty:
        raise ValueError('Filter input is empty')
    catalog['date'] = pd.to_datetime(catalog.date)
    clean_index = get_clean_data(catalog).index
    catalog['eligible_hips_ftir_diagnostic'] = (
        catalog.has_hips_ftir_pair & catalog.index.isin(clean_index) &
        ~catalog.filter_type_conflict & catalog.filter_type.notna())
    return catalog, evidence


def discover_portal_files(portal_dir=None):
    """Use explicit/env/local/Drive paths; do not mix current and archived exports."""
    if portal_dir is None:
        configured = os.environ.get('AETHMODULAR_PORTAL_DIR')
        local = DATA_ROOT / 'Filter Data' / 'SPARTAN portal downloads'
        if configured:
            portal_dir = Path(configured).expanduser()
        elif local.is_dir():
            portal_dir = local
        else:
            try:
                portal_dir = maia_data_root() / 'EC-HIPS-Aeth Comparison/Data/SPARTAN portal downloads'
            except FileNotFoundError:
                return []
    root = Path(portal_dir)
    return sorted(root.glob('*/FilterBased_ChemSpecPM25_*.csv'))


def load_portal_metadata(paths, sites=SITES):
    """Retain portal measurement codes and reported windows, with source-row links.

    Hours_sampled equal to elapsed UTC hours is consistent with continuous
    collection, but does not independently verify an actual on/off log. Local
    clock times ambiguous/nonexistent at DST transitions remain unresolved.
    """
    timing = [f'{edge}_{part}_local' for edge in ['Start', 'End']
              for part in ['Year', 'Month', 'Day', 'hour']] + ['Hours_sampled']
    required = {'Site_Code', 'Filter_ID', 'Parameter_Code', 'Parameter_Name',
                'Collection_Description', 'Conditions', *timing}
    frames = []
    for path in paths:
        frame = pd.read_csv(path, skiprows=3, skipinitialspace=True)
        if not required.issubset(frame):
            raise ValueError(f'Unsupported portal schema: {path}')
        for column in frame.select_dtypes('object'):
            frame[column] = frame[column].str.strip()
        frame = frame.rename(columns={'Filter_ID': 'FilterId'})
        frame = add_base_filter_id(frame)
        frame['portal_source_file'] = str(Path(path).resolve())
        # Zero-based data row after the three preamble lines and header.
        frame['portal_source_row'] = np.arange(len(frame))
        frames.append(frame)
    metadata_columns = ['site_code', 'base_filter_id', 'portal_interval_start_utc',
        'portal_interval_end_utc', 'portal_hours_sampled', 'portal_elapsed_hours',
        'portal_timing_conflict', 'portal_timing_status', 'portal_continuous_schedule_consistent',
        'portal_collection_descriptions', 'portal_conditions', 'portal_source_links']
    if not frames:
        return pd.DataFrame(columns=metadata_columns), pd.DataFrame()
    evidence = pd.concat(frames, ignore_index=True)
    code_config = {config['code']: config for config in sites.values()}
    records = []
    for (site_code, filter_id), group in evidence.groupby(['Site_Code', 'base_filter_id']):
        if site_code not in code_config:
            raise ValueError(f'Unconfigured portal site: {site_code}')
        variants = group[timing].drop_duplicates()
        start = end = pd.NaT
        hours = elapsed = np.nan
        conflict = len(variants) != 1
        if not conflict:
            row = variants.iloc[0]
            timestamps = []
            for edge in ['Start', 'End']:
                try:
                    parts = [float(row[f'{edge}_{part}_local']) for part in ['Year', 'Month', 'Day']]
                    hour = float(row[f'{edge}_hour_local'])
                    if any(not value.is_integer() for value in parts) or not 0 <= hour <= 24:
                        raise ValueError('Invalid local date/hour components')
                    date = pd.Timestamp(year=int(row[f'{edge}_Year_local']),
                                        month=int(row[f'{edge}_Month_local']),
                                        day=int(row[f'{edge}_Day_local']))
                    stamp = date + pd.to_timedelta(hour, unit='h')
                    timestamps.append(stamp.tz_localize(code_config[site_code]['timezone'],
                        ambiguous='NaT', nonexistent='NaT').tz_convert('UTC'))
                except (ValueError, TypeError, OverflowError):
                    timestamps.append(pd.NaT)
            start, end = timestamps
            hours = pd.to_numeric(row.Hours_sampled, errors='coerce')
            elapsed = (end - start).total_seconds() / 3600
        continuous = bool(elapsed > 0 and hours > 0 and np.isclose(hours, elapsed, atol=0.05, rtol=0))
        if conflict:
            status = 'conflicting_portal_timing_rows'
        elif not np.isfinite(elapsed) or elapsed <= 0:
            status = 'invalid_or_nonpositive_collection_window'
        elif not np.isfinite(hours) or hours <= 0:
            status = 'invalid_or_missing_sampled_hours'
        elif not continuous:
            status = 'sampled_hours_differ_from_elapsed_window; active_periods_needed'
        else:
            status = 'reported_hours_consistent_with_continuous_window; observation_log_unverified'
        records.append({
            'site_code': site_code, 'base_filter_id': filter_id,
            'portal_interval_start_utc': start, 'portal_interval_end_utc': end,
            'portal_hours_sampled': hours, 'portal_elapsed_hours': elapsed,
            'portal_timing_conflict': conflict,
            'portal_timing_status': status,
            # Exported hours may be rounded to a tenth of an hour. This test
            # only labels consistency; it never switches validation on.
            'portal_continuous_schedule_consistent': continuous,
            'portal_collection_descriptions': _json(_values(group.Collection_Description)),
            'portal_conditions': _json(_values(group.Conditions)),
            'portal_source_links': _json(group[['portal_source_file', 'portal_source_row']].to_dict('records')),
        })
    metadata = pd.DataFrame(records, columns=metadata_columns)
    for column in ['portal_interval_start_utc', 'portal_interval_end_utc']:
        metadata[column] = pd.to_datetime(metadata[column], utc=True)
    return metadata, evidence


def attach_portal_metadata(catalog, metadata):
    result = catalog.merge(metadata, on=['site_code', 'base_filter_id'], how='left', validate='one_to_one')
    result['portal_window_available'] = (
        result.portal_interval_start_utc.notna() & result.portal_interval_end_utc.notna() &
        (result.portal_elapsed_hours > 0))
    result['portal_start_matches_sample_date'] = pd.Series(pd.NA, index=result.index, dtype='boolean')
    for site, indices in result.groupby('site').groups.items():
        local_start = pd.to_datetime(result.loc[indices, 'portal_interval_start_utc'], utc=True).dt.tz_convert(
            SITES[site]['timezone']).dt.tz_localize(None).dt.normalize()
        known = local_start.notna() & result.loc[indices, 'date'].notna()
        result.loc[indices[known], 'portal_start_matches_sample_date'] = (
            local_start.loc[known] == result.loc[indices[known], 'date'])
    result.loc[result.portal_window_available, 'sampling_schedule_reason'] = (
        'portal_collection_window_recovered; active_sampling_log_not_verified')
    return result


def audit_aeth_frame(frame, site, config):
    """Inspect saved aggregates without assigning unsupported interval semantics."""
    timestamp_positions = [i for i, col in enumerate(frame.columns) if col == 'datetime_local']
    timestamps = [frame.iloc[:, i] for i in timestamp_positions
                  if isinstance(frame.iloc[:, i].dtype, pd.DatetimeTZDtype)]
    hours = timestamps[0].dt.tz_convert(config['timezone']).dt.hour.value_counts().to_dict() if len(timestamps) == 1 else {}
    duplicate_names = frame.columns[frame.columns.duplicated()].tolist()
    coverage = _numeric(frame['data_completeness_pct']) if 'data_completeness_pct' in frame else pd.Series(dtype=float)
    days = pd.to_datetime(frame['day_9am'], errors='coerce')
    return {
        'site': site, 'site_code': config['code'], 'n_rows': len(frame),
        'date_min': str(days.min()), 'date_max': str(days.max()),
        'raw_ir_available_days': int(_numeric(frame['IR BCc']).notna().sum()),
        'duplicate_column_names': duplicate_names,
        'duplicate_days': int(days.duplicated(keep=False).sum()),
        'invalid_days': int(days.isna().sum()),
        'local_timestamp_hours': {str(k): int(v) for k, v in sorted(hours.items())},
        'coverage_column_present': 'data_completeness_pct' in frame,
        'coverage_outside_0_100': int(((coverage < 0) | (coverage > 100)).sum()),
        'coverage_provenance': frame.attrs.get('coverage_basis', 'unverified'),
        'resampling_version': frame.attrs.get('resampling_version', 'unverified'),
        'device_ids': _values(frame.Device_ID) if 'Device_ID' in frame else [],
        'issue': 'saved_aggregate_not_verified_against_raw_observation_intervals',
    }


def attach_legacy_candidates(catalog, aeth_by_site, tolerance_days=1):
    """Expose historical date-window candidates and row links, never validate them."""
    if not isinstance(tolerance_days, int) or tolerance_days < 0:
        raise ValueError('tolerance_days must be a nonnegative integer')
    result = catalog.copy()
    links = []
    result['candidate_ir_ebc_ugm3'] = np.nan
    result['candidate_ir_smoothed_ebc_ugm3'] = np.nan
    result['candidate_aeth_days'] = 0
    result['candidate_aeth_valid_ir_days'] = 0
    result['candidate_method'] = f'legacy_date_window_plus_minus_{tolerance_days}_days_UNVALIDATED'
    result['candidate_ir_wavelength_nm'] = WAVELENGTHS_NM['IR']
    result['eligible_aeth_hips_calibration'] = False
    result['eligible_aeth_ec_calibration'] = False
    result['observed_coverage_fraction'] = np.nan
    result['aeth_ineligibility_reason'] = (
        'collection_intervals_unverified; observed_coverage_unverified; '
        'wavelength_conversion_unverified; particle_size_and_reference_conditions_unverified')
    for site, indexes in result.groupby('site').groups.items():
        frame = aeth_by_site.get(site)
        if frame is None:
            result.loc[indexes, 'aeth_ineligibility_reason'] += '; missing_aeth_file'
            continue
        days = pd.to_datetime(frame.day_9am, errors='coerce')
        if days.dt.tz is not None:
            raise ValueError('Legacy candidate dates must be timezone-naive calendar labels')
        raw = _numeric(frame['IR BCc'])
        smooth = _numeric(frame['IR BCc smoothed']) if 'IR BCc smoothed' in frame else pd.Series(np.nan, index=frame.index)
        for index in indexes:
            sample = result.loc[index]
            mask = (days >= sample.date - pd.Timedelta(days=tolerance_days)) & (days <= sample.date + pd.Timedelta(days=tolerance_days))
            positions = np.flatnonzero(mask.to_numpy())
            result.loc[index, 'candidate_aeth_days'] = len(positions)
            result.loc[index, 'candidate_aeth_valid_ir_days'] = int(raw.iloc[positions].notna().sum())
            result.loc[index, 'candidate_ir_ebc_ugm3'] = raw.iloc[positions].mean() / 1000
            result.loc[index, 'candidate_ir_smoothed_ebc_ugm3'] = smooth.iloc[positions].mean() / 1000
            if pd.isna(sample.date):
                result.loc[index, 'aeth_ineligibility_reason'] += '; missing_or_conflicting_sample_date'
            elif not raw.iloc[positions].notna().any():
                result.loc[index, 'aeth_ineligibility_reason'] += '; no_finite_ir_candidate'
            for pos in positions:
                links.append({
                    'site': site, 'base_filter_id': sample.base_filter_id,
                    'aeth_source_row': int(pos), 'aeth_day_label': days.iloc[pos],
                    'raw_ir_ebc_ngm3': raw.iloc[pos], 'smoothed_ir_ebc_ngm3': smooth.iloc[pos],
                    'used_in_raw_candidate_mean': bool(pd.notna(raw.iloc[pos])),
                    'match_status': 'UNVALIDATED_DATE_CANDIDATE',
                })
    columns = ['site', 'base_filter_id', 'aeth_source_row', 'aeth_day_label',
               'raw_ir_ebc_ngm3', 'smoothed_ir_ebc_ngm3', 'used_in_raw_candidate_mean', 'match_status']
    return result, pd.DataFrame(links, columns=columns)


def selection_summary(catalog):
    rows = []
    for site, group in catalog.groupby('site', sort=False):
        rows.append({
            'site': site, 'physical_filters': len(group),
            'finite_ftir_ec': int(group.ftir_ec_ugm3.notna().sum()),
            'finite_hips': int(group.hips_fabs_Mm1.notna().sum()),
            'hips_ftir_pairs': int(group.has_hips_ftir_pair.sum()),
            'registry_flagged': int(group.is_excluded.sum()),
            'hips_ftir_diagnostic_eligible': int(group.eligible_hips_ftir_diagnostic.sum()),
            'hips_with_ir_candidate': int((group.hips_fabs_Mm1.notna() & group.candidate_ir_ebc_ugm3.notna()).sum()),
            'ftir_with_ir_candidate': int((group.ftir_ec_ugm3.notna() & group.candidate_ir_ebc_ugm3.notna()).sum()),
            'missing_sample_date': int(group.date.isna().sum()),
            'conflicting_chemspec_ec': int(group.chemspec_ec_ugm3_conflict.sum()),
            'portal_windows_recovered': int(group.portal_window_available.sum()),
            'hips_with_portal_window': int((group.hips_fabs_Mm1.notna() & group.portal_window_available).sum()),
            'portal_start_date_disagreements': int(group.portal_start_matches_sample_date.eq(False).sum()),
            'portal_timing_discrepancies': int(group.portal_continuous_schedule_consistent.eq(False).sum()),
            'validated_aeth_hips': int(group.eligible_aeth_hips_calibration.sum()),
        })
    return pd.DataFrame(rows)


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def _git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO_ROOT, text=True).strip()


def write_report(path, summary, inventory):
    selection = summary[['site', 'physical_filters', 'finite_ftir_ec', 'finite_hips',
                         'hips_ftir_diagnostic_eligible', 'hips_with_ir_candidate']].rename(columns={
        'site': 'Site', 'physical_filters': 'Filter IDs', 'finite_ftir_ec': 'FTIR EC',
        'finite_hips': 'HIPS', 'hips_ftir_diagnostic_eligible': 'Filter-only pairs',
        'hips_with_ir_candidate': 'HIPS–IR candidates'})
    readiness = summary[['site', 'missing_sample_date', 'conflicting_chemspec_ec',
        'portal_windows_recovered', 'hips_with_portal_window', 'portal_timing_discrepancies',
        'portal_start_date_disagreements']].rename(columns={
        'site': 'Site', 'missing_sample_date': 'Undated', 'conflicting_chemspec_ec': 'EC conflicts',
        'portal_windows_recovered': 'Portal windows', 'hips_with_portal_window': 'HIPS + window',
        'portal_timing_discrepancies': 'Timing issues', 'portal_start_date_disagreements': 'Date conflicts'})
    lines = [
        '# Matched-sample readiness audit', '',
        'This run audits existing local inputs. It exports a filter catalog and legacy date-window '
        '**candidates**, not a validated interval-matched calibration dataset. '
        'All original filter measurement rows are retained in `filter_measurements.parquet`; '
        'source row numbers refer to the hashed input pickles in `manifest.json`.', '',
        '## Sample selection', '', selection.to_markdown(index=False), '',
        '`Filter IDs` counts distinct site/base FilterId pairs, including undated and chemistry-only samples. '
        '`Filter-only pairs` requires a finite, unit-checked same-filter pair, an unambiguous '
        'filter type, and no existing registry exclusion. It is a measurement comparison, not independent '
        'validation of FTIR EC. Candidate counts precede registry exclusions and do not assert coverage. '
        'Low/nonpositive EC and EC below its stored MDL are flagged, not deleted. '
        'Empirical concentration thresholds are not applied in this inventory.', '',
        '## Metadata readiness', '', readiness.to_markdown(index=False), '',
        'Portal windows count positive-duration reported collection windows, not verified active sampling '
        'schedules. Timing issues count reported sampled hours inconsistent with elapsed duration or '
        'invalid windows. These flags and the original parameter codes remain in the portal exports. '
        'A zero count of portal windows means no positive-duration window was recovered from the files '
        'loaded in this run; it does not prove metadata are unavailable elsewhere.', '',
        '## Saved aethalometer inputs', '',
    ]
    for item in inventory:
        lines += [f"- **{item['site']}**: {item['n_rows']} saved days; {item['raw_ir_available_days']} with finite IR. "
                  f"Timestamp-hour counts: {item['local_timestamp_hours']}. Duplicate columns: "
                  f"{item['duplicate_column_names']}. Coverage present: {item['coverage_column_present']}; "
                  f"outside 0–100%: {item['coverage_outside_0_100']}."]
    lines += [
        '', '## What prevents validated optical calibration', '',
        '1. The unified input has `SampleDate`, `FilterType`, and sometimes `Volume_m3`, '
        'but no collection start/end or active sampling intervals. Available original portal exports '
        'restore reported local collection start/end times, hours sampled, collection descriptions and '
        'reference conditions. Their UTC windows are joined by physical filter ID and source-linked. '
        '`portal_continuous_schedule_consistent` tests whether reported sampled hours agree with '
        'elapsed UTC hours within 0.05 h; it does not establish an observed on/off schedule. '
        'Recover missing sites and resolve timing discrepancies before calibration.',
        '2. Historical date-window averaging uses up to three daily values, without measuring observed '
        'coverage over collection intervals. Existing daily files cannot recover within-day gaps or '
        'interpolation history. The resampler source has been corrected; existing pickles have not been '
        'regenerated and remain unverified. Rebuild from original observations after schedule review.',
        '3. HIPS absorption is stored in Mm⁻¹; aethalometer BCc is stored as eBC in ng/m³. '
        'This export only converts ng/m³ to µg/m³ and labels the IR wavelength from config. '
        'It does not multiply by the HIPS MAC constant to invent aethalometer absorption. Verify '
        'instrument conversion coefficients, loading/flow corrections, size fraction, reference '
        'conditions, and HIPS wavelength before optical comparison.',
        '4. The repository provenance investigation identifies ChemSpec EC as the FTIR EC product '
        'and ChemSpec BC as HIPS Fabs/10, subject to rounding. Neither is an independent reference. '
        'This input also contains conflicting ChemSpec EC records within physical filters; these '
        'remain unresolved with original rows preserved. Do not average them or silently take the first.',
        '5. HIPS uncertainty and MDL come from their own parameter rows. Their reported values are '
        'retained. Confirm uncertainty semantics before choosing an EIV estimator; no arbitrary '
        'equal-error assumption or AIRSpec RMSE is applied to these FTIR predictions.', '',
        '## Reproduce and inspect', '',
        '```bash', 'uv run aeth doctor',
        'uv run python research/ftir_hips_chem/workflows/audit_matched_samples.py', '```', '',
        'Portal files resolve through `--portal-dir`, `AETHMODULAR_PORTAL_DIR`, a local '
        '`Filter Data/SPARTAN portal downloads` copy, or the existing Drive resolver. '
        'Only current site-level ChemSpec exports are loaded; archives are not mixed in. '
        'Use `--skip-portal` for a local-pickle-only run. Every loaded portal CSV is hashed '
        'and its original parameter codes and source rows are retained.', '',
        'Read any table with `pd.read_parquet(...)`. Join the catalog to candidate links on '
        '`site, base_filter_id`; dereference `aeth_source_row` against the corresponding input '
        'pickle with `.iloc`. Filter measurement offsets are recorded per parameter and in '
        '`source_rows`. Parquet evidence keeps numbers as numbers and serializes original object '
        'fields to strings for stable storage; the unchanged input pickle remains the source of truth.', '',
        '## Next milestone and acceptance checks', '',
        'Resolve remaining filter intervals and intermittent duty cycles, regenerate observed channel '
        'coverage from timestamped data, then replace candidate means with interval-overlap means. '
        'If schedules or flow records are unavailable, state that limitation and define a justified '
        'sensitivity analysis rather than declaring the date match validated. Preserve separate '
        'eligibility flags for each comparison.', '',
        'Once matching and units pass review: compare optical absorption at an aligned wavelength; '
        'describe absorption versus FTIR-predicted EC separately; validate proposed adjustments with '
        'calendar blocks keeping each physical filter entirely in one fold. Report bias, MAE, slopes '
        'with appropriate measurement-error treatment and uncertainty, then predefined coverage, '
        'smoothing, timing, low-EC and influence sensitivity checks. The current run makes no '
        'numerical calibration or mechanism claims.', '',
    ]
    path.write_text('\n'.join(lines))


def run(output_dir, filter_path=FILTER_DATA_PATH, sites_dir=PROCESSED_SITES_DIR, portal_files=()):
    output_dir = Path(output_dir).resolve()
    input_paths = [Path(filter_path)] + [Path(sites_dir) / config['file'] for config in SITES.values()] + list(portal_files)
    input_hashes = [_hash(path) for path in input_paths]
    filters = pd.read_pickle(filter_path)
    aeth = {site: pd.read_pickle(Path(sites_dir) / config['file']) for site, config in SITES.items()}
    catalog, evidence = build_filter_catalog(filters)
    portal_metadata, portal_evidence = load_portal_metadata(portal_files)
    catalog = attach_portal_metadata(catalog, portal_metadata)
    catalog, links = attach_legacy_candidates(catalog, aeth)
    inventory = [audit_aeth_frame(aeth[site], site, config) for site, config in SITES.items()]
    summary = selection_summary(catalog)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Some source object columns mix numeric calibration IDs and strings.
    # Normalize only storage types; do not overwrite the input measurements.
    for column in evidence.select_dtypes(include='object'):
        evidence[column] = evidence[column].astype('string')
    tables = {'matched_sample_candidates': catalog, 'filter_measurements': evidence,
              'aeth_candidate_links': links, 'selection_summary': summary,
              # Always replace these, including on --skip-portal runs, so a
              # previous run's portal evidence cannot look like current output.
              'portal_filter_metadata': portal_metadata, 'portal_measurement_rows': portal_evidence}
    for name, table in tables.items():
        table.to_parquet(output_dir / f'{name}.parquet', index=False)
    (output_dir / 'aeth_inventory.json').write_text(json.dumps(inventory, indent=2))
    write_report(output_dir / 'selection_report.md', summary, inventory)
    # Hash imported analysis sources as well as the Git revision: a dirty tree
    # and a dependency lock hash alone do not identify the code that was run.
    code_paths = sorted(SCRIPTS_DIR.rglob('*.py')) + [Path(__file__),
        REPO_ROOT / 'scripts/pipelines/create_9am_resampled_datasets.py',
        REPO_ROOT / 'pyproject.toml', REPO_ROOT / 'uv.lock', REPO_ROOT / 'AGENTS.md']
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'AUDIT_ONLY_UNVALIDATED_CANDIDATES',
        'git_revision': _git('rev-parse', 'HEAD'), 'git_status': _git('status', '--short'),
        'python': platform.python_version(), 'executable': sys.executable,
        'packages': {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()},
        'configuration': {'sites': SITES, 'legacy_date_tolerance_days': 1,
                          'wavelengths_nm': WAVELENGTHS_NM, 'exclusion_date_tolerance_days': 1,
                          'portal_hours_consistency_tolerance_h': 0.05,
                          'portal_files': [str(Path(path).resolve()) for path in portal_files],
                          'threshold_flags_applied': False, 'missing_coverage_policy': 'unknown_not_zero'},
        'inputs': input_hashes, 'code': [_hash(path) for path in code_paths],
        'outputs': [_hash(output_dir / name) for name in [
            *(f'{key}.parquet' for key in tables), 'selection_report.md', 'aeth_inventory.json']],
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    print(f'\nAudit written to {output_dir / "selection_report.md"}')
    return catalog, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DATA_ROOT / 'output/tables/matched_sample_audit')
    parser.add_argument('--filter-path', type=Path, default=FILTER_DATA_PATH)
    parser.add_argument('--sites-dir', type=Path, default=PROCESSED_SITES_DIR)
    parser.add_argument('--portal-dir', type=Path, help='Directory with current site-level portal CSV exports.')
    parser.add_argument('--skip-portal', action='store_true', help='Audit local pickles without portal metadata.')
    args = parser.parse_args()
    portal_files = [] if args.skip_portal else discover_portal_files(args.portal_dir)
    run(args.output_dir, args.filter_path, args.sites_dir, portal_files)
