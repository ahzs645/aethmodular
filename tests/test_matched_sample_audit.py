"""Protect filter identity, conflicting records, units and audit-only eligibility."""

import json

import numpy as np
import pandas as pd
import pytest

from research.ftir_hips_chem.workflows.audit_matched_samples import (
    attach_legacy_candidates,
    audit_aeth_frame,
    build_filter_catalog,
    attach_portal_metadata,
    load_portal_metadata,
)
from research.ftir_hips_chem.scripts.config import SITES


def measurement(filter_id, parameter, concentration, date='2024-01-10', **overrides):
    row = {'Site': 'ETAD', 'FilterId': filter_id, 'SampleDate': date,
           'Parameter': parameter, 'Concentration': concentration,
           'Concentration_Units': 'Mm-1' if parameter.startswith('HIPS') else 'ug/m3',
           'MDL': np.nan, 'Uncertainty': np.nan, 'FilterType': 'PM2.5', 'Volume_m3': 7.2}
    return {**row, **overrides}


def test_distinct_filters_on_same_date_and_undated_samples_survive():
    frame = pd.DataFrame([
        measurement('ETAD-0001-1', 'EC_ftir', 1.),
        measurement('ETAD-0001', 'HIPS_Fabs', 10.),
        measurement('ETAD-0002-1', 'EC_ftir', 9.),
        measurement('ETAD-0003-1', 'EC_ftir', 0., date=None),
    ])
    catalog, evidence = build_filter_catalog(frame)
    assert len(evidence) == len(frame)
    assert len(catalog) == 3
    assert catalog.base_filter_id.tolist() == ['ETAD-0001', 'ETAD-0002', 'ETAD-0003']
    assert catalog.eligible_hips_ftir_diagnostic.tolist() == [True, False, False]
    assert catalog.ftir_ec_nonpositive.tolist() == [False, False, True]
    assert pd.isna(catalog.date.iloc[-1])
    assert json.loads(catalog.source_rows.iloc[0]) == [0, 1]


def test_conflicting_parameter_values_do_not_choose_first_or_mean():
    frame = pd.DataFrame([
        measurement('ETAD-0001', 'ChemSpec_EC_PM2.5', 3.),
        measurement('ETAD-0001', 'ChemSpec_EC_PM2.5', 0.06),
        measurement('ETAD-0001-1', 'EC_ftir', 3.),
        measurement('ETAD-0001-1', 'HIPS_Fabs', 10.),
    ])
    catalog, _ = build_filter_catalog(frame)
    row = catalog.iloc[0]
    assert row.chemspec_ec_ugm3_conflict
    assert np.isnan(row.chemspec_ec_ugm3)
    assert row.chemspec_ec_ugm3_n_rows == 2
    # A conflict in an auxiliary product must not require listwise deletion.
    assert row.eligible_hips_ftir_diagnostic
    assert not row.independent_ec_reference_available


def test_exact_repeats_are_audited_without_increasing_sample_count():
    row = measurement('ETAD-0001', 'EC_ftir', 2.)
    catalog, evidence = build_filter_catalog(pd.DataFrame([row, row]))
    assert len(catalog) == 1
    assert len(evidence) == 2
    assert catalog.ftir_ec_ugm3.iloc[0] == 2
    assert not catalog.ftir_ec_ugm3_conflict.iloc[0]
    assert catalog.ftir_ec_ugm3_n_rows.iloc[0] == 2


def test_wrong_units_cannot_supply_a_comparison_value():
    frame = pd.DataFrame([
        measurement('ETAD-0001', 'EC_ftir', 2.),
        measurement('ETAD-0001', 'HIPS_Fabs', 20., Concentration_Units='ug/m3'),
    ])
    catalog, _ = build_filter_catalog(frame)
    assert np.isnan(catalog.hips_fabs_Mm1.iloc[0])
    assert not catalog.eligible_hips_ftir_diagnostic.iloc[0]


def test_sibling_hips_uncertainty_and_mdl_are_retained():
    frame = pd.DataFrame([
        measurement('ETAD-0001', 'HIPS_Fabs', 20.),
        measurement('ETAD-0001', 'HIPS_Uncertainty', 2., Uncertainty=2.),
        measurement('ETAD-0001', 'HIPS_MDL', 1., MDL=1.),
    ])
    catalog, _ = build_filter_catalog(frame)
    assert catalog.hips_uncertainty_Mm1.iloc[0] == 2
    assert catalog.hips_mdl_Mm1.iloc[0] == 1


def test_date_conflict_prevents_aeth_candidate_but_keeps_same_filter_pair():
    frame = pd.DataFrame([
        measurement('ETAD-0001', 'HIPS_Fabs', 20., date='2024-01-11'),
        measurement('ETAD-0001', 'EC_ftir', 2.),
    ])
    catalog, _ = build_filter_catalog(frame)
    assert catalog.date_conflict.iloc[0]
    assert catalog.eligible_hips_ftir_diagnostic.iloc[0]
    aeth = pd.DataFrame({'day_9am': pd.to_datetime(['2024-01-10']), 'IR BCc': [1000.]})
    result, links = attach_legacy_candidates(catalog, {'Addis_Ababa': aeth})
    assert links.empty
    assert np.isnan(result.candidate_ir_ebc_ugm3.iloc[0])


def test_legacy_candidates_keep_row_links_and_do_not_claim_validation():
    frame = pd.DataFrame([measurement('ETAD-0001', 'HIPS_Fabs', 20.)])
    catalog, _ = build_filter_catalog(frame)
    aeth = pd.DataFrame({'day_9am': pd.to_datetime(['2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12']),
                         'IR BCc': [1000., np.nan, 3000., 9000.]})
    result, links = attach_legacy_candidates(catalog, {'Addis_Ababa': aeth})
    assert result.candidate_ir_ebc_ugm3.iloc[0] == 2.
    assert result.candidate_aeth_valid_ir_days.iloc[0] == 2
    assert links.aeth_source_row.tolist() == [0, 1, 2]
    assert links.used_in_raw_candidate_mean.tolist() == [True, False, True]
    assert not result.eligible_aeth_hips_calibration.any()
    assert result.observed_coverage_fraction.isna().all()
    assert not result.sampling_schedule_verified.any()


def test_registry_exclusions_match_original_replicate_ids():
    frame = pd.DataFrame([
        measurement('INDH-0172-4', 'EC_ftir', 60.64, date='2024-06-21', Site='INDH'),
        measurement('INDH-0172', 'HIPS_Fabs', 20., date='2024-06-21', Site='INDH'),
    ])
    catalog, _ = build_filter_catalog(frame)
    assert catalog.has_hips_ftir_pair.iloc[0]
    assert catalog.is_excluded.iloc[0]
    assert 'Extreme FTIR EC' in catalog.exclusion_reason.iloc[0]
    assert not catalog.eligible_hips_ftir_diagnostic.iloc[0]


def test_audit_detects_duplicate_timestamp_fields_and_impossible_coverage():
    frame = pd.DataFrame([[pd.Timestamp('2024-01-01 15:00', tz='Asia/Shanghai'), 123.,
                           pd.Timestamp('2024-01-01'), 101., 1000.]],
                         columns=['datetime_local', 'datetime_local', 'day_9am',
                                  'data_completeness_pct', 'IR BCc'])
    result = audit_aeth_frame(frame, 'Beijing', SITES['Beijing'])
    assert result['local_timestamp_hours'] == {'15': 1}
    assert result['duplicate_column_names'] == ['datetime_local']
    assert result['coverage_outside_0_100'] == 1


def test_missing_identity_is_not_silently_dropped_or_grouped():
    with pytest.raises(ValueError, match='identity'):
        build_filter_catalog(pd.DataFrame([measurement(None, 'EC_ftir', 2.)]))


def portal_row(**overrides):
    return {**{'Site_Code': 'ETAD', 'Filter_ID': ' ETAD-0001 ',
        'Start_Year_local': 2024, 'Start_Month_local': 1, 'Start_Day_local': 10,
        'Start_hour_local': 9, 'End_Year_local': 2024, 'End_Month_local': 1,
        'End_Day_local': 11, 'End_hour_local': 9, 'Hours_sampled': 24,
        'Parameter_Code': 1, 'Parameter_Name': 'BC PM2.5', 'Value': 2.,
        'Collection_Description': 'Sampler with cyclone', 'Conditions': 'Ambient local'}, **overrides}


def portal_file(tmp_path, rows):
    path = tmp_path / 'portal.csv'
    path.write_text('File Updated: test\nData version test\nSite: test\n' +
                    pd.DataFrame(rows).to_csv(index=False))
    return path


def test_portal_windows_recover_by_id_without_claiming_observed_schedule(tmp_path):
    path = portal_file(tmp_path, [portal_row(), portal_row(Parameter_Code=2)])
    metadata, evidence = load_portal_metadata([path])
    assert len(metadata) == 1
    assert len(evidence) == 2
    row = metadata.iloc[0]
    assert row.portal_interval_start_utc == pd.Timestamp('2024-01-10 06:00', tz='UTC')
    assert row.portal_continuous_schedule_consistent
    assert len(json.loads(row.portal_source_links)) == 2
    catalog, _ = build_filter_catalog(pd.DataFrame([measurement('ETAD-0001', 'EC_ftir', 1.)]))
    result = attach_portal_metadata(catalog, metadata)
    assert result.portal_window_available.iloc[0]
    assert result.portal_start_matches_sample_date.iloc[0]
    assert not result.sampling_schedule_verified.iloc[0]


def test_portal_intermittency_and_conflicts_remain_explicit(tmp_path):
    path = portal_file(tmp_path, [portal_row(Hours_sampled=22.8),
        portal_row(Filter_ID='ETAD-0002'),
        portal_row(Filter_ID='ETAD-0002', Start_Day_local=9)])
    metadata, _ = load_portal_metadata([path])
    assert not metadata.portal_continuous_schedule_consistent.any()
    assert metadata.portal_timing_conflict.tolist() == [False, True]
    assert metadata.portal_interval_start_utc.isna().tolist() == [False, True]


def test_portal_dst_duration_uses_utc_not_naive_clock_difference(tmp_path):
    path = portal_file(tmp_path, [portal_row(Site_Code='USPA', Filter_ID='USPA-0001',
        Start_Month_local=3, Start_Day_local=9, End_Month_local=3, End_Day_local=10,
        Hours_sampled=24)])
    metadata, _ = load_portal_metadata([path])
    assert metadata.portal_elapsed_hours.iloc[0] == 23
    assert not metadata.portal_continuous_schedule_consistent.iloc[0]


def test_no_portal_inputs_keeps_all_samples_and_unknown_timing():
    catalog, _ = build_filter_catalog(pd.DataFrame([measurement('ETAD-0001', 'EC_ftir', 1.)]))
    metadata, evidence = load_portal_metadata([])
    result = attach_portal_metadata(catalog, metadata)
    assert len(result) == 1
    assert evidence.empty
    assert not result.portal_window_available.any()
    assert result.portal_start_matches_sample_date.isna().all()


def test_zero_length_and_multiday_portal_windows_are_not_daily_schedules(tmp_path):
    path = portal_file(tmp_path, [portal_row(End_Day_local=10),
                                portal_row(Filter_ID='ETAD-0002', End_Day_local=18)])
    metadata, _ = load_portal_metadata([path])
    assert metadata.portal_elapsed_hours.tolist() == [0, 192]
    assert not metadata.portal_continuous_schedule_consistent.any()
    assert metadata.portal_timing_status.iloc[0] == 'invalid_or_nonpositive_collection_window'
    catalog, _ = build_filter_catalog(pd.DataFrame([measurement('ETAD-0001', 'EC_ftir', 1.)]))
    result = attach_portal_metadata(catalog, metadata)
    assert not result.portal_window_available.any()
