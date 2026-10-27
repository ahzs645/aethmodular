"""Offline checks for the AERONET fetch (parsing) and the gallery export (window means)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load('fetch_aeronet_spartan', 'research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py')
export = load('export_aeronet', 'gallery/data/export_aeronet.py')

DOWNLOAD = """AERONET Data Download (Version 3 Direct Sun)
AERONET Version 3
Version 3: AOD Level 1.5
The following data are cloud cleared and quality controls have been applied.
Contact: PI=Someone
AERONET_Site,Date(dd:mm:yyyy),Time(hh:mm:ss),Day_of_Year,AOD_500nm,AOD_440nm,440-870_Angstrom_Exponent
AAU_Jackros_ET,07:12:2022,04:10:50,341,0.080,-999.,1.10
AAU_Jackros_ET,07:12:2022,05:10:50,341,0.100,0.120,-999.000000
"""


def test_parse_keeps_named_columns_and_drops_missing_codes():
    out = fetch.parse(DOWNLOAD, {'AOD_500nm': 'aod500', 'AOD_440nm': 'aod440', 'Precipitable_Water(cm)': 'pw_cm'})
    assert list(out.columns) == ['station', 'time_utc', 'aod500', 'aod440', 'pw_cm']
    assert out['time_utc'].iloc[0] == pd.Timestamp('2022-12-07 04:10:50')
    assert out['aod500'].tolist() == [0.08, 0.1]
    assert np.isnan(out['aod440'].iloc[0]) and out['aod440'].iloc[1] == 0.12
    assert out['pw_cm'].isna().all()          # a column the product lacks is all missing


def test_parse_of_an_empty_download_has_the_columns():
    out = fetch.parse('AERONET Data Download\nno data for this period\n', {'AOD_500nm': 'aod500'})
    assert out.empty and list(out.columns) == ['station', 'time_utc', 'aod500']


def test_window_days_lists_every_calendar_day_a_window_touches():
    windows = pd.DataFrame({'start': [pd.Timestamp('2023-01-01 09:00')], 'end': [pd.Timestamp('2023-01-02 09:00')]})
    assert fetch.window_days(windows) == {'2023-01-01', '2023-01-02'}


def test_window_means_average_only_the_retrievals_inside_each_window():
    points = pd.DataFrame({
        'time_utc': pd.to_datetime(['2023-01-01 06:00', '2023-01-01 10:00', '2023-01-01 14:00', '2023-01-02 10:00']),
        'aod500': [9.0, 0.1, 0.3, 0.5],
        'aaod675': [np.nan, np.nan, 0.01, np.nan],
    })
    starts = pd.to_datetime(['2023-01-01 09:00', '2023-01-03 09:00']).to_numpy(dtype='datetime64[ns]')
    ends = pd.to_datetime(['2023-01-02 09:00', '2023-01-04 09:00']).to_numpy(dtype='datetime64[ns]')
    means, lo, hi = export.window_means(points, ['aod500', 'aaod675'], starts, ends)
    assert np.allclose(means[0], [0.2, 0.01])     # 06:00 is before the window, the next day's 10:00 after it
    assert np.isnan(means[1]).all()               # nothing retrieved in the second window
    assert (hi - lo).tolist() == [2, 0]


def test_two_band_exponent_recovers_a_power_law_and_skips_non_positive_pairs():
    # AAOD ∝ λ^-1.4 between 440 and 675 nm gives AAE 1.4; a zero, missing or negative AAOD gives none
    a440 = np.array([0.1, 0.0, np.nan, 0.05])
    a675 = np.array([0.1 * (675 / 440) ** -1.4, 0.02, 0.02, -0.01])
    out = export.two_band_exponent(a440, a675, 440, 675)
    assert np.isclose(out[0], 1.4)
    assert np.isnan(out[1:]).all()


def test_derive_adds_two_band_absorption_and_scattering_exponents_per_retrieval():
    a675 = 0.08 * (675 / 440) ** -2.0
    inv = pd.DataFrame({
        'aaod440': [0.08], 'aaod675': [a675], 'aaod870': [a675 * (870 / 675) ** -1.0],
        'inv_aod440': [0.5], 'inv_aod675': [0.5 * (675 / 440) ** -1.5], 'ssa440': [0.9], 'ssa675': [0.9], 'ssa870': [0.85],
    })
    out = export.derive('inv', inv)
    assert np.isclose(out['aae440_675'].iloc[0], 2.0) and np.isclose(out['aae675_870'].iloc[0], 1.0)
    assert np.isclose(out['sae440_675'].iloc[0], 1.5)        # equal SSA: scattering follows extinction
    assert np.isclose(out['dssa'].iloc[0], 0.05)
    aod = export.derive('aod', pd.DataFrame({'aod440': [0.6], 'aod870': [0.2]}))
    assert np.isclose(aod['daod'].iloc[0], 0.4)


def test_aerosol_type_follows_the_giles_fine_fraction_and_ssa_bands():
    fmf = [0.2, 0.33, 0.5, 0.8, 0.8, 0.8, 0.8, np.nan]
    ssa = [0.85, 0.99, 0.99, 0.97, 0.93, 0.88, np.nan, 0.9]
    # coarse at FMF ≤ 0.33 and mixed to 0.66 whatever the SSA; fine split at SSA 0.95 and 0.90; no FMF, no type
    assert export.aerosol_type(fmf, ssa).tolist() == [1, 1, 2, 3, 4, 5, 6, 0]
    assert set(export.AEROSOL_TYPES) == {1, 2, 3, 4, 5, 6}
