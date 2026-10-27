"""The AERONET part of the Data availability export (gallery/data/export_availability.py)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# run as a script, the exporter finds its sibling site_labels.py on the path; as a module it needs it added
sys.path.insert(0, str(ROOT / 'gallery/data'))
spec = importlib.util.spec_from_file_location('export_availability', ROOT / 'gallery/data/export_availability.py')
availability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(availability)


def test_photometer_days_split_aod_and_inversions():
    daily = {'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
             'aod440': [0.1, None, None], 'aod500': [None, None, None], 'aod675': [None, 0.2, None],
             'aaod440': [None, None, None], 'aaod675': [0.01, None, None], 'ssa440': [None, None, 0.9]}
    aod, inv = availability.photometer_days(daily)
    assert aod == ['2024-01-01', '2024-01-02']          # any AOD wavelength counts
    assert inv == ['2024-01-01', '2024-01-03']          # an inversion day need not have a direct-sun AOD


def test_monthly_counts_keep_filter_and_photometer_months():
    filters = [('2024-01-05', True, False), ('2024-01-20', False, False), ('2024-03-02', True, True)]
    out = availability.monthly_counts(filters, ['2024-01-05', '2024-02-10'], ['2024-03-02'])
    assert [m['month'] for m in out] == ['2024-01', '2024-02', '2024-03']
    assert out[0] == {'month': '2024-01', 'filters': 2, 'filters_aod': 1, 'filters_inv': 0, 'aod_days': 1, 'inv_days': 0}
    assert out[1]['filters'] == 0 and out[1]['aod_days'] == 1
    assert out[2]['filters_inv'] == 1 and out[2]['inv_days'] == 1
