"""Offline checks for the MAIA catalogue's parsing (scripts/pipelines/maia_catalog.py)."""
import importlib.util
from pathlib import Path

from aethmodular_cli import cli

PATH = Path(__file__).resolve().parents[1] / 'scripts/pipelines/maia_catalog.py'
spec = importlib.util.spec_from_file_location('maia_catalog', PATH)
maia = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maia)


def test_parses_surface_monitor_and_level4_titles():
    assert maia.parse_granule_title(
        'MAIA_ANC_SURFACEMONITOR_PM_TOTAL_IND-Delhi_20261018T000000Z_F01_V2026p08p26.nc'
    ) == {'target': 'IND-Delhi', 'day': '2026-10-18', 'version': 'V2026p08p26'}
    parsed = maia.parse_granule_title('MAIA_L4_GFPM_20181231T000000Z_FB_NOM_R01_USA-Boston_F01_VSIM01p01p01p01.nc')
    assert parsed['target'] == 'USA-Boston'
    assert parsed['day'] == '2018-12-31'


def test_newest_version_wins_per_target_day():
    base = 'MAIA_ANC_SURFACEMONITOR_PM_TOTAL_ETH-AddisAbaba_20230101T000000Z_F01_'
    titles = [base + 'V01p01p01_V001.nc', base + 'V2026p06p10.nc', base + 'V2025p08p28.nc',
              'MAIA_ANC_SURFACEMONITOR_PM_TOTAL_ETH-AddisAbaba_20230102T000000Z_F01_V001.nc']
    assert maia.newest_per_day(titles) == [base + 'V2026p06p10.nc', titles[3]]


def test_coverage_counts_missing_days_and_longest_gap():
    out = maia.coverage(['2023-01-01', '2023-01-02', '2023-01-06', '2023-01-06', ''])
    assert out == {'first': '2023-01-01', 'last': '2023-01-06', 'n_days': 3,
                   'span_days': 6, 'missing_days': 3, 'longest_gap_days': 3}
    assert maia.coverage([])['n_days'] == 0


def test_target_outline_drawn_as_linestring():
    square = {'type': 'LineString', 'coordinates': [[38, 8], [40, 8], [40, 10], [38, 10], [38, 8]]}
    outlines = {'ETH-AddisAbaba': maia.rings(square)}
    assert maia.target_of(9.0, 38.76, outlines, {'ETH-AddisAbaba': 'PTA'}) == 'ETH-AddisAbaba'
    assert maia.target_of(9.0, 41.0, outlines, {'ETH-AddisAbaba': 'PTA'}) == ''


def test_layer_names_drop_target_prefix_even_when_mistyped():
    row = maia.layer_row('VNM-Hanoi', {'name': 'VNN Land Cover 1km', 'type': 'tile',
                                       'url': 'Layers/VNM-Hanoi_LANDCOV_1km_v5/{z}/{x}/{y}.png'})
    assert row['layer'] == 'Land Cover'
    assert row['resolution'] == '1km'
    assert row['url'].startswith(maia.MMGIS_MISSION)
    ndvi = maia.layer_row('ETH-AddisAbaba', {'name': 'ETH-AddisAbaba NDVI 250m',
                                             'url': 'Layers/ETH-AddisAbaba_NDVI_20231117_250m_v5/{z}/{x}/{y}.png'})
    assert ndvi['date'] == '2023-11-17'


def test_data_links_skip_collection_links():
    entry = {'links': [
        {'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#', 'href': 'https://data.asdc.earthdata.nasa.gov/x.nc'},
        {'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#', 'href': 'https://search.earthdata.nasa.gov/s', 'inherited': True},
    ]}
    assert maia.data_links(entry) == ['https://data.asdc.earthdata.nasa.gov/x.nc']


def test_cli_routes_maia_workflows():
    args = cli.build_parser().parse_args(['maia', 'pull', '--dry-run'])
    assert args.func is cli.command_maia
    assert args.workflow == 'pull'
    assert args.script_args == ['--dry-run']
