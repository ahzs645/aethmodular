"""Offline checks for the research-tab HIPS exporters (gallery/data/export_research_*.py)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gallery/data'))
sys.path.insert(0, str(ROOT / 'research/ftir_hips_chem/scripts'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'gallery/data/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


optics = load('export_research_hips_optics')
envelope = load('export_research_improve_envelope')


def long_table():
    """Two HIPS pieces of one sample and one blank, with FTIR mass on the sample."""
    rows = []
    def add(fid, ftype, param, conc, mass=np.nan, date='2023-01-03'):
        rows.append({'Site': fid[:4], 'FilterId': fid, 'SampleDate': pd.Timestamp(date) if date else pd.NaT, 'FilterType': ftype,
                     'LotId': 251.0, 'AnalysisDate': pd.Timestamp('2023-03-01'), 'DepositArea_cm2': 3.53, 'Volume_m3': 7.2,
                     'Parameter': param, 'Concentration': conc, 'MassLoading_ug': mass})
    for p, v in [('HIPS_t', 0.4), ('HIPS_r', 0.5), ('HIPS_tau', 0.3), ('HIPS_T1', 900.0), ('HIPS_R1', 1100.0),
                 ('HIPS_Intercept', 1416.2), ('HIPS_Slope', 0.5), ('HIPS_Fabs', 51.5), ('HIPS_MDL', 1.6)]:
        add('ETAD-0035-3', 'PM2.5', p, v)
        add('ETAD-0099-7', 'FB', p, v / 2, date=None)
    add('ETAD-0035-3', 'PM2.5', 'EC_ftir', 4.43, mass=31.9)
    add('ETAD-0035-3', 'PM2.5', 'OC_ftir', 6.2, mass=44.6)
    return pd.DataFrame(rows)


def test_optics_rows_join_on_the_gallery_id_and_keep_blanks():
    frame = optics.build(long_table())
    assert sorted(frame['type']) == ['FB', 'PM2.5']
    sample = frame[frame['type'].eq('PM2.5')].iloc[0]
    assert sample['id'] == 'ETAD-0035' and sample['filter_id'] == 'ETAD-0035-3'
    assert sample['ec_ug'] == pytest.approx(31.9) and sample['fabs'] == pytest.approx(51.5)
    assert sample['lot'] == '251' and sample['date'] == '2023-01-03'
    blank = frame[frame['type'].eq('FB')].iloc[0]
    assert blank['date'] is None or pd.isna(blank['date'])
    payload = optics.to_json(frame)
    assert payload['columns'][0] == 'id' and len(payload['rows']) == 2


def improve_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ec = rng.uniform(0.05, 6, n)
    vol = np.where(np.arange(n) % 10 == 0, np.nan, 33.0)
    fabs = 10 * ec + 1 + rng.normal(0, 0.5, n)
    return envelope.prepare_improve(pd.DataFrame({
        'SiteCode': np.where(np.arange(n) % 2, 'AAAA1', 'BBBB1'),
        'Date': pd.to_datetime('2010-01-01') + pd.to_timedelta(np.arange(n), 'D'),
        'ECf_Val': ec, 'OCf_Val': ec * np.where(np.arange(n) % 3 == 0, 1.5, 6.0), 'fAbs_Val': fabs,
        'FEf_Val': 0.05, 'MF_Val': 5.0, 'volume_m3': vol, 'EC_loading_ug': ec * vol, 'EC_loading_ug_cm2_primary': ec * vol / 3.5,
        'MF_loading_ug': 5.0 * vol,
    }))


def spartan_frame():
    ec = np.linspace(1, 8, 30)
    return pd.DataFrame({'code': 'ETAD', 'fabs': 4 * ec + 28, 'ec_conc': ec, 'ec_ug': ec * 7.2, 'oc_ug': ec * 14,
                         'ec_ug_cm2': ec * 7.2 / 3.53, 'volume_m3': 7.2, 'area_cm2': 3.53})


def test_bounds_and_screens_count_rows_inside_each_addis_range():
    sp = spartan_frame()
    bounds = envelope.etad_bounds(sp)
    assert bounds['fabs']['min'] == pytest.approx(32) and bounds['fabs']['max'] == pytest.approx(60)
    imp = improve_frame()
    steps = {s['step']: s for s in envelope.screens(imp, bounds)}
    with_loading = steps['… with a volume (EC mass and surface loading)']['n']
    assert with_loading == int(imp['valid_loading'].sum())
    alone = steps['fAbs in the Addis p05–p95']['n']
    both = steps['fAbs and EC mass']['n']
    assert both <= alone <= with_loading


def test_intercept_ranges_recover_a_known_line():
    imp = improve_frame()
    out = {o['range']: o for o in envelope.intercepts(imp, min_site_n=50)}
    allrows = out['All post-2003']
    assert allrows['slope'] == pytest.approx(10, abs=0.1) and allrows['intercept'] == pytest.approx(1, abs=0.2)
    assert out[next(k for k in out if k.startswith('Per-site'))]['n'] == 2


def test_density_counts_every_row_in_range_once():
    imp = improve_frame()
    post = imp[imp['post']]
    d = envelope.density(post['EC_loading_ug'], post['fAbs_Val'], 1000, 1000, bins=10)
    assert sum(c for _, _, c in d['cells']) + d['beyond'] == d['n'] == int(post[['EC_loading_ug', 'fAbs_Val']].dropna().shape[0])


def test_rt_bins_and_the_initial_line():
    imp = improve_frame()
    rng = np.random.default_rng(1)
    n = len(imp)
    for label, (rc, tc) in envelope.RT_PAIRS.items():
        ref = rng.uniform(80, 100, n)
        imp[rc], imp[tc] = ref, 0.5 * ref + 10 - (imp['fAbs_Val'] * 0.1 if label == 'minimum' else 0)
    bounds = envelope.etad_bounds(spartan_frame())
    out = envelope.rt(imp, bounds, bins=10, min_cell=1)
    assert set(out['pairs']) == {'initial', 'final', 'minimum'}
    assert out['initial_line']['slope'] == pytest.approx(0.5) and out['initial_line']['intercept'] == pytest.approx(10)
    # minimum readings sit below the initial line by 0.1 x fAbs: the deficit grows with fAbs
    medians = [d['p50'] for d in out['deficit']]
    assert medians == sorted(medians, reverse=True) and medians[0] < 0


def test_bundle_runs_end_to_end_without_rt_fields():
    payload = envelope.bundle(improve_frame(), spartan_frame())
    assert payload['rt'] is None
    assert [s['threshold'] for s in payload['sweep']] == envelope.SWEEP
    assert {m['group'] for m in payload['matched']} >= {'ETAD (Addis)'}
    assert any(r['group'] == 'IMPROVE' and r['quantity'] == 'fAbs × V / A' for r in payload['mass_space'])


origin = load('export_research_improve_origin')


def test_site_sufficient_statistics_refit_ols_exactly():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({'Site': np.repeat(['A', 'B', 'C'], 50), 'EC_ugm3': rng.uniform(0, 2, 150)})
    frame['Fabs'] = 9 * frame['EC_ugm3'] + 0.4 + rng.normal(0, 0.1, 150)
    slope, intercept = origin.ols_from_stats(origin.site_stats(frame))
    ref = np.polyfit(frame['EC_ugm3'], frame['Fabs'], 1)
    assert slope == pytest.approx(ref[0]) and intercept == pytest.approx(ref[1])
    lo, hi = origin.cluster_ci(frame, reps=200)
    assert lo <= intercept <= hi


def test_deming_intercept_rises_monotonically_to_the_ols_value():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 3, 400)
    y = 10 * x + 1 + rng.normal(0, 1, 400)
    x_obs = x + rng.normal(0, 0.2, 400)
    ints = [origin.deming(x_obs, y, lam)[1] for lam in origin.LAMBDAS]
    assert all(b >= a - 1e-9 for a, b in zip(ints, ints[1:]))
    assert ints[-1] == pytest.approx(np.polyfit(x_obs, y, 1)[1], abs=1e-3)


met = load('export_research_addis_met')


def test_sample_day_met_joins_the_station_day_and_means_the_hours():
    daily = pd.DataFrame({'date': ['12/6/2022 0:00', '12/7/2022 0:00'], 'tavg': [17.3, 20.0], 'tmin': [8.6, 10.0], 'tmax': [24.2, 26.0], 'prcp': [0, 1.5]})
    hourly = pd.DataFrame({'time': ['12/6/2022 0:00', '12/6/2022 1:00', '12/7/2022 0:00'], 'temp': [10, 11, 12], 'dwpt': [1, 'NA', 3],
                           'rhum': [50, 60, 'NA'], 'wspd': [10, 20, 30]})
    m = met.daily_met(daily, hourly)
    day = m.loc[pd.Timestamp('2022-12-06')]
    assert day['rhum_mean'] == pytest.approx(55) and day['dwpt_mean'] == pytest.approx(1) and day['trange'] == pytest.approx(15.6)
    rows = met.per_filter([{'id': 'ETAD-0001', 'code': 'ETAD', 'date': '2022-12-06'}, {'id': 'CHTS-0001', 'code': 'CHTS', 'date': '2022-12-06'},
                           {'id': 'ETAD-0002', 'code': 'ETAD', 'date': '2023-01-01'}], m)
    assert list(rows) == ['ETAD-0001']
    assert rows['ETAD-0001'][list(met.FIELDS).index('Minimum temperature (station, sample day)')] == 8.6
