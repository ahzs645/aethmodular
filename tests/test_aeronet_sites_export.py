"""Offline checks for the per-site AERONET summary (gallery/data/export_aeronet_sites.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_aeronet_sites', ROOT / 'gallery/data/export_aeronet_sites.py')
sites = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sites)


def frame(**columns):
    base = {'PM2.5 mass': [np.nan], 'HIPS Fabs': [np.nan], 'BC (public HIPS)': [np.nan],
            sites.AOD500: [np.nan], sites.AAOD675: [np.nan],
            'AAE 440-870 (AERONET)': [np.nan], 'SSA 440 nm (AERONET)': [np.nan]}
    n = len(next(iter(columns.values())))
    base = {k: v * n for k, v in base.items()}
    base.update(columns)
    return pd.DataFrame(base)


def test_eta_needs_positive_pm_and_aod():
    out = sites.derive(frame(**{'PM2.5 mass': [30.0, -1.0, 20.0, 10.0], sites.AOD500: [0.5, 0.5, 0.0, np.nan]}))
    assert out['eta'].iloc[0] == 60.0
    assert out['eta'].iloc[1:].isna().all()


def test_scale_height_prefers_hips_fabs_then_public_bc():
    out = sites.derive(frame(**{sites.AAOD675: [0.01, 0.01, 0.01], 'HIPS Fabs': [20.0, np.nan, 0.0],
                                'BC (public HIPS)': [5.0, 2.0, 2.0]}))
    assert out['h'].iloc[0] == pytest.approx(500)       # HIPS Fabs wins where present
    assert out['h'].iloc[1] == pytest.approx(500)       # else public BC x 10
    assert np.isnan(out['h'].iloc[2])                 # a zero HIPS Fabs gives no H (not the BC fallback)


def test_summarise_and_spearman():
    s = sites.summarise(pd.Series([1.0, 2.0, 3.0, np.nan, 10.0]), 2)
    assert s == {'n': 4, 'median': 2.5, 'mean': 4.0, 'p25': 1.75, 'p75': 4.75}
    assert sites.summarise(pd.Series([np.nan]), 2) is None
    x = pd.Series(range(10), dtype=float)
    assert sites.spearman(x, x ** 3) == 1.0
    assert sites.spearman(x[:5], x[:5]) is None       # too few pairs


def test_site_summary_bins_by_the_site_calendar():
    calendars = {'XXXX': {'seasons': [{'name': 'Dry', 'months': [1, 2, 3, 4, 5, 6]},
                                      {'name': 'Wet', 'months': [7, 8, 9, 10, 11, 12]}]}}
    months = sites.season_lookup(calendars)['XXXX']
    part = frame(**{'date': ['2024-01-05', '2024-02-05', '2024-08-05'], 'PM2.5 mass': [10.0, 20.0, 30.0],
                    sites.AOD500: [0.1, 0.2, 0.3]})
    out = sites.site_summary(part, sites.derive(part), months, ['Dry', 'Wet'])
    assert out['n_filters'] == 3
    assert out['all']['eta']['median'] == 100.0
    assert out['seasons']['Dry']['n_filters'] == 2 and out['seasons']['Wet']['eta'] == {'n': 1, 'median': 100.0}
    assert 'aaod675' not in out['all']                # nothing to summarise is left out, not zero
