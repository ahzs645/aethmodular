"""Scientific boundaries and preservation for the notebook-coverage additions."""
import json

import numpy as np
import pandas as pd
import pytest

from gallery.data import export_research_spectral as spectral
from gallery.data.export_processing_comparison import compare
from src.data.qc.smoothing import dema, ona


def test_ona_inclusive_threshold_and_boundaries():
    out, windows = ona(np.array([1., 2., 3., 20., 30.]),
                       np.array([0., .02, .05, .06, .07]), np.array([0, 0, 0, 0, 1]))
    np.testing.assert_allclose(out, [2, 2, 2, 20, 30])
    assert windows.points.tolist() == [3, 1, 1]
    assert windows.threshold_reached.tolist() == [True, False, False]


def test_ona_keeps_missing_observations_and_unknown_attenuation_missing():
    out, windows = ona(np.array([2., np.nan, 6., 9.]), np.array([0., .01, .02, np.nan]), np.zeros(4))
    np.testing.assert_allclose(out, [4., np.nan, 4., np.nan], equal_nan=True)
    assert windows.points.tolist() == [3]
    with pytest.raises(ValueError):
        ona(np.ones(2), np.ones(3), np.zeros(2))


def test_shared_dema_matches_pipeline_and_restarts_at_a_spot():
    x = pd.Series([1., 2., 3., 100., 100.])
    segments = pd.Series([0, 0, 0, 1, 1])
    actual = dema(x, segments, 15)
    first = x.iloc[:3].ewm(span=15, adjust=False).mean()
    np.testing.assert_allclose(actual.iloc[:3], 2 * first - first.ewm(span=15, adjust=False).mean())
    assert actual.iloc[3] == 100


def minute_frame(n):
    t = pd.date_range('2026-01-01 09:00', periods=n, freq='min')
    x = 2000 + 500 * np.sin(np.arange(n) / 5)
    return pd.DataFrame({'tu': t, 't': t, 'IR BCc': x, 'Blue BCc': x,
                         'IR ATN1': np.arange(n) / 100, 'Blue ATN1': np.arange(n) / 100})


def test_short_sample_does_not_invent_daily_agreement():
    tables = compare(minute_frame(60))
    assert tables['daily'].empty and tables['agreement'].empty
    assert set(tables['summary'].n) == {60}
    for _, group in tables['windows'].groupby('channel'):
        assert (group.minutes * group['count']).sum() == 60


def test_daily_methods_use_identical_minutes_and_do_not_cross_gaps():
    df = minute_frame(1440)
    df.loc[10, 'IR BCc'] = np.nan
    tables = compare(df)
    daily = tables['daily'].query("channel == 'IR'")
    assert set(daily.n_minutes) == {1439}
    assert len(daily) == 4
    # A constant signal remains constant under every method.
    df[['IR BCc', 'Blue BCc']] = 2000.
    df.loc[20:, 'tu'] += pd.Timedelta(hours=2)
    tables = compare(df)
    assert set(tables['summary'].RMSE) == {0.}


def test_optional_export_preserves_values_lineage_and_original_date(tmp_path, monkeypatch):
    monkeypatch.setattr(spectral, 'ROOT', tmp_path)
    out = tmp_path / 'gallery/app/public/data/ml'
    out.mkdir(parents=True)
    saved = {'schema_version': 1, 'generated': '2026-10-30',
             'tables': {'curve': {'k': [1, 2], 'value': [1.123456789, 2.987654321]}},
             'sources': [{'table': 'curve', 'notebook': 'ftir_32', 'path': 'source.csv', 'read': 'csv'}]}
    (out / 'test.json').write_text(json.dumps(saved))
    page = spectral.Page('test', 'Test', out=out)
    frame = page.optional('curve', tmp_path, 'ftir_32', 'missing.csv', 'curve')
    assert frame.to_dict('list') == saved['tables']['curve']
    assert not page.missing
    assert page.sources[0]['source_export_generated'] == '2026-10-30'
    assert page.sources[0]['path'] == 'source.csv'
    # A real source supersedes the retained export and transforms exactly once.
    pd.DataFrame({'value': [3.]}).to_csv(tmp_path / 'fresh.csv', index=False)
    page.optional('curve', tmp_path, 'ftir_32', 'fresh.csv', 'curve', transform=lambda f: f * 2)
    assert page.tables['curve'].value.tolist() == [6.]


def test_charcoal_export_arrays_require_aligned_metadata():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / 'research/ftir_etad_char/scripts/diagnostic_exports.py'
    spec = importlib.util.spec_from_file_location('diagnostic_exports', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(ValueError):
        mod.score_rows(np.ones((2, 2)), pd.DataFrame({'id': [1]}), 'Addis', 'PCA')
    meta = pd.DataFrame({'MediaId': [7, 8], '2000.0': [1., 2.]})
    scores = mod.score_rows(np.ones((2, 2)), meta, 'Addis', 'PCA')
    assert scores.MediaId.tolist() == [7, 8] and '2000.0' not in scores
    vals = np.array([[0., 2.], [10., 4.]])
    envelope = mod.envelope_rows([1000., 2000.], vals, 'Addis', 'Raw')
    np.testing.assert_allclose(envelope['mean'], [5., 3.])
    np.testing.assert_allclose(envelope.p05, [.5, 2.1])
    np.testing.assert_allclose(envelope.p95, [9.5, 3.9])


def test_small_derivative_envelopes_survive_shared_serialization(tmp_path, monkeypatch):
    monkeypatch.setattr(spectral, 'ROOT', tmp_path)
    page = spectral.Page('precision', 'Precision', out=tmp_path / 'gallery/app/public/data/ml',
                         exporter=tmp_path / 'export.py')
    page.add('envelope', pd.DataFrame({'mean': [1.2345e-8, -4.321e-7]}), digits=None)
    saved = json.loads(page.write().read_text())
    assert saved['tables']['envelope']['mean'] == [1.2345e-8, -4.321e-7]
    # A later rebuild retaining this table must retain the precision policy too.
    later = spectral.Page('precision', 'Precision', out=page.out, exporter=tmp_path / 'export.py')
    later.retained('envelope')
    assert json.loads(later.write().read_text())['tables']['envelope'] == saved['tables']['envelope']
