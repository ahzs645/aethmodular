"""A corrected source result cannot relabel or alter historical target evidence."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'research/ftir_hips_chem/scripts'), str(ROOT / 'gallery/data')]
from candidate_metrics import SCORE_PROTOCOL, paired_candidate_metrics, sha256_file  # noqa: E402
from export_continuum_baseline import refresh_source_evaluation  # noqa: E402


@pytest.fixture
def frozen_comparison(tmp_path):
    metadata = pd.DataFrame({'filter_id': ['A', 'B', 'C'], 'site': ['X', 'Y', 'Z'],
                             'tor_ug': [1., 2., 3.], 'ocec': [1., 1., 1.]})
    models = ['raw', 'spline', 'vibes', 'vibes_deflate', 'vibes_anchorhp', 'epo']
    scores, population, ledger = paired_candidate_metrics(metadata, {m: np.array([2., 4., 6.]) for m in models})
    scores.loc[scores.model.eq('epo'), 'evaluation_status'] = 'corrected'
    directory = tmp_path / 'source_scores'
    directory.mkdir()
    scores.to_csv(directory / 'baseline_scoreboard.csv', index=False)
    ledger.to_csv(directory / 'baseline_scoreboard_filters.csv', index=False)
    receipt = {'status': 'corrected', 'protocol': SCORE_PROTOCOL, 'population': population,
               'provenance': {'epo_protocol': 'epo_train_only_site_cv_v2', 'epo_predictions_sha256': 'a' * 64},
               'outputs': {name: {'sha256': sha256_file(directory / name)} for name in
                           ('baseline_scoreboard.csv', 'baseline_scoreboard_filters.csv')}}
    (directory / 'baseline_scoreboard_receipt.json').write_text(json.dumps(receipt))
    original = {
        'schema_version': 1, 'generated': '2026-11-04T00:00:00Z', 'source': 'historical_run',
        'settings': {'k': 18, 'fit': {'fit_rows': 12387}},
        'scoreboard': {'model': models, 'improve_error': [9.] * 6, 'target_intercept': [-.9] * 6},
        'sweep': {'model': ['vibes', 'epo'], 'k': [18, 18], 'improve_mae': [1.1, 1.24],
                  'improve_heavy_lowocec_ratio': [.73, .91], 'intercept_aeth': [-.9, .2]},
        'rule_sim': {'model': ['VIBES k18 (before)', 'EPO k18'], 'pass': [.7, .9]},
        'mae': {'model': ['vibes'], 'k': [18], 'mae': [1.1]},
        'addis_filters': {'filter_id': ['ETAD-0001'], 'vibes k18': [5.43]},
        'season_aeth': {'season': ['Bega', 'Kiremt'], 'value': [-.8, 1.4]},
    }
    saved = tmp_path / 'saved.json'
    saved.write_text(json.dumps(original))
    return saved, directory, original


def test_refresh_preserves_target_values_and_the_frozen_input(frozen_comparison, tmp_path):
    saved, scores, original = frozen_comparison
    before = saved.read_bytes()
    result = refresh_source_evaluation(saved, scores, tmp_path / 'refreshed.json', expected_sha256=sha256_file(saved))
    assert saved.read_bytes() == before
    assert result['addis_filters'] == original['addis_filters']
    assert result['season_aeth'] == original['season_aeth']
    assert result['settings']['fit'] == original['settings']['fit']
    assert result['settings']['target_predictions']['generated'] == original['generated']
    assert result['settings']['source_refresh']['base_export_sha256'] == sha256_file(saved)
    scoreboard = pd.DataFrame(result['scoreboard']).set_index('model')
    assert scoreboard.loc['epo', 'improve_median_absolute_error'] == 2.
    assert scoreboard.loc['epo', 'evaluation_status'] == 'corrected'
    assert scoreboard.loc['epo', 'target_intercept'] == -.9
    assert 'common_filter_ids' not in result['settings']['source_evaluation']['population']
    assert 'common_filter_ids_sha256' in result['settings']['source_evaluation']['population']
    assert len(pd.read_csv(scores / 'baseline_scoreboard_filters.csv')) == 3


def test_new_source_scores_do_not_certify_saved_sweep_or_simulation(frozen_comparison, tmp_path):
    saved, scores, original = frozen_comparison
    result = refresh_source_evaluation(saved, scores, tmp_path / 'refreshed.json', expected_sha256=sha256_file(saved))
    sweep = pd.DataFrame(result['sweep']).set_index('model')
    assert sweep.loc['vibes', 'improve_median_absolute_error'] == 1.1
    assert pd.isna(sweep.loc['vibes', 'improve_mae'])
    assert pd.isna(sweep.loc['epo', 'improve_median_absolute_error'])
    assert pd.isna(sweep.loc['epo', 'improve_heavy_lowocec_ratio'])
    assert sweep.loc['epo', 'intercept_aeth'] == .2
    assert result['rule_sim']['model'] == ['VIBES k18 (before)']
    assert result['mae']['median_absolute_error'] == original['mae']['mae']


@pytest.mark.parametrize('corruption', ['base_hash', 'ledger', 'same_output'])
def test_refresh_rejects_unverified_or_destructive_inputs(frozen_comparison, tmp_path, corruption):
    saved, scores, _ = frozen_comparison
    output = tmp_path / 'refreshed.json'
    expected = sha256_file(saved)
    if corruption == 'base_hash':
        expected = '0' * 64
    elif corruption == 'ledger':
        with (scores / 'baseline_scoreboard_filters.csv').open('a') as stream:
            stream.write('changed\n')
    else:
        output = saved
    before = saved.read_bytes()
    with pytest.raises(ValueError):
        refresh_source_evaluation(saved, scores, output, expected_sha256=expected)
    assert saved.read_bytes() == before
    if output != saved:
        assert not output.exists()


@pytest.mark.parametrize('field,value', [('k', 12), ('models', ['raw'])])
def test_refresh_requires_the_declared_candidate_and_component_comparison(frozen_comparison, tmp_path, field, value):
    saved, scores, original = frozen_comparison
    changed = copy.deepcopy(original)
    if field == 'k':
        changed['settings']['k'] = value
    else:
        changed['scoreboard'] = {key: values[:1] for key, values in changed['scoreboard'].items()}
    saved.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='candidate identities|component count'):
        refresh_source_evaluation(saved, scores, tmp_path / 'refreshed.json', expected_sha256=sha256_file(saved))
