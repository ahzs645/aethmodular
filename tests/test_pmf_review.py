"""Regression tests for integration review; all EPA evidence is synthetic."""
import numpy as np
import pytest

from src.pmf.epa import export_epa, import_epa
from src.pmf.io import Dataset, write_json, write_matrix
from src.pmf.runner import project_case, selected_model


def _import_fixture(tmp_path, settings):
    g = np.array([[1.], [2.], [3.]])
    h = np.array([[1., 2., 3.]])
    data = Dataset(g @ h, np.ones((3, 3)), np.array(['a', 'b', 'c']),
                   np.array(['1', '2', '3']), {'units': 'synthetic'})
    export_epa(data, tmp_path/'handoff', dataset_id='synthetic', factors=[1])
    write_matrix(tmp_path/'G.csv', data.sample_ids, ['F1'], g)
    write_matrix(tmp_path/'H.csv', ['F1'], data.features, h)
    (tmp_path/'fixture.log').write_text('Synthetic fixture, not native execution.\n')
    write_json(tmp_path/'evidence.json', {
        'engine': 'EPA PMF5', 'engine_version': 'synthetic-fixture',
        'native_execution': True, 'convergence_status': 'converged',
        'settings': settings, 'reported_q_true': 0., 'raw_files': ['fixture.log'],
    })
    import_epa(tmp_path/'handoff', tmp_path/'G.csv', tmp_path/'H.csv',
               tmp_path/'evidence.json', tmp_path/'imported')
    return data


@pytest.mark.parametrize('settings', [
    {'robust': True}, {'loss': 'native-robust'},
    {'requested_native_loss': 'native-robust'},
])
def test_operator_robust_import_cannot_silently_project_quadratically(tmp_path, settings):
    data = _import_fixture(tmp_path, settings)
    _, model, manifest = selected_model(tmp_path/'imported', 1)
    assert model['config']['requested_native_loss'] == 'native-robust'
    assert model['metrics']['loss'] == 'quadratic'  # Q(true), not native Q_main
    assert model['provenance']['operator_settings'] == settings
    assert manifest['native_execution_independently_observed'] is False
    with pytest.raises(ValueError, match='Native robust fixed-profile projection'):
        project_case(data, tmp_path/'imported', tmp_path/'projection', 1)


@pytest.mark.parametrize('settings', [{}, {'robust': False}, {'loss': 'quadratic'}])
def test_operator_ordinary_import_still_supports_projection(tmp_path, settings):
    data = _import_fixture(tmp_path, settings)
    result = project_case(data, tmp_path/'imported', tmp_path/'projection', 1)
    assert result['projection_info']['loss'] == 'quadratic'


@pytest.mark.parametrize('settings', [
    {'robust': 'true'}, {'robust': True, 'loss': 'quadratic'}, {'loss': 'huber'},
])
def test_operator_import_rejects_ambiguous_native_loss(tmp_path, settings):
    with pytest.raises(ValueError, match='boolean|Conflicting|Native loss'):
        _import_fixture(tmp_path, settings)
