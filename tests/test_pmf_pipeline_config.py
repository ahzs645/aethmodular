"""Reject ambiguous or escaping pipeline paths before creating any output."""
import json

import pytest

from src.pmf.cli import pipeline


@pytest.mark.parametrize('field,value', [
    ('dataset', '.'), ('dataset', '..'), ('dataset', '../escape'),
    ('run', '.'), ('run', '..'), ('run', '../escape'), ('run', 'prepared'),
])
def test_pipeline_rejects_unsafe_components_before_writes(tmp_path, field, value):
    dataset = {'dataset_id': value if field == 'dataset' else 'example',
               'preparation_config': 'absent.json',
               'runs': [{'name': value if field == 'run' else 'ordinary'}]}
    config = tmp_path / 'pipeline.json'
    config.write_text(json.dumps({'output': 'runs', 'datasets': [dataset]}))
    with pytest.raises(ValueError, match='safe'):
        pipeline(config)
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('defect', ['empty', 'no_runs', 'duplicate_dataset', 'duplicate_run', 'operation'])
def test_pipeline_requires_unambiguous_executable_jobs(tmp_path, defect):
    dataset = {'dataset_id': 'example', 'preparation_config': 'absent.json',
               'runs': [{'name': 'ordinary'}]}
    datasets = [dataset]
    if defect == 'empty':
        datasets = []
    elif defect == 'no_runs':
        dataset['runs'] = []
    elif defect == 'duplicate_dataset':
        datasets.append(dataset.copy())
    elif defect == 'duplicate_run':
        dataset['runs'].append({'name': 'ordinary'})
    else:
        dataset['runs'][0]['operation'] = 'unknown'
    config = tmp_path / 'pipeline.json'
    config.write_text(json.dumps({'output': 'runs', 'datasets': datasets}))
    with pytest.raises(ValueError):
        pipeline(config)
    assert not (tmp_path / 'runs').exists()
