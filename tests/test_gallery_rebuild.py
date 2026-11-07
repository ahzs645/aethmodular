"""The gallery rebuild's step list stays runnable, ordered, complete and documented."""
import importlib.util
import json
import sys
from pathlib import Path

from aethmodular_cli import cli

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module      # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


rebuild = load('gallery_rebuild', 'gallery/data/rebuild.py')
mirror = load('mirror_davis_drive', 'scripts/pipelines/mirror_davis_drive.py')


def test_steps_have_unique_keys_and_follow_their_dependencies():
    keys = [s.key for s in rebuild.STEPS]
    assert len(keys) == len(set(keys))
    position = {s.key: s.order for s in rebuild.STEPS}
    for s in rebuild.STEPS:
        for dep in s.after:
            assert dep in position, f'{s.key} runs after unknown step {dep}'
            assert position[dep] < s.order, f'{s.key} is listed before {dep}, which it needs'


def test_every_step_script_exists_and_carries_no_machine_path():
    for s in rebuild.STEPS:
        if not s.node and not s.run[0].startswith('-'):
            assert (ROOT / s.run[0]).is_file(), s.run[0]
        assert not any(p.startswith(('/home/', '/Users/')) for p in s.run), s.key


def test_every_new_gallery_exporter_is_a_step():
    registered = {s.run[0] for s in rebuild.STEPS if not s.node}
    exporters = sorted(p.relative_to(ROOT).as_posix() for pat in ('export_aeth*.py', 'export_aeronet*.py', 'export_research_*.py')
                       for p in (ROOT / 'gallery/data').glob(pat))
    missing = [e for e in exporters if e not in registered]
    assert not missing, f'add these to gallery/data/rebuild.py STEPS: {missing}'


def test_the_pipeline_doc_carries_the_generated_step_table():
    text = rebuild.DOC.read_text()
    assert rebuild.doc_table(text) == rebuild.markdown(), 'run `uv run aeth gallery markdown --write`'


def test_each_private_or_unlisted_section_has_data_paths_to_check():
    markers = {**rebuild.section_markers(), **rebuild.section_markers(rebuild.unlisted_sections())}
    assert markers and all(markers.values()), markers


def test_mirror_leaves_out_archives_the_app_mirror_duplicates_and_figures():
    assert mirror.wanted('Aethalometry Data/Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv', False, False)
    assert not mirror.wanted('AERONET/Jacros/_archive/bundle.zip', False, False)
    assert mirror.wanted('AERONET/Jacros/_archive/bundle.zip', True, False)
    assert not mirror.wanted('FTIR/shiny.aqrc.ucdavis.edu/index.html', True, False)
    assert not mirror.wanted('Aethalometry Data/x/df_jacros_uncleaned_API.pkl', True, False)
    assert not mirror.wanted('DAVIS/ETAD FTIR/plots/fig.png', True, False)
    assert mirror.wanted('DAVIS/ETAD FTIR/plots/fig.png', False, True)


def test_cli_routes_gallery_and_drive():
    args = cli.build_parser().parse_args(['gallery', 'rebuild', '--only', 'aeronet'])
    assert args.func is cli.command_gallery and args.workflow == 'rebuild' and args.script_args == ['--only', 'aeronet']
    args = cli.build_parser().parse_args(['drive', 'mirror', '--list'])
    assert args.func is cli.command_drive and args.script_args == ['--list']


def _isolated_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, 'ROOT', tmp_path)
    monkeypatch.setattr(rebuild, 'APP', tmp_path)
    monkeypatch.setattr(rebuild, 'RUN_LOG', tmp_path / 'last_run.json')
    monkeypatch.setattr(rebuild, 'roots', lambda: {})
    monkeypatch.setattr(rebuild, 'git', lambda *args: '')


def test_unexpected_node_failure_is_fatal_and_blocks_transitive_dependents(tmp_path, monkeypatch):
    _isolated_runner(tmp_path, monkeypatch)
    fail = rebuild.Step('node-tests', 'check', (sys.executable, '-c', 'raise SystemExit(7)'), 'unexpected test failure', '-', node=True)
    child = rebuild.Step('dependent', 'check', (sys.executable, '-c', "from pathlib import Path; Path('child-ran').touch()"),
                         'must not run', '-', after=('node-tests',), node=True)
    grandchild = rebuild.Step('grandchild', 'check', (sys.executable, '-c', "from pathlib import Path; Path('grandchild-ran').touch()"),
                              'must not run', '-', after=('dependent',), node=True)
    assert rebuild.run([fail, child, grandchild], dry=False) == 1
    log = json.loads(rebuild.RUN_LOG.read_text())
    assert log['steps'][0]['status'] == 'failed (exit 7)'
    assert log['steps'][1]['status'].startswith('skipped:')
    assert log['steps'][2]['status'].startswith('skipped:')
    assert not (tmp_path / 'child-ran').exists()
    assert not (tmp_path / 'grandchild-ran').exists()
    assert log['exit_code'] == 1


def test_missing_check_input_is_unavailable_and_fails_without_running(tmp_path, monkeypatch):
    _isolated_runner(tmp_path, monkeypatch)
    step = rebuild.Step('scientific-check', 'check', (sys.executable, '-c', "from pathlib import Path; Path('ran').touch()"),
                        'needs the frozen export', '-', (rebuild.Need('frozen/evidence.csv', lambda: False),), node=True)
    assert rebuild.run([step], dry=False) == 1
    assert not (tmp_path / 'ran').exists()
    record = json.loads(rebuild.RUN_LOG.read_text())['steps'][0]
    assert record['status'] == 'unavailable: missing frozen/evidence.csv'
    assert record['missing_inputs'] == ['frozen/evidence.csv']


def test_missing_export_input_keeps_the_existing_rebuild_skip_behavior(tmp_path, monkeypatch):
    _isolated_runner(tmp_path, monkeypatch)
    step = rebuild.Step('optional-export', 'export', (sys.executable, '-c', 'raise SystemExit(99)'),
                        'keep existing output', '-', (rebuild.Need('upstream/raw.csv', lambda: False),), node=True)
    assert rebuild.run([step], dry=False) == 0
    assert json.loads(rebuild.RUN_LOG.read_text())['steps'][0]['status'].startswith('skipped: missing')


def test_source_public_and_research_profiles_make_their_scope_explicit():
    profiles = {name: rebuild.check_steps(name) for name in rebuild.CHECK_PROFILES}
    keys = {name: {s.key for s in steps} for name, steps in profiles.items()}
    assert keys['source'] < keys['public'] < keys['research']
    assert 'node-tests' in keys['source']
    assert 'node-data-public' in keys['public'] - keys['source']
    assert {'node-data-research', 'python-data-tests'} <= keys['research'] - keys['public']
    assert '--source-only' in next(s for s in profiles['source'] if s.key == 'public-build').run
    assert '--source-only' not in next(s for s in profiles['public'] if s.key == 'public-build').run


def test_check_defaults_to_full_research_and_passes_the_requested_scope(monkeypatch):
    calls = []
    monkeypatch.setattr(rebuild, 'run', lambda steps, dry, profile: calls.append((steps, dry, profile)) or 1)
    monkeypatch.setattr(rebuild, 'sizes', lambda: 0)
    assert rebuild.main(['check']) == 1
    assert calls[-1][2] == 'research'
    assert rebuild.main(['check', '--profile', 'source']) == 1
    assert calls[-1][2] == 'source'
