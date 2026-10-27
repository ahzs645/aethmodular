"""The gallery rebuild's step list stays runnable, ordered, complete and documented."""
import importlib.util
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


def test_each_private_section_has_data_paths_to_check():
    markers = rebuild.private_markers()
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
