"""The flagged-section manifest and how the publish scripts keep a private section's data out."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gallery/share'))
spec = importlib.util.spec_from_file_location('gallery_sections', ROOT / 'gallery/share/sections.py')
sections = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sections)


def test_manifest_lists_tabs_and_data_for_every_section():
    manifest = sections.load()
    assert manifest, 'no sections declared'
    for key, s in manifest.items():
        assert isinstance(s['public'], bool), key
        assert s['tabs'] and all(isinstance(t, str) for t in s['tabs']), key
        assert (ROOT / 'gallery/app/src/sections' / key).is_dir(), f'no module for section {key}'


def test_private_paths_are_matched_by_folder_not_prefix():
    dirs = ['aethalometer']
    assert sections.is_private('aethalometer/index.json', dirs)
    assert sections.is_private('aethalometer/daily/ETAD.json', dirs)
    assert not sections.is_private('aethalometer_notes.json', dirs)
    assert not sections.is_private('aeronet/index.json', dirs)


def test_private_lists_follow_the_flag(tmp_path):
    manifest = tmp_path / 'sections.json'
    manifest.write_text(json.dumps({'sections': {
        'a': {'public': False, 'tabs': ['t1'], 'data': ['d1/']},
        'b': {'public': True, 'tabs': ['t2'], 'data': ['d2']},
    }}))
    assert sections.private_data_dirs(manifest) == ['d1']
    assert sections.private_tabs(manifest) == ['t1']


def test_publish_packer_skips_private_data(tmp_path, monkeypatch):
    pub = importlib.util.spec_from_file_location('publish_pages', ROOT / 'gallery/share/publish_pages.py')
    publish = importlib.util.module_from_spec(pub)
    pub.loader.exec_module(publish)
    data = tmp_path / 'data'
    (data / 'aethalometer').mkdir(parents=True)
    (data / 'aethalometer/index.json').write_text('{}')
    (data / 'meta.json').write_text('{}')
    monkeypatch.setattr(publish, 'DATA', data)
    monkeypatch.setattr(publish, 'private_data_dirs', lambda: ['aethalometer'])
    import tarfile
    out = tmp_path / 'data.tar.gz'
    publish.pack_data(out)
    with tarfile.open(out) as tar:
        assert tar.getnames() == ['meta.json']
