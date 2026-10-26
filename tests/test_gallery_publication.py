"""Publication must retain linked data assets and their relative URLs."""
import gzip
import getpass
import importlib.util
import json
from pathlib import Path
import re
import tarfile

import pytest

PATH = Path(__file__).resolve().parents[1] / 'gallery/share/publish_pages.py'
spec = importlib.util.spec_from_file_location('gallery_publication', PATH)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def test_pack_includes_linked_directories_and_scrubs_compressed_metadata(tmp_path, monkeypatch):
    data, exports = tmp_path / 'data', tmp_path / 'exports'
    data.mkdir()
    (exports / 'native').mkdir(parents=True)
    (data / 'index.json').write_text(json.dumps({'file': 'network/native/site.json.gz'}))
    source = {'source': '/Users/example/github/aethmodular/research/input.csv', 'value': 1.25}
    (exports / 'native/site.json.gz').write_bytes(gzip.compress(json.dumps(source).encode()))
    (exports / 'native/trace.bin').write_bytes(bytes(range(256)))
    (exports / '.DS_Store').write_bytes(b'ignored')
    (data / 'network').symlink_to(exports, target_is_directory=True)
    monkeypatch.setattr(publication, 'DATA', data)
    monkeypatch.setattr(publication, 'FORBIDDEN', re.compile(r'/Users/|GoogleDrive-'))

    archive = tmp_path / 'first.tar.gz'
    digest, count, scrubbed = publication.pack_data(archive)
    assert count == 3 and scrubbed == 1
    with tarfile.open(archive) as tar:
        assert tar.getnames() == ['index.json', 'network/native/site.json.gz', 'network/native/trace.bin']
        metadata = json.loads(gzip.decompress(tar.extractfile('network/native/site.json.gz').read()))
        assert metadata == {'source': 'research/input.csv', 'value': 1.25}
        assert tar.extractfile('network/native/trace.bin').read() == bytes(range(256))
    assert publication.pack_data(tmp_path / 'second.tar.gz') == (digest, count, scrubbed)


def test_pack_rejects_missing_linked_data(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'missing').symlink_to(tmp_path / 'absent', target_is_directory=True)
    monkeypatch.setattr(publication, 'DATA', data)
    with pytest.raises(SystemExit, match='broken data link'):
        publication.pack_data(tmp_path / 'data.tar.gz')


def test_pack_rejects_link_cycles(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'loop').symlink_to(data, target_is_directory=True)
    monkeypatch.setattr(publication, 'DATA', data)
    with pytest.raises(SystemExit, match='directory cycle'):
        publication.pack_data(tmp_path / 'data.tar.gz')


def test_release_readme_identifies_the_successful_build():
    text = publication.README_RELEASE.format(sha='abc123', tag='data-example', run_id=123)
    assert '/aethmodular/commit/abc123' in text
    assert '/releases/tag/data-example' in text
    assert '/actions/runs/123' in text


def test_privacy_check_accepts_browser_identifiers_and_retains_personal_path_checks(monkeypatch):
    monkeypatch.setattr(getpass, 'getuser', lambda: 'agent')
    monkeypatch.setattr(publication.subprocess, 'run', lambda *a, **kw:
                        publication.subprocess.CompletedProcess(a, 0, stdout='person@example.test\n'))
    pattern = publication.forbidden_pattern()
    assert not pattern.search('navigator.userAgent; personalization; user-agent')
    assert pattern.search('/Users/example/file.json')
    assert pattern.search('GoogleDrive-example')
    assert pattern.search('person@example.test')
    assert pattern.search('/person/input.csv')
