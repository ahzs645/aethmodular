"""The public SPARTAN file check must notice same-name revisions safely."""

from pathlib import Path

from scripts.pipelines import spartan_pull_and_summarize as pull


class FakeResponse:
    def __init__(self, body: bytes, fail_after_chunk: bool = False):
        self.body = body
        self.fail_after_chunk = fail_after_chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.body
        if self.fail_after_chunk:
            raise OSError("connection dropped")


def spec(path: Path) -> pull.FileSpec:
    return pull.FileSpec("FilterBased", "ChemSpecPM25", "ETAD", path.name,
                         "https://example.test/file.csv", path)


def test_check_detects_same_name_revision_without_writing(tmp_path, monkeypatch):
    path = tmp_path / "ChemSpecPM25_ETAD.csv"
    path.write_bytes(b"original")
    monkeypatch.setattr(pull.requests, "get", lambda *a, **k: FakeResponse(b"revised"))
    assert pull.download_one(spec(path), check_only=True)[2] == "changed"
    assert path.read_bytes() == b"original"
    assert not path.with_suffix(".csv.part").exists()
    assert pull.download_one(spec(path))[2] == "updated"
    assert path.read_bytes() == b"revised"
    assert pull.download_one(spec(path))[2] == "unchanged"


def test_failed_refresh_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "ChemSpecPM25_ETAD.csv"
    path.write_bytes(b"original")
    monkeypatch.setattr(pull.requests, "get", lambda *a, **k: FakeResponse(b"partial", True))
    assert pull.download_one(spec(path))[2].startswith("error:")
    assert path.read_bytes() == b"original"
    assert not path.with_suffix(".csv.part").exists()


def test_missing_cache_reports_changed_s3_object(tmp_path, monkeypatch, capsys):
    item = spec(tmp_path / "ChemSpecPM25_ETAD.csv")
    item.remote_etag = "new-tag"
    monkeypatch.setattr(pull, "inventory_metadata", lambda: {
        pull.spec_key(item): {"remote_etag": "old-tag"}
    })
    assert pull.download_one(item, check_only=True)[2] == "new"
    assert pull.print_update_check([item], [(item, 0, "new")]) == 0
    output = capsys.readouterr().out
    assert "Objects with changed S3 ETags and no local bytes: 1" in output
    assert "S3 object changed: FilterBased/ChemSpecPM25/ChemSpecPM25_ETAD.csv" in output
