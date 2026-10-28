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


# ---- reconcile: keep what the bucket drops, take what it adds, set aside what is impossible

HEAD = ("# File Updated: 2026-10-13\n# Data version 3.1.4\n# Site: Haifa, Israel\n"
        "Site_Code,Filter_ID,Start_Year_local,Start_Month_local,Start_Day_local,Start_hour_local,"
        "End_Year_local,End_Month_local,End_Day_local,End_hour_local,Parameter_Code,Parameter_Name,Value,Method_Code\n")


def row(fid, start="2025,12,14,9.0", end="2025,12,22,9.0", code="28101", value="10.0"):
    return f"ILHA,{fid},{start},{end},{code},Filter PM2.5 mass,{value},116\n"


def test_empty_remote_never_replaces_rows(tmp_path, monkeypatch):
    """TWKA on 2026-10-28: header lines and no rows."""
    path = tmp_path / "FilterBased_ChemSpecPM25_TWKA.csv"
    path.write_text(HEAD + row("TWKA-0001") + row("TWKA-0002"))
    monkeypatch.setattr(pull.requests, "get", lambda *a, **k: FakeResponse(HEAD.encode()))
    item = spec(path)
    assert pull.download_one(item)[2] == "held"
    assert "TWKA-0002" in path.read_text()
    assert pull.remote_copy(path).read_text() == HEAD          # SPARTAN's bytes kept for the next comparison
    assert pull.download_one(spec(path))[2] == "unchanged"     # the same empty file is not news


def test_backwards_sampling_is_set_aside(tmp_path, monkeypatch):
    """ILHA-0216 on 2026-10-28: starts 2026-11-17, after the file's update (2026-10-13)."""
    path = tmp_path / "FilterBased_ChemSpecPM25_ILHA.csv"
    path.write_text(HEAD + row("ILHA-0213"))
    remote = HEAD + row("ILHA-0213") + row("ILHA-0216", start="2026,11,17,9.0", end="2026,1,9,9.0")
    monkeypatch.setattr(pull.requests, "get", lambda *a, **k: FakeResponse(remote.encode()))
    item = spec(path)
    assert pull.download_one(item)[2] == "updated"
    assert "ILHA-0216" not in path.read_text()
    assert "ILHA-0216" in pull.quarantine_copy(path).read_text()
    assert item.sync.quarantined == ["ILHA-0216"]


def test_end_before_a_plausible_start_is_installed_and_flagged(tmp_path, monkeypatch):
    """ETAD-0262 on 2026-10-28: 2025-01-25 to 2024-02-02, with good data; exporters key on the start."""
    path = tmp_path / "FilterBased_ChemSpecPM25_ETAD.csv"
    remote = HEAD + row("ETAD-0262", start="2025,1,25,9.0", end="2024,2,2,9.0")
    monkeypatch.setattr(pull.requests, "get", lambda *a, **k: FakeResponse(remote.encode()))
    item = spec(path)
    assert pull.download_one(item)[2] == "downloaded"
    assert "ETAD-0262" in path.read_text() and not pull.quarantine_copy(path).exists()
    assert item.sync.flagged == ["ETAD-0262"]


def test_dropped_rows_are_kept_and_revisions_taken(tmp_path, monkeypatch):
    path = tmp_path / "FilterBased_ChemSpecPM25_ILHA.csv"
    path.write_text(HEAD + row("ILHA-0001", value="10.0") + row("ILHA-0002"))
    remote = HEAD + row("ILHA-0001", value="12.5") + row("ILHA-0003")
    monkeypatch.setattr(pull.requests, "get", lambda *a, **k: FakeResponse(remote.encode()))
    item = spec(path)
    assert pull.download_one(item)[2] == "updated"
    text = path.read_text()
    ilha1 = [ln for ln in text.splitlines() if ",ILHA-0001," in ln]
    assert len(ilha1) == 1 and ",12.5," in ilha1[0]              # SPARTAN's revision wins, once
    assert "ILHA-0002" in text and "ILHA-0003" in text           # kept, and added
    assert item.sync.retained == ["ILHA-0002"]
    assert text.startswith("# File Updated")                     # SPARTAN's header lines stay


def test_unchanged_remote_installs_byte_for_byte(tmp_path, monkeypatch):
    path = tmp_path / "FilterBased_ChemSpecPM25_ILHA.csv"
    remote = (HEAD + row("ILHA-0001")).encode()
    monkeypatch.setattr(pull.requests, "get", lambda *a, **k: FakeResponse(remote))
    assert pull.download_one(spec(path))[2] == "downloaded"
    assert path.read_bytes() == remote and not pull.quarantine_copy(path).exists()


def test_as_published_takes_the_bucket_as_is(tmp_path, monkeypatch):
    path = tmp_path / "FilterBased_ChemSpecPM25_TWKA.csv"
    path.write_text(HEAD + row("TWKA-0001"))
    monkeypatch.setattr(pull.requests, "get", lambda *a, **k: FakeResponse(HEAD.encode()))
    assert pull.download_one(spec(path), as_published=True)[2] == "updated"
    assert path.read_text() == HEAD
