"""Accepted source scores must reach both a new archive and the served build."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gallery/share"))
import continuum_publication as gate  # noqa: E402


@pytest.fixture
def review_root(tmp_path):
    root = tmp_path / "source"
    paths = {gate.MANIFEST_PATH, gate.OVERLAY_PATH, gate.ACCEPTANCE_PATH} | gate.SCIENTIFIC_PATHS
    for relative in paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return root


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    destination = root / gate.EXPORT_PATH
    destination.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / gate.OVERLAY_PATH, destination)
    return root


def archive_with_members(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, body in members:
            entry = tarfile.TarInfo(name)
            entry.size = len(body)
            tar.addfile(entry, io.BytesIO(body))
    return "data-" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def test_accepted_review_checks_all_scientific_sources(review_root, data_root):
    original = (data_root / gate.EXPORT_PATH).read_bytes()
    payload, receipt = gate.prepare_overlay(review_root, data_root)
    assert payload == original
    assert (data_root / gate.EXPORT_PATH).read_bytes() == original
    assert set(receipt["scientific_sources"]) == gate.SCIENTIFIC_PATHS
    assert receipt["served_export"]["source_evaluation_status"] == "corrected"
    assert receipt["historical_review_base"]["data_tag"] == "data-82d701248232"
    assert receipt["historical_review_base"]["export_sha256"] != receipt["served_export"]["sha256"]
    assert receipt["target_preservation"]["verification"] == "exact_accepted_corrected_export"
    assert receipt["input_snapshot"]["kind"] == "local_directory"


@pytest.mark.parametrize("relative", sorted(gate.SCIENTIFIC_PATHS))
def test_a_changed_scientific_source_blocks_publication(review_root, data_root, relative):
    with (review_root / relative).open("ab") as stream:
        stream.write(b"\n# changed source\n")
    with pytest.raises(gate.PublicationError, match="review input hash differs"):
        gate.prepare_overlay(review_root, data_root)


@pytest.mark.parametrize("relative", [gate.MANIFEST_PATH, gate.OVERLAY_PATH, gate.ACCEPTANCE_PATH])
def test_changed_review_artifacts_cannot_redefine_the_accepted_result(review_root, data_root, relative):
    with (review_root / relative).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(gate.PublicationError, match="review input hash differs"):
        gate.prepare_overlay(review_root, data_root)


@pytest.mark.parametrize("relative", [gate.MANIFEST_PATH, gate.OVERLAY_PATH, sorted(gate.SCIENTIFIC_PATHS)[0]])
def test_sparse_checkout_must_contain_the_required_evidence(review_root, data_root, relative):
    (review_root / relative).unlink()
    with pytest.raises(gate.PublicationError, match="required review input is missing"):
        gate.prepare_overlay(review_root, data_root)


def test_an_unreviewed_export_fails_before_any_write(review_root, data_root):
    path = data_root / gate.EXPORT_PATH
    data = json.loads(path.read_bytes())
    data["scoreboard"]["improve_mae"][0] += 1
    unreviewed = gate.json_bytes(data)
    path.write_bytes(unreviewed)
    with pytest.raises(gate.PublicationError, match="neither the frozen historical export nor the accepted corrected export"):
        gate.apply_overlay(review_root, data_root)
    assert path.read_bytes() == unreviewed
    assert not (data_root / gate.DATA_RECEIPT_PATH).exists()


def test_target_preservation_compares_all_unmodified_tables_and_settings():
    base = {"schema_version": 1, "generated": "original-date", "settings": {"calendar": "belg_feb"},
            "addis_fits": {"n": [172], "intercept": [0.1]}, "scoreboard": {"old": [1]}}
    changed = json.loads(json.dumps(base))
    changed["schema_version"] = 2
    changed["scoreboard"] = {"new": [2]}
    changed["settings"]["source_refresh"] = {"base_generated": "original-date"}
    assert gate.verify_targets(base, changed) == ["addis_fits"]
    changed["addis_fits"]["intercept"] = [0]
    with pytest.raises(gate.PublicationError, match="historical table changed: addis_fits"):
        gate.verify_targets(base, changed)
    changed["addis_fits"] = base["addis_fits"]
    changed["settings"]["calendar"] = "dry_feb"
    with pytest.raises(gate.PublicationError, match="historical setting changed: calendar"):
        gate.verify_targets(base, changed)


def test_new_archive_identity_is_not_mislabeled_as_the_historical_review(review_root, data_root, tmp_path):
    body = (data_root / gate.EXPORT_PATH).read_bytes()
    archive = tmp_path / "prospective.tar.gz"
    tag = archive_with_members(archive, [(gate.EXPORT_PATH, body), ("new-context.json", b"{}")])
    _, receipt = gate.prepare_overlay(review_root, data_root, archive=archive, data_tag=tag)
    assert receipt["input_snapshot"]["data_tag"] == tag
    assert receipt["input_snapshot"]["archive_sha256"] == gate.file_sha256(archive)
    assert receipt["input_snapshot"]["archive_sha256"] != receipt["historical_review_base"]["archive_sha256"]
    with pytest.raises(gate.PublicationError, match="requires its exact original archive"):
        gate.prepare_overlay(review_root, data_root, archive=archive, data_tag=tag, require_historical_base=True)


def test_archive_identity_and_member_bytes_are_both_required(review_root, data_root, tmp_path):
    body = (data_root / gate.EXPORT_PATH).read_bytes()
    archive = tmp_path / "data.tar.gz"
    tag = archive_with_members(archive, [(gate.EXPORT_PATH, body)])
    with pytest.raises(gate.PublicationError, match="archive hash does not match"):
        gate.prepare_overlay(review_root, data_root, archive=archive, data_tag="data-" + "0" * 12)
    tag = archive_with_members(archive, [(gate.EXPORT_PATH, body + b"\n")])
    with pytest.raises(gate.PublicationError, match="restored continuum bytes differ"):
        gate.prepare_overlay(review_root, data_root, archive=archive, data_tag=tag)
    tag = archive_with_members(archive, [(gate.EXPORT_PATH, body), (gate.EXPORT_PATH, body)])
    with pytest.raises(gate.PublicationError, match="exactly one continuum export"):
        gate.prepare_overlay(review_root, data_root, archive=archive, data_tag=tag)
    tag = archive_with_members(archive, [("unrelated.json", b"{}")])
    with pytest.raises(gate.PublicationError, match="exactly one continuum export"):
        gate.prepare_overlay(review_root, data_root, archive=archive, data_tag=tag)


def test_built_site_cannot_fall_back_to_stale_or_changed_scores(review_root, data_root):
    receipt = gate.apply_overlay(review_root, data_root)
    verified = gate.verify_built(review_root, data_root, receipt)
    assert verified["built_export"]["sha256"] == receipt["overlay"]["sha256"]
    path = data_root / gate.EXPORT_PATH
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(gate.PublicationError, match="built continuum export differs"):
        gate.verify_built(review_root, data_root, receipt)


def test_built_receipt_cannot_relabel_the_historical_data_archive(review_root, data_root):
    receipt = gate.apply_overlay(review_root, data_root)
    receipt["historical_review_base"]["data_tag"] = "data-" + "f" * 12
    with pytest.raises(gate.PublicationError, match="different accepted evidence: historical_review_base"):
        gate.verify_built(review_root, data_root, receipt)


def publication_module():
    spec = importlib.util.spec_from_file_location("publication_gate_test", ROOT / "gallery/share/publish_pages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prospective_archive_contains_accepted_bytes_and_separate_evidence(review_root, data_root, tmp_path, monkeypatch):
    publication = publication_module()
    monkeypatch.setattr(publication, "ROOT", review_root)
    monkeypatch.setattr(publication, "DATA", data_root)
    original = (data_root / gate.EXPORT_PATH).read_bytes()
    archive = tmp_path / "gallery-data.tar.gz"
    digest, count, scrubbed = publication.pack_verified_data(archive)
    assert count == 2 and scrubbed == 0
    with tarfile.open(archive) as tar:
        assert tar.extractfile(gate.EXPORT_PATH).read() == original
        embedded = json.load(tar.extractfile(gate.DATA_RECEIPT_PATH))
        assert embedded["served_export"]["sha256"] == gate.sha256(original)
        assert embedded["historical_review_base"]["data_tag"] == "data-82d701248232"
    sidecar = json.loads(archive.with_name(publication.DATA_VERIFICATION_ASSET).read_text())
    assert sidecar["publication_snapshot"]["archive_sha256"] == digest
    assert sidecar["publication_snapshot"]["data_tag"] == "data-" + digest[:12]
    assert (data_root / gate.EXPORT_PATH).read_bytes() == original
    assert publication.pack_verified_data(tmp_path / "same.tar.gz") == (digest, count, scrubbed)


def test_release_entrypoint_refuses_unverified_data_before_upload(review_root, data_root, tmp_path, monkeypatch):
    publication = publication_module()
    monkeypatch.setattr(publication, "ROOT", review_root)
    monkeypatch.setattr(publication, "DATA", data_root)
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return publication.subprocess.CompletedProcess(args, 0, stdout="a" * 40 if "rev-parse" in args else "")

    monkeypatch.setattr(publication, "run", fake_run)
    (data_root / gate.EXPORT_PATH).write_text("{}")
    with pytest.raises(gate.PublicationError, match="neither the frozen historical export"):
        publication.publish_release(dry_run=False)
    assert not any(call[0] == "gh" for call in calls)


def test_dry_run_can_review_a_local_candidate_without_fetching_or_publishing(review_root, data_root, monkeypatch):
    publication = publication_module()
    monkeypatch.setattr(publication, "ROOT", review_root)
    monkeypatch.setattr(publication, "DATA", data_root)
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return publication.subprocess.CompletedProcess(args, 0, stdout="a" * 40)

    monkeypatch.setattr(publication, "run", fake_run)
    publication.publish_release(dry_run=True)
    assert calls == [("git", "rev-parse", "HEAD")]


def test_local_fallback_restores_inputs_after_a_failed_build(review_root, data_root, monkeypatch):
    publication = publication_module()
    monkeypatch.setattr(publication, "ROOT", review_root)
    monkeypatch.setattr(publication, "DATA", data_root)
    original = (data_root / gate.EXPORT_PATH).read_bytes()
    with pytest.raises(RuntimeError, match="build failed"):
        with publication.staged_verified_continuum():
            assert (data_root / gate.DATA_RECEIPT_PATH).is_file()
            raise RuntimeError("build failed")
    assert (data_root / gate.EXPORT_PATH).read_bytes() == original
    assert not (data_root / gate.DATA_RECEIPT_PATH).exists()


def test_live_readback_binds_source_archive_and_served_scientific_bytes():
    publication = publication_module()
    source, tag, archive_hash = "a" * 40, "data-" + "b" * 12, "b" * 64
    corrected = (ROOT / gate.OVERLAY_PATH).read_bytes()
    digest = gate.sha256(corrected)
    receipt = {"source_sha": source, "data_tag": tag, "data_sha256": archive_hash,
               "continuum_source": {"status": "verified", "served_export": {"sha256": digest},
                                    "built_export": {"sha256": digest}}}
    seen = []

    def fetch(url):
        seen.append(url)
        return gate.json_bytes(receipt) if "/verification-receipt.json?" in url else corrected

    publication.verify_served_publication("https://example.test/gallery/", source, tag, archive_hash, digest, fetch=fetch)
    assert len(seen) == 2 and all(f"source={source}&data={tag}" in url for url in seen)
    receipt["source_sha"] = "c" * 40
    with pytest.raises(gate.PublicationError, match="served source/data receipt differs"):
        publication.verify_served_publication("https://example.test/gallery/", source, tag, archive_hash, digest, fetch=fetch)
    receipt["source_sha"] = source

    def stale_file(url):
        return gate.json_bytes(receipt) if "/verification-receipt.json?" in url else b'{"schema_version":1}'

    with pytest.raises(gate.PublicationError, match="live continuum bytes differ"):
        publication.verify_served_publication("https://example.test/gallery/", source, tag, archive_hash, digest, fetch=stale_file)


def test_normal_deploy_and_closeout_use_the_same_gate_before_data_checks():
    for path in (ROOT / "gallery/share/pages-deploy.yml", ROOT / ".github/workflows/gallery-closeout.yml"):
        workflow = path.read_text()
        apply = workflow.index("gallery/share/continuum_publication.py apply")
        public_checks = workflow.index("npm run test:data:public", apply)
        build = workflow.index("npm run build", public_checks)
        built_gate = workflow.index("gallery/share/continuum_publication.py verify-built", build)
        assert apply < public_checks < build < built_gate
    normal = (ROOT / "gallery/share/pages-deploy.yml").read_text()
    assert "gallery/review" in normal and "research/ftir_hips_chem/workflows" in normal
    historical = (ROOT / ".github/workflows/gallery-closeout.yml").read_text()
    assert "--require-historical-base" in historical
    assert "DATA_SHA256: 82d701248232b1975ea2e452bc1d682877e811103a2aa1040aff9d4adc989051" in historical
    fallback = (ROOT / "gallery/share/pages-static-deploy.yml").read_text()
    assert fallback.index("continuum_publication.py verify-built") < fallback.index("actions/upload-pages-artifact")
    assert "--root verification-source" in fallback
