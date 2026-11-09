"""Verify and apply the accepted source-score refresh before gallery publication.

This is an artifact/provenance gate, not a scientific rerun. The historical
review archive retains its original identity. A newer publication archive may
contain either that exact continuum export or the exact accepted refresh; its
own tag/hash are recorded separately. Any other continuum version needs a new
review contract, not an override flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = "gallery/review/closeout_20261107"
MANIFEST_PATH = f"{REVIEW}/manifest.json"
MANIFEST_SHA256 = "3b2c35dc8cd8ce7ca7f835d71b8771e6d827e8bcda9520a87ab56611d5076c64"
OVERLAY_PATH = f"{REVIEW}/continuum_baseline.json"
ACCEPTANCE_PATH = f"{REVIEW}/source_acceptance.json"
EXPORT_PATH = "experimental/continuum_baseline.json"
DATA_RECEIPT_PATH = "_verification/continuum-source.json"
SCIENTIFIC_PATHS = frozenset({
    "research/ftir_hips_chem/workflows/verify_closeout_source_20261107.py",
    "research/ftir_hips_chem/workflows/run_transfer_diagnosis_20261102.py",
    "research/ftir_hips_chem/workflows/run_intercept_fixes_20261103.py",
    "research/ftir_hips_chem/scripts/epo_validation.py",
    "research/ftir_hips_chem/scripts/continuum_anchor.py",
    "research/ftir_hips_chem/scripts/candidate_metrics.py",
    "research/ftir_hips_chem/scripts/spectral_provenance.py",
    "research/ftir_hips_chem/workflows/run_hips_response_20261103.py",
})
CHANGED_TABLES = frozenset({"schema_version", "generated", "settings", "scoreboard", "sweep", "rule_sim", "mae"})
CHANGED_SETTINGS = frozenset({
    "source_evaluation", "metric_definitions", "legacy_metric_aliases", "source_refresh", "target_predictions",
})


class PublicationError(ValueError):
    """The supplied source/data pair is not covered by the accepted review."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_artifact(root: Path, relative: str, digest: str) -> bytes:
    path = root / relative
    require(path.is_file(), f"required review input is missing: {relative}")
    body = path.read_bytes()
    require(sha256(body) == digest, f"review input hash differs: {relative}")
    return body


def accepted_overlay(root: Path = ROOT) -> tuple[bytes, dict]:
    """Check the immutable review and all scientific sources it attests to."""
    manifest_bytes = read_artifact(root, MANIFEST_PATH, MANIFEST_SHA256)
    manifest = json.loads(manifest_bytes)
    require(manifest["schema"] == 1, "unsupported review manifest schema")
    require(manifest["overlay"]["path"] == OVERLAY_PATH, "review overlay path differs")
    require(manifest["source_acceptance"]["path"] == ACCEPTANCE_PATH, "source acceptance path differs")
    require(manifest["scientific_status"] == "fold_local_source_verified", "source evaluation is not verified")
    overlay_bytes = read_artifact(root, OVERLAY_PATH, manifest["overlay"]["sha256"])
    acceptance = json.loads(read_artifact(root, ACCEPTANCE_PATH, manifest["source_acceptance"]["sha256"]))
    require(acceptance["protocol"] == "closeout_source_artifact_acceptance_v1", "unsupported source acceptance protocol")
    require(acceptance["status"] == "passed", "source acceptance did not pass")
    require(acceptance["on_disk_input_hashes_preserved"] is True, "source input hashes were not preserved")
    require(acceptance["same_frozen_site_folds"] is True, "source site folds differ")
    require(set(acceptance["source_code"]) == SCIENTIFIC_PATHS, "source acceptance must cover the eight scientific files")
    for relative, digest in acceptance["source_code"].items():
        read_artifact(root, relative, digest)

    payload = json.loads(overlay_bytes)
    require(payload["schema_version"] == 2, "corrected export must use the verified source schema")
    settings = payload["settings"]
    evaluation, refresh, target = (settings[name] for name in ("source_evaluation", "source_refresh", "target_predictions"))
    require(evaluation["status"] == "corrected", "source evaluation is not corrected")
    require(evaluation["protocol"] == "fixed_k_common_physical_filter_metrics_v1", "source scoring protocol differs")
    require(refresh["protocol"] == "verified_source_scores_on_frozen_export_v1", "source refresh protocol differs")
    require(refresh["base_export_sha256"] == manifest["base_export_sha256"], "source refresh historical export differs")
    require(refresh["source_score_receipt_sha256"] == manifest["source_scores_receipt_sha256"], "source-score receipt differs")
    score_receipt_path = f"{acceptance['paths']['scores_output']}/baseline_scoreboard_receipt.json"
    require(acceptance["input_hashes"][score_receipt_path] == manifest["source_scores_receipt_sha256"],
            "acceptance covers a different source-score receipt")
    require(acceptance["population"]["n_common"] == evaluation["population"]["n_common"], "source populations differ")
    require(acceptance["population"]["source_qc_policy"]["mask_sha256"] ==
            evaluation["population"]["source_qc_policy"]["mask_sha256"], "source QC masks differ")
    perturbation = acceptance["heldout_perturbation"]
    require(perturbation["spectra_unchanged"] is True and perturbation["splits_exactly_unchanged"] is True,
            "held-out integrity check did not preserve spectra and splits")
    require(perturbation["artificial_metric_check"]["metrics_changed"] is True, "held-out integrity scores did not change")
    require(target["status"] == "saved_full_source_comparisons", "historical target status differs")
    require(target["base_export_sha256"] == manifest["base_export_sha256"], "historical target export differs")
    require(target["generated"] == refresh["base_generated"], "historical target generation date differs")

    evidence = {
        "manifest": {"path": MANIFEST_PATH, "sha256": MANIFEST_SHA256},
        "historical_review_base": {
            "data_tag": manifest["base_data_tag"], "archive_sha256": manifest["base_data_sha256"],
            "export_sha256": manifest["base_export_sha256"], "target_generated": target["generated"],
        },
        "overlay": {
            **manifest["overlay"], "scientific_status": manifest["scientific_status"],
            "source_scores_receipt_sha256": manifest["source_scores_receipt_sha256"],
            "source_acceptance": manifest["source_acceptance"],
        },
        "scientific_sources": acceptance["source_code"],
    }
    return overlay_bytes, evidence


def verify_targets(base: dict, overlay: dict) -> list[str]:
    """Only the declared source-score/label sections may change on a refresh."""
    for key in set(base) | set(overlay):
        if key not in CHANGED_TABLES:
            require(key in base and key in overlay and base[key] == overlay[key], f"historical table changed: {key}")
    for key in set(base["settings"]) | set(overlay["settings"]):
        if key not in CHANGED_SETTINGS:
            require(key in base["settings"] and key in overlay["settings"] and
                    base["settings"][key] == overlay["settings"][key], f"historical setting changed: {key}")
    require(overlay["settings"]["source_refresh"]["base_generated"] == base["generated"],
            "historical generation date changed")
    return sorted(set(base) - CHANGED_TABLES)


def verify_archive(archive: Path, tag: str, incoming: bytes) -> dict:
    """Bind the actual archive and its continuum member, not a historical tag."""
    require(bool(re.fullmatch(r"data-[0-9a-f]{12,64}", tag)), "data tag must contain 12–64 lowercase hash characters")
    digest = file_sha256(archive)
    require(digest.startswith(tag.removeprefix("data-")), "publication archive hash does not match its data tag")
    matches = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name not in {EXPORT_PATH, f"./{EXPORT_PATH}"}:
                continue
            matches += 1
            require(member.isfile() and member.size <= 5_000_000, "continuum archive member must be a bounded regular file")
            archived = tar.extractfile(member).read()
            require(archived == incoming, "restored continuum bytes differ from the publication archive member")
    require(matches == 1, "publication archive must contain exactly one continuum export")
    return {"kind": "release_archive", "data_tag": tag, "archive_sha256": digest,
            "verification": "hash_tag_and_continuum_member_verified"}


def prepare_overlay(root: Path, data_root: Path, *, archive: Path | None = None,
                    data_tag: str | None = None, require_historical_base: bool = False) -> tuple[bytes, dict]:
    """Return accepted bytes and evidence without changing the caller's inputs."""
    overlay_bytes, evidence = accepted_overlay(root)
    path = data_root / EXPORT_PATH
    require(path.is_file(), f"publication continuum export is missing: {EXPORT_PATH}")
    incoming = path.read_bytes()
    incoming_hash = sha256(incoming)
    historical_hash = evidence["historical_review_base"]["export_sha256"]
    corrected_hash = evidence["overlay"]["sha256"]
    require(incoming_hash in {historical_hash, corrected_hash},
            "continuum export is neither the frozen historical export nor the accepted corrected export")
    require((archive is None) == (data_tag is None), "archive and data tag must be supplied together")
    snapshot = (verify_archive(archive, data_tag, incoming) if archive is not None else
                {"kind": "local_directory", "verification": "continuum_artifact_verified; no archive hash asserted"})
    if require_historical_base:
        historical = evidence["historical_review_base"]
        require(snapshot.get("data_tag") == historical["data_tag"] and
                snapshot.get("archive_sha256") == historical["archive_sha256"] and incoming_hash == historical_hash,
                "historical closeout CI requires its exact original archive and export")
    payload = json.loads(overlay_bytes)
    if incoming_hash == historical_hash:
        tables = verify_targets(json.loads(incoming), payload)
        preservation = {"verification": "compared_with_hash_verified_historical_export", "unchanged_tables": tables}
    else:
        preservation = {"verification": "exact_accepted_corrected_export", "unchanged_tables":
                        sorted(set(payload) - CHANGED_TABLES)}
    receipt = {
        "schema_version": 1, "protocol": "accepted_continuum_publication_v1", "status": "verified",
        "input_snapshot": snapshot, "incoming_export": {"path": EXPORT_PATH, "sha256": incoming_hash},
        **evidence, "target_preservation": preservation,
        "served_export": {"path": EXPORT_PATH, "sha256": corrected_hash, "source_evaluation_status": "corrected"},
        "hash_scope": "The incoming publication archive and historical review archive are distinct identities. "
                      "The accepted overlay and served continuum bytes have their own hashes.",
        "not_established": ["new execution of the scientific fits", "full research reconciliation", "independent target EC accuracy"],
    }
    return overlay_bytes, receipt


def apply_overlay(root: Path, data_root: Path, *, archive: Path | None = None,
                  data_tag: str | None = None, require_historical_base: bool = False,
                  legacy_fixture: Path | None = None) -> dict:
    overlay_bytes, receipt = prepare_overlay(root, data_root, archive=archive, data_tag=data_tag,
                                             require_historical_base=require_historical_base)
    export = data_root / EXPORT_PATH
    if legacy_fixture is not None:
        require(receipt["incoming_export"]["sha256"] == receipt["historical_review_base"]["export_sha256"],
                "legacy browser fixture requires the exact historical export")
        require(legacy_fixture.resolve() != export.resolve(), "legacy fixture must not replace the served export")
        legacy_fixture.parent.mkdir(parents=True, exist_ok=True)
        legacy_fixture.write_bytes(export.read_bytes())
    export.write_bytes(overlay_bytes)
    evidence_file = data_root / DATA_RECEIPT_PATH
    evidence_file.parent.mkdir(parents=True, exist_ok=True)
    evidence_file.write_bytes(json_bytes(receipt))
    require(sha256(export.read_bytes()) == receipt["served_export"]["sha256"], "served continuum write did not verify")
    return receipt


def verify_built(root: Path, data_root: Path, receipt: dict) -> dict:
    """Check the actual built file immediately before publishing the site."""
    overlay_bytes, evidence = accepted_overlay(root)
    require(receipt.get("protocol") == "accepted_continuum_publication_v1" and receipt.get("status") == "verified",
            "continuum publication receipt is missing or incomplete")
    for key, value in evidence.items():
        require(receipt.get(key) == value, f"built receipt describes different accepted evidence: {key}")
    require(receipt.get("served_export", {}).get("path") == EXPORT_PATH, "served receipt path differs")
    require(receipt.get("served_export", {}).get("sha256") == sha256(overlay_bytes), "served receipt digest differs")
    built = data_root / EXPORT_PATH
    require(built.is_file() and built.read_bytes() == overlay_bytes, "built continuum export differs from the accepted corrected export")
    require(json.loads(built.read_bytes())["settings"]["source_evaluation"]["status"] == "corrected",
            "built continuum has no corrected source evaluation")
    return {**receipt, "built_export": {"path": f"data/{EXPORT_PATH}", "sha256": sha256(overlay_bytes),
                                        "verification": "exact_served_bytes_verified"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("apply", "verify-built"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--data-tag")
    parser.add_argument("--require-historical-base", action="store_true")
    parser.add_argument("--legacy-fixture", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "apply":
            receipt = apply_overlay(args.root, args.data_root, archive=args.archive, data_tag=args.data_tag,
                                    require_historical_base=args.require_historical_base, legacy_fixture=args.legacy_fixture)
        else:
            require(not any((args.archive, args.data_tag, args.require_historical_base, args.legacy_fixture)),
                    "verify-built uses the recorded input receipt, not replacement archive arguments")
            receipt = verify_built(args.root, args.data_root, json.loads(args.receipt.read_text()))
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(json_bytes(receipt))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        parser.exit(1, f"continuum publication blocked: {exc}\n")
    print(f"Verified corrected continuum: {receipt['served_export']['sha256']}")


if __name__ == "__main__":
    main()
