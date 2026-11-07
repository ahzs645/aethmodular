#!/usr/bin/env python3
"""Verify and reassemble the closeout archive parts without extracting them.

Uses only the Python 3.9+ standard library. Download ARCHIVE_INDEX.json and
the part folders together, preserving their relative paths, then run:

    python3 restore_closeout_archives.py --check-only
    python3 restore_closeout_archives.py

The default manifest is beside this script; --manifest selects another one.
Outputs default to a ``restored`` directory beside the selected manifest.
Part order comes from each manifest ``parts`` list, never a filename glob.
Existing matching archives are verified and reused; different files are never
replaced. A verified temporary file is published with an atomic, no-overwrite
hard link, so the destination filesystem must support ordinary hard links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Optional


CHUNK_BYTES = 8 * 1024 * 1024


class RestoreError(ValueError):
    """The manifest, part bytes or existing output failed verification."""


@dataclass(frozen=True)
class Identity:
    size: int
    sha256: str
    md5: Optional[str]


@dataclass(frozen=True)
class Part:
    path: Path
    relative_path: str
    identity: Identity


@dataclass(frozen=True)
class Archive:
    filename: str
    identity: Identity
    parts: tuple[Part, ...]


class Digests:
    def __init__(self) -> None:
        self.size = 0
        self.sha256 = hashlib.sha256()
        # MD5 is checked only for agreement with provider metadata. SHA-256 is
        # the required archive identity, including when no MD5 is supplied.
        self.md5 = hashlib.md5(usedforsecurity=False)

    def update(self, block: bytes) -> None:
        self.size += len(block)
        self.sha256.update(block)
        self.md5.update(block)

    def record(self) -> dict:
        return {
            "bytes": self.size,
            "sha256": self.sha256.hexdigest(),
            "md5": self.md5.hexdigest(),
        }

    def verify(self, identity: Identity, label: str) -> None:
        actual = self.record()
        for key, expected in (
            ("bytes", identity.size),
            ("sha256", identity.sha256),
            ("md5", identity.md5),
        ):
            if expected is not None and actual[key] != expected:
                raise RestoreError(
                    f"{label}: {key} differs (expected {expected}, found {actual[key]})"
                )


def identity_from(record: object, label: str) -> Identity:
    if not isinstance(record, dict):
        raise RestoreError(f"{label}: expected an object")
    size = record.get("bytes")
    if type(size) is not int or size < 0:
        raise RestoreError(f"{label}: bytes must be a non-negative integer")
    sha = record.get("sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
        raise RestoreError(f"{label}: a SHA-256 identity is required")
    md5 = record.get("md5")
    if md5 is not None and (not isinstance(md5, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", md5)):
        raise RestoreError(f"{label}: invalid MD5 identity")
    return Identity(size, sha.lower(), md5.lower() if md5 is not None else None)


def relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RestoreError(f"{label}: a relative path is required")
    # Reject both POSIX and Windows traversal/drive spellings regardless of
    # the platform on which the download is being restored.
    if any(character in value for character in ("\\", ":", "\x00")):
        raise RestoreError(f"{label}: invalid relative path")
    if value.startswith("/") or any(piece in ("", ".", "..") for piece in value.split("/")):
        raise RestoreError(f"{label}: path must stay beneath the manifest directory")
    return PurePosixPath(value)


def unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RestoreError(f"manifest contains a duplicate key: {key}")
        result[key] = value
    return result


def read_manifest(path: Path) -> tuple[list[Archive], str]:
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_json_object)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise RestoreError("manifest schema_version must be 1")
    records = data.get("archives")
    if not isinstance(records, list) or not records:
        raise RestoreError("manifest archives must be a non-empty list")
    root = path.parent
    archives = []
    names = set()
    part_paths = set()
    for record in records:
        identity = identity_from(record, "archive")
        filename = relative_path(record.get("filename"), "archive filename")
        if len(filename.parts) != 1 or str(filename) in names:
            raise RestoreError("archive filenames must be unique plain filenames")
        names.add(str(filename))
        entries = record.get("parts")
        if not isinstance(entries, list) or not entries:
            raise RestoreError(f"{filename}: parts must be a non-empty ordered list")
        parts = []
        for item in entries:
            expected = identity_from(item, f"{filename} part")
            relative = relative_path(item.get("path"), f"{filename} part path")
            candidate = root.joinpath(*relative.parts).resolve(strict=True)
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise RestoreError(
                    f"{relative}: resolved path escapes the manifest directory"
                ) from error
            if not candidate.is_file() or candidate in part_paths:
                raise RestoreError(f"{relative}: expected a unique regular part file")
            part_paths.add(candidate)
            parts.append(Part(candidate, str(relative), expected))
        if sum(part.identity.size for part in parts) != identity.size:
            raise RestoreError(f"{filename}: part sizes do not sum to the archive size")
        archives.append(Archive(str(filename), identity, tuple(parts)))
    # Provider fields, standalone payloads and a prior local archive_root are
    # deliberately ignored. Only relative part paths authorize file reads.
    return archives, hashlib.sha256(raw).hexdigest()


def read_verified(
    path: Path,
    identity: Identity,
    label: str,
    combined: Optional[Digests] = None,
    destination: Optional[BinaryIO] = None,
) -> Digests:
    digest = Digests()
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != identity.size:
            raise RestoreError(f"{label}: file type or byte length differs")
        while block := source.read(CHUNK_BYTES):
            digest.update(block)
            if combined is not None:
                combined.update(block)
            if destination is not None:
                destination.write(block)
        after = os.fstat(source.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RestoreError(f"{label}: file changed while it was being read")
    digest.verify(identity, label)
    return digest


def verify_parts(archive: Archive, destination: Optional[BinaryIO] = None) -> Digests:
    combined = Digests()
    for part in archive.parts:
        read_verified(part.path, part.identity, part.relative_path, combined, destination)
    combined.verify(archive.identity, archive.filename)
    return combined


def verify_existing(path: Path, archive: Archive) -> bool:
    try:
        existing = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(existing.st_mode):
        raise RestoreError(f"{archive.filename}: existing output is not a regular file")
    read_verified(path, archive.identity, f"existing {archive.filename}")
    return True


def restore_archive(archive: Archive, output_dir: Path, check_only: bool) -> dict:
    if check_only:
        digest = verify_parts(archive)
        status = "checked"
    else:
        output = output_dir / archive.filename
        if verify_existing(output, archive):
            digest = verify_parts(archive)
            status = "reused_verified_existing"
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=output_dir,
                    prefix=f".{archive.filename}.",
                    suffix=".tmp",
                    delete=False,
                ) as destination:
                    temporary = Path(destination.name)
                    digest = verify_parts(archive, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                # Read the completed temporary file back before publishing it.
                digest = read_verified(temporary, archive.identity, archive.filename)
                try:
                    # Unlike os.replace or a POSIX rename, link fails if a
                    # destination appeared while the parts were being read.
                    os.link(temporary, output)
                    status = "restored"
                except FileExistsError:
                    if not verify_existing(output, archive):
                        raise RestoreError(
                            f"{archive.filename}: output changed during publication; retry"
                        ) from None
                    status = "reused_verified_existing"
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return {
        "filename": archive.filename,
        "status": status,
        "parts_verified": len(archive.parts),
        **digest.record(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().with_name("ARCHIVE_INDEX.json"),
        help="archive index (default: ARCHIVE_INDEX.json beside this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: restored beside the manifest; explicit relative paths use the current directory)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify parts and concatenated hashes without writing archives",
    )
    args = parser.parse_args()
    try:
        manifest = args.manifest.resolve(strict=True)
        archives, manifest_sha = read_manifest(manifest)
        output_dir = (args.output_dir or manifest.parent / "restored").resolve()
        if not args.check_only:
            output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for archive in archives:
            print(
                f"Verifying {archive.filename} ({len(archive.parts)} parts)",
                file=sys.stderr,
                flush=True,
            )
            results.append(restore_archive(archive, output_dir, args.check_only))
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol": "verified_ordered_archive_restoration_v1",
                    "status": "passed",
                    "mode": "check_only" if args.check_only else "restore",
                    "manifest_sha256": manifest_sha,
                    "archive_count": len(results),
                    "part_count": sum(result["parts_verified"] for result in results),
                    "archives": results,
                    "extraction_performed": False,
                },
                indent=2,
            )
        )
    except (OSError, ValueError) as error:
        print(f"Restoration stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
