"""Import the approved 26-slide meeting bundle without refitting or overwriting.

Usage from the repository root:
    uv run python research/ftir_hips_chem/workflows/import_ann_weekly_20260917.py \
        ~/Downloads/Aethmodular_Weekly_Meeting_Package_2026-09-17.zip --check
Remove --check to copy verified files. This command never commits or pushes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from zipfile import ZipFile

DELIVERABLE = Path("deliverables/ann_weekly_2026-09-17")
MAX_BYTES = 30_000_000
MAX_FILES = 200


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_member(info, archive_root: str) -> Path | None:
    """Reject traversal, platform-specific paths, and symbolic links."""
    name = info.filename
    p = PurePosixPath(name)
    if (p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name
            or "\x00" in name or not p.parts or p.parts[0] != archive_root):
        raise ValueError(f"Unsafe or unexpected archive path: {name!r}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"Archive symbolic link is not allowed: {name!r}")
    if info.is_dir():
        return None
    if len(p.parts) < 2:
        raise ValueError(f"Missing file path below archive root: {name!r}")
    return Path(*p.parts[1:])


def reject_symlinks(path: Path) -> None:
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Refusing a symlink destination: {parent}")


def load_verified(archive: Path, manifest: dict) -> dict[Path, bytes]:
    """Validate the complete archive and primary artifact identities first."""
    if archive.stat().st_size != manifest["source_archive_size_bytes"]:
        raise ValueError("Archive size differs from the approved meeting package")
    raw = archive.read_bytes()
    if digest(raw) != manifest["source_archive_sha256"]:
        raise ValueError("Archive SHA-256 differs from the approved meeting package")
    payload: dict[Path, bytes] = {}
    with ZipFile(io.BytesIO(raw)) as z:
        if len(z.infolist()) > MAX_FILES:
            raise ValueError("Archive exceeds the file-count safety limit")
        if sum(i.file_size for i in z.infolist()) > MAX_BYTES:
            raise ValueError("Archive exceeds the expanded-size safety limit")
        for info in z.infolist():
            rel = validate_member(info, manifest["archive_root"])
            if rel is None:
                continue
            if rel in payload:
                raise ValueError(f"Duplicate archive entry: {rel}")
            payload[rel] = z.read(info)  # ZipFile also validates CRC.
    for name, expected in manifest["artifacts"].items():
        data = payload.get(Path(name))
        if data is None or len(data) != expected["size_bytes"]:
            raise ValueError(f"Missing or incorrect artifact size: {name}")
        if digest(data) != expected["sha256"]:
            raise ValueError(f"Artifact SHA-256 mismatch: {name}")
    for rel, data in payload.items():
        if rel.suffix == ".pptx":
            with ZipFile(io.BytesIO(data)) as pptx:
                names = pptx.namelist()
                slides = sum(bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", n)) for n in names)
                notes = sum(bool(re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n)) for n in names)
                expected_n = manifest["main_slides"] + manifest["backup_slides"]
                if slides != expected_n or notes != expected_n:
                    raise ValueError(f"Expected {expected_n} slides and note parts, got {slides}/{notes}")
        if rel.suffix == ".pdf" and not data.startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF: {rel}")
    return payload


def install(archive: Path, repo: Path, manifest: dict, check: bool = False) -> dict:
    payload = load_verified(archive, manifest)
    destination = repo / DELIVERABLE / "bundle"
    reject_symlinks(destination)
    missing = []
    # Complete preflight: a differing file aborts before any files are written.
    for rel, data in payload.items():
        target = destination / rel
        reject_symlinks(target)
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise FileExistsError(f"Refusing to replace an existing artifact: {target}")
        else:
            missing.append((target, data))
    receipt = {
        "archive_sha256": manifest["source_archive_sha256"],
        "files_verified": len(payload),
        "already_identical": len(payload) - len(missing),
        "files_to_copy": len(missing),
        "check_only": check,
        "destination": str(destination),
        "analyses_refit": False,
        "git_commit_or_push_performed": False,
    }
    if not check:
        for target, data in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:  # No overwrite, including a race after preflight.
                f.write(data)
        receipt["files_copied"] = len(missing)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify without creating files")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    repo = args.repo_root.absolute()
    try:
        manifest = json.loads((repo / DELIVERABLE / "package_manifest.json").read_text())
        result = install(args.archive.expanduser(), repo, manifest, args.check)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Import stopped: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if not args.check:
        print("Verified package is installed locally. Review git status and commit only the intended files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
