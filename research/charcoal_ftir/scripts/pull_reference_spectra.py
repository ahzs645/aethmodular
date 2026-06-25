#!/usr/bin/env python3
"""Download open charcoal/biochar FTIR reference datasets.

Raw files are written under ``data/raw/<source_id>/``. The data directories are
ignored by git; keep this script and ``sources.json`` committed, not the data.

Examples
--------
    /opt/anaconda3/bin/python3.13 scripts/pull_reference_spectra.py --dry-run
    /opt/anaconda3/bin/python3.13 scripts/pull_reference_spectra.py
    /opt/anaconda3/bin/python3.13 scripts/pull_reference_spectra.py --source mccall_acs_figshare_biochar_stability
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCES_PATH = PROJECT_DIR / "sources.json"
RAW_DIR = PROJECT_DIR / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "_download_manifest.json"

FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"
ZENODO_API = "https://zenodo.org/api/records/{record_id}"
MENDELEY_PUBLIC_API = "https://data.mendeley.com/public-api/datasets/{dataset_id}"
DRYAD_DATASET_API = "https://datadryad.org/api/v2/datasets/doi%3A{doi}"
DRYAD_FILE_DOWNLOAD = "https://datadryad.org/downloads/file_stream/{file_id}"
DATAONE_META = "https://knb.ecoinformatics.org/knb/d1/mn/v2/meta/{object_id}"
DATAONE_OBJECT = "https://knb.ecoinformatics.org/knb/d1/mn/v2/object/{object_id}"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "aethmodular-charcoal-ftir-downloader/0.1"
)
CHUNK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 120


class DownloadError(RuntimeError):
    """Raised when a file cannot be downloaded or validated."""


def request_json(url: str, *, referer: str | None = None) -> dict[str, Any]:
    request = Request(url, headers=build_headers(referer=referer))
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(url: str, *, referer: str | None = None) -> str:
    request = Request(url, headers=build_headers(referer=referer))
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8")


def build_headers(*, referer: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/csv, application/octet-stream, */*",
        "User-Agent": USER_AGENT,
    }
    if referer:
        headers["Referer"] = referer
    return headers


def load_sources() -> list[dict[str, Any]]:
    with SOURCES_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["sources"]


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"downloads": []}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at_unix"] = int(time.time())
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def upsert_manifest_record(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    downloads = [
        item for item in manifest.get("downloads", []) if item.get("path") != record.get("path")
    ]
    downloads.append(record)
    manifest["downloads"] = downloads


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(normalize_hash_name(algorithm))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_hash_name(name: str) -> str:
    return name.lower().replace("-", "")


def file_is_valid(path: Path, expected_hash: str | None, hash_type: str | None) -> bool:
    if not path.exists():
        return False
    if not expected_hash or not hash_type:
        return True
    return hash_file(path, hash_type) == expected_hash.lower()


def download_file(
    *,
    url: str,
    destination: Path,
    expected_size: int | None,
    expected_hash: str | None,
    hash_type: str | None,
    force: bool,
    referer: str | None = None,
) -> dict[str, Any]:
    if destination.exists() and not force:
        if file_is_valid(destination, expected_hash, hash_type):
            return {
                "path": str(destination.relative_to(PROJECT_DIR)),
                "status": "exists",
                "bytes": destination.stat().st_size,
                "hash_type": hash_type,
                "hash": expected_hash,
            }
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    if temporary_path.exists():
        temporary_path.unlink()

    request = Request(url, headers=build_headers(referer=referer))
    digest = hashlib.new(normalize_hash_name(hash_type)) if hash_type else None
    bytes_written = 0

    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            with temporary_path.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_written += len(chunk)
                    if digest:
                        digest.update(chunk)
    except HTTPError as error:
        cleanup_partial(temporary_path)
        body = compact_error_body(error.read(400).decode("utf-8", errors="replace"))
        message = f"HTTP {error.code} for {url}"
        if body:
            message = f"{message}: {body}"
        raise DownloadError(message) from error
    except URLError as error:
        cleanup_partial(temporary_path)
        raise DownloadError(f"Network error for {url}: {error.reason}") from error

    if expected_size is not None and bytes_written != expected_size:
        cleanup_partial(temporary_path)
        raise DownloadError(
            f"Size mismatch for {destination.name}: got {bytes_written}, expected {expected_size}"
        )

    actual_hash = digest.hexdigest() if digest else None
    if expected_hash and actual_hash and actual_hash != expected_hash.lower():
        cleanup_partial(temporary_path)
        raise DownloadError(
            f"{hash_type} mismatch for {destination.name}: got {actual_hash}, expected {expected_hash}"
        )

    temporary_path.replace(destination)
    return {
        "path": str(destination.relative_to(PROJECT_DIR)),
        "status": "downloaded",
        "bytes": bytes_written,
        "hash_type": hash_type,
        "hash": actual_hash or expected_hash,
    }


def cleanup_partial(path: Path) -> None:
    if path.exists():
        path.unlink()


def compact_error_body(body: str) -> str:
    text = " ".join(body.replace("\r", " ").replace("\n", " ").split())
    if text.startswith("<html>") or "<title>403 Forbidden</title>" in text:
        return "repository blocked the command-line file request"
    return text[:240]


def collect_figshare_files(source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    article = request_json(FIGSHARE_API.format(article_id=source["article_id"]))
    files = []
    for item in article.get("files", []):
        files.append(
            {
                "name": item["name"],
                "url": item["download_url"],
                "size": item.get("size"),
                "hash": item.get("computed_md5") or item.get("supplied_md5"),
                "hash_type": "md5" if (item.get("computed_md5") or item.get("supplied_md5")) else None,
            }
        )
    return article, filter_files(source, files)


def collect_zenodo_files(source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = request_json(ZENODO_API.format(record_id=source["record_id"]))
    files = []
    for item in record.get("files", []):
        hash_type, expected_hash = split_checksum(item.get("checksum"))
        files.append(
            {
                "name": item["key"],
                "url": item["links"]["self"],
                "size": item.get("size"),
                "hash": expected_hash,
                "hash_type": hash_type,
            }
        )
    return record, filter_files(source, files)


def collect_mendeley_files(source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = request_json(MENDELEY_PUBLIC_API.format(dataset_id=source["dataset_id"]))
    files = []
    for item in dataset.get("files", []):
        details = item.get("content_details", {})
        files.append(
            {
                "name": item["filename"],
                "url": details.get("download_url"),
                "size": item.get("size") or details.get("size"),
                "hash": details.get("sha256_hash"),
                "hash_type": "sha256" if details.get("sha256_hash") else None,
            }
        )
    return dataset, filter_files(source, [file for file in files if file.get("url")])


def collect_direct_files(source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    files = []
    for item in source["files"]:
        files.append(
            {
                "name": item["name"],
                "url": item["url"],
                "size": item.get("size"),
                "hash": item.get("hash"),
                "hash_type": item.get("hash_type"),
            }
        )
    return source, filter_files(source, files)


def collect_dataone_files(source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {"package_id": source.get("package_id"), "files": []}
    files = []
    for item in source["files"]:
        file_metadata = parse_dataone_system_metadata(
            request_text(DATAONE_META.format(object_id=item["object_id"]))
        )
        metadata["files"].append(file_metadata)
        files.append(
            {
                "name": item.get("name") or file_metadata.get("file_name") or item["object_id"],
                "url": DATAONE_OBJECT.format(object_id=item["object_id"]),
                "size": file_metadata.get("size"),
                "hash": file_metadata.get("checksum"),
                "hash_type": file_metadata.get("checksum_algorithm"),
            }
        )
    return metadata, filter_files(source, files)


def collect_dryad_files(source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    encoded_doi = quote(source["doi"], safe="")
    dataset = request_json(DRYAD_DATASET_API.format(doi=encoded_doi))
    version_href = dataset["_links"]["stash:version"]["href"]
    version = request_json("https://datadryad.org" + version_href)
    files_href = version["_links"]["stash:files"]["href"]
    files_payload = request_json("https://datadryad.org" + files_href)

    files = []
    for item in files_payload["_embedded"]["stash:files"]:
        file_id = item["_links"]["self"]["href"].rstrip("/").split("/")[-1]
        files.append(
            {
                "name": item["path"],
                "url": DRYAD_FILE_DOWNLOAD.format(file_id=file_id),
                "size": item.get("size"),
                "hash": item.get("digest"),
                "hash_type": item.get("digestType"),
            }
        )

    metadata = {"dataset": dataset, "version": version, "files": files_payload}
    return metadata, filter_files(source, files)


def split_checksum(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    if ":" not in value:
        return None, value
    hash_type, expected_hash = value.split(":", 1)
    return hash_type, expected_hash


def parse_dataone_system_metadata(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    checksum = root.find("checksum")
    checksum_algorithm = checksum.attrib.get("algorithm") if checksum is not None else None
    size_text = root.findtext("size")
    return {
        "identifier": root.findtext("identifier"),
        "file_name": root.findtext("fileName"),
        "format_id": root.findtext("formatId"),
        "size": int(size_text) if size_text and size_text.isdigit() else None,
        "checksum": checksum.text if checksum is not None else None,
        "checksum_algorithm": checksum_algorithm,
    }


def filter_files(source: dict[str, Any], files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    include_names = set(source.get("include_names", []))
    include_globs = source.get("include_globs", [])
    if not include_names and not include_globs:
        return files
    filtered = []
    for file_info in files:
        name = file_info["name"]
        if name in include_names or any(fnmatch.fnmatch(name, pattern) for pattern in include_globs):
            filtered.append(file_info)
    return filtered


def collect_files(source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if source["kind"] == "figshare":
        return collect_figshare_files(source)
    if source["kind"] == "dryad":
        return collect_dryad_files(source)
    if source["kind"] == "zenodo":
        return collect_zenodo_files(source)
    if source["kind"] == "mendeley":
        return collect_mendeley_files(source)
    if source["kind"] == "direct":
        return collect_direct_files(source)
    if source["kind"] == "dataone":
        return collect_dataone_files(source)
    raise ValueError(f"Unsupported source kind: {source['kind']}")


def write_source_metadata(source: dict[str, Any], metadata: dict[str, Any]) -> None:
    target = RAW_DIR / source["id"] / "_source_metadata.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def format_size(size: int | None) -> str:
    if size is None:
        return "unknown size"
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def parse_args() -> argparse.Namespace:
    sources = load_sources()
    source_ids = [source["id"] for source in sources]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=source_ids,
        action="append",
        help="Download one source id. Repeat for multiple. Defaults to all sources.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading.")
    parser.add_argument("--force", action="store_true", help="Re-download even if a valid local file exists.")
    parser.add_argument("--list-sources", action="store_true", help="Print configured sources and exit.")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write repository metadata JSON files into data/raw.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = load_sources()

    if args.list_sources:
        for source in sources:
            print(f"{source['id']} ({source['kind']}): {source['title']}")
        return 0

    selected_ids = set(args.source or [source["id"] for source in sources])
    selected_sources = [source for source in sources if source["id"] in selected_ids]

    manifest = load_manifest()
    failures: list[str] = []

    for source in selected_sources:
        print(f"\n== {source['id']} ==")
        try:
            metadata, files = collect_files(source)
        except Exception as error:  # noqa: BLE001 - CLI should report all source failures.
            failures.append(f"{source['id']}: metadata failed: {error}")
            print(f"metadata failed: {error}", file=sys.stderr)
            continue

        if not args.no_metadata and not args.dry_run:
            write_source_metadata(source, metadata)

        if not files:
            print("No files found.")
            continue

        for file_info in files:
            destination = RAW_DIR / source["id"] / file_info["name"]
            print(f"- {file_info['name']} ({format_size(file_info.get('size'))})")
            if args.dry_run:
                print(f"  {file_info['url']}")
                continue

            try:
                record = download_file(
                    url=file_info["url"],
                    destination=destination,
                    expected_size=file_info.get("size"),
                    expected_hash=file_info.get("hash"),
                    hash_type=file_info.get("hash_type"),
                    force=args.force,
                    referer=source.get("repository_url"),
                )
                record.update(
                    {
                        "source_id": source["id"],
                        "source_title": source["title"],
                        "repository_url": source.get("repository_url"),
                        "download_url": file_info["url"],
                    }
                )
                upsert_manifest_record(manifest, record)
                print(f"  {record['status']}: {record['path']}")
            except DownloadError as error:
                failures.append(f"{source['id']} / {file_info['name']}: {error}")
                print(f"  failed: {error}", file=sys.stderr)

        if source["kind"] == "dryad" and any(
            failure.startswith(source["id"]) for failure in failures
        ):
            print(
                "  Dryad metadata was readable, but file downloads may need a browser session. "
                f"Repository page: {source.get('repository_url')}",
                file=sys.stderr,
            )

    if not args.dry_run:
        save_manifest(manifest)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
