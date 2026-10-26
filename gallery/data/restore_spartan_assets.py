"""Restore a published SPARTAN export, verifying each content-addressed asset.

This copies an existing gallery export; it does not refit spectra or change
measurements. Run before packaging a preview that lacks spartan_network/:
  uv run python gallery/data/restore_spartan_assets.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from site_labels import display_site_name

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = "https://projects.ahmadjalil.com/aethmodular-gallery/data/"


def restore(source: str, output: Path, workers: int):
    source = source.rstrip("/") + "/"
    receipts = {}

    def fetch(path: str):
        if not re.fullmatch(r"spartan_network/[A-Za-z0-9_./-]+", path) or ".." in path:
            raise ValueError(f"Unsafe export path: {path}")
        target = output / path
        signature = re.search(r"-([0-9a-f]{16})\.[a-z]+\.gz$", path)
        expected = signature[1] if signature else None
        for attempt in range(3):
            try:
                body = target.read_bytes() if expected and target.exists() else b""
                if not body or hashlib.sha256(body).hexdigest()[:16] != expected:
                    request = urllib.request.Request(source + path, headers={"Accept-Encoding": "identity"})
                    with urllib.request.urlopen(request, timeout=40) as response:
                        body = response.read()
                digest = hashlib.sha256(body).hexdigest()
                if expected and digest[:16] != expected:
                    raise ValueError(f"Checksum mismatch: {path}")
                decoded = gzip.decompress(body) if body.startswith(b"\x1f\x8b") else body
                if ".json" in path:
                    json.loads(decoded)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(body)
                temporary.replace(target)
                return path, decoded, {"path": path, "bytes": len(body), "sha256": digest}
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    # Keep the entrypoint out of the app until all its dependencies are present.
    request = urllib.request.Request(source + "spartan_network/index.json", headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=40) as response:
        index_body = response.read()
    index = json.loads(index_body)
    if index.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported SPARTAN export schema")
    metadata_paths = {s[key] for s in index["sites"] for key in ("filters", "native")}
    spectra = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, path) for path in sorted(metadata_paths)]):
            path, decoded, receipt = future.result()
            receipts[path] = receipt
            if path.startswith("spartan_network/native/"):
                spectra.update(entry["file"] for entry in json.loads(decoded)["spectra"].values())
        print(f"Verified metadata for {len(index['sites'])} sites; restoring {len(spectra)} spectrum assets", flush=True)
        futures = [pool.submit(fetch, path) for path in sorted(spectra)]
        for i, future in enumerate(as_completed(futures), 1):
            path, _, receipt = future.result()
            receipts[path] = receipt
            if i % 250 == 0 or i == len(futures):
                print(f"Verified {i}/{len(futures)} spectrum assets", flush=True)
    folder = output / "spartan_network"
    folder.mkdir(parents=True, exist_ok=True)
    # Preserve the gallery's readable labels while retaining the upstream data
    # and its exact receipt. No counts, dates, traces or identifiers are changed.
    for site in index["sites"]:
        site["name"] = display_site_name(site["code"], site["name"])
        site["location"] = display_site_name(site["code"], site.get("location"), site["name"])
    published_index = json.dumps(index, ensure_ascii=False, separators=(",", ":")).encode()
    (folder / "index.json").write_bytes(published_index)
    receipt = {"source": source, "retrieved": datetime.now(timezone.utc).isoformat(),
               "source_index_sha256": hashlib.sha256(index_body).hexdigest(),
               "index_sha256": hashlib.sha256(published_index).hexdigest(),
               "display_labels": "Known compact place names expanded; measurement data unchanged",
               "sites": len(index["sites"]), "spectra_assets": len(spectra),
               "files": [receipts[key] for key in sorted(receipts)]}
    (folder / "restoration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Restored {len(receipts)} verified assets ({sum(r['bytes'] for r in receipts.values()):,} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "gallery/app/public/data")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    restore(args.source, args.output, max(1, min(16, args.workers)))
