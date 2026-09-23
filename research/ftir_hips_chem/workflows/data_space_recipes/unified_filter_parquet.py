"""Materialize a trusted local AETH pickle as Parquet; emit separate provenance JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "unified_filter_parquet.manifest.json"
DEFAULT_DATA_ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("AETHMODULAR_DATA_ROOT", DEFAULT_DATA_ROOT)))
    parser.add_argument("--output", type=Path, help="Override the Parquet output path, useful for a temporary verification run")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("id") != "aeth.unified-filter-parquet" or manifest.get("schemaVersion") != 1:
        raise ValueError("Unexpected recipe manifest")
    if manifest.get("script") != Path(__file__).name or manifest.get("output", {}).get("format") != "parquet":
        raise ValueError("Recipe manifest does not describe this script's Parquet output")

    data_root = args.data_root.expanduser().resolve()
    source = data_root / manifest["source"]["pathFromDataRoot"]
    output = (args.output or data_root / manifest["output"]["pathFromDataRoot"]).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output:
        raise ValueError("Output must differ from source")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Pickle is only accepted from this trusted local research checkout. The raw file is retained.
    frame = pd.read_pickle(source)
    with tempfile.NamedTemporaryFile(prefix=".filter-parquet-", suffix=".parquet", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression=manifest["output"]["compression"], index=True)
        read_back = pd.read_parquet(temporary, engine="pyarrow")
        pd.testing.assert_frame_equal(frame, read_back, check_dtype=True, check_exact=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    provenance = {
        "recipeId": manifest["id"],
        "recipeVersion": manifest["version"],
        "manifestSha256": sha256(manifest_path),
        "scriptSha256": sha256(Path(__file__)),
        "source": {"path": str(source), "sha256": sha256(source), "bytes": source.stat().st_size},
        "output": {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size},
        "rows": len(frame),
        "columns": list(frame.columns),
        "validation": "read-back-frame-equality",
    }
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"parquet": str(output), "provenance": str(provenance_path), "rows": len(frame), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
