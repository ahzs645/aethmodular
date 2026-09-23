"""Create a compact, path-sanitized IMPROVE table for hosted analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1]
MANIFEST = HERE / "improve_clean_parquet.manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_name(value: object) -> object:
    if not isinstance(value, str) or not value:
        return value
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if (manifest.get("schemaVersion"), manifest.get("id"), manifest.get("script")) != (
        1, "aeth.improve-clean-parquet", Path(__file__).name
    ):
        raise ValueError("Unexpected recipe manifest")
    source = DATA_ROOT / manifest["sourceFromDataRoot"]
    output = DATA_ROOT / manifest["outputFromDataRoot"]
    frame = pd.read_csv(source, low_memory=False)
    for column in manifest["pathColumns"]:
        if column not in frame.columns:
            raise ValueError(f"Missing provenance path column: {column}")
        frame[column] = frame[column].map(file_name)
        if frame[column].dropna().str.contains(r"[/\\]").any():
            raise ValueError(f"Local paths remain in {column}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".improve-clean-", suffix=".parquet", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        read_back = pd.read_parquet(temporary, engine="pyarrow")
        pd.testing.assert_frame_equal(frame, read_back, check_dtype=True, check_exact=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "recipeId": manifest["id"], "recipeVersion": manifest["version"],
        "manifestSha256": sha256(MANIFEST), "scriptSha256": sha256(Path(__file__)),
        "source": {"path": str(source.relative_to(DATA_ROOT)), "sha256": sha256(source), "bytes": source.stat().st_size},
        "output": {"path": str(output.relative_to(DATA_ROOT)), "sha256": sha256(output), "bytes": output.stat().st_size},
        "rows": len(frame), "columns": list(frame.columns),
        "pathColumns": manifest["pathColumns"], "validation": manifest["validation"],
    }
    receipt_path = output.with_suffix(".provenance.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(frame), "bytes": output.stat().st_size, "provenance": str(receipt_path)}))


if __name__ == "__main__":
    main()
