"""Convert trusted local four-site ChemSpec exports to verified Parquet files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MANIFEST = HERE / "four_site_chemspec_parquet.manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if (manifest.get("schemaVersion"), manifest.get("id"), manifest.get("script")) != (
        1, "aeth.four-site-chemspec-parquet", Path(__file__).name
    ):
        raise ValueError("Unexpected recipe manifest")
    sources = manifest["sources"]
    if len(sources) != 4 or len(set(sources)) != 4 or any("/" in name or "\\" in name for name in sources):
        raise ValueError("Expected four distinct ChemSpec CSV file names")

    source_dir = REPO / manifest["sourceRootFromRepo"]
    output_dir = REPO / manifest["outputDirFromRepo"]
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name in sources:
        source = source_dir / name
        with source.open(encoding="utf-8-sig") as stream:
            comments = [stream.readline().rstrip("\r\n") for _ in range(3)]
        if not all(line.startswith("# ") for line in comments):
            raise ValueError(f"Unexpected ChemSpec header: {source}")
        frame = pd.read_csv(source, skiprows=3, low_memory=False)
        output = output_dir / f"{source.stem}.parquet"
        with tempfile.NamedTemporaryFile(prefix=".chemspec-", suffix=".parquet", dir=output_dir, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
            read_back = pd.read_parquet(temporary, engine="pyarrow")
            pd.testing.assert_frame_equal(frame, read_back, check_dtype=True, check_exact=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        provenance.append({
            "source": str(source.relative_to(REPO)), "sourceSha256": sha256(source),
            "sourceBytes": source.stat().st_size, "sourceComments": comments,
            "output": str(output.relative_to(REPO)), "outputSha256": sha256(output),
            "outputBytes": output.stat().st_size, "rows": len(frame),
            "columns": list(frame.columns),
        })
    receipt = {
        "recipeId": manifest["id"], "recipeVersion": manifest["version"],
        "manifestSha256": sha256(MANIFEST), "scriptSha256": sha256(Path(__file__)),
        "validation": manifest["validation"], "files": provenance,
    }
    receipt_path = output_dir / "four_site_chemspec.provenance.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "files": [{"output": item["output"], "rows": item["rows"], "bytes": item["outputBytes"]} for item in provenance]}))


if __name__ == "__main__":
    main()
