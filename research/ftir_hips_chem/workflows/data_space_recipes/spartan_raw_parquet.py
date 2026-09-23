"""Create verified Parquet copies of the local public SPARTAN CSV snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MANIFEST = HERE / "spartan_raw_parquet.manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if (manifest.get("schemaVersion"), manifest.get("id"), manifest.get("script")) != (
        1, "aeth.spartan-raw-parquet", Path(__file__).name
    ):
        raise ValueError("Unexpected recipe manifest")
    source_root = REPO / manifest["sourceRootFromRepo"]
    output_root = REPO / manifest["outputDirFromRepo"]
    known_groups = set(manifest["groups"])
    actual_groups = {str(path.parent.relative_to(source_root)) for path in source_root.rglob("*.csv")}
    if actual_groups != known_groups:
        raise ValueError(f"Raw SPARTAN group set changed: {actual_groups ^ known_groups}")

    receipt = {
        "recipeId": manifest["id"], "recipeVersion": manifest["version"],
        "manifestSha256": sha256(MANIFEST), "scriptSha256": sha256(Path(__file__)),
        "validation": manifest["validation"], "groups": {},
    }
    for group, config in manifest["groups"].items():
        sources = sorted((source_root / group).glob("*.csv"))
        if len(sources) != config["expectedFiles"]:
            raise ValueError(f"File count changed in {group}: {len(sources)}")
        output_dir = output_root / group
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for source in sources:
            with source.open(encoding="utf-8-sig", errors="replace") as stream:
                comments = [stream.readline().rstrip("\r\n") for _ in range(config["skipRows"])]
            frame = pd.read_csv(source, skiprows=config["skipRows"], low_memory=False)
            output = output_dir / f"{source.stem}.parquet"
            with tempfile.NamedTemporaryFile(prefix=".spartan-", suffix=".parquet", dir=output_dir, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
                read_back = pd.read_parquet(temporary, engine="pyarrow")
                pd.testing.assert_frame_equal(frame, read_back, check_dtype=True, check_exact=True)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            records.append({
                "source": str(source.relative_to(REPO)), "sourceSha256": sha256(source),
                "sourceBytes": source.stat().st_size, "releaseComments": comments,
                "output": str(output.relative_to(REPO)), "outputSha256": sha256(output),
                "outputBytes": output.stat().st_size, "rows": len(frame), "columns": list(frame.columns),
            })
        receipt["groups"][group] = records
        print(json.dumps({"group": group, "files": len(records), "rows": sum(item["rows"] for item in records), "parquetBytes": sum(item["outputBytes"] for item in records)}), flush=True)
    receipt_path = output_root / "spartan_raw.provenance.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "files": sum(map(len, receipt["groups"].values())), "rows": sum(item["rows"] for records in receipt["groups"].values() for item in records)}))


if __name__ == "__main__":
    main()
