"""Build a lossless, query-friendly Parquet view of the four Kyan hourly CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MANIFEST = HERE / "kyan_hourly_parquet.manifest.json"
RENAME = {
    "Unnamed: 0": "source_row_index",
    "IR BCc": "ir_bcc",
    "UV BCc": "uv_bcc",
    "Red BCc": "red_bcc",
    "Green BCc": "green_bcc",
    "Blue BCc": "blue_bcc",
    "Fossil fuel BCc": "fossil_fuel_bcc",
    "Biomass BCc": "biomass_bcc",
    "AAE calculated": "aae_calculated",
    "Sample temp (C)": "sample_temp_c",
    "Sample RH (%)": "sample_rh_pct",
}
EXPECTED = {"datetime_local", "source", *RENAME}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Aethalometry Data/Kyan Data directory")
    parser.add_argument("--output", type=Path, help="Override the default ignored output Parquet")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if (manifest.get("schemaVersion"), manifest.get("id"), manifest.get("script")) != (
        1, "aeth.kyan-hourly-parquet", Path(__file__).name
    ):
        raise ValueError("Unexpected recipe manifest")
    output = args.output or REPO / manifest["outputFromRepo"]
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    sources = []
    for site, name in manifest["sources"].items():
        path = args.source_dir / manifest["sourceSubdirectory"] / name
        frame = pd.read_csv(path)
        if set(frame.columns) != EXPECTED or len(frame.columns) != len(EXPECTED):
            raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
        frame = frame.rename(columns=RENAME)
        frame.insert(0, "site", site)
        frame.insert(2, "datetime_utc", pd.to_datetime(frame["datetime_local"], utc=True, errors="raise"))
        if frame[["site", "datetime_local", "source"]].duplicated().any():
            raise ValueError(f"Duplicate site/local-hour/source key in {path.name}")
        sources.append({
            "site": site, "fileName": name, "sha256": sha256(path), "bytes": path.stat().st_size,
            "rows": len(frame), "repeatedLocalHourRows": int(frame["datetime_local"].duplicated().sum()),
        })
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    with tempfile.NamedTemporaryFile(prefix=".kyan-hourly-", suffix=".parquet", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        combined.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        read_back = pd.read_parquet(temporary, engine="pyarrow")
        pd.testing.assert_frame_equal(combined, read_back, check_dtype=True, check_exact=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    receipt = {
        "recipeId": manifest["id"], "recipeVersion": manifest["version"],
        "manifestSha256": sha256(MANIFEST), "scriptSha256": sha256(Path(__file__)),
        "sources": sources, "output": str(output), "outputSha256": sha256(output),
        "outputBytes": output.stat().st_size, "rows": len(combined),
        "columns": list(combined.columns), "status": manifest["status"],
        "interpretation": manifest["interpretation"],
    }
    receipt_path = output.with_suffix(".provenance.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "parquet": str(output), "provenance": str(receipt_path),
        "rows": len(combined), "bytes": output.stat().st_size,
        "repeatedCentralHours": next(s["repeatedLocalHourRows"] for s in sources if s["site"] == "Central"),
    }))


if __name__ == "__main__":
    main()
