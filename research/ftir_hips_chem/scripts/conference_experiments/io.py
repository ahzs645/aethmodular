"""Validated inputs and traceable, non-overwriting research outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
STATUSES = {"completed", "blocked", "insufficient_data", "failed", "planned"}


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError("Identifiers must contain letters, numbers, underscores or hyphens")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [json_safe(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def resolve_input(path: str, base: Path) -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError("Input paths must be non-empty strings")
    expanded = os.path.expandvars(path)
    if "$" in expanded:
        raise ValueError(f"An input environment variable is not configured: {path}")
    candidate = Path(expanded).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def file_receipt(path: Path, *, supplied_path: str) -> dict:
    # Preserve the portable manifest spelling (including environment variables),
    # rather than embedding this machine's resolved private directory in a report.
    return {"path": supplied_path, "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_table(path: Path, *, columns=None) -> pd.DataFrame:
    if columns is not None:
        if not isinstance(columns, list) or not columns or any(not isinstance(c, str) or not c for c in columns) or len(set(columns)) != len(columns):
            raise ValueError("Table columns must be a nonempty list of unique names")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False, usecols=columns)
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError("Table JSON must be an array of row objects")
        frame = pd.DataFrame(value)
        return frame if columns is None else frame.loc[:, columns]
    raise ValueError("Tables must be CSV, Parquet or JSON records; pickle is not accepted")


def load_dataset(spec: dict, base: Path):
    """Load tables or explicitly ID-aligned spectra; never infer row order."""
    kind = spec.get("kind", "table")
    receipts = []
    if kind == "table":
        path = resolve_input(spec["path"], base)
        table = read_table(path, columns=spec.get("columns"))
        receipts.append(file_receipt(path, supplied_path=spec["path"]))
        return table, receipts
    if kind != "spectra":
        raise ValueError(f"Unknown dataset kind: {kind}")
    path = resolve_input(spec["path"], base)
    meta_path = resolve_input(spec["metadata"], base)
    metadata = read_table(meta_path)
    row_col = spec.get("row_id_column", "analysis_id")
    if row_col not in metadata or metadata[row_col].isna().any():
        raise ValueError("Spectral metadata needs a nonmissing row identity column")
    row_ids = metadata[row_col].astype(str)
    if row_ids.duplicated().any():
        raise ValueError("Spectral row identities must be unique; physical-filter repeats use separate rows")
    with np.load(path, allow_pickle=False) as arrays:
        matrix = np.asarray(arrays[spec.get("matrix_key", "X")], dtype=float)
        ids = np.asarray(arrays[spec.get("ids_key", "analysis_ids")]).astype(str)
    if matrix.ndim != 2 or ids.ndim != 1 or len(ids) != len(matrix):
        raise ValueError("A 2-D spectral matrix and one identity per row are required")
    if pd.Index(ids).has_duplicates or any(not i.strip() or i == "nan" for i in ids):
        raise ValueError("Array row identities must be nonmissing and unique")
    if set(ids) != set(row_ids):
        raise ValueError("Array and metadata identities must match exactly")
    # The matrix's order is authoritative only after explicit identity alignment.
    metadata = metadata.assign(_row_identity=row_ids).set_index("_row_identity").loc[ids].reset_index(drop=True)
    receipts.extend([file_receipt(path, supplied_path=spec["path"]),
                     file_receipt(meta_path, supplied_path=spec["metadata"])])
    return (matrix, metadata), receipts


def runtime_receipt() -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = sorted(Path(__file__).parent.glob("*.py"))
    sources += [ROOT / "research/ftir_hips_chem/workflows/run_conference_experiments.py"]
    sources += [ROOT / "research/ftir_hips_chem/workflows/prepare_conference_inputs.py"]
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git("rev-parse", "HEAD"),
            "working_tree_changed": bool(git("status", "--porcelain")),
            "packages": packages,
            "implementation_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources if p.is_file()}}


def new_output_directory(path: Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new run directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_result(output: Path, identifier: str, kind: str, result: dict) -> dict:
    identifier = safe_name(identifier)
    summary = dict(result.get("summary", {}))
    status = summary.setdefault("status", "completed")
    if status not in STATUSES:
        raise ValueError(f"Unsupported experiment status: {status}")
    destination = output / identifier
    destination.mkdir()
    tables = []
    for name, frame in result.get("tables", {}).items():
        safe_name(name)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Result table {name} must be a DataFrame")
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=False)
        tables.append({"name": name, "path": str(path.relative_to(output)),
                       "rows": len(frame), "columns": list(frame.columns), "sha256": sha256(path)})
    write_json(destination / "summary.json", summary)
    return {"id": identifier, "kind": kind, "status": status, "summary": json_safe(summary),
            "tables": tables, "summary_path": f"{identifier}/summary.json"}


def write_report(output: Path, receipt: dict) -> None:
    scope = receipt["data_scope"]
    heading = "SYNTHETIC DEMONSTRATION" if scope == "synthetic" else "EXPLORATORY RESEARCH RUN"
    lines = [f"# Conference experiments — {heading}", "",
             "These diagnostics do not establish independent Addis elemental-carbon accuracy.", "",
             f"Input scope: **{scope}**. Implementation and input hashes are in `receipt.json`.", "",
             "| Experiment | Status | Output tables |", "|---|---|---|"]
    for item in receipt["experiments"]:
        links = ", ".join(f"[{t['name']}]({t['path']}) ({t['rows']} rows)" for t in item["tables"])
        lines.append(f"| {item['id']} | {item['status']} | {links or 'None'} |")
    for item in receipt["experiments"]:
        lines.extend(["", f"## {item['id']}", "", f"Kind: `{item['kind']}`. Status: **{item['status']}**.",
                      "", f"[Complete summary]({item['summary_path']})", "", "```json",
                      json.dumps(item["summary"], indent=2, allow_nan=False), "```"])
    lines.extend(["", "## Interpretation", "",
                  "Completed means the declared calculation ran on its eligible inputs. It does not mean a hypothesis was confirmed. "
                  "Blocked and insufficient-data results retain the missing-evidence or support reasons. "
                  "Synthetic results exercise the implementation and are not observations.", ""])
    (output / "report.md").write_text("\n".join(lines))
