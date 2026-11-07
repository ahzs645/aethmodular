"""Verify frozen spectral manifests and their physical-filter row ordering."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_metrics import sha256_file


def verify_manifest_files(directory, names):
    directory = Path(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    hashes = {}
    for name in names:
        entry = manifest.get("files", {}).get(name)
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        actual = sha256_file(directory / name)
        if not expected or expected != actual:
            raise ValueError(f"{name}: file differs from or is missing from its frozen manifest")
        hashes[name] = actual
    return hashes


def verify_corrected_spectra(directory, metadata, wavenumbers, variants):
    """Check the collector's saved row map before assigning metadata to arrays.

    The continuum collector slices each group from its manifest-bound rows.csv
    in that file's order.  Checking lengths alone cannot establish this join.
    Corrected output bytes are also hashed into the new fit's provenance.
    """
    directory = Path(directory)
    bundle = directory / "bundle"
    hashes = {f"bundle/{name}": value for name, value in verify_manifest_files(bundle, ("rows.csv", "wn.npy")).items()}
    rows = pd.read_csv(bundle / "rows.csv", dtype={"filter_id": str})
    if not {"group", "group_row", "filter_id", "row"}.issubset(rows.columns):
        raise ValueError("corrected-spectra row manifest lacks group/order/filter identity columns")
    if not np.array_equal(rows.row.to_numpy(), np.arange(len(rows))):
        raise ValueError("corrected-spectra global row order is incomplete or repeated")
    if set(rows.group) != set(metadata):
        raise ValueError("corrected-spectra groups differ from the current bundle metadata")
    grid = np.load(bundle / "wn.npy", allow_pickle=False)
    if not np.array_equal(grid, np.asarray(wavenumbers)):
        raise ValueError("corrected-spectra wavenumber grid differs from the current bundle")
    for group, frame in metadata.items():
        selected = rows[rows.group.eq(group)]
        if (not np.array_equal(selected.group_row.to_numpy(), np.arange(len(frame)))
                or not np.array_equal(selected.filter_id.to_numpy(), frame.filter_id.astype(str).to_numpy())):
            raise ValueError(f"{group}: corrected-spectra physical-filter row order differs from the current bundle")
        for variant in variants:
            filename = f"corrected_{variant}_{group}.npy"
            values = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
            if values.shape != (len(frame), len(grid)):
                raise ValueError(f"{filename}: corrected spectra shape differs from verified rows/grid")
            hashes[filename] = sha256_file(directory / filename)
    return hashes
