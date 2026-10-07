"""Export site summaries from the canonical cleaned IMPROVE data for the bubble map.

Run from the repository root: uv run python gallery/data/export_improve_map.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from improve_io import improve_clean_path, load_improve_clean  # noqa: E402
from aethmodular_cli.env import display_path  # noqa: E402

FIELDS = {
    "EC (TOR)": "ECf_Val",
    "OC (TOR)": "OCf_Val",
    "Absorption": "fAbs_Val",
    "PM2.5 mass": "MF_Val",
    "Iron": "FEf_Val",
    "Soil": "SOILf_Val",
}


def main():
    columns = ["SiteCode", "SiteName", "State", "Latitude", "Longitude", "Date", *FIELDS.values()]
    data = load_improve_clean(usecols=columns)
    sites = []
    for code, rows in data.groupby("SiteCode", sort=True):
        latitudes, longitudes = rows.Latitude.dropna(), rows.Longitude.dropna()
        lat = latitudes.median() if len(latitudes) else None
        lon = longitudes.median() if len(longitudes) else None
        def text(column):
            values = rows[column].dropna()
            return str(values.iloc[0]) if len(values) else ""
        stats = {}
        for label, column in FIELDS.items():
            values = pd.to_numeric(rows[column], errors="coerce")
            values = values[np.isfinite(values)]
            stats[label] = {"n": len(values), "mean": float(values.mean()) if len(values) else None,
                            "median": float(values.median()) if len(values) else None}
        sites.append(dict(code=str(code), name=text("SiteName") or str(code),
                          location=text("State"), lat=float(lat) if pd.notna(lat) else None,
                          lon=float(lon) if pd.notna(lon) else None, n_filters=len(rows),
                          date_min=str(rows.Date.min())[:10], date_max=str(rows.Date.max())[:10], stats=stats))
    source = improve_clean_path()
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    out = ROOT / "gallery/app/public/data/improve_map.json"
    out.write_text(json.dumps(dict(
        generated=datetime.now(timezone.utc).isoformat(), source=display_path(source), source_sha256=digest,
        note="All sites in the locally available cleaned IMPROVE export. The canonical loader deduplicates FED sample keys and retains dated samples with positive EC and absorption. Counts describe this available population, not the full historical network or the FTIR calibration set. Study-site, season, date and exclusion controls do not apply to these summaries.",
        field_units={label: "Mm⁻¹" if label == "Absorption" else "µg/m³" for label in FIELDS},
        sites=sites,
    ), separators=(",", ":"), allow_nan=False))
    print(f"Wrote {out.relative_to(ROOT)}: {len(sites)} sites, {len(data):,} samples")


if __name__ == "__main__":
    main()
