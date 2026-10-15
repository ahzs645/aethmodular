"""Write one meeting's data for the gallery's spec-driven meeting pages.

A meeting page in the gallery is a spec (gallery/app/src/meetings/<id>.ts):
sections of text plus figures, each figure naming a chart kind and the columns
of a table to draw. This module writes the tables and text that spec reads,
as one file: gallery/app/public/data/meetings/<id>.json

    {"schema_version": 1, "id": ..., "generated_from": ..., "colors": {...},
     "text": {key: markdown}, "tables": {name: {column: [values...]}}}

Tables are long ("tidy"): one row per point, one column per variable, so a
figure only has to name its x, y, colour and facet columns. Numbers are
rounded to `digits` significant decimals; NaN becomes null.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gallery/app/public/data/meetings"


def _num(v, digits):
    f = float(v)
    return round(f, digits) if np.isfinite(f) else None


def columns(frame: pd.DataFrame, digits: int = 4) -> dict[str, list]:
    out = {}
    for c in frame.columns:
        s = frame[c]
        if pd.api.types.is_bool_dtype(s):
            out[str(c)] = [bool(v) for v in s]
        elif pd.api.types.is_numeric_dtype(s):
            out[str(c)] = [_num(v, digits) for v in s.to_numpy()]
        else:
            out[str(c)] = [None if pd.isna(v) else str(v) for v in s]
    return out


def write_bundle(meeting_id: str, *, generated_from: str, tables: dict[str, pd.DataFrame],
                 text: dict[str, str] | None = None, colors: dict[str, str] | None = None,
                 digits: dict[str, int] | None = None) -> Path:
    digits = digits or {}
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": 1,
        "id": meeting_id,
        "generated_from": generated_from,
        "colors": colors or {},
        "text": text or {},
        "tables": {name: columns(df.reset_index(drop=True), digits.get(name, 4)) for name, df in tables.items()},
    }
    path = OUT / f"{meeting_id}.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(ROOT)} ({kb:.0f} KB): " + ", ".join(f"{k} {len(v)} rows" for k, v in tables.items()))
    return path
