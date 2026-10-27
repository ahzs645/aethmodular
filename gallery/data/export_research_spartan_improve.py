"""The SPARTAN vs IMPROVE shared-parameter tables, for the Research diagnostics tab.

Run with:  uv run python gallery/data/export_research_spartan_improve.py

Inputs (tracked in the repo, written by research/spartan/spartan_vs_improve_comparison.ipynb and
research/spartan/follow_up_mass_and_ratio_plots.ipynb):
  research/spartan/improve_comparison/tables/{shared_parameter_map,coverage_summary,distribution_stats,
      ratio_stats,relationship_fits}.csv
  research/spartan/follow_up_plots/tables/addis_analog_candidates.csv

The IMPROVE per-sample exports are not in the repo, so the gallery cannot redraw these panels from the
filters; the notebooks' own summary tables are what it can show. Re-run the notebooks where the IMPROVE
cache exists (research/ftir_hips_chem/scripts/improve_io.py) and then this script to refresh them.

Writes gallery/app/public/data/research/spartan_improve/tables.json (private: the research section's data).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/spartan"
OUT = ROOT / "gallery/app/public/data/research/spartan_improve"
TABLES = {
    "parameter_map": "improve_comparison/tables/shared_parameter_map.csv",
    "coverage": "improve_comparison/tables/coverage_summary.csv",
    "distribution": "improve_comparison/tables/distribution_stats.csv",
    "ratios": "improve_comparison/tables/ratio_stats.csv",
    "fits": "improve_comparison/tables/relationship_fits.csv",
    "analogs": "follow_up_plots/tables/addis_analog_candidates.csv",
}


def records(frame: pd.DataFrame) -> list[dict]:
    """Rows as plain dicts: floats to 4 significant decimals, missing values as null."""
    out = []
    for row in frame.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, float):
                clean[k] = None if not np.isfinite(v) else float(f"{v:.4g}") if abs(v) < 1e4 else round(v)
            elif v is None or (not isinstance(v, str) and pd.isna(v)):
                clean[k] = None
            else:
                clean[k] = v.item() if hasattr(v, "item") else v
        out.append(clean)
    return out


def main() -> None:
    tables = {}
    for key, rel in TABLES.items():
        path = SRC / rel
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: run the notebook that writes it first")
        tables[key] = records(pd.read_csv(path))
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {k: f"research/spartan/{v}" for k, v in TABLES.items()},
        "notebooks": ["research/spartan/spartan_vs_improve_comparison.ipynb", "research/spartan/follow_up_mass_and_ratio_plots.ipynb"],
        "tables": tables,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.json").write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {OUT / 'tables.json'}: " + ", ".join(f"{k} {len(v)}" for k, v in tables.items()))


if __name__ == "__main__":
    main()
