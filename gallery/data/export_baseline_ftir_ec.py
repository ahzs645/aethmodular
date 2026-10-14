"""FTIR EC at Addis under each spectral baseline, for the gallery's "FTIR baseline" switch.

Run with:  uv run python gallery/data/export_baseline_ftir_ec.py

The crossplots can show FTIR EC with no baseline correction (the SPARTAN product,
already in filters.json as "EC (FTIR)"), the spline baseline (AIRSpec) or the VIBES
baseline. All three come from calibrations on the whole IMPROVE pool (both lots), so
they differ only in the baseline:

    none    the current SPARTAN calibration (raw spectra)
    spline  all-IMPROVE both lots, AIRSpec spline-corrected spectra
    vibes   all-IMPROVE both lots, completed rank-70 full-range VIBES spectra

Source: the completed rank-70 meeting run (Addis test set, 233 filters),
research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/meeting/addis_predictions.csv.
Writes gallery/app/public/data/baseline_ftir_ec.json (gitignored with the other baseline_* exports).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/meeting/addis_predictions.csv"
OUT = ROOT / "gallery/app/public/data/baseline_ftir_ec.json"
FITS = {
    "spline": "allimprove|AIRSpec|all-IMPROVE both lots (frozen Colab)",
    "vibes": "allimprove|VIBES-full-r70|all-IMPROVE both lots (frozen Colab)",
}


def base_id(external: str) -> str:
    """ETAD-0084-1 -> ETAD-0084 (the gallery's per-filter id)."""
    return re.sub(r"-\d+$", "", external) if re.fullmatch(r"[A-Z]{4}-\d{4}-\d+", external) else external


def main():
    pred = pd.read_csv(SRC)
    rows: dict[str, dict[str, float]] = {}
    for method, fit_id in FITS.items():
        part = pred.loc[pred.fit_id.eq(fit_id)]
        if len(part) == 0:
            raise SystemExit(f"no predictions for {fit_id}")
        for r in part.itertuples():
            if pd.notna(r.ftir_ec_ugm3):
                rows.setdefault(base_id(str(r.ExternalFilterId)), {})[method] = round(float(r.ftir_ec_ugm3), 4)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": str(SRC.relative_to(ROOT)),
        "models": FITS,
        "note": "Addis test set only. None = the current SPARTAN FTIR EC; spline and VIBES use the same all-IMPROVE (both lots) calibration pool.",
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} Addis filters")


if __name__ == "__main__":
    main()
