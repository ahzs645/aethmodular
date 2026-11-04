"""FTIR EC at Addis under each spectral baseline, for the gallery's "FTIR baseline" switch.

Run with:  uv run python gallery/data/export_baseline_ftir_ec.py

The crossplots can show FTIR EC with no baseline correction (the SPARTAN product,
already in filters.json as "EC (FTIR)"), the spline baseline (AIRSpec) or the VIBES
baseline. All three come from calibrations on the whole IMPROVE pool (both lots), so
they differ only in the baseline:

    none      the current SPARTAN calibration (raw spectra)
    spline    all-IMPROVE both lots, AIRSpec spline-corrected spectra
    vibes     all-IMPROVE both lots, completed rank-70 full-range VIBES spectra
    anchored  all-IMPROVE both lots, VIBES on PTFE-anchored spectra, k 18 (3 Nov 2026 Colab rerun)
    epo       the same spectra with the IMPROVE OC, other-PM, Fe and K signatures projected out, k 18

anchored and epo come from research/ftir_hips_chem/workflows/run_hips_response_20261103.py stage options
(network_candidates.csv); see the Continuum page in the Experimental tab for why they are here.

Sources, by filter:
  Addis test set (233)   the completed rank-70 meeting run, as the calibration analyses used it:
      research/ftir_hips_chem/output/tables/vibes_fullrange/gallery_rank70_20261002/meeting/addis_predictions.csv
  every other SPARTAN PM2.5 filter with a native spectrum and a sampled volume: the same two
      models applied to the network baselines (research/ftir_hips_chem/workflows/
      network_baselines_20261019.py, stage `ec` -> network_ec.csv). On the Addis test set those
      reproduce the frozen spline EC to 0.002 ug/m3 (median; max 0.023) and VIBES to 0.001
      (median; 5 filters differ by >0.1 where VIBES's optimiser found another solution).
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
NETWORK = ROOT / "research/ftir_hips_chem/output/tables/network_baselines_20261019/network_ec.csv"
CANDIDATES = ROOT / "research/ftir_hips_chem/output/tables/intercept_fixes_20261103/network_candidates.csv"
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
    addis = set(rows)
    added = 0
    if NETWORK.exists():
        net = pd.read_csv(NETWORK)
        for r in net.itertuples():
            if r.id in addis:
                continue
            vals = {m: round(float(v), 4) for m, v in (("spline", r.spline_ugm3), ("vibes", r.vibes_ugm3)) if pd.notna(v)}
            if vals:
                rows[r.id] = vals
                added += 1
    n_candidates = 0
    if CANDIDATES.exists():      # the two 3 Nov options, on every filter the IMPROVE models scored
        for r in pd.read_csv(CANDIDATES).itertuples():
            vals = {m: round(float(v), 4) for m, v in (("anchored", r.anchored), ("epo", r.epo)) if pd.notna(v) and abs(v) != float("inf")}
            if vals:
                rows.setdefault(r.filter_id, {}).update(vals)
                n_candidates += 1
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": [str(SRC.relative_to(ROOT)), str(NETWORK.relative_to(ROOT)), str(CANDIDATES.relative_to(ROOT))],
        "models": FITS,
        "note": ("None = the current SPARTAN FTIR EC; spline and VIBES use the same all-IMPROVE (both lots) calibration pool. "
                 f"Addis test set ({len(addis)}): the frozen calibration runs; {added} other SPARTAN PM2.5 filters: those models "
                 "applied to the network baselines (filters with a native spectrum and a sampled volume). "
                 f"Anchored and EPO: {n_candidates} SPARTAN filters, k 18, on the same IMPROVE pool."),
        "n_addis_frozen": len(addis),
        "n_network": added,
        "n_candidates": n_candidates,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))   # Infinity would make the whole file unparsable
    print(f"wrote {OUT.relative_to(ROOT)}: {len(addis)} Addis (frozen) + {added} network filters")


if __name__ == "__main__":
    main()
