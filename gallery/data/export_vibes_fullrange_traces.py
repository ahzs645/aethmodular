"""Full-range VIBES display traces (4000-500 cm-1) for the gallery.

Same format as export_similarity.py's traces_{method}.bin (float32 scale per
filter, then int16 bins of 8 channels) and the same filter order as
similarity.json, so the browser's `Traces` decoder reads it unchanged. The bin
grid differs (the full range has 2722 channels), so it is written alongside:

    gallery/app/public/data/similarity/traces_VIBES-full.bin
    gallery/app/public/data/similarity/traces_VIBES-full.json   {bin_wn, n, source}

Run after workflows/assemble_vibes_fullrange.py:
    uv run --no-sync python gallery/data/export_vibes_fullrange_traces.py
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/ftir_hips_chem/output/tables/vibes_fullrange/rank30"
OUT = ROOT / "gallery/app/public/data/similarity"
BIN = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank', type=int, choices=[30, 70], default=30)
    rank = ap.parse_args().rank
    src = SRC if rank == 30 else ROOT / 'research/ftir_hips_chem/output/tables/vibes_fullrange/interpretation_20261002'
    if rank == 30:
        info = json.loads((src / "ASSEMBLY.json").read_text())
    else:
        summary = json.loads((src / 'summary.json').read_text())
        audit = json.loads((src.parent / 'failure_check_20261002/audit.json').read_text())
        info = dict(signature=audit['original_signature'], rank_cap=85, background_rank=70,
                    n_cases=summary['n_cases'], n_corrected=summary['n_cases'], n_failed=0,
                    n_recovered=len(summary['recovered_rows']), channels=2722,
                    wavenumber_range=[3998.42320552162, 500.124317346017])
    if info["n_corrected"] != info["n_cases"]:
        raise SystemExit("full-range run is not complete")
    X = np.load(src / ("corrected_VIBES_full.npy" if rank == 30 else 'corrected_VIBES_full_recovered.npy'))
    wn = np.load(SRC / "wn.npy")
    cases = pd.read_csv(src / "cases.csv")
    if X.shape != (info["n_cases"], info["channels"]) or not np.isfinite(X).all():
        raise ValueError("full-range corrected spectra are incomplete or invalid")
    if info["n_failed"] != 0 or len(cases) != info["n_cases"]:
        raise ValueError("full-range run has failed fits or inconsistent case counts")
    rows = np.flatnonzero(cases.kind.isin(["calibration", "target"]).to_numpy())
    ids = cases.sample_id.iloc[rows].tolist()
    meta = json.loads((OUT / "similarity.json").read_text())
    if ids != meta["filters"]["sample_id"]:
        raise ValueError("filter order differs from similarity.json")
    X = X[rows].astype(float)
    bins = [np.arange(i, min(i + BIN, len(wn))) for i in range(0, len(wn), BIN)]
    bin_wn = np.array([wn[b].mean() for b in bins])
    binned = np.stack([X[:, b].mean(axis=1) for b in bins], axis=1)
    scale = np.abs(binned).max(axis=1) / 32767
    scale[scale == 0] = 1
    q = np.round(binned / scale[:, None]).astype("<i2")
    stem = 'traces_VIBES-full' if rank == 30 else 'traces_VIBES-full-r70'
    with (OUT / f"{stem}.bin").open("wb") as out:
        out.write(scale.astype("<f4").tobytes())
        out.write(q.tobytes())
    (OUT / f"{stem}.json").write_text(json.dumps({
        "bin_wn": [round(float(w), 3) for w in bin_wn], "n": len(ids),
        "source": str(src.relative_to(ROOT)), "signature": info["signature"],
        "rank_cap": info["rank_cap"], "n_cases": info["n_cases"],
        "n_corrected": info["n_corrected"], "n_failed": info["n_failed"],
        "channels": info["channels"], "wavenumber_range": info["wavenumber_range"],
        "case_counts": {str(k): int(v) for k, v in cases.kind.value_counts().items()},
        "background_rank": info.get('background_rank', 30), "n_recovered": info.get('n_recovered', 0),
        "presentation_status": "preliminary_capped" if rank == 30 else "completed_uncapped",
    }))
    print(f"wrote {stem}.bin ({len(ids)} filters x {len(bins)} bins, {bin_wn.max():.0f}-{bin_wn.min():.0f} cm-1)")


if __name__ == "__main__":
    main()
