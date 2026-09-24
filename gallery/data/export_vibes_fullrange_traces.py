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
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/ftir_hips_chem/output/tables/vibes_fullrange/rank30"
OUT = ROOT / "gallery/app/public/data/similarity"
BIN = 8


def main():
    info = json.loads((SRC / "ASSEMBLY.json").read_text())
    if info["n_corrected"] != info["n_cases"]:
        raise SystemExit("full-range run is not complete")
    X = np.load(SRC / "corrected_VIBES_full.npy")
    wn = np.load(SRC / "wn.npy")
    cases = pd.read_csv(SRC / "cases.csv")
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
    with (OUT / "traces_VIBES-full.bin").open("wb") as out:
        out.write(scale.astype("<f4").tobytes())
        out.write(q.tobytes())
    (OUT / "traces_VIBES-full.json").write_text(json.dumps({
        "bin_wn": [round(float(w), 3) for w in bin_wn], "n": len(ids),
        "source": str(SRC.relative_to(ROOT)), "signature": info["signature"]}))
    print(f"wrote traces_VIBES-full.bin ({len(ids)} filters x {len(bins)} bins, {bin_wn.max():.0f}-{bin_wn.min():.0f} cm-1)")


if __name__ == "__main__":
    main()
