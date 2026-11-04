"""Seasonal analog calibrations at every cohort size, for the meeting page's analog picker.

Reads research/ftir_hips_chem/output/tables/analog_ladder_20261103 (one npz per selection rule,
baseline and group, from workflows/run_analog_ladder_20261103.py) and writes
gallery/app/public/data/meeting-r70/analog_ladder/: manifest.json, and for each npz a header
(`<rule>.<baseline>.<group>.json`: the ranked analogs' references and each size's fit) and its
predictions (`.bin.gz`). lib/analogLadder.ts reads them. The layout of the binary, all
little-endian, per cohort size in order:

    float32  addis[n_sizes][n_addis]    FTIR EC at Addis, µg/m³ (NaN at a size that could not be fitted)
    uint8    held[n_sizes][mask_bytes]  bitset over rank (first bit, high end of the first byte, is rank 1)
    float32  ftir[sum(test_n)]          FTIR EC of the held-out analogs, rank order, µg/m³

Float32, not a scaled int16: a poor fit can predict tens of µg/m³ at Addis, and that is the answer to show.

Run: uv run --no-sync python gallery/data/export_analog_ladder.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "research/ftir_hips_chem"
SRC = AREA / "output/tables/analog_ladder_20261103"
REFERENCES = AREA / "output/tables/vibes_fullrange/gallery_rank70_20261002/meeting/improve_references.csv"
ADDIS = AREA / "output/tables/vibes_fullrange/gallery_rank70_20261002/meeting/addis_predictions.csv"
OUT = ROOT / "gallery/app/public/data/meeting-r70/analog_ladder"

METHODS = {"spline": "AIRSpec", "vibes-full": "VIBES-full-r70", "vibes-cut": "VIBES-full-cut-r70"}
GROUPS = {"all": "All Addis", "dry": "Dry (Oct-Feb)", "belg": "Belg (Mar-May)", "kiremt": "Kiremt (Jun-Sep)"}
RULES = [
    {"key": "similarity-sep17", "label": "Most similar, 17 Sep region", "per_group": True,
     "note": "Pearson r to the median Addis spectrum of the season, with CO₂ (1800–2500) and everything above 3600 cm⁻¹ left out: the meeting's selection"},
    {"key": "similarity-oct1", "label": "Most similar, 1 Oct region", "per_group": True,
     "note": "Pearson r to the median Addis spectrum of the season over 600–3600 cm⁻¹, with CO₂ and the 1050–1250 PTFE band left out. "
             "It differs from the 17 Sep region only on full-range VIBES: spline and cut VIBES spectra stop at 1425 cm⁻¹, so they pick the same analogs"},
    {"key": "lowest-ocec", "label": "Lowest TOR OC/EC", "per_group": False,
     "note": "The IMPROVE filters with the lowest TOR OC/EC. A property of the filter, so every season gets the same analogs"},
]


def clean(v, digits=None):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (float, np.floating)):
        return float(f"{v:.{digits}g}") if digits else float(v)
    return v.item() if isinstance(v, np.generic) else v


def col(s, digits=None):
    return [clean(v, digits) for v in s.tolist()]


def float32(values):
    return np.round(np.asarray(values, float), 4).astype("<f4")  # to the page's 4 dp: gzips far better


def main():
    if not (SRC / "rank70_inputs.json").exists():
        raise SystemExit(f"{SRC} is not a finished run (no rank70_inputs.json)")
    refs = pd.read_csv(REFERENCES).set_index("filter_id")
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = None
    written = 0
    for path in sorted(SRC.glob("*.npz")):
        rule, method, group = path.stem.split(".")
        z = np.load(path, allow_pickle=True)
        fits = pd.read_json(str(z["fits"]), orient="records")
        sizes = fits.n.to_numpy()
        ref = refs.loc[z["filter_id"]].reset_index()
        addis, mask, held = z["addis"], z["held_mask"], z["held"]
        if int(fits.test_n.sum()) != len(held):
            raise ValueError(f"{path.name}: held-out rows do not match test_n")
        head = {
            "rule": rule, "method": METHODS[method], "group": GROUPS[group],
            "n_min": int(sizes[0]), "n_max": int(sizes[-1]), "n_addis": addis.shape[1],
            "fits": {c: col(fits[c], 5 if c.startswith(("TOR_", "lambda")) else None)
                     for c in ["k", "train_n", "test_n", "test_sites", "TOR_test_R2", "TOR_test_RMSE_ug",
                               "lambda_tor_fabs", "lambda_ftir_tor", "lambda_ftir_fabs"]},
            "analogs": {
                "filter_id": col(ref.filter_id), "Site": col(ref.Site), "date": [str(d)[:10] for d in ref.date],
                "lot": col(ref.lot), "tor_ec_ugm3": col(ref.tor_ec_ugm3.round(4)),
                "fabs10": col((ref.fabs_Mm1 / 10).round(4)), "op_tor_frac": col(ref.op_tor_frac.round(4)),
                "tor_oc_ec": col(ref.tor_oc_ec.round(4)), "y": col(ref.y.round(4)), "vol": col(ref.tor_volume_m3.round(4)),
            },
        }
        stem = f"{rule}.{method}.{group}"
        (OUT / f"{stem}.json").write_text(json.dumps(head, allow_nan=False, separators=(",", ":")), encoding="utf8")
        body = float32(addis).tobytes() + np.packbits(mask, axis=1).tobytes() + float32(held).tobytes()
        (OUT / f"{stem}.bin.gz").write_bytes(gzip.compress(body, mtime=0))
        written += 1
    # grids that stop at 1425 cm⁻¹ pick the same analogs under both regions: the run fitted them once
    alias = {}
    for method in METHODS:
        for group in GROUPS:
            if not (SRC / f"similarity-oct1.{method}.{group}.npz").exists():
                alias[f"similarity-oct1.{method}.{group}"] = f"similarity-sep17.{method}.{group}"
    # the column order of `addis` above: the run's Addis table, as every saved fit lists it
    media = pd.read_csv(ADDIS, usecols=["fit_id", "MediaId"]).groupby("fit_id", sort=False).MediaId.apply(list).iloc[0]
    manifest = {
        "schema_version": 1, "n_min": int(sizes[0]), "n_max": int(sizes[-1]), "default_rule": RULES[0]["key"], "default_n": 500,
        "rules": RULES, "method_slug": {v: k for k, v in METHODS.items()}, "group_slug": {v: k for k, v in GROUPS.items()},
        "addis_media_id": media, "alias": alias,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    size = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"{written} ladders, {size / 1e6:.1f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
