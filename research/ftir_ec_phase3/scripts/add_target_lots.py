"""Backfill the per-filter LotId column into existing explorer target tables.

`build_spartan_target.py` now writes LotId itself, but the four site targets
already on disk were built before that. Rather than re-export every spectrum
from the AQRC database, join the lot straight from the SPARTAN HIPS primary
table on ExternalFilterId - exactly the join `app.py::_load_all` already does
for the built-in ETAD target.

    python research/ftir_ec_phase3/scripts/add_target_lots.py [name ...]

Idempotent: a target that already carries LotId is left untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_common import PATHS  # noqa: E402

TARGETS = Path(__file__).resolve().parents[3] / "calibration_explorer" / "targets"


def lot_lookup() -> dict[str, str]:
    hips = pd.read_csv(PATHS.spartan_hips_primary, encoding="cp1252",
                       usecols=["FilterId", "LotId"]).dropna(subset=["LotId"])
    hips = hips.drop_duplicates("FilterId")
    def label(v):
        # lot ids are mostly integers but not all of them ('241a' exists)
        try:
            return str(int(float(v)))
        except (TypeError, ValueError):
            return str(v).strip()
    return {str(f): label(v) for f, v in zip(hips["FilterId"], hips["LotId"])}


def main(names: list[str]) -> None:
    lots = lot_lookup()
    print(f"HIPS lot lookup: {len(lots)} filters")
    dirs = [TARGETS / n for n in names] if names else sorted(
        d for d in TARGETS.iterdir() if (d / "reference.csv").exists())
    for d in dirs:
        path = d / "reference.csv"
        if not path.exists():
            print(f"{d.name}: no reference.csv, skipped")
            continue
        ref = pd.read_csv(path)
        if "LotId" in ref.columns:
            print(f"{d.name}: already has LotId, unchanged")
            continue
        if "ExternalFilterId" not in ref.columns:
            print(f"{d.name}: no ExternalFilterId to join on, skipped")
            continue
        ref["LotId"] = [lots.get(str(f), "") for f in ref["ExternalFilterId"]]
        matched = int((ref["LotId"] != "").sum())
        if not matched:
            print(f"{d.name}: no filters matched the HIPS table, left unchanged")
            continue
        ref.to_csv(path, index=False)
        counts = ref.loc[ref["LotId"] != "", "LotId"].value_counts().to_dict()
        print(f"{d.name}: wrote LotId for {matched}/{len(ref)} filters -> {counts}")


if __name__ == "__main__":
    main(sys.argv[1:])
