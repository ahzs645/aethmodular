"""Publish the executed Adama/Bishoftu frozen pilot to the local gallery.

Run after the research workflow:
``uv run --locked --no-sync python gallery/data/export_baseline_external_pilot.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/ftir_hips_chem/output/tables/adama_bishoftu_frozen_pilot"
DEST = ROOT / "gallery/app/public/data"


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def main() -> None:
    adama = pd.read_csv(SOURCE / "adama_candidate_pairs.csv")
    bish = pd.read_csv(SOURCE / "bishoftu_frozen_predictions.csv")
    addis = pd.read_csv(SOURCE / "addis_frozen_predictions.csv")
    matched = pd.read_csv(SOURCE / "addis_bishoftu_loading_matched.csv")
    sensitivity = pd.read_csv(SOURCE / "adama_id_mapping_sensitivity.csv")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    if (len(adama), len(bish), len(addis), len(matched)) != (5, 26, 253, 5):
        raise ValueError("Frozen pilot cohort changed; review before export")
    if len(sensitivity) != 2 * 2 * 2 * 120:
        raise ValueError("Adama mapping sensitivity is incomplete")
    adama_cols = ["date", "FilterId_ptfe", "FilterId_quartz", "flag",
                  "SampleAnalysisId_provisional", "ECTR", "ECTT",
                  "AIRSpec_EC_ugm3", "VIBES_EC_ugm3",
                  "AIRSpec_over_ECTR", "VIBES_over_ECTR",
                  "AIRSpec_over_ECTT", "VIBES_over_ECTT"]
    site_metrics = ["AIRSpec_EC_ugm3", "VIBES_EC_ugm3",
                    "AIRSpec_band1617", "AIRSpec_band2920",
                    "VIBES_band1617", "VIBES_band2920",
                    "HIPS_EC_equivalent_ugm3"]
    bish_cols = ["ExternalFilterId", "date", "LotId", "Fabs", *site_metrics]
    addis_cols = ["sample_id", "filter_id", "date", "lot", "Fabs", *site_metrics]
    matched_cols = ["bishoftu_filter_id", "addis_filter_id", "bishoftu_date",
                    "addis_date", "Fabs_ratio",
                    "bishoftu_AIRSpec_EC_ugm3", "addis_AIRSpec_EC_ugm3",
                    "bishoftu_VIBES_EC_ugm3", "addis_VIBES_EC_ugm3"]
    summarized = (sensitivity.groupby(["method", "reference", "scope"])
                  .median_prediction_over_reference.agg(["min", "max"]).reset_index())
    payload = {
        "schema_version": 1,
        "frozen_run_signature": manifest["frozen_run_signature"],
        "evidence_date": "2026-09-22",
        "adama": records(adama[adama_cols]),
        "bishoftu": records(bish[bish_cols]),
        "addis": records(addis[addis_cols]),
        "matched": records(matched[matched_cols]),
        "mapping_ranges": records(summarized),
        "limitations": {
            "adama": "Date-candidate quartz/PTFE pairs; spectral ID crosswalk and sampler equivalence unconfirmed.",
            "bishoftu": "HIPS/MAC-10 is optical EC-equivalent, not independent thermal EC.",
            "matching": "Lot 251, Oct-Dec, shared HIPS Fabs support, unique nearest pairs within 20% Fabs. Different years.",
        },
    }
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "baseline_external_pilot.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {path}: 5 Adama, 26 Bishoftu, 253 Addis, 5 matched pairs")


if __name__ == "__main__":
    main()
