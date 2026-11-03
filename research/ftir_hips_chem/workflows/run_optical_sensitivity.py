"""Run direct-HIPS optical scenarios on the committed physical-filter ledger.

No downloads, fitting, calibration replacement, or original-data changes occur.
All generated artifacts go to the dedicated ignored output directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))

from optical_sensitivity import (  # noqa: E402
    CLAIM_LIMIT, DEFAULT_EC_THRESHOLDS, DEFAULT_MAC_SCENARIOS,
    add_mac_scenarios, direct_sensitivity_ledger, effective_ratio_summary,
    ethiopia_overlap_summary, scenario_summary,
)
from portable_cohort_screen import PUBLIC_CHEMISTRY_PATH, load_public_chemistry  # noqa: E402


def main():
    output = ROOT / "research/ftir_hips_chem/output/tables/optical_sensitivity_20261102"
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / PUBLIC_CHEMISTRY_PATH
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = load_public_chemistry(ROOT)
    ledger = direct_sensitivity_ledger(frame)
    scenarios = add_mac_scenarios(ledger)
    summary = scenario_summary(scenarios)
    ratios = effective_ratio_summary(ledger)
    overlap, overlap_info = ethiopia_overlap_summary(scenarios)
    ledger.to_csv(output / "physical_filter_ledger.csv", index=False)
    scenario_columns = [
        "id", "code", "date", "optical_source", "pair_eligible", "is_excluded",
        "exclusion_reason", "direct_scenario_eligible", "direct_scenario_reason",
        "ec_ftir_ugm3", "hips_fabs_Mm1", "scenario_mac_m2_g",
        "scenario_equivalent_bc_ugm3", "scenario_ec_minus_optical_ugm3", "scenario_label",
    ]
    scenarios[scenario_columns].to_csv(output / "scenario_filter_ledger.csv", index=False)
    summary.to_csv(output / "site_mac_scenarios.csv", index=False)
    ratios.to_csv(output / "effective_ratio_threshold_sensitivity.csv", index=False)
    overlap.to_csv(output / "ethiopia_shared_date_window.csv", index=False)
    if overlap_info["status"] == "available":
        dates = pd.to_datetime(ledger.date, errors="raise")
        common_ledger = ledger.loc[
            ledger.code.isin(("ETAD", "ETBI"))
            & dates.between(overlap_info["start"], overlap_info["end"])
        ]
        common_ledger.to_csv(output / "ethiopia_shared_date_filter_panel.csv", index=False)
        common_ratios = effective_ratio_summary(common_ledger)
        common_ratios["cohort"] = "shared_ethiopia_collection_date_window"
        common_ratios["window_start_inclusive"] = overlap_info["start"]
        common_ratios["window_end_inclusive"] = overlap_info["end"]
        common_ratios.to_csv(output / "ethiopia_shared_date_ratio_thresholds.csv", index=False)
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    if after != before:
        raise RuntimeError("source changed while optical scenarios were running")
    receipt = {
        "schema_version": 1, "source": PUBLIC_CHEMISTRY_PATH.as_posix(),
        "source_sha256": before, "source_unchanged": after == before,
        "source_generated": frame.attrs.get("generated"),
        "source_population": "committed PM2.5 physical-filter ledger; exported exclusion flags retained",
        "mac_scenarios_m2_g": DEFAULT_MAC_SCENARIOS, "wavelength_nm": 633,
        "mac_note": (
            "Illustrative scenarios at the HIPS wavelength, not validated site MAC estimates. "
            "The frequently quoted 7.5 m2/g literature value is at 550 nm and cannot be "
            "transferred to 633 nm without a wavelength-dependent absorption assumption."
        ),
        "ec_thresholds_ugm3_inclusive": DEFAULT_EC_THRESHOLDS,
        "ratio_note": "Fabs/FTIR EC is an effective ratio, not an independently measured MAC",
        "n_physical_filters": len(ledger), "n_sites": int(ledger.code.nunique()),
        "n_direct_pairs": int(ledger.direct_scenario_eligible.sum()),
        "n_public_extension_pairs_not_used": int((ledger.pair_eligible & ~ledger.direct_scenario_eligible).sum()),
        "ethiopia_overlap": overlap_info, "claim_limit": CLAIM_LIMIT,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "code_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__), ROOT / "research/ftir_hips_chem/scripts/optical_sensitivity.py",
                ROOT / "research/ftir_hips_chem/scripts/portable_cohort_screen.py",
            )
        },
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "n_physical_filters", "n_sites", "n_direct_pairs", "n_public_extension_pairs_not_used",
        "ethiopia_overlap", "source_unchanged",
    )}, indent=2))
    print(summary.loc[summary.code.isin(("ETAD", "ETBI")), [
        "code", "scenario_mac_m2_g", "n_direct_pairs", "n_ftir_below_optical",
        "median_ec_minus_optical_ugm3", "median_absolute_disagreement_ugm3",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
