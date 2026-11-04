"""Response decomposition on native Addis spectra and current direct-HIPS pairs.

Reuses the native audit's loader and band definitions. Saved signed per-scan
AIRSpec corrections are averaged by physical filter; no baseline or EC refit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MAC_VALUE  # noqa: E402
from portable_cohort_screen import PUBLIC_CHEMISTRY_PATH, load_public_chemistry  # noqa: E402
from run_native_addis_baseline_audit import SOURCE_NAMES, features, load_native, safe_json  # noqa: E402
from spectral_response_screen import (  # noqa: E402
    CONTROLS, INDICATORS, TARGETS, average_physical_scans,
    build_response_ledger, response_associations,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True, type=Path)
    args = parser.parse_args()
    output = ROOT / "research/ftir_hips_chem/output/tables/spectral_response_20261102"
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {name: sha256(args.native_dir / name) for name in SOURCE_NAMES}
    input_hashes[PUBLIC_CHEMISTRY_PATH.as_posix()] = sha256(ROOT / PUBLIC_CHEMISTRY_PATH)
    scans, wn, raw, corrected, region = load_native(args.native_dir)
    physical, mean_raw, mean_corrected = average_physical_scans(scans, raw, corrected)
    feature_frame, _ = features(wn, mean_raw, mean_corrected, region)
    physical = pd.concat([physical, feature_frame], axis=1)
    scans["prior_documented_review_flag"] = np.select([
        scans.SampleAnalysisId.eq(406), scans.SampleAnalysisId.eq(1626),
        scans.base_filter_id.eq("ETAD-0158"),
    ], [
        "ETAD-0037 nearly flat acquisition; retained pending review",
        "ETAD-0139 scan 1626 nearly flat; retained in physical mean pending review",
        "ETAD-0158 documented filter hole; retained with existing canonical flags",
    ], default="")
    flags = scans.groupby("base_filter_id").prior_documented_review_flag.agg(
        lambda values: " | ".join(sorted({value for value in values if value}))
    )
    physical["prior_documented_review_flag"] = physical.base_filter_id.map(flags)
    current = load_public_chemistry(ROOT)
    ledger = build_response_ledger(physical, current)
    associations, membership = response_associations(ledger)
    scans.to_csv(output / "native_scan_provenance.csv", index=False)
    ledger.to_csv(output / "physical_filter_response_ledger.csv", index=False)
    associations.to_csv(output / "response_associations.csv", index=False)
    membership.to_csv(output / "indicator_cohort_membership.csv", index=False)
    display = associations.pivot(index="indicator", columns="target", values="partial_rank_r_squared")
    display.to_csv(output / "partial_rank_r_squared_comparison.csv")
    for name in SOURCE_NAMES:
        if sha256(args.native_dir / name) != input_hashes[name]:
            raise RuntimeError(f"native source changed during analysis: {name}")
    if sha256(ROOT / PUBLIC_CHEMISTRY_PATH) != input_hashes[PUBLIC_CHEMISTRY_PATH.as_posix()]:
        raise RuntimeError("current public chemistry changed during analysis")
    receipt = {
        "schema_version": 1, "scope": "exploratory native Addis spectral response decomposition",
        "input_sha256": input_hashes, "inputs_unchanged": True,
        "scans": len(scans), "physical_filters": len(physical),
        "physical_types": physical.ExternalFilterType.value_counts().to_dict(),
        "native_pm25": int(physical.ExternalFilterType.eq("PM2.5").sum()),
        "replicated_physical_filters": int(physical.n_scans.gt(1).sum()),
        "extra_scans": len(scans) - len(physical),
        "response_base_eligible": int(ledger.response_base_eligible.sum()),
        "base_eligibility_reasons": ledger.response_base_reason.value_counts().to_dict(),
        "all_target_comparisons_share_feature_specific_cohort": bool(
            associations.groupby("indicator").cohort_id_sha256.nunique().eq(1).all()
        ),
        "indicators": INDICATORS, "targets": TARGETS, "covariate_fields": CONTROLS,
        "control_design": "ranked PM2.5, ranked public volume, monthly sin/cos, ranked collection day, drop-first lot indicators",
        "no_optical_or_EC_response_used_as_covariate": True,
        "mac_m2_g": MAC_VALUE, "native_1617_channel_cm1": float(wn[np.argmin(abs(wn - 1617))]),
        "baseline": "saved signed AIRSpec per-scan correction; average after correction; no refit",
        "original_scan_selection": "all acquired scans retained; no outcome-based scan selection",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "research/ftir_hips_chem/scripts/spectral_response_screen.py",
            ROOT / "research/ftir_hips_chem/workflows/run_native_addis_baseline_audit.py",
            ROOT / "research/ftir_hips_chem/scripts/portable_cohort_screen.py",
            ROOT / "research/ftir_hips_chem/scripts/portable_baseline_screen.py",
            ROOT / "research/ftir_hips_chem/scripts/config.py",
        )},
        "limitations": [
            "Reported FTIR EC derives from spectra; association can recover estimator structure.",
            "HIPS absorption is an optical response, not an independent thermal EC reference.",
            "Partial rank association is not causal, predictive validation, or a baseline selection rule.",
            "The three rank-correlations are not an additive variance partition of the difference response.",
            "Missing lot is a category; acquisition date, background identities and calibration versions remain unresolved.",
            "Public sampled volume has mixed provenance as recorded by the source loader.",
            "Contrasts were specified after earlier investigations; no new hypothesis-confirmation claim or p-value search.",
        ],
    }
    (output / "receipt.json").write_text(json.dumps(safe_json(receipt), indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "scans", "physical_filters", "response_base_eligible", "base_eligibility_reasons",
        "all_target_comparisons_share_feature_specific_cohort", "inputs_unchanged",
    )}, indent=2))
    print(display.to_string())


if __name__ == "__main__":
    main()
