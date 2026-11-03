"""Descriptive joins of existing PMF amplitudes and Addis spectral/optical data.

No factor model, calibration or forward-prediction model is fitted here. The
saved etad246_huber factors used the larger cohort and are not independent
chemical measurements or leakage-safe predictors for the seasonal ML screen.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "research/ftir_hips_chem/output/tables"
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from config import resolve_seasons, season_for_month  # noqa: E402

CASE = "etad246_huber"
CALENDAR = "belg_feb"
FACTOR_UNITS = "summed-channel spectral amplitude per m³; not chemical mass"
TARGETS = {
    "optical_disagreement_ugm3": ("FTIR EC minus HIPS BC", "µg/m³"),
    "ec_ftir_ugm3": ("FTIR EC", "µg/m³"),
    "optical_bc_ugm3": ("HIPS BC", "µg/m³"),
    "signed_spline_mean_1800_2250": ("Mean signed spline-corrected absorbance, 1800–2250 cm⁻¹", "absorbance"),
}


def fingerprint(paths):
    return {role: {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in paths.items()}


def base_id(identifier):
    match = re.fullmatch(r"(ETAD-\d{4})(?:-\d+)?", str(identifier))
    if match is None:
        raise ValueError(f"unexpected Addis physical filter identity: {identifier}")
    return match.group(1)


def spearman_rho(x, y):
    """Pearson correlation of average ranks; no significance test is calculated."""
    a, b = pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average")
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run(args):
    started_at = datetime.now(timezone.utc).isoformat()
    paths = {"workflow": Path(__file__).resolve(),
             "calendar_config": ROOT / "research/ftir_hips_chem/scripts/config.py",
             "ledger": args.ledger, "native_arrays": args.arrays, "saved_pmf": args.pmf}
    hashes_start = fingerprint(paths)
    ledger = pd.read_csv(args.ledger)
    if ledger.id.duplicated().any():
        raise ValueError("ledger must contain one row per physical filter")
    direct = ledger.loc[ledger.code.eq("ETAD") & ledger.spectral_pair_eligible.eq(True)
                        & ledger.optical_source.eq("direct_hips_fabs")].copy()
    if len(direct) != 232 or direct.id.nunique() != 232:
        raise ValueError("expected the existing 232 unique direct-HIPS Addis filters")
    direct["date"] = pd.to_datetime(direct.date, errors="raise").dt.date.astype(str)
    if direct.date.isna().any():
        raise ValueError("all current direct filters need an observed collection date")
    data = json.loads(args.pmf.read_text())
    case = data["contributions"][CASE]
    contribution_ids = [base_id(identifier) for identifier in case["id"]]
    g = np.asarray(case["g"], dtype=float)
    if len(contribution_ids) != 246 or len(set(contribution_ids)) != 246 or g.shape != (5, 246):
        raise ValueError("saved etad246_huber must contain 246 unique filters and five factors")
    if not np.isfinite(g).all():
        raise ValueError("saved factor amplitudes must be finite; no silent value replacement")
    factors = [f"factor_{i}" for i in range(1, 6)]
    pmf = pd.DataFrame(g.T, columns=factors)
    pmf["id"] = contribution_ids
    pmf["pmf_sample_id"] = case["id"]
    pmf["pmf_date"] = pd.to_datetime(case["date"], errors="raise").date.astype(str)
    missing = sorted(set(direct.id) - set(pmf.id))
    if missing != ["ETAD-0158"]:
        raise ValueError("expected only the previously documented damaged ETAD-0158 to lack this PMF case")
    joined = direct.merge(pmf, on="id", how="inner", validate="one_to_one")
    if len(joined) != 231 or not joined.date.eq(joined.pmf_date).all():
        raise ValueError("PMF/current-filter join must contain 231 matching physical IDs and collection dates")
    joined = joined.sort_values(["date", "id"]).reset_index(drop=True)
    with np.load(args.arrays, allow_pickle=False) as arrays:
        native_ids = arrays["physical_filter_ids"].astype(str)
        if pd.Index(native_ids).duplicated().any():
            raise ValueError("native arrays must contain one row per physical filter")
        order = pd.Index(native_ids).get_indexer(joined.id)
        if (order < 0).any():
            raise ValueError("joined PMF filters lack native spectra")
        wn = arrays["wavenumbers"]
        region = (wn >= 1800) & (wn <= 2250)
        if not region.any() or not arrays["shared_mask"][region].all():
            raise ValueError("the fixed signed spectral interval must be entirely supported")
        corrected = arrays["physical_corrected__spline_default"][order][:, region]
        if not np.isfinite(corrected).all():
            raise ValueError("the spectral mean needs finite signed values on every included channel")
        joined["signed_spline_mean_1800_2250"] = corrected.mean(axis=1)
        spectral_region = {"requested_lower_cm_1": 1800, "requested_upper_cm_1": 2250,
                           "observed_lower_cm_1": float(wn[region].min()),
                           "observed_upper_cm_1": float(wn[region].max()),
                           "n_native_channels": int(region.sum()),
                           "representation": "physical_corrected__spline_default",
                           "aggregation": "Arithmetic mean of signed absorbance; no clipping or area normalization"}
    target_columns = list(TARGETS)
    if not np.isfinite(joined[[*factors, *target_columns]]).all().all():
        raise ValueError("all joined factor amplitudes and target observations must be finite")
    if not np.allclose(joined.optical_disagreement_ugm3,
                       joined.ec_ftir_ugm3 - joined.optical_bc_ugm3, rtol=1e-7, atol=1e-8):
        raise ValueError("recorded disagreement must retain the FTIR-minus-HIPS sign")
    seasons = resolve_seasons(CALENDAR)
    joined["season"] = pd.to_datetime(joined.date).dt.month.map(lambda month: season_for_month(month, seasons))
    joined["calendar"] = CALENDAR
    joined["pmf_case"] = CASE
    joined["factor_units"] = FACTOR_UNITS
    columns = ["id", "pmf_sample_id", "date", "pmf_date", "calendar", "season", "pmf_case",
               "lot_group", "volume_m3", "factor_units", *factors, *target_columns]
    joined = joined[columns]
    groups = [("all", "All joined filters", joined)]
    groups += [("season", season, joined.loc[joined.season.eq(season)]) for season in seasons]
    correlations, medians = [], []
    for scope, season, group in groups:
        for factor in factors:
            values = group[factor].to_numpy(float)
            medians.append({"scope": scope, "season": season, "calendar": CALENDAR,
                            "factor": factor, "factor_label": factor.replace("_", " ").capitalize(),
                            "n": len(group), "median": float(np.median(values)),
                            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
                            "units": FACTOR_UNITS})
            for target, (label, units) in TARGETS.items():
                rho = spearman_rho(values, group[target].to_numpy(float))
                correlations.append({"scope": scope, "season": season, "calendar": CALENDAR,
                                     "factor": factor, "factor_label": factor.replace("_", " ").capitalize(),
                                     "target": target, "target_label": label, "rho": rho,
                                     "n": len(group), "factor_units": FACTOR_UNITS, "target_units": units,
                                     "status": "finite" if rho is not None else "constant_or_too_small"})
    hashes_end = fingerprint(paths)
    if hashes_start != hashes_end:
        raise RuntimeError("source or data changed while constructing the descriptive join")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / "joined_filters.csv", index=False)
    pd.DataFrame(correlations).to_csv(out / "correlations.csv", index=False)
    pd.DataFrame(medians).to_csv(out / "factor_season_medians.csv", index=False)
    summary = {
        "schema_version": 1, "analysis": "Descriptive existing PMF triangulation", "pmf_case": CASE,
        "calendar": CALENDAR, "counts": {"current_direct_filters": len(direct), "saved_pmf_filters": len(pmf),
                                          "joined_filters": len(joined), "missing_current_ids": missing,
                                          "season_counts": {name: int(joined.season.eq(name).sum()) for name in seasons},
                                          "date_mismatches": 0, "correlation_rows": len(correlations)},
        "factor_units": FACTOR_UNITS, "spectral_region": spectral_region,
        "correlations": correlations, "factor_medians": medians,
        "provenance": {"started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
                       "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "hashes_start": hashes_start, "hashes_end": hashes_end,
                       "source_and_inputs_unchanged": True,
                       "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                       "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in sorted(out.glob("*.csv"))}},
        "limitations": [
            "The saved PMF was fitted globally to the larger 246-filter cohort. These are descriptive associations, not forward-prediction validation or leakage-safe covariates.",
            "Factors 1–5 are spectral amplitudes, not identified chemical sources or calibrated chemical masses. Existing source labels are not transferred onto them.",
            "Both PMF factors and the signed spectral mean derive from FTIR spectra; they are not independent chemical measurements.",
            "FTIR-minus-HIPS disagreement contains a spectrum-derived EC component; separate EC and optical BC associations are retained to show this coupling.",
            "The saved nonnegative PMF factors cannot reproduce signed negative input cells; baseline and uncertainty limitations remain.",
            "Within-season correlations do not control loading, lot, volume, sampling year or other correlated features. Factor amplitudes are volume-normalized whereas the signed absorbance mean is a filter-level measurement.",
            "All available matched dates are used, including dates outside the ML evaluation folds. No p-values, source fractions, causal claims or new factor fit are produced.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps(summary["counts"], ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=TABLES / "baseline_review_20261102/filter_diagnostic_ledger.csv")
    parser.add_argument("--arrays", type=Path, default=TABLES / "baseline_existing_data_20261103/native_method_arrays.npz")
    parser.add_argument("--pmf", type=Path, default=ROOT / "gallery/app/public/data/research/python_pmf/python_pmf.json")
    parser.add_argument("--output", type=Path, default=TABLES / "seasonal_pmf_triangulation_20261103")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
