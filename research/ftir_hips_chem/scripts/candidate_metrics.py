"""Physical-filter scores for a fixed, paired source-domain comparison.

These functions do not fit or select a model.  A caller must establish that its
predictions are out of fold before passing them here.  Missing predictions are
reported as coverage loss; they are never filled with a zero error.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


SCORE_PROTOCOL = "fixed_k_common_physical_filter_metrics_v1"
EPO_RESOLUTION_CONTROL_PROTOCOL = "matched_resolution_anchorhp_epo_v1"
EPO_RESOLUTION_CONTROL_FILES = ("epo_resolution_control.csv", "epo_resolution_control_filters.csv")
SOURCE_QC_PROTOCOL = "recorded_decision_bundle_raw_qc_v1"
METRIC_DEFINITIONS = {
    "improve_median_absolute_error": "median(abs(predicted - TOR)); micrograms per physical filter",
    "improve_mae": "mean(abs(predicted - TOR)); micrograms per physical filter",
    "improve_rmse": "sqrt(mean((predicted - TOR)^2)); micrograms per physical filter",
    "improve_mean_bias": "mean(predicted - TOR); micrograms per physical filter",
    "improve_median_bias": "median(predicted - TOR); micrograms per physical filter",
    "improve_p90_absolute_error": "90th percentile of abs(predicted - TOR); micrograms per physical filter",
    "improve_p95_absolute_error": "95th percentile of abs(predicted - TOR); micrograms per physical filter",
    "improve_error": "compatibility alias of improve_median_absolute_error, never mean absolute error",
    "coverage": "candidate finite predictions / physical filters eligible after the recorded raw-spectrum QC and TOR domain, before common-filter intersection",
    "common_coverage": "physical filters with finite predictions for every compared candidate / eligible physical filters",
}
ERROR_COLUMNS = tuple(key for key in METRIC_DEFINITIONS if key.startswith("improve_"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids) -> str:
    """Hash an ordered, explicit list (not ambiguous delimiter-joined IDs)."""
    return hashlib.sha256(json.dumps(list(map(str, ids)), separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_physical_ids(ids, *, label="source") -> np.ndarray:
    values = np.asarray(ids)
    if values.ndim != 1 or pd.isna(values).any():
        raise ValueError(f"{label}: physical-filter IDs must be a nonmissing vector")
    values = values.astype(str)
    if any(not identifier.strip() for identifier in values):
        raise ValueError(f"{label}: empty physical-filter ID")
    if len(set(values)) != len(values):
        raise ValueError(f"{label}: repeated physical-filter IDs; aggregate scans under a declared rule before scoring")
    return values


def recorded_source_qc(metadata: pd.DataFrame):
    """Apply the decision workbook's existing all-candidate raw-spectrum QC.

    Missing flags remain readable for legacy synthetic/artifact fixtures, and
    their absence is recorded. A present flag may contain only explicit Boolean
    values or canonical true/false strings; nulls and ambiguous values fail.
    This mask never selects folds and never suppresses saved predictions.
    """
    present = "raw_qc_ok" in metadata
    if not present:
        mask = np.ones(len(metadata), dtype=bool)
    else:
        flags = metadata["raw_qc_ok"]
        if flags.isna().any():
            raise ValueError("raw_qc_ok contains null values; recorded source QC must be explicit")
        values = []
        for value in flags.to_numpy():
            if isinstance(value, (bool, np.bool_)):
                values.append(bool(value))
            elif isinstance(value, str) and value.strip().lower() in ("true", "false"):
                values.append(value.strip().lower() == "true")
            else:
                raise ValueError("raw_qc_ok contains malformed values; use explicit true/false flags")
        mask = np.asarray(values, dtype=bool)
    policy = dict(
        protocol=SOURCE_QC_PROTOCOL, flag_column="raw_qc_ok", flag_present=present,
        rule="retain only raw_qc_ok=true for every candidate's source training and scored reference domain",
        scope="fixed recorded spectral QC; retain all physical-filter IDs, frozen site folds, and held-out predictions",
        source="build_decision_workbook_20261022.py: the corrupt raw filter leaves every comparison",
        missing_flag_policy="all true for legacy inputs lacking the column; absence is explicitly recorded",
        n_total=len(mask), n_pass=int(mask.sum()), n_excluded=int((~mask).sum()),
        mask_sha256=hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest(),
    )
    return mask, policy


def validate_archive_identity(archive, receipt, metadata, metadata_path, prediction_path, *, label="source"):
    """Reject legacy positional arrays and prove the current bundle's identity.

    Recomputing a site split from a new metadata file cannot establish which
    records produced an old positional NPZ.  Both the saved identities and the
    metadata/prediction byte hashes are required here.
    """
    expected_ids = validate_physical_ids(metadata.filter_id.to_numpy(), label=label)
    identity = receipt.get("source_identity", {})
    if identity.get("protocol") != "physical_filter_order_and_site_folds_v1":
        raise ValueError(f"{label}: legacy prediction archive has no verified physical-filter identity; rerun its fit stage")
    expected = {
        "improve_metadata_sha256": sha256_file(metadata_path),
        "predictions_sha256": sha256_file(prediction_path),
        "filter_ids_sha256": ids_sha256(expected_ids),
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{label}: prediction or metadata hash differs from its source identity receipt")
    required = ("improve_filter_ids", "improve_site_ids", "improve_fold_id")
    if any(key not in archive for key in required):
        raise ValueError(f"{label}: prediction archive lacks physical-filter IDs, sites, or folds")
    ids = validate_physical_ids(archive["improve_filter_ids"], label=label)
    sites = np.asarray(archive["improve_site_ids"]).astype(str)
    folds = np.asarray(archive["improve_fold_id"])
    if not np.array_equal(ids, expected_ids) or not np.array_equal(sites, metadata.site.astype(str).to_numpy()):
        raise ValueError(f"{label}: prediction IDs/sites differ from bundle order")
    qc_mask, qc_policy = recorded_source_qc(metadata)
    if qc_policy["flag_present"] or "source_qc_policy" in receipt:
        if receipt.get("source_qc_policy") != qc_policy:
            raise ValueError(f"{label}: fit receipt does not certify the recorded source QC policy; rerun its fit stage")
        if ("improve_raw_qc_ok" not in archive or np.asarray(archive["improve_raw_qc_ok"]).dtype != np.dtype(bool)
                or not np.array_equal(archive["improve_raw_qc_ok"], qc_mask)):
            raise ValueError(f"{label}: saved source QC mask differs from the bundle")
    if folds.shape != expected_ids.shape or not np.issubdtype(folds.dtype, np.integer) or np.any(folds < 0):
        raise ValueError(f"{label}: each physical filter must have one nonnegative held-out fold")
    if set(folds.tolist()) != set(range(5)):
        raise ValueError(f"{label}: the comparison requires its declared five outer folds")
    if pd.DataFrame({"site": sites, "fold": folds}).groupby("site").fold.nunique().max() != 1:
        raise ValueError(f"{label}: a source site crosses held-out folds")
    return folds


def paired_candidate_metrics(metadata: pd.DataFrame, predictions: Mapping[str, np.ndarray], *, k=18):
    """Return scoreboard, population receipt, and one-row-per-filter ledger.

    All error and composition-subset metrics use the same intersection.  The
    per-candidate availability columns retain the number lost at that step.
    The declared source domain is the recorded all-candidate raw-spectrum QC
    plus TOR <150 ug/filter, including finite zero/negative references for error
    scores. Ratios use positive TOR. QC-excluded rows remain in the ledger.
    """
    ids = validate_physical_ids(metadata["filter_id"].to_numpy())
    if not predictions:
        raise ValueError("at least one candidate is required")
    y = metadata["tor_ug"].to_numpy(float)
    ocec = metadata["ocec"].to_numpy(float)
    sites = metadata["site"].astype(str).to_numpy()
    qc_mask, qc_policy = recorded_source_qc(metadata)
    reference_domain = np.isfinite(y) & (y < 150)
    eligible = reference_domain & qc_mask
    arrays = {}
    for model, prediction in predictions.items():
        value = np.asarray(prediction, float)
        if value.shape != y.shape:
            raise ValueError(f"{model}: prediction shape {value.shape} does not match physical filters {y.shape}")
        arrays[model] = value
    common = eligible & np.logical_and.reduce([np.isfinite(value) for value in arrays.values()])
    heavy = common & (y >= 15) & (ocec < 2.5)
    light = common & (y > 3.5) & (y < 10) & (ocec > 4.5) & (ocec < 8)
    n_eligible, n_common = int(eligible.sum()), int(common.sum())
    rows = []
    for model, value in arrays.items():
        available = eligible & np.isfinite(value)
        residual = value[common] - y[common]
        absolute = np.abs(residual)
        medae = float(np.median(absolute)) if n_common else None
        row = dict(
            model=model, k=k, evaluation_status="source_site_cv", source_population="common_physical_filters",
            improve_median_absolute_error=medae,
            improve_mae=float(np.mean(absolute)) if n_common else None,
            improve_rmse=float(np.sqrt(np.mean(residual ** 2))) if n_common else None,
            improve_mean_bias=float(np.mean(residual)) if n_common else None,
            improve_median_bias=float(np.median(residual)) if n_common else None,
            improve_p90_absolute_error=float(np.quantile(absolute, .9)) if n_common else None,
            improve_p95_absolute_error=float(np.quantile(absolute, .95)) if n_common else None,
            improve_error=medae,
            improve_heavy_ratio=float(np.median(value[heavy] / y[heavy])) if heavy.any() else None,
            improve_light_ratio=float(np.median(value[light] / y[light])) if light.any() else None,
            n_eligible=n_eligible, n_available=int(available.sum()), n_failed=int((eligible & ~available).sum()),
            n_reference_domain=int(reference_domain.sum()), n_excluded_raw_qc=int((reference_domain & ~qc_mask).sum()),
            n_common=n_common, n_heavy=int(heavy.sum()), n_light=int(light.sum()),
            coverage=float(available.sum() / n_eligible) if n_eligible else None,
            common_coverage=float(n_common / n_eligible) if n_eligible else None,
        )
        rows.append(row)
    ledger = metadata[["filter_id", "site", "tor_ug", "ocec"]].copy()
    ledger["raw_qc_ok"] = qc_mask
    ledger["raw_qc_flag_present"] = qc_policy["flag_present"]
    ledger["reference_domain"] = reference_domain
    ledger["eligible"] = eligible
    ledger["eligibility_reason"] = np.select(
        [~qc_mask, ~np.isfinite(y), y >= 150],
        ["recorded_raw_spectrum_QC", "missing_or_nonfinite_TOR", "TOR_at_or_above_150_ug"], default="eligible")
    ledger["common"] = common
    for model, value in arrays.items():
        ledger[f"{model}_predicted_ug"] = value
        ledger[f"{model}_available"] = eligible & np.isfinite(value)
    population = dict(
        unit="physical_filter", reference="IMPROVE TOR EC", units="micrograms per filter", k=k,
        domain=("recorded raw-spectrum QC retained; " if qc_policy["flag_present"]
                else "no recorded raw-spectrum QC flag supplied; ") + "finite TOR EC <150 micrograms per filter",
        source_qc_policy=qc_policy, aggregation="pooled physical-filter out-of-fold predictions",
        filter_ids_sha256=ids_sha256(ids), n_total=len(ids), n_eligible=n_eligible,
        n_reference_domain=int(reference_domain.sum()), n_excluded_raw_qc=int((reference_domain & ~qc_mask).sum()),
        n_missing_reference=int((~np.isfinite(y)).sum()), n_outside_domain=int((np.isfinite(y) & (y >= 150)).sum()),
        n_common=n_common, n_sites=int(len(set(sites[common]))),
        common_coverage=float(n_common / n_eligible) if n_eligible else None,
        common_filter_ids=ids[common].tolist(), common_filter_ids_sha256=ids_sha256(ids[common]),
        compared_models=list(arrays), n_heavy=int(heavy.sum()), n_light=int(light.sum()),
        heavy_domain="TOR EC >=15 and <150 micrograms per filter; TOR OC/EC <2.5",
        light_domain="TOR EC >3.5 and <10 micrograms per filter; TOR OC/EC >4.5 and <8",
    )
    return pd.DataFrame(rows), population, ledger


def legacy_source_status(reason="No verified common-filter source-score receipt is available"):
    return dict(
        schema_version=1, status="legacy_unverified", protocol=SCORE_PROTOCOL, reason=reason,
        metrics=METRIC_DEFINITIONS, population=None, provenance={},
        limitations=[
            "Legacy EPO preprocessing used held-out source labels; EPO source scores are withheld.",
            "Full-source Addis/network optical comparisons are separate historical target diagnostics.",
        ],
    )


def verified_score_receipt(directory: Path) -> dict:
    """Recognize only a receipt bound to the scoreboard and filter ledger bytes.

    A new receipt adjacent to an old CSV is insufficient. Missing legacy receipts
    are expected and result in withheld EPO scores. A malformed or stale receipt
    is also withheld, with a reason that the UI can expose.
    """
    path = Path(directory) / "baseline_scoreboard_receipt.json"
    if not path.exists():
        return legacy_source_status()
    try:
        receipt = json.loads(path.read_text())
        if receipt.get("protocol") != SCORE_PROTOCOL or receipt.get("status") != "corrected":
            raise ValueError("unrecognized source-score protocol or status")
        outputs = receipt["outputs"]
        if not isinstance(outputs, dict):
            raise ValueError("source-score outputs must be a filename-to-hash mapping")
        required = {"baseline_scoreboard.csv", "baseline_scoreboard_filters.csv"}
        control = receipt.get("matched_resolution_control")
        if control is not None:
            if (not isinstance(control, dict) or control.get("protocol") != EPO_RESOLUTION_CONTROL_PROTOCOL
                    or control.get("status") != "corrected"):
                raise ValueError("unrecognized matched-resolution control protocol or status")
            required.update(EPO_RESOLUTION_CONTROL_FILES)
            control_outputs = control.get("outputs")
            if not isinstance(control_outputs, dict) or any(control_outputs.get(name) != outputs.get(name) or name not in outputs
                   for name in EPO_RESOLUTION_CONTROL_FILES):
                raise ValueError("matched-resolution control output hashes are missing or inconsistent")
        if required - set(outputs):
            raise ValueError("source-score receipt lacks required output files")
        for name, entry in outputs.items():
            if (not isinstance(name, str) or Path(name).name != name or name in ("", ".", "..")
                    or "/" in name or "\\" in name):
                raise ValueError("source-score outputs must use safe local filenames")
            expected = entry["sha256"]
            if not expected or sha256_file(Path(directory) / name) != expected:
                raise ValueError(f"{name} does not match its source-score receipt")
        if not receipt.get("provenance", {}).get("epo_protocol") or not receipt.get("population", {}).get("common_filter_ids_sha256"):
            raise ValueError("source-score receipt lacks EPO or physical-filter provenance")
        receipt["metrics"] = METRIC_DEFINITIONS
        return receipt
    except (ValueError, KeyError, OSError, TypeError) as error:
        return legacy_source_status(f"Source scores withheld: {error}")


def export_scoreboard(frame: pd.DataFrame, status: Mapping) -> pd.DataFrame:
    """Migrate historical metric naming without promoting historical EPO scores."""
    result = frame.copy()
    if status.get("status") == "corrected":
        required = {"model", "evaluation_status", *ERROR_COLUMNS, "n_eligible", "n_available", "n_common", "coverage"}
        missing = required - set(result.columns)
        if missing:
            raise ValueError(f"corrected scoreboard missing columns: {sorted(missing)}")
        return result
    # Older fixed-k tables called MedAE improve_error.  Their MAE, RMSE, and
    # coverage cannot be reconstructed from that scalar.
    if "improve_median_absolute_error" not in result:
        result["improve_median_absolute_error"] = result.get("improve_error", np.nan)
    for column in (*ERROR_COLUMNS, "n_eligible", "n_available", "n_failed", "n_common", "coverage", "common_coverage"):
        if column not in result:
            result[column] = np.nan
    result["evaluation_status"] = "historical_source_cv"
    epo = result.model.eq("epo")
    withheld = [c for c in result if c.startswith("improve_") or c.startswith("n_") or c in ("coverage", "common_coverage")]
    result.loc[epo, withheld] = np.nan
    result.loc[epo, "evaluation_status"] = "legacy_withheld"
    return result
