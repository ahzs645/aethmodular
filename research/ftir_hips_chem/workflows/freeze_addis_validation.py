"""Freeze Addis EC predictions and score only confirmed independent TOR pairs.

Run from the repository root with ``uv run --locked --no-sync python``.
The freeze is deliberately independent of any subsequently supplied reference EC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "research/ftir_hips_chem/scripts"
sys.path.insert(0, str(SCRIPTS))
from config import season_for_month, resolve_seasons  # noqa: E402
from data_matching import base_filter_id  # noqa: E402


RUN = ROOT / (
    "research/ftir_hips_chem/output/tables/vibes_colab_cloud/persistent_results/"
    "full-83dcf32e86bc0f09"
)
OUT = ROOT / "research/ftir_hips_chem/output/tables/addis_validation_freeze"
DELIVERABLE = ROOT / "deliverables/addis_validation_freeze_2026-09-22"
METHODS = ("AIRSpec", "VIBES")
EXPECTED_RUN_SIGNATURE = "83dcf32e86bc0f092f49f7f59671ac8a89f8efcc28d990287f8b1f895d2b2ae5"
SOURCE_FILES = (
    "RUN_MANIFEST.json",
    "cases.csv",
    "addis_predictions.csv",
    "corrected_AIRSpec.npy",
    "corrected_VIBES.npy",
    "pls_full_pool_AIRSpec.npz",
    "pls_full_pool_VIBES.npz",
)
PREDICTION_COLUMNS = (
    "sample_id", "media_id", "ptfe_filter_id", "base_filter_id", "sampling_date",
    "ethiopia_season_dry_feb", "original_ptfe_volume_m3", "volume_status",
    "AIRSpec_ec_ug_filter", "VIBES_ec_ug_filter",
    "AIRSpec_ec_ug_m3", "VIBES_ec_ug_m3",
)
PAIR_COLUMNS = (
    "sample_id", "ptfe_filter_id", "eligibility_status", "eligibility_reason",
    "quartz_filter_id", "identity_provenance",
    "sampling_equivalence_provenance", "identity_confirmed", "sampling_equivalent",
    "sampling_start", "sampling_end", "time_zone", "quartz_volume_m3",
    "ptfe_volume_m3", "volume_correction_reason",
    "tor_protocol", "tor_ec_ug_filter", "tor_uncertainty_ug_filter",
    "tor_mdl_ug_filter", "tor_qa_pass", "laboratory_source_file",
)
OUTCOME_COLUMNS = (
    "tor_ec_ug_filter", "tor_uncertainty_ug_filter", "tor_mdl_ug_filter",
    "tor_qa_pass", "laboratory_source_file",
)
LOCK_COLUMNS = tuple(name for name in PAIR_COLUMNS if name not in OUTCOME_COLUMNS)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frozen_predictions() -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((RUN / "RUN_MANIFEST.json").read_text())
    assert manifest["signature"] == EXPECTED_RUN_SIGNATURE
    cases = pd.read_csv(RUN / "cases.csv", low_memory=False)
    saved = pd.read_csv(RUN / "addis_predictions.csv", low_memory=False)
    assert len(cases) == manifest["n_cases"] == 12808
    assert cases.sample_id.is_unique
    target = cases.loc[cases.kind.eq("target")].copy()
    assert len(target) == 253 and target.sample_id.is_unique
    assert target.Site.eq("ETAD").all()
    assert set(saved.cohort) == {"full_pool", "locked800"}
    assert not target.sample_id.isna().any()
    assert all(target.filter_id.map(base_filter_id).notna())

    dates = pd.to_datetime(target.date, errors="coerce")
    seasons = resolve_seasons("dry_feb")
    frozen = pd.DataFrame({
        "sample_id": target.sample_id.to_numpy(),
        "media_id": target.sample_id.str.removeprefix("etad:").astype(int).to_numpy(),
        "ptfe_filter_id": target.filter_id.to_numpy(),
        "base_filter_id": target.filter_id.map(base_filter_id).to_numpy(),
        "sampling_date": target.date.to_numpy(),
        "ethiopia_season_dry_feb": [
            season_for_month(date.month, seasons) if pd.notna(date) else "unknown"
            for date in dates
        ],
        "original_ptfe_volume_m3": target.volume.to_numpy(float),
    })
    volume = frozen.original_ptfe_volume_m3.to_numpy(float)
    valid_volume = np.isfinite(volume) & (volume > 0)
    frozen["volume_status"] = np.where(valid_volume, "recorded_positive", "missing_or_invalid")

    for method in METHODS:
        matrix = np.load(RUN / f"corrected_{method}.npy", mmap_mode="r")
        fit = np.load(RUN / f"pls_full_pool_{method}.npz")
        assert matrix.shape == (len(cases), 2002)
        x = np.asarray(matrix[target.index.to_numpy()], dtype=np.float64)
        mass = ((x - fit["x_mean"]) @ fit["coefficient"].T + fit["y_mean"]).ravel()
        assert np.isfinite(mass).all()
        concentration = np.full(len(mass), np.nan)
        concentration[valid_volume] = mass[valid_volume] / volume[valid_volume]
        old = saved.loc[
            saved.cohort.eq("full_pool") & saved.method.eq(method),
            ["sample_id", "filter_id", "prediction_ugm3"],
        ]
        assert len(old) == 253 and old.sample_id.is_unique
        check = frozen[["sample_id", "ptfe_filter_id"]].merge(
            old, on="sample_id", validate="one_to_one", sort=False
        )
        assert len(check) == 253
        assert check.ptfe_filter_id.eq(check.filter_id).all()
        assert np.array_equal(check.prediction_ugm3.notna().to_numpy(), valid_volume)
        np.testing.assert_allclose(
            concentration[valid_volume], check.prediction_ugm3.to_numpy(float)[valid_volume],
            rtol=0, atol=5e-6,
        )
        frozen[f"{method}_ec_ug_filter"] = mass
        frozen[f"{method}_ec_ug_m3"] = concentration

    assert frozen.sample_id.is_unique and frozen.ptfe_filter_id.is_unique
    assert frozen.base_filter_id.is_unique
    frozen = frozen.loc[:, PREDICTION_COLUMNS].sort_values("media_id").reset_index(drop=True)
    provenance = {
        "source_run_signature": manifest["signature"],
        "source_sha256": {name: digest(RUN / name) for name in SOURCE_FILES},
        "n_filters": len(frozen),
        "n_with_recorded_positive_volume": int(valid_volume.sum()),
        "n_without_usable_volume": int((~valid_volume).sum()),
    }
    return frozen, provenance


def freeze(out: Path) -> None:
    predictions, provenance = frozen_predictions()
    out.mkdir(parents=True, exist_ok=True)
    prediction_path = out / "frozen_full_pool_predictions.csv"
    if (out / "freeze_manifest.json").exists():
        old = json.loads((out / "freeze_manifest.json").read_text())
        if (digest(prediction_path) != old["prediction_sha256"]
                or provenance != old["source"]):
            raise ValueError("Existing freeze has changed; refusing to overwrite")
    predictions.to_csv(prediction_path, index=False, float_format="%.15g")
    pairing = predictions[["sample_id", "ptfe_filter_id"]].copy()
    for name in PAIR_COLUMNS[2:]:
        pairing[name] = ""
    pairing.to_csv(out / "independent_tor_pairing_template.csv", index=False)
    record = {
        "schema_version": 1,
        "purpose": "Frozen exploratory Colab full-pool AIRSpec/VIBES outputs for future independent Addis TOR evaluation",
        "cohort": "full_pool",
        "reference_status": "unavailable; no Addis accuracy calculated",
        "prediction_units": {"mass": "ug/filter", "concentration": "ug/m3"},
        "primary_reference": "independent thermal-optical reflectance EC, ug/filter",
        "season_convention": "dry_feb",
        "prediction_file": prediction_path.name,
        "prediction_sha256": digest(prediction_path),
        "pairing_template_file": "independent_tor_pairing_template.csv",
        "source": provenance,
        "notes": [
            "Each sample_id is one physical PTFE filter; replicate scans were combined upstream.",
            "Mass predictions come from frozen full-pool coefficients and corrected arrays, including six filters whose saved concentration was blank because volume was unavailable.",
            "No thermal reference, HIPS equivalent, or ChemSpec EC is used to make or select predictions.",
            "A future lab-pairing table must pass authoritative identity and sampling checks before scoring.",
        ],
    }
    (out / "freeze_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"Frozen {len(predictions)} Addis physical filters; "
        f"{provenance['n_without_usable_volume']} lack usable volume. "
        f"Predictions SHA-256: {record['prediction_sha256']}"
    )


def package(out: Path, destination: Path) -> None:
    manifest = json.loads((out / "freeze_manifest.json").read_text())
    prediction = out / manifest["prediction_file"]
    if digest(prediction) != manifest["prediction_sha256"]:
        raise ValueError("Frozen predictions differ from their manifest")
    files = {
        "frozen_full_pool_predictions.csv": prediction,
        "independent_tor_pairing_template.csv": out / manifest["pairing_template_file"],
        "freeze_manifest.json": out / "freeze_manifest.json",
        "addis-validation-freeze.md": ROOT / "docs/addis-validation-freeze.md",
    }
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "addis_prediction_freeze.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
        for name, path in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            zipped.writestr(info, path.read_bytes())
    with TemporaryDirectory() as tmp:
        with ZipFile(archive) as zipped:
            zipped.extractall(tmp)
        for name, path in files.items():
            if digest(Path(tmp) / name) != digest(path):
                raise ValueError(f"Archive recovery check failed for {name}")
    receipt = {
        "archive": archive.name,
        "archive_sha256": digest(archive),
        "members_sha256": {name: digest(path) for name, path in sorted(files.items())},
        "recovery_check": "All four archive members extracted and matched the source hashes",
        "scope": "Frozen existing-ETAD predictions and protocol only; full Colab source arrays not bundled",
    }
    (destination / "release_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Packaged and recovery-checked {archive}; SHA-256 {receipt['archive_sha256']}")


def restore(out: Path, destination: Path) -> None:
    receipt = json.loads((destination / "release_receipt.json").read_text())
    archive = destination / receipt["archive"]
    if digest(archive) != receipt["archive_sha256"]:
        raise ValueError("Release archive differs from its receipt")
    expected = receipt["members_sha256"]
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    with ZipFile(archive) as zipped:
        if set(zipped.namelist()) != set(expected):
            raise ValueError("Unexpected archive members")
        for name, expected_hash in expected.items():
            payload = zipped.read(name)
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ValueError(f"Archive member hash mismatch: {name}")
            if name in {"freeze_manifest.json", "frozen_full_pool_predictions.csv",
                        "independent_tor_pairing_template.csv"}:
                path = out / name
                if path.exists() and digest(path) != expected_hash:
                    raise ValueError(f"Existing freeze differs from release: {path}")
                files[path] = payload
    for path, payload in files.items():
        path.write_bytes(payload)
    manifest = json.loads((out / "freeze_manifest.json").read_text())
    if digest(out / manifest["prediction_file"]) != manifest["prediction_sha256"]:
        raise ValueError("Restored predictions differ from freeze manifest")
    print(f"Restored and verified {manifest['source']['n_filters']} frozen filters at {out}")


def read_pairing(out: Path, pairing_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((out / "freeze_manifest.json").read_text())
    path = out / manifest["prediction_file"]
    if digest(path) != manifest["prediction_sha256"]:
        raise ValueError("Frozen predictions differ from their manifest")
    predictions = pd.read_csv(path)
    pairs = pd.read_csv(pairing_path, dtype=str).fillna("")
    missing = set(PAIR_COLUMNS) - set(pairs.columns)
    if missing:
        raise ValueError(f"Pairing table is missing columns: {sorted(missing)}")
    if len(pairs) != len(predictions) or pairs.sample_id.duplicated().any():
        raise ValueError("Pairing table must contain exactly one row for each frozen ETAD filter")
    if set(pairs.sample_id) != set(predictions.sample_id):
        raise ValueError("Pairing table must contain the complete frozen ETAD cohort")
    joined = pairs.merge(predictions, on="sample_id", suffixes=("_submitted", "_frozen"),
                         validate="one_to_one", sort=False)
    if not joined.ptfe_filter_id_submitted.eq(joined.ptfe_filter_id_frozen).all():
        raise ValueError("Submitted PTFE IDs disagree with frozen identities")
    if not joined.eligibility_status.isin(["eligible", "not_eligible"]).all():
        raise ValueError("Every filter needs eligible or not_eligible status")
    ineligible = joined.eligibility_status.eq("not_eligible")
    if joined.loc[ineligible, "eligibility_reason"].str.strip().eq("").any():
        raise ValueError("Every ineligible filter needs an eligibility reason")
    eligible = joined.eligibility_status.eq("eligible")
    if not eligible.any():
        raise ValueError("No eligible independent TOR pairs")
    if joined.loc[eligible, "eligibility_reason"].str.strip().ne("").any():
        raise ValueError("Eligible filters cannot have an exclusion reason")
    if joined.loc[eligible, "quartz_filter_id"].duplicated().any():
        raise ValueError("Eligible quartz filter identifiers must be unique")
    required = ["quartz_filter_id", "identity_provenance", "sampling_equivalence_provenance",
                "sampling_start", "sampling_end", "time_zone"]
    for name in required:
        if joined.loc[eligible, name].str.strip().eq("").any():
            raise ValueError(f"Missing required independent pairing evidence: {name}")
    for name in ("sampling_start", "sampling_end"):
        if not joined.loc[eligible, name].str.contains(
            r"(?:Z|[+-]\d{2}:\d{2})$", regex=True
        ).all():
            raise ValueError(f"{name} must include an explicit UTC offset")
    for name in ["identity_confirmed", "sampling_equivalent"]:
        if not joined.loc[eligible, name].str.lower().eq("true").all():
            raise ValueError(f"All evaluated pairs must have {name}=true")
    selected = joined.loc[eligible].copy()
    if not selected.tor_protocol.str.upper().eq("TOR").all():
        raise ValueError("Primary reference must explicitly use the TOR protocol")
    starts = pd.to_datetime(selected.sampling_start, utc=True, errors="raise")
    ends = pd.to_datetime(selected.sampling_end, utc=True, errors="raise")
    if not (ends > starts).all():
        raise ValueError("Each sampling end must follow its start")
    known_dates = selected.sampling_date.notna()
    if not selected.loc[known_dates, "sampling_date"].eq(
        selected.loc[known_dates, "sampling_start"].str.slice(0, 10)
    ).all():
        raise ValueError("Submitted sampling starts disagree with frozen ETAD dates")
    for name in ("quartz_volume_m3", "ptfe_volume_m3"):
        volume = pd.to_numeric(selected[name], errors="coerce")
        if not np.isfinite(volume).all() or (volume <= 0).any():
            raise ValueError(f"{name} must be finite and positive")
        selected[name] = volume
    original = selected.original_ptfe_volume_m3.to_numpy(float)
    submitted = selected.ptfe_volume_m3.to_numpy(float)
    recorded = np.isfinite(original) & (original > 0)
    changed_volume = ~recorded | ~np.isclose(submitted, original, rtol=1e-3, atol=0)
    if selected.loc[changed_volume, "volume_correction_reason"].str.strip().eq("").any():
        raise ValueError("Missing provenance for absent or changed original PTFE volume")
    selected = selected.sort_values("sample_id")
    canonical = pairs.loc[:, LOCK_COLUMNS].sort_values("sample_id").to_csv(index=False)
    lock_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return pairs, selected, {"metadata_sha256": lock_hash,
                             "prediction_sha256": manifest["prediction_sha256"],
                             "n_eligible": len(selected), "n_ineligible": int(ineligible.sum())}


def lock_pairs(out: Path, pairing_path: Path) -> None:
    pairs, _, lock = read_pairing(out, pairing_path)
    if pairs.loc[:, OUTCOME_COLUMNS].ne("").any().any():
        raise ValueError("Lock pairing eligibility before entering TOR outcomes or lab results")
    target = out / "pairing_lock.json"
    if target.exists() and json.loads(target.read_text()) != lock:
        raise ValueError("An incompatible pairing lock already exists; refusing to replace it")
    target.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Locked {lock['n_eligible']} eligible pairs; metadata SHA-256 {lock['metadata_sha256']}")


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    residual = prediction - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    return {
        "rmse_ug_m3": float(np.sqrt(np.mean(residual**2))),
        "mae_ug_m3": float(np.mean(np.abs(residual))),
        "bias_ug_m3": float(np.mean(residual)),
        "predictive_r2": float(1 - np.sum(residual**2) / denominator)
        if denominator > 0 else None,
    }


def date_block_interval(joined: pd.DataFrame, repeats: int = 10000) -> dict:
    """Paired resampling of Monday–Sunday local sampling-date blocks."""
    dates = pd.to_datetime(joined.sampling_start.str.slice(0, 10), errors="raise")
    blocks = dates.dt.to_period("W-SUN").astype(str)
    groups = [np.flatnonzero(blocks.eq(block).to_numpy()) for block in sorted(blocks.unique())]
    result = {"n_date_blocks": len(groups), "bootstrap_repeats": repeats,
              "seed": 20260922, "interval_level": 0.95, "delta_definition": "VIBES_minus_AIRSpec"}
    if len(groups) < 2:
        result.update({"delta_rmse_ci_low": None, "delta_rmse_ci_high": None,
                       "interval_status": "unavailable_fewer_than_two_date_blocks"})
        return result
    rng = np.random.default_rng(20260922)
    truth = joined.tor_ec_ug_m3.to_numpy(float)
    air = joined.AIRSpec_ec_ug_m3_confirmed.to_numpy(float)
    vibes = joined.VIBES_ec_ug_m3_confirmed.to_numpy(float)
    deltas = np.empty(repeats)
    for i in range(repeats):
        index = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        deltas[i] = np.sqrt(np.mean((vibes[index] - truth[index]) ** 2)) - np.sqrt(
            np.mean((air[index] - truth[index]) ** 2)
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    result.update({"delta_rmse_ci_low": float(low), "delta_rmse_ci_high": float(high),
                   "interval_status": "descriptive_date_block_bootstrap_conditional_on_frozen_fits"})
    return result


def score(out: Path, pairing_path: Path) -> None:
    lock_path = out / "pairing_lock.json"
    if not lock_path.exists():
        raise ValueError("Pairing eligibility must be locked before TOR outcomes are entered")
    pairs, joined, lock = read_pairing(out, pairing_path)
    if lock != json.loads(lock_path.read_text()):
        raise ValueError("Submitted pairing metadata or eligibility differs from its pre-outcome lock")
    ineligible = pairs.eligibility_status.eq("not_eligible")
    if pairs.loc[ineligible, OUTCOME_COLUMNS].ne("").any().any():
        raise ValueError("Ineligible rows must not carry TOR outcome values")
    if joined.laboratory_source_file.str.strip().eq("").any():
        raise ValueError("Missing required independent laboratory source file")
    if not joined.tor_qa_pass.str.lower().eq("true").all():
        raise ValueError("All evaluated pairs must have tor_qa_pass=true")
    tor = pd.to_numeric(joined.tor_ec_ug_filter, errors="coerce")
    if not np.isfinite(tor).all() or (tor < 0).any():
        raise ValueError("TOR EC must be finite, nonnegative, and in ug/filter")
    uncertainty = pd.to_numeric(joined.tor_uncertainty_ug_filter, errors="coerce")
    mdl = pd.to_numeric(joined.tor_mdl_ug_filter, errors="coerce")
    if not np.isfinite(uncertainty).all() or (uncertainty < 0).any():
        raise ValueError("TOR uncertainty must be finite and nonnegative")
    if not np.isfinite(mdl).all() or (mdl < 0).any():
        raise ValueError("TOR MDL must be finite and nonnegative")
    if (tor < mdl).any():
        raise ValueError("Below-MDL pairs require an explicit predeclared handling rule")
    joined["tor_ec_ug_filter"] = tor
    joined["tor_uncertainty_ug_filter"] = uncertainty
    joined["tor_mdl_ug_filter"] = mdl
    joined["tor_ec_ug_m3"] = tor.to_numpy(float) / joined.quartz_volume_m3.to_numpy(float)
    joined["tor_uncertainty_ug_m3"] = uncertainty.to_numpy(float) / joined.quartz_volume_m3.to_numpy(float)
    for method in METHODS:
        joined[f"{method}_ec_ug_m3_confirmed"] = (
            joined[f"{method}_ec_ug_filter"].to_numpy(float)
            / joined.ptfe_volume_m3.to_numpy(float)
        )
    missing_season = joined.ethiopia_season_dry_feb.eq("unknown")
    if missing_season.any():
        months = pd.to_datetime(joined.loc[missing_season, "sampling_start"].str.slice(0, 10)).dt.month
        seasons = resolve_seasons("dry_feb")
        joined.loc[missing_season, "ethiopia_season_dry_feb"] = months.map(
            lambda month: season_for_month(month, seasons)
        )
    results = {method: metrics(joined.tor_ec_ug_m3.to_numpy(float),
                              joined[f"{method}_ec_ug_m3_confirmed"].to_numpy(float))
               for method in METHODS}
    delta = results["VIBES"]["rmse_ug_m3"] - results["AIRSpec"]["rmse_ug_m3"]
    interval = date_block_interval(joined)
    season_counts = joined.ethiopia_season_dry_feb.value_counts().to_dict()
    # The historical proposal is 36 pairs spanning all three named seasons.
    coverage_gate = len(joined) >= 36 and all(
        season_counts.get(name, 0) >= minimum
        for name, minimum in (("Dry (Oct-Feb)", 13), ("Belg (Mar-May)", 12),
                              ("Kiremt (Jun-Sep)", 11))
    )
    improvement = (
        "exploratory_only_insufficient_seasonal_coverage" if not coverage_gate else
        "inconclusive_interval_unavailable" if interval["delta_rmse_ci_high"] is None else
        "VIBES_improves_RMSE" if interval["delta_rmse_ci_high"] < 0 else
        "AIRSpec_improves_RMSE" if interval["delta_rmse_ci_low"] > 0 else
        "inconclusive_interval_includes_zero"
    )
    report = {
        "scope": "submitted identity-confirmed independent TOR pairs, conditional on frozen 2026-09-21 models",
        "reference_sha256": digest(pairing_path),
        "prediction_sha256": lock["prediction_sha256"],
        "pairing_lock_sha256": digest(lock_path),
        "n_pairs": len(joined), "n_frozen_filters": lock["n_eligible"] + lock["n_ineligible"],
        "season_convention": "dry_feb", "season_counts": season_counts,
        "primary_metric_units": "ug/m3, using each paired filter's confirmed sample volume",
        "retained_mass_units": "ug/filter for TOR and each FTIR prediction",
        "metrics": results,
        "delta_rmse_VIBES_minus_AIRSpec": delta, "paired_interval": interval,
        "coverage_gate_met": coverage_gate,
        "relative_improvement_reading": improvement,
        "absolute_accuracy_reading": "not_assessed; application-specific threshold remains to be agreed",
        "below_mdl_rule": "fail evaluation; no clipping, substitution or post hoc exclusion",
        "provenance_limit": "Identity, sampling equivalence and laboratory method flags require review against original records; this command validates their presence and consistency, not their authenticity.",
    }
    result_dir = out / f"evaluation_{report['reference_sha256'][:12]}"
    result_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(result_dir / "paired_predictions_and_TOR.csv", index=False)
    (result_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Evaluated {len(joined)} confirmed pairs; {improvement}. Report: {result_dir / 'report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "package", "restore", "lock-pairs", "check-pairs"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--destination", type=Path, default=DELIVERABLE)
    parser.add_argument("--pairs", type=Path, help="Independent TOR pairing CSV for check-pairs")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    elif args.command == "package":
        package(args.out, args.destination)
    elif args.command == "restore":
        restore(args.out, args.destination)
    elif not args.pairs:
        parser.error("--pairs is required for lock-pairs and check-pairs")
    elif args.command == "lock-pairs":
        lock_pairs(args.out, args.pairs)
    else:
        score(args.out, args.pairs)


if __name__ == "__main__":
    main()
