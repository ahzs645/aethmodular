"""End-to-end training/test boundaries for EPO, on synthetic physical filters."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/ftir_hips_chem/scripts"), str(ROOT / "research/ftir_hips_chem/workflows")]
import epo_validation as epo  # noqa: E402
import run_intercept_fixes_20261103 as workflow  # noqa: E402
from candidate_metrics import sha256_file, validate_archive_identity, verified_score_receipt  # noqa: E402
from spectral_provenance import verify_corrected_spectra, verify_manifest_files  # noqa: E402


def synthetic_source():
    rng = np.random.default_rng(812)
    n, channels = 90, 36
    sites = np.repeat(np.array(list("ABCDEF")), n // 6)
    ids = np.array([f"{site}-{i:03d}" for i, site in enumerate(sites)])
    chemistry = rng.lognormal(1, .45, (n, 5))
    chemistry[:, 0] += 8
    spectra = chemistry @ rng.normal(0, .05, (5, channels)) + rng.normal(0, .08, (n, channels))
    return spectra, chemistry[:, 0].copy(), chemistry, ids, sites


def test_heldout_tor_threshold_and_chemistry_changes_cannot_change_own_fold():
    X, y, Z, ids, sites = synthetic_source()
    splits = epo.site_folds(ids, sites)
    original = epo.fit_site_cv(X, y, Z, ids, sites, 3, splits=splits)
    heldout = splits[0][1]
    changed_y, changed_Z = y.copy(), Z.copy()
    # Move test labels across the source-domain threshold, including missing
    # labels. A global eligibility pass must not alter this fold's training.
    changed_y[heldout] = 10000
    changed_y[heldout[0]] = np.nan
    changed_Z[heldout, 0] = changed_y[heldout]
    changed_Z[heldout, 1:] *= 500
    changed_Z[heldout[1], 2] = np.nan
    changed = epo.fit_site_cv(X, changed_y, changed_Z, ids, sites, 3, splits=splits)
    np.testing.assert_array_equal(original[1], changed[1])
    np.testing.assert_array_equal(original[2][0].projection.basis, changed[2][0].projection.basis)
    np.testing.assert_array_equal(original[2][0].projection.signatures, changed[2][0].projection.signatures)
    for before, after in zip(original[2][0].pls_model, changed[2][0].pls_model):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(original[0][heldout], changed[0][heldout])
    assert original[3][0] == changed[3][0]
    assert np.nanmedian(np.abs(changed[0][heldout, 2] - changed_y[heldout])) > 1000
    # Other folds and the intentionally separate all-source model can change:
    # the perturbed site is training data for those fits.
    assert len(changed[4].training_rows) < len(original[4].training_rows)


def test_projection_uses_complete_training_chemistry_and_no_heldout_sites():
    X, y, Z, ids, sites = synthetic_source()
    Z[0, 4] = np.nan
    y[1] = 170
    Z[1, 0] = 170
    predictions, fold_ids, fits, receipts, full = epo.fit_site_cv(X, y, Z, ids, sites, 3)
    assert np.isfinite(predictions).all()  # Reference-ineligible rows are still held out and predicted.
    for fitted, receipt in zip(fits, receipts):
        assert set(receipt["training_sites"]).isdisjoint(receipt["test_sites"])
        assert set(receipt["projection_training_filter_ids"]).isdisjoint(receipt["test_filter_ids"])
        assert ids[0] not in receipt["projection_training_filter_ids"]
        assert ids[1] not in receipt["training_filter_ids"]
        assert receipt["projection_rank"] == 4 and receipt["chemistry_design_rank"] == 5
        np.testing.assert_allclose(fitted.projection.basis.T @ fitted.projection.basis, np.eye(4), atol=1e-12)
        assert set(fold_ids[np.isin(ids, receipt["test_filter_ids"])]) == {receipt["fold_id"]}
    assert 0 in full.training_rows and 0 not in full.projection.training_rows


def test_rank_deficiency_is_reported_and_zero_nuisance_has_no_arbitrary_basis():
    X, y, Z, ids, sites = synthetic_source()
    bad = Z.copy()
    bad[:, 4] = bad[:, 3]
    with pytest.raises(ValueError, match="chemistry design has rank"):
        epo.fit_training_epo(X, y, bad, np.arange(len(y)), 2)
    with pytest.raises(ValueError, match="six complete training"):
        epo.fit_projection(X, Z, np.arange(5))
    ec_only = Z[:, :1] @ np.linspace(.1, 1, X.shape[1])[None, :]
    projection = epo.fit_projection(ec_only, Z, np.arange(len(y)))
    assert projection.rank == 0
    np.testing.assert_array_equal(projection.transform(ec_only), ec_only)


def test_custom_splits_cannot_cross_site_or_physical_filter_boundaries():
    X, y, Z, ids, sites = synthetic_source()
    duplicate_ids = ids.copy()
    duplicate_ids[1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="repeated physical-filter"):
        epo.site_folds(duplicate_ids, sites)
    train, test = np.arange(1, len(y)), np.array([0])
    with pytest.raises(ValueError, match="sites or physical filters overlap"):
        epo.fit_site_cv(X, y, Z, ids, sites, 2, splits=[(train, test)])


def write_manifest(directory, names, *, plain_hashes=False):
    entries = {name: sha256_file(directory / name) for name in names}
    (directory / "MANIFEST.json").write_text(json.dumps({"files": entries if plain_hashes else
                                                        {name: {"sha256": value} for name, value in entries.items()}}))


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    X, y, Z, ids, sites = synthetic_source()
    bundle, corrected = tmp_path / "bundle", tmp_path / "corrected"
    bundle.mkdir()
    (corrected / "bundle").mkdir(parents=True)
    volume = 2.5
    im = pd.DataFrame(dict(filter_id=ids, site=sites, ec_ug=y, volume_m3=volume,
                           tor_ec_ugm3=y / volume, tor_oc_ugm3=Z[:, 1] / volume,
                           fe_ugm3=Z[:, 3] / volume, k_xrf_ugm3=Z[:, 4] / volume, fabs_Mm1=10.))
    mass = pd.DataFrame(dict(filter_id=ids, pm25_ugm3=(Z[:, 2] + 1.8 * Z[:, 1] + y) / volume))
    ad = pd.DataFrame(dict(filter_id=[f"ETAD-{i:04d}" for i in range(8)], fabs_Mm1=20., volume_m3=2.5, ec_published_ugm3=3.))
    nw = pd.DataFrame(dict(filter_id=[f"SITE-{i:04d}" for i in range(10)], fabs_Mm1=30., volume_m3=3., ec_ftir_ugm3=2.))
    frames = {"improve": im, "addis": ad, "network": nw}
    for group, frame in frames.items():
        frame.to_csv(bundle / f"{group}_meta.csv", index=False)
    mass.to_csv(bundle / "improve_mass.csv", index=False)
    wn = np.linspace(3990, 500, X.shape[1] * 2)
    np.save(bundle / "wn_full.npy", wn)
    np.save(corrected / "bundle/wn.npy", wn)
    matrices = {"improve": X, "addis": X[:len(ad)] + .2, "network": X[:len(nw)] - .1}
    rows = []
    for group, frame in frames.items():
        rows.append(pd.DataFrame(dict(group=group, group_row=np.arange(len(frame)), filter_id=frame.filter_id)))
        np.save(corrected / f"corrected_anchorhp_{group}.npy", np.repeat(matrices[group], 2, axis=1))
    row_map = pd.concat(rows, ignore_index=True)
    row_map["row"] = np.arange(len(row_map))
    row_map.to_csv(corrected / "bundle/rows.csv", index=False)
    write_manifest(corrected / "bundle", ("rows.csv", "wn.npy"), plain_hashes=True)
    write_manifest(bundle, ("improve_meta.csv", "addis_meta.csv", "network_meta.csv", "improve_mass.csv", "wn_full.npy"))
    monkeypatch.setattr(workflow.td, "VC", corrected)
    monkeypatch.setattr(workflow, "EPO_OUT", tmp_path / "versioned" / "epo_site_cv_v2")
    return bundle, corrected, frames, matrices, wn


def test_real_workflow_boundary_saves_transforms_and_preserves_historical_outputs(frozen_inputs, tmp_path, monkeypatch):
    bundle, corrected, frames, matrices, wn = frozen_inputs
    def forbidden(*args, **kwargs):
        raise AssertionError("bounded base/EPO fitting must not read optical comparators or run unrelated candidates")
    monkeypatch.setattr(workflow, "addis_frame", forbidden)
    monkeypatch.setattr(workflow, "domain_weights", forbidden)
    monkeypatch.setattr(workflow, "dipls_fit", forbidden)
    historical = tmp_path / "historical"
    historical.mkdir()
    sentinel = historical / "predictions.npz"
    sentinel.write_bytes(b"historical saved artifact")
    monkeypatch.setattr(workflow, "OUT", historical)
    destination = workflow.fit(bundle, models=("base", "epo"), kmax=3)
    assert destination == workflow.EPO_OUT and sentinel.read_bytes() == b"historical saved artifact"
    receipt = json.loads((destination / "fit_receipt.json").read_text())
    assert receipt["epo_protocol"] == epo.EPO_PROTOCOL
    assert set(receipt["models"]) == {"base", "epo"}
    assert receipt["epo"]["final_model"]["role"] == "all_source_target_model"
    im, _, _ = workflow.td.load(bundle)
    with np.load(destination / "predictions.npz", allow_pickle=False) as archive:
        folds = validate_archive_identity(archive, receipt, im, bundle / "improve_meta.csv", destination / "predictions.npz")
        for fold in range(5):
            prefix = f"epo_fold_{fold}"
            test = folds == fold
            Q = archive[f"{prefix}|basis"]
            transformed = matrices["improve"][test] - matrices["improve"][test] @ Q @ Q.T
            restored = (transformed - archive[f"{prefix}|xm"]) @ archive[f"{prefix}|B"] + archive[f"{prefix}|ym"]
            np.testing.assert_allclose(restored, archive["epo|improve_oof"][test], atol=1e-12)
            assert set(archive[f"{prefix}|projection_training_ids"]).isdisjoint(archive["improve_filter_ids"][test])
        Q = archive["epo|basis"]
        transformed = matrices["addis"] - matrices["addis"] @ Q @ Q.T
        restored = (transformed - archive["epo|xm"]) @ archive["epo|B"] + archive["epo|ym"]
        np.testing.assert_allclose(restored, archive["epo|addis"], atol=1e-12)
        assert set(archive["epo|training_ids"]) == set(archive["improve_filter_ids"])
    assert receipt["corrected_spectra_provenance"]["bundle/rows.csv"] == sha256_file(corrected / "bundle/rows.csv")


def test_workflow_heldout_labels_and_chemistry_do_not_influence_own_fold(frozen_inputs, tmp_path):
    bundle, _, _, _, _ = frozen_inputs
    before_dir = workflow.fit(bundle, models=("epo",), output=tmp_path / "before", kmax=3)
    with np.load(before_dir / "predictions.npz", allow_pickle=False) as archive:
        original = {name: archive[name] for name in archive.files}
    heldout = original["improve_fold_id"] == 0
    im = pd.read_csv(bundle / "improve_meta.csv")
    im.loc[heldout, "ec_ug"] = 10000
    im.loc[heldout, "tor_ec_ugm3"] = 4000
    im.loc[heldout, "tor_oc_ugm3"] *= 20
    im.loc[np.flatnonzero(heldout)[0], "k_xrf_ugm3"] = np.nan
    im.to_csv(bundle / "improve_meta.csv", index=False)
    mass = pd.read_csv(bundle / "improve_mass.csv")
    mass.loc[heldout, "pm25_ugm3"] *= 50
    mass.to_csv(bundle / "improve_mass.csv", index=False)
    write_manifest(bundle, ("improve_meta.csv", "addis_meta.csv", "network_meta.csv", "improve_mass.csv", "wn_full.npy"))
    after_dir = workflow.fit(bundle, models=("epo",), output=tmp_path / "after", kmax=3)
    with np.load(after_dir / "predictions.npz", allow_pickle=False) as changed:
        for suffix in ("basis", "signatures", "xm", "ym", "B", "projection_training_ids", "training_ids"):
            key = f"epo_fold_0|{suffix}"
            np.testing.assert_array_equal(original[key], changed[key])
        np.testing.assert_array_equal(original["epo|improve_oof"][heldout], changed["epo|improve_oof"][heldout])


def test_actual_epo_archive_integrates_with_the_common_filter_scorer(frozen_inputs, tmp_path):
    import run_hips_response_20261103 as response
    bundle, _, _, _, _ = frozen_inputs
    destination = workflow.fit(bundle, models=("base", "epo"), output=tmp_path / "fitted", kmax=3)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with np.load(destination / "predictions.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("improve_filter_ids", "improve_site_ids", "improve_fold_id")}
        # Deliberately identical synthetic comparison candidates: this tests the
        # artifact boundary, not a scientific ranking of the baseline methods.
        arrays.update({f"{name}|improve_oof": archive["base|improve_oof"] for name in
                       ("raw", "spline", "vibes", "vibes_deflate", "vibes_anchorhp")})
    np.savez(baseline / "predictions.npz", **arrays)
    identity = json.loads((destination / "fit_receipt.json").read_text())["source_identity"]
    identity["predictions_sha256"] = sha256_file(baseline / "predictions.npz")
    (baseline / "fit_receipt.json").write_text(json.dumps({"source_identity": identity}))
    scores = tmp_path / "source_scores_v2"
    response.source_scores(bundle, epo_output=destination, baseline_output=baseline, scores_output=scores, k=3)
    status = verified_score_receipt(scores)
    assert status["status"] == "corrected"
    assert status["provenance"]["epo_protocol"] == epo.EPO_PROTOCOL
    assert status["population"]["n_common"] == len(arrays["improve_filter_ids"])
    scoreboard = pd.read_csv(scores / "baseline_scoreboard.csv").set_index("model")
    assert scoreboard.loc["epo", "evaluation_status"] == "corrected"
    assert scoreboard.loc["epo", "coverage"] == 1
    assert np.isfinite(scoreboard.loc["epo", "improve_median_absolute_error"])


def test_corrected_spectra_cannot_borrow_identity_from_equal_length_metadata(frozen_inputs):
    bundle, corrected, frames, _, wn = frozen_inputs
    verify_corrected_spectra(corrected, frames, wn, ("anchorhp",))
    reordered = {name: frame.copy() for name, frame in frames.items()}
    reordered["improve"] = reordered["improve"].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="physical-filter row order differs"):
        verify_corrected_spectra(corrected, reordered, wn, ("anchorhp",))
    with pytest.raises(ValueError, match="wavenumber grid differs"):
        verify_corrected_spectra(corrected, frames, wn + .01, ("anchorhp",))
    with (corrected / "bundle/rows.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="frozen manifest"):
        verify_corrected_spectra(corrected, frames, wn, ("anchorhp",))


def test_original_spectral_file_cannot_change_under_a_frozen_manifest(frozen_inputs):
    bundle, _, _, _, wn = frozen_inputs
    verify_manifest_files(bundle, ("wn_full.npy",))
    np.save(bundle / "wn_full.npy", wn + .01)
    with pytest.raises(ValueError, match="frozen manifest"):
        verify_manifest_files(bundle, ("wn_full.npy",))
