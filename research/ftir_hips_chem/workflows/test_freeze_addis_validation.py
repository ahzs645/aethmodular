"""Validation gates for the Addis prediction freeze; TOR values are synthetic."""

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from freeze_addis_validation import DELIVERABLE, PAIR_COLUMNS, lock_pairs, restore, score


def synthetic_pairs(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "freeze"
    package.mkdir()
    with ZipFile(DELIVERABLE / "addis_prediction_freeze.zip") as zipped:
        for name in ("freeze_manifest.json", "frozen_full_pool_predictions.csv"):
            zipped.extract(name, path=package)
    predictions = pd.read_csv(package / "frozen_full_pool_predictions.csv")
    counts = {"Dry (Oct-Feb)": 13, "Belg (Mar-May)": 12, "Kiremt (Jun-Sep)": 11}
    chosen = pd.concat([
        predictions.loc[predictions.ethiopia_season_dry_feb.eq(season)
                        & predictions.original_ptfe_volume_m3.notna()].head(n)
        for season, n in counts.items()
    ], ignore_index=True)
    assert len(chosen) == 36
    pairs = predictions[["sample_id", "ptfe_filter_id"]].copy()
    for name in PAIR_COLUMNS[2:]:
        pairs[name] = ""
    pairs["eligibility_status"] = "not_eligible"
    pairs["eligibility_reason"] = "no paired quartz in synthetic fixture"
    for i, row in enumerate(chosen.itertuples()):
        index = pairs.index[pairs.sample_id.eq(row.sample_id)][0]
        pairs.loc[index, "eligibility_status"] = "eligible"
        pairs.loc[index, "eligibility_reason"] = ""
        pairs.loc[index, "quartz_filter_id"] = f"SYNTHETIC-Q-{i:03d}"
        pairs.loc[index, "identity_provenance"] = "synthetic fixture only"
        pairs.loc[index, "sampling_equivalence_provenance"] = "synthetic fixture only"
        pairs.loc[index, "identity_confirmed"] = "true"
        pairs.loc[index, "sampling_equivalent"] = "true"
        pairs.loc[index, "sampling_start"] = row.sampling_date + "T00:00:00+03:00"
        pairs.loc[index, "sampling_end"] = row.sampling_date + "T23:59:00+03:00"
        pairs.loc[index, "time_zone"] = "Africa/Addis_Ababa"
        pairs.loc[index, "quartz_volume_m3"] = str(row.original_ptfe_volume_m3 * 0.8)
        pairs.loc[index, "ptfe_volume_m3"] = str(row.original_ptfe_volume_m3)
        pairs.loc[index, "tor_protocol"] = "TOR"
    path = tmp_path / "synthetic_pairs.csv"
    pairs.to_csv(path, index=False)
    lock_pairs(package, path)
    for i, row in enumerate(chosen.itertuples()):
        index = pairs.index[pairs.sample_id.eq(row.sample_id)][0]
        pairs.loc[index, "tor_ec_ug_filter"] = str(float(i + 1))
        pairs.loc[index, "tor_uncertainty_ug_filter"] = "0.5"
        pairs.loc[index, "tor_mdl_ug_filter"] = "0.1"
        pairs.loc[index, "tor_qa_pass"] = "true"
        pairs.loc[index, "laboratory_source_file"] = "synthetic_fixture.csv"
    pairs.to_csv(path, index=False)
    return package, path


def test_synthetic_pair_scoring_is_reproducible(tmp_path: Path) -> None:
    package, pairs = synthetic_pairs(tmp_path)
    score(package, pairs)
    reports = list(package.glob("evaluation_*/report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["n_pairs"] == 36
    assert report["coverage_gate_met"]
    assert report["paired_interval"]["n_date_blocks"] >= 2
    assert report["absolute_accuracy_reading"].startswith("not_assessed")
    paired = pd.read_csv(reports[0].parent / "paired_predictions_and_TOR.csv")
    first = paired.iloc[0]
    assert first.tor_ec_ug_m3 == pytest.approx(first.tor_ec_ug_filter / first.quartz_volume_m3)
    assert first.AIRSpec_ec_ug_m3_confirmed == pytest.approx(
        first.AIRSpec_ec_ug_filter / first.ptfe_volume_m3
    )
    assert first.quartz_volume_m3 != first.ptfe_volume_m3


@pytest.mark.parametrize("column,value,reason", [
    ("identity_confirmed", "false", "identity_confirmed"),
    ("tor_protocol", "FTIR", "TOR protocol"),
    ("sampling_start", "2022-12-07", "UTC offset"),
    ("tor_ec_ug_filter", "not measured", "finite"),
])
def test_incomplete_or_surrogate_pairs_fail(
    tmp_path: Path, column: str, value: str, reason: str
) -> None:
    package, path = synthetic_pairs(tmp_path)
    pairs = pd.read_csv(path, dtype=str)
    index = pairs.index[pairs.eligibility_status.eq("eligible")][0]
    pairs.loc[index, column] = value
    pairs.to_csv(path, index=False)
    with pytest.raises(ValueError, match=reason):
        score(package, path)
    assert not list(package.glob("evaluation_*"))


def test_changed_predictions_fail_hash_gate(tmp_path: Path) -> None:
    package, path = synthetic_pairs(tmp_path)
    prediction = package / "frozen_full_pool_predictions.csv"
    prediction.write_text(prediction.read_text().replace("6.08855193532864", "99.08855193532864", 1))
    with pytest.raises(ValueError, match="Frozen predictions differ"):
        score(package, path)


def test_eligibility_change_after_lock_fails(tmp_path: Path) -> None:
    package, path = synthetic_pairs(tmp_path)
    pairs = pd.read_csv(path, dtype=str)
    index = pairs.index[pairs.eligibility_status.eq("not_eligible")][0]
    pairs.loc[index, "eligibility_reason"] = "changed after seeing TOR"
    pairs.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differs from its pre-outcome lock"):
        score(package, path)


def test_release_restores_without_colab_arrays(tmp_path: Path) -> None:
    restored = tmp_path / "restored"
    restore(restored, DELIVERABLE)
    assert pd.read_csv(restored / "frozen_full_pool_predictions.csv").shape[0] == 253
    restore(restored, DELIVERABLE)
