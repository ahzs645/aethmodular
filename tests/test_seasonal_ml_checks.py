import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(ROOT / "gallery/data"))

from seasonal_ml_checks import ECLine, fill_lot_groups, lot_season_counts, regression_scores, spartan_lot_map  # noqa: E402


def filters_table(rows):
    return pd.DataFrame(rows, columns=["ExternalFilterId", "ExternalFilterType", "ExternalLotId"])


def test_lot_map_uses_pm25_physical_ids_and_rejects_conflicts():
    lots = spartan_lot_map(filters_table([
        ("ETAD-0001-1", "PM2.5", "248"), ("ETAD-0001-2", "PM2.5", "248"),
        ("ETAD-0002-1", "PM2.5", "251"), ("ETAD-0002-1", "PM10", "999"), ("ETAD-0003-1", "PM2.5", None),
    ]))
    assert lots.to_dict() == {"ETAD-0001": "248", "ETAD-0002": "251"}
    with pytest.raises(ValueError, match="conflicting lots"):
        spartan_lot_map(filters_table([("ETAD-0001-1", "PM2.5", "248"), ("ETAD-0001-2", "PM2.5", "251")]))


def test_fill_lots_keeps_recorded_lots_fills_unknown_and_refuses_disagreement():
    cohort = pd.DataFrame({"id": ["ETAD-0001", "ETAD-0002", "ETAD-0003"], "lot_group": ["248.0", "unknown", np.nan]})
    lots = pd.Series({"ETAD-0001": "248", "ETAD-0002": "251"})
    out = fill_lot_groups(cohort, lots)
    assert out.lot_group.tolist() == ["248", "251", "unknown"]
    assert out.lot_source.tolist() == ["native lot map", "SPARTAN filter table", "unknown"]
    assert cohort.lot_group.tolist()[1] == "unknown", "input is not mutated"
    with pytest.raises(ValueError, match="disagrees"):
        fill_lot_groups(cohort, pd.Series({"ETAD-0001": "251"}))


def test_ec_line_is_training_only_least_squares_and_scores_are_signed():
    ec = np.array([1., 2., 3., 4.])
    line = ECLine().fit(ec, 0.5 + 0.9 * ec)
    assert line.slope_ == pytest.approx(0.9) and line.intercept_ == pytest.approx(0.5)
    assert line.predict([10.])[0] == pytest.approx(9.5)
    s = regression_scores([1., 2., 3.], [2., 3., 4.])
    assert s["rmse"] == pytest.approx(1) and s["bias"] == pytest.approx(1) and s["r2"] == pytest.approx(-0.5)
    with pytest.raises(ValueError):
        ECLine().fit([1., 1., 1.], [1., 2., 3.])


def test_lot_season_counts_are_long_and_complete():
    cohort = pd.DataFrame({"cycle_end": [2023, 2023, 2024], "season": ["Bega", "Bega", "Bega"], "lot_group": ["248", "248", "251"]})
    assert lot_season_counts(cohort).to_dict("records") == [
        {"cycle_end": 2023, "season": "Bega", "lot_group": "248", "n": 2},
        {"cycle_end": 2024, "season": "Bega", "lot_group": "251", "n": 1}]


def checks_payload(**extra):
    return {"schema_version": 1, "lots": {"counts": []}, "ec_line": [{"rmse": .5}], "residual_scores": [{"rmse": .4}],
            "residual_importance": [], "lot_control": [], "lot_separability": [], "limitations": [],
            "provenance": {"inputs_unchanged_during_run": True}, **extra}


def test_checks_export_copies_served_json_and_owns_only_its_download_folder(tmp_path):
    import export_baseline_lab as ex
    out, src = tmp_path / "out", tmp_path / "checks"
    src.mkdir()
    (src / "checks.json").write_text(json.dumps(checks_payload()))
    (src / "ec_line_scores.csv").write_text("rmse\n0.5\n")
    stale = out / "tables/seasonal_checks/old.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("x")
    keep = out / "tables/seasonal_ml/keep.csv"
    keep.parent.mkdir(parents=True)
    keep.write_text("y")
    assert ex.export_seasonal_checks(out, src) == [("seasonal_checks", src)]
    assert (out / "seasonal_checks.json").read_bytes() == (src / "checks.json").read_bytes()
    assert not stale.exists() and keep.exists()
    downloads = ex.copy_download_folders(out, [("seasonal_checks", src)])
    assert {d["path"] for d in downloads} == {"research/baseline_lab/tables/seasonal_checks/checks.json",
                                               "research/baseline_lab/tables/seasonal_checks/ec_line_scores.csv"}


def test_checks_export_refuses_incomplete_or_changed_runs_and_clears_when_absent(tmp_path):
    import export_baseline_lab as ex
    out, src = tmp_path / "out", tmp_path / "checks"
    src.mkdir()
    for bad in (checks_payload(ec_line=[]), checks_payload(provenance={"inputs_unchanged_during_run": False}),
                {k: v for k, v in checks_payload().items() if k != "lot_control"}):
        (src / "checks.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            ex.export_seasonal_checks(out, src)
    (src / "checks.json").unlink()
    (out / "seasonal_checks.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "seasonal_checks.json").write_text("{}")
    assert ex.export_seasonal_checks(out, src) == []
    assert not (out / "seasonal_checks.json").exists()
