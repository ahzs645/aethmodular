"""Regression checks for the standalone PMF runner."""
import csv
import json

import numpy as np
import pytest

from aethmodular_cli.pmf import load_pair, main


def write(path, ids, features, data):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", *features])
        for name, values in zip(ids, data):
            writer.writerow([name, *values])


def inputs(tmp_path):
    x = tmp_path / "x.csv"
    u = tmp_path / "u.csv"
    ids = ["ETAD-0001-1", "NA", "ETAD-0003-3", "ETAD-0004-4"]
    features = ["4000", "3900", "1700"]
    values = [[1.0, 0.0, -0.1], [0.0, 1.0, 0.2],
              [1.0, 1.0, 0.1], [0.4, 0.6, 0.0]]
    write(x, ids, features, values)
    write(u, ids, features, [[0.1] * 3 for _ in ids])
    return x, u


def test_roundtrip_and_ids(tmp_path):
    x, u = inputs(tmp_path)
    ids, features, xx, uu = load_pair(x, u)
    assert ids[1] == "NA"
    assert xx[0, 2] == -0.1
    assert (uu > 0).all()
    out = tmp_path / "epa"
    assert main(["epa-export", "--x", str(x), "--u", str(u),
                 "--dataset", "ETAD-test", "--output", str(out)]) == 0
    assert load_pair(out / "concentrations.csv", out / "uncertainties.csv")[0] == ids
    receipt = json.loads((out / "manifest.json").read_text())
    assert receipt["status"] == "not_run"
    assert receipt["target_engine"] == "EPA PMF5/ME-2"


def test_reject_swapped_rows_and_invalid_uncertainty(tmp_path):
    x, u = inputs(tmp_path)
    ids, features, _, _ = load_pair(x, u)
    write(u, ids[::-1], features, [[0.1] * 3 for _ in ids])
    with pytest.raises(ValueError, match="identities"):
        load_pair(x, u)
    write(u, ids, features, [[0.1] * 3, [0, 0.1, 0.1],
                            [0.1] * 3, [0.1] * 3])
    with pytest.raises(ValueError, match="strictly positive"):
        load_pair(x, u)


def test_fit_and_frozen_projection(tmp_path):
    x, u = inputs(tmp_path)
    fit = tmp_path / "fit"
    assert main(["fit", "--x", str(x), "--u", str(u), "--dataset", "ETAD",
                 "--output", str(fit), "--factors", "2", "--seeds", "1", "2",
                 "--iterations", "50"]) == 0
    manifest = json.loads((fit / "manifest.json").read_text())
    assert manifest["engine"] == "python-independent"
    assert len(manifest["runs"]) == 2
    _, _, g = __import__("aethmodular_cli.pmf", fromlist=["_read"])._read(fit / "contributions.csv")
    _, _, h = __import__("aethmodular_cli.pmf", fromlist=["_read"])._read(fit / "profiles.csv")
    assert g.shape == (4, 2)
    assert h.shape == (2, 3)
    assert np.min(g) >= 0 and np.min(h) >= 0
    before = (fit / "profiles.csv").read_bytes()
    out = tmp_path / "project"
    assert main(["project", "--x", str(x), "--u", str(u), "--dataset", "ETAD-new",
                 "--output", str(out), "--profiles", str(fit / "profiles.csv")]) == 0
    assert (fit / "profiles.csv").read_bytes() == before
    assert json.loads((out / "manifest.json").read_text())["frozen_profiles_sha256"]


def test_esat_is_blocked_not_fake_success(tmp_path):
    x, u = inputs(tmp_path)
    out = tmp_path / "esat"
    assert main(["esat-check", "--x", str(x), "--u", str(u),
                 "--dataset", "ETAD", "--output", str(out)]) == 0
    assert json.loads((out / "manifest.json").read_text())["status"] == "blocked"
