"""Checkpoint identity, held-out-site isolation and paired inference checks."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cvxpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from vibes_large_run import (
    RunConfig,
    prepare_experiment,
    atomic_checkpoint,
    read_checkpoint,
    paired_site_bootstrap,
)


@pytest.fixture
def portable_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    wn = np.linspace(3998, 1426, 200)
    rng = np.random.default_rng(5)
    np.save(data / "wn.npy", wn)
    pool = pd.DataFrame(
        {
            "FilterId": np.arange(60),
            "Site": [f"S{i % 10}" for i in range(60)],
            "LotNumber": 251,
            "FilterPurposeId": [1] * 40 + [2] * 20,
            "finite_spectrum": True,
            "all_scans_have_purpose": True,
            "eligible": [True] * 40 + [False] * 20,
            "locked800": [True] * 20 + [False] * 40,
            "TOR_EC_loading_ug": 1.0,
            "date": "2020-01-01",
        }
    )
    pool["split"] = np.where(pool.Site.isin(["S8", "S9"]), "test", "train")
    pool.to_csv(data / "pool_metadata.csv", index=False)
    np.save(data / "pool_raw.npy", rng.normal(size=(60, len(wn))))
    etad = pd.DataFrame(
        {
            "MediaId": np.arange(10),
            "role": ["blank_train"] * 4 + ["blank_test"] * 2 + ["sample"] * 4,
            "LotId": 251,
            "SampleVolume_m3": 7.0,
            "ExternalFilterId": [f"ETAD-{i}" for i in range(10)],
        }
    )
    etad.to_csv(data / "etad_metadata.csv", index=False)
    np.save(data / "etad_raw.npy", rng.normal(size=(10, len(wn))))
    return tmp_path


def test_no_physical_or_outer_site_leakage(portable_data):
    wn, X, cases, blanks, ids, bank, shape = prepare_experiment(portable_data, RunConfig())
    assert set(ids).isdisjoint(cases.sample_id)
    assert set(bank.loc[bank.source.eq("IMPROVE"), "Site"]).isdisjoint({"S8", "S9"})
    assert cases.sample_id.is_unique
    assert X.shape[1] == len(wn) == len(shape)
    for _, row in cases[cases.kind.eq("injection")].iterrows():
        assert cases.loc[int(row.parent), "kind"] == "blank"
    assert len(blanks) == len(ids)


def test_test_labels_do_not_change_background_selection(portable_data):
    before = prepare_experiment(portable_data, RunConfig())
    path = portable_data / "data/pool_metadata.csv"
    d = pd.read_csv(path)
    d.loc[d.split.eq("test"), "TOR_EC_loading_ug"] = 123456
    d.to_csv(path, index=False)
    after = prepare_experiment(portable_data, RunConfig())
    assert before[4] == after[4]
    np.testing.assert_array_equal(before[3], after[3])
    np.testing.assert_array_equal(before[1], after[1])


def test_checkpoint_roundtrip_and_identity(tmp_path):
    path = tmp_path / "batch.npz"
    idx = np.array([3, 4])
    a = np.arange(8).reshape(2, 4)
    diagnostics = [{"case_row": 3, "success": True}, {"case_row": 4, "success": True}]
    atomic_checkpoint(path, "signature", idx, a, a + 1, diagnostics)
    air, vib, diag = read_checkpoint(path, "signature", idx, 4)
    np.testing.assert_array_equal(air, a)
    np.testing.assert_array_equal(vib, a + 1)
    assert diag == diagnostics
    with pytest.raises(ValueError, match="Stale"):
        read_checkpoint(path, "changed-settings", idx, 4)
    with pytest.raises(ValueError, match="Stale"):
        read_checkpoint(path, "signature", idx[::-1], 4)
    with pytest.raises(ValueError, match="shape"):
        read_checkpoint(path, "signature", idx, 5)
    assert not path.with_suffix(".partial").exists()


def test_paired_site_bootstrap_identical_and_worse():
    y = np.arange(30, dtype=float)
    sites = np.repeat(["A", "B", "C"], 10)
    same = paired_site_bootstrap(y, y + 0.2, y + 0.2, sites, repeats=100)
    assert same["delta_rmse"] == same["ci_low"] == same["ci_high"] == 0
    worse = paired_site_bootstrap(y, y + 0.2, y + 1, sites, repeats=100)
    assert worse["ci_low"] > 0 and worse["ci_high"] > 0
    assert worse["n_sites"] == 3


def test_invalid_config():
    with pytest.raises(ValueError, match="profile"):
        RunConfig(profile="typo").validate()
    with pytest.raises(ValueError, match="batch_size"):
        RunConfig(batch_size=0).validate()
