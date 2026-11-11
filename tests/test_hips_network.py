"""The network HIPS analysis (research/ftir_hips_chem/workflows/run_hips_network_20261111.py) and its committed summary."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "ftir_hips_chem" / "workflows"))

import run_hips_network_20261111 as hips  # noqa: E402

SUMMARY = json.loads((ROOT / "docs" / "hips-network-2026-11-11.json").read_text())


def test_under_read_is_optical_depth_over_ceiling():
    assert np.allclose(hips.under_read([0.0, 0.5, 1.0, 5.0], 2.0), [0.0, 0.25, 0.5, 1.0])   # clipped at the ceiling


def test_replaying_a_filter_at_its_own_volume_changes_nothing_and_less_air_reads_lower():
    tau, volume = np.array([0.1, 0.5, 1.2]), np.array([7.2, 7.2, 3.6])
    assert np.allclose(hips.at_volume(tau, volume, volume, 2.26), tau)
    half = hips.at_volume(tau, volume, volume / 2, 2.26)
    assert (half < tau).all() and (half > tau / 2).all()        # a bent response falls by less than half
    assert np.allclose(hips.at_volume(np.array([3.0]), np.array([7.2]), 3.6, 2.26), 2.26)   # past the ceiling it stays there


def test_bands_cover_every_share_once():
    labels = hips.band_of(pd.Series([0.0, 0.049, 0.05, 0.19, 0.2, 0.3, 0.99]))
    assert list(labels) == ["under 5 %", "under 5 %", "5 to 10 %", "10 to 20 %", "20 to 30 %", "over 30 %", "over 30 %"]


def test_load_reads_the_committed_spartan_table():
    top = hips.ceiling()
    f = hips.load(top)
    assert len(f) == SUMMARY["counts"]["filters"] and f["code"].nunique() == SUMMARY["counts"]["sites"]
    assert np.allclose(f["tau"], f["fabs"] * f["volume"] / 353.0)
    sites = hips.stage_sites(f, top)["sites"]
    assert sites.iloc[0]["code"] == "ETAD"                      # Addis has the darkest median filter
    saved = pd.DataFrame(SUMMARY["sites"]).set_index("code")
    assert np.allclose(sites.set_index("code")["under_read"], saved["under_read"].reindex(sites["code"]).to_numpy(), atol=1e-6)


def test_summary_uses_no_aethalometer_and_keeps_its_own_arithmetic():
    assert SUMMARY["aethalometer_used"] is False
    assert not any("aeth" in json.dumps(SUMMARY[key]).lower() for key in SUMMARY if key not in ("aethalometer_used",))
    bands = pd.DataFrame(SUMMARY["bands"])
    assert bands["filters"].sum() == SUMMARY["counts"]["filters"]
    assert abs(bands["share_of_network"].sum() - 1.0) < 1e-9
    assert (bands["plus_minus"].diff().dropna() > 0).all()      # the correction is less certain on darker filters
    replay = pd.DataFrame(SUMMARY["volume_replay"])
    assert (replay["at_1.8"] < replay["at_3.6"]).all() and (replay["at_3.6"] < replay["at_7.2"]).all()
