"""The HIPS loading procedure (research/ftir_hips_chem/scripts/hips_loading.py): screen, identify, propagate, check."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research" / "ftir_hips_chem" / "scripts"), str(ROOT / "research" / "ftir_hips_chem" / "workflows")]

import hips_loading as hl  # noqa: E402
import hips_screen  # noqa: E402


def site(ceiling, *, n=400, level=30.0, volume=7.2, seed=1):
    """A made-up site: absorption in the air, a reference and an instrument that follow it, and HIPS read through a bend."""
    rng = np.random.default_rng(seed)
    air = level * rng.lognormal(0.0, 0.6, n)                                   # Mm-1 carried by the air
    loading = air * volume / 353.0
    observed = loading / (1.0 + loading / ceiling) if np.isfinite(ceiling) else loading
    dates = pd.date_range("2024-01-01", periods=n, freq="2D")
    return pd.DataFrame({"fabs": observed * 353.0 / volume * rng.lognormal(0.0, 0.03, n), "volume": volume,
                         "ftir_ec": air / 10.0 * rng.lognormal(0.0, 0.15, n), "pm25": air * 1.2 * rng.lognormal(0.0, 0.25, n),
                         "month": dates.month, "truth": air})


def test_the_ceiling_is_the_one_in_the_committed_improve_summary():
    rows = pd.DataFrame(json.loads((ROOT / "docs" / "addis-reference-closure-2026-11-10.json").read_text())["improve_ceiling"]).set_index("population")
    main = rows.loc["all sites"]
    assert hl.IMPROVE_CEILING.central == pytest.approx(main["tau_inf"], abs=1e-4)
    assert (hl.IMPROVE_CEILING.lo, hl.IMPROVE_CEILING.hi) == pytest.approx((main["tau_inf_lo"], main["tau_inf_hi"]), abs=1e-4)
    assert hl.IMPROVE_CEILING.reach == pytest.approx(rows.loc["six urban IMPROVE sites", "max_stratum_tau"], abs=1e-3)


def test_screen_gives_each_filter_a_status_and_a_corrected_value_with_its_range():
    top = hl.Ceiling(2.0, 1.6, 2.4, 1.0)
    frame = pd.DataFrame({"fabs": [10.0, 20.0, 30.0, 40.0, 90.0], "volume": 7.06})       # tau 0.2, 0.4, 0.6, 0.8, 1.8
    out = hl.screen(frame, ceiling=top)
    assert np.allclose(out["tau"], [0.2, 0.4, 0.6, 0.8, 1.8])
    assert np.allclose(out["under_read"], [0.1, 0.2, 0.3, 0.4, 0.9])
    assert list(out["status"]) == ["use, linearised", "use, linearised", "caution", "shape only", "shape only"]
    assert list(out["beyond_reference"]) == [False, False, False, False, True]
    assert np.allclose(out["fabs_linearised"], frame["fabs"] / (1 - out["under_read"]))
    assert (out["fabs_linearised_lo"] <= out["fabs_linearised"]).all() and (out["fabs_linearised"] <= out["fabs_linearised_hi"]).all()
    assert np.isinf(out["fabs_linearised_hi"].iloc[-1])                                   # past the low end of the ceiling: no finite value
    same = hl.screen(frame.assign(lab_tau=out["tau"]), tau="lab_tau", ceiling=top)
    assert np.allclose(same["fabs_linearised"], out["fabs_linearised"])


def test_a_site_bent_as_the_curve_says_is_seen_traced_and_corrected():
    top = hl.Ceiling(2.26, 1.82, 2.70, 1.18)
    screened = hl.screen(site(2.26), ceiling=top)
    found = hl.identify(screened, n_boot=200, ceiling=top)
    assert found["reading"] == "expected and seen" and found["exponent_hi"] < 1 and found["ratio_hi"] < 1
    assert found["slope"] > 1 and found["intercept"] < 0                                  # the signature a bent HIPS leaves in a straight-line fit
    traced = hl.propagate(screened).set_index("key")
    assert traced.loc["median_bc", "linearised"] == pytest.approx(screened["truth"].median() / 10.0, rel=0.03)
    assert traced.loc["median_bc", "reported"] < 0.8 * traced.loc["median_bc", "linearised"]
    assert abs(traced.loc["intercept", "linearised"]) < abs(traced.loc["intercept", "reported"])
    checked = hl.check(screened, n_boot=200)
    assert checked["verdict"] == "corrected"
    assert checked["exponent_before"] / checked["exponent_predicted"] == pytest.approx(checked["exponent_after"], rel=1e-6)


def test_a_site_that_bends_more_or_less_than_the_curve_is_called_out():
    top = hl.Ceiling(2.26, 1.82, 2.70, 1.18)
    assert hl.check(hl.screen(site(1.0), ceiling=top), n_boot=200)["verdict"] == "under-corrected"
    assert hl.check(hl.screen(site(np.inf), ceiling=top), n_boot=200)["verdict"] == "over-corrected"


def test_a_light_site_shows_nothing_and_a_small_one_cannot_be_tested():
    top = hl.Ceiling(2.26, 1.82, 2.70, 1.18)
    light = hl.identify(hl.screen(site(2.26, level=3.0), ceiling=top), n_boot=200, ceiling=top)
    assert light["reading"] == "not expected, not seen" and light["share_use"] == 1.0
    few = hl.screen(site(2.26, n=20), ceiling=top)
    assert hl.identify(few, n_boot=50, ceiling=top)["reading"] == "expected, cannot be tested"
    assert hl.check(few, n_boot=50)["verdict"] == "cannot be tested"


def test_less_air_through_the_filter_reads_closer_to_the_truth():
    tau, volume = np.array([0.1, 0.5, 1.2]), np.array([7.2, 7.2, 3.6])
    assert np.allclose(hl.at_volume(tau, volume, volume, 2.26), tau)
    half = hl.at_volume(tau, volume, volume / 2, 2.26)
    assert (half < tau).all() and (half > tau / 2).all()


def test_the_command_screens_a_plain_file(tmp_path, capsys):
    frame = site(2.26, n=120).rename(columns={"fabs": "Fabs", "volume": "Vol", "ftir_ec": "EC", "pm25": "PM"})
    frame["Day"] = pd.date_range("2024-01-01", periods=len(frame), freq="3D").astype(str)
    frame.to_csv(tmp_path / "batch.csv", index=False)
    out = tmp_path / "out"
    code = hips_screen.main(["--input", str(tmp_path / "batch.csv"), "--fabs", "Fabs", "--volume", "Vol", "--reference", "EC", "--instrument", "PM",
                             "--date", "Day", "--out", str(out), "--boot", "100"])
    assert code == 0 and "1 identify" in capsys.readouterr().out
    filters, sites = pd.read_csv(out / "filters.csv"), pd.read_csv(out / "sites.csv")
    assert len(filters) == 120 and {"tau", "status", "fabs_linearised", "fabs_linearised_lo", "fabs_linearised_hi"} <= set(filters)
    assert sites.loc[0, "reading"] == "expected and seen" and sites.loc[0, "check_verdict"] == "corrected"
    with pytest.raises(SystemExit):
        hips_screen.main(["--input", str(tmp_path / "batch.csv"), "--fabs", "Missing", "--volume", "Vol"])
