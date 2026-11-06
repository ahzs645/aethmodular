#!/usr/bin/env python3
"""Independent numerical checks for resampling, gauge constraints and rotations."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.optimize import minimize, brentq
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pmf.uncertainty import (weighted_simplex_projection, temporal_segments,
    moving_block_indices, match_profiles, constrained_refit,
    exact_pair_rotation_envelope, profile_displacement_search, run_bootstrap)
from src.pmf.solver import fit, objective_from_residual


def main():
    rng = np.random.default_rng(24519)
    report = {}
    errors = []
    for _ in range(20):
        v = rng.normal(size=8)
        d = np.exp(rng.normal(size=8))
        total = rng.uniform(.1, 3)
        projected = weighted_simplex_projection(v, d, total)
        opt = minimize(lambda h: .5 * np.sum(d * (h - v) ** 2), np.ones(8) * total / 8,
                       jac=lambda h: d * (h - v), method="SLSQP", bounds=[(0, None)] * 8,
                       constraints={"type": "eq", "fun": lambda h: h.sum() - total,
                                    "jac": lambda h: np.ones(8)},
                       options={"ftol": 1e-12, "maxiter": 200})
        assert opt.success, opt.message
        errors.append(abs(.5 * np.sum(d * (projected - v) ** 2) - opt.fun))
        assert abs(projected.sum() - total) < 1e-12 and projected.min() >= 0
        active = projected > 1e-10
        gradient = d * (projected - v)
        multiplier = -np.mean(gradient[active])
        assert np.max(abs(gradient[active] + multiplier)) < 1e-10
        assert np.min(gradient[~active] + multiplier, initial=0) > -1e-10
    assert max(errors) < 1e-9
    report["weighted_simplex_max_objective_difference_vs_scipy_slsqp"] = max(errors)
    free = weighted_simplex_projection([.2, .4, -1], [1, 0, 0], 1)
    assert np.allclose(free, [.2, .8, 0])
    constrained = weighted_simplex_projection([2, 1, 0], [1, 0, 0], .4)
    assert np.allclose(constrained, [.4, 0, 0])
    report["zero_curvature_free_coordinates"] = True

    dates = np.asarray(["2023-01-01", "2023-01-04", "2023-01-07", "2023-01-10",
                        "2023-03-01", "2023-03-04", "2023-03-07", "2023-03-10"])
    lots = np.asarray([1, 1, 2, 2, 2, 2, 2, 2])
    segments = temporal_segments(dates, lots, 14)
    assert [s.tolist() for s in segments] == [[0, 1], [2, 3], [4, 5, 6, 7]]
    index, blocks = moving_block_indices(8, np.random.default_rng(17), 3, segments)
    index2, blocks2 = moving_block_indices(8, np.random.default_rng(17), 3, segments)
    assert np.array_equal(index, index2) and blocks == blocks2
    for block in blocks:
        assert set(block["source_indices"]).issubset(set(segments[block["segment"]]))
    offset = 0
    for segment in segments:
        assert set(index[offset:offset + len(segment)]).issubset(set(segment))
        offset += len(segment)
    report["segmented_bootstrap_boundaries_counts_and_reproducibility"] = True
    h = rng.uniform(.2, 2, (3, 9))
    mapping = match_profiles(h, h[[2, 0, 1]], np.geomspace(.01, 10, 9))
    assert mapping["candidate_index"].tolist() == [1, 2, 0]
    assert mapping["accepted"].all() and np.allclose(mapping["cosine"], 1)
    collapsed = h.copy(); collapsed[0] = 0
    assert not match_profiles(h, collapsed)["accepted"].all()
    report["factor_permutation_and_collapsed_rejection"] = True

    # With one factor and two channels, fixing the normalized first channel
    # fixes the entire H row, and every G has a closed-form NNLS optimum.
    gt = np.asarray([[2.0], [3.0], [5.0], [.2]])
    ht = np.asarray([[.3, .7]])
    x = gt @ ht
    u = np.asarray([[.1, .2], [.15, .1], [.2, .3], [.1, .1]])
    target = .6
    hc = np.asarray([[target, 1 - target]])
    expected_g = np.maximum(0, np.sum(x * hc / u**2, axis=1) / np.sum(hc**2 / u**2, axis=1))[:, None]
    expected_q = float(np.sum(((x - expected_g @ hc) / u) ** 2))
    with threadpool_limits(limits=1):
        result = constrained_refit(x, u, gt, ht, factor=0, band_mask=[True, False], target=target,
                                   max_iter=300, tol=1e-12)
    assert np.allclose(result["G"], expected_g, atol=1e-12, rtol=1e-12)
    assert np.array_equal(result["H"], hc)
    assert abs(result["metrics"]["objective"] - expected_q) < 1e-9 and expected_q > 0
    assert np.all(np.diff(result["history"][:, 1]) <= 1e-9)
    report["constrained_rank1_objective_error_vs_analytic"] = abs(result["metrics"]["objective"] - expected_q)
    report["gauge_constraint_prevents_trivial_scaling_escape"] = True

    # Huber constrained fit versus an independent small SLSQP implementation.
    g0 = np.asarray([[1], [2], [3], [4], [2.5], [1.5]], float)
    h0 = np.asarray([[.2, .1, .4, .3]])
    xr = g0 @ h0 + rng.normal(0, .01, (6, 4)); xr[1, 2] += 4
    ur = np.full_like(xr, .1)
    def robust_objective(v):
        gr, hr = v[:6, None], v[6:][None, :]
        return objective_from_residual((xr - gr @ hr) / ur, "huber", 4)
    def robust_gradient(v):
        gr, hr = v[:6, None], v[6:][None, :]
        r = (gr @ hr - xr) / ur
        deriv = 2 * np.clip(r, -4, 4) / ur
        return np.r_[np.sum(deriv * hr, axis=1), np.sum(deriv * gr, axis=0)]
    with threadpool_limits(limits=1):
        rr = constrained_refit(xr, ur, g0, h0, factor=0, band_mask=[1, 1, 0, 0], target=.4,
                              loss="huber", delta=4, max_iter=2000, tol=1e-12)
    v0 = np.r_[rr["G"].ravel(), rr["H"].ravel()]
    opt = minimize(robust_objective, v0, jac=robust_gradient, method="SLSQP", bounds=[(0, None)] * 10,
                   constraints=[{"type": "eq", "fun": lambda v: v[6:].sum() - 1},
                                {"type": "eq", "fun": lambda v: v[6:8].sum() - .4}],
                   options={"ftol": 1e-11, "maxiter": 1000})
    assert opt.success, opt.message
    robust_difference = rr["metrics"]["objective"] - opt.fun
    assert abs(robust_difference) < 1e-6
    assert np.all(np.diff(rr["history"][:, 1]) <= 1e-8)
    report["constrained_huber_objective_difference_vs_slsqp"] = robust_difference
    report["huber_target_error"] = rr["metrics"]["target_band_error"]

    gr = rng.uniform(.1, 2, (7, 3)); hr = rng.uniform(.2, 3, (3, 9))
    rotations = exact_pair_rotation_envelope(gr, hr)
    max_product_error = max(r["product_max_relative_error"] for r in rotations["records"])
    assert max_product_error < 1e-12
    assert np.max(rotations["profile_max"] - rotations["profile_min"]) > .01
    assert np.max(abs(rotations["profiles"].sum(axis=2) - 1)) < 1e-12
    report["exact_pair_shear_max_relative_product_error"] = max_product_error
    report["feasible_rotations_produce_distinct_unit_sum_profiles"] = True

    with threadpool_limits(limits=1):
        search = profile_displacement_search(x, u, gt, ht, factor=0, band_mask=[1, 0], thresholds=[1, 4],
                                             max_iter=100, tol=1e-12, max_steps=12, relative_target_tol=1e-5)
    assert not search["any_objective_decrease"] and not search["any_factor_assignment_changed"]
    assert not search["any_iteration_cap"]
    assert all(p["status"] == "local_threshold_bracket" for p in search["intervals"])
    assert all(p["inside_delta_loss"] <= p["threshold_delta_loss"] + 1e-7 and
               p["outside_delta_loss"] >= p["threshold_delta_loss"] - 1e-7 for p in search["intervals"])
    report["profile_threshold_brackets_verified_against_analytic_rank1"] = True

    # Small end-to-end run verifies checkpoint identity and explicit rejection
    # of changed bootstrap settings; this is not a real-filter bootstrap.
    with tempfile.TemporaryDirectory(prefix="pmf_uncertainty_fixture_") as temp:
        p = Path(temp)
        gx = rng.uniform(.3, 1.5, (12, 2)); hx = rng.uniform(.2, 1, (2, 8))
        xx = gx @ hx; uu = np.full_like(xx, .1)
        d = np.arange(np.datetime64("2023-01-01"), np.datetime64("2023-02-06"), np.timedelta64(3, "D"))
        ids = np.asarray([f"fixture_{i}" for i in range(12)])
        np.savez_compressed(p / "input.npz", X=xx, U=uu, sample_ids=ids, wavenumbers=np.arange(8)[::-1], dates=d.astype(str), lot_ids=np.ones(12))
        np.savez_compressed(p / "reference.npz", G=gx, H=hx, sample_ids=ids, wavenumbers=np.arange(8)[::-1])
        with contextlib.redirect_stdout(io.StringIO()):
            run_bootstrap(p / "input.npz", p / "reference.npz", p / "out", repeats=2, starts=1, block_length=3,
                          max_iter=300, loss="quadratic", tol=1e-8)
            first = (p / "out/resample_0000.npz").read_bytes()
            with np.load(p / "out/ensemble.npz", allow_pickle=False) as ensemble:
                assert np.array_equal(ensemble["eligible"], ensemble["accepted"] & ensemble["converged"][:, None])
                for factor in range(2):
                    if not ensemble["eligible"][:, factor].any():
                        assert np.isnan(ensemble["profile_percentiles"][:, factor]).all()
            run_bootstrap(p / "input.npz", p / "reference.npz", p / "out", repeats=2, starts=1, block_length=3,
                          max_iter=300, loss="quadratic", tol=1e-8)
            assert first == (p / "out/resample_0000.npz").read_bytes()
            try:
                run_bootstrap(p / "input.npz", p / "reference.npz", p / "out", repeats=3, starts=1, block_length=3,
                              max_iter=300, loss="quadratic", tol=1e-8)
            except ValueError:
                pass
            else:
                raise AssertionError("Changed bootstrap settings were accepted")
    report["bootstrap_checkpoint_reuse_and_changed_setting_rejection"] = True
    report["bootstrap_percentiles_require_convergence_and_mapping"] = True
    report["native_EPA_parity_validated"] = False
    destination = Path(__file__).resolve().parent / "uncertainty_validation.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
