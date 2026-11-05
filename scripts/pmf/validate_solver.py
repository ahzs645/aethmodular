#!/usr/bin/env python3
"""Independent convex-reference, gradient, boundary and controlled-recovery checks."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import scipy
from scipy.optimize import minimize, nnls
from scipy.special import huber
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pmf.solver import (_g_block, _h_block, fit, normalize_factors,
    objective_from_residual, project_contributions, rho, robust_weights_from_residual,
    diagnostics, validate_arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "validation" / "solver_validation.json")
    parser.add_argument("--benchmark", type=Path, help="Optional real input NPZ for one 187-filter performance check")
    args = parser.parse_args()
    rng = np.random.default_rng(391)
    checks = {}
    x = rng.normal(1, 1, (20, 35)); u = rng.uniform(.5, 2, x.shape); w = 1 / u ** 2
    h = rng.uniform(.1, 2, (3, 35)); g = rng.uniform(.1, 2, (20, 3))
    with threadpool_limits(limits=1):
        got_g = _g_block(x, w, h, g.copy(), 500)
        ref_g = np.vstack([nnls((h / u[i]).T, x[i] / u[i])[0] for i in range(len(x))])
        got_h = _h_block(x, w, g, h.copy(), 500)
        ref_h = np.column_stack([nnls(g / u[:, j, None], x[:, j] / u[:, j])[0] for j in range(x.shape[1])])
        checks["G_block_vs_scipy_nnls_max_error"] = float(np.max(abs(got_g - ref_g)))
        checks["H_block_vs_scipy_nnls_max_error"] = float(np.max(abs(got_h - ref_h)))
        assert checks["G_block_vs_scipy_nnls_max_error"] < 1e-9
        assert checks["H_block_vs_scipy_nnls_max_error"] < 1e-9

        rtest = np.r_[np.linspace(-20, 20, 401), -4, 4, 0]
        checks["twice_huber_vs_scipy_special_max_error"] = float(np.max(abs(rho(rtest, "huber", 4) - 2 * huber(4, rtest))))
        assert checks["twice_huber_vs_scipy_special_max_error"] < 1e-12
        # Weighted-gradient reference is derived independently of the block code.
        for loss in ("quadratic", "huber"):
            signed = (g @ h - x) / u
            derivative = 2 * (signed if loss == "quadratic" else np.clip(signed, -4, 4)) / u
            dg, dh = derivative @ h.T, g.T @ derivative
            eps = 1e-5
            gp, gm = g.copy(), g.copy(); gp[3, 1] += eps; gm[3, 1] -= eps
            hp, hm = h.copy(), h.copy(); hp[1, 7] += eps; hm[1, 7] -= eps
            ng = (objective_from_residual((x - gp @ h) / u, loss) - objective_from_residual((x - gm @ h) / u, loss)) / (2 * eps)
            nh = (objective_from_residual((x - g @ hp) / u, loss) - objective_from_residual((x - g @ hm) / u, loss)) / (2 * eps)
            checks[f"{loss}_G_gradient_error"] = abs(float(ng - dg[3, 1]))
            checks[f"{loss}_H_gradient_error"] = abs(float(nh - dh[1, 7]))
            assert abs(ng - dg[3, 1]) < 1e-5 and abs(nh - dh[1, 7]) < 1e-5

        # Exact scalar check prevents the displayed EPA-style Q from becoming
        # confused with the distinct objective used to optimize robust fits.
        d = diagnostics(np.array([[-5., -4., 0., 4., 5.]]), np.ones((1, 5)), np.zeros((1, 1)), np.ones((1, 5)), "huber")
        assert d["Q_true"] == 82 and d["Q_robust_excluded_4"] == 32 and d["Q_huber"] == 80
        checks["Q_definitions_scalar_fixture"] = {k: d[k] for k in ("Q_true", "Q_robust_excluded_4", "Q_huber")}

        # The Huber IRLS quadratic majorizer touches the current objective and
        # bounds every tested new residual; this underlies monotone iteration.
        r0, r1 = rng.normal(0, 8, (20, 35)), rng.normal(0, 8, (20, 35))
        rw = robust_weights_from_residual(r0, "huber", 4)
        upper = rho(r0, "huber", 4) + rw * (r1 ** 2 - r0 ** 2)
        violation = float(np.max(rho(r1, "huber", 4) - upper))
        assert violation < 1e-10
        checks["huber_majorizer_max_violation"] = max(0., violation)

        # Check the robust fixed-profile convex problem against a completely
        # separate bounded optimizer and analytic gradient for every sample.
        px = x.copy(); px[1, 2] += 30; px[5, 4] -= 25
        projected, projection_info = project_contributions(px, u, h, loss="huber", max_iter=3000, tol=1e-12, return_info=True)
        reference = []
        for i in range(len(px)):
            design, target = (h / u[i]).T, px[i] / u[i]
            def fun(v):
                rr = design @ v - target
                return float(np.sum(2 * huber(4, rr))), 2 * design.T @ np.clip(rr, -4, 4)
            result = minimize(fun, ref_g[i], method="L-BFGS-B", jac=True, bounds=[(0, None)] * len(h), options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000, "maxls": 50})
            assert np.isfinite(result.fun)
            reference.append(result.x)
        reference = np.asarray(reference)
        checks["huber_fixed_H_vs_independent_LBFGSB_max_error"] = float(np.max(abs(projected - reference)))
        assert checks["huber_fixed_H_vs_independent_LBFGSB_max_error"] < 2e-5
        assert projection_info["objective_plateau"]
        checks["quadratic_fixed_H_vs_scipy_nnls_max_error"] = float(np.max(abs(project_contributions(x, u, h) - ref_g)))
        assert checks["quadratic_fixed_H_vs_scipy_nnls_max_error"] < 1e-10

        rs, cs = rng.uniform(.1, 3, len(x)), rng.uniform(.01, 5, x.shape[1])
        z1 = (x - g @ h) / u
        z2 = (x / rs[:, None] / cs - (g / rs[:, None]) @ (h / cs)) / (u / rs[:, None] / cs)
        checks["positive_diagonal_scaling_residual_max_error"] = float(np.max(abs(z1 - z2)))
        assert checks["positive_diagonal_scaling_residual_max_error"] < 1e-12
        gn, hn, order = normalize_factors(g, h)
        checks["normalization_product_max_error"] = float(np.max(abs(g @ h - gn @ hn)))
        assert checks["normalization_product_max_error"] < 1e-12
        assert np.allclose(hn.sum(axis=1), 1)

        # Known-rank reconstruction is a product check, not a unique-source test.
        true_g = rng.gamma(1.5, 1, (40, 3)); true_h = rng.gamma(1.5, 1, (3, 30))
        true_x = true_g @ true_h; unit_u = np.full_like(true_x, .3)
        clean = [fit(true_x, unit_u, 3, seed, loss="quadratic", max_iter=2500, tol=1e-9) for seed in (1, 10, 100)]
        best = min(clean, key=lambda z: z["metrics"]["objective"])
        checks["noiseless_rank3_relative_product_RMSE"] = float(np.linalg.norm(true_x - best["G"] @ best["H"]) / np.linalg.norm(true_x))
        assert checks["noiseless_rank3_relative_product_RMSE"] < 1e-3

        signed = true_x + rng.normal(0, 1, true_x.shape)
        signed[:5, :8] = -rng.uniform(.1, 2, (5, 8))
        original = signed.copy()
        for loss in ("quadratic", "huber"):
            result = fit(signed, np.ones_like(signed), 3, 100, loss=loss, max_iter=1600, tol=1e-8)
            assert np.array_equal(signed, original)
            assert (result["G"] >= 0).all() and (result["H"] >= 0).all()
            assert np.isfinite(result["history"]).all()
            assert (np.diff(result["history"][:, 1]) <= 1e-9 * np.maximum(1, result["history"][:-1, 1])).all()
            assert result["metrics"]["objective"] >= result["metrics"]["negative_observation_objective_lower_bound"] - 1e-8
            checks[f"{loss}_signed_fixture"] = {"input_unchanged": True, "nonnegative_factors": True, "objective_monotone": True, "converged": result["metrics"]["converged"], "objective": result["metrics"]["objective"]}

        for label, bx in (("zero", np.zeros((6, 8))), ("negative", -np.arange(1, 49, dtype=float).reshape(6, 8) / 10)):
            for loss in ("quadratic", "huber"):
                result = fit(bx, np.ones_like(bx), 2, 1, loss=loss, max_iter=100)
                assert np.array_equal(result["G"] @ result["H"], np.zeros_like(bx))
                assert result["metrics"]["converged"]
                assert np.isclose(result["metrics"]["objective"], result["metrics"]["negative_observation_objective_lower_bound"])
        checks["zero_and_all_negative_exact_boundaries"] = "passed"

        rejected = 0
        for invalid in (np.zeros_like(u), -u, np.full_like(u, np.nan), np.full_like(u, np.inf)):
            try: validate_arrays(x, invalid)
            except ValueError: rejected += 1
        assert rejected == 4
        checks["invalid_uncertainty_cases_rejected"] = rejected
        try: fit(x, u, 3, initial_G=np.zeros((20, 3)), initial_H=np.zeros((3, 35)))
        except ValueError: checks["dead_zero_initialization_rejected"] = True
        else: raise AssertionError("Dead zero initial factors accepted")

        # Fixed seed must reproduce numerical factors and histories exactly.
        a = fit(x, u, 3, 2026, loss="huber", max_iter=350)
        b = fit(x, u, 3, 2026, loss="huber", max_iter=350)
        assert all(np.array_equal(a[key], b[key]) for key in ("G", "H", "history"))
        checks["same_seed_factor_and_history_reproducibility"] = True

        # Contamination is a controlled simulation, not a real-data validation.
        rng2 = np.random.default_rng(9382)
        tg = rng2.gamma(2, 1, (60, 3)); th = rng2.gamma(1.5, 1, (3, 28))
        truth = tg @ th
        sim_u = np.full_like(truth, .2)
        contaminated = truth + rng2.normal(0, .2, truth.shape)
        selected = rng2.choice(truth.size, size=50, replace=False)
        contaminated.flat[selected] += rng2.choice([-1, 1], size=len(selected)) * rng2.uniform(20, 35, size=len(selected))
        recovery = {}
        for loss in ("quadratic", "huber"):
            candidates = [fit(contaminated, sim_u, 3, seed, loss=loss, max_iter=2500, tol=1e-8) for seed in (1, 10, 100)]
            winner = min(candidates, key=lambda z: z["metrics"]["objective"])
            recovery[loss] = float(np.sqrt(np.mean((winner["G"] @ winner["H"] - truth) ** 2)))
        assert recovery["huber"] < .5 * recovery["quadratic"], recovery
        checks["controlled_outlier_clean_product_RMSE"] = recovery

    report = {"status": "passed", "numpy": np.__version__, "scipy": scipy.__version__, "checks": checks,
              "scope": "Numerical correctness, convex references and controlled recovery; not native ME-2 parity or real-source validation.",
              "source_sha256": hashlib.sha256((Path(__file__).resolve().parents[2] / "src/pmf/solver.py").read_bytes()).hexdigest()}
    if args.benchmark:
        with np.load(args.benchmark, allow_pickle=False) as archive:
            benchmark = fit(archive["X"], archive["U"], 5, 1, loss="huber", max_iter=3000, tol=1e-7)
        report["single_real_case_benchmark"] = {"input": str(args.benchmark), "metrics": benchmark["metrics"], "provenance": benchmark["provenance"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
