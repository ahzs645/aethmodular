"""Scientific parity checks for the isolated, optional batched PB MAP solver."""

from pathlib import Path
import sys
import unittest

import numpy as np
try:
    import cvxpy as cp
except ImportError:
    cp = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"))
from vibes_map_batch import solve_pinball_map_batch


def reference(y, mean, W, tau, sigma):
    x = cp.Variable(W.shape[1])
    r = y - mean - W @ x
    problem = cp.Problem(cp.Minimize(cp.sum(0.5 * cp.abs(r) + (tau - 0.5) * r)
                                     + 0.5 * sigma * cp.sum_squares(x)))
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                  tol_feas=1e-10, max_iter=500)
    if problem.status != cp.OPTIMAL:
        raise AssertionError(problem.status)
    return mean + W @ x.value, problem.value


class TestBatchedPinballMAP(unittest.TestCase):
    def test_scalar_problem_matches_closed_form(self):
        # For W=1, x*=clip(y-mean, (tau-1)/sigma, tau/sigma).
        # This reference needs neither CVXPY nor another iterative algorithm.
        y = np.array([[-4.0], [-0.3], [0.0], [0.03], [5.0]])
        tau = np.array([0.1, 0.2, 0.1, 0.35, 0.4])
        sigma = np.array([2.0, 0.5, 0.3, 3.0, 4.0])
        result = solve_pinball_map_batch(y, np.zeros(1), np.ones((1, 1)),
                                        sigma=sigma, tau=tau, atol=1e-10, rtol=1e-10)
        self.assertTrue(result.converged.all(), result.diagnostics)
        expected = np.clip(y[:, 0], (tau - 1.0) / sigma, tau / sigma)
        np.testing.assert_allclose(result.latent[:, 0], expected, atol=2e-9)
        self.assertGreater(len(np.unique(result.iterations)), 1)

    def fixture(self):
        rng = np.random.default_rng(84)
        t = np.linspace(0, 1, 140)
        Q, _ = np.linalg.qr(np.stack([np.ones_like(t), t, t**2, np.sin(2 * np.pi * t)], axis=1))
        W = Q * np.array([0.5, 0.2, 0.08, 0.03])
        mean = 0.1 + 0.05 * t
        narrow = 0.03 * np.exp(-0.5 * ((t - 0.4) / 0.025)**2)
        broad = 0.015 * np.exp(-0.5 * ((t - 0.55) / 0.23)**2)
        signals = np.stack([narrow, broad, broad + narrow, 0.04 * broad])
        y = mean + rng.normal(size=(4, 4)) @ W.T + signals
        y += rng.normal(scale=2e-5, size=y.shape)
        return y, mean, W

    @unittest.skipIf(cp is None, "Install the optional VIBES/CVXPY dependencies")
    def test_parity_with_clarabel_broad_and_weak_signals(self):
        y, mean, W = self.fixture()
        tau = np.array([0.05, 0.1, 0.3, 0.45])
        sigma = np.array([0.01, 0.03, 0.1, 0.2])
        result = solve_pinball_map_batch(y, mean, W, sigma=sigma, tau=tau,
                                        atol=2e-8, rtol=2e-8, maxiter=20000)
        self.assertTrue(result.converged.all(), result.diagnostics)
        for i in range(len(y)):
            baseline, objective = reference(y[i], mean, W, tau[i], sigma[i])
            np.testing.assert_allclose(result.baseline[i], baseline, atol=2e-5, rtol=1e-5)
            self.assertLess(abs(result.primal_objective[i] - objective), 2e-7)
        np.testing.assert_allclose(result.baseline + result.corrected, y, atol=2e-16)
        self.assertTrue(np.all(result.duality_gap <= result.gap_tolerance))
        self.assertEqual(result.baseline.dtype, np.float64)

    @unittest.skipIf(cp is None, "Install the optional VIBES/CVXPY dependencies")
    def test_nonorthogonal_and_rank_deficient_basis(self):
        y, mean, W = self.fixture()
        W = np.column_stack([W[:, 0] + W[:, 1], W[:, 1], W[:, 2], W[:, 2]])
        result = solve_pinball_map_batch(y[:2], mean, W, sigma=0.07,
                                        atol=2e-8, rtol=2e-8, maxiter=20000)
        self.assertTrue(result.converged.all(), result.diagnostics)
        for i in range(2):
            expected, objective = reference(y[i], mean, W, 0.1, 0.07)
            np.testing.assert_allclose(result.baseline[i], expected, atol=2e-5, rtol=1e-5)
            self.assertLess(abs(result.primal_objective[i] - objective), 2e-7)
        np.testing.assert_allclose(mean + result.latent @ W.T, result.baseline, atol=1e-13)

    def test_independent_rows_and_channel_permutation(self):
        y, mean, W = self.fixture()
        kwargs = dict(sigma=0.1, atol=1e-8, rtol=1e-8, maxiter=20000)
        batch = solve_pinball_map_batch(y, mean, W, **kwargs)
        singleton = solve_pinball_map_batch(y[1:2], mean, W, **kwargs)
        reverse = solve_pinball_map_batch(y[:, ::-1], mean[::-1], W[::-1], **kwargs)
        self.assertTrue(batch.converged.all(), batch.diagnostics)
        self.assertTrue(singleton.converged.all(), singleton.diagnostics)
        self.assertTrue(reverse.converged.all(), reverse.diagnostics)
        np.testing.assert_allclose(batch.baseline[1], singleton.baseline[0], atol=1e-10)
        np.testing.assert_allclose(batch.baseline, reverse.baseline[:, ::-1], atol=2e-7)

    def test_unconverged_output_is_nan_with_diagnostics(self):
        y, mean, W = self.fixture()
        result = solve_pinball_map_batch(y, mean, W, sigma=0.1, maxiter=1)
        self.assertFalse(result.converged.any())
        self.assertTrue(np.isnan(result.baseline).all())
        self.assertTrue(np.isnan(result.corrected).all())
        self.assertTrue(np.isnan(result.latent).all())
        self.assertTrue(np.isfinite(result.primal_objective).all())
        self.assertEqual(result.diagnostics[0]["status"], "max_iterations")

    @unittest.skipIf(cp is None, "Install the optional VIBES/CVXPY dependencies")
    def test_small_sigma_stress_has_valid_bounds_and_no_silent_acceptance(self):
        y, mean, W = self.fixture()
        sigma = np.array([1e-5, 3e-5, 1e-4, 1e-3])
        result = solve_pinball_map_batch(y, mean, W, sigma=sigma, maxiter=2000,
                                        atol=2e-8, rtol=2e-8)
        # An intentionally bounded run may fail; bounds must bracket the
        # independent optimum, and no uncertified spectrum may escape.
        for i in range(len(y)):
            baseline, optimum = reference(y[i], mean, W, 0.1, sigma[i])
            self.assertLessEqual(result.dual_objective[i], optimum + 2e-9)
            self.assertGreaterEqual(result.primal_objective[i], optimum - 2e-9)
            if result.converged[i]:
                self.assertLessEqual(result.duality_gap[i], result.gap_tolerance[i])
                np.testing.assert_allclose(result.baseline[i], baseline, atol=2e-5, rtol=1e-5)
            else:
                self.assertTrue(np.isnan(result.baseline[i]).all())
                self.assertTrue(np.isnan(result.corrected[i]).all())
                self.assertEqual(result.diagnostics[i]["status"], "max_iterations")

    def test_invalid_inputs(self):
        y, mean, W = self.fixture()
        for kwargs in ({"sigma": 0}, {"sigma": -0.1}, {"sigma": [1, 2]},
                       {"sigma": 0.1, "tau": 0}, {"sigma": 0.1, "rho": 0},
                       {"sigma": 0.1, "backend": "automatic"},
                       {"sigma": 0.1, "maxiter": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solve_pinball_map_batch(y, mean, W, **kwargs)
        with self.assertRaises(ValueError):
            solve_pinball_map_batch(y * np.nan, mean, W, sigma=0.1)


if __name__ == "__main__":
    unittest.main()
