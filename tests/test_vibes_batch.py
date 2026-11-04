"""Parity checks for research/ftir_hips_chem/scripts/vibes_batch.py (CPU backend)."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/vendor/pyvibes/src"))

import vibes_batch as vb  # noqa: E402

try:
    import cvxpy as cp
    from vibes.absorbance_estimators.vibes import vibes_pb_elbo, vibes_pb_jac_elbo, vibes_pb_optim_prep
except ImportError:  # pragma: no cover
    cp = None


def fixture(rows=5, p=240, c=6, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, p)
    Q, _ = np.linalg.qr(np.cos(np.pi * t[:, None] * np.arange(c)[None, :]))
    W = Q * np.geomspace(0.3, 0.01, c)
    mu = 0.1 + 0.05 * t
    sig = 0.02 * np.exp(-0.5 * ((t - 0.4) / 0.03) ** 2)[None, :] * rng.uniform(0.2, 2, (rows, 1))
    sig += 0.01 * np.exp(-0.5 * ((t - 0.6) / 0.2) ** 2)[None, :] * rng.uniform(0, 1, (rows, 1))
    Y = mu + rng.normal(size=(rows, c)) @ W.T + sig + rng.normal(scale=1e-4, size=(rows, p))
    return Y, mu, W


@unittest.skipIf(cp is None, "needs the vendored pyvibes + cvxpy environment")
class TestObjective(unittest.TestCase):
    def test_batched_elbo_and_gradient_match_upstream(self):
        Y, mu, W = fixture()
        rng = np.random.default_rng(0)
        c = W.shape[1]
        nu = rng.normal(size=(len(Y), c))
        d = rng.uniform(0.2, 2, (len(Y), c))
        be = vb._Backend("numpy")
        f, g, s = vb._elbo_and_grad(be, Y - mu, W, W * W, 0.1, nu, d)
        for i, y in enumerate(Y):
            self.assertAlmostEqual(-f[i], vibes_pb_elbo(y, 0.1, nu[i], d[i], mu, W), places=8)
            _, dnu, dd = vibes_pb_jac_elbo(y, 0.1, nu[i], d[i], mu, W)
            np.testing.assert_allclose(-g[i], np.r_[dnu, dd], rtol=1e-11, atol=1e-9)

    def test_fused_scipy_objective_matches_upstream(self):
        Y, mu, W = fixture()
        c = W.shape[1]
        fun, init, bounds, _ = vb.fused_pb_objective(Y[0], mu, W, 0.1)
        ref_f, ref_j, ref_init, ref_b = vibes_pb_optim_prep(Y[0], 0.1, np.zeros(c), np.ones(c), mu, W, 0.1, 0.1, 1e-10)
        x = ref_init + np.r_[0, np.linspace(-0.3, 0.3, c), np.linspace(-0.5, 0.5, c)]
        val, grad = fun(x)
        self.assertAlmostEqual(val, ref_f(x), places=8)
        np.testing.assert_allclose(grad, ref_j(x), rtol=1e-11, atol=1e-9)
        np.testing.assert_array_equal(init, ref_init)
        self.assertEqual(bounds, ref_b)


class TestLineSearch(unittest.TestCase):
    def test_vectorised_more_thuente_matches_scipy_dcsrch(self):
        from scipy.optimize._dcsrch import DCSRCH

        rng = np.random.default_rng(1)
        n = 300
        A, T = rng.uniform(0.1, 5, n), rng.uniform(0.01, 20, n)
        P, Q = rng.uniform(-2, 2, n), rng.uniform(0.5, 3, n)
        phi = lambda a, i: A[i] * (a - T[i]) ** 2 + P[i] * np.sin(Q[i] * a)
        dphi = lambda a, i: 2 * A[i] * (a - T[i]) + P[i] * Q[i] * np.cos(Q[i] * a)
        idx = np.flatnonzero(dphi(0.0, np.arange(n)) < 0)
        stp0 = rng.uniform(0.01, 3, len(idx))
        expected, evals = [], []
        for j, i in enumerate(idx):
            dc = DCSRCH(None, None, 1e-3, 0.9, 0.1, 0.0, 1e10)
            s, f, g, task = dc._iterate(stp0[j], phi(0.0, i), dphi(0.0, i), b"START")
            k = 0
            while task[:2] == b"FG":
                f, g = phi(s, i), dphi(s, i)
                k += 1
                s, f, g, task = dc._iterate(s, f, g, task)
            expected.append(s)
            evals.append(k)
        mt = vb._MoreThuente(stp0.copy(), phi(0.0, idx), dphi(0.0, idx), np.full(len(idx), 1e10))
        status, k = mt.status.copy(), np.zeros(len(idx), int)
        while (status == 0).any():
            j = np.flatnonzero(status == 0)
            k[j] += 1
            status[j] = mt.iterate(j, phi(mt.stp[j], idx[j]), dphi(mt.stp[j], idx[j]))
        np.testing.assert_array_equal(mt.stp, expected)
        np.testing.assert_array_equal(k, evals)


@unittest.skipIf(cp is None, "needs the vendored pyvibes + cvxpy environment")
class TestElboFit(unittest.TestCase):
    def test_batch_reaches_scipy_optimum_on_well_posed_rows(self):
        from scipy.optimize import minimize

        Y, mu, W = fixture()
        c = W.shape[1]
        e = vb.fit_elbo_batch(Y, mu, W, pool=3)  # pool < rows exercises slot refill
        self.assertTrue(e.success.all(), e.message)
        for i, y in enumerate(Y):
            f, j, init, b = vibes_pb_optim_prep(y, 0.1, np.zeros(c), np.ones(c), mu, W, 0.1, 0.1, 1e-10)
            o = minimize(f, init, method="L-BFGS-B", jac=j, bounds=b, options={"maxiter": 10000})
            self.assertLess(abs(-o.fun - e.elbo[i]), 1e-3 * max(1, abs(o.fun)))
            self.assertLess(abs(e.sigma[i] / vb.fused_pb_objective(y, mu, W, 0.1)[3](o.x) - 1), 1e-3)

    def test_polish_never_lowers_elbo(self):
        Y, mu, W = fixture(rows=4, seed=9)
        a = vb.fit_elbo_batch(Y, mu, W, pool=4)
        b = vb.fit_elbo_batch(Y, mu, W, pool=4, polish_restarts=3)
        self.assertTrue(np.all(b.elbo >= a.elbo - 1e-9))

    def test_rejects_estimated_tau(self):
        Y, mu, W = fixture(rows=1)
        with self.assertRaises(ValueError):
            vb.fit_elbo_batch(Y, mu, W, tau=None)


class TestMap(unittest.TestCase):
    def test_scalar_closed_form(self):
        y = np.array([[-4.0], [-0.3], [0.0], [0.03], [5.0]])
        tau = np.array([0.1, 0.2, 0.1, 0.35, 0.4])
        sigma = np.array([2.0, 0.5, 0.3, 3.0, 4.0])
        r = vb.solve_map_batch(y, np.zeros(1), np.ones((1, 1)), sigma=sigma, tau=tau)
        self.assertTrue(r.converged.all())
        np.testing.assert_allclose(r.latent[:, 0], np.clip(y[:, 0], (tau - 1) / sigma, tau / sigma), atol=1e-7)

    @unittest.skipIf(cp is None, "needs cvxpy")
    def test_matches_clarabel_including_tiny_sigma(self):
        Y, mu, W = fixture(rows=6)
        sigma = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2])
        r = vb.solve_map_batch(Y, mu, W, sigma=sigma, tau=0.1)
        self.assertTrue(r.converged.all(), r.duality_gap)
        for i, y in enumerate(Y):
            x = cp.Variable(W.shape[1])
            res = y - mu - W @ x
            prob = cp.Problem(cp.Minimize(cp.sum(0.5 * cp.abs(res) + (0.1 - 0.5) * res)
                                          + 0.5 * sigma[i] * cp.sum_squares(x)))
            prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertLess(r.primal_objective[i] - prob.value, 1e-7 * max(1, abs(prob.value)))
            np.testing.assert_allclose(r.baseline[i], mu + W @ x.value, atol=2e-6)

    def test_float32_system_matrix_still_certifies(self):
        # What a WGSL GEMM gives: float32 M as preconditioner, float64 operator and certificate.
        Y, mu, W = fixture(rows=6)
        sigma = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2])
        a = vb.solve_map_batch(Y, mu, W, sigma=sigma, tau=0.1)
        b = vb.solve_map_batch(Y, mu, W, sigma=sigma, tau=0.1, hess_dtype="float32")
        self.assertTrue(b.converged.all(), b.duality_gap)
        np.testing.assert_allclose(b.baseline, a.baseline, atol=2e-6)

    def test_unconverged_rows_are_masked(self):
        Y, mu, W = fixture(rows=3)
        r = vb.solve_map_batch(Y, mu, W, sigma=1e-5, maxiter=2)
        self.assertFalse(r.converged.any())
        self.assertTrue(np.isnan(r.baseline).all())

    def test_rejects_nonpositive_sigma(self):
        Y, mu, W = fixture(rows=1)
        with self.assertRaises(ValueError):
            vb.solve_map_batch(Y, mu, W, sigma=0.0)


class TestNewton(unittest.TestCase):
    def test_hessian_matches_finite_differences(self):
        Y, mu, W = fixture()
        c = W.shape[1]
        rng = np.random.default_rng(0)
        nu, d = rng.normal(size=(len(Y), c)), rng.uniform(0.2, 2, (len(Y), c))
        be = vb._Backend("numpy")
        K = vb._khatri_rao(be, W, np.float64)
        V = Y - mu
        f, g, A, u, _ = vb._elbo_newton_terms(be, V, W, W * W, K, 0.1, nu, d, np.float64)
        f0, g0, _ = vb._elbo_and_grad(be, V, W, W * W, 0.1, nu, d)
        np.testing.assert_allclose(f, f0, rtol=1e-14)
        np.testing.assert_allclose(g, g0, rtol=1e-10, atol=1e-10)
        H = A - u[:, :, None] * u[:, None, :]
        x, h = np.concatenate([nu, d], axis=1), 1e-6
        for k in range(0, 2 * c, 3):
            e = np.zeros(2 * c)
            e[k] = h
            gp = vb._elbo_and_grad(be, V, W, W * W, 0.1, (x + e)[:, :c], (x + e)[:, c:])[1]
            gm = vb._elbo_and_grad(be, V, W, W * W, 0.1, (x - e)[:, :c], (x - e)[:, c:])[1]
            np.testing.assert_allclose(H[:, :, k], (gp - gm) / (2 * h), rtol=1e-6, atol=1e-6 * np.abs(H).max())

    def test_reaches_at_least_the_polished_lbfgs_optimum(self):
        Y, mu, W = fixture(rows=6, seed=9)
        lb = vb.fit_elbo_batch(Y, mu, W, pool=6, polish_restarts=5)
        for hd in ("float64", "float32"):
            r = vb.fit_elbo_newton(Y, mu, W, chunk=4, hess_dtype=hd)  # chunk < rows
            self.assertTrue(r.success.all(), r.message)
            self.assertTrue(np.all(r.elbo >= lb.elbo - 1e-9))
            self.assertLess(r.iterations.max(), 30)
            np.testing.assert_allclose(r.sigma, lb.sigma, rtol=1e-3)

    def test_external_float32_gemm_reaches_the_same_optimum(self):
        Y, mu, W = fixture(rows=4, seed=9)
        be = vb._Backend("numpy")
        K = vb._khatri_rao(be, W, np.float32)
        calls = []

        def gemm(name, w):
            calls.append(name)
            return np.asarray(w, np.float32) @ K[name]

        r = vb.fit_elbo_newton(Y, mu, W, gemm=gemm)
        ref = vb.fit_elbo_newton(Y, mu, W)
        self.assertTrue(r.success.all(), r.message)
        self.assertEqual(set(calls), {"WW", "WW2", "W2W2"})
        np.testing.assert_allclose(r.elbo, ref.elbo, atol=1e-8)
        np.testing.assert_allclose(r.sigma, ref.sigma, rtol=1e-6)

    def test_rejects_estimated_tau(self):
        Y, mu, W = fixture(rows=1)
        with self.assertRaises(ValueError):
            vb.fit_elbo_newton(Y, mu, W, tau=None)


try:
    import vibes_jax as vj  # noqa: E402
except ImportError:  # pragma: no cover
    vj = None


@unittest.skipIf(vj is None, "needs jax")
class TestJax(unittest.TestCase):
    def test_compiled_lbfgs_follows_the_numpy_batch(self):
        Y, mu, W = fixture(rows=5, seed=3)
        for it in (1, 5):
            a = vb.fit_elbo_batch(Y, mu, W, pool=5, maxiter=it, retry_failed=False)
            b = vj.fit_elbo_jax(Y, mu, W, pool=4, chunk_iters=3, min_pool=2, maxiter=it, retry_failed=False)
            np.testing.assert_array_equal(a.evaluations, b.evaluations)
            np.testing.assert_allclose(np.c_[a.nu, a.d], np.c_[b.nu, b.d], atol=1e-9)

    def _scipy(self, y, mu, W, k):
        from scipy.optimize import minimize

        fun, init, bounds, _ = vb.fused_pb_objective(y, mu, W, 0.1)
        return minimize(fun, init, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": k})

    def test_lbfgsb_port_follows_scipy_iterate_by_iterate(self):
        Y, mu, W = fixture(rows=3, seed=3)
        for k in (1, 2, 5, 12):
            e = vj.fit_elbo_jax(Y, mu, W, pool=3, chunk_iters=4, min_pool=2, maxiter=k, retry_failed=False, engine="lbfgsb")
            for i, y in enumerate(Y):
                o = self._scipy(y, mu, W, k)
                self.assertEqual(o.nit, e.iterations[i])
                self.assertEqual(o.nfev, e.evaluations[i])
                np.testing.assert_allclose(o.x[1:], np.r_[e.nu[i], e.d[i]], atol=1e-7)

    def test_lbfgsb_port_with_cauchy_breakpoints(self):
        # Large early gradients cross d's bound inside the Cauchy search, and the theta term from the
        # fixed tau is in play. The ELBO's -log d keeps the iterates off the bound, so this does not
        # exercise a d resting at its bound across iterations.
        Y, mu, W = fixture(rows=2, seed=11, c=8, p=120)
        Y = Y + 0.2 * np.exp(-0.5 * ((np.arange(120) - 60) / 10.0) ** 2)
        for k in (1, 3, 8, 20):
            e = vj.fit_elbo_jax(Y, mu, W, pool=2, chunk_iters=5, min_pool=2, maxiter=k, retry_failed=False, engine="lbfgsb")
            for i, y in enumerate(Y):
                o = self._scipy(y, mu, W, k)
                self.assertEqual(o.nit, e.iterations[i])
                np.testing.assert_allclose(o.x[1:], np.r_[e.nu[i], e.d[i]], atol=1e-6)

    def test_full_fit_with_polish_and_refill(self):
        Y, mu, W = fixture(rows=7, seed=9)
        a = vb.fit_elbo_batch(Y, mu, W, pool=3, polish_restarts=3)
        b = vj.fit_elbo_jax(Y, mu, W, pool=4, chunk_iters=7, min_pool=2, polish_restarts=3)
        self.assertTrue(b.success.all(), b.message)
        np.testing.assert_allclose(b.elbo, a.elbo, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
