"""Focused tests for the local di-PLS implementation."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from domain_invariant_pls import block_average, fit_domain_invariant_pls


def test_block_average_keeps_partial_final_block():
    X = np.arange(14, dtype=float).reshape(2, 7)
    got = block_average(X, 3)
    expected = np.array([[1, 4, 6], [8, 11, 13]], dtype=float)
    np.testing.assert_allclose(got, expected)


def test_lambda_zero_source_centering_matches_pls1():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(90, 12))
    y = 2 * X[:, 1] - 0.5 * X[:, 4] + rng.normal(scale=0.1, size=90)
    Xt = rng.normal(loc=0.2, size=(20, 12))
    ours = fit_domain_invariant_pls(X, y, Xt, n_components=5)
    sklearn = PLSRegression(n_components=5, scale=False).fit(X, y)
    np.testing.assert_allclose(
        ours.predict(X, centering="source"),
        sklearn.predict(X).ravel(),
        atol=1e-8,
    )


def test_heuristic_fit_is_finite_and_uses_no_target_response():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 10))
    y = X[:, 0] - X[:, 2] + rng.normal(scale=0.2, size=80)
    Xt = rng.normal(loc=0.3, size=(25, 10))
    fit = fit_domain_invariant_pls(X, y, Xt, n_components=3, heuristic=True)
    assert np.isfinite(fit.predict(Xt)).all()
    assert (fit.lambdas >= 0).all()
