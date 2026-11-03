"""Protect numerical-status and no-silent-filtering semantics of arPLS audit."""

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("pybaselines")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
from neutral_baseline import neutral_baseline_matrix  # noqa: E402
from neutral_convergence import neutral_with_diagnostics, tolerance_status  # noqa: E402


def test_finite_or_empty_stopping_history_is_not_proof_of_convergence():
    assert not tolerance_status([0.5, 0.2])["converged"]
    assert not tolerance_status([])["converged"]
    assert not tolerance_status([0.1, np.nan])["converged"]
    assert tolerance_status([0.1, 1e-4])["converged"]
    assert tolerance_status([0.1, 1e-4])["iterations"] == 2


def test_default_replays_repository_method_with_identical_grid():
    wn = np.linspace(4000, 500, 1200)
    baseline = 0.03 + 0.01 * (wn - 500) / 3500
    peak = 0.2 * np.exp(-0.5 * ((wn - 1600) / 25) ** 2)
    values = np.stack([baseline + peak, 2 * baseline + 0.5 * peak])
    grid, expected = neutral_baseline_matrix(wn, values, lam=1e6)
    actual_grid, actual, diagnostics = neutral_with_diagnostics(wn, values, max_iter=50)
    np.testing.assert_array_equal(actual_grid, grid)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert len(diagnostics) == len(values)
    assert diagnostics.finite_output.all()
    assert list(diagnostics.row) == [0, 1]
    assert not ((grid > 1100) & (grid < 1300)).any()


def test_iteration_limit_retains_spectra_and_reports_failure():
    wn = np.linspace(4000, 500, 1200)
    values = (0.01 + 0.1 * np.exp(-0.5 * ((wn - 1600) / 100) ** 2))[None, :]
    _, corrected, diagnostics = neutral_with_diagnostics(wn, values, max_iter=1)
    assert len(corrected) == len(values)
    assert np.isfinite(corrected).all()
    assert not diagnostics.success.any()
