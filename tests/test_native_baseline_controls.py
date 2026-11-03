"""Concrete leakage, grid, boundary and recovery risks in native controls."""

from pathlib import Path
import sys
import unittest
from importlib.util import find_spec

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from native_baseline_controls import (  # noqa: E402
    assert_parent_holdout, blank_fold_assignments, fixed_segment2,
    neutral_correction, paired_injection_metrics, segmented_spline_variants,
    validate_native_grid,
)
from airspec_baseline import airspec_baseline_matrix, make_mask, SEG2  # noqa: E402


class NativeBaselineControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wn = np.linspace(3998.423206, 500.124318, 2722)
        x = cls.wn
        cls.parent = (.12 + .04*(x-2300)/1700 + .025*((x-2300)/1700)**2
                      + .012*np.exp(-.5*((x-2920)/35)**2)
                      + .006*np.exp(-.5*((x-1715)/30)**2))

    def test_native_reference_and_segment_support(self):
        _, reference = airspec_baseline_matrix(self.wn, self.parent[None, :])
        _, corrected, log = segmented_spline_variants(self.wn, self.parent[None, :])
        np.testing.assert_allclose(corrected["spline_default"], reference,
                                   atol=1e-11, rtol=1e-10, equal_nan=True)
        outside_second = (self.wn >= 2000) & (self.wn < 4000)
        for method in ("spline_anchor1500", "spline_anchor1560"):
            np.testing.assert_allclose(corrected[method][:, outside_second],
                                       reference[:, outside_second], atol=1e-11)
        self.assertTrue((log.extension_native_channels == 20).all())
        np.testing.assert_allclose(log.extension_span_cm_1,
                                   20*abs(np.diff(self.wn)[0]), rtol=1e-9)

    def test_protected_interval_does_not_fit_added_peak(self):
        mask = make_mask(self.wn, SEG2)
        x, parent = self.wn[mask], self.parent[mask]
        added = np.zeros_like(x)
        inside = (x >= 1580) & (x <= 1650)
        added[inside] = .005*np.sin(np.pi*(x[inside]-1580)/70)**2
        before = fixed_segment2(x, parent, bound=1500)
        after = fixed_segment2(x, parent+added, bound=1500)
        np.testing.assert_allclose(after["baseline"], before["baseline"], atol=1e-12)
        np.testing.assert_allclose(after["absorbance"]-before["absorbance"], added, atol=1e-12)

    def test_coarse_or_reversed_native_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "coarsened"):
            validate_native_grid(self.wn[::6], self.parent[None, ::6])
        with self.assertRaisesRegex(ValueError, "descending"):
            validate_native_grid(self.wn[::-1], self.parent[None, ::-1])

    def test_physical_blank_folds_stable_and_complete(self):
        blank = pd.DataFrame({"MediaId": range(100, 137),
                              "base_filter_id": [f"B{i}" for i in range(37)],
                              "lot_group": ["248"]*5+["251"]*23+["unknown"]*9})
        first = blank_fold_assignments(blank).set_index("MediaId").fold.sort_index()
        shuffled = blank_fold_assignments(blank.sample(frac=1, random_state=7)).set_index("MediaId").fold.sort_index()
        pd.testing.assert_series_equal(first, shuffled)
        self.assertEqual(set(first), set(range(5)))
        for fold in range(5):
            train = first.index[first.ne(fold)]
            test = first.index[first.eq(fold)]
            assert_parent_holdout(train, test)
        with self.assertRaisesRegex(ValueError, "unique physical"):
            blank_fold_assignments(pd.concat([blank, blank.iloc[[0]]]))

    def test_injected_child_cannot_hide_parent_leakage(self):
        assert_parent_holdout(["101", "102"], ["103", "103", "104"])
        # Child sample IDs differ, but its physical parent is still training data.
        with self.assertRaisesRegex(ValueError, "parent blank leaked"):
            assert_parent_holdout(["101", "102"], ["102", "103"])

    def test_paired_recovery_subtracts_parent_and_preserves_negative_result(self):
        added = .01*np.exp(-.5*((self.wn-1617)/25)**2)
        offset = self.parent - .03
        result = paired_injection_metrics(self.wn, offset, offset-.4*added,
                                           added, bounds=(1510, 1730))
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["area_recovery_fraction"], -.4, places=11)
        reverse = paired_injection_metrics(self.wn[::-1], offset[::-1],
                                            (offset-.4*added)[::-1], added[::-1],
                                            bounds=(1510, 1730))
        self.assertAlmostEqual(result["area_recovery_fraction"],
                               reverse["area_recovery_fraction"], places=12)

    @unittest.skipUnless(find_spec("pybaselines"), "repository explorer dependency not installed")
    def test_neutral_wrapper_matches_repository_baseline(self):
        from neutral_baseline import neutral_baseline_matrix, neutral_grid
        grid, expected = neutral_baseline_matrix(self.wn, self.parent[None, :])
        _, indices = neutral_grid(self.wn)
        _, corrected, diagnostics = neutral_correction(self.wn, self.parent[None, :])
        np.testing.assert_allclose(corrected[:, indices], expected, atol=1e-12)
        self.assertTrue(diagnostics.success.all())
        self.assertGreater(len(grid), 2000)

    @unittest.skipUnless(find_spec("pybaselines"), "repository explorer dependency not installed")
    def test_finite_unconverged_fit_is_visible_but_not_successful(self):
        sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/workflows"))
        from run_existing_data_baseline_controls import metrics, with_convergence
        _, corrected, log = neutral_correction(self.wn, self.parent[None, :], max_iter=1)
        self.assertTrue(log.success.iloc[0])  # finite numerical values are retained
        self.assertFalse(log.converged.iloc[0])
        shared = (self.wn > 1425) & (self.wn < 4000)
        report = with_convergence(metrics(self.wn, self.parent[None, :], corrected, shared), log.converged)
        self.assertTrue(report.finite_output.iloc[0])
        self.assertFalse(report.success.iloc[0])
        self.assertTrue(np.isfinite(report.corrected_rms.iloc[0]))


if __name__ == "__main__":
    unittest.main()
