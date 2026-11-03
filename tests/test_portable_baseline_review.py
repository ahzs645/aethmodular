"""Counterfactual tests of the chronological optical-disagreement screen."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/workflows"))
from run_portable_baseline_review import SPECTRAL_FEATURES, chronological_screen


def chronological_fixture():
    """Five quarters, each with twenty distinct physical filters."""
    rng = np.random.default_rng(76031)
    dates = pd.DatetimeIndex(np.concatenate([
        pd.date_range(start, periods=20).to_numpy()
        for start in pd.date_range("2023-01-01", periods=5, freq="QS")
    ]))
    n = len(dates)
    spectral = rng.normal(size=(n, len(SPECTRAL_FEATURES)))
    optical = rng.uniform(1, 6, size=n)
    frame = pd.DataFrame(spectral, columns=SPECTRAL_FEATURES)
    frame["id"] = [f"ETAD-{i:04d}" for i in range(n)]
    frame["date"] = dates.strftime("%Y-%m-%d")
    frame["code"] = "ETAD"
    frame["spectral_pair_eligible"] = True
    frame["optical_bc_ugm3"] = optical
    frame["pm25_ugm3"] = 3 * optical + rng.uniform(1, 5, size=n)
    frame["volume_m3"] = rng.uniform(6.9, 7.4, size=n)
    frame["lot_group"] = "251"
    frame["optical_source"] = "direct_hips_fabs"
    frame["optical_disagreement_ugm3"] = (
        -0.7 + 0.12 * optical + 0.2 * spectral[:, 0] + rng.normal(scale=0.07, size=n)
    )
    return frame


class PortableBaselineChronologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = chronological_fixture()
        cls.predictions, cls.summary = chronological_screen(cls.frame)
        cls.outputs = [
            "metadata_prediction", "baseline_prediction", "bands_prediction", "spectra_prediction",
            "metadata_alpha", "baseline_alpha", "bands_alpha", "spectra_alpha",
        ]

    def test_each_physical_filter_is_predicted_once_using_strictly_earlier_quarters(self):
        predictions = self.predictions
        self.assertEqual(len(predictions), 40)
        self.assertEqual(predictions.id.nunique(), 40)
        self.assertTrue((pd.to_datetime(predictions.train_end) < pd.to_datetime(predictions.date)).all())
        self.assertEqual(set(predictions.quarter), {"2023Q4", "2024Q1"})
        for quarter, group in predictions.groupby("quarter"):
            expected_train = int((pd.to_datetime(self.frame.date).dt.to_period("Q").astype(str) < quarter).sum())
            self.assertTrue(group.train_n.eq(expected_train).all())
            self.assertTrue(group[self.outputs].notna().all().all())

    def test_final_quarter_outcomes_cannot_change_predictions_or_selected_penalties(self):
        changed = self.frame.copy()
        changed.loc[changed.date.ge("2024-01-01"), "optical_disagreement_ugm3"] += 1000
        after, _ = chronological_screen(changed)
        pd.testing.assert_frame_equal(after[["id", *self.outputs]], self.predictions[["id", *self.outputs]])
        self.assertTrue((after.actual_disagreement - self.predictions.actual_disagreement).abs().max() > 900)

    def test_future_covariates_and_new_categories_cannot_change_an_earlier_test_quarter(self):
        changed = self.frame.copy()
        future = changed.date.ge("2024-01-01")
        numeric = ["optical_bc_ugm3", "pm25_ugm3", "volume_m3", *SPECTRAL_FEATURES]
        changed.loc[future, numeric] = changed.loc[future, numeric] * 1000 + 10000
        changed.loc[future, "lot_group"] = "future_only_lot"
        changed.loc[future, "optical_source"] = "public_hips_extension"
        after, _ = chronological_screen(changed)
        earlier = self.predictions.quarter.eq("2023Q4")
        self.assertEqual(after.loc[earlier, "id"].tolist(), self.predictions.loc[earlier, "id"].tolist())
        np.testing.assert_allclose(
            after.loc[earlier, self.outputs].to_numpy(float),
            self.predictions.loc[earlier, self.outputs].to_numpy(float),
            rtol=1e-11, atol=1e-11,
        )

    def test_duplicate_physical_filter_identity_is_rejected(self):
        duplicated = pd.concat([self.frame, self.frame.iloc[[-1]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "one row per physical filter"):
            chronological_screen(duplicated)


if __name__ == "__main__":
    unittest.main()
