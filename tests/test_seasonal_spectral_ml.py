"""Scientific guards: fold coverage, training-only learning and block permutation."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from seasonal_spectral_ml import (  # noqa: E402
    CycleFold, build_model, cycle_inventory, evaluate_folds,
    grouped_region_permutation, leave_region_out_refit, make_cycle_folds,
    make_feature_frame, permute_region, row_permutations, season_labels,
    spectral_regions,
)


def sample_cycles():
    dates = np.concatenate([
        pd.date_range("2022-12-01", "2023-09-01", freq="MS"),
        pd.date_range("2023-10-01", "2024-09-01", freq="MS"),
        pd.date_range("2024-10-01", "2025-08-01", freq="MS"),
        pd.date_range("2025-11-01", "2026-01-01", freq="MS"),
    ])
    return dates, np.asarray([f"ETAD-{i:04d}" for i in range(len(dates))])


class SeasonalSpectralMLTests(unittest.TestCase):
    def test_calendar_labels_explicit_february_and_invalid_dates(self):
        dates = ["2024-01-15", "2024-02-15", "2024-06-15"]
        belg = season_labels(dates, "belg_feb")
        dry = season_labels(dates, "dry_feb")
        self.assertTrue(belg[0].startswith("Bega"))
        self.assertTrue(belg[1].startswith("Belg"))
        self.assertTrue(dry[1].startswith("Dry"))
        self.assertTrue(belg[2].startswith("Kiremt"))
        with self.assertRaises(ValueError):
            season_labels(["2024-02-01", None])
        with self.assertRaisesRegex(ValueError, "timezone"):
            season_labels(pd.date_range("2024-01-01", periods=2, tz="UTC"))

    def test_primary_complete_test_and_secondary_sparse_cycle_are_distinct(self):
        dates, ids = sample_cycles()
        primary, inventory = make_cycle_folds(dates, ids, min_train=6, min_test=3)
        self.assertEqual([fold.test_cycle for fold in primary], [2024])
        self.assertEqual(len(primary[0].train), 10)
        self.assertEqual(len(primary[0].test), 12)
        self.assertTrue(primary[0].complete_test)
        strict, _ = make_cycle_folds(dates, ids, train_coverage="all_months", min_train=6, min_test=3)
        self.assertEqual(strict, [])
        extended, _ = make_cycle_folds(dates, ids, test_coverage="all_seasons", min_train=6, min_test=3)
        self.assertEqual([fold.test_cycle for fold in extended], [2024, 2025])
        self.assertEqual([fold.complete_test for fold in extended], [True, False])
        inv = inventory.set_index("cycle")
        self.assertEqual(inv.loc[2023, "missing_months"], [10, 11])
        self.assertEqual(inv.loc[2025, "missing_months"], [9])
        self.assertFalse(inv.loc[2026, "all_seasons"])
        for fold in extended:
            self.assertLess(pd.Timestamp(dates[fold.train].max()), pd.Timestamp(dates[fold.test].min()))
            self.assertFalse(set(ids[fold.train]) & set(ids[fold.test]))

    def test_folds_deterministic_under_row_reordering_and_group_dates_checked(self):
        dates, ids = sample_cycles()
        rng = np.random.default_rng(12)
        order = rng.permutation(len(ids))
        first, _ = make_cycle_folds(dates, ids, test_coverage="all_seasons", min_train=6, min_test=3)
        second, _ = make_cycle_folds(dates[order], ids[order], test_coverage="all_seasons", min_train=6, min_test=3)
        for one, two in zip(first, second):
            self.assertEqual(set(ids[one.train]), set(ids[order][two.train]))
            self.assertEqual(set(ids[one.test]), set(ids[order][two.test]))
        # Inventoried repeat acquisitions count once; conflicting ownership/date fails.
        inv = cycle_inventory(np.append(dates, dates[0]), np.append(ids, ids[0]))
        self.assertEqual(inv.n_filters.sum(), len(ids))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            cycle_inventory(np.append(dates, dates[-1]), np.append(ids, ids[0]))

    def test_preprocessing_fits_only_training_and_unknown_lot_stays_unknown(self):
        rng = np.random.default_rng(3)
        spectra = rng.normal(size=(30, 5))
        spectra[20:] += 1000  # conspicuous held-out distribution shift
        meta = pd.DataFrame({"log_bc": np.r_[np.arange(20), np.repeat(999., 10)],
                             "lot": ["248"]*10+["251"]*10+["test_only"]*10})
        ids = [f"F{i}" for i in range(30)]
        X, columns = make_feature_frame(spectra, meta, physical_ids=ids)
        model = build_model(task="regression", model="linear", spectral_columns=columns,
                            numeric_metadata=["log_bc"], categorical_metadata=["lot"], n_components=3)
        fold = CycleFold("heldout", np.arange(20), np.arange(20, 30), 2024, True, (2023,))
        result = evaluate_folds(X, np.arange(30)/3, ids, [fold], model, task="regression")
        fitted = result["fitted_folds"]["heldout"]["estimator"]
        parts = fitted.named_steps["features"].named_transformers_
        np.testing.assert_allclose(parts["spectrum"].named_steps["center"].mean_, spectra[:20].mean(axis=0))
        self.assertAlmostEqual(parts["numeric"].named_steps["scale"].mean_[0], 9.5)
        self.assertEqual(list(parts["categorical"].named_steps["encode"].categories_[0]), ["248", "251"])
        self.assertFalse(hasattr(model.named_steps["features"], "transformers_"))
        self.assertEqual(result["predictions"].physical_id.tolist(), ids[20:])
        self.assertEqual(result["metrics"].n_test.iloc[0], 10)

    def test_no_calendar_predictors_for_classification_and_no_nan_spectra(self):
        for name in ("season", "month_sin", "sample_date", "cycle_end_year"):
            with self.assertRaisesRegex(ValueError, "leak season"):
                build_model(task="classification", spectral_columns=["sp_0000"], numeric_metadata=[name])
        with self.assertRaises(ValueError):
            make_feature_frame([[0., np.nan], [1., 2.]])
        X, columns = make_feature_frame([[-2., 1.], [-1., 3.]])
        self.assertEqual(X[columns[0]].tolist(), [-2., -1.])

    def test_evaluation_rejects_duplicate_or_misaligned_physical_ids(self):
        X, columns = make_feature_frame(np.arange(16).reshape(8, 2), physical_ids=[f"F{i}" for i in range(8)])
        model = build_model(task="regression", model="linear", spectral_columns=columns, n_components=1)
        fold = CycleFold("one", np.arange(5), np.arange(5, 8), 2024, True, (2023,))
        with self.assertRaisesRegex(ValueError, "one row per physical"):
            evaluate_folds(X, np.arange(8), ["same"]*8, [fold], model, task="regression")
        with self.assertRaisesRegex(ValueError, "row order"):
            evaluate_folds(X, np.arange(8), list(reversed(X.index)), [fold], model, task="regression")
        overlap = CycleFold("bad", np.arange(6), np.arange(5, 8), 2024, True, (2023,))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            evaluate_folds(X, np.arange(8), X.index, [overlap], model, task="regression")

    def test_pls_direct_spectral_regression_caps_components_and_flattens(self):
        rng = np.random.default_rng(14)
        X, columns = make_feature_frame(rng.normal(size=(25, 4)))
        target = 2*X.iloc[:, 0].to_numpy() - 3*X.iloc[:, 2].to_numpy()
        model = build_model(task="regression", model="pls", spectral_columns=columns, n_components=8)
        clone(model).fit(X.iloc[:20], target[:20])
        model.fit(X.iloc[:20], target[:20])
        self.assertEqual(model.named_steps["estimator"].n_components_, 4)
        self.assertNotIn("pca", model.named_steps["features"].named_transformers_["spectrum"].named_steps)
        prediction = model.predict(X.iloc[20:])
        self.assertEqual(prediction.shape, (5,))
        np.testing.assert_allclose(prediction, target[20:], atol=1e-9)

    def test_joint_region_permutation_preserves_adjacent_channel_relationships(self):
        X = pd.DataFrame({"a": np.arange(8.), "b": 2*np.arange(8.)+5, "metadata": np.arange(8)+100},
                         index=[f"filter{i}" for i in range(8)])
        old = X.copy(deep=True)
        order = np.array([3, 7, 0, 6, 1, 5, 2, 4])
        permuted = permute_region(X, ["a", "b"], order)
        np.testing.assert_allclose(permuted.b, 2*permuted.a+5)
        np.testing.assert_array_equal(permuted.a, X.a.iloc[order].to_numpy())
        pd.testing.assert_series_equal(permuted.metadata, X.metadata)
        pd.testing.assert_frame_equal(X, old)

    def test_stratified_permutations_preserve_seasons_and_are_reproducible(self):
        labels = np.array(["Bega"]*4+["Belg"]*4+["Kiremt"]*4)
        first = row_permutations(12, repeats=4, seed=18, strata=labels)
        second = row_permutations(12, repeats=4, seed=18, strata=labels)
        for one, two in zip(first, second):
            np.testing.assert_array_equal(one, two)
            np.testing.assert_array_equal(labels[one], labels)
            np.testing.assert_array_equal(np.sort(one), np.arange(12))

    def test_permutation_precedes_latent_transform_and_negative_values_retained(self):
        rng = np.random.default_rng(24)
        X, columns = make_feature_frame(rng.normal(size=(60, 4)))
        y = 5*X[columns[0]].to_numpy()
        model = build_model(task="regression", model="pls", spectral_columns=columns, n_components=4)
        model.fit(X.iloc[:40], y[:40])
        result = grouped_region_permutation(model, X.iloc[40:], y[40:],
                                            {"driver": columns[:1], "irrelevant": columns[3:]},
                                            task="regression", repeats=4, seed=8)
        self.assertGreater(result.loc[result.region.eq("driver"), "importance"].median(), 1.)
        self.assertLess(abs(result.loc[result.region.eq("irrelevant"), "importance"]).max(), 1e-9)

        class WrongPredictor:
            def predict(self, frame):
                return frame.iloc[:, 0].to_numpy()

        wrong = grouped_region_permutation(WrongPredictor(), X.iloc[40:], -X.iloc[40:, 0].to_numpy(),
                                           {"harmful": columns[:1]}, task="regression", repeats=4, seed=9)
        self.assertTrue(wrong.importance.lt(0).all())

    def test_ablation_refits_latent_components_and_whole_spectrum_uses_train_mean(self):
        rng = np.random.default_rng(7)
        X, columns = make_feature_frame(rng.normal(size=(36, 4)))
        y = 3*X.iloc[:, 0].to_numpy()
        model = build_model(task="regression", model="pls", spectral_columns=columns, n_components=8)
        result = leave_region_out_refit(model, X.iloc[:24], y[:24], X.iloc[24:], y[24:],
                                        {"drop_driver": columns[:1], "whole": columns}, task="regression")
        self.assertFalse(result.loc[result.region.eq("drop_driver"), "intercept_only"].iloc[0])
        whole = result.loc[result.region.eq("whole")].iloc[0]
        self.assertTrue(whole.intercept_only)
        self.assertAlmostEqual(whole.ablated_loss, np.sqrt(np.mean((y[24:]-y[:24].mean())**2)))
        self.assertGreater(result.importance.min(), 1.)
        self.assertFalse(hasattr(model.named_steps["features"], "transformers_"))

    def test_region_partition_and_classification_output(self):
        columns = [f"sp_{i:04d}" for i in range(6)]
        regions = spectral_regions([1599, 1550, 1500, 1499, 1450, 1400], columns,
                                    named_regions={"wide": (1450, 1600)})
        bins = [v for name, v in regions.items() if name.startswith("bin_")]
        self.assertCountEqual([c for block in bins for c in block], columns)
        self.assertEqual(regions["whole_spectrum"], columns)
        labels = np.tile(["Bega", "Belg", "Kiremt"], 12)
        data = np.eye(3)[np.tile(np.arange(3), 12)]
        X, columns = make_feature_frame(data, physical_ids=[f"C{i}" for i in range(len(data))])
        fold = CycleFold("class", np.arange(24), np.arange(24, 36), 2024, True, (2023,))
        for method in ("rf", "linear"):
            model = build_model(task="classification", model=method, spectral_columns=columns,
                                n_components=2, n_jobs=1,
                                rf_params={"n_estimators": 12, "min_samples_leaf": 2})
            result = evaluate_folds(X, labels, X.index, [fold], model, task="classification")
            self.assertEqual(result["metrics"].balanced_accuracy.iloc[0], 1.)
            probabilities = result["predictions"].filter(like="proba__")
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.)


if __name__ == "__main__":
    unittest.main()
