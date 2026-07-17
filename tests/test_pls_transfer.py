import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression


SCRIPTS = Path(__file__).resolve().parents[1] / "research/ftir_hips_chem/scripts"
sys.path.insert(0, str(SCRIPTS))

from pls_transfer import (  # noqa: E402
    component_cv_curve,
    ftir_source_band_features,
    mahalanobis_distance_squared,
    pairwise_score_distance_squared,
    predict_pls_components,
    regression_metrics,
    score_metric,
    select_first_major_minimum,
    spectral_q_residual,
    vip_overlap_summary,
    vip_scores,
)


def _synthetic_model(seed=7):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(120, 4))
    mixing = rng.normal(size=(4, 35))
    X = latent @ mixing + rng.normal(scale=0.02, size=(120, 35))
    y = latent @ np.array([2.0, -1.2, 0.5, 0.1]) + rng.normal(scale=0.05, size=120)
    return X, y, PLSRegression(n_components=4, scale=False).fit(X, y)


def test_vip_has_unit_root_mean_square():
    _, _, model = _synthetic_model()
    vip = vip_scores(model)
    assert vip.shape == (35,)
    assert np.isclose(np.mean(vip**2), 1.0, atol=1e-10)


def test_score_mahalanobis_matches_direct_quadratic_form():
    _, _, model = _synthetic_model()
    scores, inverse_ss = score_metric(model)
    calculated = mahalanobis_distance_squared(scores, inverse_ss)
    direct = np.array([row @ inverse_ss @ row for row in scores])
    assert np.allclose(calculated, direct)
    centered = scores.mean(axis=0)
    assert np.allclose(
        pairwise_score_distance_squared(scores, centered, inverse_ss),
        mahalanobis_distance_squared(scores - centered, inverse_ss),
    )


def test_q_residual_is_nonnegative_and_small_for_training_latent_structure():
    X, _, model = _synthetic_model()
    q = spectral_q_residual(model, X)
    assert np.all(q >= -1e-12)
    assert np.median(q) < 0.1


def test_regression_metrics_and_vip_overlap():
    observed = np.arange(1.0, 11.0)
    predicted = 1.5 * observed - 2.0
    metrics = regression_metrics(observed, predicted)
    assert np.isclose(metrics["slope"], 1.5)
    assert np.isclose(metrics["intercept"], -2.0)
    assert np.isclose(metrics["R2"], 1.0)

    a = np.array([2.0, 1.5, 0.2, 0.1])
    b = np.array([2.1, 1.4, 0.1, 0.2])
    overlap = vip_overlap_summary(a, b, threshold=1.0, top_n=2)
    assert overlap["top_n_overlap"] == 2
    assert overlap["important_jaccard"] == 1.0


def test_successive_component_prediction_matches_full_model():
    X, y, model = _synthetic_model()
    predicted = predict_pls_components(model, X, [1, 3, 4])
    assert predicted.shape == (120, 3)
    assert np.allclose(predicted[:, -1], model.predict(X).ravel())


def test_first_major_minimum_uses_earliest_local_value_within_one_se():
    curve = pd.DataFrame(
        {
            "n_components": [1, 2, 3, 4, 5, 6],
            "rmse_mean": [5.0, 3.2, 3.0, 3.1, 2.9, 3.0],
            "rmse_se": [0.2, 0.2, 0.15, 0.15, 0.2, 0.2],
        }
    )
    selected, annotated = select_first_major_minimum(curve)
    assert selected == 3
    assert annotated.loc[
        annotated["selected_first_major_minimum"], "n_components"
    ].item() == 3


def test_source_band_features_detect_inserted_carbonyl_peak():
    wn = np.linspace(4000, 1400, 1301)
    baseline = 0.1 + 2e-5 * (wn - 1400)
    X = np.vstack([baseline, baseline.copy()])
    X[1] += 0.05 * np.exp(-0.5 * ((wn - 1710) / 20) ** 2)
    features = ftir_source_band_features(X, wn)
    assert features.loc[1, "carbonyl_peak"] > features.loc[0, "carbonyl_peak"] + 0.04


def test_component_cv_curve_returns_fold_uncertainty():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(60, 8))
    y = X[:, 0] + rng.normal(scale=0.1, size=60)
    curve = component_cv_curve(X, y, range(1, 5), n_splits=3)
    assert curve["n_components"].tolist() == [1, 2, 3, 4]
    assert curve["rmse_se"].notna().all()
