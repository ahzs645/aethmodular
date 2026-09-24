"""Paired audit of frozen EC predictions; never refits or excludes observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("AIRSpec", "VIBES")
COHORTS = ("full_pool", "locked800")


def pair_predictions(predictions, cases):
    """Reject incomplete/ambiguous joins and changed truth, split or membership."""
    if cases.sample_id.isna().any() or cases.sample_id.duplicated().any():
        raise ValueError("Case identifiers must be present and unique")
    keys = ["cohort", "sample_id", "method"]
    if predictions[keys].isna().any().any() or predictions.duplicated(keys).any():
        raise ValueError("Prediction keys must be present and unique")
    if set(predictions.method) != set(METHODS) or set(predictions.cohort) != set(COHORTS):
        raise ValueError("Unexpected methods or cohorts")
    for col in ("paired_valid", "locked800"):
        if cases[col].isna().any() or not cases[col].isin([True, False]).all():
            raise ValueError(f"Invalid boolean metadata: {col}")
    cal = cases.loc[cases.kind.eq("calibration") & cases.paired_valid]
    if cal[["Site", "lot", "split", "y"]].isna().any().any():
        raise ValueError("Missing calibration metadata")
    if not cal.split.isin(["train", "test"]).all() or not np.isfinite(cal.y).all():
        raise ValueError("Invalid calibration split or truth")
    joined = predictions.merge(
        cases, on="sample_id", how="left", validate="many_to_one",
        suffixes=("_prediction", ""), indicator=True,
    )
    if not joined._merge.eq("both").all():
        raise ValueError("Unmatched prediction identifiers")
    if not joined.Site.eq(joined.Site_prediction).all() or not np.allclose(
        joined.y, joined.y_prediction, rtol=0, atol=1e-12
    ):
        raise ValueError("Prediction site or truth differs from frozen cases")
    if not np.isfinite(joined.prediction).all():
        raise ValueError("Nonfinite predictions")
    if not joined.split.eq("test").all() or not joined.kind.eq("calibration").all():
        raise ValueError("Predictions include non-test calibration rows")
    pairs = []
    for cohort in COHORTS:
        eligible = cal if cohort == "full_pool" else cal.loc[cal.locked800]
        train = eligible.loc[eligible.split.eq("train")]
        test = eligible.loc[eligible.split.eq("test")]
        if set(train.Site) & set(test.Site):
            raise ValueError("Training and test sites overlap")
        pred = predictions.loc[predictions.cohort.eq(cohort)]
        for method in METHODS:
            if set(pred.loc[pred.method.eq(method), "sample_id"]) != set(test.sample_id):
                raise ValueError("Incomplete or changed held-out membership")
        wide = pred.pivot(index="sample_id", columns="method", values="prediction")
        frame = test.merge(wide, on="sample_id", validate="one_to_one").assign(cohort=cohort)
        pairs.append(frame)
    return pd.concat(pairs, ignore_index=True)


def loading_bands(training_y, test_y):
    """Training quartiles, with thresholds assigned to the lower band."""
    cuts = np.quantile(np.asarray(training_y, dtype=float), [0.25, 0.5, 0.75])
    if not np.isfinite(cuts).all() or not (np.diff(cuts) > 0).all():
        raise ValueError("Training quartiles must be finite and distinct")
    labels = np.array(["Q1", "Q2", "Q3", "Q4"])
    return labels[np.searchsorted(cuts, test_y, side="left")], cuts


def metrics(frame, *, repeats=10000, seed=20260921):
    """Sample-weighted errors and paired percentile CIs from site-cluster draws.

    Intervals condition on the fitted models. They do not include refitting
    uncertainty or multiplicity correction. One-site groups get no interval.
    """
    if len(frame) == 0 or repeats < 1:
        raise ValueError("Metrics need observations and positive bootstrap repeats")
    result = {"n": len(frame), "n_sites": frame.Site.nunique()}
    sums = pd.DataFrame({"site": frame.Site, "n": 1})
    for method in METHODS:
        error = frame[method] - frame.y
        result[f"{method}_RMSE"] = float(np.sqrt(np.mean(error**2)))
        result[f"{method}_MAE"] = float(np.mean(abs(error)))
        result[f"{method}_bias"] = float(np.mean(error))
        denom = np.sum((frame.y - frame.y.mean())**2)
        result[f"{method}_predictive_R2"] = float(1 - np.sum(error**2) / denom) if denom > 0 else np.nan
        sums[f"{method}_sq"] = error**2
        sums[f"{method}_abs"] = abs(error)
        sums[f"{method}_err"] = error
    result["fraction_VIBES_smaller_abs_error"] = float(
        ((frame.VIBES - frame.y).abs() < (frame.AIRSpec - frame.y).abs()).mean()
    )
    totals = sums.groupby("site", sort=True).sum()
    if len(totals) >= 2:
        draws = np.random.default_rng(seed).integers(0, len(totals), (repeats, len(totals)))
        boot = totals.to_numpy()[draws].sum(axis=1)
        b = dict(zip(totals.columns, boot.T))
    for metric, suffix in (("RMSE", "sq"), ("MAE", "abs"), ("bias", "err")):
        key = f"delta_{metric}"
        result[key] = result[f"VIBES_{metric}"] - result[f"AIRSpec_{metric}"]
        result[key + "_ci_low"] = result[key + "_ci_high"] = np.nan
        if len(totals) >= 2:
            air = b[f"AIRSpec_{suffix}"] / b["n"]
            vib = b[f"VIBES_{suffix}"] / b["n"]
            delta = np.sqrt(vib) - np.sqrt(air) if metric == "RMSE" else vib - air
            lo, hi = np.quantile(delta, [0.025, 0.975])
            result[key + "_ci_low"], result[key + "_ci_high"] = float(lo), float(hi)
    result["interval_status"] = (
        "unavailable_single_site" if len(totals) < 2 else
        "few_sites_exploratory" if len(totals) < 10 else "exploratory_site_cluster"
    )
    return result


def audit_groups(paired, cases, *, repeats=10000, seed=20260921):
    rows, definitions, annotated, influence = [], [], [], []
    for cohort in COHORTS:
        frame = paired.loc[paired.cohort.eq(cohort)].copy()
        train = cases.loc[cases.kind.eq("calibration") & cases.split.eq("train") & cases.paired_valid]
        if cohort == "locked800":
            train = train.loc[train.locked800]
        frame["loading_band"], cuts = loading_bands(train.y, frame.y)
        definitions.append(dict(cohort=cohort, n_train=len(train), q25=cuts[0], q50=cuts[1], q75=cuts[2]))
        for method in METHODS:
            frame[f"{method}_error"] = frame[method] - frame.y
        frame["delta_squared_error"] = frame.VIBES_error**2 - frame.AIRSpec_error**2
        frame["delta_absolute_error"] = frame.VIBES_error.abs() - frame.AIRSpec_error.abs()
        frame["absolute_prediction_difference"] = (frame.VIBES - frame.AIRSpec).abs()
        annotated.append(frame)
        groups = [("overall", "all", frame)]
        for dimension in ("Site", "lot", "loading_band"):
            groups.extend((dimension, str(value), part) for value, part in frame.groupby(dimension, sort=True, observed=True))
        for dimension, value, part in groups:
            rows.append(dict(cohort=cohort, dimension=dimension, group=value,
                             **metrics(part, repeats=repeats, seed=seed)))
        for site, part in frame.groupby("Site", sort=True):
            # Algebraic leave-one-site-out sensitivity, never changing the primary population.
            remain_n = len(frame) - len(part)
            remaining = {}
            for method in METHODS:
                err = frame[f"{method}_error"]
                site_err = part[f"{method}_error"]
                remaining[method] = np.sqrt((np.sum(err**2) - np.sum(site_err**2)) / remain_n)
            influence.append(dict(
                cohort=cohort, site=site, n=len(part),
                contribution_to_overall_delta_MSE=part.delta_squared_error.sum() / len(frame),
                delta_RMSE_without_site=remaining["VIBES"] - remaining["AIRSpec"],
            ))
    return pd.DataFrame(rows), pd.DataFrame(definitions), pd.concat(annotated), pd.DataFrame(influence)
