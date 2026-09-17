"""Fixed calendar-block validation of frozen, same-filter reported products."""

import hashlib
import json

import numpy as np
import pandas as pd

from plotting.utils import calculate_regression_stats

SITE_ORDER = ("Addis_Ababa", "Beijing", "Delhi", "JPL")
POPULATIONS = {
    "diagnostic": None,
    "ratio_baseline": 1.0,
    "mdl_1_5x": 1.5,
    "mdl_2x": 2.0,
    "mdl_3x": 3.0,
    "mdl_5x": 5.0,
}
MIN_TRAIN = 10
TARGET = "Addis_Ababa:ETAD-0037"
X, Y = "ftir_ec_ugm3", "hips_fabs_Mm1"


def ids_json(frame):
    return json.dumps(sorted(frame.point_id.tolist()))


def prepare_points(points, measurements):
    """Attach metadata using exact EC source-row references, never approximate dates."""
    points = points.copy()
    if points.point_id.duplicated().any():
        raise ValueError("Physical filter identities must be unique")
    points["date"] = pd.to_datetime(points.date)
    points["reported_date_block"] = points.date.dt.to_period("Q-DEC").astype(str)
    source = measurements.set_index("source_row", verify_integrity=True)
    records = []
    for p in points.loc[points.eligible_filter_diagnostic].to_dict("records"):
        rows = source.loc[json.loads(p["ftir_ec_ugm3_source_rows"])]
        if (
            not rows.Parameter.eq("EC_ftir").all()
            or not rows.base_filter_id.eq(p["base_filter_id"]).all()
        ):
            raise ValueError("EC source identity/parameter mismatch")
        if not np.allclose(rows.Concentration.astype(float), p[X], rtol=1e-10):
            raise ValueError("EC source value mismatch")
        record = {"point_id": p["point_id"]}
        for field in ["CalibrationSetId", "AnalysisDate", "AnalysisTime", "LotId"]:
            vals = sorted(rows[field].dropna().astype(str).unique())
            record["ftir_" + field] = "|".join(vals) if vals else "unreported"
        record["unified_source_file_hash"] = "|".join(
            sorted(rows.unified_source_file_hash.unique())
        )
        records.append(record)
    return points.merge(pd.DataFrame(records), on="point_id", how="left", validate="one_to_one")


def population_memberships(points, links):
    diag = points.loc[points.eligible_filter_diagnostic]
    rows = []
    for population, multiple in POPULATIONS.items():
        ids = (
            set(diag.point_id)
            if multiple is None
            else set(links.loc[links.minimum_ec_mdl_multiple.eq(multiple), "point_id"])
        )
        if not ids.issubset(set(diag.point_id)):
            raise ValueError("Saved sensitivity includes an ineligible diagnostic point")
        if population == "ratio_baseline" and ids != set(
            points.loc[points.eligible_ec_ratio_analysis, "point_id"]
        ):
            raise ValueError("Saved 1x membership differs from frozen ratio population")
        rows.extend({"point_id": p, "population": population} for p in sorted(ids))
    return pd.DataFrame(rows).merge(points, on="point_id", validate="many_to_one")


def fit_training(train):
    if len(train) < MIN_TRAIN:
        return {
            "status": "insufficient_training_n",
            "slope": np.nan,
            "intercept": np.nan,
            "median": np.nan,
        }
    if train[X].nunique() < 2:
        return {
            "status": "constant_training_ec",
            "slope": np.nan,
            "intercept": np.nan,
            "median": np.nan,
        }
    if not np.isfinite(train[[X, Y]].to_numpy(float)).all():
        raise ValueError("Frozen eligible values must be finite; no silent deletion")
    fit = calculate_regression_stats(train[X], train[Y])
    return {
        "status": "evaluated",
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "median": train[Y].median(),
    }


def error_stats(frame):
    result = {}
    for model in ["ols", "median"]:
        e = frame[model + "_error"].dropna()
        result[model + "_mean_signed_error"] = e.mean()
        result[model + "_mae"] = e.abs().mean()
        result[model + "_rmse"] = np.sqrt((e**2).mean())
    result["mae_improvement"] = result["median_mae"] - result["ols_mae"]
    result["mae_improvement_pct"] = (
        100 * result["mae_improvement"] / result["median_mae"]
        if result["median_mae"] > 0
        else np.nan
    )
    return result


def evaluate(points, memberships):
    """Both split schemes and all populations; test outcomes never enter fitting."""
    blocks, predictions, coefficients, cohorts = [], [], [], []
    for site in SITE_ORDER:
        diagnostic = points.loc[points.site.eq(site) & points.eligible_filter_diagnostic]
        if diagnostic.empty:
            continue
        calendar = pd.period_range(diagnostic.date.min(), diagnostic.date.max(), freq="Q-DEC")
        for population in POPULATIONS:
            original = memberships.loc[
                memberships.site.eq(site) & memberships.population.eq(population)
            ]
            variants = ["baseline", "without_ETAD_0037"] if site == "Addis_Ababa" else ["baseline"]
            reference = fit_training(original)
            for variant in variants:
                cohort = (
                    original
                    if variant == "baseline"
                    else original.loc[original.point_id.ne(TARGET)]
                )
                full = fit_training(cohort)
                key = {"site": site, "population": population, "variant": variant}
                cohorts.append(
                    dict(
                        **key,
                        n=len(cohort),
                        ec_min=cohort[X].min(),
                        ec_max=cohort[X].max(),
                        reported_date_min=cohort.date.min(),
                        reported_date_max=cohort.date.max(),
                        point_ids=ids_json(cohort),
                    )
                )
                coefficients.append(
                    dict(
                        **key,
                        **full,
                        full_cohort_reference_only=True,
                        delta_slope_from_original=full["slope"] - reference["slope"],
                        delta_intercept_from_original=full["intercept"] - reference["intercept"],
                    )
                )
                for scheme in ["leave_quarter_out", "later_period"]:
                    for quarter in calendar:
                        block = str(quarter)
                        test = cohort.loc[cohort.reported_date_block.eq(block)]
                        train = (
                            cohort.loc[cohort.reported_date_block.ne(block)]
                            if scheme == "leave_quarter_out"
                            else cohort.loc[cohort.date.lt(quarter.start_time)]
                        )
                        assert not set(train.point_id) & set(test.point_id)
                        if scheme == "later_period" and len(train) and len(test):
                            assert train.date.max() < test.date.min()
                        fit = fit_training(train)
                        status = "empty_test_block" if test.empty else fit["status"]
                        split_id = hashlib.sha256(
                            (ids_json(train) + "/" + ids_json(test)).encode()
                        ).hexdigest()
                        row = dict(
                            **key,
                            scheme=scheme,
                            block=block,
                            calendar_start=quarter.start_time,
                            calendar_end_exclusive=(quarter + 1).start_time,
                            status=status,
                            small_test_block=0 < len(test) < 5,
                            train_n=len(train),
                            test_n=len(test),
                            train_ids=ids_json(train),
                            test_ids=ids_json(test),
                            split_sha256=split_id,
                            train_date_min=train.date.min(),
                            train_date_max=train.date.max(),
                            test_date_min=test.date.min(),
                            test_date_max=test.date.max(),
                            train_ec_min=train[X].min(),
                            train_ec_max=train[X].max(),
                            test_ec_min=test[X].min(),
                            test_ec_max=test[X].max(),
                            ols_slope=fit["slope"],
                            ols_intercept=fit["intercept"],
                            training_median=fit["median"],
                            slope_delta_from_full=fit["slope"] - full["slope"],
                            intercept_delta_from_full=fit["intercept"] - full["intercept"],
                        )
                        pred = test.copy()
                        for k, v in {
                            **key,
                            "scheme": scheme,
                            "block": block,
                            "status": status,
                            "split_sha256": split_id,
                        }.items():
                            pred[k] = v
                        for k in [
                            "train_n",
                            "ols_slope",
                            "ols_intercept",
                            "training_median",
                            "train_ec_min",
                            "train_ec_max",
                        ]:
                            pred[k] = row[k]
                        pred["outside_training_ec"] = (
                            pd.array(
                                (test[X].lt(train[X].min()) | test[X].gt(train[X].max())).tolist(),
                                dtype="boolean",
                            )
                            if len(train)
                            else pd.array([pd.NA] * len(test), dtype="boolean")
                        )
                        pred["ols_prediction"] = fit["intercept"] + fit["slope"] * pred[X]
                        pred["median_prediction"] = fit["median"]
                        for model in ["ols", "median"]:
                            pred[model + "_error"] = pred[model + "_prediction"] - pred[Y]
                        row.update(error_stats(pred))
                        row["outside_training_ec_n"] = (
                            int(pred.outside_training_ec.sum()) if len(train) else np.nan
                        )
                        row["outside_training_ec_fraction"] = (
                            pred.outside_training_ec.mean() if len(pred) else np.nan
                        )
                        blocks.append(row)
                        if len(pred):
                            predictions.append(pred)
    return (
        pd.DataFrame(blocks),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(coefficients),
        pd.DataFrame(cohorts),
    )


def summarize(blocks, predictions):
    keys = ["site", "population", "variant", "scheme"]
    rows = []
    for key, b in blocks.groupby(keys, sort=False):
        p = predictions
        for col, value in zip(keys, key):
            p = p.loc[p[col].eq(value)]
        good = b.loc[b.status.eq("evaluated")]
        rows.append(
            dict(
                zip(keys, key),
                **error_stats(p),
                evaluated_filters=int(p.ols_error.notna().sum()),
                evaluated_blocks=len(good),
                nonempty_blocks=int(b.test_n.gt(0).sum()),
                unavailable_nonempty_blocks=int((b.test_n.gt(0) & b.status.ne("evaluated")).sum()),
                improved_blocks=int(good.mae_improvement.gt(0).sum()),
                equal_block_mean_mae_improvement=good.mae_improvement.mean(),
                worst_block_mae_improvement=good.mae_improvement.min(),
                best_block_mae_improvement=good.mae_improvement.max(),
            )
        )
    return pd.DataFrame(rows)


def paired_influence(predictions, coefficients, blocks):
    keys = ["site", "population", "scheme", "block", "point_id"]
    base = predictions.loc[predictions.site.eq("Addis_Ababa") & predictions.variant.eq("baseline")]
    omit = predictions.loc[predictions.variant.eq("without_ETAD_0037")]
    paired = base.merge(
        omit[keys + ["ols_error", "median_error", "ols_prediction"]],
        on=keys,
        suffixes=("_baseline", "_without"),
        validate="one_to_one",
    )
    paired = paired.loc[paired.ols_error_baseline.notna() & paired.ols_error_without.notna()].copy()
    rows = []
    for (population, scheme), group in paired.groupby(["population", "scheme"], sort=False):
        for block, g in [("ALL_COMMON_FILTERS", group), *group.groupby("block", sort=False)]:
            rows.append(
                dict(
                    site="Addis_Ababa",
                    population=population,
                    scheme=scheme,
                    block=block,
                    common_n=len(g),
                    baseline_ols_mae=g.ols_error_baseline.abs().mean(),
                    without_ols_mae=g.ols_error_without.abs().mean(),
                    mae_change_without_minus_baseline=g.ols_error_without.abs().mean()
                    - g.ols_error_baseline.abs().mean(),
                    mean_abs_prediction_change=(
                        g.ols_prediction_without - g.ols_prediction_baseline
                    )
                    .abs()
                    .mean(),
                    max_abs_prediction_change=(g.ols_prediction_without - g.ols_prediction_baseline)
                    .abs()
                    .max(),
                )
            )
    return pd.DataFrame(rows), paired


def processing_summary(predictions):
    p = predictions.loc[predictions.variant.eq("baseline") & predictions.ols_error.notna()]
    rows = []
    for field in ["ftir_CalibrationSetId", "ftir_AnalysisDate", "ftir_LotId"]:
        for key, g in p.groupby(
            ["site", "population", "scheme", "block", field], dropna=False, sort=False
        ):
            site, pop, scheme, block, value = key
            rows.append(
                dict(
                    site=site,
                    population=pop,
                    scheme=scheme,
                    block=block,
                    metadata_field=field,
                    reported_value=value,
                    n=len(g),
                    ec_min=g[X].min(),
                    ec_max=g[X].max(),
                    mdl_min=g.ftir_ec_mdl_ugm3.min(),
                    mdl_max=g.ftir_ec_mdl_ugm3.max(),
                    **error_stats(g),
                )
            )
    return pd.DataFrame(rows)
