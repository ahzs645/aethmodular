"""Separately specified proportional benchmark and common-test ID-11 sensitivity."""

import hashlib
import json

import numpy as np
import pandas as pd

from filter_relationship_stability import X, Y, fit_training, ids_json

MODELS = ("median", "proportional", "ols")
MODEL_NAMES = {"median": "Training median", "proportional": "Proportional", "ols": "With intercept"}
KEYS = ["site", "population", "scheme"]


def proportional_fit(train):
    """Equal-filter least squares through zero; no clipping or positive-only filter."""
    x, y = train[X].to_numpy(float), train[Y].to_numpy(float)
    if not np.isfinite(np.column_stack([x, y])).all():
        raise ValueError("Proportional training values must be finite")
    denominator = np.dot(x, x)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Proportional denominator must be finite and positive")
    return np.dot(x, y) / denominator


def metrics(p):
    out = {}
    for model in MODELS:
        e = p[model + "_error"].dropna()
        out[model + "_mean_signed_error"] = e.mean()
        out[model + "_mae"] = e.abs().mean()
        out[model + "_rmse"] = np.sqrt((e**2).mean())
        out[model + "_nonpositive_prediction_n"] = int(p[model + "_prediction"].le(0).sum())
    out["delta_mae_proportional_minus_ols"] = out["proportional_mae"] - out["ols_mae"]
    out["intercept_mae_improvement_pct"] = (
        100 * out["delta_mae_proportional_minus_ols"] / out["proportional_mae"]
        if out["proportional_mae"] > 0
        else np.nan
    )
    return out


def benchmark(previous_blocks, previous_predictions, members):
    """Copy existing median/OLS predictions and train proportional on exact saved IDs."""
    blocks = previous_blocks.loc[previous_blocks.variant.eq("baseline")].copy()
    previous = previous_predictions.loc[previous_predictions.variant.eq("baseline")].copy()
    points = members.drop_duplicates("point_id").set_index("point_id", verify_integrity=True)
    output, block_rows = [], []
    for r in blocks.to_dict("records"):
        train = points.loc[json.loads(r["train_ids"])]
        test_ids = set(json.loads(r["test_ids"]))
        p = previous
        for col in KEYS + ["block"]:
            p = p.loc[p[col].eq(r[col])]
        p = p.copy()
        if set(p.point_id) != test_ids or len(p) != r["test_n"]:
            raise ValueError("Saved test membership changed")
        if set(train.index) & test_ids:
            raise ValueError("Train/test filter overlap")
        k = proportional_fit(train) if r["status"] == "evaluated" else np.nan
        p["proportional_k"] = k
        p["proportional_prediction"] = k * p[X]
        p["proportional_error"] = p.proportional_prediction - p[Y]
        p["paired_abs_error_difference"] = p.proportional_error.abs() - p.ols_error.abs()
        r["proportional_k"] = k
        r["test_nonpositive_ec_n"] = int(p[X].le(0).sum())
        r.update(metrics(p))
        for model in MODELS:
            if not p[model + "_error"].notna().equals(p.ols_error.notna()):
                raise ValueError("Models must use identical supported test filters")
        block_rows.append(r)
        if len(p):
            output.append(p)
    return pd.DataFrame(block_rows), pd.concat(output, ignore_index=True)


def summaries(blocks, predictions, keys=KEYS):
    """Both estimands: equally weighted filters, or equally weighted test quarters."""
    rows = []
    for group, g in blocks.groupby(keys, sort=False):
        group = group if isinstance(group, tuple) else (group,)
        label = dict(zip(keys, group))
        p = predictions
        for key, val in label.items():
            p = p.loc[p[key].eq(val)]
        valid = g.loc[g.status.eq("evaluated")]
        p = p.loc[p.ols_error.notna()]
        largest = (
            valid.sort_values(["test_n", "block"], ascending=[False, True]).iloc[0]
            if len(valid)
            else None
        )
        context = dict(
            **label,
            evaluated_filters=len(p),
            evaluated_blocks=len(valid),
            nonempty_blocks=int(g.test_n.gt(0).sum()),
            unavailable_nonempty_blocks=int((g.test_n.gt(0) & g.status.ne("evaluated")).sum()),
            intercept_better_blocks=int(valid.delta_mae_proportional_minus_ols.gt(0).sum()),
            largest_test_block=largest.block if largest is not None else None,
            largest_test_block_n=int(largest.test_n) if largest is not None else 0,
            largest_test_block_fraction=float(largest.test_n / len(p)) if len(p) else np.nan,
        )
        rows.append(dict(**context, weighting="equal_filter", **metrics(p)))
        equal = {}
        for model in MODELS:
            for metric in ["mean_signed_error", "mae"]:
                equal[model + "_" + metric] = valid[model + "_" + metric].mean()
            equal[model + "_rmse"] = np.sqrt((valid[model + "_rmse"] ** 2).mean())
            equal[model + "_nonpositive_prediction_n"] = int(p[model + "_prediction"].le(0).sum())
        equal["delta_mae_proportional_minus_ols"] = equal["proportional_mae"] - equal["ols_mae"]
        equal["intercept_mae_improvement_pct"] = (
            100 * equal["delta_mae_proportional_minus_ols"] / equal["proportional_mae"]
            if equal["proportional_mae"] > 0
            else np.nan
        )
        rows.append(dict(**context, weighting="equal_quarter", **equal))
    return pd.DataFrame(rows)


def id11_sensitivity(previous_blocks, members):
    """Addis only, strictly earlier training; compare only mutually supported folds."""
    source = previous_blocks.loc[
        previous_blocks.variant.eq("baseline")
        & previous_blocks.site.eq("Addis_Ababa")
        & previous_blocks.scheme.eq("later_period")
    ]
    rows, predictions = [], []
    for r in source.to_dict("records"):
        cohort = members.loc[
            members.site.eq("Addis_Ababa") & members.population.eq(r["population"])
        ]
        earlier = cohort.loc[cohort.point_id.isin(json.loads(r["train_ids"]))]
        id11 = earlier.loc[earlier.ftir_CalibrationSetId.eq("11")]
        test = cohort.loc[
            cohort.point_id.isin(json.loads(r["test_ids"])) & cohort.ftir_CalibrationSetId.eq("11")
        ]
        fits = {
            name: fit_training(train)
            for name, train in [("all_earlier", earlier), ("id11_earlier", id11)]
        }
        common = bool(len(test) and all(f["status"] == "evaluated" for f in fits.values()))
        reason = (
            "evaluated"
            if common
            else (
                "empty_id11_test_block"
                if test.empty
                else ";".join(
                    name + ":" + f["status"]
                    for name, f in fits.items()
                    if f["status"] != "evaluated"
                )
            )
        )
        for choice, train in [("all_earlier", earlier), ("id11_earlier", id11)]:
            fit = fits[choice]
            key = dict(
                site="Addis_Ababa",
                population=r["population"],
                scheme="later_period",
                block=r["block"],
                training_choice=choice,
            )
            assert not set(train.point_id) & set(test.point_id)
            if len(train) and len(test):
                assert train.date.max() < test.date.min()
            row = dict(
                **key,
                status=reason,
                own_training_status=fit["status"],
                train_n=len(train),
                test_n=len(test),
                train_ids=ids_json(train),
                test_ids=ids_json(test),
                all_earlier_train_n=len(earlier),
                id11_earlier_train_n=len(id11),
                train_ec_min=train[X].min(),
                train_ec_max=train[X].max(),
                test_ec_min=test[X].min(),
                test_ec_max=test[X].max(),
                train_date_min=train.date.min(),
                train_date_max=train.date.max(),
                test_date_min=test.date.min(),
                test_date_max=test.date.max(),
                previous_split_sha256=r["split_sha256"],
                common_support=common,
                ols_slope=fit["slope"],
                ols_intercept=fit["intercept"],
                training_median=fit["median"],
                proportional_k=proportional_fit(train) if fit["status"] == "evaluated" else np.nan,
            )
            row["split_sha256"] = hashlib.sha256(
                (row["train_ids"] + "/" + row["test_ids"]).encode()
            ).hexdigest()
            p = test.copy()
            for k, v in key.items():
                p[k] = v
            p["status"] = reason
            p["split_sha256"] = row["split_sha256"]
            p["proportional_prediction"] = row["proportional_k"] * p[X] if common else np.nan
            p["ols_prediction"] = fit["intercept"] + fit["slope"] * p[X] if common else np.nan
            p["median_prediction"] = fit["median"] if common else np.nan
            p["outside_training_ec"] = p[X].lt(row["train_ec_min"]) | p[X].gt(row["train_ec_max"])
            p["ols_slope"] = fit["slope"]
            p["ols_intercept"] = fit["intercept"]
            p["proportional_k"] = row["proportional_k"]
            for model in MODELS:
                p[model + "_error"] = p[model + "_prediction"] - p[Y]
            row["outside_training_ec_n"] = (
                int(p.outside_training_ec.sum()) if len(train) else np.nan
            )
            row.update(metrics(p))
            rows.append(row)
            if len(p):
                predictions.append(p)
    blocks = pd.DataFrame(rows)
    pred = pd.concat(predictions, ignore_index=True)
    pair_keys = KEYS + ["block", "point_id"]
    a = pred.loc[pred.training_choice.eq("all_earlier")]
    i = pred.loc[pred.training_choice.eq("id11_earlier")]
    pairs = a.merge(
        i[pair_keys + [m + "_error" for m in MODELS]],
        on=pair_keys,
        suffixes=("_all", "_id11"),
        validate="one_to_one",
    )
    for model in MODELS:
        pairs[model + "_mae_change_id11_minus_all"] = (
            pairs[model + "_error_id11"].abs() - pairs[model + "_error_all"].abs()
        )
    common = pairs.loc[pairs.ols_error_all.notna() & pairs.ols_error_id11.notna()]
    changes = []
    for pop, group in common.groupby("population", sort=False):
        for block, g in [("ALL_COMMON_FILTERS", group), *group.groupby("block", sort=False)]:
            row = dict(population=pop, block=block, common_n=len(g))
            for model in MODELS:
                row[model + "_mae_change_id11_minus_all"] = g[
                    model + "_mae_change_id11_minus_all"
                ].mean()
            changes.append(row)
    return blocks, pred, pairs, pd.DataFrame(changes)
