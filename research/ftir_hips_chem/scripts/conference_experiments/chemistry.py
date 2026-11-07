"""Forward tests of independent chemistry and optical/FTIR response interactions.

The target of the interaction test is optical disagreement, not true EC error.
The chemistry-prediction test accepts documented IC/XRF/ICP-MS measurements;
FTIR carbon and FTIR-derived factors cannot be relabelled as independent data.
All model arms use the same physical filters and training-only preprocessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from config import MAC_VALUE
    from data_matching import _param_to_column_name, base_filter_id, pivot_filter_by_id
    from outliers import apply_exclusion_flags, get_clean_data
    from seasonal_ml_checks import ECLine, regression_scores
    from seasonal_spectral_ml import build_model, make_cycle_folds, season_labels
except ImportError:
    from ..config import MAC_VALUE
    from ..data_matching import _param_to_column_name, base_filter_id, pivot_filter_by_id
    from ..outliers import apply_exclusion_flags, get_clean_data
    from ..seasonal_ml_checks import ECLine, regression_scores
    from ..seasonal_spectral_ml import build_model, make_cycle_folds, season_labels


ION_ANALYTES = frozenset({
    "ammonium", "bromide", "calcium", "chloride", "fluoride", "magnesium", "nitrate",
    "nitrite", "phosphate", "potassium", "sodium", "sulfate",
})
ELEMENT_ANALYTES = frozenset({
    "aluminum", "antimony", "arsenic", "barium", "cadmium", "calcium", "cerium",
    "chlorine", "chromium", "cobalt", "copper", "iron", "lead", "lithium",
    "magnesium", "manganese", "nickel", "phosphorus", "potassium", "rubidium",
    "selenium", "silicon", "sodium", "strontium", "sulfur", "tin", "titanium",
    "vanadium", "zinc",
})
SEED = 20261107


def _mass_unit_factor(units: str) -> float:
    unit = str(units).strip().replace("µ", "u").replace("μ", "u").replace("³", "3")
    if unit not in {"ug/m3", "ng/m3"}:
        raise ValueError("chemical concentration units must be explicit ug/m3 or ng/m3")
    return 1.0 if unit == "ug/m3" else 0.001


@dataclass(frozen=True)
class ChemicalMeasurement:
    """Reviewed method provenance for one independently measured constituent.

    ``source_evidence`` identifies a method table, lab receipt or other reviewed
    record. A ChemSpec parameter name alone does not establish the instrument.
    Values are converted to ug/m3 for reports, with the input unit retained.
    """

    column: str
    analyte: str
    method: str
    units: str
    source_evidence: str
    independence_verified: bool = False

    def __post_init__(self):
        if not isinstance(self.column, str) or not self.column.strip():
            raise ValueError("a chemical measurement requires a column")
        forbidden = ("ftir", "chemspec_ec", "chemspec_oc", "hips", "pmf", "factor", "equivalent_bc")
        if any(token in self.column.lower() for token in forbidden):
            raise ValueError("FTIR carbon, optical BC and derived factors are not independent chemistry")
        allowed = ION_ANALYTES if self.method == "ion_chromatography" else ELEMENT_ANALYTES
        if self.method not in {"ion_chromatography", "xrf", "icp_ms"} or self.analyte not in allowed:
            raise ValueError("independent chemistry requires a supported measured ion or element and method")
        _mass_unit_factor(self.units)
        if not isinstance(self.source_evidence, str) or not self.source_evidence.strip():
            raise ValueError("reviewed chemical method source_evidence is required")
        if not isinstance(self.independence_verified, (bool, np.bool_)) or not self.independence_verified:
            raise ValueError("independence_verified must explicitly be true after method review")


def _required(frame: pd.DataFrame, columns: Sequence[str], label: str = "filters") -> None:
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError(f"{label} must be a DataFrame with unique column names")
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{label} lacks required columns: {sorted(missing)}")


def _explicit_bool(values: pd.Series, name: str) -> pd.Series:
    if not values.dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError(f"{name} must contain explicit booleans, not strings or numbers")
    return values.astype("boolean").fillna(False).astype(bool)


def _flag(frame: pd.DataFrame, mask, reason: str) -> None:
    """Retain every row and its reasons before any analysis population is chosen."""
    mask = pd.Series(np.asarray(mask, dtype=bool), index=frame.index)
    frame.loc[mask, "is_excluded"] = True
    previous = frame.loc[mask, "exclusion_reason"].fillna("").astype(str)
    frame.loc[mask, "exclusion_reason"] = previous.map(
        lambda value: "; ".join(dict.fromkeys([s for s in value.split("; ") if s] + [reason]))
    )


def _base_filters(filters: pd.DataFrame, *, site_name="Addis Ababa", calendar="belg_feb",
                  require_lot=True) -> pd.DataFrame:
    _required(filters, ["physical_id", "date"])
    out = filters.copy().reset_index(drop=True)
    ids = out.physical_id.map(base_filter_id).mask(out.physical_id.isna())
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("one nonmissing physical filter ID is required per row")
    if ids.duplicated().any():
        raise ValueError("duplicate physical filters: aggregate repeat analyses before model evaluation")
    out["physical_id"] = ids
    out["base_filter_id"] = ids
    if any(pd.notna(v) and pd.Timestamp(v).tzinfo is not None for v in out.date):
        raise ValueError("date must be an explicit timezone-naive local sampling date")
    out["date"] = pd.to_datetime(out.date, errors="coerce").dt.normalize()
    supplied_flag = _explicit_bool(out.get("is_excluded", pd.Series(False, index=out.index)), "is_excluded")
    supplied_reason = out.get("exclusion_reason", pd.Series("", index=out.index)).fillna("").astype(str)
    out = apply_exclusion_flags(out, site_name)
    out["registry_excluded"] = out.is_excluded
    for i in out.index[supplied_flag]:
        _flag(out, out.index == i, supplied_reason.at[i] or "upstream_exclusion")
    if "is_outlier" in out:
        _flag(out, _explicit_bool(out.is_outlier, "is_outlier"), "upstream_threshold_flag")
    _flag(out, out.date.isna(), "missing_local_sampling_date")
    if require_lot:
        if "lot" not in out:
            out["lot"] = pd.NA
        lot = out.lot.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
        _flag(out, lot.isna() | lot.fillna("").str.lower().isin(["", "unknown", "nan", "none"]), "unknown_filter_lot")
        out["lot"] = lot.fillna("unknown").astype(str)
    out["season"] = "unknown"
    dated = out.date.notna()
    if dated.any():
        out.loc[dated, "season"] = season_labels(out.loc[dated, "date"], calendar)
    out["calendar"] = calendar
    return out


def _finite_flags(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    _required(frame, columns)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        _flag(frame, ~np.isfinite(frame[column]), f"missing_or_nonfinite:{column}")


def _clean_population(frame: pd.DataFrame) -> pd.DataFrame:
    frame["eligible_for_analysis"] = ~frame.is_excluded
    return get_clean_data(frame).reset_index(drop=True)


def _utc(values, name: str) -> pd.DatetimeIndex:
    if any(pd.isna(v) or pd.Timestamp(v).tzinfo is None for v in values):
        raise ValueError(f"{name} must contain complete timezone-aware timestamps")
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def _split_folds(frame, *, calendar, min_train, min_test, test_coverage, group_column=None):
    """Canonical forward cycles, with event and overlapping-window purges."""
    folds, inventory = make_cycle_folds(
        frame.date, frame.physical_id, convention=calendar, mode="forward",
        train_coverage="all_seasons", test_coverage=test_coverage,
        min_train=min_train, min_test=min_test,
    )
    result, audit = [], []
    for fold in folds:
        train, test = fold.train.copy(), fold.test.copy()
        keep = np.ones(len(train), bool)
        if group_column is not None:
            _required(frame, [group_column])
            if frame[group_column].isna().any() or frame[group_column].astype(str).eq("").any():
                raise ValueError("complete event/block IDs are required to keep episodes together")
            keep &= ~frame.iloc[train][group_column].isin(frame.iloc[test][group_column]).to_numpy()
        if {"window_start_utc", "window_end_utc"} <= set(frame):
            starts = _utc(frame.window_start_utc, "window_start_utc")
            ends = _utc(frame.window_end_utc, "window_end_utc")
            keep &= np.asarray(ends[train] <= starts[test].min())
        retained = train[keep]
        reason = "" if len(retained) >= min_train else "insufficient_training_after_episode_or_window_purge"
        audit.append({
            "fold": fold.name, "initial_n_train": len(train), "n_train": len(retained),
            "n_test": len(test), "purged_train_ids": frame.iloc[train[~keep]].physical_id.tolist(),
            "status": "eligible" if not reason else "blocked", "reason": reason,
        })
        if not reason:
            result.append(replace(fold, train=retained))
    return result, inventory, pd.DataFrame(audit)


def _paired_differences(predictions: pd.DataFrame, *, baseline: str, bootstrap_reps: int,
                        seed: int) -> pd.DataFrame:
    """Paired errors with block resampling, never independent minute resampling."""
    if predictions.empty:
        return pd.DataFrame()
    if bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be nonnegative")
    rows = []
    rng = np.random.default_rng(seed)
    for fold, group in predictions.groupby("fold", sort=True):
        base = group.loc[group.arm.eq(baseline)].set_index("physical_id")
        for arm, candidate in group.loc[~group.arm.eq(baseline)].groupby("arm", sort=True):
            candidate = candidate.set_index("physical_id").reindex(base.index)
            if candidate.prediction.isna().any() or len(candidate) != len(base):
                raise ValueError("paired model arms must retain exactly the same test filters")
            actual = base.actual.to_numpy(float)
            bp, cp = base.prediction.to_numpy(float), candidate.prediction.to_numpy(float)
            bs, cs = regression_scores(actual, bp), regression_scores(actual, cp)
            record = {"fold": fold, "baseline": baseline, "arm": arm, "n_filters": len(base)}
            for metric in ("rmse", "mae", "bias"):
                record[f"delta_{metric}"] = cs[metric] - bs[metric]
            blocks = base.block_id.astype(str).to_numpy()
            unique = np.unique(blocks)
            record["n_blocks"] = len(unique)
            record["bootstrap_reps"] = bootstrap_reps if len(unique) >= 3 else 0
            deltas = []
            if len(unique) >= 3:
                for _ in range(bootstrap_reps):
                    selected = rng.choice(unique, size=len(unique), replace=True)
                    indices = np.concatenate([np.flatnonzero(blocks == block) for block in selected])
                    delta = regression_scores(actual[indices], cp[indices])["rmse"] - regression_scores(actual[indices], bp[indices])["rmse"]
                    deltas.append(delta)
            record["delta_rmse_ci_low"] = float(np.quantile(deltas, 0.025)) if deltas else np.nan
            record["delta_rmse_ci_high"] = float(np.quantile(deltas, 0.975)) if deltas else np.nan
            record["uncertainty_note"] = "paired block bootstrap; exploratory" if deltas else "insufficient blocks or bootstrap disabled"
            rows.append(record)
    return pd.DataFrame(rows)


def _result(summary, tables):
    return {"summary": summary, "tables": tables}


def _evaluate_residual_models(
    audit, *, extra_arms: Mapping[str, Sequence[str]], numeric_controls: Sequence[str],
    calendar="belg_feb", min_train=30, min_test=6, test_coverage="all_months",
    group_column=None, alpha=1.0, bootstrap_reps=300, seed=SEED, fold_transform=None,
):
    """Shared train-only EC-line-residual protocol for chemistry and rain tests."""
    forbidden = {"hips_fabs_mm1", "hips_bc_ugm3", "d", "disagreement", "target"}
    if forbidden & {str(name).lower() for name in numeric_controls}:
        raise ValueError("the response or its HIPS component cannot be a baseline predictor")
    all_extra = list(dict.fromkeys(c for columns in extra_arms.values() for c in columns))
    _required(audit, ["ftir_ec_ugm3", "hips_fabs_Mm1", "hips_source", "lot"])
    _finite_flags(audit, ["ftir_ec_ugm3", "hips_fabs_Mm1", *numeric_controls])
    _flag(audit, ~audit.hips_source.eq("direct_HIPS"), "direct_HIPS_required_not_SSR_or_public_BC")
    if fold_transform is None:
        _finite_flags(audit, all_extra)
    clean = _clean_population(audit)
    summary = {
        "status": "insufficient_data", "n_input": len(audit), "n_eligible": len(clean),
        "target": "D = FTIR EC - direct HIPS absorption / MAC", "target_units": "ug/m3",
        "mac_m2_g": float(MAC_VALUE), "calendar": calendar,
        "parameters": {"min_train": min_train, "min_test": min_test, "test_coverage": test_coverage,
                       "ridge_alpha": alpha, "seed": seed, "bootstrap_reps": bootstrap_reps,
                       "numeric_controls": list(numeric_controls), "categorical_controls": ["lot", "season"],
                       "extra_arms": {k: list(v) for k, v in extra_arms.items()}, "block_column": group_column},
        "limitations": [
            "D contains FTIR-derived EC and an optical BC conversion; this is not independent EC validation.",
            "Prespecified forward comparisons are exploratory; no coefficients are deployed as corrections.",
            "Composition associations cannot distinguish calibration sensitivity from aerosol optical changes.",
            "Bootstrap intervals condition on the declared blocks; physical-filter blocks do not model longer temporal dependence.",
            "The summary lists the actual controls; PM concentration alone is not filter areal loading, and season alone is not measured weather.",
        ],
    }
    tables = {"eligibility": audit, "metrics": pd.DataFrame(), "predictions": pd.DataFrame(),
              "paired_differences": pd.DataFrame(), "folds": pd.DataFrame(), "cycle_inventory": pd.DataFrame(),
              "training_parameters": pd.DataFrame()}
    if len(clean) < min_train + min_test:
        summary["reason"] = "insufficient eligible physical filters"
        return _result(summary, tables)
    folds, inventory, fold_audit = _split_folds(
        clean, calendar=calendar, min_train=min_train, min_test=min_test,
        test_coverage=test_coverage, group_column=group_column,
    )
    tables.update(cycle_inventory=inventory, folds=fold_audit)
    predictions, metrics, parameters = [], [], []
    for fold in folds:
        train, test = clean.iloc[fold.train].copy(), clean.iloc[fold.test].copy()
        if train.ftir_ec_ugm3.nunique() < 2:
            parameters.append({"fold": fold.name, "status": "blocked", "reason": "constant_training_FTIR_EC"})
            continue
        train_d = (train.ftir_ec_ugm3 - train.hips_fabs_Mm1 / MAC_VALUE).to_numpy(float)
        test_d = (test.ftir_ec_ugm3 - test.hips_fabs_Mm1 / MAC_VALUE).to_numpy(float)
        line = ECLine().fit(train.ftir_ec_ugm3.to_numpy(float), train_d)
        train_line = line.predict(train.ftir_ec_ugm3)
        test_line = line.predict(test.ftir_ec_ugm3)
        p = {"fold": fold.name, "status": "completed", "ec_line_slope": line.slope_, "ec_line_intercept": line.intercept_}
        if fold_transform is not None:
            train, test, learned = fold_transform(train, test)
            p.update(learned)
        parameters.append(p)
        arm_predictions = {"ec_line": test_line}
        arms = {"controlled_baseline": (), **extra_arms}
        for name, columns in arms.items():
            model = build_model(
                task="regression", model="linear", numeric_metadata=[*numeric_controls, *columns],
                categorical_metadata=["lot", "season"], alpha=alpha, seed=seed,
            )
            model.fit(train, train_d - train_line)
            arm_predictions[name] = test_line + np.asarray(model.predict(test), float)
        for arm, predicted in arm_predictions.items():
            metrics.append({"fold": fold.name, "arm": arm, "n_train": len(train), "n_test": len(test),
                            "complete_test_cycle": fold.complete_test, **regression_scores(test_d, predicted)})
            for j, row in enumerate(test.itertuples()):
                block = str(getattr(row, group_column)) if group_column else row.physical_id
                predictions.append({"fold": fold.name, "arm": arm, "physical_id": row.physical_id,
                                    "date": row.date, "season": row.season, "lot": row.lot,
                                    "block_id": block, "actual": test_d[j], "prediction": float(predicted[j]),
                                    "ec_line_residual": float(test_d[j] - test_line[j]),
                                    "predicted_residual": float(predicted[j] - test_line[j])})
    tables.update(metrics=pd.DataFrame(metrics), predictions=pd.DataFrame(predictions), training_parameters=pd.DataFrame(parameters))
    tables["paired_differences"] = _paired_differences(tables["predictions"], baseline="controlled_baseline", bootstrap_reps=bootstrap_reps, seed=seed)
    summary.update(status="completed" if predictions else "insufficient_data", n_folds=len({p["fold"] for p in predictions}),
                   n_test_filters=len({p["physical_id"] for p in predictions}))
    if not predictions:
        summary["reason"] = "no eligible forward cycles after coverage and event/window checks"
    return _result(summary, tables)


def evaluate_regime_interactions(
    filters: pd.DataFrame, *, chemistry: ChemicalMeasurement,
    numeric_controls: Sequence[str] = ("pm25_ugm3",), site_name="Addis Ababa",
    calendar="belg_feb", min_train=30, min_test=6, test_coverage="all_months",
    alpha=1.0, bootstrap_reps=300, seed=SEED,
) -> dict:
    """Compare EC-line/control, chemistry-main-effect and slope-interaction arms.

    Chemistry and FTIR EC are centered on the training rows before constructing
    their product. This tests a continuous prespecified interaction, without
    searching a chemical cutoff on test outcomes. Predictions are never clipped.
    ``hips_source`` must explicitly identify direct HIPS observations.
    """
    _required(filters, [chemistry.column])
    if chemistry.column in numeric_controls:
        raise ValueError("the prespecified chemistry column must not already be in baseline controls")
    audit = _base_filters(filters, site_name=site_name, calendar=calendar)
    audit["chemical_ugm3"] = pd.to_numeric(audit[chemistry.column], errors="coerce") * _mass_unit_factor(chemistry.units)
    _finite_flags(audit, ["chemical_ugm3"])

    def transform(train, test):
        cm, em = float(train.chemical_ugm3.mean()), float(train.ftir_ec_ugm3.mean())
        for frame in (train, test):
            frame["chemical_centered"] = frame.chemical_ugm3 - cm
            frame["ec_chemistry_interaction"] = (frame.ftir_ec_ugm3 - em) * frame.chemical_centered
        return train, test, {"chemical_training_mean_ugm3": cm, "ec_training_mean_ugm3": em}

    result = _evaluate_residual_models(
        audit, extra_arms={"chemistry_main": ["chemical_centered"],
                           "chemistry_slope_interaction": ["chemical_centered", "ec_chemistry_interaction"]},
        numeric_controls=numeric_controls, calendar=calendar, min_train=min_train,
        min_test=min_test, test_coverage=test_coverage, alpha=alpha,
        bootstrap_reps=bootstrap_reps, seed=seed, fold_transform=transform,
    )
    result["summary"].update(experiment="chemistry_regime_interaction", chemistry=asdict(chemistry))
    # The interaction must improve on the chemistry main effect as well as the baseline.
    extra = _paired_differences(result["tables"]["predictions"], baseline="chemistry_main", bootstrap_reps=bootstrap_reps, seed=seed)
    if not extra.empty:
        extra = extra.loc[extra.arm.eq("chemistry_slope_interaction")]
        result["tables"]["paired_differences"] = pd.concat([result["tables"]["paired_differences"], extra], ignore_index=True)
    return result


def _optical_window_flags(audit, minimum_coverage, minimum_segment_coverage):
    if not 0 < minimum_coverage <= 1 or not 0 < minimum_segment_coverage <= 1:
        raise ValueError("coverage thresholds must be in (0, 1]")
    for column in ("sampling_schedule_verified", "observation_provenance_verified", "processing_history_verified"):
        if column not in audit:
            audit[column] = False
        _flag(audit, ~_explicit_bool(audit[column], column), f"unverified:{column}")
    for column, threshold in [("valid_observed_coverage_fraction", minimum_coverage),
                              ("minimum_segment_valid_observed_coverage", minimum_segment_coverage)]:
        if column not in audit:
            audit[column] = np.nan
        values = pd.to_numeric(audit[column], errors="coerce")
        _flag(audit, ~np.isfinite(values) | values.lt(threshold) | values.gt(1), f"insufficient:{column}")
    if "window_evidence_id" not in audit:
        audit["window_evidence_id"] = ""
    _flag(audit, audit.window_evidence_id.isna() | audit.window_evidence_id.astype(str).str.strip().eq(""), "missing_window_evidence_id")
    for column in ("window_start_utc", "window_end_utc"):
        if column not in audit:
            audit[column] = pd.NaT
        bad = audit[column].map(lambda v: pd.isna(v) or pd.Timestamp(v).tzinfo is None)
        _flag(audit, bad, f"missing_or_naive:{column}")
        audit[column] = pd.to_datetime(audit[column].where(~bad), utc=True)
    _flag(audit, audit.window_end_utc.le(audit.window_start_utc), "nonpositive_collection_window")
    if "optical_quantity" not in audit:
        audit["optical_quantity"] = "unknown"
    _flag(audit, ~audit.optical_quantity.isin(["b_ATN", "b_abs"]), "optical_quantity_must_be_explicit_b_ATN_or_b_abs")
    if "optical_units" not in audit:
        audit["optical_units"] = "unknown"
    _flag(audit, ~audit.optical_units.isin(["Mm-1", "Mm^-1", "Mm⁻¹"]), "optical_units_must_be_inverse_megametres")
    if audit.loc[~audit.is_excluded, "optical_quantity"].nunique() > 1:
        raise ValueError("do not pool attenuation-scale and corrected absorption features in one experiment")


def evaluate_chemistry_prediction(
    filters: pd.DataFrame, *, target: ChemicalMeasurement,
    baseline_numeric: Sequence[str], shape_columns: Sequence[str],
    temporal_columns: Sequence[str] = (), site_name="Addis Ababa", calendar="belg_feb",
    min_train=30, min_test=6, test_coverage="all_months", minimum_coverage=0.75,
    minimum_segment_coverage=0.5, alpha=1.0, bootstrap_reps=300, seed=SEED,
) -> dict:
    """Predict independent measured chemistry at verified collection-window scale.

    The caller obtains optical features over documented *active intervals*, not
    a 9am daily fallback or inferred continuous envelope. Input evidence and
    coverage gates are mandatory, and no hourly/minute source claim is made.
    """
    chosen = list(baseline_numeric) + list(shape_columns) + list(temporal_columns)
    if not baseline_numeric or not shape_columns or len(set(chosen)) != len(chosen):
        raise ValueError("distinct nonempty baseline and spectral-shape columns are required")
    if target.column in chosen:
        raise ValueError("chemical target cannot be a predictor")
    if any(any(token in name.lower() for token in ("ftir", "chemspec", "pmf", "factor", "hips", "target")) for name in chosen):
        raise ValueError("use routine optical/weather predictors, not chemical targets, FTIR products or factors")
    _required(filters, [target.column, *chosen])
    audit = _base_filters(filters, site_name=site_name, calendar=calendar)
    audit["chemical_target_ugm3"] = pd.to_numeric(audit[target.column], errors="coerce") * _mass_unit_factor(target.units)
    _finite_flags(audit, ["chemical_target_ugm3", *chosen])
    _optical_window_flags(audit, minimum_coverage, minimum_segment_coverage)
    clean = _clean_population(audit)
    summary = {
        "experiment": "optics_predict_independent_chemistry", "status": "blocked",
        "n_input": len(audit), "n_eligible": len(clean), "target": asdict(target), "target_units": "ug/m3",
        "calendar": calendar, "parameters": {"baseline_numeric": list(baseline_numeric),
            "shape_columns": list(shape_columns), "temporal_columns": list(temporal_columns),
            "minimum_coverage": minimum_coverage, "minimum_segment_coverage": minimum_segment_coverage,
            "min_train": min_train, "min_test": min_test, "test_coverage": test_coverage,
            "ridge_alpha": alpha, "bootstrap_reps": bootstrap_reps, "seed": seed},
        "limitations": [
            "Results apply to the verified filter collection timescale, not minute-level source mass.",
            "Attenuation-scale b_ATN features are not automatically corrected absorption b_abs.",
            "Independent measured chemistry does not provide a thermal EC accuracy reference.",
            "Physical-filter bootstrap intervals do not account for longer temporal dependence between filters.",
        ],
    }
    tables = {"eligibility": audit, "metrics": pd.DataFrame(), "predictions": pd.DataFrame(),
              "paired_differences": pd.DataFrame(), "folds": pd.DataFrame(), "cycle_inventory": pd.DataFrame()}
    if len(clean) < min_train + min_test:
        summary["reason"] = "insufficient filters passing independent-chemistry and verified active-window evidence gates"
        return _result(summary, tables)
    folds, inventory, fold_audit = _split_folds(clean, calendar=calendar, min_train=min_train, min_test=min_test, test_coverage=test_coverage)
    tables.update(cycle_inventory=inventory, folds=fold_audit)
    arms = {"routine_baseline": list(baseline_numeric), "plus_optical_shape": [*baseline_numeric, *shape_columns]}
    if temporal_columns:
        arms["plus_optical_temporal"] = [*baseline_numeric, *shape_columns, *temporal_columns]
    predictions, metrics = [], []
    for fold in folds:
        train, test = clean.iloc[fold.train], clean.iloc[fold.test]
        for arm, columns in arms.items():
            model = build_model(task="regression", model="linear", numeric_metadata=columns,
                                categorical_metadata=["lot", "season"], alpha=alpha, seed=seed)
            model.fit(train, train.chemical_target_ugm3)
            predicted = np.asarray(model.predict(test), float)
            actual = test.chemical_target_ugm3.to_numpy(float)
            metrics.append({"fold": fold.name, "arm": arm, "n_train": len(train), "n_test": len(test),
                            "complete_test_cycle": fold.complete_test, **regression_scores(actual, predicted)})
            for j, row in enumerate(test.itertuples()):
                predictions.append({"fold": fold.name, "arm": arm, "physical_id": row.physical_id,
                                    "date": row.date, "season": row.season, "lot": row.lot,
                                    "block_id": row.physical_id, "actual": actual[j], "prediction": predicted[j]})
    tables.update(metrics=pd.DataFrame(metrics), predictions=pd.DataFrame(predictions))
    tables["paired_differences"] = _paired_differences(tables["predictions"], baseline="routine_baseline", bootstrap_reps=bootstrap_reps, seed=seed)
    summary.update(status="completed" if predictions else "insufficient_data", n_folds=len(folds),
                   n_test_filters=len({r["physical_id"] for r in predictions}))
    if not predictions:
        summary["reason"] = "no eligible forward cycles after collection-window and coverage checks"
    return _result(summary, tables)


def adapt_unified_filters(
    long: pd.DataFrame, *, chemical_specs: Mapping[str, ChemicalMeasurement],
    site_code="ETAD", site_name="Addis Ababa",
) -> pd.DataFrame:
    """Audit the tracked long table, then use the canonical physical-filter pivot.

    Conflicting dates, lots, units or repeated values are flagged and cannot be
    resolved by the pivot's historical ``first`` rule. Direct HIPS is admitted
    only from ``HIPS_Fabs``/``DataSource=HIPS``; public BC/SSR never substitutes.
    Chemical method provenance is supplied from a reviewed external method
    record because the unified table does not retain that method field.
    """
    _required(long, ["Site", "FilterId", "SampleDate", "LotId", "FilterType", "Parameter", "Concentration", "Concentration_Units", "DataSource"], "unified filter table")
    if len({spec.column for spec in chemical_specs.values()}) != len(chemical_specs):
        raise ValueError("chemical output columns must be distinct")
    for parameter, spec in chemical_specs.items():
        suffix = "_Ion_PM2.5" if spec.method == "ion_chromatography" else "_PM2.5"
        expected = f"ChemSpec_{spec.analyte.capitalize()}{suffix}"
        if parameter != expected:
            raise ValueError("unified chemical parameter must agree with the reviewed analyte and method")
    reserved = {"physical_id", "date", "lot", "ftir_ec_ugm3", "hips_fabs_Mm1", "pm25_ugm3", "hips_source"}
    if reserved & {s.column for s in chemical_specs.values()}:
        raise ValueError("chemical output column collides with canonical metadata")
    core = {"EC_ftir": "ftir_ec_ugm3", "HIPS_Fabs": "hips_fabs_Mm1", "ChemSpec_Filter_PM2.5_mass": "pm25_ugm3"}
    mapping = {**core, **{p: spec.column for p, spec in chemical_specs.items()}}
    data = long.loc[long.Site.eq(site_code)].copy()
    data["physical_id"] = data.FilterId.map(base_filter_id).mask(data.FilterId.isna())
    if data.physical_id.isna().any():
        raise ValueError("unified table contains missing physical IDs")
    catalog, measurements = [], []
    for physical_id, group in data.groupby("physical_id", sort=True):
        why = []
        dates = pd.to_datetime(group.SampleDate, errors="coerce").dropna().dt.normalize().unique()
        date = pd.Timestamp(dates[0]) if len(dates) == 1 else pd.NaT
        if len(dates) != 1:
            why.append("conflicting_or_missing_sampling_date")
        lots = group.LotId.astype("string").str.replace(r"\.0$", "", regex=True).str.strip().dropna()
        lots = lots.loc[~lots.str.lower().isin(["", "unknown", "none", "nan"])].unique()
        lot = str(lots[0]) if len(lots) == 1 else "unknown"
        if len(lots) != 1:
            why.append("conflicting_or_missing_lot")
        types = set(group.FilterType.dropna().astype(str))
        if types != {"PM2.5"}:
            why.append("filter_type_not_unambiguous_PM2.5")
        # Retain the archived physical-filter sampling metadata separately from
        # chemistry provenance. It supports a loading calculation, not proof of
        # the pump's actual start/stop intervals.
        metadata, metadata_reasons = {}, []
        for source, dest in (("Volume_m3", "collection_volume_m3"), ("DepositArea_cm2", "deposit_area_cm2")):
            available = group.loc[group.Parameter.eq("EC_ftir") & group.DataSource.eq("FTIR")]
            values = pd.to_numeric(available.get(source, pd.Series(dtype=float)), errors="coerce")
            known = values[np.isfinite(values) & values.gt(0)].unique()
            metadata[dest] = float(known[0]) if len(known) == 1 else np.nan
            if len(known) != 1:
                metadata_reasons.append(f"conflicting_or_missing:{dest}")
        direct = False
        for parameter, column in mapping.items():
            selected = group.loc[group.Parameter.eq(parameter)]
            if selected.empty:
                continue
            values, bad_units = [], False
            for row in selected.itertuples():
                value = pd.to_numeric(pd.Series([row.Concentration]), errors="coerce").iloc[0]
                if parameter == "HIPS_Fabs":
                    units_ok = row.Concentration_Units in {"Mm-1", "Mm^-1", "Mm⁻¹"}
                    factor = 1.0
                else:
                    try:
                        factor = _mass_unit_factor(row.Concentration_Units)
                        units_ok = True
                    except ValueError:
                        units_ok, factor = False, np.nan
                bad_units |= not units_ok
                if np.isfinite(value) and units_ok:
                    values.append(float(value) * factor)
            unique = np.unique(values)
            value = float(unique[0]) if len(unique) == 1 and not bad_units else np.nan
            if bad_units:
                why.append(f"invalid_units:{column}")
            if len(unique) > 1:
                why.append(f"conflicting_repeat_values:{column}")
            if parameter == "HIPS_Fabs":
                direct = selected.DataSource.fillna("").str.lower().eq("hips").all() and not bad_units and len(unique) == 1
                if not direct:
                    why.append("unverified_direct_HIPS_provenance")
            if parameter == "EC_ftir" and not selected.DataSource.fillna("").str.lower().eq("ftir").all():
                value = np.nan
                why.append("unverified_FTIR_provenance")
            if parameter in chemical_specs:
                if not selected.DataSource.fillna("").str.lower().eq("chemspec").all():
                    value = np.nan
                    why.append(f"unverified_chemical_provenance:{column}")
                # The canonical pivot receives values in the spec's declared unit.
                value /= _mass_unit_factor(chemical_specs[parameter].units)
            measurements.append({"Site": site_code, "base_filter_id": physical_id, "SampleDate": date,
                                 "Parameter": parameter, "Concentration": value})
        catalog.append({"physical_id": physical_id, "date": date, "lot": lot, "site": site_name,
                        "hips_source": "direct_HIPS" if direct else "unverified",
                        "is_excluded": bool(why), "exclusion_reason": "; ".join(why),
                        **metadata, "loading_metadata_reason": "; ".join(metadata_reasons)})
    output = pd.DataFrame(catalog, columns=["physical_id", "date", "lot", "site", "hips_source", "is_excluded", "exclusion_reason",
                                          "collection_volume_m3", "deposit_area_cm2", "loading_metadata_reason"])
    if measurements:
        pivoted = pivot_filter_by_id(pd.DataFrame(measurements), site_code, params=list(mapping), hips_units="both")
        pivoted = pivoted.rename(columns={"base_filter_id": "physical_id", **{_param_to_column_name(p): c for p, c in mapping.items()}})
        output = output.merge(pivoted.drop(columns=["date"], errors="ignore"), on="physical_id", how="left", validate="one_to_one")
    for column in mapping.values():
        if column not in output:
            output[column] = np.nan
    output["pm25_areal_ugcm2"] = output.pm25_ugm3 * output.collection_volume_m3 / output.deposit_area_cm2
    output["loading_derivation"] = "PM25 ug/m3 * archived collection volume m3 / deposit area cm2; not active-interval verification"
    if len(output):
        existing = output.exclusion_reason.copy()
        flags = output.is_excluded.copy()
        output = apply_exclusion_flags(output.assign(base_filter_id=output.physical_id), site_name)
        for i in output.index[flags]:
            _flag(output, output.index == i, existing.at[i])
    return output
