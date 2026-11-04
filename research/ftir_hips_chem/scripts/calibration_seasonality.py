"""Season-aware description of frozen EC calibration transfer predictions.

The Ethiopian target has an optical comparator, not independent thermal EC.
This module never fits models, chooses components, adjusts MAC, or changes
exclusions. Every scan aggregation remains labelled. Calendar labels are fixed
month bins; they do not classify observed rain on the sampling day.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from config import resolve_seasons, season_convention_name, season_for_month


CALENDARS = ("belg_feb", "dry_feb")
PRIMARY_CALENDAR = "belg_feb"
UNDATED = "Undated"
PUBLISHED_CONTEXT = "published_ftir_context"
CLAIM_LIMIT = "Optical agreement diagnostic; not independent Ethiopian EC accuracy"
IDENTITY = ["arm", "aggregation_sensitivity", "SiteCode", "base_filter_id"]
METADATA = [
    "MediaId", "SiteCode", "base_filter_id", "native_date", "SamplingEndDate",
    "ExternalFilterType", "lot_group", "n_scans", "SampleVolume_m3",
    "volume_valid", "current_date", "current_pair_eligible",
    "current_optical_source", "current_optical_bc_ugm3",
    "current_ec_ftir_ugm3", "hips_fabs_Mm1", "prior_problem_filter",
]


def _assert_inputs(frame: pd.DataFrame) -> None:
    required = set(IDENTITY + METADATA + ["prediction_ugm3"])
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"missing seasonal input columns: {missing}")
    if frame.empty or frame[IDENTITY].isna().any().any():
        raise ValueError("nonempty arm and physical-filter identities are required")
    if frame.duplicated(IDENTITY).any():
        raise ValueError("duplicate physical filters within an arm/aggregation")
    if not frame.SiteCode.isin(("ETAD", "ETBI")).all():
        raise ValueError("Ethiopian seasons cannot label non-Ethiopian source sites")
    # A method must not change the comparator, collection date, volume, or lot.
    for _, group in frame.groupby(["aggregation_sensitivity", "SiteCode", "base_filter_id"], sort=False):
        inconsistent = [name for name in METADATA if group[name].nunique(dropna=False) > 1]
        if inconsistent:
            raise ValueError(f"inconsistent physical metadata across arms: {inconsistent}")
    for name in ("volume_valid", "current_pair_eligible", "prior_problem_filter"):
        if not frame[name].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"{name} must contain explicit booleans")


def with_published_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the already published FTIR value as labelled context, once per filter.

    This product was not recalibrated by the matched experiment. Its relation
    to HIPS is descriptive and is never included in source validation scores.
    """
    _assert_inputs(frame)
    if frame.arm.eq(PUBLISHED_CONTEXT).any():
        raise ValueError("published contextual arm is already present")
    out = frame.copy(deep=True)
    out["prediction_product"] = "matched_source_calibration"
    context = frame.drop_duplicates(["aggregation_sensitivity", "SiteCode", "base_filter_id"]).copy()
    context["arm"] = PUBLISHED_CONTEXT
    context["prediction_product"] = "current_published_FTIR_context_not_recalibrated"
    context["prediction_ugm3"] = context.current_ec_ftir_ugm3
    if "prediction_loading_ug" in context:
        context["prediction_loading_ug"] = np.nan
    # Do not preserve an old arm's residual under the contextual product label.
    if "experiment_ftir_minus_hips_ugm3" in context:
        context["experiment_ftir_minus_hips_ugm3"] = (
            context.prediction_ugm3 - context.current_optical_bc_ugm3
        ).where(context.current_pair_eligible)
    return pd.concat([out, context], ignore_index=True)


def season_ledger(frame: pd.DataFrame, conventions: Iterable[str] = CALENDARS) -> pd.DataFrame:
    """Retain every physical prediction with explicit calendar and eligibility.

    Inputs must already be aggregated by physical filter, with the aggregation
    recipe named. No repeat is selected by proximity to the optical value.
    Missing dates remain ``Undated`` and cannot enter a named seasonal result.
    """
    _assert_inputs(frame)
    conventions = tuple(conventions)
    if not conventions or len(set(conventions)) != len(conventions):
        raise ValueError("provide distinct named season conventions")
    dates = pd.to_datetime(frame.native_date, errors="raise")
    end_dates = pd.to_datetime(frame.SamplingEndDate, errors="raise")
    current_dates = pd.to_datetime(frame.current_date, errors="raise")
    if ((end_dates < dates) & dates.notna() & end_dates.notna()).any():
        raise ValueError("sampling end precedes sampling start")
    outputs = []
    for convention in conventions:
        seasons = resolve_seasons(convention)
        if season_convention_name(seasons) != convention:
            raise ValueError("a registered calendar name is required")
        out = frame.copy(deep=True)
        out["season_convention"] = convention
        out["parsed_date"] = dates
        out["calendar_year"] = dates.dt.year.astype("Int64")
        out["calendar_month"] = dates.dt.month.astype("Int64")
        out["month_block"] = dates.dt.strftime("%Y-%m")
        out["season"] = dates.dt.month.map(lambda month: season_for_month(month, seasons)).fillna(UNDATED)
        # Normalized family is only a comparison key; exact labels remain above.
        out["season_family"] = out.season.map(
            lambda label: "Dry/Bega" if label.startswith(("Dry", "Bega")) else label.split(" (")[0]
        )
        end_seasons = end_dates.dt.month.map(lambda month: season_for_month(month, seasons))
        out["spans_season_boundary"] = dates.notna() & end_dates.notna() & out.season.ne(end_seasons)
        out["date_missing"] = dates.isna()
        out["current_date_matches_native"] = dates.notna() & current_dates.notna() & dates.eq(current_dates)
        out["is_pm25"] = out.ExternalFilterType.eq("PM2.5")
        out["direct_hips_available"] = out.current_optical_source.eq("direct_hips_fabs") & np.isfinite(out.hips_fabs_Mm1)
        out["prediction_finite"] = np.isfinite(out.prediction_ugm3)
        out["base_direct_eligible"] = (
            out.is_pm25 & out.volume_valid & ~out.date_missing
            & out.current_date_matches_native & out.current_pair_eligible & out.direct_hips_available
        )
        out["direct_prediction_eligible"] = out.base_direct_eligible & out.prediction_finite
        group_keys = ["aggregation_sensitivity", "SiteCode", "base_filter_id"]
        expected_arms = out.arm.nunique()
        groups = out.groupby(group_keys, sort=False)
        out["common_arm_direct_eligible"] = (
            groups.direct_prediction_eligible.transform("all")
            & groups.arm.transform("nunique").eq(expected_arms)
        )
        reasons = []
        for row in out.itertuples(index=False):
            flags = [
                (not row.is_pm25, "not_PM2.5"),
                (row.date_missing, "missing_native_date"),
                (not row.volume_valid, "missing_or_invalid_native_volume"),
                (not row.current_date_matches_native, "current_date_missing_or_mismatch"),
                (not row.current_pair_eligible, "not_in_frozen_current_pair_cohort"),
                (not row.direct_hips_available, "direct_HIPS_unavailable"),
                (not row.prediction_finite, "nonfinite_prediction"),
                (not row.common_arm_direct_eligible and row.direct_prediction_eligible,
                 "not_complete_across_all_arms"),
            ]
            reasons.append(";".join(reason for flag, reason in flags if flag) or "eligible")
        out["seasonal_pair_reason"] = reasons
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def coverage_summary(ledger: pd.DataFrame, *, by_lot: bool = False, by_year: bool = False) -> pd.DataFrame:
    """Count physical filters once, preserving distinct aggregation sensitivities."""
    identity = ["season_convention", "aggregation_sensitivity", "SiteCode", "base_filter_id"]
    frame = ledger.drop_duplicates(identity)
    group_keys = ["season_convention", "aggregation_sensitivity", "SiteCode", "ExternalFilterType", "season"]
    if by_lot:
        group_keys.append("lot_group")
    if by_year:
        group_keys.append("calendar_year")
    rows = []
    for key, part in frame.groupby(group_keys, sort=True, dropna=False):
        rows.append(dict(zip(group_keys, key)) | {
            "n_physical_filters": len(part), "n_original_scans": int(part.n_scans.sum()),
            "n_direct_pairs": int(part.base_direct_eligible.sum()),
            "n_common_arm_direct_pairs": int(part.common_arm_direct_eligible.sum()),
            "n_missing_native_date": int(part.date_missing.sum()),
            "n_invalid_native_volume": int((~part.volume_valid).sum()),
            "n_prior_problem_filters": int(part.prior_problem_filter.sum()),
            "n_spanning_season_boundary": int(part.spans_season_boundary.sum()),
            "n_calendar_month_blocks": int(part.month_block.nunique()),
            "date_start": part.parsed_date.min(), "date_end": part.parsed_date.max(),
        })
    return pd.DataFrame(rows)


def mac_scenario_ledger(ledger: pd.DataFrame, macs: Iterable[float] = (7.5, 10.0, 12.5, 15.0)) -> pd.DataFrame:
    """Apply predeclared MAC scenarios only to direct measured HIPS Fabs.

    The scenarios are not estimated MAC values; extension products never
    substitute for an absent Fabs measurement. Signed predictions are retained.
    """
    macs = np.asarray(tuple(macs), dtype=float)
    if not len(macs) or not np.isfinite(macs).all() or (macs <= 0).any() or len(np.unique(macs)) != len(macs):
        raise ValueError("MAC scenarios must be distinct finite positive values")
    outputs = []
    for mac in macs:
        out = ledger.copy()
        out["scenario_mac_m2_g"] = float(mac)
        eligible = out.common_arm_direct_eligible
        out["scenario_optical_bc_ugm3"] = (out.hips_fabs_Mm1 / mac).where(eligible)
        out["scenario_ftir_minus_optical_ugm3"] = (
            out.prediction_ugm3 - out.scenario_optical_bc_ugm3
        ).where(eligible)
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def _descriptive_stats(part: pd.DataFrame) -> dict:
    difference = part.scenario_ftir_minus_optical_ugm3.to_numpy(float)
    prediction = part.prediction_ugm3.to_numpy(float)
    optical = part.scenario_optical_bc_ugm3.to_numpy(float)
    nonempty = len(part) > 0
    r = float(np.corrcoef(prediction, optical)[0, 1]) if len(part) > 2 and np.std(prediction) > 0 and np.std(optical) > 0 else np.nan
    return {
        "n_direct_pairs": len(part), "n_month_blocks": int(part.month_block.nunique()),
        "n_calendar_years": int(part.calendar_year.nunique()),
        "date_start": part.parsed_date.min(), "date_end": part.parsed_date.max(),
        "median_prediction_ugm3": float(np.median(prediction)) if nonempty else np.nan,
        "median_optical_bc_ugm3": float(np.median(optical)) if nonempty else np.nan,
        "mean_ftir_minus_optical_ugm3": float(np.mean(difference)) if nonempty else np.nan,
        "median_ftir_minus_optical_ugm3": float(np.median(difference)) if nonempty else np.nan,
        "q25_ftir_minus_optical_ugm3": float(np.quantile(difference, .25)) if nonempty else np.nan,
        "q75_ftir_minus_optical_ugm3": float(np.quantile(difference, .75)) if nonempty else np.nan,
        "median_absolute_optical_disagreement_ugm3": float(np.median(np.abs(difference))) if nonempty else np.nan,
        "root_mean_square_optical_disagreement_ugm3": float(np.sqrt(np.mean(difference**2))) if nonempty else np.nan,
        "n_prediction_below_optical": int(np.sum(difference < 0)),
        "n_negative_predictions": int(np.sum(prediction < 0)),
        "signed_pearson_r": r, "pearson_r2": r**2,
        "claim_limit": CLAIM_LIMIT,
    }


def seasonal_optical_summary(scenarios: pd.DataFrame, *, by_lot: bool = False, by_year: bool = False) -> pd.DataFrame:
    """Summarize the identical physical-filter panel across all matched arms."""
    group_keys = ["season_convention", "arm", "aggregation_sensitivity", "SiteCode", "season", "season_family", "scenario_mac_m2_g"]
    if by_lot:
        group_keys.append("lot_group")
    if by_year:
        group_keys.append("calendar_year")
    rows = []
    for key, part in scenarios.groupby(group_keys, sort=False, dropna=False):
        for cohort in ("all_common_direct_pairs", "without_prior_problem_filters"):
            eligible = part.common_arm_direct_eligible
            if cohort == "without_prior_problem_filters":
                eligible = eligible & ~part.prior_problem_filter
            chosen = part.loc[eligible]
            rows.append(dict(zip(group_keys, key)) | {"cohort": cohort} | _descriptive_stats(chosen))
    return pd.DataFrame(rows)


def february_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    """Join both conventions by season family without equating their cohorts."""
    group_keys = ["arm", "aggregation_sensitivity", "SiteCode", "season_family", "scenario_mac_m2_g", "cohort"]
    measurements = ["season", "n_direct_pairs", "median_ftir_minus_optical_ugm3", "median_prediction_ugm3", "median_optical_bc_ugm3"]
    parts = []
    for convention in CALENDARS:
        part = summary.loc[summary.season_convention.eq(convention), group_keys + measurements]
        if part.duplicated(group_keys).any():
            raise ValueError("February sensitivity requires summaries without year/lot grouping")
        parts.append(part.rename(columns={name: f"{name}_{convention}" for name in measurements}))
    paired = parts[0].merge(parts[1], on=group_keys, how="outer", validate="one_to_one")
    paired["delta_n_belg_feb_minus_dry_feb"] = paired.n_direct_pairs_belg_feb - paired.n_direct_pairs_dry_feb
    paired["delta_median_disagreement_belg_feb_minus_dry_feb_ugm3"] = (
        paired.median_ftir_minus_optical_ugm3_belg_feb - paired.median_ftir_minus_optical_ugm3_dry_feb
    )
    return paired


def complete_year_summary(scenarios: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat optical summaries for years with observations in all twelve months.

    The rule uses direct measurement/date coverage, not prediction values or
    residuals. It equalizes the represented months, not the number of filters
    per month, and is a coverage sensitivity rather than an exclusion policy.
    """
    frame = scenarios.loc[scenarios.aggregation_sensitivity.eq("all_original_scans")]
    physical = frame.drop_duplicates(["SiteCode", "base_filter_id"])
    known = physical.loc[physical.base_direct_eligible]
    audit_rows, selected_years = [], {}
    for (site, year), part in known.groupby(["SiteCode", "calendar_year"], sort=True):
        months = sorted(part.calendar_month.dropna().astype(int).unique())
        complete = set(months) == set(range(1, 13))
        audit_rows.append({"SiteCode": site, "calendar_year": int(year),
                           "n_direct_pairs": len(part), "n_observed_months": len(months),
                           "observed_months": ";".join(map(str, months)),
                           "complete_twelve_month_coverage": complete})
        if complete:
            selected_years.setdefault(site, []).append(int(year))
    included = pd.Series(False, index=frame.index)
    for site, years in selected_years.items():
        included |= frame.SiteCode.eq(site) & frame.calendar_year.isin(years)
    if not included.any():
        return pd.DataFrame(), pd.DataFrame(audit_rows)
    summary = seasonal_optical_summary(frame.loc[included])
    summary["coverage_control"] = "calendar_years_with_all_12_months_observed"
    summary["included_calendar_years"] = summary.SiteCode.map(
        lambda site: ";".join(map(str, selected_years.get(site, [])))
    )
    summary["coverage_note"] = "All12 months represented; sampling counts remain unequal"
    return summary, pd.DataFrame(audit_rows)


def source_calendar_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Describe held-out IMPROVE errors by calendar month and quarter.

    These are source calendar bins, not Ethiopian rainy seasons. This does not
    make a new seasonal model or turn a held-out-site score into a time split.
    """
    required = {"arm", "AnalysisId", "Site", "SampleDate", "TOR_EC_loading_ug", "prediction_loading_ug"}
    if not required.issubset(predictions):
        raise ValueError(f"missing source prediction columns: {sorted(required-set(predictions))}")
    if predictions.duplicated(["arm", "AnalysisId"]).any():
        raise ValueError("duplicate source held-out predictions")
    frame = predictions.copy()
    dates = pd.to_datetime(frame.SampleDate, errors="raise")
    if dates.isna().any():
        raise ValueError("source calendar bins require known sampling dates")
    frame["calendar_month"] = dates.dt.month
    frame["calendar_quarter"] = dates.dt.quarter
    rows = []
    for bin_type in ("calendar_month", "calendar_quarter"):
        for (arm, bin_number), part in frame.groupby(["arm", bin_type], sort=True):
            y, pred = part.TOR_EC_loading_ug.to_numpy(float), part.prediction_loading_ug.to_numpy(float)
            if not np.isfinite(y).all() or not np.isfinite(pred).all():
                raise ValueError("source held-out results contain nonfinite values")
            error = pred - y
            sst = np.sum((y - y.mean())**2)
            rows.append({
                "arm": arm, "bin_type": bin_type, "bin_number": int(bin_number),
                "n_filters": len(part), "n_sites": int(part.Site.nunique()),
                "rmse_ug_filter": float(np.sqrt(np.mean(error**2))),
                "mae_ug_filter": float(np.mean(np.abs(error))),
                "mean_bias_ug_filter": float(np.mean(error)),
                "median_bias_ug_filter": float(np.median(error)),
                "predictive_r2": float(1-np.sum(error**2)/sst) if sst > 0 else np.nan,
                "median_TOR_EC_loading_ug": float(np.median(y)),
                "date_start": part.SampleDate.min(), "date_end": part.SampleDate.max(),
                "scope": "Held-out IMPROVE site predictions; calendar bins, not Ethiopian seasons",
            })
    return pd.DataFrame(rows)
