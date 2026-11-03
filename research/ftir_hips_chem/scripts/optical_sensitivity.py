"""Source-separated optical sensitivity scenarios, without fitting an EC correction.

The HIPS absorption divided by an assumed MAC is an equivalent-BC comparator.
It is not an independent EC reference. Fabs / reported FTIR EC is an *effective
ratio* that inherits FTIR error and optical effects; it is not a measured MAC.
All physical filters and reasons survive in the returned ledger. Neither the
configured MAC nor any source measurement is changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portable_cohort_screen import DIRECT_HIPS, optical_pair_ledger


DEFAULT_MAC_SCENARIOS = (7.5, 10.0, 12.5, 15.0)
DEFAULT_EC_THRESHOLDS = (0.1, 0.5, 1.0)
CLAIM_LIMIT = "optical disagreement only; no independent target-domain EC reference"


def _positive_values(values, name):
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or not len(out) or not np.isfinite(out).all() or np.any(out <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    if len(np.unique(out)) != len(out):
        raise ValueError(f"{name} must contain distinct values")
    return tuple(map(float, out))


def fabs_to_equivalent_bc(fabs_Mm1, mac_m2_g):
    """Convert Mm^-1 / (m^2 g^-1) to µg m^-3; both factors of 10^-6 cancel.

    A scalar, finite positive MAC is required. Nonfinite Fabs remains missing;
    negative Fabs is not clipped. No extra 10^6 factor belongs in this ratio.
    """
    mac = float(mac_m2_g)
    if not np.isfinite(mac) or mac <= 0:
        raise ValueError("MAC must be finite and positive")
    fabs = np.asarray(fabs_Mm1, dtype=float)
    return np.where(np.isfinite(fabs), fabs / mac, np.nan)


def direct_sensitivity_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag direct-HIPS eligibility; extensions remain labelled and unconverted."""
    out = optical_pair_ledger(frame)
    out["direct_scenario_eligible"] = out.pair_eligible & out.optical_source.eq(DIRECT_HIPS)
    out["direct_scenario_reason"] = np.where(
        ~out.pair_eligible, out.pair_reason,
        np.where(out.optical_source.eq(DIRECT_HIPS), "eligible_direct_hips_scenario",
                 "not_direct_hips_absorption"),
    )
    finite_ec = np.isfinite(out.ec_ftir_ugm3)
    out["ratio_denominator_finite"] = finite_ec
    out["ratio_denominator_positive"] = finite_ec & out.ec_ftir_ugm3.gt(0)
    out["effective_ratio_eligible"] = (
        out.direct_scenario_eligible & out.ratio_denominator_positive
        & out.hips_fabs_Mm1.gt(0)
    )
    out["effective_fabs_per_ftir_ec_m2_g"] = np.divide(
        out.hips_fabs_Mm1.to_numpy(float), out.ec_ftir_ugm3.to_numpy(float),
        out=np.full(len(out), np.nan), where=out.effective_ratio_eligible.to_numpy(bool),
    )
    out["effective_ratio_reason"] = np.where(
        ~out.direct_scenario_eligible, out.direct_scenario_reason,
        np.where(~out.ratio_denominator_positive, "nonpositive_or_missing_ftir_ec_denominator",
                 np.where(out.hips_fabs_Mm1.le(0), "nonpositive_hips_absorption",
                          "eligible_effective_ratio_not_measured_mac")),
    )
    out.attrs.update(frame.attrs)
    out.attrs["claim_limit"] = CLAIM_LIMIT
    return out


def add_mac_scenarios(ledger: pd.DataFrame, mac_values=DEFAULT_MAC_SCENARIOS) -> pd.DataFrame:
    """Return one row per original filter and scenario, including ineligible rows."""
    mac_values = _positive_values(mac_values, "MAC scenarios")
    if "direct_scenario_eligible" not in ledger:
        raise ValueError("construct a direct_sensitivity_ledger first")
    parts = []
    for mac in mac_values:
        part = ledger.copy()
        part["scenario_mac_m2_g"] = mac
        part["scenario_equivalent_bc_ugm3"] = np.where(
            part.direct_scenario_eligible,
            fabs_to_equivalent_bc(part.hips_fabs_Mm1, mac), np.nan,
        )
        part["scenario_ec_minus_optical_ugm3"] = (
            part.ec_ftir_ugm3 - part.scenario_equivalent_bc_ugm3
        )
        part["scenario_label"] = "assumed_633nm_MAC_not_fitted_or_validated"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _summarize_scenarios(scenarios, cohort):
    rows = []
    for (code, mac), full in scenarios.groupby(["code", "scenario_mac_m2_g"], sort=True):
        paired = full.loc[full.direct_scenario_eligible]
        diff = paired.scenario_ec_minus_optical_ugm3.to_numpy(float)
        rows.append({
            "cohort": cohort, "code": code, "scenario_mac_m2_g": mac,
            "n_physical_filters": len(full), "n_direct_pairs": len(paired),
            "n_public_extension_pairs_not_used": int((
                full.pair_eligible & ~full.optical_source.eq(DIRECT_HIPS)
            ).sum()),
            "n_excluded_filters": int(full.is_excluded.sum()),
            "n_nonpositive_ftir_ec": int(paired.ec_ftir_ugm3.le(0).sum()),
            "n_nonpositive_hips_fabs": int(paired.hips_fabs_Mm1.le(0).sum()),
            "n_ftir_below_optical": int((diff < 0).sum()),
            "n_ftir_equal_optical": int((diff == 0).sum()),
            "n_ftir_above_optical": int((diff > 0).sum()),
            "fraction_ftir_below_optical": float(np.mean(diff < 0)) if len(diff) else np.nan,
            "median_ec_minus_optical_ugm3": float(np.median(diff)) if len(diff) else np.nan,
            "median_absolute_disagreement_ugm3": float(np.median(np.abs(diff))) if len(diff) else np.nan,
            "optical_disagreement_rmse_ugm3": float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan,
            "claim_limit": CLAIM_LIMIT,
        })
    return pd.DataFrame(rows)


def scenario_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Full available coverage; ineligible and source-extension counts remain visible."""
    return _summarize_scenarios(scenarios, "all_available_collection_dates")


def effective_ratio_summary(ledger: pd.DataFrame, ec_thresholds=DEFAULT_EC_THRESHOLDS) -> pd.DataFrame:
    """Denominator-sensitive ratio summaries; thresholds are µg m^-3, inclusive.

    Thresholds define separately labelled sensitivity cohorts, not exclusions.
    The unfiltered ledger retains every denominator and eligibility reason.
    """
    thresholds = _positive_values(ec_thresholds, "EC thresholds")
    rows = []
    for code, full in ledger.groupby("code", sort=True):
        direct = full.loc[full.direct_scenario_eligible]
        for threshold in thresholds:
            use = direct.effective_ratio_eligible & direct.ec_ftir_ugm3.ge(threshold)
            ratios = direct.loc[use, "effective_fabs_per_ftir_ec_m2_g"].to_numpy(float)
            rows.append({
                "code": code, "ftir_ec_min_ugm3_inclusive": threshold,
                "n_direct_pairs": len(direct), "n_ratios": len(ratios),
                "n_direct_hips_missing_ec": int((
                    full.optical_source.eq(DIRECT_HIPS) & ~full.ratio_denominator_finite
                ).sum()),
                "n_nonpositive_ec_denominator": int(direct.ec_ftir_ugm3.le(0).sum()),
                "n_positive_ec_below_threshold": int((
                    direct.ec_ftir_ugm3.gt(0) & direct.ec_ftir_ugm3.lt(threshold)
                ).sum()),
                "n_nonpositive_fabs": int(direct.hips_fabs_Mm1.le(0).sum()),
                "ratio_q25_m2_g": float(np.quantile(ratios, 0.25)) if len(ratios) else np.nan,
                "ratio_median_m2_g": float(np.median(ratios)) if len(ratios) else np.nan,
                "ratio_q75_m2_g": float(np.quantile(ratios, 0.75)) if len(ratios) else np.nan,
                "interpretation": "Fabs/reported_FTIR_EC_effective_ratio_not_measured_MAC",
            })
    return pd.DataFrame(rows)


def ethiopia_overlap_summary(scenarios: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Descriptive Addis/Bishoftu comparison on their common direct-pair date range.

    This is a shared collection-date window, not simultaneous collocation,
    same-lot matching, or a controlled site effect. The complete scenarios
    remain available and the selected window and membership counts are explicit.
    """
    unique = scenarios.drop_duplicates("id")
    ranges = {}
    for code in ("ETAD", "ETBI"):
        dates = pd.to_datetime(unique.loc[
            unique.direct_scenario_eligible & unique.code.eq(code), "date"
        ], errors="raise").dropna()
        if dates.empty:
            return pd.DataFrame(), {"status": "unavailable", "reason": f"no direct pairs at {code}"}
        ranges[code] = (dates.min(), dates.max())
    start = max(r[0] for r in ranges.values())
    end = min(r[1] for r in ranges.values())
    if end < start:
        return pd.DataFrame(), {"status": "unavailable", "reason": "nonoverlapping direct-pair dates"}
    dates = pd.to_datetime(scenarios.date, errors="raise")
    selected = scenarios.loc[scenarios.code.isin(("ETAD", "ETBI")) & dates.between(start, end)]
    out = _summarize_scenarios(selected, "shared_ethiopia_collection_date_window")
    out["window_start_inclusive"] = start.date().isoformat()
    out["window_end_inclusive"] = end.date().isoformat()
    return out, {
        "status": "available", "start": start.date().isoformat(), "end": end.date().isoformat(),
        "interpretation": "shared date range, not matched or collocated filters; no causal site contrast",
    }
