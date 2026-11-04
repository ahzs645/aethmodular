"""Separate spectral associations with FTIR EC, measured HIPS, and their gap.

This is an exploratory response decomposition, not a causal analysis or an EC
correction. The EC target already derives from spectra. An indicator associated
with the EC-minus-optical target can therefore recover the FTIR estimator's own
behavior. All three responses use exactly the same complete cohort and nuisance
design for each prespecified indicator. Neither response is a nuisance control.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import MAC_VALUE
from portable_cohort_screen import DIRECT_HIPS, optical_pair_ledger


INDICATORS = (
    "raw_1617", "corrected_1617", "removed_1617",
    "raw_broad_1600_contrast", "raw_near_1617_contrast",
    "raw_ch_contrast", "raw_oh_nh_contrast", "raw_continuum_3500_3900",
)
TARGETS = {
    "ec_ftir_ugm3": "reported_FTIR_EC_ugm3",
    "hips_fabs_Mm1": "measured_direct_HIPS_Fabs_Mm1",
    "ec_minus_hips_ugm3": "reported_FTIR_EC_minus_HIPS_over_config_MAC_ugm3",
}
CONTROLS = ("pm25_ugm3", "volume_m3", "date", "lot_group")


def average_physical_scans(scans, raw, corrected):
    """Average already-corrected scans by MediaId, keeping every acquisition ID.

    Baseline correction is nonlinear: corrected scans must be calculated before
    this call. This function never fits a baseline to an averaged raw spectrum.
    Known suspicious acquisitions remain included until separately adjudicated.
    """
    scans = scans.reset_index(drop=True).copy()
    raw, corrected = np.asarray(raw, float), np.asarray(corrected, float)
    required = ("MediaId", "SampleAnalysisId", "base_filter_id", "ExternalFilterType", "lot_group", "native_date")
    if any(name not in scans for name in required):
        raise ValueError("native scan metadata is incomplete")
    if raw.ndim != 2 or raw.shape != corrected.shape or len(raw) != len(scans):
        raise ValueError("scan metadata and raw/corrected spectral matrices must align")
    if scans.MediaId.isna().any() or scans.SampleAnalysisId.isna().any():
        raise ValueError("scan and physical identities must not be missing")
    if scans.SampleAnalysisId.duplicated().any():
        raise ValueError("duplicate scan identity")
    groups = list(scans.groupby("MediaId", sort=True).indices.values())
    for indices in groups:
        for column in ("base_filter_id", "ExternalFilterType", "lot_group", "native_date"):
            if scans.iloc[indices][column].nunique(dropna=False) != 1:
                raise ValueError(f"conflicting {column} within a physical filter")
    physical = scans.iloc[[indices[0] for indices in groups]].drop(columns="SampleAnalysisId").reset_index(drop=True)
    if physical.base_filter_id.isna().any() or physical.base_filter_id.duplicated().any():
        raise ValueError("physical filter IDs must be present and unique across MediaId")
    physical["n_scans"] = [len(indices) for indices in groups]
    physical["analysis_ids"] = [
        ";".join(scans.iloc[indices].SampleAnalysisId.astype(str)) for indices in groups
    ]
    mean_raw = np.vstack([raw[indices].mean(axis=0) for indices in groups])
    mean_corrected = np.vstack([corrected[indices].mean(axis=0) for indices in groups])
    return physical, mean_raw, mean_corrected


def build_response_ledger(physical, chemistry):
    """Exact physical-ID and collection-date joins; preserve unmatched rows/flags."""
    if physical.base_filter_id.duplicated().any():
        raise ValueError("native features must have one row per physical filter")
    current = optical_pair_ledger(chemistry)
    current = current.loc[current.code.eq("ETAD")].copy()
    current = current.rename(columns={"id": "base_filter_id"})
    overlap = set(physical.columns) & set(current.columns) - {"base_filter_id"}
    if overlap:
        raise ValueError(f"ambiguous native/current columns: {sorted(overlap)}")
    out = physical.merge(current, on="base_filter_id", how="outer", validate="one_to_one", indicator="join_status")
    out["native_date"] = pd.to_datetime(out.native_date, errors="raise")
    out["current_collection_date"] = pd.to_datetime(out.date, errors="raise")
    out["exact_collection_date_match"] = out.native_date.eq(out.current_collection_date)
    out["source_pair_eligible"] = out.pair_eligible.eq(True)
    out["direct_hips_source"] = out.optical_source.eq(DIRECT_HIPS)
    out["native_pm25"] = out.ExternalFilterType.eq("PM2.5")
    out["response_base_eligible"] = (
        out.join_status.eq("both") & out.native_pm25 & out.exact_collection_date_match
        & out.source_pair_eligible & out.direct_hips_source
    )
    out["response_base_reason"] = np.select([
        ~out.join_status.eq("both"), ~out.native_pm25,
        ~out.exact_collection_date_match, ~out.source_pair_eligible, ~out.direct_hips_source,
    ], [
        "unmatched_native_or_current_physical_filter", "not_native_PM2.5",
        "missing_or_mismatched_collection_date", "ineligible_current_measurement_pair",
        "not_direct_HIPS_absorption",
    ], default="eligible_direct_HIPS_native_PM2.5_pair")
    out["ec_minus_hips_ugm3"] = out.ec_ftir_ugm3 - out.hips_fabs_Mm1 / MAC_VALUE
    out["lot_group"] = out.lot_group.fillna("unknown").astype(str)
    return out


def nuisance_design(frame):
    """Shared response-independent design with explicit rank/collinearity checks.

    Rank PM2.5, public sampled volume, and collection day; retain monthly sine
    and cosine terms and drop-first lot indicators. Missing lot is an explicit
    category. Every nonconstant numeric design column is centered and scaled;
    this changes conditioning, not the nuisance subspace.
    """
    if any(name not in frame for name in CONTROLS):
        raise ValueError("nuisance controls missing")
    dates = pd.to_datetime(frame.date, errors="raise")
    if dates.isna().any() or not np.isfinite(frame[["pm25_ugm3", "volume_m3"]].to_numpy(float)).all():
        raise ValueError("nuisance controls must be complete")
    month = dates.dt.month.to_numpy(float)
    days = (dates - pd.Timestamp("2020-01-01")).dt.total_seconds().to_numpy() / 86400
    columns = {
        "rank_pm25": rankdata(frame.pm25_ugm3.to_numpy(float)),
        "rank_volume": rankdata(frame.volume_m3.to_numpy(float)),
        "month_sin": np.round(np.sin(2 * np.pi * month / 12), 14),
        "month_cos": np.round(np.cos(2 * np.pi * month / 12), 14),
        "rank_collection_day": rankdata(days),
    }
    lots = pd.get_dummies(frame.lot_group.fillna("unknown").astype(str), prefix="lot", drop_first=True, dtype=float)
    columns.update({name: lots[name].to_numpy(float) for name in lots})
    arrays, names, constant = [np.ones(len(frame))], ["intercept"], []
    for name, values in columns.items():
        spread = float(np.std(values))
        if spread <= 1e-12:
            constant.append(name)
        else:
            arrays.append((values - np.mean(values)) / spread)
            names.append(name)
    design = np.column_stack(arrays)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design))
    return design, {
        "design_columns": names, "omitted_constant_controls": constant,
        "design_n_columns": design.shape[1], "design_rank": rank,
        "design_rank_deficient": rank < design.shape[1],
        "design_condition_number": condition,
        "design_ill_conditioned": not np.isfinite(condition) or condition > 1e10,
    }


def _rank_residual(values, design):
    ranked = rankdata(np.asarray(values, float))
    residual = ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    spread = float(np.std(ranked))
    fraction = float(np.std(residual) / spread) if spread > 0 else 0.0
    return ranked, residual, fraction


def response_associations(ledger, indicators=INDICATORS, *, min_n=20):
    """Return long association results plus complete per-indicator membership.

    Each indicator uses one shared complete cohort across *all three* targets.
    No p-value, selected wavelength, source assignment or correction is produced.
    Rank-deficient nuisance designs use their estimable joint column space and
    retain an explicit warning. Degenerate residuals return missing correlations.
    """
    if ledger.base_filter_id.duplicated().any():
        raise ValueError("one row per physical filter required")
    ledger = ledger.reset_index(drop=True)
    if not indicators or len(set(indicators)) != len(indicators):
        raise ValueError("indicators must be nonempty and unique")
    if any(name not in ledger for name in (*indicators, *TARGETS, *CONTROLS, "response_base_eligible")):
        raise ValueError("response ledger is missing a required feature, target, or control")
    results, membership = [], []
    for indicator in indicators:
        numeric = [indicator, *TARGETS, "pm25_ugm3", "volume_m3"]
        complete_numeric = np.isfinite(ledger[numeric].to_numpy(float)).all(axis=1)
        complete_date = pd.to_datetime(ledger.date, errors="raise").notna().to_numpy()
        eligible = ledger.response_base_eligible.to_numpy(bool) & complete_numeric & complete_date
        part = ledger.loc[eligible].copy()
        ids = sorted(part.base_filter_id.astype(str))
        cohort_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        for index, row in ledger.iterrows():
            reasons = []
            if not bool(row.response_base_eligible):
                reasons.append(str(row.response_base_reason))
            missing = [name for name in numeric if not np.isfinite(float(row[name]))]
            if missing:
                reasons.append("missing_or_nonfinite:" + ";".join(missing))
            if pd.isna(pd.to_datetime(row.date, errors="raise")):
                reasons.append("missing_collection_date")
            membership.append({
                "indicator": indicator, "base_filter_id": row.base_filter_id,
                "common_cohort_member": bool(eligible[index]),
                "cohort_reason": "eligible_common_complete_cohort" if not reasons else " | ".join(reasons),
                "cohort_id_sha256": cohort_hash,
            })
        if len(part):
            design, design_info = nuisance_design(part)
        else:
            design = np.empty((0, 0))
            design_info = {
                "design_columns": [], "omitted_constant_controls": [],
                "design_n_columns": 0, "design_rank": 0,
                "design_rank_deficient": False, "design_condition_number": np.nan,
                "design_ill_conditioned": False,
            }
        for target, target_label in TARGETS.items():
            result = {
                "indicator": indicator, "target": target, "target_label": target_label,
                "n_physical_filters": len(part), "n_base_eligible": int(ledger.response_base_eligible.sum()),
                "n_incomplete_among_base_eligible": int(ledger.response_base_eligible.sum()) - len(part),
                "cohort_id_sha256": cohort_hash, **design_info,
                "residual_degrees_of_freedom": len(part) - design_info["design_rank"] - 1,
                "spearman_r": np.nan, "spearman_r_squared": np.nan,
                "partial_rank_r": np.nan, "partial_rank_r_squared": np.nan,
                "feature_residual_rank_fraction": np.nan, "target_residual_rank_fraction": np.nan,
                "analysis_status": "insufficient_complete_physical_filters",
                "claim_limit": "exploratory response decomposition; not causal attribution or EC accuracy",
            }
            if len(part) >= min_n and result["residual_degrees_of_freedom"] >= 10:
                xrank, xr, xf = _rank_residual(part[indicator], design)
                yrank, yr, yf = _rank_residual(part[target], design)
                result["feature_residual_rank_fraction"] = xf
                result["target_residual_rank_fraction"] = yf
                if np.std(xrank) > 0 and np.std(yrank) > 0:
                    raw = float(np.corrcoef(xrank, yrank)[0, 1])
                    result.update(spearman_r=raw, spearman_r_squared=raw**2)
                if xf > 1e-10 and yf > 1e-10:
                    partial = float(np.corrcoef(xr, yr)[0, 1])
                    result.update(partial_rank_r=partial, partial_rank_r_squared=partial**2,
                                  analysis_status="estimable_with_collinear_controls" if (
                                      design_info["design_rank_deficient"] or design_info["design_ill_conditioned"]
                                  ) else "estimable")
                else:
                    result["analysis_status"] = "degenerate_feature_or_target_after_adjustment"
            for name in ("design_columns", "omitted_constant_controls"):
                result[name] = json.dumps(result[name], separators=(",", ":"))
            results.append(result)
    return pd.DataFrame(results), pd.DataFrame(membership)
