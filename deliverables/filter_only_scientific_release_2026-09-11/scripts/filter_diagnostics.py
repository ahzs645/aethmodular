"""Unweighted filter diagnostics from frozen, identity-linked eligibility flags."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import SITES
from data_matching import add_base_filter_id
from interval_evidence import file_hash
from outliers import apply_exclusion_flags, get_clean_data
from plotting.utils import calculate_regression_stats

SENSITIVITY_MULTIPLES = (1.0, 1.5, 2.0, 3.0, 5.0)


def load_frozen_points(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "matched_active_intervals.parquet"
    record = next(r for r in manifest["outputs"] if Path(r["path"]).name == path.name)
    if file_hash(path) != record["sha256"]:
        raise ValueError("Frozen interval decision table changed")
    for name in ["config.py", "outliers.py"]:
        record = next(r for r in manifest["code"] if Path(r["path"]).name == name)
        if file_hash(Path(__file__).parent / name) != record["sha256"]:
            raise ValueError("Frozen identity/exclusion rules changed; rebuild the source audit")
    table = pd.read_parquet(path)
    filter_fields = [
        "eligible_filter_diagnostic",
        "eligible_ec_ratio_analysis",
        "ftir_ec_ugm3",
        "hips_fabs_Mm1",
        "ftir_ec_mdl_ugm3",
    ]
    if (
        (table.groupby(["site", "base_filter_id"])[filter_fields].nunique(dropna=False) > 1)
        .any()
        .any()
    ):
        raise ValueError("Filter-only values or decisions disagree across channels")
    points = table.drop_duplicates(["site", "base_filter_id"]).copy()
    for site in SITES:
        mask = points.site.eq(site)
        probes = points.loc[mask, ["date", "filter_ids"]].copy()
        probes["filter_id"] = probes.filter_ids.map(json.loads)
        probes = probes.explode("filter_id")
        flagged = apply_exclusion_flags(probes, site).groupby(level=0).is_excluded.any()
        if not flagged.reindex(points.loc[mask].index).equals(points.loc[mask, "is_excluded"]):
            raise ValueError("Frozen registry flags differ from canonical reapplication")
    if (points.eligible_ec_ratio_analysis & ~points.eligible_filter_diagnostic).any():
        raise ValueError("Every ratio point must belong to the diagnostic population")
    if (points.eligible_filter_diagnostic.sum(), points.eligible_ec_ratio_analysis.sum()) != (
        545,
        480,
    ):
        raise ValueError("This frozen-cohort report expects 545 diagnostic and 480 ratio pairs")
    points["ratio"] = np.nan
    eligible = points.eligible_ec_ratio_analysis
    points.loc[eligible, "ratio"] = (
        points.loc[eligible, "hips_fabs_Mm1"] / points.loc[eligible, "ftir_ec_ugm3"]
    )
    points["ec_mdl_multiple"] = np.nan
    known_mdl = points.ftir_ec_mdl_ugm3.gt(0) & ~points.ftir_ec_mdl_conflict
    points.loc[known_mdl, "ec_mdl_multiple"] = (
        points.loc[known_mdl, "ftir_ec_ugm3"] / points.loc[known_mdl, "ftir_ec_mdl_ugm3"]
    )
    points["near_mdl_1_to_2"] = eligible & points.ec_mdl_multiple.lt(2)
    points["diagnostic_not_ratio"] = points.eligible_filter_diagnostic & ~eligible
    points["point_id"] = points.site + ":" + points.base_filter_id
    return points.sort_values(["site", "date", "base_filter_id"]).reset_index(drop=True)


def summarize_points(points):
    clean = get_clean_data(points)
    diagnostics = clean.loc[clean.eligible_filter_diagnostic]
    ratios = clean.loc[clean.eligible_ec_ratio_analysis]
    stats = []
    long_summary = []
    exclusions = []
    sensitivity = []
    membership = []
    for site in SITES:
        d = diagnostics.loc[diagnostics.site.eq(site)]
        r = ratios.loc[ratios.site.eq(site)]
        fit = calculate_regression_stats(d.ftir_ec_ugm3, d.hips_fabs_Mm1)
        stats.append(
            dict(
                site=site,
                diagnostic_n=len(d),
                ratio_n=len(r),
                diagnostic_not_ratio_n=len(d) - len(r),
                diagnostic_date_min=d.date.min(),
                diagnostic_date_max=d.date.max(),
                ratio_date_min=r.date.min(),
                ratio_date_max=r.date.max(),
                descriptive_ols_slope=fit["slope"],
                descriptive_ols_intercept=fit["intercept"],
                descriptive_r_squared=fit["r_squared"],
                near_mdl_n=int(r.near_mdl_1_to_2.sum()),
                fit_interpretation="within-site descriptive unweighted OLS; not independent validation",
            )
        )
        for population, frame in [("diagnostic", d), ("ratio", r)]:
            for variable in ["hips_fabs_Mm1", "ftir_ec_ugm3", "ratio", "ec_mdl_multiple"]:
                values = frame[variable].dropna()
                long_summary.append(
                    dict(
                        site=site,
                        population=population,
                        variable=variable,
                        n=len(values),
                        median=values.median(),
                        q25=values.quantile(0.25),
                        q75=values.quantile(0.75),
                        minimum=values.min(),
                        maximum=values.max(),
                        reported_date_min=frame.date.min(),
                        reported_date_max=frame.date.max(),
                    )
                )
        excluded = d.loc[~d.eligible_ec_ratio_analysis]
        for reason, g in excluded.groupby("ec_ratio_ineligibility_reason"):
            exclusions.append(
                dict(
                    site=site,
                    reason=reason,
                    count=len(g),
                    point_ids=json.dumps(g.point_id.tolist()),
                    accounting="mutually exclusive complete reason strings",
                )
            )
        for multiple in SENSITIVITY_MULTIPLES:
            selected = r.loc[r.ec_mdl_multiple.ge(multiple)]
            sensitivity.append(
                dict(
                    site=site,
                    minimum_ec_mdl_multiple=multiple,
                    n=len(selected),
                    retained_fraction=len(selected) / len(r),
                    ratio_median=selected.ratio.median(),
                    ratio_q25=selected.ratio.quantile(0.25),
                    ratio_q75=selected.ratio.quantile(0.75),
                    selection_role="descriptive sensitivity; baseline eligibility unchanged",
                )
            )
            membership.extend(
                dict(point_id=p, site=site, minimum_ec_mdl_multiple=multiple)
                for p in selected.point_id
            )
    if sum(r["count"] for r in exclusions) != len(diagnostics) - len(ratios):
        raise ValueError("Ratio exclusion accounting does not reconcile")
    return {
        "site_results": pd.DataFrame(stats),
        "distribution_summary": pd.DataFrame(long_summary),
        "ratio_exclusion_reasons": pd.DataFrame(exclusions),
        "denominator_sensitivity": pd.DataFrame(sensitivity),
        "sensitivity_point_links": pd.DataFrame(membership),
    }


def evidence_queue(points, inventory):
    """Use unverified envelope overlap to prioritize retrieval, never admission."""
    sources = {s["site"]: s for s in inventory["staged_sources"]}
    rows = []
    for p in points.to_dict("records"):
        source = sources.get(p["site"])
        start, end = p["portal_interval_start_utc"], p["portal_interval_end_utc"]
        valid_bounds = pd.notna(start) and pd.notna(end) and end > start
        usable = (
            np.isfinite(p["hips_fabs_Mm1"])
            and p["hips_fabs_Mm1_units_valid"]
            and not p["hips_fabs_Mm1_conflict"]
            and not p["is_excluded"]
            and p["filter_type"] == "PM2.5"
        )
        low = pd.Timestamp(source["timestamp_min_utc"]) if source else pd.NaT
        high = pd.Timestamp(source["timestamp_max_utc"]) if source else pd.NaT
        overlap = bool(source and valid_bounds and start <= high and end > low)
        hours = (
            max(0.0, (min(end, high) - max(start, low)).total_seconds() / 3600)
            if source and valid_bounds
            else np.nan
        )
        consistent = p["schedule_status"] == "bounds_consistent_mode_unverified"
        if not usable:
            priority = 5
            gate = "usable HIPS/filter identity or registry decision"
        elif not source:
            priority = 4
            gate = "timestamped instrument export, active schedule and stream history"
        elif not valid_bounds:
            priority = 3
            gate = "collection bounds and active schedule, then instrument overlap"
        elif not overlap:
            priority = 3
            gate = "instrument records covering this collection period, then active schedule"
        elif consistent:
            priority = 1
            gate = "filter-linked continuous-operation evidence and scoped observation/processing history"
        else:
            priority = 2
            gate = "actual active periods or contradictory timing resolution, plus scoped stream history"
        rows.append(
            dict(
                point_id=p["point_id"],
                site=p["site"],
                base_filter_id=p["base_filter_id"],
                priority=priority,
                usable_hips=bool(usable),
                reported_start_utc=start,
                reported_end_utc=end,
                reported_bounds_overlap_export=overlap,
                envelope_export_overlap_hours=hours,
                export_min_utc=low,
                export_max_utc=high,
                source_file=source["source_file"] if source else None,
                source_file_hash=source["source_file_hash"] if source else None,
                schedule_status=p["schedule_status"],
                reported_hours_sampled=p["portal_hours_sampled"],
                missing_evidence=gate,
                collection_evidence_links=p["portal_source_links"],
                bounds_evidence_source=p.get("retrieval_bounds_source", "frozen_current_catalog"),
                instrument_metadata=json.dumps(source["metadata_values"], sort_keys=True)
                if source
                else "{}",
                retrieval_route=f"{p['site']} sampler operating/run record tied to {p['base_filter_id']}; instrument acquisition and cleaned-export processing records",
                run_log_located=False,
                active_schedule_verified=False,
                queue_is_primary_eligibility=False,
                priority_basis="metadata overlap and gate burden; not observed coverage or measurement agreement",
            )
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["priority", "site", "reported_start_utc", "base_filter_id"])
        .reset_index(drop=True)
    )


def trace_earlier_chemspec(export_paths, frozen_measurements):
    """Match original export rows to the frozen unified rows without choosing values."""
    frames = []
    for path in export_paths:
        raw = pd.read_csv(path, skiprows=3)
        raw.insert(0, "export_source_row", range(len(raw)))
        ec = raw.loc[raw.Parameter_Name.eq("EC PM2.5")].copy()
        ec["export_file"] = str(Path(path).resolve())
        ec["export_sha256"] = file_hash(path)
        with Path(path).open() as f:
            ec["export_header"] = "".join(next(f) for _ in range(3)).strip()
        ec = add_base_filter_id(ec.assign(FilterId=ec.Filter_ID))
        frames.append(ec)
    earlier = pd.concat(frames, ignore_index=True)
    current = frozen_measurements.loc[frozen_measurements.Parameter.eq("ChemSpec_EC_PM2.5")].copy()
    # Tolerance only reconciles floating-point serialization; it does not select
    # or combine the two distinct values within an EC group.
    earlier["value_key"] = earlier.Value.round(12)
    earlier["mdl_key"] = earlier.MDL.round(12)
    earlier["method_key"] = "ChemSpec_" + earlier.Method_Code.astype(str)
    current["value_key"] = current.Concentration.round(12)
    current["mdl_key"] = current.MDL.round(12)
    joined = current.merge(
        earlier,
        left_on=["Site", "base_filter_id", "value_key", "mdl_key", "CalibrationSetId"],
        right_on=["Site_Code", "base_filter_id", "value_key", "mdl_key", "method_key"],
        how="left",
        validate="one_to_one",
        suffixes=("_unified", "_export"),
    )
    joined["source_value_match"] = np.isclose(
        joined.Concentration, joined.Value, rtol=0, atol=1e-10
    )
    joined["source_mdl_match"] = np.isclose(
        joined.MDL_unified, joined.MDL_export, rtol=0, atol=1e-10, equal_nan=True
    )
    joined["interpretation"] = (
        "both competing concentration values already present in earlier CSV; upstream role unresolved"
    )
    joined["authoritative_value_selected"] = False
    return earlier, joined
