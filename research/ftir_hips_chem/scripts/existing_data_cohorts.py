"""Descriptive cohort controls for the existing-data baseline investigation.

Matching uses collection dates, optical concentration, PM mass concentration
and volume, never the FTIR outcome. Missing lot is not inferred. No returned
selection changes canonical exclusions or source measurement products.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from config import resolve_seasons, season_for_month

REVIEW_IDS = ("ETAD-0037", "ETAD-0139", "ETAD-0158")
MATCH_LIMITS = {"days": 30, "optical_ratio": 1.35, "pm25_ratio": 1.5, "volume_ratio": 1.2}


def match_existing_ethiopian_filters(frame, *, same_lot=False):
    """Maximum-cardinality, minimum-cost one-to-one matching within fixed calipers.

    This is a descriptive sample selection, not a causal adjustment. Acquisition
    batch is unavailable and all recent Addis lot values are unknown.
    """
    if frame["id"].duplicated().any():
        raise ValueError("one row per physical filter is required")
    f = frame.loc[frame["spectral_pair_eligible"].eq(True)
                  & frame["optical_source"].eq("direct_hips_fabs")].copy()
    f["parsed_date"] = pd.to_datetime(f["date"], errors="coerce")
    required = ["optical_bc_ugm3", "pm25_ugm3", "volume_m3"]
    f = f.loc[f["parsed_date"].notna() & (f[required] > 0).all(axis=1)]
    left = f.loc[f["code"].eq("ETAD")].sort_values("id").reset_index(drop=True)
    right = f.loc[f["code"].eq("ETBI")].sort_values("id").reset_index(drop=True)
    columns = ["addis_id", "bishoftu_id", "addis_date", "bishoftu_date", "days_apart",
               "optical_ratio", "pm25_ratio", "volume_ratio", "match_cost",
               "addis_lot", "bishoftu_lot", "same_known_lot",
               "addis_disagreement", "bishoftu_disagreement"]
    if not len(left) or not len(right):
        return pd.DataFrame(columns=columns)
    dates = abs((left["parsed_date"].to_numpy()[:, None]
                 - right["parsed_date"].to_numpy()[None, :]) / np.timedelta64(1, "D"))
    valid = dates <= MATCH_LIMITS["days"]
    cost = dates / MATCH_LIMITS["days"]
    ratios = {}
    for col, key in zip(required, ("optical_ratio", "pm25_ratio", "volume_ratio")):
        ratio = left[col].to_numpy()[:, None] / right[col].to_numpy()[None, :]
        ratios[key] = np.maximum(ratio, 1 / ratio)
        valid &= ratios[key] <= MATCH_LIMITS[key]
        cost += abs(np.log(ratio)) / np.log(MATCH_LIMITS[key])
    a_lot = left["lot_group"].fillna("unknown").astype(str).to_numpy()
    b_lot = right["lot_group"].fillna("unknown").astype(str).to_numpy()
    known = ((a_lot[:, None] != "unknown") & (b_lot[None, :] != "unknown")
             & (a_lot[:, None] == b_lot[None, :]))
    if same_lot:
        valid &= known
    # Invalid cost exceeds the maximum total possible valid cost, so the
    # assignment first maximizes feasible pairs, then minimizes their cost.
    invalid_cost = 10 * (min(len(left), len(right)) + 1)
    ai, bi = linear_sum_assignment(np.where(valid, cost, invalid_cost))
    rows = []
    for a, b in zip(ai, bi):
        if not valid[a, b]:
            continue
        rows.append({
            "addis_id": left.at[a, "id"], "bishoftu_id": right.at[b, "id"],
            "addis_date": left.at[a, "date"], "bishoftu_date": right.at[b, "date"],
            "days_apart": float(dates[a, b]),
            **{key: float(values[a, b]) for key, values in ratios.items()},
            "match_cost": float(cost[a, b]), "addis_lot": a_lot[a], "bishoftu_lot": b_lot[b],
            "same_known_lot": bool(known[a, b]),
            "addis_disagreement": left.at[a, "optical_disagreement_ugm3"],
            "bishoftu_disagreement": right.at[b, "optical_disagreement_ugm3"],
        })
    return pd.DataFrame(rows, columns=columns)


def cohort_summaries(frame):
    """Return chart summaries, exact matched membership, and selection settings."""
    if frame["id"].duplicated().any():
        raise ValueError("one row per physical filter is required")
    f = frame.loc[frame["spectral_pair_eligible"].eq(True)].copy()
    rows = []

    def add(comparison, code, g, group=None, **extra):
        y = g["optical_disagreement_ugm3"].dropna().to_numpy(float)
        rows.append({
            "comparison": comparison, "group": group or code, "code": code,
            "n": len(g), "median_disagreement": float(np.median(y)) if len(y) else None,
            "median_abs_disagreement": float(np.median(abs(y))) if len(y) else None,
            "date_start": g["date"].min() if len(g) else None,
            "date_end": g["date"].max() if len(g) else None, **extra,
        })

    for code, g in f.groupby("code", sort=True):
        add("All eligible spectral pairs", code, g)
        add("Direct HIPS only", code, g.loc[g.optical_source.eq("direct_hips_fabs")])
    etbi = f.loc[f.code.eq("ETBI")]
    overlap = None
    if len(etbi):
        overlap = [etbi.date.min(), etbi.date.max()]
        for code in ("ETAD", "ETBI"):
            add("Overlapping collection dates", code,
                f.loc[f.code.eq(code) & f.date.between(*overlap)],
                note="Same date range; not necessarily simultaneous samples or matched lots.")

    matched = match_existing_ethiopian_filters(f)
    strict = match_existing_ethiopian_filters(f, same_lot=True)
    for comparison, matches, note in (
        ("Date and loading matched", matched, "Direct HIPS; one-to-one matches; recent Addis lot is unknown."),
        ("Same known lot, date and loading", strict, "No lot is inferred from date; an empty group means unavailable matching."),
    ):
        for code, key in (("ETAD", "addis_id"), ("ETBI", "bishoftu_id")):
            add(comparison, code, f.loc[f.id.isin(matches[key])], note=note)

    for code in ("ETAD", "ETBI"):
        g = f.loc[f.code.eq(code)].copy()
        add("Documented scan flags", code, g, group=code + " · retained")
        add("Documented scan flags", code, g.loc[~g.id.isin(REVIEW_IDS)],
            group=code + " · sensitivity subset",
            note="Previously documented acquisition/damage flags; not new canonical exclusions.")
        for convention in ("dry_feb", "belg_feb"):
            seasons = resolve_seasons(convention)
            g["season"] = pd.to_datetime(g.date).dt.month.map(lambda m: season_for_month(m, seasons))
            for season, subset in g.groupby("season", sort=True):
                add("Season calendar: " + convention, code, subset,
                    group=code + " · " + season, season=season)
            for (lot, season), subset in g.groupby(["lot_group", "season"], sort=True):
                add("Lot × season: " + convention, code, subset,
                    group=code + " · " + str(lot) + " · " + season, season=season, lot=str(lot))
    return pd.DataFrame(rows), matched, {
        "matching_calipers": MATCH_LIMITS, "matching_uses_ftir_outcome": False,
        "same_known_lot_pairs": len(strict), "date_loading_pairs": len(matched),
        "overlapping_collection_date_range": overlap,
        "review_ids": list(REVIEW_IDS),
        "limits": ["Descriptive selection, not causal identification.",
                   "Collection dates do not establish identical sampling intervals.",
                   "Missing lot, acquisition batch and instrument information are not inferred.",
                   "Calendar bins do not establish measured meteorological conditions."],
    }
