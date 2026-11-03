"""Controls for the seasonal spectral ML: filter lot and the FTIR EC term in D.

D = FTIR EC(X) − HIPS BC, and FTIR EC is itself computed from the spectrum X. A
spectral model of D can therefore score well just by re-learning the EC
calibration. These helpers supply the obvious comparator (a straight line on
FTIR EC, no spectrum) and the target that is left once it is removed. They also
fill the filter lot from the SPARTAN filter table, which covers filters newer
than the native lot map.

Nothing here refits a baseline or changes an EC product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_ID = r"^([A-Z]{4}-\d{4})"


def spartan_lot_map(filters: pd.DataFrame, *, filter_type: str = "PM2.5") -> pd.Series:
    """Physical filter id (e.g. ETAD-0128) -> ExternalLotId string.

    Uses the SPARTAN ``spartan_Filters.csv`` columns ExternalFilterId,
    ExternalFilterType and ExternalLotId. A filter listed with two different
    lots is an error, never resolved silently.
    """
    need = {"ExternalFilterId", "ExternalFilterType", "ExternalLotId"}
    if not need <= set(filters.columns):
        raise ValueError(f"filter table lacks {sorted(need - set(filters.columns))}")
    rows = filters.loc[filters.ExternalFilterType.astype(str).eq(filter_type)].copy()
    rows["id"] = rows.ExternalFilterId.astype(str).str.extract(BASE_ID)[0]
    rows["lot"] = rows.ExternalLotId.astype("string").str.strip()
    rows = rows.dropna(subset=["id", "lot"])
    rows = rows.loc[rows.lot.ne("")]
    lots = rows.groupby("id").lot.agg(lambda s: sorted(set(s)))
    conflicts = lots[lots.map(len) > 1]
    if len(conflicts):
        raise ValueError(f"filters listed with conflicting lots: {list(conflicts.index[:5])}")
    return lots.map(lambda v: v[0]).rename("lot")


def fill_lot_groups(cohort: pd.DataFrame, lot_map: pd.Series) -> pd.DataFrame:
    """Fill ``lot_group`` where it is unknown; keep and verify recorded lots.

    Adds ``lot_source``: 'native lot map' for lots already recorded,
    'SPARTAN filter table' for filled ones, 'unknown' where neither has it.
    Raises if a recorded lot disagrees with the filter table.
    """
    out = cohort.copy()
    recorded = (out.lot_group.astype("string").fillna("unknown")
                .str.replace(r"\.0$", "", regex=True))
    spartan = out.id.map(lot_map).astype("string")
    known = recorded.ne("unknown")
    clash = known & spartan.notna() & recorded.ne(spartan)
    if clash.any():
        raise ValueError(f"recorded lot disagrees with the SPARTAN filter table for {list(out.id[clash][:5])}")
    filled = recorded.where(known, spartan.fillna("unknown"))
    out["lot_group"] = filled.astype(str)
    out["lot_source"] = np.where(known, "native lot map",
                                 np.where(filled.ne("unknown"), "SPARTAN filter table", "unknown"))
    return out


class ECLine:
    """D ≈ a + b·EC_FTIR, fitted by least squares on training filters only."""

    def fit(self, ec, y):
        ec, y = np.asarray(ec, float), np.asarray(y, float)
        if ec.shape != y.shape or ec.ndim != 1 or len(ec) < 3 or not np.isfinite(ec).all() or not np.isfinite(y).all():
            raise ValueError("EC line needs matching, finite one-dimensional training vectors")
        if np.ptp(ec) == 0:
            raise ValueError("EC line needs variation in training FTIR EC")
        self.slope_, self.intercept_ = (float(v) for v in np.polyfit(ec, y, 1))
        return self

    def predict(self, ec):
        return self.intercept_ + self.slope_ * np.asarray(ec, float)


def regression_scores(actual, predicted) -> dict:
    """RMSE, MAE, prediction R² (against the test mean) and mean bias."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.shape != p.shape or not len(a):
        raise ValueError("actual and predicted must match")
    err = p - a
    ss = float(np.sum((a - a.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / ss) if ss > 0 else float("nan"), "bias": float(err.mean())}


def lot_season_counts(cohort: pd.DataFrame) -> pd.DataFrame:
    """Long table of physical-filter counts by sampling cycle, season and lot."""
    return (cohort.groupby(["cycle_end", "season", "lot_group"]).size()
            .rename("n").reset_index().sort_values(["cycle_end", "season", "lot_group"], ignore_index=True))
