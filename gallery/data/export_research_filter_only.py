"""Filter-only transfer limits: the per-filter HIPS / FTIR EC pairs the release fitted, and its own result tables.

Run from the repo (no Drive data needed; every input is tracked):
    uv run python gallery/data/export_research_filter_only.py

Inputs (deliverables/filter_only_scientific_release_2026-09-11/data/):
  diagnostic/analysis_points.parquet            the 545 diagnostic filters: reported date, FTIR EC, its MDL, HIPS Fabs
  diagnostic/sensitivity_point_links.parquet    the saved 1x / 1.5x / 2x / 3x / 5x MDL denominator memberships
  stability/population_point_links.parquet      each filter's EC_ftir CalibrationSetId (from its source rows)
  stability/*, proportionality/*, diagnostic/denominator_sensitivity.parquet   the frozen results, for comparison

The page (gallery/app/src/sections/research/pages/FilterOnly.tsx) refits every model in the browser: each one
is closed-form (one-predictor OLS, the training median, the through-origin least-squares slope) over
calendar-quarter blocks, so `points.json` is all it needs. `release.json` carries the release's own block and
summary tables so the page can show how closely its refit agrees.

Notebooks: research/ftir_hips_chem/filter_only_diagnostics.ipynb (05), filter_proportionality.ipynb (8 figures),
filter_relationship_stability.ipynb (9 figures). Fabs is in Mm⁻¹ and EC in µg/m³: nothing here depends on a MAC.

Writes gallery/app/public/data/research/filter_only/{points,release}.json.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "deliverables/filter_only_scientific_release_2026-09-11/data"
OUT = ROOT / "gallery/app/public/data/research/filter_only"

SITE_NAME = {"Addis_Ababa": "Addis Ababa", "Beijing": "Beijing", "Delhi": "Delhi", "JPL": "JPL"}
POPULATIONS = ["diagnostic", "ratio_baseline", "mdl_1_5x", "mdl_2x", "mdl_3x", "mdl_5x"]
MULTIPLES = [1.0, 1.5, 2.0, 3.0, 5.0]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(v):
    """JSON-safe scalars: NaN -> None, numpy -> python, timestamps -> ISO dates."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def table(frame: pd.DataFrame, columns: list[str]) -> dict[str, list]:
    """A frozen table as columns (name -> values), floats to 7 significant digits (they are only compared against)."""
    frame = frame[columns].copy()
    if "site" in frame:
        frame["site"] = frame["site"].map(lambda s: SITE_NAME.get(s, s))
    short = lambda v: float(f"{v:.7g}") if isinstance(v, float) else v  # noqa: E731
    return {c: [short(clean(v)) for v in frame[c].tolist()] for c in columns}


def points() -> tuple[list[dict], list[Path]]:
    """One row per diagnostic filter, with the largest saved MDL multiple whose population holds it (0: diagnostic only)."""
    sources = [RELEASE / "diagnostic/analysis_points.parquet", RELEASE / "diagnostic/sensitivity_point_links.parquet",
               RELEASE / "stability/population_point_links.parquet"]
    pts = pd.read_parquet(sources[0])
    links = pd.read_parquet(sources[1])
    meta = pd.read_parquet(sources[2])
    diag = pts.loc[pts.eligible_filter_diagnostic].copy()
    if diag.point_id.duplicated().any():
        raise ValueError("physical filter identities must be unique")
    member = links.groupby("point_id").minimum_ec_mdl_multiple.max()
    # the saved memberships are nested (each stricter set inside the looser one); the page relies on that
    for m in MULTIPLES:
        saved = set(links.loc[links.minimum_ec_mdl_multiple.eq(m), "point_id"])
        if saved != set(member.index[member >= m]):
            raise ValueError(f"saved {m}x membership is not nested")
    if set(links.loc[links.minimum_ec_mdl_multiple.eq(1.0), "point_id"]) != set(diag.loc[diag.eligible_ec_ratio_analysis, "point_id"]):
        raise ValueError("saved 1x membership differs from the frozen ratio population")
    cal = meta.loc[meta.population.eq("diagnostic")].set_index("point_id", verify_integrity=True).ftir_CalibrationSetId
    diag["mult"] = diag.point_id.map(member).fillna(0.0)
    diag["cal"] = diag.point_id.map(cal)
    if diag.cal.isna().any():
        raise ValueError("every diagnostic filter has a CalibrationSetId (or 'unreported') in the stability links")
    diag = diag.sort_values(["site", "date", "base_filter_id"])
    rows = [{
        "id": r.base_filter_id, "site": SITE_NAME[r.site], "date": r.date.strftime("%Y-%m-%d"),
        "ec": clean(r.ftir_ec_ugm3), "fabs": clean(r.hips_fabs_Mm1), "mdl": clean(r.ftir_ec_mdl_ugm3),
        "mult": clean(r.mult), "cal": str(r.cal),
    } for r in diag.itertuples()]
    return rows, sources


def release() -> tuple[dict, list[Path]]:
    files = {
        "stab_blocks": RELEASE / "stability/block_performance_and_influence.parquet",
        "prop_blocks": RELEASE / "proportionality/proportionality_blocks.parquet",
        "id11_blocks": RELEASE / "proportionality/id11_training_blocks.parquet",
        "perf": RELEASE / "stability/performance_summary.parquet",
        "prop_summary": RELEASE / "proportionality/proportionality_summary.parquet",
        "coefficients": RELEASE / "stability/full_cohort_coefficients.parquet",
        "influence": RELEASE / "stability/ETAD_0037_paired_influence.parquet",
        "cohorts": RELEASE / "stability/population_membership_summary.parquet",
        "denominator": RELEASE / "diagnostic/denominator_sensitivity.parquet",
    }
    t = {k: pd.read_parquet(v) for k, v in files.items()}
    keys = ["site", "population", "variant", "scheme", "block"]
    stab_cols = ["status", "train_n", "test_n", "ols_slope", "ols_intercept", "training_median", "slope_delta_from_full",
                 "intercept_delta_from_full", "ols_mean_signed_error", "ols_mae", "median_mean_signed_error", "median_mae",
                 "outside_training_ec_n", "train_ec_min", "train_ec_max", "test_ec_min", "test_ec_max"]
    prop_cols = ["proportional_k", "proportional_mean_signed_error", "proportional_mae"]
    blocks = t["stab_blocks"][keys + stab_cols].merge(t["prop_blocks"][keys + prop_cols], on=keys, how="left", validate="one_to_one")
    payload = {
        "blocks": table(blocks, keys + stab_cols + prop_cols),
        "id11": table(t["id11_blocks"], ["site", "population", "block", "training_choice", "status", "common_support", "train_n", "test_n",
                                         "all_earlier_train_n", "id11_earlier_train_n", "ols_mae", "ols_mean_signed_error",
                                         "proportional_mae", "median_mae"]),
        "performance": table(t["perf"], ["site", "population", "variant", "scheme", "evaluated_filters", "evaluated_blocks", "improved_blocks",
                                         "ols_mae", "median_mae", "ols_mean_signed_error", "mae_improvement_pct"]),
        "proportionality": table(t["prop_summary"], ["site", "population", "scheme", "weighting", "evaluated_filters", "evaluated_blocks",
                                                     "intercept_better_blocks", "median_mae", "proportional_mae", "ols_mae",
                                                     "proportional_mean_signed_error", "ols_mean_signed_error"]),
        "coefficients": table(t["coefficients"], ["site", "population", "variant", "status", "slope", "intercept", "median",
                                                  "delta_slope_from_original", "delta_intercept_from_original"]),
        "influence": table(t["influence"], ["site", "population", "scheme", "block", "common_n", "baseline_ols_mae", "without_ols_mae",
                                            "mae_change_without_minus_baseline"]),
        "cohorts": table(t["cohorts"], ["site", "population", "variant", "n"]),
        "denominator": table(t["denominator"], ["site", "minimum_ec_mdl_multiple", "n", "ratio_median", "ratio_q25", "ratio_q75"]),
    }
    return payload, list(files.values())


def main() -> None:
    rows, point_sources = points()
    frozen, release_sources = release()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    notebooks = [f"research/ftir_hips_chem/{n}.ipynb" for n in ("filter_only_diagnostics", "filter_proportionality", "filter_relationship_stability")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "points.json").write_text(json.dumps({
        "generated": stamp, "notebooks": notebooks,
        "sources": [{"path": rel(p), "sha256": sha(p)} for p in point_sources],
        "populations": POPULATIONS, "min_train": 10, "omitted_default": "ETAD-0037",
        "points": rows,
    }, separators=(",", ":"), ensure_ascii=False))
    (OUT / "release.json").write_text(json.dumps({
        "generated": stamp, "notebooks": notebooks,
        "sources": [{"path": rel(p), "sha256": sha(p)} for p in release_sources],
        **frozen,
    }, separators=(",", ":"), ensure_ascii=False))
    by_site = pd.Series([r["site"] for r in rows]).value_counts().to_dict()
    print(f"wrote {rel(OUT / 'points.json')}: {len(rows)} diagnostic filters {by_site}")
    print(f"wrote {rel(OUT / 'release.json')}: {len(frozen['blocks']['site'])} blocks, {len(frozen['id11']['site'])} ID-11 blocks")


if __name__ == "__main__":
    main()
