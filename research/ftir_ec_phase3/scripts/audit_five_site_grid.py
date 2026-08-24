"""Audit the five-site calibration grid from its saved target readouts.

This guards against a subtle but consequential reporting error: ``heldout_R2``
describes the IMPROVE TOR-EC calibration cohort, whereas ``R2``/``all_R2``
describe the named SPARTAN evaluation target.  They must never be substituted
for one another when ranking site performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / "calibration_explorer/cache/batch_results.jsonl"
OUT = HERE.parent / "output/tables/variation_closure"
TARGETS = ["addis", "etbi", "indh", "chts", "uspa"]
KEY = [
    "cohort", "cutoff", "selection_space", "spectra", "mode", "lot", "k",
]


def _load() -> pd.DataFrame:
    rows = []
    with SOURCE.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["source_line"] = line_number
            rows.append(row)
    return pd.DataFrame(rows)


def _in_launch_grid(frame: pd.DataFrame) -> pd.Series:
    cutoff = pd.to_numeric(frame["cutoff"], errors="coerce")
    cohort_ok = (
        (frame["cohort"].eq("eth_shaped") & cutoff.between(100, 900)
         & cutoff.mod(10).eq(0))
        | (frame["cohort"].eq("analogs") & cutoff.between(100, 1500)
           & cutoff.mod(10).eq(0))
        | (frame["cohort"].eq("ocec") & cutoff.between(100, 2000)
           & cutoff.mod(10).eq(0))
        | frame["cohort"].isin(["smoke", "pool"])
    )
    selection_ok = (
        (frame["cohort"].isin(["eth_shaped", "analogs"])
         & frame["selection_space"].isin(["raw", "airspec"]))
        | (~frame["cohort"].isin(["eth_shaped", "analogs"])
           & frame["selection_space"].eq("raw"))
    )
    return (
        cohort_ok & selection_ok & frame["spectra"].isin(["raw", "airspec", "deriv2"])
        & frame["mode"].eq("site_heldout") & frame["lot"].astype(str).eq("all")
        & frame["target"].isin(TARGETS) & frame["eval_lot"].astype(str).eq("all")
    )


def _target_view(frame: pd.DataFrame) -> pd.DataFrame:
    view = frame.copy()
    addis = view["target"].eq("addis")
    view["target_ols_slope"] = np.where(
        addis, view["ols_slope"], view["all_ols_slope"]
    )
    view["target_ols_intercept"] = np.where(
        addis, view["ols_intercept"], view["all_ols_intercept"]
    )
    view["target_deming_slope"] = np.where(
        addis, view["deming_slope"], view["all_deming_slope"]
    )
    view["target_deming_intercept"] = np.where(
        addis, view["deming_intercept"], view["all_deming_intercept"]
    )
    view["target_R2"] = np.where(addis, view["R2"], view["all_R2"])
    view["target_RMSE"] = np.where(addis, view["RMSE"], view["all_RMSE"])
    return view


def audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = _load()
    grid = raw[_in_launch_grid(raw)].copy()
    duplicate_key = KEY + ["target", "eval_lot"]
    duplicate_rows = grid[grid.duplicated(duplicate_key, keep=False)].copy()
    grid = grid.sort_values("source_line").drop_duplicates(duplicate_key, keep="first")
    grid = _target_view(grid)

    best_rows = []
    for estimator in ("ols", "deming"):
        slope_col = f"target_{estimator}_slope"
        intercept_col = f"target_{estimator}_intercept"
        for target, group in grid.groupby("target"):
            base = group[
                group["heldout_R2"].ge(0.85) & group[slope_col].between(0.85, 1.18)
            ].copy()
            base["score"] = (
                base[intercept_col].abs() + 0.5 * (base[slope_col] - 1).abs()
            )
            for minimum_target_r2 in (0.0, 0.5, 0.7, 0.85):
                eligible = base[base["target_R2"].ge(minimum_target_r2)]
                if eligible.empty:
                    best_rows.append({
                        "estimator": estimator, "target": target,
                        "minimum_target_R2": minimum_target_r2, "n_eligible": 0,
                    })
                    continue
                winner = eligible.sort_values(
                    ["score", "target_R2"], ascending=[True, False]
                ).iloc[0]
                best_rows.append({
                    "estimator": estimator, "target": target,
                    "minimum_target_R2": minimum_target_r2,
                    "n_eligible": len(eligible),
                    **{column: winner[column] for column in KEY},
                    "target_slope": winner[slope_col],
                    "target_intercept": winner[intercept_col],
                    "target_R2": winner["target_R2"],
                    "target_RMSE": winner["target_RMSE"],
                    "heldout_R2": winner["heldout_R2"],
                    "extrap_pct": winner["extrap_pct"],
                    "score": winner["score"],
                })
    best = pd.DataFrame(best_rows)

    pivot = grid.pivot(index=KEY, columns="target", values=[
        "target_ols_slope", "target_ols_intercept", "target_deming_slope",
        "target_deming_intercept", "target_R2", "target_RMSE", "extrap_pct",
        "heldout_R2",
    ])
    pivot.columns = [f"{metric}_{target}" for metric, target in pivot.columns]
    pivot = pivot.reset_index()
    joint = []
    for estimator in ("ols", "deming"):
        slope_addis = f"target_{estimator}_slope_addis"
        slope_delhi = f"target_{estimator}_slope_indh"
        intercept_addis = f"target_{estimator}_intercept_addis"
        intercept_delhi = f"target_{estimator}_intercept_indh"
        both = pivot[
            pivot[slope_addis].between(0.85, 1.18)
            & pivot[slope_delhi].between(0.85, 1.18)
            & pivot["heldout_R2_addis"].ge(0.85)
        ].copy()
        both["estimator"] = estimator
        both["target_slope_addis"] = both[slope_addis]
        both["target_intercept_addis"] = both[intercept_addis]
        both["target_slope_delhi"] = both[slope_delhi]
        both["target_intercept_delhi"] = both[intercept_delhi]
        both["joint_score"] = (
            both[intercept_addis].abs() + both[intercept_delhi].abs()
            + 0.5 * (both[slope_addis] - 1).abs()
            + 0.5 * (both[slope_delhi] - 1).abs()
        )
        joint.append(both)
    both = pd.concat(joint, ignore_index=True).sort_values(
        ["estimator", "joint_score"]
    )

    metadata = {
        "source_rows": len(raw),
        "grid_rows_before_dedup": int(_in_launch_grid(raw).sum()),
        "grid_duplicate_rows": len(duplicate_rows),
        "grid_rows": len(grid),
        "rows_per_target": grid.groupby("target").size().to_dict(),
        "unique_k_configurations": int(grid[KEY].drop_duplicates().shape[0]),
        "simultaneous_addis_delhi": {},
    }
    for estimator, group in both.groupby("estimator"):
        metadata["simultaneous_addis_delhi"][estimator] = {
            "slope_box": len(group),
            "target_R2_ge_0_5": int(
                (group["target_R2_addis"].ge(0.5)
                 & group["target_R2_indh"].ge(0.5)).sum()
            ),
            "target_R2_ge_0_7": int(
                (group["target_R2_addis"].ge(0.7)
                 & group["target_R2_indh"].ge(0.7)).sum()
            ),
            "target_R2_ge_0_85": int(
                (group["target_R2_addis"].ge(0.85)
                 & group["target_R2_indh"].ge(0.85)).sum()
            ),
        }
    return grid, best, both, metadata


def main() -> None:
    grid, best, both, metadata = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "five_site_grid_audited_rows.csv", index=False)
    best.to_csv(OUT / "five_site_grid_best_by_target.csv", index=False)
    both.to_csv(OUT / "five_site_grid_addis_etbi_joint.csv", index=False)
    (OUT / "five_site_grid_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    print("\nBest in slope box with held-out TOR R2 >= 0.85 (no target-R2 floor):")
    columns = [
        "target", "n_eligible", "cohort", "cutoff", "selection_space",
        "spectra", "k", "target_slope", "target_intercept", "target_R2",
        "heldout_R2", "extrap_pct",
    ]
    print(best[best["minimum_target_R2"].eq(0)][
        ["estimator", *columns]
    ].round(4).to_string(index=False))
    print("\nBest joint Addis/Delhi rows:")
    joint_columns = ["estimator", *KEY] + [
        "target_slope_addis", "target_intercept_addis", "target_R2_addis",
        "target_slope_delhi", "target_intercept_delhi", "target_R2_indh",
        "heldout_R2_addis", "joint_score",
    ]
    print(both.groupby("estimator", group_keys=False).head(6)[joint_columns]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
