"""Export Research → Python PMF (Oct 2026): the independent Python PMF of the Addis and Bishoftu FTIR spectra.

    AETHMODULAR_PYTHON_PMF_DIR=<python_pmf> uv run --locked --no-sync python -I gallery/data/export_python_pmf.py

An independent uncertainty-weighted nonnegative factorisation (ordinary and Huber losses) of the ETAD (246 filters;
the original 102 as a subset) and ETBI (53) spectra, delivered on 30 Oct 2026 as a package of code, prepared inputs
and saved results (Drive: the three Addis_PMF_0*.zip archives, extracted together into one `python_pmf/` folder;
its Addis_Python_PMF_Analysis_Report.md holds the numbers this page reproduces). It is not EPA PMF5/ME-2 and its
factors are not validated named sources: the page calls them Factor 1…5 only.

This script reads the package's saved CSV, JSON and NPZ results as data (numpy with allow_pickle=False, pandas); it
never imports or runs the package's code. Set AETHMODULAR_PYTHON_PMF_DIR to the extracted `python_pmf` folder (the
one holding Addis_Python_PMF_Analysis_Report.md). Without it the script stops and the committed export is kept.

Writes gallery/app/public/data/research/python_pmf/ (the Research section's data folder: private):

* python_pmf.json   every five-factor case's fit summary and per-filter contributions (G), the rank screen and its
                    starts, the spectral-region residuals, the case-to-case and archived-KF matches, the supporting
                    correlations, the held-out and conditional total-OM predictions with their fold metrics, the
                    bootstrap summaries and the 368-filter QC inventory;
* profiles.json     each case's normalised profiles (H rows sum to one) on its wavenumber grid, its per-channel
                    mean scaled residual and negative-input share, and the segmented-bootstrap 5/50/95 % profile
                    percentiles for the two cases the package resampled (etad246_huber, etad102_huber).

Numbers are rounded to 5 significant figures; no machine path is written.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gallery/app/public/data/research/python_pmf"
ENV = "AETHMODULAR_PYTHON_PMF_DIR"
REPORT = "Addis_Python_PMF_Analysis_Report.md"
GENERATED_FROM = "gallery/data/export_python_pmf.py"
SIG = 5

sys.path.insert(0, str(ROOT))
try:  # the repo .env, as every resolver loads it
    from aethmodular_cli.env import load_repo_env
    load_repo_env()
except ImportError:
    pass

# the five-factor cases: key → (results folder, cohort, preparation/loss)
COHORTS = {"etad246": "ETAD 246", "etad102": "ETAD 102", "etbi53": "ETBI 53", "etad244": "ETAD 244"}
PREPS = {"huber": "Robust (Huber)", "quadratic": "Ordinary (quadratic)", "blanksub_huber": "Blank-subtracted, robust",
         "df4_3_huber": "Baseline DF(4,3), robust", "df8_5_huber": "Baseline DF(8,5), robust"}
CASES = {
    "etad246_huber": "five_factor", "etad102_huber": "five_factor", "etbi53_huber": "five_factor",
    "etad246_quadratic": "five_factor", "etad102_quadratic": "five_factor", "etbi53_quadratic": "five_factor",
    "etad246_blanksub_huber": "five_factor", "etad102_blanksub_huber": "five_factor",
    "etbi53_blanksub_huber": "preparation_sensitivity", "etad244_huber": "preparation_sensitivity",
    "etad246_df4_3_huber": "preparation_sensitivity", "etad102_df4_3_huber": "preparation_sensitivity",
    "etad246_df8_5_huber": "preparation_sensitivity", "etad102_df8_5_huber": "preparation_sensitivity",
}
# the rank screens and the five-factor case that supplies their k = 5
SCREENS = {"etad246": "etad246_huber", "etad102": "etad102_huber", "etbi53": "etbi53_huber"}
BOOTSTRAPS = {"etad246_huber": "etad246_bootstrap", "etad102_huber": "etad102_bootstrap"}
FACTORS = [f"Factor_{i}" for i in range(1, 6)]


def package_root() -> Path:
    """The extracted python_pmf folder, from AETHMODULAR_PYTHON_PMF_DIR; stop (keeping the committed export) without it."""
    env = os.environ.get(ENV)
    root = Path(env).expanduser() if env else None
    if root is None or not (root / REPORT).is_file() or not (root / "results/five_factor").is_dir():
        where = "is not set" if root is None else "does not hold the package (no Addis_Python_PMF_Analysis_Report.md and results/five_factor/)"
        raise SystemExit(f"{ENV} {where}: point it at the extracted python_pmf folder (the Drive's Addis_PMF_01/02/03 zips, "
                         f"extracted into one directory). The committed {OUT.relative_to(ROOT).as_posix()}/ is left as it is.")
    return root


def sig(v, n: int = SIG):
    """A JSON number at n significant figures; None for missing or non-finite values."""
    if v is None or isinstance(v, bool):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
    if f == 0:
        return 0
    r = float(f"{f:.{n}g}")
    return int(r) if r.is_integer() and abs(r) < 1e15 else r


def arr(values, n: int = SIG) -> list:
    return [sig(v, n) for v in np.asarray(values, dtype=float).ravel()]


def columns(frame: pd.DataFrame, n: int = SIG) -> dict[str, list]:
    """A table as columns, numbers at n significant figures (lib/table rowsOf reads it)."""
    out: dict[str, list] = {}
    for c in frame.columns:
        s = frame[c]
        if pd.api.types.is_bool_dtype(s):
            out[str(c)] = [bool(v) for v in s]
        elif pd.api.types.is_numeric_dtype(s):
            out[str(c)] = [sig(v, n) for v in s.to_numpy()]
        else:
            out[str(c)] = [None if pd.isna(v) else str(v) for v in s]
    return out


class Package:
    """Reads the package's saved results and records each file read (relative to the package root)."""

    def __init__(self, root: Path):
        self.root = root
        self.sources: list[dict] = []

    def _note(self, table: str, rel: str, rows: int, read: str) -> None:
        self.sources.append({"table": table, "rows": int(rows), "notebook": f"python_pmf/{rel}", "read": read, "path": f"python_pmf/{rel}"})

    def csv(self, table: str, rel: str) -> pd.DataFrame:
        frame = pd.read_csv(self.root / rel)
        self._note(table, rel, len(frame), "csv")
        return frame

    def json(self, table: str, rel: str) -> dict:
        data = json.loads((self.root / rel).read_text())
        self._note(table, rel, 1, "json")
        return data

    def npz(self, table: str, rel: str) -> dict[str, np.ndarray]:
        with np.load(self.root / rel, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        self._note(table, rel, 1, "npz")
        return data

    def case_dir(self, case: str) -> str:
        return f"results/{CASES[case]}/{case}"


def split_case(case: str) -> tuple[str, str]:
    cohort, _, prep = case.partition("_")
    return cohort, prep


def case_summaries(p: Package, diag: pd.DataFrame, stab: pd.DataFrame) -> list[dict]:
    """One row per five-factor case: the selected fit's diagnostics (analysis/rank_diagnostics.csv) and its starts."""
    out = []
    for case in CASES:
        d = diag[(diag.case == case) & (diag.factors == 5)]
        if d.empty:
            raise SystemExit(f"analysis/rank_diagnostics.csv has no five-factor row for {case}")
        d = d.iloc[0]
        s = stab[(stab.case == case) & (stab.factors == 5)].iloc[0]
        cohort, prep = split_case(case)
        cells = float(d.n_observed_cells)
        out.append({
            "key": case, "cohort": cohort, "cohort_label": COHORTS[cohort], "prep": prep, "prep_label": PREPS[prep],
            "folder": CASES[case], "grid": "etbi" if cohort == "etbi53" else "etad",
            "n": int(d.n_samples), "features": int(d.n_features), "cells": int(cells), "seed": int(d.selected_seed),
            "converged_starts": int(d.converged_starts), "total_starts": int(d.total_starts), "converged": bool(d.converged),
            "within_1pct": int(s.converged_within_1percent_of_best_attained),
            "objective": sig(d.objective, 8), "loss_per_cell": sig(d.objective / cells), "q_true": sig(d.Q_true, 8),
            "rmse": sig(d.weighted_rmse_true), "frac_gt3": sig(d.fraction_abs_scaled_residual_gt3),
            "frac_negative": sig(d.fraction_negative_observations), "negative_q_lower_bound": sig(d.negative_observation_Q_lower_bound, 8),
            "negative_rmse_lower_bound": sig(math.sqrt(d.negative_observation_Q_lower_bound / cells)),
            "lower_bound_share_of_q": sig(d.negative_observation_Q_lower_bound / d.Q_true),
            "near_best_min_cosine": sig(d.near_best_min_profile_cosine), "near_best_min_r": sig(d.near_best_min_contribution_r),
            "relative_objective_range": sig(d.relative_objective_range),
        })
    return out


def contributions(p: Package, case: str) -> dict:
    """Per-filter contributions (G, summed-channel spectral amplitude per m³) and each filter's scaled RMSE."""
    base = p.case_dir(case)
    g = p.csv(f"contributions:{case}", f"{base}/k05/contributions.csv")
    r = p.csv(f"sample_residuals:{case}", f"{base}/k05/sample_residuals.csv").set_index("sample_id")
    if list(g.columns[2:]) != FACTORS:
        raise SystemExit(f"{base}/k05/contributions.csv: unexpected columns {list(g.columns)}")
    return {"id": g.sample_id.astype(str).tolist(), "date": g.date.astype(str).tolist(),
            "g": [arr(g[f]) for f in FACTORS],
            "rmse": arr(r.reindex(g.sample_id)["scaled_rmse"]), "gt3": arr(r.reindex(g.sample_id)["fraction_abs_gt3"], 4)}


def profiles(p: Package, case: str, grids: dict[str, list]) -> tuple[list, dict]:
    base = p.case_dir(case)
    h = p.csv(f"profiles:{case}", f"{base}/k05/profiles.csv")
    f = p.csv(f"feature_residuals:{case}", f"{base}/k05/feature_residuals.csv")
    grid = "etbi" if case.startswith("etbi") else "etad"
    wn = arr(h.wavenumber_cm1, 8)
    if grid in grids and grids[grid] != wn:
        raise SystemExit(f"{base}: its wavenumber grid differs from the other {grid.upper()} cases")
    grids[grid] = wn
    if not np.allclose(f.wavenumber_cm1, h.wavenumber_cm1):
        raise SystemExit(f"{base}: feature_residuals.csv and profiles.csv are on different grids")
    return [arr(h[c]) for c in FACTORS], {"mean": arr(f.mean_scaled_residual, 4), "rmse": arr(f.scaled_rmse, 4), "negative": arr(f.negative_input_fraction, 3)}


def bootstrap(p: Package, case: str, wn: list) -> dict:
    """The segmented block bootstrap's per-channel profile percentiles (ensemble.npz, aligned to the case's factors)."""
    rel = f"results/uncertainty/{BOOTSTRAPS[case]}"
    z = p.npz(f"bootstrap:{case}", f"{rel}/ensemble.npz")
    levels = [int(v) for v in z["percentile_levels"]]
    if not np.allclose(z["wavenumbers"], wn, atol=1e-6):
        raise SystemExit(f"{rel}/ensemble.npz is not on {case}'s grid")
    P = z["profile_percentiles"]  # level × factor × channel
    pick = {lvl: P[levels.index(lvl)] for lvl in (5, 50, 95)}
    return {"p05": [arr(pick[5][k]) for k in range(5)], "p50": [arr(pick[50][k]) for k in range(5)], "p95": [arr(pick[95][k]) for k in range(5)],
            "accepted": [int(v) for v in z["accepted"].sum(axis=0)], "eligible": [int(v) for v in z["eligible"].sum(axis=0)],
            "resamples": int(z["converged"].shape[0]), "converged": int(z["converged"].sum())}


def rank_tables(p: Package, diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rank screen (k = 2…13; k = 5 from the five-factor primary case) and every start behind it."""
    rows, starts = [], []
    for cohort, primary in SCREENS.items():
        screen = diag[diag.case == f"{cohort}_rank_screen"]
        k5 = diag[(diag.case == primary) & (diag.factors == 5)]
        for _, d in pd.concat([screen, k5]).sort_values("factors").iterrows():
            rows.append({"cohort": cohort, "k": int(d.factors), "converged_starts": int(d.converged_starts), "total_starts": int(d.total_starts),
                         "selection": d.selection_pool, "converged": bool(d.converged), "loss_per_cell": d.objective / d.n_observed_cells,
                         "rmse": d.weighted_rmse_true, "near_best_min_cosine": d.near_best_min_profile_cosine,
                         "negative_rmse_lower_bound": math.sqrt(d.negative_observation_Q_lower_bound / d.n_observed_cells)})
        for rel in (f"results/rank_screen/{cohort}_rank_screen/all_starts.csv", f"results/five_factor/{primary}/all_starts.csv"):
            a = p.csv(f"starts:{cohort}", rel)
            for _, s in a.iterrows():
                starts.append({"cohort": cohort, "k": int(s.factors), "seed": int(s.seed), "loss_per_cell": s.objective / s.n_observed_cells,
                               "rmse": s.weighted_rmse_true, "converged": bool(s.converged)})
    return pd.DataFrame(rows), pd.DataFrame(starts)


def main() -> int:
    root = package_root()
    p = Package(root)
    diag = p.csv("rank_diagnostics", "analysis/rank_diagnostics.csv")
    stab = p.csv("multistart_stability", "analysis/multistart_stability.csv")
    manifest = p.json("package_manifest", "PACKAGE_MANIFEST.json")
    summary = p.json("numerical_summary", "analysis/numerical_summary.json")
    comparison = p.json("comparison_manifest", "inputs/comparison_manifest.json")

    cases = case_summaries(p, diag, stab)
    contrib = {c: contributions(p, c) for c in CASES}
    grids: dict[str, list] = {}
    prof, resid = {}, {}
    for c in CASES:
        prof[c], resid[c] = profiles(p, c, grids)
    boot = {c: bootstrap(p, c, grids["etad"]) for c in BOOTSTRAPS}
    boot_summary = pd.concat([p.csv(f"bootstrap_summary:{c}", f"results/uncertainty/{b}/summary.csv").assign(case=c) for c, b in BOOTSTRAPS.items()])
    bands = pd.concat([p.csv(f"band_fractions:{c}", f"results/uncertainty/{b}/band_fraction_summary.csv").assign(case=c) for c, b in BOOTSTRAPS.items()]).drop(columns="units")

    rank, starts = rank_tables(p, diag)
    regions = p.csv("spectral_region_residuals", "analysis/spectral_region_residuals.csv")
    matches = p.csv("five_factor_case_matches", "analysis/five_factor_case_matches.csv")
    historical = p.csv("historical_factor_pattern_matches", "analysis/historical_factor_pattern_matches.csv").drop(columns="interpretation")
    supporting = p.csv("supporting_variable_correlations", "analysis/supporting_variable_correlations.csv")
    comparison_ids = p.csv("comparison_filters", "inputs/sample_comparison.csv")[["sample_id"]]

    hold = p.csv("heldout_predictions", "results/mass_holdout/predictions.csv")
    hold = hold[["sample_id", "date", "fold", "observed_OM_ug_m3", "predicted_OM_ug_m3", "training_mean_baseline_ug_m3", "selected_seed"]]
    hold_folds = p.csv("heldout_fold_metrics", "results/mass_holdout/fold_metrics.csv")
    hold_folds = hold_folds[["fold", "training_n", "heldout_n", "heldout_date_min", "heldout_date_max", "selected_seed", "n", "RMSE_ug_m3", "MAE_ug_m3",
                             "R_squared", "Pearson_r", "mean_bias_ug_m3", "baseline_RMSE_ug_m3", "baseline_MAE_ug_m3", "baseline_R_squared"]]
    hold_summary = p.json("heldout_summary", "results/mass_holdout/summary.json")
    coefs = p.csv("heldout_coefficients", "results/mass_holdout/coefficients.csv")
    cond_base = "analysis/conditional_mass/etad246_huber"
    cond = p.csv("conditional_predictions", f"{cond_base}/conditional_predictions.csv")
    cond = cond[["sample_id", "date", "date_block_fold", "OM_ug_m3", "conditional_heldout_OM_prediction_ug_m3", "training_mean_baseline_ug_m3", "full_reference_fit_OM_ug_m3"]]
    cond_folds = p.csv("conditional_fold_metrics", f"{cond_base}/fold_metrics.csv")
    cond_summary = p.json("conditional_summary", f"{cond_base}/summary.json")
    alloc = p.csv("conditional_allocations", f"{cond_base}/experimental_mass_allocations.csv")

    qc = p.csv("qc_inventory", "analysis/All_Filter_QC_Inventory.csv")
    qc = qc[["site_code", "sample_id", "filter_type", "date", "sample_volume_m3", "filter_lot", "selected_analysis_qc_code", "metadata_eligible", "model_eligible", "hold_reason"]]

    # the report's own headline numbers, checked against the tables they come from (the page quotes these)
    pooled = hold_summary["pooled_heldout_metrics"]
    checks = {"heldout RMSE": (pooled["RMSE_ug_m3"], 0.516), "heldout MAE": (pooled["MAE_ug_m3"], 0.367), "heldout R²": (pooled["R_squared"], 0.954)}
    for what, (v, want) in checks.items():
        if round(v, 3) != want:
            raise SystemExit(f"{what} {v} differs from the report's {want}: the package is not the 30 Oct 2026 delivery this page was built for")

    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    labels = {str(k): v for k, v in comparison.get("archived_factor_labels", {}).items()}
    main_file = {
        "schema_version": 1, "id": "python_pmf", "title": "Python PMF (Oct 2026)", "generated": when, "generated_from": GENERATED_FROM,
        "file": "research/python_pmf/python_pmf.json", "text": {}, "missing": [],
        "package": {"report": REPORT, "report_date": "2026-10-30", "created_utc": manifest.get("created_utc"),
                    "solver_sha256": manifest.get("solver_sha256"), "files": manifest.get("file_count_excluding_this_manifest"),
                    "fits": summary.get("completed_scientific_fits"), "converged_fits": summary.get("converged_scientific_fits"),
                    "native_epa_parity": summary.get("native_EPA_ME2_parity"), "factor_count_finalized": summary.get("factor_count_finalized"),
                    "physical_filters": manifest.get("primary_unique_physical_filters")},
        "kf_labels": labels,
        "comparison_filters": comparison_ids.sample_id.astype(str).tolist(),
        "cases": cases,
        "contributions": contrib,
        "heldout_pooled": {k: sig(v) for k, v in pooled.items()},
        "heldout_baseline": {k: sig(v) for k, v in hold_summary["pooled_training_mean_baseline_metrics"].items()},
        "conditional": {"heldout": {k: sig(v) for k, v in cond_summary["heldout_OM_only"].items()},
                        "baseline": {k: sig(v) for k, v in cond_summary["training_mean_baseline"].items()},
                        "in_sample": {k: sig(v) for k, v in cond_summary["in_sample"].items()},
                        "coefficients": arr(cond_summary["full_reference_coefficients"]),
                        "outside_reference_dates": cond_summary["rows_outside_reference_date_range"],
                        "reference_dates": [cond_summary["date_min"], cond_summary["date_max"]],
                        "allocated_filters": int(len(alloc))},
        "bootstrap": {c: {k: v for k, v in b.items() if k in ("accepted", "eligible", "resamples", "converged")} for c, b in boot.items()},
        "tables": {
            "rank": columns(rank), "starts": columns(starts), "regions": columns(regions), "case_matches": columns(matches),
            "historical": columns(historical), "supporting": columns(supporting),
            "heldout": columns(hold, 7), "heldout_folds": columns(hold_folds), "heldout_coefficients": columns(coefs),
            "conditional": columns(cond, 7), "conditional_folds": columns(cond_folds),
            "bootstrap_summary": columns(boot_summary), "band_fractions": columns(bands), "qc": columns(qc),
        },
    }
    main_file["sources"] = p.sources
    profile_file = {"schema_version": 1, "generated": when, "generated_from": GENERATED_FROM, "grids": grids,
                    "profiles": prof, "residuals": resid, "bootstrap": {c: {k: b[k] for k in ("p05", "p50", "p95")} for c, b in boot.items()}}

    text = json.dumps(main_file, ensure_ascii=False, separators=(",", ":"))
    if "/home/" in text or "/Users/" in text or "/tmp/" in text:
        raise SystemExit("refusing to write a machine path into the export")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "python_pmf.json").write_text(text)
    (OUT / "profiles.json").write_text(json.dumps(profile_file, separators=(",", ":")))
    for name in ("python_pmf.json", "profiles.json"):
        print(f"wrote {(OUT / name).relative_to(ROOT).as_posix()} ({(OUT / name).stat().st_size / 1024:.0f} KB)")
    print(f"{len(cases)} cases, {sum(len(c['id']) for c in contrib.values())} case-filter rows, {len(p.sources)} package files read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
