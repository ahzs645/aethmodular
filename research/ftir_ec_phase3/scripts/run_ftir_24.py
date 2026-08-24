"""Season × HIPS interaction for the two locked OCEC-800 Addis models.

Runs Ann's requested dry-versus-wet check without interpreting separate short
range slopes as different mechanisms.  The primary test is a pooled robust
regression with dry-season intercept and slope interactions; per-season OLS
and York fits are retained as descriptive readouts.  Both Ethiopian February
conventions and MAC 10/6 are reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "output/tables/ftir24"

sys.path.insert(0, str(REPO / "calibration_explorer"))
from hips_lab import york_site  # noqa: E402

MODELS = {
    "OCEC-800 raw k6": {
        "cohort": "ocec", "cutoff": 800, "selection_space": "raw",
        "spectra": "raw", "mode": "site_heldout", "k": 6,
    },
    "OCEC-800 AIRSpec k5": {
        "cohort": "ocec", "cutoff": 800, "selection_space": "raw",
        "spectra": "airspec", "mode": "site_heldout", "k": 5,
    },
}


def _post(base_url: str, body: dict) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}/api/run", json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=900) as response:  # noqa: S310
        output = json.load(response)
    if "error" in output:
        raise RuntimeError(output["error"])
    return output


def _season(month: int, convention: str) -> str:
    if convention == "dry_feb":
        if month in (10, 11, 12, 1, 2):
            return "Dry"
        if month in (3, 4, 5):
            return "Belg"
    elif convention == "belg_feb":
        if month in (10, 11, 12, 1):
            return "Dry"
        if month in (2, 3, 4, 5):
            return "Belg"
    if month in (6, 7, 8, 9):
        return "Kiremt"
    raise ValueError(f"unrecognized month {month}")


def _descriptive(group: pd.DataFrame, mac: float) -> dict:
    x = group["Fabs"].to_numpy(float) / mac
    y = group["predicted"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    correlation = np.corrcoef(x, y)[0, 1]
    york = york_site(y, group["Fabs"].to_numpy(float), "ETAD")
    # york_site is parameterized at MAC=10; transform its slope for other MACs.
    york_slope = york["slope"] * mac / 10.0
    return {
        "n": len(group), "x_min": x.min(), "x_max": x.max(),
        "x_range": np.ptp(x), "ols_slope": slope, "ols_intercept": intercept,
        "R2": correlation ** 2, "RMSE_identity": np.sqrt(np.mean((y - x) ** 2)),
        "mean_residual": np.mean(y - x), "york_slope": york_slope,
        "york_intercept": york["intercept"],
    }


def _interaction(frame: pd.DataFrame, mac: float) -> dict:
    x = frame["Fabs"].to_numpy(float) / mac
    dry = frame["binary_season"].eq("Dry").to_numpy(float)
    design = pd.DataFrame({
        "constant": 1.0, "hips_bc": x, "dry_intercept": dry,
        "dry_slope": dry * x,
    })
    fit = sm.OLS(frame["predicted"].to_numpy(float), design).fit(cov_type="HC3")
    confidence = fit.conf_int(alpha=0.05)
    joint = fit.wald_test("dry_intercept = 0, dry_slope = 0", scalar=True)
    return {
        "n": len(frame), "wet_intercept": fit.params["constant"],
        "wet_slope": fit.params["hips_bc"],
        "dry_intercept_delta": fit.params["dry_intercept"],
        "dry_intercept_delta_se": fit.bse["dry_intercept"],
        "dry_intercept_delta_p": fit.pvalues["dry_intercept"],
        "dry_intercept_delta_ci_low": confidence.loc["dry_intercept", 0],
        "dry_intercept_delta_ci_high": confidence.loc["dry_intercept", 1],
        "dry_slope_delta": fit.params["dry_slope"],
        "dry_slope_delta_se": fit.bse["dry_slope"],
        "dry_slope_delta_p": fit.pvalues["dry_slope"],
        "dry_slope_delta_ci_low": confidence.loc["dry_slope", 0],
        "dry_slope_delta_ci_high": confidence.loc["dry_slope", 1],
        "dry_intercept": fit.params["constant"] + fit.params["dry_intercept"],
        "dry_slope": fit.params["hips_bc"] + fit.params["dry_slope"],
        "joint_wald_chi2": float(joint.statistic), "joint_p": float(joint.pvalue),
        "model_R2": fit.rsquared,
    }


def run(base_url: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples, descriptive, interactions = [], [], []
    for model_name, config in MODELS.items():
        output = _post(base_url, {**config, "target": "addis"})
        evaluation = output["eval"]
        frame = pd.DataFrame({
            "Fabs": evaluation["ref"], "predicted": evaluation["pred"],
            "Date": pd.to_datetime(evaluation["date"], errors="coerce"),
            "fixed": evaluation["fixed"],
        })
        if frame["Date"].isna().any():
            raise ValueError(f"{model_name}: dated evaluation unexpectedly has null dates")
        frame["model"] = model_name
        samples.append(frame)
        for convention in ("dry_feb", "belg_feb"):
            seasonal = frame.copy()
            seasonal["season"] = seasonal["Date"].dt.month.map(
                lambda month: _season(int(month), convention)
            )
            seasonal["binary_season"] = np.where(
                seasonal["season"].eq("Dry"), "Dry", "Belg+Kiremt"
            )
            for scope, selected in (
                ("all", seasonal), ("fixed", seasonal[seasonal["fixed"]]),
            ):
                for mac in (10.0, 6.0):
                    for season, group in selected.groupby("season"):
                        descriptive.append({
                            "model": model_name, "convention": convention,
                            "scope": scope, "MAC": mac, "season": season,
                            **_descriptive(group, mac),
                        })
                    interactions.append({
                        "model": model_name, "convention": convention,
                        "scope": scope, "MAC": mac, **_interaction(selected, mac),
                    })
    return pd.concat(samples, ignore_index=True), pd.DataFrame(descriptive), pd.DataFrame(interactions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5058")
    args = parser.parse_args()
    samples, descriptive, interactions = run(args.base_url)
    OUT.mkdir(parents=True, exist_ok=True)
    samples.to_csv(OUT / "locked_model_predictions.csv", index=False)
    descriptive.to_csv(OUT / "season_crossplots.csv", index=False)
    interactions.to_csv(OUT / "season_interactions.csv", index=False)
    show = interactions[
        interactions["scope"].eq("all") & interactions["MAC"].eq(10)
    ]
    columns = [
        "model", "convention", "n", "wet_slope", "wet_intercept",
        "dry_slope", "dry_intercept", "dry_slope_delta",
        "dry_slope_delta_p", "dry_intercept_delta",
        "dry_intercept_delta_p", "joint_p",
    ]
    print(show[columns].round(4).to_string(index=False))
    print("\nPer-season descriptive fits (all, MAC 10):")
    desc = descriptive[descriptive["scope"].eq("all") & descriptive["MAC"].eq(10)]
    print(desc[[
        "model", "convention", "season", "n", "x_range", "ols_slope",
        "ols_intercept", "R2", "mean_residual", "york_slope", "york_intercept",
    ]].round(3).to_string(index=False))
    print(f"\nwrote ftir_24 tables to {OUT}")


if __name__ == "__main__":
    main()
