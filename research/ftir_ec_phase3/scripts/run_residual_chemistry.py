"""Test potassium and dust tracers against locked calibration residuals."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "output/tables/variation_closure"
PREDICTIONS = OUT / "locked_reconstruction_predictions.csv"

sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
from data_matching import base_filter_id, load_filter_data  # noqa: E402

TARGET_SITE = {"addis_augmented": "ETAD", "indh_augmented": "INDH"}
PARAMETERS = {
    "K_ion": "ChemSpec_Potassium_Ion_PM2.5",
    "K_total": "ChemSpec_Potassium_PM2.5",
    "Al": "ChemSpec_Aluminum_PM2.5",
    "Si": "ChemSpec_Silicon_PM2.5",
    "Ti": "ChemSpec_Titanium_PM2.5",
    "Fe": "ChemSpec_Iron_PM2.5",
    "OC": "ChemSpec_OC_PM2.5",
    "EC_chem": "ChemSpec_EC_PM2.5",
}


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(controls, values, rcond=None)
    return values - controls @ beta


def _partial_test(frame: pd.DataFrame, predictor: str,
                  seed: int = 20260824, permutations: int = 5000) -> dict:
    columns = ["calibration_residual", "observed_bc_mac10_ugm3", "month", predictor]
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20 or data[predictor].nunique() < 5:
        return {"n": len(data)}
    month = data["month"].to_numpy(float)
    controls = np.column_stack([
        np.ones(len(data)), data["observed_bc_mac10_ugm3"].to_numpy(float),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
    ])
    y = _residualize(data["calibration_residual"].to_numpy(float), controls)
    x = _residualize(np.log1p(data[predictor].clip(lower=0)).to_numpy(float), controls)
    correlation = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for index in range(permutations):
        null[index] = np.corrcoef(x, rng.permutation(y))[0, 1]
    pvalue = (1 + np.sum(np.abs(null) >= abs(correlation))) / (permutations + 1)
    spearman, spearman_p = spearmanr(
        data[predictor], data["calibration_residual"], nan_policy="omit"
    )
    return {
        "n": len(data), "partial_r": correlation, "permutation_p": pvalue,
        "spearman_r": spearman, "spearman_p": spearman_p,
    }


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[predictions["target"].isin(TARGET_SITE)].copy()
    predictions["calibration_residual"] = (
        predictions["predicted_ec_ugm3"] - predictions["observed_bc_mac10_ugm3"]
    )
    predictions["base_filter_id"] = predictions["ExternalFilterId"].map(base_filter_id)
    predictions["month"] = pd.to_datetime(predictions["Date"], errors="coerce").dt.month

    filters = load_filter_data()
    chemistry = filters[
        filters["Parameter"].isin(PARAMETERS.values())
    ][["FilterId", "Site", "Parameter", "Concentration"]].copy()
    chemistry["base_filter_id"] = chemistry["FilterId"].map(base_filter_id)
    wide = chemistry.pivot_table(
        index=["base_filter_id", "Site"], columns="Parameter", values="Concentration",
        aggfunc="first",
    ).reset_index().rename(columns={value: key for key, value in PARAMETERS.items()})
    predictions["Site"] = predictions["target"].map(TARGET_SITE)
    joined = predictions.merge(
        wide, on=["base_filter_id", "Site"], how="left", validate="many_to_one",
    )
    joined["dust_index"] = joined[["Al", "Si", "Ti", "Fe"]].apply(
        lambda column: (np.log1p(column.clip(lower=0))
                        - np.log1p(column.clip(lower=0)).mean())
        / np.log1p(column.clip(lower=0)).std(),
        axis=0,
    ).mean(axis=1, skipna=False)
    joined["Kion_to_Al"] = joined["K_ion"] / joined["Al"].replace(0, np.nan)
    joined["OC_to_EC"] = joined["OC"] / joined["EC_chem"].replace(0, np.nan)

    predictors = ["K_ion", "K_total", "Al", "Si", "Ti", "Fe", "dust_index",
                  "Kion_to_Al", "OC_to_EC"]
    rows = []
    for (config, target), group in joined.groupby(["config", "target"]):
        for predictor in predictors:
            rows.append({
                "config": config, "target": target, "predictor": predictor,
                **_partial_test(group, predictor),
            })
    results = pd.DataFrame(rows)
    results["fdr_q"] = np.nan
    for _, indices in results.groupby(["config", "target"]).groups.items():
        usable = results.loc[indices, "permutation_p"].notna()
        use_indices = np.asarray(indices)[usable.to_numpy()]
        if len(use_indices):
            results.loc[use_indices, "fdr_q"] = multipletests(
                results.loc[use_indices, "permutation_p"], method="fdr_bh"
            )[1]
    OUT.mkdir(parents=True, exist_ok=True)
    joined.to_csv(OUT / "residual_chemistry_per_filter.csv", index=False)
    results.to_csv(OUT / "residual_chemistry_tests.csv", index=False)
    focus = results[
        ((results["config"].eq("addis_winner_k8")
          & results["target"].eq("addis_augmented"))
         | (results["config"].eq("delhi_winner_k20")
            & results["target"].eq("indh_augmented")))
    ]
    print(focus.round(4).to_string(index=False))
    print(f"\nwrote residual chemistry tables to {OUT}")


if __name__ == "__main__":
    main()
