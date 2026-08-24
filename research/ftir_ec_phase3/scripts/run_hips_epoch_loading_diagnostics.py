"""Diagnose HIPS epoch, loading, and calibration-line sensitivity.

Consumes the locked per-filter predictions, joins them to raw HIPS internals,
and evaluates three reference variants: the deployed calibration line, a
linear refit, and a quadratic refit.  Refits are calibration-set-specific
(``lot × deployed intercept/slope``), because pooling all blank lines within a
lot would mix known instrument/configuration epochs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "output/tables/variation_closure"
PREDICTIONS = OUT / "locked_reconstruction_predictions.csv"
RECONSTRUCTED = HERE.parent / "output/tables/hips/reconstructed_fabs.csv"
RAW = Path("/Users/ahmadjalil/Downloads/hips/spartan_hips_raw_all.csv")

sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(REPO / "calibration_explorer"))

from pls_transfer import FTIRTransferPaths  # noqa: E402
from hips_lab import york_site  # noqa: E402

PATHS = FTIRTransferPaths.defaults()
SITE_CODE = {
    "addis_augmented": "ETAD",
    "etbi_augmented": "ETBI",
    "indh_augmented": "INDH",
}


def _lot(value) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _line_key(lot, intercept, slope) -> tuple[str, float, float] | None:
    if pd.isna(intercept) or pd.isna(slope):
        return None
    return _lot(lot), round(float(intercept), 3), round(float(slope), 4)


def _epoch(timestamp) -> str:
    if pd.isna(timestamp):
        return "missing"
    day = pd.Timestamp(timestamp).normalize()
    if day <= pd.Timestamp("2023-03-17"):
        return "E1 (through 2023-03-17)"
    if pd.Timestamp("2023-05-03") <= day <= pd.Timestamp("2023-08-22"):
        return "E2 (2023-05-03 to 2023-08-22)"
    if day >= pd.Timestamp("2023-09-22"):
        return "E3 (from 2023-09-22)"
    return "transition/unobserved"


def _calibration_set_refits(batch: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    blanks = batch[batch["FilterType"].isin(["FB", "LB"])].dropna(
        subset=["T1", "R1", "Intercept", "Slope"]
    )
    refits, ledger = {}, []
    for values, group in blanks.groupby(["LotId", "Intercept", "Slope"]):
        lot, deployed_intercept, deployed_slope = values
        key = _line_key(lot, deployed_intercept, deployed_slope)
        if key is None or len(group) < 5:
            continue
        reflectance = group["R1"].to_numpy(float)
        transmittance = group["T1"].to_numpy(float)
        linear = np.polyfit(reflectance, transmittance, 1)
        quadratic = np.polyfit(reflectance, transmittance, 2)
        refits[key] = {"linear": linear, "quadratic": quadratic}
        ledger.append({
            "lot": key[0], "deployed_intercept": key[1],
            "deployed_slope": key[2], "n_blanks": len(group),
            "blank_r1_min": reflectance.min(), "blank_r1_max": reflectance.max(),
            "linear_intercept": linear[1], "linear_slope": linear[0],
            "quadratic_a2": quadratic[0], "quadratic_a1": quadratic[1],
            "quadratic_a0": quadratic[2],
            "rms_deployed": np.sqrt(np.mean(
                (transmittance - (float(deployed_intercept)
                                  + float(deployed_slope) * reflectance)) ** 2
            )),
            "rms_linear": np.sqrt(np.mean((transmittance - np.polyval(
                linear, reflectance
            )) ** 2)),
            "rms_quadratic": np.sqrt(np.mean((transmittance - np.polyval(
                quadratic, reflectance
            )) ** 2)),
        })
    return refits, pd.DataFrame(ledger)


def _instrument_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW, encoding="utf-8-sig", low_memory=False)
    sample = raw[raw["ResultTypeId"].eq(0)].copy()
    duplicate = sample[sample["ExternalFilterId"].duplicated(False)]
    sample = sample.sort_values("Timestamp").drop_duplicates(
        "ExternalFilterId", keep="last"
    )
    sample["AnalysisTimestamp_raw"] = pd.to_datetime(
        sample["Timestamp"], format="mixed", errors="coerce"
    )
    sample["SamplingStartDate_raw"] = pd.to_datetime(
        sample["SamplingStartDate"], format="mixed", errors="coerce"
    )
    sample["instrument_gain"] = sample["Transmittance"] / sample["TransmittanceRaw"]
    keep = [
        "ExternalFilterId", "SiteCode", "ExternalLotId", "AnalysisTimestamp_raw",
        "SamplingStartDate_raw", "Transmittance", "Reflectance",
        "TransmittanceRaw", "ReflectanceRaw", "instrument_gain",
    ]
    quality = pd.DataFrame([{
        "raw_rows": len(raw), "result0_rows": int(raw["ResultTypeId"].eq(0).sum()),
        "result0_unique_filters": sample["ExternalFilterId"].nunique(),
        "duplicate_result0_rows": len(duplicate),
        "telemetry_nonnull": int(raw[[
            "LaserPower", "LaserDiodeTemperature", "TransmittanceSensorTemperature",
            "ReflectanceSensorTemperature",
        ]].notna().sum().sum()),
    }])
    return sample[keep], quality


def _enrich_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[predictions["target"].isin(SITE_CODE)].copy()
    raw, quality = _instrument_frame()
    batch_columns = [
        "FilterId", "Site", "FilterType", "LotId", "T1", "R1", "Intercept",
        "Slope", "DepositArea", "Volume", "Fabs",
    ]
    batch = pd.read_csv(
        PATHS.spartan_hips_primary, encoding="cp1252", usecols=batch_columns,
        low_memory=False,
    ).drop_duplicates("FilterId")
    reconstructed = pd.read_csv(RECONSTRUCTED).rename(columns={
        "FilterId": "ExternalFilterId",
        "Lot": "Lot_reconstructed",
        "AnalysisTimestamp": "AnalysisTimestamp_reconstructed",
        "T1": "T1_reconstructed", "R1": "R1_reconstructed",
        "DepositArea": "DepositArea_reconstructed",
        "CalibrationIntercept": "Intercept_reconstructed",
        "CalibrationSlope": "Slope_reconstructed",
    })
    joined = predictions.merge(raw, on="ExternalFilterId", how="left", validate="many_to_one")
    joined = joined.merge(
        batch, left_on="ExternalFilterId", right_on="FilterId", how="left",
        validate="many_to_one", suffixes=("", "_batch"),
    )
    joined = joined.merge(reconstructed[[
        "ExternalFilterId", "Lot_reconstructed", "AnalysisTimestamp_reconstructed",
        "T1_reconstructed", "R1_reconstructed", "DepositArea_reconstructed",
        "Intercept_reconstructed", "Slope_reconstructed",
    ]], on="ExternalFilterId", how="left", validate="many_to_one")

    joined["analysis_timestamp"] = joined["AnalysisTimestamp_raw"].fillna(
        pd.to_datetime(joined["AnalysisTimestamp_reconstructed"], format="mixed",
                       errors="coerce")
    )
    joined["epoch"] = joined["analysis_timestamp"].map(_epoch)
    joined["analysis_day"] = joined["analysis_timestamp"].dt.date.astype(str).where(
        joined["analysis_timestamp"].notna(), "missing"
    )
    joined["lot"] = joined["LotId"].fillna(joined["Lot_reconstructed"]).map(_lot)
    joined["hips_t1"] = joined["T1"].fillna(joined["T1_reconstructed"])
    joined["hips_r1"] = joined["R1"].fillna(joined["R1_reconstructed"])
    joined["line_intercept"] = joined["Intercept"].fillna(
        joined["Intercept_reconstructed"]
    )
    joined["line_slope"] = joined["Slope"].fillna(joined["Slope_reconstructed"])
    joined["deposit_area"] = joined["DepositArea"].fillna(
        joined["DepositArea_reconstructed"]
    )
    joined["volume"] = joined["Volume"].fillna(joined["Volume_m3"])
    joined["line_key"] = [
        _line_key(lot, intercept, slope) for lot, intercept, slope in zip(
            joined["lot"], joined["line_intercept"], joined["line_slope"]
        )
    ]

    refits, ledger = _calibration_set_refits(batch)
    for variant in ("linear", "quadratic"):
        top = np.full(len(joined), np.nan)
        for idx, (key, reflectance) in enumerate(zip(joined["line_key"], joined["hips_r1"])):
            if key in refits and pd.notna(reflectance):
                top[idx] = np.polyval(refits[key][variant], float(reflectance))
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.log(top / joined["hips_t1"].to_numpy(float))
        joined[f"fabs_{variant}"] = (
            100 * tau * joined["deposit_area"].to_numpy(float)
            / joined["volume"].to_numpy(float)
        )
    joined["fabs_deployed"] = joined["Fabs"]
    joined["tau_deployed"] = (
        joined["fabs_deployed"] * joined["volume"] / (100 * joined["deposit_area"])
    )
    limits = ledger.set_index([
        "lot", "deployed_intercept", "deployed_slope"
    ])["blank_r1_min"].to_dict()
    joined["blank_r1_min"] = joined["line_key"].map(limits)
    joined["below_blank_r1"] = joined["hips_r1"] < joined["blank_r1_min"]

    match = (joined.groupby(["target", "ReferenceSource"], as_index=False)
             .agg(rows=("ExternalFilterId", "size"),
                  unique_filters=("ExternalFilterId", "nunique"),
                  raw_match=("AnalysisTimestamp_raw", "count"),
                  line_match=("line_intercept", "count"),
                  alternative_refit_match=("fabs_linear", "count")))
    return joined, ledger, pd.concat([quality, pd.DataFrame([{
        "prediction_rows": len(joined), "prediction_unique_filters": joined["ExternalFilterId"].nunique(),
        "raw_timestamp_coverage": float(joined["AnalysisTimestamp_raw"].notna().mean()),
        "calibration_line_coverage": float(joined["line_intercept"].notna().mean()),
        "alternative_refit_coverage": float(joined["fabs_linear"].notna().mean()),
    }])], ignore_index=True), match


def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept = np.polyfit(x, y, 1)
    correlation = np.corrcoef(x, y)[0, 1]
    return {
        "ols_slope": slope, "ols_intercept": intercept,
        "R2": correlation ** 2,
        "RMSE_identity": np.sqrt(np.mean((y - x) ** 2)),
        "mean_bias": np.mean(y - x),
    }


def _summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, loading = [], []
    groupings = {
        "overall": [], "reference_source": ["ReferenceSource"],
        "epoch": ["epoch"], "analysis_day": ["analysis_day"],
        "below_blank_range": ["below_blank_r1"],
    }
    for (config, target), base in frame.groupby(["config", "target"]):
        site_code = SITE_CODE[target]
        for grouping, columns in groupings.items():
            iterator = [("all", base)] if not columns else base.groupby(columns[0], dropna=False)
            for label, group in iterator:
                for variant in ("deployed", "linear", "quadratic"):
                    fabs = group[f"fabs_{variant}"].to_numpy(float)
                    predicted = group["predicted_ec_ugm3"].to_numpy(float)
                    ok = np.isfinite(fabs) & np.isfinite(predicted)
                    if ok.sum() < 8 or np.ptp(fabs[ok]) == 0:
                        continue
                    observed = fabs[ok] / 10.0
                    fit = _ols(observed, predicted[ok])
                    york = york_site(predicted[ok], fabs[ok], site_code)
                    rows.append({
                        "config": config, "target": target, "grouping": grouping,
                        "group": str(label), "reference_variant": variant,
                        "n": int(ok.sum()), "below_blank_r1_pct": 100 * group.loc[
                            group.index[ok], "below_blank_r1"
                        ].mean(), **fit,
                        "york_slope": york["slope"],
                        "york_slope_se": york["slope_se"],
                        "york_intercept": york["intercept"],
                        "york_intercept_se": york["intercept_se"],
                        "york_kappa": york["kappa"],
                    })
        residual = base["predicted_ec_ugm3"] - base["fabs_deployed"] / 10.0
        for predictor in ("tau_deployed", "hips_r1", "instrument_gain"):
            ok = np.isfinite(residual) & np.isfinite(base[predictor])
            if ok.sum() < 8 or base.loc[ok, predictor].nunique() < 3:
                continue
            rho, pvalue = spearmanr(base.loc[ok, predictor], residual[ok])
            loading.append({
                "config": config, "target": target, "predictor": predictor,
                "n": int(ok.sum()), "spearman_r": rho, "p_value": pvalue,
            })
    return pd.DataFrame(rows), pd.DataFrame(loading)


def main() -> None:
    enriched, ledger, quality, match = _enrich_predictions()
    summary, loading = _summaries(enriched)
    OUT.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUT / "hips_epoch_loading_per_filter.csv", index=False)
    ledger.to_csv(OUT / "hips_calibration_set_blank_refits.csv", index=False)
    quality.to_csv(OUT / "hips_epoch_loading_quality.csv", index=False)
    match.to_csv(OUT / "hips_epoch_loading_join_coverage.csv", index=False)
    summary.to_csv(OUT / "hips_epoch_loading_summary.csv", index=False)
    loading.to_csv(OUT / "hips_loading_correlations.csv", index=False)
    headline = summary[
        summary["grouping"].isin(["reference_source", "epoch"])
        & summary["reference_variant"].eq("deployed")
        & summary["config"].isin(["addis_winner_k8", "delhi_winner_k20"])
    ]
    columns = [
        "config", "target", "grouping", "group", "n", "ols_slope",
        "ols_intercept", "R2", "york_slope", "york_intercept",
        "below_blank_r1_pct",
    ]
    print(headline[columns].round(3).to_string(index=False))
    print("\njoin coverage:")
    print(match.to_string(index=False))
    print("\nquality:")
    print(quality.to_string(index=False))
    print(f"\nwrote epoch/loading diagnostics to {OUT}")


if __name__ == "__main__":
    main()
