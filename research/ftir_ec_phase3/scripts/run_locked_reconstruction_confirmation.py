"""Apply pre-selected FTIR calibrations to reconstructed HIPS holdouts.

The configurations in ``LOCKED_CONFIGS`` were selected on the exhaustive
five-site grid before the 54 raw HIPS measurements were reconstructed.  This
script deliberately does no model or component re-selection.  It queries the
calibration explorer, independently recomputes the crossplot metrics, and
writes both per-filter predictions and a bootstrap summary.

Start the explorer first (Flask can be supplied ephemerally if needed)::

    PORT=5058 uv run --with flask python calibration_explorer/app.py
    uv run python research/ftir_ec_phase3/scripts/\
        run_locked_reconstruction_confirmation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TARGETS = REPO / "calibration_explorer/targets"
OUT = HERE.parent / "output/tables/variation_closure"

LOCKED_CONFIGS = {
    "addis_winner_k8": {
        "cohort": "ocec", "cutoff": 440, "selection_space": "raw",
        "spectra": "airspec", "mode": "site_heldout", "k": 8,
    },
    "addis_winner_k9": {
        "cohort": "ocec", "cutoff": 440, "selection_space": "raw",
        "spectra": "airspec", "mode": "site_heldout", "k": 9,
    },
    "delhi_winner_k20": {
        "cohort": "analogs", "cutoff": 530, "selection_space": "airspec",
        "spectra": "deriv2", "mode": "site_heldout", "k": 20,
    },
    "common_candidate_k20": {
        "cohort": "analogs", "cutoff": 440, "selection_space": "airspec",
        "spectra": "deriv2", "mode": "site_heldout", "k": 20,
    },
}

TARGET_NAMES = (
    "addis_reconstructed_holdout", "etbi_reconstructed_holdout",
    "indh_reconstructed_holdout", "addis_augmented", "etbi_augmented",
    "indh_augmented",
)


def _post(base_url: str, body: dict) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}/api/run",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=900) as response:  # noqa: S310
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc
    except URLError as exc:
        raise RuntimeError(f"calibration explorer is unavailable: {exc}") from exc
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def _metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    resid = y - fitted
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "R2": float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean((y - x) ** 2))),
        "mean_bias": float(np.mean(y - x)),
        "median_bias": float(np.median(y - x)),
    }


def _bootstrap(x: np.ndarray, y: np.ndarray, seed: int = 20260824,
               n_boot: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    slopes, intercepts = [], []
    for _ in range(n_boot):
        take = rng.integers(0, len(x), len(x))
        if np.ptp(x[take]) == 0:
            continue
        slope, intercept = np.polyfit(x[take], y[take], 1)
        slopes.append(slope)
        intercepts.append(intercept)
    return {
        "slope_ci_low": float(np.percentile(slopes, 2.5)),
        "slope_ci_high": float(np.percentile(slopes, 97.5)),
        "intercept_ci_low": float(np.percentile(intercepts, 2.5)),
        "intercept_ci_high": float(np.percentile(intercepts, 97.5)),
    }


def _reference(target: str) -> pd.DataFrame:
    frame = pd.read_csv(TARGETS / target / "reference.csv")
    required = {"ExternalFilterId", "Fabs", "Volume_m3", "ReferenceSource"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{target}: missing reference columns {sorted(missing)}")
    if frame["ExternalFilterId"].duplicated().any():
        raise ValueError(f"{target}: duplicate ExternalFilterId values")
    return frame.reset_index(drop=True)


def run(base_url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, summaries = [], []
    for config_name, config in LOCKED_CONFIGS.items():
        for target in TARGET_NAMES:
            output = _post(base_url, {**config, "target": target})
            reference = _reference(target)
            observed_fabs = np.asarray(output["eval"]["ref"], float)
            predicted = np.asarray(output["eval"]["pred"], float)
            if len(reference) != len(observed_fabs) or len(predicted) != len(reference):
                raise ValueError(
                    f"{config_name}/{target}: API/reference length mismatch "
                    f"({len(predicted)}, {len(observed_fabs)}, {len(reference)})"
                )
            if not np.allclose(reference["Fabs"], observed_fabs, atol=5e-4):
                raise ValueError(f"{config_name}/{target}: API/reference order mismatch")
            observed = observed_fabs / 10.0
            stats = _metrics(observed, predicted)
            ci = _bootstrap(observed, predicted)
            summaries.append({
                "config": config_name, "target": target, "n": len(reference),
                "reference_source": "+".join(sorted(reference["ReferenceSource"].unique())),
                "heldout_tor_R2": output["heldout"]["R2"],
                "extrap_pct": output["target"]["extrap_pct"],
                **config, **stats, **ci,
            })
            per_filter = reference.copy()
            per_filter["observed_bc_mac10_ugm3"] = observed
            per_filter["predicted_ec_ugm3"] = predicted
            per_filter["residual_ugm3"] = predicted - observed
            per_filter.insert(0, "target", target)
            per_filter.insert(0, "config", config_name)
            predictions.append(per_filter)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5058")
    args = parser.parse_args()
    predictions, summary = run(args.base_url)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "locked_reconstruction_predictions.csv", index=False)
    summary.to_csv(OUT / "locked_reconstruction_summary.csv", index=False)
    holdout = summary[summary["target"].str.endswith("_holdout")]
    columns = [
        "config", "target", "n", "ols_slope", "ols_intercept", "R2",
        "RMSE", "mean_bias", "extrap_pct", "slope_ci_low", "slope_ci_high",
    ]
    print(holdout[columns].round(3).to_string(index=False))
    print(f"\nwrote {len(predictions):,} predictions and {len(summary)} summaries to {OUT}")


if __name__ == "__main__":
    main()
