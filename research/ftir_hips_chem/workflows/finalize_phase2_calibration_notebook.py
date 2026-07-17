#!/usr/bin/env python3
"""Replace phase-2 notebook placeholder summaries with executed values."""

from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "ftir_10_ec_calibration_cohort_comparison.ipynb"
TABLES = ROOT / "output/tables/pls_calibration_phase2"


def metric_row(metrics, model, cohort="fixed common cohort", mac=10.0):
    return metrics[
        metrics["model"].eq(model)
        & metrics["cohort"].eq(cohort)
        & metrics["MAC_m2_g"].eq(mac)
    ].iloc[0]


if __name__ == "__main__":
    nb = nbformat.read(NOTEBOOK, as_version=4)
    metrics = pd.read_csv(TABLES / "addis_calibration_metrics.csv")
    holdout = pd.read_csv(TABLES / "site_held_out_tor_metrics.csv")
    audit = pd.read_csv(TABLES / "addis_calibration_data_audit.csv").iloc[0]

    deployed_name = "Deployed SPARTAN FTIR EC"
    smoke_name = "Smoke IMPROVE (906, k=31)"
    shaped_name = next(value for value in metrics["model"].unique() if value.startswith("Ethiopia-shaped"))
    locked_name = next(value for value in metrics["model"].unique() if value.startswith("Locked analog raw—first"))
    upper_name = next(value for value in metrics["model"].unique() if value.startswith("Locked analog >1500"))
    prior_name = "Prior top-400 analog—unlocked sensitivity"

    deployed = metric_row(metrics, deployed_name)
    smoke = metric_row(metrics, smoke_name)
    shaped = metric_row(metrics, shaped_name)
    locked = metric_row(metrics, locked_name)
    upper = metric_row(metrics, upper_name)
    prior = metric_row(metrics, prior_name)
    held = holdout[holdout["model"].str.startswith("Full-spectrum analog—first")].iloc[0]

    summary = f"""## tl;dr

On the fixed **{int(deployed['n'])}-filter Addis cohort** at MAC=10, the deployed SPARTAN values give
intercept **{deployed['intercept']:.2f}** and RMSE **{deployed['RMSE']:.2f} µg m⁻³**. The
meeting-described 906-sample smoke calibration does not fix the intercept
(**{smoke['intercept']:.2f}**). Selecting the 300 smoke spectra closest to Addis in CH, carbonyl,
and ~1600 cm⁻¹ feature space moves the intercept toward zero to **{shaped['intercept']:.2f}** and
gives RMSE **{shaped['RMSE']:.2f}**, but this remains HIPS/MAC-dependent.

The earlier top-400 analog result (RMSE **{prior['RMSE']:.2f}**) is not stable after locking a
site-held-out TOR test: the first-major-minimum refit has held-out TOR loading RMSE
**{held['RMSE']:.2f} µg/filter** and Addis RMSE **{locked['RMSE']:.2f}**. Restricting to
>1500 cm⁻¹ with simple offset correction also performs poorly (Addis RMSE **{upper['RMSE']:.2f}**),
confirming that offset correction cannot stand in for validated AIRSpec processing.
"""

    takeaways = f"""## Takeaways

- **The original smoke-only strategy is not the answer.** Its fixed-cohort intercept is
  {smoke['intercept']:.2f}, compared with {deployed['intercept']:.2f} for deployed SPARTAN EC.
- **Spectral selection helps the stated intercept objective.** The Ethiopia-shaped smoke cohort
  reaches {shaped['intercept']:.2f}, but only **7** smoke spectra fall inside the Addis 5th–95th
  percentile range for all three source-shape features; the 300-sample cohort is therefore nearest,
  not truly matched.
- **The prior analog improvement is split-sensitive.** The locked, disjoint-site TOR test does not
  reproduce the optimistic top-400 result, so that calibration should not replace the deployed model.
- **The >1500 cm⁻¹ sensitivity is a negative result, not an AIRSpec result.** A complete EDF 6–8
  preprocessing run is still required before concluding that the lower-wavenumber region should be
  excluded.
- **Validation disposition: needs revision before deployment.** The exported coefficient files are
  research candidates for controlled testing; the next decisive data item is an independent Addis
  TOR/EC reference or an agreed MAC/protocol bridge.
"""

    replaced = 0
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        if cell.source.startswith("# FTIR EC calibration cohort comparison"):
            marker = "## Context & Methods"
            context = cell.source[cell.source.index(marker):]
            cell.source = "# FTIR EC calibration cohort comparison: deployed, smoke, Ethiopia-shaped, and full-pool analog models\n\n" + summary + "\n" + context
            replaced += 1
        elif cell.source.startswith("## Takeaways"):
            cell.source = takeaways
            replaced += 1
    if replaced != 2:
        raise RuntimeError(f"expected to replace 2 summary cells, replaced {replaced}")
    nbformat.write(nb, NOTEBOOK)
    print(f"finalized {NOTEBOOK}; Addis HIPS n={int(audit['Addis_with_newer_HIPS'])}")
