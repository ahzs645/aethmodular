"""Build data-linked speaker notes and a concise supplementary analysis report."""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "research/ftir_hips_chem"
TABLES = ACTIVE / "output/tables/ann_weekly_20260910_diagnostics"
DELIVERY = ROOT / "deliverables/ann_weekly_2026-09-10"
NOTEBOOK = ACTIVE / "notebooks/archive/executed/ann_weekly_20260910_diagnostics.ipynb"
FIGURES = ACTIVE / "output/plots/ann_weekly_20260910_diagnostics"
BUILD = DELIVERY / ".build/diagnostics"
BUILD.mkdir(parents=True, exist_ok=True)
pair = pd.read_csv(TABLES / "paired_model_comparison.csv")
stability = pd.read_csv(TABLES / "analog_stability_summary.csv")
shape = pd.read_csv(TABLES / "pca_season_summary.csv")
sites = pd.read_csv(TABLES / "site_concentration.csv")
monthly = pd.read_csv(TABLES / "monthly_discrepancy.csv")
methods = json.loads((TABLES / "methods.json").read_text())

paired_detail = "\n".join(
    f"{r.season}, {r.candidate}: RMS {r.baseline_rmse:.3f} → {r.candidate_rmse:.3f}; change {r.delta_rmse:+.3f} [{r.delta_rmse_low:+.3f}, {r.delta_rmse_high:+.3f}] µg/m³."
    for r in pair.itertuples()
)
stable_detail = "\n".join(
    f"{r.group}: median retention {100 * r.median_retained:.1f}%; 5th–95th resample percentiles {100 * r.p05_retained:.1f}–{100 * r.p95_retained:.1f}%; {r.stable_original_n}/500 original filters selected in at least 80% of draws; union {r.union_selected_n}."
    for r in stability.itertuples()
)
shape_detail = "\n".join(
    f"{r.season}: {r.above_source_q95}/{r.n} ({100 * r.fraction_above:.1f}%) above source 95th-percentile reconstruction residual."
    for r in shape.itertuples()
)
site_detail = "\n".join(
    f"{r.group}: {r.n_sites} distinct sites; {r.effective_sites:.1f} effective sites; top five supply {100 * r.top5_fraction:.1f}% of filters."
    for r in sites.itertuples()
)

specs = [
    dict(
        number=13,
        title="Dry and Belg increase disagreement with HIPS",
        footer="Positive = greater disagreement. Same filters; 95% month-bootstrap intervals.",
        talk="I compared the models on exactly the same Addis filters, with the historical calibration as the reference. The Dry and Belg seasonal models increase RMS disagreement with HIPS by about 1.11 and 1.00 micrograms per cubic metre. Kiremt improves by only 0.09, and its interval includes no improvement. The pooled analog model increases disagreement in all three seasons. This gives a more direct comparison than ranking models by correlation alone.",
        detail="HIPS Fabs/MAC is a proxy, not chemical EC truth. Differences are candidate minus historical RMS discrepancy. Four thousand whole-calendar-month bootstrap draws keep all models and HIPS paired. Predictions, model choices and MAC remain fixed; intervals exclude fitting/selection uncertainty and ignore dependence between months. Existing Addis data remain exploratory.\n"
        + paired_detail,
        files=["paired_model_comparison.csv", "paired_addis_predictions.csv"],
        urls=["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html"],
    ),
    dict(
        number=14,
        title="Seasonal matching does not guarantee agreement",
        footer="Each column uses identical Addis filters. HIPS Fabs/MAC is the comparison proxy.",
        talk="Each column holds the Addis evaluation filters constant; each row changes the calibration. The left heatmap shows RMS disagreement and the right shows correlation. The Dry-selected calibration has large disagreement in every season. The Kiremt calibration gives the smallest RMS discrepancy in Dry and Kiremt, while the historical calibration is slightly lower in Belg. This is a useful clue about transfer, but it does not make the Kiremt calibration acceptable: its source-site TOR screening was weak.",
        detail="All five models were already fitted before this diagnostic. The seasonal selection masks exclude 1800–2500 cm⁻¹; PLS uses the full grid. R² means squared Pearson correlation. This heatmap provides no paired uncertainty for the cross-season Kiremt-versus-historical comparison on Dry, so the apparent ranking is descriptive. No model is selected or promoted based on this display. Fixed HIPS MAC, previously explored target data, differing source cohorts and their weak TOR validation limit inference.",
        files=["cross_season_transfer.csv", "paired_addis_predictions.csv"],
        urls=[],
    ),
    dict(
        number=15,
        title="Monthly differences remain below zero",
        footer="Monthly means; sample counts vary. Seasonal line switches calibration by season.",
        talk="The monthly plot shows where the disagreement occurs in time. All three calibration approaches remain below the HIPS proxy in every observed monthly mean. The historical calibration is generally closer, while the seasonal approach changes sharply as its calibration switches between seasons. These jumps could arise from the modelling rule as well as changes in the aerosol, so this plot does not establish a physical cause.",
        detail=f"There are {monthly.month.nunique()} observed calendar months, with {int(monthly.n.min())}–{int(monthly.n.max())} filters per month. Means are calculated over the available filters; months with one filter are retained and do not have a meaningful within-month correlation. Missing calendar months break the lines. Month counts and per-model RMS, MAE, bias and R² are retained in the table. This is not a uniformly sampled monthly climatology. No meteorological adjustment is applied.",
        files=["monthly_discrepancy.csv", "paired_addis_predictions.csv"],
        urls=[],
    ),
    dict(
        number=16,
        title="Changing sampled months changes the analog lists",
        footer="Boxes: 25–75%; whiskers: 5–95% of 200 resamples. The 80% marker is descriptive.",
        talk="I resampled the observed months and repeated the entire analog selection, including the median spectrum. Median overlap with the original list ranges from 76 to 87 percent. The pooled and Belg lists are less stable than Dry and Kiremt in this experiment. The curves show a stable core and a less stable edge. That suggests we should examine sensitivity to calibration-list membership before treating an exact top-500 list as definitive.",
        detail="Every draw samples as many whole year-month blocks as observed, with replacement, then keeps all filters in each sampled block. The All-Addis draw is not season-stratified, so seasonal balance can vary. Correlation is signed, centred Pearson r over retained channels; duplicate scans collapse to physical filters using the same deterministic ranking helper. The baseline exactly reproduces every saved primary list. These resample distributions are conditional sampling sensitivity, not uncertainty intervals for calibration performance. No PLS model is refit.\n"
        + stable_detail,
        files=[
            "analog_stability_summary.csv",
            "analog_bootstrap_draws.csv",
            "analog_selection_frequencies.csv",
        ],
        urls=[],
    ),
    dict(
        number=17,
        title="Belg more often exceeds the source residual threshold",
        footer="Source-relative spectral diagnostic; it does not establish prediction accuracy.",
        talk="The map summarizes spectral shape after the CO₂ region is removed. Grey points are the eligible IMPROVE library; colours show Addis seasons. The two displayed components explain about 75 percent of source shape variation. A separate ten-component reconstruction check finds 10 of 61 Belg spectra above the source residual threshold, compared with 5 of 105 Dry and 3 of 73 Kiremt spectra. Belg therefore deserves a closer inspection of the individual spectra, without automatically excluding these filters.",
        detail=f"Each masked spectrum is centred across channels and normalized to unit length, matching Pearson geometry. PCA is fitted only to {methods['pca_source_n']:,} unique eligible source filters, choosing the lowest AnalysisId for duplicate source filters. Ten fixed PCs explain {100 * sum(methods['pca_variance_ratio']):.2f}% of source variance. Addis is projected using the source basis. The threshold is the in-sample source 95th percentile of squared reconstruction residual ({methods['source_q95']:.6g}); it is descriptive, not a calibrated acceptance rule. The 5% dashed line is the source reference fraction. No chemical meaning is assigned to PCs, and no uncertainty or significance test is claimed for seasonal percentages.\n"
        + shape_detail,
        files=[
            "pca_target_scores.csv",
            "pca_source_scores.csv",
            "pca_season_summary.csv",
            "methods.json",
        ],
        urls=["https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"],
    ),
    dict(
        number=18,
        title="Rainy-season cohorts are more concentrated by site",
        footer="Effective sites = 1/Σp². A concentration measure, not an independent sample count.",
        talk="All cohorts contain 500 filters, but those filters are distributed differently across source sites. Dry has 121 distinct sites and about 70 effective sites. Belg and Kiremt have only about 32 and 29 effective sites. In each rainy-season list, five sites supply 32 percent of filters. This suggests a concrete sensitivity check: repeat calibration with source-site contributions capped, then compare models on a common source test set.",
        detail="The effective site count is the inverse of the sum of squared site shares, also called inverse-Simpson concentration. Equal site contributions recover the distinct count; concentration reduces it. This is not an estimate of independent filters or a replacement for grouped validation. It is computed over the full selected 500-filter cohort, including its source training and test roles. Site identities and complete ranked contribution curves are supplied. Site capping and calibration refitting are a next experiment, not a result of this diagnostic.\n"
        + site_detail,
        files=["site_concentration.csv", "site_concentration_curves.csv"],
        urls=[],
    ),
]
for s in specs:
    s["image"] = str(FIGURES / f"slide_{s['number']}.png")
    sources = [str(TABLES / name) for name in s["files"]] + [str(NOTEBOOK), s["image"]] + s["urls"]
    s["notes"] = f"Talk track\n{s['talk']}\n\nIf asked\n{s['detail']}\n\nSources\n" + "\n".join(
        sources
    )
    s["notes"] += "\nFigure format: PNG exported from Jupyter, as requested."
(BUILD / "content.json").write_text(json.dumps(specs, indent=2))

parts = [
    "# Additional Addis FTIR calculations — 10 September 2026\n\n"
    "The paired comparison reinforces the concern about the Dry and Belg calibrations. New diagnostics also show that analog lists depend on sampled months and that rainy-season cohorts concentrate heavily in a few source sites. "
    "The six new notebook figures are appended as slides 13–18, leaving the main weekly update intact.\n\n"
    "All Addis results are exploratory and use HIPS Fabs/MAC as a comparison proxy. They do not establish chemical-EC accuracy. No PLS calibration is refit and the proposed validation split remains unscored.\n"
]
for s in specs:
    parts.append(
        f"## {s['title']}\n\n{s['talk']}\n\n{s['detail']}\n\n![{s['title']}]({s['image']})\n\n"
        + "Sources: "
        + ", ".join(f"[{n}]({TABLES / n})" for n in s["files"])
        + ".\n"
    )
parts.append(
    "## Reproduce and interpret\n\n"
    "Run `uv run aeth doctor`, then `uv run python research/ftir_hips_chem/workflows/create_ann_weekly_diagnostics_notebook.py` from the repository root. "
    "The source notebook is `research/ftir_hips_chem/ann_weekly_20260910_diagnostics.ipynb`; run it from that active folder. "
    "The archive copy retains six displayed figures and all calculation outputs. Tables include exact paired predictions, resample-level overlap, per-filter selection frequencies, named site contributions and PCA scores. `methods.json` records seeds and input hashes; `pca_basis.npz` retains the source mean and component basis.\n\n"
    "The paired-resampling principle is described in [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); "
    "this analysis applies it to whole observed year-month clusters. Source-fitted PCA transforms follow [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).\n\n"
    "Useful next experiments are a common source-site test set for all candidates, source-site contribution caps, a fixed grid of analog counts, and direct inspection of the flagged Belg spectra. "
    "Set these choices before evaluating any newly reserved target filters. The present diagnostics do not supply results for those experiments.\n"
)
(DELIVERY / "output/diagnostics_report.md").write_text("\n".join(parts))
print(f"Prepared {len(specs)} slides and diagnostics_report.md")
