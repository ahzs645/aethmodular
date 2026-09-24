"""Build the provenance audit and weekly report from the completed rerun tables.

Run after run_ann_weekly_20260910.py. No PLS fitting or target-side scoring here.
The ten meeting-figure labels below were transcribed from its original legend.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "research/ftir_hips_chem"
P3 = ROOT / "research/ftir_ec_phase3"
OUT = ACTIVE / "output/tables/ann_weekly_20260910"
PLOTS = ACTIVE / "output/plots/ann_weekly_20260910"
DELIVERABLE = ROOT / "deliverables/ann_weekly_2026-09-10/output"
sys.path.insert(0, str(P3 / "scripts"))
sys.path.insert(0, str(ACTIVE / "scripts"))

import pandas as pd
from aethmodular_cli.env import display_path
from config import ETHIOPIA_SEASONS
from phase3_common import PATHS
from plotting.utils import deming


def read(name):
    return pd.read_csv(OUT / f"{name}.csv")


def link_target(path):
    """Link target from the report: relative inside the repo, ``~/`` outside it."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(ROOT):
        return Path(os.path.relpath(resolved, DELIVERABLE)).as_posix()
    return display_path(path)


def link(path, label=None):
    return f"[{label or path.name}](<{link_target(path)}>)"


def table(headers, rows):
    def line(values):
        return "| " + " | ".join(str(v) for v in values) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *map(line, rows)])


def interval(row, term):
    return (
        f"{row['deming_' + term]:.3f} [{row[term + '_ci_low']:.3f}, {row[term + '_ci_high']:.3f}]"
    )


def main():
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT / "summary.json").read_text())
    library = read("library_eligibility")
    history = read("historical_addis_winner_membership")
    roles = history.set_index("FilterId").role.to_dict()
    legend = [
        ("Addis", "CHAS1", "2023-02-02"),
        ("Addis", "NOGA1", "2022-08-15"),
        ("Addis", "EGBE1", "2023-05-18"),
        ("Addis", "SOGP1", "2022-12-04"),
        ("Addis", "PUSO1", "2023-02-14"),
        ("Bishoftu", "DOSO1", "2023-01-15"),
        ("Bishoftu", "SHEN1", "2023-04-24"),
        ("Bishoftu", "BADL1", "2023-02-14"),
        ("Bishoftu", "BADL1", "2023-03-22"),
        ("Bishoftu", "DOSO1", "2023-03-28"),
    ]
    members = []
    for target, site, date in legend:
        match = library.loc[library.Site.eq(site) & library.date.eq(date)].copy()
        assert len(match) == 1, f"Ambiguous figure label: {site} {date}"
        match["target"] = target
        match["role"] = match.FilterId.map(roles).fillna("outside_cohort")
        members.append(match)
    meeting = pd.concat(members, ignore_index=True)
    meeting.to_csv(OUT / "meeting_figure_top5_membership.csv", index=False)

    # Audit lot identity separately from whether the HIPS value was shipped.
    bishoftu = pd.read_csv(ROOT / "calibration_explorer/targets/etbi_augmented/reference.csv")
    shipped = pd.read_csv(
        PATHS.spartan_hips_primary,
        encoding="cp1252",
        usecols=["FilterId", "LotId"],
    ).drop_duplicates()
    assert not shipped.FilterId.duplicated().any()
    raw_path = Path(
        os.environ.get(
            "AETHMODULAR_WEEKLY_RAW_HIPS_CSV",
            Path.home() / "Downloads/hips/spartan_hips_raw_all.csv",
        )
    )
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    raw.columns = [c.strip('"') for c in raw]
    raw = raw.loc[raw.ResultTypeId.eq(0), ["ExternalFilterId", "ExternalLotId"]].drop_duplicates()
    assert not raw.ExternalFilterId.duplicated().any()
    lot = (
        bishoftu.rename(columns={"LotId": "LotId_staged"})
        .merge(
            shipped.rename(columns={"LotId": "LotId_shipped"}),
            left_on="ExternalFilterId",
            right_on="FilterId",
            how="left",
            validate="one_to_one",
        )
        .merge(raw, on="ExternalFilterId", how="left", validate="one_to_one")
    )
    for column in ["ExternalLotId", "LotId_staged", "LotId_shipped"]:
        lot[column] = pd.to_numeric(lot[column], errors="coerce")
    assert lot.ExternalLotId.notna().all() and lot.ExternalLotId.eq(251).all()
    assert lot.LotId_staged.eq(lot.ExternalLotId).all()
    lot.to_csv(OUT / "bishoftu_primary_lot_check.csv", index=False)

    fits, metrics, predictions = (
        read("calibration_fits"),
        read("regression_metrics"),
        read("predictions"),
    )
    sensitivity = []
    for _, fit in fits.iterrows():
        p = predictions.loc[predictions.fit_id.eq(fit.fit_id)]
        group = fit.selection_group
        if group in ETHIOPIA_SEASONS:
            p = p.loc[p.season.eq(group)]
        elif group != "All Addis":
            p = p.loc[p.pmf_source.eq(group)]
        for lam in [1.0, 2.96, summary["deming_lambda"], 10.0]:
            slope, intercept = deming(p.hips_equivalent_ugm3, p.prediction_ugm3, lam)
            sensitivity.append(
                {
                    "fit_id": fit.fit_id,
                    "evaluation_group": group,
                    "lambda": lam,
                    "slope": slope,
                    "intercept": intercept,
                }
            )
    pd.DataFrame(sensitivity).to_csv(OUT / "lambda_sensitivity.csv", index=False)
    provenance = {
        "figure_labels_source": display_path(
            ROOT / "deliverables/spectral_comparison_2026-09-01/figures/viz_nearest_analogs.png"
        ),
        "figure_label_method": "manually transcribed site/date labels; required a unique library match",
        "lot_source": display_path(raw_path),
        "lot_source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "shipped_lot_source": display_path(PATHS.spartan_hips_primary),
        "raw_lot_confirmed_n": int(lot.ExternalLotId.eq(251).sum()),
        "shipped_lot_confirmed_n": int(lot.LotId_shipped.eq(251).sum()),
    }
    (OUT / "supplementary_provenance.json").write_text(json.dumps(provenance, indent=2))

    own = metrics.loc[metrics.selection_group.eq(metrics.evaluation_group)]
    seasonal = own.loc[own["mask"].eq("no_co2") & own.selection_group.isin(ETHIOPIA_SEASONS)]
    pmf = own.loc[
        own["mask"].eq("no_co2") & ~own.selection_group.isin(["All Addis", *ETHIOPIA_SEASONS])
    ]
    changes = read("mask_membership_changes")
    changes = changes.loc[changes.group.eq("All Addis")]
    overlaps = read("seasonal_overlap")
    overlaps = overlaps.loc[overlaps["mask"].eq("no_co2")]
    pooled = own.loc[own.selection_group.eq("All Addis")]
    split = read("split_counts")
    # Keep all precision in the CSV; only round values for the report.
    pieces = [
        "# Addis FTIR weekly follow-up — 10 September 2026",
        "The requested reruns and audits are complete. Spectral exclusions change the analog sets, "
        "and Dry selects almost entirely different filters from the rainy seasons. However, the three "
        "season-specific calibrations have site-held-out TOR R² of 0.428–0.700, below the prior 0.85 "
        "screening threshold. Their intercepts therefore should not be interpreted as evidence that "
        "the Addis offset is resolved. Keep the existing OC/EC calibration as the reference for now.",
        "The proposed independent Addis split is supplied as an **unscored retrospective design**. "
        "All performance results below use already explored Addis data and are exploratory.",
        "## Completion against Ann’s task list",
        table(
            ["Task", "Result"],
            [
                [
                    "Exclude CO₂ and test upper end",
                    "Seven selection masks; four main masks recalibrated",
                ],
                [
                    "Keep baseline-corrected matching",
                    "AIRSpec-corrected spectra throughout; raw selection not reopened",
                ],
                [
                    "Audit original matched filters and CV",
                    "All 10 meeting-figure analogs outside historical 440; site-grouped CV confirmed",
                ],
                [
                    "Separate seasons and compare overlap",
                    "Three season-specific cohorts and fits; full overlap tables",
                ],
                [
                    "Extend to PMF factors",
                    "Five normalized dominant-source cohorts and fits; 102 matched target filters",
                ],
                [
                    "Report slope and R² with intercept",
                    "Seasonal and PMF crossplots and complete regression table",
                ],
                [
                    "Plot complete calibration versus season",
                    "All actual training traces versus all seasonal Addis traces; appendix summaries",
                ],
                [
                    "Confirm Deming uncertainty",
                    "Original routine returned estimates only; added tested bootstrap SEs and 95% CIs",
                ],
                [
                    "Scope held-out Addis design",
                    "166 selection / 73 validation filters; whole-month split; not scored",
                ],
                [
                    "Confirm Bishoftu lot",
                    "All 40 matched to raw HIPS metadata: lot 251; 26 also confirmed in shipped HIPS",
                ],
            ],
        ),
        "## Data and fixed choices",
        f"The user-supplied {link(Path.home() / 'Downloads/results_tor.csv')} supplies TOR values. "
        "The corrected cache contains 13,634 IMPROVE scans representing 13,632 physical filters; "
        "13,010 physical filters have finite spectra and positive TOR EC loading and can enter calibration. "
        "Availability and calibration eligibility are retained as flags. The 239 shipped Addis filters "
        "were checked against the previous evaluation IDs and Fabs values. Replicate Addis scans are averaged "
        "by MediaId, and the canonical exclusion registry is applied.",
        "The primary analysis correlates each IMPROVE spectrum with the **median spectrum of the target group**, "
        "preserving the earlier seasonal analysis definition. Correlation is centred Pearson r and is ranked "
        "with its sign so an inverted spectrum cannot become a good match. The 500 highest-ranked unique "
        "TOR-eligible physical filters form each calibration cohort. Ties are resolved deterministically by AnalysisId. "
        "A separately retained mean-across-target-correlations sensitivity tests aggregation dependence.",
        "**Masks affect analog selection only.** All PLS fits use the identical complete 2,002-channel "
        "AIRSpec grid, approximately 1426–3998 cm⁻¹. The primary mask removes 1800–2500 cm⁻¹ inclusive. "
        "Two variants also remove >3600 or >3500 cm⁻¹. A full-grid control and three variants starting "
        "at 1850 cm⁻¹ complete the seven matching runs. The 1850 variants were used for membership "
        "sensitivity; they were not separately recalibrated. This task does not establish chemical "
        "attribution of the features removed by the upper cut.",
        "Each new calibration reserves 20% of its unique IMPROVE sites (seed 20260717). PLS components "
        "are chosen by five-fold site-grouped CV on the remaining sites using the existing first-major-minimum "
        "rule, with a maximum of 30. The historical model retains its locked k=8. The final model is fit "
        "only on the training subset. **TOR tests differ across cohorts**, so the numbers below are "
        "screening results rather than paired estimates of model superiority. The current eligible pool "
        "is lots 248/251 and is not a claim about every IMPROVE filter ever collected.",
        "There are 22 primary fits: four masks × four target selections (all Addis plus three seasons), "
        "five PMF fits for the primary mask, and the historical model. The separate mean sensitivity "
        "adds 22 fits. Season labels use **dry_feb**: Dry October–February (105), Belg March–May (61), "
        "Kiremt June–September (73). The same convention is used for all matching, plots and split counts.",
        "## Effect of spectral exclusions",
        table(
            ["Mask", "Changed vs full / 500", "Changed vs primary / 500"],
            [
                [r["mask"], int(r.changed_vs_full), int(r.changed_vs_no_co2)]
                for _, r in changes.iterrows()
            ],
        ),
        "The 1800 versus 1850 cm⁻¹ lower limit changes only five filters for the pooled CO₂-only selection. "
        "This supports keeping the stated 1800 cm⁻¹ primary cut while preserving the exact sensitivity lists.",
        table(
            ["Season pair", "Shared / 500", "Jaccard"],
            [
                [f"{r.group_a} / {r.group_b}", int(r.intersection), f"{r.jaccard:.2%}"]
                for _, r in overlaps.iterrows()
            ],
        ),
        "Jaccard is intersection divided by union; it is not the shared fraction of a 500-filter list. "
        "Only 1–6 Dry analogs overlap each rainy-season cohort, while Belg and Kiremt share 160.",
        "## Calibration and seasonal regression",
        "Addis R² below is squared Pearson correlation between **x = HIPS Fabs / MAC** and **y = predicted FTIR EC**, "
        "both in µg/m³. MAC is 10 m²/g from the current config. TOR R² is squared correlation with measured "
        "TOR loadings on held-out source sites. **TOR Q² = 1 − SSE/SST** measures prediction accuracy and "
        "can be negative even when squared correlation is positive. RMSE in the full fit table is µg/filter.",
        table(
            [
                "Selection",
                "k",
                "n Addis",
                "Addis R²",
                "Deming slope [95% CI]",
                "Intercept [95% CI]",
                "TOR R²",
                "TOR Q²",
            ],
            [
                [
                    r.selection_group,
                    int(r.k),
                    int(r["n"]),
                    f"{r.R2:.3f}",
                    interval(r, "slope"),
                    interval(r, "intercept"),
                    f"{r.TOR_R2:.3f}",
                    f"{r.TOR_Q2:.3f}",
                ]
                for _, r in seasonal.iterrows()
            ],
        ),
        "Dry’s intercept is compatible with zero, but its slope is only 0.305 and its 95% interval "
        "does not include 1. Kiremt’s slope interval includes 1 while its intercept interval stays negative. "
        "Neither result demonstrates simultaneous agreement. All three season-specific fits fail the prior "
        "TOR R² screening threshold, so an attractive Addis intercept cannot justify adopting them.",
        f"![Seasonal regressions]({link_target(PLOTS / 'seasons_crossplots.png')})",
        "For the pooled selection, the CO₂-only mask passes the prior TOR correlation screen, but adding "
        "the upper cuts does not improve that screening result. Addis R² remains lower than the historical "
        "reference in these exploratory readouts.",
        table(
            ["Pooled selection", "k", "Addis R²", "Deming slope", "Intercept", "TOR R²", "TOR Q²"],
            [
                [
                    r["mask"],
                    int(r.k),
                    f"{r.R2:.3f}",
                    f"{r.deming_slope:.3f}",
                    f"{r.deming_intercept:.3f}",
                    f"{r.TOR_R2:.3f}",
                    f"{r.TOR_Q2:.3f}",
                ]
                for _, r in pooled.iterrows()
            ],
        ),
        "The report also retains **every fitted seasonal model evaluated in every season**, plus common "
        "pooled/historical-model readouts by season and PMF group. These distinguish changes in target "
        "composition from changes caused by selecting a separate calibration: "
        + link(OUT / "regression_metrics.csv")
        + ".",
        "## PMF extension",
        "The canonical helper normalizes GF1–GF5 before assigning dominant source, then joins filter IDs "
        "through parsed sampling dates. There are 102 matched filters and 137 without a PMF match; these "
        "are retained and labelled unmatched. A dominant-source label is not a pure-source spectrum.",
        table(
            [
                "Dominant source",
                "n",
                "Addis R²",
                "Deming slope [95% CI]",
                "Intercept [95% CI]",
                "TOR R²",
                "TOR Q²",
            ],
            [
                [
                    r.selection_group,
                    int(r["n"]),
                    f"{r.R2:.3f}",
                    interval(r, "slope"),
                    interval(r, "intercept"),
                    f"{r.TOR_R2:.3f}",
                    f"{r.TOR_Q2:.3f}",
                ]
                for _, r in pmf.iterrows()
            ],
        ),
        "All five PMF calibrations also fail the 0.85 TOR correlation screen. Sea Salt is a useful caution: "
        "its Addis R² is 0.831, but source-held-out TOR Q² is −0.893. Strong correlation on the target "
        "comparison can coexist with poor source prediction. Small PMF groups also limit interval precision.",
        f"![PMF regressions]({link_target(PLOTS / 'pmf_crossplots.png')})",
        "Pairwise analog overlap is recorded in " + link(OUT / "pmf_overlap.csv") + ".",
        "## Full calibration spectra versus all seasonal Addis spectra",
        "The following panels contain every spectrum in the actual fitted calibration subset, compared "
        "with every Addis spectrum in that season, on shared axes. Held-out TOR filters are not counted as "
        "training filters. The shaded CO₂ band is excluded from correlation selection only. The deck’s "
        "editable appendix charts summarize these same complete sets using minima, maxima and medians; "
        "the complete individual-trace panels below remain available for inspection.",
    ]
    for season in ETHIOPIA_SEASONS:
        slug = "".join(c if c.isalnum() else "_" for c in season).strip("_")
        pieces.append(
            f"![{season}: complete calibration and Addis spectra]({link_target(PLOTS / f'full_calibration_vs_addis_{slug}.png')})"
        )
    pieces += [
        "## Historical membership and CV",
        "The historical Addis reference is the lowest OC/EC 440-filter AIRSpec calibration with k=8. "
        "Reconstructing its original cohort order and site split gives **387 fit filters and 53 TOR-test filters**. "
        "It used site-grouped CV, not interleaved CV. Its rerun TOR R² is 0.924 (Q² 0.905); this is on its own "
        "test cohort, not the same test set as the new analog models.",
        "Each of the ten IMPROVE site/date labels in the September 1 Addis/Bishoftu nearest-analog figure "
        "was matched uniquely to the library. **All ten are outside the entire historical 440-filter cohort**:",
        table(
            ["Target", "Site", "Date", "AnalysisId", "FilterId", "Historical role"],
            [
                [r.target, r.Site, r.date, int(r.AnalysisId), int(r.FilterId), r.role]
                for _, r in meeting.iterrows()
            ],
        ),
        "The prior seasonal top-five lists were audited separately: all 15 are also outside the historical "
        "cohort. New site-median top-five matches are saved separately and are not asserted to reproduce "
        "the earlier representative/medoid display. The new 500-filter lists each carry historical cohort "
        "and actual training membership flags.",
        "## Deming uncertainty and assumptions",
        "The existing Deming implementation returned slope and intercept point estimates only. The new "
        "`deming_bootstrap` helper returns bootstrap standard errors, percentile intervals, group counts "
        "and successful-draw counts. Here it uses 2,000 whole-calendar-month bootstrap resamples; this "
        "retains dependence within each month but does not model dependence across months. Resampling "
        "pairs/groups for errors-in-variables regression is consistent with documented bootstrap approaches. "
        "See [mcr bootstrap documentation](https://search.r-project.org/CRAN/refmans/mcr/html/mc.bootstrap.html) "
        "and [deming package documentation](https://stat.ethz.ch/CRAN/web/packages/deming/refman/deming.html).",
        f"The median HIPS uncertainty is read from **HIPS_Uncertainty parameter rows**, not the empty "
        f"uncertainty column on Fabs rows. After MAC conversion, σx = {summary['sigma_x_ugm3']:.6f} µg/m³. "
        f"The historical TOR RMSE of {summary['sigma_y_proxy_ugm3']:.3f} µg/m³ is used as an approximate shared "
        f"σy proxy, giving λ = (σy/σx)² = **{summary['deming_lambda']:.6f}**. The older λ≈2.96 used a "
        "different conversion; it is not silently carried forward. This constant-ratio approximation is "
        "not a per-filter uncertainty model, and it is not recalibrated for each new PLS fit.",
        "The intervals condition on fixed predictions, calibration choice, MAC and λ. They do not include "
        "PLS fitting/selection uncertainty or uncertainty in the HIPS and FTIR error estimates. No multiplicity "
        "adjustment is applied across exploratory groups. Compatibility with slope 1 or intercept 0 is not "
        "an equivalence test or proof of practically acceptable error. Point-estimate sensitivity at λ=1, "
        "2.96, the current estimate, and 10 is supplied in "
        + link(OUT / "lambda_sensitivity.csv")
        + ".",
        "## Proposed Addis selection / validation design",
        table(
            [
                "Season",
                "Selection filters",
                "Selection months",
                "Validation filters",
                "Validation months",
            ],
            [
                [
                    season,
                    int(
                        (part := split.loc[split.season.eq(season)].set_index("role")).loc[
                            "selection", "n"
                        ]
                    ),
                    int(part.loc["selection", "months"]),
                    int(part.loc["validation", "n"]),
                    int(part.loc["validation", "months"]),
                ]
                for season in ETHIOPIA_SEASONS
            ],
        ),
        "The saved split reserves approximately one-third of each season’s observed year-month blocks "
        "with seed 20260910: **166 filters for selection and 73 for validation**. No physical filter or month "
        "appears on both sides. This is a stratified month-block proposal, not a chronological future-period "
        "forecast test. It may leave adjacent months on opposite sides. It preserves representation of each season.",
        "1. Freeze the calendar, masks, aggregation, cohort size and PLS component-selection rule.\n"
        "2. Use only the 166 selection-side Addis spectra to define the analog cohorts.\n"
        "3. Fit and validate the calibration using IMPROVE TOR training/CV/test sites.\n"
        "4. Apply the frozen calibration once to the 73 reserved Addis spectra and compare with their HIPS "
        "values using R², slope/intercept intervals, RMSE and bias.\n"
        "5. Report the complete reserved result; do not retune the analog rule on those outcomes.",
        "The proposed validation rows have already participated in earlier exploratory analyses, including "
        "this full-data follow-up. Calling them pristine independent test data would be misleading. New or "
        "previously unexamined filters offer stronger confirmation. The split is supplied for review and **has "
        "not been used to choose or score a new calibration**. Keeping model selection separate from test "
        "evaluation follows [scikit-learn’s leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) "
        "and [grouped-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).",
        "## Bishoftu filter lot",
        "All **40 Bishoftu filters are lot 251**, independently matched by ExternalFilterId to the raw HIPS "
        "`ExternalLotId`. The shipped HIPS table independently confirms the same lot for its 26 released "
        "filters. The remaining 14 have provisional reconstructed Fabs, so lot confirmation does not make "
        "those optical values an official release. Per-filter evidence is in "
        + link(OUT / "bishoftu_primary_lot_check.csv")
        + ".",
        "## Sensitivity and limits",
        "The alternative mean-across-target-correlations run gives seasonal TOR R² of 0.446, 0.569 and "
        "0.781 (Dry, Belg, Kiremt). Each still fails the prior 0.85 screening threshold. That supports "
        "the screening conclusion, while the changed memberships and predictions show that aggregation "
        "must be fixed before future validation. The mean run is retained in "
        + link(OUT.parent / "ann_weekly_20260910_mean_sensitivity", "mean sensitivity tables")
        + ".",
        "Cohort size is held at 500; this follow-up does not establish that 500 is optimal. The 0.85 threshold "
        "is a prior screening convention, not a universal acceptance criterion. HIPS Fabs/MAC is a comparison "
        "proxy rather than direct chemical EC truth. The mask test diagnoses matching sensitivity, not the "
        "physical cause of the Addis offset. The source and target libraries were already used in prior "
        "exploration, so the historical TOR tests are not newly untouched confirmatory data either.",
        "## Reproduction and review files",
        "Run from the repository root with the configured `uv` environment:",
        "```sh\nuv run aeth doctor\n"
        "AETHMODULAR_WEEKLY_TOR_CSV=~/Downloads/results_tor.csv uv run python research/ftir_hips_chem/workflows/run_ann_weekly_20260910.py\n"
        "AETHMODULAR_WEEKLY_AGGREGATION=mean AETHMODULAR_WEEKLY_TOR_CSV=~/Downloads/results_tor.csv uv run python research/ftir_hips_chem/workflows/run_ann_weekly_20260910.py\n"
        "uv run python research/ftir_hips_chem/workflows/summarize_ann_weekly_20260910.py\n"
        "uv run python research/ftir_hips_chem/workflows/prepare_ann_weekly_slides_20260910.py\n"
        "uv run pytest -q tests/test_ann_weekly_analogs.py tests/test_overlays_crossplot.py tests/test_pls_transfer.py\n```",
        "The source arrays and TOR CSV are SHA-256 fingerprinted in "
        + link(OUT / "summary.json")
        + "; "
        "the supplementary source audit is in " + link(OUT / "supplementary_provenance.json") + ". "
        "Each fit has an exact cohort/role table, CV curve and prediction table. The checks cover mask "
        "boundaries, signed ranking, duplicate physical-filter handling, split isolation, Deming identities "
        "and bootstrap reproducibility, alongside existing regression/transfer checks.",
        "The presentation contains nine main slides and three spectral appendix slides, with speaker notes. "
        "Its native editable charts retain every channel; chart-workbook numbers are rounded to ten "
        "significant digits for Excel compatibility, while the analysis arrays retain full precision. "
        "No new calibration has been promoted into production and no email has been sent.",
        "Suggested opening: “I applied the spectral cuts and reran the seasonal and PMF selections. "
        "The seasons choose different analogs, but their TOR validation is weak. The next decision is "
        "how to reserve Addis data before we judge a new calibration.”",
    ]
    (DELIVERABLE / "analysis_report.md").write_text("\n\n".join(pieces) + "\n")
    print(f"Wrote report; confirmed raw lot 251 for {len(lot)} Bishoftu filters")


if __name__ == "__main__":
    main()
