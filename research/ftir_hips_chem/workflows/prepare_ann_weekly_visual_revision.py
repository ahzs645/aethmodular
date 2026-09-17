"""Create graph-focused slide content and spoken notes from completed results."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from config import ETHIOPIA_SEASONS

OUT = ROOT / "research/ftir_hips_chem/output/tables/ann_weekly_20260910"
BUILD = ROOT / "deliverables/ann_weekly_2026-09-10/.build/visual_revision"
BUILD.mkdir(parents=True, exist_ok=True)
S = json.loads((OUT / "summary.json").read_text())
SEASONS = list(ETHIOPIA_SEASONS)
COLORS = [ETHIOPIA_SEASONS[s]["color"] for s in SEASONS]
BLUE, GREY, INK = "#246A9B", "#AAB4BC", "#17384A"
met = pd.read_csv(OUT / "regression_metrics.csv")
pred = pd.read_csv(OUT / "predictions.csv")
fits = pd.read_csv(OUT / "calibration_fits.csv")
changes = pd.read_csv(OUT / "mask_membership_changes.csv")
overlap = pd.read_csv(OUT / "seasonal_overlap.csv")
split = pd.read_csv(OUT / "split_counts.csv")
anchors = [
    "sl/y90nupkv",
    "sl/hwbqtkby",
    "sl/ofy9wn61",
    "sl/jyx0ra1s",
    "sl/i107q5of",
    "sl/x8f69ofe",
    "sl/gnmp4jqx",
    "sl/fu1gfa1s",
    "sl/udsvah03",
    "sl/m90b6t0r",
    "sl/jetc3ut0",
    "sl/8f2psfyx",
]


def stats(group, mask="no_co2"):
    return met.loc[
        met["mask"].eq(mask) & met.selection_group.eq(group) & met.evaluation_group.eq(group)
    ].iloc[0]


def notes(talk, detail, *sources):
    return (
        "Talk track\n"
        + talk
        + "\n\nIf asked\n"
        + detail
        + "\n\nSources\n"
        + "\n".join(
            str(OUT / ("summary.json" if name == "summary" else f"{name}.csv")) for name in sources
        )
    )


def txt(text, x=70, y=180, w=1120, h=40, size=26, color=INK, bold=False):
    return dict(kind="text", text=text, x=x, y=y, w=w, h=h, size=size, color=color, bold=bold)


def axis(title=None, low=None, high=None, step=None, size=22, grid=True, **kwargs):
    a = {
        "textStyle": {"fontSize": size, "fill": INK},
        "majorGridlines": {"fill": "#E3E9EE", "width": 1} if grid else None,
        **kwargs,
    }
    for k, v in [("title", title), ("min", low), ("max", high), ("majorUnit", step)]:
        if v is not None:
            a[k] = v
    return a


def series(name, values, color=BLUE, x=None, width=2.5, marker="none", **kwargs):
    r = {
        "name": name,
        "values": list(values),
        "fill": color,
        "line": {"fill": color if width else "none", "width": width},
        "marker": {"symbol": marker, "size": 7},
        **kwargs,
    }
    if x is not None:
        r["xValues"] = list(x)
    return r


def scatter(series_list, x=72, y=220, w=1100, h=370, **kwargs):
    return dict(
        kind="chart",
        type="scatter",
        x=x,
        y=y,
        w=w,
        h=h,
        series=series_list,
        scatterStyle="lineWithMarkers",
        **kwargs,
    )


def bar(categories, series_list, x=72, y=210, w=1100, h=390, **kwargs):
    for item in series_list:
        item["valuesFormatCode"] = (
            "0" if all(float(v).is_integer() for v in item["values"]) else "0.000"
        )
    return dict(
        kind="chart",
        type="bar",
        categories=categories,
        series=series_list,
        x=x,
        y=y,
        w=w,
        h=h,
        **kwargs,
    )


slides = []
slides.append(
    dict(
        preserve="all",
        notes=notes(
            "I followed up on the spectral exclusions and reran the seasonal and PMF analog selections. "
            "The main result is that the seasons choose different IMPROVE filters, but the new seasonal "
            "calibrations still have weak TOR validation. I’ll show the spectral changes, what happens to "
            "the regressions, and the proposed Addis split for the next evaluation.",
            "All new Addis readouts use already explored target data. They are exploratory. The three "
            "spectral comparisons at the end are backup slides. The analysis uses the September 3 task list.",
            "calibration_fits",
        ),
    )
)

# Actual all-Addis median, with spectral ranges colored as matching masks.
e = pd.read_csv(OUT / "addis_evaluation.csv")
z = np.load(
    ROOT / "research/ftir_ec_phase3/output/corrected/etad_corrected_df6.npz", allow_pickle=True
)
wn = z["wn"].astype(float)
X = np.vstack([z["corrected"][z["media_id"].astype(int) == m].mean(axis=0) for m in e.MediaId])
median = np.median(X, axis=0)
co2, upper = (wn >= 1800) & (wn <= 2500), wn > 3500
slides.append(
    dict(
        title="Spectral regions for analog selection",
        subtitle="Median of all 239 Addis filters",
        blocks=[
            scatter(
                [
                    series("Full spectrum", median, BLUE, wn),
                    series("CO₂ cut: 1800–2500", median[co2], "#C64F44", wn[co2], width=5),
                    series(
                        "Upper cut: >3500 or >3600", median[upper], "#BD8A29", wn[upper], width=4
                    ),
                ],
                y=220,
                h=390,
                legend=True,
                xAxis=axis("Wavenumber (cm⁻¹)", 1400, 4000, 500, grid=False, numberFormatCode="0"),
                yAxis=axis("Corrected absorbance", numberFormatCode="0.000"),
            )
        ],
        footnote="Masks change analog selection. PLS uses the full spectrum.",
        notes=notes(
            "This is the median baseline-corrected spectrum for all 239 Addis filters. The red section "
            "marks the CO₂ region removed from matching, from 1800 to 2500 inverse centimetres. The gold "
            "section shows the upper end where I tested cutoffs at 3500 and 3600. I applied these masks "
            "when selecting analogs. Every PLS calibration still uses the same full spectral grid, so "
            "we can isolate the effect of changing the selected filters.",
            "Primary selection uses signed Pearson correlation to the target group’s median corrected "
            "spectrum, retaining 500 unique TOR-eligible IMPROVE filters. The displayed gold trace starts "
            "at 3500 and encompasses both upper-cut tests. Seven masks include an 1850 lower-bound "
            "sensitivity. The full available grid has 2002 channels, about 1426–3998 cm⁻¹. Raw-spectrum "
            "selection remains closed. Mean-across-target correlation is a separate sensitivity.",
            "mask_channels",
            "library_eligibility",
            "addis_evaluation",
        ),
    )
)

pooled_changes = changes.loc[changes.group.eq("All Addis")].set_index("mask")
shared = overlap.loc[overlap["mask"].eq("no_co2")]
slides.append(
    dict(
        title="The masks and seasons select different filters",
        blocks=[
            txt("Replaced after the spectral cuts", x=76, y=160, w=545, size=26, bold=True),
            txt("Shared between seasons", x=695, y=160, w=510, size=26, bold=True),
            bar(
                ["CO₂ cut", "+ >3600", "+ >3500"],
                [
                    series(
                        "Replaced",
                        [
                            int(pooled_changes.loc[m, "changed_vs_full"])
                            for m in ["no_co2", "no_co2_max3600", "no_co2_max3500"]
                        ],
                    )
                ],
                x=66,
                y=215,
                w=555,
                h=377,
                yAxis=axis("Filters", 0, 100, 20),
            ),
            bar(
                ["Dry / Belg", "Dry / Kiremt", "Belg / Kiremt"],
                [series("Shared", shared.intersection.astype(int))],
                x=675,
                y=215,
                w=540,
                h=377,
                yAxis=axis("Filters", 0, 200, 50),
            ),
        ],
        footnote="500 filters per cohort. Seasonal overlap uses the CO₂ exclusion.",
        notes=notes(
            "On the left, the CO₂ cut replaces 22 of the pooled selection’s 500 filters. Adding the upper "
            "cut replaces 61 or 70 relative to the full-spectrum selection. On the right, the Dry "
            "selection shares only six filters with Belg and one with Kiremt. Belg and Kiremt share 160. "
            "So the seasonal spectral differences translate into quite different calibration cohorts, "
            "especially for Dry. The next question is whether those separate cohorts predict well.",
            "Jaccard overlap is 0.6% for Dry/Belg, 0.1% for Dry/Kiremt and 19.0% for Belg/Kiremt. "
            "Jaccard uses intersection/union, unlike the shared count out of 500 shown here. The 1800 "
            "versus 1850 lower boundary changes five pooled CO₂-only analogs. All counts deduplicate "
            "physical IMPROVE filters. These are membership changes, not a causal chemical attribution.",
            "mask_membership_changes",
            "seasonal_overlap",
        ),
    )
)

blocks = []
season_details = []
for i, (season, col) in enumerate(zip(SEASONS, COLORS)):
    r = stats(season)
    p = pred.loc[pred.fit_id.eq(r.fit_id) & pred.season.eq(season)]
    xx = np.array([0.0, 10.0])
    fit_x = np.array(
        [
            max(0, -r.deming_intercept / r.deming_slope),
            min(10, (10 - r.deming_intercept) / r.deming_slope),
        ]
    )
    left = 58 + i * 405
    blocks += [
        txt(
            f"{season.split(' ')[0]}   n = {len(p)}",
            x=left + 16,
            y=155,
            w=350,
            size=27,
            color=col,
            bold=True,
        ),
        txt(f"R² {r.R2:.3f}   Slope {r.deming_slope:.2f}", x=left + 16, y=193, w=365, size=23),
        scatter(
            [
                series("1:1", xx, GREY, xx, width=1.4),
                series(
                    "Deming fit", r.deming_slope * fit_x + r.deming_intercept, INK, fit_x, width=2
                ),
                series(
                    "Addis filters",
                    p.prediction_ugm3,
                    col,
                    p.hips_equivalent_ugm3,
                    width=0,
                    marker="circle",
                ),
            ],
            x=left,
            y=241,
            w=382,
            h=361,
            xAxis=axis("HIPS / MAC (µg/m³)", 0, 10, 2, size=19, grid=False),
            yAxis=axis("FTIR EC (µg/m³)" if i == 0 else None, 0, 10, 2, size=19),
        ),
    ]
    season_details.append(
        f"{season}: R²={r.R2:.3f}; Deming slope {r.deming_slope:.3f} "
        f"[{r.slope_ci_low:.3f}, {r.slope_ci_high:.3f}]; intercept {r.deming_intercept:.3f} "
        f"[{r.intercept_ci_low:.3f}, {r.intercept_ci_high:.3f}] µg/m³; k={int(r.k)}."
    )
slides.append(
    dict(
        title="Dry’s near-zero intercept hides a shallow slope",
        blocks=blocks,
        footnote="Dark line: Deming fit. Grey line: 1:1. All Addis comparisons are exploratory.",
        notes=notes(
            "Each panel uses a calibration selected for that season and shows all of its Addis filters. "
            "The grey diagonal is one-to-one agreement and the dark line is the Deming fit. Dry has an "
            "intercept near zero, but the slope is only about 0.31. Its predictions change far less than "
            "the HIPS comparison. Belg is also shallow. Kiremt gets closer to a slope of one, but its "
            "intercept remains negative. Looking at the intercept alone would miss these differences.",
            "\n".join(season_details) + "\nAxes share the 0–10 scale. MAC=10 m²/g. The target-side "
            "comparison is HIPS Fabs/MAC versus predicted FTIR EC, in µg/m³. February belongs to Dry. "
            "The same Addis spectra defined the analog selection, so these panels are exploratory. "
            "Intervals are 95% month-block bootstrap intervals at fixed λ=3.33538.",
            "predictions",
            "regression_metrics",
        ),
    )
)

pmf_names = sorted(set(S["pmf_counts"]) - {"unmatched"})
pmf_rows = [stats(name) for name in pmf_names]
pmf_labels = ["Charcoal", "Fossil fuel", "Polluted\nmarine", "Sea salt\nmixed", "Wood"]
slides.append(
    dict(
        title="PMF correlation can hide weak TOR prediction",
        subtitle="102 Addis filters with a PMF match",
        blocks=[
            bar(
                pmf_labels,
                [
                    series("Addis R²", [r.R2 for r in pmf_rows], BLUE),
                    series("TOR R²", [r.TOR_R2 for r in pmf_rows], "#7B959F"),
                ],
                y=209,
                h=405,
                legend=True,
                yAxis=axis("R²", 0, 1, 0.2, numberFormatCode="0.0"),
                dataLabels={"showValue": True, "position": "outEnd", "textStyle": {"fontSize": 22}},
            )
        ],
        footnote="All five TOR R² values remain below the prior 0.85 screen.",
        notes=notes(
            "Here I split Addis by its dominant PMF source and selected a calibration for each group. "
            "Blue shows correlation on the Addis comparison, and grey shows correlation with TOR on "
            "held-out IMPROVE sites. Sea Salt is the clearest warning: the Addis R² is 0.831 while TOR "
            "R² is only 0.285, and its TOR prediction Q² is negative. All five source calibrations fall "
            "below the previous TOR screen. These results don’t support adopting a PMF-specific calibration yet.",
            "GF fractions are normalized by the canonical ETAD helper before dominant-source assignment. "
            "There are 137 unmatched Addis filters. Dominant source does not mean pure-source aerosol. "
            "TOR cohorts differ. R² is squared correlation; predictive Q² is distinct. Sea Salt TOR Q²=-0.893.\n"
            + "\n".join(
                f"{name}: n={int(r['n'])}; Deming slope={r.deming_slope:.3f}; "
                f"intercept={r.deming_intercept:.3f} µg/m³; Addis R²={r.R2:.3f}; TOR R²={r.TOR_R2:.3f}."
                for name, r in zip(pmf_names, pmf_rows)
            ),
            "regression_metrics",
            "pmf_overlap",
        ),
    )
)

history = pd.read_csv(OUT / "historical_addis_winner_membership.csv")
lots = pd.read_csv(OUT / "bishoftu_primary_lot_check.csv")
meeting = pd.read_csv(OUT / "meeting_figure_top5_membership.csv")
assert len(meeting) == 10 and meeting.role.eq("outside_cohort").all()
assert len(lots) == 40 and lots.ExternalLotId.eq(251).all()
slides.append(
    dict(
        title="Training membership and Bishoftu lot",
        blocks=[
            txt("Historical 440-filter cohort", x=92, y=157, w=535, bold=True),
            txt("Bishoftu: lot 251 for all 40 filters", x=666, y=157, w=548, bold=True),
            bar(
                ["IMPROVE filters"],
                [
                    series("Fit", [int(history.role.eq("train").sum())], BLUE),
                    series("TOR test", [int(history.role.eq("TOR_test").sum())], GREY),
                ],
                x=82,
                y=215,
                w=520,
                h=345,
                legend=True,
                grouping="stacked",
                gapWidth=200,
                yAxis=axis("Filters", 0, 440, 100),
                dataLabels={"showValue": True, "position": "center", "textStyle": {"fontSize": 25}},
            ),
            bar(
                ["HIPS values"],
                [
                    series("Shipped", [int(lots.ReferenceSource.eq("shipped").sum())], BLUE),
                    series(
                        "Reconstructed", [int(lots.ReferenceSource.eq("reconstructed").sum())], GREY
                    ),
                ],
                x=687,
                y=215,
                w=500,
                h=345,
                legend=True,
                grouping="stacked",
                gapWidth=200,
                yAxis=axis("Filters", 0, 40, 10),
                dataLabels={"showValue": True, "position": "center", "textStyle": {"fontSize": 25}},
            ),
        ],
        footnote="All 10 meeting-figure analogs fall outside the historical cohort.",
        notes=notes(
            "I checked the actual historical calibration membership. Of its 440 filters, 387 entered "
            "the fit and 53 formed the TOR site test. All ten nearest analogs shown in the original "
            "Addis and Bishoftu figure were outside that entire cohort. I also checked the Bishoftu "
            "lot directly against raw HIPS metadata. All 40 filters are lot 251. Their HIPS values "
            "include 26 shipped values and 14 provisional reconstructions.",
            "The historical model uses the lowest OC/EC 440-filter cohort, AIRSpec correction and "
            "k=8, with site-grouped CV and the original site split seed 20260717. The 15 earlier "
            "seasonal top-five analogs were audited separately and are also outside the cohort. Raw "
            "ExternalLotId independently confirms all 40 Bishoftu lot IDs. Lot confirmation does "
            "not make the 14 reconstructed Fabs values an official release.",
            "historical_addis_winner_membership",
            "meeting_figure_top5_membership",
            "bishoftu_primary_lot_check",
        ),
    )
)

parts = [split.loc[split.season.eq(s)].set_index("role") for s in SEASONS]
slides.append(
    dict(
        title="Proposed Addis split: 166 select, 73 validate",
        blocks=[
            bar(
                ["Dry", "Belg", "Kiremt"],
                [
                    series("Selection", [int(p.loc["selection", "n"]) for p in parts], BLUE),
                    series("Validation", [int(p.loc["validation", "n"]) for p in parts], GREY),
                ],
                y=195,
                h=420,
                legend=True,
                grouping="stacked",
                yAxis=axis("Addis filters", 0, 120, 30),
                dataLabels={"showValue": True, "position": "center", "textStyle": {"fontSize": 28}},
            )
        ],
        footnote="Whole-month blocks. Retrospective proposal, not yet scored.",
        notes=notes(
            "This is the proposed split for the next Addis evaluation. Blue is the selection portion "
            "and grey is the validation portion. We would use 166 filters to choose the analog cohorts "
            "and reserve 73 for prediction. Whole calendar months stay together within each season. "
            "Before scoring, we need to lock the mask, aggregation rule, cohort size and PLS selection "
            "procedure. Because these samples have already been explored, this is a retrospective "
            "check. New samples would provide stronger confirmation.",
            "Selection/validation counts: Dry 76/29, Belg 40/21, Kiremt 50/23. Validation blocks are "
            "5, 3 and 4 months respectively. Seed 20260910. No month or physical filter crosses the "
            "split. It is stratified by season rather than a future-period forecast. The validation "
            "spectra and HIPS values must not choose the calibration. No split-based model has been "
            "selected or scored. Leakage reference: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
            "proposed_addis_split",
            "split_counts",
        ),
    )
)

interval_blocks = []
for j, term in enumerate(["slope", "intercept"]):
    guide = 1 if term == "slope" else 0
    series_list = []
    for i, (season, col) in enumerate(zip(SEASONS, COLORS)):
        r = stats(season)
        vals = [r[term + "_ci_low"], r[term + "_ci_high"]]
        series_list.append(
            series(
                season.split(" ")[0],
                [3 - i] * 2,
                col,
                vals,
                width=4,
            )
        )
    series_list.append(series(f"Agreement: {guide}", [0.5, 3.5], GREY, [guide, guide], width=1.5))
    interval_blocks += [
        txt(
            "Deming slope" if j == 0 else "Intercept (µg/m³)",
            x=95 + j * 590,
            y=162,
            w=510,
            bold=True,
        ),
        scatter(
            series_list,
            x=76 + j * 590,
            y=230,
            w=540,
            h=367,
            legend=True,
            xAxis=axis(
                None,
                0 if j == 0 else -2.5,
                1.25 if j == 0 else 1,
                0.25 if j == 0 else 0.5,
                numberFormatCode="0.00",
                grid=False,
            ),
            yAxis=axis(
                None,
                0.5,
                3.5,
                visible=False,
                grid=False,
                tickLabelPosition="none",
                numberFormatCode=";;;",
                line={"fill": "none", "width": 0},
            ),
        ),
    ]
slides.append(
    dict(
        title="Uncertainty makes the seasonal differences clear",
        blocks=interval_blocks,
        footnote="95% bootstrap intervals. Fixed calibration, MAC and λ = 3.34.",
        notes=notes(
            "These horizontal intervals show the uncertainty around the seasonal Deming estimates. "
            "The left panel compares slopes with the agreement value of one. The right compares "
            "intercepts with zero. Dry’s intercept can include zero while its slope is clearly below "
            "one. Kiremt shows the opposite pattern: its slope interval includes one, but its intercept "
            "stays negative. Covering an agreement value means compatibility with that value. It does "
            "not establish practical equivalence.",
            "The original Deming routine returned point estimates only. The added helper returns "
            "standard errors and percentile intervals from 2000 whole-month bootstrap samples. "
            "Median HIPS_Uncertainty/MAC gives sigma_x=0.290751 µg/m³. A historical FTIR held-out "
            "RMSE proxy of 0.531 gives lambda=3.335383. These conditional intervals omit uncertainty "
            "in lambda, MAC and calibration selection/fitting. No multiplicity adjustment or "
            "equivalence margin is defined. Bootstrap documentation: "
            "https://search.r-project.org/CRAN/refmans/mcr/html/mc.bootstrap.html\n"
            + "\n".join(season_details),
            "regression_metrics",
            "lambda_sensitivity",
        ),
    )
)

benchmark = [stats(s).TOR_R2 for s in SEASONS] + [
    stats("All Addis").TOR_R2,
    stats("All Addis", "historical_ocec440").TOR_R2,
]
slides.append(
    dict(
        title="The reference calibration remains the benchmark",
        blocks=[
            bar(
                ["Dry", "Belg", "Kiremt", "All Addis", "Historical\nreference"],
                [
                    series(
                        "TOR R²",
                        benchmark,
                        BLUE,
                        points=[
                            {"idx": i, "fill": c} for i, c in enumerate(COLORS + ["#7B959F", BLUE])
                        ],
                    )
                ],
                y=195,
                h=400,
                yAxis=axis("TOR R²", 0, 1, 0.2, numberFormatCode="0.0"),
                dataLabels={"showValue": True, "position": "outEnd", "textStyle": {"fontSize": 26}},
            )
        ],
        footnote="Prior TOR R² screen: 0.85. Each cohort uses different test filters.",
        notes=notes(
            "This brings the calibration results together. The three seasonal TOR R² values range "
            "from about 0.43 to 0.70, below the prior 0.85 screen. The pooled CO₂-only selection reaches "
            "about 0.86. The historical OC/EC reference remains the benchmark at 0.924 on its own TOR "
            "test. These test populations differ, so this is a screening comparison rather than a "
            "paired claim of superiority. My proposed next step is to agree on the Addis split and "
            "freeze the selection rule before evaluating the reserved predictions.",
            "The bars use the primary median-spectrum, CO₂-only selection. All new fits reserve "
            "source sites and choose components by five-fold site-grouped training CV. The historical "
            "fit retains k=8. No new model is promoted. The separate mean-correlation sensitivity "
            "also leaves all three seasonal fits below 0.85, at TOR R² 0.446, 0.569 and 0.781. "
            "R² is squared correlation, with predictive Q² available in the full fit table.",
            "calibration_fits",
            "regression_metrics",
        ),
    )
)

for season in SEASONS:
    env = S["envelopes"][season]
    short = season.split(" ")[0]
    slides.append(
        dict(
            preserve="charts",
            title=f"{short}: complete spectral comparison",
            subtitle=f"{env['train_n']} IMPROVE training filters and all {env['addis_n']} Addis filters",
            footnote="Grey: IMPROVE. Colour: Addis. Thin lines: full range. Thick lines: median.",
            notes=notes(
                f"This backup slide summarizes every spectrum in the {short} comparison. Grey represents "
                f"the {env['train_n']} IMPROVE filters that actually entered the fit, and the seasonal "
                f"colour represents all {env['addis_n']} Addis filters. Thin lines show the minimum and "
                "maximum at each wavenumber, and thick lines show the median. This lets us compare "
                "the complete training set with the whole season, rather than selecting five examples.",
                "The full 2002-channel grid appears here. The 500 selected analogs include a separate "
                "TOR-test subset, which does not enter these training envelopes. This is a range "
                "summary, not a confidence interval. The report supplies every individual spectral "
                "trace. The matching mask is 1800–2500 cm⁻¹, while the model uses the full grid. "
                "No extra normalization or smoothing was applied for these comparisons.",
                "calibration_fits",
                "summary",
            ),
        )
    )

assert len(slides) == 12
for anchor, slide in zip(anchors, slides):
    slide["anchor"] = anchor


def portable(value):
    """Keep chart data within Excel's 15-digit numeric precision limit."""
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(f"{value:.10g}")
    if isinstance(value, np.integer):
        return int(value)
    return value


(BUILD / "content.json").write_text(json.dumps(portable({"slides": slides}), indent=2))
print(f"Prepared {len(slides)} slides with graph data and spoken notes")
