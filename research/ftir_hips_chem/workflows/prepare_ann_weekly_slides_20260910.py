"""Build the meeting narrative only from the completed analysis tables."""

import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from config import ETHIOPIA_SEASONS

OUT = ROOT / "research/ftir_hips_chem/output/tables/ann_weekly_20260910"
BUILD = ROOT / "deliverables/ann_weekly_2026-09-10/.build"
BUILD.mkdir(parents=True, exist_ok=True)
S = json.loads((OUT / "summary.json").read_text())


def read(n):
    return pd.read_csv(OUT / f"{n}.csv")


met = read("regression_metrics")
fits = read("calibration_fits")
changes = read("mask_membership_changes")
overlap = read("seasonal_overlap")
select = read("selection_summary")
old = read("old_deck_top5_membership")
split = read("split_counts")
new = read("representative_top5_membership")
meeting = read("meeting_figure_top5_membership")
seasons = list(ETHIOPIA_SEASONS)
short = ["Dry", "Belg", "Kiremt"]
primary = "no_co2"


def fmt(x):
    return f"{x:.2f}"


def ci(r, key):
    return f"{r['deming_' + key]:.2f} [{r[key + '_ci_low']:.2f}, {r[key + '_ci_high']:.2f}]"


def stats(mask, group, eval_group):
    return met.loc[
        (met["mask"] == mask)
        & (met.selection_group == group)
        & (met.evaluation_group == eval_group)
    ].iloc[0]


def tb(values, **kw):
    return {"kind": "table", "values": values, **kw}


def tx(text, **kw):
    return {"kind": "text", "text": text, **kw}


def source(*files):
    return "\n".join(str(OUT / f"{f}.csv") for f in files)


slides = []
slides.append(
    {
        "cover": True,
        "title": "Addis FTIR calibration\nWeekly update",
        "subtitle": "Spectral exclusions, seasonal analogs\nand the next validation design",
        "notes": "Follow-up to the September 3 meeting and Ann’s approved task email. Baseline-corrected spectra throughout. All new Addis readouts in this deck are exploratory.",
    }
)

vals = [
    ["Selection space", "Channels used"],
    ["Full AIRSpec control", "1426–3998 cm⁻¹"],
    ["Primary exclusion", "Remove 1800–2500 cm⁻¹"],
    ["Upper-end tests", "Also remove >3600 or >3500 cm⁻¹"],
]
slides.append(
    {
        "title": "Spectral regions for analog selection",
        "subtitle": "Rank IMPROVE spectra by Pearson correlation to each group’s median Addis spectrum.",
        "blocks": [
            tb(vals, y=230, h=270, widths=[420, 728]),
            tx(
                "500 unique, TOR-eligible filters per selection\nAll PLS fits retain the same full AIRSpec grid.",
                y=535,
                h=80,
                size=27,
            ),
        ],
        "footnote": "1850 cm⁻¹ lower-bound sensitivity also checked. Raw-spectra selection stays closed.",
        "notes": source("mask_channels", "library_eligibility")
        + "\nAvailable grid: 2002 channels. Only correlation selection is masked. Signed r avoids rewarding inverted spectra. The group median preserves the earlier seasonal selection definition. Physical IMPROVE FilterIds are deduplicated after ranking. Mean-across-filter correlation is retained as a separate sensitivity run.",
    }
)

sub = changes.loc[
    (changes.group == "All Addis")
    & changes["mask"].isin(["no_co2", "no_co2_max3600", "no_co2_max3500"])
].set_index("mask")
shared = overlap.loc[overlap["mask"] == primary]
rows = [["Season pair", "Shared filters / 500", "Jaccard overlap"]] + [
    [
        f"{r.group_a.split(' ')[0]} / {r.group_b.split(' ')[0]}",
        str(r.intersection),
        f"{r.jaccard:.1%}",
    ]
    for _, r in shared.iterrows()
]
slides.append(
    {
        "title": "Region exclusions change the analog sets",
        "subtitle": "Number of filters replaced relative to the full-spectrum selection, out of 500.",
        "blocks": [
            {
                "kind": "chart",
                "type": "bar",
                "x": 65,
                "y": 213,
                "w": 575,
                "h": 360,
                "categories": ["CO₂ cut", "+ >3600 cut", "+ >3500 cut"],
                "series": [
                    {
                        "name": "Replaced filters",
                        "values": [
                            int(sub.loc[m, "changed_vs_full"])
                            for m in ["no_co2", "no_co2_max3600", "no_co2_max3500"]
                        ],
                        "fill": "#246A9B",
                    }
                ],
                "yAxis": {
                    "min": 0,
                    "max": 100,
                    "majorUnit": 20,
                    "textStyle": {"fontSize": 22},
                    "majorGridlines": {"fill": "#E3E9EE", "width": 1},
                },
            },
            tb(rows, x=675, y=257, w=535, h=258, widths=[185, 190, 160], fontSize=22),
        ],
        "footnote": "Seasonal overlap uses the primary CO₂ exclusion. Jaccard = shared filters / combined unique filters.",
        "notes": source("mask_membership_changes", "seasonal_overlap"),
    }
)

rows = [["Season", "n", "Addis R²", "Deming slope [95% CI]", "Intercept [95% CI]"]]
for season, label in zip(seasons, short):
    r = stats(primary, season, season)
    rows.append([label, str(int(r["n"])), f"{r.R2:.3f}", ci(r, "slope"), ci(r, "intercept")])
primary_fits = fits.loc[(fits["mask"] == primary) & fits.selection_group.isin(seasons)]
slides.append(
    {
        "title": "A near-zero Dry intercept hides a shallow slope",
        "subtitle": "All three seasonal models fall below the previous TOR R² screening threshold of 0.85.",
        "blocks": [
            tb(rows, y=231, h=272, widths=[160, 75, 95, 380, 438], fontSize=25),
            tx(
                f"Site-held-out TOR R²: {primary_fits.TOR_R2.min():.3f}–{primary_fits.TOR_R2.max():.3f}\nTOR tests differ between cohorts, so these are not paired model comparisons.",
                y=532,
                h=82,
                size=25,
            ),
        ],
        "footnote": "x = HIPS Fabs/MAC, y = FTIR EC, both µg/m³. February belongs to Dry (dry_feb).",
        "notes": source("regression_metrics", "calibration_fits")
        + f"\nMAC={S['mac']}. 95% percentile intervals from 2000 month-block bootstrap draws. PLS k comes from site-grouped CV on training sites only, first-major-minimum rule. Readouts use all seasonal Addis filters that also defined spectral selection. They are exploratory, not an independent Addis holdout. TOR R² is squared Pearson correlation, Q² prediction accuracy is separately saved in calibration_fits.csv. Error ratio lambda={S['deming_lambda']:.4f}.",
    }
)

pmf_names = sorted(set(S["pmf_counts"]) - {"unmatched"})
rows = [["PMF dominant source", "n", "Addis R²", "TOR R²", "Slope", "Intercept"]]
for name in pmf_names:
    r = stats(primary, name, name)
    rows.append(
        [
            name,
            str(int(r["n"])),
            f"{r.R2:.3f}",
            f"{r.TOR_R2:.3f}",
            fmt(r.deming_slope),
            fmt(r.deming_intercept),
        ]
    )
slides.append(
    {
        "title": "PMF models remain exploratory",
        "subtitle": f"{239 - S['pmf_counts'].get('unmatched', 0)} matched Addis filters support five separate source selections.",
        "blocks": [tb(rows, y=220, h=360, widths=[340, 70, 160, 160, 200, 218], fontSize=25)],
        "footnote": f"Slope and intercept use Deming (intercept in µg/m³). {S['pmf_counts'].get('unmatched', 0)} filters have no PMF match.",
        "notes": source("pmf_overlap", "regression_metrics")
        + "\nGF fractions normalized with the canonical ETAD helper before dominant-source assignment. Exact calendar-date join. Dominant source does not mean pure-source aerosol. Each row uses its own selected calibration and only its own PMF target group. All 95% uncertainty intervals, common-model group readouts and TOR tests are in the analysis report.",
    }
)

hist = read("historical_addis_winner_membership")
oldin = int(old.historical_winner_filter_role.eq("train").sum())
newsub = new.loc[new["mask"] == primary]
bi = newsub.loc[newsub.target == "Bishoftu"]
ad = newsub.loc[newsub.target == "Addis"]
slides.append(
    {
        "title": "Training membership and Bishoftu lot",
        "subtitle": "The historical Addis calibration used site-grouped CV and a separate TOR site test.",
        "blocks": [
            tx(
                f"Historical model: lowest OC/EC 440, AIRSpec, k = 8\n{sum(hist.role == 'train')} filters in the fit, {sum(hist.role == 'TOR_test')} in the TOR test.",
                y=220,
                h=100,
                size=29,
            ),
            tx(
                "All 10 IMPROVE filters in the meeting’s Addis / Bishoftu figure\nwere outside the historical 440-filter calibration cohort.",
                y=358,
                h=110,
                size=28,
            ),
            tx(
                "Bishoftu: all 40 filters confirmed as lot 251 in raw metadata.\nHIPS values: 26 shipped + 14 reconstructed.",
                y=514,
                h=85,
                size=28,
            ),
        ],
        "notes": source(
            "historical_addis_winner_membership",
            "meeting_figure_top5_membership",
            "old_deck_top5_membership",
            "representative_top5_membership",
            "bishoftu_lot_audit",
            "bishoftu_primary_lot_check",
        )
        + "\nNew representative top-five lists use each site median, so do not claim they reproduce the old medoid display. Historical training membership is reconstructed from exact original OC/EC ordering and the locked site split seed 20260717. Distinguish membership in the 440-filter cohort from the subset actually used to fit.",
    }
)

rows = [["Season", "Selection filters", "Validation filters", "Validation months"]]
for se, shortn in zip(seasons, short):
    part = split.loc[split.season == se].set_index("role")
    rows.append(
        [
            shortn,
            str(int(part.loc["selection", "n"])),
            str(int(part.loc["validation", "n"])),
            str(int(part.loc["validation", "months"])),
        ]
    )
slides.append(
    {
        "title": "A separate Addis selection and validation set",
        "subtitle": "Proposed design: hold out whole calendar months within each season.",
        "blocks": [
            tb(rows, y=218, h=260, widths=[220, 295, 315, 318], fontSize=26),
            tx(
                "166 filters select the analog cohorts.\n73 filters are reserved for a later prediction check.",
                y=507,
                h=84,
                size=29,
            ),
        ],
        "footnote": "Retrospective proposal, not yet scored. Earlier Addis exploration prevents a pristine holdout claim.",
        "notes": source("proposed_addis_split", "split_counts")
        + "\nSeed 20260910. Approximately one-third of each season’s observed calendar-month blocks are reserved. Lock masks, aggregation, N and k-selection before scoring. Select using only selection-side spectra, then fit using IMPROVE TOR. Neither validation spectra nor HIPS values may choose a calibration. For a stronger confirmatory test collect new samples or use a previously unexamined cohort. https://scikit-learn.org/stable/common_pitfalls.html#data-leakage https://scikit-learn.org/stable/modules/cross_validation.html",
    }
)

slides.append(
    {
        "title": "What the uncertainty intervals establish",
        "subtitle": "The existing Deming functions returned point estimates only. Bootstrap intervals are now available.",
        "blocks": [
            tx(
                "Intervals use whole-month resampling to keep nearby filters together.",
                y=220,
                h=85,
                size=31,
            ),
            tx(
                f"HIPS uncertainty: {S['sigma_x_ugm3']:.3f} µg/m³ after MAC conversion.\nThe FTIR error proxy gives λ = {S['deming_lambda']:.2f}.",
                y=335,
                h=100,
                size=30,
            ),
            tx(
                "An interval covering slope 1 or intercept 0 means compatibility.\nIt does not prove agreement or acceptable practical error.",
                y=470,
                h=100,
                size=29,
            ),
        ],
        "footnote": "Intervals condition on the chosen calibration and fixed MAC and λ. They exclude model-selection uncertainty.",
        "notes": source("regression_metrics")
        + f"\nMedian HIPS_Uncertainty / MAC read from parameter rows. sigma_y={S['sigma_y_proxy_ugm3']} µg/m³ is the historical held-out TOR RMSE proxy, not a measured per-filter uncertainty. It is approximate and shared across models; sensitivity to lambda belongs in interpretation. References: https://search.r-project.org/CRAN/refmans/mcr/html/mc.bootstrap.html and https://stat.ethz.ch/CRAN/web/packages/deming/refman/deming.html. No equivalence test or practical margin has been established.",
    }
)

slides.append(
    {
        "title": "The reference calibration remains the benchmark",
        "subtitle": "Distinct spectral analog sets have not yet produced a stronger seasonal calibration.",
        "blocks": [
            tx(
                "Dry shares only 1–6 filters with the rainy-season analog sets.",
                y=225,
                h=80,
                size=31,
            ),
            tx(
                "Seasonal TOR R² ranges from 0.43 to 0.70.\nThe historical OC/EC reference gives 0.924 on its own TOR test.",
                y=340,
                h=110,
                size=31,
            ),
            tx(
                "Next: agree on the Addis split and freeze the selection rule\nbefore assessing the reserved predictions.",
                y=496,
                h=90,
                size=30,
            ),
        ],
        "footnote": "Cohort-specific TOR tests differ. The mean-correlation sensitivity reaches the same screening conclusion.",
        "notes": source("calibration_fits", "seasonal_overlap")
        + "\nThe primary results preserve the prior median-spectrum selection. The first pass using mean-across-filter correlation is retained under output/tables/ann_weekly_20260910_mean_sensitivity. Both approaches fail the 0.85 screening floor for each of the three season-specific fits. Do not claim statistical superiority from different TOR test populations or adopt a calibration selected on already explored Addis outcomes.",
    }
)

# Three spectrum summaries, preserving the complete channel grid as editable data.
colors = [ETHIOPIA_SEASONS[season]["color"] for season in seasons]
pales = ["#F3D1AD", "#B4DEC6", "#B8D7EC"]
for se, label, col, pale in zip(seasons, short, colors, pales):
    # Excel chart workbooks support 15 significant digits; ten is ample for display.
    # Preserve the unrounded source arrays in summary.json.
    env = {
        k: ([float(f"{v:.10g}") for v in val] if isinstance(val, list) else val)
        for k, val in S["envelopes"][se].items()
    }
    series = []
    for name, color, bold in [("calibration", "#AAB4BC", False), ("addis", pale, False)]:
        for q in [0, 100]:
            series.append(
                {
                    "name": f"{name} range {q}%",
                    "xValues": env["wavenumber"],
                    "values": env[f"{name}_q{q}"],
                    "line": {"fill": color, "width": 1},
                    "marker": {"symbol": "none"},
                }
            )
    series += [
        {
            "name": "Calibration median",
            "xValues": env["wavenumber"],
            "values": env["calibration_q50"],
            "line": {"fill": "#66747F", "width": 2.5},
            "marker": {"symbol": "none"},
        },
        {
            "name": "Addis median",
            "xValues": env["wavenumber"],
            "values": env["addis_q50"],
            "line": {"fill": col, "width": 3},
            "marker": {"symbol": "none"},
        },
    ]
    slides.append(
        {
            "title": f"{label}: full spectral range of both sets",
            "subtitle": f"{env['train_n']} IMPROVE filters in the actual fit and all {env['addis_n']} Addis filters. Grey = IMPROVE. Colour = Addis.",
            "blocks": [
                {
                    "kind": "chart",
                    "type": "scatter",
                    "y": 225,
                    "h": 375,
                    "series": series,
                    "legend": False,
                    "xAxis": {
                        "title": "Wavenumber (cm⁻¹)",
                        "min": 1400,
                        "max": 4000,
                        "majorUnit": 500,
                        "textStyle": {"fontSize": 22},
                        "majorGridlines": None,
                    },
                    "yAxis": {
                        "title": "AIRSpec-corrected absorbance",
                        "numberFormatCode": "0.00",
                        "textStyle": {"fontSize": 22},
                        "majorGridlines": {"fill": "#E3E9EE", "width": 1},
                    },
                }
            ],
            "footnote": "Thin lines show the full minimum–maximum envelope. Thick lines show the median. All individual traces are supplied separately.",
            "notes": str(
                ROOT
                / "research/ftir_hips_chem/output/plots/ann_weekly_20260910"
                / f"full_calibration_vs_addis_{''.join(c if c.isalnum() else '_' for c in se).strip('_')}.png"
            )
            + "\nComplete 2002-channel data in the editable chart. Full-range envelope represents all actual training spectra and every seasonal Addis spectrum. The 500 selected filters include the TOR test subset, which does not enter the fit; the chart uses only actual training filters. This display summarizes the full set rather than choosing five examples. The downloadable PNG shows every individual trace. No extra smoothing or area normalization applied.",
        }
    )

(BUILD / "content.json").write_text(json.dumps({"slides": slides}, indent=2))
print(f"{len(slides)} slides prepared")
