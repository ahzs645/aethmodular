"""Export the 10 Nov 2026 page "Which measurement carries the Addis offset" for the gallery's Meeting tab.

Run from the repo root with the repo environment:

    uv run python gallery/data/export_meeting_20261110.py

A results page, not a recorded meeting. It answers the question the 9 Nov status page left open, with
data already in hand and without the aethalometer (the Addis MA350 is not a reference). Every number
comes from the workflow research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py, which
this runs in memory. Its IMPROVE stage reads the Drive tables of TOR EC and HIPS, or the stratum table
saved in docs/addis-reference-closure-2026-11-10.json where the Drive is absent; everything else is
committed. docs/addis-reference-closure-2026-11-10.md is the written account.

Writes gallery/app/public/data/meetings/closure-20261110.json (see meeting_bundle.py).
The figures are declared in gallery/app/src/meetings/closure-20261110.meeting.ts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research/ftir_hips_chem/workflows"), str(ROOT / "research/ftir_hips_chem/scripts")]

import reference_closure as rc  # noqa: E402
import run_addis_reference_closure_20261110 as closure  # noqa: E402
from meeting_bundle import write_bundle  # noqa: E402

MEETING_ID = "closure-20261110"
SEASON_COLORS = {"Bega": "#E67E22", "Belg": "#27AE60", "Kiremt": "#3498DB"}   # config.ETHIOPIA_SEASONS colours
COLORS = {**SEASON_COLORS, "all sites": "#22252A", "sites with median OC/EC < 4": "#2C6E9E", "six urban IMPROVE sites": "#B23327",
          "PM2.5 mass (gravimetric)": "#27AE60", "Potassium (XRF)": "#2C6E9E", "FTIR EC, SPARTAN published": "#B23327",
          "HIPS Fabs": "#5B5852", "measured (instrument: PM2.5 mass)": "#C49442", "measured (instrument: potassium)": "#2C6E9E",
          "predicted from IMPROVE thermal EC": "#8a8f98", "HIPS as measured": "#B23327", "HIPS linearised": "#2C6E9E",
          "this filter's own TOR (selects on its error)": "#B23327", "the previous sample at the site": "#2C6E9E", "the site": "#22252A",
          "IMPROVE (FTIR EC ≥ 5 µg per filter)": "#8a8f98", "assumed": "#8a8f98", "measured with thermal EC": "#2C6E9E", "needed at Addis": "#B23327",
          "Raw spectra + EPO": "#7A4FA3", "as trained": "#22252A", "recalibrated on other sites": "#2C6E9E",
          "As fitted": "#22252A", "Weighted toward heavy filters": "#B23327", "Blanks added at zero": "#2C6E9E", "Weighted, and blanks at zero": "#7A4FA3", "Adama, Ethiopia (5 filters with TOR)": "#C49442",
          "urban IMPROVE site": "#B23327", "other IMPROVE site": "#8a8f98", "TOR's own OC/EC on this filter": "#B23327",
          "FTIR's own OC/EC on this filter": "#7A4FA3", "TOR OC/EC of the previous sample at the site": "#2C6E9E",
          "By site: the six urban IMPROVE sites": "#22252A", "By the filter's own TOR (heavy, low OC/EC)": "#B23327", "Addis Ababa": "#C49442", "2005–2014": "#8F8C84", "2015–2025": "#7A4FA3", "optical pyrolysis > 0": "#27AE60",
          "Raw spectra": "#22252A", "PTFE-anchored VIBES": "#C49442", "Spline baseline": "#2C6E9E"}


tables, table = closure.run(n_boot=2000)
PUB = closure.LABEL[closure.PUBLISHED]
ceiling = closure.thermal_ceiling(tables)

# ---------------------------------------------------------------- 1. IMPROVE: HIPS against thermal EC
strata = tables["improve_strata"].rename(columns={"loading_ug": "TOR EC (µg per filter)", "tau": "HIPS optical depth",
                                                  "fabs_over_ec": "Fabs ÷ TOR EC (m²/g)"})
from pls_transfer import deming_regression  # noqa: E402

import json  # noqa: E402

closeout = json.loads(closure.CLOSEOUT.read_text())
pilot = pd.DataFrame(closeout["adama"])
adama_points = pd.DataFrame({"population": "Adama, Ethiopia (5 filters with TOR)", "n": 1, "TOR EC (µg per filter)": pilot.tor_ug,
                             "HIPS optical depth": pilot.tau, "Fabs ÷ TOR EC (m²/g)": pilot.fabs_over_tor}).sort_values("HIPS optical depth")
adama_filters = pilot[["date", "flag", "tor_ug", "ocec", "tau", "fabs_over_tor"]].assign(flag=pilot.flag.fillna("none"))
summary = pd.DataFrame(closeout["adama_summary"])
adama_ftir = summary.assign(calibration=summary.model.map({"VIBES k18 (before)": "Production VIBES", "VIBES anchored k18 (after)": "PTFE-anchored VIBES",
                                                         "raw k18": "Raw spectra", "EPO k18": "Anchored + EPO (historical)"}))[
    ["calibration", "median_unflagged", "median_all", "mapped_lo", "mapped_hi"]]
ceilings = tables["improve_ceiling"].copy()
LAMBDA = 2.96   # closeout LAMBDA_ADDIS, as in the workflow
implied = []
for value in ceilings.tau_inf:
    lin = rc.linearise(table.tau, value) * 100.0 * closure.DEPOSIT_AREA_CM2 / table.volume_m3 / closure.MAC_VALUE
    ok = np.isfinite(lin) & table[closure.PUBLISHED].notna()
    d = deming_regression(lin[ok], table.loc[ok, closure.PUBLISHED], LAMBDA)
    implied.append({"addis_predicted_exponent": float(np.mean(np.clip(1.0 - table.tau / value, 0.0, 1.0))),
                    "addis_slope": d["slope"], "addis_intercept": d["intercept"]})
ceilings = pd.concat([ceilings.reset_index(drop=True), pd.DataFrame(implied)], axis=1)
ceilings["kind"] = np.where(ceilings.population.isin(closure.IMPROVE_POPULATIONS), "choice of sites", "robustness, all sites")
era = ceilings.set_index("population")
primary = ceilings.set_index("population").loc[closure.IMPROVE_PRIMARY]
light = strata[strata.population.eq(closure.IMPROVE_PRIMARY)].iloc[0]
heavy = strata[strata.population.eq(closure.IMPROVE_PRIMARY)].sort_values("HIPS optical depth").iloc[-1]
text_improve = f"""**IMPROVE's own thermal EC shows HIPS falling behind as the filter darkens.** {int(primary.samples):,} IMPROVE samples
({int(primary.sites)} sites, {primary.years}) have TOR EC and HIPS on the same day. Grouped by the previous sample's EC loading at the site,
so that neither instrument's error decides the group, HIPS Fabs ÷ TOR EC falls from {light['Fabs ÷ TOR EC (m²/g)']:.1f} m²/g on the
lightest filters to {heavy['Fabs ÷ TOR EC (m²/g)']:.1f} at optical depth {heavy['HIPS optical depth']:.2f}. A saturating curve through the group
medians has a **ceiling of {primary.tau_inf:.2f}** ({primary.tau_inf_lo:.2f}–{primary.tau_inf_hi:.2f}, resampling sites); the low-OC/EC sites and
the six urban IMPROVE sites give the same. Addis optical depth has a median of {table.tau.median():.2f} and reaches {table.tau.max():.2f}, so
Addis sits on the bend. No FTIR and no aethalometer enter this. **The ceiling is not the same in every period:** {era.loc['2005–2014', 'tau_inf']:.2f}
for 2005–2014 and {era.loc['2015–2025', 'tau_inf']:.2f} ({era.loc['2015–2025', 'tau_inf_lo']:.2f}–{era.loc['2015–2025', 'tau_inf_hi']:.2f}) for 2015–2025, the years
that include the Addis record. The later value predicts an Addis exponent of {era.loc['2015–2025', 'addis_predicted_exponent']:.2f}, which is what Addis
shows, and a positive intercept ({era.loc['2015–2025', 'addis_intercept']:+.2f}). So the ceiling is the main uncertainty: it decides whether the
intercept left after linearising is slightly negative, zero or positive, though not whether the −6 goes away."""

# ---------------------------------------------------------------- 2. Addis: not proportional, and by how much
lin = tables["linearity"]
whole = lin[lin.group.eq("all")].copy()
whole["pair"] = whole.y + " against " + whole.x
exponents = whole[["pair", "instrument", "n", "exponent", "lo", "hi"]]
bounds = tables["frisch_bounds"].set_index("x").loc[PUB]
h = whole[(whole.y.eq("HIPS Fabs")) & (whole.x.eq(PUB))].set_index("instrument")
raw, linearised = tables["raw_hips"].set_index("model"), tables["linearised_hips"]
row = {w: linearised[(linearised.which.eq(w)) & (linearised.model.eq(PUB))].iloc[0] for w in ("central", "lo", "hi")}
text_addis = f"""**At Addis, HIPS and FTIR EC are far from proportional, and the IMPROVE curve predicts most of it.** HIPS rises as FTIR EC
to the power {h.loc['PM2.5 mass (gravimetric)', 'exponent']:.2f} (instrument: gravimetric PM2.5) or {h.loc['Potassium (XRF)', 'exponent']:.2f}
(XRF potassium); with no instrument at all the exponent is bounded to {bounds.lower:.2f}–{bounds.upper:.2f}. If FTIR EC were exactly proportional
to EC, the IMPROVE curve predicts {row['central'].predicted_hips_exponent:.2f} ({row['lo'].predicted_hips_exponent:.2f}–{row['hi'].predicted_hips_exponent:.2f})
at Addis optical depths. So HIPS compression accounts for most of the departure from 1. What is left over is small and not resolved:
a ceiling at the low end of its interval, or FTIR EC rising a little faster than EC. {len(table)} Addis filters; {int(table.pm25.notna().sum())} with
PM2.5 mass, {int(table[closure.PUBLISHED].notna().sum())} with a published EC."""

linear_bc = rc.linearise(table.tau, ceiling["central"]) * 100.0 * closure.DEPOSIT_AREA_CM2 / table.volume_m3 / closure.MAC_VALUE
points = pd.DataFrame({"filter": table.index, "season": table.season, "hips_bc": table.hips_bc, "hips_linearised": linear_bc,
                       "ftir_published": table[closure.PUBLISHED]})
against_hips = pd.concat([
    points.assign(comparator="HIPS as measured (Fabs / 10)", x=points.hips_bc),
    points.assign(comparator="HIPS linearised (/ 10), IMPROVE thermal ceiling", x=points.hips_linearised),
])[["filter", "season", "comparator", "x", "ftir_published"]].dropna()
fits = pd.concat([
    tables["raw_hips"].rename(columns={"slope_vs_hips": "slope", "intercept_vs_hips": "intercept"}).assign(comparator="HIPS as measured"),
    linearised.rename(columns={"slope_vs_linearised_hips": "slope", "intercept_vs_linearised_hips": "intercept"}).assign(
        comparator=lambda d: "HIPS linearised, ceiling " + d.tau_inf.map("{:.2f}".format)),
])[["comparator", "model", "n", "slope", "intercept"]]
text_response = f"""**Linearising HIPS with the IMPROVE ceiling takes the intercept to zero.** The published fit moves from
{raw.loc[PUB, 'slope_vs_hips']:.2f}x − {abs(raw.loc[PUB, 'intercept_vs_hips']):.2f} to **{row['central'].slope_vs_linearised_hips:.2f}x {row['central'].intercept_vs_linearised_hips:+.2f}**
(intercept {row['lo'].intercept_vs_linearised_hips:+.2f} to {row['hi'].intercept_vs_linearised_hips:+.2f} across the ceiling's interval). The correction is large
(median × {row['central'].median_correction_factor:.2f}) and multiplies noise on the heaviest filters. Against linearised HIPS, FTIR EC is close to
proportional but not exactly: linearised HIPS rises as FTIR EC to the power {row['central'].linearised_exponent_vs_ftir:.2f}
({row['central'].linearised_exponent_lo:.2f}–{row['central'].linearised_exponent_hi:.2f}). The slope that remains is the level question: FTIR EC would
need a MAC near {row['central'].mac_for_agreement:.0f} m²/g to agree."""

# ---------------------------------------------------------------- 3. seasons and lots
season = tables["season_lot"]
season_rows = season[~season.lot.eq("all")].copy()
season_rows["group"] = season_rows.season + ", lot " + season_rows.lot.astype(str)
s251 = season_rows[season_rows.lot.astype(str).eq("251")].set_index("season")
s248 = season_rows[season_rows.lot.astype(str).eq("248")].set_index("season")
text_season = f"""**Most of the seasonal swing was HIPS; a smaller one remains, and lot 248 reads low.** Against HIPS as measured, FTIR EC ÷ HIPS
goes from {s251.loc['Bega', 'published_over_hips']:.2f} in Bega to {s251.loc['Kiremt', 'published_over_hips']:.2f} in Kiremt within lot 251. Against
linearised HIPS it goes from {s251.loc['Bega', 'published_over_linearised_hips']:.2f} to {s251.loc['Kiremt', 'published_over_linearised_hips']:.2f}. Kiremt
also has the heaviest filters, so season and loading cannot be told apart here. Lot 248 reads lower than lot 251 in the same season
against linearised HIPS ({s248.loc['Bega', 'published_over_linearised_hips']:.2f} vs {s251.loc['Bega', 'published_over_linearised_hips']:.2f} in Bega) and
against PM2.5 mass ({s248.loc['Bega', 'published_over_pm25']:.2f} vs {s251.loc['Bega', 'published_over_pm25']:.2f}), so that offset is not a HIPS effect."""

# ---------------------------------------------------------------- 4. the level
anchors = tables["level_anchors"].rename(columns={"published_over_true": "published ÷ true"})
mass = tables["mass_closure"].rename(columns={"ec_multiplier": "EC multiplier", "reconstructed_over_gravimetric": "reconstructed ÷ gravimetric PM2.5",
                                              "ec_share_of_pm25": "EC share of PM2.5"})
ratios = anchors["published ÷ true"].dropna()
flevel = tables["ftir_level"].set_index("filters")
comp = tables["ftir_composition"].iloc[0]
urban_level = flevel.loc["six urban IMPROVE sites"]
urban_mac = float(ceilings.set_index("population").loc["six urban IMPROVE sites", "unsaturated_fabs_over_ec"])
implied_mac = closure.MAC_VALUE * urban_level.ftir_over_tor * comp.published_over_refit / row["central"].slope_vs_linearised_hips
text_level = f"""**The slope that is left has two parts, and they fit together.** After linearising, FTIR EC is {row['central'].slope_vs_linearised_hips:.2f} of
HIPS BC at MAC 10. That is the FTIR level times 10 ÷ the real MAC. On the urban IMPROVE sites the raw-spectra calibration reads
{urban_level.ftir_over_tor:.2f} of thermal EC, and the published product reads {comp.published_over_refit:.2f}× that calibration at Addis. If the same bias
holds at Addis, the slope implies a MAC of about {implied_mac:.0f} m²/g; IMPROVE's urban sites give {urban_mac:.1f} on light filters and the five
Adama filters about 19. **Working range: published EC × 1.0 to 1.4, most likely about × 1.1.** The anchors put published ÷ true between
{ratios.min():.2f} and {ratios.max():.2f}. Mass closure caps it: published EC reconstructs {mass.iloc[0, 2] * 100:.0f} % of gravimetric PM2.5, × 1.5 reaches
{mass.iloc[3, 2] * 100:.0f} % and × 1.9 gives {mass.iloc[5, 2] * 100:.0f} %. None of this is thermal EC measured at Addis, and section 8 shows why IMPROVE
cannot settle it."""

# the MAC argument, drawn: which MAC you believe decides the FTIR level
cset = ceilings.set_index("population")
adama_linearised = float(np.median(pilot.fabs_over_tor * rc.linearise(pilot.tau, ceiling["central"]) / pilot.tau))
needed = {which: float(row[which].mac_for_agreement) for which in ("central", "lo", "hi")}
mac_bars = pd.DataFrame([
    ("MAC 10 (the convention behind HIPS BC)", closure.MAC_VALUE, "assumed"),
    ("IMPROVE, all sites, light filters", float(cset.loc["all sites", "unsaturated_fabs_over_ec"]), "measured with thermal EC"),
    ("IMPROVE, sites with median OC/EC < 4, light filters", float(cset.loc["sites with median OC/EC < 4", "unsaturated_fabs_over_ec"]), "measured with thermal EC"),
    ("IMPROVE's six urban sites, light filters", float(cset.loc["six urban IMPROVE sites", "unsaturated_fabs_over_ec"]), "measured with thermal EC"),
    ("Adama, five filters, as measured", float(pilot.fabs_over_tor.median()), "measured with thermal EC"),
    ("Adama, five filters, HIPS linearised", adama_linearised, "measured with thermal EC"),
    ("What Addis needs for the published FTIR EC to be right", needed["central"], "needed at Addis"),
], columns=["source", "mac", "kind"])
grid = np.arange(8.0, 26.5, 0.5)
mac_levels = pd.concat([pd.DataFrame({"ceiling": f"HIPS ceiling {ceiling[which]:.2f}", "mac": grid, "published_over_true": grid / needed[which]})
                        for which in ("lo", "central", "hi")], ignore_index=True)
text_mac = f"""**The MAC you believe decides the FTIR level.** Once HIPS is linearised, the published FTIR EC agrees with HIPS only if the MAC is
{needed['central']:.1f} m²/g (between {needed['hi']:.1f} and {needed['lo']:.1f} across the ceiling's interval). At the conventional MAC of 10, FTIR would be
reading {closure.MAC_VALUE / needed['central']:.2f} of true EC. But every MAC measured against thermal EC is above 10: {mac_bars.mac[1]:.1f} m²/g on light IMPROVE
filters, {mac_bars.mac[3]:.1f} at IMPROVE's six urban sites, and {mac_bars.mac[4]:.1f} on the five Adama filters ({adama_linearised:.1f} after the same
linearisation). At the urban IMPROVE value FTIR reads {mac_bars.mac[3] / needed['central']:.2f} of true EC; at the Adama value, {adama_linearised / needed['central']:.2f}.
Adama is the only thermal EC in Ethiopia, so it is the most relevant MAC and the least certain: five filters, another site and sampler."""

# ---------------------------------------------------------------- 5. the network, site by site
net = tables["network_exponents"].copy()
net["readable"] = (net.hi - net.lo) < 0.6
readable = net[net.readable].sort_values("median_optical_depth")
network = pd.concat([
    readable.assign(what="measured (instrument: PM2.5 mass)", exponent=readable.hips_exponent_vs_ftir_ec),
    readable.assign(what="measured (instrument: potassium)", exponent=readable.exponent_potassium, lo=readable.potassium_lo, hi=readable.potassium_hi),
    readable.assign(what="predicted from IMPROVE thermal EC", exponent=readable.predicted_from_improve, lo=np.nan, hi=np.nan),
]).dropna(subset=["exponent"])[["site", "code", "what", "n", "median_optical_depth", "exponent", "lo", "hi"]]
site_fits = pd.concat([
    net.assign(comparator="HIPS as measured", slope=net.slope_measured, intercept=net.intercept_measured),
    net.assign(comparator="HIPS linearised", slope=net.slope_linearised, intercept=net.intercept_linearised),
])
site_fits["intercept_share"] = site_fits.intercept / site_fits.median_ftir_ec
site_fits = site_fits.sort_values("median_optical_depth")[["site", "code", "comparator", "n", "median_optical_depth", "slope", "intercept", "intercept_share"]]
pts = tables["network_points"]
site_points = pd.concat([
    pts.assign(comparator="HIPS as measured (Fabs / 10)", x=pts.hips_bc),
    pts.assign(comparator="HIPS linearised (/ 10)", x=pts.hips_linearised),
])[["id", "site", "date", "comparator", "x", "ftir_ec", "tau"]].dropna()
site_points["year"] = site_points.date.str[:4]
site_table = net[["site", "n", "median_optical_depth", "share_above_half", "hips_exponent_vs_ftir_ec", "lo", "hi", "exponent_potassium",
                  "predicted_from_improve", "intercept_measured", "intercept_linearised", "median_ftir_ec"]]
by = net.set_index("code")
closer = int(((net.intercept_linearised.abs() < net.intercept_measured.abs()) & net.readable).sum())
text_network = f"""**Across SPARTAN, the exponent falls with a site's optical depth about as the IMPROVE curve says, but one ceiling does not suit
every site.** Addis is the heaviest site (median optical depth {by.loc['ETAD', 'median_optical_depth']:.2f}) and the least proportional
({by.loc['ETAD', 'hips_exponent_vs_ftir_ec']:.2f} measured, {by.loc['ETAD', 'predicted_from_improve']:.2f} predicted). Linearising takes the Addis intercept from
{by.loc['ETAD', 'intercept_measured']:.2f} to {by.loc['ETAD', 'intercept_linearised']:.2f} µg/m³, Delhi from {by.loc['INDH', 'intercept_measured']:.2f} to
{by.loc['INDH', 'intercept_linearised']:.2f}, Santiago from {by.loc['CLST', 'intercept_measured']:.2f} to {by.loc['CLST', 'intercept_linearised']:.2f} and Beijing from
{by.loc['CHTS', 'intercept_measured']:.2f} to {by.loc['CHTS', 'intercept_linearised']:.2f}. It moves the intercept toward zero at {closer} of the {int(net.readable.sum())}
sites with a readable exponent. It over-corrects where HIPS and FTIR EC were already close to proportional at moderate optical depth:
Pretoria ({by.loc['ZAPR', 'intercept_measured']:.2f} to {by.loc['ZAPR', 'intercept_linearised']:+.2f}), Johannesburg, Jaipur, Mexico City and Dhaka
({by.loc['BDDU', 'intercept_measured']:.2f} to {by.loc['BDDU', 'intercept_linearised']:+.2f}). The two Chilean sites are far less proportional than their optical
depth predicts with either instrument (0.56), which is a real difference, not noise. Pick a site below to see its filters."""

# ---------------------------------------------------------------- 5b. every saved calibration, by season
cand = tables["candidates_by_season"]
by_season = cand[~cand.season.eq("All")]
overall = cand[cand.season.eq("All")][["calibration", "comparator", "n", "slope", "intercept", "median_ratio"]]
swing = by_season.pivot_table(index=["calibration", "comparator"], columns="season", values="median_ratio")
swing = (swing["Kiremt"] / swing["Bega"]).unstack("comparator")
pub_swing, spline_swing = swing.loc["SPARTAN published"], swing.loc["Spline baseline"]
inter = overall.pivot(index="calibration", columns="comparator", values="intercept")
text_candidates = f"""**Every saved calibration loses its intercept and more than half of its seasonal swing once HIPS is linearised.** Across the twelve
calibrations the Addis intercept runs from {inter['HIPS as measured'].min():.1f} to {inter['HIPS as measured'].max():.1f} µg/m³ against HIPS as measured and from
{inter['HIPS linearised'].min():.1f} to {inter['HIPS linearised'].max():+.1f} against linearised HIPS. For the published product, Kiremt reads
{pub_swing['HIPS as measured']:.2f}× Bega as measured and {pub_swing['HIPS linearised']:.2f}× once linearised. What remains differs by calibration: the spline
baseline keeps {spline_swing['HIPS linearised']:.2f}×, the full-spectrum calibrations about 1.3×. That is consistent with a composition-linked part in
FTIR (more organic signal in Kiremt), but season and loading still cannot be separated at Addis."""

# ---------------------------------------------------------------- 5c. the FTIR side, against thermal EC
source = tables["improve_ftir_linearity"]
lots = tables["improve_lot"]
ftir_level = tables["ftir_level"].copy()
ftir_candidates = tables["ftir_candidates"]
ftir_values = tables["ftir_ocec_values"]
ftir_epo, sens, blank_table = tables["ftir_epo"], tables["ftir_sensitivity"].set_index("calibration"), tables["ftir_blanks"].set_index("what")
epo_urban = ftir_epo[ftir_epo.filters.eq("six urban IMPROVE sites")].set_index("calibration")
raw_s, epo_s = sens.loc["Raw spectra"], sens.loc["Raw spectra + EPO"]
sensitivity_table = tables["ftir_sensitivity"][["calibration", "offset_ug", "gain", "gain_lo", "gain_hi", "oc_response", "level_urban", "level_addis", "level_addis_lo", "level_addis_hi"]]
co = blank_table.loc["Three urban IMPROVE filters co-added on one blank"]
text_steps = f"""**Both next steps are now run.** *EPO, re-scored.* No saved EPO predictions exist, so EPO was refitted on the raw spectra with the
repository's own implementation, inside each training fold. On IMPROVE's six urban sites it reads {epo_urban.loc['Raw spectra + EPO', 'ftir_over_tor']:.2f}
({epo_urban.loc['Raw spectra + EPO', 'lo']:.2f}–{epo_urban.loc['Raw spectra + EPO', 'hi']:.2f}) of thermal EC against {epo_urban.loc['Raw spectra', 'ftir_over_tor']:.2f} for
the plain calibration: a small real gain, at the cost of a higher median error overall ({epo_urban.loc['Raw spectra + EPO', 'overall_median_abs_error']:.2f} against
{epo_urban.loc['Raw spectra', 'overall_median_abs_error']:.2f} µg per filter).

*What the calibration responds to.* Across IMPROVE site means, the raw calibration's prediction is {raw_s.offset_ug:.1f} µg per filter plus
**{raw_s.gain:.2f} × thermal EC** ({raw_s.gain_lo:.2f}–{raw_s.gain_hi:.2f}) plus {raw_s.oc_response:.3f} × OC. It under-responds to EC and makes up the
difference with a fixed offset, which works at IMPROVE's loadings and not at Addis's. Blank filters show an offset too, a larger one: the calibration reads
{blank_table.loc['IMPROVE field blanks', 'predicted_ec_ug']:.1f} µg of EC on IMPROVE field blanks and {blank_table.loc['Addis field blanks', 'predicted_ec_ug']:.1f} µg on
Addis field blanks. EPO raises the gain to {epo_s.gain:.2f} and removes the OC response. Read at Addis loading and an OC/EC of 1, the raw
calibration gives {raw_s.level_addis:.2f} ({raw_s.level_addis_lo:.2f}–{raw_s.level_addis_hi:.2f}) of true EC and EPO {epo_s.level_addis:.2f}
({epo_s.level_addis_lo:.2f}–{epo_s.level_addis_hi:.2f}). That is a linear extrapolation, not a measurement.

*Spectral augmentation.* Co-adding three urban IMPROVE filters on one blank gives {co.tor_ec_ug:.0f} µg of EC, an Addis-like loading, and FTIR reads
{co.ftir_over_tor:.2f} of it. But for a linear calibration that number is exactly the sum of the three predictions minus twice the blank reading, so
co-adding shows the offset and nothing else, and a sum can never have a lower OC/EC than its parts. Augmentation cannot test Addis composition."""
ftir_by_site = tables["ftir_by_site"]
ftir_by_site = ftir_by_site[ftir_by_site.median_tor_ec_ug >= 1.5]    # below this a site median ratio is mostly noise
candidates_long = pd.concat([
    ftir_candidates.assign(chosen="By site: the six urban IMPROVE sites", ftir_over_tor=ftir_candidates.urban_sites),
    ftir_candidates.assign(chosen="By the filter's own TOR (heavy, low OC/EC)", ftir_over_tor=ftir_candidates.selected_on_own_tor),
])[["calibration", "chosen", "ftir_over_tor"]]
sel = tables["ftir_selection"]
low_bin = sel[sel.ocec < 2].groupby("chosen_by").ftir_over_tor.first()
pilot_ocec, pilot_ec = float(pilot.ocec.median()), float(pilot.tor_ug.median())
pilot_ftir = adama_ftir.set_index("calibration").median_unflagged
raw_low = source[(source.calibration.eq("Raw spectra")) & (source.filters.eq("TOR OC/EC < 2.5"))].iloc[0]
raw_all = source[(source.calibration.eq("Raw spectra")) & (source.filters.eq("all held-out filters"))].iloc[0]
own = flevel.loc["TOR EC ≥ 15 µg and OC/EC < 2.5"]
prev = flevel.loc["previous EC ≥ 15 µg and OC/EC < 2.5"]
cand_urban = ftir_candidates.set_index("calibration")
text_source = f"""**FTIR does read thermal EC low on Addis-like IMPROVE filters, by about a tenth, not a quarter.** The closeout's heavy low-OC/EC
subset (TOR EC ≥ 15 µg, TOR OC/EC < 2.5) gives FTIR ÷ TOR of {own.ftir_over_tor:.2f}. But that subset is chosen by each filter's own TOR
values, so it collects filters whose TOR EC happened to read high, and the ratio comes out too low. Choosing by the previous sample at the
site gives {prev.ftir_over_tor:.2f} ({prev.lo:.2f}–{prev.hi:.2f}); the six urban IMPROVE sites give {urban_level.ftir_over_tor:.2f} ({urban_level.lo:.2f}–{urban_level.hi:.2f}).
The saved calibrations then differ little: {cand_urban.urban_sites.min():.2f} to {cand_urban.urban_sites.max():.2f} on the urban sites, against
{cand_urban.selected_on_own_tor.min():.2f} to {cand_urban.selected_on_own_tor.max():.2f} on the selected subset. FTIR EC is proportional to thermal EC there
(exponent {raw_all.exponent:.2f}, {raw_all.lo:.2f}–{raw_all.hi:.2f}), and lot 248 reads no lower than lot 251.

**IMPROVE cannot say what the bias is at Addis.** With EC and OC calibrations fitted the same way on both, Addis FTIR OC/EC has a median of
{comp.addis_median:.2f}; {comp.addis_share_below_improve_p05 * 100:.0f} % of Addis filters are below IMPROVE's 5th percentile ({comp.improve_p05:.2f}), and the
urban IMPROVE median is {comp.improve_urban_median:.2f}. The bias grows as OC/EC falls across IMPROVE sites
({flevel.loc['sites with median OC/EC 3.5–5', 'ftir_over_tor']:.2f} at site OC/EC 3.5–5, {flevel.loc['sites with median OC/EC 2.5–3.5', 'ftir_over_tor']:.2f} at 2.5–3.5),
so it is probably larger at Addis, by an amount no IMPROVE filter can show. IMPROVE is a mostly rural and remote network; "urban" here means
its six city sites (Atlanta, Birmingham, Fresno, Phoenix, Pittsburgh, Seattle), which are its low-OC/EC end.

**A correction cannot be learned by binning on OC/EC.** In the lowest bin the same IMPROVE filters read
{low_bin["TOR's own OC/EC on this filter"]:.2f} of thermal EC when binned by TOR's own OC/EC and {low_bin["FTIR's own OC/EC on this filter"]:.2f} when binned by
FTIR's own. Each ratio contains the error of the EC it is built from, so the bin collects that error.

**The Adama TOR filters help with the MAC, not with the FTIR level at Addis.** The five Adama pairs are the only thermal EC in Ethiopia.
They carry {pilot_ec:.0f} µg of EC per filter at TOR OC/EC {pilot_ocec:.1f}; Addis filters carry about 34 µg at an OC/EC near 1. So Adama sits at
the opposite end from Addis, where the calibrations disagree most (FTIR ÷ TOR from {pilot_ftir.min():.2f} to {pilot_ftir.max():.2f} across four
calibrations on three clean pairs). What Adama does show: per unit of thermal EC, HIPS reads {pilot.fabs_over_tor.min():.0f} to
{pilot.fabs_over_tor.max():.0f} m²/g there, above IMPROVE's curve at the same optical depth. Ethiopian aerosol absorbs more per unit EC than
MAC 10 assumes, which is the other half of the 0.55 slope."""

# ---------------------------------------------------------------- 5d. can the calibration be fixed from IMPROVE alone
gain_file = json.loads((ROOT / "docs/calibration-gain-2026-11-10.json").read_text())
gv = pd.DataFrame(gain_file["variants"])
gv["calibration"] = np.where(gv.recalibrated.eq("no"), gv.variant, gv.variant + ", then offset removed and divided by the gain")
gv["step"] = np.where(gv.recalibrated.eq("no"), "as trained", "recalibrated on other sites")
gain_rows = gv[["preprocessing", "calibration", "variant", "step", "gain", "gain_lo", "gain_hi", "offset", "blank_reading", "heavy_level", "heavy_sum_ratio",
                "urban_level", "light_level", "light_median_abs_error", "overall_median_abs_error", "overall_rmse"]]
gain_k = pd.DataFrame(gain_file["components"])[["preprocessing", "variant", "k", "gain", "offset"]]
gain_closeout = pd.DataFrame(gain_file["closeout_calibrations"])[["calibration", "filters", "sites", "gain", "gain_lo", "gain_hi", "offset", "urban_level", "heavy_level"]]
g = gv[gv.preprocessing.eq("Raw spectra")].set_index(["variant", "recalibrated"])
base, fixed = g.loc[("As fitted", "no")], g.loc[("As fitted", "offset removed, divided by the gain")]
weighted, with_blanks = g.loc[("Weighted toward heavy filters", "no")], g.loc[("Blanks added at zero", "no")]
kk = gain_k[(gain_k.preprocessing.eq("Raw spectra")) & (gain_k.variant.eq("As fitted"))].set_index("k").gain
cl = gain_closeout.set_index("calibration")
best = g.loc[("Blanks added at zero", "offset removed, divided by the gain")]
refit_addis = tables["addis_refit"].set_index("filter_id")
volume_addis = table.volume_m3.reindex(refit_addis.index)
recal_addis = ((refit_addis.ftir_ec_ugm3 * volume_addis - base.offset) / base.gain) / volume_addis
addis_factor_raw = float((recal_addis / refit_addis.ftir_ec_ugm3).median())
addis_factor_published = float((recal_addis / table[closure.PUBLISHED].reindex(refit_addis.index)).median())
text_gain = f"""**Recalibrating from the offset and gain works on sites it has not seen.** Removing the offset and dividing by the gain, both
fitted on other IMPROVE sites, takes the gain from {base.gain:.2f} to {fixed.gain:.2f} ({fixed.gain_lo:.2f}–{fixed.gain_hi:.2f}), the level on IMPROVE's six
urban sites from {base.urban_level:.2f} to {fixed.urban_level:.2f}, and on heavy filters from {base.heavy_level:.2f} to {fixed.heavy_level:.2f}. Light filters are
not harmed: their level goes from {base.light_level:.2f} to {fixed.light_level:.2f} and their median error from {base.light_median_abs_error:.2f} to
{fixed.light_median_abs_error:.2f} µg per filter. The cost is in the largest errors, which are scaled up with everything else (RMSE
{base.overall_rmse:.2f} to {fixed.overall_rmse:.2f}); the overall median error is unchanged ({base.overall_median_abs_error:.2f}, {fixed.overall_median_abs_error:.2f}).

**Why the gain is low: more components raise it only slowly, and how the training is weighted does not move it.** The gain rises with the number of components and
levels off below 1: {kk.loc[6]:.2f} at 6, {kk.loc[12]:.2f} at 12, {kk.loc[18]:.2f} at 18, {kk.loc[25]:.2f} at 25. Weighting the training toward heavy filters
leaves it at {weighted.gain:.2f} and only raises the offset (a blank then reads {weighted.blank_reading:.1f} µg and the light-filter error nearly
doubles, {weighted.light_median_abs_error:.2f}). Adding blank filters at zero is a small free gain at the low end: the blank reading falls from
{base.blank_reading:.1f} to {with_blanks.blank_reading:.1f} µg and the overall median error from {base.overall_median_abs_error:.2f} to
{with_blanks.overall_median_abs_error:.2f}, with the gain at {with_blanks.gain:.2f}. Blanks at zero and the recalibration together give gain
{best.gain:.2f}, urban sites {best.urban_level:.2f}, light filters {best.light_level:.2f}, and a blank reading of {best.blank_reading:.1f} µg.

**The baseline does not change the gain much.** On the 1,821 heavier filters the closeout exported, the gain is {cl.loc['Raw spectra', 'gain']:.2f} for
raw spectra, {cl.loc['Spline baseline', 'gain']:.2f} for the spline baseline, {cl.loc['Production VIBES', 'gain']:.2f} for production VIBES and
{cl.loc['PTFE-anchored VIBES', 'gain']:.2f} for PTFE-anchored VIBES; deflated VIBES is lower ({cl.loc['Deflated VIBES', 'gain']:.2f}) and the OC/EC-balanced
weighting higher ({cl.loc['Anchored, OC/EC-balanced weights', 'gain']:.2f}). That subset leaves out the light filters, which lowers every gain, so
compare the rows with each other and not with the full refit. `run_calibration_gain_20261110.py --bundle` repeats the full test on the
baseline-corrected spectra where they are stored.

**At Addis** the same recalibration raises the raw calibration by × {addis_factor_raw:.2f} at the median filter, which is × {addis_factor_published:.2f}
on the published product. It is fitted on IMPROVE and checked on IMPROVE: it corrects the loading part of the bias, and says nothing about
whether Addis composition adds to it."""

# ---------------------------------------------------------------- 6. what changes, what is open
spline_level = float((table["spline k18"] / table.hips_bc).median())
published_level = float((table[closure.PUBLISHED] / table.hips_bc).median())
rereads = pd.DataFrame([
    ("The problem is calibration transfer, not HIPS", "Most of the −6 µg/m³ intercept is on the HIPS side: it goes to about zero once HIPS is linearised with a ceiling from IMPROVE thermal EC. A level question about FTIR remains."),
    ("Baseline correction is the largest single improvement", f"The spline calibration reads Addis at {spline_level:.2f} of HIPS / 10 against {published_level:.2f} for the published product. It lowers the whole response, which narrows the gap to a compressed HIPS. Judge baselines on IMPROVE thermal EC, not on the Addis HIPS intercept."),
    ("The seasons fail differently", f"Within lot 251, FTIR ÷ HIPS swings {s251.loc['Bega', 'published_over_hips']:.2f} to {s251.loc['Kiremt', 'published_over_hips']:.2f} as measured and {s251.loc['Bega', 'published_over_linearised_hips']:.2f} to {s251.loc['Kiremt', 'published_over_linearised_hips']:.2f} once HIPS is linearised. A smaller season or loading effect remains."),
    ("A MAC of 28 to 32 would be needed in Dry and Belg", f"Those are slopes against compressed HIPS. Against linearised HIPS it is about {row['central'].mac_for_agreement:.0f} m²/g."),
    ("FTIR reads a quarter low on heavy, low-OC/EC IMPROVE filters (closeout heavy subset, 0.72 to 0.78)", f"That subset selects on TOR's own error. Grouped by site or by the previous sample, FTIR ÷ TOR is {urban_level.ftir_over_tor:.2f} ({urban_level.lo:.2f}–{urban_level.hi:.2f}), and the calibrations differ by {cand_urban.urban_sites.min():.2f} to {cand_urban.urban_sites.max():.2f}, not {cand_urban.selected_on_own_tor.min():.2f} to {cand_urban.selected_on_own_tor.max():.2f}."),
    ("Organic deficit: partial correlation with HIPS held fixed", "Holding a compressed variable fixed leaves real EC variation in the residual. Redo with linearised HIPS held fixed."),
], columns=["earlier reading", "with HIPS linearised"])
pathways = pd.DataFrame([
    ("Re-read the 28 Oct season, cohort, deficit and MAC results against linearised HIPS", "open, data in hand", "1"),
    ("Lot 248 reads 15 to 20 % low at Addis; IMPROVE shows no lot effect, so it is the SPARTAN lot-248 blanks or the 2022–23 aerosol", "narrowed here", "2"),
    ("Why the HIPS ceiling differs between 2005–2014 and 2015–2025 in IMPROVE (instrument history), and which applies to SPARTAN", "new, data in hand", "3"),
    ("Site-specific HIPS response: why South Africa, Jaipur and Mexico City stay near-proportional and Chile does not", "new, data in hand", "3"),
    ("A composition-aware level factor learned on IMPROVE", "tested here: not possible, Addis is outside IMPROVE's composition range", ""),
    ("Spectral augmentation by co-adding IMPROVE spectra", "run here: shows only the blank offset; cannot reach Addis composition", ""),
    ("EPO re-scored with fair grouping (refitted on raw spectra; no saved predictions exist)", "run here: 0.93 against 0.91 on the urban sites", ""),
    ("A recalibration from the offset and EC gain, fitted on other IMPROVE sites", "run here: urban sites 0.91 to 1.02, light filters unharmed", ""),
    ("Why the EC gain is 0.78: weighting, blanks at zero, more components", "run here: it is the spectra, not the training", ""),
    ("The same gain test on spline, VIBES and anchored spectra (run_calibration_gain_20261110.py --bundle)", "script committed; needs the saved bundle", "4"),
    ("Whether to carry the IMPROVE recalibration to Addis as the stated correction (about × 1.1 on the published product)", "a decision, with Ann", "5"),
    ("FTIR proportional to thermal EC on held-out IMPROVE sites (exponent 0.94 to 1.04)", "done here", ""),
    ("The exponent and the linearised fit at every SPARTAN site, with potassium as a second instrument", "done here", ""),
    ("PLS on the masked regions; raw model without 3600–4000 and below 1425 cm⁻¹", "open, lower priority", "7"),
    ("OC/EC-selected cohorts, organic deficit, dust, season-specific MAC, OP = 0, urban sites", "done 28 Oct (HIPS / 10 on x)", ""),
    ("Adama TOR pairs (the only Ethiopian thermal EC): used for the MAC; too light and too organic to test the Addis FTIR level", "used here", ""),
    ("Addis quartz + PTFE pairs, with HIPS and some heavy loadings", "blocked: filters not in hand", ""),
], columns=["pathway", "status", "rank"])
text_next = """**Proposed position.** Carry the published FTIR EC forward with no intercept correction and a stated level range. Stop using
Fabs / 10 as the Addis yardstick without linearising. A modest season or loading effect and a lot-248 offset remain open. The proposed TOR
pass band (0.8 to 1.25) now contains the most likely answer (about 0.9 to 1.0), with the lower edge still inside the plausible range
(0.7 to 1.15). Ten pairs fix a mean bias only to about ±27 % (3 Nov simulation), which is wider than the question that is left, so decide before they
arrive whether the deliverable is pass or fail, or an estimated level factor, and ask for HIPS on the same filters with a few loaded to Addis optical depth. Optical agreement is still not thermal EC."""

intro = f"""Which measurement carries the Addis offset: an answer to the question the 9 Nov status page left open, from data already in
hand ({len(table)} Addis filters of the 7 Nov closeout, and the IMPROVE record of thermal EC and HIPS). **The aethalometer is not used**: the
Addis MA350 is not a reference. A results page, not a recorded meeting. Exploratory: no Addis filter has thermal (TOR) EC, so the absolute
level stays a range. Written account: docs/addis-reference-closure-2026-11-10.md."""

write_bundle(
    MEETING_ID,
    generated_from="IMPROVE TOR and HIPS tables, gallery/review/closeout_20261107/continuum_baseline.json, filters.json, spartan_reference.json "
                   "and the unified filter dataset via research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py",
    colors=COLORS,
    text={"intro": intro, "improve": text_improve, "addis": text_addis, "response": text_response, "season": text_season,
          "level": text_level, "network": text_network, "candidates": text_candidates, "source": text_source, "mac": text_mac, "steps": text_steps, "gain": text_gain, "next": text_next},
    tables={"strata": pd.concat([strata[["population", "n", "TOR EC (µg per filter)", "HIPS optical depth", "Fabs ÷ TOR EC (m²/g)"]], adama_points], ignore_index=True),
            "adama_filters": adama_filters, "adama_ftir": adama_ftir, "ftir_by_site": ftir_by_site, "ftir_selection": tables["ftir_selection"],
            "ftir_candidates_long": candidates_long, "mac_bars": mac_bars, "mac_levels": mac_levels, "ftir_epo": ftir_epo,
            "ftir_sensitivity": sensitivity_table, "ftir_blanks": tables["ftir_blanks"],
            "gain_rows": gain_rows, "gain_k": gain_k, "gain_closeout": gain_closeout,
            "ceilings": ceilings, "exponents": exponents, "against_hips": against_hips, "fits": fits,
            "season": season_rows[["group", "season", "lot", "n", "median_optical_depth", "published_over_hips",
                                   "published_over_linearised_hips", "published_over_pm25"]],
            "anchors": anchors, "ftir_level": ftir_level, "ftir_candidates": ftir_candidates, "ftir_values": ftir_values, "mass": mass, "network": network, "site_fits": site_fits, "site_points": site_points,
            "site_table": site_table, "cand_season": by_season, "cand_overall": overall, "source": source, "lots": lots,
            "rereads": rereads, "pathways": pathways},
)
