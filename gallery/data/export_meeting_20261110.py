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
COLORS = {**SEASON_COLORS, "all sites": "#22252A", "sites with median OC/EC < 4": "#2C6E9E", "six urban sites": "#B23327",
          "PM2.5 mass (gravimetric)": "#27AE60", "Potassium (XRF)": "#2C6E9E", "FTIR EC, SPARTAN published": "#B23327",
          "HIPS Fabs": "#5B5852", "measured (instrument: PM2.5 mass)": "#C49442", "measured (instrument: potassium)": "#2C6E9E",
          "predicted from IMPROVE thermal EC": "#8a8f98", "HIPS as measured": "#B23327", "HIPS linearised": "#2C6E9E",
          "2005–2014": "#8F8C84", "2015–2025": "#7A4FA3", "optical pyrolysis > 0": "#27AE60",
          "Raw spectra": "#22252A", "PTFE-anchored VIBES": "#C49442", "Spline baseline": "#2C6E9E"}


tables, table = closure.run(n_boot=2000)
PUB = closure.LABEL[closure.PUBLISHED]
ceiling = closure.thermal_ceiling(tables)

# ---------------------------------------------------------------- 1. IMPROVE: HIPS against thermal EC
strata = tables["improve_strata"].rename(columns={"loading_ug": "TOR EC (µg per filter)", "tau": "HIPS optical depth",
                                                  "fabs_over_ec": "Fabs ÷ TOR EC (m²/g)"})
from pls_transfer import deming_regression  # noqa: E402

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
the six urban sites give the same. Addis optical depth has a median of {table.tau.median():.2f} and reaches {table.tau.max():.2f}, so
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
ocec = tables["improve_ratio_by_ocec"]
anchors = tables["level_anchors"].rename(columns={"published_over_true": "published ÷ true"})
mass = tables["mass_closure"].rename(columns={"ec_multiplier": "EC multiplier", "reconstructed_over_gravimetric": "reconstructed ÷ gravimetric PM2.5",
                                              "ec_share_of_pm25": "EC share of PM2.5"})
addis_ocec = tables["addis_ocec"].iloc[0]
ratios = anchors["published ÷ true"].dropna()
text_level = f"""**What is left for FTIR is mainly one level factor, bracketed at about × 1.0 to 1.5.** Linearised HIPS needs
{row['central'].mac_for_agreement:.1f} m²/g to agree with the published EC. MACs measured with thermal EC are lower: {primary.unsaturated_fabs_over_ec:.1f} m²/g
across IMPROVE on light filters, {ceilings.set_index('population').loc['six urban sites', 'unsaturated_fabs_over_ec']:.1f} at the urban sites, about 19 on the
five Adama filters. Those put published ÷ true between {ratios.min():.2f} and {ratios.max():.2f}. On held-out IMPROVE sites the calibration also reads
thermal EC lower the lower the OC/EC ({ocec.iloc[-1]['Raw spectra']:.2f} above OC/EC 4, {ocec.iloc[0]['Raw spectra']:.2f} below 1.5; Addis FTIR OC/EC is
{addis_ocec['median']:.2f}). Mass closure caps it: published EC reconstructs {mass.iloc[0, 2] * 100:.0f} % of gravimetric PM2.5, × 1.5 reaches
{mass.iloc[3, 2] * 100:.0f} % and × 1.9 gives {mass.iloc[5, 2] * 100:.0f} %. None of this is thermal EC measured at Addis."""

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

# ---------------------------------------------------------------- 5c. inside the calibration set
source = tables["improve_ftir_linearity"]
lots = tables["improve_lot"]
raw_low = source[(source.calibration.eq("Raw spectra")) & (source.filters.eq("TOR OC/EC < 2.5"))].iloc[0]
raw_all = source[(source.calibration.eq("Raw spectra")) & (source.filters.eq("all held-out filters"))].iloc[0]
text_source = f"""**On held-out IMPROVE sites FTIR EC is proportional to thermal EC, and filter lot does not matter there.** The exponent of the
held-out FTIR prediction against TOR EC (instrument: HIPS) is {raw_all.exponent:.2f} ({raw_all.lo:.2f}–{raw_all.hi:.2f}) over all filters and
{raw_low.exponent:.2f} ({raw_low.lo:.2f}–{raw_low.hi:.2f}) at TOR OC/EC below 2.5. So the calibration does not rise faster than EC where EC is known, and
the small leftover at Addis points at the HIPS ceiling there rather than at FTIR. IMPROVE filters of lot 248 read thermal EC no lower than lot
251, so the low Addis lot-248 readings are not a general property of that filter lot. This is the calibration set: it does not show how FTIR
behaves on Addis aerosol."""

# ---------------------------------------------------------------- 6. what changes, what is open
spline_level = float((table["spline k18"] / table.hips_bc).median())
published_level = float((table[closure.PUBLISHED] / table.hips_bc).median())
rereads = pd.DataFrame([
    ("The problem is calibration transfer, not HIPS", "Most of the −6 µg/m³ intercept is on the HIPS side: it goes to about zero once HIPS is linearised with a ceiling from IMPROVE thermal EC. A level question about FTIR remains."),
    ("Baseline correction is the largest single improvement", f"The spline calibration reads Addis at {spline_level:.2f} of HIPS / 10 against {published_level:.2f} for the published product. It lowers the whole response, which narrows the gap to a compressed HIPS. Judge baselines on IMPROVE thermal EC, not on the Addis HIPS intercept."),
    ("The seasons fail differently", f"Within lot 251, FTIR ÷ HIPS swings {s251.loc['Bega', 'published_over_hips']:.2f} to {s251.loc['Kiremt', 'published_over_hips']:.2f} as measured and {s251.loc['Bega', 'published_over_linearised_hips']:.2f} to {s251.loc['Kiremt', 'published_over_linearised_hips']:.2f} once HIPS is linearised. A smaller season or loading effect remains."),
    ("A MAC of 28 to 32 would be needed in Dry and Belg", f"Those are slopes against compressed HIPS. Against linearised HIPS it is about {row['central'].mac_for_agreement:.0f} m²/g."),
    ("Organic deficit: partial correlation with HIPS held fixed", "Holding a compressed variable fixed leaves real EC variation in the residual. Redo with linearised HIPS held fixed."),
], columns=["earlier reading", "with HIPS linearised"])
pathways = pd.DataFrame([
    ("Re-read the 28 Oct season, cohort, deficit and MAC results against linearised HIPS", "open, data in hand", "1"),
    ("Lot 248 reads 15 to 20 % low at Addis; IMPROVE shows no lot effect, so it is the SPARTAN lot-248 blanks or the 2022–23 aerosol", "narrowed here", "2"),
    ("Why the HIPS ceiling differs between 2005–2014 and 2015–2025 in IMPROVE (instrument history), and which applies to SPARTAN", "new, data in hand", "3"),
    ("Site-specific HIPS response: why South Africa, Jaipur and Mexico City stay near-proportional and Chile does not", "new, data in hand", "3"),
    ("A composition-aware level factor learned on held-out IMPROVE sites", "open, data in hand", "4"),
    ("Spectral augmentation: co-added IMPROVE spectra at Addis-like loading and OC/EC", "never run", "5"),
    ("FTIR proportional to thermal EC on held-out IMPROVE sites (exponent 0.94 to 1.04)", "done here", ""),
    ("The exponent and the linearised fit at every SPARTAN site, with potassium as a second instrument", "done here", ""),
    ("PLS on the masked regions; raw model without 3600–4000 and below 1425 cm⁻¹", "open, lower priority", "7"),
    ("OC/EC-selected cohorts, organic deficit, dust, season-specific MAC, OP = 0, urban sites", "done 28 Oct (HIPS / 10 on x)", ""),
    ("Addis quartz + PTFE pairs, with HIPS and some heavy loadings", "blocked: filters not in hand", ""),
], columns=["pathway", "status", "rank"])
text_next = """**Proposed position.** Carry the published FTIR EC forward with no intercept correction and a stated level range. Stop using
Fabs / 10 as the Addis yardstick without linearising. A modest season or loading effect and a lot-248 offset remain open. The proposed TOR
pass band (0.8 to 1.25) sits where the existing evidence predicts the answer (about 0.65 to 1.0), so ten filters would most likely read
"inconclusive": decide before they arrive whether the deliverable is pass or fail, or an estimated level factor, and ask for HIPS on the
same filters with a few loaded to Addis optical depth. Optical agreement is still not thermal EC."""

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
          "level": text_level, "network": text_network, "candidates": text_candidates, "source": text_source, "next": text_next},
    tables={"strata": strata[["population", "n", "TOR EC (µg per filter)", "HIPS optical depth", "Fabs ÷ TOR EC (m²/g)"]],
            "ceilings": ceilings, "exponents": exponents, "against_hips": against_hips, "fits": fits,
            "season": season_rows[["group", "season", "lot", "n", "median_optical_depth", "published_over_hips",
                                   "published_over_linearised_hips", "published_over_pm25"]],
            "ocec": ocec, "anchors": anchors, "mass": mass, "network": network, "site_fits": site_fits, "site_points": site_points,
            "site_table": site_table, "cand_season": by_season, "cand_overall": overall, "source": source, "lots": lots,
            "rereads": rereads, "pathways": pathways},
)
