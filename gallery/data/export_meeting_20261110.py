"""Export the 10 Nov 2026 page "Which measurement carries the Addis offset" for the gallery's Meeting tab.

Run from the repo root with the repo environment:

    uv run python gallery/data/export_meeting_20261110.py

A results page, not a recorded meeting. It answers the question the 9 Nov status page left open, with
data already in the repository. Every number comes from the workflow
research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py, which this runs in memory
(committed inputs only: the 7 Nov closeout export, filters.json, spartan_reference.json and the unified
filter dataset). docs/addis-reference-closure-2026-11-10.md is the written account.

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
BOOTSTRAP = 2000
SEASON_COLORS = {"Bega": "#E67E22", "Belg": "#27AE60", "Kiremt": "#3498DB"}   # config.ETHIOPIA_SEASONS colours
COLORS = {**SEASON_COLORS, "HIPS Fabs": "#5B5852", "FTIR EC, SPARTAN published": "#B23327",
          "FTIR EC, PTFE-anchored VIBES": "#C49442", "Aethalometer IR BCc": "#2C6E9E",
          "HIPS, saved linearisation": "#8F8C84", "Inferred TOR comparator": "#7A4FA3",
          "PM2.5 mass (gravimetric)": "#27AE60", "Addis Ababa": "#C49442", "other SPARTAN sites": "#8a8f98"}
SITE_NAMES = {"ETAD": "Addis Ababa", "BDDU": "Dhaka", "USPA": "Pasadena", "CHTS": "Beijing", "INDH": "Delhi",
              "TWTA": "Taipei", "TWKA": "Kaohsiung", "CLST": "Santiago", "CLTA": "Talca", "ZAPR": "Pretoria",
              "ZAJB": "Johannesburg", "AUMN": "Melbourne", "ETBI": "Bishoftu"}

table, closeout = closure.load()
tables: dict[str, pd.DataFrame] = {}
for stage in closure.STAGES:
    tables |= closure.RUN[stage](table, closeout, BOOTSTRAP)

# ---------------------------------------------------------------- 1. who is proportional to whom
lin = tables["linearity"]
whole = lin[lin.group.eq("all")].copy()
whole["pair"] = whole.y + " against " + whole.x
exponents = whole[["pair", "y", "instrument", "n", "exponent", "lo", "hi"]]
by_season = lin[~lin.group.eq("all")][["group", "y", "n", "exponent", "lo", "hi"]].rename(columns={"group": "season"})


def exponent(y, x, z):
    row = whole[(whole.y.eq(closure.LABEL[y])) & (whole.x.eq(closure.LABEL[x])) & (whole.instrument.eq(closure.LABEL[z]))].iloc[0]
    return row.exponent, row.lo, row.hi


h_pm, h_k = exponent("fabs", "aeth_ir", "pm25"), exponent("fabs", "aeth_ir", "potassium")
f_pm, f_k = exponent(closure.PUBLISHED, "aeth_ir", "pm25"), exponent(closure.PUBLISHED, "aeth_ir", "potassium")
counts = {"filters": len(table), "aeth": int(table.aeth_ir.notna().sum()), "pm": int(table.pm25.notna().sum()),
          "published": int(table[closure.PUBLISHED].notna().sum())}

# per-filter points (already public in the closeout export), with the linearised HIPS beside the measured one
stress = closeout["settings"]["stress"]
tau_inf, tau_lo, tau_hi = stress["curve"]["tau_inf"], stress["tau_inf_boot"][0], stress["tau_inf_boot"][2]
linear_bc = rc.linearise(table.tau, tau_inf) * 100.0 * closure.DEPOSIT_AREA_CM2 / table.volume_m3 / closure.MAC_VALUE
points = pd.DataFrame({
    "filter": table.index, "season": table.season, "lot": table.lot,
    "aethalometer": table.aeth_ir, "pm25": table.pm25, "hips_bc": table.hips_bc, "hips_linearised": linear_bc,
    "optical_depth": table.tau, "ftir_published": table[closure.PUBLISHED], "ftir_anchored": table[closure.ANCHORED],
})
against_aeth = pd.concat([
    points.assign(measure="HIPS Fabs / 10", value=points.hips_bc),
    points.assign(measure="FTIR EC, SPARTAN published", value=points.ftir_published),
])[["filter", "season", "measure", "aethalometer", "value"]].dropna()
against_hips = pd.concat([
    points.assign(comparator="HIPS as measured (Fabs / 10)", x=points.hips_bc),
    points.assign(comparator="HIPS linearised (/ 10), Addis held out", x=points.hips_linearised),
])[["filter", "season", "comparator", "x", "ftir_published"]].dropna()

raw, linearised = tables["raw_hips"].set_index("model"), tables["linearised_hips"]
pub_label = closure.LABEL[closure.PUBLISHED]
central = linearised[(linearised.which.eq("central")) & (linearised.model.eq(pub_label))].iloc[0]
lo_row = linearised[(linearised.which.eq("lo")) & (linearised.model.eq(pub_label))].iloc[0]
hi_row = linearised[(linearised.which.eq("hi")) & (linearised.model.eq(pub_label))].iloc[0]
fits = pd.concat([
    tables["raw_hips"].rename(columns={"slope_vs_hips": "slope", "intercept_vs_hips": "intercept"}).assign(comparator="HIPS as measured"),
    linearised.rename(columns={"slope_vs_linearised_hips": "slope", "intercept_vs_linearised_hips": "intercept"}).assign(
        comparator=lambda d: "HIPS linearised, ceiling " + d.tau_inf.round(2).astype(str)),
])[["comparator", "model", "n", "slope", "intercept"]]

text_linearity = f"""**HIPS bends; FTIR EC does not.** Against HIPS optical BC the published FTIR EC reads
{raw.loc[pub_label, 'slope_vs_hips']:.2f}x − {abs(raw.loc[pub_label, 'intercept_vs_hips']):.2f} µg/m³. Two instruments cannot say which one is not
proportional, so this uses a third as the instrument: the exponent b in Y ∝ X^b, free of the flattening that noise in X causes
in a log-log fit. Against the aethalometer, **HIPS rises to the power {h_pm[0]:.2f}** ({h_pm[1]:.2f}–{h_pm[2]:.2f}; instrument PM2.5 mass) or
{h_k[0]:.2f} (potassium), while **FTIR EC rises to the power {f_pm[0]:.2f}** ({f_pm[1]:.2f}–{f_pm[2]:.2f}) or {f_k[0]:.2f}. PM2.5 mass is gravimetric
and potassium is XRF: neither is optical or FTIR, so this does not rest on trusting one optical instrument over another.
{counts['filters']} Addis filters; {counts['aeth']} with an aethalometer match, {counts['pm']} with PM2.5 mass, {counts['published']} with a published EC."""

text_response = f"""**Linearising HIPS with a constant fitted elsewhere removes the offset.** The saturation ceiling fitted on Beijing,
Pasadena and Delhi with Addis held out (3 Nov) is {tau_inf:.2f} ({tau_lo:.2f}–{tau_hi:.2f}); Addis optical depth has a median of
{table.tau.median():.2f} and reaches {table.tau.max():.2f}. Applying that ceiling, the published fit moves from
{raw.loc[pub_label, 'slope_vs_hips']:.2f}x − {abs(raw.loc[pub_label, 'intercept_vs_hips']):.2f} to **{central.slope_vs_linearised_hips:.2f}x + {central.intercept_vs_linearised_hips:.2f}**
(intercept {lo_row.intercept_vs_linearised_hips:+.2f} to {hi_row.intercept_vs_linearised_hips:+.2f} across the ceiling's interval), and linearised HIPS
becomes proportional to the aethalometer (power {central.linearised_exponent_vs_aethalometer:.2f}). The correction is large
(median × {central.median_correction_factor:.2f}) and multiplies noise on the heaviest filters. The remaining slope is the
level question: FTIR EC would need a MAC near {central.mac_for_agreement:.0f} m²/g to agree."""

# ---------------------------------------------------------------- 2. random error with no reference
tc = tables["triple_collocation"]
first = tc[tc.triplet.eq(tc.triplet.iloc[0])].copy()
for c in ("error_sd", "error_sd_lo", "error_sd_hi"):
    first[c] = first[c] * 100.0
errors = first[["measurement", "n", "error_sd", "error_sd_lo", "error_sd_hi", "exponent"]]
text_errors = "**FTIR EC is about as precise as the aethalometer at Addis.** " + "; ".join(
    f"{r.measurement} {r.error_sd:.0f} % ({r.error_sd_lo:.0f}–{r.error_sd_hi:.0f})" for r in errors.itertuples()) + (
    ". Random error from three-way agreement alone (log-space triple collocation), on a common scale. It says how tightly each "
    "follows the shared day-to-day signal, not how accurate its level is.")

# ---------------------------------------------------------------- 3. seasons and lots
season = tables["season_lot"]
season_rows = season[~season.lot.eq("all")].copy()
season_rows["group"] = season_rows.season + ", lot " + season_rows.lot.astype(str)
in251 = season_rows[season_rows.lot.astype(str).eq("251")].set_index("season").published_over_aethalometer
b248 = season_rows[(season_rows.lot.astype(str).eq("248")) & (season_rows.season.eq("Bega"))].iloc[0]
text_season = f"""**Within lot 251 the published FTIR EC has no season effect against the aethalometer:** median FTIR ÷ aethalometer is
{in251['Bega']:.2f} in Bega, {in251['Belg']:.2f} in Belg and {in251['Kiremt']:.2f} in Kiremt. The low Bega value seen before comes from the
{int(b248.n)} lot-248 filters of December 2022 to early 2023 ({b248.published_over_aethalometer:.2f}), a lot or early-record question rather
than seasonal chemistry. HIPS / 10 over the same rows falls from {season_rows.hips_mac10_over_aethalometer.max():.2f} to
{season_rows.hips_mac10_over_aethalometer.min():.2f}: Kiremt has the highest concentrations, so compressed HIPS reads lowest there."""

# ---------------------------------------------------------------- 4. the level
ocec = tables["improve_ratio_by_ocec"]
anchors = tables["level_anchors"].rename(columns={"published_over_true": "published ÷ true", "value": "value"})
mass = tables["mass_closure"].rename(columns={"ec_multiplier": "EC multiplier", "reconstructed_over_gravimetric": "reconstructed ÷ gravimetric PM2.5",
                                              "ec_share_of_pm25": "EC share of PM2.5"})
addis_ocec = tables["addis_ocec"].iloc[0]
lowest = ocec.iloc[0]
text_level = f"""**What is left for FTIR is one level factor, and the existing data bracket it at about × 1.0 to 1.5.** On held-out
IMPROVE sites the calibration reads thermal EC lower the lower the OC/EC: median FTIR ÷ TOR is {ocec.iloc[-1]['Raw spectra']:.2f} above OC/EC 4
and {lowest['Raw spectra']:.2f} below 1.5 (n {int(lowest.n)}). Addis FTIR OC/EC has a median of {addis_ocec['median']:.2f}. The aethalometer scaled by
Pasadena or Delhi, linearised HIPS at the Adama MAC, and PM2.5 mass closure point the same way. Mass closure also caps it: published EC
reconstructs {mass.iloc[0, 2] * 100:.0f} % of gravimetric PM2.5, × 1.5 reaches {mass.iloc[3, 2] * 100:.0f} % and × 1.9 gives {mass.iloc[5, 2] * 100:.0f} %, so
the aethalometer or linearised HIPS read at MAC 10 cannot be EC. None of this is thermal EC measured at Addis."""

# ---------------------------------------------------------------- 5. the network
net = tables["network_exponents"].copy()
net["site"] = net.code.map(SITE_NAMES).fillna(net.code)
net["which"] = np.where(net.code.eq("ETAD"), "Addis Ababa", "other SPARTAN sites")
net["interval_width"] = net.hi - net.lo
readable = net[net.interval_width < 0.6].sort_values("median_optical_depth")
addis_net, dhaka_net = net[net.code.eq("ETAD")].iloc[0], net[net.code.eq("BDDU")].iloc[0]
text_network = f"""**Addis is the heaviest SPARTAN site optically, and the most compressed.** With PM2.5 mass as the instrument no
aethalometer is needed, so the same exponent can be read at every site. Addis has the highest median optical depth
({addis_net.median_optical_depth:.2f}) and the lowest exponent of HIPS against FTIR EC ({addis_net.hips_exponent_vs_ftir_ec:.2f}); Dhaka is next on
both ({dhaka_net.median_optical_depth:.2f}; {dhaka_net.hips_exponent_vs_ftir_ec:.2f}). A first pass: sites where the interval is wider than 0.6 are
left off the chart ({len(net) - len(readable)} of {len(net)}), and the two Chilean sites need a second look before anyone quotes them."""

# ---------------------------------------------------------------- 6. what changes, what is open
rereads = pd.DataFrame([
    ("The problem is calibration transfer, not HIPS", "The −6 µg/m³ intercept is on the HIPS side. A level question about FTIR remains."),
    ("Baseline correction is the largest single improvement", "The spline calibration reads 0.25 of the aethalometer against 0.65 for the published product: it closed the HIPS intercept by giving up most of the response."),
    ("The seasons fail differently", "Kiremt has the highest concentrations, so compressed HIPS reads relatively low there. Within lot 251 the published product has no season effect against the aethalometer."),
    ("A MAC of 28 to 32 would be needed in Dry and Belg", f"Those are slopes against compressed HIPS. Against linearised HIPS it is about {central.mac_for_agreement:.0f} m²/g at every loading."),
    ("Organic deficit: partial correlation with HIPS held fixed", "Holding a compressed variable fixed leaves real EC variation in the residual. Redo with the aethalometer held fixed."),
], columns=["earlier reading", "with a linear comparator"])
pathways = pd.DataFrame([
    ("Re-read the 28 Oct season, cohort, deficit and MAC results against the aethalometer and linearised HIPS", "open, data in hand", "1"),
    ("The lot-248 early record: filter lot, or the aethalometer's first months", "open, data in hand", "2"),
    ("A thermal anchor for the HIPS curve inside IMPROVE (TOR EC on low-OC/EC urban sites)", "open, data in hand", "3"),
    ("A composition-aware level factor learned on held-out IMPROVE sites", "open, data in hand", "4"),
    ("Spectral augmentation: co-added IMPROVE spectra at Addis-like loading and OC/EC", "never run", "5"),
    ("The exponent test across SPARTAN, with potassium as a second instrument", "first pass here", "6"),
    ("PLS on the masked regions; raw model without 3600–4000 and below 1425 cm⁻¹", "open, lower priority", "7"),
    ("OC/EC-selected cohorts, organic deficit, dust, season-specific MAC, OP = 0, urban sites", "done 28 Oct (HIPS on x)", ""),
    ("Addis quartz + PTFE pairs, with HIPS and some heavy loadings", "blocked: filters not in hand", ""),
], columns=["pathway", "status", "rank"])
text_next = """**Proposed position.** Carry the published FTIR EC forward with no intercept or seasonal correction and a stated level range.
Stop using Fabs / 10 as the Addis yardstick without linearising. The proposed TOR pass band (0.8 to 1.25) sits where the existing
evidence predicts the answer (about 0.67 to 1.0), so ten filters would most likely read "inconclusive": decide before they arrive
whether the deliverable is pass or fail, or an estimated level factor, and ask for HIPS on the same filters with a few loaded to
Addis optical depth. Optical agreement is still not thermal EC."""

intro = f"""Which measurement carries the Addis offset: an answer to the question the 9 Nov status page left open, from data already in
the repository ({counts['filters']} Addis filters of the 7 Nov closeout). A results page, not a recorded meeting. Exploratory: no Addis
filter has thermal (TOR) EC, so the absolute level stays a range. Written account: docs/addis-reference-closure-2026-11-10.md."""

write_bundle(
    MEETING_ID,
    generated_from="gallery/review/closeout_20261107/continuum_baseline.json, filters.json, spartan_reference.json and the unified "
                   "filter dataset via research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py",
    colors=COLORS,
    text={"intro": intro, "linearity": text_linearity, "response": text_response, "errors": text_errors, "season": text_season,
          "level": text_level, "network": text_network, "next": text_next},
    tables={"exponents": exponents, "by_season": by_season, "against_aeth": against_aeth, "against_hips": against_hips, "fits": fits,
            "errors": errors, "season": season_rows[["group", "season", "lot", "n", "published_over_aethalometer",
                                                     "anchored_over_aethalometer", "hips_mac10_over_aethalometer"]],
            "ocec": ocec, "anchors": anchors, "mass": mass, "network": readable[["site", "code", "which", "n", "median_optical_depth",
                                                                              "hips_exponent_vs_ftir_ec", "lo", "hi"]],
            "rereads": rereads, "pathways": pathways},
)
