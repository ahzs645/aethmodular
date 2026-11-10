"""Export the 10 Nov 2026 page "The calibration, step by step" for the gallery's Meeting tab.

    uv run python gallery/data/export_meeting_20261110_process.py

Walks the whole process in order: which IMPROVE filters the calibration is built on, the spectra and baseline,
how the PLS model is chosen (components, cross-validation), how it is scored, what it responds to, the
recalibration, what comes out on filters that were never in training (held-out IMPROVE sites and the Addis
filters), and the baselines. Each step says what was done before, what is done now, and why.

Reads only committed files, so it runs from a plain checkout:
docs/addis-reference-closure-2026-11-10.json (run_addis_reference_closure_20261110.py --summary),
docs/calibration-gain-2026-11-10.json (run_calibration_gain_20261110.py --summary, with the --bundle runs merged),
and the 7 Nov closeout export for the Addis filters. No aethalometer data.

Writes gallery/app/public/data/meetings/process-20261110.json; figures in src/meetings/process-20261110.meeting.ts.
"""

from __future__ import annotations

import json
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
from pls_transfer import deming_regression  # noqa: E402

MEETING_ID = "process-20261110"
LAMBDA = 2.96
C = json.loads((ROOT / "docs/addis-reference-closure-2026-11-10.json").read_text())
G = json.loads((ROOT / "docs/calibration-gain-2026-11-10.json").read_text())
frame = lambda source, key: pd.DataFrame(source[key])  # noqa: E731
table, _closeout = closure.load()
pop = G["population"]

# ---------------------------------------------------------------- 1. the calibration set
by_site = frame(C, "ftir_by_site")
ocec_values = frame(C, "ftir_ocec_values")
comp = frame(C, "ftir_composition").iloc[0]
lots = ", ".join(f"lot {k}: {v:,}" for k, v in sorted(pop["lots"].items()))
population = pd.DataFrame([
    ("IMPROVE filters with a spectrum and thermal (TOR) EC", f"{pop['filters']:,}"), ("IMPROVE sites", f"{pop['sites']}"),
    ("Sampling dates", f"{pop['first_date']} to {pop['last_date']}"), ("PTFE filter lots", lots),
    ("Left out: TOR EC of 150 µg per filter or more", f"{pop['tor_ec_150_or_more']}"), ("Left out: recorded raw-spectrum QC flag", f"{pop['raw_qc_excluded']}"),
    ("Filters at IMPROVE's six urban sites", f"{pop['urban_filters']}"), ("Median TOR EC per filter (µg)", f"{pop['median_tor_ec_ug']:.1f}"),
    ("Median TOR OC/EC", f"{pop['median_tor_ocec']:.1f}"), ("Addis filters (the test set, never used to fit or choose)", f"{len(table)}"),
], columns=["what", "value"])
text_set = f"""**What we chose.** Every IMPROVE filter of lots 248 and 251 that has a spectrum and thermal (TOR) EC: {pop['filters']:,} filters from
{pop['sites']} sites. Nothing is picked to resemble Addis. **Why.** Any subset chosen to look like Addis has to be chosen by something
measured on the filter, and that selects on the measurement's own error (step 4). **Before.** Sets picked to resemble Addis: 906
smoke-flagged filters, the 800 and 450 lowest in OC/EC, the 500 nearest spectral look-alikes, one set per season. Each was judged by how
close it brought the Addis intercept against HIPS to zero. **What it cannot do.** IMPROVE is a mostly rural and remote network. Its
median OC/EC is {pop['median_tor_ocec']:.1f}; even its six urban sites sit near 3, and Addis is near 1. {comp.addis_share_below_improve_p05 * 100:.0f} % of Addis
filters are below IMPROVE's 5th percentile of FTIR OC/EC."""

# ---------------------------------------------------------------- 2. the model: components and cross-validation
gain_k = frame(G, "components")
sel = frame(G, "bundle_selection") if "bundle_selection" in G else pd.DataFrame(columns=["preprocessing", "scheme", "k", "rmse", "median_abs_error", "gain"])
kk = gain_k[(gain_k.preprocessing.eq("Raw spectra")) & (gain_k.variant.eq("As fitted"))].set_index("k").gain
if len(sel):
    at = sel[sel.k.eq(18)].set_index("scheme")
    grouped, inter = at.loc["Whole sites held out (5 folds)"], at.loc["Interleaved (10 folds, the app's protocol)"]
    cv_sentence = (f"At 18 components the interleaved protocol reports a median error of {inter.median_abs_error:.2f} µg per filter and an RMSE of "
                   f"{inter.rmse:.2f}; with whole sites held out the same model gives {grouped.median_abs_error:.2f} and {grouped.rmse:.2f}. "
                   f"The whole-site RMSE is lowest at {int(sel[sel.scheme.eq('Whole sites held out (5 folds)')].set_index('k').rmse.idxmin())} components; "
                   f"the interleaved one keeps falling to {int(sel[sel.scheme.str.startswith('Interleaved')].set_index('k').rmse.idxmin())}. On this broad set the two "
                   "schemes differ by a few per cent; on narrow sets such as the smoke-flagged one the gap was much larger.")
else:
    cv_sentence = "The comparison of the two fold schemes is added when its run is merged."
text_model = f"""**What we chose.** Partial least squares on unscaled spectra, 18 components, cross-validated with **whole sites held out**: five
folds, and no site is ever in training and test together. **Why whole sites.** Addis is a site the calibration has never seen, so the
honest error is the error on unseen sites. **Before.** The calibration app deals filters into ten interleaved folds, so every test filter
has neighbours from its own site in training, and it picks the component count at the first minimum of that error. {cv_sentence}
**Why 18, and fixed.** It is the count that reproduces SPARTAN's published Addis values, and one fixed count keeps every comparison on the
same model. The count matters more than it looks: the calibration's response to thermal EC rises with it, {kk.loc[6]:.2f} at 6,
{kk.loc[12]:.2f} at 12, {kk.loc[18]:.2f} at 18 and {kk.loc[25]:.2f} at 25 components, and cross-validation error on IMPROVE barely sees that."""

# ---------------------------------------------------------------- 3. how it is scored
selection = frame(C, "ftir_selection")
level = frame(C, "ftir_level")
cand = frame(C, "ftir_candidates")
candidates_long = pd.concat([
    cand.assign(chosen="Now: IMPROVE's six urban sites", ftir_over_tor=cand.urban_sites),
    cand.assign(chosen="Before: heavy, low-OC/EC filters chosen by their own TOR", ftir_over_tor=cand.selected_on_own_tor),
])[["calibration", "chosen", "ftir_over_tor"]]
lv = level.set_index("filters")
low = selection[selection.ocec < 2].groupby("chosen_by").ftir_over_tor.first()
text_score = f"""**What we chose.** Score against thermal EC on IMPROVE, on groups chosen by site or by the *previous* sample at the site.
**Why.** A filter picked because its own TOR EC is high, or its own TOR OC/EC is low, is more likely to be one where TOR happened to read
high, so FTIR looks low on it. **Before.** Two yardsticks, both misleading: the Addis intercept against HIPS / 10, which is mostly HIPS
compression (the Nov 10 offset page), and FTIR ÷ TOR on the "heavy, low-OC/EC" IMPROVE subset chosen by each filter's own TOR. **What
changed.** On that subset FTIR ÷ TOR is {lv.loc['TOR EC ≥ 15 µg and OC/EC < 2.5', 'ftir_over_tor']:.2f}; chosen by the previous sample it is
{lv.loc['previous EC ≥ 15 µg and OC/EC < 2.5', 'ftir_over_tor']:.2f}, and on the six urban IMPROVE sites {lv.loc['six urban IMPROVE sites', 'ftir_over_tor']:.2f}. In the
lowest OC/EC bin the same filters read {low["TOR's own OC/EC on this filter"]:.2f} of thermal EC when binned by TOR's own ratio and
{low["FTIR's own OC/EC on this filter"]:.2f} when binned by FTIR's own."""

# ---------------------------------------------------------------- 4. what it responds to, and the recalibration
gv = frame(G, "variants")
gv["step"] = np.where(gv.recalibrated.eq("no"), "as trained", "recalibrated on other sites")
gv["calibration"] = np.where(gv.recalibrated.eq("no"), gv.variant, gv.variant + ", then offset removed and divided by the gain")
g = gv.set_index(["variant", "recalibrated"])
base, fixed = g.loc[("As fitted", "no")], g.loc[("As fitted", "offset removed, divided by the gain")]
weighted, blanks = g.loc[("Weighted toward heavy filters", "no")], g.loc[("Blanks added at zero", "no")]
text_response = f"""**What we found.** Across IMPROVE site means the calibration's prediction is {base.offset:.1f} µg per filter plus
**{base.gain:.2f} × thermal EC** ({base.gain_lo:.2f}–{base.gain_hi:.2f}). It under-responds to EC and makes up the difference with a fixed offset: a
blank filter reads {base.blank_reading:.1f} µg. That is right at IMPROVE's usual loadings and low on heavy filters. **What we tried in
training.** Weighting toward heavy filters leaves the gain at {weighted.gain:.2f} and only raises the offset (a blank then reads
{weighted.blank_reading:.1f} µg, and light filters get worse). Adding blank filters at zero helps the low end (blank {blanks.blank_reading:.1f} µg)
and leaves the gain at {blanks.gain:.2f}. **What works.** Removing the offset and dividing by the gain, with both fitted on *other* sites:
gain {fixed.gain:.2f} ({fixed.gain_lo:.2f}–{fixed.gain_hi:.2f}), the six urban IMPROVE sites from {base.urban_level:.2f} to {fixed.urban_level:.2f}, heavy filters from
{base.heavy_level:.2f} to {fixed.heavy_level:.2f}, light filters from {base.light_level:.2f} to {fixed.light_level:.2f}. **The cost.** The largest errors are scaled
up with everything else (RMSE {base.overall_rmse:.2f} to {fixed.overall_rmse:.2f}); the median error is unchanged."""

# ---------------------------------------------------------------- 5. the output on filters never in training: Addis
refit = frame(C, "addis_refit").drop_duplicates("filter_id").set_index("filter_id")
ceiling = frame(C, "improve_ceiling").set_index("population").loc["all sites", "tau_inf"]
volume = table.volume_m3.reindex(refit.index)
recal = ((refit.ftir_ec_ugm3 * volume - base.offset) / base.gain) / volume
hips = table.hips_bc.reindex(refit.index)
linear = (rc.linearise(table.tau, ceiling) * 100.0 * closure.DEPOSIT_AREA_CM2 / table.volume_m3 / closure.MAC_VALUE).reindex(refit.index)
outputs = {"SPARTAN published": table[closure.PUBLISHED].reindex(refit.index), "Raw spectra, as trained": refit.ftir_ec_ugm3,
           "Raw spectra, recalibrated on IMPROVE": recal}
points, fits = [], []
for name, y in outputs.items():
    for comparator, x in (("HIPS as measured (Fabs / 10)", hips), ("HIPS linearised (/ 10)", linear)):
        pair = pd.DataFrame({"filter": refit.index, "season": refit.season.to_numpy(), "calibration": name, "comparator": comparator,
                             "x": x.to_numpy(), "ftir_ec": y.to_numpy()}).replace([np.inf, -np.inf], np.nan).dropna()
        d = deming_regression(pair.x, pair.ftir_ec, LAMBDA)
        points.append(pair)
        fits.append({"calibration": name, "comparator": comparator, "n": d["n"], "slope": d["slope"], "intercept": d["intercept"],
                     "median_ec": float(pair.ftir_ec.median())})
addis_points, addis_fits = pd.concat(points, ignore_index=True), pd.DataFrame(fits)
f = addis_fits.set_index(["calibration", "comparator"])
factor_raw = float((recal / refit.ftir_ec_ugm3).median())
factor_pub = float((recal / outputs["SPARTAN published"]).median())
text_addis = f"""**What comes out at Addis.** The Addis filters were never used to fit or choose anything. The recalibration learned on
IMPROVE raises the raw calibration by × {factor_raw:.2f} at the median filter, which is × {factor_pub:.2f} on SPARTAN's published values (median
{f.loc[('SPARTAN published', 'HIPS linearised (/ 10)'), 'median_ec']:.2f} to {f.loc[('Raw spectra, recalibrated on IMPROVE', 'HIPS linearised (/ 10)'), 'median_ec']:.2f} µg/m³).
Against HIPS as measured every version has a large negative intercept ({f.loc[('SPARTAN published', 'HIPS as measured (Fabs / 10)'), 'intercept']:.1f} for the published
product), which is HIPS compression. Against linearised HIPS the published product reads
{f.loc[('SPARTAN published', 'HIPS linearised (/ 10)'), 'slope']:.2f}x {f.loc[('SPARTAN published', 'HIPS linearised (/ 10)'), 'intercept']:+.2f} and the recalibrated one
{f.loc[('Raw spectra, recalibrated on IMPROVE', 'HIPS linearised (/ 10)'), 'slope']:.2f}x {f.loc[('Raw spectra, recalibrated on IMPROVE', 'HIPS linearised (/ 10)'), 'intercept']:+.2f}.
**What this is not.** No Addis filter has thermal EC. The recalibration corrects the loading part of the bias, checked on IMPROVE sites it
had not seen; it says nothing about whether Addis's composition adds to it. HIPS is optical, and its MAC at Addis is not known."""

# ---------------------------------------------------------------- 6. baselines
bv = frame(G, "bundle_variants")
bv = bv[bv.variant.eq("As fitted")].copy()
bv["step"] = np.where(bv.recalibrated.eq("no"), "as trained", "recalibrated on other sites")
order = ["Raw spectra", "Spline baseline", "Production VIBES", "PTFE-anchored VIBES", "Deflated VIBES"]
bv["preprocessing"] = pd.Categorical(bv.preprocessing, order)
baselines = bv.sort_values(["preprocessing", "step"]).astype({"preprocessing": str})[
    ["preprocessing", "step", "gain", "gain_lo", "gain_hi", "offset", "heavy_level", "urban_level", "light_level", "light_median_abs_error",
     "overall_median_abs_error", "overall_rmse"]]
baseline_k = frame(G, "bundle_components")
baseline_k = baseline_k[baseline_k.variant.eq("As fitted")][["preprocessing", "k", "gain"]]
cs = frame(C, "candidates_by_season")
names = {"Raw spectra": "Raw spectra", "Spline baseline": "Spline baseline", "Production VIBES": "Production VIBES",
         "PTFE-anchored VIBES": "PTFE-anchored VIBES", "Deflated VIBES": "Deflated VIBES"}
baseline_addis = cs[cs.season.eq("All") & cs.calibration.isin(names)][["calibration", "comparator", "n", "slope", "intercept", "median_ratio"]]
b = baselines[baselines.step.eq("as trained")].set_index("preprocessing")
r = baselines[baselines.step.eq("recalibrated on other sites")].set_index("preprocessing")
ba = baseline_addis.set_index(["calibration", "comparator"])
text_baselines = f"""**What a baseline is for.** A PTFE filter has its own broad spectrum under the aerosol's. Raw spectra leave it in; the spline
(AIRSpec) baseline fits and removes a smooth curve above 1425 cm⁻¹; VIBES models the filter from a bank of blanks, and the PTFE-anchored
version protects the broad continuum that EC itself produces. **The test.** The same {pop['filters']:,} IMPROVE filters, 18 components, whole
sites held out, for each. **What it shows.** No baseline changes the calibration's response to thermal EC: {b.loc['Raw spectra', 'gain']:.2f} raw,
{b.loc['Spline baseline', 'gain']:.2f} spline, {b.loc['Production VIBES', 'gain']:.2f} production VIBES, {b.loc['PTFE-anchored VIBES', 'gain']:.2f} PTFE-anchored VIBES.
Deflated VIBES, which removes smooth directions, lowers it to {b.loc['Deflated VIBES', 'gain']:.2f}. The six urban IMPROVE sites read
{b.loc['Raw spectra', 'urban_level']:.2f}, {b.loc['Spline baseline', 'urban_level']:.2f}, {b.loc['Production VIBES', 'urban_level']:.2f} and {b.loc['PTFE-anchored VIBES', 'urban_level']:.2f}
of thermal EC as trained, and {r.loc['Raw spectra', 'urban_level']:.2f}, {r.loc['Spline baseline', 'urban_level']:.2f}, {r.loc['Production VIBES', 'urban_level']:.2f} and
{r.loc['PTFE-anchored VIBES', 'urban_level']:.2f} after the same recalibration. Baselines differ a little in median error
({b.overall_median_abs_error.min():.2f} to {b.overall_median_abs_error.max():.2f} µg per filter) and not in the thing that limits Addis.
**Before.** The spline baseline looked like the largest single improvement because it moved the Addis intercept against HIPS from
{ba.loc[('Raw spectra', 'HIPS as measured'), 'intercept']:.1f} to {ba.loc[('Spline baseline', 'HIPS as measured'), 'intercept']:.1f} µg/m³. Against linearised HIPS the two
intercepts are {ba.loc[('Raw spectra', 'HIPS linearised'), 'intercept']:+.2f} and {ba.loc[('Spline baseline', 'HIPS linearised'), 'intercept']:+.2f}: the difference was HIPS
compression meeting a calibration that reads Addis at {ba.loc[('Spline baseline', 'HIPS as measured'), 'median_ratio']:.2f} of HIPS / 10
(raw: {ba.loc[('Raw spectra', 'HIPS as measured'), 'median_ratio']:.2f})."""

# ---------------------------------------------------------------- 0. the choices at a glance
choices = pd.DataFrame([
    ("1. Calibration set", "Filters picked to resemble Addis (smoke, lowest OC/EC, spectral look-alikes, per season)", f"All {pop['filters']:,} IMPROVE filters with thermal EC, {pop['sites']} sites",
     "Picking by a filter's own measurements selects on their error"),
    ("2. Spectra and baseline", "Chosen by the Addis intercept against HIPS", "Raw spectra; baselines tested side by side on thermal EC", "No baseline changes the response to EC (step 6)"),
    ("3. Model", "Component count at the first minimum of interleaved cross-validation", "18 components, fixed; whole sites held out", "Addis is an unseen site; one count keeps comparisons fair"),
    ("4. Scoring", "Addis intercept against HIPS / 10; FTIR ÷ TOR on filters chosen by their own TOR", "FTIR ÷ TOR on groups chosen by site or by the previous sample", "HIPS is compressed at Addis; own-TOR groups read low by construction"),
    ("5. Correction", "A different calibration set per season; flat offsets", "Offset removed and divided by the gain, fitted on other IMPROVE sites", f"Urban IMPROVE sites {base.urban_level:.2f} → {fixed.urban_level:.2f}; light filters unharmed"),
    ("6. Addis output", "Judged by agreement with HIPS / 10", f"Published × {factor_pub:.2f}, stated as a range (× 1.0 to 1.4)", "No Addis thermal EC; composition is outside IMPROVE"),
], columns=["step", "before", "now", "why"])
intro = f"""The calibration, step by step: which filters it is built on, how the model is chosen and scored, what it responds to, how it is
corrected, and what comes out on filters that were never in training. Each step says what was done before, what is done now, and why,
with the chart behind it. {pop['filters']:,} IMPROVE filters with thermal (TOR) EC; {len(table)} Addis filters as the test set. No Addis filter
has thermal EC, so the Addis level stays a range, and no aethalometer data are used."""

COLORS = {"Bega": "#E67E22", "Belg": "#27AE60", "Kiremt": "#3498DB", "urban IMPROVE site": "#B23327", "other IMPROVE site": "#8a8f98",
          "IMPROVE (FTIR EC ≥ 5 µg per filter)": "#8a8f98", "Addis Ababa": "#C49442", "as trained": "#22252A", "recalibrated on other sites": "#2C6E9E",
          "Whole sites held out (5 folds)": "#22252A", "Interleaved (10 folds, the app's protocol)": "#B23327",
          "TOR's own OC/EC on this filter": "#B23327", "FTIR's own OC/EC on this filter": "#7A4FA3", "TOR OC/EC of the previous sample at the site": "#2C6E9E",
          "this filter's own TOR (selects on its error)": "#B23327", "the previous sample at the site": "#2C6E9E", "the site": "#22252A",
          "Now: IMPROVE's six urban sites": "#22252A", "Before: heavy, low-OC/EC filters chosen by their own TOR": "#B23327",
          "As fitted": "#22252A", "Weighted toward heavy filters": "#B23327", "Blanks added at zero": "#2C6E9E", "Weighted, and blanks at zero": "#7A4FA3",
          "Raw spectra": "#22252A", "Spline baseline": "#2C6E9E", "Production VIBES": "#d95f0e", "PTFE-anchored VIBES": "#C49442", "Deflated VIBES": "#8F8C84",
          "HIPS as measured": "#B23327", "HIPS linearised": "#2C6E9E"}
write_bundle(
    MEETING_ID,
    generated_from="docs/addis-reference-closure-2026-11-10.json, docs/calibration-gain-2026-11-10.json and the 7 Nov closeout export "
                   "via gallery/data/export_meeting_20261110_process.py",
    colors=COLORS,
    text={"intro": intro, "set": text_set, "model": text_model, "score": text_score, "response": text_response, "addis": text_addis,
          "baselines": text_baselines},
    tables={"choices": choices, "population": population, "by_site": by_site[by_site.median_tor_ec_ug >= 1.5], "ocec_values": ocec_values,
            "selection_cv": sel, "gain_k": gain_k[gain_k.preprocessing.eq("Raw spectra")][["variant", "k", "gain"]],
            "selection_bias": selection, "level": level, "candidates": candidates_long,
            "variants": gv[["calibration", "variant", "step", "gain", "gain_lo", "gain_hi", "offset", "blank_reading", "heavy_level", "urban_level",
                            "light_level", "light_median_abs_error", "overall_median_abs_error", "overall_rmse"]],
            "addis_points": addis_points, "addis_fits": addis_fits, "baselines": baselines, "baseline_k": baseline_k, "baseline_addis": baseline_addis},
)
