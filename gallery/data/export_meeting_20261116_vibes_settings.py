"""Export the 16 Nov 2026 page "Which VIBES setting: solver, rank and tau" for the gallery's Meeting tab.

    uv run python gallery/data/export_meeting_20261116_vibes_settings.py

The investigation step by step: the solver (October run against converged), the rank, tau with the frozen
calibration, tau with the calibration rebuilt, the rule that picks the number of PLS components, one QC-failed
training spectrum, and the choices that follow. Nothing is fitted here. It reads the saved outputs of

    research/ftir_hips_chem/workflows/run_vibes_rank_tau_ec_20261116.py        (network scans at 12 settings)
    research/ftir_hips_chem/workflows/run_vibes_tau_calibration_20261116.py    (IMPROVE calibration at each tau,
                                                                               and the k 4 model without one spectrum)
    research/ftir_hips_chem/workflows/vibes_newton_refit_20261103.py           (the converged v2 run, 3 Nov transfer)
    docs/vibes-rank-check-2026-11-10.json                                      (band injection by rank)

EC is µg per filter, read with the frozen k 4 IMPROVE model unless a table says the calibration was rebuilt.

Writes gallery/app/public/data/meetings/vibes-settings-20261116.json; figures in
src/meetings/vibes-settings-20261116.meeting.ts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from meeting_bundle import write_bundle  # noqa: E402

MEETING_ID = "vibes-settings-20261116"
TABLES = ROOT / "research/ftir_hips_chem/output/tables"
SETTINGS, CALIBRATION, V2 = TABLES / "vibes_rank_tau_ec_20261116", TABLES / "vibes_tau_calibration_20261116", TABLES / "network_baselines_v2_20261103"
REFERENCE = (70, 0.1)
tau_name = lambda t: f"tau {float(t):g}"  # noqa: E731
rank_name = lambda r: f"Rank {int(r)}"  # noqa: E731
COLORS = {"tau 0.05": "#4c78a8", "tau 0.1": "#54a24b", "tau 0.2": "#f58518",
          "Rank 70": "#4c78a8", "Rank 40": "#72b7b2", "Rank 20": "#f58518", "Rank 10": "#e45756",
          "Addis": "#e45756", "Other sites": "#4c78a8",
          "Lower without the spectrum": "#4c78a8", "Higher without the spectrum": "#f58518",
          "SPARTAN PM2.5 scans": "#4c78a8", "IMPROVE held-out filters": "#f58518",
          # the six training sets: one ramp from every spectrum down to the lightest, and the distance trim apart
          "All QC-passed": "#1f3b63", "EC below 150 µg": "#2f6aa8", "EC below 100 µg": "#4f9bc9", "EC below 50 µg": "#f58518",
          "EC below 25 µg": "#e45756", "Nearest 99 % of spectra": "#7f7f7f"}
f1, f2, f3 = (lambda v: f"{v:.1f}"), (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}")  # noqa: E731

# ---------------------------------------------------------------- step 1: the solver
diagnostics = pd.read_csv(V2 / "vibes_fit_diagnostics.csv")
early_by_type = diagnostics[diagnostics.premature_stop].filter_type.value_counts()
d = diagnostics[diagnostics.vibes_model.eq("b87") & diagnostics.filter_type.eq("PM2.5")].copy()
d["october"], d["converged"] = d.ec_saved_baseline_frozen_model_ug, d.ec_newton_baseline_frozen_model_ug
d["change"] = d.converged - d.october
d["change_pct"] = 100 * d.change / d.october.where(d.october >= 1)
masks = [("All PM2.5 scans", d.october.notna()), ("Not an early stop", ~d.premature_stop), ("Early stop in October", d.premature_stop),
         ("Addis early stops", d.premature_stop & d.code.eq("ETAD"))]
solver_groups = pd.DataFrame([{
    "group": name, "scans": int(m.sum()), "median_abs_ug": d.change[m].abs().median(), "median_abs_pct": d.change_pct[m].abs().median(),
    "largest_ug": d.change[m].abs().max(), "down": int((d.change[m] < 0).sum()), "up": int((d.change[m] > 0).sum())} for name, m in masks])
early = d[d.premature_stop]
solver_scans = pd.DataFrame({"filter": early.filter_key, "analysis": early.analysis_id.astype(str), "site": early.code,
                             "group": np.where(early.code.eq("ETAD"), "Addis", "Other sites"), "october": early.october, "converged": early.converged,
                             "change (%)": early.change_pct.round(1), "October steps": early.saved_iterations})
example = d[d.filter_key.eq("CHTS-0830")]
solver_example = pd.DataFrame({"scan": "Analysis " + example.analysis_id.astype(str), "early_stop": np.where(example.premature_stop, "yes", "no"),
                               "october_steps": example.saved_iterations, "october": example.october, "converged": example.converged, "change_pct": example.change_pct})
sg = solver_groups.set_index("group")

# ---------------------------------------------------------------- steps 2 and 3: rank and tau with the frozen calibration
scans = pd.read_csv(SETTINGS / "scans.csv", dtype={"analysis_id": str})
fits = {}
for path in sorted(SETTINGS.glob("fit_rank*_tau*.npz")):
    z = np.load(path)
    fits[(int(z["rank"]), float(z["tau"]))] = {k: z[k] for k in z.files}
ref = fits[REFERENCE]
pm = scans.filter_type.eq("PM2.5").to_numpy()
blank = scans.filter_type.eq("FB").to_numpy()
addis = pm & scans.code.eq("ETAD").to_numpy()
# repeat scans of one filter, without the filters the lab review marks as having one weak scan
reviewed = set(re.findall(r"'([A-Z]{4}-\d{4}-\d)':", (ROOT / "gallery/app/src/lib/nativeScanReview.ts").read_text()))
kept = scans[pm & ~scans.filter_id.isin(reviewed)]
pairs = np.array([v[:2] for v in kept.groupby("filter_id").indices.values() if len(v) >= 2])
pairs = kept.index.to_numpy()[pairs]


def setting(key):
    f = fits[key]
    m = pm & f["ok"] & ref["ok"]
    base, change = ref["ec"][m], f["ec"][m] - ref["ec"][m]
    loaded = base >= 1
    relative = np.abs(change[loaded]) / base[loaded]
    a = addis & f["ok"] & ref["ok"]
    gap = np.abs(f["ec"][pairs[:, 0]] - f["ec"][pairs[:, 1]])
    blanks_ec = f["ec"][blank & f["ok"]]
    return {"rank": rank_name(key[0]), "tau": tau_name(key[1]), "rank_n": key[0], "tau_n": key[1], "scans": int(m.sum()),
            "median_abs_pct": 100 * np.median(relative), "moved_gt10_pct": 100 * np.mean(relative > 0.1),
            "median_abs_ug": np.median(np.abs(change)), "largest_ug": np.abs(change).max(),
            "slope": np.polyfit(base, f["ec"][m], 1)[0] if key != REFERENCE else 1.0,
            "addis_median_change_ug": np.median(f["ec"][a] - ref["ec"][a]), "addis_slope": np.polyfit(ref["ec"][a], f["ec"][a], 1)[0] if key != REFERENCE else 1.0,
            "blank_iqr_ug": float(np.subtract(*np.quantile(blanks_ec, [0.75, 0.25]))), "repeat_gap_ug": np.median(gap), "repeat_gap_p90_ug": np.quantile(gap, 0.9),
            "noise_e4": 1e4 * np.median(f["sigma"][pm & f["ok"]]), "featureless_e3": 1e3 * np.median(f["mean featureless 2000-2500"][pm & f["ok"]])}


settings = pd.DataFrame([setting(k) for k in sorted(fits, key=lambda k: (-k[0], k[1]))])
at = settings.set_index(["rank_n", "tau_n"])
by_rank = settings[settings.tau_n.eq(0.1)].copy()
by_tau = settings[settings.rank_n.eq(70)].copy()
certified = int(min(f["ok"].sum() for f in fits.values()))

injection = pd.DataFrame(json.loads((ROOT / "docs/vibes-rank-check-2026-11-10.json").read_text())["inject"]["table"])
injection = injection.assign(rank=injection["rank"].map(rank_name), height=injection.height.map(lambda h: f"{h:g}")).sort_values(["group", "rank", "fwhm"])
injection = injection[["group", "rank", "fwhm", "height", "median", "p10", "p90", "n"]]

# ---------------------------------------------------------------- step 4: tau with the calibration rebuilt
TRAINING = {"all": "Whole train split (the frozen procedure)", "qc": "Without the QC-failed spectrum"}
scores = pd.read_csv(CALIBRATION / "calibration_scores.csv")
heldout = scores.sort_values(["training", "tau", "k"]).assign(tau=lambda f: f.tau.map(tau_name), training=lambda f: f.training.map(TRAINING)).rename(
    columns={"median |error|": "median_abs"})[["tau", "training", "k", "RMSE", "MAE", "median_abs", "bias", "selected"]]
qc = scores[scores.training.eq("qc")].set_index(["tau", "k"])
whole = scores[scores.training.eq("all")].set_index(["tau", "k"])
heldout_fixed = scores[scores.training.eq("qc") & scores.k.isin([4, 8, 11, 18])].pivot(index="k", columns="tau", values="RMSE").reset_index()
heldout_fixed.columns = ["k"] + [tau_name(t) for t in heldout_fixed.columns[1:]]

boot = pd.read_csv(CALIBRATION / "bootstrap.csv")
boot["reading"] = np.where(boot.hi < 0, "smaller error than tau 0.1", np.where(boot.lo > 0, "larger error than tau 0.1", "no different"))
boot = boot.assign(tau=boot.tau.map(tau_name), row="k " + boot.k.astype(str))[["score", "row", "k", "tau", "difference", "lo", "hi", "reading"]]
worse = lambda t: int((boot.tau.eq(tau_name(t)) & boot.reading.str.startswith("larger")).sum())  # noqa: E731
better = lambda t: int((boot.tau.eq(tau_name(t)) & boot.reading.str.startswith("smaller")).sum())  # noqa: E731
comparisons = int(boot.tau.eq("tau 0.05").sum())


def held_ks(tau, reading):
    """The k values at which a tau's held-out error differs from tau 0.1 in one direction, written as a phrase."""
    ks = sorted(set(boot[boot.tau.eq(tau_name(tau)) & boot.score.str.startswith("Held-out") & boot.reading.str.startswith(reading)].k))
    return " and ".join(str(k) for k in ks)


def held_reading(tau):
    """One tau's held-out comparison with tau 0.1 in words."""
    smaller, larger = held_ks(tau, "smaller"), held_ks(tau, "larger")
    if smaller and larger:
        return f"{tau_name(tau)} is better at k {smaller} and worse at k {larger}"
    return f"{tau_name(tau)} is " + (f"better at k {smaller}" if smaller else f"never better, and worse at k {larger}" if larger else "no different at any k")

transfer = pd.read_csv(CALIBRATION / "transfer_scores.csv").rename(columns={
    "IMPROVE out-of-site median |error| (µg)": "oos_median_abs", "Addis slope vs HIPS": "addis_slope",
    "Addis intercept vs HIPS (µg)": "addis_intercept", "Addis R2": "addis_r2"})
tr = transfer.set_index(["tau", "k"])
transfer = transfer.sort_values(["tau", "k"]).assign(tau=lambda f: f.tau.map(tau_name))[["tau", "k", "oos_median_abs", "addis_slope", "addis_intercept", "addis_r2"]]

# ---------------------------------------------------------------- step 5: the rule that picks k
curves = pd.read_csv(CALIBRATION / "calibration_cv_curves.csv")
cv = curves.pivot_table(index=["tau", "n_components"], columns="training", values="rmse_mean").reset_index().rename(
    columns={"n_components": "k", "all": "rmse_whole", "qc": "rmse_qc"})
cv = cv[cv.k <= 25].assign(tau=cv.tau.map(tau_name))
picked = scores[scores.selected].pivot(index="training", columns="tau", values="k")
rule = pd.DataFrame([{"training": TRAINING[t], **{tau_name(tau): int(picked.loc[t, tau]) for tau in (0.05, 0.1, 0.2)},
                      **{f"RMSE at that k, {tau_name(tau)}": float((whole if t == "all" else qc).loc[(tau, int(picked.loc[t, tau])), "RMSE"]) for tau in (0.05, 0.1, 0.2)}}
                     for t in ("all", "qc")])

# ---------------------------------------------------------------- step 6: one QC-failed training spectrum
old = pd.read_csv(TABLES / "vibes_newton_20261103/transfer/ratios_by_loading.csv")
old = old[old.model.eq("vibes_newton") & old.what.str.contains("OLS|out-of-site")].set_index(["what", "k"])["median"]
if not old.index.is_unique:
    raise ValueError("3 Nov transfer summary: more than one row per score and k")
qc_effect = pd.DataFrame([{"k": k,
                           "slope_with": old[("Addis OLS slope vs HIPS", k)], "slope_without": tr.loc[(0.1, k), "addis_slope"],
                           "intercept_with": old[("Addis OLS intercept vs HIPS (µg)", k)], "intercept_without": tr.loc[(0.1, k), "addis_intercept"],
                           "error_with": old[("IMPROVE out-of-site MAE (µg)", k)], "error_without": tr.loc[(0.1, k), "oos_median_abs"]} for k in (5, 8, 12, 18)])
rows_ = pd.read_csv(CALIBRATION / "improve_rows.csv")
failed = rows_[~rows_.raw_qc_ok]
frozen_with, frozen_without = whole.loc[(0.1, 4), "RMSE"], qc.loc[(0.1, 4), "RMSE"]
# the k 4 model with and without that spectrum, on every network scan (qc_spectrum_effect in the tau calibration workflow)
Q = lambda name: pd.read_csv(CALIBRATION / f"qc_spectrum_{name}.csv")  # noqa: E731
qc_groups, qc_sites, qc_leverage, qc_k = Q("groups"), Q("sites"), Q("leverage"), Q("by_k")
qc_sites["direction"] = np.where(qc_sites.change_pct < 0, "Lower without the spectrum", "Higher without the spectrum")
qc_leverage["filter"] = qc_leverage["filter"].astype(str)
c = Q("checks").set_index("model")
qc_checks = pd.DataFrame([(label, c.loc["With the spectrum", key], c.loc["Without it", key], verdict) for label, key, verdict in (
    ("IMPROVE held-out RMSE (µg)", "heldout_rmse", "the same"),
    ("IMPROVE held-out median |error| (µg)", "heldout_median_abs", "slightly lower without"),
    ("Addis slope of FTIR EC on HIPS BC", "addis_slope", "the same"),
    ("Addis R² against HIPS", "addis_r2", "slightly looser without"),
    ("Field blanks, median EC (µg)", "blank_median", "closer to zero without"),
    ("Repeat scans of one filter, median gap (µg)", "repeat_gap", "larger without"),
    ("PM2.5 scans reading below zero (%)", "negative_share_pct", "slightly more without"),
)], columns=["check", "with", "without", "reading"])
qc_by_k = pd.concat([
    pd.DataFrame({"k": qc_k.k, "scans": "SPARTAN PM2.5 scans", "median_abs_pct": qc_k.network_median_abs_pct}),
    pd.DataFrame({"k": qc_k.k, "scans": "IMPROVE held-out filters", "median_abs_pct": qc_k.improve_test_median_abs_pct})])
qg = qc_groups.set_index("group")

# ---------------------------------------------------------------- step 7: training without the heaviest filters
domain = pd.read_csv(CALIBRATION / "domain_policies.csv")
domain["k"] = "k " + domain.k.astype(str)
domain_pairs = pd.read_csv(CALIBRATION / "domain_pairs.csv")
domain_pairs = domain_pairs[domain_pairs.k.eq(4)].drop(columns="k")
domain_sites = pd.read_csv(CALIBRATION / "domain_sites.csv")
dk4 = domain[domain.k.eq("k 4")].set_index("training")
everything, lightest = dk4.loc["All QC-passed"], dk4.loc["EC below 25 µg"]
clean_ec = pd.read_csv(CALIBRATION / "qc_spectrum_scans.csv").query("filter_type == 'PM2.5'").ec_without
between = domain_pairs[domain_pairs.training.ne(domain_pairs.against)].median_abs_pct
addis_site = domain_sites[domain_sites.site.eq("ETAD")].set_index("training").median_ec
lightest_by_k = domain[domain.training.eq("EC below 25 µg")].set_index("k")
# the Addis line against HIPS at every k for each training set, with the 5th-95th percentile over resampled IMPROVE sites
domain_addis = pd.read_csv(CALIBRATION / "domain_addis_by_k.csv")
if not np.allclose(domain_addis.set_index(["training", "k"]).addis_slope.reindex(list(zip(domain.training, domain.k.str[2:].astype(int)))), domain.addis_slope, atol=1e-9):
    raise ValueError("domain_addis_by_k.csv and domain_policies.csv disagree on the Addis slope")
slope_k = domain_addis.pivot(index="k", columns="training", values="addis_slope")
ratio_light = slope_k["EC below 25 µg"] / slope_k["All QC-passed"]
crosses = {name: (int(slope_k.index[(slope_k[name] >= 1).to_numpy().argmax()]) if (slope_k[name] >= 1).any() else None) for name in slope_k}
da = domain_addis.set_index(["training", "k"])
far = qc_leverage.iloc[0]
others = qc_leverage[qc_leverage.raw_qc.eq("passed") & qc_leverage.distance.notna()]
random_median = float(qc_leverage.loc[qc_leverage["filter"].str.contains("median of"), "network_median_abs_pct"].iloc[0])
at_k = qc_k.set_index("k").network_median_abs_pct

# ---------------------------------------------------------------- the words
r40, r20, r10 = at.loc[(40, 0.1)], at.loc[(20, 0.1)], at.loc[(10, 0.1)]
t05, t20 = at.loc[(70, 0.05)], at.loc[(70, 0.2)]
k4, k11 = qc.xs(4, level="k").RMSE, qc.xs(11, level="k").RMSE
gap = max(float(qc.xs(k, level="k").RMSE.max() - qc.xs(k, level="k").RMSE.min()) for k in (4, 8, 11, 18))
slope4, slope18 = tr.xs(4, level="k").addis_slope, tr.xs(18, level="k").addis_slope
ex = solver_example.set_index("scan")
picked_all = ", ".join(str(int(picked.loc["all", t])) for t in (0.1, 0.05, 0.2))
picked_qc = ", ".join(str(int(picked.loc["qc", t])) for t in (0.1, 0.05, 0.2))

terms = pd.DataFrame([
    ("VIBES", "The baseline method. It fits the part of a spectrum that a blank filter explains and subtracts it; what is left is the corrected spectrum.", "87-blank background"),
    ("Noise level", "How closely VIBES lets the baseline follow one spectrum. It is fitted for every scan before the baseline.", "fitted per scan"),
    ("Solver", "The routine that fits the noise level. The October run used one that sometimes stopped early; the converged solver runs until the fit cannot improve.", "Converged"),
    ("Rank", "How many patterns of the blank background the baseline may use, largest first.", "70"),
    ("tau", "The quantile the baseline aims for under the spectrum. A lower tau puts the baseline lower.", "0.1"),
    ("k", "The number of PLS components in the EC calibration, which reads EC from the corrected spectrum.", "4 (frozen model)"),
    ("Frozen model", "The IMPROVE EC calibration at k 4, trained on spectra corrected at rank 70 and tau 0.1. EC from it is only valid at those settings.", "k 4"),
], columns=["term", "meaning", "value"])

glance = pd.DataFrame([
    ("1. Solver", "Does solving to convergence change EC?",
     f"October run against converged on {int(sg.loc['All PM2.5 scans', 'scans']):,} PM2.5 scans",
     f"Unchanged on {int(sg.loc['Not an early stop', 'scans']):,} scans. The {int(sg.loc['Early stop in October', 'scans'])} early stops move by a median {f1(sg.loc['Early stop in October', 'median_abs_pct'])} %, Addis by {f1(sg.loc['Addis early stops', 'median_abs_pct'])} %",
     "Converged for new work"),
    ("2. Rank", "Does the number of background components matter?",
     f"Every scan refitted at ranks 10, 20, 40 and 70 ({certified:,} of {len(scans):,} certified at every setting)",
     f"Rank 40 gives the same EC as rank 70 (median {f2(r40.median_abs_pct)} %). Rank 20 moves it {f1(r20.median_abs_pct)} %, rank 10 {f1(r10.median_abs_pct)} %",
     "Rank 70"),
    ("3. tau, frozen model", "What happens if tau changes and the calibration does not?",
     "Every scan refitted at tau 0.05, 0.1 and 0.2, EC read with the frozen model",
     f"EC moves by a median {f1(t05.median_abs_pct)} % at tau 0.05 and {f1(t20.median_abs_pct)} % at tau 0.2",
     "Do not mix: tau must match the calibration"),
    ("4. tau, calibration rebuilt", "Is another tau more accurate?",
     f"IMPROVE calibration rebuilt at each tau ({len(rows_):,} spectra), scored on held-out filters at fixed k",
     f"No tau is better at every k: {held_reading(0.05)}; {held_reading(0.2)}",
     "tau 0.1"),
    ("5. The k rule", "Does the rule that picks k give a stable answer?",
     "The frozen selection rule run at each tau, with and without one bad spectrum",
     f"It picks k {picked_all} at tau 0.1, 0.05, 0.2, and {picked_qc} without the bad spectrum",
     "Compare settings at a fixed k"),
    ("6. One bad spectrum", "Does a QC-failed training spectrum matter?",
     f"The k 4 model fitted with and without IMPROVE filter {failed.filter_id.iloc[0]} ({failed.site.iloc[0]}), both applied to every network scan",
     f"IMPROVE scores stay the same, but SPARTAN EC from the k 4 model moves by a median {f1(qg.loc['PM2.5 scans', 'median_abs_pct'])} %, and site medians by {qc_sites.change_pct.min():.0f} % to +{qc_sites.change_pct.max():.0f} %",
     "Leave it out; EC at k 4 is not pinned down by IMPROVE scores"),
    ("7. Lighter training set", "Is a model trained without the heaviest filters stable?",
     "Six training sets, from every QC-passed spectrum down to thermal EC below 25 µg; each refitted with one spectrum dropped and with IMPROVE sites resampled",
     f"One spectrum stops mattering ({f1(everything.one_spectrum_max_pct)} % to {f1(lightest.one_spectrum_max_pct)} %), but EC still spreads {f1(lightest.scan_spread_pct)} % when sites are resampled, and the cut-off itself moves EC by {between.min():.0f} to {between.max():.0f} %. "
     f"The Addis slope against HIPS rises with k under every training set, and below 25 µg it is about half the slope at every k",
     "Steadier, not pinned down; the cut-off is a choice, and so is k"),
    ("8. Choices", "What do we do?", "The seven steps together",
     "The solver, rank and tau are settled. What moves EC is the calibration: its k and its training spectra",
     "See the last table"),
], columns=["step", "question", "did", "found", "choice"])

choices = pd.DataFrame([
    ("Solver", "Converged", f"Same objective, never a worse fit, all scans fit. EC is unchanged except on the {int(sg.loc['Early stop in October', 'scans'])} early stops", "Recommended; the switch of the gallery's default run waits for Ann and Satoshi"),
    ("Rank", "70", f"Rank 40 gives the same EC, so nothing is gained by lowering it, and field blanks are tightest at 70 (spread {f3(at.loc[(70, 0.1), 'blank_iqr_ug'])} µg against {f3(r20.blank_iqr_ug)} at rank 20)", "Settled"),
    ("tau", "0.1", f"No tau is better at every k once the calibration is rebuilt; the largest gap between tau values at one k (4, 8, 11, 18) is {f2(gap)} µg", "Settled"),
    ("Presets in the browser tool", "None", "It opens on rank 70, tau 0.1, converged. No other setting has evidence behind it", "Settled"),
    ("Comparing settings", "At a fixed k", f"The selection rule picks a different k when tau changes ({picked_all}), so a comparison at the rule's k compares k, not the setting", "Settled"),
    ("QC-failed training spectrum", "Leave it out", f"It is {far.distance:.0f} times farther from the mean spectrum than a typical train spectrum. IMPROVE held-out RMSE is {f3(frozen_with)} with it and {f3(frozen_without)} without, but SPARTAN PM2.5 EC moves by a median {f1(qg.loc['PM2.5 scans', 'median_abs_pct'])} % and field blanks read {f2(qg.loc['Field blanks', 'median_without'])} µg instead of {f2(qg.loc['Field blanks', 'median_with'])}", "Recommended for new calibrations. Refreezing the k 4 model changes network EC, so it needs Ann and Satoshi"),
    ("EC from the k 4 model", "Treat as not pinned down", f"Two k 4 models that IMPROVE cannot tell apart differ by a median {f1(qg.loc['PM2.5 scans', 'median_abs_pct'])} % on SPARTAN. With IMPROVE sites resampled, a scan's EC spreads by {f1(everything.scan_spread_pct)} % and a site's median by {f1(everything.site_spread_median_pct)} % (PRFJ {everything.site_spread_max_pct:.0f} %)", "Open; report EC with this spread"),
    ("Training set", "Not chosen here", f"No network scan reads above 50 µg and {100 * (clean_ec > 25).mean():.1f} % read above 25 µg, so the heaviest training filters lie outside SPARTAN's range. Training below 25 µg removes the pull of single spectra and lowers held-out RMSE on light filters from {f2(everything.heldout_rmse_below25)} to {f2(lightest.heldout_rmse_below25)} µg, but it moves network EC by {f1(lightest.change_from_all_pct)} %, Addis's median from {f1(addis_site['All QC-passed'])} to {f1(addis_site['EC below 25 µg'])} µg, and the Addis slope against HIPS to about half at every k. Nothing here says the lower reading is more correct", "Open; the frozen model stays as it is until Ann and Satoshi decide"),
    ("Number of PLS components (k)", "Not chosen here", f"Held-out RMSE falls from {f2(k4.mean())} µg at k 4 to {f2(k11.mean())} at k 11, but the Addis slope against HIPS rises from {f2(slope4.mean())} to {f2(slope18.mean())} at k 18. IMPROVE accuracy and the Addis transfer pull in opposite directions. "
     f"The slope rises with k under every training set, but the k where it reaches 1 depends on the set: k {crosses['All QC-passed']} with every spectrum, k {crosses['EC below 50 µg']} below 50 µg, never below 25 µg", "Open; decide it together with the training set"),
    ("The background", "Not tested here", "A feature about 1,500 cm⁻¹ wide is removed at every rank (10 Nov check), so the continuum loss belongs to the background, not to rank or tau", "Open"),
], columns=["decision", "choice", "why", "status"])

text = {
    "intro": f"""The question was which VIBES settings to use for new work: the solver, the rank and tau. Each step below asks one
question, shows the measurement and states the choice. EC is in µg per filter. Unless a figure says the calibration was rebuilt, EC is
read with the frozen k 4 IMPROVE model, which was trained on spectra corrected at rank 70 and tau 0.1.""",
    "solver": f"""VIBES first fits a noise level for each scan, then the baseline. In the October run the optimiser stopped within 30
steps on {int(early_by_type.sum())} scans ({int(early_by_type.get('PM2.5', 0))} PM2.5, {int(early_by_type.get('PM10', 0))} PM10, {int(early_by_type.get('FB', 0))} field blanks),
leaving the noise level too high and the baseline too close to the average blank. The converged solver fits the same objective until it
cannot improve. The scans here are the PM2.5 scans saved under the 87-blank background. A flatter corrected spectrum is shown for the
Addis early stops (3 Nov tests); that their lower EC is more correct is not shown.""",
    "rank": f"""The rank was lowered by keeping the first components of the rank-70 background, not by building a new background at each
rank. Rank 40 changes EC by a median {f2(r40.median_abs_pct)} % and at most {f2(r40.largest_ug)} µg, so components 41 to 70 do nothing to EC.
The 10 Nov injection test explains why: a band up to 150 cm⁻¹ wide survives at every rank on a loaded filter.""",
    "tau_fixed": f"""These numbers are the cost of correcting a spectrum one way and calibrating another. They do not say which tau is
best. The fitted noise level follows tau (median {f1(t05.noise_e4)}, {f1(at.loc[(70, 0.1), 'noise_e4'])} and {f1(t20.noise_e4)} ×10⁻⁴ at tau 0.05, 0.1 and 0.2), so a noise
level only compares between fits made at one tau.""",
    "tau_refit": f"""IMPROVE ({len(rows_):,} spectra) and Addis were corrected at each tau and each tau was scored with its own model. At
tau 0.1 the rebuild returns the 3 Nov calibration. Scores are compared at a fixed k. Of {comparisons} comparisons with tau 0.1, tau 0.05 has a larger
error in {worse(0.05)} and a smaller one in {better(0.05)}; tau 0.2 has a larger error in {worse(0.2)} and a smaller one in {better(0.2)}. The intervals
resample sites 2,000 times.""",
    "krule": f"""The frozen procedure draws a site-grouped cross-validation curve on the train split and takes the first major minimum. That
rule picks k {picked_all} at tau 0.1, 0.05 and 0.2. With one QC-failed spectrum left out of training it picks {picked_qc}. The held-out RMSE
at k 4 is about {f2(k4.mean())} µg and at k 11 about {f2(k11.mean())} µg at every tau, so a setting judged at the rule's k is mostly being judged on k.""",
    "qc": f"""IMPROVE filter {failed.filter_id.iloc[0]} ({failed.site.iloc[0]}) is marked as failing raw QC in the decision bundle, and it is in the
train split of the frozen k 4 model. Its corrected spectrum is {far.distance:.0f} times farther from the mean than a typical train spectrum.
The k 4 model was fitted with and without it on the converged tau 0.1 spectra and both were applied to the converged network run. The
model with it is the 3 Nov rebuilt k 4 model. IMPROVE cannot tell the two apart (held-out RMSE {f3(frozen_with)} against {f3(frozen_without)} µg),
but on SPARTAN PM2.5 scans EC moves by a median {f1(qg.loc['PM2.5 scans', 'median_abs_pct'])} %, {f1(qg.loc['PM2.5 scans', 'moved_gt10_pct'])} % of scans move by more than 10 %, and
Addis moves {f1(qg.loc['Addis PM2.5 scans', 'median_abs_pct'])} %. Nothing here says which model is right: blanks read closer to zero without the
spectrum, repeat scans agree less well, and the Addis slope against HIPS is the same. The 3 Nov transfer table was made with the
spectrum in training; the scoring harness now leaves it out.""",
    "domain": f"""No network PM2.5 scan reads above 50 µg, and {100 * (clean_ec > 25).mean():.1f} % read above 25 µg, so the heaviest training
filters lie outside SPARTAN's range. Six training sets were tried: every QC-passed train spectrum, only those with thermal EC below 150,
100, 50 and 25 µg, and the 99 % of spectra nearest the mean. Each model was refitted with one of its six most extreme spectra dropped,
and on IMPROVE sites drawn with replacement 30 times. Trimming removes the pull of single spectra ({f1(everything.one_spectrum_max_pct)} % with everything in,
{f1(lightest.one_spectrum_max_pct)} % below 25 µg). It does not pin EC down: with sites resampled a scan's EC still spreads by {f1(lightest.scan_spread_pct)} % below 25 µg
({f1(lightest_by_k.loc['k 8', 'scan_spread_pct'])} % at k 8, {f1(lightest_by_k.loc['k 12', 'scan_spread_pct'])} % at k 12), and the models differ from each other by {between.min():.0f} to {between.max():.0f} %. The Addis slope against HIPS
goes from {f2(everything.addis_slope)} to {f2(lightest.addis_slope)} at k 4.

The earlier finding that the Addis slope follows k was reread with the training set held fixed. It holds: under every one of the six
sets the slope rises with k, from {f2(slope_k.loc[4].min())} to {f2(slope_k.loc[4].max())} at k 4 to {f2(slope_k.loc[18].min())} to {f2(slope_k.loc[18].max())} at k 18. The training set moves it as well. Below 25 µg
the slope is {f2(ratio_light.min())} to {f2(ratio_light.max())} of the slope with every spectrum in, at every k, and it never reaches 1. From k 2 up, cut-offs of 150 and 100 µg stay
inside the band that resampling IMPROVE sites draws around "all" ({f2(da.loc[('All QC-passed', 4), 'slope_lo'])} to {f2(da.loc[('All QC-passed', 4), 'slope_hi'])} at k 4, {f2(da.loc[('All QC-passed', 12), 'slope_lo'])} to {f2(da.loc[('All QC-passed', 12), 'slope_hi'])} at k 12). How closely FTIR EC
follows HIPS barely changes (R² {f2(domain_addis.addis_r2[domain_addis.k.ge(3)].min())} to {f2(domain_addis.addis_r2[domain_addis.k.ge(3)].max())} from k 3 up), so R² cannot choose between them: k and the training set set the scale of Addis EC, not its
pattern.""",
    "choices": """The solver, the rank and tau are settled by this work. The calibration is not. The number of PLS components moves
held-out accuracy far more than any of the three settings, and moves the Addis comparison with HIPS the other way. And at k 4 one training
spectrum changes SPARTAN EC by more than the solver, the rank and tau together.""",
}

write_bundle(
    MEETING_ID,
    generated_from="research/ftir_hips_chem/workflows/run_vibes_rank_tau_ec_20261116.py and run_vibes_tau_calibration_20261116.py",
    tables={"terms": terms, "glance": glance, "solver_groups": solver_groups, "solver_scans": solver_scans, "solver_example": solver_example,
            "settings": settings, "by_rank": by_rank, "by_tau": by_tau, "injection": injection,
            "heldout": heldout, "heldout_fixed": heldout_fixed, "boot": boot, "transfer": transfer,
            "cv": cv, "rule": rule, "qc_effect": qc_effect, "qc_groups": qc_groups, "qc_sites": qc_sites, "qc_checks": qc_checks,
            "qc_by_k": qc_by_k, "qc_leverage": qc_leverage, "domain": domain, "domain_pairs": domain_pairs, "domain_sites": domain_sites, "domain_addis": domain_addis,
            "choices": choices},
    text=text, colors=COLORS,
)
