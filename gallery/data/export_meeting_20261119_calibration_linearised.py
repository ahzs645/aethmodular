"""Export the 19 Nov 2026 page "Which calibration, once HIPS is linearised" for the gallery's Meeting tab.

    uv run python gallery/data/export_meeting_20261119_calibration_linearised.py

The question: now that HIPS is linearised, is there an IMPROVE calibration (raw spectra or a baseline, a training set,
a number of components) that brings Addis FTIR EC to a slope of 1 and an intercept of 0 against it? Nothing is fitted
here. It reads the saved outputs of

    research/ftir_hips_chem/workflows/run_linearised_calibration_sweep_20261119.py   (990 models, each against HIPS as
                                                                                     measured and linearised, and
                                                                                     against thermal EC on IMPROVE)
    gallery/app/public/data/calibration_runs.json                                    (the nine saved configurations)
    docs/addis-reference-closure-2026-11-10.json                                     (MACs measured against thermal EC)

Slopes are FTIR EC on HIPS BC in µg/m³, Deming with the closeout's error ratio. A slope at another MAC is the slope at
MAC 10 times MAC ÷ 10 and the intercept does not move (the error ratio scales with MAC², lib/macMath).

Writes gallery/app/public/data/meetings/calibration-linearised-20261119.json; figures in
src/meetings/calibration-linearised-20261119.meeting.ts.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research/ftir_hips_chem/scripts")]

from hips_loading import DEPOSIT_AREA_CM2, IMPROVE_CEILING  # noqa: E402
from meeting_bundle import write_bundle  # noqa: E402
from pls_transfer import deming_regression  # noqa: E402

MEETING_ID = "calibration-linearised-20261119"
SWEEP = ROOT / "research/ftir_hips_chem/output/tables/linearised_calibration_sweep_20261119"
DATA = ROOT / "gallery/app/public/data"
CLOSURE = ROOT / "docs/addis-reference-closure-2026-11-10.json"
LAMBDA = 2.96
SPECTRA = {"raw": "Raw spectra", "spline": "Spline baseline", "VIBES": "VIBES baseline"}
CEILINGS = IMPROVE_CEILING.values()
HIPS = {"measured": "As measured", "central": f"Linearised, ceiling {CEILINGS['central']:.2f}", "lo": f"Linearised, ceiling {CEILINGS['lo']:.2f}",
        "hi": f"Linearised, ceiling {CEILINGS['hi']:.2f}"}
ALL = "All IMPROVE filters"
TOP = 12
COLORS = {"Raw spectra": "#4a4a4a", "Spline baseline": "#2C6E9E", "VIBES baseline": "#7A4FA3",
          "Bega": "#E67E22", "Belg": "#27AE60", "Kiremt": "#3498DB",
          # the training sets: the EC trims on one ramp, the cohorts apart
          ALL: "#1f3b63", "Thermal EC below 150 µg": "#2f6aa8", "Thermal EC below 50 µg": "#4f9bc9", "Thermal EC below 25 µg": "#9ecae1",
          "Six urban sites": "#54a24b", "Lowest OC/EC (locked cohort)": "#b279a2", "Smokiest tenth (non-soil potassium)": "#f58518",
          "Thermal EC 15 µg or more": "#e45756", "Lot 251 only": "#7f7f7f", "500 nearest the Addis median spectrum": "#d6a700",
          "All filters, weighted toward heavy ones": "#72b7b2"}
f1, f2 = (lambda v: f"{v:.1f}"), (lambda v: f"{v:.2f}")  # noqa: E731
signed = lambda v: f"{'+' if v >= 0 else '−'} {abs(v):.2f}"  # noqa: E731
line = lambda s, i: f"{s:.2f}x {signed(i)}"  # noqa: E731
minus = lambda v: f"{v:+.2f}".replace("-", "−")  # noqa: E731

sweep = pd.read_csv(SWEEP / "sweep.csv")
sweep["Spectra"], sweep["HIPS"] = sweep.spectra.map(SPECTRA), sweep.hips.map(HIPS)
sweep = sweep.rename(columns={"training": "Training set"})
sweep["model"] = sweep.Spectra + ", " + sweep["Training set"].str[0].str.lower() + sweep["Training set"].str[1:]
closure = json.loads(CLOSURE.read_text())
anchor = {a["anchor"]: a["value"] for a in closure["level_anchors"]}
MAC_URBAN = round(next(v for k, v in anchor.items() if "six urban IMPROVE sites" in k and "unsaturated MAC" in k), 1)
MAC_LIGHT = round(next(v for k, v in anchor.items() if "unsaturated MAC" in k and "all sites" in k), 1)
MAC_LOW = round(next(v for k, v in anchor.items() if "unsaturated MAC" in k and "OC/EC < 4" in k), 1)
MAC_ADAMA = round(next(v for k, v in anchor.items() if "Adama MAC" in k and "2.26" in k), 1)
MACS = {"MAC 10 (the convention)": 10.0, f"MAC {MAC_URBAN} (urban IMPROVE, against thermal EC)": MAC_URBAN}
MAC10, MACU = list(MACS)

central = sweep[sweep.hips.eq("central")].copy()
central["slope_urban"] = central.slope * MAC_URBAN / 10


def distance(frame, mac):
    """How far a fit is from slope 1 and intercept 0 at a MAC: the intercept counted as a share of the mean HIPS BC."""
    return np.hypot(frame.slope * mac / 10 - 1, frame.intercept / (frame.mean_hips_ugm3 * 10 / mac))


# ---------------------------------------------------------------- 1. slope, intercept and R² against k
by_k = sweep[["Spectra", "Training set", "k", "HIPS", "slope", "intercept", "r2", "median_ec_ugm3", "mac_for_slope_1", "n_training"]].copy()
by_k["slope_urban"] = by_k.slope * MAC_URBAN / 10

# ---------------------------------------------------------------- 2. every model on the slope and intercept plane
plane = central[central.k >= 2][["Spectra", "Training set", "k", "slope", "slope_urban", "intercept", "r2", "improve_urban_level", "improve_median_abs_error_ug"]].copy()
plane["R²"] = plane.r2.round(2)
plane["IMPROVE urban FTIR ÷ thermal EC"] = plane.improve_urban_level.round(2)
# one panel per MAC: the slope scales with the MAC and the intercept stays
plane = pd.concat([plane.assign(MAC=label, slope_at_mac=plane.slope * mac / 10) for label, mac in MACS.items()], ignore_index=True)

# ---------------------------------------------------------------- 3. the closest models, each at its best k
best_rows = []
for label, mac in MACS.items():
    scored = central[central.k >= 2].assign(distance=lambda d, mac=mac: distance(d, mac))
    top = scored.loc[scored.groupby("model").distance.idxmin()].sort_values("distance").head(TOP)
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        best_rows.append({"MAC": label, "rank": rank, "Spectra": r.Spectra, "Training set": r["Training set"], "n_training": r.n_training, "k": r.k,
                          "slope": r.slope * mac / 10, "intercept": r.intercept, "r2": r.r2, "distance": r.distance, "median_ec": r.median_ec_ugm3,
                          "urban": r.improve_urban_level, "heavy": r.improve_heavy_level, "light": r.improve_light_level, "error": r.improve_median_abs_error_ug,
                          "model": r.model})
best = pd.DataFrame(best_rows)
first = {label: best[best.MAC.eq(label)].iloc[0] for label in MACS}
best_10, best_urban = (best[best.MAC.eq(label)].drop(columns="MAC") for label in MACS)

# the plain models for comparison: every filter in training, at the k of the published-like (18) and the frozen (4) model
plain_rows = []
for spectra, k, name in (("raw", 18, "Raw spectra, every filter, k 18 (as SPARTAN's published EC is built)"), ("VIBES", 4, "VIBES baseline, every filter, k 4 (as the frozen model)"),
                         ("spline", 5, "Spline baseline, every filter, k 5")):
    r = central[central.spectra.eq(spectra) & central["Training set"].eq(ALL) & central.k.eq(k)].iloc[0]
    plain_rows.append({"model": name, "slope_10": r.slope, "slope_urban": r.slope_urban, "intercept": r.intercept, "r2": r.r2, "mac_for_slope_1": r.mac_for_slope_1,
                       "urban": r.improve_urban_level, "heavy": r.improve_heavy_level, "error": r.improve_median_abs_error_ug})
plain = pd.DataFrame(plain_rows)

# ---------------------------------------------------------------- 4. what the models read where EC is known
cost = central[central.k >= 2][["Spectra", "Training set", "k", "slope", "improve_urban_level", "improve_heavy_level", "improve_light_level", "improve_median_abs_error_ug"]].copy()
cost = cost[cost.improve_urban_level.between(0, 6)]
cost["Median error on IMPROVE (µg)"] = cost.improve_median_abs_error_ug.round(2)
# the models that are right at IMPROVE's urban sites (within 5 % of thermal EC, held out): where they put Addis
right = cost[cost.improve_urban_level.between(0.95, 1.05)]
right_by = right.groupby("Spectra").slope.agg(["min", "max", "median", "size"])

# ---------------------------------------------------------------- 5. chosen models, filter by filter
z = np.load(SWEEP / "predictions.npz", allow_pickle=False)
ids = z["filter_id"].astype(str)
dates = {r["id"]: r["date"] for r in json.loads((DATA / "spartan_reference.json").read_text())["rows"]}
month = pd.to_datetime(pd.Series([dates.get(i) for i in ids])).dt.month
season = month.map(lambda m: "Bega" if m in (10, 11, 12, 1) else "Belg" if m in (2, 3, 4, 5) else "Kiremt" if m in (6, 7, 8, 9) else None)
back = {v: k for k, v in SPECTRA.items()}
shown = {}
for label, r in first.items():
    shown[f"Closest at {label.split(' (')[0]}: {r.model}, k {int(r.k)}"] = (back[r.Spectra], r["Training set"], int(r.k))
shown["Raw spectra, every filter, k 18"] = ("raw", ALL, 18)
shown["VIBES baseline, every filter, k 4"] = ("VIBES", ALL, 4)
shown["Spline baseline, every filter, k 5"] = ("spline", ALL, 5)
drawn = pd.concat([pd.DataFrame({"Model": name, "filter": ids, "season": season.to_numpy(), "date": [dates.get(i) for i in ids],
                                 "hips": z["hips_central"], "hips_urban": z["hips_central"] * 10 / MAC_URBAN, "hips_measured": z["hips_measured"],
                                 "ftir": z[f"ec|{spectra}|{training}"][:, k - 1]}) for name, (spectra, training, k) in shown.items()], ignore_index=True).dropna(subset=["hips", "ftir"])

# ---------------------------------------------------------------- 6. the nine saved configurations of the Calibration tab
runs = json.loads((DATA / "calibration_runs.json").read_text())
index = json.loads((DATA / "spartan_network/index.json").read_text())
site_file = next(s["filters"] for s in index["sites"] if s["code"] == "ETAD")
volumes = {r["id"]: r.get("Sampled volume") for r in json.loads(gzip.open(DATA / site_file).read())["rows"]}
PRESET = {"pool": "Every IMPROVE filter", "smoke": "Smoke cohort", "eth_shaped": "Ethiopia-shaped", "analogs": "Analogs of Addis", "ocec": "Lowest OC/EC"}
SPACE = {"raw": "Raw spectra", "airspec": "Spline baseline", "deriv2": "Second derivative"}
saved_rows = []
for run in runs["runs"]:
    if run["target"] != "addis":
        continue
    preset = next(p for p in runs["presets"] if p["key"] == run["preset"])
    e = run["eval"]
    fabs, pred = np.array(e["ref"], float), np.array(e["pred"], float)
    volume = np.array([v if isinstance(v := volumes.get(i), (int, float)) else np.nan for i in e["id"]], float)
    tau = fabs * volume / (100 * DEPOSIT_AREA_CM2)
    linear = fabs / (1 - tau / CEILINGS["central"])
    ok = np.isfinite(linear) & np.isfinite(pred)
    a = deming_regression(pd.Series(fabs / 10), pd.Series(pred), LAMBDA)
    b = deming_regression(pd.Series(linear[ok] / 10), pd.Series(pred[ok]), LAMBDA)
    saved_rows.append({"configuration": f"{PRESET.get(preset['cohort'], preset['cohort'])}{' (' + str(preset['cutoff']) + ')' if preset.get('cutoff') else ''}, {SPACE.get(preset['spectra'], preset['spectra']).lower()}",
                       "k": run["k"], "measured": line(a["slope"], a["intercept"]), "linearised": line(b["slope"], b["intercept"]),
                       "slope": b["slope"], "intercept": b["intercept"], "r2": float(np.corrcoef(linear[ok], pred[ok])[0, 1] ** 2),
                       "slope_urban": b["slope"] * MAC_URBAN / 10, "mac_for_slope_1": 10 / b["slope"], "n": int(ok.sum())})
saved = pd.DataFrame(saved_rows).sort_values("slope", ascending=False)

# ---------------------------------------------------------------- 7. the MAC, measured against thermal EC
mac_table = pd.DataFrame([
    {"where": "The convention", "mac": 10.0, "source": "Fabs ÷ 10; not measured against thermal EC"},
    {"where": "IMPROVE, light filters, all sites", "mac": MAC_LIGHT, "source": "347,936 samples, 210 sites (10 Nov closure)"},
    {"where": "IMPROVE, sites with median OC/EC below 4", "mac": MAC_LOW, "source": "32,898 samples, 32 sites"},
    {"where": "IMPROVE, six urban sites", "mac": MAC_URBAN, "source": "11,020 samples"},
    {"where": "Adama, Ethiopia, after the same linearising", "mac": MAC_ADAMA, "source": "five filters, July 2024"},
])
mac_table["slope_if_ftir_right"] = 10 / mac_table.mac

# ---------------------------------------------------------------- numbers for the titles and text
every = central[central["Training set"].eq(ALL)]
at = lambda spectra, k: every[every.spectra.eq(spectra) & every.k.eq(k)].iloc[0]  # noqa: E731
raw_range = every[every.spectra.eq("raw") & every.k.between(2, 30)]
vibes_range = every[every.spectra.eq("VIBES") & every.k.between(2, 30)]
spline_range = every[every.spectra.eq("spline") & every.k.between(2, 30)]
b10, bu = first[MAC10], first[MACU]
raw18 = at("raw", 18)

top_slope = central[central.k >= 2].loc[central[central.k >= 2].slope.idxmax()]
n_models, n_fits = central.model.nunique(), int((central.k >= 2).sum())
rr, rv, rs = right_by.loc["Raw spectra"], right_by.loc["VIBES baseline"], right_by.loc["Spline baseline"]

glance = pd.DataFrame([
    {"question": "Does any calibration reach a slope of 1 against linearised HIPS at MAC 10?",
     "found": f"No. Of {n_models} models at 2 to 30 components ({n_fits} fits), none passes {top_slope.slope:.2f}, and {int((central[central.k >= 2].slope >= 0.9).sum())} reach 0.9."},
    {"question": "Which comes closest to a slope of 1 and an intercept of 0?",
     "found": f"{b10.model}, k {int(b10.k)}: {line(b10.slope, b10.intercept)}, R² {b10.r2:.2f}. It is trained on {int(b10.n_training)} filters and its median error on held-out IMPROVE sites is {b10.error:.2f} µg, against {raw18.improve_median_abs_error_ug:.2f} µg for raw spectra with every filter (k 18)."},
    {"question": "With or without a baseline?",
     "found": f"Without. Raw spectra stay proportional at every k: intercept {minus(raw_range.intercept.min())} to {minus(raw_range.intercept.max())} µg/m³, R² {raw_range.r2.min():.2f} to {raw_range.r2.max():.2f}. The VIBES intercept falls to {minus(vibes_range.intercept.min())} µg/m³ as k rises. The spline slope never passes {spline_range.slope.max():.2f}."},
    {"question": "Where do the models that are right on IMPROVE put Addis?",
     "found": (f"Of the fits within 5 % of thermal EC at IMPROVE's six urban sites, raw spectra give Addis slopes of {rr['min']:.2f} to {rr['max']:.2f} (median {rr['median']:.2f}), "
               f"VIBES {rv['min']:.2f} to {rv['max']:.2f} and the spline baseline {rs['min']:.2f} to {rs['max']:.2f}. With the FTIR level right, a slope of {rr['median']:.2f} is a MAC of {10 / rr['median']:.0f} m²/g.")},
    {"question": "And at a MAC measured against thermal EC?",
     "found": f"At {MAC_URBAN} m²/g (urban IMPROVE) the closest is {bu.model}, k {int(bu.k)}: {line(bu.slope, bu.intercept)}, reading {bu.urban:.2f} of thermal EC on IMPROVE. Raw spectra with every filter in training give {line(raw18.slope_urban, raw18.intercept)} at k 18."},
])

text = {
    "intro": f"""**The question (19 Nov).** Linearising HIPS took the Addis intercept to about zero and left a slope near 0.55. Is there a calibration,
with or without a baseline, that brings the slope to 1 as well?

**What was run.** {n_models} calibrations, each at 1 to 30 components: three kinds of spectra (raw, spline baseline, VIBES baseline) and eleven training sets.
Each is applied to the {int(central.n.max())} Addis filters and fitted against HIPS BC, as measured and linearised filter by filter. The same model is also
scored against thermal EC on IMPROVE sites held out of its training.

**How to read a slope.** After linearising, the slope is the FTIR level × 10 ÷ MAC. A slope of 1 at MAC 10 says FTIR EC equals Fabs ÷ 10. It does not say
FTIR EC equals EC: every MAC measured against thermal EC is above 10 ({MAC_LIGHT} to {MAC_ADAMA} m²/g).""",
    "by_k": f"""With every IMPROVE filter in training, raw spectra give {line(at('raw', 4).slope, at('raw', 4).intercept)} at k 4 and {line(raw18.slope, raw18.intercept)} at k 18:
the intercept stays near zero and k moves only the level. VIBES gives {line(at('VIBES', 4).slope, at('VIBES', 4).intercept)} at k 4 and
{line(at('VIBES', 18).slope, at('VIBES', 18).intercept)} at k 18. The spline baseline stays low at every k. Switch HIPS to "As measured" to see the old reading, where the same
raw model is {sweep[sweep.hips.eq('measured') & sweep.spectra.eq('raw') & sweep['Training set'].eq(ALL) & sweep.k.eq(18)].slope.item():.2f}x with a large negative intercept.""",
    "plane": f"""Each dot is one calibration at one k. The target is where the two lines cross. At MAC 10 every dot is left of it: the largest slope is
{top_slope.slope:.2f}. At a MAC of {MAC_URBAN} the same dots move right by a factor of {MAC_URBAN / 10:.2f}, and the raw-spectra models land on it.""",
    "best": """Each model is listed at the k that brings it closest. Distance is √((slope − 1)² + (intercept ÷ mean HIPS BC)²), so a slope 10 % off counts like an
intercept of 10 % of the mean. The IMPROVE columns are the same model on IMPROVE sites it did not train on, against thermal EC: 1.00 is right.""",
    "cost": f"""Each dot is one calibration at one k: its Addis slope against what it reads at IMPROVE's six urban sites, held out of training. The models
within 5 % of thermal EC there give Addis slopes of {rr['min']:.2f} to {rr['max']:.2f} with raw spectra ({int(rr['size'])} fits), {rv['min']:.2f} to {rv['max']:.2f} with VIBES
({int(rv['size'])}) and {rs['min']:.2f} to {rs['max']:.2f} with the spline baseline ({int(rs['size'])}). No fit, right or wrong on IMPROVE, reaches 0.9; a slope of 1 would be off the right edge.""",
    "drawn": f"The chosen models on the Addis filters. Switch the MAC to see the same points against Fabs ÷ {MAC_URBAN}.",
    "saved": """The nine configurations saved on the Calibration tab, read again. Against HIPS as measured the spline configurations looked best (slope near 0.9).
Against linearised HIPS they read about a quarter of it, and the raw-spectra configurations are the ones with an intercept near zero. The Calibration tab now
follows the linearised view for these nine (Plot, HIPS, Linearised).""",
    "mac": f"""If FTIR EC were exactly right, the slope against linearised HIPS at MAC 10 would be 10 ÷ MAC: {10 / MAC_LIGHT:.2f} to {10 / MAC_ADAMA:.2f} for the MACs
measured against thermal EC. Raw spectra with every filter in training give {raw18.slope:.2f} at k 18 and {at('raw', 30).slope:.2f} at k 30: at the low end of that range, which is the Adama MAC.""",
    "limits": """- HIPS is optical. A slope against it compares two instruments; it is not a score against EC.
- The ceiling is IMPROVE's (2.26, interval 1.82 to 2.70). Across it the intercept of one model moves by about ±1 µg/m³; the order of the models does not.
- "Closest" is chosen on Addis, so the top rows are fitted to the target they are scored on. The IMPROVE columns are the independent check.
- The slope keeps rising slowly with k, so many models are closest at 25 to 30 components, where IMPROVE scores have stopped improving.
- IMPROVE's urban sites have a thermal OC/EC near 3; Addis is near 1. No IMPROVE filter tests a calibration at Addis composition.
- The nearest-to-Addis training set uses Addis spectra (not Addis HIPS or EC) to choose its 500 filters.
- No thermal EC has been measured on an Addis filter.""",
}

write_bundle(
    MEETING_ID,
    generated_from="research/ftir_hips_chem/workflows/run_linearised_calibration_sweep_20261119.py",
    tables={"glance": glance, "by_k": by_k, "plane": plane, "best_10": best_10, "best_urban": best_urban, "plain": plain, "cost": cost, "drawn": drawn, "saved": saved, "mac": mac_table},
    text=text, colors=COLORS,
)
print(glance.to_string(index=False))
print(right_by.round(2).to_string())
print(best[["MAC", "rank", "model", "k", "slope", "intercept", "r2", "urban", "error"]].round(2).to_string(index=False))
print(plain.round(2).to_string(index=False))
print(saved[["configuration", "k", "measured", "linearised", "r2"]].round(2).to_string(index=False))
