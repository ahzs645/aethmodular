"""Export the 9 Nov 2026 investigation status page for the gallery's spec-driven Meeting tab.

Run from the repo root with the repo environment:

    uv run python gallery/data/export_meeting_20261109.py

A status page, not a recorded meeting: where the existing-data FTIR EC investigation stands
after the 7 Nov closeout, for the supervisor's end-of-December check-in. Every number is read
from an export that already exists; nothing is refitted.

- gallery/review/closeout_20261107/continuum_baseline.json: the six-candidate source
  scoreboard (12,386 common IMPROVE filters, fixed k 18, five site folds) and the saved Addis
  fits of each candidate against HIPS optical BC and the aethalometer.
- gallery/app/public/data/meetings/ann-20261009.json: Addis vs IMPROVE loading.
- gallery/app/public/data/meetings/meeting-20261028.json: EC share of PM2.5, and the Addis
  slope/intercept by season for five ways of choosing calibration filters.
- git log: commits per week since 31 Aug 2026, and whether a manuscript draft exists.

Writes gallery/app/public/data/meetings/status-20261109.json (see meeting_bundle.py).
The figures are declared in gallery/app/src/meetings/status-20261109.meeting.ts.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from meeting_bundle import OUT, write_bundle  # noqa: E402

MEETING_ID = "status-20261109"
CLOSEOUT = ROOT / "gallery/review/closeout_20261107/continuum_baseline.json"
ANN_1009 = OUT / "ann-20261009.json"
MEETING_1028 = OUT / "meeting-20261028.json"
WEEKS_FROM = dt.date(2026, 8, 31)
WEEKS_TO = dt.date(2026, 11, 9)

# reader-facing candidate names, as in docs/current-research-summary.md
CANDIDATES = {
    "published": "SPARTAN published",
    "raw": "Raw spectra",
    "spline": "Spline baseline",
    "vibes": "Production VIBES",
    "vibes_deflate": "Deflated VIBES",
    "vibes_anchorhp": "PTFE-anchored VIBES",
    "epo": "Anchored + EPO",
    "HIPS BC": "HIPS optical BC itself",
}
# the saved Addis fits keep the historical EPO pipeline; only the corrected source score is reported for EPO
ADDIS_MODELS = ["published", "raw", "spline", "vibes", "vibes_deflate", "vibes_anchorhp", "HIPS BC"]
COMPARATORS = {"hips_bc": "HIPS optical BC (Fabs / MAC 10)", "aeth_ir": "Aethalometer IR BC"}
METRICS = {
    "improve_median_absolute_error": "Median absolute error",
    "improve_mae": "Mean absolute error",
    "improve_rmse": "RMSE",
    "improve_p95_absolute_error": "95th percentile absolute error",
}
COLORS = {
    "SPARTAN published": "#B23327", "Raw spectra": "#22252A", "Spline baseline": "#2C6E9E",
    "Production VIBES": "#d95f0e", "Deflated VIBES": "#8F8C84", "PTFE-anchored VIBES": "#C49442",
    "Anchored + EPO": "#7A4FA3", "HIPS optical BC itself": "#5B5852",
    "Addis": "#C49442", "IMPROVE": "#8a8f98",
    "Done": "#27AE60", "On track": "#2C6E9E", "Not started": "#B23327", "Waiting on others": "#8F8C84",
    "Analysis and tooling": "#2C6E9E",
}


def table(bundle: dict, name: str) -> pd.DataFrame:
    return pd.DataFrame(bundle["tables"][name])


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


closeout = json.loads(CLOSEOUT.read_text())
ann = json.loads(ANN_1009.read_text())
m1028 = json.loads(MEETING_1028.read_text())

# ---------------------------------------------------------------- 1. Addis sits outside the library
loading = table(ann, "loading_summary")
conc, mass = loading.iloc[0], loading.iloc[1]
spec = table(m1028, "speciation_table")
keep = {"Addis Dry": "Addis", "Addis Belg": "Addis", "Addis Kiremt": "Addis",
        "IMPROVE, urban sites": "IMPROVE", "IMPROVE, all library": "IMPROVE"}
ec_share = spec.loc[spec.population.isin(keep)].assign(
    network=lambda d: d.population.map(keep), ec_share=lambda d: 100 * d.EC)[["population", "network", "ec_share", "OC / EC"]]
addis_share = ec_share.loc[ec_share.network.eq("Addis"), "ec_share"]
library_share = float(ec_share.loc[ec_share.population.eq("IMPROVE, all library"), "ec_share"].iloc[0])
text_domain = f"""**Addis is outside what the IMPROVE library has seen.** In µg/m³ the Addis median is
{conc['Addis median'] / conc['IMPROVE median']:.0f}× the IMPROVE median; per filter, which is what FTIR sees, it is
{mass['Addis median'] / mass['IMPROVE median']:.0f}×, and {mass['% of Addis above IMPROVE p99']:.0f} % of Addis filters are still above the IMPROVE
99th percentile. EC is {addis_share.min():.0f}–{addis_share.max():.0f} % of Addis PM2.5 by season against about {library_share:.0f} % across the
IMPROVE library. A calibration cannot be checked on a composition it never trained on."""

# ---------------------------------------------------------------- 2. preprocessing on the Addis cross-plots
af = pd.DataFrame(closeout["addis_fits"])
af["model"] = af.model.str.replace(r"\s+k\d+$", "", regex=True)
addis = (af.loc[af.model.isin(ADDIS_MODELS) & af.x.isin(COMPARATORS)]
         .assign(candidate=lambda d: d.model.map(CANDIDATES), comparator=lambda d: d.x.map(COMPARATORS),
                 order=lambda d: d.model.map(ADDIS_MODELS.index))
         .sort_values(["comparator", "order"])[["candidate", "comparator", "n", "slope", "intercept", "r2"]])
hb = addis.loc[addis.comparator.eq(COMPARATORS["hips_bc"])].set_index("candidate")
ae = addis.loc[addis.comparator.eq(COMPARATORS["aeth_ir"])].set_index("candidate")
pub, spl = hb.loc["SPARTAN published"], hb.loc["Spline baseline"]
hips_vs_aeth = af.loc[af.model.eq("HIPS BC") & af.x.eq("aeth_ir")].iloc[0]
text_preprocess = f"""**Baseline correction is the largest single improvement, and it is not a fix.** Against HIPS optical BC
(MAC 10), the SPARTAN published product has intercept {pub.intercept:.2f} µg/m³ and slope {pub.slope:.2f} (n = {pub.n:.0f});
the spline-baselined calibration moves that to {spl.intercept:.2f} and {spl.slope:.2f} (n = {spl.n:.0f}).

**The aethalometer does not separate the candidates.** Against it, most intercepts are already near zero (published
{ae.loc['SPARTAN published', 'intercept']:+.2f}, PTFE-anchored VIBES {ae.loc['PTFE-anchored VIBES', 'intercept']:+.2f}; production VIBES
{ae.loc['Production VIBES', 'intercept']:+.2f} µg/m³), while HIPS optical BC against the same aethalometer has intercept
{hips_vs_aeth.intercept:+.2f} (n = {hips_vs_aeth.n:.0f}). Which measurement carries the offset stays open: agreement between optical
methods is a diagnostic, **not** proof of correct EC, and no Addis filter has thermal (TOR) EC."""

# ---------------------------------------------------------------- 3. seasons
cals = table(m1028, "cals")
COLORS.update({k: v for k, v in m1028["colors"].items() if k in set(cals.calibration)})
seasonal = cals.loc[cals.calibration.eq("spectral look-alikes (Figure 9)")].set_index("season")
allimp = cals.loc[cals.calibration.eq("all IMPROVE (Figure 7)")].set_index("season")
text_seasons = f"""**The seasons fail differently.** With the same baseline-corrected all-IMPROVE calibration the Addis slope is
{allimp.loc['Dry', 'slope']:.2f} in Dry, {allimp.loc['Belg', 'slope']:.2f} in Belg and {allimp.loc['Kiremt', 'slope']:.2f} in Kiremt. Choosing each season's own spectral
look-alikes lifts Kiremt to {seasonal.loc['Kiremt', 'slope']:.2f} but leaves Dry at {seasonal.loc['Dry', 'slope']:.2f} and Belg at {seasonal.loc['Belg', 'slope']:.2f}, and no other
choice of calibration filters (lowest OC/EC, urban IMPROVE) brings Dry or Belg near 1. Per µg of EC, Addis carries far less
organic absorbance than its IMPROVE look-alikes, least in Dry: the likely cause is **composition, not concentration**."""

# ---------------------------------------------------------------- 4. the source scoreboard
sb = pd.DataFrame(closeout["scoreboard"]).assign(candidate=lambda d: d.model.map(CANDIDATES))
assert sb.n_common.nunique() == 1 and sb.n_failed.eq(0).all(), "candidates must share one population with no failed predictions"
n_common, n_heavy = int(sb.n_common.iloc[0]), int(sb.n_heavy.iloc[0])
scores = sb[["candidate", "k", "improve_median_absolute_error", "improve_mae", "improve_rmse", "improve_mean_bias",
             "improve_p95_absolute_error", "improve_heavy_ratio", "n_common", "n_heavy"]].rename(columns={
    "improve_median_absolute_error": "MedAE", "improve_mae": "MAE", "improve_rmse": "RMSE", "improve_mean_bias": "mean bias",
    "improve_p95_absolute_error": "P95 abs error", "improve_heavy_ratio": "heavy median FTIR / TOR"})
scores_long = sb.melt(id_vars=["candidate"], value_vars=list(METRICS), var_name="key", value_name="value").assign(
    metric=lambda d: d.key.map(METRICS))[["metric", "candidate", "value"]]
best = {m: sb.loc[sb[k].idxmin(), "candidate"] for k, m in METRICS.items()}
heavy = sb[["candidate", "improve_heavy_ratio"]].rename(columns={"improve_heavy_ratio": "ratio"})
epo, anch = sb.set_index("candidate").loc["Anchored + EPO"], sb.set_index("candidate").loc["PTFE-anchored VIBES"]
text_scores = f"""**No universal winner.** Six preprocessing pipelines, the same {n_common:,} IMPROVE filters, 18 PLS components and
five site-grouped folds, every eligible filter predicted. Lowest median absolute error: {best['Median absolute error']}; lowest
mean absolute error and 95th percentile error: {best['Mean absolute error']}; lowest RMSE: {best['RMSE']}. EPO moves the {n_heavy} heavy,
low-OC/EC IMPROVE filters toward 1:1 (median FTIR/TOR {anch.improve_heavy_ratio:.2f} → {epo.improve_heavy_ratio:.2f}) at the cost of overall error
(MAE {anch.improve_mae:.2f} → {epo.improve_mae:.2f} µg/filter). Descriptive differences, not significance tests; prior exploratory settings mean this is
a retrospective comparison, not independent model selection."""

# ---------------------------------------------------------------- 5. where the time went
dates = pd.to_datetime(pd.Series(git("log", f"--since={WEEKS_FROM}", f"--until={WEEKS_TO}", "--format=%ad", "--date=short").split()))
week0 = pd.Timestamp(WEEKS_FROM)
weeks = pd.date_range(week0, pd.Timestamp(WEEKS_TO) - pd.Timedelta(days=1), freq="7D")
counts = ((dates - week0).dt.days // 7).value_counts()
commits = pd.DataFrame({"week": [f"{w:%b} {w.day}" for w in weeks],
                        "commits": [int(counts.get(i, 0)) for i in range(len(weeks))],
                        "kind": "Analysis and tooling"})
peak = commits.loc[commits.commits.idxmax()]
since_oct = int(commits.loc[weeks >= pd.Timestamp("2026-10-05"), "commits"].sum())
draft_suffixes = {".docx", ".doc", ".tex", ".qmd", ".md", ".rmd"}
drafts = [p for p in (ROOT / "manuscript").rglob("*")
          if p.is_file() and p.suffix.lower() in draft_suffixes and "citations" not in p.parts and p.name != "README.md"]
text_time = f"""**The analysis is closed; the writing has not started.** {since_oct} commits since 5 Oct, {peak.commits} of them in the week of
{peak.week}, almost all analysis, gallery and release work. {"A manuscript draft is in the repository." if drafts else
"The repository holds no manuscript draft: `manuscript/` has citation tooling only."} The supervisor's definition of finished is the
**chapter written and the paper ready**, so the next six weeks go to writing."""

# ---------------------------------------------------------------- 6. schedule and decisions
milestones = pd.DataFrame([
    ("Existing-data investigation closed out (reproducible release)", "7 Nov 2026", "Done"),
    ("AAAR poster presented", "late Oct 2026", "Done"),
    ("Paper 1 outline to Ann", "~20 Nov 2026", "Not started"),
    ("Paper 1 full draft to Ann", "early Dec 2026", "Not started"),
    ("Paper and thesis chapter to the supervisor", "~18 Dec 2026", "Not started"),
    ("End-of-semester status to the supervisor", "end Dec 2026", "On track"),
    ("Go/no-go on Addis and choice of next objective", "Jan 2027 at the latest", "On track"),
    ("Next committee meeting", "by ~Apr 2027", "Not started"),
    ("~10 paired FTIR + TOR filters from Addis city (Ann)", "next few months", "Waiting on others"),
    ("SPARTAN deployed EC model file (AQRC)", "requested 21 Oct", "Waiting on others"),
], columns=["milestone", "due", "status"])

claims = pd.DataFrame([
    ("Absolute EC accuracy at Addis", "No Addis filter has thermal (TOR) EC; HIPS / MAC 10 is an optical equivalent"),
    ("A near-zero aethalometer intercept proves correct EC", "Agreement between two optical methods, not a thermal reference"),
    ("A seasonal residual identifies its chemical cause", "The residuals are diagnostics; composition is the leading explanation"),
    ("One preprocessing method wins", "Each metric names a different best candidate; no significance test"),
    ("Three Adama TOR pairs validate a method", "Different site, lot and sampler; provisional mapping"),
    ("MAC 10 is the Addis value", "A convention; the MAC is part of the slope problem, the calibration the intercept"),
], columns=["do not claim", "why"])

text_next = """**Proposed position for the December–January decision: a defensible pathway exists.** Write paper 1 on the
existing-data result now: why an IMPROVE-trained FTIR EC calibration does not transfer to Addis, how much preprocessing and
cohort choice recover, the seasonal structure, and the compositional explanation, with absolute accuracy stated as the open
limit. The incoming Addis TOR filters then become the validation step, not a prerequisite. Agree what agreement would count as
evidence before they arrive.

Open analysis from the AAAR list (OC/EC-selected cohorts for Dry and Belg, the dust band, a season-specific MAC, optical pyrolysis
on dark IMPROVE filters) waits until the draft exists or goes to future work. The next objective (Brian's glacier black carbon
project, international collaborations across SPARTAN/MAIA, or the PurpleAir objective the committee still expects) is chosen in January."""

intro = f"""Where the existing-data FTIR EC investigation stands on 9 Nov 2026, after the 7 Nov closeout, for the end-of-December
check-in with the supervisor. A status page, not a recorded meeting: every chart reads an export that already exists
(the closeout source scoreboard and saved Addis fits, the 9 Oct and 28 Oct meeting bundles, and the repository history).
Source scores are µg per filter on {n_common:,} common IMPROVE filters; Addis comparisons are µg/m³ against optical comparators."""

write_bundle(
    MEETING_ID,
    generated_from="gallery/review/closeout_20261107/continuum_baseline.json, meetings/ann-20261009.json, "
                   "meetings/meeting-20261028.json and git history via gallery/data/export_meeting_20261109.py",
    colors=COLORS,
    text={"intro": intro, "domain": text_domain, "preprocess": text_preprocess, "seasons": text_seasons,
          "scores": text_scores, "time": text_time, "next": text_next},
    tables={"loading": loading, "ec_share": ec_share, "addis": addis, "cals": cals, "scores": scores,
            "scores_long": scores_long, "heavy": heavy, "commits": commits, "milestones": milestones, "claims": claims},
)
