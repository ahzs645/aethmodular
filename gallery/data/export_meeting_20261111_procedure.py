"""Export the 11 Nov 2026 page "HIPS on loaded filters: the procedure" for the gallery's Meeting tab.

    uv run python gallery/data/export_meeting_20261111_procedure.py

The end product of the HIPS work as four steps that can be run on any set of filters: identify the bend, trace
what it does to the numbers read off HIPS, correct it with a stated range, and check that the correction
worked. The steps are research/ftir_hips_chem/scripts/hips_loading.py; the command is `aeth hips screen`; the
protocol is docs/hips-loading-protocol.md. This page shows each step on every SPARTAN site, from
research/ftir_hips_chem/workflows/run_hips_network_20261111.py (committed files only, no aethalometer data).

Writes gallery/app/public/data/meetings/hips-procedure-20261111.json; figures in
src/meetings/hips-procedure-20261111.meeting.ts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research/ftir_hips_chem/workflows"), str(ROOT / "research/ftir_hips_chem/scripts")]

import hips_loading as hl  # noqa: E402
import run_hips_network_20261111 as hips  # noqa: E402
from meeting_bundle import write_bundle  # noqa: E402

MEETING_ID = "hips-procedure-20261111"
STATUS_COLORS = {"use, linearised": "#1b9e77", "caution": "#e6ab02", "shape only": "#a50f15"}
KIND_COLORS = {"Seen at the site": "#4c78a8", "Predicted by the curve": "#f58518", "Before correcting": "#a50f15", "After correcting": "#1b9e77"}
READING_COLORS = {"corrected": "#1b9e77", "under-corrected": "#a50f15", "over-corrected": "#4292c6"}

tables, top = hips.run(("procedure",))
found, traced, checked, screened = tables["identified"], tables["propagated"], tables["checked"], tables["screened"]
order = list(found["site"])
pct = lambda v: f"{100 * v:.0f} %"  # noqa: E731
at = found.set_index("code")
trace = traced.set_index(["code", "key"])
chk = checked.set_index("code")

# ---------------------------------------------------------------- the steps
steps = pd.DataFrame([
    ("1 Identify", "Is this filter, or this site, dark enough for HIPS to under-read, and do its own data show it?",
     "Per filter: optical depth = Fabs × volume ÷ 353 (the laboratory's HIPS tau) and the expected under-read, tau ÷ ceiling. "
     "Per site: the exponent of HIPS against FTIR EC with weighed PM2.5 as the instrument, and Fabs ÷ FTIR EC on the most loaded third of filters over the least loaded third.",
     "Fabs and volume. The site tests also need FTIR EC and PM2.5 on 40 filters.",
     f"A status per filter: use, linearised (up to {pct(hl.USE_UP_TO)}), caution (to {pct(hl.SHAPE_ONLY_FROM)}), shape only (beyond). A reading per site."),
    ("2 Trace", "Which numbers read off HIPS does it move, and by how much?",
     "Each number recomputed on linearised HIPS: median and mean BC, the seasonal contrast, BC's share of PM2.5, Fabs ÷ FTIR EC, and the slope and intercept of FTIR EC on HIPS BC.",
     "Step 1, a date, and for the last three FTIR EC or PM2.5.", "Reported and linearised side by side, with the range the ceiling's interval allows."),
    ("3 Correct", "What value should be used, and how sure is it?",
     "Fabs × 1 ÷ (1 − tau ÷ ceiling), evaluated at the ceiling and at both ends of its interval.",
     "Step 1.", "A linearised Fabs with a low and a high for every filter; filters past the low end of the ceiling get no high value."),
    ("4 Check", "Did the correction remove the bend at this site?",
     "The step 1 exponent again, on linearised HIPS. It should include 1. Below 1 the site is under-corrected, above 1 over-corrected.",
     "As the site tests of step 1.", "A verdict per site: corrected, under-corrected, over-corrected, or cannot tell."),
], columns=["step", "question", "computed", "needs", "gives"])

# ---------------------------------------------------------------- 1. identify
status = (screened.groupby(["site", "status"]).size().rename("filters").reset_index())
status["share"] = status["filters"] / status.groupby("site")["filters"].transform("sum")
grid = pd.MultiIndex.from_product([order, hl.STATUSES], names=["site", "status"])
status = status.set_index(["site", "status"])[["filters", "share"]].reindex(grid, fill_value=0).reset_index()
tested = found[found["ratio_heavy_over_light"].notna()]
ratios = pd.concat([
    tested.assign(kind="Seen at the site", value=tested["ratio_heavy_over_light"], lo=tested["ratio_lo"], hi=tested["ratio_hi"])[["site", "kind", "value", "lo", "hi", "reading"]],
    tested.assign(kind="Predicted by the curve", value=tested["ratio_predicted"], lo=np.nan, hi=np.nan)[["site", "kind", "value", "lo", "hi", "reading"]],
])
ratios["order"] = ratios["site"].map({s: i for i, s in enumerate(order)})
ratios = ratios.sort_values(["order", "kind"], ascending=[True, False]).drop(columns="order")
readings = found["reading"].value_counts()
seen = found[found["reading"] == "expected and seen"]
untestable = found[found["reading"] == "expected, cannot be tested"]
faster = found[found["reading"] == "HIPS rises faster than the reference"]
names = lambda frame: ", ".join(frame["site"].str.split(",").str[0])  # noqa: E731
network = screened["status"].value_counts(normalize=True)
text_identify = f"""**Two levels.** A filter is identified by its own optical depth, which needs nothing but Fabs and the sampled volume:
{pct(network['use, linearised'])} of the network's filters are *use, linearised*, {pct(network['caution'])} *caution* and {pct(network['shape only'])}
*shape only*. A site is identified by two tests on its own data that assume no curve. **What the tests found.** The bend is expected and seen
at {len(seen)} sites ({names(seen)}). It is expected at {len(untestable)} more that have too few filters with FTIR EC to test
({names(untestable)}). **A reading that points the other way.** At {len(faster)} light sites ({names(faster)}) HIPS rises *faster* than FTIR EC.
A bend in HIPS cannot produce that; an offset in FTIR EC on light filters would. It is why each test is read beside its
prediction and not alone."""

# ---------------------------------------------------------------- 2. trace
change = traced[traced["key"] != "intercept"].copy()
change["order"] = change["site"].map({s: i for i, s in enumerate(order)})
change = change.sort_values(["order", "quantity"]).drop(columns="order")
ranks = traced[traced["key"] == "median_bc"].copy()
ranks["rank_reported"] = ranks["reported"].rank(ascending=False).astype(int)
ranks["rank_linearised"] = ranks["linearised"].rank(ascending=False).astype(int)
ranks = ranks.sort_values("rank_linearised")
fits = traced[traced["key"].isin(["slope", "intercept"])].pivot(index=["code", "site"], columns="key", values=["reported", "linearised"])
fits.columns = [f"{b}_{a}" for a, b in fits.columns]
fits = fits.reset_index()
fits["order"] = fits["site"].map({s: i for i, s in enumerate(order)})
fits = fits.sort_values("order").drop(columns="order")
a_bc, d_bc = trace.loc[("ETAD", "median_bc")], trace.loc[("INDH", "median_bc")]
a_share, a_season = trace.loc[("ETAD", "bc_share_of_pm25")], trace.loc[("ETAD", "seasonal_contrast")]
text_trace = f"""**Everything read off HIPS moves, and not by the same amount.** At Addis the median BC goes from {a_bc.reported:.1f} to
{a_bc.linearised:.1f} µg/m³ [{a_bc.lo:.1f}, {a_bc.hi:.1f}], BC's share of PM2.5 from {pct(a_share.reported)} to {pct(a_share.linearised)}, and the
seasonal contrast from {a_season.reported:.2f} to {a_season.linearised:.2f}: the bend flattens seasons because it takes most from the darkest
months. **It changes the order of sites.** Reported, Delhi has the highest median BC ({d_bc.reported:.1f}) and Addis is third; linearised they are
level ({d_bc.linearised:.1f} and {a_bc.linearised:.1f}). **It bends straight-line fits.** A compressed HIPS on the x-axis gives a slope above 1 and
a negative intercept. At Addis the fit of FTIR EC on HIPS BC goes from {trace.loc[('ETAD', 'slope'), 'reported']:.2f}x − {abs(trace.loc[('ETAD', 'intercept'), 'reported']):.2f}
to {trace.loc[('ETAD', 'slope'), 'linearised']:.2f}x − {abs(trace.loc[('ETAD', 'intercept'), 'linearised']):.2f}; at Delhi from
{trace.loc[('INDH', 'slope'), 'reported']:.2f}x − {abs(trace.loc[('INDH', 'intercept'), 'reported']):.2f} to {trace.loc[('INDH', 'slope'), 'linearised']:.2f}x −
{abs(trace.loc[('INDH', 'intercept'), 'linearised']):.2f}. **Means move more than medians**, because a few very dark filters carry most of the correction."""

# ---------------------------------------------------------------- 3. correct
by_status = []
for name in hl.STATUSES:
    part = screened[screened["status"] == name]
    tau = float(part["tau"].median())
    f_c, f_lo, f_hi = (float(hl.factor(tau, t)) for t in (top["central"], top["hi"], top["lo"]))
    by_status.append({"status": name, "under_read": {"use, linearised": f"up to {pct(hl.USE_UP_TO)}", "caution": f"{pct(hl.USE_UP_TO)} to {pct(hl.SHAPE_ONLY_FROM)}",
                                                     "shape only": f"over {pct(hl.SHAPE_ONLY_FROM)}"}[name],
                      "tau": {"use, linearised": f"up to {hl.USE_UP_TO * top['central']:.2f}", "caution": f"to {hl.SHAPE_ONLY_FROM * top['central']:.2f}",
                              "shape only": f"over {hl.SHAPE_ONLY_FROM * top['central']:.2f}"}[name],
                      "filters": len(part), "share": len(part) / len(screened), "factor": f_c, "factor_lo": f_lo, "factor_hi": f_hi,
                      "plus_minus": (f_hi - f_lo) / (2 * f_c),
                      "use": {"use, linearised": "Level and shape. Quote the linearised value with its range.",
                              "caution": "Shape, and level only with the range stated.",
                              "shape only": "Shape only: rank order, timing, season. Not a level."}[name]})
by_status = pd.DataFrame(by_status)
points = screened.assign(date=screened["date"].dt.date.astype(str))
points = points[["id", "site", "date", "fabs", "volume", "tau", "under_read", "status", "fabs_linearised", "fabs_linearised_lo", "fabs_linearised_hi"]]
points = points[np.isfinite(points["fabs_linearised"])]
st = by_status.set_index("status")
text_correct = f"""**One formula, three levels of trust.** Linearised Fabs = Fabs ÷ (1 − tau ÷ ceiling). What differs between filters is how much
the answer depends on the ceiling, which is known as {top['central']:.2f} [{top['lo']:.2f}, {top['hi']:.2f}]. For a *use, linearised* filter the factor is
{st.loc['use, linearised', 'factor']:.2f} ± {pct(st.loc['use, linearised', 'plus_minus'])}; for *caution*, {st.loc['caution', 'factor']:.2f} ± {pct(st.loc['caution', 'plus_minus'])};
for *shape only*, {st.loc['shape only', 'factor']:.2f} ± {pct(st.loc['shape only', 'plus_minus'])}. **So the correction is always computed and the status says what it
may be used for.** Nothing is deleted. **Mitigation comes before correction where it is possible:** less air through the filter lowers the
optical depth itself (the Nov 11 network page, section 2)."""

# ---------------------------------------------------------------- 4. check
told = checked[checked["verdict"].isin(["corrected", "under-corrected", "over-corrected"])]
after = pd.concat([
    told.assign(kind="Before correcting", value=told["exponent_before"], lo=told["before_lo"], hi=told["before_hi"])[["site", "kind", "value", "lo", "hi", "verdict", "median_tau"]],
    told.assign(kind="After correcting", value=told["exponent_after"], lo=told["after_lo"], hi=told["after_hi"])[["site", "kind", "value", "lo", "hi", "verdict", "median_tau"]],
]).sort_values(["median_tau", "kind"], ascending=[False, False])
verdicts = checked["verdict"].value_counts()
group = lambda name: names(checked[checked["verdict"] == name])  # noqa: E731
text_check = f"""**The acceptance test.** After linearising, HIPS should be proportional to the reference: an exponent whose interval
includes 1. {len(told)} sites can be read. **Corrected** at {int(verdicts.get('corrected', 0))}: {group('corrected')}. **Under-corrected** at
{int(verdicts.get('under-corrected', 0))}: {group('under-corrected')}; Addis goes from {chk.loc['ETAD', 'exponent_before']:.2f} to
{chk.loc['ETAD', 'exponent_after']:.2f} [{chk.loc['ETAD', 'after_lo']:.2f}, {chk.loc['ETAD', 'after_hi']:.2f}] and Delhi from {chk.loc['INDH', 'exponent_before']:.2f} to
{chk.loc['INDH', 'exponent_after']:.2f} [{chk.loc['INDH', 'after_lo']:.2f}, {chk.loc['INDH', 'after_hi']:.2f}], so most of the compression goes but not all.
**Over-corrected** at {int(verdicts.get('over-corrected', 0))}: {group('over-corrected')}. **What a failed check means.** Either the site's filters bend
differently from IMPROVE's, or FTIR EC is not proportional to the absorber there. The check cannot say which, because FTIR EC is not
thermal EC. It does say where a single ceiling should not be trusted for level: the under- and over-corrected sites."""

# ---------------------------------------------------------------- what comes out
outputs = pd.DataFrame([
    ("filters.csv", "One row per filter", "tau, under_read (with low and high), status, beyond_reference, factor, fabs_linearised (with low and high)",
     "The corrected value and what it may be used for. Join it to any analysis that uses HIPS."),
    ("sites.csv", "One row per site", "Shares by status, the two site tests beside their predictions, the reading, and the check's verdict",
     "Which sites to treat as corrected, and which to report as shape only."),
    ("propagation.csv", "One row per site and quantity", "Reported, linearised, low, high for each number read off HIPS",
     "What to quote, and by how much a result already reported would move."),
], columns=["file", "rows", "columns", "use"])
revisit = pd.DataFrame([
    ("A new HIPS delivery", "Run `aeth hips screen`. New filters get a status the day they arrive."),
    ("A sampled volume changes at a site", "Rerun; expect a step in reported Fabs that the linearised series should not show."),
    ("Thermal EC arrives for a heavy site (the Addis TOR filters)", "Fit that site's own ceiling against thermal EC and compare with IMPROVE's. This is the only thing that can settle the under-corrected sites."),
    ("The IMPROVE ceiling is re-estimated", "Change the one constant in scripts/hips_loading.py; a test fails until the two agree."),
    ("A site fails the check", "Report its HIPS as shape only for level, whatever its filters' statuses."),
], columns=["when", "what to do"])

text_intro = f"""**What this page is.** The HIPS finding turned into a procedure that can be run on any set of filters, now or when new data
arrive: **identify** the bend, **trace** what it does to the numbers read off HIPS, **correct** it with a stated range, and **check** that
the correction worked. One command runs all four (`aeth hips screen`) and writes three tables. **What it does for us now.** Across
{len(screened):,} SPARTAN filters it marks {pct(network['use, linearised'])} as usable once linearised, {pct(network['caution'])} as caution and
{pct(network['shape only'])} as shape only; it shows the bend in the data of {len(seen)} sites; and its own check passes at {int(verdicts.get('corrected', 0))} of the
{len(told)} sites that can be read. The ceiling is IMPROVE's, measured against thermal EC. HIPS is optical, FTIR EC is FTIR-derived,
no SPARTAN filter here has thermal EC, and no aethalometer data are used."""

write_bundle(
    MEETING_ID,
    generated_from="research/ftir_hips_chem/scripts/hips_loading.py via workflows/run_hips_network_20261111.py",
    tables={"steps": steps, "status": status, "found": found, "ratios": ratios, "change": change, "traced": traced, "ranks": ranks, "fits": fits,
            "by_status": by_status, "points": points, "after": after, "checked": checked, "outputs": outputs, "revisit": revisit},
    text={"intro": text_intro, "identify": text_identify, "trace": text_trace, "correct": text_correct, "check": text_check},
    colors=STATUS_COLORS | KIND_COLORS | READING_COLORS,
)
