"""Export the 11 Nov 2026 page "HIPS across the network" for the gallery's Meeting tab.

    uv run python gallery/data/export_meeting_20261111.py

Applies the 10 Nov finding (HIPS reads low as a filter darkens, with the ceiling measured on IMPROVE against
thermal EC) to every SPARTAN filter: which sites and months are affected, how much is the air and how much
the sampled volume, which filters form the affected group, whether each site's own data show the bend, and
what that leaves us able to do. Reruns research/ftir_hips_chem/workflows/run_hips_network_20261111.py, which
reads only committed files. No aethalometer data.

Writes gallery/app/public/data/meetings/hips-network-20261111.json; figures in
src/meetings/hips-network-20261111.meeting.ts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research/ftir_hips_chem/workflows")]

import run_hips_network_20261111 as hips  # noqa: E402
from meeting_bundle import write_bundle  # noqa: E402

MEETING_ID = "hips-network-20261111"
BAND_COLORS = {"under 5 %": "#9ecae1", "5 to 10 %": "#4292c6", "10 to 20 %": "#fdae6b", "20 to 30 %": "#e6550d", "over 30 %": "#a50f15"}
REGIME_COLORS = {"1.8 m³": "#1b9e77", "3.6 m³": "#7570b3", "7.2 m³": "#d95f02", "14 m³ or more": "#666666"}
READING_COLORS = {"as predicted": "#1b9e77", "more compressed than predicted": "#a50f15", "less compressed than predicted": "#4292c6", "cannot tell": "#999999"}

tables, top = hips.run()
f, sites = tables["filters"], tables["sites"]
order = list(sites["site"])
rank = {name: i for i, name in enumerate(order)}
pct = lambda v: f"{100 * v:.0f} %"  # noqa: E731
site = sites.set_index("code")
addis, delhi, jaipur = site.loc["ETAD"], site.loc["INDH"], site.loc["INJA"]

# ---------------------------------------------------------------- 1. where it bites
heavy = sites[sites["under_read"] > 0.20]
light = sites[sites["under_read"] < 0.10]
text_sites = f"""**What this is.** HIPS reports a share *optical depth ÷ ceiling* less absorption than the filter carries. The ceiling,
{top['central']:.2f} [{top['lo']:.2f}, {top['hi']:.2f}], was measured on IMPROVE filters against thermal EC. The optical depth is the
laboratory's own HIPS tau (Fabs × sampled volume ÷ 353). **What it shows.** {len(heavy)} of {len(sites)} sites lose more than a fifth on
their median filter, led by Addis Ababa at {pct(addis.under_read)} [{pct(addis.under_read_lo)}, {pct(addis.under_read_hi)}]. {len(light)}
sites lose under a tenth. Across {len(f):,} filters, {pct((f.under_read > 0.10).mean())} lose more than a tenth, {pct((f.under_read > 0.20).mean())}
more than a fifth and {pct((f.under_read > 0.30).mean())} more than three tenths. **What it rests on.** The curve is IMPROVE's. SPARTAN has no
thermal EC to check it against, so these are predictions for SPARTAN filters, tested as far as SPARTAN data allow in section 5."""

# ---------------------------------------------------------------- 2. air or volume
replay = tables["volume_replay"].set_index("code")
matched = tables["volume_matched"]
both = matched.iloc[-1]
regimes = tables["volume_regimes"].copy()
delhi_cold = f[(f.code == "INDH") & f.month.isin([10, 11, 12, 1]) & (f.regime == "1.8 m³")]
delhi_cold_full = float(hips.under_read(pd.Series(hips.at_volume(delhi_cold.tau, delhi_cold.volume, 7.2, top["central"])).median(), top["central"]))
longer = regimes[regimes["regime"] == "14 m³ or more"].set_index("code")
# years as 22.5 for mid-2022: the chart's axis writes four-digit numbers rounded
points = f.assign(year=f["date"].dt.year - 2000 + (f["date"].dt.dayofyear - 1) / 365.25, date=f["date"].dt.date.astype(str))
points = points[["id", "code", "site", "date", "year", "fabs", "volume", "regime", "tau", "under_read", "band"]]
text_volume = f"""**Both, and they pull apart.** Optical depth is the air's absorption times the volume drawn through the filter. Between
sites the air does almost all of it: {int((sites.median_volume.between(6.5, 8)).sum())} of {len(sites)} sites sample about 7.2 m³. Delhi and
Jaipur are the exception and the lesson. Their air absorbs as much as Addis's ({delhi.linearised_fabs:.0f} and {jaipur.linearised_fabs:.0f}
against {addis.linearised_fabs:.0f} Mm⁻¹ once linearised), but they sample half the volume or less, so their median filter loses
{pct(delhi.under_read)} and {pct(jaipur.under_read)} against {pct(addis.under_read)}. At 7.2 m³ Delhi would lose {pct(replay.loc['INDH', 'at_7.2'])},
the same as Addis. **Delhi's volume was cut twice**, from 7 to 3.6 to 1.8 m³; Addis has stayed at 7.2 m³ throughout. Delhi's October to January
filters at 1.8 m³ lose {pct(delhi_cold.under_read.median())}; the same air at 7.2 m³ would lose {pct(delhi_cold_full)}. At 3.6 m³ Addis's
median filter would lose {pct(replay.loc['ETAD', 'at_3.6'])}, and at 1.8 m³ {pct(replay.loc['ETAD', 'at_1.8'])}. **A check that uses no FTIR.**
In the same calendar months, Delhi filters sampled at 1.8 m³ report {both.observed_ratio:.2f} [{both.lo:.2f}, {both.hi:.2f}] times the
absorption per unit of weighed PM2.5 that filters sampled at 3.6 m³ do; the curve predicts {both.predicted_ratio:.2f}. The two volumes
fall in different years, which this cannot separate. **Volume also works the other way.** Clean sites that moved to long samples carry more
on the filter: Norman's 44 m³ filters lose {pct(longer.loc['USNO', 'under_read'])} and Melbourne's 29 m³ filters {pct(longer.loc['AUMN', 'under_read'])}."""

# ---------------------------------------------------------------- 3. season
delhi_filters = f[f.code == "INDH"]
delhi_winter = delhi_filters[delhi_filters.month.isin([10, 11, 12, 1])]
months = tables["months"].copy()
months["order"] = months["site"].map(rank)
months = months.sort_values(["order", "month"]).drop(columns="order")
swing = tables["season_swing"]
big = swing[swing["swing"] >= 0.15]
row = swing.set_index("code")
text_season = f"""**Yes, where the air is seasonal.** {len(big)} sites swing by 15 points or more between their best and worst month.
Santiago goes from {pct(row.loc['CLST', 'lowest'])} in {row.loc['CLST', 'best_month']} to {pct(row.loc['CLST', 'highest'])} in
{row.loc['CLST', 'worst_month']}, Pretoria from {pct(row.loc['ZAPR', 'lowest'])} to {pct(row.loc['ZAPR', 'highest'])} and Johannesburg from
{pct(row.loc['ZAJB', 'lowest'])} to {pct(row.loc['ZAJB', 'highest'])}, all in the southern winter. Delhi goes from {pct(row.loc['INDH', 'lowest'])}
in {row.loc['INDH', 'best_month']} to {pct(delhi_winter.under_read.median())} from October to January, even at its reduced volume.
**Addis is different: it never comes down.** Its best month loses {pct(row.loc['ETAD', 'lowest'])} and its worst, {row.loc['ETAD', 'worst_month']},
{pct(row.loc['ETAD', 'highest'])}. So at most sites a season can be set aside; at Addis no season is clean. **Read with the volume in mind.**
Jaipur's December includes filters sampled at 6.6 m³, so its swing is part season and part volume."""

# ---------------------------------------------------------------- 4. the affected group
bands = tables["bands"].copy()
band = bands.set_index("band")
# every site in every band (0 where it has none), so the grid's rows and columns come out in order
grid = pd.MultiIndex.from_product([order, [b[0] for b in hips.BANDS]], names=["site", "band"])
by_site = tables["bands_by_site"].set_index(["site", "band"])[["filters", "share"]].reindex(grid, fill_value=0).reset_index()
safe = bands[bands["band"].isin(["under 5 %", "5 to 10 %", "10 to 20 %"])]
worst = band.loc["over 30 %"]
addis_filters = f[f.code == "ETAD"]
text_group = f"""**The group is defined by the filter, not the site.** Any filter's expected under-read follows from its own optical
depth, so the affected group can be picked out anywhere: optical depth above {band.loc['20 to 30 %', 'tau_from']:.2f} loses more than a
fifth, above {worst.tau_from:.2f} more than three tenths. **How well the correction is known decides what each band is good for.** Up to a
fifth, the correction factor moves by ±{pct(band.loc['10 to 20 %', 'plus_minus'])} across the ceiling's interval, and by
{pct(band.loc['10 to 20 %', 'by_period'])} between IMPROVE's two half-periods; that covers {pct(safe.share_of_network.sum())} of the network's
filters. Above three tenths it moves by ±{pct(worst.plus_minus)} and {pct(worst.by_period)}, and {pct(worst.beyond_improve)} of those filters
are darker than any IMPROVE stratum the curve was fitted on. **Addis sits in that last band:** {pct((addis_filters.under_read > 0.30).mean())}
of its filters, {int((addis_filters.under_read > 0.30).sum())} of the {int(worst.filters)} in it."""

# ---------------------------------------------------------------- 5. each site's own evidence
evidence = tables["evidence"].copy()
tally = tables["evidence_tally"].set_index("reading")
agree = tables["evidence_agreement"].iloc[0]
ev = evidence.set_index("code")
told = evidence[evidence["reading"] != "cannot tell"]
# observed and predicted on one axis, for the sites whose interval is narrow enough to read
exponents = pd.concat([
    told.assign(kind="Observed against FTIR EC", value=told["observed"])[["site", "kind", "value", "lo", "hi", "n", "median_tau", "reading"]],
    told.assign(kind="Predicted from the IMPROVE curve", value=told["predicted"], lo=told["predicted_lo"], hi=told["predicted_hi"])[
        ["site", "kind", "value", "lo", "hi", "n", "median_tau", "reading"]],
]).sort_values(["median_tau", "kind"], ascending=[False, True])
count = lambda name: int(tally.loc[name, "sites"]) if name in tally.index else 0  # noqa: E731
text_evidence = f"""**The test.** If HIPS bends, a site's HIPS should rise more slowly than its FTIR EC, by an exponent the curve
predicts for that site's filters. The exponent uses weighed PM2.5 as the instrument, so random error in FTIR EC does not flatten it.
The prediction is the exponent of reported against linearised HIPS, so it allows for filters sampled at different volumes.
**What it shows.** {len(told)} of {len(evidence)} sites have an interval narrow enough to tell. {count('as predicted')} match the prediction.
{count('more compressed than predicted')} are more compressed than predicted: Addis Ababa, the two Chilean sites, and Delhi by a little
({ev.loc['INDH', 'observed']:.2f} [{ev.loc['INDH', 'lo']:.2f}, {ev.loc['INDH', 'hi']:.2f}] against {ev.loc['INDH', 'predicted']:.2f}).
{count('less compressed than predicted')} are less (Johannesburg, Taipei). Across the {len(told)}, observed follows predicted with a correlation
of {agree.correlation:.2f}. **What it does not show.** That one ceiling fits every site. FTIR EC is not thermal EC (our refit of it carries
an offset on light filters), the exponent cannot separate a HIPS bend from an FTIR one, and where EC is a small or unsteady share of
PM2.5 the instrument is weak. The direction holds across the network; the size is not confirmed site by site."""

# ---------------------------------------------------------------- 6. what to do
share = lambda mask: pct(mask.mean())  # noqa: E731
keep = f["under_read"] <= 0.20
options = pd.DataFrame([
    ("Leave HIPS as reported", "Fabs ÷ 10 everywhere.",
     f"Simple. Fine at the {len(light)} sites that lose under a tenth.",
     f"{share(f.under_read > 0.10)} of filters read more than a tenth low and {share(f.under_read > 0.30)} more than three tenths. Sites are "
     f"mis-ranked: Delhi reports {delhi.median_fabs:.0f} Mm⁻¹ against Addis's {addis.median_fabs:.0f}, but linearised they are "
     f"{delhi.linearised_fabs:.0f} and {addis.linearised_fabs:.0f}.", "Not for heavy sites"),
    ("Carry the optical depth with every HIPS value", "Report tau and the expected under-read beside Fabs. No value changes.",
     "Costs nothing; tau is the laboratory's own number. Every reader can see which filters to trust.",
     "Corrects nothing by itself.", "Do it"),
    ("Linearise filters that lose up to a fifth; flag the rest", "Correct where tau ≤ 0.45; mark darker filters as not usable for level.",
     f"Covers {share(keep)} of filters. The correction is known to ±{pct(band.loc['10 to 20 %', 'plus_minus'])} "
     f"({pct(band.loc['10 to 20 %', 'by_period'])} between IMPROVE's half-periods).",
     f"Leaves out every Addis filter, {share(~keep[f.code == 'INDH'])} of Delhi's and {share(~keep[f.code == 'BDDU'])} of Dhaka's.",
     "Use for comparisons with FTIR EC"),
    ("Linearise every filter with one ceiling", "One corrected HIPS product for the network.",
     "Removes the bend on average, and moves the Addis fit from 2.31x − 6.17 to 0.55x + 0.01.",
     f"Above three tenths the factor is uncertain by ±{pct(worst.plus_minus)} ({pct(worst.by_period)} between half-periods). Only "
     f"{count('as predicted')} of {len(told)} sites match the curve, and SPARTAN has no thermal EC to check it.",
     "Not as a product; as a stated sensitivity"),
    ("Sample less air at the heavy sites", "Ask SPARTAN for 3.6 or 1.8 m³ at Addis, as at Delhi.",
     f"Removes the problem instead of modelling it: Addis falls from {pct(addis.under_read)} to {pct(replay.loc['ETAD', 'at_3.6'])} at 3.6 m³ and "
     f"{pct(replay.loc['ETAD', 'at_1.8'])} at 1.8 m³. Delhi's filters at two volumes are consistent with it.",
     "Only helps new filters. Less mass on the filter for every other analysis, and a break in the record at the change.", "Ask"),
    ("Fit each site its own curve", "A ceiling per site from its own HIPS and FTIR EC.",
     "Would absorb real differences between sites.",
     "Needs a reference that is not FTIR EC, or the fit tunes HIPS to the thing it is meant to check.", "Not with the data on file"),
], columns=["option", "what", "for", "against", "verdict"])
text_decision = f"""**What we can do with the data as it stands.** Three things hold on the evidence. HIPS under-reads in proportion to the
filter's optical depth; the size is known well up to about a fifth and poorly beyond three tenths; and the bend is a property of the
loaded filter, so sampled volume moves it. **So:** carry the optical depth with every HIPS value; when HIPS is the yardstick for FTIR
EC, use the {share(keep)} of filters that lose a fifth or less, linearised; and treat darker filters as evidence about shape, not level.
**For Addis that means** HIPS cannot set the FTIR EC level on the filters on file, because none qualifies. It still shows that the
published slope and intercept come from the bend. **For Delhi** {share(keep[f.code == 'INDH'])} of filters qualify,
{share(~delhi_filters[delhi_filters.under_read <= 0.20].month.isin([10, 11, 12, 1]))} of them outside October to January. **The request that would change this** is a lower sampled volume at Addis."""

text_intro = f"""**The question.** The 10 Nov work showed that HIPS reads low on dark filters and that this, not FTIR, produces most of
the Addis disagreement. Does that apply across SPARTAN, to which sites, seasons and filters, and what does it let us do? **The short
answers.** It applies wherever filters are dark: {len(heavy)} of {len(sites)} sites lose more than a fifth on their median filter. Addis is
the worst because it pairs heavy air with the full 7.2 m³; Delhi's air is as heavy but its volume was cut, and it loses half as much.
The effect is seasonal at most heavy sites and constant at Addis. The correction is usable up to about a fifth and uncertain beyond
three tenths. No aethalometer data are used. FTIR EC is FTIR-derived, HIPS is optical, and no SPARTAN filter here has thermal EC."""

write_bundle(
    MEETING_ID,
    generated_from="research/ftir_hips_chem/workflows/run_hips_network_20261111.py",
    tables={
        "sites": sites, "points": points, "replay": tables["volume_replay"], "regimes": regimes, "matched": matched,
        "months": months, "swing": swing, "bands": bands, "bands_by_site": by_site,
        "evidence": evidence, "told": told, "exponents": exponents, "tally": tables["evidence_tally"], "options": options,
    },
    text={"intro": text_intro, "sites": text_sites, "volume": text_volume, "season": text_season, "group": text_group,
          "evidence": text_evidence, "decision": text_decision},
    colors=BAND_COLORS | REGIME_COLORS | READING_COLORS,
)
