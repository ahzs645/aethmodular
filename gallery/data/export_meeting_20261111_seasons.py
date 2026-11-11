"""Export the 11 Nov 2026 page "HIPS and seasons" for the gallery's Meeting tab.

    uv run python gallery/data/export_meeting_20261111_seasons.py

Two questions: should the HIPS correction be applied per site, per season or per filter, and do the season
calendars themselves hold up against the climate? Reruns
research/ftir_hips_chem/workflows/run_hips_seasons_20261111.py, which reads committed files only: the gallery's
SPARTAN table, the season calendars (src/siteSeasons.network.json) and the 1991-2020 ERA5 normals
(public/data/weather/normals_1991_2020.json, from fetch_climate_normals_era5.py). No aethalometer data.

Writes gallery/app/public/data/meetings/hips-seasons-20261111.json; figures in
src/meetings/hips-seasons-20261111.meeting.ts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "research/ftir_hips_chem/workflows")]

import run_hips_seasons_20261111 as seasons  # noqa: E402
from meeting_bundle import write_bundle  # noqa: E402

MEETING_ID = "hips-seasons-20261111"
COLORS = {"After correcting, all seasons pooled": "#a50f15", "After correcting, within seasons": "#1b9e77",
          "source": "#1b9e77", "regional": "#4c78a8", "Rain (mm/day)": "#4c78a8", "Temperature (°C)": "#e6550d"}

tables, top = seasons.run()
levels, by_site, variance = tables["levels"], tables["levels_by_site"], tables["variance"]
site_seasons, within, tally = tables["site_seasons"], tables["within"], tables["within_tally"]
calendars, alternatives = tables["calendars"], tables["calendar_alternatives"]
pct = lambda v: f"{100 * v:.0f} %"  # noqa: E731
names = lambda frame: ", ".join(frame["site"].str.split(",").str[0])  # noqa: E731

# ---------------------------------------------------------------- 1. site, season or filter
heavy_scope = levels["scope"].iloc[-1]
lv = levels.set_index(["scope", "level"])
var = variance.set_index(["scope", "part"])["share"]
heavy_var = variance["scope"].iloc[-1]
one = lambda level, column="median": lv.loc[(heavy_scope, level), column]  # noqa: E731
text_levels = f"""**The correction belongs to the filter.** The factor depends on the filter's own optical depth, so the question is
how much is lost by using one factor for a group. At the sites that lose more than a tenth, one factor per site is wrong by {pct(one('One per site'))}
on the typical filter, one per site and season by {pct(one('One per site and season'))}, one per site and month by {pct(one('One per site and month'))}.
A season factor still misses by more than a tenth on {pct(one('One per site and season', 'share_over_10'))} of filters. For scale, the ceiling's
own uncertainty is {pct(one("The ceiling's own uncertainty, per filter"))}. **Why seasons do not capture it.** At those sites {pct(var[(heavy_var, 'Between filters in a season')])}
of the spread in the correction is between filters of the same site and season, {pct(var[(heavy_var, 'Between sites')])} between sites and only
{pct(var[(heavy_var, 'Between seasons at a site')])} between seasons. A season is a mix of light and dark filters, and at Delhi, Jaipur, Dhaka and
Bujumbura of filters sampled at different volumes. **So:** correct each filter. The per-filter factor costs nothing extra, because the optical depth is
already on file for every filter."""

# ---------------------------------------------------------------- 2. does it hold season by season
w = within.set_index("code")
long = pd.concat([
    within.assign(kind="After correcting, all seasons pooled", value=within["pooled_after"], lo=within["pooled_lo"], hi=within["pooled_hi"]),
    within.assign(kind="After correcting, within seasons", value=within["within_after"], lo=within["within_lo"], hi=within["within_hi"]),
])[["site", "kind", "value", "lo", "hi", "reading", "median_tau"]].sort_values(["median_tau", "kind"], ascending=[False, True])
group = lambda reading: names(within[within["reading"] == reading])  # noqa: E731
ss = site_seasons.set_index(["code", "season"])
addis = site_seasons[site_seasons["code"] == "ETAD"].set_index("season")
bega, kiremt = addis.loc["Bega (Oct–Jan)"], addis.loc["Kiremt (Jun–Sep)"]
differs = within[within["reading"] == "differs by season"]
failing = "; ".join(f"{r.site.split(',')[0]} ({r.seasons_failing.lower()})" for r in differs.itertuples())
text_within = f"""**Seasons are where to check the correction, not where to apply it.** The check of the procedure page (is linearised HIPS
proportional to FTIR EC?) is run three ways: with all of a site's filters pooled, with each season's own level taken out, and season by
season where a season has 30 filters. **Addis passes within every season.** Bega {bega.exponent_after:.2f} [{bega.after_lo:.2f}, {bega.after_hi:.2f}],
Belg {addis.loc['Belg (Feb–May)', 'exponent_after']:.2f}, Kiremt {kiremt.exponent_after:.2f} [{kiremt.after_lo:.2f}, {kiremt.after_hi:.2f}]; within seasons
together {w.loc['ETAD', 'within_after']:.2f} [{w.loc['ETAD', 'within_lo']:.2f}, {w.loc['ETAD', 'within_hi']:.2f}]. Pooled it fails
({w.loc['ETAD', 'pooled_after']:.2f}). **What is left at Addis is a difference between seasons, not the bend.** After correcting, HIPS per unit of
FTIR EC is {bega.fabs_per_ec_linearised:.1f} m²/g in Bega and {kiremt.fabs_per_ec_linearised:.1f} in Kiremt, a ratio of {w.loc['ETAD', 'season_spread_linearised']:.2f}
(it was {w.loc['ETAD', 'season_spread_reported']:.2f} before, so the per-filter correction removes about half). **At {len(differs)} sites a season fails
while the site passes on average:** {failing}. A pooled check hides these. **Where it stays compressed within seasons** ({group('still compressed within seasons')}),
season is not the explanation: the Chilean sites stay compressed in both seasons, and Talca's filters are too light for loading to be the cause.
Delhi is on the line ({w.loc['INDH', 'within_after']:.2f} [{w.loc['INDH', 'within_lo']:.2f}, {w.loc['INDH', 'within_hi']:.2f}]). **What a season difference
could be:** a real change in how much the aerosol absorbs per unit EC, or FTIR EC reading differently by season. Neither is a HIPS loading
effect, so neither belongs in the HIPS correction."""

# ---------------------------------------------------------------- 3. calendars against the climate
structure = tables["calendar_structure"].iloc[0]
scored = calendars[calendars["together"].notna()].copy()
scored["checked"] = scored["rain_labels_hold"].map({True: "hold", False: "do not hold"}).fillna("no wet or dry names")
scored["temperature_checked"] = scored["temperature_labels_hold"].map({True: "hold", False: "do not hold"}).fillna("no cold or warm names")
scored = scored.sort_values("headroom", ascending=False)
hips_sites = scored[scored["hips_filters"] > 0]
alt = alternatives.set_index(["code", "calendar"])
normals = json.loads(seasons.NORMALS.read_text())
climate = pd.DataFrame([{"site": entry["site"], "code": code, "month": m, "month_name": pd.Timestamp(2000, m, 1).strftime("%b"), "variable": label, "value": entry[key][m - 1]}
                        for code, entry in normals["sites"].items() for m in range(1, 13)
                        for label, key in (("Rain (mm/day)", "precipitation"), ("Temperature (°C)", "temperature"))])
in_use = seasons.load_calendars()
climate["season"] = [next(iter(in_use[c]["calendars"].values()))[m] if c in in_use else "" for c, m in zip(climate["code"], climate["month"])]
failing = scored[scored["rain_labels_hold"] == False]  # noqa: E712
text_calendars = f"""**Structure: nothing to fix.** All {int(structure.sites_with_a_calendar)} calendars cover each month exactly once, every site
with HIPS has one, and the Python calendars and the gallery's agree. **New: every calendar scored against the climate.** The calendars are sourced
from agencies and papers, but until now nothing in the repo measured them against weather over a standard period: the gallery's weather covers
each site's sampling period, under a year at {structure.hips_sites_with_under_a_year_of_gallery_weather}. This adds 1991–2020 ERA5 normals for
{int(structure.sites_with_normals)} sites and asks how much of the year's cycle in rain and temperature falls between a calendar's seasons.
**Addis holds.** Its calendar in use (Bega Oct–Jan, Belg Feb–May, Kiremt Jun–Sep) is the best three-season split there is
({alt.loc[('ETAD', 'belg_feb'), 'together']:.2f}); the older one with February in the dry season scores {alt.loc[('ETAD', 'dry_feb'), 'together']:.2f}, better
on rain and worse on temperature. **Candidates for a second look.** The three Indian sites: the alternative with December in winter fits better at
Kanpur ({alt.loc[('INKA', 'dec_winter'), 'together']:.2f} against {alt.loc[('INKA', 'imd_national'), 'together']:.2f}) and Jaipur
({alt.loc[('INJA', 'imd_jaipur'), 'together']:.2f} against {alt.loc[('INJA', 'imd_national'), 'together']:.2f}), and Delhi's older research calendar
(winter Nov–Feb) also scores above the one in use ({alt.loc[('INDH', 'research_2026_04'), 'together']:.2f} against {alt.loc[('INDH', 'imd_national'), 'together']:.2f}).
Kaohsiung fits standard quarters better than its shifted ones ({alt.loc[('TWKA', 'outlook'), 'together']:.2f} against {alt.loc[('TWKA', 'review'), 'together']:.2f});
Taipei the reverse. Bujumbura is the weakest of the sites with HIPS ({alt.loc[('BIBU', 'ndamuzi'), 'together']:.2f}), and its one-month "short dry season"
is not drier than its rains in ERA5. **What the score is not.** It does not replace a national definition, and ERA5 rain is least reliable in mountains and on coasts. A low
score at a mid-latitude site mostly says rain has no season there."""

# ---------------------------------------------------------------- 4. seasons and the HIPS cycle
cycle = calendars[calendars["hips_cycle_between_seasons"].notna()].sort_values("hips_cycle_range", ascending=False)
c = cycle.set_index("code")
wet_heavy = cycle[(cycle["hips_follows_rain"] > 0.5) & (cycle["hips_cycle_range"] >= 0.08)]
dry_heavy = cycle[(cycle["hips_follows_rain"] < -0.5) & (cycle["hips_cycle_range"] >= 0.08)]
text_cycle = f"""**The calendars do carry most of the monthly cycle in filter darkness** where there is one: {pct(c.loc['ETAD', 'hips_cycle_between_seasons'])} at Addis,
{pct(c.loc['INDH', 'hips_cycle_between_seasons'])} at Delhi, {pct(c.loc['ZAPR', 'hips_cycle_between_seasons'])} at Pretoria. They carry less at Jaipur
({pct(c.loc['INJA', 'hips_cycle_between_seasons'])}) and Johannesburg ({pct(c.loc['ZAJB', 'hips_cycle_between_seasons'])}). **But "wet" and "dry" do not mean
the same thing for HIPS everywhere.** At {len(dry_heavy)} sites the dark filters come in the dry months ({names(dry_heavy)}). At {len(wet_heavy)} they come in
the wet months ({names(wet_heavy)}): at Addis the heaviest filters are in Kiremt, the main rains. So a rule such as "the dry season is the affected
season" would be wrong at Addis. This is one more reason to identify by the filter and use the season only as a label."""

# ---------------------------------------------------------------- 5. what to do
actions = pd.DataFrame([
    ("Apply the HIPS correction per filter", "Do it",
     f"A season factor is wrong by {pct(one('One per site and season'))} on the typical filter at heavy sites and by more than a tenth on "
     f"{pct(one('One per site and season', 'share_over_10'))}; the per-filter factor is already available."),
    ("Run the check within each season as well as pooled", "Added to the procedure",
     f"At Addis the pooled check fails although the correction holds in every season; at {names(differs)} the site passes on average and one season fails. "
     "`aeth hips screen` now reports both."),
    ("Where one season fails the check, treat that season's HIPS as shape only", "Proposed",
     f"{failing}. The failure is in one season's aerosol or its FTIR EC, so it should not downgrade the site's other seasons."),
    ("Report HIPS per unit FTIR EC by season at Addis", "Report, do not correct",
     f"After correcting, Bega is {w.loc['ETAD', 'season_spread_linearised']:.2f} times Kiremt. That is a season effect in absorption or in FTIR EC, not in HIPS loading."),
    ("Keep the sourced calendars as the labels", "No change",
     "Structure is sound and Addis's calendar in use is the best three-season split of the climate."),
    ("Score any new or changed calendar against the 1991–2020 normals", "Make it the standard check",
     "One method for every site, from data now in the repo. It turns 'which calendar' from a choice of citation into something measured."),
    ("Revisit December at the Indian sites, Kaohsiung's quarters, and Bujumbura", "Decide",
     "The alternatives on file fit the climate better at Kanpur, Jaipur and Kaohsiung. Frozen results name their calendar, so a change needs a version, not an edit."),
    ("For claims about rain or temperature, use each filter's own weather", "Already possible",
     "The gallery's ERA5 fields give the rain and temperature of each filter's sampling window. A calendar bin says which months, not what the weather was."),
], columns=["action", "status", "why"])

text_intro = f"""**Two questions.** Should the HIPS correction be applied per site, per season or per filter? And are the season definitions
themselves sound? **Short answers.** Per filter: season explains {pct(var[(heavy_var, 'Between seasons at a site')])} of the spread in the correction at
the heavy sites, and a season factor misses the typical filter by {pct(one('One per site and season'))}. Seasons are still worth having, as the place
to *check* the correction: at Addis it holds within every season, and what is left is a difference between seasons that is not a loading effect;
at {len(differs)} other sites a single season fails.
The calendars hold up: all {int(structure.sites_with_a_calendar)} are complete and consistent, and Addis's is the best split of its climate. A few
are worth a second look, and there is now one measured way to score any of them. FTIR EC is FTIR-derived, HIPS is optical, and no aethalometer
data are used."""

write_bundle(
    MEETING_ID,
    generated_from="research/ftir_hips_chem/workflows/run_hips_seasons_20261111.py",
    tables={"levels": levels, "variance": variance, "levels_by_site": by_site, "within": within, "within_long": long, "within_tally": tally,
            "site_seasons": site_seasons, "calendars": scored, "alternatives": alternatives, "climate": climate, "cycle": cycle,
            "consistency": tables["calendar_consistency"], "actions": actions},
    text={"intro": text_intro, "levels": text_levels, "within": text_within, "calendars": text_calendars, "cycle": text_cycle},
    colors=COLORS,
)
del hips_sites, failing, ss
