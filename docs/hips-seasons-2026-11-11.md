# HIPS and seasons: correct per filter, check per season, and how the season calendars hold up

Date: 11 November 2026. Follows [hips-network-2026-11-11.md](hips-network-2026-11-11.md) and
[hips-loading-protocol.md](hips-loading-protocol.md). The numbers are in [`hips-seasons-2026-11-11.json`](hips-seasons-2026-11-11.json);
the charts are in the gallery's Meeting tab, Nov 11, "HIPS and seasons". No aethalometer data are used. FTIR EC is
FTIR-derived and HIPS is optical.

## Summary

| Question | Answer |
|---|---|
| Per site, per season or per filter? | **Per filter.** At the heavy sites a factor per site and season is wrong by 5 % on the typical filter and by more than a tenth on 27 %. Season carries 15 % of the spread in the correction; filters within a season carry 44 %. |
| Then what are seasons for? | **Checking the correction.** At Addis the pooled check fails (0.84) but every season passes; what is left is a difference between seasons. At Johannesburg, Kaohsiung and Beijing the site passes on average and one season fails. |
| What is left at Addis? | After correcting, HIPS per unit FTIR EC is 1.36 times higher in Bega than in Kiremt (it was 1.69). That is a season effect in absorption or in FTIR EC, not in HIPS loading. |
| Are the season definitions sound? | **Yes in structure.** All 42 calendars cover each month once, every HIPS site has one, and the Python and gallery calendars agree. |
| Do they fit the climate? | **Now measured.** Against 1991–2020 ERA5 normals, Addis's calendar in use is the best three-season split there is. December-in-winter fits better at the Indian sites, standard quarters at Kaohsiung, and Bujumbura is weak. |
| A new source? | **Added:** ERA5 monthly normals for 1991–2020 at 40 sites, one method for all. The gallery's weather covered only each site's sampling period, under a year at CLTA, ETBI, INJA, USSL. |

## 1. Per site, per season or per filter

The correction factor depends on the filter's own optical depth. The test is how far a group's factor (that of the group's
median filter) is from each filter's own.

| Factor applied | Every site, typical filter | Heavy sites, typical filter | Heavy sites, worst tenth at least | Heavy sites, missed by over a tenth |
|---|---|---|---|---|
| One factor for the network | 9 % | 11 % | 33 % | 52 % |
| One per site | 4 % | 7 % | 20 % | 36 % |
| One per site and season | 3 % | 5 % | 17 % | 27 % |
| One per site and month | 3 % | 5 % | 16 % | 23 % |
| The ceiling's own uncertainty, per filter | 3 % | 6 % | 15 % | 23 % |

Heavy sites: those whose median filter is expected to lose more than a tenth (2,083 filters).

Where the spread in the (log) correction factor lies:

| | Every site | Heavy sites |
|---|---|---|
| Between sites | 58 % | 41 % |
| Between seasons at a site | 10 % | 15 % |
| Between filters in a season | 31 % | 44 % |

A season is a mix of light and dark filters, and at Delhi, Jaipur, Dhaka and Bujumbura of filters sampled at different
volumes. The per-filter factor costs nothing extra: the optical depth is on file for every filter.

## 2. The correction, checked within seasons

The check is whether linearised HIPS is proportional to FTIR EC (an exponent that includes 1), with weighed PM2.5 as the
instrument. Three ways: pooled, with each season's own level taken out, and season by season where a season has 30 filters.

| Site | Pooled | Within seasons | Seasons that fail | Highest ÷ lowest season of Fabs ÷ FTIR EC, reported → linearised | Reading |
|---|---|---|---|---|---|
| Addis Ababa, Ethiopia | 0.84 (under-corrected) | 1.09 [0.99, 1.21] | — | 1.69 → 1.36 | holds within seasons; the seasons sit at different levels |
| Johannesburg, South Africa | 1.10 (over-corrected) | 1.09 [0.96, 1.25] | Winter under, Spring over | 1.17 → 1.17 | differs by season |
| Delhi, India | 0.92 (under-corrected) | 0.91 [0.82, 1.00] | — | 1.40 → 1.32 | still compressed within seasons |
| Santiago, Chile | 0.72 (under-corrected) | 0.75 [0.68, 0.80] | Warm under, Cold under | 2.03 → 1.57 | still compressed within seasons |
| Pretoria, South Africa | 1.10 (corrected) | 1.00 [0.87, 1.15] | — | 1.14 → 1.35 | holds within seasons and overall |
| Kaohsiung, Taiwan | 0.92 (corrected) | 0.94 [0.79, 1.10] | Winter under | 1.17 → 1.12 | differs by season |
| Beijing, China | 1.06 (corrected) | 1.01 [0.89, 1.20] | Summer under | 1.44 → 1.40 | differs by season |
| Talca, Chile | 0.65 (under-corrected) | 0.62 [0.55, 0.67] | Cold under, Warm under | 1.86 → 1.77 | still compressed within seasons |
| Taipei, Taiwan | 1.14 (over-corrected) | 1.17 [1.05, 1.32] | — | 1.24 → 1.17 | over-corrected within seasons |
| Melbourne, Australia | 0.82 (corrected) | 0.73 [0.46, 1.16] | — | 1.08 → 1.36 | cannot tell |
| Pasadena, United States | 1.08 (corrected) | 1.03 [0.82, 1.29] | — | 1.12 → 1.14 | holds within seasons and overall |

Addis, season by season:

| Season | Filters | Optical depth | Expected under-read | Fabs ÷ FTIR EC (m²/g), reported → linearised | Exponent after | Verdict |
|---|---|---|---|---|---|---|
| Bega (Oct–Jan) | 87 | 0.88 | 39 % | 13.2 → 22.3 | 1.10 [0.87, 1.39] | corrected |
| Belg (Feb–May) | 89 | 0.92 | 41 % | 11.6 → 19.6 | 1.04 [0.89, 1.25] | corrected |
| Kiremt (Jun–Sep) | 71 | 1.14 | 50 % | 7.8 → 16.4 | 1.13 [1.00, 1.29] | corrected |

- **Addis:** the per-filter correction holds in all three seasons. Pooled it fails because the seasons sit at different
  levels of HIPS per unit FTIR EC, and the heaviest season (Kiremt) is the lowest. The correction removes about half of
  that difference. The rest is not a loading effect.
- **Johannesburg, Kaohsiung, Beijing:** a season fails while the site passes on average (two at Johannesburg, in opposite
  directions). A pooled check hides it.
- **Santiago, Talca:** compressed in both seasons, and Talca's filters are too light for loading to explain it.
- **Delhi:** on the line, 0.91 [0.82, 1.00] within seasons; each of its three testable seasons passes alone.
- A made-up site with a season offset and no extra bend reproduces the Addis pattern: every season passes, the pooled
  check fails (`tests/test_hips_seasons.py`).

## 3. The season calendars

**Structure.** 42 calendars in `gallery/app/src/siteSeasons.network.json`; each covers every month once; no HIPS
site lacks one; `src/config/multi_site_seasons.py` (`local_2026_10`) and `config.py` (`belg_feb`) give the same months as the
gallery for the four focus sites.

**Against the climate.** New in the repo: `gallery/app/public/data/weather/normals_1991_2020.json`, ERA5 monthly normals of
temperature and precipitation at 40 sites (`fetch_climate_normals_era5.py`). A calendar is scored by the share of the
month-to-month spread in rain and in temperature that falls between its seasons, and compared with the best split of the
year into as many runs of consecutive months. Sites with HIPS, largest gap first:

| Site | Seasons in use | Rain | Temperature | Together | Best possible | Months the best split would move |
|---|---|---|---|---|---|---|
| Bujumbura, Burundi | Short dry season (Jan); Long rains (Feb–May); Long dry season (Jun–Aug); Short rains (Sep–Dec) | 0.59 | 0.26 | 0.42 | 0.78 | Jan to Long rains, Nov to Long rains, Dec to Long rains |
| Singapore, Singapore | Northeast monsoon (Dec–Mar); Inter-monsoon (Apr–May); Southwest monsoon (Jun–Sep); Inter-monsoon (Oct–Nov) | 0.47 | 0.76 | 0.61 | 0.82 | Mar to Inter-monsoon, Oct to Southwest monsoon, Dec to Inter-monsoon |
| Ilorin, Nigeria | Harmattan/dry (Nov–Mar); Wet (Apr–Oct) | 0.77 | 0.38 | 0.58 | 0.78 | Apr to Harmattan/dry, May to Harmattan/dry |
| Halifax, Canada | Winter (Dec–Feb); Spring (Mar–May); Summer (Jun–Aug); Autumn (Sep–Nov) | 0.58 | 0.83 | 0.71 | 0.90 | Mar to Winter, Sep to Summer, Dec to Autumn |
| Jaipur, India | Winter (Jan–Feb); Pre-monsoon (Mar–May); Southwest monsoon (Jun–Sep); Post-monsoon (Oct–Dec) | 0.72 | 0.72 | 0.72 | 0.89 | Jan to Post-monsoon, Feb to Post-monsoon, Mar to Post-monsoon, Jun to Pre-monsoon, Sep to Post-monsoon |
| Norman, United States | Winter (Dec–Feb); Spring (Mar–May); Summer (Jun–Aug); Autumn (Sep–Nov) | 0.59 | 0.84 | 0.71 | 0.88 | Mar to Winter, Jun to Spring, Nov to Winter |
| Kaohsiung, Taiwan | Winter (Nov–Jan); Spring (Feb–Apr); Summer (May–Jul); Autumn (Aug–Oct) | 0.71 | 0.78 | 0.75 | 0.90 | Feb to Winter, Aug to Summer, Sep to Summer, Nov to Autumn |
| Bandung, Indonesia | Rainy (Dec–Feb); Wet-to-dry transition (Mar–May); Dry (Jun–Aug); Dry-to-wet transition (Sep–Nov) | 0.63 | 0.77 | 0.70 | 0.84 | Mar to Rainy, Apr to Rainy, May to Dry |
| Pretoria, South Africa | Summer (Dec–Feb); Autumn (Mar–May); Winter (Jun–Aug); Spring (Sep–Nov) | 0.75 | 0.85 | 0.80 | 0.93 | Mar to Summer, May to Winter, Nov to Summer |
| Sherbrooke, Canada | Winter (Dec–Feb); Spring (Mar–May); Summer (Jun–Aug); Autumn (Sep–Nov) | 0.72 | 0.83 | 0.78 | 0.90 | Mar to Winter, May to Summer, Sep to Summer, Oct to Summer, Dec to Autumn |
| Ulsan, Korea | Winter (Dec–Feb); Spring (Mar–May); Summer (Jun–Aug); Autumn (Sep–Nov) | 0.75 | 0.84 | 0.79 | 0.92 | Mar to Winter, Sep to Summer, Nov to Winter |
| Pasadena, United States | Wet window (Oct–Apr); Dry window (May–Sep) | 0.52 | 0.72 | 0.62 | 0.74 | Apr to Dry window, Oct to Dry window, Nov to Dry window |

Sites with two calendars on file:

| Site | Calendar | Seasons | Rain | Temperature | Together |
|---|---|---|---|---|---|
| Bujumbura, Burundi | `ndamuzi` (in use) | Short dry season (Jan); Long rains (Feb–May); Long dry season (Jun–Aug); Short rains (Sep–Dec) | 0.59 | 0.26 | 0.42 |
| Bujumbura, Burundi | `nkunzimana` | Semidry (Jan–Feb); Long rains (Mar–May); Dry (Jun–Aug); Short rains (Sep–Dec) | 0.62 | 0.26 | 0.44 |
| Palmira, Colombia | `bogota` (in use) | First dry (Dec–Feb); First wet (Mar–May); Second dry (Jun–Aug); Second wet (Sep–Nov) | 0.65 | 0.11 | 0.38 |
| Palmira, Colombia | `palmira` | First drier (Dec–Feb); First wetter (Mar–May); Second drier (Jun–Sep); Second wetter (Oct–Nov) | 0.87 | 0.26 | 0.56 |
| Addis Ababa, Ethiopia | `belg_feb` (in use) | Bega (Oct–Jan); Belg (Feb–May); Kiremt (Jun–Sep) | 0.88 | 0.88 | 0.88 |
| Addis Ababa, Ethiopia | `dry_feb` | Dry (Oct-Feb); Belg (Mar-May); Kiremt (Jun-Sep) | 0.91 | 0.76 | 0.84 |
| Delhi, India | `imd_national` (in use) | Winter (Jan–Feb); Pre-monsoon (Mar–May); Monsoon (Jun–Sep); Post-monsoon (Oct–Dec) | 0.77 | 0.75 | 0.76 |
| Delhi, India | `research_2026_04` | Winter (Nov-Feb); Pre-monsoon (Mar-Jun); Monsoon (Jul-Sep); Post-monsoon (Oct) | 0.83 | 0.80 | 0.81 |
| Jaipur, India | `imd_national` (in use) | Winter (Jan–Feb); Pre-monsoon (Mar–May); Southwest monsoon (Jun–Sep); Post-monsoon (Oct–Dec) | 0.72 | 0.72 | 0.72 |
| Jaipur, India | `imd_jaipur` | Winter (Dec–Feb); Pre-monsoon (Mar–May); Southwest monsoon (Jun–Sep); Post-monsoon (Oct–Nov) | 0.72 | 0.80 | 0.76 |
| Kanpur, India | `imd_national` (in use) | Winter (Jan–Feb); Pre-monsoon (Mar–May); Southwest monsoon (Jun–Sep); Post-monsoon (Oct–Dec) | 0.82 | 0.74 | 0.78 |
| Kanpur, India | `dec_winter` | Winter (Dec–Feb); Pre-monsoon (Mar–May); Southwest monsoon (Jun–Sep); Post-monsoon (Oct–Nov) | 0.82 | 0.82 | 0.82 |
| Kaohsiung, Taiwan | `review` (in use) | Winter (Nov–Jan); Spring (Feb–Apr); Summer (May–Jul); Autumn (Aug–Oct) | 0.71 | 0.78 | 0.75 |
| Kaohsiung, Taiwan | `outlook` | Winter (Dec–Feb); Spring (Mar–May); Summer (Jun–Aug); Autumn (Sep–Nov) | 0.80 | 0.85 | 0.82 |
| Taipei, Taiwan | `review` (in use) | Winter (Nov–Jan); Spring (Feb–Apr); Summer (May–Jul); Autumn (Aug–Oct) | 0.87 | 0.81 | 0.84 |
| Taipei, Taiwan | `outlook` | Winter (Dec–Feb); Spring (Mar–May); Summer (Jun–Aug); Autumn (Sep–Nov) | 0.74 | 0.84 | 0.79 |

- **Addis:** the calendar in use (`belg_feb`) is the best three-season split. `dry_feb` is better on rain alone and worse on
  temperature. No change indicated.
- **Delhi, Jaipur, Kanpur:** the calendars with December in winter score higher than the IMD national bins in use.
- **Kaohsiung:** standard quarters score higher than the shifted quarters in use; Taipei the reverse.
- **Bujumbura:** weak under both calendars, and the one-month "short dry season" is not drier than the rains in ERA5.
- **Mid-latitude sites on standard quarters:** temperature is separated well (about 0.83) and rain poorly, because rain has
  little season there. The gap to the best split is mostly the one-month lag of temperature behind the calendar.

Wet, dry, cold and warm names were also checked against the normals: they hold at every site that has them, except
Bujumbura's calendar in use.

## 4. Seasons and filter darkness

| Site | Best to worst month, expected under-read | Carried by the calendar's seasons | Follows rain | Follows temperature |
|---|---|---|---|---|
| Jaipur, India | 37 % | 57 % | -0.71 | -0.67 |
| Delhi, India | 28 % | 73 % | -0.90 | -0.42 |
| Santiago, Chile | 26 % | 62 % | +0.82 | -0.81 |
| Bujumbura, Burundi | 22 % | 98 % | -0.79 | +0.01 |
| Talca, Chile | 22 % | 72 % | +0.72 | -0.75 |
| Addis Ababa, Ethiopia | 18 % | 75 % | +0.85 | +0.03 |
| Pretoria, South Africa | 18 % | 78 % | -0.96 | -0.96 |
| Johannesburg, South Africa | 17 % | 59 % | -0.91 | -0.82 |
| Mexico City, Mexico | 16 % | 72 % | -0.83 | -0.34 |
| Bandung, Indonesia | 15 % | 73 % | -0.72 | +0.22 |
| Seoul, Korea | 11 % | 64 % | -0.85 | -0.82 |
| Rehovot, Israel | 11 % | 62 % | +0.92 | -0.83 |

"Follows" is the rank correlation across calendar months. Dark filters come in the dry months at most heavy sites, but in
the wet months at Addis, Santiago, Talca, Rehovot and Haifa. At Addis the heaviest filters are in Kiremt. A rule such as
"the dry season is the affected season" would be wrong there.

## What to do

| Action | Status |
|---|---|
| Apply the HIPS correction per filter | Do it |
| Run the check within each season as well as pooled | Added: `aeth hips screen` reports both and writes `seasons.csv` |
| Where one season fails the check, treat that season's HIPS as shape only | Proposed |
| Report HIPS per unit FTIR EC by season at Addis; do not fold it into the HIPS correction | Proposed |
| Keep the sourced calendars as the labels | No change |
| Score any new or changed calendar against the 1991–2020 normals | Proposed as the standard check |
| Revisit December at the Indian sites, Kaohsiung's quarters, Bujumbura | Decide; frozen results name their calendar, so a change needs a new version |
| For claims about rain or temperature, use each filter's own ERA5 window | Already possible in the gallery |

## Limits

- The checks use FTIR EC. A season difference in HIPS per unit FTIR EC can come from the aerosol or from FTIR EC.
- A season needs 30 filters with FTIR EC and PM2.5 for its own verdict; 55 of 101 site-seasons have fewer.
- ERA5 is a reanalysis on a grid. Rain in mountains and on coasts is its weakest point, which matters at Addis, Bujumbura
  and Bandung.
- The calendar score rewards separating rain and temperature. A national definition can rest on other grounds (winds,
  onset dates), so a low score is a prompt to look, not a verdict.
- Seasons are fixed month bins. Monsoon onset and withdrawal move from year to year.

## Reproduce

```bash
python research/ftir_hips_chem/workflows/fetch_climate_normals_era5.py        # only to refresh the normals
python research/ftir_hips_chem/workflows/run_hips_seasons_20261111.py --summary
python gallery/data/export_meeting_20261111_seasons.py
aeth hips screen --site ETAD ZAJB --out OUT                                   # the check pooled and by season
python -m pytest -q tests/test_hips_seasons.py
```
