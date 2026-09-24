# Site season calendars for the four-site filter gallery

The gallery defaults to **each site's local calendar** (the table below): every
filter is binned by its own site's months, and labels carry the site, e.g.
“Beijing · Winter (Dec–Feb)”. Month-shaded charts shade each site panel with its
own calendar and leave a shared month axis unshaded when several calendars are
in force. The Ethiopian `dry_feb`/`belg_feb` bins remain as a **shared** option
for like-month comparison; under it, “Belg” on a Beijing, Delhi, or JPL chart
only means the selected months, not the local weather regime. These are fixed
month approximations, not classifications from observed weather on each filter day.

| Site | Recommended local bins | Evidence and qualification |
|---|---|---|
| Beijing (CHTS) | Spring Mar–May; summer Jun–Aug; autumn Sep–Nov; winter Dec–Feb | [Beijing municipal visitor service](https://english.visitbeijing.com.cn/article/4JQ27e27tU6) gives these months. [China Meteorological Administration](https://www.cma.gov.cn/zfxxgk/gknr/flfgbz/bz/202209/P020220921552240920203.pdf) also defines *climate season onset* from persistent daily temperature thresholds; the actual onset can differ from fixed months. Summer is the main rainy period, but “spring” and “autumn” are not wet/dry equivalents. |
| Delhi (INDH) | Winter Jan–Feb; pre-monsoon/hot weather Mar–May; southwest monsoon Jun–Sep; post-monsoon Oct–Dec | [India Meteorological Department seasonal classification](https://mausam.imd.gov.in/imd_latest/contents/pdf/IMDposoco.pdf). The [IMD Delhi onset/withdrawal series](https://mausam.imd.gov.in/newdelhi/mcdata/seasonal_report.pdf) shows monsoon onset and withdrawal vary by year; June–September is a reporting bin, not a claim every June filter was collected in monsoon conditions. |
| JPL / Pasadena area (USPA) | Wet-season window Oct–Apr; dry-season window May–Sep | JPL is in the [Pasadena area](https://www.jpl.nasa.gov/who-we-are/); [California Department of Water Resources](https://water.ca.gov/News/Blog/2023/July-23/Roadmap-for-a-Climate-Resilient-Forecasting-Framework) describes the California wet season as October through April. This is a regional precipitation window, not a site-specific wet-day classification. California’s [water year](https://water.ca.gov/Water-Basics/Glossary) is Oct–Sep and is a different concept. |
| Addis Ababa (ETAD) | Bega/dry Oct–Jan; Belg Feb–May; Kiremt Jun–Sep | The [Ethiopian Meteorological Institute annual bulletin](https://www.ethiomet.gov.et/documents/108/Annual_bull_for_2023.pdf) uses these bins. The repo’s `belg_feb` option matches this calendar. Its older `dry_feb` default puts February in dry (Oct–Feb) and Belg in Mar–May; preserve its name on legacy plots because moving February changes seasonal means. |

## Coverage in the current gallery export

Counts below are included **base filter records**, grouped using the local
bins above from `gallery/app/public/data/filters.json` (939 records total).
They describe sampling coverage, not weather observations. The separate public
SPARTAN CSV inventory was refreshed on 2026-09-22; the gallery export has not
been rebuilt from it.

| Site | Local bin counts |
|---|---|
| Beijing | Winter 73; spring 84; summer 100; autumn 118 |
| Delhi | Winter 7; pre-monsoon 45; monsoon 22; post-monsoon 22 |
| JPL | Wet-season window 163; dry-season window 115 |
| Addis Ababa | Bega 55; Belg 69; Kiremt 66 |

Delhi has **zero September filters** in this export, and only seven winter
filters. Avoid interpreting its fixed-bin winter/monsoon contrasts as a balanced
seasonal experiment. Check year coverage within each bin before comparing means.

## Analysis rules

- Label every seasonal estimate with its site, calendar, month range, sample
  count, and date coverage. A site’s missing months or years can imitate a
  seasonal difference.
- For a multi-site comparison, either use the **same explicit month bins** at
  every site and call them month bins, or fit/report a within-site seasonal
  effect before comparing effects. “Belg” and “monsoon” are not common classes.
- Do not infer meteorological conditions from a calendar bin alone. For claims
  about rain or temperature mechanisms, join contemporaneous weather data and
  classify from those observations.
- The per-site gallery calendar relabels and filters by month; it does not add
  weather observations or PMF solutions for Beijing, Delhi, or JPL. Pick the
  shared Ethiopian month bins for a like-month comparison across sites.
