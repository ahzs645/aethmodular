# Site season calendars for the four-site filter gallery

The gallery defaults to **each site's local calendar** (the table below): every
filter is binned by its own site's months, and labels carry the site, e.g.
“Beijing · Winter (Dec–Feb)”. Month-shaded charts shade each site panel with its
own calendar and leave a shared month axis unshaded when several calendars are
in force. The Ethiopian `dry_feb`/`belg_feb` bins remain as a **shared** option
for like-month comparison; under it, “Belg” on a Beijing, Delhi, or JPL chart
only means the selected months, not the local weather regime. These are fixed
month approximations, not classifications from observed weather on each filter day.

Every other SPARTAN site, and the IMPROVE Santiago sites ANTU1 and LCON1, has its own
calendar in [site-seasonality-network.md](site-seasonality-network.md); the gallery uses
those under the same per-site default.

| Site | Recommended local bins | Evidence and qualification |
|---|---|---|
| Beijing (CHTS) | Spring Mar–May; summer Jun–Aug; autumn Sep–Nov; winter Dec–Feb | [Beijing municipal visitor service](https://english.visitbeijing.com.cn/article/4JQ27e27tU6) gives these months. [China Meteorological Administration](https://www.cma.gov.cn/zfxxgk/gknr/flfgbz/bz/202209/P020220921552240920203.pdf) also defines *climate season onset* from persistent daily temperature thresholds; the actual onset can differ from fixed months. Summer is the main rainy period, but “spring” and “autumn” are not wet/dry equivalents. |
| Delhi (INDH) | Winter Jan–Feb; pre-monsoon/hot weather Mar–May; southwest monsoon Jun–Sep; post-monsoon Oct–Dec | [India Meteorological Department seasonal classification](https://mausam.imd.gov.in/imd_latest/contents/pdf/IMDposoco.pdf). The [IMD Delhi onset/withdrawal series](https://mausam.imd.gov.in/newdelhi/mcdata/seasonal_report.pdf) shows monsoon onset and withdrawal vary by year; June–September is a reporting bin, not a claim every June filter was collected in monsoon conditions. |
| JPL / Pasadena area (USPA) | Wet-season window Oct–Apr; dry-season window May–Sep | JPL is in the [Pasadena area](https://www.jpl.nasa.gov/who-we-are/); [California Department of Water Resources](https://water.ca.gov/News/Blog/2023/July-23/Roadmap-for-a-Climate-Resilient-Forecasting-Framework) describes the California wet season as October through April. This is a regional precipitation window, not a site-specific wet-day classification. California’s [water year](https://water.ca.gov/Water-Basics/Glossary) is Oct–Sep and is a different concept. |
| Addis Ababa (ETAD) | Bega/dry Oct–Jan; Belg Feb–May; Kiremt Jun–Sep | The [Ethiopian Meteorological Institute annual bulletin](https://www.ethiomet.gov.et/documents/108/Annual_bull_for_2023.pdf) uses these bins, and [Shlipak et al. (2026)](https://doi.org/10.1021/acsestair.6c00140) use them for Addis PM2.5 and source-apportioned BC. The repo's `belg_feb` option matches this calendar; use it for new reporting. The older `dry_feb` calendar puts February in dry (Oct–Feb) and Belg in Mar–May. [LaPolt et al. (2025)](https://doi.org/10.1371/journal.pgph.0005021) state it exactly; [Yirga et al. (2024)](https://doi.org/10.3389/feart.2024.1452075) only imply its dry season. Every frozen calibration result was binned with `dry_feb`; keep its name on those results, because moving February changes them (see below). |

Delhi, Beijing and JPL keep their agency sources here; the network table
([site-seasonality-network.md](site-seasonality-network.md)) adds the papers for every site, with
how far each carries (`match`, `scope`, `checked`).

## Calendars in the Python research code

`src/config/multi_site_seasons.py` names two calendars for the four focus sites. Pick one by id;
`assign_season(df, site, calendar=...)` records it in `df.attrs['season_calendar']`.

| Site | `research_2026_04` (default; April 2026 board assets, `multisite_diurnal_wavelength_analysis`, `warren_meeting_slides`) | `local_2026_10` (this page's bins, for new work) |
|---|---|---|
| Addis Ababa | Dry Oct–Feb; Belg Mar–May; Kiremt Jun–Sep (`dry_feb`) | Bega Oct–Jan; Belg Feb–May; Kiremt Jun–Sep (`belg_feb`) |
| Delhi | Winter Nov–Feb; pre-monsoon Mar–Jun; monsoon Jul–Sep; post-monsoon Oct (provisional) | IMD national: winter Jan–Feb; pre-monsoon Mar–May; monsoon Jun–Sep; post-monsoon Oct–Dec |
| JPL | Quarters Dec–Feb, Mar–May, Jun–Aug, Sep–Nov | Wet window Oct–Apr; dry window May–Sep |
| Beijing | Quarters | Quarters |

`generate_apr2026_board_assets.py` pins `research_2026_04` and writes it into its seasonal
summaries, so its outputs reproduce. `research/ftir_hips_chem/scripts/config.py` (`dry_feb`
default, both Ethiopian conventions registered) is unchanged, because frozen audits pin its hash.

## Addis February sensitivity (28 Oct 2026)

[`addis_feb_season_sensitivity_20261028.ipynb`](../research/ftir_hips_chem/addis_feb_season_sensitivity_20261028.ipynb)
moves February between the two calendars on the poster's 189-filter cohort (18 filters move: 9 in
2023, 9 in 2024). Calibrations that do not use season (SPARTAN published, all-IMPROVE Figure 7, the
500 lowest TOR OC/EC) change by at most 0.05 in any seasonal Deming slope. **Figure 9's season-specific
look-alike calibration does not hold**. With February in Belg, Belg's look-alike set changes (315 of
500 shared), and its slope goes from 0.32 (95% CI 0.15–0.47) to 0.67 (0.56–0.79). That holds at every
PLS k, so the cause is the set, not k. The within-Belg organic-signal correlation falls from 0.63 to 0.13.
Dry/Bega and Kiremt results hold. Name the calendar on every seasonal number from a season-specific
calibration, and choose it on meteorological grounds, not by which one fits Belg better.

## Coverage in the current gallery export

Counts below are included **base filter records**, grouped using the local
bins above from `gallery/app/public/data/filters.json` (938 records total).
They describe sampling coverage, not weather observations. The separate public
SPARTAN CSV inventory was refreshed on 2026-09-22; the gallery export has not
been rebuilt from it.

| Site | Local bin counts |
|---|---|
| Beijing | Winter 73; spring 84; summer 100; autumn 118 |
| Delhi | Winter 7; pre-monsoon 45; monsoon 22; post-monsoon 22 |
| JPL | Wet-season window 163; dry-season window 115 |
| Addis Ababa | Bega 55; Belg 69; Kiremt 65 |

ETAD-0248 (2024-08-07) is excluded: it is a PM10 filter, outside every PM2.5
comparison, so Addis has 189 included filters (the frozen calibration runs never
had it).

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
