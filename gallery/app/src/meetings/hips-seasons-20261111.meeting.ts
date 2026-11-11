import type { MeetingSpec } from './types'

// "HIPS and seasons": should the HIPS correction be applied per site, per season or per filter, and do the season
// calendars hold up against the climate? Tables from gallery/data/export_meeting_20261111_seasons.py, which reruns
// research/ftir_hips_chem/workflows/run_hips_seasons_20261111.py on committed files (the SPARTAN table, the season
// calendars, and 1991-2020 ERA5 normals). Titles carry the message: section text shows only with Explain on.
const CURVE = 'The ceiling is IMPROVE\'s, 2.26 [1.82, 2.70], measured against thermal EC. SPARTAN has no thermal EC to check it on its own filters.'
const REFERENCE = 'The checks use FTIR EC, which is FTIR-derived and not thermal EC, so they cannot separate HIPS from FTIR EC. No aethalometer data are used.'
const CLIMATE = 'Climate: ERA5 monthly normals for 1991–2020 (Copernicus Climate Change Service, via Open-Meteo, CC BY 4.0). A reanalysis on a grid; least reliable for rain in mountains and on coasts. Calendars are fixed month bins from agencies and papers (docs/site-seasonality-network.md).'

export default {
  id: 'hips-seasons-20261111',
  label: 'Nov 11, 2026',
  date: '2026-11-11',
  short: 'HIPS and seasons',
  title: 'HIPS and seasons: correct per filter, check per season, and how the season calendars hold up',
  command: 'uv run python gallery/data/export_meeting_20261111_seasons.py',
  intro: '@intro',
  sections: [
    {
      id: 'levels',
      title: '1. Per site, per season or per filter? Per filter: a season factor misses the typical filter by 5 %',
      text: '@levels',
      figures: [
        {
          kind: 'bars', id: 'hs-levels', table: 'levels', label: 'level', value: 'median', format: 'pct', sort: 'none',
          choose: [{ column: 'scope', label: 'Sites', default: 'Sites that lose more than 10% on the median filter' }],
          xLabel: 'how far the group\'s factor is from the filter\'s own (median filter)',
          title: 'One factor for a group against each filter\'s own: finer groups help, but a season is still wrong by about as much as the ceiling is uncertain',
          subtitle: 'Each filter\'s correction factor is compared with the factor of its group\'s median filter. The last bar is the uncertainty of the per-filter factor itself, from the ceiling\'s interval.',
          noteAlways: true, note: CURVE,
        },
        {
          kind: 'table', id: 'hs-levels-table', table: 'levels', title: 'The same, with the worst tenth of filters and the share missed by more than a tenth',
          columns: [{ key: 'scope', label: 'Sites' }, { key: 'level', label: 'Factor applied' }, { key: 'filters', label: 'Filters', format: 'int' },
            { key: 'median', label: 'Typical filter', format: 'pct' }, { key: 'p90', label: 'Worst tenth, at least', format: 'pct' }, { key: 'share_over_10', label: 'Missed by over a tenth', format: 'pct' }],
        },
        {
          kind: 'bars', id: 'hs-variance', table: 'variance', label: 'part', value: 'share', format: 'pct', sort: 'none',
          choose: [{ column: 'scope', label: 'Sites', default: 'Sites that lose more than 10%' }], xLabel: 'share of the spread in the correction factor',
          title: 'Where the spread in the correction lies: seasons carry the least',
          subtitle: 'Variance of the log correction factor, split between sites, between seasons at a site, and between filters in the same site and season.',
        },
        {
          kind: 'table', id: 'hs-levels-site', table: 'levels_by_site', title: 'Site by site: how far each coarser factor is from the filter\'s own',
          columns: [{ key: 'site', label: 'Site' }, { key: 'median_tau', label: 'Optical depth', format: 'f2' }, { key: 'One per site', label: 'One per site', format: 'pct' },
            { key: 'One per site and season', label: 'per site and season', format: 'pct' }, { key: 'One per site and month', label: 'per site and month', format: 'pct' },
            { key: 'The ceiling\'s own uncertainty, per filter', label: 'Ceiling\'s uncertainty', format: 'pct' }],
          subtitle: 'Median filter at each site.',
        },
      ],
    },
    {
      id: 'within',
      title: '2. Check per season: Addis holds within every season; at three sites one season fails',
      text: '@within',
      figures: [
        {
          kind: 'intervals', id: 'hs-within', table: 'within_long', row: 'site', series: 'kind',
          title: 'The check pooled and within seasons: Addis fails pooled but holds within seasons',
          subtitle: 'b in linearised HIPS ∝ (FTIR EC)^b with weighed PM2.5 as the instrument; it should include 1. "Within seasons" takes out each season\'s own level first. Bars: 95 % bootstrap. Heaviest sites first.',
          panels: [{ value: 'value', label: 'Response exponent after correcting', reference: 1, lo: 'lo', hi: 'hi' }], labelWidth: 220,
          noteAlways: true, note: REFERENCE,
        },
        {
          kind: 'table', id: 'hs-within-table', table: 'within', title: 'Each site\'s reading, and how far HIPS per unit FTIR EC differs between its seasons',
          columns: [{ key: 'site', label: 'Site' }, { key: 'tested_filters', label: 'Filters', format: 'int' }, { key: 'pooled_after', label: 'Pooled', format: 'f2' },
            { key: 'within_after', label: 'Within seasons', format: 'f2' }, { key: 'within_lo', label: 'low', format: 'f2' }, { key: 'within_hi', label: 'high', format: 'f2' },
            { key: 'season_spread_reported', label: 'Highest ÷ lowest season, reported', format: 'f2' }, { key: 'season_spread_linearised', label: 'linearised', format: 'f2' },
            { key: 'highest_season', label: 'Highest' }, { key: 'lowest_season', label: 'Lowest' }, { key: 'seasons_failing', label: 'Seasons that fail' }, { key: 'reading', label: 'Reading' }],
          subtitle: 'Highest ÷ lowest: the ratio of the season medians of Fabs ÷ FTIR EC. Seasons with at least 15 tested filters.',
          noteAlways: true, note: REFERENCE,
        },
        {
          kind: 'table', id: 'hs-site-seasons', table: 'site_seasons', choose: [{ column: 'site', label: 'Site', default: 'Addis Ababa, Ethiopia' }],
          title: 'One site, season by season: how dark the filters are, and whether the correction holds',
          columns: [{ key: 'season', label: 'Season' }, { key: 'filters', label: 'Filters', format: 'int' }, { key: 'median_tau', label: 'Optical depth', format: 'f2' },
            { key: 'under_read', label: 'Expected under-read', format: 'pct' }, { key: 'share_use', label: 'Usable', format: 'pct' },
            { key: 'fabs_per_ec', label: 'Fabs ÷ FTIR EC, reported', format: 'f2' }, { key: 'fabs_per_ec_linearised', label: 'linearised', format: 'f2' },
            { key: 'exponent_after', label: 'Exponent after', format: 'f2' }, { key: 'after_lo', label: 'low', format: 'f2' }, { key: 'after_hi', label: 'high', format: 'f2' }, { key: 'verdict', label: 'Verdict' }],
          subtitle: 'Choose a site. A season needs 30 filters with FTIR EC and PM2.5 for its own verdict.',
          noteAlways: true, note: CURVE + ' ' + REFERENCE,
        },
      ],
    },
    {
      id: 'calendars',
      title: '3. The season calendars against the climate: complete, consistent, and Addis\'s is the best split there is',
      text: '@calendars',
      figures: [
        {
          kind: 'table', id: 'hs-calendars', table: 'calendars', title: 'Every calendar in use, scored against 1991–2020 rain and temperature',
          subtitle: 'Rain and temperature: the share of the month-to-month spread that falls between the calendar\'s seasons. Best possible: the same for the best split into as many runs of consecutive months. Sorted by the gap between them.',
          columns: [{ key: 'site', label: 'Site' }, { key: 'bins', label: 'Seasons in use' }, { key: 'confidence', label: 'Source' }, { key: 'rain', label: 'Rain', format: 'f2' },
            { key: 'temperature', label: 'Temperature', format: 'f2' }, { key: 'together', label: 'Together', format: 'f2' }, { key: 'best_possible', label: 'Best possible', format: 'f2' },
            { key: 'headroom', label: 'Gap', format: 'f2' }, { key: 'checked', label: 'Wet and dry names' }, { key: 'temperature_checked', label: 'Cold and warm names' }, { key: 'would_move', label: 'Months the best split would move' }],
          noteAlways: true, note: CLIMATE,
        },
        {
          kind: 'table', id: 'hs-alternatives', table: 'alternatives', title: 'Sites with two calendars on file: December in winter fits better at the Indian sites; Addis\'s calendar in use is the best',
          columns: [{ key: 'site', label: 'Site' }, { key: 'calendar', label: 'Calendar' }, { key: 'in_use', label: 'In use' }, { key: 'bins', label: 'Seasons' },
            { key: 'rain', label: 'Rain', format: 'f2' }, { key: 'temperature', label: 'Temperature', format: 'f2' }, { key: 'together', label: 'Together', format: 'f2' },
            { key: 'best_possible', label: 'Best possible', format: 'f2' }],
          noteAlways: true, note: CLIMATE,
        },
        {
          kind: 'sweep', id: 'hs-climate', table: 'climate', x: 'month', y: 'value', xLabel: 'calendar month', yLabel: 'monthly normal',
          choose: [{ column: 'site', label: 'Site', default: 'Addis Ababa, Ethiopia' }, { column: 'variable', label: 'Show', default: 'Rain (mm/day)' }],
          title: 'The climate the calendars are scored against: one site\'s year',
          subtitle: 'Monthly normal, 1991–2020. Choose a site, and rain or temperature.',
          noteAlways: true, note: CLIMATE,
        },
        {
          kind: 'table', id: 'hs-consistency', table: 'consistency', title: 'The calendars in the Python code and in the gallery are the same',
          columns: [{ key: 'pair', label: 'Python calendar' }, { key: 'against', label: 'Compared with' }, { key: 'same_months', label: 'Same months' }],
        },
      ],
    },
    {
      id: 'cycle',
      title: '4. Seasons and filter darkness: dark filters come in the dry months at eleven sites, in the wet months at five',
      text: '@cycle',
      figures: [
        {
          kind: 'table', id: 'hs-cycle', table: 'cycle', title: 'How much of the monthly cycle in expected under-read the calendar carries, and whether it follows rain or temperature',
          columns: [{ key: 'site', label: 'Site' }, { key: 'hips_cycle_range', label: 'Best to worst month', format: 'pct' }, { key: 'hips_cycle_between_seasons', label: 'Carried by the seasons', format: 'pct' },
            { key: 'hips_follows_rain', label: 'Follows rain', format: 'f2' }, { key: 'hips_follows_temperature', label: 'Follows temperature', format: 'f2' }, { key: 'bins', label: 'Seasons in use' }],
          subtitle: 'Follows: rank correlation across calendar months between the median expected under-read and the 1991–2020 normal. Negative with rain: darker filters in drier months.',
          noteAlways: true, note: CURVE + ' ' + CLIMATE,
        },
      ],
    },
    {
      id: 'actions',
      title: '5. What to do',
      figures: [
        {
          kind: 'table', id: 'hs-actions', table: 'actions', title: 'Correct per filter, check per season, keep the calendars and score them',
          columns: [{ key: 'action', label: 'Action' }, { key: 'status', label: 'Status' }, { key: 'why', label: 'Why' }],
          noteAlways: true, note: CURVE + ' ' + REFERENCE,
        },
      ],
    },
  ],
} satisfies MeetingSpec
