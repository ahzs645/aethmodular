import type { ReactNode } from 'react'
import type { AxesOpts } from '@/lib/axes'
import type { FilterRow, MetaFile, PmfFile, PmfRow } from '@/lib/types'
import type { Scope } from '@/lib/dashboard'
import { StatTile } from '@/charts/summary/StatTile'
import { SiteCards } from '@/charts/summary/SiteCards'
import { SiteSummary } from '@/charts/summary/SiteSummary'
import { SiteRank } from '@/charts/ranking/SiteRank'
import { CVCurveTile, MonthlyClimateTile, PolarPlotTile, PollutionRoseTile, RainWashoutTile, RunCrossplotTile, WeatherCorrelationsTile, WeatherSummaryTile, WeatherTimelineTile, WindRosesTile, WindSpeedTile, type CalibrationData } from '@/components/dashboard/extraTiles'
import { Scatterplot } from '@/charts/correlation/Scatterplot'
import { Correlogram } from '@/charts/correlation/Correlogram'
import { AvailabilityHeatmap } from '@/charts/correlation/AvailabilityHeatmap'
import { DensityHexbin } from '@/charts/correlation/DensityHexbin'
import { ConnectedScatter } from '@/charts/correlation/ConnectedScatter'
import { SpeciesGraph } from '@/charts/correlation/SpeciesGraph'
import { DistributionPanel } from '@/charts/distribution/DistributionPanel'
import { Histogram } from '@/charts/distribution/Histogram'
import { Ridgeline } from '@/charts/distribution/Ridgeline'
import { Timeseries } from '@/charts/evolution/Timeseries'
import { MonthlyBand } from '@/charts/evolution/MonthlyBand'
import { SeasonalClock } from '@/charts/evolution/SeasonalClock'
import { MonthlyHeatmap } from '@/charts/evolution/MonthlyHeatmap'
import { Barplot } from '@/charts/ranking/Barplot'
import { CorrelationRanking } from '@/charts/ranking/CorrelationRanking'
import { SiteRadar } from '@/charts/ranking/SiteRadar'
import { ParallelCoordinates } from '@/charts/ranking/ParallelCoordinates'
import { CompositionTreemap } from '@/charts/partOfWhole/CompositionTreemap'
import { FunctionalGroups } from '@/charts/partOfWhole/FunctionalGroups'
import { SiteBubbleMap, SpartanCompositionMap } from '@/charts/map/lazy'
import { MethodSankey } from '@/charts/flow/MethodSankey'
import { SourceStack } from '@/charts/pmf/SourceStack'
import { resolveField } from '@/lib/fieldEquivalents'

/** What App hands every tile: the Filters panel's rows and meta, and what a few charts need besides. */
export interface DashboardData {
  rows: FilterRow[]
  /** every filter of the sites in view before the date, season and exclusion filters (what each site measured at all) */
  measured?: FilterRow[]
  meta: MetaFile
  axes: AxesOpts
  pmf: PmfFile | null
  pmfRows: PmfRow[]
  /** the field a chart draws for a picked one (FTIR EC under the chosen spectral baseline) */
  eff: (field: string) => string
  /** the sidebar's x, y and measurement: a new tile starts on them */
  defaults: { x: string; y: string; field: string }
  /** open one site on its own (the One site scope); a map bubble or a site card calls it */
  openSite?: (code: string) => void
  /** a date's PMF source, for the wind roses' split by source */
  pmfLabelOfDate?: (d: string | null | undefined) => string | null
  /** the calibration explorer's runs, for the calibration tiles (absent: those tiles are not offered) */
  calibration?: CalibrationData | null
}

/**
 * One kind of tile. `fields` says which pickers the tile shows: an x/y pair, one
 * measurement, or none (the chart picks its own). `render` gets the tile's fields already
 * mapped through `eff`, in that order.
 */
export interface DashboardChart {
  kind: string
  label: string
  /** the gallery tab it comes from, for the Add chart menu */
  group: string
  blurb: string
  fields: 'pair' | 'single' | 'trio' | 'none'
  /** a new tile's columns (of 12) and rows (of 12, the window's height) */
  size: [number, number]
  /** offered only when this holds (PMF charts need the PMF solution) */
  available?: (d: DashboardData) => boolean
  render: (d: DashboardData, fields: string[]) => ReactNode
}

const CHARTS: DashboardChart[] = [
  { kind: 'sitecards', label: 'Site cards', group: 'Summary', blurb: 'one card per site: three numbers, bars and sparklines; click to open the site', fields: 'trio', size: [12, 6],
    render: (d, f) => <SiteCards rows={d.rows} meta={d.meta} fields={f} onOpenSite={d.openSite} /> },
  { kind: 'sitesummary', label: 'Site summary', group: 'Summary', blurb: 'where, when, and which measurements exist (and on how many filters)', fields: 'none', size: [3, 2],
    render: (d) => <SiteSummary rows={d.rows} meta={d.meta} /> },
  { kind: 'stat', label: 'Headline number', group: 'Summary', blurb: 'median, spread and monthly sparkline of one field', fields: 'single', size: [3, 2],
    render: (d, [f]) => <StatTile rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'scatter', label: 'Scatterplot', group: 'Correlation', blurb: 'x against y, fit lines, per-site panels', fields: 'pair', size: [4, 5],
    render: (d, [x, y]) => <Scatterplot rows={d.rows} meta={d.meta} xField={x} yField={y} axes={d.axes} /> },
  { kind: 'hexbin', label: 'Density hexbin', group: 'Correlation', blurb: 'x against y where points overlap', fields: 'pair', size: [4, 5],
    render: (d, [x, y]) => <DensityHexbin rows={d.rows} meta={d.meta} xField={x} yField={y} axes={d.axes} /> },
  { kind: 'connected', label: 'Connected scatter', group: 'Correlation', blurb: 'x against y in date order', fields: 'pair', size: [4, 5],
    render: (d, [x, y]) => <ConnectedScatter rows={d.rows} meta={d.meta} xField={x} yField={y} axes={d.axes} /> },
  { kind: 'correlogram', label: 'Correlogram', group: 'Correlation', blurb: 'r between many fields at once', fields: 'none', size: [6, 6],
    render: (d) => <Correlogram rows={d.rows} meta={d.meta} /> },
  { kind: 'species', label: 'Species graph', group: 'Correlation', blurb: 'which species move together', fields: 'none', size: [6, 6],
    render: (d) => <SpeciesGraph rows={d.rows} meta={d.meta} /> },
  { kind: 'availability', label: 'Availability', group: 'Correlation', blurb: 'where both fields were measured', fields: 'pair', size: [6, 4],
    render: (d, [x, y]) => <AvailabilityHeatmap rows={d.rows} meta={d.meta} xField={x} yField={y} /> },

  { kind: 'distribution', label: 'Box / violin', group: 'Distribution', blurb: 'one field by site and season', fields: 'single', size: [6, 5],
    render: (d, [f]) => <DistributionPanel rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'histogram', label: 'Histogram', group: 'Distribution', blurb: 'overlay, density or small multiples', fields: 'single', size: [4, 5],
    render: (d, [f]) => <Histogram rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'ridgeline', label: 'Ridgeline', group: 'Distribution', blurb: 'one density per site', fields: 'single', size: [4, 5],
    render: (d, [f]) => <Ridgeline rows={d.rows} meta={d.meta} field={f} /> },

  { kind: 'timeseries', label: 'Time series', group: 'Evolution', blurb: 'one field over time', fields: 'single', size: [8, 5],
    render: (d, [f]) => <Timeseries rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'monthlyband', label: 'Monthly band', group: 'Evolution', blurb: 'median and spread by month', fields: 'single', size: [4, 5],
    render: (d, [f]) => <MonthlyBand rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'clock', label: 'Seasonal clock', group: 'Evolution', blurb: 'months around a circle', fields: 'single', size: [4, 5],
    render: (d, [f]) => <SeasonalClock rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'monthlyheatmap', label: 'Monthly heatmap', group: 'Evolution', blurb: 'year × month', fields: 'single', size: [6, 5],
    render: (d, [f]) => <MonthlyHeatmap rows={d.rows} meta={d.meta} field={f} /> },

  { kind: 'corrrank', label: 'Correlation ranking', group: 'Ranking', blurb: 'what tracks one field best', fields: 'single', size: [4, 6],
    render: (d, [f]) => <CorrelationRanking rows={d.rows} meta={d.meta} target={f} /> },
  { kind: 'siterank', label: 'Sites ranked', group: 'Ranking', blurb: 'median and IQR per site, sorted; reads at 38 sites; click to open one', fields: 'single', size: [4, 8],
    render: (d, [f]) => <SiteRank rows={d.rows} meta={d.meta} field={f} onOpenSite={d.openSite} /> },
  { kind: 'barplot', label: 'Barplot', group: 'Ranking', blurb: 'one field by site', fields: 'single', size: [6, 5],
    render: (d, [f]) => <Barplot rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'radar', label: 'Site radar', group: 'Ranking', blurb: 'sites across many fields', fields: 'none', size: [4, 5],
    render: (d) => <SiteRadar rows={d.rows} meta={d.meta} /> },
  { kind: 'parallel', label: 'Parallel coordinates', group: 'Ranking', blurb: 'every filter across many fields', fields: 'none', size: [8, 5],
    render: (d) => <ParallelCoordinates rows={d.rows} meta={d.meta} /> },

  { kind: 'composition', label: 'Composition', group: 'Composition', blurb: 'what the PM2.5 is made of', fields: 'none', size: [4, 5],
    render: (d) => <CompositionTreemap rows={d.rows} meta={d.meta} /> },
  { kind: 'fgroups', label: 'Functional groups', group: 'Composition', blurb: 'FTIR organic functional groups, stacked', fields: 'none', size: [4, 5],
    render: (d) => <FunctionalGroups rows={d.rows} meta={d.meta} /> },
  { kind: 'map', label: 'Site map', group: 'Map', blurb: 'one field per site on the map', fields: 'single', size: [6, 5],
    render: (d, [f]) => <SiteBubbleMap rows={d.rows} meta={d.meta} field={f} onSite={d.openSite} /> },
  { kind: 'spartancomposition', label: 'SPARTAN composition map', group: 'Map', blurb: "a donut of SPARTAN's PM2.5 reconstruction per site", fields: 'none', size: [10, 7],
    render: () => <SpartanCompositionMap /> },
  { kind: 'sankey', label: 'Method flow', group: 'Flow', blurb: 'site → EC method → HIPS / season', fields: 'none', size: [6, 5],
    render: (d) => <MethodSankey rows={d.rows} meta={d.meta} /> },

  { kind: 'wxtimeline', label: 'Weather timeline', group: 'Weather', blurb: 'daily weather under the filters (one site)', fields: 'single', size: [8, 6],
    render: (d, [f]) => <WeatherTimelineTile rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'wxclimate', label: 'Monthly climate', group: 'Weather', blurb: 'temperature and rain by month against the season calendar', fields: 'none', size: [4, 6],
    render: (d) => <MonthlyClimateTile rows={d.rows} meta={d.meta} /> },
  { kind: 'wxwashout', label: 'Rain washout', group: 'Weather', blurb: "the measurement by how wet each filter's window was", fields: 'single', size: [4, 6],
    render: (d, [f]) => <RainWashoutTile rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'wxroses', label: 'Wind roses', group: 'Weather', blurb: 'where the air came from while the filters sampled', fields: 'none', size: [4, 6],
    render: (d) => <WindRosesTile rows={d.rows} meta={d.meta} pmfLabelOfDate={d.pmfLabelOfDate} /> },
  { kind: 'wxpollrose', label: 'Pollution rose', group: 'Weather', blurb: 'the measurement by the direction the air came from', fields: 'single', size: [4, 6],
    render: (d, [f]) => <PollutionRoseTile rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'wxpolar', label: 'Polar plot', group: 'Weather', blurb: 'the measurement by wind direction and speed', fields: 'single', size: [4, 6],
    render: (d, [f]) => <PolarPlotTile rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'wxspeed', label: 'By wind speed', group: 'Weather', blurb: 'median and IQR of the measurement by wind speed, per site', fields: 'single', size: [4, 5],
    render: (d, [f]) => <WindSpeedTile rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'wxcorr', label: 'Weather correlations', group: 'Weather', blurb: 'which weather goes with each measurement, by season', fields: 'none', size: [6, 6],
    render: (d) => <WeatherCorrelationsTile rows={d.rows} meta={d.meta} /> },
  { kind: 'wxsummary', label: 'Weather by site', group: 'Weather', blurb: 'per site: which weather goes with the measurement; click to open a site', fields: 'single', size: [8, 6],
    render: (d, [f]) => <WeatherSummaryTile rows={d.rows} meta={d.meta} field={f} onOpenSite={d.openSite} /> },

  { kind: 'calcross', label: 'Calibration crossplot', group: 'Calibration', blurb: 'one run: predicted FTIR EC against the reference, per filter', fields: 'none', size: [6, 7],
    available: (d) => !!d.calibration, render: (d) => d.calibration && <RunCrossplotTile cal={d.calibration} /> },
  { kind: 'calcv', label: 'Cross-validation curve', group: 'Calibration', blurb: 'one run: error against the number of PLS components', fields: 'none', size: [6, 5],
    available: (d) => !!d.calibration, render: (d) => d.calibration && <CVCurveTile cal={d.calibration} /> },

  { kind: 'sourcestack', label: 'PMF sources over time', group: 'PMF sources', blurb: 'factor contributions, Addis 2023', fields: 'none', size: [8, 5],
    available: (d) => !!d.pmf, render: (d) => d.pmf && <SourceStack pmf={d.pmf} rows={d.pmfRows} meta={d.meta} /> },
  { kind: 'sourceseason', label: 'PMF sources by season', group: 'PMF sources', blurb: 'factor shares per season', fields: 'none', size: [4, 5],
    available: (d) => !!d.pmf, render: (d) => d.pmf && <SourceStack pmf={d.pmf} rows={d.pmfRows} meta={d.meta} id="sourceseason" initialGrouping="by season" title="Source apportionment by season" /> },
]

/**
 * Add a chart kind to the Dashboard's Add chart menu. Any component that takes the
 * gallery's rows and meta can be a tile; register it once, next to the chart, and every
 * dashboard can use it. A kind already registered is replaced.
 */
export function registerDashboardChart(chart: DashboardChart) {
  const i = CHARTS.findIndex((c) => c.kind === chart.kind)
  if (i >= 0) CHARTS[i] = chart
  else CHARTS.push(chart)
}

export const dashboardCharts = (): readonly DashboardChart[] => CHARTS
export const dashboardChart = (kind: string) => CHARTS.find((c) => c.kind === kind)

const presence = new WeakMap<FilterRow[], Set<string>>()
/** the fields with at least one value among these rows (cached per rows array) */
export function fieldsPresent(rows: FilterRow[]): Set<string> {
  let out = presence.get(rows)
  if (!out) {
    out = new Set()
    for (const r of rows) for (const k in r) if (!out.has(k) && typeof r[k] === 'number' && Number.isFinite(r[k] as number)) out.add(k)
    presence.set(rows, out)
  }
  return out
}

/**
 * A starter field as the rows in view have it (lib/fieldEquivalents): itself where measured, else SPARTAN's
 * documented stand-in (public HIPS or reflectometer BC for HIPS BC, public FTIR or TOR carbon for FTIR carbon),
 * else itself, so the tile can say why it is empty. Measured means under the chosen baseline (`eff`).
 */
export function resolveStarter(field: string, d: DashboardData): string {
  const has = fieldsPresent(d.measured?.length ? d.measured : d.rows)
  return resolveField(field, (f) => has.has(d.eff(f))).field
}

/** a new tile's fields, from the sidebar's choices */
export const startFields = (chart: DashboardChart, d: DashboardData): string[] =>
  chart.fields === 'pair' ? [resolveStarter(d.defaults.x, d), resolveStarter(d.defaults.y, d)] : chart.fields === 'single' ? [resolveStarter(d.defaults.field, d)]
    : chart.fields === 'trio' ? ['PM2.5 mass', 'EC (FTIR)', 'HIPS BC'].map((f) => resolveStarter(f, d)) : []

type Start = { kind: string; w: number; h: number; fields?: string[] }
const KPIS: Start[] = ['PM2.5 mass', 'EC (FTIR)', 'HIPS BC', 'OC (FTIR)'].map((f) => ({ kind: 'stat', w: 3, h: 2, fields: [f] }))
/** one site's top row: what the site is, then three headline numbers */
const SITE_TOP: Start[] = [{ kind: 'sitesummary', w: 3, h: 2 }, ...KPIS.slice(0, 3)]
const EC = ['EC (FTIR)']

/**
 * What each scope opens with before the reader changes it: a few named sheets, each laid
 * out to fill the 12 × 12 grid. One site: how it varies over time and season. Compare: the
 * crossplots that matter here, with the sites in one colour each (BC against EC is the
 * calibration question; OC, mass, potassium and dust say what kind of carbon it is), then what
 * tracks EC and BC with a column per site (a Compare ranking opens side by side), then the
 * same numbers per site (cards), then the charts that set sites side by side. Network: map
 * first (a click opens a site), then the network's charts. A field the sites in view never measured moves
 * to its documented stand-in (resolveStarter); one this network does not have at all, with no stand-in,
 * falls back to the chart's own start fields.
 */
export const STARTER: Record<Scope, { name: string; tiles: Start[] }[]> = {
  site: [
    { name: 'Overview', tiles: [...SITE_TOP,
      { kind: 'timeseries', w: 8, h: 5, fields: EC }, { kind: 'clock', w: 4, h: 5, fields: EC },
      { kind: 'scatter', w: 4, h: 5 }, { kind: 'composition', w: 4, h: 5 }, { kind: 'histogram', w: 4, h: 5, fields: EC }] },
    { name: 'Weather', tiles: [
      { kind: 'wxtimeline', w: 8, h: 7, fields: EC }, { kind: 'wxclimate', w: 4, h: 7 },
      { kind: 'wxwashout', w: 4, h: 5, fields: EC }, { kind: 'wxroses', w: 4, h: 5 }, { kind: 'wxcorr', w: 4, h: 5 }] },
    { name: 'Seasons', tiles: [
      { kind: 'monthlyband', w: 6, h: 6, fields: EC }, { kind: 'monthlyheatmap', w: 6, h: 6, fields: EC },
      { kind: 'distribution', w: 6, h: 6, fields: EC }, { kind: 'ridgeline', w: 6, h: 6, fields: EC }] },
  ],
  sites: [
    { name: 'Crossplots', tiles: [
      { kind: 'scatter', w: 4, h: 6, fields: ['HIPS BC', 'EC (FTIR)'] }, { kind: 'scatter', w: 4, h: 6, fields: ['EC (FTIR)', 'OC (FTIR)'] },
      { kind: 'scatter', w: 4, h: 6, fields: ['PM2.5 mass', 'EC (FTIR)'] },
      { kind: 'scatter', w: 4, h: 6, fields: ['Potassium', 'EC (FTIR)'] }, { kind: 'scatter', w: 4, h: 6, fields: ['Silicon', 'Iron'] },
      { kind: 'corrrank', w: 4, h: 6, fields: EC }] },
    { name: 'Side by side', tiles: [{ kind: 'corrrank', w: 6, h: 12, fields: EC }, { kind: 'corrrank', w: 6, h: 12, fields: ['HIPS BC'] }] },
    { name: 'Cards', tiles: [{ kind: 'sitecards', w: 12, h: 4 }, { kind: 'barplot', w: 6, h: 8, fields: EC }, { kind: 'ridgeline', w: 6, h: 8, fields: EC }] },
    { name: 'Charts', tiles: [...KPIS,
      { kind: 'map', w: 5, h: 5, fields: EC }, { kind: 'barplot', w: 7, h: 5, fields: EC },
      { kind: 'scatter', w: 4, h: 5 }, { kind: 'ridgeline', w: 4, h: 5, fields: EC }, { kind: 'monthlyband', w: 4, h: 5, fields: EC }] },
  ],
  network: [
    { name: 'Map', tiles: [{ kind: 'map', w: 8, h: 12, fields: ['PM2.5 mass'] }, { kind: 'sitecards', w: 4, h: 12 }] },
    { name: 'Charts', tiles: [...KPIS,
      { kind: 'siterank', w: 4, h: 10, fields: ['PM2.5 mass'] }, { kind: 'siterank', w: 4, h: 10, fields: EC },
      { kind: 'scatter', w: 4, h: 5 }, { kind: 'corrrank', w: 4, h: 5, fields: EC }] },
  ],
}
