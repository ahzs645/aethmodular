import type { ReactNode } from 'react'
import type { AxesOpts } from '@/lib/axes'
import type { FilterRow, MetaFile, PmfFile, PmfRow } from '@/lib/types'
import type { Scope } from '@/lib/dashboard'
import { StatTile } from '@/charts/summary/StatTile'
import { SiteCards } from '@/charts/summary/SiteCards'
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
import { SiteBubbleMap } from '@/charts/map/SiteBubbleMap'
import { MethodSankey } from '@/charts/flow/MethodSankey'
import { SourceStack } from '@/charts/pmf/SourceStack'
import { SourceSeasonality } from '@/charts/pmf/SourceSeasonality'

/** What App hands every tile: the Filters panel's rows and meta, and what a few charts need besides. */
export interface DashboardData {
  rows: FilterRow[]
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
  { kind: 'barplot', label: 'Barplot', group: 'Ranking', blurb: 'one field by site', fields: 'single', size: [6, 5],
    render: (d, [f]) => <Barplot rows={d.rows} meta={d.meta} field={f} /> },
  { kind: 'radar', label: 'Site radar', group: 'Ranking', blurb: 'sites across many fields', fields: 'none', size: [4, 5],
    render: (d) => <SiteRadar rows={d.rows} meta={d.meta} /> },
  { kind: 'parallel', label: 'Parallel coordinates', group: 'Ranking', blurb: 'every filter across many fields', fields: 'none', size: [8, 5],
    render: (d) => <ParallelCoordinates rows={d.rows} meta={d.meta} /> },

  { kind: 'composition', label: 'Composition', group: 'Part of a whole', blurb: 'what the PM2.5 is made of', fields: 'none', size: [4, 5],
    render: (d) => <CompositionTreemap rows={d.rows} meta={d.meta} /> },
  { kind: 'map', label: 'Site map', group: 'Map', blurb: 'one field per site on the map', fields: 'single', size: [6, 5],
    render: (d, [f]) => <SiteBubbleMap rows={d.rows} meta={d.meta} field={f} onSite={d.openSite} /> },
  { kind: 'sankey', label: 'Method flow', group: 'Flow', blurb: 'site → EC method → HIPS / season', fields: 'none', size: [6, 5],
    render: (d) => <MethodSankey rows={d.rows} meta={d.meta} /> },

  { kind: 'sourcestack', label: 'PMF sources over time', group: 'PMF sources', blurb: 'factor contributions, Addis 2023', fields: 'none', size: [8, 5],
    available: (d) => !!d.pmf, render: (d) => d.pmf && <SourceStack pmf={d.pmf} rows={d.pmfRows} meta={d.meta} /> },
  { kind: 'sourceseason', label: 'PMF sources by season', group: 'PMF sources', blurb: 'factor shares per season', fields: 'none', size: [4, 5],
    available: (d) => !!d.pmf, render: (d) => d.pmf && <SourceSeasonality pmf={d.pmf} rows={d.pmfRows} meta={d.meta} /> },
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

/** a new tile's fields, from the sidebar's choices */
export const startFields = (chart: DashboardChart, d: DashboardData): string[] =>
  chart.fields === 'pair' ? [d.defaults.x, d.defaults.y] : chart.fields === 'single' ? [d.defaults.field]
    : chart.fields === 'trio' ? ['PM2.5 mass', 'EC (FTIR)', 'HIPS BC'] : []

type Start = { kind: string; w: number; h: number; fields?: string[] }
const KPIS: Start[] = ['PM2.5 mass', 'EC (FTIR)', 'HIPS BC', 'OC (FTIR)'].map((f) => ({ kind: 'stat', w: 3, h: 2, fields: [f] }))
const EC = ['EC (FTIR)']

/**
 * What each scope opens with before the reader changes it: a few named sheets, each laid
 * out to fill the 12 × 12 grid. One site: how it varies over time and season. Compare: the
 * same numbers per site (cards), then the charts that set sites side by side. Network: map
 * first (a click opens a site), then the network's charts.
 */
export const STARTER: Record<Scope, { name: string; tiles: Start[] }[]> = {
  site: [
    { name: 'Overview', tiles: [...KPIS,
      { kind: 'timeseries', w: 8, h: 5, fields: EC }, { kind: 'clock', w: 4, h: 5, fields: EC },
      { kind: 'scatter', w: 4, h: 5 }, { kind: 'composition', w: 4, h: 5 }, { kind: 'histogram', w: 4, h: 5, fields: EC }] },
    { name: 'Seasons', tiles: [
      { kind: 'monthlyband', w: 6, h: 6, fields: EC }, { kind: 'monthlyheatmap', w: 6, h: 6, fields: EC },
      { kind: 'distribution', w: 6, h: 6, fields: EC }, { kind: 'ridgeline', w: 6, h: 6, fields: EC }] },
  ],
  sites: [
    { name: 'Cards', tiles: [{ kind: 'sitecards', w: 12, h: 4 }, { kind: 'barplot', w: 6, h: 8, fields: EC }, { kind: 'ridgeline', w: 6, h: 8, fields: EC }] },
    { name: 'Charts', tiles: [...KPIS,
      { kind: 'map', w: 5, h: 5, fields: EC }, { kind: 'barplot', w: 7, h: 5, fields: EC },
      { kind: 'scatter', w: 4, h: 5 }, { kind: 'ridgeline', w: 4, h: 5, fields: EC }, { kind: 'monthlyband', w: 4, h: 5, fields: EC }] },
  ],
  network: [
    { name: 'Map', tiles: [{ kind: 'map', w: 8, h: 12, fields: ['PM2.5 mass'] }, { kind: 'sitecards', w: 4, h: 12 }] },
    { name: 'Charts', tiles: [...KPIS,
      { kind: 'distribution', w: 8, h: 5, fields: ['PM2.5 mass'] }, { kind: 'corrrank', w: 4, h: 5, fields: EC },
      { kind: 'scatter', w: 4, h: 5 }, { kind: 'availability', w: 8, h: 5 }] },
  ],
}
