import { createContext, useContext, useMemo, type ComponentType, type ReactNode } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { mediaKey, useEtadByMedia } from '@/lib/etadMedia'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { ScatterPlot, type ScatterPoint } from '@/components/ScatterPlot'
import { baseFilterId } from '@/lib/highlight'
import { ecdf } from '@/lib/stats'
import { useHighlightActions } from '@/lib/highlight'
import { PanelAxesControl } from '@/lib/panelAxes'
import type { PanelDim } from '@/lib/axes'
import { FacetGrid } from '@/components/FacetGrid'
import { HeatMatrix } from '@/components/HeatMatrix'
import { BarRows, type BarRow } from '@/components/BarRows'
import { IntervalStrip, type IntervalBlock, type IntervalMark, type IntervalPanel } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { Legend, useLegend } from '@/components/Legend'
import { ANALOG_SELECTION_EXCLUDED, OCT1_ALSO_EXCLUDED, type Region } from '@/components/SpectralRegions'
import { SpectrumPlot, type SpectrumAnnotation, type SpectrumBand, type SpectrumSeries } from '@/components/SpectrumPlot'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { coef, f2, f3, fmt, pct, percent } from '@/lib/format'
import { Markdown } from '@/lib/markdown'
import { densities, num, rowsOf, summarize, uniq, where, INTERVAL_LABEL, type Interval, type Row, type Table, type Value } from '@/lib/table'
import { BASELINE_COLOR, FONT, INK, SPECTRAL_REGION } from '@/lib/theme'
import { useChartOptions, type OptionDef } from '@/lib/chartOptions'
import { useColourScheme } from '@/lib/colourSchemes'
import { TABLEAU } from '@/lib/palettes'

/**
 * A chart from a description: the chart kind and which columns of a long
 * table to draw. The same few chart types every page already uses
 * (Crossplot, SpectrumPlot, SweepPlot, IntervalStrip, HeatMatrix, BarRows, table), so a
 * figure declared here looks, hovers, pins and exports like a hand-built one.
 *
 *   <Figure kind="spectrum" data={rows} wn="wn" y="median" series="series" lo="q25" hi="q75" title="…" />
 *
 * On a spec-driven meeting page (pages/MeetingSpecPage) the data is a table
 * name in the meeting's bundle instead of rows. Anything a description cannot
 * say (linked charts, live refits, a bespoke map) is a `custom` figure: a
 * component registered with `registerFigure`, which receives the bundle.
 */
export type FormatName = 'f2' | 'f3' | 'coef' | 'pct' | 'int' | 'text' | 'percent'
interface Base {
  title: string
  subtitle?: string
  tip?: string
  /** anchor id; also the prefix of this figure's URL keys */
  id?: string
  /** a table in the meeting bundle, or rows passed directly with `data` */
  table?: string
  data?: Table
  where?: Record<string, Value | Value[]>
  /** controls that keep only the rows with the chosen value of a column */
  choose?: { column: string; label?: string; default?: string }[]
  /**
   * reader-facing variants of the same figure: each choice patches the spec (a log y scale,
   * another column on x, a facet). Rules as everywhere in the gallery (lib/chartOptions): a
   * variant or a choice with `when` is offered only while it holds for the values chosen so far
   * (earlier variants and the `choose` columns), and a choice no longer offered falls back.
   * A patch changes how the figure is drawn, not where its rows come from (`table`, `data`,
   * `where` are read before the choices resolve).
   */
  variants?: FigureVariant[]
  /** colours for category values, over the bundle's and the gallery's (baselines, seasons) */
  colors?: Record<string, string>
  /** a note under the chart, Markdown */
  note?: string
  /** Measurement or scope information needed even with Explain off. */
  noteAlways?: boolean
  /**
   * Filters → excluded: Mark (lib/markedRows useExclusionSplit): rows of the same shape the gallery's exclusions take
   * out of `data`; a crossplot draws them as muted ×, never fitted or counted (the same `where`, `choose` and facets apply)
   */
  markedData?: Table
}
export type FigureSpec =
  | (Base & { kind: 'crossplot'; x: string; y: string; color?: string; idColumn?: string; tipColumns?: string[]; xLabel: string; yLabel: string; fit?: boolean; errorRatio?: number; facet?: string; maxSide?: number; sameScale?: boolean })
  | (Base & { kind: 'scatter'; x: string; y: string; color?: string; idColumn?: string; tipColumns?: string[]; xLabel: string; yLabel: string; refX?: number[]; refY?: number[]; facet?: string; maxSide?: number })
  | (Base & { kind: 'spectrum'; wn: string; y: string; series: string; lo?: string; hi?: string; bandLabel?: string; role?: 'trace' | 'line'; window?: [number, number]; yLabel?: string; yClip?: [number, number]; includeZero?: boolean; zeroLine?: boolean; regions?: RegionName[]; annotations?: SpectrumAnnotation[]; hover?: 'nearest' | 'readout'; facet?: string; height?: number })
  | (Base & { kind: 'sweep'; x: string; y: string; series?: string; lo?: string; hi?: string; bandLabel?: string; aggregate?: { stat?: 'median' | 'mean'; interval?: Interval }; xLabel: string; yLabel: string; xScale?: 'linear' | 'log' | 'time' | 'point'; yScale?: 'linear' | 'log'; reference?: number; xMarkers?: { at: number; label?: string }[]; chosen?: { series: string; x: Value }[]; interval?: 'band' | 'whisker'; yClip?: [number, number]; facet?: string; height?: number })
  | (Base & { kind: 'intervals'; row: string; series?: string; block?: string; panels: { value: string; label: string; reference?: number; lo?: string; hi?: string }[]; aggregate?: { stat?: 'median' | 'mean'; interval?: Interval }; labelWidth?: number })
  | (Base & { kind: 'histogram'; value: string; series: string; log?: boolean; bins?: number; xLabel: string; medians?: boolean; height?: number })
  | (Base & { kind: 'ecdf'; value: string; count?: string; series?: string; xLabel: string; height?: number })
  | (Base & { kind: 'heatmap'; row: string; column: string; value: string; domain?: [number, number]; ramp?: 'diverging' | 'sequential'; format?: FormatName; valueLabel?: string; cell?: [number, number] })
  | (Base & { kind: 'bars'; label: string; value: string; color?: string; sort?: 'none' | 'desc' | 'asc'; format?: FormatName; xLabel?: string; reference?: number; domain?: [number, number] })
  | (Base & { kind: 'table'; columns?: { key: string; label?: string; format?: FormatName }[] })
  | (Base & { kind: 'custom'; component: string; props?: Record<string, unknown> })
export type RegionName = 'analog' | 'oct1' | 'ptfe' | 'co2'
export interface FigureVariant {
  key: string
  label?: string
  default?: string
  title?: string
  when?: (chosen: Record<string, unknown>) => boolean
  choices: { value: string; label?: string; patch?: Record<string, unknown>; when?: (chosen: Record<string, unknown>) => boolean }[]
}

export interface MeetingBundle {
  schema_version: number
  id: string
  generated_from: string
  colors: Record<string, string>
  text: Record<string, string>
  tables: Record<string, Table>
}
export const BundleContext = createContext<MeetingBundle | null>(null)

const CUSTOM = new Map<string, ComponentType<{ bundle: MeetingBundle | null; props?: Record<string, unknown> }>>()
/** Register a hand-built chart under a name, for `{ kind: 'custom', component: name }`. */
export function registerFigure(name: string, component: ComponentType<{ bundle: MeetingBundle | null; props?: Record<string, unknown> }>) { CUSTOM.set(name, component) }

const REGIONS: Record<RegionName, readonly Region[]> = {
  analog: ANALOG_SELECTION_EXCLUDED, oct1: OCT1_ALSO_EXCLUDED, ptfe: [SPECTRAL_REGION.ptfe], co2: [SPECTRAL_REGION.co2],
}
export const FORMAT: Record<FormatName, (v: unknown) => string> = {
  f2, f3, coef, pct: (v) => pct(v), percent: (v) => percent(v),
  int: (v) => (typeof v === 'number' ? Math.round(v).toLocaleString() : '—'), text: (v) => (v == null ? '—' : String(v)),
}
const PALETTE = [TABLEAU.blue, TABLEAU.orange, TABLEAU.green, TABLEAU.red, TABLEAU.teal, TABLEAU.yellow, TABLEAU.purple, TABLEAU.pink, TABLEAU.brown, TABLEAU.grey]

/** One colour per category: the figure's own map, then the bundle's, then the gallery's baselines, then a fixed palette. */
function useColors(spec: Base) {
  const bundle = useContext(BundleContext)
  return useMemo(() => {
    const seen = new Map<string, string>()
    return (v: Value) => {
      const k = String(v)
      const c = spec.colors?.[k] ?? bundle?.colors?.[k] ?? BASELINE_COLOR[k]
      if (c) return c
      if (!seen.has(k)) seen.set(k, PALETTE[seen.size % PALETTE.length])
      return seen.get(k)!
    }
  }, [spec.colors, bundle])
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40)

// research-spectra: `provenance` (which notebook a figure stands in for) passes through to its ChartFrame
export function Figure(props: FigureSpec & { about?: ReactNode; provenance?: ReactNode; aside?: ReactNode
  /** what a chart with no rows says, where the reader can do something about it (e.g. which Samples to pick) */
  empty?: ReactNode }) {
  let spec: FigureSpec = props
  const bundle = useContext(BundleContext)
  const id = spec.id ?? slug(spec.title)
  const all = useMemo(() => where(rowsOf(spec.data ?? (spec.table ? bundle?.tables[spec.table] : null)), spec.where), [spec.data, spec.table, spec.where, bundle])
  // `choose` columns and `variants` resolve together, in order, through the gallery's options
  // engine; both are kept in the URL (`${id}.${column}` / `${id}.${key}`)
  const variants = spec.variants ?? []
  const o = useChartOptions(id, [
    ...(spec.choose ?? []).map((c): OptionDef => {
      const options = uniq(all, c.column).map(String)
      return { key: c.column, label: c.label ?? c.column, persist: true, default: c.default ?? options[0] ?? '', style: options.length > 4 ? 'select' : 'segmented', choices: options }
    }),
    ...variants.map((v): OptionDef => ({ key: v.key, label: v.label ?? v.key, persist: true, default: v.default ?? v.choices[0]?.value ?? '', title: v.title,
      when: v.when, choices: v.choices.map((c) => ({ value: c.value, label: c.label, when: c.when })) })),
  ])
  const chosen = (r: Row) => (spec.choose ?? []).every((c) => String(r[c.column]) === o.values[c.column])
  const rows = useMemo(() => all.filter(chosen),
    [all, (spec.choose ?? []).map((c) => o.values[c.column]).join('|')]) // eslint-disable-line react-hooks/exhaustive-deps
  const marked = useMemo(() => (spec.markedData ? where(rowsOf(spec.markedData), spec.where).filter(chosen) : []),
    [spec.markedData, spec.where, (spec.choose ?? []).map((c) => o.values[c.column]).join('|')]) // eslint-disable-line react-hooks/exhaustive-deps
  // the chosen variants' patches, applied over the spec in the order the variants are declared
  spec = variants.reduce((acc, v) => ({ ...acc, ...(v.choices.find((c) => c.value === o.values[v.key])?.patch ?? {}) }), spec) as FigureSpec
  // a faceted crossplot offers the gallery's Panel axes (lib/panelAxes) while it draws more than one panel
  const panelAxes = spec.kind === 'crossplot' && !!spec.facet && uniq(rows, spec.facet).length > 1
  const facetDim = spec.kind === 'crossplot' ? facetDimOf(spec.facet) : null
  const controls = (spec.choose?.length || variants.length || panelAxes) ? <>{(spec.choose?.length || variants.length) ? o.controls() : null}{panelAxes && <PanelAxesControl dims={facetDim ? [facetDim] : []} />}</> : undefined

  const Custom = spec.kind === 'custom' ? CUSTOM.get(spec.component) : null
  const body = spec.kind === 'custom'
    ? (Custom ? <Custom bundle={bundle} props={spec.props} /> : <Empty>No figure registered as “{spec.component}”.</Empty>)
    : !rows.length ? <Empty>{props.empty ?? (spec.table && bundle && !bundle.tables[spec.table] ? `No table “${spec.table}” in this meeting’s data.` : 'No rows to draw.')}</Empty>
    : spec.kind === 'crossplot' ? <CrossplotFigure spec={spec} rows={rows} marked={marked} />
    : spec.kind === 'scatter' ? <ScatterFigure spec={spec} rows={rows} marked={marked} />
    : spec.kind === 'spectrum' ? <SpectrumFigure spec={spec} rows={rows} />
    : spec.kind === 'sweep' ? <SweepFigure spec={spec} rows={rows} />
    : spec.kind === 'intervals' ? <IntervalsFigure spec={spec} rows={rows} />
    : spec.kind === 'histogram' ? <HistogramFigure spec={spec} rows={rows} />
    : spec.kind === 'ecdf' ? <EcdfFigure spec={spec} rows={rows} />
    : spec.kind === 'heatmap' ? <HeatmapFigure spec={spec} rows={rows} />
    : spec.kind === 'bars' ? <BarsFigure spec={spec} rows={rows} />
    : <TableFigure spec={spec} rows={rows} />
  return (
    <ChartFrame id={id} title={spec.title} subtitle={spec.subtitle} tip={spec.tip} controls={controls} about={props.about} provenance={props.provenance}
      exportable={spec.kind !== 'table'} source={bundle ? `gallery/app/src/meetings/${bundle.id}.meeting.ts › ${id}${spec.table ? `; data: table "${spec.table}" in gallery/app/public/data/meetings/${bundle.id}.json, from ${bundle.generated_from}` : ''}` : undefined}>
      {body}
      {spec.note && <div className={spec.noteAlways ? 'legend-definition' : 'legend-note'} data-export={spec.noteAlways ? 'text' : undefined} style={{ marginTop: 6 }}><Markdown text={spec.note} /></div>}
      {props.aside}
    </ChartFrame>
  )
}

/** a tipColumns line: a column the row does not carry (undefined) is left out, so a column can annotate some points
 * only; null prints '—'; a fractional number prints to 3 decimals instead of every digit of the float */
const tipLine = (c: string, v: Value | undefined): string[] =>
  v === undefined ? [] : [`${c}: ${typeof v === 'number' && !Number.isInteger(v) ? fmt(v, 3) : v ?? '—'}`]

/** what a crossplot's facets split by, for Panel axes (a facet on neither site nor season collapses Per site/Per season to Shared) */
const facetDimOf = (facet?: string): PanelDim | null => (facet === 'season' ? 'season' : facet === 'site' || facet === 'Site' ? 'site' : null)

const facetsOf = (rows: Row[], facet?: string) => (facet ? uniq(rows, facet).map((f) => ({ key: String(f), rows: rows.filter((r) => r[facet] === f) })) : [{ key: '', rows }])

// ------------------------------------------------------------------ crossplot
function CrossplotFigure({ spec, rows, marked = [] }: { spec: Extract<FigureSpec, { kind: 'crossplot' }>; rows: Row[]; marked?: Row[] }) {
  const color = useColors(spec)
  const groups = spec.color ? uniq(rows, spec.color) : []
  // a MediaId column joins the gallery's Addis filters through etad_ids.json, so the points follow Filters → excluded
  const byMedia = useEtadByMedia(spec.idColumn === 'MediaId')
  // a point that maps to a gallery filter opens it in the sample drawer on click, as the other crossplots do
  const { openSample } = useHighlightActions()
  const toPoints = (rs: Row[]): CrossPoint[] => rs.flatMap((r) => {
    const x = num(r[spec.x]), y = num(r[spec.y])
    if (x == null || y == null) return []
    const raw = spec.idColumn ? String(r[spec.idColumn]) : undefined
    const etad = byMedia && raw !== undefined ? byMedia.get(mediaKey(raw)) : undefined
    const idv = etad ?? raw
    return [{ x, y, color: spec.color ? color(r[spec.color]) : INK.text, id: idv, open: etad ? () => openSample(etad) : undefined,
      tip: [...(etad ? [`${etad} · MediaId ${mediaKey(raw)}`] : idv ? [idv] : []), ...(spec.tipColumns ?? (spec.color ? [spec.color] : [])).flatMap((c) => tipLine(c, r[c])), `${spec.xLabel}: ${f2(x)}`, `${spec.yLabel}: ${f2(y)}`] }]
  })
  const facets = facetsOf(rows, spec.facet)
  const sameScale = spec.sameScale ?? true
  const markedOf = (key: string) => toPoints(spec.facet && key ? marked.filter((r) => String(r[spec.facet!]) === key) : marked)
  const plot = (f: { key: string; rows: Row[] }, side?: number) => <Crossplot title={f.key || ''} points={toPoints(f.rows)} marked={marked.length ? markedOf(f.key) : undefined} xLabel={spec.xLabel} yLabel={spec.yLabel}
    fit={spec.fit ?? true} errorRatio={spec.errorRatio} maxSide={side ?? spec.maxSide ?? 480} sameScale={sameScale} />
  return <>
    {/* Panel axes (the facets' panel group): Shared puts every facet on the range of all of them, Own each on its own;
        the control is in the figure's options, so the group leaves its own out */}
    {facets.length > 1 ? <FacetGrid items={facets} minCell={300} maxCols={3} itemKey={(f) => f.key} share axesControl={false}
      site={facetDimOf(spec.facet) === 'site' ? (f) => f.key : undefined} render={(f, w) => plot(f, w - 70)} /> : plot(facets[0])}
    {groups.length > 0 && <Key items={[...groups.map((g) => ({ label: String(g), color: color(g) })),
      ...(spec.sameScale ?? true ? [{ label: '1:1', color: INK.identity, kind: 'dash' as const }] : [])]} />}
  </>
}

// Different quantities (PCA scores, residual vs domain distance): no 1:1 or
// origin-based crossplot ranges. ScatterPlot owns marks, axes, exclusions and hover.
function ScatterFigure({ spec, rows, marked }: { spec: Extract<FigureSpec, { kind: 'scatter' }>; rows: Row[]; marked: Row[] }) {
  const color = useColors(spec)
  const byMedia = useEtadByMedia(spec.idColumn === 'MediaId')
  const located = useMemo(() => [...rows, ...marked].flatMap((r): { point: ScatterPoint; facet: string }[] => {
    const x = num(r[spec.x]), y = num(r[spec.y])
    if (x == null || y == null) return []
    const raw = spec.idColumn && r[spec.idColumn] != null ? String(r[spec.idColumn]) : undefined
    const id = raw ? (spec.idColumn === 'MediaId' ? byMedia?.get(mediaKey(raw)) : baseFilterId(raw)) : undefined
    return [{ facet: spec.facet ? String(r[spec.facet]) : '', point: { x, y, id,
      key: `${id ?? ''}|${r[spec.color ?? ''] ?? ''}|${x}|${y}`, color: spec.color ? color(r[spec.color]) : INK.accent,
      tip: [...(id ? [id] : []), ...(spec.tipColumns ?? (spec.color ? [spec.color] : [])).map((k) => `${k}: ${r[k] ?? '—'}`), `${spec.xLabel}: ${f3(x)}`, `${spec.yLabel}: ${f3(y)}`] } }]
  }), [rows, marked, spec.x, spec.y, spec.idColumn, spec.color, spec.facet, spec.xLabel, spec.yLabel, spec.tipColumns?.join('|'), byMedia, color])
  const points = useMemo(() => located.map((p) => p.point), [located])
  const groups = spec.color ? uniq(rows, spec.color) : []
  const facets = spec.facet ? uniq([...rows, ...marked], spec.facet).map(String) : ['']
  const facetPoints = useMemo(() => new Map(facets.map((key) => [key, located.filter((p) => p.facet === key).map((p) => p.point)])), [located, facets.join('|')])
  return <>
    {facets.length > 1 ? <FacetGrid items={facets} minCell={300} maxCols={3} render={(key, w) =>
      <ScatterPlot points={facetPoints.get(key) ?? []} title={key} xLabel={spec.xLabel} yLabel={spec.yLabel} maxSide={w - 70} refX={spec.refX} refY={spec.refY} note="mark" swap />} />
      : <ScatterPlot points={points} xLabel={spec.xLabel} yLabel={spec.yLabel} maxSide={spec.maxSide ?? 480} refX={spec.refX} refY={spec.refY} note="mark" swap />}
    {groups.length > 0 && <Key items={groups.map((g) => ({ label: String(g), color: color(g) }))} />}
  </>
}

function EcdfFigure({ spec, rows }: { spec: Extract<FigureSpec, { kind: 'ecdf' }>; rows: Row[] }) {
  const color = useColors(spec), lg = useLegend()
  const names = spec.series ? uniq(rows, spec.series).map(String) : [spec.xLabel]
  const series = useMemo(() => names.filter(lg.show).map((name): SweepSeries => ({
    id: name, label: name, color: color(name), curve: 'step', marks: false,
    points: (() => { const group = rows.filter((r) => !spec.series || String(r[spec.series]) === name)
      const pts = ecdf(group.map((r) => num(r[spec.value])), spec.count ? group.map((r) => num(r[spec.count!])) : undefined)
      return pts.length ? [{ x: pts[0].x, y: 0 }, ...pts] : [] })(),
  })), [rows, spec.value, spec.count, spec.series, color, names.join('|'), lg.show])
  return <><SweepPlot series={series} xScale="linear" xLabel={spec.xLabel} yLabel="Cumulative fraction ≤ x" yDomain={[0, 1]} height={spec.height ?? 280} />
    <Legend {...lg.props} items={names.map((name) => ({ label: name, color: color(name), shape: 'line' as const }))} /></>
}

// ------------------------------------------------------------------ spectrum
function SpectrumFigure({ spec, rows }: { spec: Extract<FigureSpec, { kind: 'spectrum' }>; rows: Row[] }) {
  const color = useColors(spec)
  const lg = useLegend()
  const names = uniq(rows, spec.series).map(String)
  const role = spec.role ?? (names.length > 12 ? 'trace' : 'line')
  const build = (rs: Row[]) => {
    const series: SpectrumSeries[] = [], bands: SpectrumBand[] = []
    for (const name of names) {
      if (!lg.show(name)) continue
      const sr = rs.filter((r) => String(r[spec.series]) === name)
      if (!sr.length) continue
      const wn = sr.map((r) => num(r[spec.wn]) ?? NaN)
      const n = sr[0].n
      series.push({ id: name, label: name, wn, y: sr.map((r) => num(r[spec.y]) ?? NaN), color: color(name), role, group: name,
        tip: [name, ...(typeof n === 'number' ? [`n = ${n}`] : [])] })
      if (spec.lo && spec.hi) bands.push({ id: name, wn, lo: sr.map((r) => num(r[spec.lo!]) ?? NaN), hi: sr.map((r) => num(r[spec.hi!]) ?? NaN), color: color(name), group: name })
    }
    return { series, bands }
  }
  const common = { window: spec.window, yLabel: spec.yLabel ?? 'Absorbance', yClip: spec.yClip, includeZero: spec.includeZero, zeroLine: spec.zeroLine,
    regions: (spec.regions ?? []).flatMap((r) => REGIONS[r]), annotations: spec.annotations, hover: spec.hover ?? (role === 'line' ? 'readout' as const : 'nearest' as const), dim: lg.dim }
  const facets = facetsOf(rows, spec.facet)
  return <>
    {facets.length > 1
      ? <FacetGrid items={facets} minCell={300} render={(f, w) => <SpectrumPlot {...build(f.rows)} {...common} width={w} height={spec.height ?? 220} title={f.key} />} />
      : <SpectrumPlot {...build(rows)} {...common} height={spec.height ?? 320} />}
    <Legend {...lg.props} items={names.map((n) => ({ label: n, color: color(n), shape: 'line' as const }))}
      note={spec.lo && spec.hi ? spec.bandLabel ?? 'line: median · band: interquartile range' : undefined} noteAlways={!!spec.bandLabel} />
  </>
}

// ------------------------------------------------------------------ sweep
function SweepFigure({ spec, rows }: { spec: Extract<FigureSpec, { kind: 'sweep' }>; rows: Row[] }) {
  const color = useColors(spec)
  const lg = useLegend()
  const names = spec.series ? uniq(rows, spec.series).map(String) : [spec.yLabel]
  const toX = (v: Value) => (spec.xScale === 'time' ? new Date(String(v)) : typeof v === 'number' ? v : String(v))
  const build = (rs: Row[]): SweepSeries[] => names.filter(lg.show).map((name) => {
    const sr = spec.series ? rs.filter((r) => String(r[spec.series!]) === name) : rs
    const pts = spec.aggregate
      ? summarize(sr, [spec.x], spec.y, spec.aggregate).map((s) => ({ x: toX(s[spec.x] as Value), y: s.mid, lo: s.lo, hi: s.hi, tip: [name, `${spec.xLabel}: ${s[spec.x]}`, `${spec.yLabel}: ${f2(s.mid)} (${f2(s.lo)} to ${f2(s.hi)}), n=${s.n}`] }))
      : sr.map((r) => ({ x: toX(r[spec.x]), y: num(r[spec.y]), lo: spec.lo ? num(r[spec.lo]) : null, hi: spec.hi ? num(r[spec.hi]) : null }))
    return { id: name, label: name, color: color(name), points: pts, interval: spec.interval ?? 'band', group: name }
  })
  const common = { xScale: spec.xScale, xLabel: spec.xLabel, yLabel: spec.yLabel, yScale: spec.yScale, reference: spec.reference,
    xMarkers: spec.xMarkers, chosen: spec.chosen?.map((c) => ({ series: c.series, x: c.x as string | number })), yClip: spec.yClip, dim: lg.dim }
  const facets = facetsOf(rows, spec.facet)
  return <>
    {facets.length > 1
      ? <FacetGrid items={facets} minCell={280} maxCols={3} render={(f, w) => <SweepPlot series={build(f.rows)} {...common} width={w} height={spec.height ?? 220} title={f.key} />} />
      : <SweepPlot series={build(rows)} {...common} height={spec.height ?? 280} />}
    {spec.series && <Legend {...lg.props} items={names.map((n) => ({ label: n, color: color(n), shape: 'line' as const }))}
      note={spec.bandLabel ?? (spec.aggregate ? `${spec.aggregate.stat ?? 'median'} and ${INTERVAL_LABEL[spec.aggregate.interval ?? 'iqr']}` : undefined)} noteAlways={!!spec.bandLabel} />}
  </>
}

// ------------------------------------------------------------------ intervals
function IntervalsFigure({ spec, rows }: { spec: Extract<FigureSpec, { kind: 'intervals' }>; rows: Row[] }) {
  const color = useColors(spec)
  const series = spec.series ? uniq(rows, spec.series).map(String) : ['']
  const blocks = spec.block ? uniq(rows, spec.block).map(String) : ['']
  const agg = spec.aggregate ?? { stat: 'median' as const, interval: 'iqr' as const }
  const by = [spec.block, spec.row, spec.series].filter(Boolean) as string[]
  const stats = spec.panels.map((p) => p.lo && p.hi
    ? rows.map((r) => ({ ...r, mid: num(r[p.value]), lo: num(r[p.lo!]), hi: num(r[p.hi!]) }))
    : summarize(rows, by, p.value, agg))
  const find = (pi: number, b: string, row: string, s: string) => (stats[pi] as Row[]).find((r) =>
    (!spec.block || String(r[spec.block]) === b) && String(r[spec.row]) === row && (!spec.series || String(r[spec.series]) === s))
  const out: IntervalBlock[] = blocks.map((b) => {
    const rs = spec.block ? rows.filter((r) => String(r[spec.block!]) === b) : rows
    return { key: b, title: b || undefined, rows: uniq(rs, spec.row).map(String).map((row) => ({
      key: `${b}|${row}`, label: row,
      marks: spec.panels.map((_, pi) => series.map((s, si): IntervalMark => {
        const r = find(pi, b, row, s)
        return { mid: num(r?.mid ?? null), lo: num(r?.lo ?? null), hi: num(r?.hi ?? null), color: spec.series ? color(s) : INK.text, dy: (si - (series.length - 1) / 2) * 7 }
      })),
      tip: [row, ...series.flatMap((s) => spec.panels.map((p, pi) => { const r = find(pi, b, row, s); return `${s ? s + ' · ' : ''}${p.label}: ${f2(r?.mid)}${r?.lo != null ? ` (${f2(r.lo)} to ${f2(r.hi)})` : ''}` }))],
    })) }
  })
  const panels: IntervalPanel[] = spec.panels.map((p) => ({ label: p.label, reference: p.reference, format: f2 }))
  return <>
    <IntervalStrip blocks={out} panels={panels} labelWidth={spec.labelWidth ?? 180} rowHeight={Math.max(20, series.length * 8 + 6)} />
    {spec.series && <Key items={series.map((s) => ({ label: s, color: color(s) }))} />}
    {!spec.panels.some((p) => p.lo) && <div className="legend-note">{agg.stat ?? 'median'} and {INTERVAL_LABEL[agg.interval ?? 'iqr']}</div>}
  </>
}

// ------------------------------------------------------------------ histogram
function HistogramFigure({ spec, rows }: { spec: Extract<FigureSpec, { kind: 'histogram' }>; rows: Row[] }) {
  const color = useColors(spec)
  const lg = useLegend()
  const names = uniq(rows, spec.series).map(String)
  const d = useMemo(() => densities(names.map((n) => ({ key: n, values: rows.filter((r) => String(r[spec.series]) === n).map((r) => num(r[spec.value])).filter((v): v is number => v != null) })),
    { bins: spec.bins ?? 40, log: spec.log, tail: spec.log ? 0.005 : 0 }), [rows, spec.series, spec.value, spec.bins, spec.log])
  const counts = new Map(names.map((n) => [n, rows.filter((r) => String(r[spec.series]) === n && num(r[spec.value]) != null).length]))
  const series: SweepSeries[] = d.series.filter((s) => lg.show(s.key)).map((s) => ({ id: s.key, label: s.key, color: color(s.key), curve: 'monotone', marks: false, interval: 'none', group: s.key,
    points: s.centers.map((c, i) => ({ x: c, y: s.density[i], tip: [s.key, `${spec.xLabel}: ${f2(c)}`, `density ${f3(s.density[i])}`] })) }))
  return <>
    <SweepPlot series={series} xScale={spec.log ? 'log' : 'linear'} xLabel={spec.xLabel} yLabel={spec.log ? 'Density (per log₁₀ unit)' : 'Density'} include={[0]} height={spec.height ?? 260} dim={lg.dim}
      xMarkers={(spec.medians ?? true) ? d.series.filter((s) => s.median != null && lg.show(s.key)).map((s) => ({ at: s.median!, label: `median ${f2(s.median)}`, color: color(s.key) })) : []} />
    <Legend {...lg.props} items={names.map((n) => ({ label: n, color: color(n), shape: 'line' as const, detail: `n=${(counts.get(n) ?? 0).toLocaleString()}` }))} note="each curve integrates to 1, so populations of different sizes compare" />
  </>
}

// ------------------------------------------------------------------ heat map
function HeatmapFigure({ spec, rows }: { spec: Extract<FigureSpec, { kind: 'heatmap' }>; rows: Row[] }) {
  const rs = useMemo(() => uniq(rows, spec.row).map(String), [rows, spec.row])
  const cs = useMemo(() => uniq(rows, spec.column).map(String), [rows, spec.column])
  const byCell = useMemo(() => new Map(rows.map((r) => [`${r[spec.row]}\u0000${r[spec.column]}`, num(r[spec.value])])), [rows, spec.row, spec.column, spec.value])
  const vals = [...byCell.values()].filter((v): v is number => v != null)
  const dom = spec.domain ?? (spec.ramp === 'sequential' ? (d3.extent(vals) as [number, number]) : [-1, 1] as [number, number])
  // the palette is picked from the colour ramp below (kept in the URL as <id>.palette), as on every correlation chart
  const colours = useColourScheme(spec.id ?? 'figure-heatmap', spec.ramp === 'sequential' ? 'sequential' : 'diverging')
  const ramp = colours.stops
  const scale = useMemo(() => d3.scaleLinear<string>().domain(ramp.map((_, i) => dom[0] + (i / (ramp.length - 1)) * (dom[1] - dom[0]))).range(ramp).clamp(true),
    [ramp.join(), dom[0], dom[1]]) // eslint-disable-line react-hooks/exhaustive-deps
  const f = FORMAT[spec.format ?? 'f2']
  // `cell` enlarges the grid; the labels grow with the cell height
  const cell = { w: spec.cell?.[0] ?? 64, h: spec.cell?.[1] ?? 26 }, k = Math.max(1, cell.h / 26)
  const at = useMemo(() => (i: number, j: number) => byCell.get(`${rs[i]}\u0000${cs[j]}`) ?? null, [byCell, rs, cs])
  return <HeatMatrix rows={rs} columns={cs} cell={at} color={scale} format={(v) => f(v)} cellWidth={cell.w} cellHeight={cell.h} gap={2} center
    rowLabelSize={11 * k} colLabelSize={11 * k} colLabelWeight={600} colLabels="slant" colLabelAngle={-35} labelSize={10 * k} maxLabelWidth={200 * k}
    legend={{ scale: scale as unknown as { domain(): number[]; (v: number): string }, scheme: colours.scheme, label: spec.valueLabel ?? spec.value, format: (v) => f(v) }} />
}

// ------------------------------------------------------------------ bars
function BarsFigure({ spec, rows }: { spec: Extract<FigureSpec, { kind: 'bars' }>; rows: Row[] }) {
  const color = useColors(spec)
  const f = FORMAT[spec.format ?? 'f2']
  const bars = rows.map((r, i): BarRow => ({ key: `${r[spec.label]}|${i}`, label: String(r[spec.label] ?? '—'), value: num(r[spec.value]),
    color: spec.color ? color(r[spec.color]) : undefined, opacity: spec.color ? 0.85 : undefined,
    tip: [String(r[spec.label] ?? '—'), `${spec.xLabel ?? spec.value}: ${f(num(r[spec.value]))}`, ...(spec.color ? [`${spec.color}: ${r[spec.color] ?? '—'}`] : [])] }))
  const groups = spec.color ? uniq(rows, spec.color) : []
  return <>
    <BarRows rows={bars} sort={spec.sort ?? 'none'} format={(v) => f(v)} reference={spec.reference} axis={{ label: spec.xLabel ?? spec.value }} domain={spec.domain} />
    {groups.length > 1 && <Key items={groups.map((g) => ({ label: String(g), color: color(g), kind: 'square' as const }))} />}
  </>
}

// ------------------------------------------------------------------ table
function TableFigure({ spec, rows }: { spec: Extract<FigureSpec, { kind: 'table' }>; rows: Row[] }) {
  const cols: { key: string; label?: string; format?: FormatName }[] = spec.columns ?? Object.keys(rows[0] ?? {}).map((key) => ({ key }))
  const cell = (c: { format?: FormatName }, v: Value): ReactNode => (c.format ? FORMAT[c.format](v) : typeof v === 'number' ? fmt(v, 2) : v == null ? '—' : String(v))
  return <div style={{ overflowX: 'auto' }}>
    <table className="census-table">
      <thead><tr>{cols.map((c) => <th key={c.key} style={{ textTransform: 'none', letterSpacing: 0 }}>{c.label ?? c.key}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c.key} style={typeof r[c.key] === 'number' ? { fontFamily: FONT.mono, textAlign: 'right' } : undefined}>{cell(c, r[c.key])}</td>)}</tr>)}</tbody>
    </table>
  </div>
}
