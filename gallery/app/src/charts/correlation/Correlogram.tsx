import { fitUncertainty } from '@/lib/fitUncertainty'
import { robustDomain } from '@/lib/axes'
import { RegressionLines, RegressionGroupLines } from '@/components/RegressionOverlay'
import { useRegressions } from '@/lib/regressionPrefs'
import { fitGroupingOption, useChartOptions } from '@/lib/chartOptions'
import { groupedRegressions, type FitGrouping } from '@/lib/siteRegression'
import { memo, useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { FieldMultiSelect, type FieldPreset } from '@/components/FieldMultiSelect'
import { isPmfField } from '@/lib/pmfGroups'
import { useSearchParam } from '@/lib/url'
import { ColorLegend } from '@/components/ColorLegend'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { FocusLayer, PinRing, dotRim, focusStyle, useHighlightActions } from '@/lib/highlight'
import { HitLayer } from '@/components/HitLayer'
import { useFilterEdits } from '@/lib/filterEdits'
import { RESTATES, isBookkeeping, sameProduct } from '@/fieldLineage'
import { corrMatrix, fmt, regression } from '@/lib/stats'
import { INK, FONT, inkOn } from '@/lib/theme'
import { useColourScheme } from '@/lib/colourSchemes'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'
import { wrapCategory } from '@/components/CategoryAxis'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { MENU_ACTION } from '@/lib/filterEdits'
import { XIcon } from '@/components/Icons'
import { createPortal } from 'react-dom'
import { chemText } from '@/lib/format'
import { ExcludedNote } from '@/lib/markedRows'

const ENCODINGS = ['Colour', 'Circle size', 'Scatter matrix'] as const
const MIN_N = 20
// the scatter matrix draws up to twelve fields (the 'Top 12' preset in full); past that the button says how many are drawn
const SPLOM_MAX = 12

/** the named field sets (FieldMultiSelect presets): only the fields this view has count */
const listed = (names: string[]) => (av: string[]) => names.filter((f) => av.includes(f))
const isWeather = (f: string) => /\((ERA5|station)\)$/.test(f)
const isAeronet = (f: string) => /\(AERONET\)$/.test(f)
const BASE_PRESETS: FieldPreset[] = [
  { key: 'carbon', label: 'Carbon & optical', fields: listed(['EC (FTIR)', 'OC (FTIR)', 'OM (FTIR)', 'EC (TOR)', 'OC (TOR)', 'HIPS BC', 'HIPS Fabs', 'BC (public HIPS)', 'Equivalent BC (SSR)', 'PM2.5 mass']) },
  { key: 'ions', label: 'Ions & mass', fields: listed(['PM2.5 mass', 'Sulfate', 'Nitrate', 'Ammonium', 'Potassium', 'Sodium', 'Calcium', 'Magnesium', 'Nitrite', 'Phosphate', 'Bromide']) },
  { key: 'dust', label: 'Dust & metals', fields: listed(['Silicon', 'Aluminum', 'Iron', 'Titanium', 'Calcium (element)', 'Potassium (element)', 'Manganese', 'Soil', 'Zinc', 'Copper', 'Lead']) },
  { key: 'combustion', label: 'Combustion tracers', note: 'EC, OC and BC with potassium, zinc, lead and the combustion PMF factors',
    fields: listed(['EC (FTIR)', 'OC (FTIR)', 'HIPS BC', 'EC (TOR)', 'Potassium', 'Zinc', 'Lead', 'Wood Burning share (PMF)', 'Charcoal share (PMF)', 'Fossil Fuel Combustion share (PMF)']) },
  { key: 'groups', label: 'FTIR functional groups', fields: listed(['Alkane CH', 'Alcohol COH', 'Carboxylic COOH', 'Non-acid CO', 'OM (FTIR)', 'OC (FTIR)', 'EC (FTIR)']) },
  { key: 'pmf', label: 'PMF sources', note: 'the PMF factor shares with EC and BC', fields: (av) => [...listed(['EC (FTIR)', 'HIPS BC'])(av), ...av.filter(isPmfField)] },
  { key: 'weather', label: 'Weather', note: 'EC, BC and mass with the weather over each filter', fields: (av) => [...listed(['EC (FTIR)', 'HIPS BC', 'PM2.5 mass'])(av), ...av.filter(isWeather)].slice(0, 14) },
  { key: 'aeronet', label: 'AERONET column', note: 'BC, EC and mass with the sun photometer over each filter', fields: (av) => [...listed(['HIPS Fabs', 'HIPS BC', 'BC (public HIPS)', 'EC (FTIR)', 'PM2.5 mass'])(av), ...av.filter(isAeronet)].slice(0, 16) },
  { key: 'all', label: 'Everything', note: 'every field with enough values', fields: (av) => av },
]

/**
 * Correlogram — the chemistry correlation matrix, 14 figures in the estate.
 * AGENTS.md asks for R² when ranking relationship strength and signed r only
 * where direction matters; here direction is the whole point, so cells carry
 * signed r and the tooltip adds R².
 *
 * Restatements (the HIPS unit conversion and public FTIR carbon duplicates)
 * are dropped because their near-perfect correlations are not new findings.
 *
 * `Scatter matrix` is react-graph-gallery's own correlogram: the pairwise
 * scatterplots of a handful of species with their histograms on the diagonal.
 * The r-matrix says *how much* two species co-vary; the scatter matrix shows
 * *how* — a single Beijing dust day can carry a correlation on its own, and
 * only the cloud shows it.
 */
export function Correlogram({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  // pins, the brush and the handlers only: a hover anywhere in the gallery does not re-render the chart
  const hl = useHighlightActions()
  const lg = useLegend()
  const colours = useColourScheme('correlogram', 'diverging')

  const siteCodes = useMemo(
    () => ['All', ...meta.sites.filter((s) => rows.some((r) => r.code === s.code)).map((s) => s.code)],
    [meta.sites, rows]
  )
  const { methods, grouping: chosenGrouping, setGrouping } = useRegressions()
  // the options and when each applies (lib/chartOptions): species and fit grouping belong to the
  // scatter matrix; with one site in the matrix (a site picked, or one in view) a fit per site is the pooled fit
  const o = useChartOptions('correlogram', [
    { key: 'site', label: 'Site', style: 'select', default: 'ETAD', choices: siteCodes.map((c) => ({ value: c, label: c === 'All' ? 'All sites' : meta.sites.find((s) => s.code === c)?.name ?? c })) },
    { key: 'encoding', label: 'Encoding', choices: ENCODINGS, default: 'Colour' },
    // which fields: a preset or the reader's own (components/FieldMultiSelect); the scatter matrix draws the first SPLOM_MAX
    { key: 'fields', node: () => <FieldMultiSelect meta={meta} available={avail} value={chosen} preset={presetKey === 'custom' ? null : presetKey} presets={presets}
      max={encoding === 'Scatter matrix' ? SPLOM_MAX : undefined} onChange={(fs, p) => { setCustom(fs.join('|')); setPreset(p ?? 'custom') }} /> },
    fitGroupingOption({ bind: [chosenGrouping, setGrouping as (v: never) => void],
      when: (c) => c.encoding === 'Scatter matrix' && (c.estimators as number) > 0,
      offerSite: (c) => c.site === 'All' && (c.nSites as number) > 1 }),
  ], { estimators: methods.length, nSites: siteCodes.length - 1 })
  const effectiveSite = o.values.site as string
  const encoding = o.values.encoding as (typeof ENCODINGS)[number]
  const matrixGrouping = o.values.grouping as FitGrouping
  const sub = useMemo(() => (effectiveSite === 'All' ? rows : rows.filter((r) => r.code === effectiveSite)), [rows, effectiveSite])

  // the fields that can be shown here: measured (no restatements, uncertainties, MDLs) with ≥ MIN_N filters
  const { avail, nOf } = useMemo(() => {
    const nOf: Record<string, number> = {}
    for (const f of meta.fields) if (!(f in RESTATES) && !isBookkeeping(f)) nOf[f] = sub.filter((r) => typeof r[f] === 'number').length
    return { avail: Object.keys(nOf).filter((f) => nOf[f] >= MIN_N), nOf }
  }, [sub, meta.fields])
  const presets = useMemo<FieldPreset[]>(() => [
    { key: 'top', label: 'Top 12 by coverage', note: 'the twelve fields with the most filters here', fields: (av) => [...av].sort((a, b) => (nOf[b] ?? 0) - (nOf[a] ?? 0)).slice(0, 12) },
    ...BASE_PRESETS,
  ], [nOf])
  const [presetKey, setPreset] = useSearchParam<string>('corrp', 'top', [...presets.map((p) => p.key), 'custom'])
  const [customParam, setCustom] = useSearchParam<string>('corrf', '')
  const chosen = useMemo(() => {
    if (presetKey !== 'custom') return (presets.find((p) => p.key === presetKey) ?? presets[0]).fields(avail)
    return customParam.split('|').filter((f) => avail.includes(f))
  }, [presetKey, customParam, presets, avail])

  // Computed from the live subset rather than a precomputed file, so the
  // season/site/date filters above actually move these numbers.
  const block = useMemo(() => {
    const fields = chosen
    const matrix = corrMatrix(sub, fields, sameProduct)
    const n: Record<string, number> = {}
    for (const f of fields) n[f] = sub.filter((r) => typeof r[f] === 'number').length
    const name = effectiveSite === 'All' ? 'All sites' : meta.sites.find((s) => s.code === effectiveSite)?.name ?? effectiveSite
    return { code: effectiveSite, name, fields, matrix, n }
  }, [sub, effectiveSite, chosen, meta.sites])

  // the scatter matrix draws the first SPLOM_MAX of the chosen fields. The same array until the choice
  // changes: the matrix's scales key on it, and new scales redraw every panel's dots (a tooltip re-renders this chart)
  const splomFields = useMemo(() => chosen.slice(0, SPLOM_MAX), [chosen])
  // stable inputs for the matrix's memoised panels: the rows only change with the data or the legend
  const hiddenKey = meta.sites.filter((x) => !lg.show(x.name)).map((x) => x.name).join('|')
  const splomRows = useMemo(() => sub.filter((r) => lg.show(r.site)), [sub, hiddenKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const dimKey = `${hiddenKey}#${lg.hover ?? ''}`
  // a matrix cell opened as a crossplot (right-click a cell: lib/filterEdits MENU_ACTION, detail "x|y")
  const [pair, setPair] = useState<[string, string] | null>(null)
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const open = (e: Event) => { const [x, y] = String((e as CustomEvent).detail).split('|'); if (x && y) setPair([x, y]) }
    el.addEventListener(MENU_ACTION, open)
    return () => el.removeEventListener(MENU_ACTION, open)
  }, [])

  const labelPad = 128
  const fit = useFitHeight()
  // a readable size on the page; fitted (a tile, the single-chart view) as large as the room allows
  const size = fit === null ? Math.min(Math.max(320, width - labelPad - 40), 660)
    : Math.max(120, Math.min(width - labelPad - 40, fit - labelPad - 20))
  const cell = size / Math.max(1, block.fields.length)

  const color = useMemo(
    () => d3.scaleLinear<string>().domain([-1, -0.66, -0.33, 0, 0.33, 0.66, 1]).range(colours.stops).clamp(true),
    [colours.stops.join()] // eslint-disable-line react-hooks/exhaustive-deps
  )

  return (
    <ChartFrame
      id="correlogram"
      title="Correlogram — how the chemistry co-varies"
      subtitle="Pearson r between species at one site; the colour scale below runs from negative to positive r (click the scale for other colours). The in-house and ChemSpec EC feeds both report FTIR EC, so the duplicate ChemSpec carbon fields are omitted. 'Scatter matrix' shows the clouds behind a few cells, with each species' histogram on the diagonal."
      provenance="stands in for 14 correlogram + heatmap figures · react-graph-gallery.com/correlogram"
      controls={o.controls()}
    >
      <div ref={wrapRef} className="chart-wrap centered" style={{ overflowX: 'auto' }}>
        {block.fields.length < 2 ? (
          <p className="frame-empty">{avail.length < 2 ? `Fewer than two fields with ≥${MIN_N} filters in this subset.` : 'Choose at least two fields (Fields ▾).'}</p>
        ) : encoding === 'Scatter matrix' ? (
          splomFields.length < 2 ? (
            <Empty>Choose at least two fields (Fields ▾).</Empty>
          ) : (
            <ScatterMatrix grouping={matrixGrouping} meta={meta} rows={splomRows} fields={splomFields} width={fit === null ? Math.min(width, Math.max(760, 160 + 100 * splomFields.length)) : width} tip={tip} hl={hl} units={meta.field_units} dim={lg.dim} dimKey={dimKey} />
          )
        ) : (
          <svg width={size + labelPad + 30} height={size + labelPad + 20} className="animated">
            <g transform={`translate(${labelPad},${labelPad})`}>
              {block.fields.map((fa, i) =>
                block.fields.map((fb, j) => {
                  const r = block.matrix[i][j]
                  if (r === null) return null
                  const cx = j * cell
                  const cy = i * cell
                  const rad = (cell / 2 - 1.5) * Math.sqrt(Math.abs(r))
                  return (
                    <g
                      key={`${i}-${j}`}
                      onMouseEnter={(e) =>
                        tip.show(e, [fa, `vs ${fb}`, `r = ${fmt(r, 3)} · R² = ${fmt(r * r, 3)}`, `n = ${Math.min(block.n[fa] ?? 0, block.n[fb] ?? 0)}`])
                      }
                      onMouseLeave={tip.hide}
                    >
                      <rect x={cx} y={cy} width={cell} height={cell} fill={encoding === 'Colour' ? color(r) : INK.panel} stroke={INK.panel} strokeWidth={1} />
                      {encoding === 'Circle size' && <circle cx={cx + cell / 2} cy={cy + cell / 2} r={Math.max(1, rad)} fill={color(r)} />}
                      {cell > 26 && encoding === 'Colour' && (
                        <text
                          x={cx + cell / 2} y={cy + cell / 2} dy="0.34em" textAnchor="middle"
                          fontSize={Math.min(10, cell / 3.4)} fill={inkOn(color(r))}
                          fontFamily={FONT.mono} pointerEvents="none"
                        >
                          {r.toFixed(2)}
                        </text>
                      )}
                    </g>
                  )
                })
              )}
              {block.fields.map((f, i) => (
                <text key={`r${i}`} x={-8} y={i * cell + cell / 2} dy="0.32em" textAnchor="end" fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>
                  {f}
                </text>
              ))}
              {block.fields.map((f, j) => (
                <text key={`c${j}`} transform={`translate(${j * cell + cell / 2},-8) rotate(-52)`} fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>
                  {f}
                </text>
              ))}
            </g>
          </svg>
        )}
        {encoding === 'Scatter matrix' && pair && <PairFocus pair={pair} rows={splomRows} meta={meta} site={block.name} onClose={() => setPair(null)} />}
        <div className="legend">
          {encoding === 'Scatter matrix' ? (
            <>
              <Legend items={meta.sites.filter((s) => sub.some((r) => r.site === s.name)).map((s) => ({ label: s.name, color: s.color }))} {...lg.props} note="diagonal: histogram of the species · off-diagonal: the pair, selected regression fits, r" />
              {/* a fit per season: the lines' colours, one entry per season drawn */}
              {matrixGrouping === 'season' && methods.length > 0 && <Legend items={meta.seasons.filter((se) => splomRows.some((r) => r.season === se.name))
                .map((se) => ({ label: se.name, color: se.color, shape: 'line' as const, parent: 'Fit per season' }))} />}
            </>
          ) : (
            <ColorLegend scale={color} scheme={colours.scheme} label="Pearson r" width={180} ticks={5} format={(v) => (v > 0 ? `+${v}` : String(v))} note="Repeated fields omitted: HIPS BC is Fabs ÷ MAC; ChemSpec EC/OC repeat the FTIR products" />
          )}
        </div>
        <ExcludedNote />
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/**
 * A field name in at most `max` lines of `room` px (wrapCategory's 6 px a character, at the matrix's 10.5 px
 * text): word breaks first, then an ellipsis for what still does not fit, so no name runs into its neighbour.
 */
export function wrapName(name: string, room: number, max = 3): string[] {
  const chars = Math.max(2, Math.floor(room / 6))
  const lines = wrapCategory(name, room)
  const out = lines.slice(0, max)
  if (lines.length > max) out[max - 1] = `${out[max - 1]}…`
  return out.map((l) => (l.length > chars ? `${l.slice(0, chars - 1)}…` : l))
}
const LINE = 13

/**
 * One cell of the scatter matrix, large: the shared Crossplot with the gallery's selected fits, their statistics
 * and the gallery's fit grouping (its control here is the matrix's: the same setting), each axis on its own range.
 * Hover links the filter as everywhere; a click opens its record in the sample drawer, over this dialog.
 */
function PairFocus({ pair: [fx, fy], rows, meta, site, onClose }: {
  pair: [string, string]; rows: FilterRow[]; meta: MetaFile; site: string; onClose: () => void
}) {
  const hl = useHighlightActions()
  const points = useMemo<CrossPoint[]>(() => rows
    .filter((r) => typeof r[fx] === 'number' && typeof r[fy] === 'number' && Number.isFinite(r[fx] as number) && Number.isFinite(r[fy] as number))
    .map((r) => ({ x: r[fx] as number, y: r[fy] as number, color: r.color as string, id: r.id,
      site: r.site, season: r.season ? String(r.season) : undefined, seasonColor: meta.seasons.find((se) => se.name === r.season)?.color,
      open: () => hl.openSample(r.id),
      tip: [r.id, `${r.site} · ${r.date}`, `${chemText(fx)}: ${fmt(r[fx] as number)}`, `${chemText(fy)}: ${fmt(r[fy] as number)}`] })), [rows, fx, fy, meta.seasons]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    // a record opened from a point is closed first, by its own Esc
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape' && !e.defaultPrevented && !document.querySelector('.drawer')) onClose() }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])
  const unit = (f: string) => meta.field_units?.[f] ? ` (${meta.field_units[f].replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹')})` : ''
  return createPortal(
    <div className="export-backdrop pair-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="export-dialog pair-focus" role="dialog" aria-label={`${chemText(fy)} vs ${chemText(fx)}`}>
        <header>
          <div><strong>{chemText(fy)} vs {chemText(fx)}</strong><span className="export-title">{site} · {points.length} filters</span></div>
          <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close" title="Close (Esc)"><XIcon /></button>
        </header>
        {/* the title is the dialog's; the fit grouping control is Crossplot's own, on the gallery-wide setting */}
        <Crossplot points={points} title="" xLabel={`${fx}${unit(fx)}`} yLabel={`${fy}${unit(fy)}`}
          maxSide={Math.max(320, Math.min(640, window.innerHeight - 300))} sameScale={false} />
      </div>
    </div>,
    document.body,
  )
}

/** Pairwise scatter matrix for a few species, histograms on the diagonal. */
function ScatterMatrix({ rows, fields, width, tip, hl, units, dim, dimKey, meta, grouping }: {
  meta: MetaFile
  rows: FilterRow[]
  fields: string[]
  width: number
  tip: ReturnType<typeof useTooltip>
  hl: ReturnType<typeof useHighlightActions>
  units: Record<string, string>
  dim: (site: string) => number
  /** changes when the legend's dimming does (dim itself is a new function every render) */
  dimKey: string
  /** the grouping drawn (one site in view: per site is the pooled fit) */
  grouping: FitGrouping
}) {
  const n = fields.length
  const pad = 6
  // the row labels sit in the left gutter, as wide as the longest name up to ROW_ROOM; longer names wrap
  const ROW_ROOM = 132
  const rowRoom = Math.min(ROW_ROOM, Math.max(48, ...fields.map((f) => f.length * 6)))
  const labelW = rowRoom + 12
  // as wide as the panel allows, and no taller than a fitted height (a Dashboard tile, the single-chart view);
  // the column labels under the matrix take up to three lines
  const fit = useFitHeight()
  const byWidth = Math.floor((width - labelW - 20 - pad * (n - 1)) / n)
  const cellS = Math.max(70, fit === null ? byWidth : Math.min(byWidth, Math.floor((fit - (10 + 3 * LINE) - pad * (n - 1)) / n)))
  const colLines = fields.map((f) => wrapName(chemText(f), cellS - 4))
  const rowLines = fields.map((f) => wrapName(chemText(f), rowRoom))
  const bottomH = 10 + Math.max(1, ...colLines.map((l) => l.length)) * LINE
  const scales = useMemo(
    () =>
      fields.map((f) => {
        const vals = rows.map((r) => r[f]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v)).sort(d3.ascending)
        // 1–99 % so a single spike does not push the cloud into a corner
        const [lo, hi] = robustDomain(vals, 0.01, 0.99, { include: [0] }).domain
        return d3.scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).range([0, cellS]).nice()
      }),
    [rows, fields, cellS]
  )
  const W = labelW + n * cellS + (n - 1) * pad + 10
  const H = n * cellS + (n - 1) * pad + bottomH
  const unitOf = (f: string) => (units[f] ? units[f].replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹') : '')
  // the dots are drawn in one <svg>, and everything a hover changes (the dimming, each panel's focused dot, the
  // pointer layers) in a second <svg> over it, on its own compositing layer (will-change), with the fits and r.
  // In one <svg> a hover re-rasterised every dot in the matrix (138k at all sites, ~60–90 ms a move at 12 fields).
  // The export composes both <svg>s; the hover's marks are data-export="skip".
  const [overlay, setOverlay] = useState<SVGGElement | null>(null)

  return (
    <div style={{ position: 'relative', width: W, height: H, marginLeft: 'auto', marginRight: 'auto' }}>
    <svg width={W} height={H} className="animated">
      <g transform={`translate(${labelW},0)`}>
        {fields.map((fy, i) =>
          fields.map((fx, j) => {
            const ox = j * (cellS + pad)
            const oy = i * (cellS + pad)
            const sx = scales[j]
            const sy = scales[i]
            if (i === j) {
              const vals = rows.map((r) => r[fx]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
              const bins = d3.bin().domain(sx.domain() as [number, number]).thresholds(18)(vals)
              const ymax = d3.max(bins, (b) => b.length) ?? 1
              // the title (up to two lines) and its unit line sit over the bars
              const title = wrapName(chemText(fx), cellS - 8, 2)
              const top = title.length * 11 + 16
              return (
                <g key={`${i}-${j}`} transform={`translate(${ox},${oy})`}>
                  <rect width={cellS} height={cellS} fill={INK.zebra} stroke={INK.grid} />
                  {bins.map((b, k) => (
                    <rect key={k} x={sx(b.x0 ?? 0)} y={cellS - (b.length / ymax) * (cellS - top)} width={Math.max(0, sx(b.x1 ?? 0) - sx(b.x0 ?? 0) - 0.5)} height={(b.length / ymax) * (cellS - top)} fill={INK.muted} fillOpacity={0.5} />
                  ))}
                  <text textAnchor="middle" fontSize={10.5} fontWeight={600} fill={INK.text} fontFamily={FONT.family}>
                    {title.map((l, k) => <tspan key={k} x={cellS / 2} y={11 + k * 11}>{l}</tspan>)}<title>{fx}</title>
                  </text>
                  <text x={cellS / 2} y={title.length * 11 + 11} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.family}>{unitOf(fx)} · n={vals.length}</text>
                </g>
              )
            }
            return <g key={`${i}-${j}`} transform={`translate(${ox},${oy})`}>
              <SplomCell rows={rows} fx={fx} fy={fy} sx={sx} sy={sy} cellS={cellS} ox={ox} oy={oy} overlay={overlay} grouping={grouping} meta={meta}
                selected={hl.selected} dimKey={dimKey} dim={dim} tip={tip} setHover={hl.setHover} open={hl.openSample} />
            </g>
          })
        )}
        {/* every name wrapped to its room (wrapName): the rows to the gutter, the columns to their cell */}
        {fields.map((f, i) => (
          <text key={`l${i}`} textAnchor="end" fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>
            {rowLines[i].map((l, k) => <tspan key={k} x={-8} y={i * (cellS + pad) + cellS / 2 + (k - (rowLines[i].length - 1) / 2) * LINE} dy="0.32em">{l}</tspan>)}<title>{f}</title>
          </text>
        ))}
        {fields.map((f, j) => (
          <text key={`b${j}`} textAnchor="middle" fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>
            {colLines[j].map((l, k) => <tspan key={k} x={j * (cellS + pad) + cellS / 2} y={n * (cellS + pad) - pad + 16 + k * LINE}>{l}</tspan>)}<title>{f}</title>
          </text>
        ))}
      </g>
    </svg>
    <svg width={W} height={H} style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none', willChange: 'transform' }}>
      <g ref={setOverlay} transform={`translate(${labelW},0)`} pointerEvents="visiblePainted" />
    </svg>
    </div>
  )
}

/**
 * One off-diagonal panel of the scatter matrix. Thirty of these hold a thousand dots each, so the dots are drawn
 * once and do not follow the hover: a hover anywhere (this panel, another, another chart) dims them under a veil
 * and draws the focused filter again on top (lib/highlight FocusLayer), and one hit layer finds the dot under the
 * pointer (components/HitLayer). Those, a veil that does the dimming and the fits are portalled into the matrix's
 * overlay <svg>, so a hover never touches the <svg> that holds the dots. Hovering a dot lights the same filter up in every panel and chart; clicking pins it
 * and opens its record. The fits are cached per panel, and the dots redraw only with the data, the brush or the legend.
 */
const SplomCell = memo(function SplomCell({ rows, fx, fy, sx, sy, cellS, ox, oy, overlay, grouping, meta, selected, dimKey, dim, tip, setHover, open }: {
  rows: FilterRow[]; fx: string; fy: string
  sx: d3.ScaleLinear<number, number>; sy: d3.ScaleLinear<number, number>; cellS: number
  /** the panel's place in the matrix, and the overlay <svg>'s group its hover marks and pointer layer go into */
  ox: number; oy: number; overlay: SVGGElement | null
  grouping: FitGrouping; meta: MetaFile
  selected: Set<string> | null; dimKey: string; dim: (site: string) => number
  tip: ReturnType<typeof useTooltip>; setHover: (id: string | null) => void; open: (id: string) => void
}) {
  const edits = useFilterEdits()
  const pts = useMemo(() => rows
    .map((r) => ({ row: r, x: r[fx], y: r[fy] }))
    .filter((p): p is { row: FilterRow; x: number; y: number } => typeof p.x === 'number' && typeof p.y === 'number' && Number.isFinite(p.x) && Number.isFinite(p.y)), [rows, fx, fy])
  // Deming (λ=1) rather than y-on-x least squares: both species are measured with error
  const { st, fitGroups } = useMemo(() => ({
    st: regression(pts.map((p) => p.x), pts.map((p) => p.y), fitUncertainty(pts.map(p => p.row), fx, fy, meta)),
    fitGroups: grouping === 'pooled' ? [] : groupedRegressions(pts, grouping, { x: p => p.x, y: p => p.y, site: p => p.row.site, season: p => p.row.season, seasonColor: p => meta.seasons.find(se => se.name === p.row.season)?.color ?? INK.muted,
      color: p => meta.sites.find(s => s.name === p.row.site)?.color ?? INK.muted,
      options: members => fitUncertainty(members.map(p => p.row), fx, fy, meta) }),
  }), [pts, grouping, fx, fy, meta])
  // each dot's place in the panel, clamped to it (the axes run 1–99 %, so a spike sits on the edge)
  const xy = useMemo(() => pts.map((p) => [Math.max(0, Math.min(cellS, sx(p.x))), Math.max(0, Math.min(cellS, cellS - sy(p.y)))] as const), [pts, sx, sy, cellS])
  const keys = useMemo(() => new Set(pts.map((p) => p.row.id)), [pts])
  // the rim most dots share is the group's (inherited), so only a dot that differs (a brushed one) carries its own:
  // the theme's paint rules (lib/themeCss) then match one group, not every circle, which halves the dots' style pass
  const rim = dotRim()
  const dots = useMemo(() => pts.map((p, i) => {
    const f = focusStyle(p.row.id, null, { r: 2, level: 'dense' }, selected)
    const k = dim(p.row.site)
    const own = f.stroke !== rim.stroke || f.strokeWidth !== rim.strokeWidth
    return <circle key={p.row.id} cx={xy[i][0]} cy={xy[i][1]} r={f.r} fill={p.row.color as string} fillOpacity={f.opacity * k}
      stroke={own ? f.stroke : undefined} strokeWidth={own ? f.strokeWidth : undefined} data-dot={k === 1 ? f.dot : undefined} data-fid={p.row.id} />
  }), [pts, xy, selected, dimKey, rim.stroke]) // eslint-disable-line react-hooks/exhaustive-deps
  // the overlay: a veil that dims the dots while a filter is focused, the fits and r over it, then the focused dot
  // and the pointer layer. The veil is the panel at 65 %, the same as the dots at 35 %, without touching the dots
  const veilRef = useRef<SVGGElement>(null)
  return <g className="splom-cell">
    <rect width={cellS} height={cellS} fill={INK.panel} stroke={INK.grid} />
    <g className="splom-dots" pointerEvents="none" stroke={rim.stroke} strokeWidth={rim.strokeWidth}>{dots}</g>
    {overlay && createPortal(<g transform={`translate(${ox},${oy})`} data-menu-action={`Open ${chemText(fy)} vs ${chemText(fx)} as a crossplot`} data-menu-detail={`${fx}|${fy}`}>
    <g ref={veilRef} className="splom-veil" data-export="skip" pointerEvents="none"><rect width={cellS} height={cellS} fill={INK.panel} fillOpacity={0.65} /></g>
    {grouping !== 'pooled' ? <RegressionGroupLines groups={fitGroups} x={sx} y={v => cellS - sy(v)} width={cellS} height={cellS} />
      : <RegressionLines stats={st} x={sx} y={v => cellS - sy(v)} domain={sx.domain() as number[]} width={cellS} height={cellS} />}
    {st && <text x={cellS - 4} y={11} textAnchor="end" fontSize={9.5} fill={INK.text} fontFamily={FONT.mono} pointerEvents="none">r={st.r.toFixed(2)}</text>}
    <g data-export="skip">
    <FocusLayer keys={keys} items={pts} idOf={(p) => p.row.id} groupRef={veilRef} focusClass="on" render={(p, pinned) => {
      const f = focusStyle(p.row.id, p.row.id, { r: 2, level: 'dense' })
      const cx = Math.max(0, Math.min(cellS, sx(p.x))), cy = Math.max(0, Math.min(cellS, cellS - sy(p.y)))
      return <>
        <circle cx={cx} cy={cy} r={f.r} fill={p.row.color as string} fillOpacity={f.opacity * dim(p.row.site)} stroke={f.stroke} strokeWidth={f.strokeWidth} pointerEvents="none" />
        {pinned && <PinRing cx={cx} cy={cy} r={f.r} />}
      </>
    }} />
    <HitLayer items={pts} xy={xy} width={cellS} height={cellS} hit={10}
      onHover={(e, p) => { setHover(p.row.id); tip.show(e, [p.row.id, `${p.row.site} · ${p.row.date}`, `${chemText(fx)}: ${fmt(p.x)}`, `${chemText(fy)}: ${fmt(p.y)}`, 'click to pin it on every chart']) }}
      onLeave={() => { setHover(null); tip.hide() }}
      onPick={(p) => open(p.row.id)}
      onMenu={(e, p) => edits.openMenu(e, { ids: [p.row.id], title: p.row.id })} />
    </g>
    </g>, overlay)}
  </g>
}, (a, b) => a.rows === b.rows && a.fx === b.fx && a.fy === b.fy && a.sx === b.sx && a.sy === b.sy && a.cellS === b.cellS
  && a.ox === b.ox && a.oy === b.oy && a.overlay === b.overlay
  && a.grouping === b.grouping && a.meta === b.meta && a.selected === b.selected && a.dimKey === b.dimKey)
