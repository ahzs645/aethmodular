import { fitUncertainty } from '@/lib/fitUncertainty'
import { robustDomain } from '@/lib/axes'
import { RegressionLines, RegressionGroupLines } from '@/components/RegressionOverlay'
import { useRegressions } from '@/lib/regressionPrefs'
import { fitGroupingOption, useChartOptions } from '@/lib/chartOptions'
import { groupedRegressions, type FitGrouping } from '@/lib/siteRegression'
import { memo, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { FieldMultiSelect, type FieldPreset } from '@/components/FieldMultiSelect'
import { isPmfField } from '@/lib/pmfGroups'
import { useSearchParam } from '@/lib/url'
import { ColorLegend } from '@/components/ColorLegend'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusLast, focusStyle, useHighlight } from '@/lib/highlight'
import { RESTATES, isBookkeeping, sameProduct } from '@/fieldLineage'
import { corrMatrix, fmt, regression } from '@/lib/stats'
import { INK, FONT, inkOn } from '@/lib/theme'
import { useColourScheme } from '@/lib/colourSchemes'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'

const ENCODINGS = ['Colour', 'Circle size', 'Scatter matrix'] as const
const MIN_N = 20
const SPLOM_MAX = 6

/** the named field sets (FieldMultiSelect presets): only the fields this view has count */
const listed = (names: string[]) => (av: string[]) => names.filter((f) => av.includes(f))
const isWeather = (f: string) => /\((ERA5|station)\)$/.test(f)
const BASE_PRESETS: FieldPreset[] = [
  { key: 'carbon', label: 'Carbon & optical', fields: listed(['EC (FTIR)', 'OC (FTIR)', 'OM (FTIR)', 'EC (TOR)', 'OC (TOR)', 'HIPS BC', 'HIPS Fabs', 'BC (public HIPS)', 'Equivalent BC (SSR)', 'PM2.5 mass']) },
  { key: 'ions', label: 'Ions & mass', fields: listed(['PM2.5 mass', 'Sulfate', 'Nitrate', 'Ammonium', 'Potassium', 'Sodium', 'Calcium', 'Magnesium', 'Nitrite', 'Phosphate', 'Bromide']) },
  { key: 'dust', label: 'Dust & metals', fields: listed(['Silicon', 'Aluminum', 'Iron', 'Titanium', 'Calcium (element)', 'Potassium (element)', 'Manganese', 'Soil', 'Zinc', 'Copper', 'Lead']) },
  { key: 'combustion', label: 'Combustion tracers', note: 'EC, OC and BC with potassium, zinc, lead and the combustion PMF factors',
    fields: listed(['EC (FTIR)', 'OC (FTIR)', 'HIPS BC', 'EC (TOR)', 'Potassium', 'Zinc', 'Lead', 'Wood Burning share (PMF)', 'Charcoal share (PMF)', 'Fossil Fuel Combustion share (PMF)']) },
  { key: 'groups', label: 'FTIR functional groups', fields: listed(['Alkane CH', 'Alcohol COH', 'Carboxylic COOH', 'Non-acid CO', 'OM (FTIR)', 'OC (FTIR)', 'EC (FTIR)']) },
  { key: 'pmf', label: 'PMF sources', note: 'the PMF factor shares with EC and BC', fields: (av) => [...listed(['EC (FTIR)', 'HIPS BC'])(av), ...av.filter(isPmfField)] },
  { key: 'weather', label: 'Weather', note: 'EC, BC and mass with the weather over each filter', fields: (av) => [...listed(['EC (FTIR)', 'HIPS BC', 'PM2.5 mass'])(av), ...av.filter(isWeather)].slice(0, 14) },
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
  const hl = useHighlight()
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
    // which fields: a preset or the reader's own (components/FieldMultiSelect); the scatter matrix draws the first six
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

  // the scatter matrix reads at a handful: the first six of the chosen fields
  const splomFields = chosen.slice(0, SPLOM_MAX)
  // stable inputs for the matrix's memoised panels: the rows only change with the data or the legend
  const hiddenKey = meta.sites.filter((x) => !lg.show(x.name)).map((x) => x.name).join('|')
  const splomRows = useMemo(() => sub.filter((r) => lg.show(r.site)), [sub, hiddenKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const dimKey = `${hiddenKey}#${lg.hover ?? ''}`

  const labelPad = 128
  const fit = useFitHeight()
  const size = fit === null ? Math.min(Math.max(320, width - labelPad - 40), 660)
    : Math.max(120, Math.min(width - labelPad - 40, 660, fit - labelPad - 20))
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
            <ScatterMatrix grouping={matrixGrouping} meta={meta} rows={splomRows} fields={splomFields} width={Math.min(width, 760)} tip={tip} hl={hl} units={meta.field_units} dim={lg.dim} dimKey={dimKey} />
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
        <div className="legend">
          {encoding === 'Scatter matrix' ? (
            <Legend items={meta.sites.filter((s) => sub.some((r) => r.site === s.name)).map((s) => ({ label: s.name, color: s.color }))} {...lg.props} note="diagonal: histogram of the species · off-diagonal: the pair, selected regression fits, r" />
          ) : (
            <ColorLegend scale={color} scheme={colours.scheme} label="Pearson r" width={180} ticks={5} format={(v) => (v > 0 ? `+${v}` : String(v))} note="Repeated fields omitted: HIPS BC is Fabs ÷ MAC; ChemSpec EC/OC repeat the FTIR products" />
          )}
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/** Pairwise scatter matrix for a few species, histograms on the diagonal. */
function ScatterMatrix({ rows, fields, width, tip, hl, units, dim, dimKey, meta, grouping }: {
  meta: MetaFile
  rows: FilterRow[]
  fields: string[]
  width: number
  tip: ReturnType<typeof useTooltip>
  hl: ReturnType<typeof useHighlight>
  units: Record<string, string>
  dim: (site: string) => number
  /** changes when the legend's dimming does (dim itself is a new function every render) */
  dimKey: string
  /** the grouping drawn (one site in view: per site is the pooled fit) */
  grouping: FitGrouping
}) {
  const n = fields.length
  const pad = 6
  // the row labels sit in the left gutter: size it from the longest name
  const labelW = Math.max(60, Math.ceil(Math.max(...fields.map((f) => f.length)) * 6.4) + 12)
  const cellS = Math.max(70, Math.floor((width - labelW - 20 - pad * (n - 1)) / n))
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
  const H = n * cellS + (n - 1) * pad + labelW
  const unitOf = (f: string) => (units[f] ? units[f].replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹') : '')

  return (
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
              return (
                <g key={`${i}-${j}`} transform={`translate(${ox},${oy})`}>
                  <rect width={cellS} height={cellS} fill={INK.zebra} stroke={INK.grid} />
                  {bins.map((b, k) => (
                    <rect key={k} x={sx(b.x0 ?? 0)} y={cellS - (b.length / ymax) * (cellS - 16)} width={Math.max(0, sx(b.x1 ?? 0) - sx(b.x0 ?? 0) - 0.5)} height={(b.length / ymax) * (cellS - 16)} fill={INK.muted} fillOpacity={0.5} />
                  ))}
                  <text x={cellS / 2} y={11} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={INK.text} fontFamily={FONT.family}>{fx}</text>
                  <text x={cellS / 2} y={22} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.family}>{unitOf(fx)} · n={vals.length}</text>
                </g>
              )
            }
            return <g key={`${i}-${j}`} transform={`translate(${ox},${oy})`}>
              <SplomCell rows={rows} fx={fx} fy={fy} sx={sx} sy={sy} cellS={cellS} grouping={grouping} meta={meta}
                pinnedId={hl.pinnedId} selected={hl.selected} dimKey={dimKey} dim={dim} tip={tip} open={hl.openSample} />
            </g>
          })
        )}
        {fields.map((f, i) => (
          <text key={`l${i}`} x={-8} y={i * (cellS + pad) + cellS / 2} dy="0.32em" textAnchor="end" fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>{f}</text>
        ))}
        {fields.map((f, j) => (
          <text key={`b${j}`} x={j * (cellS + pad) + cellS / 2} y={n * (cellS + pad) - pad + 16} textAnchor="middle" fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>{f}</text>
        ))}
      </g>
    </svg>
  )
}

/**
 * One off-diagonal panel of the scatter matrix. Hovering highlights the filter here only (a
 * gallery-wide hover would redraw every panel, and every chart, on each mouse move); clicking
 * pins it, which highlights it everywhere and opens its record. The fits are cached per panel,
 * and the panel redraws only when its data, the pinned filter, the brush or the legend change.
 */
const SplomCell = memo(function SplomCell({ rows, fx, fy, sx, sy, cellS, grouping, meta, pinnedId, selected, dim, tip, open }: {
  rows: FilterRow[]; fx: string; fy: string
  sx: d3.ScaleLinear<number, number>; sy: d3.ScaleLinear<number, number>; cellS: number
  grouping: FitGrouping; meta: MetaFile
  pinnedId: string | null; selected: Set<string> | null; dimKey: string; dim: (site: string) => number
  tip: ReturnType<typeof useTooltip>; open: (id: string) => void
}) {
  const [hover, setHover] = useState<string | null>(null)
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
  const focusId = hover ?? pinnedId
  return <>
    <rect width={cellS} height={cellS} fill={INK.panel} stroke={INK.grid} />
    {focusLast(pts, (p) => p.row.id, focusId).map(([p]) => {
      const f = focusStyle(p.row.id, focusId, { r: 2, opacity: 0.55 }, selected)
      const cx = Math.max(0, Math.min(cellS, sx(p.x)))
      const cy = Math.max(0, Math.min(cellS, cellS - sy(p.y)))
      return (
        <circle
          key={p.row.id} cx={cx} cy={cy} r={f.r} fill={p.row.color as string} fillOpacity={f.opacity * dim(p.row.site)} stroke={f.stroke} strokeWidth={f.strokeWidth}
          style={{ cursor: 'pointer' }}
          onMouseEnter={(e) => { setHover(p.row.id); tip.show(e, [p.row.id, `${p.row.site} · ${p.row.date}`, `${fx}: ${fmt(p.x)}`, `${fy}: ${fmt(p.y)}`, 'click to pin it on every chart']) }}
          onMouseLeave={() => { setHover(null); tip.hide() }}
          data-fid={p.row.id} onClick={() => open(p.row.id)}
        />
      )
    })}
    {grouping !== 'pooled' ? <RegressionGroupLines groups={fitGroups} x={sx} y={v => cellS - sy(v)} width={cellS} height={cellS} />
      : <RegressionLines stats={st} x={sx} y={v => cellS - sy(v)} domain={sx.domain() as number[]} width={cellS} height={cellS} />}
    {st && <text x={cellS - 4} y={11} textAnchor="end" fontSize={9.5} fill={INK.text} fontFamily={FONT.mono} pointerEvents="none">r={st.r.toFixed(2)}</text>}
  </>
}, (a, b) => a.rows === b.rows && a.fx === b.fx && a.fy === b.fy && a.sx === b.sx && a.sy === b.sy && a.cellS === b.cellS
  && a.grouping === b.grouping && a.meta === b.meta && a.pinnedId === b.pinnedId && a.selected === b.selected && a.dimKey === b.dimKey)
