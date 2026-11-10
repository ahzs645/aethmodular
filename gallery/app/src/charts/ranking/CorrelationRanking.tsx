import { confidence } from '@/lib/regressionCI'
import { useRegressions, fitsFor, equation } from '@/lib/regressionPrefs'
import { fitUncertainty } from '@/lib/fitUncertainty'
import { useMemo, useRef, useState, type ReactNode } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useChartOptions } from '@/lib/chartOptions'
import { useParamScope } from '@/lib/url'
import { Legend, useLegend } from '@/components/Legend'
import { AxisTitle, XAxis } from '@/components/Axes'
import { useAxisFields } from '@/components/AxisFieldPopover'
import { FieldMultiSelect, useFieldChoice, type FieldPreset } from '@/components/FieldMultiSelect'
import { FIELD_PRESETS } from '@/lib/fieldPresets'
import { RefLine } from '@/components/RefLines'
import { clipText, labelGutter, textWidth } from '@/lib/fitLabel'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { RESTATES, isBookkeeping, sameProduct } from '@/fieldLineage'
import { regression, spearman, fmt, type RegressionStats } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import { chemText } from '@/lib/format'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'
import { ExcludedNote } from '@/lib/markedRows'

const SAME = 'same product (unit conversion or public FTIR report)'
const kindOf = (d: { r: number; trivial: boolean }) => (d.trivial ? SAME : d.r >= 0 ? 'positive r' : 'negative r')
const colourOf = (d: { r: number; trivial: boolean }) => (d.trivial ? INK.identity : d.r >= 0 ? INK.positive : INK.negative)
/** what a lollipop's length is: R² (the default, AGENTS.md), or a signed r or Spearman ρ that keep the direction on the axis */
const MEASURES = [{ value: 'r2', label: 'R²' }, { value: 'r', label: 'r' }, { value: 'rho', label: 'Spearman ρ' }]
type Measure = 'r2' | 'r' | 'rho'
const MEASURE_TEXT: Record<Measure, string> = { r2: 'R²', r: 'r', rho: 'Spearman ρ' }
const POOLED = 'All sites (pooled)'
const VIEWS = ['One site', 'Side by side'] as const
/** side by side draws one column per site, readable up to this many */
const MAX_COLUMNS = 12
/** the label column fits the longest field name as drawn at 11.5 px, within reason */
const labelWidth = (fields: string[], cap: number) => labelGutter(fields, { size: 11.5, pad: 14, min: 60, max: cap })
const clip = (f: string, w: number, weight = 400) => clipText(f, w - 14, { size: 11.5, weight })

type Ranked = { field: string; r2: number; r: number; rho: number; n: number; trivial: boolean; stats: RegressionStats }

/** a field that can be ranked against the target: not the target, not bookkeeping, and not a
 * restatement of another candidate (HIPS BC = HIPS Fabs / MAC) unless it restates the target itself */
const isCandidate = (f: string, target: string) => f !== target && !isBookkeeping(f) && (!(f in RESTATES) || sameProduct(f, target))
const pairCount = (src: FilterRow[], f: string, target: string) => {
  let n = 0
  for (const r of src) { const a = r[f], b = r[target]; if (typeof a === 'number' && typeof b === 'number' && Number.isFinite(a) && Number.isFinite(b)) n++ }
  return n
}
/** the most common pairing first: the "Top 15 by coverage" preset */
const TOP_COVERAGE = 15

/** each candidate field against the target over these rows: R², r and the fit, for the pairs with ≥ minN filters */
function rankAgainst(src: FilterRow[], meta: MetaFile, target: string, minN: number, fields: readonly string[]): Ranked[] {
  const out: Ranked[] = []
  for (const f of fields) {
    const xs: number[] = []
    const ys: number[] = []
    const paired: FilterRow[] = []
    for (const r of src) {
      const a = r[f]
      const b = r[target]
      if (typeof a === 'number' && typeof b === 'number' && Number.isFinite(a) && Number.isFinite(b)) {
        xs.push(a)
        ys.push(b)
        paired.push(r)
      }
    }
    if (xs.length < minN) continue
    const s = regression(xs, ys, fitUncertainty(paired, f, target, meta))
    if (s) out.push({ field: f, r2: s.r2, r: s.r, rho: spearman(xs, ys) ?? NaN, n: s.n, stats: s, trivial: sameProduct(f, target) })
  }
  return out.sort((a, b) => b.r2 - a.r2)
}

/**
 * Lollipop — what actually predicts a chosen measurement, ranked.
 *
 * AGENTS.md: "Prefer R² over raw r when ranking or describing relationship
 * strength." So the bar length is R² and the sign of r is carried by colour
 * plus the tooltip, never by the length.
 *
 * This is the chart the estate is missing. 116 notebooks draw crossplots one
 * pair at a time; nothing ranks all the pairs at once. It keeps a per-site
 * select on purpose: pooling sites manufactures correlation out of the site
 * offsets alone (Simpson's), so "one site" is the honest default.
 */
/**
 * Lollipop — what actually predicts a chosen measurement, ranked.
 *
 * AGENTS.md: "Prefer R² over raw r when ranking or describing relationship
 * strength." So the bar length is R² and the sign of r is carried by colour
 * plus the tooltip, never by the length.
 *
 * This is the chart the estate is missing. 116 notebooks draw crossplots one
 * pair at a time; nothing ranks all the pairs at once. It ranks within one site
 * on purpose: pooling sites manufactures correlation out of the site offsets
 * alone (Simpson's), so "one site" is the honest default. Side by side keeps that
 * rule and draws every site in view as its own column, the fields in one shared
 * order (by R² averaged over the sites, or by one site's), so a row reads across
 * as "does this track the target everywhere, or only here?". A Compare tile opens
 * side by side.
 */
export function CorrelationRanking({ rows, meta, target }: { rows: FilterRow[]; meta: MetaFile; target: string }) {
  const { methods, showCI } = useRegressions()
  const axisPicker = !!useAxisFields()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const lg = useLegend()
  const [hoverField, setHoverField] = useState<string | null>(null)
  const inTile = useParamScope() !== ''

  const sites = useMemo(() => meta.sites.filter((s) => rows.some((r) => r.site === s.name)), [meta.sites, rows])
  const columnsFit = sites.length >= 2 && sites.length <= MAX_COLUMNS
  // the first choice offered is the default (the stored default, 'auto', is never a choice): a
  // Compare tile (a few sites) opens side by side, everything else on one site
  const sideFirst = inTile && columnsFit
  const o = useChartOptions('ranking', [
    { key: 'view', label: 'View', default: 'auto', choices: sideFirst ? [...VIEWS].reverse() : VIEWS,
      when: () => columnsFit, hidden: 'One site',
      hiddenNote: () => (sites.length > MAX_COLUMNS ? `side by side up to ${MAX_COLUMNS} sites` : null),
      title: 'One site: one ranking. Side by side: a column per site, the fields in one shared order' },
    { key: 'site', label: 'Site', style: 'select', default: 'Addis Ababa', when: (c) => c.view === 'One site',
      choices: [...sites.map((s) => s.name), POOLED], title: 'Pooled sites can rank a species highly on site offsets alone' },
    { key: 'order', label: 'Order by', style: 'select', default: 'mean', when: (c) => c.view === 'Side by side',
      choices: [{ value: 'mean', label: 'Average over sites' }, ...sites.map((s) => ({ value: s.name, label: s.name }))],
      title: 'Average: R² averaged over the sites (a site without the pair counts as 0). A site: that site\'s ranking' },
    { key: 'top', label: 'Show', default: '25', when: (c) => c.view === 'Side by side',
      choices: [{ value: '15', label: 'Top 15' }, { value: '25', label: 'Top 25' }, { value: 'all', label: 'All' }],
      title: 'The first rows in the order chosen; the rest are one click away' },
    { key: 'measure', label: 'Measure', default: 'r2', choices: MEASURES,
      title: 'R²: strength only (the sign is the colour). r and Spearman ρ: signed, so the axis runs −1 to 1; ρ ranks the values, so a few high-loading filters do not decide it. Ranked by strength either way' },
    { key: 'minN', label: 'Min filters', default: '30', choices: ['10', '30', '50', '100'] },
  ])
  const view = o.values.view as (typeof VIEWS)[number]
  const site = o.values.site as string
  const order = o.values.order as string
  const minN = Number(o.values.minN)
  const measure = (o.values.measure as Measure) ?? 'r2'
  const signed = measure !== 'r2'
  const val = (d: Ranked) => (measure === 'r2' ? d.r2 : measure === 'r' ? d.r : d.rho)
  /** ranked by strength: R², |r| or |ρ| (a pair without ρ last) */
  const strength = (d: Ranked) => { const v = val(d); return Number.isFinite(v) ? Math.abs(v) : -1 }
  const domain: [number, number] = signed ? [-1, 1] : [0, 1]
  const measureText = MEASURE_TEXT[measure]
  const top = o.values.top === 'all' ? Infinity : Number(o.values.top)

  // which fields are ranked: the candidates that pair with the target on ≥ minN filters in view,
  // narrowed by a preset or the reader's own list (components/FieldMultiSelect, kept in the URL as rankp / rankf)
  const { candidates, nOf } = useMemo(() => {
    const nOf: Record<string, number> = {}
    for (const f of meta.fields) if (isCandidate(f, target)) nOf[f] = pairCount(rows, f, target)
    return { candidates: Object.keys(nOf).filter((f) => nOf[f] >= minN), nOf }
  }, [rows, meta.fields, target, minN])
  const presets = useMemo<FieldPreset[]>(() => [
    { key: 'all', label: 'Everything', note: `every field that pairs with ${target}`, fields: (av) => av },
    { key: 'top', label: `Top ${TOP_COVERAGE} by coverage`, note: `the ${TOP_COVERAGE} fields with the most filters paired with ${target}`,
      fields: (av) => [...av].sort((a, b) => (nOf[b] ?? 0) - (nOf[a] ?? 0)).slice(0, TOP_COVERAGE) },
    ...FIELD_PRESETS.filter((p) => p.key !== 'all'),
  ], [nOf, target])
  const fieldChoice = useFieldChoice('rank', candidates, presets, 'all')
  const chosen = fieldChoice.chosen
  const narrowed = chosen.length < candidates.length

  const ranked = useMemo(
    () => (view === 'One site' ? rankAgainst(site === POOLED ? rows : rows.filter((r) => r.site === site), meta, target, minN, chosen).sort((a, b) => strength(b) - strength(a)) : []),
    [view, rows, target, site, minN, meta, measure, chosen] // eslint-disable-line react-hooks/exhaustive-deps
  )
  // side by side: one ranking per site, then the fields any site has, in one order
  const grid = useMemo(() => {
    if (view !== 'Side by side') return null
    const per = sites.map((s) => new Map(rankAgainst(rows.filter((r) => r.site === s.name), meta, target, minN, chosen).map((d) => [d.field, d])))
    const fields = [...new Set(per.flatMap((m) => [...m.keys()]))]
    const str = (d: Ranked | undefined) => (d ? strength(d) : 0)
    const score = (f: string) => {
      if (order === 'mean') return d3.sum(per, (m) => str(m.get(f))) / per.length
      const d = per[sites.findIndex((s) => s.name === order)]?.get(f)
      return d ? strength(d) : -1
    }
    const s = new Map(fields.map((f) => [f, score(f)]))
    fields.sort((a, b) => s.get(b)! - s.get(a)!)
    return { per, fields }
  }, [view, sites, rows, meta, target, minN, order, measure, chosen]) // eslint-disable-line react-hooks/exhaustive-deps

  const fit = useFitHeight()
  const narrow = width < 520
  const tipLines = (d: Ranked, where: string) => [
    `${d.field} · ${where}`,
    `R² = ${fmt(d.r2, 4)}`,
    `r = ${fmt(d.r, 3)}`,
    ...(Number.isFinite(d.rho) ? [`Spearman ρ = ${fmt(d.rho, 3)}`] : []),
    ...fitsFor(d.stats, methods).flatMap((f) => {
      const ci = showCI ? confidence(d.stats, f.method) : null
      return [equation(f, 4), ...(showCI ? [ci ? `95% slope [${fmt(ci.slope[0], 3)}, ${fmt(ci.slope[1], 3)}]; intercept [${fmt(ci.intercept[0], 3)}, ${fmt(ci.intercept[1], 3)}]` : '95% CI unavailable'] : [])]
    }),
    `n = ${d.n}`,
    ...(d.trivial ? ['derived from the target by arithmetic — not a finding'] : []),
  ]
  const rowHeight = (count: number, extraTop = 0) =>
    fit === null ? 22 : Math.max(12, Math.min(22, (fit - MARGIN.top - extraTop - MARGIN.bottom) / Math.max(1, count)))

  // `full`: a native <title> with the whole name when it is clipped, only where the row has no gallery tooltip
  const fieldLabel = (f: string, labelW: number, cy: number, muted: boolean, full = true) => {
    const cut = clip(f, labelW)
    return (
      <text x={-10} y={cy} dy="0.32em" textAnchor="end" fontSize={11.5} fill={muted ? INK.muted : INK.text} fontFamily={FONT.family}
        fontWeight={hoverField === f ? 600 : undefined}>
        {cut !== f && full ? <><title>{f}</title>{cut}</> : cut}
      </text>
    )
  }

  let plot: ReactNode
  let count: number
  if (view === 'One site') {
    const shown = ranked.filter((d) => lg.show(kindOf(d)))
    count = ranked.length
    // right gutter holds the "0.939 · n=190" annotation, which used to run off the canvas for the top-ranked rows
    const LABEL_GUTTER = 96
    const labelW = labelWidth(shown.map((d) => d.field), Math.max(110, width * 0.36))
    const innerW = Math.max(120, width - labelW - MARGIN.right - LABEL_GUTTER)
    const rowH = rowHeight(shown.length)
    const innerH = shown.length * rowH
    const x = d3.scaleLinear().domain(domain).range([0, innerW])
    plot = ranked.length === 0 ? (
      chosen.length === 0 ? <Empty>No fields chosen to rank. Pick a preset or tick some fields under Fields.</Empty>
        : <Empty>No pair reaches n ≥ {minN} at {site}{narrowed ? ' among the fields chosen' : ''}. Lower the minimum{narrowed ? ', choose more fields' : ''} or widen the subset.</Empty>
    ) : (
      <svg width={width} height={innerH + MARGIN.top + MARGIN.bottom} className="animated">
        <g transform={`translate(${labelW},${MARGIN.top})`}>
          <XAxis scale={x} y={innerH} label={`${measureText} against ${target}`} gridHeight={innerH} pick={axisPicker ? 'field' : undefined} />
          {signed && <RefLine orient="v" at={x(0)} from={0} to={innerH} />}
          {shown.map((d, i) => {
            const cy = i * rowH + rowH / 2
            const col = colourOf(d)
            return (
              <g key={d.field} opacity={lg.dim(kindOf(d), 0.2)} {...tip.bind(() => tipLines(d, site))}>
                {Number.isFinite(val(d)) && <>
                  <line x1={x(0)} x2={x(val(d))} y1={cy} y2={cy} stroke={col} strokeWidth={2} strokeOpacity={0.55} strokeDasharray={d.trivial ? '3 3' : undefined} />
                  <circle cx={x(val(d))} cy={cy} r={5} fill={d.trivial ? INK.panel : col} stroke={col} strokeWidth={d.trivial ? 1.5 : 0} />
                </>}
                {fieldLabel(d.field, labelW, cy, d.trivial, false)}
                <text x={x(Number.isFinite(val(d)) ? Math.max(0, val(d)) : 0) + 10} y={cy} dy="0.32em" fontSize={10.5} fill={INK.muted} fontFamily={FONT.mono}>
                  {Number.isFinite(val(d)) ? fmt(val(d), 3) : '–'} · n={d.n}
                </text>
              </g>
            )
          })}
        </g>
      </svg>
    )
  } else {
    const { per, fields } = grid!
    // a field is drawn while any site's dot for it is in a shown kind (the legend hides kinds)
    const shown = fields.filter((f) => per.some((m) => { const d = m.get(f); return d && lg.show(kindOf(d)) })).slice(0, top)
    count = fields.length
    const HEAD = 30
    const labelW = labelWidth(shown, Math.max(100, width * (narrow ? 0.34 : 0.3)))
    const plotW = Math.max(sites.length * 44, width - labelW - MARGIN.right)
    const colW = plotW / sites.length
    const GAP = Math.min(14, colW * 0.18)
    const x = d3.scaleLinear().domain(domain).range([0, colW - GAP])
    const rowH = rowHeight(shown.length, HEAD)
    const innerH = shown.length * rowH
    const withValues = colW >= 88
    plot = fields.length === 0 ? (
      chosen.length === 0 ? <Empty>No fields chosen to rank. Pick a preset or tick some fields under Fields.</Empty>
        : <Empty>No pair reaches n ≥ {minN} at any site in view{narrowed ? ' among the fields chosen' : ''}. Lower the minimum{narrowed ? ', choose more fields' : ''} or widen the subset.</Empty>
    ) : (
      <svg width={labelW + plotW + MARGIN.right} height={innerH + MARGIN.top + HEAD + MARGIN.bottom} className="animated">
        <g transform={`translate(${labelW},${MARGIN.top + HEAD})`}>
          {hoverField && shown.includes(hoverField) && (
            <rect x={-labelW} y={shown.indexOf(hoverField) * rowH} width={labelW + plotW} height={rowH} fill={INK.grid} opacity={0.6} />
          )}
          {shown.map((f, i) => <g key={f}>{fieldLabel(f, labelW, i * rowH + rowH / 2, per.every((m) => m.get(f)?.trivial ?? true))}</g>)}
          {sites.map((s, j) => {
            const m = per[j]
            // the code when not even nine characters of the name would fit
            const name = colW + 8 - 14 < textWidth(s.name.slice(0, 9), 11.5, 600) ? s.code : clip(s.name, colW + 8, 600)
            return (
              <g key={s.code} transform={`translate(${j * colW},0)`}>
                <text x={(colW - GAP) / 2} y={-HEAD + 8} textAnchor="middle" fontSize={11.5} fontWeight={600} fill={INK.text} fontFamily={FONT.family}>
                  <title>{s.name}</title>{name}
                </text>
                <text x={(colW - GAP) / 2} y={-HEAD + 21} textAnchor="middle" fontSize={9.5} fill={INK.muted} fontFamily={FONT.family}>
                  {m.size} pairs
                </text>
                {(signed ? [-1, 0, 1] : [0, 0.5, 1]).map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={innerH} stroke={t === 0 && signed ? INK.axis : INK.grid} />)}
                {shown.map((f, i) => {
                  const cy = i * rowH + rowH / 2
                  const d = m.get(f)
                  const hover = { onMouseEnter: () => setHoverField(f), onMouseLeave: () => { setHoverField(null); tip.hide() } }
                  if (!d) return (
                    <g key={f} {...hover} onMouseMove={(e) => tip.show(e, [`${f} · ${s.name}`, `fewer than ${minN} filters with both`])}>
                      <rect x={0} y={cy - rowH / 2} width={colW - GAP} height={rowH} fill="transparent" />
                      <text x={x(0)} y={cy} dy="0.32em" fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>–</text>
                    </g>
                  )
                  const col = colourOf(d)
                  const v = val(d)
                  const right = x(v) > (colW - GAP) * 0.62
                  return (
                    <g key={f} opacity={lg.show(kindOf(d)) ? lg.dim(kindOf(d), 0.2) : 0.08} {...hover} onMouseMove={(e) => tip.show(e, tipLines(d, s.name))}>
                      <rect x={0} y={cy - rowH / 2} width={colW - GAP} height={rowH} fill="transparent" />
                      {Number.isFinite(v) && <>
                        <line x1={x(0)} x2={x(v)} y1={cy} y2={cy} stroke={col} strokeWidth={2} strokeOpacity={0.55} strokeDasharray={d.trivial ? '3 3' : undefined} />
                        <circle cx={x(v)} cy={cy} r={Math.min(5, rowH / 2.6)} fill={d.trivial ? INK.panel : col} stroke={col} strokeWidth={d.trivial ? 1.5 : 0} />
                      </>}
                      {withValues && Number.isFinite(v) && (
                        // signed: on the empty side of zero, clear of the stem; R²: beside the dot
                        <text x={signed ? x(0) + (v >= 0 ? -5 : 5) : x(v) + (right ? -9 : 9)} y={cy} dy="0.32em" textAnchor={signed ? (v >= 0 ? 'end' : 'start') : right ? 'end' : 'start'} fontSize={10} fill={INK.muted}
                          fontFamily={FONT.mono} stroke={INK.panel} strokeWidth={3} paintOrder="stroke">
                          {fmt(v, 2)}
                        </text>
                      )}
                    </g>
                  )
                })}
                <g transform={`translate(0,${innerH})`} fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>
                  <line x1={0} x2={x(1)} stroke={INK.axis} />
                  <text x={0} y={14} textAnchor="start">{domain[0]}</text>
                  {signed && <text x={x(0)} y={14} textAnchor="middle">0</text>}
                  <text x={x(1)} y={14} textAnchor="end">1</text>
                </g>
              </g>
            )
          })}
          <AxisTitle pick="field">
            <text x={plotW / 2} y={innerH + 36} textAnchor="middle" fontSize={12} fill={INK.axis} fontFamily={FONT.family}>
              {measureText} against {chemText(target)}, each site on its own
              {axisPicker && <tspan className="axis-pick-caret" dx={4} data-export="skip">▾</tspan>}
            </text>
          </AxisTitle>
        </g>
      </svg>
    )
  }

  return (
    <ChartFrame
      id="ranking"
      title={`Lollipop — everything that predicts ${target}, ranked`}
      subtitle={`${measureText} of every other measured quantity against the measurement chosen in the bar, within one site, ranked by strength; side by side, one column per site in a shared order. ${signed ? `Length is ${measureText}, signed (the axis runs −1 to 1)` : 'Length is R² (never signed)'}; colour carries the sign of r. This is the view that tells you where to point a crossplot, instead of drawing 281 of them one pair at a time.`}
      provenance="stands in for 70 barplot + 11 lollipop figures · react-graph-gallery.com/lollipop"
      controls={<>{o.controls()}<FieldMultiSelect meta={meta} {...fieldChoice.props} /><span className="control">{view === 'One site' ? `${count} pairs` : `${Math.min(count, top)} of ${count} fields`}</span></>}
    >
      <div ref={wrapRef} className="chart-wrap">
        {plot}
        {count > 0 && <Legend
          items={[
            { label: 'positive r', color: INK.positive },
            { label: 'negative r', color: INK.negative },
            { label: SAME, color: INK.identity },
          ]}
          {...lg.props}
        />}
        {tip.node}
      </div>
      <ExcludedNote fields={[target]} />
    </ChartFrame>
  )
}
