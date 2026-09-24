import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented, Select, Toggle } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useHighlight } from '@/lib/highlight'
import { fmt } from '@/lib/stats'
import { INK, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const COLOR_BY = ['Site', 'Season'] as const
/** an axis needs this many numeric values before it earns a column */
const MIN_N = 20
const HEIGHT = 440
const BOTTOM = 14
/** half-width of the brushable strip either side of an axis, px */
const BRUSH_HALF = 8
/** rough glyph widths for sizing the gutters: 11 px sans, 9.5 px mono */
const PX_PER_CHAR = 6.5
const PX_PER_MONO = 5.8

type Axis = {
  field: string
  /** field name and its bracketed unit, drawn on two lines so the title stays narrow */
  name: string
  unit: string
  y: d3.ScaleLinear<number, number>
  /** raw value → plotted value (÷ p98 when normalised, identity otherwise) */
  plot: (v: number) => number
  ticks: { v: number; label: string }[]
  /** how many values fell outside the axis domain and sit at its end */
  nClamped: number
}
type Line = { row: FilterRow; values: (number | null)[]; ys: (number | null)[]; d: string }
type PixelRange = [number, number]

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const without = (b: Record<string, PixelRange>, field: string) => Object.fromEntries(Object.entries(b).filter(([k]) => k !== field))

/**
 * Parallel coordinates — every species of a family, every filter.
 *
 * The radar above it draws one polygon per *site* (a median per species),
 * which is the composition question. This is the same set of axes with one
 * polyline per *filter*, which is the covariance question: does a filter
 * that is high in sulfate also carry high ammonium, and is that the same set
 * of filters in every season. Nothing in the estate draws it — the closest
 * are the 123 colour-by-third-variable crossplots, which can only show one
 * pair at a time.
 *
 * Each axis has its own scale over the 2–98 % bulk of that species, because
 * a single Delhi calcium spike would otherwise flatten every other filter to
 * the bottom of the axis. Values outside are clamped to the axis end and
 * counted in the note under the chart rather than dropped. `normalise` puts
 * every axis on 0–1 (÷ p98) so the axes share one scale, at the cost of the
 * tick labels meaning something.
 *
 * Brushing is what makes the chart worth drawing: a range on one or more
 * axes keeps only the filters inside every active range. d3.brushY owns a
 * small <g> per axis; React owns everything around it, as in DateBrush.
 */
export function ParallelCoordinates({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()

  const groupNames = meta.field_groups.map((g) => g.label)
  const [group, setGroup] = useState(groupNames.includes('Ions') ? 'Ions' : groupNames[0] ?? '')
  const [colorBy, setColorBy] = useState<(typeof COLOR_BY)[number]>('Site')
  const [normalise, setNormalise] = useState(false)
  const { hidden, setHidden, dim, props: legendProps } = useLegend()
  /** active brush per field, in pixels of that axis; pixel space is what the eye selected, so clamped values at an axis end are brushable */
  const [brushes, setBrushes] = useState<Record<string, PixelRange>>({})
  // bumping this rebuilds every brush from scratch, which is how "clear" wipes the d3-owned rectangles
  const [brushEpoch, setBrushEpoch] = useState(0)
  const brushRefs = useRef(new Map<string, SVGGElement>())

  const seasonColor = useMemo(() => new Map(meta.seasons.map((s) => [s.name, s.color])), [meta.seasons])
  const groupOf = (r: FilterRow) => (colorBy === 'Season' ? r.season : r.site)
  const colorOf = (r: FilterRow) => (colorBy === 'Season' ? seasonColor.get(r.season) ?? INK.muted : r.color)

  // The fields worth an axis, with their sorted values for the quantiles.
  // Computed over every row (not just the visible groups) so hiding a site
  // in the legend does not move the axes under an active brush.
  const fields = useMemo(() => {
    const candidates = (meta.field_groups.find((g) => g.label === group)?.fields ?? []).filter((f) => !f.includes('uncertainty') && !f.includes('MDL'))
    return candidates
      .map((field) => ({ field, sorted: rows.map((r) => num(r[field])).filter((v): v is number => v !== null).sort(d3.ascending) }))
      .filter((f) => f.sorted.length >= MIN_N)
  }, [meta.field_groups, group, rows])

  // ---- layout: gutters from the widest title, a second title row when the axes are closer than a title is wide
  const titles = fields.map((f) => {
    const full = withUnit(f.field, meta.field_units)
    return { name: f.field, unit: full.slice(f.field.length).trim() }
  })
  const maxTitleW = Math.max(0, ...titles.map((t) => Math.max(t.name.length, t.unit.length))) * PX_PER_CHAR
  const halfTitle = maxTitleW / 2 + 4
  // tick labels hang to the left of every axis; the leftmost needs a gutter of
  // its own, budgeted for a 7-character label ("0.0125", "1,000") so nothing
  // leaves the svg box (the README's rendering invariant)
  const tickW = 7 * PX_PER_MONO + 12
  const gutterL = Math.max(halfTitle, tickW)
  const gutterR = halfTitle
  const innerW = Math.max(120, width - gutterL - gutterR)
  const step = fields.length > 1 ? innerW / (fields.length - 1) : innerW
  const stagger = fields.length > 1 && maxTitleW > step * 0.95
  const top = stagger ? 58 : 36
  const innerH = HEIGHT - top - BOTTOM
  const xs = fields.map((_, i) => (fields.length > 1 ? (i * innerW) / (fields.length - 1) : innerW / 2))

  const axes = useMemo<Axis[]>(
    () =>
      fields.map((f, i) => {
        const p02 = d3.quantile(f.sorted, 0.02) ?? 0
        const p98 = d3.quantile(f.sorted, 0.98) ?? 1
        // ÷ p98 needs a positive denominator; fall back to the max, then to 1
        const denom = p98 > 0 ? p98 : (f.sorted[f.sorted.length - 1] ?? 0) > 0 ? f.sorted[f.sorted.length - 1] : 1
        const plot = normalise ? (v: number) => v / denom : (v: number) => v
        const domain: [number, number] = normalise ? [0, 1] : [p02, p98 === p02 ? p02 + 1 : p98]
        const y = d3.scaleLinear().domain(domain).range([innerH, 0]).clamp(true)
        if (!normalise) y.nice()
        const [lo, hi] = y.domain() as [number, number]
        const nClamped = f.sorted.reduce((n, v) => n + (plot(v) < lo || plot(v) > hi ? 1 : 0), 0)
        // tiny magnitudes (Magnesium ion sits near 1e-4 µg/m³) would print as 7-digit decimals that overrun the axis
        const format = Math.max(Math.abs(lo), Math.abs(hi)) < 0.01 && hi !== lo ? d3.format('.1~e') : y.tickFormat(5)
        return { field: f.field, name: titles[i].name, unit: titles[i].unit, y, plot, ticks: y.ticks(5).map((v) => ({ v, label: format(v) })), nClamped }
      }),
    // titles derive from fields, so they are covered by that dependency
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [fields, normalise, innerH]
  )

  // One polyline per filter with values on at least 2 axes; d3.line().defined
  // leaves a gap where a species is missing rather than bridging it.
  const allLines = useMemo<Line[]>(() => {
    const line = d3.line<number | null>().defined((v) => v !== null).x((_, i) => xs[i]).y((v) => v as number)
    return rows
      .map((row) => {
        const values = axes.map((a) => num(row[a.field]))
        const ys = values.map((v, i) => (v === null ? null : axes[i].y(axes[i].plot(v))))
        return { row, values, ys, d: line(ys) ?? '' }
      })
      .filter((l) => l.values.filter((v) => v !== null).length >= 2)
    // xs derives from fields + innerW, both already in the axes dependency chain
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, axes, innerW])
  const lines = useMemo(() => allLines.filter((l) => !hidden.has(groupOf(l.row))), [allLines, hidden, colorBy]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---- brushing: a filter survives only if it is inside every active range
  const active = Object.entries(brushes).filter(([f]) => axes.some((a) => a.field === f))
  const anyBrush = active.length > 0
  const axisIndex = new Map(axes.map((a, i) => [a.field, i]))
  const inBrush = (l: Line) =>
    active.every(([f, [y0, y1]]) => {
      const i = axisIndex.get(f)
      if (i === undefined) return true
      const py = l.ys[i]
      return py !== null && py >= y0 && py <= y1
    })
  const selected = useMemo(() => new Set(lines.filter(inBrush).map((l) => l.row.id)), [lines, brushes]) // eslint-disable-line react-hooks/exhaustive-deps
  // the brushed set is shared with every other chart on the page (lib/highlight),
  // so a brushed sulfate range shows up as the same filters on the lollipop's
  // partner tabs; cleared here, it clears everywhere
  const hadBrush = useRef(false)
  useEffect(() => {
    // only this chart's own brushes write the shared set: mounting with no
    // brush must not wipe a selection made on the scatterplot
    if (anyBrush) hl.setSelected(selected)
    else if (hadBrush.current) hl.setSelected(null)
    hadBrush.current = anyBrush
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, anyBrush])

  const axisKey = axes.map((a) => a.field).join('|')
  useEffect(() => {
    const cleanups: (() => void)[] = []
    for (const a of axes) {
      const g = brushRefs.current.get(a.field)
      if (!g) continue
      const sel = d3.select(g)
      const brush = d3
        .brushY<unknown>()
        .extent([[-BRUSH_HALF, 0], [BRUSH_HALF, innerH]])
        .on('brush end', (ev: d3.D3BrushEvent<unknown>) => {
          if (!ev.sourceEvent) return // programmatic move — handled where it was issued
          const s = ev.selection as PixelRange | null
          setBrushes((b) => (s ? { ...b, [a.field]: s } : a.field in b ? without(b, a.field) : b))
        })
      sel.call(brush)
      // the svg's `animated` transition would make the brush trail the pointer
      sel.selectAll('rect').style('transition', 'none')
      sel.selectAll('.selection').attr('fill', INK.accent).attr('fill-opacity', 0.18).attr('stroke', INK.accent)
      sel.on('dblclick', () => {
        sel.call(brush.move, null)
        setBrushes((b) => without(b, a.field))
      })
      cleanups.push(() => {
        sel.on('.brush', null)
        sel.on('dblclick', null)
        sel.selectAll('*').remove()
      })
    }
    // a rebuild means fresh, empty brushes; pixel ranges from the old scale would lie
    setBrushes({})
    return () => cleanups.forEach((fn) => fn())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [axisKey, innerH, normalise, brushEpoch, lines.length > 0])

  const clearBrushes = () => {
    setBrushes({})
    setBrushEpoch((e) => e + 1)
  }

  const legend = useMemo(
    () =>
      (colorBy === 'Season' ? meta.seasons : meta.sites).map((s) => ({
        label: s.name,
        color: s.color,
        detail: `n=${allLines.filter((l) => groupOf(l.row) === s.name).length}`,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [colorBy, meta.seasons, meta.sites, allLines]
  )

  const nClamped = axes.reduce((n, a) => n + a.nClamped, 0)
  const focusId = hl.focusId
  const strokeOpacityOf = (l: Line) => {
    if (focusId === l.row.id) return 1
    if (anyBrush && !selected.has(l.row.id)) return 0.05
    if (focusId) return 0.12
    return 0.35
  }
  const focused = lines.find((l) => l.row.id === focusId)
  const showTip = (e: React.MouseEvent, l: Line) => {
    hl.setHover(l.row.id)
    tip.show(e, [l.row.id, `${l.row.site} · ${l.row.date}`, ...axes.map((a, i) => `${a.field}: ${fmt(l.values[i])}`)])
  }

  return (
    <ChartFrame
      id="parallel"
      title="Parallel coordinates — every species of a family, every filter"
      subtitle="One polyline per filter across one axis per species, where the radar shows only a site median per axis: the whole composition of every filter at once, so co-varying species read as parallel strands and a filter that is odd on one axis shows which others it is odd on. Each axis spans the 2–98 % bulk of its species. Drag along an axis to keep only the filters in that range; brush several axes to intersect them. Hover a line to find the filter in every other chart; click it for the full record."
      provenance="react-graph-gallery.com/parallel-plot · not in the estate (0 figures)"
      controls={
        <>
          <Select label="species family" value={group} options={groupNames} onChange={(v) => { setGroup(v); setBrushes({}) }} />
          <Segmented label="colour" value={colorBy} options={COLOR_BY} onChange={(v) => { setColorBy(v); setHidden(new Set()) }} />
          <Toggle label="normalise" checked={normalise} onChange={(v) => { setNormalise(v); setBrushes({}) }} title="every axis 0–1: value ÷ that species' 98th percentile, so the axes share one scale" />
          {anyBrush ? (
            <span className="control">
              <strong>{selected.size}</strong> of {lines.length} filters selected
              <button type="button" className="btn quiet" onClick={clearBrushes} title="Remove every axis brush (double-click an axis clears just that one)">
                clear
              </button>
            </span>
          ) : (
            <span className="control">{lines.length} filters</span>
          )}
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {axes.length < 2 ? (
          <Empty>Fewer than 2 {group} species carry ≥ {MIN_N} values in this subset.</Empty>
        ) : lines.length === 0 ? (
          <Empty>No filter in this subset carries at least 2 {group} species.</Empty>
        ) : (
          <svg width={innerW + gutterL + gutterR} height={HEIGHT} className="animated">
            <g transform={`translate(${gutterL},${top})`}>
              {/* axes, ticks and titles first so the lines paint over the grid */}
              {axes.map((a, i) => {
                const row = stagger && i % 2 === 0 ? -24 : 0
                return (
                  <g key={a.field} transform={`translate(${xs[i]},0)`} fontFamily={FONT.family}>
                    <line y1={0} y2={innerH} stroke={INK.axis} />
                    {a.ticks.map((t) => (
                      <g key={t.v} transform={`translate(0,${a.y(t.v)})`}>
                        <line x1={-4} x2={0} stroke={INK.axis} />
                        <text x={-6} dy="0.32em" textAnchor="end" fontSize={9.5} fontFamily={FONT.mono} fill={INK.muted} stroke="#fff" strokeWidth={3} paintOrder="stroke">
                          {t.label}
                        </text>
                      </g>
                    ))}
                    <text y={-22 + row} textAnchor="middle" fontSize={11} fontWeight={600} fill={INK.text}>
                      {a.name}
                    </text>
                    {a.unit && (
                      <text y={-9 + row} textAnchor="middle" fontSize={9.5} fill={INK.muted}>
                        {normalise ? '÷ p98' : a.unit}
                      </text>
                    )}
                  </g>
                )
              })}

              {/* the lines: unfocused first, the focused one last so it sits on top */}
              <g fill="none" strokeLinejoin="round" strokeLinecap="round" pointerEvents="none">
                {lines.map((l) =>
                  l.row.id === focusId ? null : (
                    <path key={l.row.id} d={l.d} stroke={colorOf(l.row)} strokeWidth={1} strokeOpacity={strokeOpacityOf(l) * dim(groupOf(l.row))} />
                  )
                )}
                {focused && (
                  <>
                    <path d={focused.d} stroke={INK.fit} strokeWidth={4.5} strokeOpacity={1} />
                    <path d={focused.d} stroke={colorOf(focused.row)} strokeWidth={2.5} strokeOpacity={1} />
                  </>
                )}
              </g>

              {/* the hit layer: a wide invisible stroke per line, so a 1 px line answers the pointer */}
              <g fill="none" stroke="transparent" strokeWidth={7} pointerEvents="stroke" style={{ cursor: 'pointer' }}>
                {lines.map((l) => (
                  <path
                    key={l.row.id}
                    d={l.d}
                    onMouseEnter={(e) => showTip(e, l)}
                    onMouseMove={(e) => tip.show(e, [l.row.id, `${l.row.site} · ${l.row.date}`, ...axes.map((a, i) => `${a.field}: ${fmt(l.values[i])}`)])}
                    onMouseLeave={() => { hl.setHover(null); tip.hide() }}
                    onClick={() => hl.openSample(l.row.id)}
                  />
                ))}
              </g>

              {/* brush strips, one per axis; d3 owns the contents of each <g> */}
              {axes.map((a, i) => (
                <g
                  key={`brush-${a.field}`}
                  transform={`translate(${xs[i]},0)`}
                  ref={(el) => {
                    if (el) brushRefs.current.set(a.field, el)
                    else brushRefs.current.delete(a.field)
                  }}
                />
              ))}
            </g>
          </svg>
        )}
        {tip.node}
      </div>
      <Legend
        items={legend}
        {...legendProps}
        note={anyBrush ? 'faded lines fall outside an active brush' : 'click a group to drop it; drag along an axis to brush'}
      />
      {nClamped > 0 && (
        <Note>
          {nClamped} value{nClamped === 1 ? '' : 's'} {normalise ? 'beyond the 98th percentile (or below zero)' : 'outside the 2–98 % axis range'} {nClamped === 1 ? 'is' : 'are'} drawn at the axis end rather than dropped.
        </Note>
      )}
    </ChartFrame>
  )
}
