import { useEffect, useId, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented, Toggle } from '@/components/ChartFrame'
import { ColorLegend, SizeLegend } from '@/components/ColorLegend'
import { FieldSelect } from '@/components/FieldSelect'
import { Legend, toggleIn, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { inDomain, pairDomain, type AxesOpts, type PairDomain } from '@/lib/axes'
import { regression, fmt, type RegressionStats } from '@/lib/stats'
import { INK, MARGIN, FONT, RAMP_SEQUENTIAL, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

type Point = { row: FilterRow; x: number; y: number; c: number | null; s: number | null }
type Scale = d3.ScaleLinear<number, number> | d3.ScaleLogarithmic<number, number>

const LAYOUTS = ['Combined', 'Per site'] as const
const COLOR_BY = ['Site', 'Season', 'Value'] as const
const NO_SIZE = 'none'

/** Sequential ramp over the 2–98 % bulk of a third variable, so one outlier does not wash the rest to the first colour. */
function valueScale(values: number[]) {
  const sorted = values.slice().sort(d3.ascending)
  const lo = d3.quantile(sorted, 0.02) ?? 0
  const hi = d3.quantile(sorted, 0.98) ?? 1
  return d3.scaleSequential(d3.interpolateRgbBasis(RAMP_SEQUENTIAL.slice(2))).domain([lo, hi === lo ? lo + 1 : hi]).clamp(true)
}

/**
 * The crossplot family — 281 of the 827 figures in this repo, by far the
 * largest single group (see gallery/census/CENSUS.md), plus the 123 that
 * colour the same crossplot by a third measured quantity ("Iron colour-coded",
 * "Fe/BC ratio"). That third variable is the `Value` colour mode here, and a
 * fourth can go into the mark size (react-graph-gallery's bubble plot).
 *
 * Interactive counterpart of `plotting/crossplots.py`, including its
 * PlotConfig.layout: "Combined" overlays every site on one axes, "Per site"
 * is the 2×2 grid, with the axis domain shared across panels so the eye can
 * compare slopes. The fit is always errors-in-variables (Deming, λ=1): both
 * axes are measurements with their own uncertainty, so y-on-x least squares
 * would be biased shallow (agreed with Ann, 23 Sep 2026: Deming only).
 *
 * Hover uses closest-point detection (d3.Delaunay) over the whole panel
 * rather than per-circle mouse events, so a dense cloud still answers the
 * pointer and a 2 px mark is as easy to reach as a 12 px one. With `brush to
 * select` on, dragging a rectangle puts the enclosed filters into the shared
 * selection (lib/highlight) and every chart on the tab dims the rest — the
 * "brush a cloud of points, see them on the timeseries" step the README
 * listed as the next one after hover/pin.
 *
 * The 1:1 toggle and the axes mode live in the subset bar so the hexbin and
 * the connected scatter below draw the same frame.
 */
export function Scatterplot({ rows, meta, xField, yField, axes }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string; axes: AxesOpts }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()

  const [colorBy, setColorBy] = useState<(typeof COLOR_BY)[number]>('Site')
  const [colorField, setColorField] = useState(meta.fields.includes('Iron') ? 'Iron' : meta.fields[0])
  const [sizeField, setSizeField] = useState(NO_SIZE)
  const [layout, setLayout] = useState<(typeof LAYOUTS)[number]>('Combined')
  const [showFit, setShowFit] = useState(true)
  const [brushing, setBrushing] = useState(false)
  const { hidden, setHidden, dim, hover, setHover } = useLegend()
  const { identity: showIdentity, log: logAxes } = axes

  const seasonColor = useMemo(() => new Map(meta.seasons.map((s) => [s.name, s.color])), [meta.seasons])
  const groupOf = (r: FilterRow) => (colorBy === 'Season' ? r.season : r.site)
  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

  const points = useMemo<Point[]>(
    () =>
      rows
        .filter((r) => !hidden.has(groupOf(r)))
        .map((r) => ({ row: r, x: r[xField] as number, y: r[yField] as number, c: colorBy === 'Value' ? num(r[colorField]) : null, s: sizeField !== NO_SIZE ? num(r[sizeField]) : null }))
        .filter(
          (p) =>
            typeof p.x === 'number' && typeof p.y === 'number' &&
            Number.isFinite(p.x) && Number.isFinite(p.y) &&
            (!logAxes || (p.x > 0 && p.y > 0))
        ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rows, xField, yField, logAxes, hidden, colorBy, colorField, sizeField]
  )

  // third variable → colour; fourth → size (sqrt, so area is proportional)
  const cScale = useMemo(() => (colorBy === 'Value' ? valueScale(points.map((p) => p.c).filter((v): v is number => v !== null)) : null), [points, colorBy])
  const sScale = useMemo(() => {
    if (sizeField === NO_SIZE) return null
    const vals = points.map((p) => p.s).filter((v): v is number => v !== null && v > 0)
    return d3.scaleSqrt().domain([0, d3.quantile(vals.slice().sort(d3.ascending), 0.98) ?? 1]).range([1.5, 9]).clamp(true)
  }, [points, sizeField])
  const nNoColor = colorBy === 'Value' ? points.filter((p) => p.c === null).length : 0
  const nNoSize = sScale ? points.filter((p) => p.s === null).length : 0

  const colorOf = (p: Point) =>
    colorBy === 'Value' ? (p.c === null || !cScale ? INK.neutral : cScale(p.c)) : colorBy === 'Season' ? seasonColor.get(p.row.season) ?? INK.muted : (p.row.color as string)
  const radiusOf = (p: Point) => (sScale ? (p.s === null ? 1.5 : sScale(p.s)) : 4)

  const panels = useMemo(() => {
    const raw = layout === 'Combined'
      ? [{ key: 'all', title: null as string | null, color: INK.text, points }]
      : meta.sites
          .map((s) => ({ key: s.code, title: s.name, color: s.color, points: points.filter((p) => p.row.site === s.name) }))
          .filter((p) => p.points.length > 0)
    // Deming lambda: when both axes are the same measured quantity we default to
    // 1.0 (orthogonal), exactly as calculate_regression_stats does.
    return raw.map((p) => ({
      ...p,
      stats: regression(p.points.map((q) => q.x), p.points.map((q) => q.y), { errorsInVariables: true }),
    }))
  }, [layout, points, meta.sites, showIdentity])

  // Shared domain across every panel — a per-site grid with different axes
  // per panel silently hides that one site's slope is twice another's.
  const domain = useMemo(() => {
    const intercepts = panels.flatMap((p) => (p.stats && p.stats.demingIntercept !== null ? [p.stats.demingIntercept] : []))
    return pairDomain(points.map((p) => p.x), points.map((p) => p.y), axes, showFit ? intercepts : [])
  }, [points, panels, axes, showFit])

  const legend = useMemo(
    () =>
      (colorBy === 'Season' ? meta.seasons : meta.sites).map((s) => ({
        label: s.name,
        color: colorBy === 'Value' ? INK.neutral : s.color,
        detail: `n=${rows.filter((r) => groupOf(r) === s.name && typeof r[xField] === 'number' && typeof r[yField] === 'number').length}`,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [colorBy, meta, rows, xField, yField]
  )

  // PlotConfig.equal_axes: a 1:1 line only reads correctly on a square panel.
  const MAX_SQUARE = 560
  const cols = layout === 'Per site' && width > 700 ? 2 : 1
  const panelOuterW = Math.floor((width - (cols - 1) * 12) / cols)
  const availW = Math.max(200, panelOuterW - MARGIN.left - MARGIN.right)
  // every crossplot panel is square, with or without the 1:1 line
  const innerW = Math.min(availW, layout === 'Per site' ? 420 : MAX_SQUARE)
  const innerH = innerW

  if (xField === yField) return <ChartFrame title="Scatterplot"><Empty>x and y are the same field — pick two different measurements in the bar above.</Empty></ChartFrame>

  const unitOf = (f: string) => withUnit(f, meta.field_units)

  return (
    <ChartFrame
      id="scatter"
      title="Scatterplot — the crossplot family"
      subtitle="Any measured quantity against any other, coloured by site, by the selected season calendar, or by a third measurement (the 123 'colour-coded' crossplots in the estate); a fourth can size the marks. The stats box reports R² and the errors-in-variables (Deming) fit, since both axes carry measurement error. Hover anywhere near a point to find it in every other chart; click it for the full record."
      provenance="stands in for 281 scatter + 123 colour-by-third-variable figures across 116 notebooks · plotting/crossplots.py · react-graph-gallery.com/scatter-plot · /bubble-plot"
      controls={
        <>
          <Segmented label="layout" value={layout} options={LAYOUTS} onChange={setLayout} title="PlotConfig.layout: combined overlays sites; per site is the grid" />
          <Segmented label="colour" value={colorBy} options={COLOR_BY} onChange={(v) => { setColorBy(v); setHidden(new Set()) }} title="Value: a third measurement on a sequential ramp (2–98 % of its range)" />
          {colorBy === 'Value' && <FieldSelect label="by" value={colorField} meta={meta} onChange={setColorField} />}
          <FieldSelect label="size" value={sizeField} meta={meta} onChange={setSizeField} extraOptions={[NO_SIZE]} title="Bubble plot: mark area proportional to a fourth measurement" />
          <Toggle label="fit lines" checked={showFit} onChange={setShowFit} />
          <Toggle label="brush to select" checked={brushing} onChange={(v) => { setBrushing(v); if (!v) hl.setSelected(null) }} title="Drag a rectangle: the enclosed filters are highlighted on every chart of this tab" />
          {hl.selected && (
            <span className="control">
              <strong>{hl.selected.size}</strong> selected
              <button type="button" className="btn quiet" onClick={() => hl.setSelected(null)}>clear</button>
            </span>
          )}
          <span className="control">{points.length} points</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap centered">
        {points.length < 3 ? (
          <Empty>Fewer than 3 filters carry both {xField} and {yField} in this subset.</Empty>
        ) : (
          <div className="facet-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
            {panels.map((p) => (
              <ScatterPanel
                key={p.key}
                title={p.title}
                titleColor={p.color}
                points={p.points}
                stats={p.stats}
                domain={domain}
                innerW={innerW}
                innerH={innerH}
                axes={axes}
                showFit={showFit}
                xLabel={unitOf(xField)}
                yLabel={unitOf(yField)}
                colorOf={colorOf}
                radiusOf={radiusOf}
                dimOf={(pt) => dim(groupOf(pt.row))}
                fitDim={p.title && colorBy !== 'Season' ? dim(p.title) : 1}
                focusId={hl.focusId}
                selected={hl.selected}
                brushing={brushing}
                onBrush={(ids) => hl.setSelected(ids)}
                onEnter={(e, pt) => {
                  const row = pt.row
                  hl.setHover(row.id)
                  tip.show(e, [
                    row.id,
                    `${row.site} · ${row.date}`,
                    row.season,
                    `${xField}: ${fmt(row[xField] as number)}`,
                    `${yField}: ${fmt(row[yField] as number)}`,
                    ...(colorBy === 'Value' ? [`${colorField}: ${fmt(pt.c)}`] : []),
                    ...(sScale ? [`${sizeField}: ${fmt(pt.s)}`] : []),
                    row.excluded ? `excluded: ${row.exclusion_reason}` : 'click for details',
                  ])
                }}
                onLeave={() => { hl.setHover(null); tip.hide() }}
                onClick={(row) => hl.openSample(row.id)}
              />
            ))}
          </div>
        )}
        <div className="legend">
          {colorBy === 'Value' && cScale ? (
            <ColorLegend scale={cScale} label={unitOf(colorField)} note={nNoColor > 0 ? `${nNoColor} points without ${colorField} in grey` : undefined} />
          ) : (
            <Legend items={legend} hidden={hidden} onToggle={(l) => setHidden((h) => toggleIn(h, l))} onHover={setHover} highlighted={hover} note={showFit ? 'dashed magenta = Deming fit (λ=1)' : undefined} />
          )}
          {sScale && <SizeLegend scale={sScale} label={unitOf(sizeField)} />}
          {sScale && nNoSize > 0 && <span className="legend-note">{nNoSize} points without {sizeField} drawn smallest</span>}
        </div>
        {colorBy === 'Value' && (
          <Legend items={legend} hidden={hidden} onToggle={(l) => setHidden((h) => toggleIn(h, l))} onHover={setHover} highlighted={hover} note="click a site to drop it; colour is the value above" />
        )}
        {domain.dropped > 0 && (
          <Note>{domain.dropped} point{domain.dropped === 1 ? '' : 's'} below zero are outside the axes and not drawn; they are still in the fit. Switch axes to “Data” in the bar to see them.</Note>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/** One axes: used once for the combined layout, once per site for the grid. */
function ScatterPanel({
  title, titleColor, points, stats, domain, innerW, innerH, axes, showFit, xLabel, yLabel, colorOf, radiusOf, dimOf, fitDim, focusId, selected, brushing, onBrush, onEnter, onLeave, onClick,
}: {
  title: string | null
  titleColor: string
  points: Point[]
  stats: RegressionStats | null
  domain: PairDomain
  innerW: number
  innerH: number
  axes: AxesOpts
  showFit: boolean
  xLabel: string
  yLabel: string
  colorOf: (p: Point) => string
  radiusOf: (p: Point) => number
  dimOf: (p: Point) => number
  fitDim: number
  focusId: string | null
  selected: Set<string> | null
  brushing: boolean
  onBrush: (ids: Set<string> | null) => void
  onEnter: (e: React.MouseEvent, p: Point) => void
  onLeave: () => void
  onClick: (row: FilterRow) => void
}) {
  const clipId = useId()
  const mk = (dom: [number, number], range: [number, number]): Scale =>
    axes.log
      ? d3.scaleLog().domain([Math.max(1e-4, dom[0]), dom[1]]).range(range).nice()
      : d3.scaleLinear().domain(dom).range(range).nice()
  const xScale = mk(domain.x, [0, innerW])
  const yScale = mk(domain.y, [innerH, 0])
  const [d0, d1] = xScale.domain() as [number, number]
  // On log axes a fitted line goes negative somewhere left of the data, and a
  // log scale maps that to NaN (React then drops the attribute). Clamp the
  // segment's y to the bottom of the domain; the clipPath hides the rest.
  const yFloor = yScale.domain()[0] as number
  const yv = (v: number) => (axes.log ? Math.max(yFloor, v) : v)
  const seg = (m: number, b: number) => ({ x1: xScale(d0), y1: yScale(yv(m * d0 + b)), x2: xScale(d1), y2: yScale(yv(m * d1 + b)) })
  const drawn = useMemo(() => points.filter((p) => inDomain(domain, p.x, p.y)), [points, domain])
  // closest-point detection: one Delaunay over the drawn points, one hit rect
  const delaunay = useMemo(() => d3.Delaunay.from(drawn, (p) => xScale(p.x), (p) => yScale(p.y)), [drawn, xScale, yScale])
  const hoverRef = useRef<Point | null>(null)
  const HIT = 18
  // d3.brush owns its own DOM, so it is attached imperatively to an empty <g>
  const brushRef = useRef<SVGGElement>(null)
  useEffect(() => {
    const g = brushRef.current
    if (!g) return
    const sel = d3.select(g)
    if (!brushing) { sel.selectAll('*').remove(); sel.on('.brush', null); return }
    const brush = d3.brush<unknown>()
      .extent([[0, 0], [innerW, innerH]])
      .on('end', (ev: d3.D3BrushEvent<unknown>) => {
        if (!ev.selection) { onBrush(null); return }
        const [[x0, y0], [x1, y1]] = ev.selection as [[number, number], [number, number]]
        const ids = new Set<string>()
        for (const p of drawn) {
          const px = xScale(p.x)
          const py = yScale(p.y)
          if (px >= x0 && px <= x1 && py >= y0 && py <= y1) ids.add(p.row.id)
        }
        onBrush(ids.size ? ids : null)
      })
    sel.call(brush)
    return () => { sel.on('.brush', null); sel.selectAll('*').remove() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brushing, drawn, innerW, innerH])
  const nearest = (e: React.MouseEvent<SVGRectElement>): Point | null => {
    if (drawn.length === 0) return null
    const [mx, my] = d3.pointer(e)
    const i = delaunay.find(mx, my)
    const p = drawn[i]
    if (!p) return null
    const dx = xScale(p.x) - mx
    const dy = yScale(p.y) - my
    return Math.hypot(dx, dy) <= HIT + radiusOf(p) ? p : null
  }
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const p = nearest(e)
    if (p !== hoverRef.current) {
      hoverRef.current = p
      if (p) onEnter(e, p)
      else onLeave()
    } else if (p) onEnter(e, p)
  }
  const markIntercepts = axes.mode === 'Show intercept' && showFit && !axes.log && stats
  const originAxes = axes.mode === 'Show intercept' && !axes.log && xScale.domain()[0] < 0 && yScale.domain()[0] < 0
  const boxRows = 3

  return (
    <div>
      {title && (
        <p className="facet-title">
          <span className="swatch" style={{ background: titleColor }} /> {title}
          <span style={{ color: 'var(--ink-muted)', fontWeight: 400, fontFamily: 'var(--mono)', fontSize: 11 }}>n={points.length}</span>
        </p>
      )}
      <svg width={innerW + MARGIN.left + MARGIN.right} height={innerH + MARGIN.top + MARGIN.bottom} className="animated">
        <defs>
          {/* fit and identity lines are clipped to the axes: a slope above 1 otherwise leaves through the top margin */}
          <clipPath id={clipId}>
            <rect x={0} y={0} width={innerW} height={innerH} />
          </clipPath>
        </defs>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* In 'Show intercept' mode the spines cross at the origin, matplotlib-style, so the
              intercept markers sit on the y axis and the x axis runs through the cloud. */}
          {originAxes ? (
            <>
              <YAxis scale={yScale} x={xScale(0)} label={yLabel} gridWidth={innerW} gridFrom={0} labelX={-50} />
              <XAxis scale={xScale} y={yScale(0)} label={xLabel} labelDy={innerH - yScale(0) + 40} />
            </>
          ) : (
            <>
              <YAxis scale={yScale} x={0} label={yLabel} gridWidth={innerW} />
              <XAxis scale={xScale} y={innerH} label={xLabel} />
            </>
          )}

          <g clipPath={`url(#${clipId})`}>
            {axes.identity && (
              <line x1={xScale(d0)} y1={yScale(d0)} x2={xScale(d1)} y2={yScale(d1)} stroke={INK.identity} strokeWidth={1.5} strokeDasharray="5 4" />
            )}
            {stats && showFit && (
              <g opacity={fitDim}>
                {stats.demingSlope !== null && stats.demingIntercept !== null && (
                  <line {...seg(stats.demingSlope, stats.demingIntercept)} stroke={INK.deming} strokeWidth={2} strokeDasharray="7 3" pointerEvents="none" />
                )}
              </g>
            )}
          </g>

          {drawn.map((p) => {
            const st = focusStyle(p.row.id, focusId, { r: radiusOf(p), opacity: 0.72 }, selected)
            const k = dimOf(p)
            return (
              <circle
                key={p.row.id}
                cx={xScale(p.x)}
                cy={yScale(p.y)}
                r={st.r}
                fill={colorOf(p)}
                fillOpacity={st.opacity * k}
                stroke={st.stroke}
                strokeOpacity={k}
                strokeWidth={st.strokeWidth}
                pointerEvents="none"
              />
            )
          })}

          {/* the hit layer: closest point wins, within HIT px of the pointer; the brush replaces it while selecting */}
          {!brushing && (
            <rect
              x={0} y={0} width={innerW} height={innerH} fill="transparent"
              style={{ cursor: hoverRef.current ? 'pointer' : 'default' }}
              onMouseMove={onMove}
              onMouseLeave={() => { hoverRef.current = null; onLeave() }}
              onClick={(e) => { const p = nearest(e); if (p) onClick(p.row) }}
            />
          )}
          <g ref={brushRef} className="brush-layer" />

          {markIntercepts && (
            <g pointerEvents="none" fontFamily={FONT.mono} fontSize={10.5}>
              {!originAxes && <line x1={xScale(0)} x2={xScale(0)} y1={0} y2={innerH} stroke={INK.axis} strokeDasharray="2 3" strokeOpacity={0.6} />}
              {stats.demingIntercept !== null && (
                <>
                  <circle cx={xScale(0)} cy={yScale(stats.demingIntercept)} r={4.5} fill="#fff" stroke={INK.deming} strokeWidth={2} />
                  <text x={xScale(0) + 8} y={yScale(stats.demingIntercept)} dy="0.35em" fill={INK.deming} stroke="#fff" strokeWidth={3} paintOrder="stroke">b = {fmt(stats.demingIntercept, 3)}</text>
                </>
              )}
            </g>
          )}

          {stats && (
            <g transform="translate(10,10)" fontFamily={FONT.mono} fontSize={11} pointerEvents="none">
              <rect width={224} height={boxRows * 15 + 12} rx={5} fill="#fff" fillOpacity={0.93} stroke={INK.border} />
              {(() => {
                const lines: { t: string; c: string }[] = [
                  { t: `n = ${stats.n}`, c: INK.text },
                  { t: `R² = ${fmt(stats.r2, 4)}`, c: INK.text },
                ]
                if (stats.demingSlope !== null && stats.demingIntercept !== null) {
                  const b = stats.demingIntercept
                  lines.push({ t: `Deming y = ${fmt(stats.demingSlope)}x ${b < 0 ? '−' : '+'} ${fmt(Math.abs(b))}`, c: INK.deming })
                }
                return lines.map((l, i) => (
                  <text key={i} x={9} y={18 + i * 15} fill={l.c}>{l.t}</text>
                ))
              })()}
            </g>
          )}
        </g>
      </svg>
    </div>
  )
}
