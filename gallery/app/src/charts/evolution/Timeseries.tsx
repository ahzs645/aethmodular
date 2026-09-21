import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Toggle } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const LAYOUTS = ['Overlay', 'Per site'] as const
type Pt = { row: FilterRow; date: Date; value: number }

/**
 * Timeseries / area — 179 evolution figures in the estate (timeseries, line,
 * fill_between bands). Seasonal shading uses config.ETHIOPIA_SEASONS colours,
 * shipped through meta.json, so it matches the matplotlib axvspan overlays
 * rather than re-declaring a calendar (AGENTS.md is emphatic about this).
 *
 * "Per site" gives each site its own time axis — Beijing's 2013–2016 run and
 * Delhi's 2022–2024 run share almost no dates, so an overlay is mostly empty
 * space. Which sites appear is the subset bar's decision.
 *
 * A synchronized cursor (react-graph-gallery's line-chart pattern) follows
 * the pointer by *date* across every panel: hovering 2023-03 on the Addis
 * panel drops the same date line on Delhi's, with each series' nearest
 * value labelled, which is how "was Delhi high the same month" gets read.
 */
export function Timeseries({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()

  const [layout, setLayout] = useState<(typeof LAYOUTS)[number]>('Per site')
  const [smooth, setSmooth] = useState(true)
  const [seasonShading, setSeasonShading] = useState(true)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [cursor, setCursor] = useState<Date | null>(null)

  const series = useMemo(
    () =>
      meta.sites
        .filter((s) => !hidden.has(s.name))
        .map((s) => {
          const pts: Pt[] = rows
            .filter((r) => r.site === s.name && typeof r[field] === 'number' && Number.isFinite(r[field] as number))
            .map((r) => ({ row: r, date: new Date(r.date + 'T00:00:00'), value: r[field] as number }))
            .sort((a, b) => +a.date - +b.date)
          return { name: s.name, color: s.color, pts }
        })
        .filter((s) => s.pts.length > 1),
    [rows, field, meta.sites, hidden]
  )

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const panelH = layout === 'Overlay' ? 430 : 250
  const innerH = panelH - MARGIN.top - MARGIN.bottom

  const panels = layout === 'Overlay' ? [{ key: 'all', title: null as string | null, color: INK.text, series }] : series.map((s) => ({ key: s.name, title: s.name, color: s.color, series: [s] }))

  // shared y so panels compare; per-panel x so each site fills its own width
  const allPts = series.flatMap((s) => s.pts)
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(allPts, (p) => p.value) ?? 0), (d3.max(allPts, (p) => p.value) ?? 1) * 1.08]).range([innerH, 0]).nice()

  /** Centred rolling median — robust to the spikes these series carry. */
  const rolling = (pts: Pt[], w = 7) =>
    pts.map((p, i) => {
      const lo = Math.max(0, i - Math.floor(w / 2))
      const win = pts.slice(lo, lo + w).map((q) => q.value).sort(d3.ascending)
      return { date: p.date, value: d3.quantile(win, 0.5) ?? p.value }
    })

  const bisect = d3.bisector<Pt, Date>((p) => p.date).center
  /** The series point nearest the cursor date, or null when further than 45 days. */
  const nearest = (pts: Pt[], d: Date): Pt | null => {
    if (!pts.length) return null
    const p = pts[bisect(pts, d)]
    return p && Math.abs(+p.date - +d) <= 45 * 86400e3 ? p : null
  }

  const bandsFor = (x: d3.ScaleTime<number, number>) => {
    if (!seasonShading) return []
    const [d0, d1] = x.domain() as [Date, Date]
    const bands: { x: number; w: number; color: string }[] = []
    const monthToSeason = new Map<number, string>()
    for (const s of meta.seasons) for (const m of s.months) monthToSeason.set(m, s.color)
    let cur = new Date(d0.getFullYear(), d0.getMonth(), 1)
    while (cur < d1) {
      const next = new Date(cur.getFullYear(), cur.getMonth() + 1, 1)
      const c = monthToSeason.get(cur.getMonth() + 1)
      if (c) {
        const x0 = Math.max(0, x(cur))
        const x1 = Math.min(innerW, x(next))
        if (x1 > x0) bands.push({ x: x0, w: x1 - x0, color: c })
      }
      cur = next
    }
    return bands
  }

  return (
    <ChartFrame
      id="timeseries"
      title="Timeseries — filter measurements over time"
      subtitle={`Every filter as a point, with an optional 7-sample rolling median. Season bands are the Ethiopian calendar under the ${meta.season_convention} convention chosen above, the same colours the matplotlib overlays use. Move the pointer across any panel and a date cursor follows on every other panel. Drag on the timeline in the subset bar to zoom every tab to a date range.`}
      provenance="stands in for 73 timeseries + 59 area + 46 line figures · plotting/timeseries.py · react-graph-gallery.com/timeseries · /line-chart-synchronized-cursors"
      controls={
        <>
          <Segmented label="layout" value={layout} options={LAYOUTS} onChange={setLayout} />
          <Toggle label="rolling median" checked={smooth} onChange={setSmooth} />
          <Toggle label="season shading" checked={seasonShading} onChange={setSeasonShading} />
          {cursor && <span className="control" style={{ fontFamily: 'var(--mono)' }}>{d3.timeFormat('%Y-%m-%d')(cursor)}</span>}
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {series.length === 0 ? (
          <Empty>No site has 2 or more filters with {field} in this subset.</Empty>
        ) : (
          panels.map((panel) => {
            const pts = panel.series.flatMap((s) => s.pts)
            const x = d3.scaleTime().domain(d3.extent(pts, (p) => p.date) as [Date, Date]).range([0, innerW])
            const line = d3.line<{ date: Date; value: number }>().x((p) => x(p.date)).y((p) => y(p.value)).curve(d3.curveMonotoneX)
            const [x0d, x1d] = x.domain() as [Date, Date]
            const cursorIn = cursor && cursor >= x0d && cursor <= x1d
            return (
              <div key={panel.key}>
                {panel.title && (
                  <p className="facet-title">
                    <span className="swatch" style={{ background: panel.color }} /> {panel.title}
                    <span style={{ color: 'var(--ink-muted)', fontWeight: 400, fontFamily: 'var(--mono)', fontSize: 11 }}>n={pts.length}</span>
                  </p>
                )}
                <svg width={width} height={panelH}>
                  <g
                    transform={`translate(${MARGIN.left},${MARGIN.top})`}
                    onMouseMove={(e) => {
                      const [mx] = d3.pointer(e, e.currentTarget)
                      if (mx >= 0 && mx <= innerW) setCursor(x.invert(mx))
                    }}
                    onMouseLeave={() => setCursor(null)}
                  >
                    {/* catches pointer moves in the empty parts of the panel */}
                    <rect x={0} y={0} width={innerW} height={innerH} fill="transparent" />
                    {bandsFor(x).map((b, i) => (
                      <rect key={i} x={b.x} y={0} width={b.w} height={innerH} fill={b.color} fillOpacity={0.1} pointerEvents="none" />
                    ))}
                    <YAxis scale={y} x={0} label={withUnit(field, meta.field_units)} gridWidth={innerW} tickCount={layout === 'Overlay' ? 6 : 4} />
                    <XAxis scale={x as any} y={innerH} format={(d: Date) => d3.timeFormat('%b %Y')(d)} tickCount={7} />
                    {panel.series.map((s) => (
                      <g key={s.name}>
                        {s.pts.map((p) => {
                          const st = focusStyle(p.row.id, hl.focusId, { r: 2.6, opacity: smooth ? 0.32 : 0.75 })
                          return (
                            <circle
                              key={p.row.id} cx={x(p.date)} cy={y(p.value)} r={st.r}
                              fill={s.color} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={hl.focusId === p.row.id ? 2 : 0}
                              pointerEvents="none"
                            />
                          )
                        })}
                        {smooth && <path d={line(rolling(s.pts)) ?? ''} fill="none" stroke={s.color} strokeWidth={2.2} pointerEvents="none" />}
                        {/* invisible hit targets on top: a 2.6 px dot is too small to hover or click */}
                        {s.pts.map((p) => (
                          <circle
                            key={`hit-${p.row.id}`} cx={x(p.date)} cy={y(p.value)} r={7} fill="transparent"
                            style={{ cursor: 'pointer' }}
                            onMouseEnter={(e) => { hl.setHover(p.row.id); tip.show(e, [p.row.id, `${p.row.date} · ${p.row.season}`, `${field}: ${fmt(p.value)}`, p.row.excluded ? `excluded: ${p.row.exclusion_reason}` : 'click for details']) }}
                            onMouseLeave={() => { hl.setHover(null); tip.hide() }}
                            onClick={() => hl.openSample(p.row.id)}
                          />
                        ))}
                      </g>
                    ))}
                    {cursorIn && cursor && (
                      <g pointerEvents="none" fontFamily={FONT.mono} fontSize={10.5}>
                        <line x1={x(cursor)} x2={x(cursor)} y1={0} y2={innerH} stroke={INK.text} strokeOpacity={0.5} strokeDasharray="3 3" />
                        {panel.series.map((s) => {
                          const p = nearest(s.pts, cursor)
                          if (!p) return null
                          const px = x(p.date)
                          const py = y(p.value)
                          const right = px < innerW - 90
                          return (
                            <g key={s.name}>
                              <circle cx={px} cy={py} r={5} fill="#fff" stroke={s.color} strokeWidth={2} />
                              <text x={right ? px + 9 : px - 9} y={py} dy="0.35em" textAnchor={right ? 'start' : 'end'} fill={s.color} stroke="#fff" strokeWidth={3} paintOrder="stroke" fontWeight={600}>
                                {fmt(p.value, 2)} · {d3.timeFormat('%d %b %y')(p.date)}
                              </text>
                            </g>
                          )
                        })}
                      </g>
                    )}
                  </g>
                </svg>
              </div>
            )
          })
        )}
        <Legend
          items={[
            ...meta.sites.filter((s) => rows.some((r) => r.site === s.name)).map((s) => ({ label: s.name, color: s.color })),
            ...(seasonShading ? meta.seasons.map((s) => ({ label: s.name, color: s.color, shape: 'band' as const })) : []),
          ]}
          hidden={hidden}
          onToggle={(l) => { if (meta.sites.some((s) => s.name === l)) setHidden((h) => toggleIn(h, l)) }}
          note="hover any panel: the dashed cursor is the same date on every panel"
        />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
