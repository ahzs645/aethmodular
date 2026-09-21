import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { boxStats, kde, fmt } from '@/lib/stats'
import { INK, MARGIN, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const MODES = ['Boxplot', 'Box + points', 'Violin', 'Beeswarm'] as const
const GROUPS = ['Site', 'Season'] as const

/**
 * Distribution family — 89 figures in the estate, almost all of them
 * matplotlib boxplots. react-graph-gallery offers violin and beeswarm as
 * alternatives for the same data, so all three share one control here:
 * the point of the gallery is being able to switch encoding without
 * re-running a notebook. The beeswarm's points are the same filters as the
 * scatterplot's, so hovering links across.
 */
export function DistributionPanel({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const height = 440

  const [groupBy, setGroupBy] = useState<(typeof GROUPS)[number]>('Site')
  const [mode, setMode] = useState<(typeof MODES)[number]>('Boxplot')

  const seasonColor = useMemo(() => new Map(meta.seasons.map((s) => [s.name, s.color])), [meta.seasons])
  const siteColor = useMemo(() => new Map(meta.sites.map((s) => [s.name, s.color])), [meta.sites])

  const groups = useMemo(() => {
    const key = (r: FilterRow) => (groupBy === 'Site' ? r.site : r.season)
    const order = groupBy === 'Site' ? meta.sites.map((s) => s.name) : meta.seasons.map((s) => s.name)
    const m = new Map<string, { v: number; row: FilterRow }[]>()
    for (const r of rows) {
      const v = r[field]
      if (typeof v !== 'number' || !Number.isFinite(v)) continue
      const k = key(r)
      if (!m.has(k)) m.set(k, [])
      m.get(k)!.push({ v, row: r })
    }
    return order
      .filter((o) => (m.get(o)?.length ?? 0) >= 4)
      .map((name) => {
        const items = m.get(name)!
        return {
          name,
          items,
          values: items.map((i) => i.v),
          color: (groupBy === 'Site' ? siteColor.get(name) : seasonColor.get(name)) ?? INK.muted,
          box: boxStats(items.map((i) => i.v))!,
        }
      })
  }, [rows, field, groupBy, meta, siteColor, seasonColor])

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom

  const x = d3.scaleBand<string>().domain(groups.map((g) => g.name)).range([0, innerW]).padding(0.34)
  const allVals = groups.flatMap((g) => g.values)
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(allVals) ?? 0), (d3.max(allVals) ?? 1) * 1.05]).range([innerH, 0]).nice()

  // deterministic jitter so the beeswarm doesn't dance on every re-render
  const jitter = (i: number, w: number) => {
    const t = Math.sin(i * 12.9898) * 43758.5453
    return (t - Math.floor(t) - 0.5) * w
  }

  return (
    <ChartFrame
      id="distribution"
      title="Boxplot · Violin · Beeswarm — one distribution, four encodings"
      subtitle="The same grouped values drawn four ways. The notebooks commit to a boxplot at write time; here the encoding is a control, which is what makes the shape of a small group (Delhi, n≈97) legible rather than hidden behind a five-number summary. 'Box + points' is the gallery's boxplot-with-jitter: the summary and every filter it summarises."
      provenance="stands in for 50 boxplot + 3 violin figures · plotting/distributions.py · react-graph-gallery.com/boxplot · /boxplot-jitter"
      controls={
        <>
          <Segmented label="group by" value={groupBy} options={GROUPS} onChange={setGroupBy} />
          <Segmented label="encoding" value={mode} options={MODES} onChange={setMode} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {groups.length === 0 ? (
          <Empty>No group has 4 or more filters with {field} in this subset.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={withUnit(field, meta.field_units)} gridWidth={innerW} />
              <XAxis scale={x} y={innerH} />

              {groups.map((g, gi) => {
                const cx = (x(g.name) ?? 0) + x.bandwidth() / 2
                const bw = x.bandwidth()

                if (mode === 'Violin') {
                  const lo = d3.min(g.values)!
                  const hi = d3.max(g.values)!
                  const grid = d3.range(40).map((i) => lo + ((hi - lo) * i) / 39)
                  const dens = kde(g.values, grid)
                  const maxD = d3.max(dens) ?? 1
                  const wScale = d3.scaleLinear().domain([0, maxD]).range([0, bw / 2])
                  const area = d3.area<number>().x0((_, i) => cx - wScale(dens[i])).x1((_, i) => cx + wScale(dens[i])).y((d) => y(d)).curve(d3.curveCatmullRom)
                  return (
                    <g key={g.name} onMouseEnter={(e) => tip.show(e, [g.name, `n = ${g.box.n}`, `median = ${fmt(g.box.median)}`])} onMouseLeave={tip.hide}>
                      <path d={area(grid) ?? ''} fill={g.color} fillOpacity={0.55} stroke={g.color} strokeWidth={1.2} />
                      <line x1={cx} x2={cx} y1={y(g.box.q1)} y2={y(g.box.q3)} stroke={INK.text} strokeWidth={5} strokeLinecap="round" />
                      <circle cx={cx} cy={y(g.box.median)} r={3} fill="#fff" stroke={INK.text} strokeWidth={1.4} />
                    </g>
                  )
                }

                if (mode === 'Beeswarm') {
                  return (
                    <g key={g.name}>
                      {g.items.map((it, i) => {
                        const st = focusStyle(it.row.id, hl.focusId, { r: 3, opacity: 0.6 })
                        return (
                          <circle
                            key={it.row.id}
                            cx={cx + jitter(gi * 997 + i, bw * 0.82)}
                            cy={y(it.v)}
                            r={st.r}
                            fill={g.color}
                            fillOpacity={st.opacity}
                            stroke={st.stroke}
                            strokeWidth={st.strokeWidth}
                            style={{ cursor: 'pointer' }}
                            onMouseEnter={(e) => { hl.setHover(it.row.id); tip.show(e, [it.row.id, `${it.row.site} · ${it.row.date}`, `${field} = ${fmt(it.v)}`, 'click for details']) }}
                            onMouseLeave={() => { hl.setHover(null); tip.hide() }}
                            onClick={() => hl.openSample(it.row.id)}
                          />
                        )
                      })}
                      <line x1={cx - bw / 2} x2={cx + bw / 2} y1={y(g.box.median)} y2={y(g.box.median)} stroke={INK.text} strokeWidth={2} pointerEvents="none" />
                    </g>
                  )
                }

                const b = g.box
                const withPoints = mode === 'Box + points'
                return (
                  <g key={g.name}>
                    <g onMouseEnter={(e) => tip.show(e, [g.name, `n = ${b.n}`, `median = ${fmt(b.median)}`, `IQR = ${fmt(b.q1)} – ${fmt(b.q3)}`, `${b.outliers.length} outside 1.5·IQR`])} onMouseLeave={tip.hide}>
                      <line x1={cx} x2={cx} y1={y(b.max)} y2={y(b.q3)} stroke={INK.text} strokeWidth={1.2} />
                      <line x1={cx} x2={cx} y1={y(b.q1)} y2={y(b.min)} stroke={INK.text} strokeWidth={1.2} />
                      <line x1={cx - bw / 4} x2={cx + bw / 4} y1={y(b.max)} y2={y(b.max)} stroke={INK.text} strokeWidth={1.2} />
                      <line x1={cx - bw / 4} x2={cx + bw / 4} y1={y(b.min)} y2={y(b.min)} stroke={INK.text} strokeWidth={1.2} />
                      <rect x={cx - bw / 2} y={y(b.q3)} width={bw} height={Math.max(1, y(b.q1) - y(b.q3))} fill={g.color} fillOpacity={withPoints ? 0.22 : 0.55} stroke={g.color} strokeWidth={1.4} rx={2} />
                      <line x1={cx - bw / 2} x2={cx + bw / 2} y1={y(b.median)} y2={y(b.median)} stroke={INK.text} strokeWidth={2.2} />
                      {!withPoints && b.outliers.map((o, i) => (
                        <circle key={i} cx={cx} cy={y(o)} r={2.6} fill="none" stroke={g.color} strokeWidth={1.2} />
                      ))}
                    </g>
                    {/* boxplot-with-jitter: every filter over its own summary, so the box can be read against the n it summarises */}
                    {withPoints && g.items.map((it, i) => {
                      const st = focusStyle(it.row.id, hl.focusId, { r: 2.4, opacity: 0.55 })
                      return (
                        <circle
                          key={it.row.id}
                          cx={cx + jitter(gi * 997 + i, bw * 0.7)}
                          cy={y(it.v)}
                          r={st.r}
                          fill={g.color}
                          fillOpacity={st.opacity}
                          stroke={st.stroke}
                          strokeWidth={st.strokeWidth}
                          style={{ cursor: 'pointer' }}
                          onMouseEnter={(e) => { hl.setHover(it.row.id); tip.show(e, [it.row.id, `${it.row.site} · ${it.row.date}`, `${field} = ${fmt(it.v)}`, 'click for details']) }}
                          onMouseLeave={() => { hl.setHover(null); tip.hide() }}
                          onClick={() => hl.openSample(it.row.id)}
                        />
                      )
                    })}
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
