import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlightActions } from '@/lib/highlight'
import { ScatterLayer } from '@/components/ScatterLayer'
import { boxStats, kde, fmt } from '@/lib/stats'
import { INK, MARGIN, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'
import { useChartGrouping } from '@/components/GroupingDialog'

import { MeasurementControl } from '@/components/AxisFieldPopover'
import { CategoryAxis, categoryAxisHeight } from '@/components/CategoryAxis'
import { useTileSetting } from '@/lib/url'

const MODES = ['Boxplot', 'Box + points', 'Violin', 'Beeswarm'] as const

/** one filter drawn as a point: a beeswarm or boxplot-with-jitter dot, or a plain boxplot's outlier */
type Pt = { v: number; row: FilterRow; g: { name: string; site?: string; color: string }; kind: 'swarm' | 'jitter' | 'outlier'; at: readonly [number, number] }

/** deterministic jitter so the points don't dance on every re-render */
const jitter = (i: number, w: number) => {
  const t = Math.sin(i * 12.9898) * 43758.5453
  return (t - Math.floor(t) - 0.5) * w
}
const GROUPS = ['Site', 'Season', 'Group'] as const

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
  // the brush selection only: the hovered filter is drawn by the points' ScatterLayer, so a hover does not re-render the chart
  const hl = useHighlightActions()
  const height = useFitHeight() ?? 440

  const [groupChoice, setGroupBy] = useTileSetting<(typeof GROUPS)[number]>('distribution.by', 'Site')
  const [mode, setMode] = useTileSetting<(typeof MODES)[number]>('distribution.encoding', 'Boxplot')
  const [comparison, setComparison] = useTileSetting<'Pooled' | 'Compare sites'>('distribution.compare', 'Compare sites')

  // Group: the reader's grouping (components/GroupingDialog), its bands in order, on the filters that carry the field
  const measured = useMemo(() => rows.filter((r) => typeof r[field] === 'number' && Number.isFinite(r[field] as number)), [rows, field])
  const { grp, picker, missingNote } = useChartGrouping('distribution', measured, () => setGroupBy('Group'))
  // Group is offered only once a grouping is picked; without one the panel groups by site
  const groupBy = groupChoice === 'Group' && !grp ? 'Site' : groupChoice
  const seasonColor = useMemo(() => new Map(meta.seasons.map((s) => [s.name, s.color])), [meta.seasons])
  const siteColor = useMemo(() => new Map(meta.sites.map((s) => [s.name, s.color])), [meta.sites])

  const groups = useMemo(() => {
    if (groupBy === 'Group' && !grp) return []
    const split = groupBy === 'Group' && comparison === 'Compare sites'
    const key = (r: FilterRow) => groupBy === 'Site' ? r.site : groupBy === 'Group' ? (grp!.labelOf(r) === null ? '' : split ? `${grp!.labelOf(r)}|${r.site}` : grp!.labelOf(r)!) : r.season
    const order = groupBy === 'Site' ? meta.sites.map(s => s.name) : groupBy === 'Group' ? split ? grp!.labels.flatMap(label => meta.sites.map(s => `${label}|${s.name}`)) : grp!.labels : meta.seasons.map(s => s.name)
    const m = new Map<string, { v: number; row: FilterRow }[]>()
    for (const r of measured) {
      const k = key(r)
      if (!k) continue
      if (!m.has(k)) m.set(k, [])
      m.get(k)!.push({ v: r[field] as number, row: r })
    }
    return order.filter(name => (m.get(name)?.length ?? 0) >= 4).map(name => {
      const items = m.get(name)!, site = items[0].row.site
      const parent = groupBy === 'Season' && name.startsWith(`${site} · `) ? site : split ? grp!.labelOf(items[0].row)! : undefined
      const label = groupBy === 'Season' && parent ? name.slice(site.length + 3) : split ? site : name
      const color = (groupBy === 'Site' || split ? siteColor.get(site) : groupBy === 'Group' ? grp!.colorOf(name) : seasonColor.get(name)) ?? INK.muted
      return { name, key: name, label, parent, parentColor: groupBy === 'Season' ? siteColor.get(site) : undefined, site: groupBy === 'Site' || groupBy === 'Season' || split ? site : undefined, items, values: items.map(i => i.v), color, box: boxStats(items.map(i => i.v))! }
    })
  }, [measured, field, groupBy, comparison, meta, siteColor, seasonColor, grp])

  const plotWidth = Math.max(width, groups.length * 112 + MARGIN.left + MARGIN.right)
  const innerW = Math.max(160, plotWidth - MARGIN.left - MARGIN.right)
  const x = d3.scaleBand<string>().domain(groups.map(g => g.name)).range([0, innerW]).padding(0.34)
  const bottom = 14 + categoryAxisHeight(groups, x)
  const innerH = Math.max(140, height - MARGIN.top - bottom)
  const allVals = groups.flatMap(g => g.values)
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(allVals) ?? 0), (d3.max(allVals) ?? 1) * 1.05]).range([innerH, 0]).nice()

  // the filters drawn as points: every one in the beeswarm and the boxplot-with-jitter, the outliers alone in the plain
  // boxplot. One layer for every group (components/ScatterLayer), drawn once; a hover is drawn over it
  const pts = useMemo<Pt[]>(() => {
    if (mode === 'Violin') return []
    const bw = x.bandwidth()
    return groups.flatMap((g, gi): Pt[] => {
      const cx = (x(g.name) ?? 0) + bw / 2
      if (mode === 'Boxplot') return g.items.filter((it) => g.box.outliers.includes(it.v)).map((it) => ({ ...it, g, kind: 'outlier' as const, at: [cx, y(it.v)] as const }))
      const kind = mode === 'Beeswarm' ? 'swarm' as const : 'jitter' as const
      return g.items.map((it, i) => ({ ...it, g, kind, at: [cx + jitter(gi * 997 + i, bw * (kind === 'swarm' ? 0.82 : 0.7)), y(it.v)] as const }))
    })
  }, [groups, mode, innerW, innerH]) // eslint-disable-line react-hooks/exhaustive-deps
  const xy = useMemo(() => pts.map((p) => p.at), [pts])
  const dotStyle = (p: Pt, focused: boolean) => focusStyle(p.row.id, focused ? p.row.id : null, p.kind === 'swarm' ? { r: 3 } : { r: 2.4, level: 'dense' }, focused ? null : hl.selected)
  const dots = useMemo(() => pts.map((p) => {
    const tags = { 'data-export-site': p.g.site, 'data-export-series': p.g.name, 'data-export-color': p.g.color, 'data-fid': p.row.id }
    if (p.kind === 'outlier') return <circle key={p.row.id} cx={p.at[0]} cy={p.at[1]} r={2.6} fill={INK.panel} stroke={p.g.color} strokeWidth={1.2} {...tags} />
    const st = dotStyle(p, false)
    return <circle key={p.row.id} cx={p.at[0]} cy={p.at[1]} r={st.r} fill={p.g.color} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} data-dot={st.dot} {...tags} />
  }), [pts, hl.selected]) // eslint-disable-line react-hooks/exhaustive-deps
  // the boxes' tooltip: the points' layer takes the pointer over the whole plot, so off a point it reads the box under it
  const onPoint = useRef(false)
  const boxTip = (e: React.MouseEvent<SVGGElement>) => {
    if (onPoint.current || mode === 'Beeswarm') return
    const [mx, my] = d3.pointer(e, e.currentTarget)
    const g = groups.find((g) => { const x0 = x(g.name) ?? 0; return mx >= x0 && mx <= x0 + x.bandwidth() && my >= y(g.box.max) - 3 && my <= y(g.box.min) + 3 })
    if (!g) { tip.hide(); return }
    const b = g.box
    tip.show(e, [g.name, `n = ${b.n}`, `median = ${fmt(b.median)}`, `IQR = ${fmt(b.q1)} – ${fmt(b.q3)}`, `${b.outliers.length} outside 1.5·IQR`])
  }

  return (
    <ChartFrame
      id="distribution"
      title="Boxplot · Violin · Beeswarm — one distribution, four encodings"
      subtitle="The same grouped values drawn four ways. The notebooks commit to a boxplot at write time; here the encoding is a control, which is what makes the shape of a small group (Delhi, n≈97) legible rather than hidden behind a five-number summary. 'Box + points' is the gallery's boxplot-with-jitter: the summary and every filter it summarises."
      provenance="stands in for 50 boxplot + 3 violin figures · plotting/distributions.py · react-graph-gallery.com/boxplot · /boxplot-jitter"
      controls={
        <>
          <MeasurementControl />
          <Segmented label="Group by" value={groupBy} options={grp ? GROUPS : GROUPS.filter((g) => g !== 'Group')} onChange={setGroupBy} />
          {picker}
          {groupBy === 'Group' && <Segmented label="Sites" value={comparison} options={['Pooled', 'Compare sites'] as const} onChange={setComparison} title="Apply shared grouping cutoffs to all filters; compare sites side by side within each band" />}
          <Segmented label="Encoding" value={mode} options={MODES} onChange={setMode} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {plotWidth > width + 1 && <p className="chart-note">Scroll horizontally to compare all {groups.length} groups.</p>}
        {groups.length === 0 ? (
          <Empty>{groupBy === 'Group' && !grp ? 'Choose a grouping, or make one with “New grouping…”.' : `No group has 4 or more filters with ${field} in this subset.`}</Empty>
        ) : (
          <div className="chart-scroll" tabIndex={0} role="region" aria-label="Distribution chart, scroll horizontally for all groups"><svg width={plotWidth} height={innerH + MARGIN.top + bottom} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={withUnit(field, meta.field_units)} gridWidth={innerW} pick="field" />
              <CategoryAxis items={groups} scale={x} y={innerH} />

              {groups.map((g) => {
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
                    <g key={g.name} data-export-site={g.site} data-export-series={g.name} data-export-color={g.color} onMouseEnter={(e) => tip.show(e, [g.name, `n = ${g.box.n}`, `median = ${fmt(g.box.median)}`])} onMouseLeave={tip.hide}>
                      <path d={area(grid) ?? ''} fill={g.color} fillOpacity={0.55} stroke={g.color} strokeWidth={1.2} />
                      <line x1={cx} x2={cx} y1={y(g.box.q1)} y2={y(g.box.q3)} stroke={INK.text} strokeWidth={5} strokeLinecap="round" />
                      <circle cx={cx} cy={y(g.box.median)} r={3} fill={INK.panel} stroke={INK.text} strokeWidth={1.4} />
                    </g>
                  )
                }

                if (mode === 'Beeswarm') {
                  return (
                    <g key={g.name} data-export-site={g.site} data-export-series={g.name} data-export-color={g.color}>
                      <line x1={cx - bw / 2} x2={cx + bw / 2} y1={y(g.box.median)} y2={y(g.box.median)} stroke={INK.text} strokeWidth={2} pointerEvents="none" />
                    </g>
                  )
                }

                const b = g.box
                const withPoints = mode === 'Box + points'
                return (
                  <g key={g.name} data-export-site={g.site} data-export-series={g.name} data-export-color={g.color}>
                    <line x1={cx} x2={cx} y1={y(b.max)} y2={y(b.q3)} stroke={INK.text} strokeWidth={1.2} />
                    <line x1={cx} x2={cx} y1={y(b.q1)} y2={y(b.min)} stroke={INK.text} strokeWidth={1.2} />
                    <line x1={cx - bw / 4} x2={cx + bw / 4} y1={y(b.max)} y2={y(b.max)} stroke={INK.text} strokeWidth={1.2} />
                    <line x1={cx - bw / 4} x2={cx + bw / 4} y1={y(b.min)} y2={y(b.min)} stroke={INK.text} strokeWidth={1.2} />
                    <rect x={cx - bw / 2} y={y(b.q3)} width={bw} height={Math.max(1, y(b.q1) - y(b.q3))} fill={g.color} fillOpacity={withPoints ? 0.22 : 0.55} stroke={g.color} strokeWidth={1.4} rx={2} />
                    <line x1={cx - bw / 2} x2={cx + bw / 2} y1={y(b.median)} y2={y(b.median)} stroke={INK.text} strokeWidth={2.2} />
                  </g>
                )
              })}
              {/* the filters as points (each outlier is one filter too; boxplot-with-jitter puts every filter over its own
                  summary, so the box can be read against the n it summarises): hover links it everywhere, a click opens its record */}
              {mode !== 'Violin' && (
                <g onMouseMove={boxTip} onMouseLeave={() => { if (!onPoint.current) tip.hide() }}>
                  <ScatterLayer items={pts} xy={xy} idOf={(p) => p.row.id} dots={dots} width={innerW} height={innerH}
                    focusR={(p) => (p.kind === 'outlier' ? 4 : dotStyle(p, true).r)}
                    focus={(p, [fx, fy]) => {
                      if (p.kind === 'outlier') return <circle cx={fx} cy={fy} r={4} fill={p.g.color} stroke={p.g.color} strokeWidth={1.2} />
                      const st = dotStyle(p, true)
                      return <circle cx={fx} cy={fy} r={st.r} fill={p.g.color} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} />
                    }}
                    onHover={(e, p) => {
                      onPoint.current = true
                      hl.setHover(p.row.id)
                      tip.show(e, p.kind === 'outlier' ? [p.row.id, `${p.row.date} · ${p.row.site}`, `${fmt(p.v)} (outside 1.5·IQR)`, 'click for the filter record']
                        : [p.row.id, `${p.row.site} · ${p.row.date}`, `${field} = ${fmt(p.v)}`, 'click for details'])
                    }}
                    onLeave={() => { onPoint.current = false; hl.setHover(null); tip.hide() }} />
                </g>
              )}
            </g>
          </svg></div>
        )}
        {groupBy === 'Group' && missingNote('left out of the groups')}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
