import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusLast, focusStyle, useHighlight } from '@/lib/highlight'
import { boxStats, kde, fmt } from '@/lib/stats'
import { INK, MARGIN, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'
import { useChartGrouping } from '@/components/GroupingDialog'

import { MeasurementControl } from '@/components/AxisFieldPopover'
import { CategoryAxis, categoryAxisHeight } from '@/components/CategoryAxis'
import { useTileSetting } from '@/lib/url'

const MODES = ['Boxplot', 'Box + points', 'Violin', 'Beeswarm'] as const
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
  const hl = useHighlight()
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
                      {focusLast(g.items, (it) => it.row.id, hl.focusId).map(([it, i]) => {
                        const st = focusStyle(it.row.id, hl.focusId, { r: 3, opacity: 0.6 }, hl.selected)
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
                            data-fid={it.row.id} onClick={() => hl.openSample(it.row.id)}
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
                  <g key={g.name} data-export-site={g.site} data-export-series={g.name} data-export-color={g.color}>
                    <g onMouseEnter={(e) => tip.show(e, [g.name, `n = ${b.n}`, `median = ${fmt(b.median)}`, `IQR = ${fmt(b.q1)} – ${fmt(b.q3)}`, `${b.outliers.length} outside 1.5·IQR`])} onMouseLeave={tip.hide}>
                      <line x1={cx} x2={cx} y1={y(b.max)} y2={y(b.q3)} stroke={INK.text} strokeWidth={1.2} />
                      <line x1={cx} x2={cx} y1={y(b.q1)} y2={y(b.min)} stroke={INK.text} strokeWidth={1.2} />
                      <line x1={cx - bw / 4} x2={cx + bw / 4} y1={y(b.max)} y2={y(b.max)} stroke={INK.text} strokeWidth={1.2} />
                      <line x1={cx - bw / 4} x2={cx + bw / 4} y1={y(b.min)} y2={y(b.min)} stroke={INK.text} strokeWidth={1.2} />
                      <rect x={cx - bw / 2} y={y(b.q3)} width={bw} height={Math.max(1, y(b.q1) - y(b.q3))} fill={g.color} fillOpacity={withPoints ? 0.22 : 0.55} stroke={g.color} strokeWidth={1.4} rx={2} />
                      <line x1={cx - bw / 2} x2={cx + bw / 2} y1={y(b.median)} y2={y(b.median)} stroke={INK.text} strokeWidth={2.2} />
                      {/* each outlier is one filter, so it opens that filter's record */}
                      {!withPoints && g.items.filter((it) => b.outliers.includes(it.v)).map((it) => (
                        <circle
                          key={it.row.id} cx={cx} cy={y(it.v)} r={hl.focusId === it.row.id ? 4 : 2.6}
                          fill={hl.focusId === it.row.id ? g.color : INK.panel} stroke={g.color} strokeWidth={1.2}
                          style={{ cursor: 'pointer' }}
                          onMouseEnter={(e) => { e.stopPropagation(); hl.setHover(it.row.id); tip.show(e, [it.row.id, `${it.row.date} · ${it.row.site}`, `${fmt(it.v)} (outside 1.5·IQR)`, 'click for the filter record']) }}
                          onMouseLeave={() => { hl.setHover(null); tip.hide() }}
                          data-fid={it.row.id} onClick={() => hl.openSample(it.row.id)}
                        />
                      ))}
                    </g>
                    {/* boxplot-with-jitter: every filter over its own summary, so the box can be read against the n it summarises */}
                    {withPoints && focusLast(g.items, (it) => it.row.id, hl.focusId).map(([it, i]) => {
                      const st = focusStyle(it.row.id, hl.focusId, { r: 2.4, opacity: 0.55 }, hl.selected)
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
                          data-fid={it.row.id} onClick={() => hl.openSample(it.row.id)}
                        />
                      )
                    })}
                  </g>
                )
              })}
            </g>
          </svg></div>
        )}
        {groupBy === 'Group' && missingNote('left out of the groups')}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
