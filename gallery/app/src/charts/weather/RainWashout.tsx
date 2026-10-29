import { useMemo, useRef } from 'react'
import { seasonOnly } from '@/lib/seasons'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { useTooltip } from '@/hooks/useTooltip'
import { focusLast, focusStyle, useHighlight } from '@/lib/highlight'
import { fidProps, fidsProps, useFilterEdits } from '@/lib/filterEdits'
import { boxStats, fmt } from '@/lib/stats'
import { FONT, INK, withUnit } from '@/lib/theme'
import { AxisTitle } from '@/components/Axes'
import { Legend } from '@/components/Legend'
import { useAxisFields } from '@/components/AxisFieldPopover'
import { RAIN_CLASSES, rainClass, spearman } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'

const MAX_SITES = 12
// the site, ρ and n are the facet's title row, above the svg
const H = 240, M = { top: 8, right: 12, bottom: 44, left: 52 }

/**
 * Rain washout: the measurement by how wet each filter's window was (mean mm/day: none,
 * light, moderate, heavy, addis_03_meteorology's bins), one panel per site; dots coloured
 * by season, ρ = Spearman between rain rate and the measurement. The Addis notebooks asked
 * whether rain lowers BC (and in meteorology_source_interaction, whether it lowers biomass
 * BC more than fossil), so a falling box ladder is the washout signature.
 */
export function RainWashout({ rows, meta, field, prcpField, id = 'weather-washout', title = "Rain washout — the measurement by how wet each filter's window was" }: {
  rows: FilterRow[]; meta: MetaFile; field: string; prcpField: string; id?: string; title?: string
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const axisPicker = !!useAxisFields()
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const edits = useFilterEdits()

  const panels = useMemo(() => {
    const ok = rows.filter((r) => typeof r[field] === 'number' && typeof r[prcpField] === 'number')
    return d3.groups(ok, (r) => r.site).map(([site, rs]) => ({
      site, color: meta.sites.find((s) => s.name === site)?.color ?? INK.muted,
      rho: spearman(rs.map((r) => r[prcpField] as number), rs.map((r) => r[field] as number)),
      classes: RAIN_CLASSES.map((c) => {
        const members = rs.filter((r) => rainClass(r[prcpField] as number) === c.key)
        return { ...c, members, box: boxStats(members.map((r) => r[field] as number)) }
      }),
      n: rs.length,
    })).filter((p) => p.n >= 5).sort((a, b) => b.n - a.n)
  }, [rows, field, prcpField, meta.sites])
  const shown = panels.slice(0, MAX_SITES)


  return (
    <ChartFrame id={id} title={title}
      subtitle={`Each filter sorted by its window's mean precipitation (${prcpField}); boxes are the median and IQR, dots are filters coloured by season. ρ is Spearman's rank correlation between rain rate and ${field} at that site.`}
      provenance="addis_03_meteorology Tasks 2 and 5, meteorology_source_interaction §1">
      <div ref={wrapRef} className="chart-wrap">
        {shown.length === 0 ? <Empty>No filters carry both {field} and {prcpField} in this subset.</Empty> : <FacetGrid items={shown} minCell={288} maxCols={4} gap={12} expandable
          title={(p) => p.site} note={(p) => `ρ = ${fmt(p.rho, 2)} · n = ${p.n}`} label={(p) => `${p.site}: ${field} by rain`}
          render={(p, pw, _, cell) => {
          const h = cell.expanded ? 380 : cell.height ?? H
          const iw = pw - M.left - M.right, ih = h - M.top - M.bottom
          const x = d3.scaleBand<string>().domain(RAIN_CLASSES.map((c) => c.key)).range([0, iw]).padding(0.25)
          const vals = p.classes.flatMap((c) => c.members.map((r) => r[field] as number))
          const y = d3.scaleLinear().domain([Math.min(0, d3.min(vals) ?? 0), d3.quantile(vals.slice().sort(d3.ascending), 0.99)! * 1.1 || 1]).range([ih, 0]).nice()
          return (
            <svg width={pw} height={h} fontFamily={FONT.family}>
              <g transform={`translate(${M.left},${M.top})`}>
                {y.ticks(4).map((t) => <g key={t}><line x1={0} x2={iw} y1={y(t)} y2={y(t)} stroke={INK.grid} /><text x={-5} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={INK.muted}>{t}</text></g>)}
                {/* the measurement is the page's: its title opens the same field picker as the crossplots */}
                <AxisTitle pick="field"><text transform={`translate(${-38},${ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={INK.text}>
                  {withUnit(field, meta.field_units)}{axisPicker && <tspan className="axis-pick-caret" dx={4} data-export="skip">▾</tspan>}</text></AxisTitle>
                {p.classes.map((c) => {
                  const cx = (x(c.key) ?? 0) + x.bandwidth() / 2, bw = x.bandwidth()
                  const title = `${p.site} · ${seasonOnly(c.key)}`
                  return (
                    <g key={c.key}>
                      {c.box && <g style={{ cursor: 'pointer' }} {...fidsProps(c.members.map((r) => r.id), title)}
                        onClick={(e) => { e.stopPropagation(); tip.hide(); edits.openGroup({ title, ids: c.members.map((r) => r.id), fields: [field, prcpField] }) }}
                        onMouseEnter={(e) => tip.show(e, [title, `n = ${c.members.length}`, `median ${fmt(c.box!.median)}`, `IQR ${fmt(c.box!.q1)} – ${fmt(c.box!.q3)}`, 'click to list them · right-click to exclude'])}
                        onMouseLeave={tip.hide}>
                        <line x1={cx} x2={cx} y1={y(Math.min(c.box.max, c.box.q3 + 1.5 * (c.box.q3 - c.box.q1), y.domain()[1]))} y2={y(Math.max(c.box.min, c.box.q1 - 1.5 * (c.box.q3 - c.box.q1)))} stroke={INK.axis} />
                        <rect x={cx - bw / 2} width={bw} y={y(c.box.q3)} height={Math.max(1, y(c.box.q1) - y(c.box.q3))} fill={c.color} fillOpacity={0.35} stroke={c.color} />
                        <line x1={cx - bw / 2} x2={cx + bw / 2} y1={y(c.box.median)} y2={y(c.box.median)} stroke={INK.text} strokeWidth={2} />
                      </g>}
                      {focusLast(c.members, (r) => r.id, hl.focusId).map(([r, i]) => {
                        const v = Math.min(r[field] as number, y.domain()[1])
                        const st = focusStyle(r.id, hl.focusId, { r: 2.2 }, hl.selected)
                        // deterministic jitter so a re-render does not shuffle the dots
                        const jx = ((i * 0.6180339) % 1 - 0.5) * bw * 0.8
                        return <circle key={r.id} cx={cx + jx} cy={y(v)} r={st.r} fill={meta.seasons.find((s) => s.name === r.season)?.color ?? INK.muted}
                          fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} data-dot={st.dot} style={{ cursor: 'pointer' }} {...fidProps(r.id)}
                          onMouseEnter={(e) => { hl.setHover(r.id); tip.show(e, [r.id, `${r.date} · ${seasonOnly(String(r.season ?? ''))}`, `${field}: ${fmt(r[field])}`, `rain ${fmt(r[prcpField], 2)} mm/day`]) }}
                          onMouseLeave={() => { hl.setHover(null); tip.hide() }} onClick={(e) => { e.stopPropagation(); hl.openSample(r.id) }} />
                      })}
                      <text x={cx} y={ih + 14} textAnchor="middle" fontSize={9.5} fill={INK.muted}>{c.key.split(' (')[0]}</text>
                      <text x={cx} y={ih + 27} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.mono}>n={c.members.length}</text>
                    </g>
                  )
                })}
              </g>
            </svg>
          )
        }} />}
        {tip.node}
      </div>
      <Legend items={RAIN_CLASSES.map((c) => ({ label: c.key, color: c.color, shape: 'square' as const }))} />
      {panels.length > MAX_SITES && <Note>Showing the {MAX_SITES} sites with the most filters of {panels.length}.</Note>}
    </ChartFrame>
  )
}
