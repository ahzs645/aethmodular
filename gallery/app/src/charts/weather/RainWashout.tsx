import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { fidProps, fidsProps, useFilterEdits } from '@/lib/filterEdits'
import { boxStats, fmt } from '@/lib/stats'
import { FONT, INK, withUnit } from '@/lib/theme'
import { RAIN_CLASSES, rainClass, spearman } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'

const MAX_SITES = 12
const H = 260, M = { top: 26, right: 12, bottom: 44, left: 52 }

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
  const { width } = useDimensions(wrapRef)
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

  const cols = Math.max(1, Math.min(shown.length, Math.floor((width + 12) / 300)))
  const pw = (width - (cols - 1) * 12) / cols
  const iw = pw - M.left - M.right, ih = H - M.top - M.bottom
  const x = d3.scaleBand<string>().domain(RAIN_CLASSES.map((c) => c.key)).range([0, iw]).padding(0.25)

  return (
    <ChartFrame id={id} title={title}
      subtitle={`Each filter sorted by its window's mean precipitation (${prcpField}); boxes are the median and IQR, dots are filters coloured by season. ρ is Spearman's rank correlation between rain rate and ${field} at that site.`}
      provenance="addis_03_meteorology Tasks 2 and 5, meteorology_source_interaction §1">
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
        {shown.length === 0 ? <Empty>No filters carry both {field} and {prcpField} in this subset.</Empty> : shown.map((p) => {
          const vals = p.classes.flatMap((c) => c.members.map((r) => r[field] as number))
          const y = d3.scaleLinear().domain([Math.min(0, d3.min(vals) ?? 0), d3.quantile(vals.slice().sort(d3.ascending), 0.99)! * 1.1 || 1]).range([ih, 0]).nice()
          return (
            <svg key={p.site} width={pw} height={H} fontFamily={FONT.family}>
              <text x={M.left} y={15} fontSize={12} fontWeight={600} fill={INK.text}>{p.site}</text>
              <text x={pw - M.right} y={15} fontSize={11} textAnchor="end" fill={INK.muted} fontFamily={FONT.mono}>ρ = {fmt(p.rho, 2)} · n = {p.n}</text>
              <g transform={`translate(${M.left},${M.top})`}>
                {y.ticks(4).map((t) => <g key={t}><line x1={0} x2={iw} y1={y(t)} y2={y(t)} stroke={INK.grid} /><text x={-5} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={INK.muted}>{t}</text></g>)}
                <text transform={`translate(${-38},${ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={10.5} fill={INK.muted}>{withUnit(field, meta.field_units)}</text>
                {p.classes.map((c) => {
                  const cx = (x(c.key) ?? 0) + x.bandwidth() / 2, bw = x.bandwidth()
                  const title = `${p.site} · ${c.key}`
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
                      {c.members.map((r, i) => {
                        const v = Math.min(r[field] as number, y.domain()[1])
                        const st = focusStyle(r.id, hl.focusId, { r: 2.2, opacity: 0.7 }, hl.selected)
                        // deterministic jitter so a re-render does not shuffle the dots
                        const jx = ((i * 0.6180339) % 1 - 0.5) * bw * 0.8
                        return <circle key={r.id} cx={cx + jx} cy={y(v)} r={st.r} fill={meta.seasons.find((s) => s.name === r.season)?.color ?? INK.muted}
                          fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} style={{ cursor: 'pointer' }} {...fidProps(r.id)}
                          onMouseEnter={(e) => { hl.setHover(r.id); tip.show(e, [r.id, `${r.date} · ${r.season}`, `${field}: ${fmt(r[field])}`, `rain ${fmt(r[prcpField], 2)} mm/day`]) }}
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
        })}
        {tip.node}
      </div>
      <div className="legend">
        {RAIN_CLASSES.map((c) => <span key={c.key} className="legend-item"><span className="swatch square" style={{ background: c.color }} />{c.key}</span>)}
      </div>
      {panels.length > MAX_SITES && <Note>Showing the {MAX_SITES} sites with the most filters of {panels.length}.</Note>}
    </ChartFrame>
  )
}
