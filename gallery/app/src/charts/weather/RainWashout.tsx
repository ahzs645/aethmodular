import { useMemo, useRef } from 'react'
import { ScatterLayer } from '@/components/ScatterLayer'
import { seasonOnly } from '@/lib/seasons'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlightActions } from '@/lib/highlight'
import { fidProps, fidsProps, useFilterEdits } from '@/lib/filterEdits'
import { boxStats, fmt } from '@/lib/stats'
import { FONT, INK, withUnit } from '@/lib/theme'
import { AxisTitle } from '@/components/Axes'
import { Legend } from '@/components/Legend'
import { useAxisFields } from '@/components/AxisFieldPopover'
import { RAIN_CLASSES, rainClass, spearman } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'
import { ExcludedNote, toMarks, useMarkReason, useMarkedRows } from '@/lib/markedRows'

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
  // the brush only: the dots read the hover themselves (components/ScatterLayer)
  const hl = useHighlightActions()

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
          render={(p, pw, _, cell) => <WashoutPanel p={p} pw={pw} h={cell.expanded ? 380 : cell.height ?? H} meta={meta} field={field} prcpField={prcpField}
            selected={hl.selected} tip={tip} axisPicker={axisPicker} />} />}
        {tip.node}
      </div>
      <Legend items={RAIN_CLASSES.map((c) => ({ label: c.key, color: c.color, shape: 'square' as const }))} />
      <ExcludedNote fields={[field, prcpField]} of={(r) => shown.some((p) => p.site === r.site)} drawn />
      {panels.length > MAX_SITES && <Note>Showing the {MAX_SITES} sites with the most filters of {panels.length}.</Note>}
    </ChartFrame>
  )
}

type WashPanel = { site: string; classes: ((typeof RAIN_CLASSES)[number] & { members: FilterRow[]; box: ReturnType<typeof boxStats> })[] }

/**
 * One site's ladder: a box per rain class over its filters as jittered dots (components/ScatterLayer), drawn once per
 * data, size or brush. The dots' pointer layer covers the plot, so the boxes answer through it: where no dot is near,
 * hovering a box shows its summary, clicking lists its filters and right-clicking offers them to exclude.
 */
function WashoutPanel({ p, pw, h, meta, field, prcpField, selected, tip, axisPicker }: {
  p: WashPanel; pw: number; h: number; meta: MetaFile; field: string; prcpField: string
  selected: Set<string> | null; tip: ReturnType<typeof useTooltip>; axisPicker: boolean
}) {
  const edits = useFilterEdits()
  const { setHover } = useHighlightActions()
  const iw = pw - M.left - M.right, ih = h - M.top - M.bottom
  const { x, y } = useMemo(() => {
    const vals = p.classes.flatMap((c) => c.members.map((r) => r[field] as number))
    return {
      x: d3.scaleBand<string>().domain(RAIN_CLASSES.map((c) => c.key)).range([0, iw]).padding(0.25),
      y: d3.scaleLinear().domain([Math.min(0, d3.min(vals) ?? 0), d3.quantile(vals.slice().sort(d3.ascending), 0.99)! * 1.1 || 1]).range([ih, 0]).nice(),
    }
  }, [p, field, iw, ih])
  const bw = x.bandwidth()
  const centre = (key: string) => (x(key) ?? 0) + bw / 2
  // every filter in its class column, with a deterministic jitter so a re-render does not shuffle the dots
  const items = useMemo(() => p.classes.flatMap((c) => c.members.map((r, i) => ({ r, jx: ((i * 0.6180339) % 1 - 0.5) * bw * 0.8, key: c.key }))), [p, bw])
  const xy = useMemo(() => items.map(({ r, jx, key }) => [centre(key) + jx, y(Math.min(r[field] as number, y.domain()[1]))] as const), [items, y, field]) // eslint-disable-line react-hooks/exhaustive-deps
  const colorOf = (r: FilterRow) => meta.seasons.find((s) => s.name === r.season)?.color ?? INK.muted
  // Filters → excluded: Mark (lib/markedRows): this site's excluded filters as × in their rain class, in no box or n
  const markedAll = useMarkedRows()
  const markReason = useMarkReason()
  const marks = useMemo(() => toMarks(markedAll.filter((r) => r.site === p.site), (r) => {
    const v = r[field], rain = r[prcpField]
    if (typeof v !== 'number' || typeof rain !== 'number' || v < y.domain()[0] || v > y.domain()[1]) return null
    return [centre(rainClass(rain)) + (((r.id.length * 0.6180339 + v) % 1) - 0.5) * bw * 0.8, y(v)]
  }, markReason, (r) => [`${field}: ${fmt(r[field])}`, `rain ${fmt(r[prcpField], 2)} mm/day`]), [markedAll, p.site, field, prcpField, y, bw, markReason]) // eslint-disable-line react-hooks/exhaustive-deps
  const dots = useMemo(() => items.map(({ r }, i) => {
    const st = focusStyle(r.id, null, { r: 2.2 }, selected)
    return <circle key={r.id} cx={xy[i][0]} cy={xy[i][1]} r={st.r} fill={colorOf(r)}
      fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} data-dot={st.dot} {...fidProps(r.id)} />
  }), [items, xy, selected, meta.seasons]) // eslint-disable-line react-hooks/exhaustive-deps
  // the box under the pointer when no dot is: its class, title and members
  const onDot = useRef(false)
  const boxAt = (e: React.MouseEvent) => {
    const [mx, my] = d3.pointer(e, e.currentTarget)
    const c = p.classes.find((c) => c.box && Math.abs(mx - centre(c.key)) <= bw / 2 && my >= y(c.box.q3) && my <= y(c.box.q1))
    return c ? { c, title: `${p.site} · ${seasonOnly(c.key)}`, ids: c.members.map((r) => r.id) } : null
  }
  return (
    <svg width={pw} height={h} fontFamily={FONT.family}>
      <g transform={`translate(${M.left},${M.top})`}>
        {y.ticks(4).map((t) => <g key={t}><line x1={0} x2={iw} y1={y(t)} y2={y(t)} stroke={INK.grid} /><text x={-5} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={INK.muted}>{t}</text></g>)}
        {/* the measurement is the page's: its title opens the same field picker as the crossplots */}
        <AxisTitle pick="field"><text transform={`translate(${-38},${ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={INK.text}>
          {withUnit(field, meta.field_units)}{axisPicker && <tspan className="axis-pick-caret" dx={4} data-export="skip">▾</tspan>}</text></AxisTitle>
        {p.classes.map((c) => {
          const cx = centre(c.key)
          return (
            <g key={c.key}>
              {c.box && <g {...fidsProps(c.members.map((r) => r.id), `${p.site} · ${seasonOnly(c.key)}`)}>
                <line x1={cx} x2={cx} y1={y(Math.min(c.box.max, c.box.q3 + 1.5 * (c.box.q3 - c.box.q1), y.domain()[1]))} y2={y(Math.max(c.box.min, c.box.q1 - 1.5 * (c.box.q3 - c.box.q1)))} stroke={INK.axis} />
                <rect x={cx - bw / 2} width={bw} y={y(c.box.q3)} height={Math.max(1, y(c.box.q1) - y(c.box.q3))} fill={c.color} fillOpacity={0.35} stroke={c.color} />
                <line x1={cx - bw / 2} x2={cx + bw / 2} y1={y(c.box.median)} y2={y(c.box.median)} stroke={INK.text} strokeWidth={2} />
              </g>}
              <text x={cx} y={ih + 14} textAnchor="middle" fontSize={9.5} fill={INK.muted}>{c.key.split(' (')[0]}</text>
              <text x={cx} y={ih + 27} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.mono}>n={c.members.length}</text>
            </g>
          )
        })}
        {/* a dot answers first (the layer stops its click and menu); otherwise the box under the pointer does */}
        <g onMouseMove={(e) => {
            if (onDot.current) return
            const b = boxAt(e)
            ;(e.target as SVGElement).style.cursor = b ? 'pointer' : ''
            if (b) tip.show(e, [b.title, `n = ${b.c.members.length}`, `median ${fmt(b.c.box!.median)}`, `IQR ${fmt(b.c.box!.q1)} – ${fmt(b.c.box!.q3)}`, 'click to list them · right-click to exclude'])
            else tip.hide()
          }}
          onMouseLeave={() => { if (!onDot.current) tip.hide() }}
          onClick={(e) => { const b = boxAt(e); if (b) { e.stopPropagation(); tip.hide(); edits.openGroup({ title: b.title, ids: b.ids, fields: [field, prcpField] }) } }}
          onContextMenu={(e) => { const b = boxAt(e); if (b) edits.openMenu(e, { ids: b.ids, title: b.title, group: { title: b.title, ids: b.ids, fields: [field, prcpField] } }) }}>
          <ScatterLayer items={items} xy={xy} idOf={({ r }) => r.id} dots={dots} width={iw} height={ih} tip={tip} marks={marks}
            onHover={(e, { r }) => { onDot.current = true; tip.show(e, [r.id, `${r.date} · ${seasonOnly(String(r.season ?? ''))}`, `${field}: ${fmt(r[field])}`, `rain ${fmt(r[prcpField], 2)} mm/day`]); setHover(r.id) }}
            onLeave={() => { onDot.current = false; tip.hide(); setHover(null) }}
            hit={8} focusR={({ r }) => focusStyle(r.id, r.id, { r: 2.2 }).r}
            focus={({ r }, [cx, cy]) => { const st = focusStyle(r.id, r.id, { r: 2.2 }); return <circle cx={cx} cy={cy} r={st.r} fill={colorOf(r)} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} /> }} />
        </g>
      </g>
    </svg>
  )
}
