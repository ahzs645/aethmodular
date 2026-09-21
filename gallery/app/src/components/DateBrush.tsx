import { useEffect, useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { INK } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

export type DateRange = [string, string] // ISO YYYY-MM-DD, inclusive

/**
 * A brushable timeline of the whole estate: filters per month, stacked by
 * site. Dragging selects a date range that every tab then respects. This is
 * the gallery's brush pattern applied where the repo actually needs it —
 * Beijing reaches back to 2013, Delhi starts mid-2022, and there was no way to
 * say "only the overlap" without editing a notebook.
 *
 * Snaps to whole months so the URL carries something a human can read.
 */
export function DateBrush({
  rows,
  meta,
  range,
  onChange,
  width,
}: {
  /** rows before the date filter (but after the exclusion toggle) */
  rows: FilterRow[]
  meta: MetaFile
  range: DateRange | null
  onChange: (r: DateRange | null) => void
  width: number
}) {
  const gRef = useRef<SVGGElement>(null)
  const height = 44
  const margin = { left: 4, right: 4, top: 4, bottom: 14 }
  const innerW = Math.max(200, width - margin.left - margin.right)
  const innerH = height - margin.top - margin.bottom

  const { months, stacks, x, yMax } = useMemo(() => {
    const byMonth = new Map<string, Record<string, number>>()
    for (const r of rows) {
      const k = r.date.slice(0, 7)
      const rec = byMonth.get(k) ?? {}
      rec[r.site] = (rec[r.site] ?? 0) + 1
      byMonth.set(k, rec)
    }
    const keys = [...byMonth.keys()].sort()
    if (!keys.length) return { months: [] as string[], stacks: [], x: d3.scaleTime(), yMax: 1 }
    const first = new Date(keys[0] + '-01T00:00:00')
    const last = new Date(keys[keys.length - 1] + '-01T00:00:00')
    const end = new Date(last.getFullYear(), last.getMonth() + 1, 1)
    const x = d3.scaleTime().domain([first, end]).range([0, innerW])
    const siteNames = meta.sites.map((s) => s.name)
    const stacks = keys.map((k) => {
      const rec = byMonth.get(k)!
      let acc = 0
      const parts = siteNames.map((s) => {
        const v = rec[s] ?? 0
        const p = { site: s, y0: acc, y1: acc + v, color: meta.sites.find((m) => m.name === s)?.color ?? INK.muted }
        acc += v
        return p
      })
      return { key: k, d: new Date(k + '-01T00:00:00'), total: acc, parts }
    })
    const yMax = d3.max(stacks, (s) => s.total) ?? 1
    return { months: keys, stacks, x, yMax }
  }, [rows, meta.sites, innerW])

  const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0])

  // d3.brush wants to own its <g>; React owns everything around it.
  useEffect(() => {
    const g = gRef.current
    if (!g || !months.length) return
    const sel = d3.select(g)
    const snap = (d: Date) => new Date(d.getFullYear(), d.getMonth(), 1)
    const iso = (d: Date) => d3.timeFormat('%Y-%m-%d')(d)
    const brush = d3
      .brushX()
      .extent([[0, 0], [innerW, innerH]])
      .on('end', (ev: d3.D3BrushEvent<unknown>) => {
        if (!ev.sourceEvent) return // programmatic move — don't loop
        if (!ev.selection) {
          onChange(null)
          return
        }
        const [a, b] = ev.selection as [number, number]
        const d0 = snap(x.invert(a))
        const d1raw = x.invert(b)
        // round the right edge up to the end of its month
        const d1 = new Date(d1raw.getFullYear(), d1raw.getMonth() + 1, 0)
        if (+d1 <= +d0) {
          onChange(null)
          return
        }
        onChange([iso(d0), iso(d1)])
      })
    sel.call(brush)
    sel.selectAll('.selection').attr('fill', INK.accent).attr('fill-opacity', 0.14).attr('stroke', INK.accent)
    sel.selectAll('.handle').attr('fill', INK.accent).attr('fill-opacity', 0.6)
    if (range) {
      const a = x(new Date(range[0] + 'T00:00:00'))
      const b = x(new Date(range[1] + 'T00:00:00'))
      sel.call(brush.move, [Math.max(0, a), Math.min(innerW, b)])
    } else {
      sel.call(brush.move, null)
    }
    return () => {
      sel.on('.brush', null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [months.join(','), innerW, innerH, range?.[0], range?.[1]])

  if (!months.length) return null
  const [d0, d1] = x.domain() as [Date, Date]
  const years = d3.timeYear.range(d3.timeYear.ceil(d0), d1)

  return (
    <svg width={innerW + margin.left + margin.right} height={height} style={{ display: 'block' }}>
      <g transform={`translate(${margin.left},${margin.top})`}>
        {stacks.map((s) => {
          const x0 = x(s.d)
          const x1 = x(new Date(s.d.getFullYear(), s.d.getMonth() + 1, 1))
          return s.parts
            .filter((p) => p.y1 > p.y0)
            .map((p) => (
              <rect
                key={`${s.key}-${p.site}`}
                x={x0}
                y={y(p.y1)}
                width={Math.max(0.8, x1 - x0 - 0.4)}
                height={y(p.y0) - y(p.y1)}
                fill={p.color}
                fillOpacity={0.8}
              />
            ))
        })}
        <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke={INK.axis} />
        {years.map((yr) => (
          <g key={+yr} transform={`translate(${x(yr)},${innerH})`}>
            <line y2={3} stroke={INK.axis} />
            <text y={12} textAnchor="middle" fontSize={9.5} fill={INK.muted} fontFamily="var(--font)">
              {yr.getFullYear()}
            </text>
          </g>
        ))}
        <g ref={gRef} />
      </g>
    </svg>
  )
}
