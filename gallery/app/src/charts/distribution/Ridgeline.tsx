import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented, Toggle } from '@/components/ChartFrame'
import { XAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { kde } from '@/lib/stats'
import { INK, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const ROWS = ['Season', 'Site', 'Year'] as const
const OVERLAPS = ['None', 'Some', 'Ridge'] as const
const OVERLAP_FACTOR: Record<(typeof OVERLAPS)[number], number> = { None: 0.95, Some: 1.35, Ridge: 1.8 }

/**
 * Ridgeline — stacked densities, one row per group. The estate has no
 * matplotlib equivalent; it is the gallery chart that most directly answers
 * "does the distribution shift by season or by year?", which several notebooks
 * currently approach with separate histogram panels.
 *
 * Which sites feed it is the subset bar's job — the old per-chart site select
 * fought with the site chips (deselect Addis up there, pick Addis down here,
 * get an empty chart).
 */
export function Ridgeline({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)

  const [groupBy, setGroupBy] = useState<(typeof ROWS)[number]>('Season')
  const [clip, setClip] = useState(true)
  const [overlapMode, setOverlapMode] = useState<(typeof OVERLAPS)[number]>('Some')

  const groups = useMemo(() => {
    const keyOf = (r: FilterRow) => (groupBy === 'Season' ? r.season : groupBy === 'Site' ? r.site : String(r.year ?? ''))
    const m = new Map<string, number[]>()
    for (const r of rows) {
      const v = r[field]
      if (typeof v !== 'number' || !Number.isFinite(v)) continue
      const k = keyOf(r)
      if (!k) continue
      if (!m.has(k)) m.set(k, [])
      m.get(k)!.push(v)
    }
    let order = [...m.keys()]
    if (groupBy === 'Season') order = meta.seasons.map((s) => s.name).filter((s) => m.has(s))
    else if (groupBy === 'Site') order = meta.sites.map((s) => s.name).filter((s) => m.has(s))
    else order = order.sort()

    const colorFor = (k: string) =>
      groupBy === 'Season'
        ? meta.seasons.find((s) => s.name === k)?.color ?? INK.muted
        : groupBy === 'Site'
          ? meta.sites.find((s) => s.name === k)?.color ?? INK.muted
          : d3.interpolateViridis(order.indexOf(k) / Math.max(1, order.length - 1))

    return order.filter((k) => m.get(k)!.length >= 8).map((k) => ({ name: k, values: m.get(k)!, color: colorFor(k) }))
  }, [rows, field, groupBy, meta])

  // Row labels can be long ("Belg (Feb-May, short rains)"), so the gutter is
  // sized from the actual text rather than a fixed margin that clips them.
  const labelPad = useMemo(() => {
    const longest = Math.max(8, ...groups.map((g) => g.name.length))
    return Math.min(210, Math.round(longest * 6.4) + 26)
  }, [groups])

  const rowH = 62
  // how far a row's peak may rise into the row above: the ridgeline's defining
  // trick, but at 1.7 rows it reads as a collision, so it is a control
  const overlap = OVERLAP_FACTOR[overlapMode]
  const margin = { top: Math.ceil(rowH * (overlap - 1)) + 16, right: 26, bottom: 66, left: labelPad }
  const innerW = Math.max(240, width - margin.left - margin.right)
  const innerH = groups.length * rowH
  const height = innerH + margin.top + margin.bottom

  const allVals = groups.flatMap((g) => g.values)
  const sorted = allVals.slice().sort(d3.ascending)
  const lo = clip ? (d3.quantile(sorted, 0.01) ?? d3.min(sorted) ?? 0) : (d3.min(sorted) ?? 0)
  const hi = clip ? (d3.quantile(sorted, 0.99) ?? d3.max(sorted) ?? 1) : (d3.max(sorted) ?? 1)
  const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]).nice()
  const [dLo, dHi] = x.domain() as [number, number]
  const nOutside = allVals.filter((v) => v < dLo || v > dHi).length

  const grid = useMemo(() => d3.range(90).map((i) => dLo + ((dHi - dLo) * i) / 89), [dLo, dHi])
  const densities = useMemo(() => groups.map((g) => kde(g.values, grid)), [groups, grid])
  const maxDensity = d3.max(densities.flat()) ?? 1

  return (
    <ChartFrame
      id="ridgeline"
      title="Ridgeline — distribution shift across groups"
      subtitle="One density per group, stacked, each scaled to the same peak height. Reading down the stack shows whether a season or a year moves the whole distribution or only its tail — a question the estate's separate histogram panels make you hold in your head. Use the site chips above to restrict it to one site."
      provenance="no matplotlib equivalent in the repo — a gallery chart the notebooks don't have · react-graph-gallery.com/ridgeline-plot"
      controls={
        <>
          <Segmented label="rows" value={groupBy} options={ROWS} onChange={setGroupBy} />
          <Segmented label="overlap" value={overlapMode} options={OVERLAPS} onChange={setOverlapMode} title="How far each density may rise into the row above" />
          <Toggle label="clip to 1–99 %" checked={clip} onChange={setClip} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {groups.length === 0 ? (
          <Empty>No group has 8 or more filters with {field} in this subset.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <defs>
              <clipPath id="ridge-clip">
                <rect x={0} y={-margin.top} width={innerW} height={innerH + margin.top} />
              </clipPath>
            </defs>
            <g transform={`translate(${margin.left},${margin.top})`}>
              <g clipPath="url(#ridge-clip)">
                {groups
                  .map((g, i) => ({ g, i }))
                  .reverse()
                  .map(({ g, i }) => {
                    const dens = densities[i]
                    const yBase = i * rowH + rowH
                    const h = d3.scaleLinear().domain([0, maxDensity]).range([0, rowH * overlap])
                    const area = d3.area<number>().x((_, k) => x(grid[k])).y0(yBase).y1((_, k) => yBase - h(dens[k])).curve(d3.curveBasis)
                    return <path key={g.name} d={area(grid) ?? ''} fill={g.color} fillOpacity={0.62} stroke={g.color} strokeWidth={1.4} />
                  })}
              </g>

              {groups.map((g, i) => {
                const yBase = i * rowH + rowH
                return (
                  <g key={`lab-${g.name}`}>
                    <line x1={0} x2={innerW} y1={yBase} y2={yBase} stroke={INK.grid} strokeWidth={1} />
                    <text x={-10} y={yBase - 5} textAnchor="end" fontSize={11.5} fill={INK.text} fontFamily={FONT.family}>{g.name}</text>
                    <text x={-10} y={yBase + 9} textAnchor="end" fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>n={g.values.length}</text>
                  </g>
                )
              })}

              <XAxis scale={x} y={innerH + 14} label={withUnit(field, meta.field_units)} />
            </g>
          </svg>
        )}
        {nOutside > 0 && (
          <Note>
            {nOutside} of {allVals.length} values fall outside the displayed range and are not drawn{clip ? ' — untick “clip” to include them' : ''}.
          </Note>
        )}
      </div>
    </ChartFrame>
  )
}
