import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, FONT, RAMP_SEQUENTIAL } from '@/lib/theme'
import type { OverlapRow } from '@/lib/types'
import { cohortColor } from './common'

const VIEWS = ['Matrix', 'Chord'] as const

/** Cohort label → the fixed cohort colour, by keyword. */
function colorForCohort(name: string): string {
  const n = name.toLowerCase()
  if (n.includes('smoke')) return cohortColor('smoke')
  if (n.includes('ethiopia')) return cohortColor('eth_shaped')
  if (n.includes('analog')) return cohortColor('analogs')
  if (n.includes('oc/ec') || n.includes('ocec') || n.includes('lowest')) return cohortColor('ocec')
  return cohortColor('pool')
}

/**
 * Are the selection methods picking the same filters? Pairwise membership
 * overlap between the cohorts at their locked cutoffs, as a share of the
 * smaller set. The raw vs corrected Ethiopia-shaped pair is the July
 * next-step: 285 of 300 shared, so baseline correction barely moves the pool
 * even though it moves the calibration a lot.
 *
 * Two views of the same rows. The matrix reads exactly; the chord diagram
 * (react-graph-gallery's flow category, otherwise empty in the estate) shows
 * the same shared memberships as ribbons whose width is the shared count, so
 * "smoke-906 contains all of both Ethiopia-shaped cohorts" is one glance.
 */
export function OverlapMatrix({ rows }: { rows: OverlapRow[] }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [view, setView] = useState<(typeof VIEWS)[number]>('Matrix')
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const lg = useLegend()

  const { names, sizes, cell } = useMemo(() => {
    const names: string[] = []
    const sizes = new Map<string, number>()
    for (const r of rows) {
      if (!names.includes(r.a)) names.push(r.a)
      if (!names.includes(r.b)) names.push(r.b)
      sizes.set(r.a, r.n_a)
      sizes.set(r.b, r.n_b)
    }
    const cell = new Map<string, number>()
    for (const r of rows) {
      cell.set(`${r.a}|${r.b}`, r.overlap)
      cell.set(`${r.b}|${r.a}`, r.overlap)
    }
    return { names, sizes, cell }
  }, [rows])

  const labelW = 230
  const size = Math.max(48, Math.min(72, (width - labelW - 20) / Math.max(1, names.length)))
  const color = d3.scaleQuantize<string>().domain([0, 1]).range(RAMP_SEQUENTIAL)
  const short = (n: string) => n.replace(' (', ' ').replace(')', '')
  const vnames = useMemo(() => names.filter((n) => !lg.hidden.has(short(n))), [names, lg.hidden])

  // ---- chord layout: a symmetric matrix of shared members; the diagonal is
  // what each cohort does NOT share with anyone, so the arc length is the
  // cohort's size and the ribbons partition it.
  const chord = useMemo(() => {
    const names = vnames
    const n = names.length
    const m: number[][] = names.map(() => names.map(() => 0))
    for (let i = 0; i < n; i++) {
      let shared = 0
      for (let j = 0; j < n; j++) {
        if (i === j) continue
        const v = cell.get(`${names[i]}|${names[j]}`) ?? 0
        m[i][j] = v
        shared += v
      }
      m[i][i] = Math.max(0, (sizes.get(names[i]) ?? 0) - shared)
    }
    const layout = d3.chord().padAngle(0.04).sortSubgroups(d3.descending)(m)
    return { m, layout }
  }, [vnames, cell, sizes])

  const R = Math.max(120, Math.min(260, (width - 40) / 2 - 80))
  const arc = d3.arc<d3.ChordGroup>().innerRadius(R).outerRadius(R + 14)
  const ribbon = d3.ribbon<d3.Chord, d3.ChordSubgroup>().radius(R - 1)
  const legendIdx = lg.hover ? vnames.findIndex((n) => short(n) === lg.hover) : -1
  const hot = hoverIdx ?? (legendIdx >= 0 ? legendIdx : null)
  const ribbonTouches = (c: d3.Chord) => hot === null || c.source.index === hot || c.target.index === hot

  return (
    <ChartFrame
      id="overlap"
      title="Cohort overlap — are the selection methods picking the same filters?"
      subtitle="Shared members between every pair of cohorts at their locked cutoffs. Matrix: shaded as a fraction of the smaller cohort, the diagonal is each cohort's size. Chord: every cohort is an arc sized by its member count, ribbons are the filters two cohorts share, and the unribboned part of an arc is what that cohort alone selects."
      provenance="calibration_explorer /api/overlap · meeting item · react-graph-gallery.com/chord-diagram"
      controls={<Segmented label="view" value={view} options={VIEWS} onChange={setView} />}
    >
      <div ref={wrapRef} className="chart-wrap centered">
        {names.length < 2 ? (
          <Empty>No overlap rows exported.</Empty>
        ) : view === 'Matrix' ? (
          <svg width={labelW + size * names.length + 150} height={size * names.length + 150}>
            <g transform={`translate(${labelW},140)`}>
              {names.map((n, j) => (
                <text key={n} transform={`translate(${j * size + size / 2},-8) rotate(-40)`} fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>{short(n)}</text>
              ))}
              {names.map((a, i) => (
                <g key={a} transform={`translate(0,${i * size})`}>
                  <text x={-8} y={size / 2} dy="0.32em" textAnchor="end" fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>{short(a)}</text>
                  {names.map((b, j) => {
                    const diag = i === j
                    const v = diag ? sizes.get(a) ?? 0 : cell.get(`${a}|${b}`) ?? 0
                    const frac = diag ? 1 : v / Math.max(1, Math.min(sizes.get(a) ?? 1, sizes.get(b) ?? 1))
                    return (
                      <g key={b} onMouseEnter={(e) => tip.show(e, diag ? [a, `${v} filters`] : [`${short(a)} ∩ ${short(b)}`, `${v} shared filters`, `${(frac * 100).toFixed(0)} % of the smaller cohort`])} onMouseLeave={tip.hide}>
                        <rect x={j * size + 1} y={1} width={size - 2} height={size - 2} rx={3} fill={diag ? INK.empty : color(frac)} stroke="#fff" />
                        <text x={j * size + size / 2} y={size / 2} dy="0.32em" textAnchor="middle" fontSize={10.5} fontFamily={FONT.mono} fill={!diag && frac > 0.6 ? '#fff' : INK.text} pointerEvents="none">
                          {v}
                        </text>
                      </g>
                    )
                  })}
                </g>
              ))}
            </g>
          </svg>
        ) : (
          <svg width={width} height={2 * R + 190}>
            <g transform={`translate(${width / 2},${R + 95})`} fontFamily={FONT.family}>
              {chord.layout.map((c, i) => (
                <path
                  key={i}
                  d={ribbon(c as any) ?? ''}
                  fill={colorForCohort(vnames[c.source.index])}
                  fillOpacity={ribbonTouches(c) ? 0.55 : 0.08}
                  stroke={colorForCohort(vnames[c.source.index])}
                  strokeOpacity={ribbonTouches(c) ? 0.8 : 0.1}
                  style={{ transition: 'fill-opacity 0.2s, stroke-opacity 0.2s' }}
                  onMouseEnter={(e) =>
                    tip.show(e, [
                      `${short(vnames[c.source.index])} ∩ ${short(vnames[c.target.index])}`,
                      `${c.source.value} shared filters`,
                      `${((c.source.value / Math.max(1, Math.min(sizes.get(vnames[c.source.index]) ?? 1, sizes.get(vnames[c.target.index]) ?? 1))) * 100).toFixed(0)} % of the smaller cohort`,
                    ])
                  }
                  onMouseLeave={tip.hide}
                />
              ))}
              {chord.layout.groups.map((g) => {
                const mid = (g.startAngle + g.endAngle) / 2
                const flip = mid > Math.PI
                const name = vnames[g.index]
                const own = chord.m[g.index][g.index]
                return (
                  <g
                    key={g.index}
                    onMouseEnter={(e) => { setHoverIdx(g.index); tip.show(e, [name, `${sizes.get(name) ?? 0} filters`, `${own} shared with no other ${lg.hidden.size ? 'shown ' : ''}cohort`]) }}
                    onMouseLeave={() => { setHoverIdx(null); tip.hide() }}
                    style={{ cursor: 'default' }}
                  >
                    <path d={arc(g) ?? ''} fill={colorForCohort(name)} fillOpacity={hot === null || hot === g.index ? 0.9 : 0.35} stroke="#fff" style={{ transition: 'fill-opacity 0.2s' }} />
                    <text
                      transform={`rotate(${(mid * 180) / Math.PI - 90}) translate(${R + 20},0) ${flip ? 'rotate(180)' : ''}`}
                      dy="0.35em"
                      textAnchor={flip ? 'end' : 'start'}
                      fontSize={11}
                      fill={INK.text}
                    >
                      {short(name)}
                    </text>
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        {view === 'Matrix' ? (
          <div className="legend">
            <ColorLegend scale={color} label="shared, as % of the smaller cohort" format={(v) => `${Math.round(v * 100)} %`} width={160} />
          </div>
        ) : (
          <Legend
            items={names.map((n) => ({ label: short(n), color: colorForCohort(n) }))}
            {...lg.props}
            highlighted={hoverIdx !== null ? short(vnames[hoverIdx]) : lg.hover}
            note="hover an arc or a cohort to isolate its ribbons · click a cohort to drop it · ribbon width = shared filters"
          />
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
