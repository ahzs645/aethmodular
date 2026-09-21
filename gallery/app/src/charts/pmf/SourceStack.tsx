import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN } from '@/lib/theme'
import type { MetaFile, PmfFile, PmfRow } from '@/lib/types'

const MODES = ['Relative (%)', 'Absolute (µg/m³)'] as const
const SHAPES = ['Stacked', 'Streamgraph'] as const

/**
 * Stacked area / streamgraph — PMF source contributions over time.
 *
 * This is the chart the estate is most missing: exactly one stacked-area cell
 * exists across 827 figures, and source apportionment is precisely the
 * question it answers. Fractions are the normalised relative contributions
 * (normalize_gf_fractions); the raw GF1-GF5 columns are PM2.5 mass fractions
 * summing to 0.03-0.46 and would be wrong to stack directly.
 *
 * The streamgraph is the same stack with a wiggle offset and inside-out
 * ordering — it trades a readable baseline for a readable *shape*, which is
 * the right trade when the question is "when does charcoal swell".
 */
export function SourceStack({ pmf, rows: subsetRows, meta }: { pmf: PmfFile; rows: PmfRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = 440

  const [mode, setMode] = useState<(typeof MODES)[number]>('Relative (%)')
  const [shape, setShape] = useState<(typeof SHAPES)[number]>('Stacked')
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  const labels = pmf.sources.map((s) => s.label).filter((l) => !hidden.has(l))
  const colorOf = useMemo(() => new Map(pmf.sources.map((s) => [s.label, s.color])), [pmf.sources])

  // Season filtering and the active calendar are applied upstream by the
  // global subset bar, so every tab means the same thing by "Kiremt".
  const rows = useMemo(
    () => subsetRows.map((r) => ({ ...r, d: new Date(r.date + 'T00:00:00') })).sort((a, b) => +a.d - +b.d),
    [subsetRows]
  )

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const relative = mode === 'Relative (%)'

  const stacked = useMemo(() => {
    const src = relative ? 'fraction' : 'ugm3'
    const gen = d3
      .stack<(typeof rows)[number]>()
      .keys(labels)
      .value((r, k) => {
        const v = (r as any)[src][k]
        return typeof v === 'number' && Number.isFinite(v) ? v : 0
      })
    if (shape === 'Streamgraph') gen.offset(d3.stackOffsetWiggle).order(d3.stackOrderInsideOut)
    return gen(rows)
  }, [rows, labels, relative, shape])

  const x = d3.scaleTime().domain(d3.extent(rows, (r) => r.d) as [Date, Date]).range([0, innerW])
  const yLo = shape === 'Streamgraph' ? (d3.min(stacked, (l) => d3.min(l, (d) => d[0])) ?? 0) : 0
  const yHi = d3.max(stacked, (l) => d3.max(l, (d) => d[1])) ?? 1
  const y = d3.scaleLinear().domain([yLo, yHi * (shape === 'Streamgraph' ? 1 : 1.02)]).range([innerH, 0]).nice()

  const area = d3.area<d3.SeriesPoint<(typeof rows)[number]>>().x((d) => x(d.data.d)).y0((d) => y(d[0])).y1((d) => y(d[1])).curve(d3.curveMonotoneX)

  const seasonBands = useMemo(() => {
    if (!rows.length) return []
    const [d0, d1] = x.domain() as [Date, Date]
    const monthToSeason = new Map<number, string>()
    for (const s of meta.seasons) for (const m of s.months) monthToSeason.set(m, s.color)
    const bands: { x: number; w: number; color: string }[] = []
    let cur = new Date(d0.getFullYear(), d0.getMonth(), 1)
    while (cur < d1) {
      const next = new Date(cur.getFullYear(), cur.getMonth() + 1, 1)
      const c = monthToSeason.get(cur.getMonth() + 1)
      if (c) {
        const a = Math.max(0, x(cur))
        const b = Math.min(innerW, x(next))
        if (b > a) bands.push({ x: a, w: b - a, color: c })
      }
      cur = next
    }
    return bands
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, innerW, meta.seasons])

  return (
    <ChartFrame
      id="sourcestack"
      title="Stacked area · Streamgraph — PMF source contributions over time"
      subtitle={`${pmf.n} ETAD filters with a PMF solution. Relative mode stacks the normalised source mix to 100 %; absolute mode stacks the K_Fn concentrations in µg/m³, so the height is total apportioned mass. Click a source in the legend to drop it from the stack.`}
      provenance="the estate has exactly 1 stacked-area figure in 827 · react-graph-gallery.com/stacked-area-plot · /streamchart"
      controls={
        <>
          <Segmented label="scale" value={mode} options={MODES} onChange={setMode} />
          <Segmented label="shape" value={shape} options={SHAPES} onChange={setShape} title="Streamgraph: wiggle offset, inside-out order — no baseline, better shape" />
          <span className="control">{rows.length} filters · {pmf.site_name}</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {!rows.length ? (
          <Empty>No PMF filters in this subset.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {seasonBands.map((b, i) => (
                <rect key={i} x={b.x} y={-6} width={b.w} height={6} fill={b.color} fillOpacity={0.55} />
              ))}
              <YAxis
                scale={y}
                x={0}
                label={shape === 'Streamgraph' ? (relative ? 'share (baseline free)' : 'µg/m³ (baseline free)') : relative ? 'share of apportioned mass' : 'µg/m³'}
                gridWidth={innerW}
                format={(v: number) => (relative ? `${Math.round(v * 100)}%` : String(v))}
              />
              <XAxis scale={x as any} y={innerH} format={(d: Date) => d3.timeFormat('%b %Y')(d)} tickCount={7} />

              {stacked.map((layer) => (
                <path
                  key={String(layer.key)}
                  d={area(layer) ?? ''}
                  fill={colorOf.get(String(layer.key)) ?? INK.muted}
                  fillOpacity={0.88}
                  stroke="#fff"
                  strokeWidth={0.4}
                  style={{ transition: 'd 0.35s' }}
                  onMouseMove={(e) => {
                    const b = wrapRef.current!.getBoundingClientRect()
                    const dt = x.invert(e.clientX - b.left - MARGIN.left)
                    const near = rows.reduce((best, r) => (Math.abs(+r.d - +dt) < Math.abs(+best.d - +dt) ? r : best), rows[0])
                    tip.show(e, [
                      `${near.date} · ${near.season}`,
                      ...labels.map((l) => {
                        const f = near.fraction[l]
                        const a = near.ugm3[l]
                        return `${l}: ${f !== null ? (f * 100).toFixed(1) + ' %' : '—'}${a !== null ? ` (${fmt(a, 2)} µg/m³)` : ''}`
                      }),
                      `dominant: ${near.dominant_source ?? '—'} ${near.dominant_fraction !== null ? `(${(near.dominant_fraction * 100).toFixed(0)} %)` : ''}`,
                    ])
                  }}
                  onMouseLeave={tip.hide}
                />
              ))}
            </g>
          </svg>
        )}
        <Legend
          items={pmf.sources.map((s) => ({ label: s.label, color: s.color, shape: 'square' as const }))}
          hidden={hidden}
          onToggle={(l) => setHidden((h) => toggleIn(h, l))}
          note="top strip = Ethiopian season"
        />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
