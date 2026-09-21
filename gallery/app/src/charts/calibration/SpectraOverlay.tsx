import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Toggle } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'

export interface SpectrumSeries {
  label: string
  color: string
  wn: number[]
  median: number[]
  q25: number[]
  q75: number[]
  dash?: string
}

/**
 * Median spectrum with an interquartile band, one per series, wavenumber
 * decreasing to the right as FTIR convention has it. Used twice on the
 * Calibration tab: the three selection cohorts against the Addis median
 * (Ann's multi-cohort ask), and every SPARTAN evaluation site against each
 * other. The spectra space (raw or AIRSpec-baselined) is a control because
 * the two disagree about what the ~1600 cm⁻¹ Addis band even looks like.
 */
export function SpectraOverlay({
  id, title, subtitle, provenance, bySpace, defaultSpace = 'raw',
}: {
  id: string
  title: string
  subtitle: string
  provenance: string
  /** series per spectra space; an empty record renders an empty state */
  bySpace: Record<string, SpectrumSeries[]>
  defaultSpace?: string
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = 380

  const spaces = Object.keys(bySpace)
  const [space, setSpace] = useState(spaces.includes(defaultSpace) ? defaultSpace : spaces[0] ?? '')
  const [bands, setBands] = useState(true)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const series = (bySpace[space] ?? []).filter((s) => !hidden.has(s.label))

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const wnExt = d3.extent(series.flatMap((s) => s.wn)) as [number, number]
  const x = d3.scaleLinear().domain([wnExt[1] ?? 4000, wnExt[0] ?? 400]).range([0, innerW])
  const vals = series.flatMap((s) => (bands ? [...s.q25, ...s.q75] : s.median))
  const y = d3.scaleLinear().domain(d3.extent(vals) as [number, number]).range([innerH, 0]).nice()
  const marks = useMemo(() => [{ wn: 1600, label: '~1600 aromatic / carboxylate' }, { wn: 2920, label: 'C–H stretch' }, { wn: 1720, label: 'C=O' }], [])

  return (
    <ChartFrame
      id={id}
      title={title}
      subtitle={subtitle}
      provenance={provenance}
      controls={
        <>
          {spaces.length > 1 && <Segmented label="spectra" value={space} options={spaces} onChange={setSpace} />}
          <Toggle label="IQR bands" checked={bands} onChange={setBands} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {series.length === 0 ? (
          <Empty>No spectra exported for this space.</Empty>
        ) : (
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={space === 'raw' ? 'absorbance' : 'baselined absorbance'} gridWidth={innerW} tickCount={5} />
              <XAxis scale={x} y={innerH} label="wavenumber (cm⁻¹)" tickCount={8} />
              {marks.filter((m) => m.wn <= x.domain()[0] && m.wn >= x.domain()[1]).map((m) => (
                <g key={m.wn}>
                  <line x1={x(m.wn)} x2={x(m.wn)} y1={0} y2={innerH} stroke={INK.grid} strokeDasharray="3 3" />
                  <text x={x(m.wn) + (x(m.wn) > innerW - 150 ? -3 : 3)} y={10} textAnchor={x(m.wn) > innerW - 150 ? 'end' : 'start'} fontSize={9.5} fill={INK.muted} fontFamily={FONT.mono}>{m.label}</text>
                </g>
              ))}
              {series.map((s) => {
                const area = d3.area<number>().x((_, i) => x(s.wn[i])).y0((_, i) => y(s.q25[i])).y1((_, i) => y(s.q75[i]))
                const line = d3.line<number>().x((_, i) => x(s.wn[i])).y((v) => y(v))
                return (
                  <g key={s.label}>
                    {bands && <path d={area(s.median) ?? ''} fill={s.color} fillOpacity={0.12} pointerEvents="none" />}
                    <path d={line(s.median) ?? ''} fill="none" stroke={s.color} strokeWidth={1.8} strokeDasharray={s.dash} pointerEvents="none" />
                  </g>
                )
              })}
              <rect x={0} y={0} width={innerW} height={innerH} fill="transparent"
                onMouseMove={(e) => {
                  const b = wrapRef.current!.getBoundingClientRect()
                  const wn = x.invert(e.clientX - b.left - MARGIN.left)
                  tip.show(e, [`${Math.round(wn)} cm⁻¹`, ...series.map((s) => { const i = d3.bisector((w: number) => -w).left(s.wn.map((w) => -w), -wn); const k = Math.min(i, s.wn.length - 1); return `${s.label}: ${fmt(s.median[k], 4)}` })])
                }}
                onMouseLeave={tip.hide} />
            </g>
          </svg>
        )}
        <Legend items={(bySpace[space] ?? []).map((s) => ({ label: s.label, color: s.color, shape: s.dash ? ('dashed' as const) : ('line' as const) }))} hidden={hidden} onToggle={(l) => setHidden((h) => toggleIn(h, l))} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
