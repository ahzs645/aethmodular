import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibFile, CalibRunsFile } from '@/lib/types'
import { cohortColor } from './common'
import { COHORT, SELECTION_SPACE, label } from '@/lib/labels'

const VIEWS = ['metric vs rank', 'distribution'] as const

/**
 * The selection diagnostic: the similarity metric each ranked cohort sorts
 * by, against rank, with the cutoff marked. A jump in the curve is a natural
 * cutoff; a smooth slope means the cutoff is a choice. Where the cohort can
 * be selected on AIRSpec-corrected spectra too, both rankings are overlaid
 * (the July next-step: do the two selections actually differ?).
 */
export function SelectionRanking({ runs, calib }: { runs: CalibRunsFile; calib: CalibFile | null }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const lg = useLegend()
  const height = 340

  const cohorts = useMemo(() => [...new Set(runs.rankings.map((r) => r.cohort))], [runs.rankings])
  const [cohort, setCohort] = useState(cohorts.includes('ocec') ? 'ocec' : cohorts[0] ?? '')
  const [view, setView] = useState<(typeof VIEWS)[number]>('metric vs rank')
  const series = runs.rankings.filter((r) => r.cohort === cohort)
  const labelOf = (sel: string) => label(SELECTION_SPACE, sel)
  const shown = series.filter((s) => lg.show(labelOf(s.selection_space)))
  const primary = series.find((s) => s.selection_space === 'raw') ?? series[0]
  const cutoff = primary?.default_cutoff ?? calib?.default_cutoff[cohort] ?? 0

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const x = d3.scaleLinear().domain([0, primary?.n_total ?? 1]).range([0, innerW]).nice()
  const allMetric = shown.flatMap((s) => s.metric)
  // The tail of the pool (OC/EC in the hundreds) would flatten the whole curve
  // near zero; clip the axis at p99, as the histogram view does, and pin the tail.
  const sortedMetric = allMetric.filter(Number.isFinite).sort(d3.ascending)
  const yHi = d3.quantile(sortedMetric, 0.99) ?? d3.max(sortedMetric) ?? 1
  const yLo = sortedMetric[0] ?? 0
  const nClipped = sortedMetric.filter((v) => v > yHi).length
  const y = d3.scaleLinear().domain([yLo, yHi]).range([innerH, 0]).nice()
  const yTop = y.domain()[1]
  const hx = d3.scaleLinear().domain(d3.extent(primary?.hist.centers ?? [0, 1]) as [number, number]).range([0, innerW]).nice()
  const hy = d3.scaleLinear().domain([0, d3.max(primary?.hist.counts ?? [1]) ?? 1]).range([innerH, 0]).nice()
  const colorOf = (sel: string) => (sel === 'raw' ? cohortColor(cohort) : INK.deming)

  return (
    <ChartFrame
      id="selection-ranking"
      title="Selection ranking — where the cutoff sits on the metric"
      subtitle={`${primary?.label ?? 'metric'} against rank for every TOR-eligible IMPROVE filter, lowest (most Addis-like) first. The dashed line is the locked cutoff. 'Distribution' shows the same metric as a histogram of the whole pool with the cutoff value marked.`}
      provenance="calibration_explorer /api/ranking · Selection tab"
      controls={
        <>
          <Select label="cohort" value={cohort} options={cohorts} onChange={setCohort} optionLabel={(c) => label(COHORT, c)} />
          <Segmented value={view} options={VIEWS} onChange={setView} />
          <span className="control">{primary ? `${primary.n_total.toLocaleString()} candidates · cutoff ${cutoff}` : ''}</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {!primary ? (
          <Empty>No ranking exported for {cohort}.</Empty>
        ) : view === 'metric vs rank' ? (
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={primary.label} gridWidth={innerW} tickCount={5} />
              <XAxis scale={x} y={innerH} label="rank (most Addis-like first)" />
              <rect x={0} y={0} width={x(cutoff)} height={innerH} fill={cohortColor(cohort)} fillOpacity={0.06} />
              <line x1={x(cutoff)} x2={x(cutoff)} y1={0} y2={innerH} stroke={INK.negative} strokeDasharray="5 3" />
              <text x={x(cutoff) + 4} y={12} fontSize={10} fill={INK.negative} fontFamily={FONT.mono}>cutoff {cutoff}</text>
              {shown.map((s) => {
                const line = d3.line<number>().x((_, i) => x(s.rank[i])).y((v) => y(Math.min(v, yTop)))
                return <path key={s.selection_space} d={line(s.metric) ?? ''} fill="none" stroke={colorOf(s.selection_space)} strokeWidth={s.selection_space === 'raw' ? 2 : 1.6} strokeDasharray={s.selection_space === 'raw' ? undefined : '6 3'} strokeOpacity={lg.dim(labelOf(s.selection_space))} />
              })}
              <rect x={0} y={0} width={innerW} height={innerH} fill="transparent"
                onMouseMove={(e) => {
                  const b = wrapRef.current!.getBoundingClientRect()
                  const rk = x.invert(e.clientX - b.left - MARGIN.left)
                  tip.show(e, [`rank ≈ ${Math.round(rk).toLocaleString()}`, ...shown.map((s) => { const i = d3.bisector((r: number) => r).left(s.rank, rk); return `${s.selection_space}: ${fmt(s.metric[Math.min(i, s.metric.length - 1)], 4)}` })])
                }}
                onMouseLeave={tip.hide} />
            </g>
          </svg>
        ) : (
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={hy} x={0} label="filters" gridWidth={innerW} tickCount={5} />
              <XAxis scale={hx} y={innerH} label={`${primary.label} (clipped at p99)`} />
              {primary.hist.centers.map((c, i) => {
                const w = innerW / primary.hist.centers.length
                const selected = c <= primary.cutoff_metric
                return <rect key={i} x={hx(c) - w / 2} y={hy(primary.hist.counts[i])} width={Math.max(0.5, w - 0.6)} height={innerH - hy(primary.hist.counts[i])} fill={selected ? cohortColor(cohort) : INK.muted} fillOpacity={selected ? 0.7 : 0.35}
                  onMouseEnter={(e) => tip.show(e, [`${fmt(c, 4)}`, `${primary.hist.counts[i]} filters`, selected ? 'inside the cutoff' : 'rest of pool'])} onMouseLeave={tip.hide} />
              })}
              <line x1={hx(primary.cutoff_metric)} x2={hx(primary.cutoff_metric)} y1={0} y2={innerH} stroke={INK.negative} strokeDasharray="5 3" />
              <text x={hx(primary.cutoff_metric) + 4} y={12} fontSize={10} fill={INK.negative} fontFamily={FONT.mono}>cutoff {cutoff} → {fmt(primary.cutoff_metric, 3)}</text>
            </g>
          </svg>
        )}
        <Legend
          items={series.map((s) => ({ label: labelOf(s.selection_space), color: colorOf(s.selection_space), shape: s.selection_space === 'raw' ? ('line' as const) : ('dashed' as const) }))}
          {...lg.props}
        />
        {view === 'metric vs rank' && nClipped > 0 && (
          <p className="chart-note">Axis clipped at p99: {nClipped.toLocaleString()} tail values (up to {fmt(sortedMetric[sortedMetric.length - 1], 3)}) run along the top edge.</p>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
