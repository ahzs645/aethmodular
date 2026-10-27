import { useRef } from 'react'
import { AxisChoices } from '@/components/AxisChoices'
import { useSearchParam } from '@/lib/url'
import { RefLine } from '@/components/RefLines'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import { dotRim } from '@/lib/highlight'
import type { CalibRun, CalibRunsFile } from '@/lib/types'
import { metricRow } from './common'

const SPLIT_LABEL: Record<string, string> = { all: 'all filters', early: 'early half', late: 'late half', odd: 'odd half', even: 'even half' }
const SPLIT_COLOR: Record<string, string> = { get all() { return INK.text }, early: '#E67E22', late: '#3498DB', odd: '#27AE60', even: '#8e44ad' }

/**
 * Blind-half check (Ann, 2026-08-27): the same fitted model scored on the two
 * date-ordered halves of the evaluation set and on the two interleaved halves.
 * A gap between early and late that odd/even does not show is a time trend;
 * a gap both show is sampling scatter. The axes follow the halves, so the
 * distance between points is the whole story. Deming only (agreed with Ann,
 * 23 Sep 2026): both axes of the crossplot carry measurement error.
 */
export function SplitCheck({ run, runs }: { run: CalibRun; runs: CalibRunsFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = 300

  const isFabs = run.ref_kind === 'fabs'
  const hasFixed = run.eval.fixed.some(Boolean)
  const rows = run.split_check
    .map((s) => ({ ...s, m: metricRow(s.metrics, hasFixed ? 'fixed' : 'all', isFabs ? runs.mac_value : null) }))
    .filter((s) => s.m && s.m.deming_slope !== null)
  const slopeOf = (r: (typeof rows)[number]) => r.m!.deming_slope as number
  const icOf = (r: (typeof rows)[number]) => r.m!.deming_intercept as number
  // either axis (click its title, components/AxisChoices): slope, intercept or R² of each half
  const SPLIT_AXES: Record<string, (r: (typeof rows)[number]) => number> = { 'Deming slope': slopeOf, 'Deming intercept': icOf, 'R²': (r) => (r.m!.R2 ?? NaN) as number }
  const sx = useSearchParam<string>('splitx', 'Deming slope', Object.keys(SPLIT_AXES))
  const sy = useSearchParam<string>('splity', 'Deming intercept', Object.keys(SPLIT_AXES))
  const fx = SPLIT_AXES[sx[0]], fy = SPLIT_AXES[sy[0]]

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const span = (vals: number[], floor: number): [number, number] => {
    const lo = d3.min(vals) ?? 0, hi = d3.max(vals) ?? 1
    const pad = Math.max((hi - lo) * 0.35, floor)
    return [lo - pad, hi + pad]
  }
  const x = d3.scaleLinear().domain(span(rows.map(fx), sx[0] === 'Deming intercept' ? 0.08 : 0.02)).range([0, innerW]).nice()
  const y = d3.scaleLinear().domain(span(rows.map(fy), sy[0] === 'Deming intercept' ? 0.08 : 0.02)).range([innerH, 0]).nice()
  const by = Object.fromEntries(rows.map((r) => [r.split, r]))
  const gap = (a: string, b: string) => by[a] && by[b] ? `${a} → ${b}: intercept ${fmt(icOf(by[a]), 2)} → ${fmt(icOf(by[b]), 2)} (Δ ${fmt(Math.abs(icOf(by[a]) - icOf(by[b])), 2)}), slope ${fmt(slopeOf(by[a]), 2)} → ${fmt(slopeOf(by[b]), 2)}` : null
  const target = runs.targets[run.target]

  return (
    <ChartFrame
      id="split-check"
      title={`Blind-half check — does the readout move between halves at ${target?.site ?? run.target}?`}
      subtitle="One fitted model, scored on the early and late halves of the test set and on the odd and even (interleaved) halves, all with the same n. The axes follow the points: the distance between them is the finding. Dotted lines mark slope 1 and intercept 0 when they are in range."
      provenance="calibration_explorer /api/run split_check · Target readout tab"
    >
      <div ref={wrapRef} className="chart-wrap">
        {rows.length < 2 ? (
          <Empty>Too few test-set filters to split into equal halves.</Empty>
        ) : (
          <AxisChoices noun="metrics" x={sx} y={sy} options={Object.keys(SPLIT_AXES).map((k) => ({ key: k, group: 'Each half', unit: k === 'Deming intercept' ? 'µg/m³' : undefined }))}>
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={`Test set ${sy[0]}${sy[0] === 'Deming intercept' ? ' (µg/m³)' : ''}`} gridWidth={innerW} tickCount={5} pick="y" />
              <XAxis scale={x} y={innerH} label={`Test set ${sx[0]}${sx[0] === 'Deming intercept' ? ' (µg/m³)' : ''}`} tickCount={6} pick="x" />
              {sx[0] === 'Deming slope' && x.domain()[0] <= 1 && 1 <= x.domain()[1] && <RefLine orient="v" at={x(1)} from={0} to={innerH} />}
              {sy[0] === 'Deming intercept' && y.domain()[0] <= 0 && 0 <= y.domain()[1] && <RefLine orient="h" at={y(0)} from={0} to={innerW} />}
              {rows.map((r, i) => (
                <g key={r.split} onMouseEnter={(e) => tip.show(e, [`${SPLIT_LABEL[r.split] ?? r.split} (n=${r.n})`, `Deming slope ${fmt(slopeOf(r), 3)} · intercept ${fmt(icOf(r), 3)} µg/m³`, `R² ${fmt(r.m!.R2 ?? null, 3)}`])} onMouseLeave={tip.hide}>
                  {r.split === 'all'
                    ? <rect x={x(fx(r)) - 6} y={y(fy(r)) - 6} width={12} height={12} transform={`rotate(45 ${x(fx(r))} ${y(fy(r))})`} fill={SPLIT_COLOR.all} />
                    : <circle cx={x(fx(r))} cy={y(fy(r))} r={6} fill={SPLIT_COLOR[r.split] ?? INK.muted} {...dotRim()} />}
                  <text x={x(fx(r))} y={y(fy(r)) + (i % 2 ? 18 : -11)} textAnchor="middle" fontSize={10} fill={INK.text} fontFamily={FONT.mono}>{r.split}</text>
                </g>
              ))}
            </g>
          </svg>
          </AxisChoices>
        )}
        <p className="chart-note">
          {[gap('early', 'late'), gap('odd', 'even')].filter(Boolean).map((t, i) => <span key={i} style={{ display: 'block' }}>{t}</span>)}
          <span style={{ display: 'block' }}>A gap early/late shows but odd/even does not is a time trend; one both show is sampling scatter.</span>
        </p>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
