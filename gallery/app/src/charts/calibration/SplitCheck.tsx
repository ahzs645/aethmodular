import { useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibRun, CalibRunsFile } from '@/lib/types'
import { metricRow } from './common'

const SPLIT_LABEL: Record<string, string> = { all: 'all filters', early: 'early half', late: 'late half', odd: 'odd half', even: 'even half' }
const SPLIT_COLOR: Record<string, string> = { all: '#1f2933', early: '#E67E22', late: '#3498DB', odd: '#27AE60', even: '#8e44ad' }

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

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const span = (vals: number[], floor: number): [number, number] => {
    const lo = d3.min(vals) ?? 0, hi = d3.max(vals) ?? 1
    const pad = Math.max((hi - lo) * 0.35, floor)
    return [lo - pad, hi + pad]
  }
  const x = d3.scaleLinear().domain(span(rows.map(slopeOf), 0.02)).range([0, innerW]).nice()
  const y = d3.scaleLinear().domain(span(rows.map(icOf), 0.08)).range([innerH, 0]).nice()
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
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label="Test set Deming intercept (µg/m³)" gridWidth={innerW} tickCount={5} />
              <XAxis scale={x} y={innerH} label="Test set Deming slope" tickCount={6} />
              {x.domain()[0] <= 1 && 1 <= x.domain()[1] && <line x1={x(1)} x2={x(1)} y1={0} y2={innerH} stroke={INK.text} strokeDasharray="2 3" />}
              {y.domain()[0] <= 0 && 0 <= y.domain()[1] && <line x1={0} x2={innerW} y1={y(0)} y2={y(0)} stroke={INK.text} strokeDasharray="2 3" />}
              {rows.map((r, i) => (
                <g key={r.split} onMouseEnter={(e) => tip.show(e, [`${SPLIT_LABEL[r.split] ?? r.split} (n=${r.n})`, `Deming slope ${fmt(slopeOf(r), 3)} · intercept ${fmt(icOf(r), 3)} µg/m³`, `R² ${fmt(r.m!.R2 ?? null, 3)}`])} onMouseLeave={tip.hide}>
                  {r.split === 'all'
                    ? <rect x={x(slopeOf(r)) - 6} y={y(icOf(r)) - 6} width={12} height={12} transform={`rotate(45 ${x(slopeOf(r))} ${y(icOf(r))})`} fill={SPLIT_COLOR.all} />
                    : <circle cx={x(slopeOf(r))} cy={y(icOf(r))} r={6} fill={SPLIT_COLOR[r.split] ?? INK.muted} />}
                  <text x={x(slopeOf(r))} y={y(icOf(r)) + (i % 2 ? 18 : -11)} textAnchor="middle" fontSize={10} fill={INK.text} fontFamily={FONT.mono}>{r.split}</text>
                </g>
              ))}
            </g>
          </svg>
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
