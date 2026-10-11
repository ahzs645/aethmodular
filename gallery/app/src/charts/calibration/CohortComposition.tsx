import { useRef } from 'react'
import { Marker } from '@/components/RefLines'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CohortInfo } from '@/lib/types'

/**
 * The cohort's OC/EC composition against the whole IMPROVE pool — the
 * explorer's "composition ruler". Each histogram is scaled to its own peak so
 * a 450-filter cohort and a 13,000-filter pool can share the axis. The Addis
 * marker (1.34, FTIR-derived) is what the cohort is being pulled toward; the
 * pool median (5.52) is where the network sits.
 */
export function CohortComposition({ info, label }: { info: CohortInfo | undefined; label: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const lg = useLegend()
  const POOL = 'IMPROVE pool', COHORT = 'this cohort'
  const height = 280

  const c = info?.composition
  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const poolMax = c ? Math.max(1, d3.max(c.pool) ?? 1) : 1
  const cohortMax = c ? Math.max(1, d3.max(c.cohort) ?? 1) : 1
  const x = d3.scaleLinear().domain([0, 25]).range([0, innerW])
  const y = d3.scaleLinear().domain([0, 1.05]).range([innerH, 0])
  const bw = c ? innerW / c.centers.length : 1

  return (
    <ChartFrame
      id="cohort-composition"
      title="Cohort composition — TOR OC/EC of the cohort against the pool"
      subtitle="Relative frequency of the TOR OC/EC ratio (clipped at 25) for this cohort and for the whole IMPROVE pool, each scaled to its own peak. Selecting on Addis-likeness should pull the cohort toward the Addis marker; the lowest-OC/EC cohort does so by construction."
      provenance="calibration_explorer /api/cohort_info · composition ruler"
      controls={
        <span className="control">
          {info ? `${info.label} · ${info.n} filters · ${info.n_sites} sites · top: ${info.top_sites.slice(0, 3).join(', ')}` : label}
        </span>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {!c ? (
          <Empty>No cohort composition exported for {label}.</Empty>
        ) : (
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label="relative frequency" gridWidth={innerW} tickCount={4} />
              <XAxis scale={x} y={innerH} label="TOR OC/EC ratio (clipped at 25)" />
              {c.centers.map((cx, i) => (
                <g key={i}>
                  <rect x={x(cx) - bw / 2} y={y(c.pool[i] / poolMax)} width={Math.max(0.5, bw - 0.5)} height={innerH - y(c.pool[i] / poolMax)} fill={INK.muted} fillOpacity={lg.show(POOL) ? 0.35 * lg.dim(POOL) : 0}
                    onMouseEnter={(e) => tip.show(e, [`OC/EC ≈ ${fmt(cx, 2)}`, `pool: ${c.pool[i]} filters`, `cohort: ${c.cohort[i]} filters`])} onMouseLeave={tip.hide} />
                  {lg.show(COHORT) && <rect x={x(cx) - bw / 2} y={y(c.cohort[i] / cohortMax)} width={Math.max(0.5, bw - 0.5)} height={innerH - y(c.cohort[i] / cohortMax)} fill={INK.accent} fillOpacity={0.55 * lg.dim(COHORT, 0.2)} pointerEvents="none" />}
                </g>
              ))}
              <Marker orient="v" at={x(c.addis_marker)} from={0} to={innerH} color={INK.negative} />
              <text x={x(c.addis_marker) + 4} y={12} fontSize={10} fill={INK.negative} fontFamily={FONT.mono}>Addis {c.addis_marker}</text>
              <Marker orient="v" at={x(c.pool_median)} from={0} to={innerH} />
              <text x={x(c.pool_median) + 4} y={26} fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>pool median {c.pool_median}</text>
            </g>
          </svg>
        )}
        <Legend items={[{ label: POOL, color: INK.muted, shape: 'square' }, { label: COHORT, color: INK.accent, shape: 'square' }]} {...lg.props} note="each scaled to its own peak" />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
